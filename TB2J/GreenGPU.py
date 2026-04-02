"""
GreenGPU: GPU-accelerated Green's function computation using JAX.
"""

import copy
import time

import numpy as np
from HamiltonIO.model.occupations import GaussOccupations

from TB2J.green import TBGreen, find_energy_ingap


class TBGreenGPU(TBGreen):
    """
    GPU-accelerated version of TBGreen using JAX.

    This class inherits from TBGreen and overrides the eigenvalue/eigenvector
    preparation to use GPU acceleration via JAX.
    """

    def __init__(
        self,
        tbmodel,
        kmesh=None,
        ibz=False,
        efermi=None,
        gamma=False,
        kpts=None,
        kweights=None,
        k_sym=False,
        use_cache=False,
        cache_path=None,
        nproc=1,
        initial_emin=-25,
        smearing_width=0.01,
    ):
        """
        Initialize TBGreenGPU.

        Note: use_gpu parameter is not needed here as this class is specifically
        for GPU computation. If GPU is not available, an error will be raised.
        """
        # Initialize base attributes (skip TBGreen.__init__ to avoid calling _prepare_eigen)
        self.initial_emin = initial_emin
        self.tbmodel = tbmodel
        self.is_orthogonal = tbmodel.is_orthogonal
        self.R2kfactor = tbmodel.R2kfactor
        self.k2Rfactor = -tbmodel.R2kfactor
        self.efermi = efermi
        self._use_cache = use_cache
        self.cache_path = cache_path
        self.use_gpu = True

        if use_cache:
            self._prepare_cache()

        self.prepare_kpts(
            kmesh=kmesh,
            ibz=ibz,
            gamma=gamma,
            kpts=kpts,
            kweights=kweights,
            tbmodel=tbmodel,
        )

        self.norb = tbmodel.norb
        self.nbasis = tbmodel.nbasis
        self.k_sym = k_sym
        self.nproc = nproc
        self.fermi_width = float(smearing_width)

        # Initialize Rmap for spin-phonon coupling
        self._Rmap = None
        self._Rmap_rev = None

        print(
            f"starting to prepare eigenvalues and eigenvectors for {self.nkpts} k-points..."
        )
        t0 = time.time()

        # Call GPU eigen preparation
        self._prepare_eigen()

        print(
            f"Finished preparing eigenvalues and eigenvectors. Time taken: {time.time() - t0:.2f} seconds"
        )

    def _prepare_eigen(self, solve=True, saveH=False):
        """
        Calculate eigenvalues and eigenvectors for all k-points using GPU acceleration.

        Uses JAX for GPU-accelerated computation of H(k), S(k) and their
        eigenvalue decomposition.
        """
        import jax.numpy as jnp

        from TB2J.gpu.jax_utils import (
            _compute_Hk_Sk_all_jax,
            _prepare_eigen_gpu,
            _prepare_HR_jax,
        )

        nkpts = len(self.kpts)
        self.nkpts = nkpts
        self.evals = np.zeros((nkpts, self.nbasis), dtype=float)
        self.H0 = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        self.evecs = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        self.H = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        if not self.is_orthogonal:
            self.S = np.zeros((nkpts, self.nbasis, self.nbasis), dtype=complex)
        else:
            self.S = None

        print("Using GPU for eigenvalue/eigenvector computation...")

        # Compute eigenvalues and eigenvectors on GPU
        self.evals, self.evecs, Sk_all = _prepare_eigen_gpu(self.tbmodel, self.kpts)

        # H is not computed on GPU, set to None
        self.H = None

        # Store S(k) if non-orthogonal
        if Sk_all is not None:
            self.S = Sk_all
        else:
            self.S = None

        # Compute H0 by averaging H(k) over all k-points on GPU
        Rpts_jax, HR_jax, SR_jax, R2kfactor = _prepare_HR_jax(self.tbmodel)
        kpts_jax = jnp.array(self.kpts)
        Hk_all, _ = _compute_Hk_Sk_all_jax(
            Rpts_jax, HR_jax, SR_jax, kpts_jax, R2kfactor
        )
        self.H0 = np.array(jnp.mean(Hk_all, axis=0))

        # Get Fermi energy if not provided
        if self.efermi is None:
            print("Calculating Fermi energy from eigenvalues")
            print(f"Number of electrons: {self.tbmodel.nel} ")

            occ = GaussOccupations(
                nel=self.tbmodel.nel, width=0.1, wk=self.kweights, nspin=2
            )
            self.efermi = occ.efermi(copy.deepcopy(self.evals))
            print(f"Fermi energy found: {self.efermi}")

        # Adjust energy range
        self.adjusted_emin = (
            find_energy_ingap(
                self.evals, rbound=self.efermi + self.initial_emin, gap=2.0
            )
            - self.efermi
        )

        self.evals, self.evecs = self._reduce_eigens(
            self.evals,
            self.evecs,
            emin=self.efermi + self.adjusted_emin,
            emax=self.efermi + 5.1,
        )

        # Handle caching if enabled
        if self._use_cache:
            evecs = self.evecs
            self.evecs_shape = self.evecs.shape
            self.evecs = np.memmap(
                self._get_cache_path("evecs.dat"),
                mode="w+",
                shape=self.evecs.shape,
                dtype=complex,
            )
            self.evecs[:, :, :] = evecs[:, :, :]
            del self.evecs

            if not self.is_orthogonal:
                S = self.S
                self.S = np.memmap(
                    self._get_cache_path("S.dat"),
                    mode="w+",
                    shape=(nkpts, self.nbasis, self.nbasis),
                    dtype=complex,
                )
                self.S[:] = S[:]
                del self.S

    def _get_cache_path(self, filename):
        """Get full path for cache file."""
        import os

        return os.path.join(self.cache_path, filename)
