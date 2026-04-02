"""
ExchangeCL2GPU: GPU-accelerated version of ExchangeCL2 using JAX.
"""

import os
from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.gpu.jax_utils import (
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)

# Import JAX after ensuring it's available
_require_jax()
import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402


@jit
def _eigen_to_G_jax(evals, evecs, efermi, energy):
    """
    JAX version: Calculate Green's function from eigenvalues/eigenvectors.
    Vectorized over k-points.

    evals: (nk, nb)
    evecs: (nk, nb, nb)
    energy: scalar
    Returns: (nk, nb, nb)
    """
    denominator = 1.0 / (-evals + (energy + efermi))
    return jnp.einsum(
        "kib,kb,kjb->kij",
        evecs,
        denominator,
        jnp.conj(evecs),
        optimize="optimal",
    )


@jit
def _compute_GR_single_e_jax(Rpts, kpts, Gks, kweights, k2Rfactor):
    """
    JAX version: Fourier transform from G(k) to G(R) for a single energy.

    Gks: (nk, nb, nb)
    kpts: (nk, 3)
    kweights: (nk,)
    Rpts: (nr, 3)
    Returns: (nr, nb, nb)
    """
    # phase = exp( k2Rfactor * R . k )
    phase = jnp.exp(k2Rfactor * jnp.einsum("ni,mi->nm", Rpts, kpts))
    phase *= kweights[None, :]  # (nr, nk)

    # GR = sum_k G(k) * phase(R, k)
    GR = jnp.einsum("kij,rk->rij", Gks, phase, optimize="optimal")
    return GR


@jit
def _compute_collinear_A_jax(Gij_up, Gji_dn, Delta_i, Delta_j):
    """
    JAX version: Compute collinear exchange A tensor for all R vectors.

    Formula: A = Delta_i @ Gij_up @ Delta_j @ Gji_dn

    Gij_up: (nR, ni, nj) - spin-up Green's function from atom i to j
    Gji_dn: (nR, nj, ni) - spin-down Green's function from atom j to i
    Delta_i: (ni, ni) - spin splitting on atom i
    Delta_j: (nj, nj) - spin splitting on atom j

    Returns: (nR, ni, nj) - orbital-resolved A tensor
    """
    # t = Delta_i @ Gij_up @ Delta_j @ Gji_dn
    # Using einsum for efficient computation
    # t[a,c] = sum_b sum_d (Delta_i[a,b] * Gij_up[b,d] * Delta_j[d,c] * Gji_dn[c,a])
    # But we want t[a,c] for each R, so:
    # t[r,a,c] = sum_b sum_d (Delta_i[a,b] * Gij_up[r,b,d] * Delta_j[d,c] * Gji_dn[r,c,a])

    # Step 1: X = Delta_i @ Gij_up -> (nR, ni, nj)
    X = jnp.einsum("ab,rbd->rad", Delta_i, Gij_up)

    # Step 2: Y = Delta_j @ Gji_dn -> (nR, nj, ni)
    Y = jnp.einsum("cd,rdc->rcc", Delta_j, Gji_dn)

    # Actually, let's be more careful about the indices
    # Gji_dn is (nR, nj, ni), we need to contract correctly
    # Y[r,c,a] = sum_d Delta_j[c,d] * Gji_dn[r,d,a]
    Y = jnp.einsum("cd,rda->rca", Delta_j, Gji_dn)

    # t[r,a,c] = sum_c X[r,a,c] * Y[r,c,a]... wait that's wrong
    # Let me reconsider: t = Delta_i @ Gij_up @ Delta_j @ Gji_dn
    # If Gij_up is (ni, nj) and Gji_dn is (nj, ni)
    # t[ai] = Delta_i[ab] @ Gij_up[bj] @ Delta_j[jk] @ Gji_dn[ki]
    # t[ai] = (Delta_i @ Gij_up @ Delta_j @ Gji_dn)[ai]

    # For batched version:
    # X[r,a,j] = Delta_i[a,b] * Gij_up[r,b,j]
    X = jnp.einsum("ab,rbj->raj", Delta_i, Gij_up)  # (nR, ni, nj)

    # Y[r,j,i] = Delta_j[j,k] * Gji_dn[r,k,i]
    Y = jnp.einsum("jk,rki->rji", Delta_j, Gji_dn)  # (nR, nj, ni)

    # t[r,a,i] = X[r,a,j] * Y[r,j,i] summed over j
    t = jnp.einsum("raj,rji->rai", X, Y)  # (nR, ni, ni)

    return t


@jit
def _compute_collinear_A_batch_jax(Gij_up, Gji_dn, Delta_i, Delta_j):
    """
    JAX version: Compute collinear exchange A tensor for all R vectors.
    Optimized batch version.

    Returns: (nR, ni, ni) orbital-resolved A tensor, and (nR,) total A values
    """
    # X = Delta_i @ Gij_up
    X = jnp.einsum("ab,rbj->raj", Delta_i, Gij_up)  # (nR, ni, nj)

    # Y = Delta_j @ Gji_dn
    Y = jnp.einsum("jk,rki->rji", Delta_j, Gji_dn)  # (nR, nj, ni)

    # t = X @ Y
    t = jnp.einsum("raj,rji->rai", X, Y)  # (nR, ni, ni)

    # Total A value for each R
    A_total = jnp.sum(t, axis=(1, 2))  # (nR,)

    return t, A_total


class ExchangeCL2GPU(ExchangeCL2):
    """
    GPU-accelerated version of ExchangeCL2 using JAX.

    This class accelerates both GR computation and exchange tensor computation on GPU.
    """

    def __init__(self, tbmodels, atoms, **kwargs):
        """Initialize ExchangeCL2GPU."""
        _require_jax()
        # Don't call super().__init__ because it will create CPU TBGreen
        # Instead, we'll set up everything manually

        # Initialize the parent class manually to avoid TBGreen creation
        self.atoms = atoms
        self.integration_method = kwargs.pop("integration_method", "CFR")
        # Call ExchangeParams __init__
        from TB2J.exchange_params import ExchangeParams

        ExchangeParams.__init__(self, **kwargs)
        self._prepare_kmesh(self._kmesh)
        self._prepare_Rlist()
        self._set_tbmodels_gpu(tbmodels)
        self._adjust_emin()
        self._prepare_elist(method=self.integration_method)
        self._prepare_basis()
        self._prepare_orb_dict()
        self._prepare_distance()

        self._is_collinear = True
        self.has_elistc = False

        # Initialize storage
        self.Jorb_list = {}
        self.JJ_list = {}
        self.Jorb = {}
        self.JJ = {}
        self.Jorb_list = {}
        self.JJ_list = {}
        self.exchange_Jdict = {}
        self.Jiso_orb = {}
        self.biquadratic = False

        self._clean_tbmodels()

    def _set_tbmodels_gpu(self, tbmodels):
        """
        Set up TB models with GPU-accelerated Green's functions.
        """
        from TB2J.GreenGPU import TBGreenGPU

        self.tbmodel_up, self.tbmodel_dn = tbmodels
        self.backend_name = self.tbmodel_up.name

        # Create GPU-accelerated Green's functions
        self.Gup = TBGreenGPU(
            tbmodel=self.tbmodel_up,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
            smearing_width=self.smearing,
        )
        self.Gdn = TBGreenGPU(
            tbmodel=self.tbmodel_dn,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
            smearing_width=self.smearing,
        )

        if self.write_density_matrix:
            self.Gup.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_up.pickle")
            )
            self.Gdn.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_dn.pickle")
            )

        self.norb = self.Gup.norb
        self.nbasis = self.Gup.nbasis + self.Gdn.nbasis

        self.rho_up = self.Gup.get_density_matrix()
        self.rho_dn = self.Gdn.get_density_matrix()
        self.Jorb_list = {}
        self.JJ_list = {}
        self.JJ = {}
        self.Jorb = {}
        self.HR0_up = self.Gup.H0
        self.HR0_dn = self.Gdn.H0
        self.Delta = self.HR0_up - self.HR0_dn

        if self.Gup.is_orthogonal and self.Gdn.is_orthogonal:
            self.is_orthogonal = True
        else:
            self.is_orthogonal = False

        self.exchange_Jdict = {}
        self.Jiso_orb = {}
        self.biquadratic = False
        self._is_collinear = True

        # Prepare JAX arrays for GPU computation
        self._evals_up_jax = None
        self._evecs_up_jax = None
        self._evals_dn_jax = None
        self._evecs_dn_jax = None
        self._kpts_jax = None
        self._kweights_jax = None
        self._Rpts_jax = None

    def set_tbmodels(self, tbmodels):
        """Override to prevent CPU TBGreen creation."""
        self._set_tbmodels_gpu(tbmodels)

    def _prepare_jax_arrays(self):
        """Prepare JAX arrays for GPU computation."""
        if self._evals_up_jax is None:
            self._evals_up_jax = numpy_to_jax(self.Gup.evals)
            self._evecs_up_jax = numpy_to_jax(self.Gup.evecs)
            self._evals_dn_jax = numpy_to_jax(self.Gdn.evals)
            self._evecs_dn_jax = numpy_to_jax(self.Gdn.evecs)
            self._kpts_jax = numpy_to_jax(self.Gup.kpts)
            self._kweights_jax = numpy_to_jax(self.Gup.kweights)
            self._Rpts_jax = numpy_to_jax(self.short_Rlist)
            print(
                f"Prepared JAX arrays for collinear: "
                f"evals_up {self._evals_up_jax.shape}, evals_dn {self._evals_dn_jax.shape}"
            )
            print(f"kpts {self._kpts_jax.shape}, Rpts {self._Rpts_jax.shape}")

    def _get_GR_gpu(self, energy, spin="up"):
        """
        Compute GR for a single energy point on GPU.

        Parameters:
        -----------
        energy : float
            Energy point
        spin : str
            "up" or "down" for spin channel

        Returns:
        --------
        jnp.ndarray : GR array of shape (nR, nbasis, nbasis)
        """
        if spin == "up":
            evals = self._evals_up_jax
            evecs = self._evecs_up_jax
            efermi = self.Gup.efermi
            k2Rfactor = self.Gup.k2Rfactor
        else:
            evals = self._evals_dn_jax
            evecs = self._evecs_dn_jax
            efermi = self.Gdn.efermi
            k2Rfactor = self.Gdn.k2Rfactor

        # Compute G(k, e) for all k-points on GPU
        Gk_all = _eigen_to_G_jax(evals, evecs, efermi, energy)

        # Fourier transform to G(r, e) on GPU
        GR = _compute_GR_single_e_jax(
            self._Rpts_jax, self._kpts_jax, Gk_all, self._kweights_jax, k2Rfactor
        )
        return GR

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated version of get_quantities_per_e for collinear case.
        All computation happens on GPU, with minimal CPU-GPU transfer.
        """
        # Get GR for both spin channels on GPU
        GR_up_jax = self._get_GR_gpu(e, spin="up")
        GR_dn_jax = self._get_GR_gpu(e, spin="down")

        # Get magnetic sites and their orbital indices
        magnetic_sites = self.ind_mag_atoms
        n_mag = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]

        # Build Delta matrices for all magnetic sites (on GPU)
        Delta_jax = [numpy_to_jax(self.get_Delta(site)) for site in magnetic_sites]

        # Initialize results
        Jorb_list = {}
        JJ_list = {}

        # Process all atom pairs
        for i, j in product(range(n_mag), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]

            # Extract Gij_up and Gji_dn for all R
            Gij_up = GR_up_jax[:, idx][:, :, jdx]  # (nR, ni, nj)
            # Flip Gdn for -R (Gji_dn for Gij_up)
            Gji_dn = jnp.flip(GR_dn_jax[:, jdx][:, :, idx], axis=0)  # (nR, nj, ni)

            # Compute exchange tensors on GPU
            t_orb, A_total = _compute_collinear_A_batch_jax(
                Gij_up, Gji_dn, Delta_jax[i], Delta_jax[j]
            )

            # Block until computation is done
            A_total.block_until_ready()

            # Convert to numpy
            t_orb_np = jax_to_numpy(t_orb)
            A_total_np = jax_to_numpy(A_total)

            mi, mj = magnetic_sites[i], magnetic_sites[j]

            # Store results for each R vector (only pairs in distance_dict)
            for iR, R_vec in enumerate(self.short_Rlist):
                if (R_vec, i, j) in self.distance_dict:
                    Jorb_list[(R_vec, mi, mj)] = t_orb_np[iR] / (4.0 * np.pi)
                    JJ_list[(R_vec, mi, mj)] = A_total_np[iR] / (4.0 * np.pi)

        return {"Jorb_list": Jorb_list, "JJ_list": JJ_list}

    def calculate_all(self, use_gpu=True, vectorize_energy=True, e_batch_size=None):
        """
        Calculate all exchange parameters using GPU acceleration.
        """
        print("Green's function Calculation started (Collinear GPU acceleration).")

        # Prepare JAX arrays once
        self._prepare_jax_arrays()

        npole = len(self.contour.path)
        weights = self.contour.weights

        # Initialize accumulators
        self.Jorb = {}
        self.JJ = {}

        # Process each energy point
        for i, e in enumerate(
            tqdm(self.contour.path, total=npole, desc="Energy integration")
        ):
            result = self.get_quantities_per_e_gpu(e)
            w = weights[i]

            for key, val in result["Jorb_list"].items():
                if key in self.Jorb:
                    self.Jorb[key] += val * w
                else:
                    self.Jorb[key] = val * w

            for key, val in result["JJ_list"].items():
                if key in self.JJ:
                    self.JJ[key] += val * w
                else:
                    self.JJ[key] = val * w

        # Apply integration factor
        if npole > 0:
            dummy = np.zeros(npole)
            dummy[0] = 1.0
            factor = self.contour.integrate_values(dummy) / weights[0]
            for key in self.Jorb:
                self.Jorb[key] *= factor
            for key in self.JJ:
                self.JJ[key] *= factor

        self.get_rho_atom()
        self.A_to_Jtensor()

    def run(
        self,
        path="TB2J_results",
        use_gpu=True,
        vectorize_energy=True,
        e_batch_size=None,
    ):
        """
        Run the exchange calculation with GPU acceleration.
        """
        self.calculate_all(
            use_gpu=use_gpu,
            vectorize_energy=vectorize_energy,
            e_batch_size=e_batch_size,
        )
        self.write_output(path=path)
        self.finalize()


# Keep compatibility with old naming
ExchangeCLGPU = ExchangeCL2GPU
ExchangeCL_JAX = ExchangeCL2GPU
