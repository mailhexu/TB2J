"""
ExchangeCL2GPU: GPU-accelerated version of ExchangeCL2 using JAX.
"""

import os
from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.gpu.jax_utils import (
    _compute_GR_single_e,
    _eigen_to_G_single_e,
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)

_require_jax()
import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402


@jit
def _compute_collinear_A_batch(Gij_up, Gji_dn, Delta_i, Delta_j):
    """
    Compute collinear exchange A tensor for all R vectors.

    Returns: (nR, ni, ni) orbital-resolved A tensor, (nR,) total A values
    """
    X = jnp.einsum("ab,rbj->raj", Delta_i, Gij_up)
    Y = jnp.einsum("jk,rki->rji", Delta_j, Gji_dn)
    t = jnp.einsum("raj,rji->rai", X, Y)
    A_total = jnp.sum(t, axis=(1, 2))
    return t, A_total


class ExchangeCL2GPU(ExchangeCL2):
    """
    GPU-accelerated version of ExchangeCL2 using JAX.

    This class accelerates both GR computation and exchange tensor computation on GPU.
    """

    def __init__(self, tbmodels, atoms, **kwargs):
        """Initialize ExchangeCL2GPU."""
        _require_jax()

        self.atoms = atoms
        self.integration_method = kwargs.pop("integration_method", "CFR")
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

        Gk_all = _eigen_to_G_single_e(evals, evecs, efermi, energy)

        GR = _compute_GR_single_e(
            self._Rpts_jax, self._kpts_jax, Gk_all, self._kweights_jax, k2Rfactor
        )
        return GR

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated version of get_quantities_per_e for collinear case.
        All computation happens on GPU, with minimal CPU-GPU transfer.
        """
        GR_up_jax = self._get_GR_gpu(e, spin="up")
        GR_dn_jax = self._get_GR_gpu(e, spin="down")

        magnetic_sites = self.ind_mag_atoms
        n_mag = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]

        Delta_jax = [numpy_to_jax(self.get_Delta(site)) for site in magnetic_sites]

        Jorb_list = {}
        JJ_list = {}

        for i, j in product(range(n_mag), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]

            Gij_up = GR_up_jax[:, idx][:, :, jdx]
            Gji_dn = jnp.flip(GR_dn_jax[:, jdx][:, :, idx], axis=0)

            t_orb, A_total = _compute_collinear_A_batch(
                Gij_up, Gji_dn, Delta_jax[i], Delta_jax[j]
            )

            A_total.block_until_ready()

            t_orb_np = jax_to_numpy(t_orb)
            A_total_np = jax_to_numpy(A_total)

            mi, mj = magnetic_sites[i], magnetic_sites[j]

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

        self._prepare_jax_arrays()

        npole = len(self.contour.path)
        weights = self.contour.weights

        self.Jorb = {}
        self.JJ = {}

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


ExchangeCLGPU = ExchangeCL2GPU
ExchangeCL_JAX = ExchangeCL2GPU
