"""
ExchangeNCLGPU: GPU-accelerated version of ExchangeNCL using JAX.
"""

from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchange import ExchangeNCL
from TB2J.gpu.jax_utils import (
    _compute_GR_single_e,
    _eigen_to_G_single_e,
    _pauli_block_interleaved,
    _require_jax,
    jax_to_numpy,
    numpy_to_jax,
)
from TB2J.gpu.jax_utils import (
    _jit as jit,
)
from TB2J.gpu.jax_utils import (
    _jnp as jnp,
)


@jit
def _pauli_decompose_pair(Gij, Gji):
    """
    Pauli decomposition for a pair of matrices (interleaved basis), with -R flip.
    Gij: (nR, ni, nj), Gji: (nR, nj, ni)
    Returns: Gij_blocks (nR, 4, ni, nj), Gji_blocks (nR, 4, nj, ni)
    """
    Gij_blocks = _pauli_block_interleaved(Gij)
    Gji_blocks = _pauli_block_interleaved(Gji)
    Gji_blocks = jnp.flip(Gji_blocks, axis=0)
    return Gij_blocks, Gji_blocks


@jit
def _compute_A_tensor(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    Compute A tensor for all R vectors and Pauli components.
    Returns: (nR, 4, 4) A tensor
    """
    X = jnp.einsum("ik,rukj->ruij", Pi, Gij_blocks)
    Y = jnp.einsum("jk,rvki->rvji", Pj, Gji_blocks)
    return jnp.einsum("ruij,rvji->ruv", X, Y) / jnp.pi


@jit
def _compute_A_tensor_orb(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    Compute A tensor with orbital decomposition for all R.
    Returns: A (nR, 4, 4), A_orb (nR, 4, 4, ni, nj)
    """
    X = jnp.einsum("ik,rukj->ruij", Pi, Gij_blocks)
    Y = jnp.einsum("jk,rvki->rvji", Pj, Gji_blocks)
    A_orb = jnp.einsum("ruij,rvji->ruvij", X, Y) / jnp.pi
    A = jnp.sum(A_orb, axis=(-2, -1))
    return A, A_orb


class ExchangeNCLGPU(ExchangeNCL):
    """
    GPU-accelerated version of ExchangeNCL using JAX.

    This class accelerates both GR computation and A tensor computation on GPU.
    """

    def __init__(self, tbmodels, atoms, **kwargs):
        """Initialize ExchangeNCLGPU."""
        _require_jax()
        super().__init__(tbmodels, atoms, **kwargs)

        self._jit_cache = {}

        self._evals_jax = None
        self._evecs_jax = None
        self._kpts_jax = None
        self._kweights_jax = None
        self._Rpts_jax = None

    def _prepare_jax_arrays(self):
        """Prepare JAX arrays for GPU computation."""
        if self._evals_jax is None:
            self._evals_jax = numpy_to_jax(self.G.evals)
            self._evecs_jax = numpy_to_jax(self.G.evecs)
            self._kpts_jax = numpy_to_jax(self.G.kpts)
            self._kweights_jax = numpy_to_jax(self.G.kweights)
            self._Rpts_jax = numpy_to_jax(self.short_Rlist)
            print(
                f"Prepared JAX arrays: evals {self._evals_jax.shape}, evecs {self._evecs_jax.shape}"
            )
            print(f"kpts {self._kpts_jax.shape}, Rpts {self._Rpts_jax.shape}")

    def _get_GR_gpu(self, energy):
        """
        Compute GR for a single energy point on GPU.

        Parameters:
        -----------
        energy : float
            Energy point

        Returns:
        --------
        jnp.ndarray : GR array of shape (nR, nbasis, nbasis)
        """
        Gk_all = _eigen_to_G_single_e(
            self._evals_jax, self._evecs_jax, self.G.efermi, energy
        )

        GR = _compute_GR_single_e(
            self._Rpts_jax, self._kpts_jax, Gk_all, self._kweights_jax, self.G.k2Rfactor
        )
        return GR

    def _get_jit_funcs(self, ni, nj):
        """Get or create JIT-compiled functions for given orbital counts."""
        key = (ni, nj)
        if key not in self._jit_cache:
            self._jit_cache[key] = {
                "pauli": _pauli_decompose_pair,
                "A": _compute_A_tensor,
                "A_orb": _compute_A_tensor_orb,
            }
        return self._jit_cache[key]

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated version of get_quantities_per_e.
        All computation happens on GPU, with minimal CPU-GPU transfer.
        """
        GR_jax = self._get_GR_gpu(e)

        magnetic_sites = self.ind_mag_atoms
        n_mag = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]

        P_jax = [numpy_to_jax(self.get_P_iatom(site)) for site in magnetic_sites]

        A_results = {}
        A_orb_results = {}

        for i, j in product(range(n_mag), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]
            ni, nj = len(idx), len(jdx)

            Gij = GR_jax[:, idx][:, :, jdx]
            Gji = GR_jax[:, jdx][:, :, idx]

            jit_funcs = self._get_jit_funcs(ni, nj)

            Gij_blocks, Gji_blocks = jit_funcs["pauli"](Gij, Gji)

            if self.orb_decomposition:
                A_val_tensor, A_orb_tensor = jit_funcs["A_orb"](
                    Gij_blocks, Gji_blocks, P_jax[i], P_jax[j]
                )
            else:
                A_val_tensor = jit_funcs["A"](
                    Gij_blocks, Gji_blocks, P_jax[i], P_jax[j]
                )
                A_orb_tensor = None

            A_results[(i, j)] = (A_val_tensor, idx, jdx)
            if A_orb_tensor is not None:
                A_orb_results[(i, j)] = A_orb_tensor

        for key, (A_val_tensor, _, _) in A_results.items():
            A_val_tensor.block_until_ready()

        A = {}
        A_orb = {}

        for (i, j), (A_val_tensor, idx, jdx) in A_results.items():
            A_val_np = jax_to_numpy(A_val_tensor)
            mi, mj = magnetic_sites[i], magnetic_sites[j]

            for iR, R_vec in enumerate(self.short_Rlist):
                if (R_vec, i, j) in self.distance_dict:
                    A[(R_vec, mi, mj)] = A_val_np[iR]

            if (i, j) in A_orb_results:
                A_orb_np = jax_to_numpy(A_orb_results[(i, j)])
                for iR, R_vec in enumerate(self.short_Rlist):
                    if (R_vec, i, j) in self.distance_dict:
                        A_orb[(R_vec, mi, mj)] = A_orb_np[iR]

        return {"AijR": A, "AijR_orb": A_orb}

    def calculate_all(self, use_gpu=True, vectorize_energy=True, e_batch_size=None):
        """
        Calculate all exchange parameters using GPU acceleration.
        """
        print("Green's function Calculation started (Full GPU acceleration).")
        self.validate()

        self._prepare_jax_arrays()

        npole = len(self.contour.path)
        weights = self.contour.weights

        for i, e in enumerate(
            tqdm(self.contour.path, total=npole, desc="Energy integration")
        ):
            result = self.get_quantities_per_e_gpu(e)
            w = weights[i]
            for key, val in result["AijR"].items():
                self.A_ijR[key] += val * w
            if self.orb_decomposition:
                for key, val in result["AijR_orb"].items():
                    if key in self.A_ijR_orb:
                        self.A_ijR_orb[key] += val * w
                    else:
                        self.A_ijR_orb[key] = val * w

        if npole > 0:
            dummy = np.zeros(npole)
            dummy[0] = 1.0
            factor = self.contour.integrate_values(dummy) / weights[0]
            for key in self.A_ijR:
                self.A_ijR[key] *= factor
            if self.orb_decomposition:
                for key in self.A_ijR_orb:
                    self.A_ijR_orb[key] *= factor

        self.get_rho_atom()
        self.compute_charge_and_magnetic_moments()
        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

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


ExchangeNCLJAX = ExchangeNCLGPU
ExchangeGPU = ExchangeNCLGPU
ExchangeNCL_GPU = ExchangeNCLGPU
