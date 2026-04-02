"""
ExchangeNCLGPU: GPU-accelerated version of ExchangeNCL using JAX.
"""

from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchange import ExchangeNCL
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
def _pauli_decompose(Gij, Gji):
    """
    JIT-compiled Pauli decomposition.

    Gij: (nR, ni, nj)
    Gji: (nR, nj, ni)
    Returns: Gij_blocks (nR, 4, ni, nj), Gji_blocks (nR, 4, nj, ni)
    """
    # Gij blocks
    M00_i = Gij[..., ::2, ::2]
    M01_i = Gij[..., ::2, 1::2]
    M10_i = Gij[..., 1::2, ::2]
    M11_i = Gij[..., 1::2, 1::2]

    I_i = (M00_i + M11_i) / 2.0
    x_i = (M01_i + M10_i) / 2.0
    y_i = (M01_i - M10_i) * 0.5j  # Note: positive 0.5j, matching CPU version
    z_i = (M00_i - M11_i) / 2.0
    Gij_blocks = jnp.stack([I_i, x_i, y_i, z_i], axis=1)  # (nR, 4, ni, nj)

    # Gji blocks
    M00_j = Gji[..., ::2, ::2]
    M01_j = Gji[..., ::2, 1::2]
    M10_j = Gji[..., 1::2, ::2]
    M11_j = Gji[..., 1::2, 1::2]

    I_j = (M00_j + M11_j) / 2.0
    x_j = (M01_j + M10_j) / 2.0
    y_j = (M01_j - M10_j) * 0.5j  # Note: positive 0.5j, matching CPU version
    z_j = (M00_j - M11_j) / 2.0
    Gji_blocks = jnp.stack([I_j, x_j, y_j, z_j], axis=1)  # (nR, 4, nj, ni)

    # Flip Gji for -R
    Gji_blocks = jnp.flip(Gji_blocks, axis=0)

    return Gij_blocks, Gji_blocks


@jit
def _compute_A_tensor_jax(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    JAX version: Compute A tensor for all R vectors and all Pauli components.
    Gij_blocks: (nR, 4, ni, nj)
    Gji_blocks: (nR, 4, nj, ni)
    Pi, Pj: (ni, ni), (nj, nj) projection matrices
    Returns: (nR, 4, 4) A tensor
    """
    # X = Pi @ Gij_blocks: (ni, ni) @ (nR, 4, ni, nj) -> (nR, 4, ni, nj)
    # einsum: "ik,rukj->ruij" - contract k (second index of Pi with third index of Gij)
    X = jnp.einsum("ik,rukj->ruij", Pi, Gij_blocks)

    # Y = Pj @ Gji_blocks: (nj, nj) @ (nR, 4, nj, ni) -> (nR, 4, nj, ni)
    # einsum: "ik,rvki->rvik" but we want output (nR, 4, nj, ni) with indices (r, v, j, i)
    # Actually: Y[r,v,j,i] = sum_k Pj[j,k] * Gji[r,v,k,i]
    # einsum: "jk,rvki->rvji"
    Y = jnp.einsum("jk,rvki->rvji", Pj, Gji_blocks)

    # A = X @ Y^T summed over orbitals: (nR, 4, 4)
    # A[r,u,v] = sum_{i,j} X[r,u,i,j] * Y[r,v,j,i] / pi
    A = jnp.einsum("ruij,rvji->ruv", X, Y) / jnp.pi

    return A


@jit
def _compute_A_tensor_orb_jax(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    JAX version: Compute A tensor with orbital decomposition for all R.
    """
    X = jnp.einsum("ik,rukj->ruij", Pi, Gij_blocks)
    Y = jnp.einsum("jk,rvki->rvji", Pj, Gji_blocks)

    # Orbital-resolved A: (nR, 4, 4, ni, nj)
    A_orb = jnp.einsum("ruij,rvji->ruvij", X, Y) / jnp.pi

    # Sum over orbitals to get total A
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

        # Cache for JIT-compiled functions per orbital count
        self._jit_cache = {}

        # Prepare JAX arrays for eigenvalues/eigenvectors (computed once)
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
        # Compute G(k, e) for all k-points on GPU
        Gk_all = _eigen_to_G_jax(
            self._evals_jax, self._evecs_jax, self.G.efermi, energy
        )

        # Fourier transform to G(R, e) on GPU
        GR = _compute_GR_single_e_jax(
            self._Rpts_jax, self._kpts_jax, Gk_all, self._kweights_jax, self.G.k2Rfactor
        )
        return GR

    def _get_jit_funcs(self, ni, nj):
        """Get or create JIT-compiled functions for given orbital counts."""
        key = (ni, nj)
        if key not in self._jit_cache:
            self._jit_cache[key] = {
                "pauli": _pauli_decompose,
                "A": _compute_A_tensor_jax,
                "A_orb": _compute_A_tensor_orb_jax,
            }
        return self._jit_cache[key]

    def get_quantities_per_e_gpu(self, e):
        """
        GPU-accelerated version of get_quantities_per_e.
        All computation happens on GPU, with minimal CPU-GPU transfer.
        """
        # Get GR for this energy on GPU
        GR_jax = self._get_GR_gpu(e)

        # Get magnetic sites and their orbital indices
        magnetic_sites = self.ind_mag_atoms
        n_mag = len(magnetic_sites)
        iorbs = [self.iorb(site) for site in magnetic_sites]

        # Build P matrices for all magnetic sites (on GPU)
        P_jax = [numpy_to_jax(self.get_P_iatom(site)) for site in magnetic_sites]

        # Collect all results first, then convert once
        A_results = {}
        A_orb_results = {}

        # Process all atom pairs
        for i, j in product(range(n_mag), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]
            ni, nj = len(idx), len(jdx)

            # Extract Gij and Gji for all R
            Gij = GR_jax[:, idx][:, :, jdx]  # (nR, ni, nj)
            Gji = GR_jax[:, jdx][:, :, idx]  # (nR, nj, ni)

            # Get JIT-compiled functions
            jit_funcs = self._get_jit_funcs(ni, nj)

            # Pauli decomposition on GPU
            Gij_blocks, Gji_blocks = jit_funcs["pauli"](Gij, Gji)

            # Compute A tensors on GPU
            if self.orb_decomposition:
                A_val_tensor, A_orb_tensor = jit_funcs["A_orb"](
                    Gij_blocks, Gji_blocks, P_jax[i], P_jax[j]
                )
            else:
                A_val_tensor = jit_funcs["A"](
                    Gij_blocks, Gji_blocks, P_jax[i], P_jax[j]
                )
                A_orb_tensor = None

            # Store JAX arrays for now, convert later
            A_results[(i, j)] = (A_val_tensor, idx, jdx)
            if A_orb_tensor is not None:
                A_orb_results[(i, j)] = A_orb_tensor

        # Block until all GPU computations are done
        for key, (A_val_tensor, _, _) in A_results.items():
            A_val_tensor.block_until_ready()

        # Now convert all results to numpy
        A = {}
        A_orb = {}

        for (i, j), (A_val_tensor, idx, jdx) in A_results.items():
            A_val_np = jax_to_numpy(A_val_tensor)
            mi, mj = magnetic_sites[i], magnetic_sites[j]

            # Store results for each R vector
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

        # Prepare JAX arrays once
        self._prepare_jax_arrays()

        npole = len(self.contour.path)
        weights = self.contour.weights

        # Process each energy point
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

        # Apply integration factor
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


# Keep compatibility with old naming
ExchangeNCLJAX = ExchangeNCLGPU
ExchangeGPU = ExchangeNCLGPU
ExchangeNCL_GPU = ExchangeNCLGPU
