"""
ExchangeGPU: GPU-accelerated version of ExchangeNCL using JAX.

This module provides a GPU-accelerated implementation of the ExchangeNCL class
using the JAX framework for high-performance matrix computations.

JAX is an optional dependency. If not installed, the module will raise an
ImportError when ExchangeNCLGPU is instantiated.
"""

from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.exchange import ExchangeNCL
from TB2J.external import p_imap

# JAX imports - lazy loading to avoid hard dependency
_jax_available = None
_jnp = None
_jax = None
_jit = None
_vmap = None


def _check_jax():
    """Check if JAX is available."""
    global _jax_available, _jnp, _jax, _jit, _vmap
    if _jax_available is None:
        try:
            import jax
            import jax.numpy as jnp
            from jax import jit, vmap

            _jax = jax
            _jnp = jnp
            _jit = jit
            _vmap = vmap
            _jax_available = True
        except ImportError:
            _jax_available = False
    return _jax_available


def _require_jax():
    """Raise ImportError if JAX is not available."""
    if not _check_jax():
        raise ImportError(
            "JAX is required for ExchangeNCLGPU but is not installed. "
            "Install it with: pip install jax jaxlib"
        )


# =============================================================================
# JAX Helper Functions
# =============================================================================


def _pauli_block_all_jax(M):
    """
    JAX version: Decompose a spinor matrix into Pauli components.
    Returns [I, x, y, z] components.
    """
    _require_jax()
    jnp = _jnp

    n = M.shape[-1] // 2
    M_reshaped = M.reshape(*M.shape[:-2], 2 * n, 2 * n)

    # Extract blocks
    M00 = M_reshaped[..., ::2, ::2]
    M01 = M_reshaped[..., ::2, 1::2]
    M10 = M_reshaped[..., 1::2, ::2]
    M11 = M_reshaped[..., 1::2, 1::2]

    # Pauli decomposition
    I = (M00 + M11) / 2.0
    x = (M01 + M10) / 2.0
    y = (M01 - M10) * (-1j / 2.0)
    z = (M00 - M11) / 2.0

    return jnp.stack([I, x, y, z], axis=0)


def _compute_A_tensor_jax(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    JAX version: Compute A tensor for all R vectors and all Pauli components.

    Parameters:
    -----------
    Gij_blocks : jnp.ndarray, shape (nR, 4, ni, nj)
        Pauli-decomposed Green's function G(i,j,R) for all R
    Gji_blocks : jnp.ndarray, shape (nR, 4, nj, ni)
        Pauli-decomposed Green's function G(j,i,-R) for all R
    Pi : jnp.ndarray, shape (ni, ni)
        P matrix for atom i
    Pj : jnp.ndarray, shape (nj, nj)
        P matrix for atom j

    Returns:
    --------
    A_tensor : jnp.ndarray, shape (nR, 4, 4)
        A tensor for all R and all Pauli component pairs
    """
    _require_jax()
    jnp = _jnp

    # Compute X = Pi @ Gij and Y = Pj @ Gji for all components
    # Gij_blocks: (nR, 4, ni, nj), Pi: (ni, ni)
    X = jnp.einsum("ij,rabj->riab", Pi, Gij_blocks)  # (nR, 4, ni, nj)
    Y = jnp.einsum("ij,rbaj->riba", Pj, Gji_blocks)  # (nR, 4, nj, ni)

    # Compute A = X @ Y for all component pairs (a, b)
    # Result: (nR, 4, 4) - sum over orbitals and R
    A = jnp.einsum("raij,rbji->rab", X, Y) / jnp.pi

    return A


def _compute_A_tensor_orb_jax(Gij_blocks, Gji_blocks, Pi, Pj):
    """
    JAX version: Compute A tensor with orbital decomposition.

    Parameters:
    -----------
    Gij_blocks : jnp.ndarray, shape (nR, 4, ni, nj)
    Gji_blocks : jnp.ndarray, shape (nR, 4, nj, ni)
    Pi : jnp.ndarray, shape (ni, ni)
    Pj : jnp.ndarray, shape (nj, nj)

    Returns:
    --------
    A_tensor : jnp.ndarray, shape (nR, 4, 4)
    A_orb_tensor : jnp.ndarray, shape (nR, 4, 4, ni, nj)
    """
    _require_jax()
    jnp = _jnp

    # Compute X = Pi @ Gij and Y = Pj @ Gji
    X = jnp.einsum("ij,rabj->riab", Pi, Gij_blocks)
    Y = jnp.einsum("ij,rbaj->riba", Pj, Gji_blocks)

    # Compute orbital-resolved A
    A_orb = jnp.einsum("raij,rbji->rabij", X, Y) / jnp.pi

    # Sum over orbitals to get total A
    A = jnp.sum(A_orb, axis=(-2, -1))

    return A, A_orb


# =============================================================================
# ExchangeNCLGPU Class
# =============================================================================


class ExchangeNCLGPU(ExchangeNCL):
    """
    GPU-accelerated version of ExchangeNCL using JAX.

    This class provides the same API as ExchangeNCL but uses JAX for
    GPU-accelerated matrix computations. The heavy operations
    (Green's function processing, Pauli decomposition, and A tensor
    calculations) are performed on the GPU.

    Parameters:
    -----------
    tbmodels : Hamiltonian model
        The tight-binding model
    atoms : ASE Atoms object
        Atomic structure
    **kwargs : dict
        Additional arguments passed to ExchangeNCL

    Examples:
    ---------
    >>> from TB2J.exchangeGPU import ExchangeNCLGPU
    >>> exchange = ExchangeNCLGPU(
    ...     tbmodels=model,
    ...     atoms=model.atoms,
    ...     kmesh=[6, 6, 6],
    ... )
    >>> exchange.run(path="TB2J_results")
    """

    def __init__(self, tbmodels, atoms, **kwargs):
        """Initialize ExchangeNCLGPU."""
        _require_jax()

        # Store JAX module references before parent init
        self._jnp = _jnp
        self._jax = _jax

        # Call parent initialization
        super().__init__(tbmodels, atoms, **kwargs)

    def _numpy_to_jax(self, arr):
        """Convert numpy array to JAX array."""
        return self._jnp.array(arr)

    def _jax_to_numpy(self, arr):
        """Convert JAX array to numpy array."""
        return np.array(arr)

    def get_all_A_vectorized_gpu(self, GR, orb_indices_map=None):
        """
        GPU-accelerated vectorized calculation of all A matrix elements.

        Parameters:
        -----------
        GR : np.ndarray, shape (nR, nbasis, nbasis)
            Green's function array
        orb_indices_map : dict, optional
            Mapping from global to reduced orbital indices

        Returns:
        --------
        A : dict
            Dictionary of A matrices with (R_vec, mi, mj) keys
        A_orb : dict
            Dictionary of orbital-resolved A matrices (if orb_decomposition=True)
        """
        # Get magnetic sites and their orbital indices
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]

        if orb_indices_map is not None:
            new_iorbs = []
            for site_orbs in iorbs:
                new_orbs = np.array(
                    [orb_indices_map[orb_idx] for orb_idx in site_orbs], dtype=int
                )
                new_iorbs.append(new_orbs)
            iorbs = new_iorbs

        # Convert P matrices to JAX
        P = [self._numpy_to_jax(self.get_P_iatom(site)) for site in magnetic_sites]

        # Convert GR to JAX
        GR_jax = self._numpy_to_jax(GR)

        # Initialize results dictionary
        A = {}
        A_orb = {}

        # Process all atom pairs
        for i, j in product(range(len(magnetic_sites)), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]

            # Extract Gij and Gji for all R
            Gij = GR_jax[:, idx][:, :, jdx]  # (nR, ni, nj)
            Gji = GR_jax[:, jdx][:, :, idx]  # (nR, nj, ni)

            # Pauli decomposition (on GPU)
            Gij_blocks = _pauli_block_all_jax(Gij)  # (4, nR, ni, nj)
            Gji_blocks = _pauli_block_all_jax(Gji)  # (4, nR, nj, ni)

            # Transpose to (nR, 4, ni, nj) format
            Gij_blocks = self._jnp.transpose(Gij_blocks, (1, 0, 2, 3))
            Gji_blocks = self._jnp.transpose(Gji_blocks, (1, 0, 2, 3))

            # Flip Gji for -R (same as numpy version)
            Gji_blocks = self._jnp.flip(Gji_blocks, axis=0)

            # Compute A tensors on GPU
            if self.orb_decomposition:
                A_val_tensor, A_orb_tensor = _compute_A_tensor_orb_jax(
                    Gij_blocks, Gji_blocks, P[i], P[j]
                )
                A_orb_tensor = self._jax_to_numpy(A_orb_tensor)
            else:
                A_val_tensor = _compute_A_tensor_jax(Gij_blocks, Gji_blocks, P[i], P[j])
                A_orb_tensor = None

            # Convert back to numpy
            A_val_tensor = self._jax_to_numpy(A_val_tensor)

            mi, mj = magnetic_sites[i], magnetic_sites[j]

            # Store results for each R vector
            for iR, R_vec in enumerate(self.short_Rlist):
                if (R_vec, i, j) in self.distance_dict:
                    A[(R_vec, mi, mj)] = A_val_tensor[iR]
                    if A_orb_tensor is not None:
                        A_orb[(R_vec, mi, mj)] = A_orb_tensor[iR]

        return A, A_orb

    def get_quantities_per_e(self, e):
        """
        GPU-accelerated version of get_quantities_per_e.

        Parameters:
        -----------
        e : complex
            Energy point on the contour

        Returns:
        --------
        dict
            Dictionary with AijR and AijR_orb
        """
        mae = None

        # Compute Green's function (on CPU, then transfer to GPU)
        GR = self.G.get_GR(self.short_Rlist, energy=e, Gk_all=None)

        # Save diagonal elements if debug option is enabled
        if self.debug_options.get("compute_charge_moments", False):
            self.save_greens_function_diagonals(GR, e)

        # Use GPU-accelerated method
        try:
            AijR, AijR_orb = self.get_all_A_vectorized_gpu(GR)
        except Exception as err:
            print(f"GPU method failed: {err}, falling back to CPU method")
            AijR, AijR_orb = self.get_all_A_vectorized(GR)

        return dict(AijR=AijR, AijR_orb=AijR_orb, mae=mae)

    def calculate_all(self, use_gpu=True):
        """
        Calculate all exchange parameters.

        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU acceleration (default: True)
        """
        print("Green's function Calculation started.")

        self.validate()

        npole = len(self.contour.path)
        weights = self.contour.weights

        if self.nproc > 1:
            results = p_imap(
                self.get_quantities_per_e, self.contour.path, num_cpus=self.nproc
            )
        else:
            results = (
                self.get_quantities_per_e(e)
                for e in tqdm(self.contour.path, total=npole)
            )

        for i, result in enumerate(results):
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

    def run(self, path="TB2J_results", use_gpu=True):
        """
        Run the exchange calculation with optional GPU acceleration.

        Parameters:
        -----------
        path : str
            Output directory path
        use_gpu : bool
            Whether to use GPU acceleration (default: True)
        """
        self.calculate_all(use_gpu=use_gpu)
        self.write_output(path=path)
        self.finalize()


# Keep compatibility with old naming
ExchangeNCLJAX = ExchangeNCLGPU
ExchangeGPU = ExchangeNCLGPU
