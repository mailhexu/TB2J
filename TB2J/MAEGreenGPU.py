"""
MAEGreenGPU: GPU-accelerated version of MAEGreen using JAX.

This module provides a GPU-accelerated implementation of the MAEGreen class
using the JAX framework for high-performance matrix computations.

JAX is an optional dependency. If not installed, the module will raise an
ImportError when MAEGreenGPU is instantiated.
"""

import numpy as np
import tqdm
from typing_extensions import DefaultDict

from TB2J.external import p_map
from TB2J.MAEGreen import MAEGreen

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
            "JAX is required for MAEGreenGPU but is not installed. "
            "Install it with: pip install jax jaxlib"
        )


# =============================================================================
# JAX Helper Functions
# =============================================================================


def _rotation_matrix_jax(theta, phi):
    """
    JAX version: The unitary operator U, that U^dagger * s3 * U
    is the rotated s3 by theta and phi.
    """
    _require_jax()
    jnp = _jnp

    return jnp.array(
        [
            [jnp.cos(theta / 2), jnp.exp(-1j * phi) * jnp.sin(theta / 2)],
            [-jnp.exp(1j * phi) * jnp.sin(theta / 2), jnp.cos(theta / 2)],
        ]
    )


def _rotate_spinor_matrix_einsum_jax(M, theta, phi):
    """
    JAX version: Rotate the spinor matrix M by theta and phi using einsum.
    """
    _require_jax()
    jnp = _jnp

    shape = M.shape
    n1 = jnp.prod(jnp.array(shape[:-1])) // 2
    n2 = shape[-1] // 2
    Mnew = jnp.reshape(M, (n1, 2, n2, 2))
    U = _rotation_matrix_jax(theta, phi)
    UT = jnp.conj(U.T)

    # einsum: UT @ M @ U for each 2x2 block
    Mnew = jnp.einsum("ij, rjsk, kl -> risl", UT, Mnew, U)
    Mnew = jnp.reshape(Mnew, shape)
    return Mnew


def _eigen_to_G_jax(evals, evecs, efermi, energy):
    """
    JAX version: Calculate Green's function from eigenvalues/eigenvectors.
    G = evecs @ diag(1.0 / (-evals + energy + efermi)) @ evecs.conj().T
    """
    _require_jax()
    jnp = _jnp

    denominator = 1.0 / (-evals + (energy + efermi))
    return jnp.einsum(
        "ib, b, jb-> ij",
        evecs,
        denominator,
        jnp.conj(evecs),
        optimize=True,
    )


def _get_Gk_all_jax(evals_all, evecs_all, efermi, energy):
    """
    JAX version: Compute Green's function for all k-points at one energy.

    Parameters:
    -----------
    evals_all : jnp.ndarray, shape (nkpts, nbasis)
        Eigenvalues for all k-points
    evecs_all : jnp.ndarray, shape (nkpts, nbasis, nbasis)
        Eigenvectors for all k-points
    efermi : float
        Fermi energy
    energy : complex
        Energy point

    Returns:
    --------
    Gk_all : jnp.ndarray, shape (nkpts, nbasis, nbasis)
        Green's function for all k-points
    """
    _require_jax()
    vmap = _vmap

    # Vectorize over k-points
    def compute_Gk(evals, evecs):
        return _eigen_to_G_jax(evals, evecs, efermi, energy)

    Gk_all = vmap(compute_Gk)(evals_all, evecs_all)
    return Gk_all


# =============================================================================
# JAX JIT-compiled Functions for Performance
# =============================================================================


def _get_perturbed_single_angle_jax(G0K, Hsoc_k, theta, phi, kweights):
    """
    JAX version: Compute perturbed energy for a single angle across all k-points.

    This computes the second-order perturbation:
    dE = sum_k Tr(G0K @ dHi @ G0K @ dHi) * kweight
    where dHi = rotate_spinor_matrix(Hsoc_k, theta, phi)

    Parameters:
    -----------
    G0K : jnp.ndarray, shape (nkpts, nbasis, nbasis)
        Green's function for all k-points
    Hsoc_k : jnp.ndarray, shape (nkpts, nbasis, nbasis)
        SOC Hamiltonian for all k-points
    theta, phi : float
        Rotation angles in spherical coordinates
    kweights : jnp.ndarray, shape (nkpts,)
        Integration weights for k-points

    Returns:
    --------
    dE : complex
        Total energy perturbation for this angle
    dE_matrix : jnp.ndarray, shape (natoms, natoms)
        Atom-atom resolved energy contributions
    """
    _require_jax()
    jnp = _jnp
    vmap = _vmap

    # Rotate SOC Hamiltonian for all k-points
    def rotate_and_compute(Gk, Hk, kw):
        dHi = _rotate_spinor_matrix_einsum_jax(Hk, theta, phi)
        GdH = Gk @ dHi
        dG2 = GdH * GdH.T  # Element-wise product, then transpose
        dG2sum = jnp.sum(dG2)
        return dG2sum * kw, dG2

    # Vectorize over k-points
    results = vmap(rotate_and_compute)(G0K, Hsoc_k, kweights)
    dE_total = jnp.sum(results[0])
    dG2_all = results[1]  # shape (nkpts, nbasis, nbasis)

    return dE_total, dG2_all


def _get_perturbed_all_angles_jax(G0K, Hsoc_k, thetas, phis, kweights):
    """
    JAX version: Compute perturbed energy for all angles using vmap.

    Parameters:
    -----------
    G0K : jnp.ndarray, shape (nkpts, nbasis, nbasis)
    Hsoc_k : jnp.ndarray, shape (nkpts, nbasis, nbasis)
    thetas, phis : jnp.ndarray, shape (nangles,)
    kweights : jnp.ndarray, shape (nkpts,)

    Returns:
    --------
    dE_angles : jnp.ndarray, shape (nangles,)
    dG2_angles : jnp.ndarray, shape (nangles, nkpts, nbasis, nbasis)
    """
    _require_jax()
    vmap = _vmap

    def compute_for_angle(theta, phi):
        return _get_perturbed_single_angle_jax(G0K, Hsoc_k, theta, phi, kweights)

    results = vmap(compute_for_angle)(thetas, phis)
    return results[0], results[1]


# =============================================================================
# MAEGreenGPU Class
# =============================================================================


class MAEGreenGPU(MAEGreen):
    """
    GPU-accelerated version of MAEGreen using JAX.

    This class provides the same API as MAEGreen but uses JAX for
    GPU-accelerated matrix computations. All heavy matrix operations
    (Green's function calculation, spinor rotations, and perturbation
    theory calculations) are performed on the GPU.

    Parameters:
    -----------
    angles : str or list, optional
        Angle preset or custom [thetas, phis] arrays
    **kwargs : dict
        Additional arguments passed to ExchangeNCL

    Examples:
    ---------
    >>> from TB2J.MAEGreenGPU import MAEGreenGPU
    >>> mae = MAEGreenGPU(
    ...     tbmodels=model,
    ...     atoms=model.atoms,
    ...     kmesh=[6, 6, 1],
    ...     angles="fib",
    ... )
    >>> mae.run(output_path="my_mae_results")
    """

    def __init__(self, angles=None, **kwargs):
        """Initialize MAEGreenGPU."""
        # Check JAX availability on initialization
        _require_jax()

        # Call parent initialization
        super().__init__(angles=angles, **kwargs)

        # Store JAX module references
        self._jnp = _jnp
        self._jax = _jax

    def _numpy_to_jax(self, arr):
        """Convert numpy array to JAX array."""
        return self._jnp.array(arr)

    def _jax_to_numpy(self, arr):
        """Convert JAX array to numpy array."""
        return np.array(arr)

    def get_perturbed_gpu(self, e, thetas, phis):
        """
        GPU-accelerated version of get_perturbed using JAX.

        Computes the second-order perturbation energy for all angles
        simultaneously on the GPU using JAX's vmap.

        Parameters:
        -----------
        e : complex
            Energy point on the contour
        thetas, phis : array-like
            Arrays of rotation angles

        Returns:
        --------
        dE_angle : np.ndarray
            Energy perturbations for each angle
        dE_angle_matrix : np.ndarray
            Atom-atom resolved contributions
        dE_angle_atom_orb : dict
            Orbital-resolved contributions
        """
        # Ensure SOC is turned off for the non-perturbed Hamiltonian
        self.tbmodel.set_so_strength(0.0)

        # Get Green's function and SOC Hamiltonian
        # These are computed on CPU then transferred to GPU
        G0K_np = self.G.get_Gk_all(e)
        Hsoc_k_np = self.tbmodel.get_Hk_soc(self.G.kpts)

        # Convert to JAX arrays and transfer to GPU
        G0K = self._numpy_to_jax(G0K_np)
        Hsoc_k = self._numpy_to_jax(Hsoc_k_np)
        kweights = self._numpy_to_jax(self.G.kweights)
        thetas_jax = self._numpy_to_jax(thetas)
        phis_jax = self._numpy_to_jax(phis)

        na = len(thetas)

        # Compute for all angles simultaneously on GPU
        dE_angles, dG2_all = _get_perturbed_all_angles_jax(
            G0K, Hsoc_k, thetas_jax, phis_jax, kweights
        )

        # Convert results back to numpy
        dE_angle = self._jax_to_numpy(dE_angles)

        # Compute atom-atom matrix and orbital-resolved contributions
        # These are done on CPU as they involve complex indexing
        dE_angle_matrix = np.zeros((na, self.natoms, self.natoms), dtype=complex)
        dE_angle_atom_orb = DefaultDict(lambda: 0)

        dG2_np = self._jax_to_numpy(dG2_all)

        for iangle in range(na):
            for ik in range(len(self.G.kpts)):
                dG2 = dG2_np[iangle, ik]
                kw = self.G.kweights[ik]

                # Calculate atom-atom matrix interactions
                for iatom in range(self.natoms):
                    iorb = self.iorb(iatom)
                    for jatom in range(self.natoms):
                        jorb = self.iorb(jatom)
                        # Calculate cross terms between atoms i and j
                        dE_ij_orb = dG2[np.ix_(iorb, jorb)] * kw
                        dE_ij_orb = (
                            dE_ij_orb[::2, ::2]
                            + dE_ij_orb[1::2, 1::2]
                            + dE_ij_orb[1::2, ::2]
                            + dE_ij_orb[::2, 1::2]
                        )
                        dE_ij = np.sum(dE_ij_orb)
                        # Transform to local orbital basis
                        mmat_i = self.mmats[iatom]
                        mmat_j = self.mmats[jatom]
                        dE_ij_orb = mmat_i.T @ dE_ij_orb @ mmat_j
                        dE_angle_matrix[iangle, iatom, jatom] += dE_ij
                        # Store orbital-resolved data for diagonal terms
                        if iatom == jatom:
                            dE_angle_atom_orb[(iangle, iatom)] += dE_ij_orb

        return dE_angle, dE_angle_matrix, dE_angle_atom_orb

    def get_band_energy_vs_angles(self, thetas, phis, with_eigen=False, use_gpu=True):
        """
        Calculate band energy vs angles using GPU acceleration.

        Parameters:
        -----------
        thetas, phis : array-like
            Arrays of rotation angles
        with_eigen : bool
            Whether to also compute using eigenvalue method
        use_gpu : bool
            Whether to use GPU acceleration (default: True)

        Returns:
        --------
        None (results stored in self.es, self.es_matrix, self.es_atom_orb)
        """
        if with_eigen:
            self.es2 = self.get_band_energy_vs_angles_from_eigen(thetas, phis)

        # Use GPU-accelerated version if requested
        if use_gpu:
            self._get_band_energy_vs_angles_gpu(thetas, phis)
        else:
            # Fall back to parent implementation
            super().get_band_energy_vs_angles(thetas, phis, with_eigen=False)

    def _get_band_energy_vs_angles_gpu(self, thetas, phis):
        """
        Internal GPU-accelerated implementation.
        """

        # Process energy contour points
        def func(e):
            return self.get_perturbed_gpu(e, thetas, phis)

        if self.nproc > 1:
            results = p_map(func, self.contour.path, num_cpus=self.nproc)
        else:
            npole = len(self.contour.path)
            results = map(func, tqdm.tqdm(self.contour.path, total=npole))

        for i, result in enumerate(results):
            dE_angle, dE_angle_matrix, dE_angle_atom_orb = result
            self.es += dE_angle * self.contour.weights[i]
            self.es_matrix += dE_angle_matrix * self.contour.weights[i]
            for key, value in dE_angle_atom_orb.items():
                self.es_atom_orb[key] += (
                    dE_angle_atom_orb[key] * self.contour.weights[i]
                )

        # Apply final normalization
        self.es = -np.imag(self.es) / (2 * np.pi)
        self.es_matrix = -np.imag(self.es_matrix) / (2 * np.pi)
        for key, value in self.es_atom_orb.items():
            self.es_atom_orb[key] = -np.imag(value) / (2 * np.pi)

    def run(self, output_path="TB2J_anisotropy", with_eigen=False, use_gpu=True):
        """
        Run the MAE calculation with optional GPU acceleration.

        Parameters:
        -----------
        output_path : str
            Directory for output files
        with_eigen : bool
            Whether to also compute using eigenvalue method
        use_gpu : bool
            Whether to use GPU acceleration (default: True)
        """
        self.get_band_energy_vs_angles(
            self.thetas, self.phis, with_eigen=with_eigen, use_gpu=use_gpu
        )
        self.output(output_path=output_path, with_eigen=with_eigen)


# =============================================================================
# Convenience Functions (Same API as MAEGreen)
# =============================================================================


def abacus_get_MAE(
    path_nosoc,
    path_soc,
    kmesh,
    thetas,
    phis,
    gamma=True,
    output_path="TB2J_anisotropy",
    nel=None,
    width=0.1,
    with_eigen=False,
    use_gpu=True,
    **kwargs,
):
    """
    Get MAE from Abacus with GPU acceleration using JAX.

    This function provides the same API as TB2J.MAEGreen.abacus_get_MAE
    but uses GPU acceleration for the heavy matrix computations.

    Parameters:
    -----------
    path_nosoc : str
        Path to calculation without SOC
    path_soc : str
        Path to calculation with SOC
    kmesh : list
        K-point mesh [nx, ny, nz]
    thetas, phis : array-like
        Arrays of rotation angles
    gamma : bool
        Use Gamma-centered k-points
    output_path : str
        Directory for output files
    nel : int, optional
        Number of electrons
    width : float
        Smearing width
    with_eigen : bool
        Also compute using eigenvalue method
    use_gpu : bool
        Use GPU acceleration (default: True)
    **kwargs : dict
        Additional arguments for MAEGreenGPU

    Returns:
    --------
    mae : MAEGreenGPU
        The MAEGreenGPU instance with computed results
    """
    from HamiltonIO.abacus.abacus_wrapper import AbacusSplitSOCParser

    parser = AbacusSplitSOCParser(
        outpath_nosoc=path_nosoc, outpath_soc=path_soc, binary=False
    )
    model = parser.parse()
    model.set_so_strength(0.0)
    if nel is not None:
        model.nel = nel

    mae = MAEGreenGPU(
        tbmodels=model,
        atoms=model.atoms,
        kmesh=kmesh,
        efermi=None,
        basis=model.basis,
        angles=[thetas, phis],
        **kwargs,
    )
    mae.run(output_path=output_path, with_eigen=with_eigen, use_gpu=use_gpu)
    return mae


# Keep compatibility with old naming
MAEGreenJAX = MAEGreenGPU
