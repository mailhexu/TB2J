"""
JAX utilities for TB2J GPU-accelerated modules.
"""

import numpy as np

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

            # Enable float64 for numerical precision
            jax.config.update("jax_enable_x64", True)
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


def _is_gpu_available():
    """Check if JAX can find a GPU/TPU."""
    if _check_jax():
        try:
            # Check for any device that is not CPU
            devices = _jax.devices()
            for d in devices:
                if d.platform != "cpu":
                    return d.platform
        except Exception:
            return None
    return None


def _require_jax():
    """Raise ImportError if JAX is not available."""
    if not _check_jax():
        raise ImportError(
            "JAX is required for GPU accelerated version of TB2J but is not installed. "
            "Install it with: pip install jax jaxlib"
        )


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

    return jnp.stack([I, x, y, z], axis=-3)


def _pauli_block_all_sep_jax(M):
    """
    JAX version: Decompose a spinor matrix into Pauli components for separated basis.
    Returns [I, x, y, z] components.
    """
    _require_jax()
    jnp = _jnp

    n = M.shape[-1] // 2
    M_reshaped = M.reshape(*M.shape[:-2], 2 * n, 2 * n)

    # Extract blocks: [up1...upn, dn1...dnn]
    # M00 = uu, M01 = ud, M10 = du, M11 = dd
    M00 = M_reshaped[..., :n, :n]
    M01 = M_reshaped[..., :n, n:]
    M10 = M_reshaped[..., n:, :n]
    M11 = M_reshaped[..., n:, n:]

    # Pauli decomposition
    I = (M00 + M11) / 2.0
    x = (M01 + M10) / 2.0
    y = (M01 - M10) * (-1j / 2.0)
    z = (M00 - M11) / 2.0

    return jnp.stack([I, x, y, z], axis=-3)


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


def numpy_to_jax(arr):
    """Convert numpy array to JAX array."""
    _require_jax()
    if arr is None:
        return None
    return _jnp.array(arr)


def jax_to_numpy(arr):
    """Convert JAX array to numpy array."""
    if arr is None:
        return None
    return np.array(arr)


def _eigen_to_G_jax(evals, evecs, efermi, energy):
    """
    JAX version: Calculate Green's function from eigenvalues/eigenvectors.
    Supports multiple energy points and k-points via broadcasting.
    evals: (..., nb)
    evecs: (..., nb, nb)
    energy: (...,) or scalar
    """
    _require_jax()
    jnp = _jnp

    # energy + efermi - evals
    # We want 1 / ( (energy + efermi) * I - H )
    # H = evecs @ diag(evals) @ evecs.conj().T
    # G = evecs @ diag(1 / (energy + efermi - evals)) @ evecs.conj().T

    # evals shape: (nk, nb)
    # energy shape: (ne,)
    # Reshape for broadcasting: evals (1, nk, nb), energy (ne, 1, 1)

    # We use einsum for the matrix product
    # ...ib, ...b, ...jb -> ...ij
    denominator = 1.0 / (-evals + (energy + efermi))
    return jnp.einsum(
        "...ib, ...b, ...jb-> ...ij",
        evecs,
        denominator,
        jnp.conj(evecs),
        optimize="optimal",
    )


def _compute_GR_jax(Rpts, kpts, Gks, kweights, k2Rfactor):
    """
    JAX version: Fourier transform from G(k) to G(R).
    Gks: (ne, nk, nb, nb)
    kpts: (nk, 3)
    kweights: (nk,)
    Rpts: (nr, 3)
    """
    _require_jax()
    jnp = _jnp

    # phase = exp( k2Rfactor * R . k )
    phase = jnp.exp(k2Rfactor * jnp.einsum("ni,mi->nm", Rpts, kpts))
    phase *= kweights[None, :]  # (nr, nk)

    # GR = sum_k G(k) * phase(R, k)
    GR = jnp.einsum("ekij,rk->erij", Gks, phase, optimize="optimal")
    return GR


def _compute_dGR_jax(GR_all, Rmap, dV_all):
    """
    JAX version: Compute dG(R)/dx for all energies.
    GR_all: (ne, nR_total, nb, nb)
    Rmap: list of indices (id_Rm, id_Rnj, id_Rj)
    dV_all: (n_map, nb, nb)
    """
    _require_jax()
    jnp = _jnp

    # Select GRm and GRnj for all energies
    # Rmap indices: Rm_idx, Rnj_idx, Rj_idx
    # we want to sum over the mapping entries for each target Rj

    # Pre-select matrices
    # GRm: (ne, n_map, nb, nb)
    # GRnj: (ne, n_map, nb, nb)
    # dV: (n_map, nb, nb)

    indices_Rm = jnp.array([entry[0] for entry in Rmap])
    indices_Rnj = jnp.array([entry[1] for entry in Rmap])
    indices_Rj = jnp.array([entry[2] for entry in Rmap])

    GRm = GR_all[:, indices_Rm]  # (ne, n_map, nb, nb)
    GRnj = GR_all[:, indices_Rnj]  # (ne, n_map, nb, nb)

    # dG_map = GRm @ dV @ GRnj
    # ne, m, i, j   @   m, j, k   @   ne, m, k, l  -> ne, m, i, l
    dG_map = jnp.einsum("emij,mjk,emkl->emil", GRm, dV_all, GRnj)

    # Final accumulation: sum over m for each Rj
    # We can use jax.ops.segment_sum or just a loop if number of Rj is small
    # but a loop over target Rj is safer.

    unique_Rj = jnp.unique(indices_Rj)
    nRj = len(unique_Rj)
    nb = GR_all.shape[-1]
    ne = GR_all.shape[0]

    dGR_target = jnp.zeros((ne, nRj, nb, nb), dtype=GR_all.dtype)

    # Vectorized accumulation using segment_sum (JAX specific)
    # we need to reshape dG_map to (ne, m, nb*nb)
    dG_flat = dG_map.reshape(ne, -1, nb * nb)

    # JAX segment_sum usually works on the first axis.
    # We transpose to (m, ne, nb*nb)
    dG_flat_m = dG_flat.transpose(1, 0, 2)

    # segment_sum(data, segment_ids)
    # data: (m, ne*nb*nb), segment_ids: (m,)
    data = dG_flat_m.reshape(dG_flat_m.shape[0], -1)

    # We need to map unique_Rj back to 0..nRj-1 for segment_ids
    # Rj_id_map = {val: i for i, val in enumerate(unique_Rj)}
    # but unique_Rj is already sorted.
    # We can use jnp.searchsorted
    segment_ids = jnp.searchsorted(unique_Rj, indices_Rj)

    integrated_flat = jnp.zeros((nRj, data.shape[1]), dtype=data.dtype)
    integrated_flat = integrated_flat.at[segment_ids].add(data)

    # Reshape back to (nRj, ne, nb, nb) -> (ne, nRj, nb, nb)
    dGR_target = integrated_flat.reshape(nRj, ne, nb, nb).transpose(1, 0, 2, 3)

    return dGR_target, unique_Rj


def _prepare_HR_jax(tbmodel):
    """
    Prepare Hamiltonian H(R) and overlap S(R) data as JAX arrays.

    Parameters:
    -----------
    tbmodel : MyTB or SiestaHamiltonian
        Tight-binding model with HR/SR arrays or data attribute

    Returns:
    --------
    Rpts_jax : jnp.ndarray, shape (nR, 3)
        R vectors
    HR_jax : jnp.ndarray, shape (nR, nbasis, nbasis)
        Hamiltonian matrices for each R
    SR_jax : jnp.ndarray or None
        Overlap matrices for each R (None if orthogonal)
    R2kfactor : complex
        Phase factor (typically 2πi)
    """
    _require_jax()
    jnp = _jnp

    # Check for HR/SR arrays (SiestaHamiltonian style)
    if hasattr(tbmodel, "HR") and hasattr(tbmodel, "Rlist"):
        HR = tbmodel.HR
        Rpts = tbmodel.Rlist
        SR = tbmodel.SR if hasattr(tbmodel, "SR") else None
    elif hasattr(tbmodel, "data"):
        # MyTB style
        Rpts = np.array(list(tbmodel.data.keys()))
        HR = np.array([tbmodel.data[tuple(R)] for R in Rpts])
        SR = None
    else:
        raise ValueError("tbmodel must have either HR/Rlist or data attributes")

    Rpts_jax = jnp.array(Rpts)
    HR_jax = jnp.array(HR)
    SR_jax = jnp.array(SR) if SR is not None else None

    return Rpts_jax, HR_jax, SR_jax, tbmodel.R2kfactor


# Cached JIT-compiled functions (initialized on first use)
_compute_Hk_Sk_all_jax_cached = None
_compute_eigen_all_jax_cached = None
_prepare_eigen_pipeline_cached = None


def _compute_Hk_Sk_all_jax(Rpts, HR, SR, kpts, R2kfactor):
    """
    Compute H(k) and S(k) for all k-points on GPU.

    H(k) = sum_R H(R) * exp(2πi * k·R) + H(R)^† * exp(-2πi * k·R)

    Parameters:
    -----------
    Rpts : jnp.ndarray, shape (nR, 3)
        R vectors
    HR : jnp.ndarray, shape (nR, nbasis, nbasis)
        Hamiltonian matrices for each R
    SR : jnp.ndarray or None, shape (nR, nbasis, nbasis)
        Overlap matrices for each R
    kpts : jnp.ndarray, shape (nk, 3)
        k-points
    R2kfactor : complex
        Phase factor (typically 2πi)

    Returns:
    --------
    Hk_all : jnp.ndarray, shape (nk, nbasis, nbasis)
        Hamiltonian matrices for all k-points
    Sk_all : jnp.ndarray or None, shape (nk, nbasis, nbasis)
        Overlap matrices for all k-points
    """
    global _compute_Hk_Sk_all_jax_cached
    _require_jax()
    jnp = _jnp

    # Create JIT-compiled function on first call
    if _compute_Hk_Sk_all_jax_cached is None:

        @_jax.jit
        def _compute_Hk_Sk_impl(Rpts, HR, SR, kpts, R2kfactor):
            # Compute phase for all (k, R) pairs: phase[k, R] = exp(R2kfactor * k·R)
            k_dot_R = jnp.einsum("ki,ri->kr", kpts, Rpts)  # (nk, nR)
            phase = jnp.exp(R2kfactor * k_dot_R)  # (nk, nR)

            # H(k) = sum_R H(R) * phase[k, R]
            Hk = jnp.einsum("rij,kr->kij", HR, phase)

            # S(k) if non-orthogonal
            if SR is not None:
                Sk = jnp.einsum("rij,kr->kij", SR, phase)
            else:
                Sk = None

            return Hk, Sk

        _compute_Hk_Sk_all_jax_cached = _compute_Hk_Sk_impl

    return _compute_Hk_Sk_all_jax_cached(Rpts, HR, SR, kpts, R2kfactor)


def _eigh_standard_single(H):
    """Compute eigenvalues and eigenvectors for a single Hermitian matrix."""
    _require_jax()
    return _jnp.linalg.eigh(H)


def _eigh_generalized_cholesky(H, S):
    """
    Solve generalized eigenvalue problem H @ v = λ * S @ v using Cholesky decomposition.

    1. S = L @ L^H (Cholesky)
    2. H' = L^{-1} @ H @ L^{-H}
    3. Solve H' @ v' = λ * v' (standard eigenvalue problem)
    4. Transform back: v = L^{-H} @ v'
    """
    # Cholesky decomposition: S = L @ L^H
    L = _jnp.linalg.cholesky(S)

    # L^{-1}
    L_inv = _jnp.linalg.inv(L)

    # H' = L^{-1} @ H @ L^{-H}
    H_prime = L_inv @ H @ L_inv.conj().T

    # Standard eigenvalue problem
    evals, evecs_prime = _jnp.linalg.eigh(H_prime)

    # Transform eigenvectors back: v = L^{-H} @ v'
    evecs = L_inv.conj().T @ evecs_prime

    return evals, evecs


def _compute_eigen_all_jax(Hk_all, Sk_all):
    """
    Compute eigenvalues and eigenvectors for all k-points on GPU.

    Parameters:
    -----------
    Hk_all : jnp.ndarray, shape (nk, nbasis, nbasis)
        Hamiltonian matrices for all k-points
    Sk_all : jnp.ndarray or None, shape (nk, nbasis, nbasis)
        Overlap matrices for all k-points

    Returns:
    --------
    evals : jnp.ndarray, shape (nk, nbasis)
        Eigenvalues for all k-points
    evecs : jnp.ndarray, shape (nk, nbasis, nbasis)
        Eigenvectors for all k-points
    """
    global _compute_eigen_all_jax_cached
    _require_jax()

    # Create JIT-compiled function on first call
    if _compute_eigen_all_jax_cached is None:

        @_jax.jit
        def _compute_eigen_impl(Hk_all, Sk_all):
            if Sk_all is not None:
                # Generalized eigenvalue problem via Cholesky decomposition
                eigh_vmap = _jax.vmap(_eigh_generalized_cholesky)
                evals, evecs = eigh_vmap(Hk_all, Sk_all)
            else:
                # Standard eigenvalue problem
                eigh_vmap = _jax.vmap(_eigh_standard_single)
                evals, evecs = eigh_vmap(Hk_all)

            return evals, evecs

        _compute_eigen_all_jax_cached = _compute_eigen_impl

    return _compute_eigen_all_jax_cached(Hk_all, Sk_all)


def _prepare_eigen_pipeline_jax(Rpts, HR, SR, kpts, R2kfactor):
    """
    Combined JIT-compiled pipeline for eigenvalue preparation.
    Computes H(k), S(k) and their eigenvalues/eigenvectors in one compiled function.

    Parameters:
    -----------
    Rpts : jnp.ndarray, shape (nR, 3)
        R vectors
    HR : jnp.ndarray, shape (nR, nbasis, nbasis)
        Hamiltonian matrices for each R
    SR : jnp.ndarray or None, shape (nR, nbasis, nbasis)
        Overlap matrices for each R
    kpts : jnp.ndarray, shape (nk, 3)
        k-points
    R2kfactor : complex
        Phase factor (typically 2πi)

    Returns:
    --------
    evals : jnp.ndarray, shape (nk, nbasis)
        Eigenvalues for all k-points
    evecs : jnp.ndarray, shape (nk, nbasis, nbasis)
        Eigenvectors for all k-points
    Sk_all : jnp.ndarray or None
        Overlap matrices for all k-points
    """
    global _prepare_eigen_pipeline_cached
    _require_jax()
    jnp = _jnp

    if _prepare_eigen_pipeline_cached is None:

        def _eigh_generalized_single(H, S):
            """Generalized eigenvalue for single matrix using Cholesky."""
            L = jnp.linalg.cholesky(S)
            # Use solve_triangular for better efficiency
            L_inv = jnp.linalg.inv(L)
            H_prime = L_inv @ H @ L_inv.conj().T
            evals, evecs_prime = jnp.linalg.eigh(H_prime)
            evecs = L_inv.conj().T @ evecs_prime
            return evals, evecs

        def _eigh_standard_single_inner(H):
            """Standard eigenvalue for single matrix."""
            return jnp.linalg.eigh(H)

        @_jax.jit
        def _pipeline_impl(Rpts, HR, SR, kpts, R2kfactor):
            # Step 1: Compute H(k) and S(k) for all k-points
            k_dot_R = jnp.einsum("ki,ri->kr", kpts, Rpts)
            phase = jnp.exp(R2kfactor * k_dot_R)
            Hk = jnp.einsum("rij,kr->kij", HR, phase)

            if SR is not None:
                Sk = jnp.einsum("rij,kr->kij", SR, phase)
                # Step 2: Generalized eigenvalue problem
                eigh_vmap = _jax.vmap(_eigh_generalized_single)
                evals, evecs = eigh_vmap(Hk, Sk)
            else:
                Sk = None
                # Step 2: Standard eigenvalue problem
                eigh_vmap = _jax.vmap(_eigh_standard_single_inner)
                evals, evecs = eigh_vmap(Hk)

            return evals, evecs, Sk

        _prepare_eigen_pipeline_cached = _pipeline_impl

    return _prepare_eigen_pipeline_cached(Rpts, HR, SR, kpts, R2kfactor)


def _prepare_eigen_gpu(tbmodel, kpts):
    """
    GPU-accelerated eigenvalue/eigenvector preparation.

    Computes H(k), S(k) and their eigenvalues/eigenvectors for all k-points on GPU.

    Parameters:
    -----------
    tbmodel : MyTB or SiestaHamiltonian
        Tight-binding model
    kpts : np.ndarray, shape (nk, 3)
        k-points

    Returns:
    --------
    evals : np.ndarray, shape (nk, nbasis)
        Eigenvalues for all k-points
    evecs : np.ndarray, shape (nk, nbasis, nbasis)
        Eigenvectors for all k-points
    Sk_all : np.ndarray or None, shape (nk, nbasis, nbasis)
        Overlap matrices for all k-points (None if orthogonal)
    """
    _require_jax()
    jnp = _jnp

    # Prepare H(R) and S(R) data
    Rpts_jax, HR_jax, SR_jax, R2kfactor = _prepare_HR_jax(tbmodel)
    kpts_jax = jnp.array(kpts)

    print(
        f"GPU eigen preparation: {len(kpts)} k-points, {Rpts_jax.shape[0]} R-vectors, {HR_jax.shape[1]} basis functions"
    )
    print(f"  Non-orthogonal: {SR_jax is not None}")

    # Use the combined pipeline for better performance
    evals, evecs, Sk_all = _prepare_eigen_pipeline_jax(
        Rpts_jax, HR_jax, SR_jax, kpts_jax, R2kfactor
    )

    # Block until computation is done
    evals.block_until_ready()

    # Convert back to numpy
    Sk_np = np.array(Sk_all) if Sk_all is not None else None
    return np.array(evals), np.array(evecs), Sk_np
