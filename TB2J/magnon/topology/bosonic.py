"""Bosonic mode solver with metric normalization and stability diagnostics.

The solver consumes the assembled Hermitian bosonic Hamiltonian ``H`` (the
legacy ``Magnon.Hq`` doubled form ``[[A1-C, B], [B^dag, A2-C]]``) and returns
physical mode energies, metric-normalized mode vectors when a complete
normalized set exists, and the diagnostics required by the validity policy
(architecture section 4). It never supplies a regularized mode set silently:
any necessary diagonal shift is refused, never applied.

Certified Colpa identities (``docs/sympy/magnon_topology_lswt_ferromagnet.py``):
with ``H = L L^dag`` (lower Cholesky), ``M = L^dag g L`` eigendecomposition
``(omega_all, u)`` has the physical energies in the ascending top half
``omega = omega_all[n:]`` (the legacy ``Magnon._magnon_energies`` convention,
without its silent shifts); ``v = L^{-dag} u`` obeys ``H v = omega g v`` with
``v^dag g v = 1/omega``; ``sqrt(omega)`` rescaling yields unit metric norm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "BosonicModes",
    "ModeSolverError",
    "solve_bosonic_modes",
]


class ModeSolverError(ValueError):
    """Raised when the mode solution is physically unusable."""


@dataclass
class BosonicModes:
    """Result of the bosonic mode solve for one momentum."""

    energies: np.ndarray  # physical mode energies, ascending (n,)
    classification: str  # positive_definite | zero_mode | energetically_unstable | dynamically_unstable
    vectors: Optional[np.ndarray] = None  # (2n, n_pos) metric-normalized positive modes
    positive_indices: Optional[np.ndarray] = (
        None  # indices of the positive modes within ``energies``
    )
    null_space: Optional[np.ndarray] = None  # (2n, n_zero) non-normalizable null modes
    h_min_eigenvalue: float = 0.0  # min eigenvalue of the Hessian H itself
    min_energy: float = 0.0  # min physical energy (top-half metric eigenvalue)
    residuals: dict = field(default_factory=dict)


def solve_bosonic_modes(
    H: np.ndarray,
    hermiticity_tol: float = 1e-10,
    zero_tol: float = 1e-8,
) -> BosonicModes:
    """Solve the bosonic eigenproblem ``H v = omega g v`` for Hermitian ``H``.

    Parameters
    ----------
    H : (2n, 2n) complex Hermitian array in the legacy ``Magnon.Hq`` form.
    hermiticity_tol : absolute tolerance on ``max|H - H^dag|`` and on the
        +-pairing residual of the metric spectrum.
    zero_tol : absolute energy tolerance separating a stable zero mode from
        resolved positive energies (eV).

    Returns
    -------
    BosonicModes with ``classification``:
      - ``positive_definite``: stable gapped set; complete metric-normalized
        positive modes attached.
      - ``zero_mode``: a stable zero-frequency direction exists (Goldstone
        or accidental touching); ``null_space`` attached and no normalized
        mode vectors (null modes are not normalizable bosons). Ticket 02
        refuses validated invariants on such meshes.
      - ``energetically_unstable``: the Hessian has a resolved negative
        direction; no diagonal shift is applied and no vectors are provided.
      - ``dynamically_unstable``: the metric spectrum violates the +-pairing
        structure expected of a bosonic BdG set; no vectors are provided.
    """
    H = np.asarray(H, dtype=complex)
    n2 = H.shape[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1] or n2 % 2:
        raise ModeSolverError(
            f"Hamiltonian must be square with even size, got shape {H.shape}."
        )
    asym = float(np.max(np.abs(H - H.conj().T)))
    if asym > hermiticity_tol:
        raise ModeSolverError(
            f"Hamiltonian is not Hermitian within tolerance: max|H-H^dag| = {asym:.3e}."
        )
    n = n2 // 2
    h_min = float(np.linalg.eigvalsh(H)[0])
    g = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))

    if h_min < -zero_tol:
        # frequency character of the dynamical matrix gH (diagnostics only;
        # no diagonal shift is applied and no vectors are provided)
        g_diag = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))
        freq = np.linalg.eigvals(g_diag @ H)
        scale = max(float(np.max(np.abs(freq))), 1.0)
        freq_character = (
            "real"
            if float(np.max(np.abs(freq.imag))) <= hermiticity_tol * scale
            else "complex"
        )
        return BosonicModes(
            energies=np.array([]),
            classification="energetically_unstable",
            h_min_eigenvalue=h_min,
            residuals={
                "hermiticity_asymmetry": asym,
                "frequency_character": freq_character,
                "max_imag_frequency": float(np.max(np.abs(freq.imag))),
                "note": "Hessian has a resolved negative direction; a diagonal "
                "shift is refused because it would not legitimize topology.",
            },
        )
    hessian_null_space = None
    try:
        L = np.linalg.cholesky(H)  # lower, H = L L^dag
    except np.linalg.LinAlgError:
        if h_min < -zero_tol:
            raise ModeSolverError(
                f"Cholesky failed with negative Hessian minimum {h_min:.3e}."
            ) from None
        # Exactly semidefinite H: build its psd square root without a shift.
        w_h, U_h = np.linalg.eigh(H)
        null_h = np.abs(w_h) <= zero_tol
        hessian_null_space = U_h[:, null_h]
        w_clamped = np.where(null_h, 0.0, w_h)
        L = (U_h * np.sqrt(w_clamped)[None, :]) @ U_h.conj().T
        if np.max(np.abs(L @ L.conj().T - H)) > zero_tol:
            raise ModeSolverError(
                f"Hessian not decomposable within tolerance (min {h_min:.3e})."
            ) from None
    M = L.conj().T @ g @ L
    omega_all, u = np.linalg.eigh(M)
    omega = omega_all[n:]  # physical top half (legacy [:, n:] convention)
    pairing_residual = float(np.max(np.abs(omega_all[:n] + omega_all[n:][::-1])))
    scale = max(float(np.max(np.abs(omega_all))), 1.0)
    if pairing_residual > hermiticity_tol * scale:
        return BosonicModes(
            energies=omega,
            classification="dynamically_unstable",
            h_min_eigenvalue=h_min,
            min_energy=float(omega[0]),
            residuals={
                "hermiticity_asymmetry": asym,
                "pairing_residual": pairing_residual,
            },
        )

    pos_mask = omega > zero_tol
    zero_mask = ~pos_mask
    null_space = None
    null_resid = 0.0
    if np.any(zero_mask):
        # Null vectors of a semidefinite Hessian cannot be recovered by
        # solving through its singular square root.
        if hessian_null_space is not None and hessian_null_space.shape[1]:
            null_space = hessian_null_space
        else:
            null_space = np.linalg.solve(L.conj().T, u[:, n:][:, zero_mask])
        null_resid = float(np.max(np.abs(H @ null_space))) if null_space.size else 0.0
    if not np.any(pos_mask):
        return BosonicModes(
            energies=omega,
            classification="zero_mode",
            h_min_eigenvalue=h_min,
            min_energy=float(omega[0]),
            residuals={
                "hermiticity_asymmetry": asym,
                "pairing_residual": pairing_residual,
                "null_space_residual": null_resid,
                "n_zero_modes": int(np.sum(zero_mask)),
            },
            null_space=null_space,
        )
    pos_u = u[:, n:][:, pos_mask]
    pos = omega[pos_mask]
    v = np.linalg.solve(L.conj().T, pos_u)  # v = L^{-dag} u for positive modes
    norm = np.real(np.sum(v.conj() * (g @ v), axis=0))
    if np.any(norm <= 0):
        raise ModeSolverError("Nonpositive metric norm in positive subspace.")
    vectors = v * np.sqrt(pos)[None, :]
    resid = H @ vectors - (g @ vectors) * pos[None, :]
    return BosonicModes(
        energies=omega,
        classification="zero_mode" if null_space is not None else "positive_definite",
        vectors=vectors,
        positive_indices=np.flatnonzero(pos_mask),
        null_space=null_space,
        h_min_eigenvalue=h_min,
        min_energy=float(omega[0]),
        residuals={
            "hermiticity_asymmetry": asym,
            "pairing_residual": pairing_residual,
            "max_generalized_residual": float(np.max(np.abs(resid))),
            "max_metric_norm_error": float(np.max(np.abs(pos * norm - 1.0))),
            "null_space_residual": null_resid,
            "n_zero_modes": int(np.sum(zero_mask)),
        },
    )
