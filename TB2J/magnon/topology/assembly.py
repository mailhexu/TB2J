"""Classical assembly: stationarity from actual physical bonds.

The constrained transverse gradient of the classical energy

    E = - sum_{ijR} S_i . J_ij(R) . S_j
      = - sum_{ijR} S_i . J'_ij(R) . S_j   (J' with the single-Q rotation)

is evaluated from the retained bonds, on-site terms, moments and reference of
the model *before* any matrix solve, providing the assembly-owned stationarity
diagnostic required by architecture section 4. A quadratic matrix alone can
never certify stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "StationarityDiagnostics",
    "classical_transverse_gradient",
    "local_frame_bosonic_hamiltonian",
]


@dataclass
class StationarityDiagnostics:
    """Constrained transverse classical gradient at the reference state."""

    transverse_field: np.ndarray  # (nspin, 3) P_i B_i
    field_scale: float  # max |B_i| over spins
    max_transverse: float  # max |P_i B_i|
    tolerance: float  # scale-aware stationarity tolerance
    stationary: bool

    def as_dict(self) -> dict:
        return {
            "field_scale": self.field_scale,
            "max_transverse": self.max_transverse,
            "tolerance": self.tolerance,
            "stationary": self.stationary,
        }


def classical_transverse_gradient(
    JR: np.ndarray,
    Rlist: np.ndarray,
    magmoms: np.ndarray,
    Q: np.ndarray = (0.0, 0.0, 0.0),
    n: np.ndarray = (1.0, 0.0, 0.0),
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> StationarityDiagnostics:
    """Transverse classical field at each spin for the given reference.

    ``B_i = sum_{j,R} Jbar_ij(R) R_n(2 pi R.Q) mu_j`` with ``mu`` the
    magnetic moments (muB), ``s_i = |mu_i|/2`` and
    ``Jbar_ij = J_ij/(s_i s_j)``, matching the legacy magnon Hamiltonian.
    The constrained gradient is ``P_i B_i`` with
    ``P_i = 1 - zhat_i zhat_i^T``; a stationary reference requires
    ``max|P_i B_i| <= max(atol, rtol * max|B|)``.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    spin = np.linalg.norm(magmoms, axis=1) / 2.0
    JR = np.asarray(JR, dtype=float) / (
        spin[None, :, None, None, None] * spin[None, None, :, None, None]
    )
    Q = np.asarray(Q, dtype=float)
    n = np.asarray(n, dtype=float)
    nspin = magmoms.shape[0]
    B = np.zeros((nspin, 3), dtype=float)
    for iR, R in enumerate(Rlist):
        J = JR[iR]
        if np.any(Q != 0):
            phi = 2 * np.pi * np.asarray(R, dtype=float) @ Q
            rot = Rotation.from_rotvec(phi * n).as_matrix()
        else:
            rot = np.eye(3)
        B += np.einsum("ijxy,jy->ix", J, magmoms @ rot.T)
    norms = np.linalg.norm(magmoms, axis=1)
    zhat = magmoms / norms[:, None]
    B_parallel = np.einsum("i,ij->ij", np.einsum("ij,ij->i", B, zhat), zhat)
    transverse = B - B_parallel
    field_scale = float(np.max(np.linalg.norm(B, axis=1)))
    max_transverse = float(np.max(np.linalg.norm(transverse, axis=1)))
    tolerance = max(atol, rtol * field_scale)
    return StationarityDiagnostics(
        transverse_field=transverse,
        field_scale=field_scale,
        max_transverse=max_transverse,
        tolerance=tolerance,
        stationary=bool(max_transverse <= tolerance),
    )


def _local_frames(
    magmoms: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right-handed local frames with ``e3`` along each moment."""
    magmoms = np.asarray(magmoms, dtype=float)
    norms = np.linalg.norm(magmoms, axis=1)
    if np.any(norms <= 0):
        raise ValueError("magmoms must have nonzero length for local frames.")
    e3 = magmoms / norms[:, None]
    u = np.asarray(u, dtype=float)
    u_norm = np.linalg.norm(u)
    if u.shape != (3,) or not np.isfinite(u).all() or u_norm <= 0:
        raise ValueError("u must be a finite nonzero three-component vector.")
    u = u / u_norm
    e1 = np.cross(u, e3)
    for i in range(e1.shape[0]):
        norm = np.linalg.norm(e1[i])
        if norm < 1e-12:
            axis = np.eye(3)[np.argmin(np.abs(e3[i]))]
            e1[i] = axis - np.dot(axis, e3[i]) * e3[i]
            norm = np.linalg.norm(e1[i])
        e1[i] /= norm
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def local_frame_bosonic_hamiltonian(
    JR: np.ndarray,
    Rlist: np.ndarray,
    magmoms: np.ndarray,
    kpoints: np.ndarray,
    u: np.ndarray = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Magnon Hamiltonian from site-local Holstein–Primakoff frames.

    ``e3_i = mu_i / |mu_i|`` is the classical moment direction.
    ``e1_i = normalize(u x e3_i)`` and ``e2_i = e3_i x e1_i`` define a
    right-handed transverse basis; the fallback for moments parallel to ``u``
    chooses any stable orthonormal transverse pair. ``u`` therefore fixes only
    the local-frame gauge, not the quantization axis.

    The returned Colpa matrix is ``[[A(k), B(k)], [B^dagger(k), A(-k)^*]]``.
    For a Hermitian Fourier block, ``A(-k)^* = A^T(-k)``. The hole block is
    accumulated from the conjugated real-space number-conserving coefficients;
    it is not generally ``A(k)^*``. Let ``M_i = |mu_i|``,
    ``s_i = M_i/2``, and ``Jbar_ij(R) = J_ij(R)/(s_i s_j)`` as in the
    legacy magnon Hamiltonian. With ``g_ab = e_a(i) . Jbar_ij(R) . e_b(j)``,
    the transverse blocks are ``A_ij(R) = -sqrt(M_i M_j)/2 *
    [(g00+g11)-i(g01-g10)]`` and ``B_ij(R) = -sqrt(M_i M_j)/2 *
    [(g00-g11)+i(g01+g10)]``; longitudinal normalization contributes
    ``A_ii += sum_jR M_j e3_i . Jbar_ij(R) . e3_j``.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    JR = np.asarray(JR, dtype=float)
    Rlist = np.asarray(Rlist, dtype=float)
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    nspin = magmoms.shape[0]
    e1, e2, e3 = _local_frames(magmoms, np.asarray(u, dtype=float))
    M = np.linalg.norm(magmoms, axis=1)
    spin = M / 2.0
    JR = JR / (spin[None, :, None, None, None] * spin[None, None, :, None, None])

    nk = kpoints.shape[0]
    phases = np.exp(-2j * np.pi * kpoints @ Rlist.T)  # (nk, nR)
    A = np.zeros((nk, nspin, nspin), dtype=complex)
    A_hole = np.zeros((nk, nspin, nspin), dtype=complex)
    B = np.zeros((nk, nspin, nspin), dtype=complex)
    stiff = np.zeros((nspin,), dtype=float)
    for iR in range(Rlist.shape[0]):
        J = JR[iR]
        frame12 = np.stack([e1, e2], axis=1)  # (nspin, 2, 3)
        g = np.einsum("iax,ijxy,jby->ijab", frame12, J, frame12)
        g33 = np.einsum("ix,ijxy,jy->ij", e3, J, e3)
        s = np.sqrt(M)[:, None] * np.sqrt(M)[None, :]
        a_contrib = (
            -0.5
            * s
            * ((g[:, :, 0, 0] + g[:, :, 1, 1]) - 1j * (g[:, :, 0, 1] - g[:, :, 1, 0]))
        )
        b_contrib = (
            -0.5
            * s
            * ((g[:, :, 0, 0] - g[:, :, 1, 1]) + 1j * (g[:, :, 0, 1] + g[:, :, 1, 0]))
        )
        ph = phases[:, iR][:, None, None]
        A += ph * a_contrib[None, :, :]
        A_hole += ph * a_contrib.conj()[None, :, :]
        B += ph * b_contrib[None, :, :]
        # transverse stiffness from the normalization (tr) terms
        for i in range(nspin):
            for j in range(nspin):
                if i != j:
                    stiff[i] += M[j] * g33[i, j]
                else:
                    stiff[i] += M[i] * g33[i, i]
    A += np.diag(stiff)[None, :, :]
    A_hole += np.diag(stiff)[None, :, :]

    H = np.empty((nk, 2 * nspin, 2 * nspin), dtype=complex)
    H[:, :nspin, :nspin] = A
    H[:, :nspin, nspin:] = B
    H[:, nspin:, :nspin] = B.conj().transpose(0, 2, 1)
    H[:, nspin:, nspin:] = A_hole
    return H
