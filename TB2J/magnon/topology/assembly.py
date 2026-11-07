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

    ``B_i = sum_{j,R} J_ij(R) R_n(2 pi R.Q) mu_j`` with ``mu`` the magnetic
    moments (muB) and the executable single-Q right-rotation convention of
    ``Magnon.Jq``. The constrained gradient is ``P_i B_i`` with
    ``P_i = 1 - zhat_i zhat_i^T``; a stationary reference requires
    ``max|P_i B_i| <= max(atol, rtol * max|B|)``.
    """
    magmoms = np.asarray(magmoms, dtype=float)
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
    """Global-axis spiral-gauge frames: e3 = u, e1 = mu_hat, e2 = u x e1.

    The fluctuation basis of site ``s`` is ``(e1_s, e2_s) = (mu_hat_s,
    normalize(cross(u, mu_hat_s)))`` with the common quantization axis
    ``e3 = u``; this matches the certificate convention of
    ``docs/sympy/magnon_topology_magnetic_cell.py``. Sites with ``mu``
    parallel to ``u`` fall back to ``e1`` = the in-plane x direction
    (y when degenerate), i.e. the standard collinear frame.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    norms = np.linalg.norm(magmoms, axis=1)
    if np.any(norms <= 0):
        raise ValueError("magmoms must have nonzero length for local frames.")
    e1 = magmoms / norms[:, None]
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    e2 = np.cross(u, e1)
    nnorm = np.linalg.norm(e2, axis=1)
    for i in range(e1.shape[0]):
        if nnorm[i] < 1e-12:  # mu parallel to u: standard collinear frame
            axis = np.array([1.0, 0.0, 0.0])
            if abs(e1[i] @ axis) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            v = axis - (axis @ e1[i]) * e1[i]
            e1[i] = v / np.linalg.norm(v)
            e2[i] = np.cross(u, e1[i])
        else:
            e2[i] = e2[i] / nnorm[i]
    return e1, e2, np.broadcast_to(u, e1.shape)


def local_frame_bosonic_hamiltonian(
    JR: np.ndarray,
    Rlist: np.ndarray,
    magmoms: np.ndarray,
    kpoints: np.ndarray,
    u: np.ndarray = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Magnon Hamiltonian for noncollinear references (spiral-gauge frames).

    Linear spin-wave expansion of ``E = -sum_ijR S_i . J_ij(R) . S_j`` about
    the classical reference ``magmoms``, with Holstein-Primakoff bosons
    quantized along the common axis ``u`` and fluctuation bases
    ``(e1_i, e2_i) = (mu_hat_i, normalize(u x mu_hat_i))``. Returns the
    Colpa block matrix

        H = [[A(k), B(k)], [B(k)^dagger, A(k)^*]]

    with shape ``(nk, 2*nspin, 2*nspin)``, consumable by
    :func:`TB2J.magnon.topology.bosonic.solve_bosonic_modes`.

    Convention (certificate ``magnon_topology_magnetic_cell``): with moment
    magnitude ``S_i = |mu_i|``,
    ``A_ij(R) = -sqrt(S_i S_j) / 2 * [(g00 + g11) - i (g01 - g10)]`` and
    ``B_ij(R) = -sqrt(S_i S_j) / 2 * [(g00 - g11) + i (g01 + g10)]`` where
    ``g_ab = e_a(i) . J_ij(R) . e_b(j)``; the transverse stiffness enters the
    diagonal as ``A_ii += sum_l S_l e3 . J_il . e3``. In the collinear limit
    this reproduces the legacy ``Magnon.Hq`` energies.
    """
    magmoms = np.asarray(magmoms, dtype=float)
    JR = np.asarray(JR, dtype=float)
    Rlist = np.asarray(Rlist, dtype=float)
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    nspin = magmoms.shape[0]
    e1, e2, e3 = _local_frames(magmoms, np.asarray(u, dtype=float))
    S = np.linalg.norm(magmoms, axis=1)  # full moment magnitude (muB)

    nk = kpoints.shape[0]
    phases = np.exp(-2j * np.pi * kpoints @ Rlist.T)  # (nk, nR)
    A = np.zeros((nk, nspin, nspin), dtype=complex)
    B = np.zeros((nk, nspin, nspin), dtype=complex)
    stiff = np.zeros((nspin,), dtype=float)
    for iR in range(Rlist.shape[0]):
        J = JR[iR]
        frame12 = np.stack([e1, e2], axis=1)  # (nspin, 2, 3)
        g = np.einsum("iax,ijxy,jby->ijab", frame12, J, frame12)
        g33 = np.einsum("ix,ijxy,jy->ij", e3, J, e3)
        s = np.sqrt(S)[:, None] * np.sqrt(S)[None, :]
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
        B += ph * b_contrib[None, :, :]
        # transverse stiffness from the normalization (tr) terms
        for i in range(nspin):
            for j in range(nspin):
                if i != j:
                    stiff[i] += S[j] * g33[i, j]
                else:
                    stiff[i] += S[i] * g33[i, i]
    A += np.diag(stiff)[None, :, :]

    H = np.empty((nk, 2 * nspin, 2 * nspin), dtype=complex)
    H[:, :nspin, :nspin] = A
    H[:, :nspin, nspin:] = B
    H[:, nspin:, :nspin] = B.conj()
    H[:, nspin:, nspin:] = A.conj()
    return H
