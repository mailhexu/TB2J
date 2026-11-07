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

__all__ = ["StationarityDiagnostics", "classical_transverse_gradient"]


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
