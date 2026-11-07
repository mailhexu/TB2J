"""Shared dimensional embedding of the magnetic cell.

The embedding declares an ordered integer real-space lattice basis (edge,
width, normal) in input-lattice fractional coordinates, plus the fixed normal
reciprocal momentum. It derives the reciprocal map rather than accepting
unrelated bulk and strip planes, and validates rank, orientation and
primitive coverage. Nonprimitive bases require an explicit supercell
expansion (ticket 05) and are refused here with actionable instructions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LatticeEmbedding"]


@dataclass(frozen=True)
class LatticeEmbedding:
    """Ordered integer real-space basis (edge, width, normal) rows.

    ``basis`` rows are integer lattice coordinates in the input cell basis.
    ``normal_k`` is the fixed normal reciprocal coordinate in the *embedding*
    basis (fractional, cycles).
    """

    basis: np.ndarray
    normal_k: float = 0.0

    def __post_init__(self):
        basis = np.asarray(self.basis, dtype=np.int64)
        if basis.shape != (3, 3):
            raise ValueError(
                f"Embedding basis must have shape (3, 3), got {basis.shape}."
            )
        if not np.all(np.isfinite(basis)):
            raise ValueError("Embedding basis must contain finite integers.")
        det = int(round(np.linalg.det(basis)))
        if det == 0:
            raise ValueError("Embedding basis is singular (rank < 3).")
        if abs(det) != 1:
            raise ValueError(
                f"Embedding basis is nonprimitive (|det| = {abs(det)}). Topology "
                "currently requires a primitive basis; a nonprimitive edge choice "
                "needs an explicit supercell expansion with site and bond "
                "remapping (not yet supported)."
            )
        object.__setattr__(self, "basis", basis)
        if not np.isfinite(self.normal_k):
            raise ValueError("normal_k must be finite.")

    @property
    def orientation_sign(self) -> int:
        """Sign of (edge, width, normal) handedness: +1 right-handed."""
        return int(round(np.linalg.det(self.basis)))

    def to_input_k(self, k_embed: np.ndarray) -> np.ndarray:
        """Map embedding-fractional momenta ``(ke, kw)`` or ``(ke, kw, kn)``.

        The normal component defaults to the fixed ``normal_k``. Returns
        input-cell fractional coordinates: ``k_in = k_embed @ basis`` with
        the normal component added at ``normal_k``.
        """
        k_embed = np.atleast_2d(np.asarray(k_embed, dtype=float))
        if k_embed.shape[1] == 2:
            kn = np.full((k_embed.shape[0], 1), self.normal_k)
            k_embed = np.concatenate([k_embed, kn], axis=1)
        elif k_embed.shape[1] != 3:
            raise ValueError("k_embed must have 2 or 3 components per row.")
        return k_embed @ self.basis


def identity_embedding() -> LatticeEmbedding:
    """Default embedding: input basis rows are (edge, width, normal)."""
    return LatticeEmbedding(basis=np.eye(3, dtype=np.int64))
