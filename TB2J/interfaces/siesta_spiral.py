"""Versioned SIESTA spiral importer (story 002, epic 001).

Assembles the rotating-frame generalized pencil (H(k), S(k)) from stored
sparse 4-component blocks exactly per the conventions pinned by
``docs/sympy/siesta_spiral_pencil.py`` (story 001):

* spiral phase ``q/2`` per spin sector (``diag2kspiral.F90`` line 141,
  ``qwh = qw*0.5``): up sector ``e^{-i(k+q/2)x}``, down sector
  ``e^{-i(k-q/2)x}``, applied to **both** H and S blocks;
* spinor packing ``H1=H11, H2=H22, H3=Re H12, H4=-Im H12`` with no
  hermitization on the Hamiltonian path;
* the assembled overlap is spin-dependent; every pencil computation must
  use it, never a scalar lab S.

The bundle records its conventions in a versioned header; the importer
fails closed (``SpiralImportError``) on any unknown or inconsistent
convention rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from TB2J.interfaces.siesta_spiral_hubbard import DftuProvenance

BUNDLE_VERSION = "1.1"

_SUPPORTED_SPIN_PACKINGS = frozenset({"H4=-ImH12"})
_SUPPORTED_PHASE_CONVENTIONS = frozenset({"q_half_per_spin_sector"})
_SUPPORTED_HERMITIZATIONS = frozenset({"none_hamiltonian_path"})
_SUPPORTED_FERMI_SHIFTS = frozenset({"none"})
_SUPPORTED_PITCH_SIGNS = frozenset({"qphs_up_exp_minus"})


class SpiralImportError(RuntimeError):
    """Raised when a bundle's conventions are unknown or inconsistent."""


@dataclass
class SpiralBundle:
    """Versioned container for a stored SIESTA spiral reference.

    Attributes
    ----------
    sparse : dict[(mu, nu, R) -> (H1, H2, H3, H4)]
        Stored 4-component spin blocks keyed by orbital pair and signed
        lattice vector, with the physical pairing ``(nu, mu, -R)``. The
        spin-diagonal components (H1, H2) must match their partner's
        exactly; the mixing components (H3, H4) may carry producer
        texture and are not compared across partners. ``R`` is either a
        signed 1D cell index (scalar path) or a 3-tuple of signed cell
        indices (vector path, requires ``cell`` and
        ``orbital_positions``).
    overlap : dict[(mu, nu, R) -> float]
        Spin-independent lab overlap values with the same pairing; every
        sparse key must have an overlap entry.
    pitch_sign : str
        Producer convention for the exported pitch: ``qphs_up_exp_minus``
        means the up sector carries ``exp(-i q x / 2)`` (diag2kspiral).
    cell : tuple or None
        Lattice vectors as 3 rows (Bohr); required for the vector path.
        Phases use the full ``xij = R @ cell + pos[nu] - pos[mu]``
        displacement (fixture-certified: dropping the intra-cell orbital
        offset part breaks the eigenvalue oracle at the 1e-2 eV level).
    orbital_positions : tuple or None
        Per-orbital positions (Bohr, matching the sparse keys' orbital
        indices); required for the vector path.
    """

    version: str
    norb: int
    ncell: int
    q: tuple
    spin_packing: str
    phase_convention: str
    hermitization: str
    fermi_shift: str
    pitch_sign: str
    sparse: dict
    overlap: dict
    metadata: dict = field(default_factory=dict)
    cell: tuple | None = None
    orbital_positions: tuple | None = None
    hubbard: "DftuProvenance | None" = None

    @staticmethod
    def _neg(R):
        if isinstance(R, tuple):
            return tuple(-v for v in R)
        return -R

    def validate(self) -> None:
        if self.version != BUNDLE_VERSION:
            if self.version == "1.0" and self.hubbard is None:
                pass  # 1.0 predates the +U payload; imports as hubbard=None
            elif self.version == "1.0":
                raise SpiralImportError(
                    "version 1.0 bundles predate the +U payload and cannot "
                    "carry hubbard metadata; refusing partial hubbard-on-1.0"
                )
            else:
                raise SpiralImportError(
                    f"version {self.version!r} unsupported (expected "
                    f"{BUNDLE_VERSION!r}); refusing to guess conventions"
                )
        if self.spin_packing not in _SUPPORTED_SPIN_PACKINGS:
            raise SpiralImportError(f"spin_packing {self.spin_packing!r} unsupported")
        if self.phase_convention not in _SUPPORTED_PHASE_CONVENTIONS:
            raise SpiralImportError(
                f"phase_convention {self.phase_convention!r} unsupported"
            )
        if self.hermitization not in _SUPPORTED_HERMITIZATIONS:
            raise SpiralImportError(f"hermitization {self.hermitization!r} unsupported")
        if self.fermi_shift not in _SUPPORTED_FERMI_SHIFTS:
            raise SpiralImportError(
                f"fermi_shift {self.fermi_shift!r} unsupported; sisl shifts H by "
                "-E_f on read -- undo or declare the shift explicitly"
            )
        if self.pitch_sign not in _SUPPORTED_PITCH_SIGNS:
            raise SpiralImportError(
                f"pitch_sign {self.pitch_sign!r} unsupported: the producer's "
                "exported pitch-sign convention must be declared (diag2kspiral "
                "applies exp(-i q x / 2) to the up sector)"
            )
        vector_keys = any(isinstance(R, tuple) for _, _, R in self.sparse)
        if vector_keys:
            if self.cell is None or self.orbital_positions is None:
                raise SpiralImportError(
                    "3-tuple cell keys require cell and orbital_positions; "
                    "refusing to guess the xij displacements"
                )
            if len(self.q) != 3:
                raise SpiralImportError(
                    f"vector path requires q of length 3, got {len(self.q)}"
                )
            cell = np.asarray(self.cell, dtype=float)
            positions = np.asarray(self.orbital_positions, dtype=float)
            if cell.shape != (3, 3) or not np.isfinite(cell).all():
                raise SpiralImportError(
                    "cell must contain three finite Cartesian lattice vectors"
                )
            if positions.shape != (self.norb, 3) or not np.isfinite(positions).all():
                raise SpiralImportError(
                    "orbital_positions must contain one finite 3-vector per orbital"
                )
            if any(not isinstance(R, tuple) or len(R) != 3 for _, _, R in self.sparse):
                raise SpiralImportError(
                    "vector bundles require consistent 3-tuple cell keys"
                )
        elif self.cell is not None or self.orbital_positions is not None:
            raise SpiralImportError(
                "cell/orbital_positions provided but all sparse keys are "
                "scalar; refusing ambiguous geometry"
            )
        if not np.isfinite(self.q).all() or (not vector_keys and len(self.q) != 1):
            raise SpiralImportError(
                "pitch must be finite and match the scalar/vector geometry"
            )
        for (mu, nu, R), value in self.overlap.items():
            partner = self.overlap.get((nu, mu, self._neg(R)))
            if partner is None or not np.isclose(partner, value):
                raise SpiralImportError(
                    f"overlap pairing violated at (mu={mu}, nu={nu}, R={R}); "
                    "expected (nu, mu, -R) with the same value"
                )
        for (mu, nu, R), comps in self.sparse.items():
            partner_key = (nu, mu, self._neg(R))
            partner = self.sparse.get(partner_key)
            if partner is None:
                raise SpiralImportError(
                    f"sparse pairing violated at (mu={mu}, nu={nu}, R={R}); "
                    f"expected partner {partner_key}"
                )
            if not (
                np.isclose(comps[0], partner[0]) and np.isclose(comps[1], partner[1])
            ):
                raise SpiralImportError(
                    f"sparse spin-diagonal components differ between "
                    f"{(mu, nu, R)} and {partner_key}; the pencil would not be "
                    "Hermitian -- refusing inconsistent bundle"
                )
            if (mu, nu, R) not in self.overlap:
                raise SpiralImportError(
                    f"missing overlap entry for sparse key (mu={mu}, nu={nu}, "
                    f"R={R}); every stored Hamiltonian block needs its phased "
                    "overlap"
                )
        if self.hubbard is not None:
            from TB2J.interfaces.siesta_spiral_hubbard import DftuProvenance

            if not isinstance(self.hubbard, DftuProvenance):
                raise SpiralImportError(
                    "hubbard payload present but not a validated "
                    "DftuProvenance; build it through "
                    "DftuProvenance.from_dict, which refuses partial +U "
                    "metadata (refusing hubbard-on-bundle with incomplete "
                    "provenance)"
                )


class SpiralImporter:
    """Assemble rotating-frame pencils from a :class:`SpiralBundle`."""

    # MathDoc: tb2j/siesta-spiral-vector (package docs/sympy authority).
    def assemble_pencil(self, bundle: SpiralBundle, k):
        """Return (H(k), S(k)) in the rotating frame.

        ``k`` is a scalar for the 1D scalar-cell path, or a 3-vector
        (Cartesian, inverse Bohr) for the vector path. The vector path
        uses the full displacement ``xij = R @ cell + pos[nu] - pos[mu]``
        exactly as ``diag2kspiral``'s ``setup_k`` does, with
        ``D11 = exp(-i (k + q/2) xij)`` on the up sector (row ``mu`` and
        the H12 block) and ``D22 = exp(-i (k - q/2) xij)`` on the down
        sector, applied to both H and S.
        """
        bundle.validate()
        n = bundle.norb
        vector = bundle.cell is not None
        if vector:
            cell = np.asarray(bundle.cell, dtype=float)
            pos = np.asarray(bundle.orbital_positions, dtype=float)
            qvec = np.asarray(bundle.q, dtype=float)
            kvec = np.atleast_1d(np.asarray(k, dtype=float))
            if kvec.shape != (3,) or not np.isfinite(kvec).all():
                raise SpiralImportError(
                    f"vector bundle needs a 3-vector k, got shape {kvec.shape}"
                )
        else:
            q = bundle.q[0]
        H = np.zeros((2 * n, 2 * n), complex)
        S = np.zeros((2 * n, 2 * n), complex)
        for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
            if vector:
                xij = np.asarray(R, dtype=float) @ cell + pos[nu] - pos[mu]
                kx = float(kvec @ xij)
                qx = float(qvec @ xij)
            else:
                kx = k * R
                qx = q * R
            kphs = np.exp(-1j * kx)
            qphs = np.exp(-1j * qx / 2)
            d11 = kphs * qphs
            d22 = kphs * np.conj(qphs)
            h12 = complex(h3, -h4)
            H[mu, nu] += h1 * d11
            H[mu, nu + n] += h12 * d11
            H[mu + n, nu] += np.conj(h12) * d22
            H[mu + n, nu + n] += h2 * d22
            sv = bundle.overlap[(mu, nu, R)]
            S[mu, nu] += sv * d11
            S[mu + n, nu + n] += sv * d22
        return H, S
