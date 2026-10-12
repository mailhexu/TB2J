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

import numpy as np

BUNDLE_VERSION = "1.0"

_SUPPORTED_SPIN_PACKINGS = frozenset({"H3=-ImH12"})
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
        texture and are not compared across partners.
    overlap : dict[(mu, nu, R) -> float]
        Spin-independent lab overlap values with the same pairing; every
        sparse key must have an overlap entry.
    pitch_sign : str
        Producer convention for the exported pitch: ``qphs_up_exp_minus``
        means the up sector carries ``exp(-i q x / 2)`` (diag2kspiral).
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

    def validate(self) -> None:
        if self.version != BUNDLE_VERSION:
            raise SpiralImportError(
                f"version {self.version!r} unsupported (expected {BUNDLE_VERSION!r}); "
                "refusing to guess conventions"
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
        for (mu, nu, R), value in self.overlap.items():
            partner = self.overlap.get((nu, mu, -R))
            if partner is None or not np.isclose(partner, value):
                raise SpiralImportError(
                    f"overlap pairing violated at (mu={mu}, nu={nu}, R={R}); "
                    "expected (nu, mu, -R) with the same value"
                )
        for (mu, nu, R), comps in self.sparse.items():
            partner_key = (nu, mu, -R)
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


class SpiralImporter:
    """Assemble rotating-frame pencils from a :class:`SpiralBundle`."""

    def assemble_pencil(self, bundle: SpiralBundle, k: float):
        """Return (H(k), S(k)) in the rotating frame at scalar 1D k."""
        bundle.validate()
        n = bundle.norb
        q = bundle.q[0]
        H = np.zeros((2 * n, 2 * n), complex)
        S = np.zeros((2 * n, 2 * n), complex)
        for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
            x = R
            kphs = np.exp(-1j * k * x)
            qphs = np.exp(-1j * q * x / 2)
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
