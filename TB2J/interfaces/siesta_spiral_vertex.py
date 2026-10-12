"""Production magnetic vertex from the total-H spin splitting (story 005).

Implements the revised ADR-004 production convention (user decision
2026-10-12): the rotating magnetic operator is the spin-dependent part of
the saved total Hamiltonian (Oroszlany, PRB 99, 224412) -- no XC
reconstruction on this path; the Durhuus-Skovhus spatial-XC vertex is a
separate optional audit (story 007).

Conventions pinned symbolically by ``docs/sympy/siesta_spiral_vertex.py``:

* Pauli decomposition of each stored spinor block
  ``W = h0*sigma0 + h.sigma`` with ``h0 = (H1+H2)/2``,
  ``hx = H3``, ``hy = +H4`` (``H12 = H3 - 1j*H4 = hx - 1j*hy``),
  ``hz = (H1-H2)/2``; the vertex block is the traceless part ``h.sigma``.
* Local rotation of a declared partition acts as a two-sided same-U
  conjugation ``U_A W U_A^dag`` on every entry whose ``mu`` or ``nu``
  belongs to site A (one-sided variants are non-production, story 007).
* Certificate fields: vertex convention, packing, sector phases,
  projector/metric conventions, allocation, and a content digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .siesta_spiral import SpiralBundle, SpiralImportError

VERTEX_VERSION = "1.0"
VERTEX_CONVENTION = "oroszlany_total_h_spin_split"
PACKING_CONVENTION = "H12=H3-1j*H4;hy=+H4"


@dataclass(frozen=True)
class MagneticVertex:
    """Traceless spin-split vertex with declared partition and certificate.

    Attributes
    ----------
    blocks : dict[(mu, nu, R) -> (v_up, v_dn, v3, v4)]
        Traceless spinor splitting per stored entry:
        ``v_up = (H1-H2)/2``, ``v_dn = -v_up``, ``v3 = H3``, ``v4 = H4``
        (so the vertex spinor block is ``h.sigma`` and the mixing part
        ``H12 = v3 - 1j*v4`` is carried verbatim from the saved H).
    partition : Mapping[site, frozenset[key]]
        Declared spatial partition: for each site label, the entry keys
        whose ``mu`` or ``nu`` orbital belongs to that site (ADR-004);
        local rotations act as two-sided same-U conjugation there.
    certificate : dict
        Provenance: conventions, bundle identity, partition declaration,
        and ``vertex_sha256`` content digest.
    """

    blocks: dict
    partition: Mapping[object, frozenset]
    certificate: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def key_text(key):
            mu, nu, R = key
            return json.dumps(
                [
                    int(mu),
                    int(nu),
                    [int(r) for r in R] if isinstance(R, tuple) else int(R),
                ]
            )

        return {
            "vertex_version": VERTEX_VERSION,
            "blocks": {
                key_text(key): list(self.blocks[key]) for key in sorted(self.blocks)
            },
            "partition": {
                str(site): sorted(key_text(key) for key in members)
                for site, members in self.partition.items()
            },
            "certificate": dict(self.certificate),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MagneticVertex":
        if data.get("vertex_version") != VERTEX_VERSION:
            raise SpiralImportError(
                "unsupported magnetic-vertex version "
                f"{data.get('vertex_version')!r}; expected {VERTEX_VERSION}"
            )

        def parse_key(text):
            mu, nu, R = json.loads(text)
            return (
                int(mu),
                int(nu),
                (tuple(int(r) for r in R) if isinstance(R, list) else int(R)),
            )

        blocks = {
            parse_key(key_text): tuple(float(v) for v in comps)
            for key_text, comps in data["blocks"].items()
        }
        partition = {
            site: frozenset(parse_key(key_text) for key_text in members)
            for site, members in data["partition"].items()
        }
        vertex = cls(
            blocks=blocks, partition=partition, certificate=dict(data["certificate"])
        )
        cert = vertex.certificate
        expected_fields = {
            "vertex_version": VERTEX_VERSION,
            "vertex_convention": VERTEX_CONVENTION,
            "packing_convention": PACKING_CONVENTION,
            "sector_phase_convention": "q_half_per_spin_sector",
            "pitch_sign": "qphs_up_exp_minus",
            "spin_packing": "H4=-ImH12",
            "hermitization": "none_hamiltonian_path",
            "fermi_shift": "none",
            "projector_convention": "dual basis A~ = S^-1 A (S-weighted traces)",
            "metric_convention": (
                "generalized pencil (H c = eps S c) with the assembled "
                "spin-dependent overlap"
            ),
            "allocation": "DZP channels retained",
            "partition_declaration": (
                "two-sided same-U conjugation on entries with mu or nu in the "
                "declared site"
                if partition
                else "undeclared; rotation gates must refuse"
            ),
        }
        for field_name, expected_value in expected_fields.items():
            if cert.get(field_name) != expected_value:
                raise SpiralImportError(
                    f"magnetic-vertex certificate {field_name} mismatch"
                )
        if not isinstance(cert.get("q"), list) or not isinstance(
            cert.get("source_bundle_sha256"), str
        ):
            raise SpiralImportError("magnetic-vertex certificate lacks source identity")
        digest_fields = {k: v for k, v in cert.items() if k != "vertex_sha256"}
        expected = _content_digest(vertex.blocks, vertex.partition, digest_fields)
        if cert.get("vertex_sha256") != expected:
            raise SpiralImportError(
                "magnetic-vertex certificate digest mismatch; blocks, partition "
                "and provenance no longer describe the same vertex"
            )
        return vertex


def _content_digest(blocks: Mapping, partition: Mapping, certificate: Mapping) -> str:
    hasher = hashlib.sha256()
    for mu, nu, R in sorted(blocks):
        r_text = ",".join(str(r) for r in R) if isinstance(R, tuple) else str(R)
        hasher.update(f"{mu}|{nu}|{r_text}|".encode())
        hasher.update("|".join(repr(float(v)) for v in blocks[(mu, nu, R)]).encode())
    for site in sorted(partition):
        hasher.update(f"site:{site}|".encode())
        for mu, nu, R in sorted(partition[site]):
            r_text = ",".join(str(r) for r in R) if isinstance(R, tuple) else str(R)
            hasher.update(f"{mu}|{nu}|{r_text}|".encode())
    hasher.update(
        json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode()
    )
    return hasher.hexdigest()


def _source_bundle_digest(bundle: SpiralBundle) -> str:
    """Bind the vertex to the exact H, S, pitch, geometry and conventions."""
    hasher = hashlib.sha256()
    for value in (
        bundle.version,
        bundle.spin_packing,
        bundle.phase_convention,
        bundle.hermitization,
        bundle.fermi_shift,
        bundle.pitch_sign,
        bundle.q,
        bundle.cell,
        bundle.orbital_positions,
    ):
        hasher.update(repr(value).encode())
        hasher.update(b"\0")
    for key in sorted(bundle.sparse):
        hasher.update(repr(key).encode())
        hasher.update(repr(bundle.sparse[key]).encode())
        hasher.update(repr(bundle.overlap[key]).encode())
    return hasher.hexdigest()


# MathDoc: tb2j/siesta-spiral-vertex (package docs/sympy authority).
def extract_magnetic_vertex(
    bundle: SpiralBundle,
    *,
    orbitals_by_site: Mapping[str, Sequence[int]] | None = None,
) -> MagneticVertex:
    """Extract the production vertex from a validated spiral bundle.

    ``orbitals_by_site`` declares orbital ownership (site label -> orbital
    indices). Every stored entry incident to a site's orbital belongs to
    that site's rotation partition. Without ownership, the partition is
    explicitly undeclared; story-006 rotation gates must refuse to guess.
    """
    bundle.validate()
    blocks = {}
    for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
        v_up = 0.5 * (h1 - h2)
        blocks[(mu, nu, R)] = (v_up, -v_up, h3, h4)
    if orbitals_by_site is not None:
        owner = {}
        for site, orbitals in orbitals_by_site.items():
            if not isinstance(site, str) or not site:
                raise SpiralImportError("site labels must be non-empty strings")
            for orb in orbitals:
                if orb in owner:
                    raise SpiralImportError(
                        f"orbital {orb} claimed by sites {owner[orb]!r} and "
                        f"{site!r}; the spatial partition must be disjoint"
                    )
                owner[orb] = site
        if set(owner) != set(range(bundle.norb)):
            raise SpiralImportError(
                "spatial partition must assign every orbital exactly once"
            )
        members_by_site = {site: set() for site in orbitals_by_site}
        for key in blocks:
            mu, nu, _R = key
            for site in {owner[mu], owner[nu]}:
                members_by_site[site].add(key)
        partition = {
            site: frozenset(members) for site, members in members_by_site.items()
        }
    else:
        partition = {}
    partition_declared = bool(partition)
    certificate = {
        "vertex_version": VERTEX_VERSION,
        "vertex_convention": VERTEX_CONVENTION,
        "vertex_convention_value": (
            "spin-dependent part of the saved total Hamiltonian; no XC "
            "reconstruction on the production path (ADR-004 revised)"
        ),
        "packing_convention": PACKING_CONVENTION,
        "sector_phase_convention": bundle.phase_convention,
        "pitch_sign": bundle.pitch_sign,
        "spin_packing": bundle.spin_packing,
        "hermitization": bundle.hermitization,
        "fermi_shift": bundle.fermi_shift,
        "bundle_version": bundle.version,
        "q": [float(v) for v in bundle.q],
        "source_bundle_sha256": _source_bundle_digest(bundle),
        "projector_convention": "dual basis A~ = S^-1 A (S-weighted traces)",
        "metric_convention": (
            "generalized pencil (H c = eps S c) with the assembled "
            "spin-dependent overlap"
        ),
        "allocation": "DZP channels retained",
        "partition_declaration": (
            "two-sided same-U conjugation on entries with mu or nu in the "
            "declared site"
            if partition_declared
            else "undeclared; rotation gates must refuse"
        ),
    }
    certificate["vertex_sha256"] = _content_digest(blocks, partition, certificate)
    return MagneticVertex(blocks=blocks, partition=partition, certificate=certificate)
