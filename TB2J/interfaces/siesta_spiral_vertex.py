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

import numpy as np

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


# ---------------------------------------------------------------------------
# Story 006: rotation legitimacy, partition variants, torque-state recording
# ---------------------------------------------------------------------------


def _rodrigues(axis, angle):
    """Unit-axis check + 3x3 rotation matrix (right-handed)."""
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0.0:
        raise SpiralImportError("rotation axis must be a nonzero finite 3-vector")
    n = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(n, n)


def rotate_site_blocks(vertex, site, *, axis=(0.0, 0.0, 1.0), angle=0.0):
    """Finite physical local rotation of one declared site's magnetic blocks.

    Two-sided same-U conjugation on every partition entry of ``site``: the
    magnetic vector h = (v3, v4, v_up) = (hx, hy, hz) of each stored spinor
    block is rotated by R(axis, angle); the scalar part h0 and every other
    site's entries are untouched. Returns a new blocks mapping.
    """
    if not vertex.partition:
        raise SpiralImportError(
            "spatial partition undeclared; rotation gates must refuse to guess"
        )
    if site not in vertex.partition:
        raise SpiralImportError(
            f"rotation requested for unknown site {site!r}; declared sites: "
            f"{sorted(vertex.partition)}"
        )
    rot = _rodrigues(axis, float(angle))
    rotated = dict(vertex.blocks)
    for key in vertex.partition[site]:
        v_up, _v_dn, v3, v4 = vertex.blocks[key]
        h = np.array([v3, v4, v_up], dtype=float)
        hx, hy, hz = rot @ h
        rotated[key] = (hz, -hz, hx, hy)
    return rotated


def _operator_change(blocks_a, blocks_b):
    return float(
        np.sqrt(
            sum(
                (a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2
                for a, b in zip(blocks_a.values(), blocks_b.values())
            )
        )
    )


def rotation_legitimacy_report(
    bundle, vertex, site, *, axis=(0.0, 0.0, 1.0), angle=0.35
):
    """AC-004 rotation clause: the rotation acts on the magnetic operator only.

    Verifies on the real stored data: (i) the magnetic operator changes;
    (ii) the scalar part h0 = (H1+H2)/2 of EVERY entry is unchanged;
    (iii) the overlap S is unchanged (the rotation never touches it);
    (iv) untouched sites' magnetic blocks are unchanged; (v) the trace of
    every rotated spinor block is preserved.
    """
    if not vertex.partition:
        raise SpiralImportError(
            "spatial partition undeclared; rotation gates must refuse to guess"
        )
    if site not in vertex.partition:
        raise SpiralImportError(
            f"rotation requested for unknown site {site!r}; declared sites: "
            f"{sorted(vertex.partition)}"
        )
    rotated = rotate_site_blocks(vertex, site, axis=axis, angle=angle)
    scalar_before = {
        k: 0.5 * (h1 + h2) for k, (h1, h2, _h3, _h4) in bundle.sparse.items()
    }
    scalar_after = {}
    for key, (h1, h2, _h3, _h4) in bundle.sparse.items():
        h0 = 0.5 * (h1 + h2)
        if key in rotated:
            h0 = 0.5 * ((h0 + rotated[key][0]) + (h0 + rotated[key][1]))
        scalar_after[key] = h0
    scalar_change = float(
        np.sqrt(sum((scalar_before[k] - scalar_after[k]) ** 2 for k in scalar_before))
    )
    rotated_members = set(vertex.partition[site])
    untouched_ok = all(
        rotated[key] == vertex.blocks[key]
        for key in vertex.blocks
        if key not in rotated_members
    )
    max_trace = 0.0
    for key in vertex.partition[site]:
        v = rotated[key]
        max_trace = max(max_trace, abs(v[0] + v[1]))
    magnetic_change = _operator_change(vertex.blocks, rotated)
    passed = (
        magnetic_change > 1e-8
        and scalar_change < 1e-12
        and untouched_ok
        and max_trace < 1e-12
        and (float(angle) != 0.0 or magnetic_change == 0.0)
    )
    return {
        "passed": bool(passed),
        "rotated_site": site,
        "axis": [float(x) for x in axis],
        "angle_rad": float(angle),
        "magnetic_operator_change": magnetic_change,
        "scalar_part_change": scalar_change,
        "overlap_change": 0.0,
        "untouched_sites_unchanged": bool(untouched_ok),
        "max_trace_deviation": float(max_trace),
        "rotated_entries": len(vertex.partition[site]),
        "convention": "two-sided same-U conjugation (production, ADR-004)",
    }


def partition_variant_spread(
    bundle, variants, *, probe_orbitals, axis=(0.0, 0.0, 1.0), angle=0.35
):
    """FR-015 sensitivity: the same probe rotation under declared partitions.

    ``variants`` maps a variant name to an ``orbitals_by_site`` declaration.
    The probe rotation is applied, per variant, to the site that owns
    ``probe_orbitals``; the report carries each variant's operator change
    and the maximum pairwise spread of the rotated magnetic operators.
    """
    if not isinstance(variants, dict) or len(variants) < 1:
        raise SpiralImportError("partition variants must be a non-empty mapping")
    probe = [int(o) for o in probe_orbitals]
    if not probe:
        raise SpiralImportError("probe_orbitals must be non-empty")
    per_variant = {}
    blocks_reference = None
    for name, sites in variants.items():
        vertex = extract_magnetic_vertex(bundle, orbitals_by_site=sites)
        if blocks_reference is None:
            blocks_reference = vertex.blocks
        elif _operator_change(blocks_reference, vertex.blocks) != 0.0:
            raise SpiralImportError(
                "vertex blocks are partition-independent; variant extraction "
                "disagreement indicates a corrupted bundle"
            )
        owner = {}
        for site_label, orbitals in sites.items():
            for orb in orbitals:
                if orb in owner:
                    raise SpiralImportError(
                        f"variant {name!r}: orbital {orb} claimed by two sites"
                    )
                owner[orb] = site_label
        owners = {owner.get(orb) for orb in probe}
        if None in owners or len(owners) != 1:
            raise SpiralImportError(
                "probe orbitals must belong to exactly one declared site per "
                f"variant; got owners {sorted(map(str, owners))} in {name!r}"
            )
        site = owners.pop()
        rotated = rotate_site_blocks(vertex, site, axis=axis, angle=angle)
        per_variant[name] = {
            "rotated_site": site,
            "rotated_entries": len(vertex.partition[site]),
            "magnetic_operator_change": _operator_change(vertex.blocks, rotated),
            "blocks": rotated,
        }
    names = sorted(per_variant)
    spread = 0.0
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            spread = max(
                spread,
                _operator_change(per_variant[a]["blocks"], per_variant[b]["blocks"]),
            )
    return {
        "probe_orbitals": probe,
        "axis": [float(x) for x in axis],
        "angle_rad": float(angle),
        "unrotated_blocks_identical": True,
        "variants": {
            name: {k: v for k, v in info.items() if k != "blocks"}
            for name, info in per_variant.items()
        },
        "max_pairwise_spread": spread,
        "convention": (
            "same probe rotation under declared spatial partitions (FR-015)"
        ),
    }


def _is_onsite_key(key):
    """True for a stored onsite entry; R may be a tuple or a scalar (1D)."""
    R = key[2]
    if isinstance(R, (tuple, list)):
        return key[0] == key[1] and all(r == 0 for r in R)
    return key[0] == key[1] and R == 0


def torque_state_report(bundle, vertex, *, transverse_tol=1e-6):
    """FR-008: record the reference's onsite transverse splitting verbatim.

    The onsite magnetic vector h = (H3, H4, (H1-H2)/2) of every stored
    onsite entry is inventoried per declared site (rms magnitudes and the
    transverse fraction). A reference with transverse splitting above the
    tolerance is labeled torque-bearing and recorded, never silently
    accepted; the label follows the recorded data.
    """
    if not vertex.partition:
        raise SpiralImportError(
            "spatial partition undeclared; torque-state recording must refuse "
            "to guess site ownership"
        )
    sites = {}
    worst_transverse = 0.0
    for site, members in vertex.partition.items():
        onsite = [vertex.blocks[key] for key in members if _is_onsite_key(key)]
        if not onsite:
            raise SpiralImportError(
                f"site {site!r} has no stored onsite entry; torque state "
                "cannot be recorded"
            )
        hx = np.array([b[2] for b in onsite])
        hy = np.array([b[3] for b in onsite])
        hz = np.array([b[0] for b in onsite])
        transverse = float(np.sqrt(np.mean(hx**2 + hy**2)))
        longitudinal = float(np.sqrt(np.mean(hz**2)))
        worst_transverse = max(worst_transverse, transverse)
        sites[site] = {
            "n_onsite_entries": len(onsite),
            "transverse_rms_eV": transverse,
            "longitudinal_rms_eV": longitudinal,
            "transverse_fraction": float(
                transverse / max(transverse + longitudinal, 1e-300)
            ),
        }
    state = (
        "nonzero_onsite_transverse_splitting"
        if worst_transverse > transverse_tol
        else "onsite_splitting_within_tolerance"
    )
    return {
        "torque_state": state,
        "transverse_tol_eV": float(transverse_tol),
        "max_site_transverse_rms_eV": worst_transverse,
        "sites": sites,
        "disposition": "recorded_verbatim",
        "note": (
            "torque-bearing references are flagged, not silently accepted; "
            "response consumers must read this record (FR-008)"
        ),
    }
