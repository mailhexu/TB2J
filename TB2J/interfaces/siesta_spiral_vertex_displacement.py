"""Vertex displacement derivative (story 018, ADR-004, ADR-008).

Forms the displacement derivative of the ADR-004 production magnetic vertex
(total-H spin splitting, story-005/006 machinery, story-017 dH/dR seam):
``dV/dR = (dH1 - dH2)/2`` on the onsite same-site blocks, packed as
``(v_up, -v_up, 0, 0)`` — the declared diagonal-spinor splitting only.

Refusals (FR-014 pattern): an ambiguous vertex request (displacement response
without a declared vertex-derivative / fixed-partition label) and a story-006
partition VARIANT requested together with derivatives are refused with
actionable diagnostics instead of silently dropping the partition term.

MathDoc: tb2j/siesta-spiral-vertex-displacement
(the executable authority is docs/sympy/siesta_spiral_vertex_displacement.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .siesta_spiral_displacement import DisplacementDerivatives
from .siesta_spiral_vertex import (
    DEFAULT_OPERATOR_MODE,
    OPERATOR_MODE_ONSITE_DELTA,
    OPERATOR_MODE_WEIGHTED_TOTAL_H,
    OPERATOR_MODES,
)

VERTEX_DISPLACEMENT_VERSION = "1.0"

PARTITION_FIXED = "fixed"

# PRD risk, verbatim from
# subprojects/siesta-fixed-spin-spiral-nonorthogonal-xc-mft/specs/prd.md
PRD_PARTITION_RISK = "Displacement changes the site/projector definition"

PARTITION_NOTE = (
    "fixed-partition observable (exact, not approximate): the site "
    "partition map enters the vertex as CONSTANT index data, so "
    "d(partition)/dR = 0 identically; the declared vertex derivative "
    "carries the dH1-dH2 splitting components only. PRD risk: "
    f"{PRD_PARTITION_RISK}."
)

TERMS_INCLUDED = ("vertex",)
TERMS_EXCLUDED = (
    "onsite_mixing_transport",
    "overlap_S",
    "partition",
    "scalar_h0",
)

# story-006 partition variants: declared partition families that propagate
# as declared variants and are REFUSED (not silently dropped) under
# displacement derivatives
PARTITION_VARIANTS = frozenset({OPERATOR_MODE_WEIGHTED_TOTAL_H})

EXPECTED_DV_UNITS = {"dV": "eV/Ang"}


# ---------------------------------------------------------------------------
# Refusal classes (FR-014)
# ---------------------------------------------------------------------------


class VertexDisplacementError(RuntimeError):
    """Base class: vertex displacement derivative requests are incoherent."""


class AmbiguousVertexRequest(VertexDisplacementError):
    """A displacement response was requested without declaring whether it
    is a vertex derivative under a fixed partition; the caller must state
    the observable family instead of letting one be guessed."""


class PartitionVariantUnderDisplacement(VertexDisplacementError):
    """A story-006 partition variant (e.g. the weighted site partition)
    was requested together with displacement derivatives; propagating it
    would require the partition derivative the fixed-partition record
    excludes by declaration, so the variant must be declared explicitly
    through its own path instead of being silently dropped."""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VertexDisplacementDerivatives:
    """Sparse dV/dR in the bundle's conventions, fixed-partition labeled.

    ``dV`` maps onsite same-site ``(mu, nu, R=0)`` keys to the 4-component
    packing ``(v_up, -v_up, 0.0, 0.0)`` with ``v_up = (dH1 - dH2)/2`` in
    eV/Ang (file packing as in :class:`DisplacementDerivatives`).
    """

    pattern: object
    dV: dict
    partition: str
    partition_note: str
    terms_included: tuple
    terms_excluded: tuple
    operator_mode: str
    partition_declared: bool
    gauge: str
    units: dict
    producer_recipe: str
    step_bohr: float
    branch: str
    provenance: dict = field(default_factory=dict)

    def declaration(self) -> dict:
        """The machine-checkable included/excluded declaration."""
        return {
            "terms_included": tuple(self.terms_included),
            "terms_excluded": tuple(self.terms_excluded),
            "partition": self.partition,
            "partition_note": self.partition_note,
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _validate_ownership(orbitals_by_site, norb):
    """Disjoint full orbital -> site ownership (story-005 rules)."""
    owner = {}
    for site, orbitals in orbitals_by_site.items():
        if not isinstance(site, str) or not site:
            raise AmbiguousVertexRequest(
                "site labels must be non-empty strings; refusing to build "
                f"the vertex derivative from ownership {orbitals_by_site!r}"
            )
        for orb in orbitals:
            if orb in owner:
                raise AmbiguousVertexRequest(
                    f"orbital {orb} claimed by sites {owner[orb]!r} and "
                    f"{site!r}; the spatial partition must be disjoint"
                )
            owner[orb] = site
    if set(owner) != set(range(norb)):
        raise AmbiguousVertexRequest(
            "spatial partition must assign every orbital exactly once; "
            "refusing to attribute same-site blocks with partial ownership"
        )
    return owner


def vertex_displacement_derivative(
    derivatives: DisplacementDerivatives,
    bundle,
    partition: str | None = PARTITION_FIXED,
    *,
    orbitals_by_site=None,
    operator: str = DEFAULT_OPERATOR_MODE,
) -> VertexDisplacementDerivatives:
    """Form the fixed-partition vertex displacement derivative.

    ``derivatives`` is the story-017 :class:`DisplacementDerivatives`
    record; ``bundle`` is the assembled reference bundle (its sparse
    pattern defines key coverage and its ``norb`` validates ownership).
    ``partition`` must be the explicit ``"fixed"`` declaration; ``None``
    or an unknown label is an ambiguous vertex request, and a story-006
    partition variant is refused under displacement.

    ``orbitals_by_site`` (site label -> orbital indices) lets same-site
    R=0 blocks with distinct orbitals (``mu != nu``) enter the site
    partition, exactly as the story-005 ``onsite_delta`` partition does.
    Without declared ownership only the diagonal onsite entries
    (``mu == nu``, ``R = 0``) are certified — every site's own orbitals —
    and the record says so via ``partition_declared=False``.
    """
    if partition is None:
        raise AmbiguousVertexRequest(
            "no partition declaration given: a displacement response must "
            "declare whether it is a vertex derivative under the fixed "
            'partition (pass partition="fixed") or an explicitly '
            "partition-resolved observable; refusing to guess which "
            "observable the dH/dR record realizes"
        )
    if partition != PARTITION_FIXED:
        if partition in PARTITION_VARIANTS:
            raise PartitionVariantUnderDisplacement(
                f"partition variant {partition!r} under displacement is a "
                "declared variant (story-006), not a fixed-partition "
                "observable: propagating it would require the partition "
                "derivative that the displacement record excludes. Declare "
                'the fixed-partition vertex derivative (partition="fixed") '
                "or resolve the variant through its own declared path; "
                "silently dropping the partition term is refused."
            )
        raise AmbiguousVertexRequest(
            f"unknown partition declaration {partition!r}; expected "
            f"{PARTITION_FIXED!r} (the only partition family with an exact "
            "displacement derivative), None (refused as ambiguous), or a "
            f"declared story-006 variant {sorted(PARTITION_VARIANTS)} "
            "(refused under displacement)"
        )
    if operator not in OPERATOR_MODES:
        raise AmbiguousVertexRequest(
            f"unknown magnetic operator mode {operator!r}; the vertex "
            f"derivative is declared for {sorted(OPERATOR_MODES)}"
        )
    if operator != OPERATOR_MODE_ONSITE_DELTA:
        raise PartitionVariantUnderDisplacement(
            f"operator mode {operator!r} is the weighted site partition "
            "(story-006): under displacement it is a declared variant, "
            "not a fixed-partition observable; refusing to drop its "
            "partition structure silently"
        )
    if derivatives.units.get("dH") != "eV/Ang":
        raise AmbiguousVertexRequest(
            f"derivative record units {derivatives.units} are not the "
            "declared eV/Ang convention; refusing to pack a vertex "
            "derivative in unknown units"
        )
    if bundle is None:
        raise AmbiguousVertexRequest(
            "no reference bundle given: the vertex derivative needs the "
            "bundle's sparse pattern to scope the onsite same-site blocks"
        )
    owner = None
    if orbitals_by_site is not None:
        owner = _validate_ownership(orbitals_by_site, bundle.norb)
    dV = {}
    for key in bundle.sparse:
        mu, nu, R = key
        if R != 0 and R != (0, 0, 0):
            continue
        if owner is not None:
            if owner[mu] != owner[nu]:
                continue
        elif mu != nu:
            # without declared ownership the same-site attribution of
            # mu != nu blocks would be a guess: only diagonal onsite
            # entries (each site's own orbital) are certified
            continue
        d1, d2, _d3, _d4 = derivatives.dH[key]
        v_up = (d1 - d2) / 2.0
        dV[key] = (v_up, -v_up, 0.0, 0.0)
    provenance = {
        "source_producer_recipe": derivatives.producer_recipe,
        "vertex_displacement_version": VERTEX_DISPLACEMENT_VERSION,
        "mathdoc": "tb2j/siesta-spiral-vertex-displacement",
        "packing": "(v_up, -v_up, 0, 0); v_up = (dH1 - dH2)/2 on onsite "
        "same-site (mu, nu, R=0) blocks",
    }
    return VertexDisplacementDerivatives(
        pattern=derivatives.pattern,
        dV=dV,
        partition=PARTITION_FIXED,
        partition_note=PARTITION_NOTE,
        terms_included=TERMS_INCLUDED,
        terms_excluded=TERMS_EXCLUDED,
        operator_mode=operator,
        partition_declared=orbitals_by_site is not None,
        gauge=derivatives.gauge,
        units=dict(EXPECTED_DV_UNITS),
        producer_recipe=(
            "vertex displacement derivative over " f"{derivatives.producer_recipe}"
        ),
        step_bohr=derivatives.step_bohr,
        branch=derivatives.branch,
        provenance=provenance,
    )
