"""Vertex displacement derivative (story 018, ADR-004, ADR-008).

Forms the displacement derivative of the ADR-004 production magnetic vertex
(total-H spin splitting, story-005/006 machinery, story-017 dH/dR seam):
``dV/dR = (dH1 - dH2)/2`` on the onsite same-site blocks, packed as
``(dV_up, -dV_up, dH3, dH4)`` — the derivative of the full story-005
vertex block, whose certified packing is ``(v_up, -v_up, H3, H4)``
(``H3 = Re H12``, ``H4 = -Im H12``).

Refusals (FR-014 pattern): an ambiguous vertex request (displacement response
without a declared vertex-derivative / fixed-partition label) and a story-006
partition VARIANT requested together with derivatives are refused with
actionable diagnostics instead of silently dropping the partition term.

MathDoc: tb2j/siesta-spiral-vertex-displacement
(the executable authority is docs/sympy/siesta_spiral_vertex_displacement.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .siesta_spiral_displacement import (
    GAUGE_LABELS,
    DisplacementDerivatives,
    GaugeMismatch,
    SparsePatternMismatch,
    displacement_payload_sha256,
)
from .siesta_spiral_displacement import (
    DisplacedReferenceFDProvider as _DisplacedReferenceFDProvider,
)
from .siesta_spiral_hubbard import BOHR_TO_ANG
from .siesta_spiral_vertex import (
    DEFAULT_OPERATOR_MODE,
    OPERATOR_MODE_ONSITE_DELTA,
    OPERATOR_MODE_WEIGHTED_TOTAL_H,
    OPERATOR_MODES,
    extract_magnetic_vertex,
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
    "carries all four packed components of the vertex block. PRD risk: "
    f"{PRD_PARTITION_RISK}."
)

TERMS_INCLUDED = ("vertex",)
TERMS_EXCLUDED = (
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
    packing ``(dV_up, -dV_up, dH3, dH4)`` with ``dV_up = (dH1 - dH2)/2``
    in eV/Ang (file packing as in :class:`DisplacementDerivatives`; the
    transverse components are the packed derivatives of ``H3 = Re H12``
    and ``H4 = -Im H12``, i.e. the derivative of the full story-005
    vertex block).
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
        d1, d2, d3, d4 = derivatives.dH[key]
        v_up = (d1 - d2) / 2.0
        dV[key] = (v_up, -v_up, d3, d4)
    provenance = {
        "source_producer_recipe": derivatives.producer_recipe,
        "vertex_displacement_version": VERTEX_DISPLACEMENT_VERSION,
        "mathdoc": "tb2j/siesta-spiral-vertex-displacement",
        "packing": "(dV_up, -dV_up, dH3, dH4); dV_up = (dH1 - dH2)/2 on "
        "onsite same-site (mu, nu, R=0) blocks (derivative of the full "
        "story-005 vertex block)",
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


# ---------------------------------------------------------------------------
# Story 019: AC-009 validation driver
# ---------------------------------------------------------------------------

# Pipeline-consistency gate pinned at story-019 approval: the vertex
# functional is LINEAR in H, so the derivative-based route and the
# reference-based route differ only by floating-point ordering.  1e-10
# eV/Ang absolute is the pinned tolerance; this is a correctness gate,
# NOT a numerical-convergence claim (NFR-006; convergence is story-022).
AC009_TOLERANCE_EV_PER_ANG = 1e-10

# Ensemble/alignment bookkeeping (ADR-005), recorded verbatim in every
# verdict.  Story-004 lesson: two independent SCFs can differ ~0.1
# eV/cell; this gate compares Hamiltonian blocks under a matched
# per-file energy alignment, never cross-SCF energies.
AC009_ENSEMBLE_BOOKKEEPING = {
    "ensemble_declaration": (
        "ADR-005 frozen-MFT response with declared ensemble: occupations "
        "fixed by the producer; each displaced reference is an INDEPENDENT "
        "SCF and the gate compares Hamiltonian/vertex blocks only — no "
        "cross-SCF energy difference enters the gate"
    ),
    "energy_alignment": (
        "per-file: sisl stores H-Ef(file)S; the certified loader "
        "(_bundle_from_hs_geometry) undoes each file's -Ef shift, so both "
        "routes consume the raw producer H (fermi_shift='none')"
    ),
    "electronic_branch": (
        "central finite differences (branch='central'); the derivative "
        "ladder is the pinned +/-0.05 Bohr displaced-SCF pair"
    ),
    "perturbation_definition": (
        "displacement of the declared atom along the declared Cartesian "
        "axis of the reference geometry, its periodic copies included "
        "(periodic-image-sum gauge)"
    ),
}


class LocalizedGaugeUnavailable(GaugeMismatch):
    """A localized-gauge AC-009 validation was requested from periodic
    displaced-SCF sampling; the localized observable requires the
    localized-gauge displaced set (dP endpoints), which the periodic
    campaign does not provide."""


def run_ac009_validation(
    bundle,
    pattern,
    plus_path=None,
    minus_path=None,
    *,
    plus_sile=None,
    minus_sile=None,
    hsx_opener=None,
    reference_fermi_ev=None,
    orbitals_by_site=None,
    operator: str = DEFAULT_OPERATOR_MODE,
    gauge: str = "periodic-image-sum",
    tolerance: float = AC009_TOLERANCE_EV_PER_ANG,
) -> dict:
    """Validate the vertex displacement derivative against independent
    displaced references (AC-009, FR-012).

    Two independent routes are run on the SAME displaced +h/-h pair:

    (a) derivative-based: :class:`DisplacedReferenceFDProvider` forms the
        sparse dH/dR record and :func:`vertex_displacement_derivative`
        packs the fixed-partition dV/dR;
    (b) reference-based: ``extract_magnetic_vertex`` is applied to each
        displaced HSX independently (through the certified loader, with
        each file's Fermi shift undone) and the vertex blocks are
        finite-differenced directly.

    Route (a) is compared against route (b) on ALL FOUR components of
    every onsite block.  Because the vertex functional is linear in H,
    agreement must be tight; the default absolute tolerance
    (``AC009_TOLERANCE_EV_PER_ANG`` = 1e-10 eV/Ang) is pinned at
    story-019 approval and is a pipeline-consistency gate, not a
    convergence claim.

    Returns a JSON-serializable verdict dict carrying the per-site /
    per-component agreement table, the gauge labels, and the ensemble /
    alignment bookkeeping verbatim. ``derivative_payload_sha256`` binds
    every field of the constructed vertex record; ``seam_payload_sha256``
    separately binds the complete dH/dS seam record, including components
    outside the onsite magnetic comparison. Either record can therefore
    be exported against this verdict without reconstructing its payload
    from the agreement table. The verdict is ``"pass"`` only when every
    compared component is within ``tolerance``; a failed gate returns an
    evidence-bearing ``"fail"`` verdict, never a fabricated agreement.
    """
    if gauge not in GAUGE_LABELS:
        raise GaugeMismatch(
            f"requested gauge {gauge!r} is unknown; expected one of "
            f"{sorted(GAUGE_LABELS)}"
        )
    if gauge == "localized":
        raise LocalizedGaugeUnavailable(
            "a localized-gauge AC-009 validation is refused: the localized "
            "observable requires the localized-gauge displaced set — the "
            "translated dP endpoints (displaced-projector references) — "
            "which the periodic displaced-SCF campaign does not provide. "
            "The periodic campaign's +/- displaced SCFs realize only the "
            "'periodic-image-sum' gauge; run a localized-gauge displaced "
            "set (dP endpoints) to certify the localized observable."
        )
    bundle_gauge = bundle.metadata.get("displacement_gauge")
    if bundle_gauge != gauge:
        raise GaugeMismatch(
            f"the reference bundle declares displacement gauge "
            f"{bundle_gauge!r} but the validation requests {gauge!r}; "
            "gauges are different observables and must not be mixed"
        )

    # route (a): derivative-based vertex displacement record
    provider = _DisplacedReferenceFDProvider(
        pattern=pattern,
        plus_path=plus_path,
        minus_path=minus_path,
        plus_sile=plus_sile,
        minus_sile=minus_sile,
        reference_bundle=bundle,
        reference_fermi_ev=reference_fermi_ev,
        hsx_opener=hsx_opener,
    )
    derivatives = provider.provide()
    if derivatives.gauge != gauge:
        raise GaugeMismatch(
            f"the seam record carries gauge {derivatives.gauge!r} but the "
            f"validation requests {gauge!r}"
        )
    record = vertex_displacement_derivative(
        derivatives,
        bundle,
        PARTITION_FIXED,
        orbitals_by_site=orbitals_by_site,
        operator=operator,
    )
    if record.gauge != gauge:
        raise GaugeMismatch(
            f"the vertex-derivative record carries gauge {record.gauge!r} "
            f"but the validation requests {gauge!r}"
        )

    # route (b): reference-based FD of the extracted vertex blocks
    opener = hsx_opener or _default_hsx_opener
    plus_sile_b = plus_sile if plus_sile is not None else opener(plus_path)
    minus_sile_b = minus_sile if minus_sile is not None else opener(minus_path)
    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    plus_bundle = _bundle_from_hs_geometry(plus_sile_b, "displaced_plus", "")
    minus_bundle = _bundle_from_hs_geometry(minus_sile_b, "displaced_minus", "")
    reference_keys = set(bundle.sparse)
    for label, displaced_bundle in (
        ("+h", plus_bundle),
        ("-h", minus_bundle),
    ):
        displaced_keys = set(displaced_bundle.sparse)
        extras = sorted(displaced_keys - reference_keys)
        missing = sorted(reference_keys - displaced_keys)
        if extras or missing:
            raise SparsePatternMismatch(
                f"route (b): the {label} displaced HSX export does not have "
                "exactly the reference bundle's sparse pattern: "
                f"extra keys {extras}, missing keys {missing}. "
                "assemble_onto_bundle requires exact coverage, so a "
                "displaced pair must carry the same keys and no others; "
                "regenerate the displaced export with the sparse pattern "
                "of the reference bundle."
            )
    if set(plus_bundle.sparse) != set(minus_bundle.sparse):
        raise SparsePatternMismatch(
            "route (b): the +h and -h displaced HSX exports carry different "
            "sparse patterns; matched displaced references are required"
        )
    plus_vertex = extract_magnetic_vertex(plus_bundle, operator=operator)
    minus_vertex = extract_magnetic_vertex(minus_bundle, operator=operator)
    route_b_gauge = "periodic-image-sum"  # loader semantics: periodic SCF,
    # per-file Fermi undone; certified loader output, fermi_shift='none'

    h_step_ang = pattern.step_bohr * BOHR_TO_ANG
    agreement = {}
    max_deviation = 0.0
    worst = None
    for key in sorted(record.dV):
        if key not in plus_vertex.blocks:
            raise SparsePatternMismatch(
                f"route (b): onsite key {key} is absent from the displaced "
                "references' vertex blocks; the displaced HSX pattern does "
                "not cover the reference bundle's onsite blocks"
            )
        derived = record.dV[key]
        reference_fd = tuple(
            (p - m) / (2.0 * h_step_ang)
            for p, m in zip(plus_vertex.blocks[key], minus_vertex.blocks[key])
        )
        devs = [abs(a - b) for a, b in zip(derived, reference_fd)]
        agreement[str(key)] = {
            "derived": [float(v) for v in derived],
            "reference_fd": [float(v) for v in reference_fd],
            "deviation": [float(d) for d in devs],
        }
        for i, d in enumerate(devs):
            if d > max_deviation:
                max_deviation = d
                worst = (key, i)
    status = "pass" if max_deviation <= tolerance else "fail"
    geom = derivatives.reference_geometry
    verdict = {
        "acceptance_criterion": "AC-009",
        "status": status,
        "derivative_payload_sha256": displacement_payload_sha256(record),
        "seam_payload_sha256": displacement_payload_sha256(derivatives),
        "tolerance_ev_per_ang": float(tolerance),
        "tolerance_basis": (
            "pipeline-consistency gate pinned at story-019 approval: the "
            "vertex functional is linear in H, so the two routes must "
            "agree to floating-point ordering only; NOT a numerical "
            "convergence claim (NFR-006, story-022)"
        ),
        "max_deviation_ev_per_ang": float(max_deviation),
        "worst_component": (
            None if worst is None else {"key": str(worst[0]), "component": worst[1]}
        ),
        "pattern": {
            "atom_index": pattern.atom_index,
            "axis": pattern.axis,
            "step_bohr": pattern.step_bohr,
            "branch": pattern.branch,
            "direction": list(pattern.direction),
        },
        "gauge": {
            "seam_record": derivatives.gauge,
            "vertex_derivative_record": record.gauge,
            "displaced_reference_route": route_b_gauge,
            "requested": gauge,
        },
        "ensemble": dict(AC009_ENSEMBLE_BOOKKEEPING),
        "agreement": agreement,
        "geometry": {
            "reference_digests": tuple(derivatives.reference_digests),
            "displaced_axis_offset_bohr": float(geom.displaced_axis_offset_bohr),
            "n_orbitals": len(geom.orbital_positions),
            "cell_bohr": [list(map(float, row)) for row in geom.cell],
            "producer_recipe_route_a": derivatives.producer_recipe,
            "producer_recipe_route_b": (
                "extract_magnetic_vertex on each displaced HSX independently "
                "through the certified loader (_bundle_from_hs_geometry, "
                "per-file -Ef shift undone), central FD of the vertex blocks"
            ),
        },
        "partition_declared": record.partition_declared,
        "operator_mode": record.operator_mode,
        "declaration": record.declaration(),
    }
    return verdict


def _default_hsx_opener(path):
    import sisl

    return sisl.get_sile(str(path))
