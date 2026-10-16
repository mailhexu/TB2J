"""Public fail-closed SIESTA spiral exchange-export contract (story-020)."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest
from ase import Atoms
from test_siesta_spiral_ac009 import (
    STEP_BOHR,
    _ext_fake_sile,
)
from test_siesta_spiral_ac009 import (
    synthetic_bundle as _synthetic_bundle,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    audit_artifact as _audit_artifact,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    contract_table as _contract_table,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    cross_check as _cross_check,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    dftu as _dftu,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    gate as _gate,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    moments as _moments,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    producer_hubbard as _producer_hubbard,
)
from test_siesta_spiral_hubbard_ac008_gate import (
    texture_provenance,
)

from TB2J.interfaces.siesta_spiral_api import (
    ExportRefusal,
    FailedGate,
    IncompatibleEnsemble,
    MissingOperatorProvenance,
    NonHeisenbergCompatibleResponse,
    ProvenanceSchema,
    UncertifiedDisplacement,
    UncertifiedPlusU,
    build_exchange_export,
)
from TB2J.interfaces.siesta_spiral_displacement import (
    DisplacedReferenceFDProvider,
    DisplacementPattern,
)
from TB2J.interfaces.siesta_spiral_hubbard import run_ac008_gate
from TB2J.interfaces.siesta_spiral_mapping import (
    build_synthetic_response_set,
    map_response_set,
)
from TB2J.interfaces.siesta_spiral_response import (
    ENSEMBLE_GAPPED_T0,
    ENSEMBLE_METALLIC_FD,
    Ensemble,
)
from TB2J.interfaces.siesta_spiral_vertex_displacement import (
    run_ac009_validation,
    vertex_displacement_derivative,
)
from TB2J.io_exchange import SpinIO

# Register shared fixtures without shadowing imported names in test parameters.
synthetic_bundle = _synthetic_bundle
audit_artifact = _audit_artifact
contract_table = _contract_table
cross_check = _cross_check
dftu = _dftu
gate = _gate
moments = _moments
producer_hubbard = _producer_hubbard

CREATED_UTC = "2026-10-16T08:00:00Z"
# Deliberately not alphabetical: SpinIO indices must follow site_order.
SITE_ORDER = ("zeta", "alpha")
AXES = {"1": (1.0, 0.0, 0.0), "2": (0.0, 1.0, 0.0)}
REFERENCES = {site: (0.0, 0.0, 1.0) for site in SITE_ORDER}
SPINIO_KEYS = {
    "exchange_Jdict",
    "dmi_ddict",
    "Jani_dict",
    "debug_dict",
    "colinear",
}


def _json_value(value):
    return json.loads(json.dumps(value))


@pytest.fixture
def exact_response():
    """Authority's -0.1 e_a.e_b pair at a parallel collinear reference.

    docs/sympy/spiral_heisenberg_mapping.py pins the onsite Hessian to
    +0.1, mixed same-axis Hessian to -0.1, and stored ordered-pair J to
    +0.05 eV. These are independent known-energy expectations, not an
    export/mapping implementation used as its own oracle.
    """
    coords = tuple((site, label) for site in SITE_ORDER for label in AXES)
    curvature = {(a, b): 0.0 for a in coords for b in coords}
    for coord in coords:
        curvature[(coord, coord)] = 0.1
    for label in AXES:
        a, b = (SITE_ORDER[0], label), (SITE_ORDER[1], label)
        curvature[(a, b)] = curvature[(b, a)] = -0.1
    response = build_synthetic_response_set(
        site_order=SITE_ORDER,
        axes_by_site={site: dict(AXES) for site in SITE_ORDER},
        gradient={coord: 0.0 for coord in coords},
        curvature=curvature,
        ensemble={
            "kind": ENSEMBLE_GAPPED_T0,
            "n_occupied": 2,
            "electron_count": 2.0,
            "gap_ev": 1.0,
            "synthetic": True,
        },
    )
    return replace(
        response,
        provenance={**response.provenance, "requires_plusu": False},
    )


@pytest.fixture
def exact_mapping(exact_response):
    return map_response_set(exact_response, reference_directions_by_site=REFERENCES)


@pytest.fixture
def incompatible_mapping(exact_response):
    # The mapping authority's site-asymmetric onsite counterexample:
    # a single pair cannot reproduce unequal same-axis onsite responses.
    curvature = dict(exact_response.curvature)
    coord = (SITE_ORDER[0], "1")
    curvature[(coord, coord)] = 0.2
    response = build_synthetic_response_set(
        site_order=exact_response.site_order,
        axes_by_site=exact_response.axes,
        gradient=exact_response.gradient,
        curvature=curvature,
        ensemble=exact_response.ensemble,
    )
    response = replace(
        response,
        provenance={**response.provenance, "requires_plusu": False},
    )
    return map_response_set(response, reference_directions_by_site=REFERENCES)


@pytest.fixture(scope="module")
def certified_displacements(synthetic_bundle):
    """Run the existing certified FD, vertex, and AC-009 public pipeline."""
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    derivatives = DisplacedReferenceFDProvider(
        pattern=pattern,
        plus_sile=_ext_fake_sile(+STEP_BOHR),
        minus_sile=_ext_fake_sile(-STEP_BOHR),
        reference_bundle=synthetic_bundle,
        reference_fermi_ev=-4.5,
    ).provide()
    vertex = vertex_displacement_derivative(derivatives, synthetic_bundle)
    verdict = run_ac009_validation(
        synthetic_bundle,
        pattern,
        plus_sile=_ext_fake_sile(+STEP_BOHR),
        minus_sile=_ext_fake_sile(-STEP_BOHR),
        reference_fermi_ev=-4.5,
    )
    assert verdict["status"] == "pass"
    return {"raw": derivatives, "vertex": vertex, "verdict": verdict}


@pytest.fixture(scope="module")
def failed_ac009(synthetic_bundle, certified_displacements):
    verdict = run_ac009_validation(
        synthetic_bundle,
        certified_displacements["raw"].pattern,
        plus_sile=_ext_fake_sile(+STEP_BOHR),
        minus_sile=_ext_fake_sile(-STEP_BOHR),
        reference_fermi_ev=-4.5,
        tolerance=0.0,
    )
    assert verdict["status"] == "fail"
    return verdict


def test_public_exchange_export_api_is_importable():
    from TB2J.interfaces.siesta_spiral_api import (
        ProvenanceSchema,
        build_exchange_export,
    )

    assert callable(ProvenanceSchema)
    assert callable(build_exchange_export)


# TEST-001: serializable provenance and real upstream/downstream consumers.


def test_provenance_roundtrip_preserves_inputs_revision_time_and_gates():
    provenance = ProvenanceSchema(
        input_digests={"response": "a" * 64, "mapping": "b" * 64},
        code_revision="0123456789abcdef0123456789abcdef01234567",
        code_sha256="c" * 64,
        created_utc=CREATED_UTC,
        gate_verdicts=({"gate": "synthetic-consumed-gate", "passed": True},),
    )
    record = provenance.to_dict()
    assert len(record["content_sha256"]) == 64
    int(record["content_sha256"], 16)
    restored = ProvenanceSchema.from_dict(_json_value(record))
    assert restored.input_digests == provenance.input_digests
    assert restored.code_revision == provenance.code_revision
    assert restored.code_sha256 == provenance.code_sha256
    assert restored.created_utc == CREATED_UTC
    assert isinstance(restored.gate_verdicts, tuple)
    assert restored.gate_verdicts == provenance.gate_verdicts
    assert restored.to_dict() == record


@pytest.mark.parametrize(
    "field",
    [
        "input_digests",
        "code_revision",
        "code_sha256",
        "created_utc",
        "gate_verdicts",
        "content_sha256",
    ],
)
def test_provenance_refuses_tampered_payload(field):
    record = ProvenanceSchema(
        input_digests={"response": "a" * 64},
        code_revision="0123456789abcdef0123456789abcdef01234567",
        code_sha256="c" * 64,
        created_utc=CREATED_UTC,
        gate_verdicts=(),
    ).to_dict()
    record[field] = {
        "input_digests": {"response": "b" * 64},
        "code_revision": "changed-revision",
        "code_sha256": "d" * 64,
        "created_utc": "2026-10-17T08:00:00Z",
        "gate_verdicts": [{"gate": "substituted", "passed": True}],
        "content_sha256": "0" * 64,
    }[field]
    with pytest.raises(
        (ExportRefusal, ValueError), match=r"(?i)(digest|sha256|tamper)"
    ):
        ProvenanceSchema.from_dict(record)


def test_provenance_refuses_missing_content_digest():
    record = ProvenanceSchema(
        input_digests={"response": "a" * 64},
        code_revision="0123456789abcdef0123456789abcdef01234567",
        code_sha256="c" * 64,
        created_utc=CREATED_UTC,
        gate_verdicts=(),
    ).to_dict()
    del record["content_sha256"]
    with pytest.raises(
        (ExportRefusal, ValueError), match=r"(?i)(digest|sha256|tamper)"
    ):
        ProvenanceSchema.from_dict(record)


def test_certified_mapping_constructs_spinio_and_preserves_fr009_metadata(
    exact_response, exact_mapping
):
    raw_before = exact_response.to_dict()
    mapping_before = exact_mapping.to_dict()
    assert exact_mapping.compatibility["label"] == "heisenberg_compatible"

    exported = build_exchange_export(
        response_set=exact_response,
        mapping_output=exact_mapping,
        created_utc=CREATED_UTC,
    )

    assert exported["raw_response"] == raw_before
    assert exported["mapping"] == mapping_before
    assert exported["reference_metadata"]["site_order"] == list(SITE_ORDER)
    assert exported["reference_metadata"]["q"] == raw_before["provenance"]["q"]
    assert (
        exported["reference_metadata"]["frozen_declarations"]
        == (raw_before["frozen_declarations"])
    )
    for field in (
        "integration",
        "ladder",
        "finite_rotation_cross_check",
        "contour_vs_band_max_abs",
    ):
        assert exported["convergence_evidence"][field] == raw_before[field]
    assert exported["units"]["energy"] == "eV"
    assert exported["units"]["gradient"] == "eV/rad"
    assert exported["units"]["torque"] == "eV/rad"
    assert exported["units"]["curvature"] == "eV/rad^2"
    assert exported["vertex_provenance"]["operator_mode"] == (
        exact_response.operator_mode
    )
    assert (
        exported["vertex_provenance"]["vertex_sha256"]
        == (exact_response.provenance["vertex_sha256"])
    )
    assert exported["ensemble"]["response"] == raw_before["ensemble"]

    provenance = ProvenanceSchema.from_dict(exported["provenance"])
    assert provenance.created_utc == CREATED_UTC
    assert provenance.code_revision
    assert len(provenance.code_sha256) == 64
    int(provenance.code_sha256, 16)
    assert len(provenance.input_digests) >= 2
    assert all(len(digest) == 64 for digest in provenance.input_digests.values())

    kwargs = exported["spinio_kwargs"]
    assert SPINIO_KEYS <= kwargs.keys()
    assert kwargs["colinear"] is False
    assert kwargs["exchange_Jdict"] == pytest.approx(
        {((0, 0, 0), 0, 1): 0.05, ((0, 0, 0), 1, 0): 0.05},
        abs=1e-12,
    )
    # Chart-local D/Gamma stay in the mapping, never fabricated 3D tensors.
    assert kwargs["dmi_ddict"] is None
    assert kwargs["Jani_dict"] is None
    pair = exported["mapping"]["parameters"]["zeta|alpha"]
    assert pair["j_pair_once_eV"] == pytest.approx(0.1, abs=1e-12)
    assert pair["gamma_transverse_eV"] is not None
    assert pair["dmi_determined_chart_eV"] is not None

    spinio = SpinIO(
        atoms=Atoms(
            "Fe2",
            positions=((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            cell=(4.0, 4.0, 4.0),
            pbc=True,
        ),
        spinat=np.array(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
        charges=np.array((8.0, 8.0)),
        index_spin=[0, 1],
        **kwargs,
    )
    assert spinio.exchange_Jdict == kwargs["exchange_Jdict"]
    assert spinio.get_J(0, 1, (0, 0, 0)) == pytest.approx(0.05, abs=1e-12)
    assert spinio.get_J(1, 0, (0, 0, 0)) == pytest.approx(0.05, abs=1e-12)
    assert spinio.colinear is False
    assert spinio.debug_dict == kwargs["debug_dict"]
    assert exact_response.to_dict() == raw_before
    assert exact_mapping.to_dict() == mapping_before


def test_response_only_full_export_does_not_invent_exchange(exact_response):
    exported = build_exchange_export(response_set=exact_response)
    assert exported["raw_response"] == exact_response.to_dict()
    assert SPINIO_KEYS <= exported["spinio_kwargs"].keys()
    for name in ("exchange_Jdict", "dmi_ddict", "Jani_dict"):
        assert exported["spinio_kwargs"][name] is None


def test_undetermined_signed_j_is_not_guessed(exact_response):
    mapping = map_response_set(exact_response)
    assert mapping.parameters["zeta|alpha"]["j_stored_eV"] is None
    exported = build_exchange_export(mapping_output=mapping)
    assert exported["raw_response"] == exact_response.to_dict()
    assert exported["mapping"] == mapping.to_dict()
    assert not exported["spinio_kwargs"]["exchange_Jdict"]
    assert exported["spinio_kwargs"]["dmi_ddict"] is None
    assert exported["spinio_kwargs"]["Jani_dict"] is None


@pytest.mark.parametrize("record_kind", ["raw", "vertex"])
def test_displacement_export_preserves_upstream_gauge_and_ac009_evidence(
    certified_displacements, record_kind
):
    source = certified_displacements[record_kind]
    verdict = certified_displacements["verdict"]
    exported = build_exchange_export(
        displacement=source,
        ac009_verdict=verdict,
        created_utc=CREATED_UTC,
    )
    record = exported["displacement"]
    assert exported["raw_response"] is None
    assert record["pattern"] == _json_value(asdict(source.pattern))
    for field in ("gauge", "units", "producer_recipe", "branch", "step_bohr"):
        assert record[field] == _json_value(getattr(source, field))
    assert record["gauge"] == "periodic-image-sum"
    if record_kind == "raw":
        assert record["reference_geometry"] == _json_value(
            asdict(source.reference_geometry)
        )
        assert record["reference_digests"] == list(source.reference_digests)
        assert record["reference_fermi_ev"] == -4.5
        assert len(record["dH"]) == len(source.dH)
        assert sorted(record["dH"].values()) == sorted(
            list(values) for values in source.dH.values()
        )
        assert sorted(record["dS"].values()) == sorted(source.dS.values())
    else:
        assert record["operator_mode"] == source.operator_mode
        assert record["provenance"] == _json_value(source.provenance)
        assert record["partition"] == "fixed"
        assert record["partition_note"] == source.partition_note
        assert record["terms_included"] == list(source.terms_included)
        assert record["terms_excluded"] == list(source.terms_excluded)
        assert sorted(record["dV"].values()) == sorted(
            list(values) for values in source.dV.values()
        )
    provenance = ProvenanceSchema.from_dict(exported["provenance"])
    assert _json_value(verdict) in _json_value(provenance.gate_verdicts)
    assert exported["ensemble"]
    # Electronic operator derivatives are not spin-model exchange derivatives.
    kwargs = exported["spinio_kwargs"]
    assert SPINIO_KEYS <= kwargs.keys()
    for name in ("exchange_Jdict", "dmi_ddict", "Jani_dict", "dJdx"):
        assert kwargs.get(name) is None


def test_genuine_ac008_verdict_is_consumed_without_inventing_an_exchange_fit(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    """Reuse story-016 fixtures, including their genuine-data skip gates."""
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["passed"] is True, verdict["failure_reasons"]
    exported = build_exchange_export(
        response_set=gate["response"],
        plusu_gate=verdict,
        requires_plusu=True,
        created_utc=CREATED_UTC,
    )
    assert exported["raw_response"] == gate["response"].to_dict()
    provenance = ProvenanceSchema.from_dict(exported["provenance"])
    assert _json_value(verdict) in _json_value(provenance.gate_verdicts)
    assert exported["spinio_kwargs"]["exchange_Jdict"] is None


# TEST-002: six distinct actionable refusal classes and a raw-only escape.


@pytest.mark.parametrize("raw_response_only", [False, True])
@pytest.mark.parametrize("missing_from", ["top_level", "provenance", "certificate"])
def test_missing_operator_provenance_is_refused_even_in_raw_mode(
    exact_response, raw_response_only, missing_from
):
    if missing_from == "top_level":
        response = replace(exact_response, operator_mode=None)
    elif missing_from == "provenance":
        provenance = dict(exact_response.provenance)
        del provenance["operator_mode"]
        response = replace(exact_response, provenance=provenance)
    else:
        declarations = deepcopy(exact_response.frozen_declarations)
        del declarations["magnetic_amplitudes"]["certificate"]["operator_mode"]
        response = replace(exact_response, frozen_declarations=declarations)
    with pytest.raises(MissingOperatorProvenance) as exc:
        build_exchange_export(
            response_set=response, raw_response_only=raw_response_only
        )
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "missing_operator_provenance"
    assert "operator" in str(exc.value).lower()


@pytest.mark.parametrize("with_displacement", [False, True])
def test_failed_ac009_gate_is_refused_when_consumed(
    exact_response, certified_displacements, failed_ac009, with_displacement
):
    with pytest.raises(FailedGate) as exc:
        build_exchange_export(
            response_set=exact_response,
            displacement=(
                certified_displacements["vertex"] if with_displacement else None
            ),
            ac009_verdict=failed_ac009,
        )
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "failed_gate"
    assert "AC-009" in str(exc.value)


@pytest.mark.parametrize(
    "failure",
    [
        {"gate": "synthetic-consumed-gate", "passed": False},
        {"gate": "synthetic-consumed-gate", "status": "fail"},
    ],
    ids=["passed-false", "status-fail"],
)
def test_consumed_generic_failing_gate_is_not_ignored(exact_response, failure):
    with pytest.raises(FailedGate) as exc:
        build_exchange_export(
            response_set=exact_response,
            gate_verdicts=({"gate": "earlier-pass", "passed": True}, failure),
        )
    assert exc.value.reason == "failed_gate"
    assert "synthetic-consumed-gate" in str(exc.value)


def test_incompatible_ensemble_names_the_conflicting_declarations(exact_response):
    metallic_response = replace(
        exact_response,
        ensemble={
            **exact_response.ensemble,
            "kind": ENSEMBLE_METALLIC_FD,
            "gap_ev": 0.0,
            "temperature_K": 300.0,
        },
    )
    mapping = map_response_set(
        metallic_response, reference_directions_by_site=REFERENCES
    )
    with pytest.raises(IncompatibleEnsemble) as exc:
        build_exchange_export(response_set=exact_response, mapping_output=mapping)
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "incompatible_ensemble"
    assert "ensemble" in str(exc.value).lower()


def test_non_heisenberg_compatible_response_refuses_exchange_parameters(
    incompatible_mapping,
):
    assert incompatible_mapping.compatibility["label"] == "incompatible"
    with pytest.raises(NonHeisenbergCompatibleResponse) as exc:
        build_exchange_export(mapping_output=incompatible_mapping)
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "non_heisenberg_compatible_response"
    assert "heisenberg" in str(exc.value).lower()


@pytest.mark.parametrize("record_kind", ["raw", "vertex"])
def test_uncertified_displacement_names_required_ac009(
    certified_displacements, record_kind
):
    with pytest.raises(UncertifiedDisplacement) as exc:
        build_exchange_export(displacement=certified_displacements[record_kind])
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "uncertified_displacement"
    assert "AC-009" in str(exc.value)


@pytest.mark.parametrize(
    "plusu_gate",
    [None, {"gate": "AC-008", "passed": False}],
    ids=["required-gate-absent", "supplied-gate-failed"],
)
def test_uncertified_plusu_names_required_ac008(exact_response, plusu_gate):
    with pytest.raises(UncertifiedPlusU) as exc:
        build_exchange_export(
            response_set=exact_response,
            plusu_gate=plusu_gate,
            requires_plusu=plusu_gate is None,
        )
    assert isinstance(exc.value, ExportRefusal)
    assert exc.value.reason == "uncertified_plusu"
    assert "AC-008" in str(exc.value)


@pytest.mark.parametrize("record_kind", ["raw", "vertex"])
def test_raw_mode_preserves_response_but_exports_no_exchange_parameters(
    incompatible_mapping, certified_displacements, record_kind
):
    exported = build_exchange_export(
        response_set=incompatible_mapping.raw_response,
        mapping_output=incompatible_mapping,
        displacement=certified_displacements[record_kind],
        requires_plusu=True,
        raw_response_only=True,
        created_utc=CREATED_UTC,
    )
    assert exported["raw_response"] == incompatible_mapping.raw_response.to_dict()
    assert exported["labels"]["uncertified_exchange_absent"] is True
    assert {
        "spinio_kwargs",
        "mapping",
        "displacement",
        "exchange_Jdict",
        "dmi_ddict",
        "Jani_dict",
        "dJdx",
    }.isdisjoint(exported)
    assert ProvenanceSchema.from_dict(exported["provenance"]).created_utc == (
        CREATED_UTC
    )


def test_altered_vertex_derivative_cannot_reuse_unrelated_ac009_pass(
    certified_displacements,
):
    source = certified_displacements["vertex"]
    key = next(iter(source.dV))
    original = source.dV[key]
    changed = dict(source.dV)
    changed[key] = (original[0] + 1.0, *original[1:])
    substituted = replace(source, dV=changed)

    with pytest.raises(UncertifiedDisplacement) as exc:
        build_exchange_export(
            displacement=substituted,
            ac009_verdict=certified_displacements["verdict"],
        )

    assert exc.value.reason == "uncertified_displacement"
    assert "AC-009" in str(exc.value)
    assert source.dV[key] == original


@pytest.mark.parametrize(
    "change",
    ["dS", "offsite_dH", "scalar_onsite_dH", "units", "geometry", "recipe"],
)
def test_full_seam_payload_cannot_reuse_ac009_pass(certified_displacements, change):
    source = certified_displacements["raw"]
    before = deepcopy(asdict(source))
    if change == "dS":
        values = dict(source.dS)
        key = next(iter(values))
        values[key] += 0.01
        changed = replace(source, dS=values)
    elif change in ("offsite_dH", "scalar_onsite_dH"):
        values = dict(source.dH)
        if change == "offsite_dH":
            key = next(key for key in values if key[0] != key[1] or any(key[2]))
            values[key] = (values[key][0] + 0.01, *values[key][1:])
        else:
            key = next(key for key in values if key[0] == key[1] and not any(key[2]))
            h1, h2, h3, h4 = values[key]
            values[key] = (h1 + 0.01, h2 + 0.01, h3, h4)
        changed = replace(source, dH=values)
    elif change == "units":
        changed = replace(source, units={**source.units, "dH": "Ry/bohr"})
    elif change == "geometry":
        changed = replace(
            source,
            reference_geometry=replace(
                source.reference_geometry, displaced_axis_offset_bohr=0.125
            ),
        )
    else:
        changed = replace(source, producer_recipe="another derivative producer")
    with pytest.raises(UncertifiedDisplacement, match="seam_payload_sha256"):
        build_exchange_export(
            displacement=changed, ac009_verdict=certified_displacements["verdict"]
        )
    assert asdict(source) == before


@pytest.mark.parametrize(
    ("record_kind", "digest_field"),
    [
        ("raw", "seam_payload_sha256"),
        ("vertex", "derivative_payload_sha256"),
    ],
)
def test_ac009_without_complete_payload_digest_is_refused(
    certified_displacements, record_kind, digest_field
):
    verdict = deepcopy(certified_displacements["verdict"])
    del verdict[digest_field]
    with pytest.raises(UncertifiedDisplacement, match=digest_field):
        build_exchange_export(
            displacement=certified_displacements[record_kind], ac009_verdict=verdict
        )


@pytest.mark.parametrize(
    "field",
    [
        "partition_declared",
        "partition",
        "partition_note",
        "terms_included",
        "terms_excluded",
    ],
)
def test_ac009_derivative_declarations_must_match(certified_displacements, field):
    source = certified_displacements["vertex"]
    changed_value = {
        "partition_declared": not source.partition_declared,
        "partition": "moving",
        "partition_note": "different partition observable",
        "terms_included": (*source.terms_included, "new contribution"),
        "terms_excluded": (*source.terms_excluded, "omitted contribution"),
    }[field]
    with pytest.raises(UncertifiedDisplacement, match="declaration"):
        build_exchange_export(
            displacement=replace(source, **{field: changed_value}),
            ac009_verdict=certified_displacements["verdict"],
        )


def test_unknown_operator_identity_requires_explicit_declaration(exact_response):
    provenance = dict(exact_response.provenance)
    del provenance["requires_plusu"]
    response = replace(exact_response, provenance=provenance)
    with pytest.raises(MissingOperatorProvenance, match="requires_plusu"):
        build_exchange_export(response_set=response)
    with pytest.raises(UncertifiedPlusU, match="AC-008"):
        build_exchange_export(response_set=response, requires_plusu=True)
    assert (
        build_exchange_export(response_set=response, requires_plusu=False)[
            "export_kind"
        ]
        == "certified_chain"
    )
    raw = build_exchange_export(response_set=response, raw_response_only=True)
    assert "spinio_kwargs" not in raw


@pytest.mark.parametrize("requires_plusu", [None, False, True])
@pytest.mark.parametrize("vertex_kind", ["hubbard", "xc+hubbard"])
def test_declared_plusu_cannot_opt_out(exact_response, requires_plusu, vertex_kind):
    response = replace(
        exact_response,
        provenance={**exact_response.provenance, "vertex_kind": vertex_kind},
    )
    with pytest.raises(UncertifiedPlusU, match="AC-008"):
        build_exchange_export(response_set=response, requires_plusu=requires_plusu)


@pytest.mark.parametrize("requires_plusu", [None, True])
def test_story016_response_requires_ac008(gate, requires_plusu):
    with pytest.raises(
        (MissingOperatorProvenance, UncertifiedPlusU),
        match="requires_plusu|AC-008",
    ):
        build_exchange_export(
            response_set=gate["response"], requires_plusu=requires_plusu
        )


@pytest.mark.parametrize("field", ["vertex_sha256", "source_bundle_sha256"])
@pytest.mark.parametrize("substitution", ["missing", "other-reference"])
def test_ac008_identity_must_match_response(exact_response, field, substitution):
    identities = {"vertex_sha256": "a" * 64, "source_bundle_sha256": "b" * 64}
    declarations = deepcopy(exact_response.frozen_declarations)
    declarations["magnetic_amplitudes"].update(identities)
    response = replace(
        exact_response,
        provenance={**exact_response.provenance, **identities},
        frozen_declarations=declarations,
    )
    verdict = {
        "gate": "AC-008",
        "passed": True,
        **identities,
    }
    if substitution == "missing":
        del verdict[field]
    else:
        verdict[field] = "0" * 64
    with pytest.raises(UncertifiedPlusU, match=field):
        build_exchange_export(
            response_set=response, plusu_gate=verdict, requires_plusu=True
        )


def test_genuine_ac008_pass_cannot_authorize_another_reference(
    dftu, gate, cross_check, contract_table, audit_artifact, moments
):
    verdict = run_ac008_gate(
        dftu["bundle"],
        cross_check=cross_check,
        contract_table=contract_table,
        ensemble=gate["ensemble"],
        texture_provenance=texture_provenance(dftu, moments),
        audit_artifact=audit_artifact,
    )
    assert verdict["passed"] is True, verdict["failure_reasons"]
    response = gate["response"]
    provenance = {**response.provenance, "source_bundle_sha256": "0" * 64}
    declarations = deepcopy(response.frozen_declarations)
    declarations["magnetic_amplitudes"]["source_bundle_sha256"] = "0" * 64
    another_reference = replace(
        response, provenance=provenance, frozen_declarations=declarations
    )
    with pytest.raises(UncertifiedPlusU, match="source_bundle_sha256"):
        build_exchange_export(
            response_set=another_reference, plusu_gate=verdict, requires_plusu=True
        )


@pytest.mark.parametrize(
    ("field", "other_value"),
    [
        ("declaration", "different occupation and free-energy convention"),
        ("n_occupied", 3),
        ("electron_count", 3.0),
        ("gap_ev", 2.0),
    ],
)
def test_ensemble_producer_fields_are_compared(exact_response, field, other_value):
    response = replace(
        exact_response,
        ensemble=Ensemble(ENSEMBLE_GAPPED_T0, 2, 2.0, 0.0, 0.0, 1.0).to_record(),
    )
    verdict = {
        "gate": "matched-ensemble-evidence",
        "passed": True,
        "ensemble": {**response.ensemble, field: other_value},
    }
    with pytest.raises(IncompatibleEnsemble, match=field):
        build_exchange_export(response_set=response, gate_verdicts=(verdict,))


@pytest.mark.parametrize(
    "value", ["", "a" * 40, "g" * 64, "+" + "a" * 63, " " + "a" * 63, None]
)
def test_code_sha256_requires_64_hex_digits(value):
    with pytest.raises(ValueError, match="code_sha256"):
        ProvenanceSchema(
            input_digests={},
            code_revision="git-revision",
            code_sha256=value,
            created_utc=CREATED_UTC,
        )
