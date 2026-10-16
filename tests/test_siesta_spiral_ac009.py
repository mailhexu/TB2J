"""Story 019: AC-009 validation — displaced-reference cross-checks and
gauge labeling.

Two independent routes on the SAME displaced +h/-h pair (derivative-based
dV/dR vs direct FD of the extracted vertex blocks) must agree within the
pinned 1e-10 eV/Ang pipeline-consistency tolerance; every output carries
an explicit gauge label, and a localized-gauge request from periodic
sampling is refused with the required sampling named (FR-012).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_siesta_spiral_displacement import (
    _BASE_H,
    _BASE_S,
    BOHR_TO_ANG,
    CELL,
    _FakeHsxFourComponent,
)
from test_siesta_spiral_displacement import (
    ALPHA as _ALPHA,
)

from TB2J.interfaces.siesta_spiral_displacement import (
    DisplacementPattern,
    GaugeMismatch,
    displacement_payload_sha256,
)
from TB2J.interfaces.siesta_spiral_vertex_displacement import (
    AC009_TOLERANCE_EV_PER_ANG,
    LocalizedGaugeUnavailable,
    run_ac009_validation,
    vertex_displacement_derivative,
)

STEP_BOHR = 0.05

# the certified loader's bundle validates overlap pairing (nu, mu, -R); the
# story-017 synthetic fixture omits the -R partners, so extend the model with
# the hermitian partners (needed here because route (b) runs the certified
# extraction, which validates the bundle)
_PARTNERED_KEYS = ((0, 0, (1, 0, 0)), (1, 1, (1, 0, 0)))
_EXT_H = dict(_BASE_H)
_EXT_S = dict(_BASE_S)
_EXT_ALPHA = dict(_ALPHA)
for _key0 in _PARTNERED_KEYS:
    _key = (_key0[0], _key0[1], tuple(-r for r in _key0[2]))
    _EXT_H[_key] = _BASE_H[_key0]
    _EXT_S[_key] = _BASE_S[_key0]
    _EXT_ALPHA[_key] = _ALPHA[_key0]


def _ext_fake_sile(z_atom1_bohr):
    h = {}
    for key, base in _EXT_H.items():
        a = _EXT_ALPHA[key]
        h[key] = tuple(
            -(base[i] + a[i] * z_atom1_bohr * BOHR_TO_ANG)
            if i == 3
            else (base[i] + a[i] * z_atom1_bohr * BOHR_TO_ANG)
            for i in range(4)
        )
    s = {
        key: (v[0] + _EXT_ALPHA[key][0] * 0.0, 0.0, 0.0, 0.0)
        for key, v in _EXT_S.items()
    }
    return _FakeHsxFourComponent(
        CELL,
        ((0.0, 0.0, z_atom1_bohr), (0.0, 0.0, 3.0)),
        # ordering matters for the loader's 1D cell decoding: iscell=-1
        # (the -R partner column) indexes sc_off[-1]
        ((0, 0, 0), (1, 0, 0), (-1, 0, 0)),
        h,
        s,
        0.0,
    )


@pytest.fixture(scope="module")
def synthetic_bundle():
    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    b = _bundle_from_hs_geometry(_ext_fake_sile(0.0), "synthetic", "")
    return replace(
        b, metadata=dict(b.metadata, displacement_gauge="periodic-image-sum")
    )


def _run(synthetic_bundle, **kw):
    pattern = DisplacementPattern(
        atom_index=kw.pop("atom_index", 1),
        axis=kw.pop("axis", "z"),
        step_bohr=STEP_BOHR,
    )
    return pattern, run_ac009_validation(
        synthetic_bundle,
        pattern,
        plus_sile=_ext_fake_sile(+STEP_BOHR),
        minus_sile=_ext_fake_sile(-STEP_BOHR),
        reference_fermi_ev=-4.5,
        **kw,
    )


# ---------------------------------------------------------------------------
# TEST-001 (synthetic): the two routes agree within the pinned tolerance
# ---------------------------------------------------------------------------


def test_routes_agree_within_pinned_tolerance(synthetic_bundle):
    pattern, verdict = _run(synthetic_bundle)
    assert verdict["status"] == "pass"
    assert verdict["max_deviation_ev_per_ang"] <= AC009_TOLERANCE_EV_PER_ANG
    assert verdict["pattern"]["atom_index"] == 1
    assert verdict["pattern"]["axis"] == "z"
    # every onsite block, all four components compared
    assert verdict["agreement"]
    for entry in verdict["agreement"].values():
        assert len(entry["derived"]) == 4
        assert len(entry["reference_fd"]) == 4
        assert len(entry["deviation"]) == 4
    # JSON-serializable
    json.dumps(verdict)


@pytest.fixture
def certified_payloads(synthetic_bundle, monkeypatch):
    """Capture the actual route-(a) objects before the producer stamps them."""
    from TB2J.interfaces import siesta_spiral_vertex_displacement as producer

    records = {}
    provide = producer._DisplacedReferenceFDProvider.provide
    construct = producer.vertex_displacement_derivative

    def capture_seam(provider):
        records["seam"] = provide(provider)
        return records["seam"]

    def capture_vertex(*args, **kwargs):
        records["vertex"] = construct(*args, **kwargs)
        return records["vertex"]

    monkeypatch.setattr(producer._DisplacedReferenceFDProvider, "provide", capture_seam)
    monkeypatch.setattr(producer, "vertex_displacement_derivative", capture_vertex)
    _, records["verdict"] = _run(synthetic_bundle)
    return records


def test_verdict_binds_both_complete_constructed_payloads(certified_payloads):
    verdict = certified_payloads["verdict"]
    assert verdict["status"] == "pass"
    assert verdict["seam_payload_sha256"] == displacement_payload_sha256(
        certified_payloads["seam"]
    )
    assert verdict["derivative_payload_sha256"] == displacement_payload_sha256(
        certified_payloads["vertex"]
    )
    assert verdict["seam_payload_sha256"] != verdict["derivative_payload_sha256"]


@pytest.mark.parametrize("component", ("dS", "offsite_dH", "scalar_onsite_dH"))
def test_seam_digest_detects_components_outside_vertex_comparison(
    certified_payloads, synthetic_bundle, component
):
    seam = certified_payloads["seam"]
    if component == "dS":
        changed = dict(seam.dS)
        key = next(iter(changed))
        changed[key] += 0.5
        modified = replace(seam, dS=changed)
    else:
        changed = dict(seam.dH)
        key = next(
            key
            for key in changed
            if (key[2] != (0, 0, 0)) == (component == "offsite_dH")
        )
        values = list(changed[key])
        # A scalar onsite shift cancels from (dH1-dH2)/2; offsite dH
        # never enters dV. Both nevertheless belong to the certified seam.
        values[0] += 0.5
        values[1] += 0.5
        changed[key] = tuple(values)
        modified = replace(seam, dH=changed)
    changed_vertex = vertex_displacement_derivative(modified, synthetic_bundle)
    for key, values in certified_payloads["vertex"].dV.items():
        assert changed_vertex.dV[key] == pytest.approx(values, abs=1e-14)
    assert (
        displacement_payload_sha256(modified)
        != (certified_payloads["verdict"]["seam_payload_sha256"])
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "pattern",
        "dH",
        "dS",
        "producer_recipe",
        "branch",
        "step_bohr",
        "reference_fermi_ev",
        "gauge",
        "units",
        "reference_digests",
        "reference_geometry",
    ),
)
def test_seam_digest_binds_each_exported_field(certified_payloads, field_name):
    seam = certified_payloads["seam"]
    changes = {
        "pattern": replace(seam.pattern, reference_digests=("changed",)),
        "dH": {**seam.dH, (9, 9, (0, 0, 0)): (1.0, 1.0, 0.0, 0.0)},
        "dS": {**seam.dS, (9, 9, (0, 0, 0)): 0.0},
        "producer_recipe": seam.producer_recipe + " changed",
        "branch": "changed",
        "step_bohr": seam.step_bohr * 2,
        "reference_fermi_ev": 123.0,
        "gauge": "localized",
        "units": dict(seam.units, dS="changed"),
        "reference_digests": ("changed",),
        "reference_geometry": replace(
            seam.reference_geometry, displaced_axis_offset_bohr=123.0
        ),
    }
    modified = replace(seam, **{field_name: changes[field_name]})
    assert (
        displacement_payload_sha256(modified)
        != (certified_payloads["verdict"]["seam_payload_sha256"])
    )


@pytest.mark.parametrize(
    "field_name",
    ("cell", "orbital_positions", "displaced_orbital_indices"),
)
def test_seam_digest_binds_nested_geometry(certified_payloads, field_name):
    seam = certified_payloads["seam"]
    geometry = seam.reference_geometry
    changes = {
        "cell": ((123.0, 0.0, 0.0), *geometry.cell[1:]),
        "orbital_positions": ((123.0, 0.0, 0.0), *geometry.orbital_positions[1:]),
        "displaced_orbital_indices": (123,),
    }
    modified = replace(
        seam, reference_geometry=replace(geometry, **{field_name: changes[field_name]})
    )
    assert (
        displacement_payload_sha256(modified)
        != (certified_payloads["verdict"]["seam_payload_sha256"])
    )


def test_vertex_digest_binds_all_dv_values(certified_payloads):
    vertex = certified_payloads["vertex"]
    for key, values in vertex.dV.items():
        for component in range(4):
            changed_values = list(values)
            changed_values[component] += 0.5
            changed = dict(vertex.dV)
            changed[key] = tuple(changed_values)
            assert (
                displacement_payload_sha256(replace(vertex, dV=changed))
                != (certified_payloads["verdict"]["derivative_payload_sha256"])
            )


@pytest.mark.parametrize(
    "field_name",
    (
        "pattern",
        "dV",
        "partition",
        "partition_note",
        "terms_included",
        "terms_excluded",
        "operator_mode",
        "partition_declared",
        "gauge",
        "units",
        "producer_recipe",
        "step_bohr",
        "branch",
        "provenance",
    ),
)
def test_vertex_digest_binds_each_exported_field(certified_payloads, field_name):
    vertex = certified_payloads["vertex"]
    changes = {
        "pattern": replace(vertex.pattern, reference_digests=("changed",)),
        "dV": {**vertex.dV, (9, 9, (0, 0, 0)): (0.0, 0.0, 0.0, 0.0)},
        "partition": "changed",
        "partition_note": vertex.partition_note + " changed",
        "terms_included": (*vertex.terms_included, "changed"),
        "terms_excluded": (*vertex.terms_excluded, "changed"),
        "operator_mode": "changed",
        "partition_declared": not vertex.partition_declared,
        "gauge": "localized",
        "units": dict(vertex.units, dV="changed"),
        "producer_recipe": vertex.producer_recipe + " changed",
        "step_bohr": vertex.step_bohr * 2,
        "branch": "changed",
        "provenance": dict(vertex.provenance, changed=True),
    }
    modified = replace(vertex, **{field_name: changes[field_name]})
    assert (
        displacement_payload_sha256(modified)
        != (certified_payloads["verdict"]["derivative_payload_sha256"])
    )


def test_payload_digest_is_mapping_order_independent(certified_payloads):
    seam = certified_payloads["seam"]
    reordered = replace(
        seam,
        dH=dict(reversed(tuple(seam.dH.items()))),
        dS=dict(reversed(tuple(seam.dS.items()))),
        units=dict(reversed(tuple(seam.units.items()))),
    )
    assert (
        displacement_payload_sha256(reordered)
        == (certified_payloads["verdict"]["seam_payload_sha256"])
    )


def test_payload_digest_refuses_unsupported_values(certified_payloads):
    from TB2J.interfaces.siesta_spiral_hubbard import HubbardBundleError

    vertex = certified_payloads["vertex"]
    modified = replace(vertex, provenance=dict(vertex.provenance, unsupported=object()))
    with pytest.raises(HubbardBundleError, match="cannot be digested canonically"):
        displacement_payload_sha256(modified)


def test_failed_gate_returns_evidence_bearing_fail_verdict(synthetic_bundle):
    """Zero tolerance: the honest fail path, never a fabricated pass."""
    _, verdict = _run(synthetic_bundle, tolerance=0.0)
    assert verdict["status"] == "fail"
    assert verdict["max_deviation_ev_per_ang"] > 0.0
    assert verdict["worst_component"] is not None
    json.dumps(verdict)


# ---------------------------------------------------------------------------
# TEST-002: gauge labels on every output + FR-012 localized refusal
# ---------------------------------------------------------------------------


def test_gauge_labels_on_seam_record_vertex_record_and_verdict(
    synthetic_bundle,
):
    from TB2J.interfaces.siesta_spiral_displacement import (
        DisplacedReferenceFDProvider,
    )

    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    provider = DisplacedReferenceFDProvider(
        pattern=pattern,
        plus_sile=_ext_fake_sile(+STEP_BOHR),
        minus_sile=_ext_fake_sile(-STEP_BOHR),
        reference_bundle=synthetic_bundle,
        reference_fermi_ev=-4.5,
    )
    derivatives = provider.provide()
    # the seam record itself
    assert derivatives.gauge == "periodic-image-sum"
    _, verdict = _run(synthetic_bundle)
    gauges = verdict["gauge"]
    assert gauges["requested"] == "periodic-image-sum"
    assert gauges["seam_record"] == "periodic-image-sum"
    assert gauges["vertex_derivative_record"] == "periodic-image-sum"
    assert gauges["displaced_reference_route"] == "periodic-image-sum"


def test_localized_gauge_request_refused_with_required_sampling_named(
    synthetic_bundle,
):
    from TB2J.interfaces.siesta_spiral_displacement import DisplacementPattern

    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    with pytest.raises(LocalizedGaugeUnavailable) as exc:
        run_ac009_validation(
            synthetic_bundle,
            pattern,
            plus_sile=_ext_fake_sile(+STEP_BOHR),
            minus_sile=_ext_fake_sile(-STEP_BOHR),
            gauge="localized",
        )
    message = str(exc.value)
    # FR-012: the refusal names the required sampling
    assert "localized-gauge displaced set" in message
    assert "dP" in message
    assert isinstance(exc.value, GaugeMismatch)


def test_unknown_gauge_refused(synthetic_bundle):
    from TB2J.interfaces.siesta_spiral_displacement import DisplacementPattern

    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    with pytest.raises(GaugeMismatch):
        run_ac009_validation(
            synthetic_bundle,
            pattern,
            plus_sile=_ext_fake_sile(+STEP_BOHR),
            minus_sile=_ext_fake_sile(-STEP_BOHR),
            gauge="bloch-summed",
        )


def test_bundle_gauge_mismatch_refused():
    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    b = _bundle_from_hs_geometry(_ext_fake_sile(0.0), "synthetic", "")
    b = replace(b, metadata=dict(b.metadata, displacement_gauge="localized"))
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    with pytest.raises(GaugeMismatch):
        run_ac009_validation(
            b,
            pattern,
            plus_sile=_ext_fake_sile(+STEP_BOHR),
            minus_sile=_ext_fake_sile(-STEP_BOHR),
        )


# ---------------------------------------------------------------------------
# Ensemble/alignment bookkeeping recorded verbatim (ADR-005)
# ---------------------------------------------------------------------------


def test_ensemble_bookkeeping_recorded_verbatim(synthetic_bundle):
    _, verdict = _run(synthetic_bundle)
    ensemble = verdict["ensemble"]
    assert "INDEPENDENT" in ensemble["ensemble_declaration"]
    assert "no" in ensemble["ensemble_declaration"]
    assert "cross-SCF energy" in ensemble["ensemble_declaration"]
    assert "-Ef" in ensemble["energy_alignment"]
    assert "fermi_shift='none'" in ensemble["energy_alignment"]
    assert "central" in ensemble["electronic_branch"]
    assert "periodic-image-sum" in ensemble["perturbation_definition"]
    assert verdict["geometry"]["displaced_axis_offset_bohr"] == pytest.approx(STEP_BOHR)


# ---------------------------------------------------------------------------
# TEST-001 (real campaign, skip-gated): fe4 displaced campaign
# ---------------------------------------------------------------------------


def _patterned_sile(z_atom1_bohr, *, extra_keys=(), missing_keys=()):
    """A displaced fake sile whose sparse pattern deviates on purpose."""
    h, s = {}, {}
    for key, base in _EXT_H.items():
        if key in missing_keys:
            continue
        a = _EXT_ALPHA[key]
        h[key] = tuple(
            -(base[i] + a[i] * z_atom1_bohr * BOHR_TO_ANG)
            if i == 3
            else (base[i] + a[i] * z_atom1_bohr * BOHR_TO_ANG)
            for i in range(4)
        )
        s[key] = (_EXT_S[key][0], 0.0, 0.0, 0.0)
    for key in extra_keys:
        h[key] = (0.0, 0.0, 0.0, 0.0)
        s[key] = (1.0, 0.0, 0.0, 0.0)
    return _FakeHsxFourComponent(
        CELL,
        ((0.0, 0.0, z_atom1_bohr), (0.0, 0.0, 3.0)),
        ((0, 0, 0), (1, 0, 0), (-1, 0, 0)),
        h,
        s,
        0.0,
    )


def test_extra_displaced_sparse_key_refused(synthetic_bundle):
    """A displaced pair carrying an EXTRA sparse key is refused: route (b)
    must match the reference bundle's pattern exactly, both directions,
    the same contract assemble_onto_bundle enforces."""
    from TB2J.interfaces.siesta_spiral_displacement import SparsePatternMismatch

    extra = (1, 0, (1, 0, 0))
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    with pytest.raises(SparsePatternMismatch) as exc:
        run_ac009_validation(
            synthetic_bundle,
            pattern,
            plus_sile=_patterned_sile(+STEP_BOHR, extra_keys=(extra,)),
            minus_sile=_patterned_sile(-STEP_BOHR, extra_keys=(extra,)),
            reference_fermi_ev=-4.5,
        )
    msg = str(exc.value)
    assert "extra keys" in msg and "route (b)" in msg
    assert str(extra) in msg


def test_missing_displaced_sparse_key_refused(synthetic_bundle):
    """A displaced pair MISSING a reference sparse key is refused with the
    missing keys named."""
    from TB2J.interfaces.siesta_spiral_displacement import SparsePatternMismatch

    missing = (0, 0, (1, 0, 0))
    pattern = DisplacementPattern(atom_index=1, axis="z", step_bohr=STEP_BOHR)
    with pytest.raises(SparsePatternMismatch) as exc:
        run_ac009_validation(
            synthetic_bundle,
            pattern,
            plus_sile=_patterned_sile(+STEP_BOHR, missing_keys=(missing,)),
            minus_sile=_patterned_sile(-STEP_BOHR, missing_keys=(missing,)),
            reference_fermi_ev=-4.5,
        )
    msg = str(exc.value)
    assert "missing keys" in msg and "route (b)" in msg
    assert str(missing) in msg


CAMPAIGN = Path(
    "/home/hexu/projects/TB2J_dev/.tmp/siesta-xc-mft/fe4_displaced_campaign"
)
VAULT_FE4 = Path(
    "/home/hexu/projects/memnotes/memnotes/Projects/TB2J/evidence/"
    "2026-10-15-siesta-finite-pitch-mapping/fe4"
)

REAL_MODES = (
    ("a1", "z"),
    ("a3", "z"),
    ("a1", "x"),
)


def _real_bundle():
    import sisl

    from TB2J.interfaces.siesta_spiral_hubbard import _bundle_from_hs_geometry

    reference = _bundle_from_hs_geometry(
        sisl.get_sile(str(VAULT_FE4 / "fe4.HSX")), "fe4.out", "Fe_nc.ion.nc"
    )
    return replace(
        reference,
        metadata=dict(reference.metadata, displacement_gauge="periodic-image-sum"),
    )


@pytest.mark.skipif(
    not CAMPAIGN.is_dir() or not (VAULT_FE4 / "fe4.HSX").is_file(),
    reason="fe4 displaced campaign or vault reference absent",
)
@pytest.mark.parametrize("atom_tag,axis", REAL_MODES)
def test_real_fe4_routes_agree(atom_tag, axis):
    bundle = _real_bundle()
    atom_index = int(atom_tag[1:])
    pattern = DisplacementPattern(atom_index=atom_index, axis=axis, step_bohr=STEP_BOHR)
    verdict = run_ac009_validation(
        bundle,
        pattern,
        plus_path=str(CAMPAIGN / f"{atom_tag}{axis}p" / f"fe4d_{atom_tag}{axis}p.HSX"),
        minus_path=str(CAMPAIGN / f"{atom_tag}{axis}m" / f"fe4d_{atom_tag}{axis}m.HSX"),
    )
    assert verdict["status"] == "pass", verdict["max_deviation_ev_per_ang"]
    assert verdict["max_deviation_ev_per_ang"] <= AC009_TOLERANCE_EV_PER_ANG
    assert verdict["gauge"]["vertex_derivative_record"] == "periodic-image-sum"
    print(
        f"\n[real fe4 {atom_tag}{axis}] route(a) vs route(b) max deviation = "
        f"{verdict['max_deviation_ev_per_ang']:.3e} eV/Ang over "
        f"{len(verdict['agreement'])} onsite blocks"
    )
