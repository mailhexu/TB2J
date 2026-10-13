"""Story-011 same-SCF q=0 anchor tests: numeric-only ExchangeCL2 comparison.

The driver splits x-collinear spin channels from the *same* noncollinear O2
fixture bundle that feeds the repaired frozen-response core, runs the
established LKAG engine on those channels, and reports the image-summed
site-pair J values and their signed difference as numbers. Per the user
decision of 2026-10-13 there is **no agreement threshold**: the tests assert
that numbers are computed, that raw responses are preserved, that the
non-identical operator/integration settings are reported unmatched, and that
the same-SCF spectrum equivalence actually holds numerically -- never that
the two engines agree.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

from TB2J.interfaces.siesta_spiral import SpiralImporter
from TB2J.interfaces.siesta_spiral_anchor import (
    AXIS_SYMMETRY_BUDGET_EV,
    COLLINEAR_TOL_EV,
    GRADIENT_STATIONARITY_BUDGET_EV_PER_RAD,
    SPECTRAL_EQUIVALENCE_BUDGET_EV,
    AnchorDeclaration,
    AnchorError,
    AnchorSettingMismatchError,
    apply_declaration,
    build_collinear_spin_channels,
    channel_pencil,
    image_summed_pair_J,
    run_anchor_comparison,
    spectral_equivalence_check,
)
from TB2J.interfaces.siesta_spiral_io import load_fixture
from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex

FIXTURE_DIR = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_collinear_o2"
FIXTURE_NPZ = FIXTURE_DIR / "o2_fm_q0.npz"
FIXTURE_FDF = FIXTURE_DIR / "nc" / "o2_fm.fdf"
SITES = ("O1", "O2")
AXES = ("y", "z")

#: The pre-repair exploratory number; the repaired response changes it, so
#: no test may pin it anywhere.
STALE_PRE_REPAIR_J_EV = 0.311528935


@pytest.fixture(scope="module")
def bundle_and_reference():
    return load_fixture(FIXTURE_NPZ)


@pytest.fixture(scope="module")
def comparison():
    pytest.importorskip("sisl")
    pytest.importorskip("HamiltonIO")
    return run_anchor_comparison(FIXTURE_NPZ, FIXTURE_FDF)


# ---------------------------------------------------------------------------
# channel split from the same NC bundle (light, no legacy engine)
# ---------------------------------------------------------------------------
def test_channels_are_x_collinear_and_fermi_shifted(bundle_and_reference):
    bundle, _reference = bundle_and_reference
    channels = build_collinear_spin_channels(bundle)
    assert channels.norb == bundle.norb
    assert channels.max_diag_split_ev <= COLLINEAR_TOL_EV
    assert channels.max_mixing_imag_ev <= COLLINEAR_TOL_EV
    # the -Ef*S shift is applied on the R=0 block of both channels: rebuild
    # the unshifted channels from the raw bundle and compare
    r0 = channels.rlist.index((0, 0, 0))
    ef = float(bundle.metadata["ef_ev"])
    s0 = channels.overlap[r0]
    raw_up = np.zeros_like(channels.h_up[r0])
    raw_dn = np.zeros_like(channels.h_dn[r0])
    for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
        if R == (0, 0, 0):
            raw_up[mu, nu] += 0.5 * (h1 + h2) + h3
            raw_dn[mu, nu] += 0.5 * (h1 + h2) - h3
    np.testing.assert_allclose(channels.h_up[r0], raw_up - ef * s0, atol=1e-12)
    np.testing.assert_allclose(channels.h_dn[r0], raw_dn - ef * s0, atol=1e-12)


def test_channel_split_refuses_non_x_collinear_texture(bundle_and_reference):
    bundle, _reference = bundle_and_reference
    corrupted = copy.deepcopy(bundle)
    (mu, nu, R), comps = next(iter(corrupted.sparse.items()))
    partner = (nu, mu, corrupted._neg(R))
    corrupted.sparse[(mu, nu, R)] = (
        comps[0],
        comps[1],
        comps[2],
        comps[3] + 0.05,
    )
    p = corrupted.sparse[partner]
    corrupted.sparse[partner] = (p[0], p[1], p[2], p[3] + 0.05)
    with pytest.raises(AnchorError, match="x-collinear"):
        build_collinear_spin_channels(corrupted)


def test_channel_split_refuses_offzero_pitch(bundle_and_reference):
    bundle, _reference = bundle_and_reference
    pitched = copy.deepcopy(bundle)
    pitched.q = (0.01, 0.0, 0.0)
    with pytest.raises(AnchorError, match="zero-pitch"):
        build_collinear_spin_channels(pitched)


def test_plain_orbital_labels_strip_every_second_spin_suffix():
    from TB2J.interfaces.siesta_spiral_anchor import _plain_orbital_labels

    labels = [
        "O1|2sZ1|up",
        "O1|2sZ1|down",
        "O2|2pZ1|down",
        "O2|2pZ1|up",
    ]
    assert _plain_orbital_labels(labels) == ["O1|2sZ1", "O2|2pZ1"]
    with pytest.raises(AnchorError, match="spin"):
        _plain_orbital_labels(["O1|2sZ1|up", "O1|2sZ1|up"])
    with pytest.raises(AnchorError, match="spin"):
        _plain_orbital_labels(["O1|2sZ1", "O1|2sZ1|down"])


def test_spectral_equivalence_is_actually_numeric(bundle_and_reference):
    """Union of channel spectra equals the NC spinor pencil, as numbers."""
    bundle, reference = bundle_and_reference
    report = spectral_equivalence_check(bundle, reference["kp"])
    assert report["n_kpoints"] == len(reference["kp"])
    assert report["pass"] is True
    assert report["max_dev_ev"] <= SPECTRAL_EQUIVALENCE_BUDGET_EV

    # independent recomputation, not trusting the helper's own verdict
    importer = SpiralImporter()
    up = {k: 0.5 * (c[0] + c[1]) + c[2] for k, c in bundle.sparse.items()}
    dn = {k: 0.5 * (c[0] + c[1]) - c[2] for k, c in bundle.sparse.items()}
    worst = 0.0
    for k in reference["kp"]:
        h1, s1 = channel_pencil(up, bundle, k)
        h2, s2 = channel_pencil(dn, bundle, k)
        union = np.concatenate(
            (eigh(h1, s1, eigvals_only=True), eigh(h2, s2, eigvals_only=True))
        )
        hs, ss = importer.assemble_pencil(bundle, k)
        spinor = eigh(hs, ss, eigvals_only=True)
        worst = max(
            worst,
            float(np.max(np.abs(np.sort(union) - np.sort(spinor)))),
        )
    assert worst <= SPECTRAL_EQUIVALENCE_BUDGET_EV
    assert worst == report["max_dev_ev"]


# ---------------------------------------------------------------------------
# full same-SCF comparison (heavy fixture run, executed once)
# ---------------------------------------------------------------------------
def test_certificate_is_json_safe_and_same_scf(comparison, bundle_and_reference):
    payload = json.loads(json.dumps(comparison.to_dict(), allow_nan=False))
    assert payload["anchor_version"] == comparison.certificate["anchor_version"]

    bundle, _reference = bundle_and_reference
    n = bundle.norb
    vertex = extract_magnetic_vertex(
        bundle,
        orbitals_by_site={"O1": list(range(n // 2)), "O2": list(range(n // 2, n))},
    )
    assert (
        payload["same_scf"]["source_bundle_sha256"]
        == vertex.certificate["source_bundle_sha256"]
    )
    assert payload["same_scf"]["channel_split"]["ef_shift_ev"] == pytest.approx(
        float(bundle.metadata["ef_ev"])
    )
    assert payload["same_scf"]["geometry_checks"]["cell_max_diff_ang"] <= 1e-4
    assert payload["same_scf"]["geometry_checks"]["all_bundle_shells_in_hsx_supercells"]


def test_legacy_pair_sum_is_one_directed_with_reciprocity(comparison):
    """One-directed R-sum is the stored pair value; reverse audits only.

    Real oracle: the established engine is untouched by the joint-rotation
    repair, so its one-directed image sum must reproduce the recorded
    pre-repair value ~311.637 meV -- a factor-2 (both-directions) mistake
    or a convention flip cannot pass.
    """
    from TB2J.utils import symbol_number

    ex = comparison.legacy_exchange
    record = image_summed_pair_J(ex, *SITES)
    total = record["forward_sum_ev"]
    # ~311.637 meV recorded pre-repair (nz=200-300 spread); budget 0.1 meV
    assert total == pytest.approx(0.311637068, abs=1e-4)
    # exactly the one directed (O1, O2): no reverse entries summed in
    forward_entries = record["forward_entries"]
    assert forward_entries
    assert all(e["i"] == SITES[0] and e["j"] == SITES[1] for e in forward_entries)
    keys = [tuple(e["R"]) for e in forward_entries]
    assert len(set(keys)) == len(keys)
    names = list(symbol_number(ex.atoms).keys())
    manual = 0.0
    stored = set()
    for key, value in ex.exchange_Jdict.items():
        R = tuple(int(v) for v in np.asarray(key[0]).ravel())
        site_i = names[ex.ind_mag_atoms[key[1]]]
        site_j = names[ex.ind_mag_atoms[key[2]]]
        stored.add((R, site_i, site_j))
        if (site_i, site_j) == SITES:
            manual += float(value)
    assert manual == pytest.approx(total, abs=0.0, rel=1e-12)
    # reciprocity: every forward bond has the reflected reverse entry with
    # the same stored value; the reverse sum equals but is never added
    reverse_entries = record["reverse_entries"]
    assert reverse_entries
    assert all(e["i"] == SITES[1] and e["j"] == SITES[0] for e in reverse_entries)
    assert record["reciprocity_complete"] is True
    assert record["reciprocity_max_abs_diff_ev"] == pytest.approx(0.0, abs=1e-12)
    assert record["reverse_sum_ev"] == pytest.approx(total, abs=1e-12)
    for entry in forward_entries:
        mirrored = (tuple(-v for v in entry["R"]), entry["j"], entry["i"])
        assert mirrored in stored
    assert total == comparison.legacy_pair_sum_ev
    cert = comparison.certificate["legacy_response"]
    assert cert["pair_image_sum_eV"] == total
    assert cert["reverse_direction_sum_eV"] == pytest.approx(total, abs=1e-12)
    assert cert["reciprocity_complete"] is True


def test_numeric_comparison_numbers_only_and_consistent(comparison):
    cert = comparison.certificate
    assert cert["agreement_status"] == "not_evaluated"
    numeric = cert["numeric_comparison"]
    forbidden = ("pass", "agreement", "matched", "threshold", "compatible")

    def _no_verdicts(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not any(word in key for word in forbidden), key
                _no_verdicts(value)
        elif isinstance(node, list):
            for item in node:
                _no_verdicts(item)

    _no_verdicts(numeric)
    pair_key = f"{SITES[0]}|{SITES[1]}"
    for axis in AXES:
        record = numeric["per_axis"][axis]
        new_j = float(comparison.j_stored_by_axis[axis][pair_key]["j_stored_eV"])
        assert record["j_stored_new_eV"] == new_j
        assert record["j_stored_legacy_image_sum_eV"] == comparison.legacy_pair_sum_ev
        assert record["difference_legacy_minus_new_eV"] == (
            comparison.legacy_pair_sum_ev - new_j
        )
        # ferromagnetic fixture: both engines report a positive stored J
        assert new_j > 0.0
        assert record["j_stored_legacy_image_sum_eV"] > 0.0
    # the difference is a computed number, not pinned to the pre-repair value
    for axis in AXES:
        assert numeric["per_axis"][axis]["j_stored_new_eV"] != STALE_PRE_REPAIR_J_EV


def test_transverse_axes_agree_and_raw_response_is_preserved(comparison):
    cert = comparison.certificate
    spread = cert["numeric_comparison"]["axis_symmetry_spread_new_eV"]
    assert spread is not None and spread <= AXIS_SYMMETRY_BUDGET_EV
    for axis in AXES:
        obs = comparison.response_by_axis[axis]
        # stationarity of the collinear reference under transverse tilts
        assert max(abs(v) for v in obs["gradient"].values()) <= (
            GRADIENT_STATIONARITY_BUDGET_EV_PER_RAD
        )
        # the raw Hessian keeps its onsite/response split verbatim
        for pair, curvature in obs["curvature"].items():
            rebuilt = obs["onsite"][pair] + obs["response"][pair]
            assert curvature == pytest.approx(rebuilt, abs=1e-12, rel=0.0)
        # repaired weighted rule: cross-site second vertex is exactly zero
        cross = ((SITES[0], axis), (SITES[1], axis))
        assert obs["onsite"][cross] == 0.0
        assert obs["response"][cross] != 0.0
        # serialized raw response matches the in-memory one
        raw = cert["new_response_raw"][axis]
        assert raw["gradient"] == {
            json.dumps([c[0], c[1]]): float(v) for c, v in obs["gradient"].items()
        }


def test_ensemble_is_gapped_fixed_occupations(comparison):
    record = comparison.certificate["ensemble_record"]
    assert record["kind"] == "gapped_fixed_occupations"
    assert record["n_occupied"] == int(record["electron_count"]) == 12
    assert record["gap_ev"] > 2.0
    assert record["occupation_smearing_width_ev"] == 0.0


def test_settings_report_operator_and_integration_unmatched(comparison):
    settings = comparison.settings
    assert settings["operator"]["matched"] is False
    assert settings["integration"]["matched"] is False
    assert settings["k_phase_gauge"]["matched"] is False
    assert settings["ensemble"]["matched"] is False
    assert "operator" in comparison.certificate["unmatched_settings"]
    assert "integration" in comparison.certificate["unmatched_settings"]
    # requested mesh vs oddized actual is reported, not silently absorbed
    mesh = settings["k_mesh"]
    assert mesh["declared"] == [1, 1, 4]
    assert mesh["actual"] == [1, 1, 5]
    assert mesh["matched"] is False
    k_mesh_record = comparison.certificate["actual_k_mesh"]
    assert len(k_mesh_record["kpoints_fractional"]) == 5
    np.testing.assert_allclose(
        sorted(k[2] for k in k_mesh_record["kpoints_fractional"]),
        [-0.4, -0.2, 0.0, 0.2, 0.4],
    )
    assert comparison.problem.site_order == SITES


def test_declared_mismatch_is_reported_then_refused(comparison):
    settings = copy.deepcopy(comparison.settings)
    unmatched = apply_declaration(settings, AnchorDeclaration(kmesh=(2, 2, 2)))
    assert unmatched == ["k_mesh"]
    assert settings["k_mesh"]["declared"] == [2, 2, 2]
    assert settings["k_mesh"]["matched"] is False
    # the mismatch is reported in place; numbers are not re-absorbed
    assert settings["operator"]["matched"] is False

    with pytest.raises(AnchorSettingMismatchError, match="k_mesh"):
        apply_declaration(
            copy.deepcopy(comparison.settings),
            AnchorDeclaration(kmesh=(2, 2, 2)),
            refuse_on_mismatch=True,
        )

    reported = apply_declaration(
        copy.deepcopy(comparison.settings),
        AnchorDeclaration(j_convention="tb2j_ordered_pair_stored"),
    )
    assert reported == []
    settings = copy.deepcopy(comparison.settings)
    apply_declaration(settings, AnchorDeclaration(j_convention="pair_once"))
    assert settings["j_convention"]["matched"] is False


# ---------------------------------------------------------------------------
# round-1 review fixes (STD-001..004, SPEC-001/002)
# ---------------------------------------------------------------------------
def test_non_transverse_axis_refused_before_any_engine():
    # axes parallel to the magnetic axis drive u = t x n to zero; the refusal
    # must precede the parser/legacy run, so a nonexistent fdf proves it
    with pytest.raises(AnchorError, match="transverse"):
        run_anchor_comparison(
            FIXTURE_NPZ,
            Path("/nonexistent/dir/o2_fm.fdf"),
            axes=("x",),
        )


def test_anchor_refuses_missing_site_pair_in_legacy_cutoff():
    # The O1-O2 bond is 1.21 Ang; a shorter Rcut leaves no directed pair.
    with pytest.raises(AnchorError, match="must not be reported as zero"):
        run_anchor_comparison(FIXTURE_NPZ, FIXTURE_FDF, rcut_ang=0.1)


def test_small_nonx_component_within_raised_tol_refused_by_gate(tmp_path):
    """A sub-tolerance non-x component must fail the spectral gate, not pass."""
    data = dict(np.load(FIXTURE_NPZ, allow_pickle=False))
    data["h"][:, 3] += 5e-3  # H4 mixing below the raised tolerance ...
    perturbed = tmp_path / "perturbed_o2_fm_q0.npz"
    np.savez(perturbed, **data)
    # ... yet the spinor spectrum it breaks is gone from the channels, so
    # the equivalence gate refuses the same-SCF certificate (before engines)
    with pytest.raises(AnchorError, match="spectral equivalence"):
        run_anchor_comparison(perturbed, FIXTURE_FDF, collinear_tol_ev=1e-2)


def test_spectral_gate_covers_comparison_mesh(comparison):
    gate = comparison.certificate["spectral_equivalence"]
    assert gate["gated"] is True
    assert gate["recorded_producer_kpoints"]["n_kpoints"] == 4
    assert gate["comparison_mesh"]["n_kpoints"] == 5
    assert gate["comparison_mesh"]["pass"] is True
    assert gate["comparison_mesh"]["max_dev_ev"] <= SPECTRAL_EQUIVALENCE_BUDGET_EV
