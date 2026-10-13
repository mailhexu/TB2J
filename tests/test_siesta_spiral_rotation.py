"""Story-006: rotation legitimacy, partition variants, torque-state recording.

Production-vertex gates on the real Fe3 fixture (27 DZP orbitals, 3 Fe sites):

* rotation legitimacy -- a finite physical local rotation changes ONLY the
  magnetic operator (the traceless spin split); the scalar part of H and the
  overlap are untouched (AC-004 rotation clause, FR-008);
* partition variants -- declared alternative spatial partitions produce a
  different rotated magnetic operator for the same probe rotation; the
  spread is reported as a sensitivity diagnostic (FR-015);
* torque state -- the reference's onsite transverse splitting is recorded
  verbatim and flagged, never silently accepted (FR-008).
"""

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral import SpiralImportError
from TB2J.interfaces.siesta_spiral_io import load_fixture
from TB2J.interfaces.siesta_spiral_vertex import (
    MagneticVertex,
    extract_magnetic_vertex,
    partition_variant_spread,
    rotate_site_blocks,
    rotation_legitimacy_report,
    torque_state_report,
)

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_fe3.npz"

SITES = {f"Fe{i+1}": list(range(i * 9, (i + 1) * 9)) for i in range(3)}


@pytest.fixture(scope="module")
def fe3():
    return load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def vertex(fe3):
    bundle, _ = fe3
    return extract_magnetic_vertex(bundle, orbitals_by_site=SITES)


# ---------------------------------------------------------------------------
# TEST-001: rotation legitimacy
# ---------------------------------------------------------------------------


def test_rotation_changes_magnetic_operator_only(fe3, vertex):
    bundle, _ = fe3
    report = rotation_legitimacy_report(bundle, vertex, "Fe2", angle=0.4)
    assert report["passed"] is True
    assert report["rotated_site"] == "Fe2"
    assert report["magnetic_operator_change"] > 1e-3
    assert report["scalar_part_change"] == pytest.approx(0.0, abs=1e-12)
    assert report["overlap_change"] == pytest.approx(0.0, abs=0.0)
    assert report["untouched_sites_unchanged"] is True
    # per-block identity: the trace of every rotated spinor block is preserved
    assert report["max_trace_deviation"] < 1e-12


def test_rotation_matches_explicit_su2_conjugation(vertex):
    rotated = rotate_site_blocks(vertex, "Fe1", axis=(0.0, 0.0, 1.0), angle=np.pi / 3)
    theta = np.pi / 3

    def expected(block, weight):
        v_up, _v_dn, v3, v4 = block
        hx_r = np.cos(theta) * v3 - np.sin(theta) * v4
        hy_r = np.sin(theta) * v3 + np.cos(theta) * v4
        hz_r = v_up
        return (
            v_up + weight * (hz_r - v_up),
            -(v_up + weight * (hz_r - v_up)),
            v3 + weight * (hx_r - v3),
            v4 + weight * (hy_r - v4),
        )

    for key in vertex.partition["Fe1"]:
        owners = sum(1 for m in vertex.partition.values() if key in m)
        assert rotated[key] == pytest.approx(expected(vertex.blocks[key], 1.0 / owners))
    # onsite entries take the full conjugation, shared entries exactly half
    onsite = next(k for k in vertex.partition["Fe1"] if k[0] == k[1])
    shared = next(
        k
        for k in vertex.partition["Fe1"]
        if sum(1 for m in vertex.partition.values() if k in m) == 2
    )
    assert rotated[onsite] == pytest.approx(expected(vertex.blocks[onsite], 1.0))
    assert rotated[shared] == pytest.approx(expected(vertex.blocks[shared], 0.5))
    outside = next(k for k in vertex.blocks if k not in vertex.partition["Fe1"])
    assert rotated[outside] == vertex.blocks[outside]


def test_shared_cross_site_entry_takes_half_rotation():
    blocks = {
        (0, 0, 0): (0.5, -0.5, 0.3, -0.2),
        (1, 1, 0): (-0.25, 0.25, 0.1, 0.4),
        (0, 1, 0): (0.2, -0.2, -0.35, 0.15),
    }
    vertex = MagneticVertex(
        blocks=blocks,
        partition={
            "A": frozenset({(0, 0, 0), (0, 1, 0)}),
            "B": frozenset({(1, 1, 0), (0, 1, 0)}),
        },
    )
    theta = np.pi / 5

    def fully_rotated(block):
        v_up, v_dn, v3, v4 = block
        hx_r = np.cos(theta) * v3 - np.sin(theta) * v4
        hy_r = np.sin(theta) * v3 + np.cos(theta) * v4
        return (v_up, v_dn, hx_r, hy_r)

    rotated = rotate_site_blocks(vertex, "A", axis=(0.0, 0.0, 1.0), angle=theta)
    # site A's sole entry: full conjugation
    assert rotated[(0, 0, 0)] == pytest.approx(fully_rotated(blocks[(0, 0, 0)]))
    # the shared bond moves exactly halfway: B + 1/2 (U B U^dag - B)
    assert rotated[(0, 1, 0)] == pytest.approx(
        tuple(
            0.5 * (b + r)
            for b, r in zip(blocks[(0, 1, 0)], fully_rotated(blocks[(0, 1, 0)]))
        )
    )
    # site B's sole entry is untouched
    assert rotated[(1, 1, 0)] == vertex.blocks[(1, 1, 0)]
    # with the same U at both owners, the 1/2 + 1/2 weighted shares sum to
    # the full conjugation on the shared bond
    full = fully_rotated(blocks[(0, 1, 0)])
    both_sites = tuple(b + 2.0 * 0.5 * (f - b) for b, f in zip(blocks[(0, 1, 0)], full))
    assert both_sites == pytest.approx(full)


def test_undeclared_partition_and_unknown_site_refuse(fe3, vertex):
    bundle, _ = fe3
    undeclared = extract_magnetic_vertex(bundle)
    with pytest.raises(SpiralImportError, match="partition"):
        rotation_legitimacy_report(bundle, undeclared, "Fe1")
    with pytest.raises(SpiralImportError, match="Fe9"):
        rotation_legitimacy_report(bundle, vertex, "Fe9")


def test_zero_angle_is_a_no_operation(vertex):
    rotated = rotate_site_blocks(vertex, "Fe1", angle=0.0)
    for key in vertex.blocks:
        assert rotated[key] == pytest.approx(vertex.blocks[key])


# ---------------------------------------------------------------------------
# TEST-002: partition-variant spread
# ---------------------------------------------------------------------------


def test_partition_variant_spread_reported(fe3):
    bundle, _ = fe3
    variants = {
        "per_atom": SITES,
        "single_metal_block": {"FeAll": list(range(27))},
    }
    report = partition_variant_spread(
        bundle, variants, probe_orbitals=SITES["Fe1"], angle=0.4
    )
    # the unrotated vertex blocks are partition-independent
    assert report["unrotated_blocks_identical"] is True
    # per-variant operator change and the pairwise spread are reported
    for name in variants:
        assert report["variants"][name]["magnetic_operator_change"] > 0.0
        assert report["variants"][name]["rotated_entries"] > 0
    assert report["max_pairwise_spread"] > 1e-3
    assert report["probe_orbitals"] == SITES["Fe1"]
    assert set(report["variants"]) == set(variants)


def test_identical_variants_report_zero_spread(fe3):
    bundle, _ = fe3
    variants = {
        "a": SITES,
        "b": {f"s{i}": orbs for i, orbs in enumerate(SITES.values())},
    }
    assert partition_variant_spread(
        bundle, variants, probe_orbitals=SITES["Fe2"], angle=0.4
    )["max_pairwise_spread"] == pytest.approx(0.0, abs=1e-12)


def test_variant_probe_spanning_sites_refuses(fe3):
    bundle, _ = fe3
    full_cover = {"Fe1": list(range(9)), "rest": list(range(9, 27))}
    with pytest.raises(SpiralImportError, match="exactly one declared site"):
        partition_variant_spread(bundle, {"a": full_cover}, probe_orbitals=[0, 26])


# ---------------------------------------------------------------------------
# TEST-003: torque-state recording
# ---------------------------------------------------------------------------


def test_torque_state_recorded_verbatim(fe3, vertex):
    bundle, _ = fe3
    report = torque_state_report(bundle, vertex)
    assert report["disposition"] == "recorded_verbatim"
    # Fe3 onsite blocks carry genuine transverse splitting (probe-verified)
    assert report["torque_state"] == "nonzero_onsite_transverse_splitting"
    for site in SITES:
        entry = report["sites"][site]
        assert entry["transverse_rms_eV"] > 0.0
        assert entry["longitudinal_rms_eV"] > 0.0
        assert 0.0 <= entry["transverse_fraction"] <= 1.0


def test_collinear_reference_labeled_collinear(fe3, vertex):
    bundle, _ = fe3
    report = torque_state_report(bundle, vertex, transverse_tol=1.0)
    # a tolerance above every transverse component labels the reference
    # collinear -- the label follows the recorded data, it never hides it
    assert report["torque_state"] == "onsite_splitting_within_tolerance"
    assert report["transverse_tol_eV"] == 1.0


def test_torque_state_accepts_scalar_r_bundle_keys():
    from TB2J.interfaces.siesta_spiral_vertex import (
        MagneticVertex,
        torque_state_report,
    )

    blocks = {(0, 0, 0): (0.5, -0.5, 0.0, 0.0), (1, 1, 0): (-0.25, 0.25, 0.0, 0.0)}
    vertex = MagneticVertex(
        blocks=blocks,
        partition={"A": frozenset({(0, 0, 0)}), "B": frozenset({(1, 1, 0)})},
    )
    report = torque_state_report(None, vertex)
    assert report["torque_state"] == "onsite_splitting_within_tolerance"
    assert report["sites"]["A"]["longitudinal_rms_eV"] == pytest.approx(0.5)
