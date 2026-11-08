"""End-to-end signed bulk-edge correspondence and failure regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from TB2J.magnon.topology import (
    BulkRequest,
    LatticeEmbedding,
    StripRequest,
    calculate_bulk,
    calculate_strip,
    compare_bulk_edge,
    prepare_topology_model,
)
from TB2J.magnon.topology.bulk_edge import _scan_edge_crossings
from tests.test_magnon_topology_strip import honeycomb_spinio


@pytest.fixture(scope="module")
def topological_model():
    return prepare_topology_model(honeycomb_spinio(dmi_z=0.0005))


def test_signed_flow_matches_both_edges_and_converges_with_width_and_mesh(
    topological_model,
):
    bulk = calculate_bulk(topological_model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert bulk.valid, bulk.diagnostics["failures"]

    flows = {"left": [], "right": []}
    for width, kpoints in ((8, 43), (12, 53), (12, 61)):
        strip = calculate_strip(
            topological_model,
            StripRequest(width=width, kpoints=kpoints, edge_layers=2),
        )
        assert strip.valid, strip.diagnostics["failures"]
        for edge, expected in (("left", -1), ("right", 1)):
            comparison = compare_bulk_edge(bulk, strip, gap=1, edge=edge)
            assert comparison.valid, comparison.diagnostics["failures"]
            assert comparison.chern_below_gap == 1
            assert comparison.signed_crossings == expected
            assert comparison.matches
            flows[edge].append(comparison.signed_crossings)

    assert flows == {"left": [-1, -1, -1], "right": [1, 1, 1]}


def test_staggered_sia_trivial_control_refuses_unmatched_in_gap_states():
    spinio = honeycomb_spinio(dmi_z=0.0)
    spinio.sia_tensor[0] = np.diag([0.0, 0.0, 0.0005])
    spinio.sia_tensor[1] = np.diag([0.0, 0.0, 0.0015])
    model = prepare_topology_model(spinio)
    bulk = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert bulk.valid, bulk.diagnostics["failures"]
    assert bulk.chern == 0
    strip = calculate_strip(model, StripRequest(width=8, kpoints=43, edge_layers=2))
    assert strip.valid, strip.diagnostics["failures"]

    for edge in ("left", "right"):
        comparison = compare_bulk_edge(bulk, strip, gap=1, edge=edge)
        assert not comparison.valid
        assert comparison.chern_below_gap == 0
        assert comparison.signed_crossings is None
        assert comparison.unmatched_crossings == 2
        assert any(
            "could not be assigned" in reason
            for reason in comparison.diagnostics["failures"]
        )


def test_actual_paired_cli_writes_three_versioned_source_and_comparison_records(
    tmp_path, monkeypatch
):
    from TB2J.magnon.magnon_cli import main

    source = tmp_path / "exchange_results"
    honeycomb_spinio(dmi_z=0.0005).write_pickle(str(source))
    bulk_path = tmp_path / "bulk.json"
    strip_path = tmp_path / "strip.json"
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "TB2J_magnon",
            "-p",
            str(source),
            "--topology-compare",
            "--topology-gap",
            "1",
            "--topology-mesh",
            "12",
            "12",
            "--topology-strip-width",
            "8",
            "--topology-strip-edge-layers",
            "2",
            "--topology-strip-kpoints",
            "43",
            "--topology-out",
            str(bulk_path),
            "--topology-strip-out",
            str(strip_path),
            "--topology-comparison-out",
            str(comparison_path),
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0

    bulk_record = json.loads(bulk_path.read_text())
    strip_record = json.loads(strip_path.read_text())
    comparison_record = json.loads(comparison_path.read_text())
    json.dumps(comparison_record, allow_nan=False)
    assert bulk_record["kind"] == "bulk"
    assert bulk_record["schema_version"] == "tb2j-topology.bulk/1"
    assert bulk_record["band_indices"] == [0]
    assert bulk_record["model_digest"] == strip_record["model_digest"]
    assert strip_record["kind"] == "strip"
    assert strip_record["schema_version"] == "tb2j-topology.strip/1"
    assert comparison_record["kind"] == "comparison"
    assert comparison_record["schema_version"] == "tb2j-topology.comparison/1"
    assert comparison_record["valid"] is True
    assert comparison_record["bulk_chern_below_gap"] == 1
    assert comparison_record["spectral_flow"]["signed_crossings"] == -1
    assert (
        comparison_record["source_results"]["strip"]["assembly"][
            "cut_directed_bond_records"
        ]
        > 0
    )
    assert (
        len(comparison_record["source_results"]["strip"]["reference_magmoms_sha256"])
        == 64
    )


@pytest.mark.parametrize(
    "arguments",
    (
        ["--topology-gap", "1"],
        ["--topology-strip-kpoints", "81"],
        ["--topology-strip-width", "8"],
    ),
)
def test_cli_rejects_unpaired_topology_options(arguments):
    from TB2J.magnon.magnon_cli import (
        create_parser,
        validate_topology_cli_options,
    )

    parser = create_parser()
    args = parser.parse_args(["-p", "unused", *arguments])
    with pytest.raises(SystemExit) as exit_info:
        validate_topology_cli_options(args, parser)
    assert exit_info.value.code == 2


def test_gapless_bulk_and_incomplete_lower_group_fail_closed(topological_model):
    strip = calculate_strip(topological_model, StripRequest(width=8, kpoints=43))
    assert strip.valid, strip.diagnostics["failures"]

    gapless_model = prepare_topology_model(honeycomb_spinio(dmi_z=0.0))
    gapless_bulk = calculate_bulk(gapless_model, BulkRequest(mesh=(12, 12), bands=[0]))
    gapless_strip = calculate_strip(gapless_model, StripRequest(width=8, kpoints=43))
    comparison = compare_bulk_edge(gapless_bulk, gapless_strip, gap=1, edge="left")
    assert not comparison.valid
    assert not comparison.diagnostics["consistency_checks"][
        "resolved_bulk_indirect_gap"
    ]
    assert comparison.signed_crossings is None

    upper_only = calculate_bulk(
        topological_model, BulkRequest(mesh=(12, 12), bands=[1])
    )
    assert upper_only.valid, upper_only.diagnostics["failures"]
    incomplete = compare_bulk_edge(upper_only, strip, gap=1, edge="left")
    assert not incomplete.valid
    assert not incomplete.diagnostics["consistency_checks"][
        "complete_valid_lower_band_groups"
    ]
    assert incomplete.chern_below_gap is None

    negative_tolerance = calculate_bulk(
        topological_model,
        BulkRequest(mesh=(12, 12), bands=[0], gap_tol=-1.0),
    )
    invalid_tolerance = compare_bulk_edge(negative_tolerance, strip, gap=1, edge="left")
    assert not invalid_tolerance.valid
    assert not invalid_tolerance.diagnostics["consistency_checks"][
        "bulk_gap_tolerance_valid"
    ]
    json.dumps(invalid_tolerance.to_json_dict(), allow_nan=False)


def test_parent_digest_mismatch_is_not_compared(topological_model):
    bulk = calculate_bulk(topological_model, BulkRequest(mesh=(12, 12), bands=[0]))
    reversed_model = prepare_topology_model(honeycomb_spinio(dmi_z=-0.0005))
    strip = calculate_strip(reversed_model, StripRequest(width=8, kpoints=43))
    assert bulk.valid
    assert strip.valid, strip.diagnostics["failures"]

    comparison = compare_bulk_edge(bulk, strip, gap=1, edge="left")
    assert not comparison.valid
    assert not comparison.diagnostics["consistency_checks"]["same_parent_digest"]
    assert comparison.signed_crossings is None


def test_embedding_orientation_and_fixed_normal_momentum_must_match(
    topological_model,
):
    bulk = calculate_bulk(topological_model, BulkRequest(mesh=(12, 12), bands=[0]))
    reversed_embedding = LatticeEmbedding(basis=[[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    reversed_strip = calculate_strip(
        topological_model,
        StripRequest(width=8, kpoints=11, embedding=reversed_embedding),
    )
    assert bulk.valid
    assert reversed_strip.valid, reversed_strip.diagnostics["failures"]
    reversed_result = compare_bulk_edge(bulk, reversed_strip, gap=1, edge="left")
    assert not reversed_result.valid
    assert not reversed_result.diagnostics["consistency_checks"][
        "same_embedding_and_orientation"
    ]

    offset_embedding = LatticeEmbedding(basis=np.eye(3, dtype=int), normal_k=0.25)
    offset_strip = calculate_strip(
        topological_model,
        StripRequest(width=8, kpoints=11, embedding=offset_embedding),
    )
    offset_result = compare_bulk_edge(bulk, offset_strip, gap=1, edge="left")
    assert not offset_result.valid
    assert offset_result.diagnostics["consistency_checks"][
        "same_embedding_and_orientation"
    ]
    assert not offset_result.diagnostics["consistency_checks"][
        "same_fixed_normal_momentum"
    ]


def _scanner_strip(energies, left, right, degenerate_subspaces=None):
    return SimpleNamespace(
        edge_kpoints=np.arange(energies.shape[0], dtype=float) / energies.shape[0],
        energies=np.asarray(energies, dtype=float),
        left_edge_fraction=np.asarray(left, dtype=float),
        right_edge_fraction=np.asarray(right, dtype=float),
        degenerate_subspaces=degenerate_subspaces or [],
    )


def test_crossing_scanner_rejects_ambiguous_and_unmatched_edge_assignments():
    energies = np.array([[1.0], [0.5], [0.0], [-0.5]])
    left = np.full_like(energies, 0.8)
    right = np.full_like(energies, 0.1)
    left[2, 0] = 0.5
    ambiguous = _scanner_strip(energies, left, right)
    crossings, _subtotal, ambiguous_count, _failures = _scan_edge_crossings(
        ambiguous, "left", 0.0, 1e-10, 1e-8
    )
    assert ambiguous_count == 1
    assert any(record["status"] == "ambiguous_edge_assignment" for record in crossings)

    plateau_energies = np.array([[1.0], [0.0], [0.0], [0.0], [-0.5], [0.5]])
    plateau = _scanner_strip(
        plateau_energies,
        np.full_like(plateau_energies, 0.8),
        np.full_like(plateau_energies, 0.1),
    )
    crossings, _subtotal, ambiguous_count, _failures = _scan_edge_crossings(
        plateau, "left", 0.0, 1e-10, 1e-8
    )
    gap_touches = [
        record for record in crossings if record["status"] == "ambiguous_gap_touch"
    ]
    assert ambiguous_count == 1
    assert len(gap_touches) == 1
    assert gap_touches[0]["k_indices"] == [1, 2, 3]

    unlocalized = _scanner_strip(
        energies, np.full_like(energies, 0.2), np.full_like(energies, 0.3)
    )
    crossings, _subtotal, ambiguous_count, failures = _scan_edge_crossings(
        unlocalized, "left", 0.0, 1e-10, 1e-8
    )
    assert ambiguous_count == 0
    assert sum(record["status"] == "unmatched" for record in crossings) == 2
    assert any("could not be assigned" in failure for failure in failures)


def test_crossing_scanner_refuses_unresolved_exact_mode_degeneracy():
    energies = np.array([[1.0, 2.0], [0.5, 1.0], [0.0, 0.0], [-0.5, -0.5]])
    left = np.full_like(energies, 0.8)
    right = np.full_like(energies, 0.1)
    strip = _scanner_strip(energies, left, right)
    crossings, _subtotal, ambiguous_count, _failures = _scan_edge_crossings(
        strip, "left", 0.0, 1e-10, 1e-8
    )
    assert ambiguous_count >= 2
    assert any(
        record["status"] == "ambiguous_degenerate_subspace" for record in crossings
    )


def test_crossing_scanner_resolves_tangent_band_swap_from_projected_subspace():
    energies = np.array([[10.0, 12.0], [11.0, 11.0], [10.0, 12.0], [10.0, 12.0]])
    left = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])
    right = 1.0 - left
    group = {
        "k_index": 1,
        "mode_indices": [0, 1],
        "resolution": "basis_invariant_projected_observables",
        "left_fraction_spectrum": [1.0, 0.0],
        "right_fraction_spectrum": [0.0, 1.0],
    }
    strip = _scanner_strip(energies, left, right, [group])
    crossings, subtotal, ambiguous_count, failures = _scan_edge_crossings(
        strip, "left", 11.0, 1e-10, 1e-8
    )
    assert not failures
    assert subtotal == 1
    assert ambiguous_count == 0
    assert [item["status"] for item in crossings] == ["resolved_degenerate_subspace"]
