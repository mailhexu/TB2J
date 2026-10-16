"""Deterministic numeric and real-fixture evidence for Story-022 reporting."""

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral_campaign import (
    assemble_campaign_report,
    build_convergence_ladder,
    compare_response_curvatures,
)
from TB2J.interfaces.siesta_spiral_io import load_fixture
from TB2J.interfaces.siesta_spiral_vertex import partition_variant_spread

FIXTURE = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_cri3.npz"


def test_response_curvature_metrics_and_derivative_axis_are_independent():
    reference = {"xx": 2.0, "xy": -4.0, "yy": 1.0}
    refined = {"xx": 2.5, "xy": -3.0, "yy": 0.5}
    assert compare_response_curvatures(
        reference, refined, relative=True
    ) == pytest.approx(1.0 / 3.0)
    assert compare_response_curvatures(reference, refined, relative=False) == 1.0
    assert compare_response_curvatures(
        {"xx": 0.0}, {"xx": 2.0e-13}, relative=True
    ) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical"):
        compare_response_curvatures({"xx": 1.0}, {"yy": 1.0}, relative=True)
    with pytest.raises(ValueError, match="finite"):
        compare_response_curvatures({"xx": np.inf}, {"xx": 1.0}, relative=True)

    basis = build_convergence_ladder(
        [
            {"rung_id": "DZ", "observation": {"curvature": reference, "basis": "DZ"}},
            {
                "rung_id": "TZ",
                "observation": {"curvature": refined, "basis": "TZ"},
            },
        ],
        metric="relative_response",
        threshold=0.05,
        units="relative",
    )
    other_ladders = {
        "grid": {"rungs": [], "passed": True},
        "kmesh": {"rungs": [], "passed": True},
        "response_integration": {"rungs": [], "passed": True},
    }

    def make_report(derivative):
        return assemble_campaign_report(
            basis=basis,
            grid=other_ladders["grid"],
            kmesh=other_ladders["kmesh"],
            response_integration=other_ladders["response_integration"],
            derivative_agreement=derivative,
            partition_sensitivity={"spread": 0.25},
        )

    report = make_report({"finite_rotation": {"max_error": 0.0002}})
    changed_derivative = make_report(
        {"finite_rotation": {"max_error": 8.0}, "eV_per_ang": {"passed": False}}
    )
    assert report["convergence_ladders"] == changed_derivative["convergence_ladders"]
    assert report["convergence_ladders"]["basis"]["rungs"][1][
        "adjacent_change"
    ] == pytest.approx(1.0 / 3.0)
    assert report["convergence_ladders"]["basis"]["passed"] is False
    assert report["convergence_ladders"]["basis"]["rungs"][0]["observation"] == {
        "curvature": reference,
        "basis": "DZ",
    }
    assert "derivative_agreement" not in report["convergence_ladders"]
    assert report["derivative_agreement"] != changed_derivative["derivative_agreement"]


def test_cri3_ligand_partition_variants_report_real_operator_spread():
    bundle, _reference = load_fixture(FIXTURE)
    positions = np.asarray(bundle.orbital_positions)
    site_positions, orbital_site, counts = np.unique(
        positions, axis=0, return_inverse=True, return_counts=True
    )
    assert len(site_positions) == 8
    ligand_sites = [int(index) for index, count in enumerate(counts) if count == 13]
    chromium_sites = [int(index) for index, count in enumerate(counts) if count == 19]
    assert len(ligand_sites) == 6
    assert len(chromium_sites) == 2
    complete = {
        f"Cr{index}": np.flatnonzero(orbital_site == index).tolist()
        for index in chromium_sites
    }
    complete.update(
        {
            f"I{index}": np.flatnonzero(orbital_site == index).tolist()
            for index in ligand_sites
        }
    )
    grouped_ligands = {
        f"Cr{index}": np.flatnonzero(orbital_site == index).tolist()
        for index in chromium_sites
    }
    grouped_ligands["I_ligands"] = np.flatnonzero(
        np.isin(orbital_site, ligand_sites)
    ).tolist()
    probe_site = ligand_sites[0]
    probe_orbitals = np.flatnonzero(orbital_site == probe_site).tolist()

    report = partition_variant_spread(
        bundle,
        {"complete_sites": complete, "ligands_grouped": grouped_ligands},
        probe_orbitals=probe_orbitals,
        axis=(1.0, 0.0, 0.0),
        operator="onsite_delta",
    )

    assert report["unrotated_blocks_identical"] is True
    assert report["variants"]["complete_sites"]["rotated_site"] == f"I{probe_site}"
    assert report["variants"]["ligands_grouped"]["rotated_site"] == "I_ligands"
    for variant in report["variants"].values():
        response = variant["magnetic_operator_change"]
        assert np.isfinite(response) and response > 1.0e-8
    assert np.isfinite(report["max_pairwise_spread"])
    assert report["max_pairwise_spread"] > 1.0e-8
