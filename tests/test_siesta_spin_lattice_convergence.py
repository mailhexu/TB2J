"""Convergence must refine the actual SIESTA grid, not only its input label."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def convergence():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "siesta_spin_lattice_bccfe_convergence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "siesta_spin_lattice_convergence", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(cutoff, grid, coupling):
    return {
        "cutoff_Ry": cutoff,
        "realspace_grid": grid,
        "bonds": [{"R": [0, 0, 0], "i": 0, "j": 1, "O_meV_A": coupling}],
    }


def test_identical_effective_grids_do_not_certify_convergence(convergence):
    # A requested cutoff can change without any change in the FFT grid.
    rows = [row(200, [30] * 3, 33.5), row(300, [30] * 3, 33.5)]
    assert not convergence.grid_converged(rows, 0.1, 0.01)


@pytest.mark.parametrize("coupling,expected", [(33.7, True), (34.5, False)])
def test_distinct_grid_plateau_requires_numerical_agreement(
    convergence, coupling, expected
):
    rows = [row(300, [30] * 3, 33.5), row(400, [36] * 3, coupling)]
    assert convergence.grid_converged(rows, 0.1, 0.01) is expected


def test_single_grid_cannot_certify_convergence(convergence):
    assert not convergence.grid_converged([row(400, [36] * 3, 33.5)], 0.1, 0.01)
