"""Real-producer oracle; sparse coverage and corrupt conventions fail closed."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral import SpiralImporter, SpiralImportError
from TB2J.interfaces.siesta_spiral_io import (
    OracleMismatch,
    load_fixture,
    run_eigenvalue_oracle,
    run_electron_count_oracle,
)

DATA = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_spiral_fe3.npz"


@pytest.fixture(scope="module")
def fixture():
    return load_fixture(DATA)


@pytest.fixture(scope="module", params=["fe3", "o2"])
def producer(request):
    return load_fixture(DATA.with_name(f"siesta_spiral_{request.param}.npz"))


def test_eigenvalues_match_real_producer(producer):
    report = run_eigenvalue_oracle(*producer)
    assert report.passed, report.detail
    assert report.max_dev < 1e-6


def test_occupation_count_matches_real_producer(producer):
    report = run_electron_count_oracle(*producer)
    assert report.passed, report.detail
    assert abs(report.deviation) < 1e-5


def test_real_pencils_are_hermitian(producer):
    bundle, reference = producer
    for kvec in reference["kp"]:
        H, S = SpiralImporter().assemble_pencil(bundle, kvec)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-8, rtol=0)
        np.testing.assert_allclose(S, S.conj().T, atol=1e-8, rtol=0)


def test_zero_overlap_does_not_remove_hamiltonian_entries(fixture):
    bundle, reference = fixture
    sparse = {
        key: value for key, value in bundle.sparse.items() if bundle.overlap[key] != 0
    }
    report = run_eigenvalue_oracle(replace(bundle, sparse=sparse), reference)
    assert not report.passed, report.detail


def test_wrong_pitch_refused_before_triangular_solver(fixture):
    bundle, reference = fixture
    bad = replace(bundle, q=tuple(-v for v in bundle.q))
    report = run_eigenvalue_oracle(bad, reference)
    assert not report.passed
    assert "Hermitian" in report.detail
    with pytest.raises(OracleMismatch, match="k-point.*Hermitian"):
        run_eigenvalue_oracle(bad, reference, strict=True)


def test_swapped_spin_diagonals_detected(fixture):
    bundle, reference = fixture
    sparse = {key: (v[1], v[0], v[2], v[3]) for key, v in bundle.sparse.items()}
    report = run_eigenvalue_oracle(replace(bundle, sparse=sparse), reference)
    assert not report.passed, report.detail
    assert report.max_dev > 2e-3


@pytest.mark.parametrize("corruption", ["nan", "no-bands", "no-kpoints"])
def test_incomplete_reference_cannot_certify(fixture, corruption):
    bundle, reference = fixture
    reference = dict(reference, eig=reference["eig"].copy())
    if corruption == "nan":
        reference["eig"][0, 0] = np.nan
    elif corruption == "no-bands":
        reference["eig"][0, :] = np.inf
    else:
        reference["kp"] = np.empty((0, 3))
    with pytest.raises(OracleMismatch):
        run_eigenvalue_oracle(bundle, reference, strict=True)


def test_gapped_fixture_has_independent_unoccupied_reference():
    bundle, reference = load_fixture(DATA.with_name("siesta_spiral_o2.npz"))
    assert np.isfinite(reference["eig"]).all()
    occupied = int(reference["electron_count"])
    valence_max = reference["eig"][:, occupied - 1].max()
    conduction_min = reference["eig"][:, occupied].min()
    assert valence_max < reference["ef"] < conduction_min
    assert (
        conduction_min - valence_max > 100 * 8.617333262e-5 * reference["temperature"]
    )
    assert run_eigenvalue_oracle(bundle, reference).passed


@pytest.mark.parametrize(
    "field,value",
    [
        ("spin_packing", "swapped_spinor_components"),
        ("pitch_sign", "opposite_up_phase"),
    ],
)
def test_corrupt_recorded_conventions_refused(tmp_path, field, value):
    with np.load(DATA, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays[field] = value
    path = tmp_path / "corrupt.npz"
    np.savez(path, **arrays)
    with pytest.raises(SpiralImportError, match=field):
        load_fixture(path)
