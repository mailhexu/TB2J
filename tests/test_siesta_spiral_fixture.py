"""Real-producer oracle; sparse coverage and corrupt conventions fail closed."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.siesta_spiral import SpiralImporter, SpiralImportError
from TB2J.interfaces.siesta_spiral_io import (
    EIGENVALUE_TOL_EV,
    HERMITICITY_TOL,
    PENCIL_HERMITICITY_REL_TOL,
    OracleMismatch,
    load_fixture,
    run_eigenvalue_oracle,
    run_electron_count_oracle,
)
from TB2J.interfaces.siesta_spiral_response import (
    FrozenResponseProblem,
    SpiralResponseError,
    occupied_band_energy,
    pencil_gradient,
)
from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex

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


def test_cri3_real_producer_uses_scale_aware_pencil_oracle():
    bundle, reference = load_fixture(DATA.with_name("siesta_spiral_cri3.npz"))
    H, S = SpiralImporter().assemble_pencil(bundle, reference["kp"][0])
    h_defect = float(np.max(np.abs(H - H.conj().T)))
    assert h_defect > HERMITICITY_TOL
    for matrix in (H, S):
        scale = max(1.0, float(np.abs(matrix).max(initial=0.0)))
        defect = float(np.max(np.abs(matrix - matrix.conj().T)))
        assert defect <= PENCIL_HERMITICITY_REL_TOL * scale

    occupied = int(reference["electron_count"])
    finite_unoccupied = reference["eig"][:, occupied:]
    finite_unoccupied = finite_unoccupied[np.isfinite(finite_unoccupied)]
    assert float(finite_unoccupied.min()) > float(reference["eig"][:, :occupied].max())

    eigenvalues = run_eigenvalue_oracle(bundle, reference)
    assert eigenvalues.passed, eigenvalues.detail
    assert eigenvalues.max_dev <= EIGENVALUE_TOL_EV
    electron_count = run_electron_count_oracle(bundle, reference)
    assert electron_count.passed, electron_count.detail


def test_cri3_response_pencil_accepts_producer_h_hermiticity_defect():
    bundle, reference = load_fixture(DATA.with_name("siesta_spiral_cri3.npz"))
    k = reference["kp"][0]
    producer_h, producer_s = SpiralImporter().assemble_pencil(bundle, k)
    h_defect = float(np.max(np.abs(producer_h - producer_h.conj().T)))
    s_defect = float(np.max(np.abs(producer_s - producer_s.conj().T)))
    assert HERMITICITY_TOL < h_defect < 1e-5
    assert s_defect <= HERMITICITY_TOL

    orbitals_by_site = {
        "Cr1": range(0, 19),
        "Cr2": range(19, 38),
        "I1": range(38, 51),
        "I2": range(51, 64),
        "I3": range(64, 77),
        "I4": range(77, 90),
        "I5": range(90, 103),
        "I6": range(103, 116),
    }
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=orbitals_by_site)
    axes_by_site = {site: (("x", (1.0, 0.0, 0.0)),) for site in orbitals_by_site}
    problem = FrozenResponseProblem(bundle, vertex, axes_by_site=axes_by_site)
    response_h, response_s = problem.pencil(k)
    occupied = int(reference["electron_count"])
    assert np.isfinite(occupied_band_energy(response_h, response_s, occupied))
    rotated_h, rotated_s = problem.rotated_pencil(k, {"Cr1": {"x": 0.1}})

    rotated_h_defect = float(np.max(np.abs(rotated_h - rotated_h.conj().T)))
    assert rotated_h_defect <= 1e-5

    bad_h = response_h.copy()
    bad_h[0, 0] += 2.0e-5j
    with pytest.raises(SpiralResponseError, match="not Hermitian"):
        occupied_band_energy(bad_h, response_s, occupied)
    bad_s = response_s.copy()
    bad_s[0, 0] += 2.0e-8j
    with pytest.raises(SpiralResponseError, match="not Hermitian"):
        occupied_band_energy(response_h, bad_s, occupied)
    bad_vertex = problem.vertex_pencil(k, ("Cr1", "x"))
    bad_vertex[0, 0] += 2.0e-8j
    with pytest.raises(SpiralResponseError, match="not Hermitian"):
        pencil_gradient(response_h, response_s, bad_vertex, occupied)
    np.testing.assert_array_equal(response_h, producer_h)
    np.testing.assert_array_equal(response_s, producer_s)
    np.testing.assert_array_equal(rotated_s, producer_s)


def test_occupation_count_matches_real_producer(producer):
    report = run_electron_count_oracle(*producer)
    assert report.passed, report.detail
    assert abs(report.deviation) < 1e-5


def test_real_pencils_are_hermitian(producer):
    bundle, reference = producer
    for kvec in reference["kp"]:
        H, S = SpiralImporter().assemble_pencil(bundle, kvec)
        for matrix in (H, S):
            scale = max(1.0, float(np.abs(matrix).max(initial=0.0)))
            defect = float(np.max(np.abs(matrix - matrix.conj().T)))
            assert defect <= PENCIL_HERMITICITY_REL_TOL * scale


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
