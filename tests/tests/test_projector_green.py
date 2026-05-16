from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    build_site_projector_indices,
    pack_site_hij,
    project_potential_to_hij,
    projector_charge_moments_from_green,
    projector_exchange_trace,
    validate_green_backend,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_projector_green_example():
    path = (
        ROOT_DIR / "examples" / "projector_green" / "build_synthetic_projector_green.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_projector_green", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gpaw_bcc_fe_example():
    path = ROOT_DIR / "examples" / "projector_green" / "gpaw_bcc_fe_projector_green.py"
    spec = importlib.util.spec_from_file_location("gpaw_bcc_fe_projector_green", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bcc_fe_projector_data():
    a = 2.86
    cell = (
        0.5
        * a
        * np.array(
            [
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        )
    )
    eigenvalues = np.array(
        [
            [[0.0, 2.0], [0.5, 2.5]],
            [[0.2, 2.2], [0.7, 2.7]],
        ]
    )
    coefficients = np.zeros((2, 2, 2, 2), dtype=complex)
    coefficients[:, :, 0, 0] = 1.0
    coefficients[:, :, 1, 1] = 1.0
    hij = np.array(
        [
            [[[1.0, 0.1j], [-0.1j, 2.0]]],
            [[[0.4, 0.0], [0.0, 1.5]]],
        ],
        dtype=complex,
    )
    return ProjectorGreenData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        weights=np.array([0.5, 0.5]),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.0,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        cell=cell,
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([26]),
        projector_l=np.array([2, 2]),
        projector_m=np.array([-2, -1]),
        projector_radial=np.array([0, 0]),
        overlap_metric=np.array([[1.0, 0.2j], [-0.2j, 1.5]]),
        site_nproj=np.array([2]),
        site_projector_indices=np.array([[0, 1]]),
        hij=hij,
        hij_definition="paw_dij_projector_hamiltonian",
        hij_units="eV",
        hij_source="GPAW dH_asp",
        hij_projection="native PAW projector Hamiltonian matrix",
        coefficient_source="gpaw.P_ani",
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_partial_wave_channel",
        overlap_metric_definition="GPAW PAW onsite dO_ii correction",
        population_metric="GPAW PAW N0_p packed density contraction",
        operator_basis="native_paw_projector_hamiltonian",
        metadata={
            "source": "synthetic bcc Fe primitive cell",
            "hij_source_name": "PAW d_ij / dH_asp",
            "hij_usage": "spin-dependent part H_ij^up - H_ij^down",
        },
    )


def test_projector_data_validates_exchange_ready_and_hij_difference():
    data = make_bcc_fe_projector_data()

    assert data.validate(exchange_ready=True)
    np.testing.assert_allclose(
        data.get_hij_spin_difference(site=0),
        np.array([[0.6, 0.1j], [-0.1j, 0.5]]),
    )


def test_projector_data_requires_explicit_hij_definition():
    data = make_bcc_fe_projector_data()
    data.hij_definition = ""

    with pytest.raises(ValueError, match="hij requires an explicit definition"):
        data.validate(exchange_ready=True)


def test_projector_green_reconstructs_gk_from_spectral_data():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j

    gk = green.get_Gk(ik=0, energy=energy, ispin=0)

    expected = np.diag(
        [
            1.0 / (energy - data.eigenvalues[0, 0, 0]),
            1.0 / (energy - data.eigenvalues[0, 0, 1]),
        ]
    )
    np.testing.assert_allclose(gk, expected)


def test_projector_green_transforms_full_bz_gk_to_gr():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j
    rpts = np.array([[0, 0, 0], [1, 0, 0]])

    gks = green.get_Gk_all(energy, ispin=0)
    gr = green.get_GR(rpts, energy, Gk_all=gks, ispin=0)

    phase = np.exp(-2.0j * np.pi * np.einsum("ri,ki->rk", rpts, data.kpoints))
    expected = np.einsum("kpq,rk,k->rpq", gks, phase, data.weights)
    np.testing.assert_allclose(gr, expected)


def test_projector_green_rejects_invalid_gr_shapes():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    gks = green.get_Gk_all(1.0 + 0.5j, ispin=0)

    with pytest.raises(ValueError, match="Rpts must have shape"):
        green.compute_GR(np.array([0, 0, 0]), data.kpoints, gks)

    with pytest.raises(ValueError, match="Gks must have shape"):
        green.compute_GR(np.array([[0, 0, 0]]), data.kpoints, gks[:, :1, :1])


def test_projector_green_netcdf_roundtrip(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    data = make_bcc_fe_projector_data()
    filename = tmp_path / "projector_green.nc"

    data.save_netcdf(filename)
    with netcdf4.Dataset(filename) as nc:
        assert "greens_k" not in nc.groups
        assert "greens_R" not in nc.groups
        assert nc.groups["projectors"].variables["coefficients"].dimensions[-1] == (
            "complex"
        )
        projectors = nc.groups["projectors"]
        assert projectors.coefficient_source == "gpaw.P_ani"
        assert projectors.coefficient_projector == "dual_paw_projector"
        assert projectors.channel_interpretation == "paw_partial_wave_channel"
        assert (
            projectors.population_metric == "GPAW PAW N0_p packed density contraction"
        )
        assert (
            projectors.variables["overlap_metric"].definition
            == "GPAW PAW onsite dO_ii correction"
        )
        assert (
            nc.groups["operators"].variables["hij"].operator_basis
            == "native_paw_projector_hamiltonian"
        )

    loaded = ProjectorGreenData.load_netcdf(filename)

    assert loaded.metadata["storage_level"] == "spectral"
    assert loaded.metadata["source"] == "synthetic bcc Fe primitive cell"
    np.testing.assert_allclose(loaded.kpoints, data.kpoints)
    np.testing.assert_allclose(loaded.eigenvalues, data.eigenvalues)
    np.testing.assert_allclose(loaded.coefficients, data.coefficients)
    np.testing.assert_array_equal(loaded.projector_site, data.projector_site)
    np.testing.assert_allclose(loaded.overlap_metric, data.overlap_metric)
    np.testing.assert_allclose(loaded.hij, data.hij)
    assert loaded.hij_definition == data.hij_definition
    assert loaded.hij_units == "eV"
    assert loaded.hij_source == "GPAW dH_asp"
    assert loaded.hij_projection == "native PAW projector Hamiltonian matrix"
    assert loaded.coefficient_source == "gpaw.P_ani"
    assert loaded.coefficient_projector == "dual_paw_projector"
    assert loaded.channel_interpretation == "paw_partial_wave_channel"
    assert loaded.overlap_metric_definition == "GPAW PAW onsite dO_ii correction"
    assert loaded.population_metric == "GPAW PAW N0_p packed density contraction"
    assert loaded.operator_basis == "native_paw_projector_hamiltonian"


def test_project_potential_to_hij_for_non_paw_projectors():
    projectors = np.array(
        [
            [1.0, 0.0, 1.0j],
            [0.0, 2.0, 1.0],
            [1.0, -1.0j, 0.0],
        ],
        dtype=complex,
    )
    potential = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ]
    )
    weights = np.array([0.2, 0.3, 0.5])

    hij_global = project_potential_to_hij(projectors, potential, weights=weights)

    expected = np.zeros((2, 3, 3), dtype=complex)
    for ispin in range(2):
        for i in range(3):
            for j in range(3):
                expected[ispin, i, j] = np.sum(
                    projectors[i].conj() * potential[ispin] * projectors[j] * weights
                )
    np.testing.assert_allclose(hij_global, expected)


def test_pack_site_hij_from_projected_non_paw_hij():
    projectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=complex)
    potential = np.array([[1.0, 2.0], [3.0, 4.0]])
    projector_site = np.array([0, 1, 1])
    site_nproj, site_projector_indices = build_site_projector_indices(projector_site)
    hij_global = project_potential_to_hij(projectors, potential)

    hij = pack_site_hij(hij_global, site_projector_indices, site_nproj)

    assert hij.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(site_nproj, np.array([1, 2]))
    np.testing.assert_array_equal(site_projector_indices, np.array([[0, -1], [1, 2]]))
    np.testing.assert_allclose(hij[:, 0, :1, :1], hij_global[:, :1, :1])
    np.testing.assert_allclose(hij[:, 1], hij_global[:, 1:, 1:])


def test_green_backend_protocol_accepts_projector_green():
    green = ProjectorGreen(make_bcc_fe_projector_data())

    assert validate_green_backend(green)


def test_green_backend_protocol_rejects_missing_members():
    with pytest.raises(TypeError, match="missing required protocol members"):
        validate_green_backend(object())


def test_projector_exchange_trace_matches_direct_reference():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j
    rpts = np.array([[0, 0, 0]])

    result = projector_exchange_trace(green, rpts, energy)

    Delta = data.get_hij_spin_difference(site=0)
    Gup = green.get_GR(rpts, energy=energy, ispin=0)[0]
    Gdn = green.get_GR(rpts, energy=energy, ispin=1)[0]
    orbital = np.einsum("ij,ji->ij", Delta @ Gup, Delta @ Gdn) / (4.0 * np.pi)

    assert result["method"] == "projector_exchange_trace"
    assert result["local_operator"] == "hij_spin_difference"
    np.testing.assert_allclose(result["orbital_trace"][((0, 0, 0), 0, 0)], orbital)
    np.testing.assert_allclose(result["trace"][((0, 0, 0), 0, 0)], np.sum(orbital))


def test_projector_charge_moments_from_green_matches_manual_contour_trace():
    class FakeContour:
        path = np.array([1.0 + 0.2j, 1.5 + 0.3j])
        weights = np.array([0.7 + 0.1j, -0.2 + 0.4j])

        def integrate_values(self, values):
            return np.einsum("e,e...->...", self.weights, values)

    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    contour = FakeContour()

    density = projector_charge_moments_from_green(green, contour)

    manual = np.zeros(2)
    for ispin in range(2):
        diags = []
        for energy in contour.path:
            GR0 = green.get_GR([(0, 0, 0)], energy=energy, ispin=ispin)[0]
            diags.append(np.diag(GR0))
        manual[ispin] = np.sum(
            -np.imag(contour.integrate_values(np.asarray(diags))) / np.pi
        )

    np.testing.assert_allclose(density["density_by_spin"][:, 0], manual)
    np.testing.assert_allclose(density["charges"], [np.sum(manual)])
    np.testing.assert_allclose(density["spinat"][:, 2], [manual[0] - manual[1]])


def test_projector_exchange_trace_rejects_unsupported_hij_definition():
    data = make_bcc_fe_projector_data()
    data.hij_definition = "projected_density_matrix"
    green = ProjectorGreen(data)

    with pytest.raises(ValueError, match="unsupported hij definition"):
        projector_exchange_trace(green, np.array([[0, 0, 0]]), 1.0 + 0.5j)


def test_projector_exchange_trace_accepts_explicit_local_operator():
    data = make_bcc_fe_projector_data()
    data.hij_definition = "projected_density_matrix"
    green = ProjectorGreen(data)
    Delta = data.get_hij_spin_difference(site=0)

    result = projector_exchange_trace(
        green,
        np.array([[0, 0, 0]]),
        1.0 + 0.5j,
        local_operators={0: Delta},
    )

    assert result["local_operator"] == "explicit"
    assert ((0, 0, 0), 0, 0) in result["trace"]


def test_synthetic_nonpaw_projector_green_example_builds_valid_data():
    example = load_projector_green_example()
    data = example.build_synthetic_projector_green_data()
    green = ProjectorGreen(data)

    assert data.validate(exchange_ready=True)
    assert data.hij_definition == "projected_spin_dependent_potential"
    assert data.hij_source == "synthetic non-PAW projected potential"
    assert data.coefficient_projector == "custom_discrete_grid_projector"
    assert data.operator_basis == "projected_spin_dependent_potential"
    GR = green.get_GR(np.array([[0, 0, 0]]), energy=0.1 + 0.02j, ispin=0)
    assert GR.shape == (1, data.nproj, data.nproj)


def test_gpaw_bcc_fe_projector_green_workflow(tmp_path):
    pytest.importorskip("gpaw")
    pytest.importorskip("netCDF4")
    example = load_gpaw_bcc_fe_example()
    filename = tmp_path / "gpaw_bcc_fe_projector_green.nc"

    data, GR, trace = example.run_gpaw_bcc_fe_projector_green_workflow(filename)
    exchange_out, exchange_Jdict = example.write_projector_exchange_out(
        data, path=tmp_path / "TB2J_results"
    )

    assert filename.exists()
    assert exchange_out.exists()
    assert data.validate(exchange_ready=True)
    assert data.metadata["source_code"] == "gpaw"
    assert data.hij_definition == "paw_dh_asp_projector_hamiltonian"
    assert data.hij_source == "GPAW dH_asp"
    assert data.coefficient_source == "gpaw.P_ani"
    assert data.coefficient_projector == "dual_paw_projector"
    assert data.channel_interpretation == "paw_partial_wave_channel"
    assert data.operator_basis == "native_paw_projector_hamiltonian"
    assert data.nspin == 2
    assert data.nkpt == np.prod(data.metadata["kmesh"])
    assert data.nproj == data.site_nproj[0]
    assert data.metadata["magnetic_moment_total"] > 2.0
    assert data.metadata["magnetic_moments"][0] > 2.0
    assert np.linalg.norm(data.get_hij_spin_difference(site=0)) > 1.0
    assert GR.shape == (1, data.nproj, data.nproj)
    assert trace["local_operator"] == "hij_spin_difference"
    assert ((0, 0, 0), 0, 0) in trace["trace"]
    assert ((1, 0, 0), 0, 0) in exchange_Jdict
    assert exchange_Jdict[((1, 0, 0), 0, 0)] > 0.0
    assert "Exchange:" in exchange_out.read_text()
