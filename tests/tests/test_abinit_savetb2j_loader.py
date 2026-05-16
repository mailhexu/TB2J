from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_savetb2j import (
    gen_exchange_abinit_projector,
    load_abinit_savetb2j,
)
from TB2J.interfaces.gpaw_projector import compute_projected_charges_moments
from TB2J.scripts.abinit_projector2J import run_abinit_projector2J

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "inputs" / "abinit_savetb2j"
)


def write_minimal_abinit_savetb2j_fixture(path):
    netcdf4 = pytest.importorskip("netCDF4")

    with netcdf4.Dataset(path, "w") as nc:
        nc.createDimension("nspin", 2)
        nc.createDimension("nkpt", 2)
        nc.createDimension("nband", 2)
        nc.createDimension("nproj", 2)
        nc.createDimension("nsite", 1)
        nc.createDimension("nproj_site_max", 2)
        nc.createDimension("natom", 1)
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.schema_name = "abinit.savetb2j.projector"
        nc.schema_version = "1.0"
        nc.source_code = "abinit"
        nc.abinit_version = "synthetic"
        nc.spin_mode = "collinear"
        nc.spin_channel_order = "up,down"
        nc.full_bz = 1
        nc.kpoint_convention = "fractional_reciprocal"
        nc.phase_convention = "exp(-2*pi*i*k.R)"
        nc.coefficient_source = "abinit.cprj"
        nc.operator_basis = "abinit_native_paw_projector"
        nc.units_json = "{}"

        structure = nc.createGroup("structure")
        structure.createVariable("cell", "f8", ("three", "three"))[:] = np.eye(3)
        structure.createVariable("positions", "f8", ("natom", "three"))[:] = 0.0
        structure.createVariable("atomic_numbers", "i4", ("natom",))[:] = [26]

        kpoints = nc.createGroup("kpoints")
        kpoints.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
        kpoints.createVariable("weights", "f8", ("nkpt",))[:] = [0.5, 0.5]

        bands = nc.createGroup("bands")
        bands.efermi = 0.0
        bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = 0.0
        bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = 1.0

        projectors = nc.createGroup("projectors")
        projectors.coefficient_source = "abinit.cprj"
        projectors.coefficient_projector = "paw_nonlocal_projector"
        projectors.channel_interpretation = "abinit_paw_lmn_channel"
        projectors.operator_basis = "abinit_native_paw_projector"
        projectors.index_base = 0
        coefficients = projectors.createVariable(
            "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
        )
        coefficients[:] = 0.0
        coefficients[0, 0, 0, 0, :] = [1.0, 0.25]
        projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_site", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_l", "i4", ("nproj",))[:] = [2, 2]
        projectors.createVariable("projector_m", "i4", ("nproj",))[:] = [-2, -1]
        projectors.createVariable("projector_radial", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = [2]
        projectors.createVariable(
            "site_projector_indices", "i4", ("nsite", "nproj_site_max")
        )[:] = [[0, 1]]
        projectors.createVariable(
            "overlap_metric", "f8", ("nproj", "nproj", "complex")
        )[:] = 0.0

        operators = nc.createGroup("operators")
        hij = operators.createVariable(
            "hij",
            "f8",
            ("nspin", "nsite", "nproj_site_max", "nproj_site_max", "complex"),
        )
        hij[:] = 0.0
        hij[0, 0, 0, 0, :] = [2.0, 0.5]
        hij[1, 0, 0, 0, :] = [0.75, -0.25]
        hij.definition = "abinit_total_paw_dij"
        hij.units = "eV"
        hij.source = "paw_ij%dij"
        hij.projection = "native ABINIT PAW projector basis"
        hij.operator_basis = "abinit_native_paw_projector"

        components = operators.createGroup("operator_components")
        for name, source in [
            ("delta_total", "paw_ij%dij(up)-paw_ij%dij(down)"),
            ("dijxc", "paw_ij%dijxc"),
            ("dijU", "paw_ij%dijU"),
            ("dijso", "paw_ij%dijso"),
        ]:
            component = components.createVariable(
                name, "f8", ("nsite", "nproj_site_max", "nproj_site_max", "complex")
            )
            component[:] = 0.0
            if name == "delta_total":
                component[:] = hij[0, :, :, :, :] - hij[1, :, :, :, :]
            component.source = source
            component.units = "eV"
            component.operator_basis = "abinit_native_paw_projector"
            component.spin_treatment = "spin_difference"
            component.completeness = (
                "complete" if name == "delta_total" else "not_present"
            )


def test_load_abinit_savetb2j_normalizes_projector_green_data(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)

    assert data.schema_name == "tb2j.projector_green"
    assert data.metadata["source_code"] == "abinit"
    assert data.metadata["abinit_schema_name"] == "abinit.savetb2j.projector"
    assert data.metadata["spin_channel_order"] == "up,down"
    assert data.coefficient_source == "abinit.cprj"
    assert data.coefficient_projector == "paw_nonlocal_projector"
    assert data.channel_interpretation == "abinit_paw_lmn_channel"
    assert data.operator_basis == "abinit_native_paw_projector"
    np.testing.assert_allclose(data.kpoints, [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    np.testing.assert_allclose(data.weights, [0.5, 0.5])
    assert data.eigenvalues.shape == (2, 2, 2)
    assert data.coefficients.shape == (2, 2, 2, 2)
    assert data.hij.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(data.coefficients[0, 0, 0, 0], 1.0 + 0.25j)
    np.testing.assert_allclose(data.atomic_numbers, [26])
    np.testing.assert_array_equal(data.site_projector_indices, [[0, 1]])
    np.testing.assert_allclose(
        data.operator_components["delta_total"], data.hij[0] - data.hij[1]
    )
    assert data.operator_component_metadata["delta_total"]["completeness"] == "complete"
    assert data.operator_component_metadata["dijU"]["source"] == "paw_ij%dijU"


def test_load_abinit_savetb2j_generated_abinit_fixture():
    pytest.importorskip("netCDF4")

    data = load_abinit_savetb2j(FIXTURE_DIR / "fe_savetb2j.nc")

    assert data.metadata["source_code"] == "abinit"
    assert data.coefficients.shape == (2, 1, 12, 18)
    assert data.hij.shape == (2, 1, 18, 18)
    assert data.operator_components["delta_total"].shape == (1, 18, 18)
    np.testing.assert_allclose(data.weights.sum(), 1.0)
    np.testing.assert_array_equal(data.atomic_numbers, [26])
    np.testing.assert_array_equal(data.site_nproj, [18])
    np.testing.assert_array_equal(data.site_projector_indices, [np.arange(18)])
    np.testing.assert_allclose(
        data.operator_components["delta_total"], data.hij[0] - data.hij[1]
    )


def test_abinit_projected_population_uses_kpoint_weights(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)
    data.occupations[:] = 0.0
    data.occupations[0, :, 0] = 1.0
    data.coefficients[:] = 0.0
    data.coefficients[0, 0, 0, 0] = 1.0
    data.coefficients[0, 1, 0, 0] = 3.0
    data.overlap_metric[:] = 0.0
    data.overlap_metric[0, 0] = 2.0

    charges, spinat, density_by_spin = compute_projected_charges_moments(data)

    np.testing.assert_allclose(density_by_spin[:, 0], [10.0, 0.0])
    np.testing.assert_allclose(charges, [10.0])
    np.testing.assert_allclose(spinat[:, 2], [10.0])


def test_load_abinit_savetb2j_rejects_missing_full_bz(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "missing_full_bz.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.delncattr("full_bz")

    with pytest.raises(ValueError, match="full_bz"):
        load_abinit_savetb2j(filename)


def test_load_abinit_savetb2j_rejects_nonzero_index_base(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "one_based_projectors.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.groups["projectors"].index_base = 1

    with pytest.raises(ValueError, match="index_base"):
        load_abinit_savetb2j(filename)


def test_load_abinit_savetb2j_rejects_operator_basis_mismatch(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "basis_mismatch.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.groups["operators"].groups["operator_components"].variables[
            "delta_total"
        ].operator_basis = "wrong_basis"

    with pytest.raises(ValueError, match="operator_basis"):
        load_abinit_savetb2j(filename)


def test_load_abinit_savetb2j_rejects_malformed_complex_dimension(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "bad_complex_dimension.nc"
    with netcdf4.Dataset(filename, "w") as nc:
        nc.createDimension("nspin", 2)
        nc.createDimension("nkpt", 1)
        nc.createDimension("nband", 1)
        nc.createDimension("nproj", 1)
        nc.createDimension("nsite", 1)
        nc.createDimension("nproj_site_max", 1)
        nc.createDimension("natom", 1)
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.schema_name = "abinit.savetb2j.projector"
        nc.schema_version = "1.0"
        nc.source_code = "abinit"
        nc.abinit_version = "synthetic"
        nc.spin_mode = "collinear"
        nc.spin_channel_order = "up,down"
        nc.full_bz = 1
        nc.kpoint_convention = "fractional_reciprocal"
        nc.phase_convention = "exp(-2*pi*i*k.R)"
        nc.coefficient_source = "abinit.cprj"
        nc.operator_basis = "abinit_native_paw_projector"
        nc.units_json = "{}"
        structure = nc.createGroup("structure")
        structure.createVariable("cell", "f8", ("three", "three"))[:] = np.eye(3)
        structure.createVariable("positions", "f8", ("natom", "three"))[:] = 0.0
        structure.createVariable("atomic_numbers", "i4", ("natom",))[:] = [26]
        kpoints = nc.createGroup("kpoints")
        kpoints.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = 0.0
        kpoints.createVariable("weights", "f8", ("nkpt",))[:] = [1.0]
        bands = nc.createGroup("bands")
        bands.efermi = 0.0
        bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = 0.0
        projectors = nc.createGroup("projectors")
        projectors.coefficient_source = "abinit.cprj"
        projectors.coefficient_projector = "paw_nonlocal_projector"
        projectors.channel_interpretation = "abinit_paw_lmn_channel"
        projectors.operator_basis = "abinit_native_paw_projector"
        projectors.index_base = 0
        projectors.createVariable(
            "coefficients", "f8", ("complex", "nspin", "nkpt", "nband", "nproj")
        )[:] = 0.0
        projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = [0]
        projectors.createVariable("projector_site", "i4", ("nproj",))[:] = [0]
        projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = [1]
        projectors.createVariable(
            "site_projector_indices", "i4", ("nsite", "nproj_site_max")
        )[:] = [[0]]
        operators = nc.createGroup("operators")
        components = operators.createGroup("operator_components")
        delta_total = components.createVariable(
            "delta_total",
            "f8",
            ("nsite", "nproj_site_max", "nproj_site_max", "complex"),
        )
        delta_total[:] = 0.0
        delta_total.source = "paw_ij%dij"
        delta_total.units = "eV"
        delta_total.operator_basis = "abinit_native_paw_projector"
        delta_total.spin_treatment = "spin_difference"
        delta_total.completeness = "complete"

    with pytest.raises(ValueError, match="final complex dimension"):
        load_abinit_savetb2j(filename)


def test_gen_exchange_abinit_projector_uses_delta_total_by_default(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    output_path = tmp_path / "TB2J_results_abinit"
    write_minimal_abinit_savetb2j_fixture(filename)

    exchange_out, exchange = gen_exchange_abinit_projector(
        filename,
        output_path=output_path,
        Rmax=0,
        nz=4,
        population_mode="none",
    )

    assert exchange_out == output_path / "exchange.out"
    assert exchange_out.exists()
    assert exchange
    text = exchange_out.read_text(encoding="utf-8")
    assert "ABINIT savetb2j" in text
    assert "delta_total" in text


def test_gen_exchange_abinit_projector_rejects_unavailable_component(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    with pytest.raises(ValueError, match="unavailable"):
        gen_exchange_abinit_projector(
            filename,
            output_path=tmp_path / "unused",
            Rmax=0,
            nz=4,
            operator_component="missing_component",
            population_mode="none",
        )


def test_abinit_projector2j_cli_writes_exchange(tmp_path, monkeypatch, capsys):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    output_path = tmp_path / "TB2J_results_cli"
    write_minimal_abinit_savetb2j_fixture(filename)
    monkeypatch.setattr(
        "sys.argv",
        [
            "abinit_projector2J.py",
            "--input",
            str(filename),
            "--output_path",
            str(output_path),
            "--Rmax",
            "0",
            "--nz",
            "4",
        ],
    )

    run_abinit_projector2J()

    assert (output_path / "exchange.out").exists()
    assert "Wrote" in capsys.readouterr().out


def test_abinit_projector2j_cli_help_documents_workflow(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["abinit_projector2J.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        run_abinit_projector2J()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "ABINIT savetb2j" in help_text
    assert "delta_total" in help_text
    assert "--operator_component" in help_text
    assert "--population_mode" in help_text
