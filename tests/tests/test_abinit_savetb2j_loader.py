from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_savetb2j import (
    compute_projector_shell_populations,
    gen_exchange_abinit_nc_pao,
    gen_exchange_abinit_projector,
    load_abinit_nc_pao_savetb2j,
    load_abinit_savetb2j,
    mask_local_operators_by_shell_selection,
    select_nc_pao_shells,
)
from TB2J.interfaces.gpaw_projector import compute_projected_charges_moments
from TB2J.scripts.abinit_nc_pao2J import run_abinit_nc_pao2J
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
            ("smooth_xc", "pawdijhat(vxc)"),
            ("paw_xc_onsite", "paw_ij%dijxc"),
            ("paw_u_onsite", "paw_ij%dijU"),
            ("delta_paw_smooth_xc", "smooth_xc+paw_xc_onsite"),
            (
                "delta_paw_smooth_xc_u",
                "smooth_xc+paw_xc_onsite+paw_u_onsite",
            ),
        ]:
            component = components.createVariable(
                name, "f8", ("nsite", "nproj_site_max", "nproj_site_max", "complex")
            )
            component[:] = 0.0
            if name == "delta_total":
                component[:] = hij[0, :, :, :, :] - hij[1, :, :, :, :]
            elif name == "smooth_xc":
                component[0, 0, 0, :] = [0.2, 0.0]
            elif name == "paw_xc_onsite":
                component[0, 0, 0, :] = [0.3, 0.0]
            elif name == "paw_u_onsite":
                component[0, 0, 0, :] = [0.4, 0.0]
            elif name == "delta_paw_smooth_xc":
                component[0, 0, 0, :] = [0.5, 0.0]
            elif name == "delta_paw_smooth_xc_u":
                component[0, 0, 0, :] = [0.9, 0.0]
            component.source = source
            component.units = "eV"
            component.operator_basis = "abinit_native_paw_projector"
            component.spin_treatment = (
                "up_minus_down"
                if name.startswith(("smooth", "paw_", "delta_paw"))
                else "spin_difference"
            )
            component.completeness = {
                "delta_total": "complete",
                "smooth_xc": "smooth_site_window",
                "paw_xc_onsite": "paw_onsite_xc",
                "paw_u_onsite": "paw_onsite_u",
                "delta_paw_smooth_xc": "smooth_plus_paw_onsite_xc",
                "delta_paw_smooth_xc_u": "smooth_plus_paw_onsite_xc_u",
            }.get(name, "not_present")
            if name == "delta_paw_smooth_xc":
                component.smooth_xc_included = "true"
                component.paw_ae_minus_ps_included = "true"
                component.hubbard_included = "false"
                component.exchange_ready = "true"
            elif name == "delta_paw_smooth_xc_u":
                component.smooth_xc_included = "true"
                component.paw_ae_minus_ps_included = "true"
                component.hubbard_included = "true"
                component.exchange_ready = "true"
            elif name in {"smooth_xc", "paw_xc_onsite", "paw_u_onsite"}:
                component.exchange_ready = "false"


def write_minimal_nc_spherical_fixture(
    path, include_projection=True, include_operator=True, natom=1
):
    netcdf4 = pytest.importorskip("netCDF4")

    with netcdf4.Dataset(path, "w") as nc:
        nc.createDimension("three", 3)
        nc.createDimension("natom", natom)
        nc.createDimension("ntypat", 1)
        nc.createDimension("nproj", natom)
        nc.createDimension("nkpt", 1)
        nc.createDimension("nsppol", 2)
        nc.createDimension("nband", 1)
        nc.schema_name = "abinit.savetb2j.nc_spherical_window"
        nc.schema_version = "1.0"
        nc.abinit_version = "synthetic"
        nc.basis_type = "spherical_window"
        nc.basis_radial = "fermi_step_constant_inside"
        nc.basis_angular = "real_spherical_harmonics"
        nc.radius_source = "ratsph"
        nc.operator_basis = "abinit_nc_spherical_window"
        nc.metric_required = 1
        nc.full_bz = 1
        nc.spin_treatment = "up_minus_down"
        nc.basis_lmax = 0
        nc.soft_cutoff_width_bohr = 0.1
        nc.fermi_energy_hartree = 0.0

        nc.createVariable("typat", "i4", ("natom",))[:] = np.ones(natom, dtype=int)
        nc.createVariable("znucl", "f8", ("ntypat",))[:] = [26]
        xred = np.zeros((3, natom), dtype=float)
        if natom > 1:
            xred[0, :] = np.linspace(0.0, 0.5, natom)
        nc.createVariable("xred", "f8", ("three", "natom"))[:] = xred
        nc.createVariable("rprimd", "f8", ("three", "three"))[:] = np.eye(3)
        nc.createVariable("ratsph", "f8", ("ntypat",))[:] = [2.0]
        nc.createVariable("soft_cutoff_width", "f8")[:] = 0.1
        nc.createVariable("projector_site", "i4", ("nproj",))[:] = np.arange(
            1, natom + 1
        )
        nc.createVariable("projector_l", "i4", ("nproj",))[:] = np.zeros(
            natom, dtype=int
        )
        nc.createVariable("projector_m", "i4", ("nproj",))[:] = np.zeros(
            natom, dtype=int
        )
        nc.createVariable("kpoints", "f8", ("three", "nkpt"))[:] = [[0.0], [0.0], [0.0]]
        nc.createVariable("kweights", "f8", ("nkpt",))[:] = [1.0]
        nc.createVariable("eigenvalues", "f8", ("nband", "nkpt", "nsppol"))[:] = 0.0
        nc.createVariable("occupations", "f8", ("nband", "nkpt", "nsppol"))[:] = 1.0

        if not include_projection:
            nc.has_coefficients = 0
            nc.has_overlap_k = 0
            return

        nc.has_coefficients = 1
        nc.has_overlap_k = 1
        nc.createVariable(
            "coefficients_real", "f8", ("nproj", "nband", "nkpt", "nsppol")
        )[:] = 1.0
        nc.createVariable(
            "coefficients_imag", "f8", ("nproj", "nband", "nkpt", "nsppol")
        )[:] = 0.0
        nc.createVariable("overlap_k_real", "f8", ("nproj", "nproj", "nkpt"))[:] = (
            np.eye(natom)[:, :, None]
        )
        nc.createVariable("overlap_k_imag", "f8", ("nproj", "nproj", "nkpt"))[:] = 0.0
        if include_operator:
            nc.createVariable("delta_xc_spherical_real", "f8", ("nproj", "nproj"))[
                :
            ] = np.eye(natom) * 0.05
            nc.createVariable("delta_xc_spherical_imag", "f8", ("nproj", "nproj"))[
                :
            ] = 0.0
            nc.createVariable("delta_spherical_xc_u_real", "f8", ("nproj", "nproj"))[
                :
            ] = np.eye(natom) * 0.1
            nc.createVariable("delta_spherical_xc_u_imag", "f8", ("nproj", "nproj"))[
                :
            ] = 0.0


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
    assert data.population_metric_matrix is None
    assert "smooth pseudo-density" in data.population_metric
    np.testing.assert_array_equal(data.site_projector_indices, [[0, 1]])
    np.testing.assert_allclose(
        data.operator_components["delta_total"], data.hij[0] - data.hij[1]
    )
    assert data.operator_component_metadata["delta_total"]["completeness"] == "complete"
    assert data.operator_component_metadata["dijU"]["source"] == "paw_ij%dijU"
    assert (
        data.operator_component_metadata["delta_paw_smooth_xc"]["exchange_ready"]
        == "true"
    )
    np.testing.assert_allclose(
        data.operator_components["delta_paw_smooth_xc"][0, 0, 0], 0.5 + 0.0j
    )
    assert (
        data.operator_component_metadata["delta_paw_smooth_xc_u"]["hubbard_included"]
        == "true"
    )
    np.testing.assert_allclose(
        data.operator_components["delta_paw_smooth_xc_u"][0, 0, 0], 0.9 + 0.0j
    )


def test_load_abinit_nc_spherical_requires_projection_arrays(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical_metadata_only.nc"
    write_minimal_nc_spherical_fixture(filename, include_projection=False)

    with pytest.raises(ValueError, match="coefficients"):
        load_abinit_nc_pao_savetb2j(filename)


def test_load_abinit_nc_spherical_window_fixture(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical.nc"
    write_minimal_nc_spherical_fixture(filename, include_projection=True)

    data = load_abinit_nc_pao_savetb2j(filename)

    assert data.metadata["abinit_schema_name"] == "abinit.savetb2j.nc_spherical_window"
    assert data.operator_basis == "abinit_nc_spherical_window"
    assert data.coefficient_projector == "nc_spherical_window"
    assert data.coefficients.shape == (2, 1, 1, 1)
    assert data.overlap_k.shape == (1, 1, 1)
    assert data.has_operator_component("delta_spherical_xc_u")
    assert data.has_operator_component("delta_total")


def test_load_abinit_nc_spherical_multi_atom_structure(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical_multi_atom.nc"
    write_minimal_nc_spherical_fixture(filename, include_projection=True, natom=4)

    data = load_abinit_nc_pao_savetb2j(filename)

    assert data.positions.shape == (4, 3)
    assert data.coefficients.shape == (2, 1, 1, 4)
    assert data.overlap_k.shape == (1, 4, 4)
    assert data.operator_components["delta_total"].shape == (4, 1, 1)


def test_load_abinit_nc_spherical_projection_only_fixture(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical_projection_only.nc"
    write_minimal_nc_spherical_fixture(
        filename, include_projection=True, include_operator=False
    )

    data = load_abinit_nc_pao_savetb2j(filename)

    assert data.coefficients.shape == (2, 1, 1, 1)
    assert data.overlap_k.shape == (1, 1, 1)
    assert not data.has_operator_component("delta_total")


def test_gen_exchange_abinit_nc_pao_accepts_spherical_window(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical.nc"
    output_path = tmp_path / "TB2J_results_nc_spherical"
    write_minimal_nc_spherical_fixture(filename, include_projection=True, natom=2)

    exchange_out, exchange = gen_exchange_abinit_nc_pao(
        filename,
        output_path=output_path,
        Rmax=0,
        nz=4,
        population_mode="projector",
        shell_charge_threshold=None,
        shell_moment_threshold=None,
    )

    assert exchange_out == output_path / "exchange.out"
    assert exchange_out.exists()
    assert exchange
    text = exchange_out.read_text(encoding="utf-8")
    assert "norm-conserving" in text
    assert "delta_total" in text
    assert "abinit.savetb2j.nc_spherical_window" in text


def test_gen_exchange_abinit_nc_pao_rejects_spherical_diagnostic_component(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_nc_spherical.nc"
    write_minimal_nc_spherical_fixture(filename, include_projection=True)

    with pytest.raises(ValueError, match="exchange_ready='false'"):
        gen_exchange_abinit_nc_pao(
            filename,
            output_path=tmp_path / "unused",
            Rmax=0,
            nz=4,
            operator_component="delta_xc_spherical",
            population_mode="none",
            shell_charge_threshold=None,
            shell_moment_threshold=None,
        )


def test_abinit_savetb2j_smooth_component_exchange_ready_policy(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)

    with pytest.raises(ValueError, match="exchange_ready='false'"):
        gen_exchange_abinit_projector(
            filename,
            output_path=tmp_path / "TB2J_results_smooth_only",
            operator_component="smooth_xc",
        )
    exchange_out, _ = gen_exchange_abinit_projector(
        filename,
        output_path=tmp_path / "TB2J_results_projector_population",
        operator_component="delta_paw_smooth_xc",
        population_mode="projector",
    )
    assert exchange_out.exists()
    assert data.has_operator_component("delta_paw_smooth_xc")


def test_gen_exchange_abinit_projector_green_population(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    exchange_out, _ = gen_exchange_abinit_projector(
        filename,
        output_path=tmp_path / "TB2J_results_green_population",
        operator_component="delta_paw_smooth_xc",
        population_mode="green",
        nz=8,
    )

    text = exchange_out.read_text()
    assert "charge" in text.lower()
    assert "mag" in text.lower()


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
    data.population_metric_matrix = np.zeros_like(data.overlap_metric)
    data.population_metric_matrix[0, 0] = 2.0

    charges, spinat, density_by_spin = compute_projected_charges_moments(data)

    np.testing.assert_allclose(density_by_spin[:, 0], [10.0, 0.0])
    np.testing.assert_allclose(charges, [10.0])
    np.testing.assert_allclose(spinat[:, 2], [10.0])


def test_abinit_paw_shell_moment_filter_masks_local_operator(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)
    data.projector_radial = np.array([0, 1])
    data.population_metric_matrix = np.eye(2, dtype=complex)
    data.occupations[:] = 0.0
    data.occupations[:, :, 0] = 1.0
    data.coefficients[:] = 0.0
    data.coefficients[0, :, 0, 0] = 1.0
    data.coefficients[1, :, 0, 1] = 0.05
    local_operators = {0: np.ones((2, 2), dtype=complex)}

    shells = select_nc_pao_shells(
        compute_projector_shell_populations(data),
        charge_threshold=0.0,
        moment_threshold=0.01,
    )
    masked = mask_local_operators_by_shell_selection(data, local_operators, shells)

    np.testing.assert_allclose(
        [shell["moment_norm"] for shell in shells], [1.0, 0.0025]
    )
    assert [shell["selected"] for shell in shells] == [True, False]
    np.testing.assert_allclose(masked[0], [[1.0, 0.0], [0.0, 0.0]])


def test_abinit_paw_shell_filter_rejects_incomplete_population_metric(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    with pytest.raises(ValueError, match="PAW-complete population_metric_matrix"):
        gen_exchange_abinit_projector(
            filename,
            output_path=tmp_path / "TB2J_results_shell_filter",
            shell_moment_threshold=0.01,
        )


def test_abinit_exchange_projector_population_uses_sij_metric(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    exchange_out, _ = gen_exchange_abinit_projector(
        filename,
        output_path=tmp_path / "TB2J_results_projector_population",
        population_mode="projector",
    )

    text = exchange_out.read_text()
    assert "not PAW-complete AE charges" in text


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


def test_abinit_nc_pao2j_cli_help_documents_spherical_window(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["abinit_nc_pao2J.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        run_abinit_nc_pao2J()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "NC spherical-window" in help_text
    assert "delta_spherical_xc_u" in help_text
    assert "delta_total" in help_text
