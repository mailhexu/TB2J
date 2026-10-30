from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_savetb2j import (
    abinit_netcdf_kind,
    compute_projector_shell_populations,
    gen_exchange_abinit_nc_pao,
    gen_exchange_abinit_projector,
    load_abinit_nc_pao_savetb2j,
    load_abinit_savetb2j,
    mask_local_operators_by_shell_selection,
    select_nc_pao_shells,
)
from TB2J.interfaces.gpaw_projector import compute_projected_charges_moments
from TB2J.scripts.abinit2J import run_abinit2J

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "inputs" / "abinit_savetb2j"
)


def write_minimal_abinit_savetb2j_fixture(path, occupations="wellformed"):
    """Write a tiny ABINIT savetb2j file.

    ``occupations`` selects the bands/occupations variable layout:
    ``"wellformed"`` (an old-schema file still carrying occupations),
    ``"malformed"`` (wrong rank and type), or ``"absent"``.
    """
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
        if occupations == "wellformed":
            bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = (
                1.0
            )
        elif occupations == "malformed":
            # Wrong rank and integer type: unreadable if any consumer tried.
            bands.createVariable("occupations", "i4", ("nband",))[:] = np.full(
                len(nc.dimensions["nband"]), -3, dtype="i4"
            )

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
    # Occupations are never read from ABINIT savetb2j files, even when the
    # variable exists (this fixture writes one), and v1.0 predates the
    # electron count.
    assert data.occupations is None
    assert data.number_of_electrons is None
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
        Rpts=[(0, 0, 0)],
        nz=4,
        population_mode="green",
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
            Rpts=[(0, 0, 0)],
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
        population_mode="green",
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


@pytest.mark.parametrize("usepawu", [0, 1, -1])
def test_abinit_schema_12_preserves_hubbard_provenance(tmp_path, usepawu):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "paw12.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.2"
        nc.usepawu = usepawu
        delta = nc.groups["operators"].groups["operator_components"]["delta_total"]
        delta.hubbard_included = "true" if usepawu else "false"
    data = load_abinit_savetb2j(filename)
    assert data.metadata["usepawu"] == usepawu
    assert data.operator_component_metadata["delta_total"]["hubbard_included"] == (
        "true" if usepawu else "false"
    )


@pytest.mark.parametrize(
    "usepawu,included",
    [(0, "true"), (1, "false"), (0, "maybe"), ("invalid", "false")],
)
def test_abinit_schema_12_rejects_invalid_hubbard_provenance(
    tmp_path, usepawu, included
):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "bad-paw12.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.2"
        nc.usepawu = usepawu
        nc.groups["operators"].groups["operator_components"][
            "delta_total"
        ].hubbard_included = included
    with pytest.raises(ValueError):
        load_abinit_savetb2j(filename)


def test_abinit_schema_12_rejects_invalid_nondefault_component_provenance(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "bad-component.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.2"
        nc.groups["operators"].groups["operator_components"][
            "smooth_xc"
        ].hubbard_included = "invalid"
    with pytest.raises(ValueError):
        load_abinit_savetb2j(filename)


def test_abinit_schema_12_missing_hubbard_provenance_is_unknown(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "unknown-paw12.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.2"
    data = load_abinit_savetb2j(filename)
    assert data.metadata["usepawu"] is None
    assert "hubbard_included" not in data.operator_component_metadata["delta_total"]


@pytest.mark.parametrize("variant", ["spinor", "nc", "magnetic"])
def test_abinit_schema_12_rejects_unsupported_variants(tmp_path, variant):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "unsupported-paw12.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.2"
        if variant == "spinor":
            nc.spin_mode = "spinor"
        elif variant == "nc":
            nc.coefficient_source = "abinit.nc_pao"
        else:
            nc.groups["operators"].groups["operator_components"][
                "delta_total"
            ].spin_treatment = "magnetic_2x2"
    with pytest.raises(ValueError):
        load_abinit_savetb2j(filename)


def write_minimal_nc_pao_projector_wrapper_fixture(
    path,
    *,
    schema_version="1.2",
    include_overlap_k=True,
    include_delta_total=True,
    delta_completeness="complete",
    coefficients=None,
    overlap_k_values=None,
    delta_total_values=None,
    number_of_electrons=None,
):
    netcdf4 = pytest.importorskip("netCDF4")

    with netcdf4.Dataset(path, "w") as nc:
        for name, size in (
            ("nspin", 2),
            ("nkpt", 2),
            ("nband", 2),
            ("nproj", 2),
            ("nsite", 1),
            ("nproj_site_max", 2),
            ("natom", 1),
            ("three", 3),
            ("complex", 2),
        ):
            nc.createDimension(name, size)
        nc.schema_name = "abinit.savetb2j.projector"
        nc.schema_version = schema_version
        nc.source_code = "abinit"
        nc.abinit_version = "synthetic"
        nc.spin_mode = "collinear"
        nc.spin_channel_order = "up,down"
        nc.full_bz = "true"
        nc.kpoint_convention = "fractional_reciprocal"
        nc.phase_convention = "exp(-2*pi*i*k.R)"
        nc.coefficient_source = "abinit.nc_pao"
        nc.operator_basis = "abinit_nc_pao"
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
        if number_of_electrons is not None:
            bands.number_of_electrons = float(number_of_electrons)
        # Older wrappers still carry occupations; readers must ignore them.
        bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = 1.0

        projectors = nc.createGroup("projectors")
        projectors.coefficient_source = "abinit.nc_pao"
        projectors.coefficient_projector = "nc_pao_tphi"
        projectors.channel_interpretation = "abinit_nc_pao_lmn_channel"
        projectors.operator_basis = "abinit_nc_pao"
        projectors.index_base = 0
        coefficients_var = projectors.createVariable(
            "coefficients",
            "f8",
            ("nspin", "nkpt", "nband", "nproj", "complex"),
        )
        if coefficients is not None:
            coefficients_var[..., 0] = np.real(coefficients)
            coefficients_var[..., 1] = np.imag(coefficients)
        else:
            coefficients_var[:] = 0.0
        projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_site", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_l", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_m", "i4", ("nproj",))[:] = [0, 0]
        projectors.createVariable("projector_radial", "i4", ("nproj",))[:] = [0, 1]
        projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = [2]
        projectors.createVariable(
            "site_projector_indices", "i4", ("nsite", "nproj_site_max")
        )[:] = [[0, 1]]
        if include_overlap_k:
            overlap_k = projectors.createVariable(
                "overlap_k", "f8", ("nkpt", "nproj", "nproj", "complex")
            )
            if overlap_k_values is not None:
                # Serialize the true producer byte order: nctkarr_t reverses
                # the Fortran dim list, so Python reads B[k, j, i, c] =
                # A[c, i, j, k] and the npm row/column pair is stored swapped.
                swapped = np.swapaxes(overlap_k_values, -1, -2)
                overlap_k[..., 0] = np.real(swapped)
                overlap_k[..., 1] = np.imag(swapped)
            else:
                overlap_values = np.zeros((2, 2, 2, 2))
                overlap_values[:, :, :, 0] = np.broadcast_to(np.eye(2), (2, 2, 2))
                overlap_k[:] = overlap_values

        operators = nc.createGroup("operators")
        components = operators.createGroup("operator_components")
        if include_delta_total:
            delta_total = components.createVariable(
                "delta_total",
                "f8",
                ("nsite", "nproj_site_max", "nproj_site_max", "complex"),
            )
            if delta_total_values is not None:
                # Producer bytes: npm pair swapped, as for overlap_k above.
                swapped = np.swapaxes(delta_total_values, -1, -2)
                delta_total[..., 0] = np.real(swapped)
                delta_total[..., 1] = np.imag(swapped)
            else:
                delta_total[:] = 0.0
                delta_total[0, 0, 0, 0] = 1.0
            delta_total.source = "nc_pao.delta_xc+delta_u"
            delta_total.units = "eV"
            delta_total.operator_basis = "abinit_nc_pao"
            delta_total.spin_treatment = "spin_difference"
            delta_total.completeness = delta_completeness


@pytest.mark.parametrize("loader", [load_abinit_savetb2j, load_abinit_nc_pao_savetb2j])
@pytest.mark.parametrize("completeness", ["complete", "zero_by_symmetry"])
def test_load_abinit_schema_12_nc_pao_projector_wrapper(loader, completeness, tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "nc-pao-projector-wrapper.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(
        filename, delta_completeness=completeness
    )

    data = loader(filename)

    assert data.schema_name == "tb2j.projector_green"
    assert data.metadata["abinit_schema_name"] == "abinit.savetb2j.projector"
    assert data.metadata["abinit_schema_version"] == "1.2"
    assert data.coefficient_source == "abinit.nc_pao"
    assert data.coefficient_projector == "nc_pao_tphi"
    assert data.channel_interpretation == "abinit_nc_pao_lmn_channel"
    assert data.operator_basis == "abinit_nc_pao"
    assert data.coefficients.shape == (2, 2, 2, 2)
    assert data.overlap_k.shape == (2, 2, 2)
    assert data.overlap_metric is None
    assert data.overlap_metric_definition == "k-dependent NC PAO overlap_k"
    # Wrapper 1.2 predates the electron count; occupations are never read.
    assert data.number_of_electrons is None
    assert data.occupations is None
    assert data.hij is None
    assert data.operator_component_metadata["delta_total"] == {
        "source": "nc_pao.delta_xc+delta_u",
        "units": "eV",
        "operator_basis": "abinit_nc_pao",
        "spin_treatment": "spin_difference",
        "completeness": completeness,
    }
    data.validate(exchange_ready=True)


def test_load_abinit_schema_13_nc_pao_projector_wrapper_electron_count(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "nc-pao-projector-wrapper-v13.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(filename, schema_version="1.3")

    # Wrapper 1.3 requires the producer valence-electron count.
    with pytest.raises(ValueError, match="number_of_electrons"):
        load_abinit_savetb2j(filename)

    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(filename, "a") as nc:
        nc.groups["bands"].number_of_electrons = 8.0

    data = load_abinit_nc_pao_savetb2j(filename)

    assert data.metadata["abinit_schema_version"] == "1.3"
    assert data.number_of_electrons == 8.0
    # The occupations variable from the old layout is still ignored.
    assert data.occupations is None


def test_load_abinit_nc_pao_projector_wrapper_physical_oracle(tmp_path):
    """Conjugation/orientation oracle with non-real Hermitian matrices.

    The wrapper producer exports coefficients as <psi|p> (verified in
    m_nc_savetb2j_netcdf.F90 / m_nc_pao_hamiltonian.F90), so the loaded
    coefficients must be <p|psi> = conj(<psi|p>).  The writer declares
    overlap_k and delta_total via nctkarr_t in Fortran order ("complex,
    nproj, nproj, nkpt" / "..., nsite"), so Python reads the duplicate npm
    row/column pair swapped (B[k, j, i, c] = A[c, i, j, k]); the fixture
    serializes that true byte order and the loader must normalize it back to
    the physical <p_i|.|p_j> rows=bra orientation.  With a complete window the
    loaded arrays satisfy the spectral identity
    S[p,q] = sum_n C[p,n] conj(C[q,n]) and the generalized eigenproblem
    sum_q H[p,q] x[q] = eps_n sum_q S[p,q] x[q] with
    H[p,q] = sum_n C[p,n] conj(C[q,n]) eps_n -- identities that fail for the
    unconjugated coefficients and for the unnormalized (still-swapped) bytes
    when the matrices are non-real Hermitian.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    import scipy.linalg as sla

    rng = np.random.default_rng(23)
    nproj = 2
    g = 5
    phi = rng.normal(size=(g, nproj)) + 1j * rng.normal(size=(g, nproj))  # <G|p>
    vmat = rng.normal(size=(g, g)) + 1j * rng.normal(size=(g, g))

    overlap = phi.conj().T @ phi
    overlap = 0.5 * (overlap + overlap.conj().T)
    assert not np.allclose(overlap.imag, 0.0), "oracle needs non-real Hermitian S"

    delta = phi.conj().T @ vmat @ phi
    delta = 0.5 * (delta + delta.conj().T)
    assert not np.allclose(delta.imag, 0.0), "oracle needs non-real Hermitian O"

    # Orthonormal band basis of the projector span (complete window).
    qvec, _ = np.linalg.qr(phi)
    proj = qvec.conj().T @ phi  # proj[n, p] = <psi_n|p_p> (producer)
    eps = np.array([0.7, -1.3])

    filename = tmp_path / "wrapper-oracle.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(
        filename,
        schema_version="1.3",
        number_of_electrons=4.0,
        coefficients=np.broadcast_to(proj, (2, 2, nproj, nproj)).copy(),
        overlap_k_values=np.broadcast_to(overlap, (2, nproj, nproj)).copy(),
        delta_total_values=delta,
    )

    # The file bytes carry the npm-swapped (transposed) matrices: this pins
    # the fixture to the true serialized producer convention.
    with netCDF4.Dataset(filename) as nc:
        projectors = nc.groups["projectors"]
        raw = np.asarray(projectors.variables["overlap_k"][:])
        raw_mat = raw[..., 0] + 1j * raw[..., 1]
        components = nc.groups["operators"].groups["operator_components"]
        raw_delta = np.asarray(components.variables["delta_total"][:])
        raw_delta_mat = raw_delta[..., 0] + 1j * raw_delta[..., 1]
    np.testing.assert_allclose(raw_mat[0], overlap.T, atol=1e-14)
    np.testing.assert_allclose(raw_delta_mat[0], delta.T, atol=1e-14)
    assert not np.allclose(overlap.T, overlap), "oracle needs non-symmetric S"
    assert not np.allclose(delta.T, delta), "oracle needs non-symmetric O"

    data = load_abinit_nc_pao_savetb2j(filename)

    # Loaded coefficients are <p|psi>, not the stored <psi|p>.
    c_loaded = data.coefficients[0, 0]  # (nband, nproj)
    np.testing.assert_allclose(c_loaded, proj.conj(), atol=1e-14)

    # The loader normalizes the swapped npm pair back to the physical
    # rows=bra orientation; both matrices are non-real Hermitian and
    # non-symmetric, so a missing swapaxes is caught here.
    np.testing.assert_allclose(data.overlap_k[0], overlap, atol=1e-14)
    np.testing.assert_allclose(
        data.operator_components["delta_total"][0], delta, atol=1e-14
    )

    # Spectral identity: S[p,q] = sum_n C[p,n] conj(C[q,n]).
    loaded_overlap = data.overlap_k[0]
    c = c_loaded.T  # (nproj, nband)
    spectral = np.einsum("pn,qn->pq", c, c.conj())
    np.testing.assert_allclose(spectral, loaded_overlap, atol=1e-12)

    # The generalized eigenproblem recovers the producer eigenvalues.
    h_mat = np.einsum("pn,qn->pq", c * eps[None, :], c.conj())
    w_vals = sla.eigh(h_mat, loaded_overlap, eigvals_only=True)
    np.testing.assert_allclose(np.sort(w_vals), np.sort(eps), atol=1e-10)

    # The unconjugated convention would fail both identities.
    c_bug = proj.T
    assert np.max(np.abs(np.einsum("pn,qn->pq", c_bug, c_bug.conj()) - overlap)) > 1e-2


def test_abinit_netcdf_kind_classifies_nc_pao_wrapper(tmp_path):
    pytest.importorskip("netCDF4")
    nc_filename = tmp_path / "nc-pao-projector-wrapper.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(nc_filename)
    assert abinit_netcdf_kind(nc_filename) == "nc_pao"

    paw_filename = tmp_path / "paw-projector.nc"
    write_minimal_abinit_savetb2j_fixture(paw_filename)
    assert abinit_netcdf_kind(paw_filename) == "paw_savetb2j"


def test_gen_exchange_abinit_projector_refuses_nc_pao_wrapper(tmp_path):
    pytest.importorskip("netCDF4")
    nc_filename = tmp_path / "nc-pao-projector-wrapper.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(nc_filename)
    with pytest.raises(ValueError, match="k-dependent overlap_k metric"):
        gen_exchange_abinit_projector(
            nc_filename,
            output_path=tmp_path / "unused",
            Rpts=[(0, 0, 0)],
            nz=2,
            population_mode="none",
        )


@pytest.mark.parametrize("loader", [load_abinit_savetb2j, load_abinit_nc_pao_savetb2j])
@pytest.mark.parametrize(
    "variant",
    [
        "spinor",
        "magnetic",
        "wrong_units",
        "wrong_completeness",
        "missing_overlap_k",
        "missing_delta_total",
        "root_basis_conflict",
        "projector_basis_conflict",
        "component_basis_conflict",
        "wrong_coefficient_projector",
        "wrong_channel_interpretation",
    ],
)
def test_load_abinit_schema_12_nc_pao_projector_wrapper_rejects_invalid_payload(
    loader, variant, tmp_path
):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "invalid-nc-pao-projector-wrapper.nc"
    write_minimal_nc_pao_projector_wrapper_fixture(
        filename,
        include_overlap_k=variant != "missing_overlap_k",
        include_delta_total=variant != "missing_delta_total",
    )
    with netcdf4.Dataset(filename, "a") as nc:
        if variant == "spinor":
            nc.nspinor = 2
        elif variant == "magnetic":
            nc.groups["operators"].groups["operator_components"][
                "delta_total"
            ].spin_treatment = "magnetic_2x2"
        elif variant == "wrong_units":
            nc.groups["operators"].groups["operator_components"][
                "delta_total"
            ].units = "Hartree"
        elif variant == "wrong_completeness":
            nc.groups["operators"].groups["operator_components"][
                "delta_total"
            ].completeness = "partial"
        elif variant == "root_basis_conflict":
            nc.operator_basis = "abinit_native_paw_projector"
        elif variant == "projector_basis_conflict":
            nc.groups["projectors"].operator_basis = "abinit_native_paw_projector"
        elif variant == "component_basis_conflict":
            nc.groups["operators"].groups["operator_components"][
                "delta_total"
            ].operator_basis = "abinit_native_paw_projector"
        elif variant == "wrong_coefficient_projector":
            nc.groups["projectors"].coefficient_projector = "nc_pao"
        elif variant == "wrong_channel_interpretation":
            nc.groups["projectors"].channel_interpretation = "norm_conserving_pao"

    with pytest.raises(ValueError):
        loader(filename)


def test_load_abinit_savetb2j_generated_abinit_fixture():
    pytest.importorskip("netCDF4")
    if not (FIXTURE_DIR / "fe_savetb2j.nc").is_file():
        pytest.skip("fe_savetb2j.nc fixture not present in this worktree")

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


def test_abinit_savetb2j_occupation_features_fail_closed(tmp_path):
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)

    with pytest.raises(ValueError, match="require occupations"):
        compute_projected_charges_moments(data)
    with pytest.raises(ValueError, match="require occupations"):
        compute_projector_shell_populations(data)
    with pytest.raises(ValueError, match="require occupations"):
        build_nc_pao_band_mask(data, n_empty=1)
    with pytest.raises(ValueError, match="do not export occupations"):
        gen_exchange_abinit_projector(
            filename,
            output_path=tmp_path / "TB2J_results_projector_population",
            population_mode="projector",
        )


def test_load_abinit_savetb2j_13_exposes_number_of_electrons(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j_v13.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.3"
        nc.groups["bands"].number_of_electrons = 8.0

    data = load_abinit_savetb2j(filename)

    assert data.number_of_electrons == 8.0
    assert isinstance(data.number_of_electrons, float)
    # The occupations variable is still present in this file; it must be ignored.
    assert data.occupations is None

    exchange_out, exchange = gen_exchange_abinit_projector(
        filename,
        output_path=tmp_path / "TB2J_results_v13",
        Rpts=[(0, 0, 0)],
        nz=4,
        population_mode="none",
    )
    assert exchange_out.exists()
    assert exchange
    text = exchange_out.read_text(encoding="utf-8")
    assert "number_of_electrons: 8.0" in text


def test_load_abinit_savetb2j_13_rejects_missing_number_of_electrons(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j_v13_missing_nelect.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.3"

    with pytest.raises(ValueError, match="number_of_electrons"):
        load_abinit_savetb2j(filename)


def test_load_abinit_savetb2j_13_rejects_invalid_number_of_electrons(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j_v13_bad_nelect.nc"
    write_minimal_abinit_savetb2j_fixture(filename)
    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "1.3"
        nc.groups["bands"].number_of_electrons = 0.0

    with pytest.raises(ValueError, match="number_of_electrons"):
        load_abinit_savetb2j(filename)


def test_load_abinit_savetb2j_ignores_malformed_occupations(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j_malformed_occupations.nc"
    # Wrong rank, wrong dimensions, non-numeric type: none of it may affect
    # the load because occupation variables are never read.
    write_minimal_abinit_savetb2j_fixture(filename, occupations="malformed")

    data = load_abinit_savetb2j(filename)

    assert data.occupations is None


def test_abinit_paw_shell_moment_filter_masks_local_operator(tmp_path):
    # Shell selection/masking operate on population records; the records
    # themselves require occupations, which ABINIT savetb2j no longer
    # exports (see test_abinit_savetb2j_occupation_features_fail_closed).
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    data = load_abinit_savetb2j(filename)
    selected = select_nc_pao_shells(
        [
            {
                "projector_indices": [0],
                "charge": 1.0,
                "moment_norm": 1.0,
                "selected": True,
            },
            {
                "projector_indices": [1],
                "charge": 0.001,
                "moment_norm": 0.0025,
                "selected": False,
            },
        ],
        charge_threshold=0.0,
        moment_threshold=0.01,
    )
    assert [shell["selected"] for shell in selected] == [True, False]
    masked = mask_local_operators_by_shell_selection(
        data, {0: np.ones((2, 2), dtype=complex)}, selected
    )
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


def test_abinit_exchange_projector_population_fails_closed_green_works(tmp_path):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "abinit_savetb2j.nc"
    write_minimal_abinit_savetb2j_fixture(filename)

    # Occupation-based projector populations fail closed on ABINIT files.
    # Exchange itself keeps working via eigenvalues and the Fermi energy;
    # that path is covered by test_gen_exchange_abinit_projector_green_population.
    with pytest.raises(ValueError, match="do not export occupations"):
        gen_exchange_abinit_projector(
            filename,
            output_path=tmp_path / "TB2J_results_projector_population",
            population_mode="projector",
        )


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
        Rpts=[(0, 0, 0)],
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
            Rpts=[(0, 0, 0)],
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
            "abinit2J.py",
            "savetb2j",
            "--input",
            str(filename),
            "--output_path",
            str(output_path),
            "--Rcut",
            "10.0",
            "--nz",
            "4",
        ],
    )

    run_abinit2J()

    assert (output_path / "exchange.out").exists()
    assert "Wrote" in capsys.readouterr().out


def test_abinit_projector2j_cli_help_documents_workflow(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["abinit2J.py", "savetb2j", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        run_abinit2J()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "savetb2j" in help_text
    assert "delta_total" in help_text
    assert "--operator_component" in help_text
    assert "--population_mode" in help_text


def test_abinit_nc_pao2j_cli_help_documents_spherical_window(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["abinit2J.py", "savetb2j", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        run_abinit2J()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "NC" in help_text
    assert "delta_total" in help_text
