from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
SCHEMA_DOC = ROOT_DIR / "docs" / "src" / "abinit_savetb2j_schema.rst"


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
        nc.units_json = json.dumps(
            {
                "length": "Angstrom",
                "energy": "eV",
                "positions": "Angstrom",
                "eigenvalues": "eV",
                "operators": "eV",
            }
        )

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


def assert_minimal_abinit_contract(nc):
    missing = [
        name
        for name in (
            "schema_name",
            "schema_version",
            "source_code",
            "abinit_version",
            "spin_mode",
            "spin_channel_order",
            "full_bz",
            "kpoint_convention",
            "phase_convention",
            "coefficient_source",
            "operator_basis",
            "units_json",
        )
        if not hasattr(nc, name)
    ]
    assert not missing, f"missing root attributes: {missing}"
    assert nc.schema_name == "abinit.savetb2j.projector"
    assert nc.schema_version == "1.0"
    assert nc.source_code == "abinit"
    assert nc.abinit_version == "synthetic"
    assert nc.spin_mode == "collinear"
    assert nc.spin_channel_order == "up,down"
    assert nc.full_bz == 1
    assert nc.kpoint_convention == "fractional_reciprocal"
    assert nc.phase_convention == "exp(-2*pi*i*k.R)"
    assert nc.coefficient_source == "abinit.cprj"
    assert nc.operator_basis == "abinit_native_paw_projector"
    assert set(json.loads(nc.units_json)) == {
        "length",
        "energy",
        "positions",
        "eigenvalues",
        "operators",
    }
    assert "atomic_numbers" in nc.groups["structure"].variables
    projectors = nc.groups["projectors"]
    missing_projector_attrs = [
        name
        for name in (
            "coefficient_source",
            "coefficient_projector",
            "channel_interpretation",
            "operator_basis",
            "index_base",
        )
        if not hasattr(projectors, name)
    ]
    assert (
        not missing_projector_attrs
    ), f"missing projector attributes: {missing_projector_attrs}"
    assert projectors.index_base == 0
    assert projectors.operator_basis == "abinit_native_paw_projector"
    components = nc.groups["operators"].groups["operator_components"]
    assert components.variables["delta_total"].operator_basis == (
        "abinit_native_paw_projector"
    )


def test_abinit_savetb2j_schema_doc_defines_required_contract():
    text = SCHEMA_DOC.read_text(encoding="utf-8")

    required_terms = [
        "schema_name",
        "abinit.savetb2j.projector",
        "schema_version",
        "full_bz",
        "spin_channel_order",
        "abinit.cprj",
        "abinit_native_paw_projector",
        "delta_total",
        "dijxc",
        "dijU",
        "dijso",
        "operator_components/delta_total",
        "index_base = 0",
        "atomic_numbers(natom)",
        'spin_treatment = "spin_difference"',
        'spin_treatment = "spin_resolved"',
    ]
    for term in required_terms:
        assert term in text

    assert re.search(r"schema_name``\s*\n\s*-\s*``abinit\.savetb2j\.projector``", text)
    assert re.search(r"schema_version``\s*\n\s*-\s*``1\.0``", text)
    assert "Version 1 files must store zero-based" in text
    assert "This array is required in version 1" in text


def test_abinit_savetb2j_schema_doc_requires_strict_validation():
    text = SCHEMA_DOC.read_text(encoding="utf-8")

    validation_terms = [
        "unsupported",
        "required groups or arrays are missing",
        "operator basis metadata is absent",
        "spin_channel_order`` is absent or differs from ``up,down``",
        "atomic_numbers`` is missing",
        "index_base`` is absent or not zero",
        "exchange is requested",
        "Synthetic Fixture Requirements",
    ]
    for term in validation_terms:
        assert term in text


def test_abinit_savetb2j_schema_doc_is_in_sphinx_toctree():
    index = (ROOT_DIR / "docs" / "index.rst").read_text(encoding="utf-8")

    assert "src/projector_green.rst" in index
    assert "src/abinit_savetb2j_schema.rst" in index


def test_minimal_abinit_savetb2j_fixture_matches_schema_contract(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "minimal_abinit_savetb2j.nc"

    write_minimal_abinit_savetb2j_fixture(filename)

    with netcdf4.Dataset(filename) as nc:
        assert_minimal_abinit_contract(nc)
        assert nc.groups["projectors"].variables["coefficients"].dimensions[-1] == (
            "complex"
        )
        np.testing.assert_allclose(
            nc.groups["projectors"].variables["coefficients"][0, 0, 0, 0, :],
            [1.0, 0.25],
        )
        components = nc.groups["operators"].groups["operator_components"]
        assert set(components.variables) == {"delta_total", "dijxc", "dijU", "dijso"}
        assert components.variables["delta_total"].completeness == "complete"
        np.testing.assert_allclose(
            components.variables["delta_total"][:],
            nc.groups["operators"].variables["hij"][0, :, :, :, :]
            - nc.groups["operators"].variables["hij"][1, :, :, :, :],
        )


def test_minimal_abinit_savetb2j_fixture_rejects_missing_required_metadata(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "invalid_abinit_savetb2j.nc"

    write_minimal_abinit_savetb2j_fixture(filename)

    with netcdf4.Dataset(filename, "a") as nc:
        nc.delncattr("full_bz")

    with netcdf4.Dataset(filename) as nc:
        try:
            assert_minimal_abinit_contract(nc)
        except AssertionError as exc:
            assert "full_bz" in str(exc)
        else:
            raise AssertionError("fixture without full_bz should fail schema contract")


def test_minimal_abinit_savetb2j_fixture_rejects_missing_operator_basis(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "invalid_abinit_savetb2j_basis.nc"

    write_minimal_abinit_savetb2j_fixture(filename)

    with netcdf4.Dataset(filename, "a") as nc:
        del nc.groups["projectors"].operator_basis

    with netcdf4.Dataset(filename) as nc:
        try:
            assert_minimal_abinit_contract(nc)
        except AssertionError as exc:
            assert "operator_basis" in str(exc)
        else:
            raise AssertionError("fixture without projector operator_basis should fail")
