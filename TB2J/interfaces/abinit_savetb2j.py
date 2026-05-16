"""ABINIT savetb2j PAW-projector NetCDF loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from TB2J.interfaces.gpaw_projector import (
    _R_grid_for_cutoff,
    write_projector_exchange_out,
)
from TB2J.projector_green import ProjectorGreenData, decode_complex

ABINIT_SAVETB2J_SCHEMA_NAME = "abinit.savetb2j.projector"
ABINIT_SAVETB2J_SCHEMA_VERSION = "1.0"
ABINIT_OPERATOR_BASIS = "abinit_native_paw_projector"
ABINIT_COEFFICIENT_SOURCE = "abinit.cprj"
ABINIT_SPIN_CHANNEL_ORDER = "up,down"


def _require_attr(obj, name, context):
    if not hasattr(obj, name):
        raise ValueError(
            f"ABINIT savetb2j {context} missing required attribute: {name}"
        )
    return getattr(obj, name)


def _require_group(parent, name, context="file"):
    if name not in parent.groups:
        raise ValueError(f"ABINIT savetb2j {context} missing required group: {name}")
    return parent.groups[name]


def _require_var(group, name, context):
    if name not in group.variables:
        raise ValueError(f"ABINIT savetb2j {context} missing required array: {name}")
    return group.variables[name]


def _optional_var(group, name):
    if name in group.variables:
        return group.variables[name][:]
    return None


def _is_true(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes"}
    return bool(value)


def _decode_complex_var(group, name, context, required=True):
    if name not in group.variables:
        if required:
            raise ValueError(
                f"ABINIT savetb2j {context} missing required array: {name}"
            )
        return None
    variable = group.variables[name]
    if not variable.dimensions or variable.dimensions[-1] != "complex":
        raise ValueError(
            f"ABINIT savetb2j array {name} must use final complex dimension"
        )
    return decode_complex(variable[:])


def _validate_required_dimensions(nc):
    for name in (
        "nspin",
        "nkpt",
        "nband",
        "nproj",
        "nsite",
        "nproj_site_max",
        "natom",
        "three",
        "complex",
    ):
        if name not in nc.dimensions:
            raise ValueError(f"ABINIT savetb2j missing required dimension: {name}")
    if len(nc.dimensions["nspin"]) != 2:
        raise ValueError("ABINIT savetb2j v1 requires nspin=2")
    if len(nc.dimensions["complex"]) != 2:
        raise ValueError("ABINIT savetb2j complex dimension must have size 2")


def _root_metadata(nc):
    metadata = {}
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
    ):
        metadata[name] = _require_attr(nc, name, "root")

    if metadata["schema_name"] != ABINIT_SAVETB2J_SCHEMA_NAME:
        raise ValueError("unsupported ABINIT savetb2j schema_name")
    if metadata["schema_version"] != ABINIT_SAVETB2J_SCHEMA_VERSION:
        raise ValueError("unsupported ABINIT savetb2j schema_version")
    if metadata["source_code"] != "abinit":
        raise ValueError("ABINIT savetb2j source_code must be 'abinit'")
    if metadata["spin_mode"] != "collinear":
        raise ValueError("ABINIT savetb2j v1 requires spin_mode='collinear'")
    if metadata["spin_channel_order"] != ABINIT_SPIN_CHANNEL_ORDER:
        raise ValueError("ABINIT savetb2j spin_channel_order must be 'up,down'")
    if not _is_true(metadata["full_bz"]):
        raise ValueError("ABINIT savetb2j full_bz metadata must be true")
    if metadata["coefficient_source"] != ABINIT_COEFFICIENT_SOURCE:
        raise ValueError("ABINIT savetb2j coefficient_source must be abinit.cprj")
    if metadata["operator_basis"] != ABINIT_OPERATOR_BASIS:
        raise ValueError("ABINIT savetb2j root operator_basis is unsupported")

    try:
        metadata["units"] = json.loads(metadata["units_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("ABINIT savetb2j units_json must be valid JSON") from exc
    metadata["abinit_schema_name"] = metadata.pop("schema_name")
    metadata["abinit_schema_version"] = metadata.pop("schema_version")
    metadata["storage_level"] = "spectral"
    metadata["kpoint_set"] = "full_bz"
    return metadata


def _projector_metadata(projectors):
    coefficient_source = _require_attr(projectors, "coefficient_source", "projectors")
    coefficient_projector = _require_attr(
        projectors, "coefficient_projector", "projectors"
    )
    channel_interpretation = _require_attr(
        projectors, "channel_interpretation", "projectors"
    )
    operator_basis = _require_attr(projectors, "operator_basis", "projectors")
    index_base = _require_attr(projectors, "index_base", "projectors")

    if coefficient_source != ABINIT_COEFFICIENT_SOURCE:
        raise ValueError("ABINIT savetb2j projector coefficient_source mismatch")
    if coefficient_projector != "paw_nonlocal_projector":
        raise ValueError("ABINIT savetb2j unsupported coefficient_projector")
    if channel_interpretation != "abinit_paw_lmn_channel":
        raise ValueError("ABINIT savetb2j unsupported channel_interpretation")
    if operator_basis != ABINIT_OPERATOR_BASIS:
        raise ValueError("ABINIT savetb2j projector operator_basis mismatch")
    if int(index_base) != 0:
        raise ValueError("ABINIT savetb2j projector index_base must be 0")
    return (
        coefficient_source,
        coefficient_projector,
        channel_interpretation,
        operator_basis,
    )


def _component_metadata(variable):
    metadata = {}
    for name in ("source", "units", "operator_basis", "spin_treatment", "completeness"):
        metadata[name] = _require_attr(variable, name, f"component {variable.name}")
    if metadata["operator_basis"] != ABINIT_OPERATOR_BASIS:
        raise ValueError(
            f"ABINIT savetb2j component {variable.name} operator_basis mismatch"
        )
    if metadata["units"] != "eV":
        raise ValueError(f"ABINIT savetb2j component {variable.name} units must be eV")
    if metadata["spin_treatment"] not in {"spin_difference", "spin_resolved"}:
        raise ValueError(
            f"ABINIT savetb2j component {variable.name} has unsupported spin_treatment"
        )
    if metadata["completeness"] not in {
        "complete",
        "not_present",
        "zero_by_symmetry",
    }:
        raise ValueError(
            f"ABINIT savetb2j component {variable.name} has unsupported completeness"
        )
    return metadata


def _operator_data(operators):
    hij = hij_definition = hij_units = hij_source = hij_projection = None
    operator_basis = ABINIT_OPERATOR_BASIS
    if "hij" in operators.variables:
        hij_var = operators.variables["hij"]
        hij = _decode_complex_var(operators, "hij", "operators")
        hij_definition = _require_attr(hij_var, "definition", "hij")
        hij_units = _require_attr(hij_var, "units", "hij")
        hij_source = _require_attr(hij_var, "source", "hij")
        hij_projection = _require_attr(hij_var, "projection", "hij")
        operator_basis = _require_attr(hij_var, "operator_basis", "hij")
        if operator_basis != ABINIT_OPERATOR_BASIS:
            raise ValueError("ABINIT savetb2j hij operator_basis mismatch")
        if hij_units != "eV":
            raise ValueError("ABINIT savetb2j hij units must be eV")

    components_group = _require_group(operators, "operator_components", "operators")
    if "delta_total" not in components_group.variables and hij is None:
        raise ValueError("ABINIT savetb2j requires delta_total or spin-resolved hij")

    operator_components = {}
    operator_component_metadata = {}
    for name, variable in components_group.variables.items():
        values = _decode_complex_var(components_group, name, "operator_components")
        metadata = _component_metadata(variable)
        if metadata["spin_treatment"] != "spin_difference":
            raise ValueError(
                "ABINIT savetb2j v1 loader only normalizes spin_difference components"
            )
        operator_components[name] = values
        operator_component_metadata[name] = metadata

    return {
        "hij": hij,
        "hij_definition": hij_definition,
        "hij_units": hij_units,
        "hij_source": hij_source,
        "hij_projection": hij_projection,
        "operator_basis": operator_basis,
        "operator_components": operator_components or None,
        "operator_component_metadata": operator_component_metadata or None,
    }


def _validate_loaded_data(data):
    data.validate(exchange_ready=True)
    if data.atomic_numbers is None:
        raise ValueError("ABINIT savetb2j requires structure/atomic_numbers")
    if not np.isclose(np.sum(data.weights), 1.0):
        raise ValueError("ABINIT savetb2j k-point weights must sum to one")
    if data.operator_basis != ABINIT_OPERATOR_BASIS:
        raise ValueError("ABINIT savetb2j operator_basis mismatch")
    if data.coefficient_source != ABINIT_COEFFICIENT_SOURCE:
        raise ValueError("ABINIT savetb2j coefficient_source mismatch")
    if data.operator_components is not None:
        for name, value in data.operator_components.items():
            expected_shape = (
                len(data.site_nproj),
                data.site_projector_indices.shape[1],
                data.site_projector_indices.shape[1],
            )
            if value.shape != expected_shape:
                raise ValueError(f"ABINIT savetb2j component {name} has invalid shape")
    return data


def load_abinit_savetb2j(filename):
    """Load and strictly validate an ABINIT ``savetb2j`` NetCDF file."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("netCDF4 is required to load ABINIT savetb2j files") from exc

    with Dataset(Path(filename)) as nc:
        _validate_required_dimensions(nc)
        metadata = _root_metadata(nc)
        structure = _require_group(nc, "structure")
        kpoints = _require_group(nc, "kpoints")
        bands = _require_group(nc, "bands")
        projectors = _require_group(nc, "projectors")
        operators = _require_group(nc, "operators")

        (
            coefficient_source,
            coefficient_projector,
            channel_interpretation,
            projector_operator_basis,
        ) = _projector_metadata(projectors)
        operator_data = _operator_data(operators)
        if operator_data["operator_basis"] != projector_operator_basis:
            raise ValueError("ABINIT savetb2j operator_basis metadata mismatch")

        if not hasattr(bands, "efermi"):
            raise ValueError("ABINIT savetb2j bands group requires efermi attribute")

        data = ProjectorGreenData(
            kpoints=_require_var(kpoints, "kpoints", "kpoints")[:],
            weights=_require_var(kpoints, "weights", "kpoints")[:],
            eigenvalues=_require_var(bands, "eigenvalues", "bands")[:],
            occupations=_optional_var(bands, "occupations"),
            coefficients=_decode_complex_var(projectors, "coefficients", "projectors"),
            efermi=float(bands.efermi),
            projector_site=_require_var(projectors, "projector_site", "projectors")[:],
            projector_atom=_require_var(projectors, "projector_atom", "projectors")[:],
            cell=_require_var(structure, "cell", "structure")[:],
            positions=_require_var(structure, "positions", "structure")[:],
            atomic_numbers=_require_var(structure, "atomic_numbers", "structure")[:],
            projector_l=_optional_var(projectors, "projector_l"),
            projector_m=_optional_var(projectors, "projector_m"),
            projector_radial=_optional_var(projectors, "projector_radial"),
            overlap_metric=_decode_complex_var(
                projectors, "overlap_metric", "projectors", required=False
            ),
            site_nproj=_require_var(projectors, "site_nproj", "projectors")[:],
            site_projector_indices=_require_var(
                projectors, "site_projector_indices", "projectors"
            )[:],
            hij=operator_data["hij"],
            hij_definition=operator_data["hij_definition"],
            hij_units=operator_data["hij_units"],
            hij_source=operator_data["hij_source"],
            hij_projection=operator_data["hij_projection"],
            operator_components=operator_data["operator_components"],
            operator_component_metadata=operator_data["operator_component_metadata"],
            coefficient_source=coefficient_source,
            coefficient_projector=coefficient_projector,
            channel_interpretation=channel_interpretation,
            overlap_metric_definition="ABINIT PAW projector overlap metric",
            operator_basis=operator_data["operator_basis"],
            metadata=metadata,
        )
    return _validate_loaded_data(data)


load_abinit_savetb2j_projector = load_abinit_savetb2j


def _component_local_operators(data, component_name, sites):
    if component_name is None or component_name == "delta_total":
        component_name = "delta_total"
    if not data.has_operator_component(component_name):
        raise ValueError(
            f"ABINIT savetb2j operator component is unavailable: {component_name}"
        )
    metadata = (data.operator_component_metadata or {}).get(component_name, {})
    completeness = metadata.get("completeness")
    if completeness not in {None, "complete", "zero_by_symmetry"}:
        raise ValueError(
            "ABINIT savetb2j operator component is not exchange-ready: "
            f"{component_name} completeness={completeness!r}"
        )
    return {
        int(site): data.get_operator_component(component_name, site=site)
        for site in sites
    }


def gen_exchange_abinit_projector(
    filename,
    output_path="TB2J_results_abinit",
    Rmax=1,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    operator_component="delta_total",
    population_mode="none",
):
    """Generate projector exchange output from an ABINIT ``savetb2j`` file."""
    data = load_abinit_savetb2j(filename)
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    Rpts = _R_grid_for_cutoff(data, sites, Rcut, Rmax)
    local_operators = _component_local_operators(data, operator_component, sites)
    description = (
        "Projector Green workflow using ABINIT savetb2j PAW projections "
        f"({data.coefficient_source}) and operator component "
        f"{operator_component or 'delta_total'} in basis {data.operator_basis}. "
        "Values are from the controlled projector exchange-like trace. "
        f"ABINIT version: {data.metadata.get('abinit_version', 'unknown')}; "
        f"schema: {data.metadata.get('abinit_schema_name')} "
        f"{data.metadata.get('abinit_schema_version')}.\n"
    )
    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        population_mode=population_mode,
        Rcut=Rcut,
        local_operators=local_operators,
    )
