"""ABINIT savetb2j PAW-projector NetCDF loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.units import Bohr, kB

from TB2J.interfaces.gpaw_projector import (
    _R_grid_for_cutoff,
    component_local_operators,
    write_projector_exchange_out,
)
from TB2J.mycfr import CFR2
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    build_site_projector_indices,
    decode_complex,
    pack_site_hij,
    projector_charge_moments_from_green,
)

ABINIT_SAVETB2J_SCHEMA_NAME = "abinit.savetb2j.projector"
ABINIT_SAVETB2J_SCHEMA_VERSION = "1.0"
ABINIT_OPERATOR_BASIS = "abinit_native_paw_projector"
ABINIT_COEFFICIENT_SOURCE = "abinit.cprj"
ABINIT_SPIN_CHANNEL_ORDER = "up,down"
ABINIT_NC_PAO_SCHEMA_NAME = "abinit.savetb2j.nc_pao"
ABINIT_NC_PAO_SCHEMA_VERSION = "1.0"
ABINIT_NC_PAO_HS_SCHEMA_NAME = "abinit.nc_pao_hs"
ABINIT_NC_PAO_HS_SCHEMA_VERSION = "2"
ABINIT_NC_PAO_OPERATOR_BASIS = "abinit_nc_pao"
ABINIT_NC_PAO_HS_OPERATOR_BASIS = "norm_conserving_pao"
ABINIT_NC_PAO_COEFFICIENT_SOURCE = "abinit.nc_pao"
ABINIT_NC_SPHERICAL_SCHEMA_NAME = "abinit.savetb2j.nc_spherical_window"
ABINIT_NC_SPHERICAL_SCHEMA_VERSION = "1.0"
ABINIT_NC_SPHERICAL_OPERATOR_BASIS = "abinit_nc_spherical_window"
ABINIT_NC_SPHERICAL_COEFFICIENT_SOURCE = "abinit.nc_spherical_window"
HARTREE_TO_EV = 27.211386245988
ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT = "spectral_spin_split"
ABINIT_NC_PAO_ACTIVE_SHELL_OPERATOR_COMPONENT = "spectral_spin_split_active_shell"


def _attr_text(variable, name):
    return str(getattr(variable, name, "")).lower()


def _abinit_nc_pao_structure(nc):
    cell_var = _require_var(nc, "primitive_vectors", "NC PAO H/S")
    position_var = _require_var(nc, "atom_positions", "NC PAO H/S")
    cell = np.asarray(
        _array_in_dimension_order(cell_var, ("three", "three")), dtype=float
    )
    positions = np.asarray(
        _array_in_dimension_order(position_var, ("natom", "three")), dtype=float
    )

    cell_units = _attr_text(cell_var, "units")
    cell_text = cell_units + " " + _attr_text(cell_var, "mnemonics")
    if "bohr" in cell_text:
        cell = cell * Bohr

    position_units = _attr_text(position_var, "units")
    position_text = position_units + " " + _attr_text(position_var, "mnemonics")
    if "reduced" in position_text or "fractional" in position_text:
        positions = positions @ cell
    elif "bohr" in position_text:
        positions = positions * Bohr

    return cell, positions


def _hermitian_error(matrix):
    matrix = np.asarray(matrix, dtype=complex)
    return float(np.max(np.abs(matrix - matrix.conj().T)))


def _validate_local_operators_hermitian(local_operators, label, tolerance=1.0e-8):
    for site, matrix in local_operators.items():
        err = _hermitian_error(matrix)
        if err > tolerance:
            raise ValueError(
                f"ABINIT NC PAO operator component {label!r} is not Hermitian "
                f"on site {site}: max |Delta-Delta^dagger|={err:.3e} eV"
            )


def _population_contour(data, nz, smearing_eV):
    temperature = smearing_eV / kB
    fermi = data.efermi if data.efermi_spin is None else data.efermi_spin[:, None, None]
    spectral_range = float(np.max(np.abs(data.eigenvalues - fermi)))
    contour = CFR2(nz=nz, T=temperature)
    while np.max(np.abs(contour.path[:-1].imag)) < 4.0 * spectral_range:
        contour = CFR2(nz=2 * contour.nz, T=temperature)
    return contour


def compute_nc_pao_projected_charges_moments(data):
    """Compute NC PAO projected populations with a dual-basis Mulliken partition."""
    if data.occupations is None:
        raise ValueError("ABINIT NC PAO projected populations require occupations")
    if data.overlap_k is None:
        raise ValueError("ABINIT NC PAO projected populations require overlap_k")
    density_by_spin = np.zeros((data.nspin, len(data.site_nproj)), dtype=float)
    for spin in range(data.nspin):
        for ikpt, weight in enumerate(data.weights):
            coeff = data.coefficients[spin, ikpt]
            rho = np.einsum(
                "b,bp,bq->pq",
                data.occupations[spin, ikpt],
                coeff,
                coeff.conj(),
                optimize="optimal",
            )
            dual_density = np.linalg.inv(data.overlap_k[ikpt]) @ rho
            for site, nproj in enumerate(data.site_nproj):
                indices = data.site_projector_indices[site, :nproj]
                density_by_spin[spin, site] += float(
                    weight * np.real(np.trace(dual_density[np.ix_(indices, indices)]))
                )
    charges = np.sum(density_by_spin, axis=0)
    spinat = np.zeros((len(data.site_nproj), 3), dtype=float)
    if data.nspin >= 2:
        spinat[:, 2] = density_by_spin[0] - density_by_spin[1]
    return charges, spinat, density_by_spin


def _require_shell_metadata(data, source_label):
    if data.projector_l is None or data.projector_radial is None:
        raise ValueError(
            f"{source_label} shell diagnostics require projector_l and "
            "projector_radial/n_quantum metadata"
        )


def _shell_record(key, indices, density):
    site, radial, angular = key
    moment_z = float(density[0] - density[1]) if density.size >= 2 else 0.0
    moment = np.array([0.0, 0.0, moment_z], dtype=float)
    return {
        "site": site,
        "atom_index": site,
        "n_quantum": radial,
        "l_quantum": angular,
        "n_projectors": len(indices),
        "projector_indices": indices,
        "charge": float(np.sum(density)),
        "moment": moment,
        "moment_z": moment_z,
        "moment_norm": float(np.linalg.norm(moment)),
        "density_by_spin": density.copy(),
    }


def _shell_map(data):
    shell_map = {}
    for iproj, site in enumerate(data.projector_site):
        key = (
            int(site),
            int(data.projector_radial[iproj]),
            int(data.projector_l[iproj]),
        )
        shell_map.setdefault(key, []).append(int(iproj))
    return shell_map


def compute_nc_pao_shell_populations(data):
    """Compute NC PAO projected charge and moment by ``(site, n, l)`` shell."""
    if data.occupations is None:
        raise ValueError("ABINIT NC PAO shell populations require occupations")
    if data.overlap_k is None:
        raise ValueError("ABINIT NC PAO shell populations require overlap_k")
    _require_shell_metadata(data, "ABINIT NC PAO")

    shell_map = _shell_map(data)
    shell_spin_density = {key: np.zeros(data.nspin, dtype=float) for key in shell_map}
    for spin in range(data.nspin):
        for ikpt, weight in enumerate(data.weights):
            coeff = data.coefficients[spin, ikpt]
            rho = np.einsum(
                "b,bp,bq->pq",
                data.occupations[spin, ikpt],
                coeff,
                coeff.conj(),
                optimize="optimal",
            )
            dual_density = np.linalg.inv(data.overlap_k[ikpt]) @ rho
            diagonal = np.real(np.diag(dual_density))
            for key, indices in shell_map.items():
                shell_spin_density[key][spin] += float(
                    weight * np.sum(diagonal[indices])
                )

    records = []
    for key in sorted(shell_map):
        records.append(_shell_record(key, shell_map[key], shell_spin_density[key]))
    return records


def compute_projector_shell_populations(data):
    """Compute projector-space shell populations from occupations and a metric."""
    if data.occupations is None:
        raise ValueError("projector shell populations require occupations")
    _require_shell_metadata(data, "projector")
    shell_map = _shell_map(data)
    shell_spin_density = {key: np.zeros(data.nspin, dtype=float) for key in shell_map}
    metric = data.population_metric_matrix
    if metric is None:
        metric = data.overlap_metric

    for spin in range(data.nspin):
        rho = np.einsum(
            "k,kb,kbp,kbq->pq",
            data.weights,
            data.occupations[spin],
            data.coefficients[spin].conj(),
            data.coefficients[spin],
            optimize="optimal",
        )
        for key, indices in shell_map.items():
            block = rho[np.ix_(indices, indices)]
            if metric is None:
                value = np.trace(block)
            else:
                metric_block = metric[np.ix_(indices, indices)]
                value = np.trace(block @ metric_block)
            shell_spin_density[key][spin] += float(np.real(value))

    records = []
    for key in sorted(shell_map):
        records.append(_shell_record(key, shell_map[key], shell_spin_density[key]))
    return records


def select_nc_pao_shells(shells, charge_threshold=0.01, moment_threshold=0.01):
    """Annotate shell-population records with threshold selection metadata."""
    selected = []
    for shell in shells:
        record = dict(shell)
        reasons = []
        is_selected = True
        if charge_threshold is not None:
            passes_charge = float(record["charge"]) >= float(charge_threshold)
            is_selected = is_selected and passes_charge
            reasons.append(
                f"charge >= {charge_threshold:g}"
                if passes_charge
                else f"charge < {charge_threshold:g}"
            )
        if moment_threshold is not None:
            moment_norm = float(
                record.get("moment_norm", abs(record.get("moment_z", 0.0)))
            )
            passes_moment = moment_norm >= float(moment_threshold)
            is_selected = is_selected and passes_moment
            reasons.append(
                f"moment_norm >= {moment_threshold:g}"
                if passes_moment
                else f"moment_norm < {moment_threshold:g}"
            )
        record["selected"] = is_selected
        record["selection_reason"] = "; ".join(reasons) or "not filtered"
        selected.append(record)
    return selected


def mask_local_operators_by_shell_selection(data, local_operators, shells):
    """Zero excluded shell rows/columns in site-local operator blocks."""
    excluded = {
        int(iproj)
        for shell in shells
        if not shell.get("selected", False)
        for iproj in shell["projector_indices"]
    }
    masked = {}
    for site, operator in local_operators.items():
        block = np.array(operator, dtype=complex, copy=True)
        projectors = (
            data.get_site_projectors(site)
            if hasattr(data, "get_site_projectors")
            else None
        )
        if projectors is None:
            nproj = data.site_nproj[int(site)]
            projectors = data.site_projector_indices[int(site), :nproj]
        local_excluded = [
            i for i, iproj in enumerate(projectors) if int(iproj) in excluded
        ]
        if local_excluded:
            block[local_excluded, :] = 0.0
            block[:, local_excluded] = 0.0
        masked[int(site)] = block
    return masked


def build_nc_pao_band_mask(
    data, emax_eV=None, emax_relative_to_fermi_eV=None, n_empty=None
):
    """Build a boolean band mask and diagnostics for NC PAO spectral sums."""
    specified = [
        value is not None for value in (emax_eV, emax_relative_to_fermi_eV, n_empty)
    ]
    if sum(specified) > 1:
        raise ValueError(
            "Specify only one of emax_eV, emax_relative_to_fermi_eV, or n_empty"
        )
    if emax_eV is None and emax_relative_to_fermi_eV is None and n_empty is None:
        mask = np.ones_like(data.eigenvalues, dtype=bool)
        cutoff = None
    elif emax_eV is not None:
        cutoff = float(emax_eV)
        mask = data.eigenvalues <= cutoff
    elif n_empty is not None:
        if data.occupations is None:
            raise ValueError("Fixed empty-band windows require occupations")
        n_empty = int(n_empty)
        if n_empty < 0:
            raise ValueError("n_empty must be non-negative")
        occupied = data.occupations > 1.0e-8
        mask = np.array(occupied, dtype=bool, copy=True)
        for spin in range(data.nspin):
            for ikpt in range(data.nkpt):
                empty = np.flatnonzero(~occupied[spin, ikpt])
                if empty.size < n_empty:
                    raise ValueError(
                        "Fixed empty-band window requested more empty bands than "
                        f"available for spin {spin}, k-point {ikpt}: "
                        f"requested {n_empty}, available {empty.size}"
                    )
                order = empty[np.argsort(data.eigenvalues[spin, ikpt, empty])]
                mask[spin, ikpt, order[:n_empty]] = True
        cutoff = None
    else:
        if data.efermi_spin is not None:
            cutoff = data.efermi_spin[:, None, None] + float(emax_relative_to_fermi_eV)
        else:
            cutoff = float(data.efermi + emax_relative_to_fermi_eV)
        mask = data.eigenvalues <= cutoff

    included = np.sum(mask, axis=2)
    if data.occupations is None:
        included_unoccupied = np.zeros_like(included)
    else:
        occupied = data.occupations > 1.0e-8
        if np.any(occupied & ~mask):
            raise ValueError(
                "Eigenvalue window excludes occupied bands for at least one spin/k-point"
            )
        included_unoccupied = np.sum(mask & (data.occupations < 1.0e-8), axis=2)
    if np.any(included == 0):
        raise ValueError(
            "Eigenvalue window excludes all bands for at least one spin/k-point"
        )

    metadata = {
        "emax_eV": None if emax_eV is None else float(emax_eV),
        "emax_relative_to_fermi_eV": (
            None
            if emax_relative_to_fermi_eV is None
            else float(emax_relative_to_fermi_eV)
        ),
        "n_empty": None if n_empty is None else int(n_empty),
        "included_band_count_min": int(np.min(included)),
        "included_band_count_max": int(np.max(included)),
        "included_unoccupied_min": int(np.min(included_unoccupied)),
        "included_unoccupied_max": int(np.max(included_unoccupied)),
        "highest_included_eigenvalue_eV": float(np.max(data.eigenvalues[mask])),
        "cutoff_eV": cutoff.tolist() if isinstance(cutoff, np.ndarray) else cutoff,
    }
    return mask, metadata


def _format_nc_pao_diagnostics_report(
    data, shells, band_window_metadata=None, shell_filter_enabled=True
):
    n_selected = sum(1 for shell in shells if shell.get("selected", True))
    n_excluded = len(shells) - n_selected
    lines = [
        "# ABINIT NC PAO Diagnostics Report",
        "",
        f"Schema: {data.metadata.get('abinit_schema_name', data.schema_name)} {data.metadata.get('abinit_schema_version', data.schema_version)}",
        f"Fermi energy: {data.efermi:.6f} eV",
        "",
        "## Shell Populations",
        "",
        "| site | n | l | nproj | charge | moment_z | moment_norm | selected | reason |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for shell in shells:
        lines.append(
            f"| {shell['site']} | {shell['n_quantum']} | {shell['l_quantum']} | "
            f"{shell['n_projectors']} | {shell['charge']:.8f} | "
            f"{shell['moment_z']:.8f} | {shell.get('moment_norm', abs(shell['moment_z'])):.8f} | "
            f"{'yes' if shell.get('selected', True) else 'no'} | "
            f"{shell.get('selection_reason', 'not filtered')} |"
        )
    if not shells:
        lines.append(
            "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | shell diagnostics unavailable |"
        )
    lines.extend(
        [
            "",
            "## Shell Selection Summary",
            "",
            f"- shell_filtering: {'enabled' if shell_filter_enabled else 'disabled'}",
            f"- selected_shell_count: {n_selected}",
            f"- excluded_shell_count: {n_excluded}",
        ]
    )
    if not shells:
        lines.append("- assessment: shell diagnostics are unavailable")
    elif not shell_filter_enabled:
        lines.append("- assessment: shell filtering is disabled")
    elif n_excluded == 0:
        lines.append("- assessment: all shells are selected by the current threshold")
    else:
        lines.append(
            "- assessment: at least one shell is excluded by the current threshold"
        )
    if band_window_metadata is not None:
        lines.extend(
            [
                "",
                "## Band Window",
                "",
                f"- emax_eV: {band_window_metadata.get('emax_eV')}",
                f"- emax_relative_to_fermi_eV: {band_window_metadata.get('emax_relative_to_fermi_eV')}",
                f"- n_empty: {band_window_metadata.get('n_empty')}",
                f"- included_band_count_min: {band_window_metadata['included_band_count_min']}",
                f"- included_band_count_max: {band_window_metadata['included_band_count_max']}",
                f"- included_unoccupied_min: {band_window_metadata['included_unoccupied_min']}",
                f"- included_unoccupied_max: {band_window_metadata['included_unoccupied_max']}",
                f"- highest_included_eigenvalue_eV: {band_window_metadata['highest_included_eigenvalue_eV']:.6f}",
            ]
        )
    return "\n".join(lines) + "\n"


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


def _decode_split_complex_var(group, name, context, required=True):
    real_name = f"{name}_real"
    imag_name = f"{name}_imag"
    if real_name not in group.variables or imag_name not in group.variables:
        if required:
            raise ValueError(
                f"ABINIT savetb2j {context} missing required split complex array: {name}"
            )
        return None
    return group.variables[real_name][:] + 1j * group.variables[imag_name][:]


def _array_in_dimension_order(variable, target_dimensions):
    array = variable[:]
    dimensions = tuple(variable.dimensions)
    target_dimensions = tuple(target_dimensions)
    if dimensions == target_dimensions:
        return array
    if sorted(dimensions) != sorted(target_dimensions):
        raise ValueError(
            f"array {variable.name} has dimensions {dimensions}, "
            f"expected a permutation of {target_dimensions}"
        )
    axes = [dimensions.index(dim) for dim in target_dimensions]
    return np.transpose(array, axes)


def _decode_split_complex_var_ordered(
    group, name, context, target_dimensions, required=True
):
    real_name = f"{name}_real"
    imag_name = f"{name}_imag"
    if real_name not in group.variables or imag_name not in group.variables:
        if required:
            raise ValueError(
                f"ABINIT savetb2j {context} missing required split complex array: {name}"
            )
        return None
    real = _array_in_dimension_order(group.variables[real_name], target_dimensions)
    imag = _array_in_dimension_order(group.variables[imag_name], target_dimensions)
    return real + 1j * imag


def _decode_packed_or_split_complex_var(group, name, context, required=True):
    if name in group.variables:
        return _decode_complex_var(group, name, context, required=required)
    return _decode_split_complex_var(group, name, context, required=required)


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
    for name in (
        "site_window",
        "smooth_xc_included",
        "paw_ae_minus_ps_included",
        "hubbard_included",
        "exchange_ready",
    ):
        if hasattr(variable, name):
            metadata[name] = getattr(variable, name)
    if metadata["operator_basis"] != ABINIT_OPERATOR_BASIS:
        raise ValueError(
            f"ABINIT savetb2j component {variable.name} operator_basis mismatch"
        )
    if metadata["units"] != "eV":
        raise ValueError(f"ABINIT savetb2j component {variable.name} units must be eV")
    if metadata["spin_treatment"] not in {
        "spin_difference",
        "spin_resolved",
        "up_minus_down",
    }:
        raise ValueError(
            f"ABINIT savetb2j component {variable.name} has unsupported spin_treatment"
        )
    if metadata["completeness"] not in {
        "complete",
        "not_present",
        "zero_by_symmetry",
        "smooth_site_window",
        "paw_onsite_xc",
        "paw_onsite_u",
        "smooth_plus_paw_onsite_xc",
        "smooth_plus_paw_onsite_xc_u",
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
        if metadata["spin_treatment"] not in {"spin_difference", "up_minus_down"}:
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


def _nc_pao_root_metadata(nc):
    metadata = {}
    for name in (
        "schema_name",
        "schema_version",
        "source_code",
        "spin_mode",
        "spin_channel_order",
        "full_bz",
        "coefficient_source",
        "operator_basis",
    ):
        metadata[name] = _require_attr(nc, name, "NC PAO root")
    if metadata["schema_name"] != ABINIT_NC_PAO_SCHEMA_NAME:
        raise ValueError("unsupported ABINIT NC PAO schema_name")
    if metadata["schema_version"] != ABINIT_NC_PAO_SCHEMA_VERSION:
        raise ValueError("unsupported ABINIT NC PAO schema_version")
    if metadata["source_code"] != "abinit":
        raise ValueError("ABINIT NC PAO source_code must be 'abinit'")
    if metadata["spin_mode"] != "collinear":
        raise ValueError("ABINIT NC PAO v1 requires spin_mode='collinear'")
    if metadata["spin_channel_order"] != ABINIT_SPIN_CHANNEL_ORDER:
        raise ValueError("ABINIT NC PAO spin_channel_order must be 'up,down'")
    if not _is_true(metadata["full_bz"]):
        raise ValueError("ABINIT NC PAO full_bz metadata must be true")
    if metadata["coefficient_source"] != ABINIT_NC_PAO_COEFFICIENT_SOURCE:
        raise ValueError("ABINIT NC PAO coefficient_source mismatch")
    if metadata["operator_basis"] != ABINIT_NC_PAO_OPERATOR_BASIS:
        raise ValueError("ABINIT NC PAO operator_basis mismatch")
    metadata["abinit_schema_name"] = metadata.pop("schema_name")
    metadata["abinit_schema_version"] = metadata.pop("schema_version")
    metadata["storage_level"] = "spectral"
    metadata["kpoint_set"] = "full_bz"
    return metadata


def _nc_pao_component_metadata(variable):
    metadata = {}
    for name in ("source", "units", "operator_basis", "spin_treatment", "completeness"):
        metadata[name] = _require_attr(
            variable, name, f"NC PAO component {variable.name}"
        )
    if metadata["operator_basis"] != ABINIT_NC_PAO_OPERATOR_BASIS:
        raise ValueError(
            f"ABINIT NC PAO component {variable.name} operator_basis mismatch"
        )
    if metadata["units"] != "eV":
        raise ValueError(f"ABINIT NC PAO component {variable.name} units must be eV")
    if metadata["spin_treatment"] != "spin_difference":
        raise ValueError(
            f"ABINIT NC PAO component {variable.name} must use spin_difference"
        )
    if metadata["completeness"] not in {"complete", "zero_by_symmetry"}:
        raise ValueError(
            f"ABINIT NC PAO component {variable.name} is not exchange-ready"
        )
    return metadata


def _nc_pao_operator_components(operators):
    components_group = _require_group(
        operators, "operator_components", "NC PAO operators"
    )
    if "delta_total" not in components_group.variables:
        raise ValueError(
            "ABINIT NC PAO exchange requires delta_total operator component"
        )
    operator_components = {}
    operator_component_metadata = {}
    for name, variable in components_group.variables.items():
        values = _decode_complex_var(
            components_group, name, "NC PAO operator_components"
        )
        metadata = _nc_pao_component_metadata(variable)
        operator_components[name] = values
        operator_component_metadata[name] = metadata
    return operator_components, operator_component_metadata


def _nc_pao_hs_attr(nc, name, default=None):
    return getattr(nc, name, default)


def _load_abinit_nc_pao_hs_v2(nc):
    """Load ABINIT's split-real/imag NC PAO H/S schema-v2 file."""
    schema_version = str(_require_attr(nc, "schema_version", "NC PAO root"))
    if schema_version != ABINIT_NC_PAO_HS_SCHEMA_VERSION:
        raise ValueError("unsupported ABINIT NC PAO H/S schema_version")
    if str(_nc_pao_hs_attr(nc, "basis_type", "")) != "pseudo_atomic_orbital":
        raise ValueError("ABINIT NC PAO H/S basis_type mismatch")
    if str(_nc_pao_hs_attr(nc, "complex_storage", "")) != "split_real_imag_variables":
        raise ValueError("ABINIT NC PAO H/S requires split_real_imag_variables")
    if int(_nc_pao_hs_attr(nc, "overlap_exchange_ready", 0)) != 1:
        raise ValueError(
            "ABINIT NC PAO H/S overlap metadata is not exchange-ready; "
            "overlap_exchange_ready must be 1"
        )

    for name in ("nproj", "nkpt_ibz", "nsppol", "nband", "natom"):
        if name not in nc.dimensions:
            raise ValueError(f"ABINIT NC PAO H/S missing required dimension: {name}")
    if len(nc.dimensions["nsppol"]) != 2:
        raise ValueError("ABINIT NC PAO H/S requires nsppol=2")

    atom_index = _require_var(nc, "atom_index", "NC PAO H/S")[:].astype(int) - 1
    if np.any(atom_index < 0):
        raise ValueError("ABINIT NC PAO H/S atom_index must be 1-based positive")
    site_nproj, site_projector_indices = build_site_projector_indices(atom_index)
    typat = _require_var(nc, "atom_types", "NC PAO H/S")[:].astype(int)
    znucl = _require_var(nc, "atomic_numbers", "NC PAO H/S")[:]
    atomic_numbers = znucl[typat - 1].astype(int)

    kpoint_set = "ibz"
    kpoints = _array_in_dimension_order(
        _require_var(nc, "kpoints_ibz", "NC PAO H/S"), ("nkpt_ibz", "three")
    )
    if "kweights_ibz" in nc.variables:
        weights = nc.variables["kweights_ibz"][:]
    else:
        weights = np.ones(kpoints.shape[0], dtype=float) / kpoints.shape[0]
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    eigenvalues = (
        _array_in_dimension_order(
            _require_var(nc, "eigenvalues", "NC PAO H/S"),
            ("nsppol", "nkpt_ibz", "nband"),
        )
        * HARTREE_TO_EV
    )
    occupations = _optional_var(nc, "occupations")
    if occupations is not None:
        occupations = _array_in_dimension_order(
            nc.variables["occupations"], ("nsppol", "nkpt_ibz", "nband")
        )
    # abinao stores <psi|phi>; ProjectorGreen uses <phi|psi> like GPAW.
    coefficients = _decode_split_complex_var_ordered(
        nc, "coefficients_ibz", "NC PAO H/S", ("nsppol", "nkpt_ibz", "nband", "nproj")
    ).conj()
    overlap_k = _decode_split_complex_var_ordered(
        nc, "overlap_ibz", "NC PAO H/S", ("nkpt_ibz", "nproj", "nproj")
    )

    if (
        "kpoints_bz" in nc.variables
        and "coefficients_bz_real" in nc.variables
        and "coefficients_bz_imag" in nc.variables
        and "overlap_bz_real" in nc.variables
        and "overlap_bz_imag" in nc.variables
    ):
        kpoint_set = "full_bz"
        kpoints = _array_in_dimension_order(
            nc.variables["kpoints_bz"], ("nkpt_bz", "three")
        )
        weights = np.ones(kpoints.shape[0], dtype=float) / kpoints.shape[0]
        coefficients = _decode_split_complex_var_ordered(
            nc, "coefficients_bz", "NC PAO H/S", ("nsppol", "nkpt_bz", "nband", "nproj")
        ).conj()
        overlap_k = _decode_split_complex_var_ordered(
            nc, "overlap_bz", "NC PAO H/S", ("nkpt_bz", "nproj", "nproj")
        )
        bz_to_ibz = np.asarray(
            _require_var(nc, "bz_to_ibz", "NC PAO H/S")[:], dtype=int
        )
        if bz_to_ibz.size and np.min(bz_to_ibz) >= 1:
            bz_to_ibz = bz_to_ibz - 1
        if np.any(bz_to_ibz < 0) or np.any(bz_to_ibz >= eigenvalues.shape[1]):
            raise ValueError("ABINIT NC PAO H/S bz_to_ibz contains invalid indices")
        eigenvalues = eigenvalues[:, bz_to_ibz, :]
        if occupations is not None:
            occupations = occupations[:, bz_to_ibz, :]
        # Use BZ-resolved eigenvalues/occupations if available (needed when
        # the IBZ-to-BZ expansion applies per-k transforms like spin-flip).
        if "eigenvalues_bz" in nc.variables:
            eigenvalues = (
                _array_in_dimension_order(
                    _require_var(nc, "eigenvalues_bz", "NC PAO H/S"),
                    ("nsppol", "nkpt_bz", "nband"),
                )
                * HARTREE_TO_EV
            )
        if occupations is not None and "occupations_bz" in nc.variables:
            occupations = _array_in_dimension_order(
                nc.variables["occupations_bz"], ("nsppol", "nkpt_bz", "nband")
            )

    component_sources = {
        "delta_xc_smooth": _decode_split_complex_var_ordered(
            nc, "delta_xc_smooth", "NC PAO H/S", ("nproj", "nproj"), required=False
        ),
        "delta_U": _decode_split_complex_var_ordered(
            nc, "delta_U", "NC PAO H/S", ("nproj", "nproj"), required=False
        ),
        "delta_total": _decode_split_complex_var_ordered(
            nc, "delta_total", "NC PAO H/S", ("nproj", "nproj"), required=True
        ),
    }
    delta_u_matrix = component_sources.get("delta_U")
    if "hamiltonian_bz_real" in nc.variables and "hamiltonian_bz_imag" in nc.variables:
        hamiltonian_bz = _decode_split_complex_var_ordered(
            nc,
            "hamiltonian_bz",
            "NC PAO H/S",
            ("nsppol", "nkpt_bz", "nproj", "nproj"),
            required=False,
        )
        if hamiltonian_bz is not None and hamiltonian_bz.shape[0] == 2:
            spectral_spin_split = np.einsum(
                "k,kij->ij",
                weights,
                hamiltonian_bz[0] - hamiltonian_bz[1],
                optimize="optimal",
            )
            component_sources["spectral_spin_split"] = spectral_spin_split
            if delta_u_matrix is not None:
                active = np.any(np.abs(delta_u_matrix) > 1.0e-12, axis=0) | np.any(
                    np.abs(delta_u_matrix) > 1.0e-12, axis=1
                )
                if np.any(active):
                    active_shell_split = np.zeros_like(spectral_spin_split)
                    active_shell_split[np.ix_(active, active)] = spectral_spin_split[
                        np.ix_(active, active)
                    ]
                    component_sources[ABINIT_NC_PAO_ACTIVE_SHELL_OPERATOR_COMPONENT] = (
                        active_shell_split
                    )
    operator_components = {}
    operator_component_metadata = {}
    for name, matrix in component_sources.items():
        if matrix is None:
            continue
        component = pack_site_hij(
            matrix[None, :, :] * HARTREE_TO_EV, site_projector_indices, site_nproj
        )[0]
        operator_components[name] = component
        operator_component_metadata[name] = {
            "source": (
                "abinit.nc_pao_hs.v2.hamiltonian_bz_spin_difference"
                if name
                in {
                    "spectral_spin_split",
                    ABINIT_NC_PAO_ACTIVE_SHELL_OPERATOR_COMPONENT,
                }
                else "abinit.nc_pao_hs.v2"
            ),
            "units": "eV",
            "operator_basis": ABINIT_NC_PAO_HS_OPERATOR_BASIS,
            "spin_treatment": "up_minus_down",
            "completeness": "complete",
        }

    efermi = float(_nc_pao_hs_attr(nc, "fermi_energy", 0.0)) * HARTREE_TO_EV
    metadata = {
        "abinit_schema_name": ABINIT_NC_PAO_HS_SCHEMA_NAME,
        "abinit_schema_version": schema_version,
        "storage_level": "spectral",
        "kpoint_set": kpoint_set,
        "pao_coefficient_kpoint_set": _nc_pao_hs_attr(
            nc, "pao_coefficient_kpoint_set", kpoint_set
        ),
        "coefficient_source": ABINIT_NC_PAO_COEFFICIENT_SOURCE,
        "operator_basis": ABINIT_NC_PAO_HS_OPERATOR_BASIS,
        "overlap_condition_threshold": 1.0e12,
        "overlap_validation_status": _nc_pao_hs_attr(
            nc, "overlap_validation_status", "unknown"
        ),
    }

    cell, positions = _abinit_nc_pao_structure(nc)

    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
        efermi=efermi,
        projector_site=atom_index,
        projector_atom=atom_index,
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        projector_l=_optional_var(nc, "l_quantum"),
        projector_m=_optional_var(nc, "m_quantum"),
        projector_radial=_optional_var(nc, "n_quantum"),
        overlap_k=overlap_k,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        operator_components=operator_components,
        operator_component_metadata=operator_component_metadata,
        coefficient_source=ABINIT_NC_PAO_COEFFICIENT_SOURCE,
        coefficient_projector="nc_pao",
        channel_interpretation="norm_conserving_pao",
        overlap_metric_definition=_nc_pao_hs_attr(
            nc, "pao_overlap_convention", "k-dependent NC PAO overlap"
        ),
        population_metric="unavailable: k-dependent NC PAO overlap_k is not a population metric",
        operator_basis=ABINIT_NC_PAO_HS_OPERATOR_BASIS,
        metadata=metadata,
    )
    data.validate(exchange_ready=data.has_operator_component("delta_total"))
    return data


def _load_abinit_nc_spherical_window(nc):
    """Load ABINIT's NC spherical-window savetb2j schema."""
    schema_version = str(_require_attr(nc, "schema_version", "NC spherical root"))
    if schema_version != ABINIT_NC_SPHERICAL_SCHEMA_VERSION:
        raise ValueError("unsupported ABINIT NC spherical-window schema_version")
    if str(_require_attr(nc, "basis_type", "NC spherical root")) != "spherical_window":
        raise ValueError("ABINIT NC spherical-window basis_type mismatch")
    if (
        str(_require_attr(nc, "operator_basis", "NC spherical root"))
        != ABINIT_NC_SPHERICAL_OPERATOR_BASIS
    ):
        raise ValueError("ABINIT NC spherical-window operator_basis mismatch")
    if int(_require_attr(nc, "metric_required", "NC spherical root")) != 1:
        raise ValueError("ABINIT NC spherical-window requires metric_required=1")
    if int(_require_attr(nc, "full_bz", "NC spherical root")) != 1:
        raise ValueError("ABINIT NC spherical-window full_bz metadata must be true")

    for name in ("nproj", "nkpt", "nsppol", "nband", "natom", "ntypat"):
        if name not in nc.dimensions:
            raise ValueError(
                f"ABINIT NC spherical-window missing required dimension: {name}"
            )
    if len(nc.dimensions["nsppol"]) != 2:
        raise ValueError("ABINIT NC spherical-window requires nsppol=2")

    coefficients = _decode_split_complex_var_ordered(
        nc,
        "coefficients",
        "NC spherical-window",
        ("nsppol", "nkpt", "nband", "nproj"),
    )
    overlap_k = _decode_split_complex_var(nc, "overlap_k", "NC spherical-window")
    nproj = len(nc.dimensions["nproj"])
    nkpt = len(nc.dimensions["nkpt"])
    if overlap_k.shape == (nproj, nproj, nkpt):
        overlap_k = np.transpose(overlap_k, (2, 0, 1))
    site_index = _require_var(nc, "projector_site", "NC spherical-window")[:].astype(
        int
    )
    if np.min(site_index) >= 1:
        site_index = site_index - 1
    if np.any(site_index < 0):
        raise ValueError("ABINIT NC spherical-window projector_site must be positive")
    site_nproj, site_projector_indices = build_site_projector_indices(site_index)

    typat = _require_var(nc, "typat", "NC spherical-window")[:].astype(int)
    znucl = _require_var(nc, "znucl", "NC spherical-window")[:]
    atomic_numbers = znucl[typat - 1].astype(int)
    cell = (
        np.asarray(
            _array_in_dimension_order(
                _require_var(nc, "rprimd", "NC spherical-window"), ("three", "three")
            ),
            dtype=float,
        )
        * Bohr
    )
    xred = np.asarray(
        _array_in_dimension_order(
            _require_var(nc, "xred", "NC spherical-window"), ("natom", "three")
        ),
        dtype=float,
    )
    positions = xred @ cell
    weights = _require_var(nc, "kweights", "NC spherical-window")[:]
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)
    eigenvalues = (
        _array_in_dimension_order(
            _require_var(nc, "eigenvalues", "NC spherical-window"),
            ("nsppol", "nkpt", "nband"),
        )
        * HARTREE_TO_EV
    )
    occupations = _optional_var(nc, "occupations")
    if occupations is not None:
        occupations = _array_in_dimension_order(
            nc.variables["occupations"], ("nsppol", "nkpt", "nband")
        )

    component_sources = {}
    delta_xc = _decode_split_complex_var(
        nc, "delta_xc_spherical", "NC spherical-window", required=False
    )
    delta_u = _decode_split_complex_var(
        nc, "delta_u_spherical", "NC spherical-window", required=False
    )
    delta_total = _decode_split_complex_var(
        nc, "delta_spherical_xc_u", "NC spherical-window", required=False
    )
    if delta_xc is not None:
        component_sources["delta_xc_spherical"] = delta_xc
    if delta_u is not None:
        component_sources["delta_u_spherical"] = delta_u
    if delta_total is not None:
        component_sources["delta_spherical_xc_u"] = delta_total
        component_sources["delta_total"] = delta_total

    operator_components = {}
    operator_component_metadata = {}
    for name, matrix in component_sources.items():
        component = pack_site_hij(
            matrix[None, :, :] * HARTREE_TO_EV, site_projector_indices, site_nproj
        )[0]
        operator_components[name] = component
        operator_component_metadata[name] = {
            "source": "abinit.savetb2j.nc_spherical_window",
            "units": "eV",
            "operator_basis": ABINIT_NC_SPHERICAL_OPERATOR_BASIS,
            "spin_treatment": "up_minus_down",
            "completeness": "xc_plus_u"
            if name in {"delta_spherical_xc_u", "delta_total"}
            else "diagnostic",
            "exchange_ready": "true"
            if name in {"delta_spherical_xc_u", "delta_total"}
            else "false",
        }

    efermi = float(_nc_pao_hs_attr(nc, "fermi_energy_hartree", 0.0)) * HARTREE_TO_EV
    metadata = {
        "abinit_schema_name": ABINIT_NC_SPHERICAL_SCHEMA_NAME,
        "abinit_schema_version": schema_version,
        "storage_level": "spectral",
        "kpoint_set": "full_bz",
        "coefficient_source": ABINIT_NC_SPHERICAL_COEFFICIENT_SOURCE,
        "operator_basis": ABINIT_NC_SPHERICAL_OPERATOR_BASIS,
        "basis_lmax": _nc_pao_hs_attr(nc, "basis_lmax", None),
        "soft_cutoff_width_bohr": _nc_pao_hs_attr(nc, "soft_cutoff_width_bohr", None),
    }

    data = ProjectorGreenData(
        kpoints=_array_in_dimension_order(
            _require_var(nc, "kpoints", "NC spherical-window"), ("nkpt", "three")
        ),
        weights=weights,
        eigenvalues=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
        efermi=efermi,
        projector_site=site_index,
        projector_atom=site_index,
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        projector_l=_optional_var(nc, "projector_l"),
        projector_m=_optional_var(nc, "projector_m"),
        projector_radial=np.zeros(coefficients.shape[-1], dtype=int),
        overlap_k=overlap_k,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        operator_components=operator_components,
        operator_component_metadata=operator_component_metadata,
        coefficient_source=ABINIT_NC_SPHERICAL_COEFFICIENT_SOURCE,
        coefficient_projector="nc_spherical_window",
        channel_interpretation="real_spherical_harmonic_window",
        overlap_metric_definition="k-dependent NC spherical-window overlap_k",
        population_metric="unavailable: k-dependent spherical-window overlap_k is not a population metric",
        operator_basis=ABINIT_NC_SPHERICAL_OPERATOR_BASIS,
        metadata=metadata,
    )
    data.validate(exchange_ready=data.has_operator_component("delta_total"))
    return data


def load_abinit_nc_pao_savetb2j(filename):
    """Load and strictly validate an ABINIT norm-conserving PAO savetb2j file."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("netCDF4 is required to load ABINIT NC PAO files") from exc

    with Dataset(Path(filename)) as nc:
        schema_name = getattr(nc, "schema_name", None)
        if schema_name == ABINIT_NC_SPHERICAL_SCHEMA_NAME:
            return _load_abinit_nc_spherical_window(nc)
        if schema_name == ABINIT_NC_PAO_HS_SCHEMA_NAME:
            return _load_abinit_nc_pao_hs_v2(nc)
        _validate_required_dimensions(nc)
        metadata = _nc_pao_root_metadata(nc)
        structure = _require_group(nc, "structure")
        kpoints = _require_group(nc, "kpoints")
        bands = _require_group(nc, "bands")
        projectors = _require_group(nc, "projectors")
        operators = _require_group(nc, "operators")
        if not hasattr(bands, "efermi"):
            raise ValueError("ABINIT NC PAO bands group requires efermi attribute")
        if (
            _require_attr(projectors, "coefficient_projector", "NC PAO projectors")
            != "nc_pao"
        ):
            raise ValueError("ABINIT NC PAO coefficient_projector must be nc_pao")
        if (
            _require_attr(projectors, "channel_interpretation", "NC PAO projectors")
            != "norm_conserving_pao"
        ):
            raise ValueError("ABINIT NC PAO channel_interpretation mismatch")

        overlap_var = _require_var(projectors, "overlap_k", "NC PAO projectors")
        if (
            not hasattr(overlap_var, "definition")
            or not str(overlap_var.definition).strip()
        ):
            raise ValueError("ABINIT NC PAO overlap_k requires validation metadata")
        overlap_k = _decode_complex_var(projectors, "overlap_k", "NC PAO projectors")
        operator_components, operator_component_metadata = _nc_pao_operator_components(
            operators
        )

        data = ProjectorGreenData(
            kpoints=_require_var(kpoints, "kpoints", "kpoints")[:],
            weights=_require_var(kpoints, "weights", "kpoints")[:],
            eigenvalues=_require_var(bands, "eigenvalues", "bands")[:],
            occupations=_optional_var(bands, "occupations"),
            coefficients=_decode_complex_var(
                projectors, "coefficients", "NC PAO projectors"
            ),
            efermi=float(bands.efermi),
            projector_site=_require_var(
                projectors, "projector_site", "NC PAO projectors"
            )[:],
            projector_atom=_require_var(
                projectors, "projector_atom", "NC PAO projectors"
            )[:],
            cell=_require_var(structure, "cell", "structure")[:],
            positions=_require_var(structure, "positions", "structure")[:],
            atomic_numbers=_require_var(structure, "atomic_numbers", "structure")[:],
            projector_l=_optional_var(projectors, "projector_l"),
            projector_m=_optional_var(projectors, "projector_m"),
            projector_radial=_optional_var(projectors, "projector_radial"),
            overlap_k=overlap_k,
            site_nproj=_require_var(projectors, "site_nproj", "NC PAO projectors")[:],
            site_projector_indices=_require_var(
                projectors, "site_projector_indices", "NC PAO projectors"
            )[:],
            operator_components=operator_components,
            operator_component_metadata=operator_component_metadata,
            coefficient_source=metadata["coefficient_source"],
            coefficient_projector="nc_pao",
            channel_interpretation="norm_conserving_pao",
            overlap_metric_definition=overlap_var.definition,
            population_metric="k-dependent NC PAO overlap_k",
            operator_basis=metadata["operator_basis"],
            metadata=metadata,
        )
    data.validate(exchange_ready=True)
    if not np.isclose(np.sum(data.weights), 1.0):
        raise ValueError("ABINIT NC PAO k-point weights must sum to one")
    return data


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

        overlap_metric = _decode_complex_var(
            projectors, "overlap_metric", "projectors", required=False
        )

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
            overlap_metric=overlap_metric,
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
            overlap_metric_definition="ABINIT PAW packed nonlocal sij overlap correction",
            population_metric=(
                "unavailable: ABINIT savetb2j v1 exports cprj and nonlocal sij "
                "but not the smooth pseudo-density contribution needed for PAW "
                "charge/moment populations"
            ),
            operator_basis=operator_data["operator_basis"],
            metadata=metadata,
        )
    return _validate_loaded_data(data)


load_abinit_savetb2j_projector = load_abinit_savetb2j


def gen_exchange_abinit_projector(
    filename,
    output_path="TB2J_results_abinit",
    Rmax=None,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    operator_component="delta_total",
    population_mode="none",
    shell_charge_threshold=None,
    shell_moment_threshold=None,
):
    """Generate projector exchange output from an ABINIT ``savetb2j`` file."""
    data = load_abinit_savetb2j(filename)
    if population_mode not in {"none", "projector", "green"}:
        raise ValueError(f"unsupported population_mode: {population_mode}")
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    Rpts = _R_grid_for_cutoff(data, sites, Rcut, Rmax)
    local_operators = component_local_operators(
        data, operator_component, sites, "ABINIT savetb2j"
    )
    if shell_charge_threshold is not None or shell_moment_threshold is not None:
        if data.population_metric_matrix is None:
            raise ValueError(
                "ABINIT PAW shell filtering requires a PAW-complete "
                "population_metric_matrix. The v1 overlap_metric is only the "
                "nonlocal sij correction and is not sufficient for PAW shell "
                "charge/moment filtering."
            )
        shell_records = select_nc_pao_shells(
            compute_projector_shell_populations(data),
            charge_threshold=shell_charge_threshold,
            moment_threshold=shell_moment_threshold,
        )
        local_operators = mask_local_operators_by_shell_selection(
            data, local_operators, shell_records
        )
    description = (
        "Projector Green workflow using ABINIT savetb2j PAW projections "
        f"({data.coefficient_source}) and operator component "
        f"{operator_component or 'delta_total'} in basis {data.operator_basis}. "
        f"Shell charge threshold: {shell_charge_threshold}. "
        f"Shell moment threshold: {shell_moment_threshold}. "
        f"Population mode: {population_mode}; projector populations use the "
        "exported nonlocal PAW sij metric and are not PAW-complete AE charges. "
        "Values are from the controlled projector exchange-like trace. "
        f"ABINIT version: {data.metadata.get('abinit_version', 'unknown')}; "
        f"schema: {data.metadata.get('abinit_schema_name')} "
        f"{data.metadata.get('abinit_schema_version')}.\n"
    )
    charges = None
    spinat = None
    output_population_mode = population_mode
    if population_mode == "green":
        density = projector_charge_moments_from_green(
            ProjectorGreen(data),
            _population_contour(data, nz, smearing_eV),
            sites=sites,
        )
        charges = np.zeros(len(data.atomic_numbers), dtype=float)
        spinat = np.zeros((len(data.atomic_numbers), 3), dtype=float)
        charges[: len(density["charges"])] = density["charges"]
        spinat[: density["spinat"].shape[0]] = density["spinat"]
        output_population_mode = "none"
    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        population_mode=output_population_mode,
        charges=charges,
        spinat=spinat,
        Rcut=Rcut,
        local_operators=local_operators,
    )


def gen_exchange_abinit_nc_pao(
    filename,
    output_path="TB2J_results_abinit_nc_pao",
    Rmax=None,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    operator_component=ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT,
    population_mode="none",
    shell_charge_threshold=0.01,
    shell_moment_threshold=0.01,
    emax_eV=None,
    emax_relative_to_fermi_eV=None,
    n_empty=None,
    report_path=None,
    overlap_mode=None,
    overlap_rcond=None,
    dftu_file=None,
):
    """Generate projector exchange output from an ABINIT NC PAO savetb2j file."""
    data = load_abinit_nc_pao_savetb2j(filename)
    if dftu_file is not None:
        from TB2J.interfaces.abinit_nc_dftu import (
            embed_dftu_potential,
            read_abinit_dftu_nc,
        )

        dftu = read_abinit_dftu_nc(dftu_file)
        delta_u = embed_dftu_potential(dftu, data.projector_l, data.site_nproj)
        data.operator_components["delta_U"] = delta_u
        data.operator_component_metadata["delta_U"] = {
            "source": "abinit_nc_dftu_file",
            "units": "eV",
            "operator_basis": "abinit_nc_pao",
            "spin_treatment": "spin_difference",
            "completeness": "complete",
        }
    if data.metadata.get("kpoint_set") != "full_bz":
        raise ValueError(
            "ABINIT NC PAO exchange requires full-BZ spectral coefficients; "
            f"got kpoint_set={data.metadata.get('kpoint_set')!r}"
        )
    if population_mode not in {"none", "projector", "green"}:
        raise ValueError(f"unsupported population_mode: {population_mode}")
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    band_window_metadata = None
    if (
        emax_eV is not None
        or emax_relative_to_fermi_eV is not None
        or n_empty is not None
    ):
        data.band_mask, band_window_metadata = build_nc_pao_band_mask(
            data,
            emax_eV=emax_eV,
            emax_relative_to_fermi_eV=emax_relative_to_fermi_eV,
            n_empty=n_empty,
        )
    Rpts = _R_grid_for_cutoff(data, sites, Rcut, Rmax)
    if (
        operator_component == ABINIT_NC_PAO_DEFAULT_OPERATOR_COMPONENT
        and not data.has_operator_component(operator_component)
    ):
        operator_component = (
            "spectral_spin_split"
            if data.has_operator_component("spectral_spin_split")
            else "delta_total"
        )
    local_operators = component_local_operators(
        data, operator_component, sites, "ABINIT NC PAO savetb2j"
    )
    shell_records = None
    if shell_charge_threshold is not None or shell_moment_threshold is not None:
        if data.projector_l is not None and data.projector_radial is not None:
            shell_records = select_nc_pao_shells(
                compute_nc_pao_shell_populations(data),
                charge_threshold=shell_charge_threshold,
                moment_threshold=shell_moment_threshold,
            )
            local_operators = mask_local_operators_by_shell_selection(
                data, local_operators, shell_records
            )
        else:
            shell_records = []
    _validate_local_operators_hermitian(local_operators, operator_component)
    charges = None
    spinat = None
    output_population_mode = population_mode
    if population_mode == "projector":
        charges, spinat, _ = compute_nc_pao_projected_charges_moments(data)
        output_population_mode = "none"
    elif population_mode == "green":
        density = projector_charge_moments_from_green(
            ProjectorGreen(
                data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond
            ),
            _population_contour(data, nz, smearing_eV),
            sites=sites,
        )
        charges = np.zeros(len(data.atomic_numbers), dtype=float)
        spinat = np.zeros((len(data.atomic_numbers), 3), dtype=float)
        charges[: len(density["charges"])] = density["charges"]
        spinat[: density["spinat"].shape[0]] = density["spinat"]
    description = (
        "Projector Green workflow using ABINIT norm-conserving PAO projections "
        "with k-dependent overlap_k and operator component "
        f"{operator_component or 'delta_total'} in basis {data.operator_basis}. "
        f"Shell charge threshold: {shell_charge_threshold}. "
        f"Shell moment threshold: {shell_moment_threshold}. "
        f"Eigenvalue cutoff: {band_window_metadata}. "
        "Values are from the controlled projector exchange-like trace. "
        f"schema: {data.metadata.get('abinit_schema_name')} "
        f"{data.metadata.get('abinit_schema_version')}.\n"
    )
    result = write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        population_mode=output_population_mode,
        charges=charges,
        spinat=spinat,
        Rcut=Rcut,
        local_operators=local_operators,
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )
    if report_path is not None:
        if shell_records is None:
            if data.projector_l is not None and data.projector_radial is not None:
                shell_records = select_nc_pao_shells(
                    compute_nc_pao_shell_populations(data),
                    charge_threshold=0.0,
                    moment_threshold=None,
                )
            else:
                shell_records = []
        Path(report_path).write_text(
            _format_nc_pao_diagnostics_report(
                data,
                shell_records,
                band_window_metadata=band_window_metadata,
                shell_filter_enabled=(
                    shell_charge_threshold is not None
                    or shell_moment_threshold is not None
                ),
            ),
            encoding="utf-8",
        )
    return result
