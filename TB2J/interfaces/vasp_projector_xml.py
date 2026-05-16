"""VASP projector Green XML parser."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from ase.units import kB

from TB2J.interfaces.gpaw_projector import (
    _R_grid_for_cutoff,
    write_projector_exchange_out,
)
from TB2J.mycfr import CFR2
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    decode_complex,
    projector_charge_moments_from_green,
)

XML_SCHEMA_NAME = "tb2j.projector_green.xml"
XML_SCHEMA_VERSION = "0.1"


def _parse_dimensions(root):
    dimensions = {"three": 3, "complex": 2}
    group = root.find("dimensions")
    if group is None:
        return dimensions
    for dim in group.findall("dim"):
        name = dim.get("name")
        value = dim.get("value")
        if not name or value is None:
            raise ValueError("dimension entries require name and value")
        dimensions[name] = int(value)
    return dimensions


def _shape_from_attr(element, dimensions):
    shape = element.get("shape")
    dims = element.get("dims")
    if shape is not None:
        return tuple(int(item) for item in shape.split())
    if dims is None:
        raise ValueError(
            f"array {element.get('name', '<unnamed>')} requires shape or dims"
        )
    out = []
    for name in dims.split():
        if name not in dimensions:
            raise ValueError(f"array dimension {name!r} is not defined")
        out.append(dimensions[name])
    return tuple(out)


def _array(parent, name, dimensions, dtype=float, required=True):
    element = parent.find(f"array[@name='{name}']") if parent is not None else None
    if element is None:
        if required:
            raise ValueError(f"missing XML array: {name}")
        return None

    shape = _shape_from_attr(element, dimensions)
    text = " ".join(element.itertext()).strip()
    if not text:
        values = np.array([], dtype=float)
    else:
        try:
            values = np.array([float(token) for token in text.split()], dtype=float)
        except ValueError as exc:
            raise ValueError(f"array {name} contains non-numeric data") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"array {name} contains non-finite values")

    if element.get("dtype", "").startswith("complex") or element.get("complex"):
        if element.get("complex", "interleaved") != "interleaved":
            raise ValueError("only interleaved complex XML arrays are supported")
        expected = 2 * math.prod(shape)
        if values.size != expected:
            raise ValueError(
                f"array {name} has {values.size} values, expected {expected} "
                "for interleaved complex data"
            )
        return decode_complex(values.reshape(shape + (2,)))

    expected = math.prod(shape)
    if values.size != expected:
        raise ValueError(f"array {name} has {values.size} values, expected {expected}")
    if np.issubdtype(np.dtype(dtype), np.integer):
        if not np.all(np.equal(values, np.rint(values))):
            raise ValueError(f"integer array {name} contains non-integer values")
    return values.astype(dtype).reshape(shape)


def _metadata(root):
    metadata = dict(root.attrib)
    group = root.find("metadata")
    if group is not None:
        for item in group.findall("item"):
            name = item.get("name")
            if not name:
                raise ValueError("metadata item requires name")
            text = (item.text or "").strip()
            try:
                metadata[name] = json.loads(text)
            except json.JSONDecodeError:
                metadata[name] = text
    return metadata


def _optional_text_attr(element, name):
    value = element.get(name) if element is not None else None
    if value is None or not value.strip():
        return None
    return value


def _metadata_int(metadata, name):
    value = metadata.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_lprj_cdij_basis_mismatch(data):
    return (
        data.coefficient_source == "vasp.LPRJ_COVL"
        and data.operator_basis == "vasp_cdij_paw_hamiltonian"
    )


def _structure_positions(structure, dimensions, metadata):
    positions_element = (
        structure.find("array[@name='positions']") if structure is not None else None
    )
    positions = _array(structure, "positions", dimensions, required=False)
    if positions is None:
        return None
    coordinate_system = (positions_element.get("coordinate_system") or "").lower()
    if coordinate_system in {"cartesian", "cartesian_angstrom"}:
        return positions
    if coordinate_system in {"direct", "fractional", "scaled"}:
        cell = _array(structure, "cell", dimensions, required=True)
        return positions @ cell
    if metadata.get("source_code") == "vasp" and np.all(
        (positions >= -1e-12) & (positions <= 1.0 + 1e-12)
    ):
        cell = _array(structure, "cell", dimensions, required=True)
        return positions @ cell
    return positions


def _read_vasp_site_block(lines, start):
    values = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if values:
                break
            continue
        if stripped.startswith("-") or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "tot":
            break
        if parts[0].isdigit() and len(parts) >= 5:
            values.append(float(parts[-1]))
    return values


def load_vasp_outcar_charges_moments(filename, natom=None):
    """Load the final VASP LORBIT site charges and collinear moments."""
    lines = Path(filename).read_text(errors="replace").splitlines()
    charges = moments = None
    for index, line in enumerate(lines):
        if line.strip() == "total charge":
            charges = _read_vasp_site_block(lines, index + 1)
        elif line.strip().startswith("magnetization ("):
            moments = _read_vasp_site_block(lines, index + 1)

    if charges is None or moments is None:
        raise ValueError(
            "OUTCAR does not contain LORBIT site charge/magnetization blocks"
        )
    charges = np.asarray(charges, dtype=float)
    moments = np.asarray(moments, dtype=float)
    if natom is not None and (charges.size != natom or moments.size != natom):
        raise ValueError("OUTCAR site charge/moment count does not match XML atoms")
    spinat = np.zeros((moments.size, 3), dtype=float)
    spinat[:, 2] = moments
    return charges, spinat


def compare_green_to_vasp_outcar_populations(
    data,
    outcar_filename,
    nz=80,
    smearing_eV=0.05,
    charge_atol=0.1,
    moment_atol=0.05,
):
    """Compare Green-derived projector populations with VASP OUTCAR values."""
    if data.population_metric_matrix is not None:
        green_pop = vasp_projected_charges_moments(data)
    else:
        green = ProjectorGreen(data)
        contour = CFR2(nz=nz, T=smearing_eV / kB)
        green_pop = projector_charge_moments_from_green(green, contour)
    vasp_charges, vasp_spinat = load_vasp_outcar_charges_moments(
        outcar_filename, natom=len(data.atomic_numbers)
    )
    charge_diff = green_pop["charges"] - vasp_charges
    moment_diff = green_pop["spinat"][:, 2] - vasp_spinat[:, 2]
    matches = bool(
        np.all(np.abs(charge_diff) <= charge_atol)
        and np.all(np.abs(moment_diff) <= moment_atol)
    )
    return {
        "matches": matches,
        "green_charges": green_pop["charges"],
        "green_spinat": green_pop["spinat"],
        "vasp_charges": vasp_charges,
        "vasp_spinat": vasp_spinat,
        "charge_diff": charge_diff,
        "moment_diff": moment_diff,
        "charge_atol": charge_atol,
        "moment_atol": moment_atol,
        "method": green_pop["method"],
    }


def vasp_projected_charges_moments(data):
    """Compute VASP LORBIT-style site populations from exported QTOT metric."""
    if data.occupations is None:
        raise ValueError("VASP projected populations require occupations")
    if data.population_metric_matrix is None:
        raise ValueError("VASP projected populations require population_metric_matrix")
    if data.site_nproj is None or data.site_projector_indices is None:
        raise ValueError("VASP projected populations require site projector indices")

    nsite = len(data.site_nproj)
    density_by_spin = np.zeros((data.nspin, nsite), dtype=float)
    rspin = 2.0 if data.nspin == 1 else 1.0
    for site, nproj in enumerate(data.site_nproj):
        indices = data.site_projector_indices[site, :nproj]
        metric = data.population_metric_matrix[np.ix_(indices, indices)]
        coeff = data.coefficients[:, :, :, indices]
        for spin in range(data.nspin):
            density_by_spin[spin, site] = float(
                rspin
                * np.einsum(
                    "knp,pq,knq,k,kn->",
                    coeff[spin],
                    metric,
                    coeff[spin].conj(),
                    data.weights,
                    data.occupations[spin],
                    optimize="optimal",
                ).real
            )
    charges = np.sum(density_by_spin, axis=0)
    spinat = np.zeros((nsite, 3), dtype=float)
    if data.nspin >= 2:
        spinat[:, 2] = density_by_spin[0] - density_by_spin[1]
    return {
        "charges": charges,
        "spinat": spinat,
        "density_by_spin": density_by_spin,
        "method": "vasp_qtot_projected_population",
    }


def load_vasp_projector_xml(filename):
    """Load a TB2J VASP projector Green XML file."""
    root = ET.parse(Path(filename)).getroot()
    if root.tag != "tb2j_projector_green":
        raise ValueError("unsupported VASP projector XML root element")
    if root.get("schema_name") != XML_SCHEMA_NAME:
        raise ValueError("unsupported VASP projector XML schema")
    if root.get("schema_version") != XML_SCHEMA_VERSION:
        raise ValueError("unsupported VASP projector XML version")

    kgroup = root.find("kpoints")
    if kgroup is None:
        raise ValueError("VASP projector XML requires kpoints group")
    if kgroup.get("bz", "").lower() != "full":
        raise ValueError("VASP projector XML must contain full-BZ k-point data")

    dimensions = _parse_dimensions(root)
    metadata = _metadata(root)
    metadata.setdefault("source_code", "vasp")
    metadata["kpoint_set"] = "full_bz"

    structure = root.find("structure")
    bands = root.find("bands")
    projectors = root.find("projectors")
    if bands is None:
        raise ValueError("VASP projector XML requires bands group")
    if projectors is None:
        raise ValueError("VASP projector XML requires projectors group")

    overlap_metric = _array(projectors, "overlap_metric", dimensions, required=False)
    population_metric_matrix = _array(
        projectors, "population_metric_matrix", dimensions, required=False
    )
    operators = root.find("operators")
    hij = hij_definition = hij_units = hij_source = hij_projection = None
    operator_components = None
    operator_basis = metadata.get("operator_basis")
    if operators is not None:
        hij = _array(operators, "hij", dimensions, required=False)
        if hij is not None:
            hij_definition = operators.get("hij_definition")
            hij_units = operators.get("hij_units")
            hij_source = operators.get("hij_source")
            hij_projection = operators.get("hij_projection")
            operator_basis = operators.get("operator_basis", operator_basis)
        component_names = (
            "delta_total",
            "delta_xc",
            "delta_u",
            "delta_xc_smooth",
            "delta_u_paw_aug",
        )
        operator_components = {
            name: value
            for name in component_names
            if (value := _array(operators, name, dimensions, required=False))
            is not None
        }
        if not operator_components:
            operator_components = None
        elif hij_definition is None:
            hij_definition = operators.get(
                "delta_definition",
                "spin-splitting matrix in VASP LOCPROJ trial-function basis",
            )
            hij_units = operators.get("delta_units", "eV")
            hij_source = operators.get("delta_source")
            hij_projection = operators.get("delta_projection")
            operator_basis = operators.get("operator_basis", operator_basis)

    if bands.get("efermi") is None:
        raise ValueError("VASP projector XML bands group requires efermi")

    return ProjectorGreenData(
        kpoints=_array(kgroup, "kpoints", dimensions),
        weights=_array(kgroup, "weights", dimensions),
        eigenvalues=_array(bands, "eigenvalues", dimensions),
        occupations=_array(bands, "occupations", dimensions, required=False),
        coefficients=_array(projectors, "coefficients", dimensions),
        efermi=float(bands.get("efermi")),
        projector_site=_array(projectors, "projector_site", dimensions, dtype=int),
        projector_atom=_array(projectors, "projector_atom", dimensions, dtype=int),
        cell=_array(structure, "cell", dimensions, required=False),
        positions=_structure_positions(structure, dimensions, metadata),
        atomic_numbers=_array(
            structure, "atomic_numbers", dimensions, dtype=int, required=False
        ),
        projector_l=_array(
            projectors, "projector_l", dimensions, dtype=int, required=False
        ),
        projector_m=_array(
            projectors, "projector_m", dimensions, dtype=int, required=False
        ),
        projector_radial=_array(
            projectors, "projector_radial", dimensions, dtype=int, required=False
        ),
        overlap_metric=overlap_metric,
        population_metric_matrix=population_metric_matrix,
        site_nproj=_array(
            projectors, "site_nproj", dimensions, dtype=int, required=False
        ),
        site_projector_indices=_array(
            projectors, "site_projector_indices", dimensions, dtype=int, required=False
        ),
        hij=hij,
        hij_definition=hij_definition,
        hij_units=hij_units,
        hij_source=hij_source,
        hij_projection=hij_projection,
        operator_components=operator_components,
        coefficient_source=_optional_text_attr(projectors, "coefficient_source"),
        coefficient_projector=_optional_text_attr(projectors, "coefficient_projector"),
        channel_interpretation=_optional_text_attr(
            projectors, "channel_interpretation"
        ),
        overlap_metric_definition=_optional_text_attr(
            projectors, "overlap_metric_definition"
        ),
        population_metric=_optional_text_attr(projectors, "population_metric"),
        operator_basis=operator_basis,
        metadata=metadata,
    )


def gen_exchange_vasp_projector_xml(
    filename,
    output_path="TB2J_results_vasp_xml",
    Rmax=None,
    Rcut=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    outcar_filename=None,
    population_source="green",
    population_nz=80,
    population_charge_atol=0.1,
    population_moment_atol=0.05,
    allow_symmetry_expanded=False,
    allow_basis_mismatch=False,
):
    """Generate controlled projector exchange output from VASP XML."""
    data = load_vasp_projector_xml(filename)
    isym = _metadata_int(data.metadata, "isym")
    if isym is not None and isym > 0 and not allow_symmetry_expanded:
        raise ValueError(
            "VASP XML generated with ISYM>0 is not enabled for exchange generation by "
            "default because symmetry-expanded LPRJ_COVL coefficients have not been "
            "validated against Green-derived populations. Rerun VASP with ISYM=0, or "
            "set allow_symmetry_expanded=True only for diagnostic output."
        )
    if _has_lprj_cdij_basis_mismatch(data) and not allow_basis_mismatch:
        raise ValueError(
            "VASP XML pairs LPRJ_COVL coefficients with native CDIJ operators. "
            "That basis combination is diagnostic-only unless CDIJ is transformed "
            "into the LOCPROJ basis. Rerun VASP so the XML uses vasp.W_CPROJ, or "
            "set allow_basis_mismatch=True only for diagnostic output."
        )
    data.validate(exchange_ready=True)
    explicit_outcar = outcar_filename is not None
    if outcar_filename is None:
        candidate = Path(filename).with_name("OUTCAR")
        outcar_filename = candidate if candidate.exists() else None
    charges = spinat = None
    population_text = (
        "Atomic charges and magnetic moments are written as zero placeholders."
    )
    if population_source == "green":
        if data.population_metric_matrix is not None:
            green_pop = vasp_projected_charges_moments(data)
        else:
            green = ProjectorGreen(data)
            contour = CFR2(nz=population_nz, T=smearing_eV / kB)
            green_pop = projector_charge_moments_from_green(green, contour)
        charges = green_pop["charges"]
        spinat = green_pop["spinat"]
        population_text = (
            "Atomic charges and magnetic moments are computed from the projector "
            f"Green function ({green_pop['method']})."
        )
        if outcar_filename is not None:
            try:
                comparison = compare_green_to_vasp_outcar_populations(
                    data,
                    outcar_filename,
                    nz=population_nz,
                    smearing_eV=smearing_eV,
                    charge_atol=population_charge_atol,
                    moment_atol=population_moment_atol,
                )
            except ValueError as exc:
                if explicit_outcar:
                    raise
                population_text += (
                    f" Auto-detected OUTCAR population comparison skipped: {exc}."
                )
                comparison = None
            if comparison is None:
                pass
            elif not comparison["matches"]:
                raise ValueError(
                    "Green-derived VASP XML populations do not match OUTCAR within "
                    f"tolerances. charge_diff={comparison['charge_diff'].tolist()}, "
                    f"moment_diff={comparison['moment_diff'].tolist()}. "
                    "Use a complete LOCPROJ basis matching VASP LORBIT projections, "
                    "rerun VASP with ISYM=0 if this is a symmetry-expansion issue, "
                    "or set population_source='outcar' for reporting-only output."
                )
            else:
                population_text += (
                    " Values match VASP OUTCAR site projections within tolerance."
                )
    elif population_source == "outcar":
        if outcar_filename is None:
            raise ValueError("population_source='outcar' requires OUTCAR data")
        charges, spinat = load_vasp_outcar_charges_moments(
            outcar_filename, natom=len(data.atomic_numbers)
        )
        population_text = (
            f"Atomic charges and magnetic moments are copied from VASP OUTCAR "
            f"site projections ({Path(outcar_filename).name})."
        )
    elif population_source != "none":
        raise ValueError(f"unsupported population_source: {population_source}")
    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    Rpts = _R_grid_for_cutoff(
        data, sites or list(range(len(data.site_nproj))), Rcut, Rmax
    )
    description = (
        "Projector Green workflow using VASP XML projections "
        f"({data.coefficient_source}) and onsite operator basis "
        f"{data.operator_basis}. Values are from the controlled projector "
        "exchange-like trace, not yet a production PAW MFT benchmark. "
        f"{population_text}\n"
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
        population_mode="none",
        charges=charges,
        spinat=spinat,
        Rcut=Rcut,
    )
