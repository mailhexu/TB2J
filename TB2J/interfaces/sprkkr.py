from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Iterable, cast

import numpy as np
from ase import Atoms
from ase.units import Bohr

from TB2J.io_exchange import SpinIO
from TB2J.magnon.magnon3 import Magnon

MomentInput = float | Iterable[float] | dict[int, float | Iterable[float]]


class SprkkrParseError(ValueError):
    pass


@dataclass(frozen=True)
class SprkkrStructure:
    lattice_parameter_au: float
    ratios: tuple[float, float]
    lattice_parameters_au: tuple[float, float, float]
    primitive_vectors_lattice_units: np.ndarray
    positions_lattice_units: np.ndarray
    site_ids: list[int]
    site_classes: list[int]
    type_ids: list[int]
    site_type_labels: dict[int, str]
    site_type_z: dict[int, int]
    symbols: list[str]
    index_spin: list[int]
    atoms: Atoms
    source_file: str


@dataclass(frozen=True)
class SprkkrExchangeRow:
    type_i: int
    site_i: int
    type_j: int
    site_j: int
    n123: tuple[int, int, int]
    displacement_lattice_units: tuple[float, float, float]
    distance_lattice_units: float
    j_xx_mev: float
    j_yy_mev: float
    j_xy_mev: float
    j_yx_mev: float
    line_number: int


@dataclass(frozen=True)
class SprkkrExchangeTable:
    rows: list[SprkkrExchangeRow]
    source_file: str
    units: dict[str, str]

    def filter_by_sites(self, site_ids: Iterable[int]) -> "SprkkrExchangeTable":
        sites = set(site_ids)
        rows = [row for row in self.rows if row.site_i in sites and row.site_j in sites]
        return replace(self, rows=rows)


@dataclass(frozen=True)
class SprkkrExchangeData:
    structure: SprkkrStructure
    rows: list[SprkkrExchangeRow]
    source_files: dict[str, str]
    units: dict[str, str]
    convention: dict[str, str]
    moments_by_site: dict[int, tuple[float, float, float]]


def _read_lines(path: str | Path) -> list[str]:
    file_path = Path(path)
    try:
        return file_path.read_text(encoding="latin1").splitlines()
    except OSError as exc:
        raise SprkkrParseError(f"Could not read {file_path}: {exc}") from exc


def _find_line(lines: list[str], needle: str, path: Path) -> int:
    needle_lower = needle.lower()
    for index, line in enumerate(lines):
        if needle_lower in line.lower():
            return index
    raise SprkkrParseError(f"{path}: missing required section '{needle}'")


def _parse_float_line(
    lines: list[str], index: int, path: Path, section: str
) -> list[float]:
    try:
        return [float(token) for token in lines[index].split()]
    except (IndexError, ValueError) as exc:
        raise SprkkrParseError(f"{path}: invalid numeric data for {section}") from exc


def _normal_symbol(label: str) -> str:
    if label.lower().startswith("vc") or label == "0":
        return "X"
    return label.split("_")[0]


def read_sprkkr_structure(
    structure_file: str | Path,
    magnetic_species: Iterable[str] | None = None,
    magnetic_sites: Iterable[int] | None = None,
) -> SprkkrStructure:
    path = Path(structure_file)
    lines = _read_lines(path)

    lattice_index = _find_line(lines, "lattice parameter A", path)
    lattice_parameter_au = _parse_float_line(
        lines, lattice_index + 1, path, "lattice parameter A"
    )[0]

    ratio_index = _find_line(lines, "ratio of lattice parameters", path)
    ratios_values = _parse_float_line(lines, ratio_index + 1, path, "lattice ratios")
    ratios = (ratios_values[0], ratios_values[1])

    params_index = _find_line(lines, "lattice parameters  a b c", path)
    lattice_parameter_values = _parse_float_line(
        lines, params_index + 1, path, "lattice parameters"
    )
    lattice_parameters_au = (
        lattice_parameter_values[0],
        lattice_parameter_values[1],
        lattice_parameter_values[2],
    )

    primitive_index = _find_line(lines, "primitive vectors", path)
    primitive_vectors = np.array(
        [
            _parse_float_line(
                lines, primitive_index + offset, path, "primitive vectors"
            )
            for offset in (1, 2, 3)
        ],
        dtype=float,
    )

    nq_index = _find_line(lines, "number of sites NQ", path)
    try:
        n_sites = int(lines[nq_index + 1].split()[0])
    except (IndexError, ValueError) as exc:
        raise SprkkrParseError(f"{path}: invalid number of sites NQ") from exc

    site_start = nq_index + 3
    site_ids: list[int] = []
    site_classes: list[int] = []
    positions = []
    for offset in range(n_sites):
        line_number = site_start + offset + 1
        tokens = lines[site_start + offset].split()
        if len(tokens) < 9:
            raise SprkkrParseError(f"{path}:{line_number}: invalid site row")
        try:
            site_ids.append(int(tokens[0]))
            site_classes.append(int(tokens[1]))
            positions.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
        except ValueError as exc:
            raise SprkkrParseError(f"{path}:{line_number}: invalid site row") from exc

    nt_index = _find_line(lines, "number of atom types NT", path)
    try:
        n_types = int(lines[nt_index + 1].split()[0])
    except (IndexError, ValueError) as exc:
        raise SprkkrParseError(f"{path}: invalid number of atom types NT") from exc

    site_type_labels: dict[int, str] = {}
    site_type_z: dict[int, int] = {}
    type_ids_by_site: dict[int, int] = {}
    type_start = nt_index + 3
    for offset in range(n_types):
        line_number = type_start + offset + 1
        tokens = lines[type_start + offset].split()
        if len(tokens) < 6:
            raise SprkkrParseError(f"{path}:{line_number}: invalid atom type row")
        try:
            type_id = int(tokens[0])
            z_value = int(tokens[1])
            label = tokens[2]
            occupied_sites = [int(token) for token in tokens[5:]]
        except ValueError as exc:
            raise SprkkrParseError(
                f"{path}:{line_number}: invalid atom type row"
            ) from exc
        for site_id in occupied_sites:
            site_type_labels[site_id] = label
            site_type_z[site_id] = z_value
            type_ids_by_site[site_id] = type_id

    missing_types = [site_id for site_id in site_ids if site_id not in type_ids_by_site]
    if missing_types:
        raise SprkkrParseError(f"{path}: missing atom type for sites {missing_types}")

    magnetic_species_set = set(magnetic_species or [])
    explicit_magnetic_sites = set(magnetic_sites or [])
    index_spin: list[int] = []
    spin_index = 0
    symbols: list[str] = []
    type_ids: list[int] = []
    for site_id in site_ids:
        label = site_type_labels[site_id]
        symbol = _normal_symbol(label)
        symbols.append(symbol)
        type_ids.append(type_ids_by_site[site_id])
        is_magnetic = (
            site_id in explicit_magnetic_sites or symbol in magnetic_species_set
        )
        if is_magnetic:
            index_spin.append(spin_index)
            spin_index += 1
        else:
            index_spin.append(-1)

    cell_angstrom = primitive_vectors * lattice_parameter_au * Bohr
    positions_angstrom = np.array(positions, dtype=float) * lattice_parameter_au * Bohr
    atoms = Atoms(
        symbols=symbols, positions=positions_angstrom, cell=cell_angstrom, pbc=True
    )

    return SprkkrStructure(
        lattice_parameter_au=lattice_parameter_au,
        ratios=ratios,
        lattice_parameters_au=lattice_parameters_au,
        primitive_vectors_lattice_units=primitive_vectors,
        positions_lattice_units=np.array(positions, dtype=float),
        site_ids=site_ids,
        site_classes=site_classes,
        type_ids=type_ids,
        site_type_labels=site_type_labels,
        site_type_z=site_type_z,
        symbols=symbols,
        index_spin=index_spin,
        atoms=atoms,
        source_file=str(path),
    )


def _looks_integer(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def read_sprkkr_exchange_table(exchange_file: str | Path) -> SprkkrExchangeTable:
    path = Path(exchange_file)
    lines = _read_lines(path)
    rows: list[SprkkrExchangeRow] = []
    table_started = False
    for line_number, line in enumerate(lines, start=1):
        tokens = line.split()
        if (
            len(tokens) < 2
            or not _looks_integer(tokens[0])
            or not _looks_integer(tokens[1])
        ):
            if table_started and tokens:
                raise SprkkrParseError(
                    f"{path}:{line_number}: unexpected text after exchange table started"
                )
            continue
        table_started = True
        if len(tokens) != 15:
            raise SprkkrParseError(f"{path}:{line_number}: invalid exchange row shape")
        try:
            ints = [int(token) for token in tokens[:7]]
            floats = [float(token) for token in tokens[7:]]
        except ValueError as exc:
            raise SprkkrParseError(
                f"{path}:{line_number}: invalid exchange row"
            ) from exc
        rows.append(
            SprkkrExchangeRow(
                type_i=ints[0],
                site_i=ints[1],
                type_j=ints[2],
                site_j=ints[3],
                n123=(ints[4], ints[5], ints[6]),
                displacement_lattice_units=(floats[0], floats[1], floats[2]),
                distance_lattice_units=floats[3],
                j_xx_mev=floats[4],
                j_yy_mev=floats[5],
                j_xy_mev=floats[6],
                j_yx_mev=floats[7],
                line_number=line_number,
            )
        )
    if not rows:
        raise SprkkrParseError(f"{path}: no SPRKKR exchange rows found")
    return SprkkrExchangeTable(
        rows=rows,
        source_file=str(path),
        units={"exchange": "meV", "displacement": "lattice_parameter"},
    )


def _moments_by_site(
    structure: SprkkrStructure,
    moment: MomentInput | None,
) -> dict[int, tuple[float, float, float]]:
    magnetic_sites = [
        site_id
        for site_id, spin_index in zip(structure.site_ids, structure.index_spin)
        if spin_index >= 0
    ]
    if moment is None:
        raise ValueError("moment is required for SPRKKR magnon inputs")
    if isinstance(moment, dict):
        missing = [site_id for site_id in magnetic_sites if site_id not in moment]
        if missing:
            raise ValueError(f"Missing moment for SPRKKR magnetic sites {missing}")
        return {site_id: _moment_vector(moment[site_id]) for site_id in magnetic_sites}

    if isinstance(moment, Real):
        vector = _z_moment(float(moment))
        return {site_id: vector for site_id in magnetic_sites}

    values = [float(value) for value in cast(Iterable[float], moment)]
    nsites = len(magnetic_sites)
    if len(values) == 1:
        vector = _z_moment(values[0])
        return {site_id: vector for site_id in magnetic_sites}
    if len(values) == nsites:
        return {
            site_id: _z_moment(value) for site_id, value in zip(magnetic_sites, values)
        }
    if len(values) == 3 * nsites:
        return {
            site_id: (values[index], values[index + 1], values[index + 2])
            for site_id, index in zip(magnetic_sites, range(0, len(values), 3))
        }
    raise ValueError(
        "moment must contain 1, N, or 3N values for the selected SPRKKR "
        f"magnetic sites; got {len(values)} values for N={nsites}"
    )


def _z_moment(value: float) -> tuple[float, float, float]:
    return (0.0, 0.0, value)


def _moment_vector(moment: float | Iterable[float]) -> tuple[float, float, float]:
    if isinstance(moment, Real):
        return _z_moment(float(moment))
    values = tuple(float(value) for value in cast(Iterable[float], moment))
    if len(values) != 3:
        raise ValueError("moment vectors must contain exactly 3 values")
    return values


def read_sprkkr_exchange(
    structure_file: str | Path,
    exchange_file: str | Path,
    magnetic_species: Iterable[str] | None = None,
    magnetic_sites: Iterable[int] | None = None,
    moment: MomentInput | None = None,
    magnetic_only: bool = True,
) -> SprkkrExchangeData:
    structure = read_sprkkr_structure(
        structure_file,
        magnetic_species=magnetic_species,
        magnetic_sites=magnetic_sites,
    )
    table = read_sprkkr_exchange_table(exchange_file)
    magnetic_site_ids = {
        site_id
        for site_id, spin_index in zip(structure.site_ids, structure.index_spin)
        if spin_index >= 0
    }
    rows = (
        table.filter_by_sites(magnetic_site_ids).rows if magnetic_only else table.rows
    )
    if magnetic_only and not rows:
        raise ValueError("No exchange rows connect selected SPRKKR magnetic sites")
    return SprkkrExchangeData(
        structure=structure,
        rows=rows,
        source_files={"structure": str(structure_file), "exchange": str(exchange_file)},
        units={
            "exchange_source": "meV",
            "exchange_spinio": "eV",
            "length_source": "atomic_unit",
            "length_atoms": "Angstrom",
            "displacement_source": "lattice_parameter",
        },
        convention={
            "exchange": "LKAG, c=1, unit spin directions",
            "tensor_policy": "transverse block from J_xx, J_yy, J_xy, J_yx",
        },
        moments_by_site=_moments_by_site(structure, moment),
    )


def _site_to_spin_map(structure: SprkkrStructure) -> dict[int, int]:
    return {
        site_id: spin_index
        for site_id, spin_index in zip(structure.site_ids, structure.index_spin)
        if spin_index >= 0
    }


def _row_tensor_ev(row: SprkkrExchangeRow, include_jzz: bool = False) -> np.ndarray:
    tensor = np.array(
        [
            [row.j_xx_mev, row.j_xy_mev, 0.0],
            [row.j_yx_mev, row.j_yy_mev, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    if include_jzz:
        tensor[2, 2] = 0.5 * (row.j_xx_mev + row.j_yy_mev)
    return tensor * 0.001


def sprkkr_to_spinio(
    data: SprkkrExchangeData,
    tensor_policy: str = "transverse-block",
) -> SpinIO:
    tensor_policies = {"transverse-block", "transverse-block-jzz", "isotropic"}
    if tensor_policy not in tensor_policies:
        raise ValueError(
            "tensor_policy must be 'transverse-block', 'transverse-block-jzz', "
            "or 'isotropic'; "
            f"got {tensor_policy!r}"
        )
    site_to_spin = _site_to_spin_map(data.structure)
    natom = len(data.structure.site_ids)
    spinat = np.zeros((natom, 3), dtype=float)
    for atom_index, site_id in enumerate(data.structure.site_ids):
        if data.structure.index_spin[atom_index] >= 0:
            spinat[atom_index] = data.moments_by_site[site_id]

    exchange_Jdict: dict[tuple[tuple[int, int, int], int, int], float] = {}
    Jani_dict: dict[tuple[tuple[int, int, int], int, int], np.ndarray] | None = (
        {} if tensor_policy in {"transverse-block", "transverse-block-jzz"} else None
    )
    distance_dict: dict[
        tuple[tuple[int, int, int], int, int], tuple[np.ndarray, float]
    ] = {}
    length_factor = data.structure.lattice_parameter_au * Bohr

    for row in data.rows:
        if row.site_i not in site_to_spin or row.site_j not in site_to_spin:
            continue
        key = (row.n123, site_to_spin[row.site_i], site_to_spin[row.site_j])
        if tensor_policy in {"transverse-block", "transverse-block-jzz"}:
            tensor_ev = _row_tensor_ev(
                row,
                include_jzz=tensor_policy == "transverse-block-jzz",
            )
            exchange_Jdict[key] = 0.0
            assert Jani_dict is not None
            Jani_dict[key] = tensor_ev
        else:
            exchange_Jdict[key] = float((row.j_xx_mev + row.j_yy_mev) * 0.0005)
        distance_dict[key] = (
            np.array(row.displacement_lattice_units, dtype=float) * length_factor,
            row.distance_lattice_units * length_factor,
        )

    if not exchange_Jdict:
        raise ValueError("No SPRKKR exchange rows could be mapped to magnetic spins")

    spinio = SpinIO(
        atoms=data.structure.atoms.copy(),
        spinat=spinat,
        charges=[
            data.structure.site_type_z[site_id] for site_id in data.structure.site_ids
        ],
        index_spin=list(data.structure.index_spin),
        colinear=True,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_Jdict,
        Jani_dict=Jani_dict,
        dmi_ddict=None,
        description=(
            "Exchange parameters converted from SPRKKR reference-format files.\n"
            f"Structure: {data.source_files['structure']}\n"
            f"Exchange: {data.source_files['exchange']}\n"
            f"Tensor policy: {tensor_policy}\n"
        ),
    )
    setattr(
        spinio,
        "sprkkr_metadata",
        {
            "source_files": data.source_files,
            "units": data.units,
            "convention": data.convention,
            "moments_by_site": data.moments_by_site,
            "tensor_policy": tensor_policy,
        },
    )
    spinio._build_Rlist()
    return spinio


def write_sprkkr_tb2j_results(
    structure_file: str | Path,
    exchange_file: str | Path,
    output_path: str | Path = "TB2J_results",
    magnetic_species: Iterable[str] | None = None,
    magnetic_sites: Iterable[int] | None = None,
    moment: MomentInput | None = None,
    tensor_policy: str = "transverse-block",
) -> SpinIO:
    data = read_sprkkr_exchange(
        structure_file=structure_file,
        exchange_file=exchange_file,
        magnetic_species=magnetic_species,
        magnetic_sites=magnetic_sites,
        moment=moment,
    )
    spinio = sprkkr_to_spinio(data, tensor_policy=tensor_policy)
    spinio.write_all(path=str(output_path))
    return spinio


def magnon_from_sprkkr(
    structure_file: str | Path,
    exchange_file: str | Path,
    magnetic_species: Iterable[str] | None = None,
    magnetic_sites: Iterable[int] | None = None,
    moment: MomentInput | None = None,
    tensor_policy: str = "transverse-block",
    **kwargs,
) -> Magnon:
    data = read_sprkkr_exchange(
        structure_file=structure_file,
        exchange_file=exchange_file,
        magnetic_species=magnetic_species,
        magnetic_sites=magnetic_sites,
        moment=moment,
    )
    spinio = sprkkr_to_spinio(data, tensor_policy=tensor_policy)
    load_kwargs = {"Jiso": True, "DMI": False, "SIA": False}
    load_kwargs["Jani"] = tensor_policy in {"transverse-block", "transverse-block-jzz"}
    load_kwargs.update(kwargs)
    magnon = Magnon.load_from_io(spinio, **load_kwargs)
    magnon.set_reference(
        Q=np.zeros(3),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([0.0, 0.0, 1.0]),
        magmoms=magnon.magmom,
    )
    return magnon
