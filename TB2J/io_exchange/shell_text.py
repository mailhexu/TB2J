"""Compact TOML v4 boundary: one representative and value per shell."""

import json
import math
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tomli
import tomli_w

import TB2J

from .shells import (
    DISTANCE_TO_ANGSTROM,
    ShellDocument,
    ShellEntry,
    ShellOperation,
    _expand_core,
    _freeze,
    _pair_from_id,
    _pair_id,
    _physical_pairs,
    _shell_orbits,
    _template_distance,
    _template_number,
    _template_operations,
    _thaw,
    _validated_operations,
    _validate_context,
)

_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_LABEL = re.compile(r"([A-Z][a-z]?)[1-9][0-9]*\Z")
_SCHEMA_VERSION = 4


def _floats(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _floats(item) for key, item in value.items()}
    return value


def _energy(value, scale, field):
    if isinstance(value, (list, tuple)):
        return tuple(_energy(item, scale, field) for item in value)
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, bool):
        raise ValueError(f"{field}: expected a real exchange value, not a boolean")
    try:
        return float(Decimal(str(value)) / scale)
    except InvalidOperation as error:
        raise ValueError(f"{field}: expected a real exchange value") from error


def _wire_energy(value, scale):
    if isinstance(value, (list, tuple)):
        return tuple(_wire_energy(item, scale) for item in value)
    if isinstance(value, str) or value is None:
        return value
    # Decimal scaling preserves the float's shortest round-trip decimal form.
    return Decimal(str(value)) * scale


def _keys(table, required, optional, field):
    if not isinstance(table, dict):
        raise ValueError(f"{field}: expected a TOML table")
    unknown = set(table) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"{field}: unknown fields {', '.join(sorted(unknown))}")
    missing = set(required) - set(table)
    if missing:
        raise ValueError(
            f"{field}: missing required fields {', '.join(sorted(missing))}"
        )


def _schema(data):
    _keys(
        data,
        ("schema_version", "exchange", "structure", "symmetry"),
        (
            "energy_unit",
            "distance_unit",
            "cutoff",
            "metadata",
            "reference_spins",
            "source",
            "tolerances",
            "shell",
        ),
        "shell document",
    )
    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(
            f"schema_version: expected the integer {_SCHEMA_VERSION}, got {version!r}"
        )
    if version != _SCHEMA_VERSION:
        raise ValueError(f"schema_version: unsupported version {version!r}")
    if data["exchange"] not in {"isotropic", "tensor"}:
        raise ValueError("exchange: expected isotropic or tensor")
    if data.get("energy_unit", "meV") not in {"meV", "eV"}:
        raise ValueError("energy_unit: expected meV or eV")
    if (
        not isinstance(data.get("distance_unit", "Angstrom"), str)
        or data.get("distance_unit", "Angstrom") not in DISTANCE_TO_ANGSTROM
    ):
        raise ValueError("distance_unit: expected Angstrom, Bohr, or nm")
    _keys(data["structure"], ("coordinates", "lattice", "atoms"), (), "structure")
    if data["structure"]["coordinates"] != "fractional":
        raise ValueError("structure.coordinates: expected fractional")
    _keys(
        data["symmetry"],
        ("mode",),
        ("symprec", "magnetic_tol", "producer", "operations"),
        "symmetry",
    )
    if data["symmetry"]["mode"] not in {"spacegroup", "magnetic"}:
        raise ValueError("symmetry.mode: expected spacegroup or magnetic")
    if "tolerances" in data:
        _keys(
            data["tolerances"],
            (),
            ("value_atol", "value_rtol", "zero_tol"),
            "tolerances",
        )
    if "reference_spins" in data:
        _keys(data["reference_spins"], ("moments",), ("source",), "reference_spins")
    if "source" in data:
        _keys(
            data["source"],
            (),
            ("format", "precision_profile", "required_base_fields", "context_sha256"),
            "source",
        )
    if (
        data.get("source", {}).get("format") == "pickle"
        and "operations" not in data["symmetry"]
    ):
        raise ValueError(
            "native shell documents require their explicit symmetry.operations snapshot"
        )
    if "operations" in data["symmetry"]:
        operations = data["symmetry"]["operations"]
        if not isinstance(operations, list):
            raise ValueError(
                "symmetry.operations: expected an array of operation tables"
            )
        for index, operation in enumerate(operations):
            _keys(
                operation,
                ("rotation", "translation", "time_reversal"),
                (),
                f"symmetry.operations[{index}]",
            )
    entries = data.get("shell", [])
    if not isinstance(entries, list):
        raise ValueError("shell: expected an array of shell tables")
    for index, entry in enumerate(entries):
        _keys(
            entry,
            ("representative", "J"),
            ("name", "distance", "n_equivalent_pairs"),
            f"shell[{index}]",
        )


def _rows(block, field):
    if not isinstance(block, str):
        raise ValueError(f"{field}: expected a multiline row string")
    rows = [line.split("#", 1)[0].split() for line in block.splitlines()]
    return [row for row in rows if row]


def _real(token, field):
    if not _REAL.fullmatch(token):
        raise ValueError(f"{field}: expected a finite decimal number, got {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"{field}: expected a finite decimal number")
    return value


def _integer(token, field):
    if not _INTEGER.fullmatch(token):
        raise ValueError(f"{field}: expected an integer, got {token!r}")
    return int(token)


def _structure(table):
    from ase.data import atomic_numbers

    lattice = _rows(table["lattice"], "structure.lattice")
    if len(lattice) != 3 or any(len(row) != 3 for row in lattice):
        raise ValueError("structure.lattice: expected exactly three 3-number rows")
    cell = tuple(
        tuple(_real(token, f"structure.lattice row {i}") for token in row)
        for i, row in enumerate(lattice, 1)
    )
    if abs(np.linalg.det(cell)) < np.finfo(float).tiny:
        raise ValueError("structure.lattice: singular cell")
    species, labels, positions, selected, indices = [], [], [], [], []
    next_spin = 0
    for i, row in enumerate(_rows(table["atoms"], "structure.atoms"), 1):
        field = f"structure.atoms row {i}"
        if len(row) != 6:
            raise ValueError(f"{field}: expected atom-index label x y z magnetic")
        if _integer(row[0], field) != i:
            raise ValueError(
                f"{field}: atom indices must be sequential one-based source order"
            )
        label = _LABEL.fullmatch(row[1])
        if label is None or label[1] not in atomic_numbers or row[1] in labels:
            raise ValueError(
                f"{field}: invalid or duplicate element-number label {row[1]!r}"
            )
        if row[5] not in {"Y", "N"}:
            raise ValueError(f"{field}: magnetic flag must be Y or N")
        species.append(label[1])
        labels.append(row[1])
        positions.append(tuple(_real(token, field) for token in row[2:5]))
        if row[5] == "Y":
            indices.append(next_spin)
            selected.append(i)
            next_spin += 1
        else:
            indices.append(-1)
    return dict(
        cell_angstrom=cell,
        species=tuple(species),
        site_labels=tuple(labels),
        positions_frac=tuple(positions),
        magnetic_atoms=tuple(selected),
        index_spin=tuple(indices),
    )


def _moments(table, labels):
    if table is None:
        return None
    values = {}
    width = None
    for index, row in enumerate(_rows(table["moments"], "reference_spins.moments"), 1):
        field = f"reference_spins.moments row {index}"
        if len(row) not in (2, 4) or (width is not None and len(row) != width):
            raise ValueError(f"{field}: use uniformly label m or label mx my mz")
        if row[0] not in labels or row[0] in values:
            raise ValueError(f"{field}: unknown or duplicate label {row[0]!r}")
        width = len(row)
        numbers = tuple(_real(token, field) for token in row[1:])
        values[row[0]] = numbers[0] if width == 2 else numbers
    missing = set(labels) - set(values)
    if missing:
        raise ValueError(
            f"reference_spins.moments: missing labels {', '.join(sorted(missing))}"
        )
    return tuple(values[label] for label in labels)


def _representative(value, labels, field):
    if not isinstance(value, str) or len(value.split()) != 5:
        raise ValueError(f"{field}: expected 'label label Rx Ry Rz'")
    first, second, *translation = value.split()
    if first not in labels or second not in labels:
        raise ValueError(f"{field}: unknown site label")
    return (
        labels.index(first) + 1,
        labels.index(second) + 1,
        *(_integer(token, field) for token in translation),
    )


def load(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def loads(text):
    try:
        data = tomli.loads(text, parse_float=Decimal)
    except tomli.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML: {error}") from error
    _schema(data)
    structure = _structure(data["structure"])
    symmetry = _floats(data["symmetry"])
    reference = data.get("reference_spins")
    source = data.get("source", {})
    distance_unit = data.get("distance_unit", "Angstrom")
    distance_scale = DISTANCE_TO_ANGSTROM[distance_unit]
    mode = {"spacegroup": "sg", "magnetic": "msg"}[symmetry["mode"]]
    kwargs = dict(
        **structure,
        exchange_mode=data["exchange"],
        schema_version=_SCHEMA_VERSION,
        energy_unit=data.get("energy_unit", "meV"),
        distance_unit=distance_unit,
        symmetry_mode=mode,
        symprec=symmetry.get("symprec", 1e-5),
        magnetic_tol=symmetry.get("magnetic_tol", 1e-5 if mode == "msg" else None),
        producer=symmetry.get("producer"),
        moments=_moments(reference, structure["site_labels"]),
        moment_source=reference.get("source", "explicit") if reference else "explicit",
        metadata=_floats(data.get("metadata", {})),
        cutoff=(
            _template_number(_floats(data["cutoff"]), "cutoff", positive=True)
            * distance_scale
            if "cutoff" in data
            else None
        ),
        source_format=source.get("format"),
        precision_profile=source.get("precision_profile"),
        context_sha256=source.get("context_sha256"),
        required_base_fields=tuple(source.get("required_base_fields", ())),
        **{key: _floats(value) for key, value in data.get("tolerances", {}).items()},
    )
    # Resolve and validate complete context before detection or orbit replay.
    context = SimpleNamespace(value_atol=1e-8, value_rtol=1e-6, zero_tol=1e-10)
    context.__dict__.update({key: _freeze(value) for key, value in kwargs.items()})
    _validate_context(context)
    if (
        isinstance(context.symprec, bool)
        or not isinstance(context.symprec, (int, float))
        or not math.isfinite(context.symprec)
        or context.symprec <= 0
    ):
        raise ValueError("symmetry.symprec: expected a positive finite tolerance")
    if "operations" in symmetry:
        operations = tuple(ShellOperation(**entry) for entry in symmetry["operations"])
    else:
        from ase import Atoms

        atoms = Atoms(
            structure["species"],
            cell=structure["cell_angstrom"],
            scaled_positions=structure["positions_frac"],
            pbc=True,
        )
        detected, _, description = _template_operations(
            atoms,
            np.asarray(structure["positions_frac"]),
            np.asarray(structure["cell_angstrom"]),
            structure["magnetic_atoms"],
            mode,
            kwargs["moments"],
            kwargs["symprec"],
            kwargs["magnetic_tol"],
        )
        operations = tuple(
            ShellOperation(rotation, translation, bool(prime))
            for rotation, translation, prime in zip(
                detected.rotations, detected.translations, detected.time_reversals
            )
        )
        if kwargs["producer"] is None:
            kwargs["producer"] = description
    scale = Decimal(1000) if kwargs["energy_unit"] == "meV" else Decimal(1)
    entries = []
    for index, entry in enumerate(data.get("shell", [])):
        representative = _representative(
            entry["representative"],
            structure["site_labels"],
            f"shell[{index}].representative",
        )
        value = _energy(entry["J"], scale, f"shell[{index}].J")
        extras = {
            key: _floats(entry[key])
            for key in ("name", "distance", "n_equivalent_pairs")
            if key in entry
        }
        if "distance" in extras:
            extras["distance"] = (
                _template_number(extras["distance"], "distance", nonnegative=True)
                * distance_scale
            )
        entries.append(
            ShellEntry(
                id=_pair_id(representative),
                representative=representative,
                **{("Jiso" if data["exchange"] == "isotropic" else "tensor"): value},
                **extras,
            )
        )
    context = SimpleNamespace(**kwargs, operations=operations, shells=tuple(entries))
    orbits = _shell_orbits(context)
    entries = [
        replace(entry, id=_pair_id(min(orbit))) for entry, orbit in zip(entries, orbits)
    ]
    return ShellDocument(**kwargs, operations=operations, shells=tuple(entries))


def _pair_text(pair, labels):
    return f"{labels[pair[0] - 1]} {labels[pair[1] - 1]} " + " ".join(
        str(x) for x in pair[2:]
    )


def _number_text(value):
    return f"{value:15.8f}"


class _FixedDecimal(Decimal):
    """Preserve fixed decimal formatting when tomli_w serializes numbers."""

    def __str__(self):
        return format(self, ".8f")


class _FixedTolerance(Decimal):
    """Write a tolerance exactly, with at least eight fixed decimal places."""

    def __str__(self):
        whole, _, fraction = format(self, "f").partition(".")
        return whole + "." + fraction.ljust(8, "0")


def _rounded(value):
    if isinstance(value, _FixedTolerance):
        return value
    if isinstance(value, (float, Decimal)):
        return _FixedDecimal(str(value))
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(item) for item in value]
    return value


def _shell_entry_text(entry):
    parts = []
    for key, value in entry.items():
        if key == "J" and isinstance(value, (list, tuple)):
            rows = "".join(
                "    [" + ", ".join(_number_text(item) for item in row) + "],\n"
                for row in value
            )
            parts.append(f"J = [\n{rows}]\n")
        else:
            parts.append(tomli_w.dumps({key: _rounded(value)}))
    return "".join(parts)


def dump(document, path, *, equivalent_comments=True, source_path=None):
    atom_width = len(str(len(document.site_labels)))
    label_width = max(map(len, document.site_labels), default=0)
    structure = {
        "coordinates": "fractional",
        "lattice": "\n".join(
            " ".join(_number_text(value) for value in row)
            for row in document.cell_angstrom
        )
        + "\n",
        "atoms": "# atom-index (1-based)  element-label  fractional-x  fractional-y  fractional-z  magnetic\n"
        "# magnetic: Y = magnetic, N = nonmagnetic.\n"
        + "\n".join(
            f"{atom:>{atom_width}} {label:<{label_width}} "
            + " ".join(_number_text(value) for value in position)
            + f" {'Y' if spin >= 0 else 'N'}"
            for atom, (label, position, spin) in enumerate(
                zip(document.site_labels, document.positions_frac, document.index_spin),
                1,
            )
        )
        + "\n",
    }
    header = {
        "schema_version": _SCHEMA_VERSION,
        "exchange": document.exchange_mode,
        "energy_unit": document.energy_unit,
        "distance_unit": document.distance_unit,
    }
    if document.cutoff is not None:
        header["cutoff"] = (
            document.cutoff / DISTANCE_TO_ANGSTROM[document.distance_unit]
        )
    header["structure"] = structure
    if document.moments is not None:
        header["reference_spins"] = {
            "moments": "\n".join(
                f"{label:<{label_width}} "
                + " ".join(
                    _number_text(value)
                    for value in (moment if isinstance(moment, tuple) else (moment,))
                )
                for label, moment in zip(document.site_labels, document.moments)
            )
            + "\n",
            "source": document.moment_source,
        }
    symmetry = {
        "mode": {"sg": "spacegroup", "msg": "magnetic"}[document.symmetry_mode],
        "symprec": _FixedTolerance(str(document.symprec)),
    }
    if document.magnetic_tol is not None:
        symmetry["magnetic_tol"] = _FixedTolerance(str(document.magnetic_tol))
    if document.producer is not None:
        symmetry["producer"] = document.producer
    header["symmetry"] = symmetry
    if document.metadata:
        header["metadata"] = _thaw(document.metadata)
    provenance = f"# Generated by TB2J {TB2J.__version__}\n"
    if source_path is not None:
        provenance += f"# Source: {json.dumps(str(source_path), ensure_ascii=True)}\n"
    parts = [
        provenance
        + "# Numerical data are rounded to eight decimal places; small values may become zero.\n"
        "# Validation tolerances retain additional decimal places as needed.\n"
        "# Exchange values use energy_unit; cutoff and distance use distance_unit.\n"
        "# Lattice rows are Cartesian vectors in Angstrom; atom positions are fractional.\n"
        "# representative: first label, second label, Rx Ry Rz; translation moves the second atom.\n"
        "# Each representative is any directed bond in its full symmetry-and-reversal orbit.\n"
        "# Values use its Cartesian frame; reversal flips DMI and transposes tensors.\n"
        "# J is a scalar or Cartesian 3x3 tensor; replace 'unfilled'.\n"
        "# name, distance and n_equivalent_pairs are optional input summaries.\n"
        "# n_equivalent_pairs counts unique physical pairs, with each bond and reverse once.\n"
        "# Native sparse key/channel support, including present zeros, is retained by the companion.\n"
        + tomli_w.dumps(_rounded(header), multiline_strings=True)
    ]
    pair_sections = []
    validated = None
    if equivalent_comments and document.shells:
        validated = _validated_operations(
            document,
            np.asarray(document.cell_angstrom),
            np.asarray(document.positions_frac),
        )
    scale = Decimal(1000) if document.energy_unit == "meV" else Decimal(1)
    for number, (shell, orbit) in enumerate(
        zip(document.shells, _shell_orbits(document)), 1
    ):
        representative = shell.representative or _pair_from_id(shell.id)
        entry = {
            "representative": _pair_text(representative, document.site_labels),
        }
        if shell.name is not None:
            entry["name"] = shell.name
        entry["distance"] = (
            _template_distance(
                representative,
                np.asarray(document.positions_frac),
                np.asarray(document.cell_angstrom),
            )
            / DISTANCE_TO_ANGSTROM[document.distance_unit]
        )
        entry["n_equivalent_pairs"] = len(_physical_pairs(orbit))
        entry["J"] = _wire_energy(
            shell.Jiso if document.exchange_mode == "isotropic" else shell.tensor, scale
        )
        parts.append(f"# Shell {number}\n[[shell]]\n" + _shell_entry_text(entry))
        if equivalent_comments:
            from .shell_witness import pair_comment_sections

            pair_sections.append(f"#\n# Shell {number}\n")
            pair_sections.append(
                pair_comment_sections(
                    document, shell, orbit, representative, validated=validated
                )
            )
    if pair_sections:
        parts.append(
            "# Stored symmetry witnesses for each generated representative:\n"
            "# operation is the 1-based number of a stored symmetry.operations entry.\n"
            "# reversed=true means that operation's image is reversed to form the pair.\n"
            "# time_reversal belongs to that stored operation.\n"
            + "".join(pair_sections)
        )
    operation_sections = []
    for number, operation in enumerate(document.operations, 1):
        rows = "".join(
            "    [" + ", ".join(str(value) for value in row) + "],\n"
            for row in operation.rotation
        )
        translation = ", ".join(_number_text(value) for value in operation.translation)
        prime = "true" if operation.time_reversal else "false"
        operation_sections.append(
            f"# Operation {number}\n[[symmetry.operations]]\nrotation = [\n{rows}]\n"
            f"translation = [{translation}]\ntime_reversal = {prime}\n"
        )
    footer = {}
    footer["tolerances"] = {
        name: _FixedTolerance(str(getattr(document, name)))
        for name in ("value_atol", "value_rtol", "zero_tol")
    }
    source = {
        key: value
        for key, value in (
            ("format", document.source_format),
            ("precision_profile", document.precision_profile),
            ("context_sha256", document.context_sha256),
            ("required_base_fields", document.required_base_fields),
        )
        if value
    }
    if source:
        footer["source"] = source
    parts.append(
        "# Generated replay snapshot and tolerances\n" + "\n".join(operation_sections)
    )
    parts.append(tomli_w.dumps(_rounded(footer)))
    text = "\n".join(parts)
    try:
        parsed = loads(text)
        source_shells = {shell.id: shell for shell in document.shells}
        parsed = replace(
            parsed,
            shells=tuple(
                replace(
                    shell,
                    **{
                        name: getattr(source_shells[shell.id], name)
                        for name in (
                            "support_members",
                            "Jiso_members",
                            "DMI_members",
                            "Jani_members",
                        )
                    },
                )
                for shell in parsed.shells
            ),
        )
        filled = tuple(
            shell
            for shell in parsed.shells
            if (shell.Jiso if parsed.exchange_mode == "isotropic" else shell.tensor)
            != "unfilled"
        )
        if filled:
            _expand_core(replace(parsed, shells=filled))
    except ValueError as error:
        raise ValueError(
            f"serialized shell validation failed at eight-decimal-place precision: {error}"
        ) from error
    Path(path).write_text(text, encoding="utf-8")
