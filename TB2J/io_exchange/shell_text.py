"""TOML v2 boundary: editable labeled rows, authoritative replay footer."""

import math
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tomli
import tomli_w

from .shells import (
    ShellDocument,
    ShellEntry,
    ShellOperation,
    _freeze,
    _pair_id,
    _shell_orbits,
    _template_distance,
    _template_operations,
    _thaw,
    _validate_context,
)

_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_LABEL = re.compile(r"([A-Z][a-z]?)[1-9][0-9]*\Z")
_MASKS = ("support_members", "Jiso_members", "DMI_members", "Jani_members")


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
        ("exchange", "structure", "symmetry"),
        (
            "schema_version",
            "energy_unit",
            "cutoff",
            "metadata",
            "reference_spins",
            "source",
            "tolerances",
            "shell",
        ),
        "shell document",
    )
    version = data.get("schema_version", 2)
    if not isinstance(version, int) or isinstance(version, bool) or version != 2:
        raise ValueError(f"schema_version: unsupported version {version!r}")
    if data["exchange"] not in {"isotropic", "tensor"}:
        raise ValueError("exchange: expected isotropic or tensor")
    if data.get("energy_unit", "meV") not in {"meV", "eV"}:
        raise ValueError("energy_unit: expected meV or eV")
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
            ("J",),
            ("id", "representative", "name", "distance", "n_equivalent_pairs", *_MASKS),
            f"shell[{index}]",
        )
        if "representative" not in entry and (
            data["exchange"] == "tensor" or "id" not in entry
        ):
            raise ValueError(
                f"shell[{index}].representative: required without isotropic id"
            )
        if "operations" not in data["symmetry"] and any(key in entry for key in _MASKS):
            raise ValueError(
                f"shell[{index}]: ordinal masks require an explicit symmetry.operations snapshot"
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
    for i, row in enumerate(_rows(table["atoms"], "structure.atoms"), 1):
        field = f"structure.atoms row {i}"
        if len(row) != 7:
            raise ValueError(
                f"{field}: expected atom-index label x y z magnetic spin-index"
            )
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
        spin = -1 if row[6] == "-" else _integer(row[6], field)
        if (row[5] == "Y") != (spin >= 0):
            raise ValueError(f"{field}: magnetic flag contradicts spin-index")
        species.append(label[1])
        labels.append(row[1])
        positions.append(tuple(_real(token, field) for token in row[2:5]))
        indices.append(spin)
        if spin >= 0:
            selected.append(i)
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
    try:
        with Path(path).open("rb") as stream:
            data = tomli.load(stream, parse_float=Decimal)
    except tomli.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML: {error}") from error
    _schema(data)
    structure = _structure(data["structure"])
    symmetry = _floats(data["symmetry"])
    reference = data.get("reference_spins")
    source = data.get("source", {})
    mode = {"spacegroup": "sg", "magnetic": "msg"}[symmetry["mode"]]
    kwargs = dict(
        **structure,
        exchange_mode=data["exchange"],
        energy_unit=data.get("energy_unit", "meV"),
        symmetry_mode=mode,
        symprec=symmetry.get("symprec", 1e-5),
        magnetic_tol=symmetry.get("magnetic_tol", 1e-5 if mode == "msg" else None),
        producer=symmetry.get("producer"),
        moments=_moments(reference, structure["site_labels"]),
        moment_source=reference.get("source", "explicit") if reference else "explicit",
        metadata=_floats(data.get("metadata", {})),
        cutoff=_floats(data.get("cutoff")),
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
        representative = (
            _representative(
                entry["representative"],
                structure["site_labels"],
                f"shell[{index}].representative",
            )
            if "representative" in entry
            else None
        )
        identifier = entry.get(
            "id", _pair_id(representative) if representative else None
        )
        value = _energy(entry["J"], scale, f"shell[{index}].J")
        extras = {
            key: _floats(entry[key])
            for key in ("name", "distance", "n_equivalent_pairs", *_MASKS)
            if key in entry
        }
        entries.append(
            ShellEntry(
                id=identifier,
                representative=representative,
                **{("Jiso" if data["exchange"] == "isotropic" else "tensor"): value},
                **extras,
            )
        )
    context = SimpleNamespace(**kwargs, operations=operations, shells=tuple(entries))
    orbits = _shell_orbits(context)
    for index, (entry, orbit) in enumerate(zip(entries, orbits)):
        if "id" not in data["shell"][index]:
            entries[index] = replace(entry, id=_pair_id(min(orbit)))
    return ShellDocument(**kwargs, operations=operations, shells=tuple(entries))


def _pair_text(pair, labels):
    return f"{labels[pair[0] - 1]} {labels[pair[1] - 1]} " + " ".join(
        str(x) for x in pair[2:]
    )


def _number_text(value):
    return repr(float(value))


def dump(document, path, *, equivalent_comments=True):
    structure = {
        "coordinates": "fractional",
        "lattice": "\n".join(
            " ".join(_number_text(value) for value in row)
            for row in document.cell_angstrom
        )
        + "\n",
        "atoms": "\n".join(
            f"{atom} {label} "
            + " ".join(_number_text(value) for value in position)
            + f" {'Y' if spin >= 0 else 'N'} {spin}"
            for atom, (label, position, spin) in enumerate(
                zip(document.site_labels, document.positions_frac, document.index_spin),
                1,
            )
        )
        + "\n",
    }
    header = {
        "schema_version": 2,
        "exchange": document.exchange_mode,
        "energy_unit": document.energy_unit,
    }
    if document.cutoff is not None:
        header["cutoff"] = document.cutoff
    header["structure"] = structure
    if document.moments is not None:
        header["reference_spins"] = {
            "moments": "\n".join(
                label
                + " "
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
        "symprec": document.symprec,
    }
    if document.magnetic_tol is not None:
        symmetry["magnetic_tol"] = document.magnetic_tol
    if document.producer is not None:
        symmetry["producer"] = document.producer
    header["symmetry"] = symmetry
    if document.metadata:
        header["metadata"] = _thaw(document.metadata)
    parts = [
        "# Exchange values use energy_unit (meV or eV); cutoff and distance are in Angstrom.\n"
        "# Lattice rows are Cartesian vectors in Angstrom; atom positions are fractional.\n"
        "# Atom rows: one-based atom index, label, x, y, z, magnetic Y/N, zero-based spin index.\n"
        "# Nonmagnetic atoms use N and a negative spin index; retain the exact existing mapping.\n"
        "# Reference moments, when present, follow the complete atom order.\n"
        + tomli_w.dumps(header, multiline_strings=True)
    ]
    scale = Decimal(1000) if document.energy_unit == "meV" else Decimal(1)
    for shell, orbit in zip(document.shells, _shell_orbits(document)):
        representative = (
            shell.representative
            or orbit[shell.support_members[0] if shell.support_members else 0]
        )
        entry = {}
        if shell.name is not None:
            entry["name"] = shell.name
        # Scalar ID-only entries can explicitly carry no observed support.
        # Do not invent a representative outside that empty support.
        if shell.support_members != ():
            entry["representative"] = _pair_text(representative, document.site_labels)
        entry["distance"] = _template_distance(
            representative,
            np.asarray(document.positions_frac),
            np.asarray(document.cell_angstrom),
        )
        entry["n_equivalent_pairs"] = len(orbit)
        entry["J"] = _wire_energy(
            shell.Jiso if document.exchange_mode == "isotropic" else shell.tensor, scale
        )
        if any(getattr(shell, key) is not None for key in _MASKS):
            entry["id"] = shell.id
        for key in _MASKS:
            if getattr(shell, key) is not None:
                entry[key] = getattr(shell, key)
        text = (
            "# representative: first label, second label, Rx Ry Rz; translation moves the second atom.\n"
            "# J is a scalar for isotropic exchange or a Cartesian 3x3 tensor; replace 'unfilled'.\n"
            "# name is optional; distance and n_equivalent_pairs may be omitted on input.\n"
            "# n_equivalent_pairs counts the full ordered-pair symmetry orbit, not coordination.\n"
            + "[[shell]]\n"
            + tomli_w.dumps(entry)
        )
        if equivalent_comments:
            text += "# Equivalent ordered pairs (informational):\n"
            for pair in orbit:
                text += f"# {_pair_text(pair, document.site_labels)}; atoms {pair[:2]}, spins {tuple(document.index_spin[atom - 1] for atom in pair[:2])}\n"
        parts.append(text)
    footer = {
        "symmetry": {
            "operations": [
                {
                    "rotation": operation.rotation,
                    "translation": operation.translation,
                    "time_reversal": operation.time_reversal,
                }
                for operation in document.operations
            ]
        }
    }
    footer["tolerances"] = {
        name: getattr(document, name)
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
    # The parent [symmetry] table was already declared in the editable header.
    # Serialize child arrays directly, avoiding a duplicate table declaration.
    footer_text = tomli_w.dumps(footer)
    footer_text = footer_text.removeprefix("[symmetry]\n")
    parts.append(
        ("# Generated replay snapshot and tolerances\n" if equivalent_comments else "")
        + footer_text
    )
    Path(path).write_text("\n".join(parts), encoding="utf-8")
