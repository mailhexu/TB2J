"""TOML v1 boundary for the immutable symmetry-shell model."""

from decimal import Decimal, InvalidOperation
from pathlib import Path

import tomli
import tomli_w

from .shells import ShellDocument, ShellEntry, ShellOperation, _pair_id, _shell_orbits


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
    # Loading with parse_float=Decimal avoids an intermediate meV float rounding.
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
        ("schema_version", "exchange_mode", "structure", "symmetry", "tolerances"),
        ("energy_unit", "reference_spins", "source", "shell"),
        "shell document",
    )
    _keys(
        data["structure"],
        ("cell_angstrom", "species", "positions_frac", "magnetic_atoms", "index_spin"),
        (),
        "structure",
    )
    _keys(
        data["symmetry"],
        ("mode", "symprec", "operations"),
        ("magnetic_tol", "producer"),
        "symmetry",
    )
    _keys(
        data["tolerances"], ("value_atol", "value_rtol", "zero_tol"), (), "tolerances"
    )
    if "reference_spins" in data:
        _keys(data["reference_spins"], (), ("moments", "source"), "reference_spins")
    if "source" in data:
        _keys(
            data["source"],
            (),
            ("format", "precision_profile", "required_base_fields"),
            "source",
        )
    operations = data["symmetry"]["operations"]
    if not isinstance(operations, list):
        raise ValueError("symmetry.operations: expected an array of operation tables")
    for index, operation in enumerate(operations):
        _keys(
            operation,
            ("rotation", "translation", "time_reversal"),
            (),
            f"symmetry.operations[{index}]",
        )
    shells = data.get("shell", [])
    if not isinstance(shells, list):
        raise ValueError("shell: expected an array of shell tables")
    for index, shell in enumerate(shells):
        _keys(
            shell,
            ("id",),
            (
                "representative",
                "tensor",
                "Jiso",
                "orbit_size",
                "support_members",
                "Jiso_members",
                "DMI_members",
                "Jani_members",
            ),
            f"shell[{index}]",
        )


def load(path):
    try:
        with Path(path).open("rb") as stream:
            data = tomli.load(stream, parse_float=Decimal)
    except tomli.TOMLDecodeError as error:
        diagnostic = (
            "duplicate TOML key"
            if "Cannot overwrite" in str(error)
            else "invalid TOML syntax/value"
        )
        raise ValueError(f"{diagnostic}: {error}") from error
    _schema(data)
    try:
        structure = _floats(data["structure"])
        symmetry = _floats(data["symmetry"])
        tolerances = _floats(data["tolerances"])
        reference = _floats(data.get("reference_spins", {}))
        source = data.get("source", {})
        unit = data.get("energy_unit", "meV")
        scale = Decimal(1000) if unit == "meV" else Decimal(1)
        operations = tuple(ShellOperation(**entry) for entry in symmetry["operations"])
        shells = []
        for entry in data.get("shell", []):
            converted = dict(entry)
            for name in ("Jiso", "tensor"):
                if name in converted:
                    converted[name] = _energy(
                        converted[name], scale, f"shell {entry['id']}.{name}"
                    )
            shells.append(ShellEntry(**converted))
        return ShellDocument(
            schema_version=data["schema_version"],
            exchange_mode=data["exchange_mode"],
            energy_unit=unit,
            cell_angstrom=structure["cell_angstrom"],
            species=structure["species"],
            positions_frac=structure["positions_frac"],
            magnetic_atoms=structure["magnetic_atoms"],
            index_spin=structure["index_spin"],
            symmetry_mode=symmetry["mode"],
            symprec=symmetry["symprec"],
            magnetic_tol=symmetry.get("magnetic_tol"),
            producer=symmetry.get("producer"),
            operations=operations,
            shells=tuple(shells),
            moments=reference.get("moments"),
            moment_source=reference.get("source", "explicit"),
            source_format=source.get("format"),
            precision_profile=source.get("precision_profile"),
            required_base_fields=source.get("required_base_fields", ()),
            value_atol=tolerances["value_atol"],
            value_rtol=tolerances["value_rtol"],
            zero_tol=tolerances["zero_tol"],
        )
    except KeyError as error:
        raise ValueError(
            f"shell document: missing required field {error.args[0]}"
        ) from error
    except TypeError as error:
        raise ValueError(f"shell document: invalid field: {error}") from error


def dump(document, path, *, equivalent_comments=True):
    structure = {
        name: getattr(document, name)
        for name in (
            "cell_angstrom",
            "species",
            "positions_frac",
            "magnetic_atoms",
            "index_spin",
        )
    }
    symmetry = {"mode": document.symmetry_mode, "symprec": document.symprec}
    if document.magnetic_tol is not None:
        symmetry["magnetic_tol"] = document.magnetic_tol
    if document.producer is not None:
        symmetry["producer"] = document.producer
    symmetry["operations"] = [
        {
            "rotation": operation.rotation,
            "translation": operation.translation,
            "time_reversal": operation.time_reversal,
        }
        for operation in document.operations
    ]
    header = {
        "schema_version": document.schema_version,
        "exchange_mode": document.exchange_mode,
        "energy_unit": document.energy_unit,
        "structure": structure,
    }
    if document.moments is not None:
        header["reference_spins"] = {
            "moments": document.moments,
            "source": document.moment_source,
        }
    header["symmetry"] = symmetry
    header["tolerances"] = {
        name: getattr(document, name)
        for name in ("value_atol", "value_rtol", "zero_tol")
    }
    source = {}
    for key, value in (
        ("format", document.source_format),
        ("precision_profile", document.precision_profile),
        ("required_base_fields", document.required_base_fields),
    ):
        if value:
            source[key] = value
    if source:
        header["source"] = source
    parts = [tomli_w.dumps(header)]
    scale = Decimal(1000) if document.energy_unit == "meV" else Decimal(1)
    orbits = _shell_orbits(document)
    for shell, orbit in zip(document.shells, orbits):
        entry = {"id": shell.id}
        for name in (
            "representative",
            "tensor",
            "Jiso",
            "orbit_size",
            "support_members",
            "Jiso_members",
            "DMI_members",
            "Jani_members",
        ):
            value = getattr(shell, name)
            if value is not None:
                entry[name] = (
                    _wire_energy(value, scale) if name in {"tensor", "Jiso"} else value
                )
        text = "[[shell]]\n" + tomli_w.dumps(entry)
        if equivalent_comments:
            pairs = "; ".join(
                f"({pair[0]}, {pair[1]}, R = {pair[2:]})" for pair in orbit
            )
            reverse = min((pair[1], pair[0], *(-x for x in pair[2:])) for pair in orbit)
            relation = "same symmetry orbit" if reverse in orbit else "separate orbit"
            text += f"# equivalent pairs (informational, removable): {pairs}\n"
            text += f"# reverse partner (reciprocity transpose, {relation}): {_pair_id(reverse)}\n"
        parts.append(text)
    Path(path).write_text("\n".join(parts), encoding="utf-8")
