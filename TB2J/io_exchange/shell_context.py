"""Data-only companion for trusted native exchange imports."""

# Closed decoding deliberately accepts only exact builtin and NumPy types.
# ruff: noqa: E721

import hashlib
import json
import math
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
from ase import Atoms

from . import shells
from .io_exchange import SpinIO

_DERIVED = {"atoms", "Rlist", "ispin_list", "nspin", "_ind_atoms"}


def _encode(value, arrays, active, path):
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, np.generic):
        encoded = _encode(np.asarray(value), arrays, active, path)
        return {"type": "scalar", "value": encoded["value"]}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite number")
        return value
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path}: non-finite complex number")
        return {"type": "complex", "value": [value.real, value.imag]}
    if type(value) in (date, datetime, time):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if type(value) is np.ndarray:
        if value.dtype.metadata is not None:
            raise ValueError(f"{path}: dtype metadata is unsupported")
        if value.dtype.fields is not None or value.dtype.kind not in "biufcSU":
            raise ValueError(f"{path}: unsupported array dtype {value.dtype}")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path}: non-finite array")
        name = f"array_{len(arrays)}"
        arrays[name] = value
        return {"type": "array", "value": name}
    if type(value) not in (dict, list, tuple):
        raise ValueError(f"{path}: unsupported {type(value).__name__}")
    identity = id(value)
    if identity in active:
        raise ValueError(f"{path}: cyclic data")
    active.add(identity)
    try:
        if type(value) is dict:
            data = [
                [
                    _encode(k, arrays, active, f"{path}.<key>"),
                    _encode(v, arrays, active, f"{path}[{k!r}]"),
                ]
                for k, v in value.items()
            ]
            return {"type": "dict", "value": data}
        return {
            "type": type(value).__name__,
            "value": [
                _encode(v, arrays, active, f"{path}[{i}]") for i, v in enumerate(value)
            ],
        }
    finally:
        active.remove(identity)


def _validate_physical_state(spinio):
    natom = len(spinio.atoms)
    for name in ("charges", "damping", "gyro_ratio"):
        value = getattr(spinio, name, None)
        if shells._base_field_present(value):
            shells._numbers(value, (natom,), f"native context {name}")
    indices = getattr(spinio, "index_spin", ())
    spin_to_atom = {spin: atom for atom, spin in enumerate(indices) if spin >= 0}
    for name, shape in (
        ("exchange_Jdict", ()),
        ("dmi_ddict", (3,)),
        ("Jani_dict", (3, 3)),
    ):
        values = getattr(spinio, name, None)
        if values is None:
            continue
        if type(values) is not dict:
            raise ValueError(f"native context {name}: expected a mapping")
        for key, value in values.items():
            shells._transform_native_key(name, key, spin_to_atom)
            shells._numbers(value, shape, f"native context {name} key {key!r}")


def write_context(spinio, path):
    atoms = spinio.atoms
    if type(atoms) is not Atoms or atoms.constraints or atoms.calc is not None:
        raise ValueError(
            "native context requires ASE Atoms without constraints or calculator"
        )
    extra = set(vars(atoms)) - {
        "_cellobj",
        "_pbc",
        "arrays",
        "_celldisp",
        "_constraints",
        "info",
        "_calc",
    }
    if extra:
        raise ValueError(
            f"native context.atoms.{sorted(extra)[0]}: unsupported attribute"
        )
    _validate_physical_state(spinio)
    payload = {
        "atoms": {
            "arrays": atoms.arrays,
            "cell": np.asarray(atoms.cell.array),
            "pbc": atoms.pbc,
            "celldisp": atoms.get_celldisp(),
            "info": atoms.info,
        },
        "state": {k: v for k, v in vars(spinio).items() if k not in _DERIVED},
    }
    arrays = {}
    encoded = _encode(payload, arrays, set(), "native context")
    manifest = json.dumps(
        {"version": 1, "payload": encoded}, allow_nan=False, separators=(",", ":")
    )
    arrays["manifest"] = np.frombuffer(manifest.encode("utf-8"), dtype=np.uint8)
    with open(path, "wb") as stream:
        np.savez_compressed(stream, **arrays)
    return _digest(path)


def _digest(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"native context companion not found: {path}") from exc


def _decode(value, archive, used):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is not dict or set(value) != {"type", "value"}:
        raise ValueError("invalid native context value")
    kind, data = value["type"], value["value"]
    if kind == "array":
        if (
            not isinstance(data, str)
            or not data.startswith("array_")
            or not data[6:].isdigit()
            or data in used
            or data not in archive
        ):
            raise ValueError("invalid native context array reference")
        used.add(data)
        result = archive[data]
        if result.dtype.fields is not None or result.dtype.kind not in "biufcSU":
            raise ValueError("unsupported native context array dtype")
        if result.dtype.kind in "fc" and not np.all(np.isfinite(result)):
            raise ValueError("non-finite native context array")
        return result
    if kind == "scalar":
        result = _decode({"type": "array", "value": data}, archive, used)
        if result.ndim != 0:
            raise ValueError("invalid native context scalar")
        return result[()]
    if kind == "complex" and isinstance(data, list) and len(data) == 2:
        parts = [_decode(x, archive, used) for x in data]
        if any(type(x) not in (int, float) for x in parts):
            raise ValueError("invalid complex value")
        return complex(*parts)
    if kind in ("date", "datetime", "time") and isinstance(data, str):
        return {"date": date, "datetime": datetime, "time": time}[kind].fromisoformat(
            data
        )
    if kind in ("list", "tuple") and isinstance(data, list):
        values = [_decode(v, archive, used) for v in data]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict" and isinstance(data, list):
        result = {}
        for pair in data:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("invalid native context mapping")
            key = _decode(pair[0], archive, used)
            try:
                if key in result:
                    raise ValueError("duplicate native context key")
                result[key] = _decode(pair[1], archive, used)
            except TypeError as exc:
                raise ValueError("unhashable native context key") from exc
        return result
    raise ValueError("invalid native context type")


def load_context(shell_path, document):
    if document.source_format != "pickle" or document.context_sha256 is None:
        raise ValueError("native shell document requires its context digest")
    path = Path(shell_path).with_suffix(".context.npz")
    if _digest(path) != document.context_sha256:
        raise ValueError("native context digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or "manifest" not in archive:
            raise ValueError("invalid native context archive")
        raw = archive["manifest"]
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("invalid native context manifest")
        manifest = json.loads(raw.tobytes().decode("utf-8"))
        if (
            type(manifest) is not dict
            or set(manifest) != {"version", "payload"}
            or type(manifest["version"]) is not int
            or manifest["version"] != 1
        ):
            raise ValueError("unsupported native context version")
        used = {"manifest"}
        payload = _decode(manifest["payload"], archive, used)
        if used != set(archive.files):
            raise ValueError("unreferenced native context arrays")
    if (
        type(payload) is not dict
        or set(payload) != {"atoms", "state"}
        or type(payload["state"]) is not dict
    ):
        raise ValueError("invalid native context payload")
    a = payload["atoms"]
    if type(a) is not dict or set(a) != {"arrays", "cell", "pbc", "celldisp", "info"}:
        raise ValueError("invalid native structure context")
    atom_arrays = a["arrays"]
    if (
        type(atom_arrays) is not dict
        or "numbers" not in atom_arrays
        or "positions" not in atom_arrays
    ):
        raise ValueError("invalid native atom arrays")
    atoms = Atoms(
        numbers=atom_arrays["numbers"],
        positions=atom_arrays["positions"],
        cell=a["cell"],
        pbc=a["pbc"],
        celldisp=a["celldisp"],
        info=a["info"],
    )
    for key, value in atom_arrays.items():
        if key not in {"numbers", "positions"}:
            atoms.new_array(key, value)
    if any(type(k) is not str or k in _DERIVED for k in payload["state"]):
        raise ValueError("invalid native state attributes")
    base = SpinIO.__new__(SpinIO)
    base.__dict__.update(payload["state"])
    base.atoms = atoms
    if not hasattr(base, "index_spin"):
        raise ValueError("native context index_spin is missing")
    shells._validate_compatible_base(document, base)
    _validate_physical_state(base)
    base._build_Rlist()
    base._build_ind_atoms()
    if set(shells._required_base_fields(base)) != set(document.required_base_fields):
        raise ValueError("native context required fields do not match shell document")
    return base
