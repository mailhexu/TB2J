"""Editable exchange shells defined by an authoritative symmetry snapshot.

Cell vectors are rows, atom selectors are one-based, and exchange values in
Python are eV. The TOML codec alone converts the declared wire energy unit.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from fractions import Fraction
from itertools import product
from types import MappingProxyType

import numpy as np

from TB2J import symmetry_core

_ID = re.compile(r"([1-9][0-9]*):([1-9][0-9]*):(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\Z")
_BASE_FIELDS = {
    "biquadratic": ("biquadratic_Jdict",),
    "charges": ("charges",),
    "damping": ("damping",),
    "djdx": ("dJdx", "dJdx2", "dJdx_orb"),
    "gyromag": ("gyro_ratio",),
    "njt": ("NJT_Jdict", "NJT_ddict"),
    "orbital": ("Jiso_orb", "DMI_orb", "Jani_orb"),
    "sia": ("sia_tensor", "k1", "k1dir"),
}


def _freeze(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value.item() if isinstance(value, np.generic) else value


def _thaw(value):
    """Restore ordinary TOML containers for serialization and native pickles."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_metadata(value, path="metadata"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: TOML table keys must be strings")
            _validate_metadata(item, f"{path}.{key}")
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_metadata(item, f"{path}[{index}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: metadata numbers must be finite")
    elif not isinstance(value, (str, bool, int, date, datetime, time)):
        raise ValueError(f"{path}: expected finite TOML data")


@dataclass(frozen=True)
class ShellOperation:
    rotation: tuple
    translation: tuple
    time_reversal: bool = False

    def __post_init__(self):
        object.__setattr__(self, "rotation", _freeze(self.rotation))
        object.__setattr__(self, "translation", _freeze(self.translation))


@dataclass(frozen=True)
class ShellEntry:
    id: str
    Jiso: float | str | None = None
    tensor: tuple | str | None = None
    representative: tuple | None = None
    n_equivalent_pairs: int | None = None
    support_members: tuple | None = None
    Jiso_members: tuple | None = None
    DMI_members: tuple | None = None
    Jani_members: tuple | None = None
    name: str | None = None
    distance: float | None = None

    def __post_init__(self):
        for name in (
            "Jiso",
            "tensor",
            "representative",
            "support_members",
            "Jiso_members",
            "DMI_members",
            "Jani_members",
            "distance",
            "n_equivalent_pairs",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class ShellDocument:
    cell_angstrom: tuple
    species: tuple
    positions_frac: tuple
    magnetic_atoms: tuple
    index_spin: tuple
    operations: tuple
    shells: tuple
    exchange_mode: str = "isotropic"
    energy_unit: str = "meV"
    symmetry_mode: str = "sg"
    symprec: float = 1e-5
    magnetic_tol: float | None = None
    moments: tuple | None = None
    moment_source: str = "explicit"
    producer: str | None = None
    value_atol: float = 1e-8
    value_rtol: float = 1e-6
    zero_tol: float = 1e-10
    source_format: str | None = None
    precision_profile: str | None = None
    required_base_fields: tuple = ()
    schema_version: int = 2
    site_labels: tuple | None = None
    metadata: Mapping = field(default_factory=dict)
    cutoff: float | None = None

    def __post_init__(self):
        for name in (
            "cell_angstrom",
            "species",
            "positions_frac",
            "magnetic_atoms",
            "index_spin",
            "operations",
            "shells",
            "moments",
            "required_base_fields",
            "symprec",
            "magnetic_tol",
            "value_atol",
            "value_rtol",
            "zero_tol",
            "site_labels",
            "metadata",
            "cutoff",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.site_labels is None:
            from TB2J.utils import symbol_number_list

            object.__setattr__(self, "site_labels", symbol_number_list(self.species))
        validate_shells(self)
        object.__setattr__(
            self,
            "operations",
            tuple(
                sorted(
                    self.operations,
                    key=lambda operation: (
                        tuple(value for row in operation.rotation for value in row),
                        operation.translation,
                        operation.time_reversal,
                    ),
                )
            ),
        )
        object.__setattr__(
            self, "shells", tuple(sorted(self.shells, key=lambda shell: shell.id))
        )


@dataclass(frozen=True)
class ValidationReport:
    producer_drift: bool | None = None
    messages: tuple = ()


def _has_boolean(value):
    if isinstance(value, (tuple, list)):
        return any(_has_boolean(item) for item in value)
    return isinstance(value, (bool, np.bool_))


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _numbers(value, shape, field):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{field}: complex values are not supported")
        if raw.dtype.kind in {"U", "S", "b"} or _has_boolean(value):
            raise ValueError(f"{field}: expected numbers, not strings or booleans")
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field}: expected real values with shape {shape}: {error}"
        ) from error
    if array.shape != shape:
        raise ValueError(f"{field}: expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field}: nonfinite value")
    return array


def _pair_from_id(identifier):
    match = _ID.fullmatch(identifier) if isinstance(identifier, str) else None
    if match is None:
        raise ValueError(f"shell {identifier!r}: invalid canonical id")
    return tuple(int(part) for part in match.groups())


def _pair_id(pair):
    i, j, x, y, z = pair
    return f"{i}:{j}:{x},{y},{z}"


def _core_key(pair):
    return pair[0] - 1, pair[1] - 1, tuple(pair[2:])


def _wire_key(key):
    return key[0] + 1, key[1] + 1, *key[2]


def _operation_key(rotation, primed):
    return tuple(int(value) for value in rotation.flat), primed


# MathDoc: tb2j/spacegroup-exchange-symmetrization#hexagonal_cartesian_rotation
# MathDoc: tb2j/spacegroup-exchange-symmetrization#bond_map_image_shift
def _validated_operations(document, cell, positions):
    """Validate the finite stored group before using the core's site maps."""
    symprec = _numbers(document.symprec, (), "symmetry.symprec")
    if symprec <= 0:
        raise ValueError("symmetry.symprec: expected a positive finite tolerance")
    rotations, translations, primes, buckets = [], [], [], {}
    column_basis = cell.T
    inverse_basis = np.linalg.inv(column_basis)
    metric_atol = document.symprec / max(np.linalg.norm(cell, axis=1))

    def same_translation(left, right):
        difference = left - right
        difference -= np.round(difference)
        return np.linalg.norm(difference @ cell) <= document.symprec

    for number, operation in enumerate(document.operations):
        prefix = f"symmetry.operations[{number}]"
        if not isinstance(operation, ShellOperation):
            raise ValueError(f"{prefix}: expected ShellOperation")
        rotation = _numbers(operation.rotation, (3, 3), f"{prefix}.rotation")
        if any(not _integer(value) for row in operation.rotation for value in row):
            raise ValueError(f"{prefix}.rotation: expected integer entries")
        rotation = rotation.astype(int)
        if round(np.linalg.det(rotation)) not in {-1, 1}:
            raise ValueError(f"{prefix}.rotation: expected a unimodular matrix")
        translation = _numbers(operation.translation, (3,), f"{prefix}.translation")
        if np.any(translation < 0) or np.any(translation >= 1):
            raise ValueError(
                f"{prefix}.translation: expected fractional values in [0, 1)"
            )
        if not isinstance(operation.time_reversal, bool):
            raise ValueError(f"{prefix}.time_reversal: expected a boolean")
        if operation.time_reversal and document.symmetry_mode == "sg":
            raise ValueError(
                f"{prefix}: primed operation is invalid in sg mode; "
                "declare msg with reference_spins.moments and magnetic_tol"
            )
        cartesian = column_basis @ rotation @ inverse_basis
        if not np.allclose(
            cartesian.T @ cartesian, np.eye(3), rtol=0, atol=metric_atol
        ):
            raise ValueError(f"{prefix}: rotation does not preserve the cell metric")
        key = _operation_key(rotation, operation.time_reversal)
        if any(
            same_translation(translation, translations[index])
            for index in buckets.get(key, ())
        ):
            raise ValueError(f"{prefix}: duplicate or tolerance-ambiguous operation")
        buckets.setdefault(key, []).append(number)
        rotations.append(rotation)
        translations.append(translation)
        primes.append(operation.time_reversal)

    identity_key = _operation_key(np.eye(3, dtype=int), False)
    if not any(
        same_translation(translations[index], np.zeros(3))
        for index in buckets.get(identity_key, ())
    ):
        raise ValueError("symmetry.operations: unprimed identity operation is required")
    for left, (rotation, translation, primed) in enumerate(
        zip(rotations, translations, primes)
    ):
        for right, (other_rotation, other_translation, other_primed) in enumerate(
            zip(rotations, translations, primes)
        ):
            key = _operation_key(rotation @ other_rotation, primed ^ other_primed)
            composed = rotation @ other_translation + translation
            if not any(
                same_translation(composed, translations[index])
                for index in buckets.get(key, ())
            ):
                raise ValueError(
                    f"symmetry.operations: closure/inverse missing for operations {left} * {right}"
                )
    operations = symmetry_core.SymmetryOperations(rotations, translations, primes)
    try:
        maps = symmetry_core._build_site_maps(
            positions, cell, operations, document.symprec
        )
    except ValueError as error:
        raise ValueError(
            f"symmetry.operations: site mapping failed: {error}"
        ) from error
    magnetic = set(document.magnetic_atoms)
    moments = (
        np.asarray(document.moments, dtype=float)
        if document.symmetry_mode == "msg"
        else None
    )
    for number, (rotation, translation, mapping) in enumerate(
        zip(rotations, translations, maps)
    ):
        site_images = positions @ rotation.T + translation
        for atom, image in enumerate(mapping):
            prefix = f"symmetry.operations[{number}]"
            if document.species[atom] != document.species[image]:
                raise ValueError(
                    f"{prefix}: species selection is not invariant at atom {atom + 1}"
                )
            if ((atom + 1) in magnetic) != ((int(image) + 1) in magnetic):
                raise ValueError(
                    f"{prefix}: magnetic site selection is not invariant at atom {atom + 1}"
                )
            displacement = site_images - positions[image]
            displacement -= np.round(displacement)
            distances = np.linalg.norm(displacement @ cell, axis=1)
            if np.count_nonzero(distances <= document.symprec) != 1:
                raise ValueError(
                    f"{prefix}: ambiguous site mapping; supply high-precision structure context"
                )
        if document.symmetry_mode == "msg":
            # MathDoc: tb2j/spacegroup-exchange-symmetrization
            sign = -1 if primes[number] else 1
            if moments.ndim == 1:
                transformed = sign * moments
                deviation = np.abs(transformed - moments[mapping])
            else:
                cartesian = column_basis @ rotation @ inverse_basis
                transformed = sign * np.linalg.det(cartesian) * (moments @ cartesian.T)
                deviation = np.linalg.norm(transformed - moments[mapping], axis=1)
            if np.any(deviation > document.magnetic_tol):
                raise ValueError(
                    f"reference_spins.moments: stored operation {number} does not preserve the reference"
                )
    return operations, maps


# MathDoc: tb2j/spacegroup-exchange-symmetrization#bond_map_image_shift
# MathDoc: tb2j/spacegroup-exchange-symmetrization#hexagonal_cartesian_rotation
def _shell_orbits(document):
    """Return full direct operation orbits; never add reciprocal partners."""
    cell = _numbers(document.cell_angstrom, (3, 3), "structure.cell_angstrom")
    if abs(np.linalg.det(cell)) < np.finfo(float).tiny:
        raise ValueError("structure.cell_angstrom: singular cell")
    positions = _numbers(
        document.positions_frac, (len(document.species), 3), "structure.positions_frac"
    )
    operations, maps = _validated_operations(document, cell, positions)
    seeds = []
    for shell in document.shells:
        seed = (
            shell.representative
            if shell.representative is not None
            else _pair_from_id(shell.id)
        )
        if len(seed) != 5 or any(not _integer(x) for x in seed):
            raise ValueError(
                f"shell {shell.id}: representative must contain five integers"
            )
        if (
            seed[0] not in document.magnetic_atoms
            or seed[1] not in document.magnetic_atoms
        ):
            raise ValueError(
                f"shell {shell.id}: representative endpoints must be magnetic atom indices"
            )
        seeds.append(_core_key(seed))
    # pair_images takes the column-basis matrix used by the MathDoc. ASE and
    # the wire format store rows; site-map distances above retain row vectors.
    images, _ = symmetry_core.pair_images(seeds, operations, maps, positions, cell.T)
    return [
        tuple(sorted({_wire_key(image[seed]) for image in images})) for seed in seeds
    ]


def _mask(value, default, size, field):
    members = default if value is None else value
    if (
        not isinstance(members, tuple)
        or any(not _integer(index) for index in members)
        or tuple(sorted(set(members))) != members
        or any(index < 0 or index >= size for index in members)
    ):
        raise ValueError(
            f"{field}: expected sorted unique zero-based full-orbit ordinals below {size}"
        )
    return members


def _channel_masks(shell, size):
    if shell.n_equivalent_pairs is not None and (
        not _integer(shell.n_equivalent_pairs) or shell.n_equivalent_pairs != size
    ):
        raise ValueError(
            f"shell {shell.id}.n_equivalent_pairs: stored {shell.n_equivalent_pairs}, computed {size}"
        )
    support = _mask(
        shell.support_members,
        tuple(range(size)),
        size,
        f"shell {shell.id}.support_members",
    )
    masks = {}
    for channel in ("Jiso", "DMI", "Jani"):
        name = channel + "_members"
        members = _mask(getattr(shell, name), support, size, f"shell {shell.id}.{name}")
        if not set(members).issubset(support):
            raise ValueError(
                f"shell {shell.id}.{name}: members must be within support_members"
            )
        masks[channel] = members
    return masks


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
def _nonzero_tensor_channels(tensor):
    """Exact authored-subspace predicates, without decomposition roundoff.

    Rational trace summation also avoids overflow and catastrophic cancellation.
    No tolerance may turn a genuinely authored component into an absent channel.
    """
    return {
        "Jiso": sum(Fraction(float(tensor[i, i])) for i in range(3)) != 0,
        "DMI": any(tensor[i, j] != tensor[j, i] for i, j in ((0, 1), (0, 2), (1, 2))),
        "Jani": not (
            tensor[0, 0] == tensor[1, 1] == tensor[2, 2]
            and all(tensor[i, j] == -tensor[j, i] for i, j in ((0, 1), (0, 2), (1, 2)))
        ),
    }


def _validate_context(document):
    from ase.data import atomic_numbers

    count = len(document.species)
    if count == 0 or any(
        not isinstance(symbol, str) or symbol not in atomic_numbers
        for symbol in document.species
    ):
        raise ValueError(
            "structure.species: expected nonempty chemical species in atom order"
        )
    selected = document.magnetic_atoms
    if (
        any(not _integer(atom) or atom < 1 or atom > count for atom in selected)
        or tuple(sorted(set(selected))) != selected
    ):
        raise ValueError(
            "structure.magnetic_atoms: expected sorted unique one-based atom indices"
        )
    indices = document.index_spin
    if len(indices) != count or any(not _integer(index) for index in indices):
        raise ValueError("structure.index_spin: expected one integer for each atom")
    if {atom + 1 for atom, index in enumerate(indices) if index >= 0} != set(
        selected
    ) or sorted(index for index in indices if index >= 0) != list(range(len(selected))):
        raise ValueError(
            "structure.index_spin: selected atoms must map uniquely onto contiguous spin indices"
        )
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#site_frame_fixed_translation_bond_error
    # Native R refers to these raw site representatives, never wrapped images.
    _numbers(document.positions_frac, (count, 3), "structure.positions_frac")
    if (
        not isinstance(document.site_labels, tuple)
        or len(document.site_labels) != count
        or any(
            not isinstance(label, str)
            or re.fullmatch(re.escape(symbol) + r"[1-9][0-9]*", label) is None
            for label, symbol in zip(document.site_labels, document.species)
        )
        or len(set(document.site_labels)) != count
    ):
        raise ValueError(
            "structure.site_labels: expected unique element-number labels in atom order"
        )
    if not isinstance(document.metadata, Mapping):
        raise ValueError("metadata: expected a TOML table")
    _validate_metadata(document.metadata)
    if document.cutoff is not None:
        cutoff = _numbers(document.cutoff, (), "cutoff")
        if cutoff <= 0:
            raise ValueError("cutoff: expected a positive radius in Angstrom")
    for name in ("value_atol", "value_rtol", "zero_tol"):
        value = _numbers(getattr(document, name), (), f"tolerances.{name}")
        if value < 0:
            raise ValueError(f"tolerances.{name}: expected a nonnegative tolerance")
    if document.symmetry_mode not in {"sg", "msg"}:
        raise ValueError("symmetry.mode: expected sg or msg")
    if document.symmetry_mode == "msg" and document.moments is None:
        raise ValueError("reference_spins.moments: required for msg symmetry")
    if document.symmetry_mode == "msg" and document.magnetic_tol is None:
        raise ValueError("symmetry.magnetic_tol: required for msg symmetry")
    if document.magnetic_tol is not None:
        value = _numbers(document.magnetic_tol, (), "symmetry.magnetic_tol")
        if value <= 0:
            raise ValueError("symmetry.magnetic_tol: expected a positive tolerance")
    if document.moments is not None:
        shape = np.shape(document.moments)
        if shape not in {(count,), (count, 3)}:
            raise ValueError(
                "reference_spins.moments: expected one scalar or Cartesian vector per atom"
            )
        _numbers(document.moments, shape, "reference_spins.moments")
    if document.moment_source not in {
        "explicit",
        "native:spinat",
        "model:unit-along-z",
    }:
        raise ValueError("reference_spins.source: unsupported provenance")
    if document.source_format not in {None, "template", "pickle", "text", "xml"}:
        raise ValueError("source.format: expected template, pickle, text or xml")
    if document.precision_profile is not None and (
        not isinstance(document.precision_profile, str)
        or not document.precision_profile
    ):
        raise ValueError("source.precision_profile: expected a nonempty profile name")
    allowed = _BASE_FIELDS
    if any(
        not isinstance(name, str) or name not in allowed
        for name in document.required_base_fields
    ) or len(set(document.required_base_fields)) != len(document.required_base_fields):
        raise ValueError(
            "source.required_base_fields: expected unique supported physical field names"
        )


def validate_shells(document, *, for_export=False, check_producer_ops=False):
    """Validate the authored document without replacing its operation snapshot."""
    if not _integer(document.schema_version) or document.schema_version != 2:
        raise ValueError(
            f"schema_version: unsupported version {document.schema_version!r}"
        )
    if document.exchange_mode not in {"isotropic", "tensor"}:
        raise ValueError("exchange_mode: expected isotropic or tensor")
    if document.energy_unit not in {"meV", "eV"}:
        raise ValueError("energy_unit: expected meV or eV")
    if not document.operations:
        raise ValueError("symmetry.operations: identity operation is required")
    _validate_context(document)
    orbits = _shell_orbits(document)
    seen = set()
    for shell, orbit in zip(document.shells, orbits):
        if shell.id in seen:
            raise ValueError(f"shell {shell.id}: duplicate canonical id")
        seen.add(shell.id)
        canonical = _pair_id(min(orbit))
        if shell.id != canonical:
            raise ValueError(
                f"shell {shell.id}: computed canonical minimum is {canonical}"
            )
        masks = _channel_masks(shell, len(orbit))
        if shell.name is not None and not isinstance(shell.name, str):
            raise ValueError(f"shell {shell.id}.name: expected a display string")
        if shell.distance is not None:
            distance = _numbers(shell.distance, (), f"shell {shell.id}.distance")
            representative = shell.representative or _pair_from_id(shell.id)
            computed = _template_distance(
                representative,
                np.asarray(document.positions_frac),
                np.asarray(document.cell_angstrom),
            )
            if distance < 0 or abs(float(distance) - computed) > document.symprec:
                raise ValueError(
                    f"shell {shell.id}.distance: stored {shell.distance}, computed {computed}"
                )
        if shell.representative is not None:
            support = _mask(
                shell.support_members,
                tuple(range(len(orbit))),
                len(orbit),
                f"shell {shell.id}.support_members",
            )
            if orbit.index(shell.representative) not in support:
                raise ValueError(
                    f"shell {shell.id}.representative: must belong to support_members"
                )
        if document.exchange_mode == "isotropic":
            if (
                shell.tensor is not None
                or shell.DMI_members is not None
                or shell.Jani_members is not None
            ):
                raise ValueError(
                    f"shell {shell.id}: tensor/DMI/Jani field conflicts with isotropic exchange_mode"
                )
            value = shell.Jiso
            if value != "unfilled":
                channels = {
                    "Jiso": bool(_numbers(value, (), f"shell {shell.id}.Jiso") != 0)
                }
        else:
            if shell.Jiso is not None:
                raise ValueError(
                    f"shell {shell.id}.Jiso: field conflicts with tensor exchange_mode"
                )
            if shell.representative is None:
                raise ValueError(
                    f"shell {shell.id}.representative: required in tensor mode"
                )
            value = shell.tensor
            if value != "unfilled":
                tensor = _numbers(value, (3, 3), f"shell {shell.id}.tensor")
                channels = _nonzero_tensor_channels(tensor)
        if for_export and value == "unfilled":
            field = "Jiso" if document.exchange_mode == "isotropic" else "tensor"
            raise ValueError(
                f"shell {shell.id}.{field}: unfilled exchange value cannot be exported"
            )
        if value != "unfilled":
            for channel, nonzero in channels.items():
                if not masks[channel] and nonzero:
                    raise ValueError(
                        f"shell {shell.id}.{channel}_members: empty mask would discard a nonzero subspace"
                    )
    return _producer_report(document) if check_producer_ops else ValidationReport()


def _producer_report(document):
    from ase import Atoms

    atoms = Atoms(
        symbols=document.species,
        cell=document.cell_angstrom,
        scaled_positions=document.positions_frac,
        pbc=True,
    )
    if document.symmetry_mode == "sg":
        detected = symmetry_core.crystal_symmetry_ops(atoms, symprec=document.symprec)
    else:
        detected = symmetry_core.magnetic_symmetry_ops(
            atoms,
            document.moments,
            symprec=document.symprec,
            magnetic_tol=document.magnetic_tol,
        )
    positions = np.asarray(document.positions_frac)
    cell = np.asarray(document.cell_angstrom)
    maps = symmetry_core._build_site_maps(positions, cell, detected, document.symprec)
    selected = np.asarray(document.index_spin) >= 0
    primes = detected.time_reversals
    if primes is None:
        primes = np.zeros(len(detected), dtype=bool)
    produced = [
        (rotation, translation, bool(primed))
        for rotation, translation, primed, mapping in zip(
            detected.rotations, detected.translations, primes, maps
        )
        if np.array_equal(selected[mapping], selected)
    ]
    matched = 0
    for stored in document.operations:
        for rotation, translation, primed in produced:
            delta = translation - np.asarray(stored.translation)
            delta -= np.round(delta)
            if (
                primed == stored.time_reversal
                and np.array_equal(rotation, stored.rotation)
                and np.linalg.norm(delta @ cell) <= document.symprec
            ):
                matched += 1
                break
    drift = matched != len(document.operations) or len(produced) != len(
        document.operations
    )
    messages = (
        ("Producer operations differ; the stored snapshot was not modified.",)
        if drift
        else ()
    )
    return ValidationReport(producer_drift=drift, messages=messages)


def check_producer_ops(document):
    """Report current-producer operation drift without changing stored context."""
    return validate_shells(document, check_producer_ops=True)


def read_shells(path):
    """Read and validate a UTF-8 shells-v1 TOML document."""
    from .shell_text import load

    return load(path)


def write_shells(document, path, *, equivalent_comments=True):
    """Write a validated document; informational pair comments are optional."""
    from .shell_text import dump

    validate_shells(document)
    dump(document, path, equivalent_comments=equivalent_comments)


@dataclass(frozen=True)
class ConversionReport:
    """Immutable evidence from one shell or native exchange conversion."""

    residuals: tuple = ()
    created_pair_keys: tuple = ()
    created_channel_keys: tuple = ()
    absent_channel_keys: tuple = ()
    snapped_values: tuple = ()
    encoding_changes: tuple = ()
    precision_profile: str | None = None
    quanta: tuple = ()
    losses: tuple = ()
    messages: tuple = ()
    metadata_actions: tuple = ()

    def __post_init__(self):
        for name in (
            "residuals",
            "created_pair_keys",
            "created_channel_keys",
            "absent_channel_keys",
            "snapped_values",
            "encoding_changes",
            "metadata_actions",
            "precision_profile",
            "quanta",
            "losses",
            "messages",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class ShellResult:
    document: ShellDocument
    report: ConversionReport


@dataclass(frozen=True)
class ExpansionResult:
    spinio: object
    report: ConversionReport


@dataclass(frozen=True)
class NativeContext:
    """Explicit identity and physical context missing from a native file."""

    atoms: object = None
    species: object = None
    index_spin: object = None
    magnetic_atoms: object = None
    reference_spins: object = None
    charges: object = None

    def __post_init__(self):
        for name in (
            "species",
            "index_spin",
            "magnetic_atoms",
            "reference_spins",
            "charges",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


@dataclass(frozen=True)
class NativeReadResult:
    spinio: object
    report: ConversionReport


def _transform_public_value(value):
    if isinstance(value, np.ndarray):
        return _freeze(value)
    return float(value) if np.ndim(value) == 0 else _freeze(value)


def _transform_close(left, right, atol, rtol):
    return bool(np.allclose(left, right, atol=atol, rtol=rtol))


def _transform_max_deviation(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
def _transform_channel_value(channel, value, cartesian, *, inverse=False):
    """Transport one authored channel without filling another channel's zeros."""
    if channel == "Jiso":
        return float(value)
    rotation = cartesian.T if inverse else cartesian
    if channel == "DMI":
        return float(np.linalg.det(cartesian)) * (rotation @ np.asarray(value))
    return rotation @ np.asarray(value) @ rotation.T


def _transform_reverse_value(channel, value):
    if channel == "Jiso":
        return float(value)
    if channel == "DMI":
        return -np.asarray(value, dtype=float)
    return np.asarray(value, dtype=float).T


def _transform_native_key(channel, key, spin_to_atom):
    if not isinstance(key, tuple) or len(key) != 3:
        raise ValueError(f"{channel} key {key!r}: expected (R, i, j)")
    R, i, j = key
    if (
        not isinstance(R, (tuple, list))
        or len(R) != 3
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in R
        )
        or any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in (i, j)
        )
    ):
        raise ValueError(
            f"{channel} key {key!r}: expected integer translation and spin indices"
        )
    R = tuple(int(value) for value in R)
    i, j = int(i), int(j)
    if i not in spin_to_atom or j not in spin_to_atom:
        raise ValueError(
            f"{channel} key {key!r}: spin endpoint is absent from index_spin"
        )
    return (R, i, j), (spin_to_atom[i] + 1, spin_to_atom[j] + 1, *R)


def _transform_moments(spinio, reference_spins, natom):
    if reference_spins is not None:
        raw = reference_spins
        source = "explicit"
    else:
        raw = getattr(spinio, "spinat", None)
        source = "native:spinat" if raw is not None else "explicit"
        if raw is None:
            return None, source
        values = np.asarray(raw)
        if values.shape == (natom, 3) and bool(getattr(spinio, "colinear", False)):
            values = _numbers(values, (natom, 3), "SpinIO.spinat")
            if np.any(values[:, :2] != 0):
                raise ValueError(
                    "SpinIO.spinat: collinear moments must be z-aligned before "
                    "scalar symmetry handling"
                )
            raw = values[:, 2]
    values = np.asarray(raw)
    if values.shape not in {(natom,), (natom, 3)}:
        raise ValueError(
            "reference_spins.moments: expected one scalar or Cartesian vector per atom"
        )
    values = _numbers(raw, values.shape, "reference_spins.moments")
    return values, source


def _transform_tolerance(value, name, *, positive=False):
    result = float(_numbers(value, (), f"tolerances.{name}"))
    if result < 0 or (positive and result <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"tolerances.{name}: expected a {qualifier} tolerance")
    return result


# MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
def _transform_traceless_symmetric(value):
    """Retain the channel's symmetry and exact zero trace in binary floats."""
    result = 0.5 * np.asarray(value) + 0.5 * np.asarray(value).T
    smallest = int(np.argmin(np.abs(np.diag(result))))
    others = [index for index in range(3) if index != smallest]
    # For a traceless diagonal the two largest magnitudes have opposite
    # signs. Their difference is exactly representable (Sterbenz), avoiding
    # a rounded trace that would falsely create an absent isotropic channel.
    result[smallest, smallest] = -(
        result[others[0], others[0]] + result[others[1], others[1]]
    )
    return result


def _transform_source_values(channel, source, spin_to_atom, zero_tol, snap_zero):
    if source is None:
        return {}, (), {}
    if not hasattr(source, "items"):
        raise ValueError(f"SpinIO {channel} channel must be a mapping or None")
    shape = {"Jiso": (), "DMI": (3,), "Jani": (3, 3)}[channel]
    values = {}
    wires = {}
    snapped = []
    for key, raw in source.items():
        native_key, wire = _transform_native_key(channel, key, spin_to_atom)
        try:
            raw_array = np.asarray(raw)
        except Exception as error:
            raise ValueError(
                f"{channel} key {native_key!r}: invalid numeric value"
            ) from error
        if raw_array.shape != shape:
            raise ValueError(
                f"{channel} key {native_key!r}: underdetermined value shape "
                f"{raw_array.shape}; expected {shape}"
            )
        value = _numbers(raw, shape, f"{channel} key {native_key!r}")
        if snap_zero:
            updated = np.where(np.abs(value) < zero_tol, 0.0, value)
            if not np.array_equal(updated, value):
                snapped.append(
                    (
                        channel,
                        native_key,
                        _transform_public_value(value),
                        _transform_public_value(updated),
                    )
                )
            value = updated
        values[native_key] = float(value) if shape == () else value
        wires[native_key] = wire
    return values, tuple(snapped), wires


def _transform_operation_snapshot(spinio, symmetry, moments, symprec, magnetic_tol):
    atoms = getattr(spinio, "atoms", None)
    if atoms is None:
        raise ValueError("SpinIO.atoms: full structure context is required")
    if symmetry == "sg":
        detected = symmetry_core.crystal_symmetry_ops(atoms, symprec=symprec)
    else:
        if moments is None:
            raise ValueError(
                "reference_spins.moments: required for msg symmetry; supply "
                "reference_spins or SpinIO.spinat"
            )
        detected = symmetry_core.magnetic_symmetry_ops(
            atoms, moments, symprec=symprec, magnetic_tol=magnetic_tol
        )
    primes = detected.time_reversals
    if primes is None:
        primes = [False] * len(detected)
    operations = tuple(
        ShellOperation(
            rotation=rotation,
            translation=translation,
            time_reversal=bool(primed),
        )
        for rotation, translation, primed in zip(
            detected.rotations, detected.translations, primes
        )
    )
    return atoms, operations, detected.description


def _transform_compose(after, before):
    return after[0] @ before[0], after[1] ^ before[1]


def _transform_inverse(transform):
    return transform[0].T, transform[1]


def _transform_apply(channel, value, transform):
    matrix, reversed_bond = transform
    if reversed_bond:
        value = _transform_reverse_value(channel, value)
    return _transform_channel_value(channel, value, matrix)


def _transform_operator_matrix(channel, transform):
    matrix, reversed_bond = transform
    if channel == "Jiso":
        return np.ones((1, 1))
    if channel == "DMI":
        sign = -1.0 if reversed_bond else 1.0
        return sign * float(np.linalg.det(matrix)) * matrix
    result = np.empty((9, 9))
    for column in range(9):
        basis = np.zeros((3, 3))
        basis.flat[column] = 1.0
        result[:, column] = _transform_apply(channel, basis, transform).reshape(-1)
    return result


def _transform_project_fixed(channel, value, cycles):
    if not cycles:
        return value
    dimension = 1 if channel == "Jiso" else 3 if channel == "DMI" else 9
    identity = np.eye(dimension)
    constraints = np.concatenate(
        [_transform_operator_matrix(channel, cycle) - identity for cycle in cycles],
        axis=0,
    )
    # There are only 1, 3 or 9 unknown components. A full left basis would
    # allocate a quadratic matrix in the number of operation paths.
    _, singular_values, right_vectors = np.linalg.svd(constraints, full_matrices=False)
    largest = singular_values[0] if singular_values.size else 0.0
    # Cartesian identity/composed cycles contain inversion roundoff. Rank
    # belongs to unit-scale transformation operators, not to that residual.
    threshold = np.finfo(float).eps * max(constraints.shape) * max(1.0, largest)
    rank = int(np.count_nonzero(singular_values > threshold))
    basis = right_vectors[rank:].T
    vector = np.asarray(value, dtype=float).reshape(-1)
    projected = (
        basis @ (basis.T @ vector)
        if basis.shape[1]
        else np.zeros(dimension, dtype=float)
    )
    return (
        float(projected[0])
        if channel == "Jiso"
        else projected.reshape((3,) if channel == "DMI" else (3, 3))
    )


def _transform_project_observations(
    channel,
    values,
    wires,
    native_to_core,
    image_maps,
    cartesian_rotations,
    shell_by_wire,
    group_records,
    value_atol,
    value_rtol,
):
    """Project only observed values onto symmetry and present-reversal constraints."""
    means = {record[0]: None for record in group_records}
    if not values:
        return means, (), []
    core_to_native = {native_to_core[key]: key for key in values}
    adjacency = {key: [] for key in values}
    for key in sorted(values):
        source_core = native_to_core[key]
        reverse_core = (
            source_core[1],
            source_core[0],
            tuple(-r for r in source_core[2]),
        )
        for op_index, image_map in enumerate(image_maps):
            image = image_map[source_core]
            target = core_to_native.get(image)
            rotation = cartesian_rotations[op_index]
            if target is not None:
                adjacency[key].append((target, (rotation, False)))
            # This is a constraint on an existing observation, not a
            # calculated entry for the possibly absent reverse bond.
            if image == reverse_core:
                adjacency[key].append((key, (rotation, True)))
    identity_rotation = np.eye(3)
    for key in sorted(values):
        R, i, j = key
        reverse = (tuple(-component for component in R), j, i)
        if reverse in values and key <= reverse:
            edge = (identity_rotation, True)
            adjacency[key].append((reverse, edge))
            if reverse != key:
                adjacency[reverse].append((key, edge))

    transforms = {}
    component_roots = {}
    projected_roots = {}
    rows = []
    failures = []
    for root in sorted(values):
        if root in transforms:
            continue
        transforms[root] = (identity_rotation, False)
        component_roots[root] = root
        pending = [root]
        vertices = []
        cycles = []
        while pending:
            source = pending.pop()
            vertices.append(source)
            for target, edge in adjacency[source]:
                candidate = _transform_compose(edge, transforms[source])
                if target not in transforms:
                    transforms[target] = candidate
                    component_roots[target] = root
                    pending.append(target)
                else:
                    cycle = _transform_compose(
                        _transform_inverse(transforms[target]), candidate
                    )
                    cycles.append(cycle)
        component_vertices = set(vertices)
        component_reciprocity = any(
            (
                tuple(-component for component in key[0]),
                key[2],
                key[1],
            )
            in component_vertices
            for key in vertices
        )
        observations = [
            _transform_apply(channel, values[key], _transform_inverse(transforms[key]))
            for key in vertices
        ]
        mean = np.mean(observations, axis=0)
        mean = float(mean) if channel == "Jiso" else mean
        projected_root = _transform_project_fixed(channel, mean, cycles)
        projected_roots[root] = projected_root
        for key in vertices:
            projected = _transform_apply(channel, projected_root, transforms[key])
            residual = np.asarray(values[key]) - np.asarray(projected)
            identifier = shell_by_wire[wires[key]]
            rows.append(
                (
                    identifier,
                    channel,
                    key,
                    _transform_public_value(projected),
                    _transform_public_value(residual),
                )
            )
            if not _transform_close(values[key], projected, value_atol, value_rtol):
                failures.append(
                    f"shell {identifier}"
                    f"{' reciprocity' if component_reciprocity else ''} "
                    f"{channel} key {key!r} "
                    f"deviation={_transform_max_deviation(values[key], projected):.8g}"
                )

    for identifier, orbit, observed_core, observed_wire in group_records:
        representative_core = _core_key(min(observed_wire))
        source_keys = sorted(key for key in values if wires[key] in observed_wire)
        if not source_keys:
            continue
        native_key = source_keys[0]
        root = component_roots[native_key]
        paths = [
            (op_index, cartesian_rotations[op_index])
            for op_index, image_map in enumerate(image_maps)
            if image_map[native_to_core[native_key]] == representative_core
        ]
        if not paths:
            raise ValueError(
                f"shell {identifier}: no stored operation reaches its representative"
            )
        rotation = paths[0][1]
        root_to_representative = _transform_compose(
            (rotation, False), transforms[native_key]
        )
        means[identifier] = _transform_apply(
            channel, projected_roots[root], root_to_representative
        )
    return means, tuple(rows), failures


def _base_field_present(value):
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size != 0
    if isinstance(value, (dict, list, tuple)):
        return len(value) != 0
    return True


def _required_base_fields(spinio):
    return tuple(
        group
        for group, fields in _BASE_FIELDS.items()
        if any(_base_field_present(getattr(spinio, field, None)) for field in fields)
    )


def compress(
    spinio,
    *,
    symmetry="sg",
    reference_spins=None,
    mode="tensor",
    on_inconsistent="error",
    snap_zero=False,
    energy_unit="meV",
    symprec=1e-5,
    magnetic_tol=1e-5,
    zero_tol=1e-10,
    value_atol=1e-8,
    value_rtol=1e-6,
):
    """Compress observed SpinIO channels over exact directed symmetry orbits."""
    from TB2J.Jtensor import combine_J_tensor, decompose_J_tensor

    if symmetry not in {"sg", "msg"}:
        raise ValueError("symmetry: expected sg or msg")
    if mode not in {"isotropic", "tensor"}:
        raise ValueError("mode: expected isotropic or tensor")
    if on_inconsistent not in {"error", "project"}:
        raise ValueError("on_inconsistent: expected error or project")
    if not isinstance(snap_zero, bool):
        raise ValueError("snap_zero: expected a boolean")
    if energy_unit not in {"meV", "eV"}:
        raise ValueError("energy_unit: expected meV or eV")
    symprec = _transform_tolerance(symprec, "symprec", positive=True)
    magnetic_tol = _transform_tolerance(magnetic_tol, "magnetic_tol", positive=True)
    zero_tol = _transform_tolerance(zero_tol, "zero_tol")
    value_atol = _transform_tolerance(value_atol, "value_atol")
    value_rtol = _transform_tolerance(value_rtol, "value_rtol")

    index_spin = getattr(spinio, "index_spin", None)
    if index_spin is None:
        raise ValueError("SpinIO.index_spin: full atom-to-spin mapping is required")
    normalized_indices = []
    for atom, index in enumerate(index_spin):
        if not isinstance(index, (int, np.integer)) or isinstance(
            index, (bool, np.bool_)
        ):
            raise ValueError(f"SpinIO.index_spin[{atom}]: expected an integer")
        normalized_indices.append(int(index))
    index_spin = tuple(normalized_indices)
    spin_to_atom = {spin: atom for atom, spin in enumerate(index_spin) if spin >= 0}
    atoms = getattr(spinio, "atoms", None)
    if atoms is None:
        raise ValueError("SpinIO.atoms: full structure context is required")
    if len(atoms) != len(index_spin):
        raise ValueError("SpinIO.index_spin: length does not match SpinIO.atoms")
    moments, moment_source = _transform_moments(
        spinio, reference_spins, len(index_spin)
    )
    if mode == "isotropic" and (
        getattr(spinio, "dmi_ddict", None) or getattr(spinio, "Jani_dict", None)
    ):
        raise ValueError(
            "mode isotropic cannot discard observed DMI or Jani channels; "
            "use mode tensor"
        )

    atoms, operation_snapshot, producer = _transform_operation_snapshot(
        spinio, symmetry, moments, symprec, magnetic_tol
    )
    cell = _numbers(atoms.get_cell().array, (3, 3), "structure.cell_angstrom")
    if abs(np.linalg.det(cell)) < np.finfo(float).tiny:
        raise ValueError("structure.cell_angstrom: singular cell")
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#site_frame_cartesian_bond_residual
    positions = _numbers(
        atoms.get_scaled_positions(wrap=False),
        (len(index_spin), 3),
        "structure.positions_frac",
    )
    species = tuple(atoms.get_chemical_symbols())
    magnetic_atoms = tuple(
        atom + 1 for atom, index in enumerate(index_spin) if index >= 0
    )
    source_channels = {
        "Jiso": getattr(spinio, "exchange_Jdict", None),
        "DMI": getattr(spinio, "dmi_ddict", None) if mode == "tensor" else None,
        "Jani": getattr(spinio, "Jani_dict", None) if mode == "tensor" else None,
    }
    values = {}
    wires = {}
    snapped = []
    for channel, source in source_channels.items():
        channel_values, channel_snapped, channel_wires = _transform_source_values(
            channel, source, spin_to_atom, zero_tol, snap_zero
        )
        values[channel] = channel_values
        wires.update(channel_wires)
        snapped.extend(channel_snapped)

    # Normalize each OBSERVED native tensor before comparing channels.
    # Decomposing an orbit mean and adding its trace afterward would sum
    # disjoint observations rather than compare them, changing pair tensors.
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
    normalized_encodings = []
    for key, raw in values["Jani"].items():
        nonzero = _nonzero_tensor_channels(raw)
        raw_iso, raw_dmi, raw_ani = decompose_J_tensor(raw)
        raw_iso = float(sum(Fraction(float(raw[i, i])) for i in range(3)) / 3)
        if nonzero["Jiso"]:
            values["Jiso"][key] = values["Jiso"].get(key, 0.0) + raw_iso
        if nonzero["DMI"]:
            values["DMI"][key] = values["DMI"].get(key, np.zeros(3)) + raw_dmi
        raw_ani = (
            _transform_traceless_symmetric(raw_ani)
            if nonzero["Jani"]
            else np.zeros((3, 3))
        )
        values["Jani"][key] = raw_ani
        if nonzero["Jiso"] or nonzero["DMI"]:
            normalized_encodings.append(
                (
                    key,
                    "Jani",
                    _transform_public_value(raw),
                    raw_iso,
                    _transform_public_value(raw_dmi),
                    _transform_public_value(raw_ani),
                )
            )

    source_format = getattr(spinio, "_shell_source_format", None)
    precision_profile = getattr(spinio, "_shell_precision_profile", None)
    required_base_fields = _required_base_fields(spinio)

    from types import SimpleNamespace

    from TB2J.utils import symbol_number_list

    labels = getattr(spinio, "_shell_site_labels", symbol_number_list(species))
    if labels is None:
        raise ValueError("structure.site_labels: invalid native shell annotation")
    metadata = _freeze(getattr(spinio, "_shell_user_metadata", {}))

    operations_context = SimpleNamespace(
        species=species,
        positions_frac=positions,
        magnetic_atoms=magnetic_atoms,
        index_spin=index_spin,
        operations=operation_snapshot,
        symmetry_mode=symmetry,
        symprec=symprec,
        magnetic_tol=magnetic_tol,
        moments=moments,
        moment_source=moment_source,
        source_format=source_format,
        precision_profile=precision_profile,
        required_base_fields=required_base_fields,
        value_atol=value_atol,
        value_rtol=value_rtol,
        zero_tol=zero_tol,
        site_labels=_freeze(labels),
        metadata=metadata,
        cutoff=None,
    )
    _validate_context(operations_context)
    core_operations, site_maps = _validated_operations(
        operations_context, cell, positions
    )
    native_to_core = {native: _core_key(wire) for native, wire in wires.items()}
    seeds = sorted(set(native_to_core.values()))
    if seeds:
        image_maps, cartesian_rotations = symmetry_core.pair_images(
            seeds, core_operations, site_maps, positions, cell.T
        )
    else:
        image_maps, cartesian_rotations = [], []

    orbit_groups = {}
    for seed in seeds:
        orbit = tuple(sorted({_wire_key(image_map[seed]) for image_map in image_maps}))
        orbit_groups.setdefault(orbit, set()).add(seed)
    ordered_groups = sorted(orbit_groups.items(), key=lambda item: min(item[0]))
    shell_by_wire = {}
    group_records = []
    for orbit, observed_core in ordered_groups:
        identifier = _pair_id(min(orbit))
        observed_wire = {_wire_key(key) for key in observed_core}
        for wire in observed_wire:
            shell_by_wire[wire] = identifier
        group_records.append((identifier, orbit, observed_core, observed_wire))

    means_by_channel = {}
    residuals = []
    failures = []
    for channel in ("Jiso", "DMI", "Jani"):
        channel_means, channel_residuals, channel_failures = (
            _transform_project_observations(
                channel,
                values[channel],
                wires,
                native_to_core,
                image_maps,
                cartesian_rotations,
                shell_by_wire,
                group_records,
                value_atol,
                value_rtol,
            )
        )
        means_by_channel[channel] = channel_means
        residuals.extend(channel_residuals)
        failures.extend(channel_failures)

    entries = []
    encoding_changes = [
        (shell_by_wire[wires[row[0]]], *row) for row in normalized_encodings
    ]
    for identifier, orbit, observed_core, observed_wire in group_records:
        representative = min(observed_wire)
        orbit_index = {pair: index for index, pair in enumerate(orbit)}
        support_members = tuple(sorted(orbit_index[pair] for pair in observed_wire))
        channel_means = {}
        channel_members = {}
        for channel in ("Jiso", "DMI", "Jani"):
            source_keys = sorted(
                key
                for key, wire in wires.items()
                if wire in observed_wire and key in values[channel]
            )
            channel_members[channel] = tuple(
                sorted(orbit_index[wires[key]] for key in source_keys)
            )
            channel_means[channel] = means_by_channel[channel][identifier]
        if mode == "isotropic":
            value = channel_means.get("Jiso")
            if value is None:
                raise ValueError(f"shell {identifier}: no observed Jiso channel")
            entries.append(
                ShellEntry(
                    id=identifier,
                    Jiso=value,
                    representative=representative,
                    n_equivalent_pairs=len(orbit),
                    distance=_template_distance(representative, positions, cell),
                    support_members=support_members,
                    Jiso_members=channel_members["Jiso"],
                )
            )
            continue

        jiso = channel_means["Jiso"]
        dmi = channel_means["DMI"]
        jani = channel_means["Jani"]
        if jani is not None:
            jani = _transform_traceless_symmetric(jani)
        total_jiso = 0.0 if jiso is None else jiso
        total_dmi = np.zeros(3) if dmi is None else np.asarray(dmi)
        tensor = combine_J_tensor(
            Jiso=total_jiso,
            D=total_dmi,
            Jani=jani,
        )
        jiso_members = set(channel_members.get("Jiso", ()))
        dmi_members = set(channel_members.get("DMI", ()))
        jani_members = set(channel_members.get("Jani", ()))
        entries.append(
            ShellEntry(
                id=identifier,
                tensor=tuple(map(tuple, tensor)),
                representative=representative,
                n_equivalent_pairs=len(orbit),
                distance=_template_distance(representative, positions, cell),
                support_members=support_members,
                Jiso_members=tuple(sorted(jiso_members)),
                DMI_members=tuple(sorted(dmi_members)),
                Jani_members=tuple(sorted(jani_members)),
            )
        )

    if failures and on_inconsistent == "error":
        raise ValueError("inconsistent exchange data: " + "; ".join(failures))

    document = ShellDocument(
        cell_angstrom=cell,
        species=species,
        positions_frac=positions,
        magnetic_atoms=magnetic_atoms,
        index_spin=index_spin,
        operations=operation_snapshot,
        shells=tuple(entries),
        exchange_mode=mode,
        energy_unit=energy_unit,
        symmetry_mode=symmetry,
        symprec=symprec,
        magnetic_tol=magnetic_tol,
        moments=moments,
        moment_source=moment_source,
        producer=producer,
        source_format=source_format,
        precision_profile=precision_profile,
        required_base_fields=required_base_fields,
        site_labels=labels,
        metadata=metadata,
        value_atol=value_atol,
        value_rtol=value_rtol,
        zero_tol=zero_tol,
    )
    present_channels = {
        channel: set(values[channel]) for channel in ("Jiso", "DMI", "Jani")
    }
    present_pairs = set(wires.values())
    native_by_wire = {wire: native for native, wire in wires.items()}
    absent_channel_keys = tuple(
        (channel, native_by_wire[wire])
        for wire in sorted(present_pairs)
        for channel in ("Jiso", "DMI", "Jani")
        if native_by_wire[wire] not in present_channels[channel]
    )
    report = ConversionReport(
        residuals=tuple(residuals),
        absent_channel_keys=absent_channel_keys,
        snapped_values=tuple(snapped),
        encoding_changes=tuple(encoding_changes),
        precision_profile=precision_profile,
        metadata_actions=tuple(
            (group, "require_base", "source-present content is not encoded in shells")
            for group in required_base_fields
        ),
    )
    return ShellResult(document=document, report=report)


def _expand_core(document, *, complete_orbits=False):
    """Expand a validated shell model to an in-memory SpinIO without base data."""
    from ase import Atoms

    from TB2J.Jtensor import combine_J_tensor, decompose_J_tensor

    if not isinstance(complete_orbits, bool):
        raise ValueError("complete_orbits: expected a boolean")
    validate_shells(document, for_export=True)
    cell = _numbers(document.cell_angstrom, (3, 3), "structure.cell_angstrom")
    positions = _numbers(
        document.positions_frac,
        (len(document.species), 3),
        "structure.positions_frac",
    )
    operations, site_maps = _validated_operations(document, cell, positions)
    representative_keys = [
        _core_key(
            shell.representative
            if shell.representative is not None
            else _pair_from_id(shell.id)
        )
        for shell in document.shells
    ]
    if representative_keys:
        image_maps, cartesian_rotations = symmetry_core.pair_images(
            representative_keys, operations, site_maps, positions, cell.T
        )
    else:
        image_maps, cartesian_rotations = [], []

    channel_values = {"Jiso": {}, "DMI": {}, "Jani": {}}
    created_pair_keys = set()
    created_channel_keys = set()
    potential_pairs = set()
    failure_messages = []
    for shell_index, shell in enumerate(document.shells):
        representative_core = representative_keys[shell_index]
        orbit = tuple(
            sorted(
                {_wire_key(image_map[representative_core]) for image_map in image_maps}
            )
        )
        masks = _channel_masks(shell, len(orbit))
        support = _mask(
            shell.support_members,
            tuple(range(len(orbit))),
            len(orbit),
            f"shell {shell.id}.support_members",
        )
        selected_members = tuple(range(len(orbit))) if complete_orbits else support
        for member_index in selected_members:
            potential_pairs.add(orbit[member_index])

        if document.exchange_mode == "isotropic":
            jiso = float(_numbers(shell.Jiso, (), f"shell {shell.id}.Jiso"))
            D = None
            Jani = None
            tensor = combine_J_tensor(Jiso=jiso)
            active_channels = ("Jiso",)
        else:
            tensor = _numbers(shell.tensor, (3, 3), f"shell {shell.id}.tensor")
            jiso, D, Jani = decompose_J_tensor(tensor)
            if tensor[0, 0] == tensor[1, 1] == tensor[2, 2]:
                jiso = float(tensor[0, 0])
            active_channels = ("Jiso", "DMI", "Jani")

        for member in orbit:
            target_core = _core_key(member)
            paths = [
                (op_index, cartesian_rotations[op_index])
                for op_index, image_map in enumerate(image_maps)
                if image_map[representative_core] == target_core
            ]
            candidates = [
                (op_index, rotation @ tensor @ rotation.T)
                for op_index, rotation in paths
            ]
            if not candidates:
                raise ValueError(
                    f"shell {shell.id}: no stored operation reaches image {_pair_id(member)}"
                )
            reference = candidates[0][1]
            for op_index, candidate in candidates:
                if not _transform_close(
                    candidate, reference, document.value_atol, document.value_rtol
                ):
                    residual = _transform_max_deviation(candidate, reference)
                    if target_core == representative_core:
                        reason = "bond fixpoint"
                    else:
                        reason = "paths"
                    failure_messages.append(
                        f"shell {shell.id}: {reason} to image {_pair_id(member)} "
                        f"disagree at operation {op_index}, deviation={residual:.8g}"
                    )
            if target_core == representative_core and not _transform_close(
                reference, tensor, document.value_atol, document.value_rtol
            ):
                failure_messages.append(
                    f"shell {shell.id}: bond fixpoint changes image {_pair_id(member)}"
                )
            reverse_core = (
                representative_core[1],
                representative_core[0],
                tuple(-r for r in representative_core[2]),
            )
            if target_core == reverse_core:
                for op_index, candidate in candidates:
                    if not _transform_close(
                        candidate, tensor.T, document.value_atol, document.value_rtol
                    ):
                        failure_messages.append(
                            f"shell {shell.id}: endpoint exchange violates transpose "
                            f"at operation {op_index}, "
                            f"deviation={_transform_max_deviation(candidate, tensor.T):.8g}"
                        )

        if document.exchange_mode == "tensor":
            representative_channels = {
                "Jiso": jiso,
                "DMI": D,
                "Jani": Jani,
            }
        else:
            representative_channels = {"Jiso": jiso}
        for channel in active_channels:
            channel_members = masks[channel]
            active_members = (
                tuple(range(len(orbit)))
                if complete_orbits and channel_members
                else channel_members
            )

            for member_index in active_members:
                member = orbit[member_index]
                native_key = (
                    tuple(member[2:]),
                    document.index_spin[member[0] - 1],
                    document.index_spin[member[1] - 1],
                )
                if native_key[1] < 0 or native_key[2] < 0:
                    raise ValueError(
                        f"shell {shell.id}: image {_pair_id(member)} has a nonmagnetic endpoint"
                    )
                target_core = _core_key(member)
                paths = [
                    (op_index, cartesian_rotations[op_index])
                    for op_index, image_map in enumerate(image_maps)
                    if image_map[representative_core] == target_core
                ]
                rotation = paths[0][1]
                value = _transform_channel_value(
                    channel,
                    representative_channels[channel],
                    rotation,
                )
                channel_values[channel][native_key] = value
                if member_index not in channel_members:
                    created_channel_keys.add((channel, native_key))
                if member_index not in support:
                    created_pair_keys.add(native_key)

    if failure_messages:
        raise ValueError("; ".join(failure_messages))

    for channel in ("Jiso", "DMI", "Jani"):
        values = channel_values[channel]
        for key in sorted(values):
            R, i, j = key
            reverse = (tuple(-component for component in R), j, i)
            if reverse not in values or key > reverse:
                continue
            expected = _transform_reverse_value(channel, values[key])
            if not _transform_close(
                values[reverse],
                expected,
                document.value_atol,
                document.value_rtol,
            ):
                raise ValueError(
                    f"shell expansion violates present-pair reciprocity for "
                    f"{channel} keys {key!r}/{reverse!r}"
                )

    spinio_channels = {
        "exchange_Jdict": channel_values["Jiso"] or None,
        "dmi_ddict": channel_values["DMI"] or None,
        "Jani_dict": channel_values["Jani"] or None,
    }
    atoms = Atoms(
        symbols=document.species,
        cell=document.cell_angstrom,
        scaled_positions=document.positions_frac,
        pbc=True,
    )
    moments = (
        None if document.moments is None else np.asarray(document.moments, dtype=float)
    )
    if moments is None:
        spinat = None
        colinear = False
    elif moments.ndim == 1:
        spinat = np.zeros((len(document.species), 3), dtype=float)
        spinat[:, 2] = moments
        colinear = True
    else:
        spinat = moments.copy()
        colinear = False

    from .io_exchange import SpinIO

    spinio = SpinIO(
        atoms=atoms,
        spinat=spinat,
        charges=None,
        index_spin=list(document.index_spin),
        colinear=colinear,
        **spinio_channels,
    )
    # SpinIO's constructor supplies numeric defaults for omitted physical
    # fields; core expansion has no authority to invent those values.
    spinio.damping = None
    spinio.gyro_ratio = None

    all_channels = ("Jiso", "DMI", "Jani")
    absent_channel_keys = tuple(
        (
            channel,
            (
                tuple(pair[2:]),
                document.index_spin[pair[0] - 1],
                document.index_spin[pair[1] - 1],
            ),
        )
        for pair in sorted(potential_pairs)
        for channel in all_channels
        if (
            (
                tuple(pair[2:]),
                document.index_spin[pair[0] - 1],
                document.index_spin[pair[1] - 1],
            )
            not in channel_values[channel]
        )
    )
    return ExpansionResult(
        spinio=spinio,
        report=ConversionReport(
            created_pair_keys=tuple(sorted(created_pair_keys)),
            created_channel_keys=tuple(sorted(created_channel_keys)),
            absent_channel_keys=absent_channel_keys,
        ),
    )


def _validate_compatible_base(document, base):
    from .io_exchange import SpinIO

    if not isinstance(base, SpinIO):
        raise TypeError("base: expected SpinIO")
    if tuple(base.atoms.get_chemical_symbols()) != document.species:
        raise ValueError("base species/order does not match the shell document")
    indices = tuple(base.index_spin)
    if (
        any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            for index in indices
        )
        or indices != document.index_spin
    ):
        raise ValueError("base index_spin does not match the exact document mapping")
    cell = _numbers(base.atoms.cell.array, (3, 3), "base cell")
    stored_cell = np.asarray(document.cell_angstrom)
    if np.any(np.linalg.norm(cell - stored_cell, axis=1) > document.symprec):
        raise ValueError("base cell differs beyond symmetry.symprec")
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#site_frame_fixed_translation_bond_error
    # Equal periodic sites are not equal native fixed-R frames.
    positions = _numbers(
        base.atoms.positions, (len(document.species), 3), "base positions"
    )
    delta = positions - np.asarray(document.positions_frac) @ stored_cell
    if np.any(np.linalg.norm(delta, axis=1) > document.symprec):
        raise ValueError("base positions differ beyond symmetry.symprec")
    native = getattr(base, "spinat", None)
    if document.moments is None:
        if native is not None:
            raise ValueError("base moments are present but absent in the document")
    else:
        if native is None:
            raise ValueError("base moments are missing")
        expected = np.asarray(document.moments)
        if expected.ndim == 1:
            expected = np.column_stack((np.zeros((len(expected), 2)), expected))
        actual = _numbers(native, expected.shape, "base moments")
        if np.ndim(document.moments) == 1 and np.any(actual[:, :2] != 0):
            raise ValueError(
                "base moments: a scalar reference requires exactly z-aligned native moments"
            )
        # Never invent a magnetic tolerance for a header that declares none.
        tolerance = document.magnetic_tol if document.magnetic_tol is not None else 0.0
        if np.any(np.linalg.norm(actual - expected, axis=1) > tolerance):
            raise ValueError(
                "base moments differ beyond the declared magnetic tolerance"
            )


def _base_exchange_equal(document, left, right):
    from TB2J.Jtensor import combine_J_tensor

    channels = ("exchange_Jdict", "dmi_ddict", "Jani_dict")
    left_keys = set().union(*(getattr(left, name, None) or {} for name in channels))
    right_keys = set().union(*(getattr(right, name, None) or {} for name in channels))
    if left_keys != right_keys:
        return False

    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
    def total(spinio, key):
        return combine_J_tensor(
            Jiso=(getattr(spinio, "exchange_Jdict", None) or {}).get(key),
            D=(getattr(spinio, "dmi_ddict", None) or {}).get(key),
            Jani=(getattr(spinio, "Jani_dict", None) or {}).get(key),
        )

    return all(
        _transform_close(
            total(left, key),
            total(right, key),
            document.value_atol,
            document.value_rtol,
        )
        for key in left_keys
    )


def _normalize_base_sia(output):
    from .io_exchange import SpinIO

    k1, directions = getattr(output, "k1", None), getattr(output, "k1dir", None)
    if k1 is None and directions is None:
        return False
    if k1 is None or directions is None:
        raise ValueError("base SIA: legacy k1 and k1dir must be supplied together")
    # Reuse the native normalization authority on the isolated copy, including
    # existing tensors; never duplicate its axis/threshold convention here.
    normalized = SpinIO(
        output.atoms,
        spinat=output.spinat,
        charges=output.charges,
        index_spin=output.index_spin,
        colinear=output.colinear,
        k1=k1,
        k1dir=directions,
        sia_tensor=getattr(output, "sia_tensor", None),
    )
    for name in (
        "sia_tensor",
        "has_sia_tensor",
        "k1",
        "k1dir",
        "has_uniaxial_anistropy",
    ):
        setattr(output, name, getattr(normalized, name))
    return True


def expand(document, *, base=None, complete_orbits=False, allow_loss=()):
    """Expand shell values, preserving compatible ancillary context explicitly."""
    from copy import copy, deepcopy
    from dataclasses import replace

    from .io_exchange import gen_distance_dict

    validate_shells(document, for_export=True)
    if isinstance(allow_loss, str):
        allow_loss = (allow_loss,)
    try:
        allow_loss = tuple(allow_loss)
    except TypeError as error:
        raise ValueError("allow_loss: expected named ancillary loss groups") from error
    if any(
        not isinstance(name, str) or name not in _BASE_FIELDS for name in allow_loss
    ):
        raise ValueError("allow_loss: unknown expansion loss group")
    allowed = set(allow_loss)
    if base is not None:
        _validate_compatible_base(document, base)
    available = set(_required_base_fields(base)) if base is not None else set()
    missing = set(document.required_base_fields) - available
    denied = sorted(missing - allowed)
    if denied:
        raise ValueError(
            "required base fields missing; supply a compatible base or named allow_loss: "
            + ", ".join(denied)
        )
    losses = [
        (name, "required source content omitted without a compatible value")
        for name in sorted(missing)
    ]
    actions = [
        (name, "drop", "explicitly allowed missing source content")
        for name in sorted(missing)
    ]
    core = _expand_core(document, complete_orbits=complete_orbits)
    if base is None:
        output = core.spinio
    else:
        stale = set(available) & {"orbital", "djdx"}
        if stale and not _base_exchange_equal(document, base, core.spinio):
            denied = sorted(stale - allowed)
            if denied:
                raise ValueError(
                    "edited exchange conflicts with stale derived data; name allow_loss: "
                    + ", ".join(denied)
                )
        else:
            stale = set()
        # Replaced pair dictionaries and explicitly dropped arrays can be large.
        # Copy only retained context, preserving its internal aliases/cycles.
        overwritten = {
            "exchange_Jdict",
            "dmi_ddict",
            "Jani_dict",
            "Rlist",
            "distance_dict",
            "_ind_atoms",
            "magmoms",
        }
        overwritten.update(field for group in stale for field in _BASE_FIELDS[group])
        output = copy(base)
        output.__dict__ = deepcopy(
            {
                name: value
                for name, value in vars(base).items()
                if name not in overwritten
            },
            {id(base): output},
        )
        for name in ("exchange_Jdict", "dmi_ddict", "Jani_dict"):
            setattr(output, name, getattr(core.spinio, name))
            actions.append(
                (name, "differ", "shell values and support replace native exchange")
            )
        for group in sorted(stale):
            for name in _BASE_FIELDS[group]:
                setattr(output, name, None)
            losses.append((group, "exchange edit invalidates derived content"))
            actions.append((group, "drop", "explicitly allowed stale derived content"))
        for group in sorted(available - stale):
            actions.append((group, "keep", "compatible base content preserved"))
        for name in (
            "cell",
            "positions",
            "species",
            "index_spin",
            "spinat",
            "mass",
            "description",
            "TB2J_version",
        ):
            actions.append((name, "keep", "compatible base context preserved"))
        output.colinear = core.spinio.colinear
        if output.colinear:
            output.magmoms = np.asarray(output.spinat)[:, 2].copy()
        elif hasattr(output, "magmoms"):
            del output.magmoms
        actions.append(
            ("colinear", "differ", "explicit scalar versus Cartesian reference shape")
        )
        if _normalize_base_sia(output):
            actions.append(("sia", "differ", "legacy fields normalized once by SpinIO"))
    output._build_Rlist()
    output._build_ind_atoms()
    magnetic_atoms = [output.ind_atoms[spin] for spin in output.ispin_list]
    output.distance_dict = gen_distance_dict(magnetic_atoms, output.atoms, output.Rlist)
    actions.extend(
        (name, "differ", "derived data rebuilt") for name in ("Rlist", "distance_dict")
    )
    output._shell_source_format = document.source_format
    output._shell_precision_profile = document.precision_profile
    output._shell_site_labels = list(document.site_labels)
    output._shell_user_metadata = _thaw(document.metadata)
    messages = []
    if output.charges is None:
        messages.append(
            "Charges are absent; charge-indexing targets require real context."
        )
    if output.spinat is None:
        messages.append(
            "Reference moments are absent; no dynamics-readiness is claimed."
        )
    if core.report.created_channel_keys:
        messages.append("Completed model channel keys are not calculated entries.")
    return ExpansionResult(
        spinio=output,
        report=replace(
            core.report,
            precision_profile=document.precision_profile,
            metadata_actions=tuple(actions),
            losses=tuple(losses),
            messages=tuple(messages),
        ),
    )


def _template_number(value, field, *, positive=False, nonnegative=False):
    number = float(_numbers(value, (), field).item())
    if positive and number <= 0:
        raise ValueError(f"{field}: expected a positive finite value")
    if nonnegative and number < 0:
        raise ValueError(f"{field}: expected a nonnegative finite value")
    return number


def _template_moments(reference_spins, natom, symmetry):
    if reference_spins is None:
        if symmetry == "msg":
            raise ValueError(
                "reference_spins.moments: required for msg symmetry; "
                "supply complete moments in atom order"
            )
        return None
    try:
        raw = np.asarray(reference_spins)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "reference_spins.moments: expected one scalar or Cartesian vector per atom"
        ) from error
    if raw.shape not in {(natom,), (natom, 3)}:
        raise ValueError(
            "reference_spins.moments: expected shape "
            f"({natom},) or ({natom}, 3), got {raw.shape}"
        )
    return _numbers(reference_spins, raw.shape, "reference_spins.moments")


def _template_structure(atoms):
    from ase import Atoms

    if not isinstance(atoms, Atoms):
        raise TypeError("atoms: expected an ASE Atoms structure")
    if len(atoms) == 0:
        raise ValueError("structure.species: expected at least one atom")
    if not np.all(np.asarray(atoms.pbc, dtype=bool)):
        raise ValueError("atoms.pbc: template generation requires a periodic 3D cell")
    cell = _numbers(atoms.get_cell().array, (3, 3), "structure.cell_angstrom")
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or abs(determinant) < np.finfo(float).tiny:
        raise ValueError("structure.cell_angstrom: expected a finite nonsingular cell")
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "structure.cell_angstrom: expected a finite nonsingular cell"
        ) from error
    if not np.isfinite(inverse).all():
        raise ValueError(
            "structure.cell_angstrom: reciprocal cell is nonfinite; "
            "supply a well-conditioned cell"
        )
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#site_frame_cartesian_bond_residual
    positions = _numbers(
        atoms.get_scaled_positions(wrap=False),
        (len(atoms), 3),
        "structure.positions_frac",
    )
    species = tuple(atoms.get_chemical_symbols())
    return cell, inverse, positions, species


def _template_selection(magnetic_atoms, natom):
    try:
        selection = tuple(magnetic_atoms)
    except TypeError as error:
        raise ValueError(
            "structure.magnetic_atoms: expected one-based integer atom indices"
        ) from error
    if any(
        isinstance(atom, (bool, np.bool_)) or not isinstance(atom, (int, np.integer))
        for atom in selection
    ):
        raise ValueError(
            "structure.magnetic_atoms: expected one-based integer atom indices "
            "without boolean or float coercion"
        )
    selection = tuple(int(atom) for atom in selection)
    if any(atom < 1 or atom > natom for atom in selection):
        raise ValueError(
            "structure.magnetic_atoms: indices must be between 1 and the atom count"
        )
    if len(set(selection)) != len(selection):
        raise ValueError("structure.magnetic_atoms: duplicate atom indices")
    return tuple(sorted(selection))


# MathDoc: tb2j/spacegroup-exchange-symmetrization#primed_bilinear_energy_difference
def _template_operations(
    atoms,
    positions,
    cell,
    magnetic_atoms,
    symmetry,
    moments,
    symprec,
    magnetic_tol,
):
    if symmetry == "sg":
        detected = symmetry_core.crystal_symmetry_ops(atoms, symprec=symprec)
    else:
        detected = symmetry_core.magnetic_symmetry_ops(
            atoms,
            moments,
            symprec=symprec,
            magnetic_tol=magnetic_tol,
        )
    site_maps = symmetry_core._build_site_maps(positions, cell, detected, symprec)
    selected = np.zeros(len(positions), dtype=bool)
    selected[np.asarray(magnetic_atoms, dtype=int) - 1] = True
    # The stabilizer of the selected-site set remains a closed subgroup.
    preserving = [
        number
        for number, mapping in enumerate(site_maps)
        if np.array_equal(selected[np.asarray(mapping, dtype=int)], selected)
    ]
    if not preserving:
        raise ValueError(
            "symmetry.operations: no detected operation preserves "
            "the magnetic-atom selection"
        )
    primes = (
        np.zeros(len(detected), dtype=bool)
        if detected.time_reversals is None
        else np.asarray(detected.time_reversals, dtype=bool)
    )
    operations = symmetry_core.SymmetryOperations(
        np.asarray(detected.rotations)[preserving],
        np.asarray(detected.translations)[preserving],
        primes[preserving],
        detected.description,
    )
    maps = [site_maps[number] for number in preserving]
    return operations, maps, detected.description


def _template_candidates(cell, inverse_cell, positions, selection, cutoff):
    """Enumerate all lattice images inside the radius via reciprocal bounds."""
    with np.errstate(over="ignore", invalid="ignore"):
        reciprocal_bounds = cutoff * np.linalg.norm(inverse_cell, axis=0)
    if not np.isfinite(reciprocal_bounds).all():
        raise ValueError(
            "cutoff: reciprocal-bound image range is nonfinite for this cell"
        )
    keys = []
    zero = (0, 0, 0)
    for ia in selection:
        for ja in selection:
            displacement = positions[ja] - positions[ia]
            ranges = []
            for axis, bound in enumerate(reciprocal_bounds):
                lower = math.floor(-displacement[axis] - bound) - 1
                upper = math.ceil(-displacement[axis] + bound) + 1
                ranges.append(range(lower, upper + 1))
            for translation in product(*ranges):
                translation = tuple(int(value) for value in translation)
                if ia == ja and translation == zero:
                    continue
                vector = (displacement + translation) @ cell
                if math.hypot(*(float(component) for component in vector)) <= cutoff:
                    keys.append((ia, ja, translation))
    return tuple(sorted(set(keys)))


def _template_distance(pair, positions, cell):
    ia, ja, translation = _core_key(pair)
    displacement = positions[ja] - positions[ia] + np.asarray(translation, dtype=float)
    vector = displacement @ cell
    return math.hypot(*(float(component) for component in vector))


# MathDoc: tb2j/spacegroup-exchange-symmetrization#bond_map_image_shift
# MathDoc: tb2j/spacegroup-exchange-symmetrization#hexagonal_cartesian_rotation
def template(
    atoms,
    magnetic_atoms,
    cutoff,
    *,
    symmetry="sg",
    reference_spins=None,
    mode="isotropic",
    symprec=1e-5,
    magnetic_tol=1e-5,
    energy_unit="meV",
    zero_tol=1e-10,
    value_atol=1e-8,
    value_rtol=1e-6,
):
    """Build an unfilled full-orbit template; cutoff is always in Angstrom."""
    if not isinstance(symmetry, str) or symmetry not in {"sg", "msg"}:
        raise ValueError("symmetry: expected sg or msg")
    if not isinstance(mode, str) or mode not in {"isotropic", "tensor"}:
        raise ValueError("mode: expected isotropic or tensor")
    if not isinstance(energy_unit, str) or energy_unit not in {"meV", "eV"}:
        raise ValueError("energy_unit: expected meV or eV")
    cutoff = _template_number(cutoff, "cutoff", positive=True)
    symprec = _template_number(symprec, "symmetry.symprec", positive=True)
    if magnetic_tol is None:
        if symmetry == "msg":
            raise ValueError("symmetry.magnetic_tol: required for msg symmetry")
    else:
        magnetic_tol = _template_number(
            magnetic_tol, "symmetry.magnetic_tol", positive=True
        )
    zero_tol = _template_number(zero_tol, "tolerances.zero_tol", nonnegative=True)
    value_atol = _template_number(value_atol, "tolerances.value_atol", nonnegative=True)
    value_rtol = _template_number(value_rtol, "tolerances.value_rtol", nonnegative=True)

    cell, inverse_cell, positions, species = _template_structure(atoms)
    selection = _template_selection(magnetic_atoms, len(species))
    moments = _template_moments(reference_spins, len(species), symmetry)

    operations, site_maps, description = _template_operations(
        atoms,
        positions,
        cell,
        selection,
        symmetry,
        moments,
        symprec,
        magnetic_tol,
    )
    candidates = _template_candidates(
        cell, inverse_cell, positions, tuple(atom - 1 for atom in selection), cutoff
    )
    if candidates:
        # ASE and the shell document store row lattice vectors; pair_images
        # uses the column-basis convention fixed by the shared MathDoc.
        image_maps, _ = symmetry_core.pair_images(
            candidates, operations, site_maps, positions, cell.T
        )
    else:
        image_maps = []
    # The cutoff selects orbits; each emitted shell retains the full orbit.
    candidate_wire = {_wire_key(key) for key in candidates}
    remaining = set(candidates)
    orbits = {}
    while remaining:
        key = min(remaining)
        full_orbit = tuple(
            sorted({_wire_key(image_map[key]) for image_map in image_maps})
        )
        supported = candidate_wire.intersection(full_orbit)
        identifier = _pair_id(full_orbit[0])
        orbits[identifier] = full_orbit
        remaining.difference_update(_core_key(member) for member in supported)

    entries = []
    for identifier, full_orbit in sorted(orbits.items()):
        # The shortest member remains supported as the cutoff only grows.
        representative = min(
            full_orbit,
            key=lambda member: (_template_distance(member, positions, cell), member),
        )
        fields = {"Jiso": "unfilled"} if mode == "isotropic" else {"tensor": "unfilled"}
        entries.append(
            ShellEntry(
                id=identifier,
                representative=representative,
                n_equivalent_pairs=len(full_orbit),
                distance=_template_distance(representative, positions, cell),
                **fields,
            )
        )
    document_operations = tuple(
        ShellOperation(
            rotation=rotation,
            translation=translation,
            time_reversal=bool(primed),
        )
        for rotation, translation, primed in zip(
            operations.rotations, operations.translations, operations.time_reversals
        )
    )
    magnetic_atoms = selection
    index_by_atom = {atom: spin_index for spin_index, atom in enumerate(magnetic_atoms)}
    index_spin = tuple(
        index_by_atom.get(atom, -1) for atom in range(1, len(species) + 1)
    )
    producer = (
        f"{description}; {len(document_operations)} operations preserve magnetic_atoms"
    )
    return ShellDocument(
        cell_angstrom=cell,
        species=species,
        positions_frac=positions,
        magnetic_atoms=magnetic_atoms,
        index_spin=index_spin,
        operations=document_operations,
        shells=tuple(entries),
        exchange_mode=mode,
        energy_unit=energy_unit,
        symmetry_mode=symmetry,
        symprec=symprec,
        magnetic_tol=magnetic_tol,
        moments=moments,
        moment_source="explicit",
        producer=producer,
        value_atol=value_atol,
        value_rtol=value_rtol,
        zero_tol=zero_tol,
        source_format="template",
        cutoff=cutoff,
    )


def read_exchange(path, format, context=None, allow_loss=()):
    """Read explicit native pickle/text/XML with missing-context and loss guards.

    Pickle must be selected explicitly and must come from a trusted source.
    Text/XML precision profiles describe printed-source bounds, not symmetry tolerances.
    """
    from .native_readers import _read_exchange

    return _read_exchange(path, format, context=context, allow_loss=allow_loss)
