"""Editable exchange shells defined by an authoritative symmetry snapshot.

Cell vectors are rows, atom selectors are one-based, and exchange values in
Python are eV. The TOML codec alone converts the declared wire energy unit.
"""

import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from TB2J import symmetry_core

_ID = re.compile(r"([1-9][0-9]*):([1-9][0-9]*):(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\Z")


def _freeze(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value.item() if isinstance(value, np.generic) else value


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
    orbit_size: int | None = None
    support_members: tuple | None = None
    Jiso_members: tuple | None = None
    DMI_members: tuple | None = None
    Jani_members: tuple | None = None

    def __post_init__(self):
        for name in (
            "Jiso",
            "tensor",
            "representative",
            "support_members",
            "Jiso_members",
            "DMI_members",
            "Jani_members",
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
    schema_version: int = 1

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
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
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
    if shell.orbit_size is not None and (
        not _integer(shell.orbit_size) or shell.orbit_size != size
    ):
        raise ValueError(
            f"shell {shell.id}.orbit_size: stored {shell.orbit_size}, computed {size}"
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
    positions = _numbers(
        document.positions_frac, (count, 3), "structure.positions_frac"
    )
    if np.any(positions < 0) or np.any(positions >= 1):
        raise ValueError(
            "structure.positions_frac: expected positions inside the fractional cell [0, 1)"
        )
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
    allowed = {
        "sia",
        "njt",
        "biquadratic",
        "orbital",
        "djdx",
        "damping",
        "gyromag",
        "charges",
    }
    if any(
        not isinstance(name, str) or name not in allowed
        for name in document.required_base_fields
    ) or len(set(document.required_base_fields)) != len(document.required_base_fields):
        raise ValueError(
            "source.required_base_fields: expected unique supported physical field names"
        )


def validate_shells(document, *, for_export=False, check_producer_ops=False):
    """Validate the authored document without replacing its operation snapshot."""
    if not _integer(document.schema_version) or document.schema_version != 1:
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
