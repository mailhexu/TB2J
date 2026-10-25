"""Strict native exchange import and explicit missing-context handling."""

import io
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as read_atoms
from ase.units import Bohr, J, eV

from TB2J.Jtensor import combine_J_tensor, decompose_J_tensor
from TB2J.spinham.spin_xml import SpinXmlParser

from .io_exchange import SpinIO
from .shells import ConversionReport, NativeContext, NativeReadResult, _numbers

# MathDoc: tb2j/spacegroup-exchange-symmetrization

_LOSS_NAMES = frozenset(
    {
        "njt",
        "sia",
        "sia_tensor",
        "biquadratic",
        "orbital",
        "djdx",
        "charges",
        "damping",
        "gyromag",
        "dmi",
        "jani",
    }
)


def _finite(raw, shape, name):
    return _numbers(raw, shape, name)


def _context_atoms(context):
    atoms = context.atoms
    if atoms is None:
        return None
    if isinstance(atoms, (str, Path)):
        atoms = read_atoms(atoms)
    if not isinstance(atoms, Atoms):
        raise TypeError("context.atoms: expected ASE Atoms or a structure file")
    atoms = atoms.copy()
    cell = _finite(atoms.cell.array, (3, 3), "context.atoms.cell")
    if not np.isfinite(np.linalg.det(cell)) or np.linalg.det(cell) == 0:
        raise ValueError("context.atoms.cell: expected a nonsingular periodic cell")
    _finite(atoms.positions, (len(atoms), 3), "context.atoms.positions")
    return atoms


def _checked_indices(raw, natom, name):
    try:
        values = tuple(raw)
    except TypeError as error:
        raise ValueError(f"{name}: expected an integer index per atom") from error
    if len(values) != natom or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"{name}: expected an integer index per atom")
    values = [int(value) for value in values]
    magnetic = sorted(value for value in values if value >= 0)
    if magnetic != list(range(len(magnetic))):
        raise ValueError(
            f"{name}: magnetic indices must be unique and contiguous from zero"
        )
    return values


def _native_indices(natom, context, *, native=None, observed_atoms=None):
    supplied = None
    if context.index_spin is not None:
        supplied = _checked_indices(context.index_spin, natom, "context.index_spin")
    selection = None
    if context.magnetic_atoms is not None:
        from .shells import _template_selection

        selection = tuple(
            atom - 1 for atom in _template_selection(context.magnetic_atoms, natom)
        )
    if native is not None:
        indices = _checked_indices(native, natom, "native index_spin")
        if supplied is not None and supplied != indices:
            raise ValueError(
                "context.index_spin conflicts with the native exact index map"
            )
    elif supplied is not None:
        indices = supplied
    else:
        if selection is None:
            if observed_atoms is None or set(observed_atoms) != set(range(natom)):
                raise ValueError(
                    "ambiguous magnetic membership: supply NativeContext(index_spin=...) "
                    "or magnetic_atoms with the full magnetic-site selection"
                )
            selection = tuple(range(natom))
        by_atom = {atom: spin for spin, atom in enumerate(selection)}
        indices = [by_atom.get(atom, -1) for atom in range(natom)]
    if selection is not None and set(selection) != {
        atom for atom, spin in enumerate(indices) if spin >= 0
    }:
        raise ValueError("context.magnetic_atoms conflicts with index_spin")
    if observed_atoms is not None and any(
        not isinstance(atom, (int, np.integer))
        or atom < 0
        or atom >= natom
        or indices[atom] < 0
        for atom in observed_atoms
    ):
        raise ValueError("exchange pair endpoint conflicts with magnetic membership")
    return indices


def _context_moments(context, natom):
    if context.reference_spins is None:
        return None, None
    raw = np.asarray(context.reference_spins)
    if raw.shape == (natom,):
        values = _finite(context.reference_spins, (natom,), "context.reference_spins")
        native = np.zeros((natom, 3))
        native[:, 2] = values
        return native, True
    return _finite(
        context.reference_spins, (natom, 3), "context.reference_spins"
    ), False


def _context_charges(context, natom):
    if context.charges is None:
        return None
    return _finite(context.charges, (natom,), "context.charges")


def _compare_context_structure(
    atoms, context, cell_bound, position_bound, *, identity_required=False
):
    supplied = _context_atoms(context)
    species = context.species
    if species is not None:
        if isinstance(species, str):
            raise ValueError("context.species: supply symbols in full atom order")
        species = tuple(species)
        if len(species) != len(atoms):
            raise ValueError("context.species: atom count differs from the native file")
        # ASE validates symbols; masses never determine identity.
        Atoms(symbols=species)
    if supplied is not None:
        if len(supplied) != len(atoms):
            raise ValueError(
                "context structure atom count differs from the native file"
            )
        if np.any(
            np.abs(supplied.cell.array - atoms.cell.array) > np.asarray(cell_bound)
        ):
            raise ValueError(
                "context structure cell conflicts with native printed precision"
            )
        if np.any(
            np.abs(supplied.positions - atoms.positions) > np.asarray(position_bound)
        ):
            raise ValueError(
                "context structure positions conflict with native printed precision"
            )
        if species is not None and tuple(supplied.get_chemical_symbols()) != species:
            raise ValueError("context.species conflicts with context.atoms identity")
        species = tuple(supplied.get_chemical_symbols())
    if identity_required and species is None:
        raise ValueError(
            "native XML lacks species identity: supply NativeContext(atoms=structure) "
            "or species in atom order; masses cannot determine species"
        )
    original_species = tuple(atoms.get_chemical_symbols())
    if not identity_required and species is not None and species != original_species:
        raise ValueError("context species/order conflicts with native atom labels")
    result = atoms.copy() if supplied is None else supplied
    if species is not None:
        result.set_chemical_symbols(species)
    result.pbc = True
    return result


def _make_native(
    atoms,
    index_spin,
    *,
    spinat=None,
    colinear=False,
    charges=None,
    exchange=None,
    dmi=None,
    jani=None,
    damping=None,
    gyro_ratio=None,
    sia_tensor=None,
):
    from .shells import _transform_native_key

    natom = len(atoms)
    indices = _checked_indices(index_spin, natom, "native index_spin")
    spin_to_atom = {spin: atom for atom, spin in enumerate(indices) if spin >= 0}
    channels = {}
    for name, values, shape in (
        ("exchange_Jdict", exchange, ()),
        ("dmi_ddict", dmi, (3,)),
        ("Jani_dict", jani, (3, 3)),
    ):
        channel = {}
        for key, value in () if values is None else values.items():
            normalized, _ = _transform_native_key(name, key, spin_to_atom)
            checked = _finite(value, shape, f"native {name} key {key!r}")
            channel[normalized] = float(checked) if shape == () else checked.copy()
        channels[name] = channel or None
    if spinat is not None:
        spinat = _finite(spinat, (natom, 3), "native spinat")
    if charges is not None:
        charges = _finite(charges, (natom,), "native charges")
    if damping is not None:
        damping = _finite(damping, (natom,), "native damping")
    if gyro_ratio is not None:
        gyro_ratio = _finite(gyro_ratio, (natom,), "native gyro_ratio")
    result = SpinIO(
        atoms,
        spinat=spinat,
        charges=charges,
        index_spin=indices,
        colinear=colinear,
        damping=damping,
        gyro_ratio=gyro_ratio,
        sia_tensor=sia_tensor,
        **channels,
    )
    result.damping = damping
    result.gyro_ratio = gyro_ratio
    result._build_Rlist()
    return result


def _guard_losses(required, allow_loss):
    rows = tuple(required)
    missing = sorted({token for token, detail in rows} - set(allow_loss))
    if missing:
        raise ValueError(
            "unsupported native input blocks require named allow_loss: "
            + ", ".join(missing)
        )
    return rows


def _native_result(
    spinio,
    format,
    precision_profile,
    *,
    quanta=(),
    losses=(),
    messages=(),
    encoding_changes=(),
):
    spinio._shell_source_format = format
    spinio._shell_precision_profile = precision_profile
    return NativeReadResult(
        spinio=spinio,
        report=ConversionReport(
            precision_profile=precision_profile,
            quanta=quanta,
            losses=losses,
            messages=messages,
            encoding_changes=encoding_changes,
        ),
    )


def _read_pickle(path, context, allow_loss):
    """Read only explicitly trusted pickle input; never a sniffing fallback."""
    path = Path(path)
    result = SpinIO.load_pickle(path=path.parent, fname=path.name)
    natom = len(result.atoms)
    _native_indices(natom, context, native=result.index_spin)
    _compare_context_structure(result.atoms, context, 0.0, 0.0)
    moments, colinear = _context_moments(context, natom)
    if moments is not None:
        if result.spinat is not None and not np.array_equal(moments, result.spinat):
            raise ValueError(
                "context.reference_spins conflicts with exact pickle moments"
            )
        result.spinat = moments
        result.colinear = colinear
        if colinear:
            result.magmoms = moments[:, 2].copy()
    charges = _context_charges(context, natom)
    if charges is not None:
        if result.charges is not None and not np.array_equal(charges, result.charges):
            raise ValueError("context.charges conflicts with exact pickle charges")
        result.charges = charges
    return _native_result(
        result,
        "pickle",
        "exact",
        messages=(
            "Trusted pickle input was explicitly selected; no dynamics-readiness claim.",
        ),
    )


_XML_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_XML_COMMENT = re.compile(rb"<!--.*?-->", re.DOTALL)
_XML_START_TAG = re.compile(rb"<([A-Za-z_][A-Za-z0-9_.:-]*)\b([^<>]*)>")
_XML_ROOT_CLOSE = re.compile(rb"</System_definition\s*>")
_XML_EXCHANGE_START = re.compile(rb"<\s*spin_exchange_list\b")
_XML_EXPECTED_UNITS = {
    "unit_cell": "bohrradius",
    "position": "bohrradius",
    "spin_exchange_list": "eV",
    "spin_exchange_term": "eV",
    "spin_DMI_list": "eV",
    "spin_DMI_term": "eV",
    "spin_bilinear_list": "eV",
    "spin_bilinear_term": "eV",
    "spin_sia_tensor_list": "eV",
    "spin_sia_tensor_term": "eV",
    "spin_uniaxial_SIA_list": "eV",
    "spin_uniaxial_SIA_term": "eV",
    "data": "eV",
    "amplitude": "eV",
}
# Parser-recognized and known extension identifiers only; this is a tag-name
# list for loss classification, not a structural XML grammar.
_XML_HANDLED_TAGS = frozenset(
    {
        "System_definition",
        "unit_cell",
        "atom",
        "position",
        "spinat",
        "spin_exchange_list",
        "spin_exchange_term",
        "spin_DMI_list",
        "spin_DMI_term",
        "spin_bilinear_list",
        "spin_bilinear_term",
        "spin_sia_tensor_list",
        "spin_sia_tensor_term",
        "spin_uniaxial_SIA_list",
        "spin_uniaxial_SIA_term",
        "nterms",
        "ijR",
        "data",
        "i",
        "amplitude",
        "direction",
        "description",
    }
)
_XML_UNSUPPORTED_TOKEN_HINTS = (
    ("biquadratic", "biquadratic"),
    ("orbital", "orbital"),
    ("djd", "djdx"),
    ("njt", "njt"),
    ("charge", "charges"),
    ("damping", "damping"),
    ("gyro", "gyromag"),
    ("dmi", "dmi"),
    ("jani", "jani"),
    ("sia", "sia"),
)


def _xml_without_comments(source):
    return _XML_COMMENT.sub(b"", source)


def _xml_start_records(source):
    """Return tag identifiers and attribute text, without interpreting fields."""
    clean = _xml_without_comments(source)
    return tuple(
        (match.group(1).decode("ascii"), match.group(2))
        for match in _XML_START_TAG.finditer(clean)
    )


def _xml_attribute(attributes, name):
    encoded_name = re.escape(name.encode("ascii"))
    match = re.search(
        rb"(?:^|\s)" + encoded_name + rb"\s*=\s*([\"'])(.*?)\1",
        attributes,
        re.DOTALL,
    )
    return None if match is None else match.group(2).decode("ascii").strip()


def _xml_leaf_tokens(source, tag, field):
    """Read raw numeric tokens solely to determine their printed precision."""
    encoded_tag = re.escape(tag.encode("ascii"))
    pattern = re.compile(
        rb"<" + encoded_tag + rb"\b[^>]*>([^<]*)</" + encoded_tag + rb"\s*>",
        re.DOTALL,
    )
    groups = []
    for match in pattern.finditer(_xml_without_comments(source)):
        try:
            tokens = match.group(1).decode("ascii").split()
        except UnicodeDecodeError as error:
            raise ValueError(f"XML {field}: expected ASCII numeric tokens") from error
        if any(_XML_NUMBER.fullmatch(token) is None for token in tokens):
            raise ValueError(f"XML {field}: expected only numeric tokens")
        groups.append(tokens)
    return groups


def _xml_decimal_bound(token, scale=1.0, *, energy=False):
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(
            f"XML numeric token {token!r} has invalid decimal precision"
        ) from error
    if not value.is_finite():
        raise ValueError(f"XML numeric token {token!r} is nonfinite")
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    bound = float(quantum) * scale / 2.0
    if not np.isfinite(bound) or bound == 0.0:
        raise ValueError(f"XML precision quantum for {token!r} is outside float range")
    if energy:
        # Validate the sole parser's eV -> Joule -> eV precision path without
        # replacing its physical values or decoding another channel grammar.
        restored = (float(value) * eV / J) * (J / eV)
        if (
            not np.isfinite(restored)
            or abs(Decimal.from_float(restored) - value) > quantum / 2
        ):
            raise ValueError(
                f"XML energy token {token!r}: shared parser conversion cannot "
                "preserve the declared precision within float range"
            )
    return bound


def _xml_atom_metadata(source):
    """Inspect only printed atom-attribute presence and precision tokens."""
    result = []
    for tag, attributes in _xml_start_records(source):
        if tag != "atom":
            continue
        record = {}
        for name in ("mass", "damping_factor", "gyroratio"):
            record[name] = _xml_attribute(attributes, name)
        record["index_spin_present"] = (
            _xml_attribute(attributes, "index_spin") is not None
        )
        result.append(record)
    return tuple(result)


def _xml_precision_rows(source, natom, atom_metadata):
    """Build half-quantum bounds from raw tokens; values come from parser."""
    rows = []
    bounds = {}

    def add_row(field, tokens, unit, scale=1.0, shape=None):
        values = [
            _xml_decimal_bound(token, scale, energy=unit == "eV") for token in tokens
        ]
        if not values:
            return None
        result = np.asarray(values, dtype=float)
        if shape is not None:
            if result.size != int(np.prod(shape)):
                raise ValueError(
                    f"XML {field}: printed precision does not match shape {shape}"
                )
            result = result.reshape(shape)
        rows.append((field, result, unit))
        return result

    cell_groups = _xml_leaf_tokens(source, "unit_cell", "cell")
    if not cell_groups:
        raise ValueError("XML unit_cell is missing")
    bounds["cell"] = add_row(
        "cell", cell_groups[0], "angstrom", scale=Bohr, shape=(3, 3)
    )

    position_groups = _xml_leaf_tokens(source, "position", "positions")
    moment_groups = _xml_leaf_tokens(source, "spinat", "moments")
    if len(position_groups) != natom or len(moment_groups) != natom:
        raise ValueError(
            "XML position/spinat token counts differ from parser atom count"
        )
    position_tokens = [token for group in position_groups for token in group]
    moment_tokens = [token for group in moment_groups for token in group]
    bounds["positions"] = add_row(
        "positions",
        position_tokens,
        "angstrom",
        scale=Bohr,
        shape=(natom, 3),
    )
    bounds["moments"] = add_row("moments", moment_tokens, "mu_B", shape=(natom, 3))

    if len(atom_metadata) != natom:
        raise ValueError("XML atom tag count differs from parser atom count")
    for attribute, field, unit in (
        ("mass", "masses", "amu"),
        ("damping_factor", "damping", "dimensionless"),
        ("gyroratio", "gyro_ratio", "dimensionless"),
    ):
        tokens = [
            record[attribute]
            for record in atom_metadata
            if record[attribute] is not None
        ]
        add_row(field, tokens, unit)

    data_tokens = [
        token
        for group in _xml_leaf_tokens(source, "data", "energy data")
        for token in group
    ]
    add_row("energy_data", data_tokens, "eV")
    for tag, field, unit in (
        ("amplitude", "uniaxial_sia_amplitude", "eV"),
        ("direction", "uniaxial_sia_direction", "dimensionless"),
    ):
        tokens = [
            token for group in _xml_leaf_tokens(source, tag, field) for token in group
        ]
        add_row(field, tokens, unit)
    return tuple(rows), bounds


def _xml_loss_token(tag):
    normalized = tag.lower().replace("_", "")
    for marker, token in _XML_UNSUPPORTED_TOKEN_HINTS:
        if marker.replace("_", "") in normalized:
            return token
    return None


def _xml_metadata(source):
    """Validate declared units and classify unhandled physical tag identifiers."""
    records = _xml_start_records(source)
    tag_names = {tag for tag, _ in records}
    losses = []
    singleton_lists = {
        "spin_exchange_list",
        "spin_DMI_list",
        "spin_bilinear_list",
        "spin_sia_tensor_list",
    }
    seen_lists = set()
    uniaxial_tags = {"spin_uniaxial_SIA_list", "spin_uniaxial_SIA_term"}
    if tag_names & uniaxial_tags:
        losses.append(("sia", "XML uniaxial SIA block"))

    for tag, attributes in records:
        if tag in singleton_lists:
            if tag in seen_lists:
                raise ValueError(
                    f"XML repeated channel list <{tag}> is not consumed by the shared parser"
                )
            seen_lists.add(tag)
        if tag == "atom" and _xml_attribute(attributes, "charge") is not None:
            losses.append(("charges", "XML atom charge attribute"))
        units = _xml_attribute(attributes, "units")
        if units is not None:
            expected = _XML_EXPECTED_UNITS.get(tag)
            if expected is not None and units != expected:
                raise ValueError(f"XML <{tag}> declares unsupported units={units!r}")
            if expected is None and tag in _XML_HANDLED_TAGS:
                raise ValueError(f"XML <{tag}> declares units without a convention")
        massunits = _xml_attribute(attributes, "massunits")
        if massunits is not None and (tag != "atom" or massunits != "atomicmassunit"):
            raise ValueError(
                f"XML <{tag}> declares unsupported massunits={massunits!r}"
            )

    for tag in sorted(tag_names - _XML_HANDLED_TAGS - uniaxial_tags):
        token = _xml_loss_token(tag)
        if token is None:
            raise ValueError(f"unsupported meaningful XML block <{tag}>")
        losses.append((token, f"XML block <{tag}>"))
    return tuple(dict.fromkeys(losses))


def _xml_parser_source(path):
    """Supply only the parser-required empty list missing in DMI-only XML."""
    source = Path(path).read_bytes()
    scan_source = _xml_without_comments(source)
    if _XML_EXCHANGE_START.search(scan_source):
        return source, False
    normalized, count = _XML_ROOT_CLOSE.subn(
        b'<spin_exchange_list units="eV"><nterms>0</nterms></spin_exchange_list>\n'
        b"\\g<0>",
        scan_source,
        count=1,
    )
    return normalized, count == 1


def _read_xml(path, context, allow_loss):
    """Read native SpinXml data without inferring species from atom masses."""
    source, inserted_empty_exchange = _xml_parser_source(path)
    # SpinXmlParser is the sole semantic XML parser: its raw channel dictionaries,
    # values, and nterms assertions remain authoritative.
    parser = SpinXmlParser(io.BytesIO(source))
    losses = _guard_losses(_xml_metadata(source), allow_loss)

    natom = len(parser.positions)
    if natom == 0:
        raise ValueError("XML contains no atoms")
    positions = _finite(parser.positions, (natom, 3), "XML positions")
    cell = _finite(parser.cell, (3, 3), "XML cell")
    moments_printed = _finite(parser.spinat, (natom, 3), "XML spinat")
    masses_printed = _finite(parser.masses, (natom,), "XML masses")
    if not np.isfinite(np.linalg.det(cell)) or np.linalg.det(cell) == 0:
        raise ValueError("XML cell must be finite and nonsingular")

    atom_metadata = _xml_atom_metadata(source)
    quanta, precision_bounds = _xml_precision_rows(source, natom, atom_metadata)
    cell_bound = precision_bounds["cell"]
    position_bound = precision_bounds["positions"]
    moment_bound = precision_bounds["moments"]
    atoms = Atoms(
        symbols=["X"] * natom,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    atoms = _compare_context_structure(
        atoms,
        context,
        cell_bound,
        position_bound,
        identity_required=True,
    )
    # The parser substitutes mass=1 when the attribute is absent. Assign only
    # explicit source masses; all actual mass values come from parser.masses.
    explicit_mass = np.asarray(
        [record["mass"] is not None for record in atom_metadata], dtype=bool
    )
    if np.any(explicit_mass):
        masses = atoms.get_masses()
        masses[explicit_mass] = masses_printed[explicit_mass]
        atoms.set_masses(_finite(masses, (natom,), "XML atom masses"))

    raw_index_spin = tuple(parser.index_spin)
    if len(raw_index_spin) != natom or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in raw_index_spin
    ):
        raise ValueError("XML index_spin values are malformed")
    index_presence = [record["index_spin_present"] for record in atom_metadata]
    if all(index_presence):
        # io_multibinit writes index_spin + 1 for every atom. Reverse the
        # offset for every explicit integer, including negative sentinels.
        candidate_index_spin = [value - 1 for value in raw_index_spin]
    else:
        # SpinXmlParser's -1 fallback is not evidence that an index was
        # explicitly present. Missing or partial maps require exact context.
        candidate_index_spin = _native_indices(natom, context)
        for atom, present in enumerate(index_presence):
            if present and raw_index_spin[atom] - 1 != candidate_index_spin[atom]:
                raise ValueError(
                    "context.index_spin conflicts with an explicitly printed "
                    f"XML index_spin for atom {atom + 1}"
                )

    def native_key(key, field):
        if (
            not isinstance(key, tuple)
            or len(key) != 3
            or not isinstance(key[0], (int, np.integer))
            or not isinstance(key[1], (int, np.integer))
            or not isinstance(key[2], tuple)
            or len(key[2]) != 3
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                for value in (*key[:2], *key[2])
            )
        ):
            raise ValueError(f"XML {field}: malformed exchange key {key!r}")
        return (tuple(int(value) for value in key[2]), int(key[0]), int(key[1]))

    pair_keys = set(parser._exchange) | set(parser._dmi) | set(parser._bilinear)
    referenced_spins = {endpoint for key in pair_keys for endpoint in key[:2]}
    spin_to_atom = {
        spin: atom for atom, spin in enumerate(candidate_index_spin) if spin >= 0
    }
    if not referenced_spins.issubset(spin_to_atom):
        bad = sorted(referenced_spins - set(spin_to_atom))
        raise ValueError(f"XML exchange endpoint references unknown spin indices {bad}")
    observed_atoms = tuple(spin_to_atom[spin] for spin in sorted(referenced_spins))
    index_spin = _native_indices(
        natom,
        context,
        native=candidate_index_spin,
        observed_atoms=observed_atoms,
    )

    energy_to_ev = J / eV
    exchange = {}
    jani = {}
    encoding_changes = []
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
    # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_anisotropic
    for key, raw in parser._exchange.items():
        native = native_key(key, "exchange")
        diagonal = _finite(raw, (3,), f"XML exchange {key!r}") * energy_to_ev
        if diagonal[0] == diagonal[1] == diagonal[2]:
            exchange[native] = float(diagonal[0])
        else:
            jiso, _, jani_diagonal = decompose_J_tensor(np.diag(diagonal))
            exchange[native] = float(jiso)
            jani[native] = _finite(
                jani_diagonal, (3, 3), f"XML exchange anisotropy {key!r}"
            )
            encoding_changes.append(
                (
                    "unequal_xml_exchange_diagonal",
                    native,
                    "preserved exactly as Jiso plus traceless Jani",
                )
            )

    dmi = {}
    for key, raw in parser._dmi.items():
        native = native_key(key, "DMI")
        dmi[native] = _finite(raw, (3,), f"XML DMI {key!r}") * energy_to_ev

    for key, raw in parser._bilinear.items():
        native = native_key(key, "bilinear")
        value = _finite(raw, (3, 3), f"XML bilinear {key!r}") * energy_to_ev
        if native in jani:
            jani[native] = _finite(
                jani[native] + value,
                (3, 3),
                f"XML combined Jani {key!r}",
            )
        else:
            jani[native] = value

    sia_tensor = {}
    nspin = len(spin_to_atom)
    for spin, raw in parser._sia_tensor.items():
        if (
            isinstance(spin, (bool, np.bool_))
            or not isinstance(spin, (int, np.integer))
            or spin < 0
            or spin >= nspin
        ):
            raise ValueError(f"XML SIA tensor references invalid spin index {spin!r}")
        value = _finite(raw, (3, 3), f"XML SIA tensor spin {spin}") * energy_to_ev
        sia_tensor[int(spin)] = _finite(
            value, (3, 3), f"XML SIA tensor spin {spin} in eV"
        )

    context_moments, colinear = _context_moments(context, natom)
    spinat = moments_printed
    if context_moments is not None:
        if np.any(np.abs(context_moments - moments_printed) > moment_bound):
            raise ValueError(
                "context.reference_spins conflicts with XML printed moments "
                "at their declared decimal precision"
            )
        spinat = context_moments
    else:
        colinear = False

    context_charges = _context_charges(context, natom)
    damping_present = [record["damping_factor"] is not None for record in atom_metadata]
    gyro_present = [record["gyroratio"] is not None for record in atom_metadata]
    if any(damping_present) and not all(damping_present):
        raise ValueError("XML damping_factor must be present for every atom or absent")
    if any(gyro_present) and not all(gyro_present):
        raise ValueError("XML gyroratio must be present for every atom or absent")
    damping = (
        _finite(parser.damping_factors, (natom,), "XML damping_factor")
        if all(damping_present)
        else None
    )
    gyro_ratio = (
        _finite(parser.gyro_ratios, (natom,), "XML gyroratio")
        if all(gyro_present)
        else None
    )

    native = _make_native(
        atoms,
        index_spin,
        spinat=spinat,
        colinear=colinear,
        charges=context_charges,
        exchange=exchange,
        dmi=dmi,
        jani=jani,
        damping=damping,
        gyro_ratio=gyro_ratio,
        sia_tensor=sia_tensor or None,
    )
    messages = [
        "XML contains no atomic symbols; species identity was supplied explicitly by NativeContext.",
    ]
    if inserted_empty_exchange:
        messages.append(
            "Added only the parser-required empty spin_exchange_list in memory; no exchange terms were synthesized."
        )
    return _native_result(
        native,
        "xml",
        "xml-%.5e",
        quanta=quanta,
        losses=losses,
        messages=tuple(messages),
        encoding_changes=tuple(encoding_changes),
    )


_TEXT_FLOAT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_TEXT_FIXED = re.compile(r"[+-]?\d+\.\d+\Z")
_TEXT_SYMBOL_NUMBER = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)\Z")
_TEXT_PAIR = re.compile(
    r"\s*(\S+)\s+(\S+)\s+\(\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*\)"
    r"\s+(\S+)\s+\(\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\)\s+(\S+)\s*\Z"
)
_TEXT_ARRAY_TOKEN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?[0-9]+)?\Z")
_TEXT_JISO = re.compile(r"J_iso:\s*(\S+)\s*\Z")
_TEXT_DMI = re.compile(r"\[Testing!\]\s+DMI:\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\Z")
_TEXT_NJT_DMI = re.compile(
    r"\[Experimental!\]\s+DMI_NJt:\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\Z"
)
_TEXT_NJT_JANI = re.compile(
    r"\[Testing!\]\s+Jani_NJt:\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\Z"
)
_TEXT_DMI2 = re.compile(r"\[Debug!\]\s+DMI2:\s*\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)\s*\Z")
_TEXT_DJDX = re.compile(r"dJ/dx:\s*(\S+)\s*\Z")
_TEXT_D2JDX2 = re.compile(r"d2J/dx2:\s*(\S+)\s*\Z")
_TEXT_BIQUADRATIC = re.compile(r"\[Testing!\]\s+Jprime:\s*(\S+)\s*,\s*B:\s*(\S+)\s*\Z")
_TEXT_SIA_SCALAR = re.compile(
    r"(\S+)\s+(\S+)\s+\(\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\)\s*\Z"
)
_TEXT_SCI = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE]([+-]?[0-9]+)\Z")


def _text_error(field, detail):
    raise ValueError(f"exchange.out {field}: {detail}")


def _text_decimal(token, field):
    if not isinstance(token, str) or not _TEXT_FLOAT.fullmatch(token):
        _text_error(
            field,
            f"invalid numeric token {token!r}; unknown printer regime, identify a supported declared precision profile/source context (accepted: exchange.out-default)",
        )
    try:
        value = Decimal(token)
    except InvalidOperation as error:
        raise ValueError(
            f"exchange.out {field}: invalid Decimal token {token!r}"
        ) from error
    if not value.is_finite():
        _text_error(field, f"non-finite numeric token {token!r}")
    converted = float(value)
    if not np.isfinite(converted):
        _text_error(field, f"numeric token {token!r} is outside finite float64 range")
    if value != 0 and converted == 0.0:
        _text_error(field, f"nonzero numeric token {token!r} underflows float64")
    return value


def _text_fixed_decimal(token, decimals, field):
    if not isinstance(token, str) or not _TEXT_FIXED.fullmatch(token):
        _text_error(
            field,
            f"token {token!r} does not match the fixed {decimals}-decimal exchange.out-default printer regime; identify a supported declared precision profile/source context",
        )
    digits = len(token.rsplit(".", 1)[1])
    if digits != decimals:
        _text_error(
            field,
            f"token {token!r} does not match the fixed {decimals}-decimal exchange.out-default printer regime; identify a supported declared precision profile/source context",
        )
    return _text_decimal(token, field)


def _text_line_values(line, count, field, decimals):
    tokens = line.strip().split()
    if len(tokens) != count:
        _text_error(field, f"expected {count} numeric fields, found {len(tokens)}")
    return tuple(_text_fixed_decimal(token, decimals, field) for token in tokens)


def _text_bound(quantum):
    return quantum / Decimal(2)


def _text_label(label, field):
    match = _TEXT_SYMBOL_NUMBER.fullmatch(label)
    if match is None:
        _text_error(field, f"invalid full-table symbol_number label {label!r}")
    return match.group(1), int(match.group(2))


def _text_parse_cell(lines, marker):
    cell = []
    for row in lines[marker + 1 : marker + 4]:
        cell.append(_text_line_values(row, 3, "Cell (Angstrom)", 8))
    if len(cell) != 3:
        _text_error("Cell (Angstrom)", "expected exactly three lattice rows")
    cell = _finite(np.asarray(cell, dtype=np.float64), (3, 3), "exchange.out cell")
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant == 0.0:
        _text_error("Cell (Angstrom)", "expected a nonsingular full cell")
    return cell


def _text_parse_atoms(lines, marker, cell):
    header_index = None
    for index in range(marker + 1, len(lines)):
        line = lines[index]
        if "w_charge" in line and (
            "w_magmom" in line or all(name in line for name in ("M(x)", "M(y)", "M(z)"))
        ):
            header_index = index
            break
        if (
            not line.strip()
            or line.strip()
            == "(Note: charge and magmoms only count the wannier functions.)"
        ):
            continue
        _text_error("Atoms", f"unrecognized atom-table preamble {line.strip()!r}")
    if header_index is None:
        _text_error("Atoms", "missing complete w_charge/moment atom-table header")

    header = lines[header_index]
    header_tokens = header.split()
    vector_mode = header_tokens == [
        "Atom_number",
        "x",
        "y",
        "z",
        "w_charge",
        "M(x)",
        "M(y)",
        "M(z)",
    ]
    scalar_mode = header_tokens == [
        "Atom",
        "number",
        "x",
        "y",
        "z",
        "w_charge",
        "w_magmom",
    ]
    if not (scalar_mode or vector_mode):
        _text_error("Atoms", "unknown atom-table columns")

    symbols = []
    labels = []
    positions = []
    charges = []
    charge_decimals = []
    moments = []
    moment_decimals = []
    species_counts = {}
    total = None
    total_decimals = None
    row_index = header_index + 1
    expected_width = 6 if scalar_mode else 8
    while row_index < len(lines):
        tokens = lines[row_index].split()
        if not tokens:
            row_index += 1
            continue
        if tokens[0] == "Total":
            expected_total_width = 3 if scalar_mode else 5
            if len(tokens) != expected_total_width:
                _text_error(
                    "Atoms.Total", f"expected {expected_total_width - 1} total fields"
                )
            total_decimals = tuple(
                _text_fixed_decimal(value, 4, "Atoms.Total") for value in tokens[1:]
            )
            total = tuple(float(value) for value in total_decimals)
            row_index += 1
            break
        if len(tokens) != expected_width:
            _text_error(
                "Atoms",
                f"atom row {lines[row_index]!r} has {len(tokens)} fields, expected {expected_width}",
            )
        label = tokens[0]
        symbol, sequence = _text_label(label, "Atoms label")
        species_counts[symbol] = species_counts.get(symbol, 0) + 1
        if sequence != species_counts[symbol]:
            _text_error(
                "Atoms label",
                f"{label!r} violates per-species full-atom-order symbol_number numbering",
            )
        if label in labels:
            _text_error("Atoms label", f"duplicate numbered label {label!r}")
        coordinate_decimals = tuple(
            _text_fixed_decimal(value, 8, f"Atoms.{label}.position")
            for value in tokens[1:4]
        )
        charge_decimal = _text_fixed_decimal(tokens[4], 4, f"Atoms.{label}.charge")
        if scalar_mode:
            moment_decimal = (
                _text_fixed_decimal(tokens[5], 4, f"Atoms.{label}.w_magmom"),
            )
        else:
            moment_decimal = tuple(
                _text_fixed_decimal(value, 4, f"Atoms.{label}.moment")
                for value in tokens[5:8]
            )
        labels.append(label)
        symbols.append(symbol)
        positions.append([float(value) for value in coordinate_decimals])
        charges.append(float(charge_decimal))
        charge_decimals.append(charge_decimal)
        if scalar_mode:
            moments.append([0.0, 0.0, float(moment_decimal[0])])
            moment_decimals.append(moment_decimal[0])
        else:
            moments.append([float(value) for value in moment_decimal])
            moment_decimals.append(moment_decimal)
        row_index += 1

    if total is None or not labels:
        _text_error(
            "Atoms", "expected a nonempty numbered atom table and one Total row"
        )
    charges_array = _finite(
        np.asarray(charges, dtype=np.float64), (len(labels),), "exchange.out charges"
    )
    spinat = _finite(
        np.asarray(moments, dtype=np.float64), (len(labels), 3), "exchange.out moments"
    )
    positions = _finite(
        np.asarray(positions, dtype=np.float64),
        (len(labels), 3),
        "exchange.out positions",
    )
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    # The writer sums the unrounded source values for Total, so allow one
    # half-place for each printed atom and one for the printed aggregate.
    aggregate_bound = _text_bound(Decimal("0.0001")) * (len(labels) + 1)
    printed_components = [sum(charge_decimals, Decimal(0))]
    if scalar_mode:
        printed_components.append(sum(moment_decimals, Decimal(0)))
    else:
        printed_components.extend(
            sum((values[component] for values in moment_decimals), Decimal(0))
            for component in range(3)
        )
    for field_index, (printed_total, summed_rows) in enumerate(
        zip(total_decimals, printed_components)
    ):
        if abs(printed_total - summed_rows) > aggregate_bound:
            _text_error(
                "Atoms.Total",
                f"aggregate field {field_index} conflicts with the atom rows at printed precision",
            )

    return {
        "atoms": atoms,
        "labels": tuple(labels),
        "label_to_atom": {label: atom for atom, label in enumerate(labels)},
        "charges": charges_array,
        "spinat": spinat,
        "scalar_mode": scalar_mode,
        "charge_decimals": tuple(charge_decimals),
        "moment_decimals": tuple(moment_decimals),
        "next_line": row_index,
    }


def _text_matrix(lines, first_row, field):
    rows = lines[first_row : first_row + 3]
    if len(rows) != 3:
        _text_error(field, "expected a three-row 3x3 matrix")
    stripped = [row.strip() for row in rows]
    if not stripped[0].startswith("[[") or not stripped[-1].endswith("]]"):
        _text_error(field, "malformed NumPy 3x3 matrix brackets")
    if any(row.count("[") != 1 + (index == 0) for index, row in enumerate(stripped)):
        _text_error(field, "malformed NumPy 3x3 matrix row brackets")
    if any(row.count("]") != 1 + (index == 2) for index, row in enumerate(stripped)):
        _text_error(field, "malformed NumPy 3x3 matrix row brackets")

    token_rows = []
    for row in stripped:
        tokens = row.replace("[", " ").replace("]", " ").split()
        if len(tokens) != 3:
            _text_error(
                field, "each NumPy matrix row must contain exactly three values"
            )
        token_rows.append(tokens)
    tokens = [token for row in token_rows for token in row]
    scientific = ["e" in token.lower() for token in tokens]
    if any(scientific) and not all(scientific):
        _text_error(
            field,
            "mixed fixed/scientific NumPy matrix tokens are an unknown printer regime; identify a supported declared precision profile/source context",
        )

    decimals = []
    bounds = []
    if all(scientific):
        for token in tokens:
            match = _TEXT_SCI.fullmatch(token)
            if match is None:
                _text_error(field, f"invalid scientific NumPy matrix token {token!r}")
            mantissa = match.group(1)
            fractional_digits = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
            if fractional_digits > 3:
                _text_error(
                    field,
                    f"scientific matrix token {token!r} exceeds the NumPy precision=3 regime; identify a supported declared precision profile/source context",
                )
            exponent = int(match.group(2))
            mantissa_value = Decimal(mantissa)
            if (
                exponent < -324
                or exponent > 308
                or (
                    mantissa_value != 0
                    and not (Decimal(1) <= abs(mantissa_value) < Decimal(10))
                )
                or (mantissa_value == 0 and exponent != 0)
            ):
                _text_error(
                    field,
                    f"scientific matrix token {token!r} is outside the float64 NumPy printer regime; identify a supported declared precision profile/source context",
                )
            value = _text_decimal(token, field)
            quantum_mev = Decimal(1).scaleb(exponent - fractional_digits)
            decimals.append(value)
            bounds.append(_text_bound(quantum_mev) / Decimal(1000))
    else:
        for token in tokens:
            if "." not in token or not re.fullmatch(r"[+-]?\d+\.\d{0,3}", token):
                _text_error(
                    field,
                    f"fixed matrix token {token!r} is outside the NumPy precision=3/suppressed-small regime; identify a supported declared precision profile/source context",
                )
            value = _text_decimal(token, field)
            decimals.append(value)
            bounds.append(Decimal("0.0005") / Decimal(1000))

    # Convert the printed decimal unit before the one float rounding, as the
    # shell codec does; a meV float followed by multiplication rounds twice.
    scaled = [float(value / Decimal(1000)) for value in decimals]
    matrix_ev = _finite(np.asarray(scaled).reshape(3, 3), (3, 3), field)
    if any(
        value != 0 and converted == 0.0 for value, converted in zip(decimals, scaled)
    ):
        _text_error(field, "nonzero meV matrix value underflows after conversion to eV")
    bound_ev = np.asarray(bounds, dtype=object).reshape(3, 3)
    return matrix_ev, bound_ev, tuple(decimals), first_row + 3


def _text_parenthesized_vector(match, field, decimals):
    values = tuple(
        _text_fixed_decimal(value, decimals, field) for value in match.groups()
    )
    return np.asarray([float(value) for value in values], dtype=np.float64), values


def _text_check_context_moments(context, atom_data):
    source = atom_data["spinat"]
    scalar_mode = atom_data["scalar_mode"]
    moment_values = atom_data["moment_decimals"]
    moments, context_colinear = _context_moments(context, len(source))
    half_quantum = 5e-5
    if scalar_mode:
        source_z = source[:, 2]
        nonzero = any(value != 0 for value in moment_values)
        unsigned_magnitudes = nonzero and all(value >= 0 for value in moment_values)
        if moments is None:
            if unsigned_magnitudes:
                _text_error(
                    "Atoms.w_magmom",
                    "nonnegative collinear values do not declare signed reference directions; supply NativeContext(reference_spins=...) with signed moments",
                )
            return source.copy(), True

        if unsigned_magnitudes:
            if context_colinear:
                context_magnitudes = np.abs(moments[:, 2])
            else:
                context_magnitudes = np.linalg.norm(moments, axis=1)
            if not np.all(
                np.abs(context_magnitudes - np.abs(source_z)) <= half_quantum
            ):
                _text_error(
                    "Atoms.w_magmom",
                    "NativeContext.reference_spins conflicts with printed magnitudes",
                )
            return moments.copy(), context_colinear

        if np.any(np.abs(moments[:, :2]) > half_quantum) or not np.all(
            np.abs(moments[:, 2] - source_z) <= half_quantum
        ):
            _text_error(
                "Atoms.w_magmom",
                "NativeContext.reference_spins conflicts with printed signed collinear moments",
            )
        return moments.copy(), context_colinear

    if moments is None:
        return source.copy(), False
    if context_colinear and np.any(np.abs(moments[:, :2]) > half_quantum):
        _text_error(
            "Atoms.moment",
            "collinear NativeContext.reference_spins conflict with vector source moments",
        )
    if not np.all(np.abs(moments - source) <= half_quantum):
        _text_error(
            "Atoms.moment",
            "NativeContext.reference_spins conflicts with printed vector moments",
        )
    return moments.copy(), context_colinear


def _text_parse_orbital_names(lines, index, stop, atom_labels, losses):
    rows = []
    seen = set()
    cursor = index + 1
    while cursor < stop:
        line = lines[cursor].strip()
        if not line or set(line) == {"="}:
            break
        if line.startswith("The name of the orbitals for the decomposition"):
            cursor += 1
            continue
        match = re.fullmatch(r"(\S+)\s*:\s*(.+)", line)
        if match is None or match.group(1) not in atom_labels:
            _text_error(
                "Orbitals used in decomposition",
                f"unrecognized orbital-name row {line!r}",
            )
        if match.group(1) in seen:
            _text_error(
                "Orbitals used in decomposition",
                f"duplicate orbital row for {match.group(1)}",
            )
        seen.add(match.group(1))
        rows.append(
            ("orbital", f"orbital names for {match.group(1)}: {match.group(2)}")
        )
        cursor += 1
    losses.extend(rows)
    return cursor


def _text_parse_sia_legacy(lines, index, stop, atom_labels, losses, quanta):
    cursor = index + 1
    header = None
    while cursor < stop:
        line = lines[cursor].strip()
        if (
            line.startswith(("Atom number", "Atom_number"))
            and "k1" in line
            and "k1dir" in line
        ):
            header = cursor
            break
        if (
            not line
            or line == "(k1 is the anisotropy constant, k1dir is the direction vector)"
        ):
            cursor += 1
            continue
        if set(line) == {"="} or line == "Exchange:":
            break
        _text_error(
            "Single Ion Anisotropy (meV)", f"unrecognized legacy SIA heading {line!r}"
        )
    if header is None:
        _text_error(
            "Single Ion Anisotropy (meV)", "missing legacy k1/k1dir table header"
        )
    cursor = header + 1
    found = 0
    seen = set()
    while cursor < stop:
        line = lines[cursor].strip()
        if not line or set(line) == {"-"} or set(line) == {"="}:
            if found:
                break
            cursor += 1
            continue
        match = _TEXT_SIA_SCALAR.fullmatch(line)
        if match is None:
            break
        label, k1, dx, dy, dz = match.groups()
        if label in seen:
            _text_error(
                "Single Ion Anisotropy (meV)", f"duplicate legacy SIA row for {label!r}"
            )
        for token in (k1, dx, dy, dz):
            _text_fixed_decimal(token, 4, "Single Ion Anisotropy (meV)")
        found += 1
        seen.add(label)
        losses.append(("sia", f"legacy k1/k1dir for {label}"))
        quanta.append((f"unsupported.sia[{label}]", Decimal("0.00005"), "meV"))
        cursor += 1
    if not found:
        _text_error(
            "Single Ion Anisotropy (meV)", "legacy SIA section contains no data"
        )
    return cursor


def _text_parse_sia_tensor(lines, index, stop, atom_labels, losses, quanta):
    cursor = index + 1
    first_data = None
    while cursor < stop:
        line = lines[cursor].strip()
        tokens = line.split()
        if tokens and tokens[0] in atom_labels:
            first_data = cursor
            break
        if set(line) == {"="} or line == "Exchange:":
            break
        if line and not any(header in line for header in ("A_xx", "A_yx", "A_zx")):
            _text_error(
                "Single Ion Anisotropy", f"unrecognized SIA tensor heading {line!r}"
            )
        cursor += 1
    if first_data is None:
        return cursor
    cursor = first_data
    seen = set()
    while cursor < stop:
        line = lines[cursor].strip()
        if not line or set(line) == {"-"}:
            cursor += 1
            continue
        if set(line) == {"="} or line == "Exchange:":
            break
        tokens = line.split()
        if not tokens:
            cursor += 1
            continue
        label = tokens[0]
        if label not in atom_labels or len(tokens) != 4 or label in seen:
            _text_error(
                "Single Ion Anisotropy",
                f"malformed or duplicate SIA tensor first row {line!r}",
            )
        matrix_rows = []
        matrix_rows.append(
            tuple(
                _text_fixed_decimal(token, 8, f"Single Ion Anisotropy.{label}")
                for token in tokens[1:]
            )
        )
        cursor += 1
        for _ in (1, 2):
            if cursor >= stop:
                _text_error("Single Ion Anisotropy", f"truncated tensor for {label}")
            continuation = lines[cursor].split()
            if len(continuation) != 3:
                _text_error(
                    "Single Ion Anisotropy", f"malformed tensor row for {label}"
                )
            matrix_rows.append(
                tuple(
                    _text_fixed_decimal(token, 8, f"Single Ion Anisotropy.{label}")
                    for token in continuation
                )
            )
            cursor += 1
        _finite(
            np.asarray(matrix_rows, dtype=np.float64),
            (3, 3),
            f"exchange.out SIA tensor {label}",
        )
        seen.add(label)
        losses.append(("sia_tensor", f"sia_tensor for {label}"))
        quanta.append(
            (f"unsupported.sia_tensor[{label}]", Decimal("0.000000005"), "meV")
        )
    return cursor


def _text_skip_separators(line):
    stripped = line.strip()
    return not stripped or (len(stripped) >= 8 and set(stripped) <= {"-", "="})


def _text_parse_optional_preexchange(lines, start, stop, atom_labels):
    losses = []
    quanta = []
    seen_sections = set()
    cursor = start
    while cursor < stop:
        stripped = lines[cursor].strip()
        if _text_skip_separators(stripped):
            cursor += 1
        elif stripped.startswith("Orbitals used in decomposition"):
            if "orbital_names" in seen_sections:
                _text_error("Orbitals used in decomposition", "duplicate section")
            seen_sections.add("orbital_names")
            cursor = _text_parse_orbital_names(lines, cursor, stop, atom_labels, losses)
        elif stripped.startswith("Single Ion Anisotropy (meV)"):
            if "sia_legacy" in seen_sections:
                _text_error("Single Ion Anisotropy (meV)", "duplicate section")
            seen_sections.add("sia_legacy")
            cursor = _text_parse_sia_legacy(
                lines, cursor, stop, atom_labels, losses, quanta
            )
        elif stripped == "Single Ion Anisotropy:":
            if "sia_tensor" in seen_sections:
                _text_error("Single Ion Anisotropy", "duplicate section")
            seen_sections.add("sia_tensor")
            cursor = _text_parse_sia_tensor(
                lines, cursor, stop, atom_labels, losses, quanta
            )
        else:
            _text_error(
                "structure",
                f"unknown meaningful pre-exchange section or data row {stripped!r}",
            )
    return losses, quanta


def _text_validate_orbital_array(lines, index, field):
    if index >= len(lines):
        _text_error(field, "missing orbital-contribution array")
    pieces = []
    depth = 0
    cursor = index
    started = False
    while cursor < len(lines):
        line = lines[cursor].strip()
        if not line:
            if started:
                break
            cursor += 1
            continue
        if not started and not line.startswith("["):
            _text_error(field, f"expected a NumPy array, found {line!r}")
        started = True
        depth += line.count("[") - line.count("]")
        if depth < 0:
            _text_error(field, "unbalanced NumPy array brackets")
        pieces.append(line)
        cursor += 1
        if depth == 0:
            break
    if not started or depth != 0:
        _text_error(field, "truncated or unbalanced orbital-contribution array")
    contents = " ".join(pieces)
    tokens = contents.replace("[", " ").replace("]", " ").split()
    if not tokens:
        _text_error(field, "empty orbital-contribution array")
    for token in tokens:
        if token == "...":
            # NumPy summarizes large unsupported arrays. Named loss drops the
            # block; no physical orbital values are reconstructed from it.
            continue
        if not _TEXT_ARRAY_TOKEN.fullmatch(token):
            _text_error(
                field, f"invalid/truncated orbital-contribution value {token!r}"
            )
        _text_decimal(token, field)
    return cursor


_TEXT_ORBITAL_HEADERS = {
    "isotropic J:",
    "DMI x:",
    "DMI y:",
    "DMI z:",
    "Jani xx:",
    "Jani xy:",
    "Jani xz:",
    "Jani yx:",
    "Jani yy:",
    "Jani yz:",
    "Jani zx:",
    "Jani zy:",
    "Jani zz:",
    "Orbital contributions to dJ/dx (meV/displacement unit):",
}


def _text_parse_orbital_contributions(lines, index, losses, record_key):
    cursor = index + (lines[index].strip() == "Orbital contributions:")
    seen_headers = set()
    while cursor < len(lines):
        stripped = lines[cursor].strip()
        if _TEXT_PAIR.fullmatch(lines[cursor]):
            break
        if not stripped or _text_skip_separators(stripped):
            cursor += 1
            continue
        if stripped in _TEXT_ORBITAL_HEADERS:
            if stripped in seen_headers:
                _text_error(
                    "Orbital contributions", f"duplicate component block {stripped!r}"
                )
            seen_headers.add(stripped)
            if stripped.startswith("Orbital contributions to dJ/dx"):
                losses.append(("djdx", f"orbital dJ/dx for pair {record_key}"))
            elif not any(token == "orbital" for token, _ in losses):
                losses.append(
                    ("orbital", f"orbital contributions for pair {record_key}")
                )
            cursor = _text_validate_orbital_array(
                lines, cursor + 1, f"orbital {stripped}"
            )
            continue
        _text_error("Orbital contributions", f"unknown orbital sub-block {stripped!r}")
    if not seen_headers:
        _text_error(
            "Orbital contributions",
            "section contains no recognized contribution arrays",
        )
    return cursor


def _text_parse_exchange(lines, marker, label_to_atom):
    losses = []
    quanta = []
    records = []
    by_atom_key = {}
    cursor = marker + 1
    header_seen = False
    while cursor < len(lines):
        line = lines[cursor]
        stripped = line.strip()
        if not stripped or (stripped and set(stripped) == {"-"}):
            cursor += 1
            continue
        if not header_seen:
            columns = stripped.split()
            if (
                len(columns) != 6
                or columns[0:3] != ["i", "j", "R"]
                or columns[3:] != ["J_iso(meV)", "vector", "distance(A)"]
            ):
                _text_error("Exchange", f"unknown exchange table header {stripped!r}")
            header_seen = True
            cursor += 1
            continue

        pair_match = _TEXT_PAIR.fullmatch(line)
        if pair_match is not None:
            label_i, label_j, rx, ry, rz, j_token, vx, vy, vz, distance = (
                pair_match.groups()
            )
            if label_i not in label_to_atom or label_j not in label_to_atom:
                _text_error(
                    "Exchange pair",
                    f"pair references unknown full-table label {label_i!r}/{label_j!r}",
                )
            j_row = _text_fixed_decimal(j_token, 4, "Exchange pair J_iso(meV)")
            for token in (vx, vy, vz, distance):
                _text_fixed_decimal(token, 3, "Exchange pair vector/distance")
            R = (int(rx), int(ry), int(rz))
            atom_key = (R, label_to_atom[label_i], label_to_atom[label_j])
            if atom_key in by_atom_key:
                _text_error("Exchange pair", f"duplicate source pair {atom_key!r}")
            record = {
                "labels": (label_i, label_j),
                "atoms": atom_key[1:],
                "R": R,
                "row_jiso": j_row,
                "jiso": None,
                "dmi": None,
                "jani": None,
                "jani_bound": None,
                "combined": None,
                "combined_bound": None,
                "extra_seen": set(),
            }
            by_atom_key[atom_key] = record
            records.append(record)
            cursor += 1
            continue

        if not records:
            _text_error(
                "Exchange",
                f"unexpected exchange content before first pair: {stripped!r}",
            )
        record = records[-1]

        def mark_extra(name):
            if name in record["extra_seen"]:
                _text_error(name, f"duplicate {name} block for {pair_description}")
            record["extra_seen"].add(name)

        pair_description = (
            f"{record['labels'][0]} {record['labels'][1]} R={record['R']}"
        )
        jiso_match = _TEXT_JISO.fullmatch(stripped)
        if jiso_match is not None:
            if record["jiso"] is not None:
                _text_error("J_iso", f"duplicate J_iso line for {pair_description}")
            record["jiso"] = _text_fixed_decimal(jiso_match.group(1), 4, "J_iso(meV)")
            if abs(record["jiso"] - record["row_jiso"]) > Decimal("0.0001"):
                _text_error(
                    "J_iso",
                    f"pair-table value conflicts with authoritative J_iso line for {pair_description}",
                )
            quanta.append(
                (
                    f"exchange_Jdict[{record['R']},{record['labels'][0]},{record['labels'][1]}]",
                    Decimal("0.00000005"),
                    "eV",
                )
            )
            cursor += 1
            continue
        dmi_match = _TEXT_DMI.fullmatch(stripped)
        if dmi_match is not None:
            if record["dmi"] is not None:
                _text_error("DMI", f"duplicate DMI line for {pair_description}")
            values = tuple(
                _text_fixed_decimal(token, 4, "DMI(meV)")
                for token in dmi_match.groups()
            )
            record["dmi"] = np.asarray(
                [float(value / Decimal(1000)) for value in values], dtype=float
            )
            quanta.append(
                (
                    f"dmi_ddict[{record['R']},{record['labels'][0]},{record['labels'][1]}]",
                    (Decimal("5e-8"),) * 3,
                    "eV",
                )
            )
            cursor += 1
            continue
        if stripped == "[Testing!]J_ani:":
            if record["jani"] is not None:
                _text_error("Jani", f"duplicate Jani matrix for {pair_description}")
            record["jani"], record["jani_bound"], _, cursor = _text_matrix(
                lines, cursor + 1, "Jani(meV)"
            )
            quanta.append(
                (
                    f"Jani_dict[{record['R']},{record['labels'][0]},{record['labels'][1]}]",
                    record["jani_bound"],
                    "eV",
                )
            )
            continue
        if (
            stripped
            == "Combined J tensor (meV) [J = Jiso*I + DMI_antisymmetric + Jani_symmetric]:"
        ):
            if record["combined"] is not None:
                _text_error(
                    "Combined J tensor",
                    f"duplicate combined tensor for {pair_description}",
                )
            record["combined"], record["combined_bound"], _, cursor = _text_matrix(
                lines, cursor + 1, "Combined J tensor(meV)"
            )
            quanta.append(
                (
                    f"combined_tensor[{record['R']},{record['labels'][0]},{record['labels'][1]}]",
                    record["combined_bound"],
                    "eV",
                )
            )
            continue
        njt_dmi_match = _TEXT_NJT_DMI.fullmatch(stripped)
        if njt_dmi_match is not None:
            mark_extra("DMI_NJt")
            _, values = _text_parenthesized_vector(njt_dmi_match, "DMI_NJt(meV)", 4)
            losses.append(
                (
                    "njt",
                    f"DMI_NJt {pair_description}: {tuple(float(value) for value in values)} meV",
                )
            )
            quanta.append(
                (f"unsupported.njt.DMI_NJt[{record['R']}]", Decimal("0.00005"), "meV")
            )
            cursor += 1
            continue
        njt_jani_match = _TEXT_NJT_JANI.fullmatch(stripped)
        if njt_jani_match is not None:
            _, values = _text_parenthesized_vector(njt_jani_match, "Jani_NJt(meV)", 4)
            mark_extra("Jani_NJt")
            losses.append(
                (
                    "njt",
                    f"Jani_NJt {pair_description}: {tuple(float(value) for value in values)} meV",
                )
            )
            quanta.append(
                (f"unsupported.njt.Jani_NJt[{record['R']}]", Decimal("0.00005"), "meV")
            )
            cursor += 1
            continue
        derivative = _TEXT_DJDX.fullmatch(stripped)
        if derivative is not None:
            value = _text_fixed_decimal(derivative.group(1), 3, "dJ/dx(meV)")
            mark_extra("dJ/dx")
            losses.append(("djdx", f"dJ/dx {pair_description}: {value} meV"))
            quanta.append(
                (f"unsupported.djdx[{record['R']}]", Decimal("0.0005"), "meV")
            )
            cursor += 1
            continue
        derivative2 = _TEXT_D2JDX2.fullmatch(stripped)
        if derivative2 is not None:
            value = _text_fixed_decimal(derivative2.group(1), 3, "d2J/dx2(meV)")
            mark_extra("d2J/dx2")
            losses.append(("djdx", f"d2J/dx2 {pair_description}: {value} meV"))
            quanta.append(
                (f"unsupported.d2jdx2[{record['R']}]", Decimal("0.0005"), "meV")
            )
            cursor += 1
            continue
        biquadratic = _TEXT_BIQUADRATIC.fullmatch(stripped)
        if biquadratic is not None:
            values = tuple(
                _text_fixed_decimal(token, 3, "biquadratic(meV)")
                for token in biquadratic.groups()
            )
            mark_extra("biquadratic")
            losses.append(
                (
                    "biquadratic",
                    f"Jprime/B {pair_description}: {tuple(float(value) for value in values)} meV",
                )
            )
            quanta.append(
                (f"unsupported.biquadratic[{record['R']}]", Decimal("0.0005"), "meV")
            )
            cursor += 1
            continue
        dmi2 = _TEXT_DMI2.fullmatch(stripped)
        if dmi2 is not None:
            _, values = _text_parenthesized_vector(dmi2, "DMI2(meV)", 4)
            mark_extra("DMI2")
            losses.append(
                (
                    "dmi",
                    f"DMI2 debug vector {pair_description}: {tuple(float(value) for value in values)} meV",
                )
            )
            quanta.append(
                (f"unsupported.dmi2[{record['R']}]", Decimal("0.00005"), "meV")
            )
            cursor += 1
            continue
        if stripped.startswith("[Debug!] DMI2 not available:"):
            cursor += 1
            continue
        if stripped == "Orbital contributions:" or stripped in _TEXT_ORBITAL_HEADERS:
            mark_extra("orbital contributions")
            cursor = _text_parse_orbital_contributions(
                lines, cursor, losses, pair_description
            )
            continue
        if stripped.startswith("Single Ion Anisotropy"):
            losses.append(("sia", f"unsupported SIA block after Exchange: {stripped}"))
            _text_error(
                "Single Ion Anisotropy",
                "SIA blocks are only accepted in the pre-exchange writer position",
            )
        if stripped == "Exchange:" or stripped.startswith("Information:"):
            _text_error("Exchange", "duplicate top-level section")
        _text_error(
            "Exchange", f"unknown meaningful exchange block or row {stripped!r}"
        )

    if not header_seen or not records:
        _text_error("Exchange", "missing writer exchange table or pair records")
    for record in records:
        if record["jiso"] is None:
            _text_error(
                "J_iso",
                f"missing authoritative J_iso line for {record['labels']} R={record['R']}",
            )
    combined_count = sum(record["combined"] is not None for record in records)
    if combined_count not in (0, len(records)):
        _text_error(
            "Combined J tensor",
            "Combined blocks must be present for every pair or absent for the entire source",
        )
    if combined_count:
        for record in records:
            jiso_ev = float(record["jiso"] * Decimal("0.001"))
            dmi_ev = record["dmi"]
            jani_ev = None if record["jani"] is None else record["jani"]
            # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_channel_isotropic
            # MathDoc: tb2j/spacegroup-exchange-symmetrization#pair_tensor_dmi_axial_vector
            channel_tensor = combine_J_tensor(Jiso=jiso_ev, D=dmi_ev, Jani=jani_ev)
            combined_tensor = record["combined"]
            allowed = np.asarray(record["combined_bound"], dtype=np.float64)
            allowed += np.eye(3) * 5e-8
            if record["dmi"] is not None:
                allowed += (np.ones((3, 3)) - np.eye(3)) * 5e-8
            if record["jani_bound"] is not None:
                allowed += np.asarray(record["jani_bound"], dtype=np.float64)
            if np.any(np.abs(combined_tensor - channel_tensor) > allowed):
                _text_error(
                    "Combined J tensor",
                    f"redundant tensor disagrees with authoritative channels outside their summed printed-precision intervals for {record['labels']} R={record['R']}",
                )
    return records, losses, quanta, bool(combined_count)


def _read_text(path, context, allow_loss):
    """Read ``exchange.out`` without version sniffing or channel fabrication."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"exchange.out is not valid UTF-8 text: {path}") from error
    lines = text.splitlines()
    if context is None:
        context = NativeContext()

    information = [
        index for index, line in enumerate(lines) if line.strip() == "Information:"
    ]
    cells = [
        index for index, line in enumerate(lines) if line.strip() == "Cell (Angstrom):"
    ]
    atoms_markers = [
        index for index, line in enumerate(lines) if line.strip() == "Atoms:"
    ]
    exchanges = [
        index for index, line in enumerate(lines) if line.strip() == "Exchange:"
    ]
    if (
        len(information) != 1
        or len(cells) != 1
        or len(atoms_markers) != 1
        or len(exchanges) != 1
    ):
        _text_error(
            "structure",
            "expected exactly one Information, Cell (Angstrom), full Atoms, and Exchange section",
        )
    info_index, cell_index, atoms_index, exchange_index = (
        information[0],
        cells[0],
        atoms_markers[0],
        exchanges[0],
    )
    if not (info_index < cell_index < atoms_index < exchange_index):
        _text_error(
            "structure",
            "Information/Cell/Atoms/Exchange sections are out of writer order",
        )
    if not lines or not any(
        set(line.strip()) == {"="} for line in lines[: info_index + 1]
    ):
        _text_error("structure", "missing native exchange.out section banner")
    if any(line.strip() and set(line.strip()) != {"="} for line in lines[:info_index]):
        _text_error("structure", "unexpected content before the Information section")

    cell = _text_parse_cell(lines, cell_index)
    # Between the embedded cell and atom table only the writer's whitespace
    # and section separators are legal.
    for line in lines[cell_index + 4 : atoms_index]:
        if not _text_skip_separators(line):
            _text_error(
                "structure",
                f"unexpected content between Cell and Atoms: {line.strip()!r}",
            )
    atom_data = _text_parse_atoms(lines, atoms_index, cell)
    if atom_data["next_line"] > exchange_index:
        _text_error("Atoms", "atom table overlaps the Exchange section")

    optional_losses, quanta = _text_parse_optional_preexchange(
        lines, atom_data["next_line"], exchange_index, set(atom_data["label_to_atom"])
    )
    records, exchange_losses, exchange_quanta, has_combined = _text_parse_exchange(
        lines, exchange_index, atom_data["label_to_atom"]
    )
    losses = optional_losses + exchange_losses
    quanta.extend(exchange_quanta)

    atoms = _compare_context_structure(atom_data["atoms"], context, 5e-9, 5e-9)
    source_charges = atom_data["charges"]
    context_charges = _context_charges(context, len(atoms))
    if context_charges is not None:
        if np.any(np.abs(context_charges - source_charges) > 5e-5):
            _text_error(
                "charges",
                "NativeContext.charges conflicts with printed w_charge values",
            )
        charges = context_charges.copy()
    else:
        charges = source_charges.copy()
    spinat, colinear = _text_check_context_moments(context, atom_data)

    observed_atoms = {atom for record in records for atom in record["atoms"]}
    index_spin = _native_indices(len(atoms), context, observed_atoms=observed_atoms)
    exchange = {}
    dmi = {}
    jani = {}
    for record in records:
        atom_i, atom_j = record["atoms"]
        spin_i, spin_j = index_spin[atom_i], index_spin[atom_j]
        key = (record["R"], spin_i, spin_j)
        if key in exchange:
            _text_error("Exchange pair", f"duplicate mapped SpinIO key {key!r}")
        exchange[key] = float(record["jiso"] * Decimal("0.001"))
        if record["dmi"] is not None:
            dmi[key] = record["dmi"].copy()
        if record["jani"] is not None:
            jani[key] = record["jani"].copy()

    losses = _guard_losses(losses, allow_loss)
    spinio = _make_native(
        atoms,
        index_spin,
        spinat=spinat,
        colinear=colinear,
        charges=charges,
        exchange=exchange,
        dmi=dmi or None,
        jani=jani or None,
    )
    quanta.extend(
        (
            ("cell", Decimal("5e-9"), "angstrom"),
            ("positions", Decimal("5e-9"), "angstrom"),
            ("charges", Decimal("0.00005"), "source charge unit"),
            ("moments", Decimal("0.00005"), "source moment unit"),
        )
    )
    message = (
        "Accepted exchange.out by structural grammar (Information, embedded Angstrom cell, complete numbered atom table, and Exchange records); "
        + (
            "Combined tensors were cross-checked against the authoritative per-channel sum using the summed channel and Combined printed-precision intervals."
            if has_combined
            else "no Combined tensor blocks were present; accepted the older structural form without parsing a version string."
        )
    )
    return _native_result(
        spinio,
        "text",
        "exchange.out-default",
        quanta=tuple(quanta),
        losses=losses,
        messages=(message,),
    )


def _read_exchange(path, format, context=None, allow_loss=()):
    if not isinstance(format, str) or format not in {"pickle", "text", "xml"}:
        raise ValueError(
            "format: select pickle, text, or xml explicitly; unknown inputs "
            "never fall back to unsafe pickle loading"
        )
    if context is None:
        context = NativeContext()
    elif not isinstance(context, NativeContext):
        raise TypeError("context: expected NativeContext or None")
    if isinstance(allow_loss, str):
        allow_loss = (allow_loss,)
    try:
        allow_loss = tuple(allow_loss)
    except TypeError as error:
        raise ValueError("allow_loss: expected named loss tokens") from error
    if any(
        not isinstance(token, str) or token not in _LOSS_NAMES for token in allow_loss
    ):
        raise ValueError(
            "allow_loss: unknown loss token; supported names: "
            + ", ".join(sorted(_LOSS_NAMES))
        )
    if format == "pickle":
        return _read_pickle(path, context, allow_loss)
    if format == "text":
        return _read_text(path, context, allow_loss)
    return _read_xml(path, context, allow_loss)
