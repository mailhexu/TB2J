"""Consumer regressions for symmetry-shell documents and their TOML wire format."""

import pytest
import tomli

from TB2J.io_exchange.shells import read_shells, write_shells

HEADER = """schema_version = 1
exchange_mode = "isotropic"

[structure]
cell_angstrom = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
species = ["Fe", "Fe", "Fe"]
positions_frac = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
magnetic_atoms = [1, 2, 3]
index_spin = [0, 1, 2]

[symmetry]
mode = "sg"
symprec = 1.0e-5

[[symmetry.operations]]
rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
translation = [0.0, 0.0, 0.0]
time_reversal = false

[tolerances]
value_atol = 1.0e-8
value_rtol = 1.0e-6
zero_tol = 1.0e-10
"""


def test_equal_distance_pairs_in_distinct_stored_orbits_stay_distinct(tmp_path):
    # Both bonds have length 1 Angstrom. Identity-only snapshot: two true orbits,
    # irrespective of the larger symmetry a fresh producer might detect.
    source = tmp_path / "editable.shells.toml"
    source.write_text(
        HEADER
        + """
[[shell]]
id = "1:2:0,0,0"
Jiso = 1.25

[[shell]]
id = "1:3:0,0,0"
Jiso = 2.5
"""
    )
    document = read_shells(source)
    assert [(shell.id, shell.Jiso) for shell in document.shells] == [
        ("1:2:0,0,0", 0.00125),
        ("1:3:0,0,0", 0.0025),
    ]
    output = tmp_path / "saved.shells.toml"
    write_shells(document, output)
    wire = tomli.loads(output.read_text())
    assert wire["energy_unit"] == "meV"
    assert [(shell["id"], shell["Jiso"]) for shell in wire["shell"]] == [
        ("1:2:0,0,0", 1.25),
        ("1:3:0,0,0", 2.5),
    ]


def one_atom_header():
    return (
        HEADER.replace('species = ["Fe", "Fe", "Fe"]', 'species = ["Fe"]')
        .replace(
            "positions_frac = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]",
            "positions_frac = [[0.0, 0.0, 0.0]]",
        )
        .replace("magnetic_atoms = [1, 2, 3]", "magnetic_atoms = [1]")
        .replace("index_spin = [0, 1, 2]", "index_spin = [0]")
    )


@pytest.mark.parametrize(
    "rotation,diagnostic",
    [
        ("[[0, -1, 0], [1, 0, 0], [0, 0, 1]]", "closure"),
        ("[[-1, 1, 0], [0, 1, 0], [0, 0, 1]]", "metric"),
    ],
)
def test_invalid_stored_group_is_rejected_even_without_shells(
    tmp_path, rotation, diagnostic
):
    # C4's identity+quarter-turn is not closed; the involutory shear is closed
    # but not a symmetry of the cubic metric. Both leave the single site fixed.
    operation = f"""
[[symmetry.operations]]
rotation = {rotation}
translation = [0.0, 0.0, 0.0]
time_reversal = false

"""
    source = tmp_path / "invalid-group.shells.toml"
    source.write_text(
        one_atom_header().replace("[tolerances]", operation + "[tolerances]")
    )
    with pytest.raises(ValueError, match=f"symmetry.operations.*{diagnostic}"):
        read_shells(source)


def centered_header():
    header = (
        HEADER.replace('species = ["Fe", "Fe", "Fe"]', 'species = ["Fe", "Fe"]')
        .replace(
            "positions_frac = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]",
            "positions_frac = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]",
        )
        .replace("magnetic_atoms = [1, 2, 3]", "magnetic_atoms = [1, 2]")
        .replace("index_spin = [0, 1, 2]", "index_spin = [0, 1]")
    )
    half_translation = """
[[symmetry.operations]]
rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
translation = [0.5, 0.0, 0.0]
time_reversal = false

"""
    return header.replace("[tolerances]", half_translation + "[tolerances]")


def test_stored_symmetry_cannot_exchange_selected_and_unselected_sites(tmp_path):
    source = tmp_path / "selection.shells.toml"
    source.write_text(
        centered_header()
        .replace("magnetic_atoms = [1, 2]", "magnetic_atoms = [1]")
        .replace("index_spin = [0, 1]", "index_spin = [0, -1]")
    )
    with pytest.raises(ValueError, match="symmetry.operations.*magnetic"):
        read_shells(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("orbit_size", "3"),
        ("support_members", "[2]"),
        ("support_members", "[1, 0]"),
        ("support_members", "[0, 0]"),
        ("support_members", "[-1]"),
    ],
)
def test_masks_and_multiplicity_use_the_full_centered_orbit(tmp_path, field, value):
    source = tmp_path / "mask.shells.toml"
    source.write_text(
        centered_header()
        + f"""
[[shell]]
id = "1:2:0,0,0"
Jiso = 0.7
{field} = {value}
"""
    )
    with pytest.raises(ValueError, match=f"shell 1:2:0,0,0.*{field}"):
        read_shells(source)


def test_centered_orbit_comments_are_removable_and_unfilled_is_draft_only(tmp_path):
    from TB2J.io_exchange.shells import validate_shells

    source = tmp_path / "draft.shells.toml"
    source.write_text(
        centered_header()
        + """
[[shell]]
id = "1:2:0,0,0"
Jiso = "unfilled"
orbit_size = 2
support_members = [1]
"""
    )
    document = read_shells(source)
    output = tmp_path / "commented.shells.toml"
    write_shells(document, output)
    text = output.read_text()
    assert "(1, 2, R = (0, 0, 0))" in text
    assert "(2, 1, R = (1, 0, 0))" in text
    stripped = tmp_path / "stripped.shells.toml"
    stripped.write_text(
        "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    )
    assert read_shells(stripped) == document
    write_shells(document, output, equivalent_comments=False)
    assert "#" not in output.read_text()
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.Jiso.*unfilled"):
        validate_shells(document, for_export=True)


@pytest.mark.parametrize(
    "before,after,field",
    [
        ("index_spin = [0, 1, 2]", "index_spin = [0, 0, 1]", "index_spin"),
        ("magnetic_atoms = [1, 2, 3]", "magnetic_atoms = [3, 1, 2]", "magnetic_atoms"),
        ("value_atol = 1.0e-8", "value_atol = -1.0e-8", "value_atol"),
        ("zero_tol = 1.0e-10", "zero_tol = nan", "zero_tol"),
        (
            "[symmetry]",
            "[reference_spins]\\nmoments = [0.0, nan, 0.0]\\n\\n[symmetry]",
            "moments",
        ),
        ('mode = "sg"', 'mode = "msg"\\nmagnetic_tol = 1.0e-5', "moments"),
    ],
)
def test_invalid_context_is_rejected_before_any_exchange_export(
    tmp_path, before, after, field
):
    source = tmp_path / "context.shells.toml"
    source.write_text(HEADER.replace(before, after.replace("\\n", "\n")))
    with pytest.raises(ValueError, match=field):
        read_shells(source)


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        (
            "[structure]",
            '[structure]\\ncell_unit = "bohr"',
            "structure.*unknown.*cell_unit",
        ),
        ("time_reversal = false", "", "operations.*time_reversal"),
        (
            "schema_version = 1",
            "schema_version = 1\\nunknown_header = 7",
            "unknown.*unknown_header",
        ),
        ("schema_version = 1", "schema_version = 2", "schema_version"),
        (
            "schema_version = 1",
            "schema_version = 1\\nschema_version = 1",
            "duplicate TOML key",
        ),
    ],
)
def test_wire_schema_rejects_ignored_fields_and_missing_operation_flags(
    tmp_path, before, after, diagnostic
):
    source = tmp_path / "invalid-schema.shells.toml"
    source.write_text(HEADER.replace(before, after.replace("\\n", "\n")))
    with pytest.raises(ValueError, match=diagnostic):
        read_shells(source)


@pytest.mark.parametrize(
    "moments",
    [
        "[1.0, 1.0]",
        "[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]",
    ],
)
def test_primed_snapshot_must_preserve_scalar_or_axial_reference_moments(
    tmp_path, moments
):
    header = centered_header().replace(
        'mode = "sg"', 'mode = "msg"\nmagnetic_tol = 1.0e-5'
    )
    header = header.replace(
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = false",
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = true",
    )
    source = tmp_path / "bad-reference.shells.toml"
    source.write_text(
        header.replace(
            "[symmetry]", f"[reference_spins]\nmoments = {moments}\n\n[symmetry]"
        )
    )
    with pytest.raises(ValueError, match="reference_spins.moments.*operation"):
        read_shells(source)


def test_producer_drift_is_report_only_and_never_rewrites_snapshot(tmp_path):
    from TB2J.io_exchange.shells import check_producer_ops, validate_shells

    source = tmp_path / "stored-p1.shells.toml"
    source.write_text(one_atom_header())
    document = read_shells(source)
    original = document.operations
    # Cubic one-atom structure has more symmetry than the deliberately stored
    # identity subgroup. Ordinary load must keep the author's snapshot.
    assert check_producer_ops(document).producer_drift is True
    assert validate_shells(document, check_producer_ops=True).producer_drift is True
    assert document.operations == original
    output = tmp_path / "still-p1.shells.toml"
    write_shells(document, output)
    assert read_shells(output).operations == original


@pytest.mark.parametrize(
    "channel,tensor",
    [
        ("Jiso", "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]"),
        ("DMI", "[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"),
        ("Jani", "[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]"),
    ],
)
def test_empty_channel_masks_cannot_discard_authored_tensor_subspaces(
    tmp_path, channel, tensor
):
    source = tmp_path / "channel-mask.shells.toml"
    entry = f"""
[[shell]]
id = "1:2:0,0,0"
representative = [1, 2, 0, 0, 0]
tensor = {tensor}
{channel}_members = []
"""
    header = HEADER.replace('exchange_mode = "isotropic"', 'exchange_mode = "tensor"')
    source.write_text(header + entry)
    with pytest.raises(
        ValueError, match=f"shell 1:2:0,0,0.*{channel}_members.*nonzero"
    ):
        read_shells(source)
    zero = "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"
    source.write_text(header + entry.replace(tensor, zero))
    document = read_shells(source)
    assert getattr(document.shells[0], channel + "_members") == ()


@pytest.mark.parametrize("value", ["nan", "inf", "true", "[1.0, 2.0]"])
def test_exchange_values_reject_nonfinite_boolean_and_wrong_shapes(tmp_path, value):
    source = tmp_path / "bad-value.shells.toml"
    source.write_text(HEADER + f'\n[[shell]]\nid = "1:2:0,0,0"\nJiso = {value}\n')
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.Jiso"):
        read_shells(source)


def test_canonical_roundtrip_preserves_float_values_and_operation_order(tmp_path):
    inversion = """
[[symmetry.operations]]
rotation = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
translation = [0.0, 0.0, 0.0]
time_reversal = false

"""
    source = tmp_path / "unordered.shells.toml"
    source.write_text(
        one_atom_header().replace("[tolerances]", inversion + "[tolerances]")
        + """
[[shell]]
id = "1:1:-1,0,0"
representative = [1, 1, 1, 0, 0]
Jiso = 0.12345678901234567
"""
    )
    document = read_shells(source)
    first = tmp_path / "first.shells.toml"
    second = tmp_path / "second.shells.toml"
    write_shells(document, first)
    reloaded = read_shells(first)
    assert reloaded == document
    write_shells(reloaded, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("wire_value,inside", [("0.00005", True), ("0.00015", False)])
def test_mev_wire_values_straddle_the_absolute_ev_tolerance(
    tmp_path, wire_value, inside
):
    import numpy as np

    source = tmp_path / "precision.shells.toml"
    source.write_text(
        HEADER.replace("value_atol = 1.0e-8", "value_atol = 5.0e-8")
        + f'\n[[shell]]\nid = "1:2:0,0,0"\nJiso = {wire_value}\n'
    )
    document = read_shells(source)
    # A consumer compares internal eV with the unscaled absolute eV bound.
    # The second wire value is one fixed-4-meV quantum above the first.
    assert (
        bool(np.isclose(document.shells[0].Jiso, 0.0, atol=document.value_atol, rtol=0))
        is inside
    )
    output = tmp_path / "precision-saved.shells.toml"
    write_shells(document, output)
    assert read_shells(output) == document


def test_tensor_mode_and_representative_errors_name_the_shell(tmp_path):
    source = tmp_path / "mode.shells.toml"
    source.write_text(HEADER + '\n[[shell]]\nid = "1:2:0,0,0"\ntensor = "unfilled"\n')
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.*tensor"):
        read_shells(source)
    source.write_text(
        source.read_text().replace(
            'exchange_mode = "isotropic"', 'exchange_mode = "tensor"'
        )
    )
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.representative"):
        read_shells(source)


def test_id_must_be_the_full_direct_orbit_minimum_not_the_representative(tmp_path):
    source = tmp_path / "nonminimum.shells.toml"
    source.write_text(centered_header() + '\n[[shell]]\nid = "2:1:1,0,0"\nJiso = 1.0\n')
    with pytest.raises(ValueError, match="2:1:1,0,0.*1:2:0,0,0"):
        read_shells(source)


def test_python_edits_reject_complex_exchange_values(tmp_path):
    from dataclasses import replace

    source = tmp_path / "real.shells.toml"
    source.write_text(HEADER + '\n[[shell]]\nid = "1:2:0,0,0"\nJiso = 1.0\n')
    document = read_shells(source)
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.Jiso.*complex"):
        replace(document, shells=(replace(document.shells[0], Jiso=1.0 + 2.0j),))


@pytest.mark.parametrize(
    "moments",
    [
        "[1.0, -1.0]",
        "[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]",
    ],
)
def test_valid_primed_half_translation_snapshot_roundtrips(tmp_path, moments):
    header = centered_header().replace(
        'mode = "sg"', 'mode = "msg"\nmagnetic_tol = 1.0e-5'
    )
    header = header.replace(
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = false",
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = true",
    )
    source = tmp_path / "afm.shells.toml"
    source.write_text(
        header.replace(
            "[symmetry]", f"[reference_spins]\nmoments = {moments}\n\n[symmetry]"
        )
        + '\n[[shell]]\nid = "1:2:0,0,0"\nJiso = -2.0\n'
    )
    document = read_shells(source)
    output = tmp_path / "afm-saved.shells.toml"
    write_shells(document, output)
    assert read_shells(output) == document
    assert tuple(operation.time_reversal for operation in document.operations) == (
        False,
        True,
    )


def test_python_tensor_edit_cannot_coerce_mixed_booleans_into_exchange(tmp_path):
    from dataclasses import replace

    source = tmp_path / "tensor.shells.toml"
    source.write_text(
        HEADER.replace('exchange_mode = "isotropic"', 'exchange_mode = "tensor"')
        + '\n[[shell]]\nid = "1:2:0,0,0"\nrepresentative = [1, 2, 0, 0, 0]\n'
        "tensor = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]\n"
    )
    document = read_shells(source)
    invalid = ((True, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 0.001))
    with pytest.raises(ValueError, match="shell 1:2:0,0,0.tensor.*boolean"):
        replace(document, shells=(replace(document.shells[0], tensor=invalid),))


def test_isotropic_tensor_preserves_exactly_absent_anisotropy(tmp_path):
    from dataclasses import replace

    source = tmp_path / "isotropic-tensor.shells.toml"
    source.write_text(
        HEADER.replace('exchange_mode = "isotropic"', 'exchange_mode = "tensor"')
        + '\n[[shell]]\nid = "1:2:0,0,0"\nrepresentative = [1, 2, 0, 0, 0]\n'
        "tensor = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]\n"
        "DMI_members = []\nJani_members = []\n"
    )
    document = read_shells(source)
    tensor = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    edited = replace(document, shells=(replace(document.shells[0], tensor=tensor),))
    output = tmp_path / "saved.shells.toml"
    write_shells(edited, output)
    assert read_shells(output) == edited
    assert edited.shells[0].DMI_members == edited.shells[0].Jani_members == ()
    nonzero = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.10000000000000002))
    with pytest.raises(ValueError, match="Jani_members.*nonzero"):
        replace(document, shells=(replace(document.shells[0], tensor=nonzero),))


def test_boolean_symprec_is_not_a_numeric_tolerance(tmp_path):
    from dataclasses import replace

    source = tmp_path / "tolerance.shells.toml"
    source.write_text(one_atom_header())
    document = read_shells(source)
    with pytest.raises(ValueError, match="symmetry.symprec"):
        replace(document, symprec=True)
    source.write_text(one_atom_header().replace("symprec = 1.0e-5", "symprec = true"))
    with pytest.raises(ValueError, match="symmetry.symprec"):
        read_shells(source)


@pytest.mark.parametrize(
    "field", ["symprec", "magnetic_tol", "value_atol", "value_rtol", "zero_tol"]
)
def test_validated_scalar_tolerances_do_not_retain_mutable_aliases(tmp_path, field):
    from dataclasses import replace

    import numpy as np

    source = tmp_path / "scalar.shells.toml"
    source.write_text(one_atom_header())
    document = read_shells(source)
    alias = np.array(1.0e-5)
    edited = replace(document, **{field: alias})
    alias[...] = -1
    assert getattr(edited, field) == 1.0e-5
    scalar = replace(edited, **{field: np.float64(1.0e-5)})
    output = tmp_path / "saved.shells.toml"
    write_shells(scalar, output)
    assert read_shells(output) == scalar


@pytest.mark.parametrize("with_shell", [False, True])
def test_primed_snapshot_requires_msg_context(tmp_path, with_shell):
    source = tmp_path / "primed-sg.shells.toml"
    header = centered_header().replace(
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = false",
        "translation = [0.5, 0.0, 0.0]\ntime_reversal = true",
    )
    entry = '\n[[shell]]\nid = "1:2:0,0,0"\nJiso = 1.0\n' if with_shell else ""
    source.write_text(header + entry)
    with pytest.raises(ValueError, match=r"symmetry.operations.*sg.*msg"):
        read_shells(source)


def test_explicit_representative_must_be_in_sparse_support(tmp_path):
    source = tmp_path / "sparse-tensor.shells.toml"
    header = centered_header().replace(
        'exchange_mode = "isotropic"', 'exchange_mode = "tensor"'
    )
    entry = (
        '\n[[shell]]\nid = "1:2:0,0,0"\nrepresentative = [1, 2, 0, 0, 0]\n'
        'tensor = "unfilled"\nsupport_members = [1]\n'
    )
    source.write_text(header + entry)
    with pytest.raises(ValueError, match="representative.*support"):
        read_shells(source)
    source.write_text(header + entry.replace("[1, 2, 0, 0, 0]", "[2, 1, 1, 0, 0]"))
    document = read_shells(source)
    assert document.shells[0].id == "1:2:0,0,0"
    assert document.shells[0].representative == (2, 1, 1, 0, 0)
