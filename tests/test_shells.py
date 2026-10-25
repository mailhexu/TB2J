"""Consumer regressions for symmetry-shell documents and their TOML wire format."""

from dataclasses import replace

import numpy as np
import pytest
import tomli

from TB2J.io_exchange.shells import read_shells, write_shells

HEADER = '''schema_version = 2
exchange = "isotropic"
[structure]
coordinates = "fractional"
lattice = """
10 0 0
0 10 0
0 0 10
"""
atoms = """
1 Fe1 0 0 0 Y 0
2 Fe2 0.1 0 0 Y 1
3 Fe3 0 0.1 0 Y 2
"""
[symmetry]
mode = "spacegroup"
symprec = 1.0e-5
[[symmetry.operations]]
rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
translation = [0.0, 0.0, 0.0]
time_reversal = false
[tolerances]
value_atol = 1.0e-8
value_rtol = 1.0e-6
zero_tol = 1.0e-10
'''


def one_atom_header():
    return HEADER.replace("2 Fe2 0.1 0 0 Y 1\n3 Fe3 0 0.1 0 Y 2\n", "")


def centered_header():
    half_translation = """[[symmetry.operations]]
rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
translation = [0.5, 0.0, 0.0]
time_reversal = false
"""
    return (
        HEADER.replace("3 Fe3 0 0.1 0 Y 2\n", "")
        .replace("2 Fe2 0.1 0 0 Y 1", "2 Fe2 0.5 0 0 Y 1")
        .replace("[tolerances]", half_translation + "[tolerances]")
    )


def assert_preserved(before, after):
    # Input summaries/representatives can be omitted. Compare every other
    # field, including tensor values, exact support and authoritative snapshot.
    restored = tuple(
        replace(
            new,
            distance=old.distance,
            n_equivalent_pairs=old.n_equivalent_pairs,
            representative=old.representative,
        )
        for old, new in zip(before.shells, after.shells, strict=True)
    )
    assert replace(after, shells=restored) == before


def test_equal_distance_pairs_in_distinct_stored_orbits_stay_distinct(tmp_path):
    source = tmp_path / "editable.shells.toml"
    source.write_text(
        HEADER
        + """
[[shell]]
id = "1:2:0,0,0"
J = 1.25
[[shell]]
id = "1:3:0,0,0"
J = 2.5
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
    assert [
        (
            shell["representative"],
            shell["J"],
            shell["distance"],
            shell["n_equivalent_pairs"],
        )
        for shell in wire["shell"]
    ] == [("Fe1 Fe2 0 0 0", 1.25, 1.0, 1), ("Fe1 Fe3 0 0 0", 2.5, 1.0, 1)]


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
    operation = f"""[[symmetry.operations]]
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


def test_stored_symmetry_cannot_exchange_selected_and_unselected_sites(tmp_path):
    source = tmp_path / "selection.shells.toml"
    source.write_text(
        centered_header().replace("2 Fe2 0.5 0 0 Y 1", "2 Fe2 0.5 0 0 N -1")
    )
    with pytest.raises(ValueError, match="symmetry.operations.*magnetic"):
        read_shells(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_equivalent_pairs", "3"),
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
        + f'\n[[shell]]\nid = "1:2:0,0,0"\nJ = 0.7\n{field} = {value}\n'
    )
    with pytest.raises(ValueError, match=field):
        read_shells(source)


def test_centered_sparse_comments_are_removable_and_unfilled_is_draft_only(tmp_path):
    from TB2J.io_exchange.shells import validate_shells

    source = tmp_path / "draft.shells.toml"
    source.write_text(
        centered_header()
        + """
[[shell]]
id = "1:2:0,0,0"
J = "unfilled"
n_equivalent_pairs = 2
support_members = [1]
"""
    )
    document = read_shells(source)
    output = tmp_path / "commented.shells.toml"
    write_shells(document, output)
    stripped = tmp_path / "stripped.shells.toml"
    stripped.write_text(
        "\n".join(
            line for line in output.read_text().splitlines() if not line.startswith("#")
        )
    )
    commented = read_shells(stripped)
    write_shells(document, output, equivalent_comments=False)
    assert read_shells(output) == commented
    assert commented.shells[0].n_equivalent_pairs == 2
    assert commented.shells[0].support_members == (1,)
    with pytest.raises(ValueError, match="unfilled"):
        validate_shells(document, for_export=True)


@pytest.mark.parametrize(
    "before,after,field",
    [
        ("2 Fe2 0.1 0 0 Y 1", "2 Fe2 0.1 0 0 Y 0", "index_spin"),
        ("3 Fe3 0 0.1 0 Y 2", "3 Fe3 0 0.1 0 N 2", "magnetic"),
        ("value_atol = 1.0e-8", "value_atol = -1.0e-8", "value_atol"),
        ("zero_tol = 1.0e-10", "zero_tol = nan", "zero_tol"),
        ("[symmetry]", '[reference_spins]\nmoments = "Fe1 nan"\n[symmetry]', "moments"),
        ('mode = "spacegroup"', 'mode = "magnetic"\nmagnetic_tol = 1.0e-5', "moments"),
    ],
)
def test_invalid_context_is_rejected_before_any_exchange_export(
    tmp_path, before, after, field
):
    source = tmp_path / "context.shells.toml"
    source.write_text(HEADER.replace(before, after))
    with pytest.raises(ValueError, match=field):
        read_shells(source)


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("[structure]", '[structure]\ncell_unit = "bohr"', "cell_unit"),
        ("time_reversal = false", "", "time_reversal"),
        (
            "schema_version = 2",
            "schema_version = 2\nunknown_header = 7",
            "unknown_header",
        ),
        ("schema_version = 2", "schema_version = 1", "schema_version"),
        ("schema_version = 2", "schema_version = 2\nschema_version = 2", "TOML"),
    ],
)
def test_wire_schema_rejects_ignored_fields_and_missing_operation_flags(
    tmp_path, before, after, diagnostic
):
    source = tmp_path / "invalid-schema.shells.toml"
    source.write_text(HEADER.replace(before, after))
    with pytest.raises(ValueError, match=diagnostic):
        read_shells(source)


def primed_header(moments):
    return (
        centered_header()
        .replace('mode = "spacegroup"', 'mode = "magnetic"\nmagnetic_tol = 1.0e-5')
        .replace(
            "translation = [0.5, 0.0, 0.0]\ntime_reversal = false",
            "translation = [0.5, 0.0, 0.0]\ntime_reversal = true",
        )
        .replace(
            "[symmetry]",
            f'[reference_spins]\nmoments = """\n{moments}\n"""\n[symmetry]',
        )
    )


@pytest.mark.parametrize("moments", ["Fe1 1\nFe2 1", "Fe1 1 0 0\nFe2 1 0 0"])
def test_primed_snapshot_must_preserve_scalar_or_axial_reference_moments(
    tmp_path, moments
):
    source = tmp_path / "bad-reference.shells.toml"
    source.write_text(primed_header(moments))
    with pytest.raises(ValueError, match="reference_spins.moments.*operation"):
        read_shells(source)


def test_producer_drift_is_report_only_and_never_rewrites_snapshot(tmp_path):
    from TB2J.io_exchange.shells import check_producer_ops, validate_shells

    source = tmp_path / "stored-p1.shells.toml"
    source.write_text(one_atom_header())
    document = read_shells(source)
    original = document.operations
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
    entry = f'\n[[shell]]\nrepresentative = "Fe1 Fe2 0 0 0"\nJ = {tensor}\n{channel}_members = []\n'
    header = HEADER.replace('exchange = "isotropic"', 'exchange = "tensor"')
    source.write_text(header + entry)
    with pytest.raises(ValueError, match=f"{channel}_members.*nonzero"):
        read_shells(source)
    zero = "[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]"
    source.write_text(header + entry.replace(tensor, zero))
    assert getattr(read_shells(source).shells[0], channel + "_members") == ()


@pytest.mark.parametrize("value", ["nan", "inf", "true", "[1.0, 2.0]"])
def test_exchange_values_reject_nonfinite_boolean_and_wrong_shapes(tmp_path, value):
    source = tmp_path / "bad-value.shells.toml"
    source.write_text(HEADER + f'\n[[shell]]\nid = "1:2:0,0,0"\nJ = {value}\n')
    with pytest.raises(ValueError):
        read_shells(source)


def test_canonical_roundtrip_preserves_float_values_and_operation_order(tmp_path):
    inversion = """[[symmetry.operations]]
rotation = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
translation = [0.0, 0.0, 0.0]
time_reversal = false
"""
    source = tmp_path / "unordered.shells.toml"
    source.write_text(
        one_atom_header().replace("[tolerances]", inversion + "[tolerances]")
        + '\n[[shell]]\nid = "1:1:-1,0,0"\nrepresentative = "Fe1 Fe1 1 0 0"\nJ = 0.12345678901234567\n'
    )
    document = read_shells(source)
    first, second = tmp_path / "first.toml", tmp_path / "second.toml"
    write_shells(document, first)
    reloaded = read_shells(first)
    assert_preserved(document, reloaded)
    write_shells(reloaded, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("wire_value,inside", [("0.00005", True), ("0.00015", False)])
def test_mev_wire_values_straddle_the_absolute_ev_tolerance(
    tmp_path, wire_value, inside
):
    source = tmp_path / "precision.shells.toml"
    source.write_text(
        HEADER.replace("value_atol = 1.0e-8", "value_atol = 5.0e-8")
        + f'\n[[shell]]\nid = "1:2:0,0,0"\nJ = {wire_value}\n'
    )
    document = read_shells(source)
    assert (
        bool(np.isclose(document.shells[0].Jiso, 0.0, atol=document.value_atol, rtol=0))
        is inside
    )
    output = tmp_path / "precision-saved.shells.toml"
    write_shells(document, output)
    assert_preserved(document, read_shells(output))


def test_tensor_requires_explicit_representative_and_matrix(tmp_path):
    source = tmp_path / "mode.shells.toml"
    header = HEADER.replace('exchange = "isotropic"', 'exchange = "tensor"')
    source.write_text(header + '\n[[shell]]\nid = "1:2:0,0,0"\nJ = "unfilled"\n')
    with pytest.raises(ValueError, match="representative"):
        read_shells(source)
    source.write_text(
        header + '\n[[shell]]\nrepresentative = "Fe1 Fe2 0 0 0"\nJ = 1.0\n'
    )
    with pytest.raises(ValueError, match="tensor"):
        read_shells(source)


def test_id_must_be_full_direct_orbit_minimum_not_representative(tmp_path):
    source = tmp_path / "nonminimum.shells.toml"
    source.write_text(centered_header() + '\n[[shell]]\nid = "2:1:1,0,0"\nJ = 1.0\n')
    with pytest.raises(ValueError, match="2:1:1,0,0.*1:2:0,0,0"):
        read_shells(source)


def test_python_edits_reject_complex_exchange_values(tmp_path):
    source = tmp_path / "real.shells.toml"
    source.write_text(HEADER + '\n[[shell]]\nid = "1:2:0,0,0"\nJ = 1.0\n')
    document = read_shells(source)
    with pytest.raises(ValueError, match="complex"):
        replace(document, shells=(replace(document.shells[0], Jiso=1.0 + 2.0j),))


@pytest.mark.parametrize("moments", ["Fe1 1\nFe2 -1", "Fe1 1 0 0\nFe2 -1 0 0"])
def test_valid_primed_half_translation_snapshot_roundtrips(tmp_path, moments):
    source = tmp_path / "afm.shells.toml"
    source.write_text(
        primed_header(moments) + '\n[[shell]]\nid = "1:2:0,0,0"\nJ = -2.0\n'
    )
    document = read_shells(source)
    output = tmp_path / "afm-saved.shells.toml"
    write_shells(document, output)
    assert_preserved(document, read_shells(output))
    assert tuple(operation.time_reversal for operation in document.operations) == (
        False,
        True,
    )


def tensor_document(tmp_path, value="100.0", masks=""):
    source = tmp_path / "tensor.shells.toml"
    source.write_text(
        HEADER.replace('exchange = "isotropic"', 'exchange = "tensor"')
        + f'\n[[shell]]\nrepresentative = "Fe1 Fe2 0 0 0"\nJ = [[{value}, 0, 0], [0, {value}, 0], [0, 0, {value}]]\n{masks}'
    )
    return read_shells(source)


def test_python_tensor_edit_cannot_coerce_mixed_booleans_into_exchange(tmp_path):
    document = tensor_document(tmp_path)
    invalid = ((True, 0.0, 0.0), (0.0, 0.001, 0.0), (0.0, 0.0, 0.001))
    with pytest.raises(ValueError, match="boolean"):
        replace(document, shells=(replace(document.shells[0], tensor=invalid),))


def test_isotropic_tensor_preserves_exactly_absent_anisotropy(tmp_path):
    document = tensor_document(tmp_path, masks="DMI_members = []\nJani_members = []\n")
    tensor = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    edited = replace(document, shells=(replace(document.shells[0], tensor=tensor),))
    output = tmp_path / "saved.shells.toml"
    write_shells(edited, output)
    assert_preserved(edited, read_shells(output))
    assert edited.shells[0].DMI_members == edited.shells[0].Jani_members == ()
    nonzero = ((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.10000000000000002))
    with pytest.raises(ValueError, match="Jani_members.*nonzero"):
        replace(document, shells=(replace(document.shells[0], tensor=nonzero),))


def test_boolean_symprec_is_not_numeric_tolerance(tmp_path):
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
    source.write_text(
        header + ('\n[[shell]]\nid = "1:2:0,0,0"\nJ = 1.0\n' if with_shell else "")
    )
    with pytest.raises(ValueError, match="symmetry.operations.*sg.*msg"):
        read_shells(source)


def test_explicit_representative_must_be_in_sparse_support(tmp_path):
    source = tmp_path / "sparse-tensor.shells.toml"
    header = centered_header().replace('exchange = "isotropic"', 'exchange = "tensor"')
    entry = '\n[[shell]]\nid = "1:2:0,0,0"\nrepresentative = "Fe1 Fe2 0 0 0"\nJ = "unfilled"\nsupport_members = [1]\n'
    source.write_text(header + entry)
    with pytest.raises(ValueError, match="representative.*support"):
        read_shells(source)
    source.write_text(header + entry.replace("Fe1 Fe2 0 0 0", "Fe2 Fe1 1 0 0"))
    document = read_shells(source)
    assert document.shells[0].id == "1:2:0,0,0"
    assert document.shells[0].representative == (2, 1, 1, 0, 0)


def test_empty_scalar_support_roundtrips_without_inventing_observed_pairs(tmp_path):
    path = tmp_path / "empty-support.shells.toml"
    path.write_text(
        centered_header()
        + '\n[[shell]]\nid = "1:2:0,0,0"\nJ = 0\nsupport_members = []\n'
    )
    document = read_shells(path)
    write_shells(document, path)
    restored = read_shells(path)
    assert restored.shells[0].support_members == ()
    assert restored.shells[0].Jiso == 0.0
    assert restored.shells[0].n_equivalent_pairs == 2
