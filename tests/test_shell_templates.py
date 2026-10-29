"""Consumer regressions for structure-derived shell templates."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells


def test_structure_generation_selects_species_and_preserves_mapping(tmp_path):
    from TB2J.io_exchange.shell_workflow import generate_shells

    atoms = Atoms(
        ["Fe", "O", "Fe"],
        cell=np.eye(3) * 4,
        scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25), (0.5, 0.5, 0.5)],
        pbc=True,
    )
    source = tmp_path / "POSCAR"
    atoms.write(source, format="vasp")
    by_species = tmp_path / "species.toml"
    by_atoms = tmp_path / "atoms.toml"
    species_result = generate_shells(
        source, by_species, magnetic_species=["Fe"], cutoff=3.6
    )
    atom_result = generate_shells(source, by_atoms, magnetic_atoms=[1, 3], cutoff=3.6)
    assert species_result.document.index_spin == (0, -1, 1)
    assert species_result.document == atom_result.document
    assert shells.read_shells(by_species) == species_result.document
    assert all(entry.Jiso == "unfilled" for entry in species_result.document.shells)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"magnetic_species": ["Fe"], "magnetic_atoms": [1]},
        {"magnetic_species": ["Co"]},
        {"magnetic_species": ["NotAnElement"]},
        {"magnetic_atoms": [1, 1]},
        {"magnetic_atoms": [2]},
        {"magnetic_species": ["Fe"], "cutoff": -1},
        {"magnetic_species": ["Fe"], "symmetry": "msg"},
        {"magnetic_species": ["Fe"], "mode": ""},
        {"magnetic_species": ["Fe"], "mode": False},
        {"magnetic_species": ["Fe"], "mode": 0},
    ],
)
def test_structure_generation_refuses_invalid_context_without_overwriting(
    tmp_path, options
):
    from TB2J.io_exchange.shell_workflow import generate_shells

    source = tmp_path / "POSCAR"
    Atoms("Fe", cell=np.eye(3) * 2, scaled_positions=[(0, 0, 0)], pbc=True).write(
        source, format="vasp"
    )
    output = tmp_path / "shells.toml"
    output.write_text("untouched")
    options = {"cutoff": 2.01, **options}
    with pytest.raises(ValueError):
        generate_shells(source, output, **options)
    assert output.read_text() == "untouched"


def test_structure_generation_msg_tensor_retains_reference_and_tolerances(tmp_path):
    from TB2J.io_exchange.shell_workflow import generate_shells

    source = tmp_path / "POSCAR"
    Atoms("Fe", cell=np.eye(3) * 2, scaled_positions=[(0, 0, 0)], pbc=True).write(
        source, format="vasp"
    )
    output = tmp_path / "shells.toml"
    result = generate_shells(
        source,
        output,
        magnetic_species=["Fe"],
        cutoff=2.01,
        symmetry="msg",
        mode="tensor",
        reference_spins=[[0, 0, 2]],
        symprec=2e-5,
        magnetic_tol=3e-5,
    )
    parsed = shells.read_shells(output)
    assert parsed == result.document
    assert parsed.moments == ((0.0, 0.0, 2.0),)
    assert parsed.symprec == 2e-5 and parsed.magnetic_tol == 3e-5
    assert all(
        entry.tensor == "unfilled" and entry.representative for entry in parsed.shells
    )


def test_shell_comments_explain_editing_and_can_be_removed(tmp_path):
    atoms = Atoms("Fe", cell=np.eye(3) * 2, scaled_positions=[(0, 0, 0)], pbc=True)
    document = shells.template(atoms, [1], 2.01, mode="tensor")
    with_pairs = tmp_path / "pairs.toml"
    without_pairs = tmp_path / "without-pairs.toml"
    without_comments = tmp_path / "without-comments.toml"
    shells.write_shells(document, with_pairs)
    shells.write_shells(document, without_pairs, equivalent_comments=False)
    explained = without_pairs.read_text()
    without_comments.write_text(
        "\n".join(
            line for line in explained.splitlines() if not line.lstrip().startswith("#")
        )
    )
    assert shells.read_shells(with_pairs) == shells.read_shells(without_pairs)
    assert shells.read_shells(without_comments) == document


def test_cubic_template_is_one_full_direct_neighbor_orbit(tmp_path):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    document = shells.template(atoms, [1], 2.01, mode="tensor")
    assert document.index_spin == (0,)
    assert [
        (entry.id, entry.n_equivalent_pairs, entry.tensor, entry.representative)
        for entry in document.shells
    ] == [("1:1:-1,0,0", 6, "unfilled", (1, 1, -1, 0, 0))]
    output = tmp_path / "fe.shells.toml"
    shells.write_shells(document, output)
    assert shells.read_shells(output) == document


def test_sg_joins_and_msg_splits_cubic_bond_orbits():
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    sg = shells.template(atoms, [1], 2.01)
    msg = shells.template(
        atoms,
        [1],
        2.01,
        symmetry="msg",
        reference_spins=[[0.0, 0.0, 2.0]],
    )

    assert len(sg.shells) == 1
    assert sg.shells[0].n_equivalent_pairs == 6
    assert len(msg.shells) == 2
    assert sorted(entry.n_equivalent_pairs for entry in msg.shells) == [2, 4]
    assert all(entry.representative is not None for entry in sg.shells + msg.shells)
    assert all(
        entry.tensor is None and entry.Jiso == "unfilled"
        for entry in sg.shells + msg.shells
    )


def test_msg_requires_complete_reference_spin_shape():
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)

    with pytest.raises(ValueError, match=r"reference_spins\.moments.*required.*msg"):
        shells.template(atoms, [1], 2.01, symmetry="msg")
    with pytest.raises(ValueError, match=r"reference_spins\.moments.*shape"):
        shells.template(atoms, [1], 2.01, symmetry="msg", reference_spins=[[1.0, 2.0]])


def test_scalar_primed_msg_and_axial_vector_moments():
    afm = Atoms(
        "Fe2",
        cell=np.diag([4.0, 4.0, 2.0]),
        scaled_positions=[(0, 0, 0), (0, 0, 0.5)],
        pbc=True,
    )
    scalar = shells.template(
        afm, [1, 2], 1.01, symmetry="msg", reference_spins=[2.0, -2.0]
    )
    half_translation = [
        operation
        for operation in scalar.operations
        if np.array_equal(operation.rotation, np.eye(3, dtype=int))
        and np.allclose(operation.translation, [0.0, 0.0, 0.5])
    ]
    assert len(half_translation) == 1
    assert half_translation[0].time_reversal is True
    assert scalar.moments == (2.0, -2.0)

    cubic = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    vector = shells.template(
        cubic, [1], 2.01, symmetry="msg", reference_spins=[[0.0, 0.0, 2.0]]
    )
    inversion = [
        operation
        for operation in vector.operations
        if np.array_equal(operation.rotation, -np.eye(3, dtype=int))
        and np.allclose(operation.translation, [0.0, 0.0, 0.0])
    ]
    primed_mirror = [
        operation
        for operation in vector.operations
        if np.array_equal(operation.rotation, np.diag([-1, 1, 1]))
    ]
    assert len(inversion) == 1 and inversion[0].time_reversal is False
    assert len(primed_mirror) == 1 and primed_mirror[0].time_reversal is True
    moment = np.array([0.0, 0.0, 2.0])
    for operation in vector.operations:
        cartesian = (
            np.asarray(cubic.cell).T
            @ np.asarray(operation.rotation)
            @ np.linalg.inv(np.asarray(cubic.cell).T)
        )
        sign = -1.0 if operation.time_reversal else 1.0
        transformed = sign * np.linalg.det(cartesian) * cartesian @ moment
        np.testing.assert_allclose(transformed, moment, atol=vector.magnetic_tol)


def test_cutoff_in_angstrom_is_monotone():
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    small = shells.template(atoms, [1], 2.1)
    larger = shells.template(atoms, [1], 2.9)

    small_by_id = {entry.id: entry for entry in small.shells}
    larger_by_id = {entry.id: entry for entry in larger.shells}
    assert set(small_by_id) < set(larger_by_id)
    for identifier, entry in small_by_id.items():
        assert larger_by_id[identifier] == entry


def test_skew_cell_images_match_independent_brute_force_oracle():
    cell = np.array([[3.0, 0.0, 0.0], [2.95, 0.1, 0.0], [0.37, 0.18, 3.0]])
    atoms = Atoms(
        ["Fe", "O"],
        cell=cell,
        scaled_positions=[(0.0, 0.0, 0.0), (0.137, 0.283, 0.419)],
        pbc=True,
    )
    cutoff = 0.46
    document = shells.template(atoms, [1], cutoff)

    expected = {
        translation
        for translation in product(range(-8, 9), repeat=3)
        if translation != (0, 0, 0)
        and np.linalg.norm(np.asarray(translation) @ cell) <= cutoff
    }
    assert expected
    assert len(document.operations) == 1
    actual = {
        tuple(entry.representative[2:])
        for entry in document.shells
        if entry.representative[:2] == (1, 1)
    }
    assert actual == expected


def test_nonmagnetic_atoms_keep_atom_numbering_and_spin_indices():
    atoms = Atoms(
        ["Fe", "O", "Fe"],
        cell=np.eye(3) * 10.0,
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.231, 0.187, 0.319),
            (0.08, 0.03, 0.02),
        ],
        pbc=True,
    )
    document = shells.template(atoms, [3, 1], 1.2)

    assert document.magnetic_atoms == (1, 3)
    assert document.index_spin == (0, -1, 1)
    pair = next(
        entry for entry in document.shells if entry.representative[:2] == (1, 3)
    )
    assert pair.representative[:2] == (1, 3)
    invalid_pair = replace(pair, representative=(2, *pair.representative[1:]))
    invalid_shells = tuple(
        invalid_pair if entry.id == pair.id else entry for entry in document.shells
    )
    with pytest.raises(ValueError, match="representative endpoints must be magnetic"):
        replace(document, shells=invalid_shells)


def test_tensor_templates_roundtrip_with_optional_comments_and_export_guard(tmp_path):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    document = shells.template(atoms, [1], 2.01, mode="tensor")
    with_comments = tmp_path / "with-comments.shells.toml"
    without_comments = tmp_path / "without-comments.shells.toml"

    assert all(
        entry.tensor == "unfilled" and entry.representative is not None
        for entry in document.shells
    )
    shells.write_shells(document, with_comments)
    shells.write_shells(document, without_comments, equivalent_comments=False)
    assert shells.read_shells(with_comments) == document
    assert shells.read_shells(without_comments) == document
    with pytest.raises(ValueError, match=r"shell .*\.tensor: unfilled"):
        shells.validate_shells(document, for_export=True)


def test_msg_reference_context_survives_saved_replay_and_sg_stays_absent(tmp_path):
    atoms = Atoms(
        ["Fe", "O"],
        cell=np.diag([4.0, 5.0, 6.0]),
        scaled_positions=[(0.0, 0.0, 0.0), (0.173, 0.287, 0.391)],
        pbc=True,
    )
    moments = [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
    msg = shells.template(atoms, [1], 2.1, symmetry="msg", reference_spins=moments)
    msg_path = tmp_path / "msg.shells.toml"
    shells.write_shells(msg, msg_path)
    replayed = shells.read_shells(msg_path)
    assert replayed.moments == ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
    assert replayed.moment_source == "explicit"
    assert shells.check_producer_ops(replayed).producer_drift is False

    sg = shells.template(atoms, [1], 2.1)
    sg_path = tmp_path / "sg.shells.toml"
    shells.write_shells(sg, sg_path)
    assert shells.read_shells(sg_path).moments is None


def test_template_serialization_is_byte_deterministic(tmp_path):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    first = shells.template(atoms, [1], 2.9, mode="tensor")
    second = shells.template(atoms, [1], 2.9, mode="tensor")
    first_path = tmp_path / "first.shells.toml"
    second_path = tmp_path / "second.shells.toml"

    shells.write_shells(first, first_path)
    shells.write_shells(second, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize("selection", ([True], [1.0], [0], [2], [1, 1]))
def test_magnetic_selection_rejects_implicit_integer_or_range_coercion(selection):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    with pytest.raises(ValueError, match="magnetic_atoms"):
        shells.template(atoms, selection, 2.01)


@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"cutoff": True}, "cutoff"),
        ({"cutoff": np.inf}, "cutoff"),
        ({"symprec": 0.0}, "symmetry.symprec"),
        ({"value_atol": np.nan}, "tolerances.value_atol"),
        ({"mode": "tensor-vector"}, "mode"),
    ],
)
def test_template_rejects_invalid_numbers_and_modes(options, field):
    atoms = Atoms("Fe", cell=np.eye(3) * 2.0, scaled_positions=[(0, 0, 0)], pbc=True)
    cutoff = options.get("cutoff", 2.01)
    arguments = {name: value for name, value in options.items() if name != "cutoff"}

    with pytest.raises(ValueError, match=field):
        shells.template(atoms, [1], cutoff, **arguments)
