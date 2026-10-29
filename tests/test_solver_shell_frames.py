"""Solver outputs preserve tensor components and native pair geometry."""

import re

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.Jtensor import combine_J_tensor


def test_tomasd_serializes_full_cartesian_tensor(tmp_path):
    key = ((1, 0, 0), 0, 0)
    dmi = np.array([0.0003, -0.0002, 0.0004])
    jani = np.array(
        [[0.0005, 0.0001, 0.0002], [0.0001, -0.0003, 0.0001], [0.0002, 0.0001, -0.0002]]
    )
    model = SpinIO(
        atoms=Atoms("Fe", cell=[3, 3, 3], pbc=True),
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=None,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
        dmi_ddict={key: dmi},
        Jani_dict={key: jani},
    )
    model.write_tom_format(tmp_path / "TomASD")
    text = (tmp_path / "TomASD" / "exchange.exch").read_text()
    rows = []
    for row in range(1, 4):
        match = re.search(rf"J1_{row}\s*=\s*\[([^]]+)\]", text)
        assert match is not None
        rows.append([float(item) for item in match.group(1).split(",")])
    np.testing.assert_allclose(
        rows, combine_J_tensor(Jiso=0.002, D=dmi, Jani=jani), rtol=1e-8, atol=1e-12
    )


def test_tomasd_rebases_wrapped_bond_without_changing_source(tmp_path):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[[3, 0, 0], [0.4, 4, 0], [0.1, 0.2, 5]],
        scaled_positions=[[-0.3, 0.2, 0.1], [1.2, 0.4, 0.2]],
        pbc=True,
    )
    key = ((0, 0, 0), 1, 0)
    model = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]),
        charges=None,
        index_spin=[1, 0],
        exchange_Jdict={key: 0.002},
    )
    original = atoms.positions.copy()
    model.write_tom_format(tmp_path / "TomASD")
    ucf = (tmp_path / "TomASD" / "exchange.ucf").read_text().splitlines()[2:]
    positions = {
        int(row.split()[0]): np.array([float(x) for x in row.split()[1:4]])
        for row in ucf
    }
    text = (tmp_path / "TomASD" / "exchange.exch").read_text()
    translation = re.search(r"UnitCell1\s*=\s*\[([^]]+)\]", text)
    assert translation is not None
    R = np.array([float(x) for x in translation.group(1).split(",")])
    actual = (positions[0] + R - positions[1]) @ atoms.cell.array
    np.testing.assert_allclose(
        actual, atoms.positions[1] - atoms.positions[0], atol=1e-12
    )
    np.testing.assert_array_equal(atoms.positions, original)
    assert list(model.exchange_Jdict) == [key]
    model.write_espins(tmp_path / "ESPInS")
    esp = (tmp_path / "ESPInS" / "espins.in").read_text()
    match = re.search(r"f1=\s*([^:]+):f2=\s*([^:]+):jij=", esp)
    assert match is not None
    f1 = np.array([float(x) for x in match.group(1).split(",")])
    f2 = np.array([float(x) for x in match.group(2).split(",")])
    np.testing.assert_allclose(
        (f2 - f1) @ atoms.cell.array, atoms.positions[1] - atoms.positions[0], atol=1e-6
    )


def test_standard_writers_preserve_tensor_and_rebased_bond(tmp_path):
    from ase.units import J

    cell = np.diag([3.0, 4.0, 5.0])
    atoms = Atoms(
        ["Fe", "Co"],
        cell=cell,
        scaled_positions=[[-0.3, 0.2, 0.1], [1.2, 0.4, 0.2]],
        pbc=True,
    )
    key = ((0, 0, 0), 1, 0)
    dmi = np.array([0.0003, -0.0002, 0.0004])
    jani = np.array(
        [[0.0005, 0.0001, 0.0002], [0.0001, -0.0003, 0.0001], [0.0002, 0.0001, -0.0002]]
    )
    dmi.setflags(write=False)
    jani.setflags(write=False)
    model = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]),
        charges=None,
        index_spin=[1, 0],
        exchange_Jdict={key: 0.002},
        dmi_ddict={key: dmi},
        Jani_dict={key: jani},
    )
    original = atoms.positions.copy()
    expected_bond = original[1] - original[0]
    expected_tensor = combine_J_tensor(Jiso=0.002, D=dmi, Jani=jani)
    path = tmp_path / "results"
    report = model.write_all(path)
    assert {"TomASD", "Vampire", "ESPInS"} <= set(report.written)
    assert "Multibinit" in dict(report.omitted)
    actual_paths = {
        str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
    }
    assert set(report.written_paths) == actual_paths
    assert "Multibinit/exchange.xml" in report.omitted_paths

    tom = (path / "TomASD" / "exchange.exch").read_text()
    tom_rows = [
        re.search(rf"J1_{row}\s*=\s*\[([^]]+)\]", tom).group(1) for row in range(1, 4)
    ]
    np.testing.assert_allclose(
        [[float(x) for x in row.split(",")] for row in tom_rows],
        expected_tensor,
        atol=1e-12,
    )

    vampire = (path / "Vampire" / "vampire.UCF").read_text().splitlines()
    atom_start = vampire.index("# Atoms")
    interaction_start = vampire.index("# Interactions")
    vpos = {
        int(row.split()[0]): np.array([float(x) for x in row.split()[1:4]])
        for row in vampire[atom_start + 2 : interaction_start]
    }
    interaction = vampire[interaction_start + 2].split()
    vi, vj = int(interaction[1]), int(interaction[2])
    vR = np.array([int(x) for x in interaction[3:6]])
    lengths = np.fromstring(vampire[1], sep=" ")
    np.testing.assert_allclose(
        (vpos[vj] + vR - vpos[vi]) * lengths, expected_bond, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(interaction[6:], dtype=float).reshape(3, 3) * J / 2,
        expected_tensor,
        atol=1e-8,
    )
    directions = np.array([np.fromstring(row, sep=" ") for row in vampire[3:6]])
    np.testing.assert_allclose(lengths[:, None] * directions, cell, atol=5e-6)

    esp = (path / "ESPInS" / "espins.in").read_text()
    match = re.search(
        r"f1=\s*([^:]+):f2=\s*([^:]+):jij=.*?:t1=\s*(\d+):t2=\s*(\d+)", esp
    )
    assert match is not None
    f1 = np.array([float(x) for x in match.group(1).split(",")])
    f2 = np.array([float(x) for x in match.group(2).split(",")])
    assert (int(match.group(3)), int(match.group(4))) == (2, 1)
    np.testing.assert_allclose((f2 - f1) @ cell, expected_bond, atol=1e-6)
    matrix = esp.split("Begin Jij_matrix\n", 1)[1].splitlines()[0]
    values = [float(x) for x in re.findall(r"J..=([-+0-9.eE]+)", matrix)]
    np.testing.assert_allclose(
        np.array(values).reshape(3, 3) / 2, expected_tensor, atol=1e-8
    )
    atom_rows = (
        esp.split("Begin Atoms_Frac\n", 1)[1].split("End Atoms_Frac", 1)[0].splitlines()
    )
    assert [row.split()[0] for row in atom_rows] == ["Co", "Fe"]
    np.testing.assert_array_equal(atoms.positions, original)
    assert list(model.exchange_Jdict) == [key] and model.index_spin == [1, 0]


def test_results_inventory_names_only_emitted_leaves_and_omission_candidates(tmp_path):
    atom = Atoms("Fe", cell=[3, 3, 3], pbc=True)
    key = ((1, 0, 0), 0, 0)
    model = SpinIO(
        atoms=atom,
        spinat=None,
        charges=None,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
    )
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "user.txt").write_text("keep")
    report = model.write_all(destination)
    assert set(report.written_paths) == {
        "TB2J.pickle",
        "structure.vasp",
        "exchange.out",
        "JvsR.pdf",
    }
    assert "TomASD/exchange.exch" in report.omitted_paths
    assert "user.txt" not in report.written_paths
    assert (destination / "user.txt").read_text() == "keep"


def test_tomasd_keeps_reversed_pair_tensor_transpose(tmp_path):
    atoms = Atoms(["Fe", "Co"], cell=[3, 4, 5], pbc=True)
    forward, reverse = ((1, 0, 0), 0, 1), ((-1, 0, 0), 1, 0)
    d = np.array([0.0002, 0.0003, -0.0004])
    model = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]]),
        charges=None,
        index_spin=[0, 1],
        exchange_Jdict={forward: 0.002, reverse: 0.002},
        dmi_ddict={forward: d, reverse: -d},
    )
    model.write_tom_format(tmp_path / "TomASD")
    text = (tmp_path / "TomASD" / "exchange.exch").read_text()
    matrices = []
    for pair in ("exchange_0_1:", "exchange_1_0:"):
        block = text.split(pair, 1)[1].split("};", 1)[0]
        rows = [
            re.search(rf"J1_{row}\s*=\s*\[([^]]+)\]", block).group(1)
            for row in range(1, 4)
        ]
        matrices.append(np.array([[float(x) for x in row.split(",")] for row in rows]))
    np.testing.assert_allclose(matrices[0], matrices[1].T, atol=1e-12)
    np.testing.assert_allclose(
        matrices[0], combine_J_tensor(Jiso=0.002, D=d), atol=1e-12
    )


def test_vampire_rejects_skew_cell_without_writing_partial_target(tmp_path):
    atoms = Atoms("Fe", cell=[[3, 0, 0], [0.4, 4, 0], [0, 0, 5]], pbc=True)
    key = ((1, 0, 0), 0, 0)
    model = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=None,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
    )
    destination = tmp_path / "results"
    report = model.write_all(destination)
    assert "cell" in dict(report.omitted)["Vampire"].lower()
    assert not (destination / "Vampire").exists()
    with pytest.raises(ValueError, match="Vampire.*cell"):
        model.write_vampire(tmp_path / "direct")
    assert not (tmp_path / "direct").exists()
