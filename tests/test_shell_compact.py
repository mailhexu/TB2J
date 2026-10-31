"""End-to-end contract for compact editable symmetry shells."""

import re
from dataclasses import replace

import numpy as np
import pytest
import tomli
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO
from TB2J.io_exchange.shell_workflow import export_shells, generate_shells


def test_writer_rejects_unrepresentable_spin_mapping(tmp_path):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[4, 5, 6],
        scaled_positions=[(0, 0, 0), (0.2, 0.3, 0.4)],
        pbc=True,
    )
    document = shells.template(atoms, [1, 2], 3)
    document = replace(document, index_spin=(1, 0))
    path = tmp_path / "mapping.toml"
    with pytest.raises(ValueError, match="noncanonical spin mapping"):
        shells.write_shells(document, path)
    assert not path.exists()


def test_pickle_shell_can_use_an_unobserved_reverse_representative(tmp_path):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[[4.1, 0, 0], [0.3, 5.2, 0], [0.2, 0.4, 6.3]],
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    forward = ((1, 0, 0), 0, 1)
    reverse = ((-1, 0, 0), 1, 0)
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2], [0, 0, 1]],
        charges=None,
        index_spin=[0, 1],
        colinear=True,
        exchange_Jdict={forward: 0.001},
    )
    source.damping = source.gyro_ratio = None
    source.write_pickle(tmp_path / "source")
    shell_path = tmp_path / "shell.toml"
    generate_shells(tmp_path / "source" / "TB2J.pickle", shell_path)
    text = shell_path.read_text()
    shell = tomli.loads(text)["shell"][0]
    assert (
        not {"id", "support_members", "Jiso_members", "DMI_members", "Jani_members"}
        & shell.keys()
    )
    old_pair = shell["representative"]
    text = text.replace(
        f'representative = "{old_pair}"', 'representative = "Co1 Fe1 -1 0 0"', 1
    )
    text = text.replace("J = 1.00000000", "J = 2.00000000", 1)
    shell_path.write_text(text)
    shells.write_shells(shells.read_shells(shell_path), shell_path)
    assert (
        "Generated pair witnesses from representative Co1 Fe1 -1 0 0."
        in shell_path.read_text()
    )
    output = tmp_path / "result"
    export_shells(shell_path, output)
    replay = SpinIO.load_pickle(output)
    assert set(replay.exchange_Jdict) == {forward, reverse}
    np.testing.assert_allclose(
        [replay.exchange_Jdict[forward], replay.exchange_Jdict[reverse]], [0.002, 0.002]
    )
    assert not replay.dmi_ddict and not replay.Jani_dict


def test_tensor_reverse_representative_preserves_sparse_channels(tmp_path):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[[4.1, 0, 0], [0.3, 5.2, 0], [0.2, 0.4, 6.3]],
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    forward = ((1, 0, 0), 0, 1)
    reverse = ((-1, 0, 0), 1, 0)
    dmi = np.array([0.002, -0.003, 0.004])
    anisotropy = np.array(
        [[0.003, 0.001, -0.002], [0.001, -0.001, 0.003], [-0.002, 0.003, -0.002]]
    )
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2], [0, 0, 1]],
        charges=None,
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict={},
        dmi_ddict={forward: dmi},
        Jani_dict={forward: anisotropy},
    )
    source.damping = source.gyro_ratio = None
    source.write_pickle(tmp_path / "source")
    shell_path = tmp_path / "tensor.toml"
    generate_shells(tmp_path / "source" / "TB2J.pickle", shell_path)
    text = shell_path.read_text()
    entry = tomli.loads(text)["shell"][0]
    assert entry["representative"] == "Fe1 Co1 1 0 0"
    matrix = np.asarray(entry["J"]).T
    replacement = (
        "J = [\n"
        + "".join(
            "  [" + ", ".join(f"{value:.8f}" for value in row) + "],\n"
            for row in matrix
        )
        + "]"
    )
    text, count = re.subn(r"(?ms)^J = \[\n.*?^\]", replacement, text, count=1)
    assert count == 1
    text = text.replace(
        'representative = "Fe1 Co1 1 0 0"', 'representative = "Co1 Fe1 -1 0 0"', 1
    )
    shell_path.write_text(text)
    export_shells(shell_path, tmp_path / "result")
    replay = SpinIO.load_pickle(tmp_path / "result")
    assert not replay.exchange_Jdict
    assert set(replay.dmi_ddict) == {forward, reverse}
    assert set(replay.Jani_dict) == {forward, reverse}
    np.testing.assert_allclose(replay.dmi_ddict[forward], dmi, atol=1e-12)
    np.testing.assert_allclose(replay.dmi_ddict[reverse], -dmi, atol=1e-12)
    np.testing.assert_allclose(replay.Jani_dict[forward], anisotropy, atol=1e-12)
    np.testing.assert_allclose(replay.Jani_dict[reverse], anisotropy.T, atol=1e-12)


def test_tensor_on_unobserved_rotated_pair_keeps_original_support(tmp_path):
    atoms = Atoms("Fe", cell=[2, 2, 2], scaled_positions=[[0, 0, 0]], pbc=True)
    forward = ((1, 0, 0), 0, 0)
    reverse = ((-1, 0, 0), 0, 0)
    anisotropy = np.diag([0.002, -0.001, -0.001])
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2]],
        charges=None,
        index_spin=[0],
        colinear=False,
        exchange_Jdict={},
        Jani_dict={forward: anisotropy},
    )
    source.damping = source.gyro_ratio = None
    source.write_pickle(tmp_path / "source")
    shell_path = tmp_path / "rotated.toml"
    generate_shells(tmp_path / "source" / "TB2J.pickle", shell_path)
    text = shell_path.read_text()
    entry = tomli.loads(text)["shell"][0]
    assert entry["representative"] == "Fe1 Fe1 1 0 0"
    text = text.replace(
        'representative = "Fe1 Fe1 1 0 0"', 'representative = "Fe1 Fe1 0 1 0"', 1
    )
    tensor = "J = [\n  [-1.0, 0.0, 0.0],\n  [0.0, 2.0, 0.0],\n  [0.0, 0.0, -1.0],\n]"
    text, count = re.subn(r"(?ms)^J = \[\n.*?^\]", tensor, text, count=1)
    assert count == 1
    shell_path.write_text(text)
    export_shells(shell_path, tmp_path / "result")
    replay = SpinIO.load_pickle(tmp_path / "result")
    assert not replay.exchange_Jdict and not replay.dmi_ddict
    assert set(replay.Jani_dict) == {forward, reverse}
    np.testing.assert_allclose(replay.Jani_dict[forward], anisotropy, atol=1e-12)
    np.testing.assert_allclose(replay.Jani_dict[reverse], anisotropy, atol=1e-12)


def test_observed_zero_is_not_an_absent_exchange_channel(tmp_path):
    atoms = Atoms("Fe", cell=[2, 2, 2], scaled_positions=[[0, 0, 0]], pbc=True)
    forward = ((1, 0, 0), 0, 0)
    reverse = ((-1, 0, 0), 0, 0)
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2]],
        charges=None,
        index_spin=[0],
        colinear=True,
        exchange_Jdict={forward: 0.0},
    )
    source.damping = source.gyro_ratio = None
    source.write_pickle(tmp_path / "source")
    shell_path = tmp_path / "zero.toml"
    generate_shells(tmp_path / "source" / "TB2J.pickle", shell_path)
    export_shells(shell_path, tmp_path / "result")
    replay = SpinIO.load_pickle(tmp_path / "result")
    assert replay.exchange_Jdict == {forward: 0.0, reverse: 0.0}
    assert not replay.dmi_ddict and not replay.Jani_dict


def test_native_spin_mapping_is_hidden_and_preserved_through_export(tmp_path):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[[4.1, 0, 0], [0.3, 5.2, 0], [0.2, 0.4, 6.3]],
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    forward = ((1, 0, 0), 1, 0)
    reverse = ((-1, 0, 0), 0, 1)
    source = SpinIO(
        atoms,
        spinat=[[0, 0, 2], [0, 0, 1]],
        charges=None,
        index_spin=[1, 0],
        colinear=True,
        exchange_Jdict={forward: 0.001},
    )
    source.damping = source.gyro_ratio = None
    source.write_pickle(tmp_path / "source")
    shell_path = tmp_path / "mapping.toml"
    generate_shells(tmp_path / "source" / "TB2J.pickle", shell_path)
    atoms_block = tomli.loads(shell_path.read_text())["structure"]["atoms"]
    atom_rows = [
        line.split()
        for line in atoms_block.splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    assert all(len(row) == 6 for row in atom_rows)
    output = tmp_path / "result"
    export_shells(shell_path, output)
    replay = SpinIO.load_pickle(output)
    assert list(replay.index_spin) == [1, 0]
    assert replay.exchange_Jdict == {forward: 0.001, reverse: 0.001}


def test_pair_comments_name_operations_that_transform_the_bond(tmp_path):
    atoms = Atoms("Fe", cell=[2, 2, 2], scaled_positions=[[0, 0, 0]], pbc=True)
    document = shells.template(atoms, [1], 2.01)
    path = tmp_path / "template.toml"
    shells.write_shells(document, path)
    text = path.read_text()
    data = tomli.loads(text)
    representative = data["shell"][0]["representative"].split()
    initial = np.array([int(value) for value in representative[2:]])
    rows = re.findall(
        r"(?m)^#\s+Fe1\s+Fe1\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(true|false)\s*$",
        text,
    )
    assert len(rows) == 3
    for x, y, z, operation, reversed_pair in rows:
        op = data["symmetry"]["operations"][int(operation) - 1]
        result = np.asarray(op["rotation"]) @ initial
        if reversed_pair == "true":
            result = -result
        np.testing.assert_array_equal(result, [int(x), int(y), int(z)])
