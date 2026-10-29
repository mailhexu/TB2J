"""End-to-end export protects input data and owned standard results."""

import os
import re
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.units import J

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO
from TB2J.io_exchange.shell_workflow import export_shells, generate_shells


def _shell(tmp_path, name, moments=True):
    directory = tmp_path / name
    directory.mkdir()
    atom = Atoms("Fe", cell=[3, 3, 3], pbc=True)
    source = SpinIO(
        atoms=atom,
        spinat=np.array([[0.0, 0.0, 2.0]]) if moments else None,
        charges=None,
        index_spin=[0],
        exchange_Jdict={
            ((1, 0, 0), 0, 0): 0.002,
            ((-1, 0, 0), 0, 0): 0.002,
            ((0, 1, 0), 0, 0): 0.002,
            ((0, -1, 0), 0, 0): 0.002,
            ((0, 0, 1), 0, 0): 0.002,
            ((0, 0, -1), 0, 0): 0.002,
        },
    )
    source.write_pickle(path=directory)
    path = directory / "exchange.shells.toml"
    generate_shells(directory / "TB2J.pickle", path)
    (directory / "TB2J.pickle").unlink()
    return path


def test_export_relocated_native_and_update_removes_only_owned_stale_outputs(tmp_path):
    first = _shell(tmp_path, "first")
    second = _shell(tmp_path, "second", moments=False)
    destination = tmp_path / "results"
    result = export_shells(first, destination)
    assert "Vampire" in result.results.written
    reloaded = SpinIO.load_pickle(path=destination)
    assert reloaded.index_spin == [0]
    np.testing.assert_allclose(reloaded.spinat, [[0, 0, 2]])
    assert reloaded.exchange_Jdict[((1, 0, 0), 0, 0)] == pytest.approx(0.002)
    (destination / "user.txt").write_text("keep")
    updated = export_shells(second, destination)
    assert "Vampire" in dict(updated.results.omitted)
    assert not (destination / "Vampire" / "vampire.UCF").exists()
    assert (destination / "user.txt").read_text() == "keep"
    assert SpinIO.load_pickle(path=destination).spinat is None


def test_export_rejects_ambiguous_collision_before_replacing_anything(tmp_path):
    shell = _shell(tmp_path, "input")
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "exchange.out").write_text("user data")
    with pytest.raises(ValueError, match="ownership.*fresh output"):
        export_shells(shell, destination)
    assert (destination / "exchange.out").read_text() == "user data"
    assert not (destination / "TB2J.pickle").exists()


def test_export_rejects_input_hardlink_alias(tmp_path):
    shell = _shell(tmp_path, "input")
    destination = tmp_path / "results"
    destination.mkdir()
    os.link(shell, destination / "exchange.out")
    before = shell.read_bytes()
    with pytest.raises(ValueError, match="aliases an input"):
        export_shells(shell, destination)
    assert shell.read_bytes() == before
    assert not (destination / "TB2J.pickle").exists()


def test_export_refuses_modified_owned_file(tmp_path):
    shell = _shell(tmp_path, "input")
    destination = tmp_path / "results"
    export_shells(shell, destination)
    (destination / "exchange.out").write_text("edited by user")
    before = (destination / "TB2J.pickle").read_bytes()
    with pytest.raises(ValueError, match="ownership.*fresh output"):
        export_shells(shell, destination)
    assert (destination / "exchange.out").read_text() == "edited by user"
    assert (destination / "TB2J.pickle").read_bytes() == before


@pytest.mark.parametrize("replacement", ["", 'format = "template"'])
@pytest.mark.parametrize("companion", ["valid", "missing", "mismatch"])
def test_declared_context_cannot_be_bypassed_by_changing_provenance(
    tmp_path, replacement, companion
):
    shell = _shell(tmp_path, "input")
    text = shell.read_text()
    assert 'format = "pickle"' in text
    shell.write_text(text.replace('format = "pickle"', replacement))
    context = shell.with_suffix(".context.npz")
    if companion == "missing":
        context.unlink()
    elif companion == "mismatch":
        context.write_bytes(context.read_bytes() + b"changed")
    destination = tmp_path / "results"
    with pytest.raises(ValueError, match="context"):
        export_shells(shell, destination)
    assert not destination.exists()


def test_cli_rejects_context_digest_with_changed_provenance(tmp_path, capsys):
    from TB2J.scripts.TB2J_shell import main

    shell = _shell(tmp_path, "input")
    shell.write_text(
        shell.read_text().replace('format = "pickle"', 'format = "template"')
    )
    destination = tmp_path / "results"
    assert main(["export", str(shell), "-o", str(destination)]) != 0
    assert "context" in capsys.readouterr().err
    assert not destination.exists()


@pytest.mark.parametrize("field", ["NJT_Jdict", "NJT_ddict"])
def test_export_preserves_list_valued_njt_context(tmp_path, field):
    atom = Atoms("Fe", cell=[3, 3, 3], pbc=True)
    key = ((1, 0, 0), 0, 0)
    model = SpinIO(
        atoms=atom,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=None,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
    )
    setattr(model, field, {key: [0.0001, -0.0002, 0.0003]})
    model.write_pickle(path=tmp_path)
    shell = tmp_path / "shell.toml"
    generate_shells(tmp_path / "TB2J.pickle", shell)
    result = export_shells(shell, tmp_path / "results")
    loaded = SpinIO.load_pickle(tmp_path / "results")
    assert getattr(loaded, field)[key] == [0.0001, -0.0002, 0.0003]
    assert getattr(result.spinio, field)[key] == [0.0001, -0.0002, 0.0003]
    label = "Jani_NJt" if field == "NJT_Jdict" else "DMI_NJt"
    text = (tmp_path / "results" / "exchange.out").read_text()
    match = re.search(rf"{label}: \(([^)]+)\)", text)
    assert match is not None
    np.testing.assert_allclose(np.fromstring(match.group(1), sep=" "), [0.1, -0.2, 0.3])


@pytest.mark.parametrize("field", ["NJT_Jdict", "NJT_ddict"])
@pytest.mark.parametrize("vector", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
def test_export_rejects_invalid_njt_shapes_without_publication(tmp_path, field, vector):
    atom = Atoms("Fe", cell=[3, 3, 3], pbc=True)
    key = ((1, 0, 0), 0, 0)
    model = SpinIO(
        atoms=atom,
        spinat=np.array([[0.0, 0.0, 2.0]]),
        charges=None,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
    )
    setattr(model, field, {key: vector})
    model.write_pickle(path=tmp_path)
    shell = tmp_path / "shell.toml"
    generate_shells(tmp_path / "TB2J.pickle", shell)
    destination = tmp_path / "results"
    destination.mkdir()
    (destination / "user.txt").write_text("keep")
    with pytest.raises(ValueError, match="shape"):
        export_shells(shell, destination)
    assert (destination / "user.txt").read_text() == "keep"
    assert not (destination / "TB2J.pickle").exists()


def test_export_does_not_snap_small_nonzero_vampire_tensor(tmp_path):
    shell = _shell(tmp_path, "input")
    doc = shells.read_shells(shell)
    shells.write_shells(
        replace(doc, shells=tuple(replace(item, Jiso=1e-12) for item in doc.shells)),
        shell,
    )
    result = export_shells(shell, tmp_path / "results")
    assert "Vampire" in result.results.written
    rows = (tmp_path / "results" / "Vampire" / "vampire.UCF").read_text().splitlines()
    interaction = rows[rows.index("# Interactions") + 2].split()
    tensor = np.array(interaction[6:], dtype=float).reshape(3, 3)
    np.testing.assert_allclose(np.diag(tensor), [2e-12 / J] * 3, rtol=1e-5, atol=0)
