"""Consumer regressions for reciprocal-pair editable shells."""

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO


def _asymmetric_source(**channels):
    atoms = Atoms(
        ["Fe", "Co"],
        cell=[[4.1, 0, 0], [0.3, 5.2, 0], [0.2, 0.4, 6.3]],
        scaled_positions=[(0.13, 0.21, 0.17), (0.37, 0.16, 0.28)],
        pbc=True,
    )
    return SpinIO(
        atoms, spinat=None, charges=None, index_spin=[0, 1], colinear=False, **channels
    )


def test_sparse_scalar_is_one_editable_shell_and_exports_reverse():
    forward = ((1, 0, 0), 0, 1)
    reverse = ((-1, 0, 0), 1, 0)
    source = _asymmetric_source(exchange_Jdict={forward: 0.123})
    document = shells.compress(source, mode="isotropic").document
    assert len(document.shells) == 1
    assert document.shells[0].n_equivalent_pairs == 1
    edited = replace(document, shells=(replace(document.shells[0], Jiso=0.25),))
    expanded = shells.expand(edited, base=source).spinio
    assert expanded.exchange_Jdict == {forward: 0.25, reverse: 0.25}
    assert not expanded.dmi_ddict and not expanded.Jani_dict


def test_sparse_full_tensor_reverses_dmi_and_transposes(tmp_path):
    forward = ((1, 0, 0), 0, 1)
    reverse = ((-1, 0, 0), 1, 0)
    dmi = np.array([0.002, -0.003, 0.004])
    anisotropy = np.array(
        [[0.003, 0.001, -0.002], [0.001, -0.001, 0.003], [-0.002, 0.003, -0.002]]
    )
    source = _asymmetric_source(
        exchange_Jdict={forward: 0.01},
        dmi_ddict={forward: dmi},
        Jani_dict={forward: anisotropy},
    )
    document = shells.compress(source, mode="tensor").document
    assert len(document.shells) == 1
    path = tmp_path / "tensor.toml"
    shells.write_shells(document, path)
    replay = shells.expand(shells.read_shells(path), base=source).spinio
    assert set(replay.exchange_Jdict) == {forward, reverse}
    assert set(replay.dmi_ddict) == {forward, reverse}
    assert set(replay.Jani_dict) == {forward, reverse}
    np.testing.assert_allclose(replay.dmi_ddict[forward], dmi, atol=1e-12)
    np.testing.assert_allclose(replay.dmi_ddict[reverse], -dmi, atol=1e-12)
    np.testing.assert_allclose(replay.Jani_dict[forward], anisotropy, atol=1e-12)
    np.testing.assert_allclose(replay.Jani_dict[reverse], anisotropy.T, atol=1e-12)


def test_sparse_channels_on_opposite_directions_remain_independent():
    forward = ((1, 0, 0), 0, 1)
    reverse = ((-1, 0, 0), 1, 0)
    dmi = np.array([0.002, -0.003, 0.004])
    source = _asymmetric_source(
        exchange_Jdict={forward: 0.01}, dmi_ddict={reverse: dmi}
    )
    document = shells.compress(source, mode="tensor").document
    assert len(document.shells) == 1
    replay = shells.expand(document, base=source).spinio
    assert replay.exchange_Jdict == {forward: 0.01, reverse: 0.01}
    assert set(replay.dmi_ddict) == {forward, reverse}
    np.testing.assert_allclose(replay.dmi_ddict[reverse], dmi, atol=1e-12)
    np.testing.assert_allclose(replay.dmi_ddict[forward], -dmi, atol=1e-12)
    assert not replay.Jani_dict


def test_self_reversing_cubic_orbit_counts_unique_physical_bonds(tmp_path):
    atoms = Atoms("Fe", cell=[2, 2, 2], scaled_positions=[[0, 0, 0]], pbc=True)
    document = shells.template(atoms, [1], 2.01)
    assert len(document.shells) == 1
    assert document.shells[0].n_equivalent_pairs == 3
    path = tmp_path / "shell.toml"
    shells.write_shells(document, path)
    text = path.read_text()
    pair_lines = [
        line
        for line in text.splitlines()
        if line.startswith("# ") and line[2:].split()[:2] == ["Fe1", "Fe1"]
    ]
    assert len(pair_lines) == 3
    filled = replace(document, shells=(replace(document.shells[0], Jiso=0.1),))
    keys = shells.expand(filled).spinio.exchange_Jdict
    assert set(key[0] for key in keys) == {
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    }


def test_version_2_file_is_rejected_without_reinterpreting_masks(tmp_path):
    atoms = Atoms("Fe", cell=[2, 2, 2], scaled_positions=[[0, 0, 0]], pbc=True)
    path = tmp_path / "shell.toml"
    shells.write_shells(shells.template(atoms, [1], 2.01), path)
    text = path.read_text()
    assert "schema_version = 3" in text
    path.write_text(text.replace("schema_version = 3", "schema_version = 2", 1))
    with pytest.raises(
        ValueError, match=r"(?i)(version 2|schema_version.*2).*(regenerat|unsupported)"
    ):
        shells.read_shells(path)
