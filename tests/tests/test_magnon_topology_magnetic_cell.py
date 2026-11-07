import numpy as np
import pytest

from TB2J.magnon.topology.magnetic_cell import expand_magnetic_cell
from TB2J.magnon.topology.preparation import (
    TopologyModel,
    TopologyPreparationError,
    _digest_model,
)


def _chain_model():
    cell = np.eye(3)
    positions = np.zeros((1, 3))
    index = np.array([0])
    moments = np.array([[2.0, 0.0, 0.0]])
    translations = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    tensors = np.zeros((3, 1, 1, 3, 3))
    tensors[0, 0, 0] = tensors[2, 0, 0] = 0.01 * np.eye(3)
    return TopologyModel(
        cell,
        positions,
        positions.copy(),
        index,
        index.copy(),
        moments,
        translations,
        tensors,
        _digest_model(cell, positions, index, index, moments, translations, tensors),
        {name: True for name in ("Jiso", "Jani", "DMI", "SIA")},
        (),
        {},
        (),
        None,
    )


def test_expand_magnetic_cell_twists_moments_and_rekeys_bonds():
    expanded = expand_magnetic_cell(_chain_model(), [1 / 3, 0, 0])
    assert expanded.nspin == 3
    assert np.allclose(expanded.cell[0], [3, 0, 0])
    assert np.allclose(np.linalg.norm(expanded.magmoms, axis=1), 2.0)
    assert np.allclose(expanded.magmoms[:, 1], [0, np.sqrt(3), -np.sqrt(3)])
    assert np.allclose(np.sum(expanded.JR), 0.18)


def test_expand_magnetic_cell_refuses_incommensurate_q():
    with pytest.raises(TopologyPreparationError, match="incommensurate"):
        expand_magnetic_cell(_chain_model(), [np.sqrt(2) / 10, 0, 0])
