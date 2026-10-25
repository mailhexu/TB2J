"""Shared SG/MSG operations and complete bond-image contracts."""

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.symmetrize_J import symmetrize_exchange
from TB2J.symmetry_core import (
    SymmetryOperations,
    _build_site_maps,
    magnetic_symmetry_ops,
    pair_images,
)


@pytest.mark.parametrize(
    "moments",
    [
        np.array([1.0, -1.0]),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    ],
)
def test_antiferromagnetic_half_translation_is_primed(moments):
    atoms = Atoms(
        "Fe2",
        cell=np.eye(3) * 3,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    operations = magnetic_symmetry_ops(atoms, moments)
    identity_rotation = np.all(
        operations.rotations == np.eye(3, dtype=int), axis=(1, 2)
    )
    half_translation = np.all(np.isclose(operations.translations, 0.5), axis=1)
    assert np.any(identity_rotation & half_translation & operations.time_reversals)
    zero_translation = np.all(np.isclose(operations.translations, 0.0), axis=1)
    assert np.any(identity_rotation & zero_translation & ~operations.time_reversals)
    assert not np.any(identity_rotation & half_translation & ~operations.time_reversals)


def test_pair_images_include_cell_shifts_for_inversion_and_half_translation():
    xfrac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cell = np.eye(3) * 2
    identity = np.eye(3, dtype=int)
    operations = SymmetryOperations(
        np.array([identity, -identity, identity]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        None,
        "inversion and half translation",
    )
    keys = [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0))]
    maps = _build_site_maps(xfrac, cell, operations, symprec=1e-5)
    images, rotations = pair_images(keys, operations, maps, xfrac, cell)
    assert images == [
        {keys[0]: keys[0], keys[1]: keys[1]},
        {keys[0]: (0, 1, (-1, 0, 0)), keys[1]: (1, 0, (1, 0, 0))},
        {keys[0]: (1, 0, (1, 0, 0)), keys[1]: (0, 1, (-1, 0, 0))},
    ]
    np.testing.assert_array_equal(rotations, [identity, -identity, identity])


def test_pair_images_reject_noninteger_images_instead_of_returning_partial_map():
    operations = SymmetryOperations(
        np.array([np.eye(3, dtype=int)]), np.zeros((1, 3)), None, "inconsistent map"
    )
    xfrac = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    keys = [(0, 0, (1, 0, 0)), (0, 1, (0, 0, 0))]
    with pytest.raises(ValueError, match="non-integer image cell"):
        pair_images(keys, operations, [np.array([1, 0])], xfrac, np.eye(3))


def test_transferred_structure_rejects_inconsistent_bond_images():
    reference = Atoms(
        "Fe2",
        cell=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    stored = reference.copy()
    stored.set_scaled_positions([[0.0, 0.0, 0.0], [0.25, 0.5, 0.5]])
    model = SpinIO(
        stored,
        [1.0, 1.0],
        [0.0, 0.0],
        [0, 1],
        colinear=True,
        exchange_Jdict={((0, 0, 0), 0, 1): 0.001, ((0, 0, 0), 1, 0): 0.001},
    )
    with pytest.raises(ValueError, match="non-integer image cell"):
        symmetrize_exchange(model, reference, symprec=0.3)


@pytest.mark.parametrize(
    "moments",
    [[1.0, -1.0001], [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0001]]],
)
def test_magnetic_tolerance_controls_primed_half_translation(moments):
    atoms = Atoms(
        "Fe2",
        cell=np.eye(3) * 3,
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    strict = magnetic_symmetry_ops(atoms, moments, magnetic_tol=1e-5)
    loose = magnetic_symmetry_ops(atoms, moments, magnetic_tol=1e-3)

    def primed_half_translation(operations):
        identity = np.all(operations.rotations == np.eye(3, dtype=int), axis=(1, 2))
        half = np.all(np.isclose(operations.translations, 0.5), axis=1)
        return np.any(identity & half & operations.time_reversals)

    assert not primed_half_translation(strict)
    assert primed_half_translation(loose)
