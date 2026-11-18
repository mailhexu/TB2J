"""R list must be bounded by the k-mesh supercell (projector interfaces).

All projector interfaces (GPAW / ABINIT NC+PAW / VASP native / QE) build
their real-space pair list with ``_R_grid_for_cutoff``.  The Bloch-sum
Green function is exactly periodic under k-mesh supercell translations,
so R vectors beyond the supercell alias cells inside it; the R list must
therefore be a subset of ``kmesh_to_R(mesh)`` with Rcut only pruning
pairs, exactly like the Wannier path in ``exchange.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.interfaces.gpaw_projector import (
    ProjectorGreenData,
    _infer_kmesh,
    _R_grid_for_cutoff,
)


def _full_mesh(mesh):
    n1, n2, n3 = mesh
    return np.array(
        [
            (i / n1, j / n2, k / n3)
            for i in range(n1)
            for j in range(n2)
            for k in range(n3)
        ]
    )


def _stub_data(kpoints, cell=None, positions=None):
    nk = len(kpoints)
    cell = cell if cell is not None else 3.0 * np.eye(3)
    positions = positions if positions is not None else np.array([[0.0, 0.0, 0.0]])
    nproj = 2
    return ProjectorGreenData(
        kpoints=np.asarray(kpoints, dtype=float),
        weights=np.full(nk, 1.0 / nk),
        eigenvalues=np.zeros((1, nk, nproj)),
        coefficients=np.zeros((1, nk, nproj, nproj), dtype=complex),
        efermi=0.0,
        projector_site=np.zeros(nproj, dtype=int),
        projector_atom=np.zeros(nproj, dtype=int),
        cell=cell,
        positions=positions,
        atomic_numbers=np.array([26]),
        projector_l=np.full(nproj, 2),
        projector_m=np.array([-2, -1]),
        projector_radial=np.zeros(nproj, dtype=int),
        overlap_metric=np.eye(nproj),
        site_nproj=np.array([nproj]),
        site_projector_indices=np.arange(nproj, dtype=int).reshape(1, -1),
    )


@pytest.mark.parametrize("mesh", [(4, 4, 4), (3, 2, 5), (2, 2, 2), (1, 1, 1)])
def test_infer_kmesh_full_and_ibz(mesh):
    assert _infer_kmesh(_full_mesh(mesh)) == mesh
    # IBZ-like reduction: keep only k in the positive octant incl. origin
    kpts = _full_mesh(mesh)
    half = np.all(np.mod(kpts + 0.5, 1.0) - 0.5 <= 1e-9, axis=1)
    if half.sum() >= 1:
        assert _infer_kmesh(kpts[half]) == mesh


def test_infer_kmesh_irregular_rejects():
    with pytest.raises(ValueError):
        _infer_kmesh(np.array([[0.0, 0.0, 0.0], [0.13, 0.0, 0.0], [0.41, 0.0, 0.0]]))


def test_r_grid_bounded_by_supercell():
    """Rcut beyond the k-mesh supercell must not enlarge the R list."""
    mesh = (5, 5, 5)  # odd: symmetric window +-2, alias-free by construction
    data = _stub_data(_full_mesh(mesh))
    Rpts = _R_grid_for_cutoff(data, sites=[0], Rcut=1.0e6)
    assert set(map(tuple, Rpts)) == {
        (i, j, k) for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)
    }


def test_r_grid_even_mesh_excludes_aliasing_boundary():
    """For even n the +-n/2 classes alias; both must be excluded."""
    data = _stub_data(_full_mesh((4, 4, 4)))
    Rpts = _R_grid_for_cutoff(data, sites=[0], Rcut=1.0e6)
    entries = set(map(tuple, Rpts))
    assert entries == {
        (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
    }
    # negation closure required by the exchange kernel
    for R in entries:
        assert tuple(-x for x in R) in entries
    # no two entries congruent modulo the mesh supercell
    for R in entries:
        for R2 in entries:
            if R != R2:
                assert any((a - b) % n for a, b, n in zip(R, R2, (4, 4, 4)))


def test_r_grid_small_rcut_subset():
    mesh = (6, 6, 6)
    data = _stub_data(_full_mesh(mesh))
    Rpts = _R_grid_for_cutoff(data, sites=[0], Rcut=3.05)
    # cubic 3 A cell: neighbours at 3.0 A in +-x/y/z only
    assert set(map(tuple, Rpts)) == {
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    }


def test_r_grid_gamma_only_collapses(capsys):
    data = _stub_data(np.array([[0.0, 0.0, 0.0]]))
    Rpts = _R_grid_for_cutoff(data, sites=[0], Rcut=6.0)
    assert set(map(tuple, Rpts)) == {(0, 0, 0)}
    assert "k-mesh supercell" in capsys.readouterr().out


def test_r_grid_warns_when_rcut_exceeds_supercell(capsys):
    data = _stub_data(_full_mesh((2, 2, 2)))
    _R_grid_for_cutoff(data, sites=[0], Rcut=6.0)
    assert "WARNING" in capsys.readouterr().out
