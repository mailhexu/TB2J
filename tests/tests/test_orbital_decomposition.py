"""Regression tests for orbital decomposition in the non-collinear exchange path.

These guard the bug where ``ExchangeNCL.get_all_A_vectorized`` (the active
per-energy path) built the orbital-resolved A tensor in the *raw* orbital basis
and never applied ``simplify_orbital_contributions`` — i.e. the zeta contraction
and element selection (``Cr_3d``) were silently skipped.  The collinear path and
the commented-out non-vectorized ``get_A_ijR`` reference were correct, so the
output matrices disagreed with the ``orbital_names`` header (e.g. a 10x10 matrix
under a 5-group header).

The tests build a bare ``ExchangeNCL`` with synthetic orbital data (no DFT files)
and assert the vectorized path contracts/ selects exactly as the reference
``map_orbs_matrix`` reduction matrix dictates.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.exchange import ExchangeNCL
from TB2J.orbmap import map_orbs_matrix
from TB2J.pauli import pauli_block_all

# Double-zeta 3d shell plus an s and a p orbital, so selection and zeta
# contraction are independently observable.
_LABELS = [
    "4sZ1",
    "4pyZ1",
    "3dxyZ1",
    "3dyzZ1",
    "3dz2Z1",
    "3dxzZ1",
    "3dx2-y2Z1",
    "3dxyZ2",
    "3dyzZ2",
    "3dz2Z2",
    "3dxzZ2",
    "3dx2-y2Z2",
]


def _make_exchange(include_only):
    """Build a bare ExchangeNCL for a single magnetic atom with ``_LABELS``.

    ``include_only`` selects orbital groups (e.g. ``["3d"]``) or ``None`` for all.
    """
    n_spatial = len(_LABELS)
    nbasis = 2 * n_spatial  # spin up/down interleaved (non-collinear)

    mmat, reduced = map_orbs_matrix(_LABELS, spinor=False, include_only=include_only)

    ex = ExchangeNCL.__new__(ExchangeNCL)
    ex.ind_mag_atoms = [0]
    ex.orb_dict = {0: np.arange(nbasis)}
    # P projector lives in the Pauli-halved (spatial) basis.
    ex.Pdict = {0: np.eye(n_spatial)}
    ex.mmats = {0: mmat}
    ex.norb_reduced = {0: len(reduced)}
    ex.orbital_names = {0: reduced}
    ex.backend_name = "SIESTA"
    ex.orb_decomposition = True
    # Sorted R list so the vectorized path's np.flip convention (-R[i] lives at
    # index nR-1-i) and the non-vectorized R_negative_index agree.
    ex.short_Rlist = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    ex.R_negative_index = {0: 2, 1: 1, 2: 0}
    ex.R_ijatom_dict = {0: [(0, 0)], 1: [(0, 0)], 2: [(0, 0)]}
    ex.distance_dict = {
        ((-1, 0, 0), 0, 0): (np.zeros(3), 1.0),
        ((0, 0, 0), 0, 0): (np.zeros(3), 0.0),
        ((1, 0, 0), 0, 0): (np.zeros(3), 1.0),
    }
    return ex, mmat


def _random_hermitian_GR(nR, nbasis, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((nR, nbasis, nbasis)) + 1j * rng.standard_normal(
        (nR, nbasis, nbasis)
    )
    return g + np.conj(np.swapaxes(g, -1, -2))


def test_ncl_orbital_selection_3d():
    """``Fe_3d`` selects only the five d orbitals and contracts zeta."""
    ex, mmat = _make_exchange(include_only=["3d"])
    assert ex.norb_reduced[0] == 5
    assert ex.orbital_names[0] == ("3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2")

    GR = _random_hermitian_GR(3, 2 * len(_LABELS), seed=1)
    _, A_orb = ex.get_all_A_vectorized(GR)

    key = ((0, 0, 0), 0, 0)
    # Before the fix this was (4, 4, 12, 12) — raw, unselected, uncontracted.
    assert A_orb[key].shape == (4, 4, 5, 5)


def test_ncl_orbital_zeta_contraction():
    """With no selection, each zeta pair is summed into one group."""
    ex, mmat = _make_exchange(include_only=None)
    # 1 s + 1 p + 5 d-types (zeta-contracted) = 7 groups.
    assert ex.norb_reduced[0] == 7
    assert ex.orbital_names[0] == (
        "4s",
        "4py",
        "3dxy",
        "3dyz",
        "3dz2",
        "3dxz",
        "3dx2-y2",
    )

    GR = _random_hermitian_GR(3, 2 * len(_LABELS), seed=2)
    _, A_orb = ex.get_all_A_vectorized(GR)

    key = ((0, 0, 0), 0, 0)
    assert A_orb[key].shape == (4, 4, 7, 7)


def test_ncl_vectorized_matches_mmat_contraction():
    """The vectorized A_orb equals the explicit mmat reduction of the raw tensor."""
    ex, mmat = _make_exchange(include_only=["3d"])
    nbasis = 2 * len(_LABELS)
    GR = _random_hermitian_GR(3, nbasis, seed=3)
    A, A_orb = ex.get_all_A_vectorized(GR)

    # Recompute the raw (uncontracted) tensor and reduce it by hand.
    idx = ex.iorb(0)
    Gij = pauli_block_all(GR[:, idx][:, :, idx])
    Gji = np.flip(pauli_block_all(GR[:, idx][:, :, idx]), axis=0)
    X = ex.Pdict[0] @ Gij
    Y = ex.Pdict[0] @ Gji
    raw = np.einsum("ruij,rvji->ruvij", X, Y) / np.pi
    manual = np.einsum("ruvij,ia,jb->ruvab", raw, mmat, mmat)

    # Check every R vector: the stored A_orb equals the hand-reduced tensor and
    # the scalar A is its orbital sum.
    for iR, R_vec in enumerate(ex.short_Rlist):
        key = (R_vec, 0, 0)
        if key not in A_orb:
            continue
        np.testing.assert_allclose(A_orb[key], manual[iR], atol=1e-12)
        np.testing.assert_allclose(A[key], A_orb[key].sum(axis=(-2, -1)), atol=1e-12)


def test_ncl_vectorized_matches_reference_path():
    """The vectorized path agrees with the non-vectorized ``get_all_A`` reference."""
    ex, _ = _make_exchange(include_only=["3d"])
    nbasis = 2 * len(_LABELS)
    GR = _random_hermitian_GR(3, nbasis, seed=4)

    A_ref, Aorb_ref = ex.get_all_A(GR)
    A_vec, Aorb_vec = ex.get_all_A_vectorized(GR)

    for key in Aorb_ref:
        assert Aorb_ref[key].shape == Aorb_vec[key].shape
        np.testing.assert_allclose(Aorb_ref[key], Aorb_vec[key], atol=1e-12)
        np.testing.assert_allclose(A_ref[key], A_vec[key], atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
