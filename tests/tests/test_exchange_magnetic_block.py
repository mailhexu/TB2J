"""The exchange energy loop computes G(R) only on the magnetic-atom orbitals.

``get_quantities_per_e`` builds the Green's function on the block of
magnetic-atom orbitals (``TBGreen.get_GR_block``) and passes the index map to
the vectorized A-tensor routine. The result must equal the reference path,
which builds the full G(R) with ``get_GR`` and indexes it directly.

The tests use bare exchange objects with random eigenstates; magnetic and
non-magnetic atoms are interleaved in the basis so that the index map matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.exchange import ExchangeNCL
from TB2J.green import TBGreen

_ENERGIES = [0.3j, -1.0 + 0.5j, -0.2 + 1e-3j]
_RLIST = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]


def _make_green(nbasis, nkpts=4, efermi=0.2, seed=0):
    rng = np.random.default_rng(seed)
    G = TBGreen.__new__(TBGreen)
    G.kpts = rng.uniform(-0.5, 0.5, (nkpts, 3))
    G.nkpts = nkpts
    G.kweights = np.full(nkpts, 1.0 / nkpts)
    G.R2kfactor = 2j * np.pi
    G.k2Rfactor = -2j * np.pi
    G.efermi = efermi
    G.nbasis = nbasis
    G.is_orthogonal = True
    G._use_cache = False
    G.evals = np.sort(rng.uniform(-6.0, 6.0, (nkpts, nbasis)), axis=1)
    a = rng.standard_normal((nkpts, nbasis, nbasis))
    a = a + 1j * rng.standard_normal((nkpts, nbasis, nbasis))
    G.evecs = np.linalg.qr(a)[0]
    return G


def _random_hermitian(n, rng):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _assert_same_dicts(new, ref):
    assert set(new) == set(ref)
    for key in ref:
        np.testing.assert_allclose(new[key], ref[key], rtol=1e-11, atol=1e-13)


def _make_ncl(orb_decomposition, seed=0):
    """Atoms 0 and 2 are magnetic, 1 and 3 are not; 3 spatial orbitals each."""
    rng = np.random.default_rng(seed)
    norb_atom = 3
    nspin_orb = 2 * norb_atom
    ex = ExchangeNCL.__new__(ExchangeNCL)
    ex.G = _make_green(4 * nspin_orb, seed=seed)
    ex.orb_dict = {a: np.arange(a * nspin_orb, (a + 1) * nspin_orb) for a in range(4)}
    ex.ind_mag_atoms = [0, 2]
    ex.Pdict = {a: _random_hermitian(norb_atom, rng) for a in ex.ind_mag_atoms}
    ex.debug_options = {"compute_charge_moments": False}
    ex.orb_decomposition = orb_decomposition
    ex.backend_name = "Wannier"  # no zeta contraction of the orbital tensor
    ex.short_Rlist = _RLIST
    ex.distance_dict = {
        (R, i, j): (np.zeros(3), 1.0)
        for R in _RLIST
        for i in range(2)
        for j in range(2)
        if not (R == (0, 0, 0) and i == j)
    }
    return ex


@pytest.mark.parametrize("orb_decomposition", [False, True])
def test_ncl_magnetic_block_matches_full_GR(orb_decomposition):
    ex = _make_ncl(orb_decomposition)
    for e in _ENERGIES:
        ref_A, ref_Aorb = ex.get_all_A_vectorized(ex.G.get_GR(_RLIST, energy=e))
        out = ex.get_quantities_per_e(e)
        _assert_same_dicts(out["AijR"], ref_A)
        _assert_same_dicts(out["AijR_orb"], ref_Aorb)


def test_ncl_magnetic_block_indices():
    ex = _make_ncl(False)
    indices, orb_map = ex._magnetic_orbital_block()
    np.testing.assert_array_equal(indices, np.r_[0:6, 12:18])
    assert orb_map == {int(g): i for i, g in enumerate(indices)}
