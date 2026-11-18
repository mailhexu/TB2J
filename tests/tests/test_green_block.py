"""Tests for ``TBGreen.get_GR_block``.

``get_GR_block`` computes the real-space Green's function on a subset of the
basis from the matching rows of the eigenvectors only. It must agree with the
corresponding block of the full ``get_GR`` for any subset, energy and k-mesh,
and the self-energy branch (no eigenvector form) must give the same block.

The tests use a bare ``TBGreen`` with random unitary eigenvectors, so no DFT
input is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.green import TBGreen

_ENERGIES = [0.3j, -1.0 + 0.5j, 2.0 + 1e-3j, -0.2 - 0.7j]
_RLIST = [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, -1), (1, 1, 1)]


def _make_green(nkpts=5, nbasis=12, efermi=0.4, seed=0):
    """Bare orthogonal TBGreen with random eigenvalues and unitary eigenvectors."""
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


class _SelfEnergyModel:
    """Minimal model with a (zero) self-energy, which selects the DMFT branch."""

    def __init__(self, G):
        self._H = {
            tuple(k): (G.evecs[ik] * G.evals[ik]) @ G.evecs[ik].conj().T
            for ik, k in enumerate(G.kpts)
        }
        self.nbasis = G.nbasis

    def get_sigma(self, energy, ispin=0):
        return np.zeros((self.nbasis, self.nbasis), dtype=complex)

    def get_hamiltonian(self, kpt, ispin=0):
        return self._H[tuple(kpt)]


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0, 1, 2, 3]),
        np.array([1, 4, 5, 9, 11]),
        np.arange(12),
    ],
)
def test_get_GR_block_matches_full(indices):
    G = _make_green()
    for e in _ENERGIES:
        full = G.get_GR(_RLIST, energy=e)
        block = G.get_GR_block(_RLIST, e, indices)
        assert block.shape == (len(_RLIST), len(indices), len(indices))
        np.testing.assert_allclose(
            block, full[:, indices][:, :, indices], rtol=1e-12, atol=1e-14
        )


def test_get_GR_block_changes_with_indices():
    """The cached eigenvector rows are rebuilt when the subset changes."""
    G = _make_green(seed=1)
    a, b = np.array([0, 2, 4]), np.array([1, 3, 5, 7])
    e = -0.5 + 0.2j
    full = G.get_GR(_RLIST, energy=e)
    for idx in (a, b, a):
        np.testing.assert_allclose(
            G.get_GR_block(_RLIST, e, idx),
            full[:, idx][:, :, idx],
            rtol=1e-12,
            atol=1e-14,
        )


def test_get_GR_block_self_energy_branch():
    """With a self-energy the block is sliced from the full G(k); same result."""
    G = _make_green(seed=2)
    idx = np.array([2, 3, 7, 8])
    ref = {e: G.get_GR_block(_RLIST, e, idx) for e in _ENERGIES}
    G.tbmodel = _SelfEnergyModel(G)
    for e in _ENERGIES:
        np.testing.assert_allclose(
            G.get_GR_block(_RLIST, e, idx), ref[e], rtol=1e-10, atol=1e-12
        )
