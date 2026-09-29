"""Story 001 (nonorth-spin-lattice): displacement vertex kernel tests.

Pins the energy-dependent pencil vertex V(z) = dH - z*dS per (Rq, Rk), the
provider transpose contract consumed by ``TBGreen.get_dGR*``, and the
R-map folding equivalence against ``fold_two_legged``.  The mathematical
identities are certified in ``docs/sympy/nonorthogonal_spin_lattice.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.green import TBGreen
from TB2J.spin_lattice_lcao import (
    DisplacementVertex,
    LocalizedDisplacementDerivative,
    fold_two_legged,
)


def _rand_herm(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / 2


@pytest.fixture
def vertex2():
    """Two-entry vertex on a 2x2 basis with distinct dH and dS blocks."""
    dH = {
        ((0, 0, 0), (0, 0, 0)): _rand_herm(2, 1),
        ((0, 0, 0), (1, 0, 0)): _rand_herm(2, 2),
    }
    dS = {
        ((0, 0, 0), (0, 0, 0)): _rand_herm(2, 3),
        ((0, 0, 0), (1, 0, 0)): _rand_herm(2, 4),
    }
    return DisplacementVertex(dH, dS, nbasis=2)


def test_vertex_assembly_is_energy_dependent(vertex2):
    z1, z2 = 3.0 + 0.5j, -1.0 - 0.25j
    key = ((0, 0, 0), (1, 0, 0))
    dh, ds = vertex2.dH[key], vertex2.dS[key]
    np.testing.assert_allclose(
        vertex2.vertex(*key, z1), dh - z1 * ds, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        vertex2.vertex(*key, z2), dh - z2 * ds, rtol=0, atol=1e-14
    )
    # a fixed-z precomputation would differ: the two energies disagree
    assert not np.allclose(vertex2.vertex(*key, z1), vertex2.vertex(*key, z2))


def test_vertex_missing_key_is_zero(vertex2):
    v = vertex2.vertex((7, 7, 7), (0, 0, 0), 2.0)
    assert v.shape == (2, 2)
    assert np.all(v == 0)


def test_vertex_wrong_shape_rejected():
    with pytest.raises(ValueError, match="shape"):
        DisplacementVertex({((0, 0, 0), (0, 0, 0)): np.zeros((3, 3))}, {}, nbasis=2)


def _bare_tbgreen(efermi=0.5):
    """TBGreen instance without running __init__ (unit seam only)."""
    g = TBGreen.__new__(TBGreen)
    g._Rmap = None
    g.efermi = efermi
    return g


def test_vertex_provider_contract(vertex2):
    """Provider returns the physical vertex (call site transposes nothing)."""
    z = 1.5 + 0.2j
    key = ((0, 0, 0), (0, 0, 0))
    np.testing.assert_allclose(
        vertex2.get_vertex_two_spin(*key, z),
        vertex2.vertex(*key, z),
        rtol=0,
        atol=1e-14,
    )


def test_get_dGR_matches_reference_folding(vertex2):
    """get_dGR with the provider equals the explicit two-legged folding."""
    Rpts = [tuple(R) for R in [(-1, 0, 0), (0, 0, 0), (1, 0, 0), (-2, 0, 0), (2, 0, 0)]]
    GR = {R: _rand_herm(2, 10 + i) for i, R in enumerate(Rpts)}
    Ru = (1, 0, 0)
    Rjlist = list(Rpts)
    energy = -2.0  # relative to efermi
    g = _bare_tbgreen(efermi=0.5)
    got_ij, got_ji = g.get_dGR(
        GR, Rpts, Rjlist, vertex2, Ru, cutoff=100.0, energy=energy
    )
    ref_ij, ref_ji = fold_two_legged(GR, vertex2, Rpts, Rjlist, Ru, g.efermi + energy)
    for R in Rpts:
        np.testing.assert_allclose(got_ij[R], ref_ij[R], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(got_ji[R], ref_ji[R], rtol=1e-12, atol=1e-14)


def test_get_dGR_vertex_requires_energy(vertex2):
    g = _bare_tbgreen()
    GR = {(0, 0, 0): np.eye(2)}
    with pytest.raises(ValueError, match="contour energy"):
        g.get_dGR(GR, [(0, 0, 0)], [(0, 0, 0)], vertex2, (0, 0, 0))


def test_get_dGR_legacy_epw_unchanged():
    """The legacy EPW branch keeps raw + transpose semantics."""

    class FakeEpw:
        Rqdict = {(0, 0, 0): None}
        Rkdict = {(0, 0, 0): None, (1, 0, 0): None}

        def get_epmat_RgRk_two_spin(self, Rq, Rk, avg=False):
            m = np.zeros((2, 2), dtype=complex)
            m[0, 1] = 1.0
            return m

    class VertexShim(DisplacementVertex):
        """Same data exposed through the provider contract."""

        def __init__(self):
            m = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
            super().__init__(
                {((0, 0, 0), Rk): m for Rk in [(0, 0, 0), (1, 0, 0)]},
                {},
                nbasis=2,
            )

    g = _bare_tbgreen()
    GR = {R: np.eye(2) for R in [(0, 0, 0), (1, 0, 0), (-1, 0, 0)]}
    Rpts = list(GR)
    legacy_ij, _ = g.get_dGR(GR, Rpts, Rpts, FakeEpw(), (0, 0, 0))
    g2 = _bare_tbgreen()
    prov_ij, _ = g2.get_dGR(GR, Rpts, Rpts, VertexShim(), (0, 0, 0), energy=0.0)
    for R in Rpts:
        np.testing.assert_allclose(legacy_ij[R], prov_ij[R], rtol=1e-13)


# ---------------------------------------------------------------------
# Story 002: derivative producers
# ---------------------------------------------------------------------
class _StubAbacusParser:
    """Minimal parser exposing read_dH_dS (file IO covered in HamiltonIO)."""

    def __init__(self, dH_up, dH_dn, dS, Rlist):
        self._data = (dH_up, dH_dn, dS, Rlist)
        self.atoms = None

    def read_dH_dS(self, direction=(1, 0, 0), binary=None):
        dH_up, dH_dn, dS, Rlist = self._data
        return dH_up.shape[1], Rlist, dH_up, dH_dn, dS


def test_from_abacus_dh_keys_and_interleave():
    rng = np.random.default_rng(21)
    Rlist = [(0, 0, 0), (1, 0, 0)]
    dH_up = np.stack([rng.normal(size=(2, 2)), rng.normal(size=(2, 2))])
    dH_dn = np.stack([rng.normal(size=(2, 2)), rng.normal(size=(2, 2))])
    dS = np.stack([rng.normal(size=(2, 2)), rng.normal(size=(2, 2))])
    parser = _StubAbacusParser(dH_up, dH_dn, dS, Rlist)
    disp = LocalizedDisplacementDerivative.from_abacus_dh(parser)
    assert disp.Ru == (0, 0, 0)
    assert disp.provenance.startswith("abacus")
    assert set(disp.dH) == {((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 0, 0))}
    for iR, R in enumerate(Rlist):
        key = ((0, 0, 0), tuple(R))
        np.testing.assert_allclose(disp.dH[key][::2, ::2], dH_up[iR], rtol=1e-14)
        np.testing.assert_allclose(disp.dH[key][1::2, 1::2], dH_dn[iR], rtol=1e-14)
        np.testing.assert_allclose(disp.dS[key][::2, ::2], dS[iR], rtol=1e-14)
        np.testing.assert_allclose(disp.dS[key][1::2, 1::2], dS[iR], rtol=1e-14)


class _CellModel:
    """Scalar-channel supercell model with HR/SR arrays and an R-list."""

    def __init__(self, HR, SR, Rlist, atoms=None):
        self.HR = HR
        self.SR = SR
        self.Rlist = Rlist
        self.atoms = atoms


def _chain_supercell(tau, ncell=4, lam=1.3, t0=-1.0, s0=0.2, de=0.0):
    """1D chain supercell: atom 0 (cell 0) displaced by tau.

    Bonds touching the displaced atom carry t(tau) = t0*exp(-lam*tau),
    s(tau) = s0*exp(-lam*tau); others stay at tau=0 values. Returns the
    supercell models for the given tau (Rlist covers the needed images).
    """
    n = ncell
    Rlist = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    HR = np.zeros((len(Rlist), n, n))
    SR = np.zeros((len(Rlist), n, n))
    for iR, R in enumerate(Rlist):
        for i in range(n):
            if R[0] == 0:
                continue  # onsite only in the R=0 block; bonds live in +-1
            j_cell = (i + R[0]) % n
            touches = (i == 0) or ((i + R[0]) % n == 0)
            tt = t0 * np.exp(-lam * tau) if touches else t0
            ss = s0 * np.exp(-lam * tau) if touches else s0
            HR[iR, i, j_cell] += tt
            SR[iR, i, j_cell] += ss
    iR0 = Rlist.index((0, 0, 0))
    HR[iR0, range(n), range(n)] += de * tau  # onsite drift on every atom
    SR[:, range(n), range(n)] = 1.0
    return _CellModel(HR, SR, Rlist)


def test_from_supercell_runs_matches_analytic_derivative():
    ncell = 4
    delta = 1e-4
    plus = _chain_supercell(+delta, ncell, de=0.3)
    minus = _chain_supercell(-delta, ncell, de=0.3)
    ref = _chain_supercell(0.0, ncell, de=0.3)
    orb_map = [(0, (c, 0, 0)) for c in (0, 1, -2, -1)]  # symmetric labels
    disp = LocalizedDisplacementDerivative.from_supercell_runs(
        ref, plus, minus, orb_map, atom_orbs=[0], delta=delta
    )
    lam, t0, s0, de = 1.3, -1.0, 0.2, 0.3
    dt = -lam * t0
    ds = -lam * s0
    # displaced atom in supercell cell 0; touching elements (legs (n_i,n_j)):
    # (0,1), (0,-1), (1,0), (-1,0) plus onsite (0,0). Keys are
    # (Rq, Rk) = (-n_i, n_j - n_i); the boundary bond uses the symmetric
    # cell label -1 (cell 3 of the 4-supercell), the alias-free choice.
    expect = {
        ((0, 0, 0), (1, 0, 0)): dt,
        ((0, 0, 0), (-1, 0, 0)): dt,
        ((-1, 0, 0), (-1, 0, 0)): dt,
        ((1, 0, 0), (1, 0, 0)): dt,
        ((0, 0, 0), (0, 0, 0)): de,
    }
    assert set(disp.dH) == set(expect)
    for key, value in expect.items():
        mat = disp.dH[key]
        nz = np.argwhere(np.abs(mat) > 1e-9)
        assert len(nz) == 1, (key, mat)
        assert abs(mat[nz[0][0], nz[0][1]] - value) < 1e-6, (key, mat)
    # dS: same bond structure with ds; onsite S_ii = 1 is constant
    for key, value in expect.items():
        mat = disp.dS[key]
        if key == ((0, 0, 0), (0, 0, 0)):
            assert np.max(np.abs(mat)) < 1e-9
        else:
            nz = np.argwhere(np.abs(mat) > 1e-9)
            assert len(nz) == 1, (key, mat)
            assert abs(mat[nz[0][0], nz[0][1]] - ds) < 1e-6, (key, mat)
    assert disp.provenance.startswith("supercell")


def test_from_supercell_runs_rejects_rlist_mismatch():
    plus = _chain_supercell(1e-4, 4)
    bad = _chain_supercell(-1e-4, 4)
    bad.Rlist = [(-1, 0, 0), (0, 0, 0)]
    with pytest.raises(ValueError, match="R-list"):
        LocalizedDisplacementDerivative.from_supercell_runs(
            plus, plus, bad, [(0, (c, 0, 0)) for c in range(4)], [0], 1e-4
        )
