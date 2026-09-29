"""Story 003 (nonorth-spin-lattice): FD validation of the spin-lattice path.

Synthetic 1D non-orthogonal chain, one magnetic orbital per cell, exchange
splitting Delta(dtau) drifting on the displaced atom.  The analytic O from
the production path (TBGreen + DisplacementVertex + ExchangePert2Sl with
dP terms) is checked against central differences of the production J
computed on a matched-discretization supercell (localized displacement of
supercell atom 0).  Negative controls: dropping -z*dS or fixing z must
disagree.
"""

from __future__ import annotations

import numpy as np
import test_spin_lattice_fd as T_mod
from scipy.linalg import eigh

from TB2J.green import TBGreen
from TB2J.spin_lattice_lcao import (
    DisplacementVertex,
    ExchangePert2Sl,
    LocalizedDisplacementDerivative,
)

# model parameters (1D chain, 1 orbital, collinear z)
LAM_T = 1.1
LAM_S = 1.1
T0 = -0.6
S0 = 0.15
EPS = 0.3
DELTA = 1.0  # exchange splitting (up: +Delta/2, dn: -Delta/2)
DEPS = 0.25  # onsite drift of the displaced atom
DDELTA = 0.4  # splitting drift of the displaced atom (dP channel)
EFERMI = 0.0
ZGRID = [-2.0 + 0.3j, -0.5 + 0.5j, 1.0 + 0.25j]

RCHAIN = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]


class _SpinorChain:
    """Spinor (interleaved [up, dn]) chain model with one orbital per cell.

    tau displaces atom 0: bonds touching it carry t(tau), s(tau); its
    onsite carries eps + d_eps*tau and splitting Delta + d_delta*tau.
    ncell=1 gives the primitive model (periodic, tau only meaningful as a
    derivative parameter around 0); ncell>1 gives a supercell in which
    only supercell atom 0 is displaced (localized within the cell).
    """

    is_orthogonal = False
    name = "synthetic-chain"
    R2kfactor = 2j * np.pi
    efermi = None

    @property
    def norb(self):
        return self.ncell

    @property
    def nbasis(self):
        return 2 * self.ncell

    def __init__(self, tau=0.0, ncell=1, nel=2.0):
        self.tau = tau
        self.ncell = ncell
        self.nel = nel
        # supercell R list covers +-1 supercell lattice vectors
        self.Rlist = [(-1, 0, 0), (0, 0, 0), (1, 0, 0)] if ncell > 1 else list(RCHAIN)

    def _bond(self, i_cell, j_cell):
        """(t, s) for the bond between cell i and neighbouring cell j=i+-1."""
        if self.ncell == 1:
            touched = True  # primitive model: every cell is the displaced one
        else:
            touched = i_cell == 0 or j_cell % self.ncell == 0
        tt = T0 * np.exp(-LAM_T * self.tau) if touched else T0
        ss = S0 * np.exp(-LAM_S * self.tau) if touched else S0
        return tt, ss

    def HSR(self):
        n = self.ncell
        HR = np.zeros((len(self.Rlist), 2 * n, 2 * n), dtype=complex)
        SR = np.zeros_like(HR)
        for iR, R in enumerate(self.Rlist):
            for i in range(n):
                if R[0] == 0:
                    eps = EPS + (
                        DEPS * self.tau if (self.ncell == 1 or i == 0) else 0.0
                    )
                    delta = DELTA + (
                        DDELTA * self.tau if (self.ncell == 1 or i == 0) else 0.0
                    )
                    HR[iR, 2 * i, 2 * i] = eps + delta / 2.0
                    HR[iR, 2 * i + 1, 2 * i + 1] = eps - delta / 2.0
                    SR[iR, 2 * i, 2 * i] = 1.0
                    SR[iR, 2 * i + 1, 2 * i + 1] = 1.0
                else:
                    j = (i + R[0]) % n
                    tt, ss = self._bond(i, i + R[0])
                    HR[iR, 2 * i, 2 * j] = tt
                    HR[iR, 2 * i + 1, 2 * j + 1] = tt
                    SR[iR, 2 * i, 2 * j] = ss
                    SR[iR, 2 * i + 1, 2 * j + 1] = ss
        return HR, SR

    def HSE_k(self, kpt, convention=2):
        HR, SR = self.HSR()
        Hk = np.zeros((2 * self.ncell, 2 * self.ncell), dtype=complex)
        Sk = np.zeros_like(Hk)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(self.R2kfactor * np.dot(R, kpt))
            Hk += HR[iR] * phase
            Sk += SR[iR] * phase
        evals, evecs = eigh(Hk, Sk)
        return Hk, Sk, evals, evecs


def _green(model, kpts):
    kw = np.full(len(kpts), 1.0 / len(kpts))
    return TBGreen(
        tbmodel=model,
        efermi=EFERMI,
        kpts=np.array(kpts),
        kweights=kw,
        use_cache=False,
    )


def _analytic_vertex():
    """Two-legged vertex for displacing atom 0 in the primitive cell."""
    dt = -LAM_T * T0
    ds = -LAM_S * S0
    n = 2

    def spinor(mat):
        out = np.zeros((n, n), dtype=complex)
        out[0, 0] = mat
        out[1, 1] = mat
        return out

    dH = {}
    dS = {}
    # bonds (0 -> +1) and (0 -> -1) with atom 0 on the first leg
    dH[((0, 0, 0), (1, 0, 0))] = spinor(dt)
    dS[((0, 0, 0), (1, 0, 0))] = spinor(ds)
    dH[((0, 0, 0), (-1, 0, 0))] = spinor(dt)
    dS[((0, 0, 0), (-1, 0, 0))] = spinor(ds)
    # bonds with atom 0 on the second leg: (nu in cell 0, mu in cell +-1)
    dH[((-1, 0, 0), (-1, 0, 0))] = spinor(dt)
    dS[((-1, 0, 0), (-1, 0, 0))] = spinor(ds)
    dH[((1, 0, 0), (1, 0, 0))] = spinor(dt)
    dS[((1, 0, 0), (1, 0, 0))] = spinor(ds)
    # onsite with splitting drift (dP channel)
    on = np.zeros((n, n), dtype=complex)
    on[0, 0] = DEPS + DDELTA / 2.0
    on[1, 1] = DEPS - DDELTA / 2.0
    dH[((0, 0, 0), (0, 0, 0))] = on
    return LocalizedDisplacementDerivative(
        dH, dS, Ru=(0, 0, 0), atoms=None, provenance="synthetic analytic"
    )


def _wire_exchange(G, nbasis):
    """ExchangePert2Sl wired for the single-species chain (unit seam)."""
    ex = ExchangePert2Sl.__new__(ExchangePert2Sl)
    ex.G = G
    ex.nbasis = nbasis
    ex.HR0 = G.H0
    ex.orb_dict = {0: [0, 1]}
    ex.ind_mag_atoms = [0]
    ex.basis_is_separated = False
    ex.Pdict = {}
    ex._prepare_Patom()
    short_Rlist = [(-2, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)]
    ex.short_Rlist = short_Rlist
    ex.R_ijatom_dict = {iR: [(0, 0)] for iR in range(len(short_Rlist))}
    ex.R_negative_index = {}
    ex.orb_decomposition = False
    return ex


def _pair_energy_sum(ex, Rs=(1, 0, 0)):
    """Sum over ZGRID of A^{00}-A^{33} and its displacement derivative."""
    val = 0.0
    dval = 0.0
    for z in ZGRID:
        A, dA, _rho, _ao, _dao = ex.get_AijR_rhoR(z)
        key = (Rs, 0, 0)
        a = A[key]
        da = dA[key]
        val += np.imag(a[0, 0] - a[3, 3])
        dval += np.imag(da[0, 0] - da[3, 3])
    return val, dval


def _supercell_J(tau, ncell=8, nk=8):
    """Production J between supercell atoms (0, m) at displacement tau.

    P is taken from the displaced model itself (the paper's convention:
    the exchange splitting drifts with the displaced configuration, which
    is what the production dP terms encode).
    """
    model = _SpinorChain(tau=tau, ncell=ncell)
    kpts = [[m / nk, 0, 0] for m in range(nk)]
    G = _green(model, kpts)
    ex = _wire_exchange(G, 2 * ncell)
    out = {}
    for m in range(1, ncell):
        val = 0.0
        for z in ZGRID:
            A = _get_sc_A(ex, m, z)
            val += np.imag(A[0, 0] - A[3, 3])
        out[m] = val
    return out


def _get_sc_A(ex, m, z):
    """A-matrix between supercell atoms 0 and m at energy z (R=0 block)."""
    from TB2J.pauli import pauli_block_all

    GR = ex.G.get_GR([(0, 0, 0)], z)
    M = GR[0]
    Gij = np.zeros((2, 2), dtype=complex)
    Gji = np.zeros((2, 2), dtype=complex)
    for s1, o1 in enumerate([0, 1]):
        for s2, o2 in enumerate([2 * m, 2 * m + 1]):
            Gij[s1, s2] = M[o1, o2]
            Gji[s2, s1] = M[o2, o1]
    a_ij = pauli_block_all(Gij)
    a_ji = pauli_block_all(Gji)
    Pi = ex.Pdict[0]
    A = np.zeros((4, 4), dtype=complex)
    for a, b in ([0, 0], [3, 3]):
        A[a, b] = np.trace(Pi @ a_ij[a] @ Pi @ a_ji[b]) / np.pi
    return A


def test_fd_localized_displacement_matches_production():
    # ---- analytic side: primitive model, vertex path ----
    prim = _SpinorChain(ncell=1)
    kpts = [[j / 64, 0, 0] for j in range(64)]
    G = _green(prim, kpts)
    ex = _wire_exchange(G, 2)
    disp = _analytic_vertex()
    ex.set_displacement(disp, J_only=False)
    J_analytic, O_analytic = _pair_energy_sum(ex)

    # ---- oracle side: supercell FD of the same production J ----
    ncell = 64
    h = 1e-6
    Jp = _supercell_J(+h, ncell=ncell, nk=1)
    Jm = _supercell_J(-h, ncell=ncell, nk=1)
    O_fd_right = (Jp[1] - Jm[1]) / (2 * h)
    O_fd_left = (Jp[ncell - 1] - Jm[ncell - 1]) / (2 * h)

    rel_r = abs(O_analytic - O_fd_right) / abs(O_fd_right)
    rel_l = abs(O_analytic - O_fd_left) / abs(O_fd_left)
    assert rel_r < 1e-8, (O_analytic, O_fd_right, rel_r)
    assert rel_l < 1e-8, (O_analytic, O_fd_left, rel_l)


def test_negative_controls_fixed_z(monkeypatch):
    """Fixed-z control on the pure-overlap model: wrong constant z is wrong.

    On the combined model the dS channel is a small fraction of O, so the
    control runs where the overlap channel carries the whole signal.
    """
    monkeypatch.setattr(T_mod, "LAM_T", 0.0)
    monkeypatch.setattr(T_mod, "DEPS", 0.0)
    monkeypatch.setattr(T_mod, "DDELTA", 0.0)
    prim = _SpinorChain(ncell=1)
    kpts = [[j / 64, 0, 0] for j in range(64)]
    G = _green(prim, kpts)
    ex = _wire_exchange(G, 2)
    ex.set_displacement(_analytic_vertex(), J_only=False)
    _, O_good = _pair_energy_sum(ex)

    class _FixedZ(DisplacementVertex):
        def __init__(self, base, z0):
            super().__init__(base.dH, base.dS, base.nbasis)
            self._z0 = z0

        def vertex(self, Rq, Rk, z):
            return super().vertex(Rq, Rk, self._z0)

    ex3 = _wire_exchange(G, 2)
    base = _analytic_vertex().to_vertex(2)
    ex3.epc = _FixedZ(base, 2.0 + 2.0j)
    ex3.Ru = (0, 0, 0)
    ex3.J_only = False
    ex3.disp = _analytic_vertex()
    _, O_fixedz = _pair_energy_sum(ex3)
    assert abs(O_fixedz - O_good) / abs(O_good) > 0.1, (O_fixedz, O_good)


def _run_analytic():
    G = _green(_SpinorChain(ncell=1), [[j / 64, 0, 0] for j in range(64)])
    ex = _wire_exchange(G, 2)
    ex.set_displacement(_analytic_vertex(), J_only=False)
    return _pair_energy_sum(ex)


def _run_fd(ncell=64, h=1e-6):
    Jp = _supercell_J(+h, ncell=ncell, nk=1)
    Jm = _supercell_J(-h, ncell=ncell, nk=1)
    return (Jp[1] - Jm[1]) / (2 * h)


def test_fd_pure_overlap_channel(monkeypatch):
    """NFR-002: dS-only model (constant t, drifting s) — no-dS control fails."""
    monkeypatch.setattr(T_mod, "LAM_T", 0.0)
    monkeypatch.setattr(T_mod, "DEPS", 0.0)
    monkeypatch.setattr(T_mod, "DDELTA", 0.0)
    _, O_good = _run_analytic()
    O_fd = _run_fd()
    assert abs(O_good - O_fd) / abs(O_fd) < 1e-8, (O_good, O_fd)
    assert abs(O_good) > 1e-6  # the overlap channel alone produces O

    disp = T_mod._analytic_vertex()
    disp.dS = {}
    G = _green(T_mod._SpinorChain(ncell=1), [[j / 64, 0, 0] for j in range(64)])
    ex = _wire_exchange(G, 2)
    ex.set_displacement(disp, J_only=False)
    _, O_nods = _pair_energy_sum(ex)
    assert abs(O_nods - O_good) / abs(O_good) > 0.1, (O_nods, O_good)


def test_fd_splitting_drift_channel(monkeypatch):
    """NFR-002 second case: only the on-site splitting drifts (dP channel)."""
    monkeypatch.setattr(T_mod, "LAM_T", 0.0)
    monkeypatch.setattr(T_mod, "LAM_S", 0.0)
    monkeypatch.setattr(T_mod, "DEPS", 0.0)
    monkeypatch.setattr(T_mod, "DDELTA", 0.6)
    _, O_good = _run_analytic()
    O_fd = _run_fd()
    assert abs(O_good) > 1e-6
    assert abs(O_good - O_fd) / abs(O_fd) < 1e-8, (O_good, O_fd)


def test_fd_orthogonal_limit(monkeypatch):
    """FR-006: S=I and dS=0 reduce to the orthogonal (EPW-form) path."""
    monkeypatch.setattr(T_mod, "S0", 0.0)
    _, O_good = _run_analytic()
    O_fd = _run_fd()
    assert abs(O_good - O_fd) / abs(O_fd) < 1e-8, (O_good, O_fd)
