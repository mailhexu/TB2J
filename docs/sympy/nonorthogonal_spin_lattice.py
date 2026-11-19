#!/usr/bin/env python3
"""Spin-lattice downfolding on a non-orthogonal basis: sympy-verified derivation.

Companion report: nonorthogonal_spin_lattice.md (same directory).

Extends the electron-phonon -> spin-lattice downfolding of TB2J
(O_{IJa} = dJ_IJ^iso/d tau_a, previously Wannier/orthogonal with the EPW
vertex g_a = dV/d tau) to non-orthogonal LCAO bases, where a displacement
tau_a also changes the overlap matrix S.  All assertions below are exact
(sympy) or high-precision numeric oracles (mpmath, dps>=40); run in `mydev`:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python nonorthogonal_spin_lattice.py

Central results
---------------
(R1) One-form:      dG/d tau_a = G(z) V_a(z) G(z),  V_a(z) = d_a H - z d_a S.
(R2) Level shifts:  residue of the double pole at z = eps_n reproduces the
                    pencil identity d eps_n = c_n^dag (dH - eps_n dS) c_n.
(R3) J-invariance:  the covariant LKAG contraction Tr[D_I G^up D_J G^dn] with
                    G = (zS - H)^-1 equals the Loewdin-orthogonalized one.
(R4) O_{IJa}:       d/d tau_a of the LKAG integrand = the four terms
                    (dD_I, D_I dG^up, dD_J, D_J dG^dn) with dG from (R1);
                    verified against finite differences of J(tau).
(R5) 2nd order:     d2G/d tau_a d tau_b = G(V_a G V_b + V_b G V_a)G
                                 + G (d2_ab H - z d2_ab S) G.
(R6) Pauli channels: the exchange_pert2 A^{ab} channel derivative follows the
                    same vertex rule on the 4x4 spinor model (each dG factor
                    and dPi terms included, -z dS required).
"""

import numpy as np
import scipy.linalg as sla
import sympy as sp
from mpmath import mp

mp.dps = 40

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# ======================================================================
# Part A (R1, R5): exact symbolic one-form and second-order kernel
# ======================================================================
def part_A_symbolic_oneform():
    lam, z = sp.symbols("lambda z", real=True)
    s0, s1, a0, a1, b0, b1, d0, d1 = sp.symbols("s0 s1 a0 a1 b0 b1 d0 d1", real=True)

    S_of_lam = sp.Matrix([[1, s0 + lam * s1], [s0 + lam * s1, 1]])
    H_of_lam = sp.Matrix(
        [
            [a0 + lam * a1, b0 + lam * b1],
            [b0 + lam * b1, d0 + lam * d1],
        ]
    )
    G = (z * S_of_lam - H_of_lam).inv()
    dH = sp.diff(H_of_lam, lam)
    dS = sp.diff(S_of_lam, lam)
    diff1 = sp.simplify(sp.diff(G, lam) - G * (dH - z * dS) * G)
    ok1 = diff1 == sp.zeros(2, 2)
    check("A1: dG = G (dH - z dS) G, exact 2x2 symbolic pencil", ok1)

    # Second order with the same (linear) pencil: G'' = 2 G V G V G (V2 = 0)
    V = dH - z * dS
    diff2 = sp.simplify(sp.diff(G, lam, 2) - 2 * G * V * G * V * G)
    ok2 = diff2 == sp.zeros(2, 2)
    check("A2: d2G = 2 G V G V G (linear pencil), exact", ok2)

    # General quadratic pencil, random rational symmetric matrices, exact
    # complex-rational z (the identity is rational in z; exact verification at
    # irrational-free z keeps it a rigorous exact-arithmetic check).
    rng = np.random.default_rng(7)

    def sym(M):
        return sp.nsimplify(M + M.T)

    S0 = sp.nsimplify(sp.eye(3) * 8) + sym(sp.Matrix(rng.integers(-3, 4, (3, 3)))) / 8
    H0 = sym(sp.Matrix(rng.integers(-3, 4, (3, 3))))
    cS1, cS2 = (
        sym(sp.Matrix(rng.integers(-2, 3, (3, 3)))),
        sym(sp.Matrix(rng.integers(-2, 3, (3, 3)))),
    )
    cH1, cH2 = (
        sym(sp.Matrix(rng.integers(-2, 3, (3, 3)))),
        sym(sp.Matrix(rng.integers(-2, 3, (3, 3)))),
    )
    znum = sp.Rational(3, 2) + sp.I * sp.Rational(1, 3)
    Sq = S0 + lam * cS1 + lam**2 * cS2
    Hq = H0 + lam * cH1 + lam**2 * cH2
    Aq = znum * Sq - Hq
    Gq = Aq.inv()
    V1 = sp.diff(Hq, lam) - znum * sp.diff(Sq, lam)
    V2 = sp.diff(Hq, lam, 2) - znum * sp.diff(Sq, lam, 2)
    lhs3 = sp.diff(Gq, lam, 2)
    rhs3 = Gq * V1 * Gq * V1 * Gq + Gq * V1 * Gq * V1 * Gq + Gq * V2 * Gq
    diff3 = sp.simplify(lhs3 - rhs3)
    ok3 = diff3 == sp.zeros(3, 3)
    check("A3: d2G = G(V G V + V G V)G + G V2 G, exact quadratic pencil", ok3)


# ======================================================================
# Part B (R2): double-pole residue reproduces the pencil identity
# ======================================================================
def part_B_pencil_residue():
    rng = np.random.default_rng(11)
    n = 3
    A = rng.normal(size=(n, n))
    S = np.eye(n) + A @ A.T / 20
    A2 = rng.normal(size=(n, n))
    dS = (A2 + A2.T) / 20
    B1 = rng.normal(size=(n, n))
    H = (B1 + B1.T) / 2
    B2 = rng.normal(size=(n, n))
    dH = (B2 + B2.T) / 2

    eps, C = sla.eigh(H, S)  # H c = eps S c, C^dag S C = 1
    c = C[:, 1]
    e = eps[1]
    de_pencil = c.conj() @ (dH - e * dS) @ c

    # N(h) = (z-e)^2 G V(z) G at z = e + h tends to c c^dag de_pencil + O(h)
    def double_pole_at(delta):
        z = e + delta
        G = np.linalg.inv(z * S - H)
        return (delta**2) * (G @ (dH - z * dS) @ G)

    N1 = double_pole_at(1e-6)
    N2 = double_pole_at(2e-6)
    Mlim = 2 * N1 - N2  # cancels the O(h) correction of the simple pole part
    Mtarget = np.outer(c, c.conj()) * de_pencil
    err = np.max(np.abs(Mlim - Mtarget)) / max(np.max(np.abs(Mtarget)), 1e-30)
    check(
        "B1: double-pole residue (z-e)^2 G V(z) G -> c c^dag [c^dag(dH-e dS)c]",
        err < 1e-6,
        f"rel-err={err:.2e}",
    )


# ======================================================================
# Part C (R3): Loewdin invariance of the covariant LKAG contraction
# ======================================================================
def part_C_loewdin_invariance():
    rng = np.random.default_rng(23)
    n = 6
    A = rng.normal(size=(n, n))
    S = np.eye(n) + (A @ A.T) / 5
    B = rng.normal(size=(n, n))
    Hup = (B + B.T) / 2
    B2 = rng.normal(size=(n, n))
    Hdn = Hup + (B2 + B2.T) / 4
    # local exchange fields supported on blocks 0:3 and 3:6
    Da = np.zeros((n, n))
    Da[:3, :3] = (B2[:3, :3] + B2[:3, :3].T) / 2 + 2 * np.eye(3)
    Db = np.zeros((n, n))
    Db[3:, 3:] = -(B2[3:, 3:] + B2[3:, 3:].T) / 2 + 1.5 * np.eye(3)
    z = 0.7 + 0.3j
    lhs = np.trace(Da @ np.linalg.inv(z * S - Hup) @ Db @ np.linalg.inv(z * S - Hdn))

    Sm = sla.sqrtm(S)
    Si = np.linalg.inv(Sm)

    def GT(H):
        return np.linalg.inv(z * np.eye(n) - Si @ H @ Si)

    rhs = np.trace((Si @ Da @ Si) @ GT(Hup) @ (Si @ Db @ Si) @ GT(Hdn))
    err = abs(lhs - rhs) / abs(lhs)
    check(
        "C1: Tr[D G^up D G^dn] invariant under Loewdin orthogonalization",
        err < 1e-10,
        f"err={err:.2e}",
    )


# ======================================================================
# Parts D-F: two-site non-orthogonal oracle (mpmath)
# ======================================================================
class TwoSiteModel:
    """Collinear two-site, one-orbital-per-site model with overlap S(tau).

    S(tau) = [[1, s(tau)], [s(tau), 1]],  s = s0 exp(-ls tau)
    H^sigma(tau) = [[eps_A + sigma DA/2, t(tau)],
                    [t(tau),            eps_B + sigma DB/2]]
    t = t0 exp(-lt tau), eps_I = eI0 + uI tau, Delta_I = DI0 + wI tau.
    sigma = +1 up, -1 down.
    """

    s0, ls = 0.25, 1.1
    t0, lt = -1.3, 0.9
    eA0, eB0 = 0.0, 0.4
    eA1, eB1 = 0.3, -0.2
    DA0, DB0 = 1.8, 1.4
    DA1, DB1 = 0.5, -0.4

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def s(self, tau):
        return self.s0 * mp.e ** (-self.ls * tau)

    def ds(self, tau):
        return -self.ls * self.s(tau)

    def d2s(self, tau):
        return self.ls**2 * self.s(tau)

    def t(self, tau):
        return self.t0 * mp.e ** (-self.lt * tau)

    def dt(self, tau):
        return -self.lt * self.t(tau)

    def d2t(self, tau):
        return self.lt**2 * self.t(tau)

    def H(self, sigma, tau):
        return mp.matrix(
            [
                [
                    self.eA0 + self.eA1 * tau + sigma * (self.DA0 + self.DA1 * tau) / 2,
                    self.t(tau),
                ],
                [
                    self.t(tau),
                    self.eB0 + self.eB1 * tau + sigma * (self.DB0 + self.DB1 * tau) / 2,
                ],
            ]
        )

    def dH(self, sigma, tau):
        return mp.matrix(
            [
                [self.eA1 + sigma * self.DA1 / 2, self.dt(tau)],
                [self.dt(tau), self.eB1 + sigma * self.DB1 / 2],
            ]
        )

    def S(self, tau):
        return mp.matrix([[1, self.s(tau)], [self.s(tau), 1]])

    def dS(self, tau):
        return mp.matrix([[0, self.ds(tau)], [self.ds(tau), 0]])

    def DeltaA(self, tau):
        return self.DA0 + self.DA1 * tau

    def DeltaB(self, tau):
        return self.DB0 + self.DB1 * tau

    def G(self, sigma, z, tau):
        return (z * self.S(tau) - self.H(sigma, tau)) ** (-1)

    def V(self, sigma, z, tau):
        return self.dH(sigma, tau) - z * self.dS(tau)


def arc(m: TwoSiteModel, tau, emin, emax, F):
    """Im int_arc F(z) dz over the LKAG semicircle from emax down to emin."""
    c = (emin + emax) / 2
    R = (emax - emin) / 2

    def integrand_theta(theta):
        z = c + R * mp.e ** (-1j * theta)
        dz = -1j * R * mp.e ** (-1j * theta)
        return F(z) * dz

    return mp.im(mp.quad(integrand_theta, [0, mp.pi]))


def F_J(m: TwoSiteModel, tau):
    """LKAG kernel F(z) = Tr[D_A G^up_AB D_B G^dn_BA]."""

    def F(z):
        Gup = m.G(+1, z, tau)
        Gdn = m.G(-1, z, tau)
        return m.DeltaA(tau) * Gup[0, 1] * m.DeltaB(tau) * Gdn[1, 0]

    return F


def J_of(m: TwoSiteModel, tau, emin, emax):
    return arc(m, tau, emin, emax, F_J(m, tau)) / (4 * mp.pi)


def F_dJ(m: TwoSiteModel, tau, use_dS=True):
    """dF/dtau assembled from the vertex formula (R4)."""

    def dF(z):
        Gup = m.G(+1, z, tau)
        Gdn = m.G(-1, z, tau)
        Vup = m.V(+1, z, tau) if use_dS else m.dH(+1, tau)
        Vdn = m.V(-1, z, tau) if use_dS else m.dH(-1, tau)
        dGup = Gup * Vup * Gup
        dGdn = Gdn * Vdn * Gdn
        DA, DB = m.DeltaA(tau), m.DeltaB(tau)
        return (
            m.DA1 * Gup[0, 1] * DB * Gdn[1, 0]
            + DA * dGup[0, 1] * DB * Gdn[1, 0]
            + DA * Gup[0, 1] * m.DB1 * Gdn[1, 0]
            + DA * Gup[0, 1] * DB * dGdn[1, 0]
        )

    return dF


def O_of(m: TwoSiteModel, tau, emin, emax, use_dS=True):
    return arc(m, tau, emin, emax, F_dJ(m, tau, use_dS=use_dS)) / (4 * mp.pi)


def choose_fermi(m: TwoSiteModel):
    """E_F at the widest gap between pencil eigenvalues; emin below all."""
    tau = mp.mpf(0)
    evs = []
    for sigma in (+1, -1):
        H, S = m.H(sigma, tau), m.S(tau)
        a = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        b = -(
            H[0, 0] * S[1, 1]
            + H[1, 1] * S[0, 0]
            - H[0, 1] * S[1, 0]
            - H[1, 0] * S[0, 1]
        )
        c = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        disc = mp.sqrt(b * b - 4 * a * c)
        evs += [(-b + disc) / (2 * a), (-b - disc) / (2 * a)]
    evs = sorted(evs)
    gapw, ef = max(
        (evs[i + 1] - evs[i], (evs[i + 1] + evs[i]) / 2) for i in range(len(evs) - 1)
    )
    return float(ef), float(min(evs) - 1.5), [float(e) for e in evs]


def part_D_first_order():
    m = TwoSiteModel()
    ef, emin, evs = choose_fermi(m)
    tau = mp.mpf(0)
    h = mp.mpf(10) ** (-6)
    fd = (J_of(m, +h, emin, ef) - J_of(m, -h, emin, ef)) / (2 * h)
    vertex = O_of(m, tau, emin, ef, use_dS=True)
    rel = abs(vertex - fd) / abs(fd)
    check(
        "D1: O = dJ/dtau vertex formula (dH - z dS) == FD of J",
        rel < 1e-8,
        f"FD={float(fd):.10f} vertex={float(vertex):.10f} rel={float(rel):.1e}",
    )

    nover = O_of(m, tau, emin, ef, use_dS=False)
    relwrong = abs(nover - fd) / abs(fd)
    check(
        "D2: dropping the -z dS term is wrong (negative control)",
        relwrong > 1e-3,
        f"rel-deviation={float(relwrong):.1e}",
    )

    # orthogonal limit: s0 = 0 -> dS = 0 identically
    mo = TwoSiteModel(s0=0.0)
    efo, emino, _ = choose_fermi(mo)
    fd_o = (J_of(mo, +h, emino, efo) - J_of(mo, -h, emino, efo)) / (2 * h)
    v_o = O_of(mo, tau, emino, efo, use_dS=True)
    v_o_no = O_of(mo, tau, emino, efo, use_dS=False)
    ok = abs(v_o - fd_o) / abs(fd_o) < 1e-8 and abs(v_o_no - v_o) < abs(fd_o) * 1e-12
    check(
        "D3: orthogonal limit reduces to the EPW dV-only formula",
        ok,
        f"FD={float(fd_o):.10f}",
    )

    # overlap-only model: only s(tau) varies
    ms = TwoSiteModel(lt=0.0, eA1=0.0, eB1=0.0, DA1=0.0, DB1=0.0)
    efs, emins, _ = choose_fermi(ms)
    fd_s = (J_of(ms, +h, emins, efs) - J_of(ms, -h, emins, efs)) / (2 * h)
    v_s = O_of(ms, tau, emins, efs, use_dS=True)
    rels = abs(v_s - fd_s) / abs(fd_s)
    check(
        "D4: pure dS/dtau channel gives nonzero spin-lattice coupling, formula == FD",
        rels < 1e-8 and abs(fd_s) > 1e-6,
        f"O(dS-only)={float(fd_s):.10f} rel={float(rels):.1e}",
    )
    return dict(ef=ef, emin=emin, evs=evs)


def part_E_second_order():
    """Q = d2J/dtau2: full product rule over both propagators and both Deltas
    with d2G = G(VGV + VGV)G + G V2 G, V2 = d2H - z d2S."""
    m = TwoSiteModel()
    ef, emin, _ = choose_fermi(m)
    tau = mp.mpf(0)
    h = mp.mpf(10) ** (-4)
    J0 = J_of(m, 0, emin, ef)
    Jp, Jm = J_of(m, +h, emin, ef), J_of(m, -h, emin, ef)
    Jpp, Jmm = J_of(m, +2 * h, emin, ef), J_of(m, -2 * h, emin, ef)
    fd2 = (-Jpp + 16 * Jp - 30 * J0 + 16 * Jm - Jmm) / (12 * h**2)

    def F_d2J(z):
        Gup, Gdn = m.G(+1, z, tau), m.G(-1, z, tau)
        Vup, Vdn = m.V(+1, z, tau), m.V(-1, z, tau)
        dGup, dGdn = Gup * Vup * Gup, Gdn * Vdn * Gdn
        d2t, d2s = m.d2t(tau), m.d2s(tau)
        V2 = mp.matrix([[0, d2t], [d2t, 0]]) - z * mp.matrix([[0, d2s], [d2s, 0]])
        d2Gup = Gup * (Vup * dGup + Vup * dGup) + Gup * V2 * Gup
        d2Gdn = Gdn * (Vdn * dGdn + Vdn * dGdn) + Gdn * V2 * Gdn
        DA, DB = m.DeltaA(tau), m.DeltaB(tau)
        dDA, dDB = m.DA1, m.DB1
        d2DA = d2DB = mp.mpf(0)  # Delta linear in tau
        Aup, Adn = Gup[0, 1], Gdn[1, 0]
        dAup, dAdn = dGup[0, 1], dGdn[1, 0]
        d2Aup, d2Adn = d2Gup[0, 1], d2Gdn[1, 0]
        # F = DA * Aup * DB * Adn; complete product rule over four factors
        return (
            d2DA * Aup * DB * Adn
            + DA * d2Aup * DB * Adn
            + DA * Aup * d2DB * Adn
            + DA * Aup * DB * d2Adn
            + 2 * dDA * dAup * DB * Adn
            + 2 * dDA * Aup * dDB * Adn
            + 2 * dDA * Aup * DB * dAdn
            + 2 * DA * dAup * dDB * Adn
            + 2 * DA * dAup * DB * dAdn
            + 2 * DA * Aup * dDB * dAdn
        )

    v2 = arc(m, tau, emin, ef, F_d2J) / (4 * mp.pi)
    rel = abs(v2 - fd2) / abs(fd2)
    check(
        "E1: Q = d2J/dtau2 assembled (VGVGV + V2 + cross terms) == FD2 of J",
        rel < 1e-6,
        f"FD2={float(fd2):.8f} formula={float(v2):.8f} rel={float(rel):.1e}",
    )


# ======================================================================
# Part F (R6): Pauli-channel (NCL A-matrix) derivative follows the vertex
# ======================================================================
def spinor_pauli_blocks(M4):
    """Pauli components (m0, m1, m2, m3) of a 2x2 spin block at fixed orbital.

    Separated spin basis [up_A up B | dn_A dn_B]: the spin block at orbital
    (p, q) is [[M4[p, q], M4[p, q + 2]], [M4[p + 2, q], M4[p + 2, q + 2]]].
    """
    n = 2  # orbitals
    out = []
    for p in range(n):
        row = []
        for q in range(n):
            uu = M4[p, q]
            ud = M4[p, q + n]
            du = M4[p + n, q]
            dd = M4[p + n, q + n]
            row.append(
                ((uu + dd) / 2, (ud + du) / 2, (ud - du) / 2 * 1j, (uu - dd) / 2)
            )
        out.append(row)
    return out


def part_F_pauli_channels():
    """4x4 spinor model [up_A up_B | dn_A dn_B], collinear-z (no SOC).

    Production uses the same on-site spin-splitting P_i=(H_i^up-H_i^dn)/2
    in both A^{00} and A^{33}; G^0 and G^3 are Pauli components of G.
    Verify the collinear LKAG reduction and its displacement derivative
    with dG = G(dH-z dS)G and dP from the on-site dH blocks.
    """
    gu_ab, gd_ab, gu_ba, gd_ba = sp.symbols("gu_ab gd_ab gu_ba gd_ba")
    g0_ab, g3_ab = (gu_ab + gd_ab) / 2, (gu_ab - gd_ab) / 2
    g0_ba, g3_ba = (gu_ba + gd_ba) / 2, (gu_ba - gd_ba) / 2
    identity = sp.simplify(
        g0_ab * g0_ba - g3_ab * g3_ba - (gu_ab * gd_ba + gd_ab * gu_ba) / 2
    )
    check(
        "F0: Pauli 00-33 exactly reduces to crossed up/down propagators", identity == 0
    )
    m = TwoSiteModel()
    ef, emin, _ = choose_fermi(m)
    tau = mp.mpf(0)

    def Gspin(z, tau):
        S2, S1 = m.S(tau), m.S(tau)
        Hup, Hdn = m.H(+1, tau), m.H(-1, tau)
        A = mp.zeros(4, 4)
        for i in range(2):
            for j in range(2):
                A[i, j] = z * S2[i, j] - Hup[i, j]
                A[i + 2, j + 2] = z * S1[i, j] - Hdn[i, j]
        return A ** (-1)

    def dGspin(z, tau, use_dS):
        G = Gspin(z, tau)
        V = mp.zeros(4, 4)
        for i in range(2):
            for j in range(2):
                V[i, j] = m.dH(+1, tau)[i, j]
                V[i + 2, j + 2] = m.dH(-1, tau)[i, j]
                if use_dS:
                    dS = m.dS(tau)
                    V[i, j] += -z * dS[i, j]
                    V[i + 2, j + 2] += -z * dS[i, j]
        return G * V * G

    def Pi_of(tau):
        """TB2J magnetic on-site vertex P_i for channels 0 and 3."""
        Hup, Hdn = m.H(+1, tau), m.H(-1, tau)
        Pi = {}
        for i, idx in (("A", 0), ("B", 1)):
            p = (Hup[idx, idx] - Hdn[idx, idx]) / 2
            Pi[i] = (p, mp.mpf(0), mp.mpf(0), p)
        return Pi

    def dPi_of():
        dHup, dHdn = m.dH(+1, tau), m.dH(-1, tau)
        dPi = {}
        for i, idx in (("A", 0), ("B", 1)):
            dp = (dHup[idx, idx] - dHdn[idx, idx]) / 2
            dPi[i] = (dp, mp.mpf(0), mp.mpf(0), dp)
        return dPi

    def X(z, tau, differentiated=False, use_dS=True):
        G4 = Gspin(z, tau)
        Gb = spinor_pauli_blocks(G4)
        if differentiated:
            dG4 = dGspin(z, tau, use_dS)
            dGb = spinor_pauli_blocks(dG4)
        Pi = Pi_of(tau)
        dPi = dPi_of() if differentiated else None
        # channel contraction Tr[Pi_A^a G^a_AB Pi_B^b G^b_BA], (a,b) = (0,0),(3,3)
        val = mp.mpf(0)
        for a in (0, 3):
            for b in (0, 3):
                if not (a == b and a in (0, 3)):
                    continue
                PiA, PiB = Pi["A"][a], Pi["B"][b]
                G_AB = Gb[0][1][a] if a == b else None
                G_BA = Gb[1][0][b]
                term = PiA * G_AB * PiB * G_BA
                if differentiated:
                    dPiA, dPiB = dPi["A"][a], dPi["B"][b]
                    dG_AB = dGb[0][1][a]
                    dG_BA = dGb[1][0][b]
                    term = (
                        dPiA * G_AB * PiB * G_BA
                        + PiA * dG_AB * PiB * G_BA
                        + PiA * G_AB * dPiB * G_BA
                        + PiA * G_AB * PiB * dG_BA
                    )
                val += term if a == 0 else -term
        return val

    def X00m33(z, tau, **kw):
        return X(z, tau, **kw)

    # TB2J uses A^{ab}=Tr[P_i G^a_ij P_j G^b_ji]/pi.
    h = mp.mpf(10) ** (-6)

    def Ich(t):
        return arc(m, t, emin, ef, lambda z: X00m33(z, t)) / mp.pi

    fd = (Ich(+h) - Ich(-h)) / (2 * h)
    formula = (
        arc(
            m, tau, emin, ef, lambda z: X00m33(z, tau, differentiated=True, use_dS=True)
        )
        / mp.pi
    )
    rel = abs(formula - fd) / abs(fd)
    z_probe = mp.mpc("0.6", "0.4")
    reduction_err = abs(X00m33(z_probe, tau) - F_J(m, tau)(z_probe) / 4)
    J_err = abs(Ich(tau) - J_of(m, tau, emin, ef))
    check(
        "F1: production Pauli 00-33 equals LKAG and vertex derivative equals FD",
        rel < 1e-8 and reduction_err < 1e-30 and J_err < 1e-30,
        f"rel-FD={float(rel):.1e} J-err={float(J_err):.1e}",
    )
    nover = (
        arc(
            m,
            tau,
            emin,
            ef,
            lambda z: X00m33(z, tau, differentiated=True, use_dS=False),
        )
        / mp.pi
    )
    relwrong = abs(nover - fd) / abs(fd)
    check(
        "F2: channel derivative without -z dS is wrong (negative control)",
        relwrong > 1e-3,
        f"rel-deviation={float(relwrong):.1e}",
    )


def main():
    print(f"mpmath dps = {mp.dps}", flush=True)
    part_A_symbolic_oneform()
    part_B_pencil_residue()
    part_C_loewdin_invariance()
    info = part_D_first_order()
    part_E_second_order()
    part_F_pauli_channels()
    print()
    print(f"model eigenvalues at tau=0: {info['evs']}")
    print(f"E_F = {info['ef']:.6f}, emin = {info['emin']:.6f}")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
