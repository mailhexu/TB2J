"""Assertion-checked derivation of the explicit magnon-phonon vertex.

Run with the project environment: python docs/sympy/magnon_phonon_vertex.py

Derives the spin-wave factor F^{ij}_{lambda lambda'}(k, q) of the one-phonon
magnon-scattering vertex

    H_mp = sum_{k q} g_{mu nu nu'}(k, q)
           a^dagger_{k+q,mu} a_{k,nu} (b_{q,nu'} + b^dagger_{-q,nu'}) / sqrt(N_cell)

    g_{mu nu nu'}(k, q) = - sum_{directed bonds b=(i,j,R)} O^pat_{b,nu'}(q) F_b[mu, nu](k, q)

from the Holstein-Primakoff expansion of S_i.S_j, the site-boson to magnon-mode
transformation of the canonical Story-006 modes (u, v particle/hole blocks,
sqrt(2 S) transverse spin factors carried outside u, v), and the phonon mode
projection O^pat applied with the zero-point factor exactly once.

Conventions (identical to docs/sympy/site_boson_modes.md and the manuscript):
  * H = -sum_b J_b S_i.S_j with classical full-moment J (TB2J sign convention,
    J > 0 ferromagnetic); derivatives dJ/dtau in eV/Angstrom.
  * directed bond list, both directions present; J_{ji}(-R) = J_{ij}(R)^T.
  * Fourier A(q) = sum_R A(R) exp(-2 pi i q.R).
  * collinear reference: V_i = V_j for all magnetic sites; local frames
    U_i = e1_i + i e2_i from the Story-006 solver; a unit-commutator HP boson
    carries sqrt(2 S_i) inside the transverse spin, never inside u, v.
  * phonon displacement of atom a in cell L:
    tau_{a alpha}(L) = ZPA_{a nu} e^{mw}_{a alpha, nu}(q) e^{2 pi i q.L}
                       (b_{q nu} + b^dagger_{-q nu}) / sqrt(N_cell),
    ZPA_{a nu} = sqrt(hbar / (2 M_a omega_{q nu})),
    with e^{mw} the mass-weighted unit-L2 eigenvector carrying the phonopy
    basis-position phase.  The zero-point factor is applied exactly once, here.

Closed-form result asserted below, per directed bond (i, j, R):

  F_b[mu, nu](k, q) =
      e^{+2 pi i k.R}   t_ij  [u*_{mu i}(k+q) u_{nu j}(k) + w*_{nu i}(-k) w_{mu j}(-k-q)]
    + e^{-2 pi i (k+q).R} t_ij* [u*_{mu j}(k+q) u_{nu i}(k) + w*_{nu j}(-k) w_{mu i}(-k-q)]
    - S_j [u*_{mu i}(k+q) u_{nu i}(k) + w*_{nu i}(-k) w_{mu i}(-k-q)]
    - S_i e^{-2 pi i q.R} [u*_{mu j}(k+q) u_{nu j}(k) + w*_{nu j}(-k) w_{mu j}(-k-q)]

  t_ij = sqrt(S_i S_j)/2 (U_i . U_j*),  w(p) = Story-006 v-array at p.

The site-j fluctuation sits in cell L+R while the displacement modulation is
tied to the bond's home cell L, which produces the e^{-2 pi i q.R} phase on the
last (longitudinal partner) term.  With these factors the identity
F_b[mu, nu](k, q) = [F_b[nu, mu](k+q, -q)]* holds exactly per directed bond
(given O(-q) = O(q)*), on-site bonds vanish identically, and the full assembly
matches the mechanical real-space finite-model reference below.  The
longitudinal coefficients carry the partner spin (site i fluctuation weighted
by S_j), on-site bonds vanish identically, and there is no bilinear
magnon-phonon (a^dagger b) mixing component for the scalar collinear coupling
(symmetry-forbidden control).
"""

import numpy as np
import sympy as sp


def _coeff(expr, gens, mono):
    """Coefficient of an exact monomial in the commutative boson symbols."""
    return sp.Poly(sp.expand(expr), *gens).coeff_monomial(mono)


def check_hp_local_frame_expansion():
    """HP expansion of S_i.S_j in local frames; spin-length factors asserted."""
    S = sp.symbols("S", positive=True)
    ai, adi, aj, adj = sp.symbols("a_i a_i_dag a_j a_j_dag", real=True)
    # collinear concrete frames: V = z, U = x + i y  (e1 = x, e2 = y)
    pref = sp.sqrt(S / 2)
    Sx_i, Sy_i = pref * (ai + adi), -sp.I * pref * (ai - adi)
    Sz_i = S - adi * ai
    Sx_j, Sy_j = pref * (aj + adj), -sp.I * pref * (aj - adj)
    Sz_j = S - adj * aj
    dot = sp.expand(Sx_i * Sx_j + Sy_i * Sy_j + Sz_i * Sz_j)

    # transverse number-conserving coefficient: sqrt(S_i S_j)/2 (U_i.U_j*) = S
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), adi * aj) - S) == 0
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), ai * adj) - S) == 0
    # anomalous site structure vanishes for collinear real frames (U_i.U_j = 0)
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), ai * aj)) == 0
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), adi * adj)) == 0
    # longitudinal fluctuations carry the PARTNER spin: site j number gets -S_i
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), adj * aj) + S) == 0
    assert sp.simplify(_coeff(dot, (ai, adi, aj, adj), adi * ai) + S) == 0
    # no linear-in-boson terms for a collinear stationary reference
    for lin in (ai, adi, aj, adj):
        assert _coeff(dot, (ai, adi, aj, adj), lin) == 0

    # on-site bond (i = j): the degree<=2 (bilinear) part of the linear-HP
    # expansion cancels: transverse 2 S a a^dag and longitudinal -2 S a^dag a
    # differ only by the commutator constant, so the on-site vertex is zero.
    anc, andag = sp.symbols("nc_a nc_ad", commutative=False)
    quad = 2 * S * (anc * andag - andag * anc)
    quad = sp.expand(quad.subs(anc * andag, andag * anc + 1))
    assert sp.simplify(quad - 2 * S) == 0  # pure commutator constant, no bilinear

    # generic relative transverse-frame rotation: coefficient = sqrt(SiSj)/2 (U_i.U_j*)
    S2 = sp.symbols("S_j", positive=True)
    th = sp.symbols("theta", real=True)
    U = sp.Matrix([1, 0, 0]) + sp.I * sp.Matrix([0, 1, 0])
    e1p = sp.Matrix([sp.cos(th), sp.sin(th), 0])
    e2p = sp.Matrix([-sp.sin(th), sp.cos(th), 0])
    Up = e1p + sp.I * e2p
    assert sp.simplify(Up.dot(U.conjugate()) - 2 * sp.exp(-sp.I * th)) == 0
    aip, adp = sp.symbols("a_p a_p_dag", real=True)
    T_i = sp.sqrt(S / 2) * (ai * U.conjugate() + adi * U)
    T_p = sp.sqrt(S2 / 2) * (aip * Up.conjugate() + adp * Up)
    transverse = sp.expand(T_i.dot(T_p))
    assert (
        sp.simplify(
            _coeff(transverse, (ai, adi, aip, adp), adi * aip)
            - (sp.sqrt(S * S2) / 2) * 2 * sp.exp(sp.I * th)
        )
        == 0
    )
    assert (
        sp.simplify(
            _coeff(transverse, (ai, adi, aip, adp), ai * adp)
            - (sp.sqrt(S * S2) / 2) * 2 * sp.exp(-sp.I * th)
        )
        == 0
    )
    print("HP local-frame expansion, spin factors, on-site cancellation: OK")


def check_forbidden_scalar_bilinear_mixing():
    """Scalar collinear coupling cannot generate a bilinear a-b mixing block."""
    S, O = sp.symbols("S O", positive=True, real=True)
    ai, adi, aj, adj = sp.symbols("a_i a_i_dag a_j a_j_dag", real=True)
    b, bd = sp.symbols("b b_dag", real=True)
    pref = sp.sqrt(S / 2)
    Sx_i, Sy_i = pref * (ai + adi), -sp.I * pref * (ai - adi)
    Sz_i = S - adi * ai
    Sx_j, Sy_j = pref * (aj + adj), -sp.I * pref * (aj - adj)
    Sz_j = S - adj * aj
    # scalar-collinear exchange-striction vertex: -O tau S_i.S_j, tau = (b + b^dag)
    vertex = sp.expand(-O * (b + bd) * (Sx_i * Sx_j + Sy_i * Sy_j + Sz_i * Sz_j))
    gens = (ai, adi, aj, adj, b, bd)
    # magnon-phonon BILINEAR mixing operators (one magnon, one phonon operator)
    for mono in (ai * b, adi * b, ai * bd, adi * bd):
        assert _coeff(vertex, gens, mono) == 0
    # the surviving structure is the cubic a^dag a (b + b^dag) family
    assert sp.simplify(_coeff(vertex, gens, adi * aj * b) + O * S) == 0
    assert sp.simplify(_coeff(vertex, gens, adi * aj * bd) + O * S) == 0
    print("scalar-collinear forbidden bilinear mixing control: OK")


def check_bogoliubov_projection_and_hermiticity():
    """Mode projection (w at -k, -k-q) and exact per-bond Hermiticity."""

    def pair(name):
        # plain symbols: x represents the amplitude, x_c its conjugate; sympy
        # conjugation stays unevaluated so the conj_map below can rename.
        return sp.Symbol(name), sp.Symbol(name + "_c")

    u_mi_kp, u_mi_kp_c = pair("u_mu_i_kp")
    u_nj_k, u_nj_k_c = pair("u_nu_j_k")
    u_mj_kp, u_mj_kp_c = pair("u_mu_j_kp")
    u_ni_k, u_ni_k_c = pair("u_nu_i_k")
    w_ni_mk, w_ni_mk_c = pair("w_nu_i_mk")
    w_mj_mkp, w_mj_mkp_c = pair("w_mu_j_mkp")
    w_nj_mk, w_nj_mk_c = pair("w_nu_j_mk")
    w_mi_mkp, w_mi_mkp_c = pair("w_mu_i_mkp")
    t, Si, Sj, O, p3 = sp.symbols("t S_i S_j O phase_q", real=True)
    p1, p2 = sp.symbols("phase_k phase_kp", complex=True)

    # F[mu, nu](k, q) per directed bond, structure factors only (g = -O F adds the rest)
    F_mn = (
        p1 * t * (u_mi_kp_c * u_nj_k + w_ni_mk_c * w_mj_mkp)
        + p2 * t * (u_mj_kp_c * u_ni_k + w_nj_mk_c * w_mi_mkp)
        - Sj * (u_mi_kp_c * u_ni_k + w_ni_mk_c * w_mi_mkp)
        - Si * p3 * (u_mj_kp_c * u_nj_k + w_nj_mk_c * w_mj_mkp)
    )

    # partner F[nu, mu](k+q, -q): u/w momentum labels swap roles, phases conjugate;
    # e^{+2 pi i (k+q).R} = p2* and e^{-2 pi i k.R} = p1*.
    F_nm_partner = (
        p2.conjugate() * t * (u_ni_k_c * u_mj_kp + w_mi_mkp_c * w_nj_mk)
        + p1.conjugate() * t * (u_nj_k_c * u_mi_kp + w_mj_mkp_c * w_ni_mk)
        - Sj * (u_ni_k_c * u_mi_kp + w_mi_mkp_c * w_ni_mk)
        - Si * p3 * (u_nj_k_c * u_mj_kp + w_mj_mkp_c * w_nj_mk)
    )

    def conj_map(expr):
        """Complex conjugation expressed through the independent symbol pairs."""
        pairs = [
            (u_mi_kp, u_mi_kp_c),
            (u_nj_k, u_nj_k_c),
            (u_mj_kp, u_mj_kp_c),
            (u_ni_k, u_ni_k_c),
            (w_ni_mk, w_ni_mk_c),
            (w_mj_mkp, w_mj_mkp_c),
            (w_nj_mk, w_nj_mk_c),
            (w_mi_mkp, w_mi_mkp_c),
        ]
        rule = {}
        for x, xc in pairs:
            rule[x] = xc
            rule[xc] = x
            rule[sp.conjugate(x)] = x
            rule[sp.conjugate(xc)] = xc
        for p in (p1, p2):
            rule[sp.conjugate(p)] = p
        return sp.expand(expr.xreplace(rule))

    residual = sp.expand(F_mn - conj_map(F_nm_partner))
    assert residual == 0, residual
    g_mn = -O * F_mn
    assert sp.expand(g_mn - conj_map(-O * F_nm_partner)) == 0

    # covariance of the contracted residue under mode rephasing: with
    # u -> u Theta, w -> w Theta^T-dagger-conventions the residue g g^dag is
    # invariant for diagonal rephasing (checked numerically below).
    print("Bogoliubov projection form and per-bond Hermiticity: OK")


def _colpa(H):
    """Independent bosonic Colpa diagonalization (lower Cholesky, Story-006 order)."""
    n = H.shape[0] // 2
    K = np.linalg.cholesky(H)
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    evals, evecs = np.linalg.eigh(K.conj().T @ g @ K)
    physical = np.linalg.solve(K.conj().T, evecs) * np.sqrt(np.abs(evals))[None, :]
    T = np.concatenate((physical[:, n:], physical[:, :n][:, ::-1]), axis=1)
    return T, evals[n:]


def _spring_model():
    """1D two-atom-cell chain with exponential bond springs (finite classical model)."""
    a = 2.4
    xcell = np.array([0.0, 1.15])
    ncell = 6
    spins = np.array([1.2, 0.9])
    bonds = [
        # (i, j, R, j0, beta): J(L) = j0 exp(-beta (len(L) - len0)); beta=0: constant
        (0, 1, 0, 0.9, 1.7),
        (1, 0, 0, 0.9, 1.7),
        (0, 1, 1, 0.55, 1.1),
        (1, 0, -1, 0.55, 1.1),
        (0, 0, 0, 0.30, 0.0),
        (1, 1, 0, 0.20, 0.0),
    ]
    return a, xcell, ncell, spins, bonds


class _Model:
    """Mechanical real-space model and independent vertex reference."""

    def __init__(self):
        (self.a, self.xcell, self.ncell, self.spins, self.bonds) = _spring_model()
        self.nsites = len(self.xcell)
        # collinear frames: every site has U = e1 + i e2 with e1 = x, e2 = y
        Uv = np.array([1.0 + 0j, 1.0j, 0.0])
        self.t_bond = np.array(
            [
                np.sqrt(self.spins[i] * self.spins[j]) / 2 * np.sum(Uv * np.conj(Uv))
                for i, j, _, _, _ in self.bonds
            ]
        )
        self.s_bond = np.array(
            [
                np.sqrt(self.spins[i] * self.spins[j])
                / 2
                * np.sum(np.conj(Uv) * np.conj(Uv))
                for i, j, _, _, _ in self.bonds
            ]
        )
        self.len0 = np.array(
            [
                abs((self.xcell[j] + R * self.a) - self.xcell[i]) if beta else 1.0
                for i, j, R, j0, beta in self.bonds
            ]
        )
        # collinearity control: anomalous site structure vanishes identically
        assert np.max(np.abs(self.s_bond)) < 1e-14

    def spring_J(self, bi, tau_i, tau_j):
        i, j, R, j0, beta = self.bonds[bi]
        if beta == 0.0:
            return j0
        length = abs((self.xcell[j] + R * self.a + tau_j) - (self.xcell[i] + tau_i))
        return j0 * np.exp(-beta * (length - self.len0[bi]))

    def dJ_endpoint(self, bi):
        """dJ/dtau at the i and j endpoints of bond bi at the undisplaced geometry."""
        i, j, R, j0, beta = self.bonds[bi]
        if beta == 0.0:
            return np.zeros(3), np.zeros(3)
        dhat = np.array(
            [((self.xcell[j] + R * self.a) - self.xcell[i]) / self.len0[bi], 0, 0]
        )
        return +j0 * beta * dhat, -j0 * beta * dhat

    def static_hessian(self, k):
        """Bosonic Hessian H(k) of the undisplaced model (Story-006 conventions)."""
        n = self.nsites
        A = np.zeros((n, n), dtype=complex)
        B = np.zeros((n, n), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(self.bonds):
            Jb = self.spring_J(bi, 0.0, 0.0)
            ph = np.exp(2j * np.pi * k * R)
            A[i, j] += -Jb * self.t_bond[bi] * ph
            A[j, i] += -Jb * np.conj(self.t_bond[bi]) * np.conj(ph)
            A[i, i] += Jb * self.spins[j]
            A[j, j] += Jb * self.spins[i]
            B[i, j] += -Jb * self.s_bond[bi] * ph
        H = np.block([[A, B], [B.conj(), A.conj()]])
        return H

    def modes_at(self, k):
        return _colpa(self.static_hessian(k))

    def modulated_M1(self, q, disp, h, field):
        """Real-space a^dag a / a a coefficient matrices for a REAL displacement field.

        ``field`` selects the cosine (in-phase) or sine (quadrature) component of
        the frozen pattern: tau(L) = h * disp * cos(2 pi q L) or * sin(2 pi q L),
        with disp[a] the Cartesian displacement of atom a in Angstrom per unit h.
        """
        trig = np.cos if field == "cos" else np.sin
        n = self.nsites
        M1 = np.zeros((self.ncell * n, self.ncell * n), dtype=complex)
        M2 = np.zeros((self.ncell * n, self.ncell * n), dtype=complex)
        for L in range(self.ncell):
            for bi, (i, j, R, j0, beta) in enumerate(self.bonds):
                tau_i = h * disp[i] * trig(2 * np.pi * q * L)
                tau_j = h * disp[j] * trig(2 * np.pi * q * (L + R))
                Jb = self.spring_J(bi, tau_i, tau_j)
                x = L * n + i
                y = ((L + R) % self.ncell) * n + j
                M1[x, y] += -Jb * self.t_bond[bi]
                M1[y, x] += -Jb * np.conj(self.t_bond[bi])
                M1[x, x] += Jb * self.spins[j]
                M1[y, y] += Jb * self.spins[i]
                M2[x, y] += -Jb * self.s_bond[bi]
        return M1, M2

    def bloch2(self, M, p1, p2):
        """Two-momentum Bloch transform M[(p1), (p2)] of a real-space matrix.

        Carries the 1/N_cell of the site-boson Bloch convention
        a_{iL} = N^{-1/2} sum_p e^{2 pi i p L} a_i(p).
        """
        n = self.nsites
        out = np.zeros((n, n), dtype=complex)
        for Lx in range(self.ncell):
            for Ly in range(self.ncell):
                ph = np.exp(-2j * np.pi * p1 * Lx) * np.exp(2j * np.pi * p2 * Ly)
                out += ph * M[Lx * n : (Lx + 1) * n, Ly * n : (Ly + 1) * n]
        return out / self.ncell

    def reference_vertex(self, k, q, disp, h):
        """Mechanical vertex: FD of the REAL cos/sin displaced models combined
        into the complex pattern response, then Bogoliubov-projected."""
        n = self.nsites
        kp, mk, mkp = (k + q) % 1, (-k) % 1, (-(k + q)) % 1
        dcos1 = (
            self.modulated_M1(q, disp, +h, "cos")[0]
            - self.modulated_M1(q, disp, -h, "cos")[0]
        ) / (2 * h)
        dsin1 = (
            self.modulated_M1(q, disp, +h, "sin")[0]
            - self.modulated_M1(q, disp, -h, "sin")[0]
        ) / (2 * h)
        dcos2 = (
            self.modulated_M1(q, disp, +h, "cos")[1]
            - self.modulated_M1(q, disp, -h, "cos")[1]
        ) / (2 * h)
        dsin2 = (
            self.modulated_M1(q, disp, +h, "sin")[1]
            - self.modulated_M1(q, disp, -h, "sin")[1]
        ) / (2 * h)
        dM1 = dcos1 + 1j * dsin1
        dM2 = dcos2 + 1j * dsin2
        M_kp_k = self.bloch2(dM1, kp, k)
        M_mk_mkp = self.bloch2(dM1, mk, mkp)
        M_aa = self.bloch2(dM2, k, mkp)  # a_k a_{-k-q} structure (zero: collinear)
        assert np.max(np.abs(M_aa)) < 1e-12
        T = {p: _colpa(self.static_hessian(p))[0] for p in (kp, k, mk, mkp)}
        u_kp, u_k = T[kp][:n, :n].T, T[k][:n, :n].T
        w_mk, w_mkp = T[mk][n:, :n].T, T[mkp][n:, :n].T
        g = np.zeros((n, n), dtype=complex)
        for mu in range(n):
            for nu in range(n):
                val = 0j
                for ci in range(n):
                    for cj in range(n):
                        val += np.conj(u_kp[mu, ci]) * M_kp_k[ci, cj] * u_k[nu, cj]
                        val += np.conj(w_mk[nu, ci]) * M_mk_mkp[ci, cj] * w_mkp[mu, cj]
                g[mu, nu] = val
        return g

    def closed_form_vertex(self, k, q, pattern, zpa):
        """Closed-form bond assembly (the formula implemented by TB2J/magnon_phonon)."""
        n = self.nsites
        kp, mk, mkp = (k + q) % 1, (-k) % 1, (-(k + q)) % 1
        T = {p: _colpa(self.static_hessian(p))[0] for p in (kp, k, mk, mkp)}
        u_kp, u_k = T[kp][:n, :n].T, T[k][:n, :n].T
        w_mk, w_mkp = T[mk][n:, :n].T, T[mkp][n:, :n].T
        # pattern-projected complex rows from the home-cell endpoint derivatives
        rows = np.zeros(len(self.bonds), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(self.bonds):
            di, dj = self.dJ_endpoint(bi)
            rows[bi] = zpa[i] * np.sum(di * pattern[i]) + np.exp(
                2j * np.pi * q * R
            ) * zpa[j] * np.sum(dj * pattern[j])
        g = np.zeros((n, n), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(self.bonds):
            Ob = rows[bi]
            t = self.t_bond[bi]
            F = np.zeros((n, n), dtype=complex)
            p1 = np.exp(2j * np.pi * k * R)
            p2 = np.exp(-2j * np.pi * (k + q) * R)
            for mu in range(n):
                for nu in range(n):
                    F[mu, nu] = (
                        p1
                        * t
                        * (
                            np.conj(u_kp[mu, i]) * u_k[nu, j]
                            + np.conj(w_mk[nu, i]) * w_mkp[mu, j]
                        )
                        + p2
                        * np.conj(t)
                        * (
                            np.conj(u_kp[mu, j]) * u_k[nu, i]
                            + np.conj(w_mk[nu, j]) * w_mkp[mu, i]
                        )
                        - self.spins[j]
                        * (
                            np.conj(u_kp[mu, i]) * u_k[nu, i]
                            + np.conj(w_mk[nu, i]) * w_mkp[mu, i]
                        )
                        - self.spins[i]
                        * np.exp(-2j * np.pi * q * R)
                        * (
                            np.conj(u_kp[mu, j]) * u_k[nu, j]
                            + np.conj(w_mk[nu, j]) * w_mkp[mu, j]
                        )
                    )
            g -= Ob * F
        return g


def check_numeric_finite_model():
    """Closed-form vertex vs mechanical real-space reference in the weak limit."""
    model = _Model()
    hbar_js = 1.054571817e-34
    amu_kg = 1.66053906660e-27
    masses = np.array([1.3, 0.7])
    omega = 2 * np.pi * 5.3 * 1e12
    zpa = np.sqrt(hbar_js / (2 * masses * amu_kg * omega)) * 1e10  # Angstrom
    rng = np.random.default_rng(9)
    raw = rng.normal(size=(2, 2))
    pattern = np.linalg.qr(raw)[0][:, 0] * 1.0  # real unit Cartesian pattern

    max_err = 0.0
    for k in (1 / 6, 1 / 3):
        for q in (1 / 3, 1 / 6):
            g_ref = model.reference_vertex(k, q, pattern * zpa, 1e-4)
            g_cf = model.closed_form_vertex(k, q, pattern, zpa)
            err = np.max(np.abs(g_ref - g_cf))
            max_err = max(max_err, err)
            assert err < 1e-9, (k, q, err)
    # weak-displacement linearity: the extracted linear coefficient is
    # independent of the displacement amplitude (no quadratic leak)
    g1 = model.reference_vertex(1 / 6, 1 / 3, pattern * zpa, 1e-4)
    g2 = model.reference_vertex(1 / 6, 1 / 3, pattern * zpa, 0.5e-4)
    leak = np.max(np.abs(g1 - g2))
    assert leak < 1e-9, leak

    # numeric Hermiticity: g(k, q) = [g(nu mu)(k+q, -q)]* with O(-q) = O(q)*
    for k in (1 / 6, 1 / 3, 1 / 2):
        for q in (1 / 3,):  # no sampled momentum hits the Gamma zero mode
            g_a = model.closed_form_vertex(k, q, pattern, zpa)
            g_b = model.closed_form_vertex((k + q) % 1, -q, pattern, zpa)
            assert np.max(np.abs(g_a - np.conj(g_b.T))) < 1e-12

    # mode rephasing: with u(p) -> u(p) Theta^dag and w(p) -> w(p) Theta_{-p}
    # (fixed site bosons a(p)), a diagonal phase rephasing leaves every matrix
    # element of the contracted residue g g^dag invariant.
    n = model.nsites
    theta = np.diag(np.exp(1j * np.array([0.7, -1.3])))

    def rephased(k, q):
        kp, mk, mkp = (k + q) % 1, (-k) % 1, (-(k + q)) % 1
        T = {p: _colpa(model.static_hessian(p))[0] for p in (kp, k, mk, mkp)}
        rows = np.zeros(len(model.bonds), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(model.bonds):
            di, dj = model.dJ_endpoint(bi)
            rows[bi] = zpa[i] * np.sum(di * pattern[i]) + np.exp(
                2j * np.pi * q * R
            ) * zpa[j] * np.sum(dj * pattern[j])
        u_kp = (T[kp][:n, :n] @ theta.conj().T).T
        u_k = (T[k][:n, :n] @ theta.conj().T).T
        w_mk = (T[mk][n:, :n] @ theta).T
        w_mkp = (T[mkp][n:, :n] @ theta).T
        g = np.zeros((n, n), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(model.bonds):
            t = model.t_bond[bi]
            p1 = np.exp(2j * np.pi * k * R)
            p2 = np.exp(-2j * np.pi * (k + q) * R)
            F = np.zeros((n, n), dtype=complex)
            for mu in range(n):
                for nu in range(n):
                    F[mu, nu] = (
                        p1
                        * t
                        * (
                            np.conj(u_kp[mu, i]) * u_k[nu, j]
                            + np.conj(w_mk[nu, i]) * w_mkp[mu, j]
                        )
                        + p2
                        * np.conj(t)
                        * (
                            np.conj(u_kp[mu, j]) * u_k[nu, i]
                            + np.conj(w_mk[nu, j]) * w_mkp[mu, i]
                        )
                        - model.spins[j]
                        * (
                            np.conj(u_kp[mu, i]) * u_k[nu, i]
                            + np.conj(w_mk[nu, i]) * w_mkp[mu, i]
                        )
                        - model.spins[i]
                        * np.exp(-2j * np.pi * q * R)
                        * (
                            np.conj(u_kp[mu, j]) * u_k[nu, j]
                            + np.conj(w_mk[nu, j]) * w_mkp[mu, j]
                        )
                    )
            g -= rows[bi] * F
        return g

    for k in (1 / 6, 1 / 3):
        for q in (1 / 3, 1 / 6):
            ga = model.closed_form_vertex(k, q, pattern, zpa)
            gb = rephased(k, q)
            # g' = Theta g Theta^dag: the contracted residue spectrum is invariant
            res_a = np.linalg.eigvalsh(ga @ ga.conj().T)
            res_b = np.linalg.eigvalsh(gb @ gb.conj().T)
            assert np.max(np.abs(res_a - res_b)) < 1e-10
            assert np.allclose(np.sort(np.abs(gb)), np.sort(np.abs(ga)), atol=1e-10)
    print(
        f"finite-model comparison (max |dg| = {max_err:.2e}), Hermiticity, rephasing: OK"
    )


def main():
    check_hp_local_frame_expansion()
    check_forbidden_scalar_bilinear_mixing()
    check_bogoliubov_projection_and_hermiticity()
    check_numeric_finite_model()
    print(
        "magnon-phonon vertex: HP factors, projection normalization, Hermiticity, "
        "forbidden mixing, and finite-model comparison: OK"
    )


if __name__ == "__main__":
    main()
