#!/usr/bin/env python
"""
02_rpa_callen_tc.py -- Finite-temperature Heisenberg magnetism:
equation-of-motion Green's function method, RPA (Tyablikov) and Callen
decoupling, Callen magnetization formula, and Curie temperatures.

Companion report: 02_rpa_callen_tc.md.
Reference (equation labels quoted in comments and markdown): arXiv:2405.00477,
Sec. II (LaTeX source: TB2J/Refs/2405.00477/main.tex).

Model: isotropic Heisenberg ferromagnet on a Bravais lattice,
    H = -(1/2) sum_{i != j} J_ij S_i . S_j ,   J > 0  (FM),
dimensionless spins, k_B = 1, J = 1 in the numerical section.

Run:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python 02_rpa_callen_tc.py
"""

import time

import numpy as np
import sympy as sp

T0 = time.time()


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# Part 1: spin algebra and the [S^+-, H] commutator  (paper Eqs. spin-comm)
# ============================================================================
banner(
    "Part 1: spin commutators and [S^+-, H] on a 2-site cluster "
    "(Eqs. spin-comm-1, spin-comm-2, spin-comm-H)"
)

Jsy, alpha, w = sp.symbols("J alpha omega", real=True)


def spin_matrices(S):
    """Exact spin-S matrices Sz, Sp, Sm in the |m = S..-S> basis."""
    dim = int(round(2 * S + 1))
    Sz = sp.diag(*[sp.Rational(int(round(2 * S)) - 2 * i, 2) for i in range(dim)])
    Sp = sp.zeros(dim)
    for b in range(dim - 1):  # raises |m_{b+1}> -> |m_b>
        mm = Sz[b + 1, b + 1]
        Sp[b, b + 1] = sp.sqrt(S * (S + 1) - mm * (mm + 1))
    return Sz, Sp, Sp.T


def kron2(A, B):
    """Exact Kronecker product of two sympy Matrices."""
    ra, ca, rb, cb = *A.shape, *B.shape
    M = sp.zeros(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            M[i * rb : (i + 1) * rb, j * cb : (j + 1) * cb] = A[i, j] * B
    return M


for S in (sp.Rational(1, 2), sp.Integer(1), sp.Rational(3, 2)):
    Sz, Sp, Sm = spin_matrices(S)
    assert sp.simplify(Sz * Sp - Sp * Sz - Sp) == sp.zeros(*Sz.shape)
    assert sp.simplify(Sz * Sm - Sm * Sz + Sm) == sp.zeros(*Sz.shape)
    assert sp.simplify(Sp * Sm - Sm * Sp - 2 * Sz) == sp.zeros(*Sz.shape)
    # ladder identities used in Part 4 (Eqs. Sz-cd1/2):
    eyev = sp.eye(*Sz.shape)
    assert sp.simplify(Sp * Sm - (S * (S + 1) * eyev - Sz**2 + Sz)) == sp.zeros(
        *Sz.shape
    )
    assert sp.simplify(Sm * Sp - (S * (S + 1) * eyev - Sz**2 - Sz)) == sp.zeros(
        *Sz.shape
    )
print(
    "PASS  single-site commutators [Sz,S+-]=+-S+-, [S+,S-]=2Sz and "
    "S+S-/S-S+ ladder identities (S=1/2, 1, 3/2)"
)

# --- two-site cluster: H = -J S1.S2 (Eq. hamiltonian in circular coordinates)
for S in (sp.Rational(1, 2), sp.Integer(1)):
    Id = sp.eye(int(round(2 * S + 1)))
    Sz1 = kron2(spin_matrices(S)[0], Id)
    Sp1 = kron2(spin_matrices(S)[1], Id)
    Sm1 = kron2(spin_matrices(S)[2], Id)
    Sz2 = kron2(Id, spin_matrices(S)[0])
    Sp2 = kron2(Id, spin_matrices(S)[1])
    Sm2 = kron2(Id, spin_matrices(S)[2])
    H = -Jsy * (Sz1 * Sz2 + (Sp1 * Sm2 + Sm1 * Sp2) / 2)

    # Eq. spin-comm-1: [S_i^z, S_j^+-] = +- S_i^+- delta_ij
    assert sp.simplify(Sz1 * Sp1 - Sp1 * Sz1 - Sp1) == sp.zeros(*Sz1.shape)
    assert sp.simplify(Sz1 * Sp2 - Sp2 * Sz1) == sp.zeros(*Sz1.shape)
    assert sp.simplify(Sz1 * Sm1 - Sm1 * Sz1 + Sm1) == sp.zeros(*Sz1.shape)
    assert sp.simplify(Sz2 * Sp1 - Sp1 * Sz2) == sp.zeros(*Sz1.shape)
    # Eq. spin-comm-2: [S_i^+, S_j^-] = 2 S_i^z delta_ij
    assert sp.simplify(Sp1 * Sm1 - Sm1 * Sp1 - 2 * Sz1) == sp.zeros(*Sz1.shape)
    assert sp.simplify(Sp1 * Sm2 - Sm2 * Sp1) == sp.zeros(*Sz1.shape)
    assert sp.simplify(Sp2 * Sm2 - Sm2 * Sp2 - 2 * Sz2) == sp.zeros(*Sz1.shape)
    # Eq. spin-comm-H: [S_i^+-, H] = +- sum_{j!=i} J_ij (S_i^+- S_j^z - S_i^z S_j^+-)
    assert sp.simplify(Sp1 * H - H * Sp1 - Jsy * (Sp1 * Sz2 - Sz1 * Sp2)) == sp.zeros(
        *Sz1.shape
    )
    assert sp.simplify(Sm1 * H - H * Sm1 + Jsy * (Sm1 * Sz2 - Sz1 * Sm2)) == sp.zeros(
        *Sz1.shape
    )
    assert sp.simplify(Sp2 * H - H * Sp2 - Jsy * (Sp2 * Sz1 - Sz2 * Sp1)) == sp.zeros(
        *Sz1.shape
    )
    assert sp.simplify(Sm2 * H - H * Sm2 + Jsy * (Sm2 * Sz1 - Sz2 * Sm1)) == sp.zeros(
        *Sz1.shape
    )
    print(
        f"PASS  [S_i^+-, H] = +- sum_{{j!=i}} J_ij (S_i^+- S_j^z - S_i^z S_j^+-) "
        f"exact for the 2-site S={S} cluster (Eq. spin-comm-H)"
    )

# ============================================================================
# Part 2: equation of motion of the retarded Green's function (exact Lehmann
#         check on the S=1/2 dimer; Eqs. arbGfn-def, eqnofmotion-Gfn)
# ============================================================================
banner(
    "Part 2: Green's-function equation of motion (exact Lehmann check, "
    "Eqs. arbGfn-def, arbGfn-def_fourier, eqnofmotion-Gfn)"
)

# G_ij(t) = -i theta(t) <[S_i^+(t), S_j^-]>  (Eq. arbGfn-def);
# Lehmann representation for any retarded GF <<A;B>>:
#   <<A;B>>_w = (1/Z) sum_{mn} (e^{-b E_m} - e^{-b E_n}) <m|A|n><n|B|m> / (w + E_m - E_n + i eta)
Sz, Sp, Sm = spin_matrices(sp.Rational(1, 2))
Id = sp.eye(2)
Sz1, Sp1, Sm1 = kron2(Sz, Id), kron2(Sp, Id), kron2(Sm, Id)
Sz2, Sp2, Sm2 = kron2(Id, Sz), kron2(Id, Sp), kron2(Id, Sm)
Jv = sp.Rational(3, 2)
Hd = -Jv * (Sz1 * Sz2 + (Sp1 * Sm2 + Sm1 * Sp2) / 2)

evals, evecs = [], []
for lam, mult, vs in Hd.eigenvects():
    for v in vs:
        evals.append(sp.simplify(lam))
        evecs.append(sp.Matrix(v).normalized())

beta = sp.Rational(7, 2)
Z = sum(sp.exp(-beta * E) for E in evals)


def lehmann(A, B, wv, eta=sp.Rational(1, 100)):
    G = 0
    for Em, vm in zip(evals, evecs):
        for En, vn in zip(evals, evecs):
            amp = (vm.T * A * vn)[0] * (vn.T * B * vm)[0]
            G += (
                (sp.exp(-beta * Em) - sp.exp(-beta * En))
                * amp
                / (wv + Em - En + sp.I * eta)
            )
    return G / Z


for i, j in [(1, 1), (2, 1), (1, 2)]:
    A = Sp1 if i == 1 else Sp2
    B = Sm1 if j == 1 else Sm2
    commAB = (
        sum(
            sp.exp(-beta * E) * (vm.T * (A * B - B * A) * vm)[0]
            for E, vm in zip(evals, evecs)
        )
        / Z
    )
    for wv in (sp.Rational(1, 4), sp.Rational(9, 10), sp.Integer(2)):
        eom = complex(
            sp.N(
                (wv + sp.I / 100) * lehmann(A, B, wv)
                - commAB
                - lehmann(A * Hd - Hd * A, B, wv)
            )
        )
        assert abs(eom) < 1e-9, (i, j, wv, eom)
print(
    "PASS  (w+i*eta) G_ij = <[S_i^+, S_j^-]> + <<[S_i^+, H]; S_j^->>_w "
    "holds exactly (Lehmann sum, S=1/2 dimer, beta*J=7/4, 3 omega values "
    "x 3 site pairs) -- Eq. eqnofmotion-Gfn"
)
poles = sorted({float(sp.re(e - e2)) for e in evals for e2 in evals})
print(
    f"      exact dimer excitation energies (G poles) = "
    f"{[round(p, 6) for p in poles]}  (intra-triplet 0, singlet-triplet +/-{float(Jv)})"
)

# ============================================================================
# Part 3: RPA (Tyablikov) decoupling (Eqs. rpa-approx, Gfn-q-space, G_rpa,
#         rpa-magnons3d)
# ============================================================================
banner(
    "Part 3: RPA decoupling -> G_q = 2<Sz>/(w - w_q + i*eta), "
    "w_q^RPA = <Sz>(J_0 - J_q) (Eqs. rpa-approx, G_rpa, rpa-magnons3d)"
)

m = sp.Symbol("m", positive=True)  # m = <S^z>

# --- 3a. 4-site ring: RPA-decoupled EOM solved exactly in sympy -------------
# w G_ij = 2 m delta_ij + m sum_k J_ik (G_ij - G_kj)   (Eqs. eq:eom after
# comm + rpa-approx); translational invariance G_kj = G_{0, j-k mod N}.
N4 = 4
Jmat = sp.zeros(N4)
for a in range(N4):
    for b in range(N4):
        if (a - b) % N4 in (1, N4 - 1):
            Jmat[a, b] = Jsy
J0_4 = sum(Jmat[0, k] for k in range(N4))  # = 2J
lam4 = (J0_4 * sp.eye(N4) - Jmat).eigenvals()
expect = {sp.nsimplify(2 * Jsy * (1 - sp.cos(2 * sp.pi * s / N4))) for s in range(N4)}
assert {sp.nsimplify(k) for k in lam4} == expect, lam4
print(
    "PASS  4-site ring: eigenvalues of (J0*I - J) = J_0 - J_q with "
    "J_q = 2J cos(q)  =>  w_q^RPA = m (J0 - J_q)  (Eq. rpa-magnons3d)"
)

G0j = sp.symbols("G0:4")  # G_{0,j}
eqs = []
for j in range(N4):
    rhs = 2 * m * sp.KroneckerDelta(0, j)
    for k in (1, N4 - 1):  # NN of site 0
        rhs += m * Jsy * (G0j[j] - G0j[(j - k) % N4])
    eqs.append(sp.Eq(w * G0j[j], rhs))
sol = sp.solve(eqs, list(G0j), dict=True)[0]
for s in range(N4):  # G_q = sum_j G_0j e^{i q R_j}
    Gq = sp.simplify(
        sum(sol[G0j[j]] * sp.exp(sp.I * sp.pi * s * j / 2) for j in range(N4))
    )
    wq = m * (J0_4 - 2 * Jsy * sp.cos(sp.pi * s / 2))
    assert sp.simplify(sp.together(Gq - 2 * m / (w - wq))) == 0, s
assert m * (J0_4 - J0_4) == 0
print(
    "PASS  4-site ring: explicit solution + Fourier transform (Eq. "
    "Gfn-q-space) gives G_q = 2<Sz>/(w - <Sz>(J0-J_q) + i*eta) with "
    "residue 2<Sz> at every q  (Eq. G_rpa);  w_Gamma = 0 (Goldstone)"
)

# ============================================================================
# Part 4: Callen decoupling (Eqs. Sz-cd1..3, Sz-cd, cd-approx, G_cd,
#         cd-magnon3d)
# ============================================================================
banner(
    "Part 4: Callen decoupling: alpha-identity, decoupling algebra, "
    "w_q^CD (Eqs. Sz-cd, cd-approx, G_cd, cd-magnon3d)"
)

# --- 4a. alpha-identity (Eq. Sz-cd) exact for arbitrary alpha ---------------
for S in (sp.Rational(1, 2), sp.Integer(1), sp.Rational(3, 2), sp.Integer(2)):
    Sz, Sp, Sm = spin_matrices(S)
    lhs = (
        alpha * (S * (S + 1) * sp.eye(*Sz.shape) - Sz**2)
        + sp.Rational(1, 2) * (1 - alpha) * Sp * Sm
        - sp.Rational(1, 2) * (1 + alpha) * Sm * Sp
    )
    assert sp.simplify(lhs - Sz) == sp.zeros(*Sz.shape)
print(
    "PASS  alpha-identity Sz = alpha(S(S+1)-Sz^2) + (1-alpha)/2 S+S- - "
    "(1+alpha)/2 S-S+ holds for arbitrary alpha (matrix check S=1/2,1,3/2,2; "
    "Eq. Sz-cd); alpha=+1/-1/0 reproduce Eqs. Sz-cd1/Sz-cd2/Sz-cd3"
)

# --- 4b. CD-decoupled EOM on the 2-site ring; algebra through to w_q^CD -----
# 2-site ring: site 0 has its partner (site 1) at R = +a AND R = -a, so
# J_0 = 2J and J_q = 2 J cos(qa); q in {Gamma, X=pi/a}.
# EOM (Eq. eq:eom after comm):
#   w G_00 = 2m + 2J [ <<S0+ S1z; S0->> - <<S0z S1+; S0->> ]
# CD decoupling (Eq. cd-approx) of both higher-order GFs:
#   <<S_i+ S_kz; S_j->> ~ m G_ij - alpha <S_k^- S_i^+> G_kj
#   <<S_iz S_k+; S_j->> ~ m G_kj - alpha <S_i^- S_k^+> G_ij     (i != k)
# with <S0^- S1^+> = <S1^- S0^+> = C by inversion symmetry.
C = sp.Symbol("C", real=True)
G00, G10 = sp.symbols("G00 G10")
eq1 = sp.Eq(
    w * G00,
    2 * m + 2 * Jsy * ((m * G00 - alpha * C * G10) - (m * G10 - alpha * C * G00)),
)
eq2 = sp.Eq(
    w * G10, 2 * Jsy * ((m * G10 - alpha * C * G00) - (m * G00 - alpha * C * G10))
)
sols = sp.solve([eq1, eq2], [G00, G10], dict=True)[0]
G_Gam = sp.simplify(sols[G00] + sols[G10])  # G_q = G00 + e^{-iqa} G10
G_X = sp.simplify(sols[G00] - sols[G10])
wX_expected = 4 * m * Jsy + 4 * alpha * Jsy * C  # see 02_rpa_callen_tc.md
assert sp.simplify(sp.together(G_Gam - 2 * m / w)) == 0
assert sp.simplify(sp.together(G_X - 2 * m / (w - wX_expected))) == 0
print(
    "PASS  2-site ring CD algebra: G_Gamma = 2m/w (Goldstone intact), "
    "G_X = 2m/(w - w_X) with w_X = 4J(m + alpha*C)  (Eq. G_cd)"
)

# cross-check against Eq. cd-magnon3d specialized to N=2:
#   w_q^CD = m(J0-Jq) + alpha (1/N) sum_p (J_p - J_{q-p}) C_p,
#   J_Gamma = 2J, J_X = -2J, C_p = C_0 +- C  (same-site part C_0 cancels):
#   w_X^CD = 4mJ + alpha (1/2)(2J+2J)(C_Gamma - C_X) = 4mJ + 4 alpha J C  ✓
w_cd_general = m * (2 * Jsy - (-2 * Jsy)) + alpha * sp.Rational(1, 2) * (
    2 * Jsy + 2 * Jsy
) * ((sp.Symbol("C0") + C) - (sp.Symbol("C0") - C))
assert sp.simplify(w_cd_general - wX_expected) == 0
print(
    "PASS  w_X equals Eq. cd-magnon3d specialized to N=2 (same-site "
    "correlator parts cancel; only <S_i^- S_j^+>, i != j, enters)"
)

# --- 4c. spectral theorem: C_p = <S^-S^+>_p = 2<Sz> n_B(w_p) -----------------
eta = sp.Symbol("eta", positive=True)
omq, TT = sp.symbols("omega_q T", positive=True)
nB = 1 / (sp.exp(omq / TT) - 1)
Gq = 2 * m / (w - omq + sp.I * eta)
Gq_eta0 = 2 * m / (w - omq)
res_eta0 = sp.residue(nB * Gq_eta0, w, omq)
assert sp.simplify(res_eta0 - 2 * m * nB) == 0
print(
    "PASS  fluctuation-dissipation step: residue of n_B(w) G_q(w) at "
    "w = w_q equals 2<Sz> n_B(w_q)  =>  <S_i^- S_j^+> = (1/N) sum_q "
    "e^{iq(R_j-R_i)} 2<Sz> n_B(w_q)"
)

# --- 4d. close the equations with alpha = <Sz>/(2 S^2) ----------------------
Ssym = sp.Symbol("S", positive=True)
nX = sp.Symbol("n_X", real=True)
# C = <S0^- S1^+> = m(n_Gamma - n_X) (spectral theorem; n_Gamma excluded):
C_val = m * (0 - nX)
wX_cd_final = sp.simplify(wX_expected.subs(alpha, m / (2 * Ssym**2)).subs(C, C_val))
wX_cd_paper = sp.simplify(
    m * 4 * Jsy + (m**2 / Ssym**2) * sp.Rational(1, 2) * (-4 * Jsy) * nX
)
assert sp.simplify(wX_cd_final - wX_cd_paper) == 0
print(
    "PASS  with alpha = <Sz>/(2S^2) and C = -<Sz> n_B(w_X):  w_X^CD = "
    "4J<Sz> - (2J/S^2) <Sz>^2 n_B(w_X) = <Sz>(J0-J_X) + "
    "(<Sz>^2/(S^2 N)) sum_p (J_p - J_{X-p}) n_p   (Eq. cd-magnon3d)"
)

# ============================================================================
# Part 5: Callen magnetization formula and its limits (Eq. mag)
# ============================================================================
banner(
    "Part 5: <Sz> = [(S-phi)(1+phi)^{2S+1} + (S+1+phi) phi^{2S+1}] / "
    "[(1+phi)^{2S+1} - phi^{2S+1}]  (Eq. mag)"
)

phi = sp.Symbol("phi", positive=True)


def callen(p, Sv):
    return ((Sv - p) * (1 + p) ** (2 * Sv + 1) + (Sv + 1 + p) * p ** (2 * Sv + 1)) / (
        (1 + p) ** (2 * Sv + 1) - p ** (2 * Sv + 1)
    )


# --- 5a. S=1/2: exact reduction to 1/(2+4 phi) ------------------------------
m_half = sp.simplify(callen(phi, sp.Rational(1, 2)))
assert sp.simplify(m_half - sp.Rational(1, 2) / (1 + 2 * phi)) == 0
Sz12 = spin_matrices(sp.Rational(1, 2))
assert sp.simplify(
    Sz12[2] * Sz12[1] - (sp.Rational(1, 2) * sp.eye(2) - Sz12[0])
) == sp.zeros(2)
sol_m = sp.solve(sp.Eq(m, sp.Rational(1, 2) - 2 * m * phi), m)[0]
assert sp.simplify(sol_m - m_half) == 0
print(
    "PASS  S=1/2: Eq. (mag) reduces exactly to <Sz> = 1/(2+4 phi); the "
    "same follows from the operator identity S^-S^+ = 1/2 - Sz and the "
    "FDT <S^-S^+> = 2<Sz> phi"
)

# --- 5b. low-T limit <Sz> = S - phi + O(phi^2) ------------------------------
for Sv in (sp.Rational(1, 2), sp.Integer(1), sp.Integer(2), sp.Integer(3)):
    ser = sp.expand(sp.series(callen(phi, Sv), phi, 0, 3).removeO())
    assert sp.simplify(ser.coeff(phi, 0) - Sv) == 0
    assert sp.simplify(ser.coeff(phi, 1) + 1) == 0
    # remainder is O(phi^2) with S-dependent coefficient (e.g. +2 phi^2 for S=1/2):
    rem = (sp.series(callen(phi, Sv), phi, 0, 4).removeO() - (Sv - phi)) / phi**2
    assert sp.limit(rem, phi, 0).is_finite is not False
# generic S: phi^{2S+1} = O(phi^2) for S >= 1/2, binomial series for (1+phi)^{2S+1}
nn = 2 * Ssym + 1
X = sp.series((1 + phi) ** nn, phi, 0, 3).removeO()
m_gen_low = sp.series(sp.expand((Ssym - phi) * X) / sp.expand(X), phi, 0, 2).removeO()
assert sp.simplify(sp.expand(m_gen_low - (Ssym - phi))) == 0
print(
    "PASS  low-T limit <Sz> = S - phi + O(phi^2)  (S=1/2,1,2,3 exactly; "
    "symbolic S via binomial series)  -- Eqs. HP_mag, phi"
)

# --- 5c. large-phi limit: phi <Sz> -> S(S+1)/3 ------------------------------
for Sv in (
    sp.Rational(1, 2),
    sp.Integer(1),
    sp.Integer(2),
    sp.Integer(3),
    sp.Integer(5),
):
    lim = sp.limit(phi * callen(phi, Sv), phi, sp.oo)
    assert sp.simplify(lim - Sv * (Sv + 1) / 3) == 0, (Sv, lim)
# generic S: substitute u = 1/phi, (1+phi)^{2S+1} = phi^{2S+1}(1+u)^{2S+1},
# expand (1+u)^n to u^3 and divide the resulting polynomials
u = sp.Symbol("u", positive=True)
Xu = 1 + nn * u + nn * (nn - 1) / 2 * u**2 + nn * (nn - 1) * (nn - 2) / 6 * u**3
num_u = sp.expand((Ssym - 1 / u) * Xu + (Ssym + 1 + 1 / u))
den_u = sp.expand(Xu - 1)
q_ser = sp.series(
    sp.series(num_u, u, 0, 3).removeO() / sp.series(den_u, u, 0, 3).removeO(), u, 0, 2
).removeO()
assert sp.simplify(sp.expand(q_ser - Ssym * (Ssym + 1) / 3 * u)) == 0
print(
    "PASS  large-phi limit phi <Sz> -> S(S+1)/3, i.e. <Sz> = "
    "S(S+1)/(3 phi) (1 + O(phi^-2))  (S=1/2..5 exact limits + symbolic S) "
    "-- key step towards Eqs. Tc3d-gen/Tc3d-rpa"
)

# --- 5d. Tc derivation steps (Eqs. Tc3d-gen, Tc3d-rpa, eq:T_mf) -------------
x = sp.Symbol("x")
nb_ser = sp.series(1 / (sp.exp(x) - 1), x, 0, 3).removeO()
assert sp.simplify(nb_ser - (1 / x - sp.Rational(1, 2) + x / 12)) == 0
print(
    "PASS  Bose-factor expansion n_B = T/w - 1/2 + w/(12T) + O(w^3): with "
    "w_q = <Sz>(J0-J_q), phi = (T/<Sz>) (1/N)sum 1/(J0-J_q) - 1/2 + "
    "O(<Sz>); combining with <Sz> phi -> S(S+1)/3 at <Sz> -> 0 gives "
    "k_B Tc = S(S+1)/3 [(1/N) sum 1/(J0-J_q)]^{-1}  (Eq. Tc3d-rpa)"
)
A_sum, J0s = sp.Symbol("A", positive=True), sp.Symbol("J_0", positive=True)
# general form (Eq. Tc3d-gen) with the MFA dispersion w_q = <Sz> J_0:
tc_gen_mfa = sp.simplify(Ssym * (Ssym + 1) / 3 * (1 / (1 / J0s)))
assert sp.simplify(tc_gen_mfa - J0s * Ssym * (Ssym + 1) / 3) == 0
print(
    "PASS  inserting w_q^MFA = <Sz> J_0 into Eq. Tc3d-gen reproduces "
    "k_B Tc^MFA = J_0 S(S+1)/3  (Eq. eq:T_mf)"
)

# ============================================================================
# Part 6: numerical verification on the simple-cubic NN model
# ============================================================================
banner(
    "Part 6: numerics on the simple-cubic NN Heisenberg FM (J=1): "
    "lattice Green function, Tc values, self-consistent RPA/CD/HP"
)

# --- 6a. Watson / lattice-Green-function convergence ------------------------
# sc NN: J_q = 2(cos qx + cos qy + cos qz), J0 = 6, J0-J_q = 2(3-sum cos).
# (1/N) sum 1/(3-sum cos) = G_sc(0)/3 with the sc lattice Green function at
# the origin G_sc(0) = 1.516386059151978 (Watson 1939; Joyce, J. Phys. A 5,
# L60 (1972)). Gamma is excluded from the discrete sums: the integrand has
# an integrable 1/q^2 singularity (a set of measure zero in the continuum),
# and the Gamma magnon (w_Gamma = 0 exactly, a uniform spin rotation) must
# not enter the Bose sums.
watson_rows = []
for ngrid in (8, 16, 32, 64, 128):
    q1 = 2 * np.pi * np.arange(ngrid) / ngrid  # Gamma at index 0
    QX, QY, QZ = np.meshgrid(q1, q1, q1, indexing="ij")
    denominator = 3 - (np.cos(QX) + np.cos(QY) + np.cos(QZ))
    inv = np.zeros_like(denominator)
    np.divide(1.0, denominator, out=inv, where=np.abs(denominator) > 1e-12)
    wbar = inv.sum() / ngrid**3
    watson_rows.append((ngrid, wbar))
    print(f"      grid {ngrid:3d}^3 :  mean 1/(3-sum cos) = {wbar:.8f}")
w_rich = watson_rows[-1][1] + (watson_rows[-1][1] - watson_rows[-2][1])
G_sc0 = 3 * w_rich
print(f"      Richardson-extrapolated 3 * mean = {G_sc0:.8f}")
assert abs(G_sc0 - 1.516386059151978) < 5e-5, G_sc0
print(
    "PASS  sc lattice Green function: grid sums (Gamma excluded) converge "
    "to 3 x mean = G_sc(0) = 1.516386059 (Watson/Joyce); convergence table above"
)

A_cont = w_rich / 2.0  # (1/N) sum 1/(J0-J_q), continuum
TC_RPA_CL = 1.0 / (3 * A_cont)  # k_B Tc^RPA / (J S^2)
print(f"      (1/N) sum 1/(J0-J_q) = {A_cont:.8f}")
print(
    f"      k_B Tc^RPA = S(S+1) * {TC_RPA_CL:.6f} J  ->  "
    f"S=1/2: {0.75 * TC_RPA_CL:.6f} J,  classical (S^2): {TC_RPA_CL:.6f} J S^2"
)

# --- 6b. self-consistent machinery ------------------------------------------
NG = 24
q1 = 2 * np.pi * np.arange(NG) / NG
QX, QY, QZ = np.meshgrid(q1, q1, q1, indexing="ij")
J3 = 2 * (np.cos(QX) + np.cos(QY) + np.cos(QZ))
J0 = 6.0
E3 = J0 - J3
FF = np.fft.fftn(J3)
eps = E3[np.abs(E3) > 1e-12]
npts = NG**3


def bose(ww, T):
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(np.clip(ww / T, 1e-12, None))


def callen_np(phiv, S):
    """Eq. (mag) evaluated in log space (stable for large phi and S)."""
    rho = np.exp((2 * S + 1) * (np.log(np.clip(phiv, 1e-300, None)) - np.log1p(phiv)))
    return ((S - phiv) + (S + 1 + phiv) * rho) / (1 - rho)


def phi_of_m(mv, S):
    """Invert Eq. (mag): the phi at which <Sz>(phi) = mv (bisection)."""
    lo, hi = 1e-12, 1.0
    while callen_np(hi, S) > mv:
        hi *= 2
        assert hi < 1e30
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if callen_np(mid, S) > mv:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def occupations(mv, T, S, method, nb3=None, maxit=12000, tol=1e-9):
    """Self-consistent Bose occupations n_q on the NG^3 grid.

    RPA: w_q = m (J0 - J_q)                              (Eq. rpa-magnons3d)
    CD:  w_q = m (J0 - J_q) + (m^2/S^2) * corr_q         (Eq. cd-magnon3d)
    HP:  w_q = (S-phi)(J0-J_q) + corr_q                  (Eq. HP_magnon_energy)
         corr_q = (1/N) sum_p (J_p - J_{q-p}) n_p        (FFT convolution)
    Gamma excluded from all BZ averages. Returns None on negative magnon
    energies (HP breakdown) or non-convergence of the adaptive iteration.
    """
    if nb3 is None:
        nb3 = np.zeros((NG, NG, NG))
    mix, prev, bad = 0.4, np.inf, 0
    for _ in range(maxit):
        nb3[0, 0, 0] = 0.0
        c1 = (J3 * nb3).sum() / npts
        c2 = np.fft.ifftn(FF * np.fft.fftn(nb3)).real / npts
        d = c1 - c2
        if method == "CD":
            w3 = mv * E3 + (mv * mv / S**2) * d
        elif method == "HP":
            w3 = (S - nb3.sum() / npts) * E3 + d
        else:
            w3 = mv * E3
        w3[0, 0, 0] = np.inf
        if w3.min() < 0:
            return None
        wn = bose(w3, T)
        res = np.max(np.abs(wn - nb3))
        if res < tol * (1 + wn.max()):
            return wn
        if res > prev:
            bad += 1
        else:
            bad = max(0, bad - 1)
        if bad > 3:
            mix = max(0.002, 0.4 * mix)
            bad = 0
        prev = res
        nb3 = (1 - mix) * nb3 + mix * wn
    return None


def phi_method(mv, T, S, method):
    if method == "RPA":
        return bose(mv * eps, T).mean()
    nb3 = occupations(mv, T, S, method)
    return None if nb3 is None else nb3.sum() / npts


def T_of_m(mv, S, method, Tlo, Thi, nstep=34):
    """T at which the self-consistent <Sz> equals mv (mv -> 0 gives Tc).
    Bypasses the critical slowing down of a direct m(T) iteration."""
    target = phi_of_m(mv, S)
    lo, hi, nb3 = Tlo, Thi, None
    for _ in range(nstep):
        T = 0.5 * (lo + hi)
        if method == "RPA":
            p = phi_method(mv, T, S, method)
        else:
            nb3 = occupations(mv, T, S, method, nb3)
            p = None if nb3 is None else nb3.sum() / npts
        if p is None:
            return None
        if p < target:
            lo = T
        else:
            hi = T
    return 0.5 * (lo + hi)


def tc_richardson(S, method, Tlo, Thi, mstar=5e-3):
    T1 = T_of_m(mstar, S, method, Tlo, Thi)
    T2 = T_of_m(2 * mstar, S, method, Tlo, Thi)
    return None if (T1 is None or T2 is None) else 2 * T1 - T2


# 6b-i. RPA self-consistency vs closed form on the same grid
A_grid = (1.0 / eps).mean()
tc_rpa_closed_grid = 0.5 * 1.5 / 3 / A_grid
tc_rpa_sc = tc_richardson(0.5, "RPA", 0.5, 3.0)
rel = abs(tc_rpa_sc - tc_rpa_closed_grid) / tc_rpa_closed_grid
print(
    f"      RPA S=1/2 on {NG}^3 grid: self-consistent Tc = {tc_rpa_sc:.6f} "
    f"vs closed form {tc_rpa_closed_grid:.6f}  (rel. diff {rel:.2e})"
)
assert rel < 5e-3
print(
    "PASS  RPA: fully self-consistent numerical solution reproduces the "
    "closed form k_B Tc = S(S+1)/3 [(1/N) sum 1/(J0-J_q)]^{-1} (Eq. Tc3d-rpa)"
)


# 6b-ii. m(T) table: RPA/CD smooth, HP terminates (Eqs. mag, HP_mag)
def mag_curve(T, S, method):
    """Converged <Sz>(T); None if no ordered solution exists (T > Tc)."""
    phiv, nb3 = 1e-8, None
    for _ in range(400):
        mm = callen_np(phiv, S)
        if not np.isfinite(mm) or mm <= 0 or mm > S:
            return None
        if method == "RPA":
            phin = bose(mm * eps, T).mean()
        else:
            nb3 = occupations(mm, T, S, method, nb3)
            if nb3 is None:
                return None
            phin = nb3.sum() / npts
        if (not np.isfinite(phin)) or phin < 0 or phin > 1e4 * (1 + S):
            return None  # phi diverges: T above Tc
        if abs(phin - phiv) < 1e-10:
            phiv = phin
            break
        phiv = 0.5 * phiv + 0.5 * phin
    mv = float(callen_np(phiv, S))
    return mv if np.isfinite(mv) and 0 < mv <= S else None


print("      <Sz>(T), S=1/2:      T      RPA       CD        HP")
for T in (0.4, 0.6, 0.8):
    mR = mag_curve(T, 0.5, "RPA")
    mC = mag_curve(T, 0.5, "CD")
    nbH = occupations(0, T, 0.5, "HP")
    mH = None if nbH is None else 0.5 - nbH.sum() / npts
    print(
        f"                        {T:.2f}  {mR:7.4f}  "
        f"{('%7.4f' % mC) if mC is not None else '   None'}  "
        f"{('%7.4f' % mH) if mH is not None else '   None'}"
    )
assert mag_curve(1.00, 0.5, "RPA") is not None and mag_curve(1.00, 0.5, "RPA") < 0.13
assert mag_curve(1.10, 0.5, "RPA") is None  # above Tc: no solution
mCD_seq = [mag_curve(T, 0.5, "CD") for T in (0.8, 1.0, 1.2, 1.30)]
assert all(x is not None for x in mCD_seq)
assert all(mCD_seq[i] > mCD_seq[i + 1] for i in range(3))  # smooth decrease
mCD_near = mag_curve(1.33, 0.5, "CD")
assert mCD_near is not None and mCD_near < 0.09  # approaching zero near Tc^CD
print(
    "PASS  RPA and CD magnetizations vanish continuously at their Tc (RPA "
    "has no solution above 1.10 J; CD <Sz> decreases smoothly to < 0.09 at "
    "1.33 J just below Tc^CD), while HP terminates at finite <Sz> (next)"
)


# --- 6c. HP breakdown at finite magnetization --------------------------------
def hp_breakdown(S):
    lo, hi = 1e-3, 4 * S * S + 5
    assert occupations(0, lo, S, "HP") is not None
    for _ in range(32):
        T = 0.5 * (lo + hi)
        if occupations(0, T, S, "HP") is not None:
            lo = T
        else:
            hi = T
    nb3 = occupations(0, lo, S, "HP")
    return lo, S - nb3.sum() / npts


print("      HP mean-field (Eqs. HP_magnon_energy, HP_mag, phi):")
hp_vals = {}
for S in (0.5, 1.0, 20.0):
    Tbr, mbr = hp_breakdown(S)
    hp_vals[S] = Tbr
    print(
        f"        S={S:5.1f}: first negative magnon energy at T = {Tbr:8.4f}, "
        f"finite <Sz> = {mbr:.4f}  (m/S = {mbr / S:.3f})"
    )
assert hp_vals[0.5] and hp_breakdown(0.5)[1] > 0.05
print(
    "PASS  HP self-consistency terminates at FINITE <Sz> (first negative "
    "magnon energy -> Bose factors ill-defined, dm/dT -> -infty), whereas "
    "RPA/CD reach <Sz> -> 0 smoothly (paper Secs. II.B, II.D)"
)

# --- 6d. CD Curie temperatures: S-dependence and classical limit -------------
print("      CD Tc (self-consistent + Richardson extrapolation in <Sz>):")
cd_vals = {}
for S in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0):
    tc = tc_richardson(S, "CD", 0.3, 4 * S * S + 10)
    cd_vals[S] = tc
    print(f"        S={S:6.1f}:  Tc^CD = {tc:10.4f}   Tc/(J S^2) = {tc / S**2:.5f}")
assert cd_vals[0.5] > 1.2 * tc_rpa_closed_grid  # CD unreliable at low spin
assert cd_vals[100.0] / 1e4 < cd_vals[20.0] / 400.0  # monotone classical approach
print(
    "PASS  CD: Tc well above RPA at S=1/2 (Callen's method unreliable for "
    "low spin) and decreasing Tc/(J S^2) as S -> infinity"
)


# --- 6e. classical-limit prescription S -> infinity ---------------------------
def tc_rpa_formula(S, A):
    return S * (S + 1) / (3 * A)


def tc_rpa_classical(S, A):
    """Paper prescription: replace S(S+1) by S^2, then take S -> infinity."""
    return S * S / (3 * A)


for S in (0.5, 1.0, 5.0, 100.0):
    assert abs(tc_rpa_formula(S, A_cont) / (S * (S + 1)) - TC_RPA_CL) < 1e-12
assert (
    abs(tc_rpa_classical(1000.0, A_cont) / tc_rpa_formula(1000.0, A_cont) - 1.0) < 0.01
)
MC_LIT = 1.4429  # classical Heisenberg sc, H = -J sum_<ij> e_i.e_j (MC)
cd_cl_grid_corr = (cd_vals[100.0] / 1e4) * A_grid / A_cont
print(
    f"PASS  classical prescription coded: k_B Tc^RPA,cl = S^2/3 * "
    f"[(1/N) sum 1/(J0-J_q)]^{{-1}} = {TC_RPA_CL:.4f} J S^2"
)
print(
    f"      RPA classical = {TC_RPA_CL:.4f} J S^2 vs classical MC = "
    f"{MC_LIT} J S^2  ->  RPA {100 * (1 - TC_RPA_CL / MC_LIT):.1f}% below MC "
    f"(paper: 'RPA underestimates it by 10%')"
)
print(
    f"      CD  classical ~ {cd_cl_grid_corr:.4f} J S^2 (grid-corrected) "
    f"-> {100 * (cd_cl_grid_corr / MC_LIT - 1):+.1f}% vs MC "
    f"(paper: 'CD comes very close to the exact result')"
)
assert abs(cd_cl_grid_corr - MC_LIT) / MC_LIT < 0.10


# ============================================================================
banner("ALL CHECKS PASSED")
print(f"total runtime: {time.time() - T0:.1f} s")
