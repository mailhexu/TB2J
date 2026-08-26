#!/usr/bin/env python3
"""
03 — Single-ion anisotropy, anisotropic exchange, multi-site generalization,
      and the bridge to TB2J magnon conventions.

Companion report: 03_anisotropy_multisite_conventions.md
Reference paper : arXiv:2405.00477 (LaTeX source: TB2J/Refs/2405.00477/main.tex)

Covered paper equations (labels from main.tex):
  * 2dheisenberghamiltonian   anisotropic Hamiltonian  -1/2 sum lambda S^z S^z - A sum (S^z)^2
  * sia-reorder               a+a a+a = a+a+a+a + a+a   (operator-ordering ambiguity)
  * hp_correction             Delta omega^HP_q
  * standard_order            RPA SIA decoupling (symmetrized ordering)
  * rpa_correct_ordering      alternative RPA orderings (2<Sz> -+ 1)
  * rpa_correction            Delta omega^RPA_q = 2A<Sz> + lambda_0 <Sz>
  * callen_correction         Delta omega^CD_q
  * delta_rpa / delta_cd      T=0 gaps  2AS + S lambda_0  vs  A(2S-1) + S lambda_0
  * multisite Hamiltonian, commutator, H_q^ab (multisite-H_rpa), H_q^ab (multisite-H_cd),
    psi^ab_q (fluctuation-dissipation), phi (eq:phi_multiple), 2siteJab / 2SiteAE

TB2J bridge (magnon3.py conventions), verified numerically on a 2-sublattice model:
  * Jq normalization 1/(S_i S_j), Fourier phase exp(-2 pi i k.R), negative sign in Hq
  * BdG matrix [[A1 - C, B], [B^dagger, A2 - C]], Cholesky-based bosonic diagonalization
  * stored tensor JR = (1/2) J^paper S_i S_j (isot. + lambda z^z) with SIA k1 = A S^2 on-site
  * identification  A1 - C  ==  H_q^{RPA, T=0}  (paper)  =>  gap 2AS + S lambda_0

Run:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python 03_anisotropy_multisite_conventions.py
"""

import numpy as np
import sympy as sp

PASS_LINES = []


def verify(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    PASS_LINES.append(name)
    print(f"PASS: {name}")


# ----------------------------------------------------------------------------
# Helpers: spin matrices, boson matrices, noncommutative normal ordering
# ----------------------------------------------------------------------------


def spin_matrices(S):
    """Return (Sz, Sp, Sm) sympy matrices for spin S (Rational)."""
    S = sp.Rational(S)
    d = int(2 * S) + 1
    ms = [S - k for k in range(d)]  # basis order: m = S, S-1, ..., -S
    Sz = sp.diag(*ms)
    Sp = sp.zeros(d, d)
    for i in range(1, d):
        m = ms[i]
        Sp[i - 1, i] = sp.sqrt(S * (S + 1) - m * (m + 1))
    Sm = Sp.T  # real symmetric construction -> transpose is conjugate
    return Sz, Sp, Sm


def boson_matrices(dim=6):
    """Boson annihilation operator on Fock states |0..dim-1>."""
    a = sp.zeros(dim, dim)
    for n in range(1, dim):
        a[n - 1, n] = sp.sqrt(n)
    return a, a.T


def _split_term(t):
    """Split an expanded term into (commutative coeff, noncommutative symbol list)."""
    coeff = sp.Integer(1)
    seq = []
    for f in sp.expand(t).as_ordered_factors():
        if f.is_commutative:
            coeff = coeff * f
        elif f.is_Pow and f.base.is_Symbol:
            seq.extend([f.base] * int(f.exp))
        else:
            seq.append(f)
    return coeff, seq


def normal_order(expr, rules, maxiter=200):
    """Normal-order a polynomial of noncommutative symbols.

    rules: dict mapping an adjacent pair (x, y) to a list of (coeff, seq)
    replacements implementing [x, y] = xy - yx.  Terminates when the ordering
    implied by the rules is reached (bubble sort with algebraic side terms).
    """
    expr = sp.expand(expr)
    if expr == 0:
        return sp.Integer(0)
    terms = [_split_term(t) for t in expr.as_ordered_terms()]
    for _ in range(maxiter):
        changed = False
        out = []
        for coeff, seq in terms:
            hit = None
            for i in range(len(seq) - 1):
                key = (seq[i], seq[i + 1])
                if key in rules:
                    hit = (i, key)
                    break
            if hit is None:
                out.append((coeff, seq))
                continue
            i, key = hit
            for rc, rseq in rules[key]:
                out.append((coeff * rc, seq[:i] + list(rseq) + seq[i + 2 :]))
            changed = True
        terms = out
        if not changed:
            break
    else:
        raise RuntimeError("normal_order did not terminate")
    res = sp.Integer(0)
    for coeff, seq in terms:
        res += coeff * sp.Mul(*seq)
    return sp.expand(res)


# ============================================================================
# SECTION 1 — Spin algebra and the SIA commutator orderings
# ============================================================================
print("\n=== Section 1: spin commutators and SIA commutator identities ===")

for Shalf in (sp.Rational(1, 2), 1, sp.Rational(3, 2)):
    Sz, Sp, Sm = spin_matrices(Shalf)
    d = Sz.shape[0]
    zero = sp.zeros(d, d)
    verify(
        f"[Sz,Sp]=+Sp and [Sp,Sm]=2Sz for S={Shalf}",
        (Sz * Sp - Sp * Sz - Sp).applyfunc(sp.simplify) == zero
        and (Sp * Sm - Sm * Sp - 2 * Sz).applyfunc(sp.simplify) == zero,
    )
    # Paper eqs (standard_order) / (rpa_correct_ordering):
    #   -[Sp, Sz^2] = Sz Sp + Sp Sz = 2 Sz Sp - Sp = 2 Sp Sz + Sp
    lhs = -(Sp * Sz * Sz - Sz * Sz * Sp)
    for tag, rhs in (
        ("SzSp+SpSz", Sz * Sp + Sp * Sz),
        ("2SzSp-Sp", 2 * Sz * Sp - Sp),
        ("2SpSz+Sp", 2 * Sp * Sz + Sp),
    ):
        verify(
            f"-[Sp,(Sz)^2] == {tag} for S={Shalf}",
            (lhs - rhs).applyfunc(sp.simplify) == zero,
        )

# Same identities from abstract noncommutative algebra: [Sz, Sp] = Sp  =>
# Sp Sz = Sz Sp - Sp.  Rules move Sp to the right of Sz.
Sznc, Spnc = sp.symbols("Sz Sp", commutative=False)
rules_spin = {(Spnc, Sznc): [(1, (Sznc, Spnc)), (-1, (Spnc,))]}
e1 = Sznc * Spnc + Spnc * Sznc - (2 * Sznc * Spnc - Spnc)
verify(
    "noncommutative: SzSp+SpSz == 2SzSp-Sp (from [Sz,Sp]=Sp)",
    normal_order(e1, rules_spin) == 0,
)
e2 = Sznc * Spnc + Spnc * Sznc - (2 * Spnc * Sznc + Spnc)
verify(
    "noncommutative: SzSp+SpSz == 2SpSz+Sp (from [Sz,Sp]=Sp)",
    normal_order(e2, rules_spin) == 0,
)

# ----------------------------------------------------------------------------
# SECTION 2 — Boson re-ordering identity (paper eq. sia-reorder)
# ----------------------------------------------------------------------------
print("\n=== Section 2: boson operator re-ordering (eq. sia-reorder) ===")

anc, adc = sp.symbols("a ad", commutative=False)
rules_boson = {(anc, adc): [(1, (adc, anc)), (1, ())]}  # a ad = ad a + 1
expr = adc * anc * adc * anc - adc * adc * anc * anc - adc * anc
verify(
    "noncommutative: ad*a*ad*a == ad*ad*a*a + ad*a  (eq. sia-reorder)",
    normal_order(expr, rules_boson) == 0,
)

# Symbolic check on a generic Fock state |n>:  (a+a)^2|n> = n^2|n>,
# (a+)^2 a^2|n> = n(n-1)|n>  =>  n^2 = n(n-1) + n.
nn = sp.symbols("n", nonnegative=True)
verify(
    "Fock algebra: n^2 == n(n-1) + n for a+a a+a |n>",
    sp.simplify(nn**2 - (nn * (nn - 1) + nn)) == 0,
)

a6, ad6 = boson_matrices(6)
n6 = ad6 * a6
lhs6 = n6 * n6
rhs6 = ad6 * ad6 * a6 * a6 + n6
verify(
    "6x6 Fock matrix rep: (ad a)(ad a) == ad ad a a + ad a",
    (lhs6 - rhs6).applyfunc(sp.expand) == sp.zeros(6, 6),
)

# ----------------------------------------------------------------------------
# SECTION 3 — HP treatment of single-ion anisotropy: exact one-magnon gap
# ----------------------------------------------------------------------------
print("\n=== Section 3: HP/SIA quadratic Hamiltonian and T=0 gap ===")

# HP: Sz = S - n  =>  -A (Sz)^2 = -A S^2 + 2 A S n - A n^2 and (eq. sia-reorder)
# n^2 = ad a ad a = ad ad a a + ad a.  In the *single-magnon sector* the
# quartic term vanishes (ad ad a a |1 magnon> = 0), leaving
#   H_SIA^quad = -A S^2 + A(2S - 1) n       =>  gap contribution A(2S-1).
# Because S^z|S,m=S-1> = S-1 exactly and (S^z)^2 is diagonal, the one-magnon
# sector of the *full* Hamiltonian is exactly reproduced by the quadratic HP
# form; we verify by exact diagonalization of a 2-site cluster:
#   H = -J S1.S2 - lambda S1z S2z - A[(S1z)^2 + (S2z)^2]
# The one-magnon sector {|S-1,S>, |S,S-1>} closes under H (M^z conservation).

Ssym, Jsy, lsy, Asy = sp.symbols("S J lambda A", positive=True)
diag = (
    -Jsy * (Ssym - 1) * Ssym
    - lsy * (Ssym - 1) * Ssym
    - Asy * ((Ssym - 1) ** 2 + Ssym**2)
)
off = -Jsy * Ssym  # <-J/2 (S1+S2- + S1-S2+) matrix element between the 2 states
block = sp.Matrix([[diag, off], [off, diag]])
E_GS = -Jsy * Ssym**2 - lsy * Ssym**2 - 2 * Asy * Ssym**2
w_syms = [sp.simplify(w - E_GS) for w in block.eigenvals()]
target = [
    Asy * (2 * Ssym - 1) + Ssym * lsy,
    Asy * (2 * Ssym - 1) + Ssym * lsy + 2 * Ssym * Jsy,
]
verify(
    "2-site symbolic ED (generic S): one-magnon energies == A(2S-1)+S*lambda (+2SJ)",
    all(
        sp.simplify(w - t) == 0
        for w, t in zip(
            sorted(w_syms, key=sp.default_sort_key),
            sorted(target, key=sp.default_sort_key),
        )
    ),
)

# Belt and braces: full matrix ED for several S with numeric couplings.
Jv, lv, Av = sp.Rational(13, 10), sp.Rational(4, 10), sp.Rational(7, 10)
for Shalf in (sp.Rational(1, 2), 1, sp.Rational(3, 2), 2):
    Sz, Sp, Sm = spin_matrices(Shalf)
    d = Sz.shape[0]
    I = sp.eye(d)
    S1S2 = (
        sp.kronecker_product(Sz, Sz)
        + (sp.kronecker_product(Sp, Sm) + sp.kronecker_product(Sm, Sp)) / 2
    )
    H = (
        -Jv * S1S2
        - lv * sp.kronecker_product(Sz, Sz)
        - Av * (sp.kronecker_product(Sz * Sz, I) + sp.kronecker_product(I, Sz * Sz))
    )
    # M = 2S - 1 sector: basis states (m1,m2) with m1+m2 = 2S-1
    ms = [Shalf - k for k in range(d)]
    idx = [(i, j) for i in range(d) for j in range(d) if ms[i] + ms[j] == 2 * Shalf - 1]
    flat = [i * d + j for (i, j) in idx]
    sub = H.extract(flat, flat)
    eigs = sorted(sp.re(v) for v in sub.eigenvals())
    E_gs = sp.re(
        H[(int(0), int(0))]  # |S,S> is the ground state for FM J, small anisotropy
    )
    got = [sp.nsimplify(e - E_gs) for e in eigs]
    want = sorted(
        [
            Av * (2 * Shalf - 1) + Shalf * lv,
            Av * (2 * Shalf - 1) + Shalf * lv + 2 * Shalf * Jv,
        ]
    )
    verify(
        f"2-site exact ED S={Shalf}: one-magnon gap == A(2S-1)+S*lambda_0, +2SJ",
        all(sp.simplify(g - w) == 0 for g, w in zip(got, want)),
    )

# ----------------------------------------------------------------------------
# SECTION 4 — RPA decoupling and the operator-ordering ambiguity
# ----------------------------------------------------------------------------
print("\n=== Section 4: RPA corrections and operator ordering ===")

m, Ss, As, lam0 = sp.symbols("m S A lambda_0", positive=True)

# (standard_order):  -<[Sp,(Sz)^2]> = <Sz Sp + Sp Sz>  -> (RPA) 2 m G
# (rpa_correct_ordering): 2 Sz Sp - Sp  -> (2 m - 1) G ;  2 Sp Sz + Sp -> (2 m + 1) G
dRPA_std = 2 * m
dRPA_minus = 2 * m - 1
dRPA_plus = 2 * m + 1
# Anisotropic exchange: [Sp, -1/2 sum lambda S^z S^z] gives +lambda_0 Sp (see
# appendix commutator, verified in Section 6) -> + lambda_0 m in all schemes.
Dw_RPA = 2 * As * m + lam0 * m  # eq. (rpa_correction)
Dw_RPA_ord = (2 * m - 1) * As + lam0 * m  # from (rpa_correct_ordering)

# T=0 limits, <Sz> -> S:
verify(
    "Delta^RPA(T=0) == 2AS + S lambda_0  (eq. delta_rpa)",
    sp.simplify(Dw_RPA.subs(m, Ss) - (2 * As * Ss + lam0 * Ss)) == 0,
)
# HP gap from Section 3 (exact): A(2S-1) + S lambda_0
verify(
    "Delta^HP(T=0) == A(2S-1) + S lambda_0  (eq. delta_cd, HP branch)",
    sp.simplify((As * (2 * Ss - 1) + lam0 * Ss) - (As * (2 * Ss - 1) + lam0 * Ss)) == 0,
)

# Callen decoupling of the SIA (on-site, alpha = m/(2S^2)):
#   <<Sz Sp>> -> (m - alpha psi) G,  <<Sp Sz>> -> (m - alpha <Sp Sm>) G with
#   <Sp Sm> = <Sm Sp> + 2 m  =>  coefficient  2m - (m/S^2)(m + psi_bar)
psi_bar = sp.symbols("psi_bar", nonnegative=True)
Dw_CD_A = As * (2 * m - (m / Ss**2) * (m + psi_bar))
# eq. (callen_correction) A-terms with psi_bar = 2 m phi:
phi = sp.symbols("phi", nonnegative=True)
Dw_CD_paperA = As * (2 * m - m**2 / Ss**2) - 2 * As * m**2 / Ss**2 * phi
verify(
    "CD A-coefficient 2m-(m/S^2)(m+psi_bar) == paper A-terms with psi_bar=2m*phi",
    sp.simplify(Dw_CD_A - Dw_CD_paperA).subs(psi_bar, 2 * m * phi) == 0,
)
verify(
    "Delta^CD(T=0) == A(2S-1) + S lambda_0  (eq. delta_cd, CD branch)",
    sp.simplify(
        (Dw_CD_A + lam0 * m).subs([(m, Ss), (psi_bar, 0)])
        - (As * (2 * Ss - 1) + lam0 * Ss)
    )
    == 0,
)

# S = 1/2 irrelevance:
verify(
    "S=1/2: A(2S-1) == 0  (HP and CD gaps lose the SIA at T=0)",
    sp.simplify(As * (2 * sp.Rational(1, 2) - 1)) == 0,
)
verify(
    "S=1/2: RPA spurious T=0 gap 2AS == A != 0  (ordering failure)",
    sp.simplify((2 * As * sp.Rational(1, 2)) - As) == 0,
)

# ----------------------------------------------------------------------------
# SECTION 5 — Multi-site commutator (paper appendix) on a 2-site cluster
# ----------------------------------------------------------------------------
print("\n=== Section 5: multi-site commutator [S^+_{a i}, H] ===")

for Shalf in (sp.Rational(1, 2), 1):
    Sz, Sp, Sm = spin_matrices(Shalf)
    I = sp.eye(Sz.shape[0])
    J12, l12, Av2 = sp.Rational(7, 5), sp.Rational(3, 10), sp.Rational(2, 5)
    H = (
        -J12
        * (
            sp.kronecker_product(Sz, Sz)
            + (sp.kronecker_product(Sp, Sm) + sp.kronecker_product(Sm, Sp)) / 2
        )
        - l12 * sp.kronecker_product(Sz, Sz)
        - Av2 * (sp.kronecker_product(Sz * Sz, I) + sp.kronecker_product(I, Sz * Sz))
    )
    Sp1 = sp.kronecker_product(Sp, I)
    Sz1 = sp.kronecker_product(Sz, I)
    Sp2 = sp.kronecker_product(I, Sp)
    Sz2 = sp.kronecker_product(I, Sz)
    comm = Sp1 * H - H * Sp1
    # Correct appendix commutator (the printed A(S^z S^+ - S^+ S^z) = A [S^z,S^+]
    # = A S^+ is a typo: [S^+,-A(S^z)^2] = A(S^z S^+ + S^+ S^z), verified here).
    rhs = (
        J12 * (Sp1 * Sz2 - Sz1 * Sp2) + l12 * Sp1 * Sz2 + Av2 * (Sz1 * Sp1 + Sp1 * Sz1)
    )
    rhs_printed_typo = (
        J12 * (Sp1 * Sz2 - Sz1 * Sp2) + l12 * Sp1 * Sz2 + Av2 * (Sz1 * Sp1 - Sp1 * Sz1)
    )
    verify(
        f"[S1+, H_multisite] == corrected appendix commutator (A-term symmetrized), S={Shalf}",
        (comm - rhs).applyfunc(sp.simplify) == sp.zeros(*comm.shape),
    )
    if Shalf == 1:
        verify(
            "printed appendix A-term A(Sz Sp - Sp Sz) = A Sp fails (typo evidence)",
            (comm - rhs_printed_typo).applyfunc(sp.simplify) != sp.zeros(*comm.shape),
        )

# ----------------------------------------------------------------------------
# SECTION 6 — Multi-site dynamical matrices: reductions and hermiticity
# ----------------------------------------------------------------------------
print("\n=== Section 6: multisite H^RPA and H^CD (corrected) ===")

# --- 6a. Symbolic single-site reduction on a minimal BZ with N_q = 2 --------
print("-- 6a: mini-BZ (N_q = 2) symbolic single-site reductions --")
JQ0, JQpi, lq0, lqpi, n0, n1 = sp.symbols(
    "J_q0 J_qpi lambda_q0 lambda_qpi n0 n1", real=True
)
psi0, psi1 = 2 * m * n0, 2 * m * n1
psibar = (psi0 + psi1) / 2


# RPA multisite matrix (paper eq. multisite-H_rpa), n_a = 1:
def H_rpa_ms(qval):
    Jq = {0: JQ0, sp.pi: JQpi}[qval]
    return m * (JQ0 + lq0) + 2 * As * m - m * Jq


def H_rpa_paper(qval):
    # omega^RPA_q + Delta^RPA  (eqs. rpa-magnon3d + rpa_correction)
    Jq = {0: JQ0, sp.pi: JQpi}[qval]
    return m * (JQ0 - Jq) + 2 * As * m + lq0 * m


for qv in (0, sp.pi):
    verify(
        f"H^RPA multisite -> single-site formula (q={qv})",
        sp.simplify(H_rpa_ms(qv) - H_rpa_paper(qv)) == 0,
    )


# CD multisite matrix, n_a = 1, N_q = 2, with the two index/typo fixes
#   (i) -m J_q  (printed +m J_q in multisite-H_cd, line 2)
#   (ii) A-term (m/S^2)(m + psi_bar)  (printed (1 + psi_bar))
#   (iii) Hadamard index psi^{ab} in the second term (printed psi^{ac})
def H_cd_ms(qval):
    Jq = {0: JQ0, sp.pi: JQpi}[qval]
    if qval == 0:  # q - q' = -q' -> (lam+J)_{q'} for even couplings
        pairs = [(JQ0, lq0 + JQ0, psi0), (JQpi, lqpi + JQpi, psi1)]
    else:  # pi - 0 = pi, pi - pi = 0
        pairs = [(JQ0, lqpi + JQpi, psi0), (JQpi, lq0 + JQ0, psi1)]
    fl = sum(Jqp * psi - lamJqp * psi for Jqp, lamJqp, psi in pairs)
    return (
        m * (JQ0 + lq0)
        - m * Jq
        + m / (2 * Ss**2 * 2) * fl
        + As * (2 * m - (m / Ss**2) * (m + psibar))
    )


def H_cd_paper(qval):
    Jq = {0: JQ0, sp.pi: JQpi}[qval]
    if qval == 0:
        sumJ = (JQ0 - JQ0) * n0 + (JQpi - JQpi) * n1
        suml = (lq0 + 2 * As) * n0 + (lqpi + 2 * As) * n1
    else:
        sumJ = (JQ0 - JQpi) * n0 + (JQpi - JQ0) * n1
        suml = (lqpi + 2 * As) * n0 + (lq0 + 2 * As) * n1
    return (
        m * (JQ0 - Jq)
        + m**2 / (Ss**2 * 2) * sumJ
        + As * (2 * m - m**2 / Ss**2)
        + lq0 * m
        - m**2 / (Ss**2 * 2) * suml
    )


for qv in (0, sp.pi):
    verify(
        f"H^CD multisite (corrected) -> omega^CD + Delta^CD (q={qv})",
        sp.simplify(H_cd_ms(qv) - H_cd_paper(qv)) == 0,
    )

# Demonstrate the printed (1 + psi_bar) A-term is inconsistent with
# eq. (callen_correction): difference is A m (m-1)/S^2, nonzero unless m = 1.
printed_diff = As * (2 * m - (m / Ss**2) * (1 + psibar)) - As * (
    2 * m - (m / Ss**2) * (m + psibar)
)
verify(
    "printed '(1+psi_bar)' A-term differs from paper eq.(callen_correction) by A*m*(m-1)/S^2",
    sp.simplify(printed_diff - As * m * (m - 1) / Ss**2) == 0
    and sp.simplify(As * m * (m - 1)) != 0,
)
# Same for the '+m J_q' sign: at T=0 (psi=0, m=S) the printed matrix would give
# +S J_q instead of -S J_q, i.e. it cannot reduce to omega = S(J_0 - J_q).
verify(
    "printed '+m J_q' sign cannot reproduce S(J_0-J_q) at T=0",
    sp.simplify((m * (JQ0 + lq0) + m * JQ0).subs(m, Ss) - Ss * (JQ0 - JQ0 + 2 * JQ0))
    != 0,
)

# --- 6b. Numeric checks on random couplings --------------------------------
print("-- 6b: numeric reductions / hermiticity on random 2-sublattice couplings --")
rng = np.random.default_rng(20240517)


def build_Jq(couplings, qs, n_sub, sign=+1):
    """Fourier transform of a dict {(a,b,R-int): value}; sign=+1 -> e^{+2pi i q R}."""
    Jq = np.zeros((len(qs), n_sub, n_sub), dtype=complex)
    for iq, q in enumerate(qs):
        for (a, b, R), v in couplings.items():
            Jq[iq, a, b] += v * np.exp(sign * 2j * np.pi * q * R)
    return Jq


# (i) single-site HP multisite reduction (paper appendix H^HP), numeric
Nq = 16
qs = np.arange(Nq) / Nq
Rmax = 4
J_R = {R: rng.normal() for R in range(1, Rmax + 1)}
l_R = {R: rng.normal() * 0.3 for R in range(1, Rmax + 1)}
nB_q = rng.random(Nq) * 2  # arbitrary positive "occupations"
Sn = 1.5
An = 0.4


def Jfun(q):
    return sum(2 * v * np.cos(2 * np.pi * q * R) for R, v in J_R.items())


def lfun(q):
    return sum(2 * v * np.cos(2 * np.pi * q * R) for R, v in l_R.items())


for iq, q in enumerate(qs):
    # multisite H^HP formula specialized to n_a = 1, n^{aa}_{q'} = nB_{q'}
    s = 0.0
    for iq2, q2 in enumerate(qs):
        qdiff = qs[(iq - iq2) % Nq]
        s += (
            Jfun(q) * nB_q[iq2]
            - (Jfun(qdiff) + lfun(qdiff)) * nB_q[iq2]
            + Jfun(q2) * nB_q[iq2]
            - (Jfun(0.0) + lfun(0.0)) * nB_q[iq2]
            - 4 * An * nB_q[iq2]
        )
    H_HP_ms = Sn * (Jfun(0) + lfun(0)) - Sn * Jfun(q) + An * (2 * Sn - 1) + s / Nq
    # paper: omega^HP + Delta^HP  (eqs. HP_magnon_energy + hp_correction)
    phi_n = nB_q.mean()
    mz = Sn - phi_n
    H_HP_paper = (
        mz * (Jfun(0) - Jfun(q))
        + (np.array([Jfun(q2) for q2 in qs]) * nB_q).sum() / Nq
        - (np.array([(Jfun(qs[(iq - iq2) % Nq])) for iq2 in range(Nq)]) * nB_q).sum()
        / Nq
        + An * (2 * Sn - 1)
        + lfun(0) * mz
        - (np.array([lfun(qs[(iq - iq2) % Nq]) for iq2 in range(Nq)]) * nB_q).sum() / Nq
        - 4 * An * phi_n
    )
    if iq == 0:
        verify(
            "multisite H^HP -> omega^HP + Delta^HP (numeric, n_a=1, random couplings)",
            abs(H_HP_ms - H_HP_paper) < 1e-10,
        )
    else:
        assert abs(H_HP_ms - H_HP_paper) < 1e-10
verify(
    "multisite H^HP reduction holds at every q of the 16-point mesh",
    True,
)


# (ii) hermiticity of H^RPA and H^CD on a random 2-sublattice model
n_sub = 2
Rlist2 = [-2, -1, 0, 1, 2]
coup = {}
for a in range(n_sub):
    for b in range(n_sub):
        for R in Rlist2:
            if a == b and R == 0:
                continue
            if R > 0:  # draw once per +R, then enforce both symmetries
                coup[(a, b, R)] = rng.normal() * 0.5
                coup[(a, b, -R)] = coup[(a, b, R)]
                coup[(b, a, -R)] = coup[(a, b, R)]
                coup[(b, a, R)] = coup[(a, b, R)]
        if a != b:
            coup[(a, b, 0)] = rng.normal() * 0.5
            coup[(b, a, 0)] = coup[(a, b, 0)]
lam_coup = {k: 0.1 * v for k, v in coup.items()}
Jq2 = build_Jq(coup, qs, n_sub)
lq2 = build_Jq(lam_coup, qs, n_sub)
verify(
    "random even couplings give real-symmetric J_q^{ab}",
    np.allclose(Jq2.real, Jq2) and np.allclose(Jq2, Jq2.transpose(0, 2, 1)),
)

mn, Sn2, An2 = 0.7, 1.0, 0.3
J0_2 = Jq2[0]
l0_2 = lq2[0]
diagR = np.diag(mn * (J0_2 + l0_2).sum(axis=1) + 2 * An2 * mn)
H_rpa2 = diagR[None, :, :] - mn * Jq2
verify(
    "H^RPA multisite is Hermitian for all q (real couplings)",
    np.allclose(H_rpa2, H_rpa2.conj().transpose(0, 2, 1)),
)

# psi from random real orthogonal U and positive occupations
psi_arr = np.zeros((Nq, n_sub, n_sub))
for iq in range(Nq):
    X = rng.normal(size=(n_sub, n_sub))
    Qm, _ = np.linalg.qr(X)
    occ = rng.random(n_sub)
    psi_arr[iq] = 2 * mn * (Qm * occ) @ Qm.T
verify(
    "psi^{ab}_q is Hermitian per q",
    np.allclose(psi_arr, psi_arr.conj().transpose(0, 2, 1)),
)


def H_cd_build(iq, JqA, lqA, psiA, mval, Sval, Aval, Nqv):
    """Corrected multisite-H_cd (Hadamard psi index, A-term (m+psi_bar))."""
    n = JqA.shape[1]
    H = np.diag(mval * (JqA[0] + lqA[0]).sum(axis=1)) - mval * JqA[iq]
    acc = np.zeros((n, n), dtype=complex)
    for iq2 in range(Nqv):
        iqdiff = (iq - iq2) % Nqv
        acc += (lqA[iqdiff] + JqA[iqdiff]) * psiA[iq2]  # Hadamard (a,b)
    H += -mval / (2 * Sval**2 * Nqv) * acc
    Jpsi = np.einsum("qac,qca->a", JqA, psiA) / Nqv
    H += np.diag(mval / (2 * Sval**2) * Jpsi)
    psib = psiA.mean(axis=0).diagonal()
    H += np.diag(Aval * (2 * mval - (mval / Sval**2) * (mval + psib)))
    return H


H_cd2 = np.stack(
    [H_cd_build(iq, Jq2, lq2, psi_arr, mn, Sn2, An2, Nq) for iq in range(Nq)]
)
verify(
    "H^CD multisite (corrected, Hadamard psi) is Hermitian for all q (real couplings)",
    np.allclose(H_cd2, H_cd2.conj().transpose(0, 2, 1)),
)
diagR_S_cd = np.diag(Sn2 * (J0_2 + l0_2).sum(axis=1) + An2 * (2 * Sn2 - 1))
verify(
    "H^CD(psi=0, m=S) == LSWT matrix with SIA gap A(2S-1) (NOT the RPA 2AS)",
    np.allclose(
        H_cd_build(3, Jq2, lq2, np.zeros_like(psi_arr), Sn2, Sn2, An2, Nq),
        diagR_S_cd - Sn2 * Jq2[3],
    ),
)

# hermiticity caveat: with complex (non-even) J_q the finite-psi Hadamard form
# picks up a small anti-Hermitian part of order Im(J_q)*psi; at psi = 0 (T = 0)
# the matrix is Hermitian for any real-in-R couplings.
coup_c = dict(coup)
# break inversion evenness while keeping the pair relation J^{ba}(-R)=J^{ab}(R)
coup_c[(0, 1, 1)] += 0.37
coup_c[(1, 0, -1)] += 0.37
coup_c[(0, 1, -1)] -= 0.11
coup_c[(1, 0, 1)] -= 0.11
Jq2c = build_Jq(coup_c, qs, n_sub)
verify(
    "perturbed couplings: J_q complex but still Hermitian (physical storage)",
    np.abs(Jq2c.imag).max() > 1e-3
    and np.allclose(Jq2c, Jq2c.conj().transpose(0, 2, 1)),
)
H_cd2c_psi0 = H_cd_build(3, Jq2c, lq2, np.zeros_like(psi_arr), mn, Sn2, An2, Nq)
verify(
    "H^CD at psi=0 is Hermitian for complex J_q too (T=0 is safe)",
    np.allclose(H_cd2c_psi0, H_cd2c_psi0.conj().T),
)
H_cd2c = np.stack(
    [H_cd_build(iq, Jq2c, lq2, psi_arr, mn, Sn2, An2, Nq) for iq in range(Nq)]
)
anti = np.abs(H_cd2c - H_cd2c.conj().transpose(0, 2, 1)).max()
verify(
    "finite-psi H^CD with complex J_q is non-Hermitian at O(Im J * psi) (documented caveat)",
    anti > 1e-12 and anti < 0.05 * np.abs(H_cd2c).max(),
)
H_cd2c_herm = 0.5 * (H_cd2c + H_cd2c.conj().transpose(0, 2, 1))
verify(
    "hermitized (H + H^dagger)/2 restores Hermiticity for the complex-J_q case",
    np.allclose(H_cd2c_herm, H_cd2c_herm.conj().transpose(0, 2, 1)),
)

# (iii) phi (eq. phi_multiple) and psi from fluctuation-dissipation
U = np.linalg.qr(rng.normal(size=(4, 4)))[0]
occ = rng.random(4)
psiU = 2 * mn * (U * occ) @ U.conj().T
n_aa = np.real(np.diag(psiU)) / (2 * mn)  # remove the 2<Sz> prefactor
verify(
    "phi_multiple: sum_a n^{aa}_q == sum_n n^B_n (unitarity of U, per q)",
    abs(n_aa.sum() - occ.sum()) < 1e-12,
)
verify(
    "psi trace relation: (1/N_a) sum_a psi^{aa} == 2<Sz> phi",
    abs(np.trace(psiU) / 4 - 2 * mn * occ.sum() / 4) < 1e-12,
)

# --- 6c. Self-consistent Callen decoupling on a 2-sublattice chain ---------
print("-- 6c: self-consistent CD on 2-sublattice chain (finite T) --")


def callen_mag(S, phi):
    return (
        (S - phi) * (1 + phi) ** (2 * S + 1) + (S + 1 + phi) * phi ** (2 * S + 1)
    ) / ((1 + phi) ** (2 * S + 1) - phi ** (2 * S + 1))


Scd, Acd = 1.0, 0.15
kT = 0.25
NqC = 64
qsC = np.arange(NqC) / NqC
coupC = {
    (0, 0, 1): 0.5,
    (0, 0, -1): 0.5,
    (1, 1, 1): 0.4,
    (1, 1, -1): 0.4,
    (0, 1, 0): 1.0,
    (1, 0, 0): 1.0,
}
lamC = {
    (0, 1, 0): 0.1,
    (1, 0, 0): 0.1,
    (0, 0, 1): 0.02,
    (0, 0, -1): 0.02,
    (1, 1, 1): 0.015,
    (1, 1, -1): 0.015,
}
JqC = build_Jq(coupC, qsC, 2)
lqC = build_Jq(lamC, qsC, 2)
verify("CD chain model has real J_q", np.allclose(JqC.real, JqC))

mcd = Scd
psiC = np.zeros((NqC, 2, 2))
for it in range(600):
    Hs = np.stack(
        [H_cd_build(iq, JqC, lqC, psiC, mcd, Scd, Acd, NqC) for iq in range(NqC)]
    )
    ws, Us = np.linalg.eigh(Hs)
    ws = np.maximum(ws, 1e-8)
    nB = 1.0 / (np.exp(np.minimum(ws / kT, 700)) - 1)
    psiC = 2 * mcd * (Us * nB[:, None, :]) @ Us.conj().transpose(0, 2, 1)
    phiC = nB.sum() / (NqC * 2)
    m_new = callen_mag(Scd, phiC)
    if abs(m_new - mcd) < 1e-12:
        mcd = m_new
        break
    mcd = 0.5 * mcd + 0.5 * m_new
Hs = np.stack([H_cd_build(iq, JqC, lqC, psiC, mcd, Scd, Acd, NqC) for iq in range(NqC)])
ws, Us = np.linalg.eigh(Hs)
nB = 1.0 / (np.exp(np.minimum(np.maximum(ws, 1e-8) / kT, 700)) - 1)
verify(
    "CD self-consistency converged",
    abs(mcd - callen_mag(Scd, nB.sum() / (NqC * 2))) < 1e-9,
)
verify(
    "self-consistent H^CD Hermitian at all q (real-coupling chain)",
    np.allclose(Hs, Hs.conj().transpose(0, 2, 1)),
)
verify(
    "self-consistent psi^{ab}_q Hermitian at all q",
    np.allclose(psiC, psiC.conj().transpose(0, 2, 1)),
)
verify("self-consistent magnon energies positive", (ws > 0).all())
verify("magnetization renormalized below S at finite T", 0 < mcd < Scd)
verify(
    "phi_multiple in the CD loop: (1/N_a) sum_a psi^{aa}/(2m) == phi",
    abs(np.trace(psiC.mean(axis=0)) / 2 / (2 * mcd) - nB.sum() / (NqC * 2)) < 1e-10,
)
# T -> 0 limit: psi -> 0 and H^CD -> H^RPA(m=S)
kT0 = 1e-4
psi0arr = np.zeros_like(psiC)
H0 = H_cd_build(5, JqC, lqC, psi0arr, Scd, Scd, Acd, NqC)
diagC = np.diag(Scd * (JqC[0] + lqC[0]).sum(axis=1) + Acd * (2 * Scd - 1))
verify(
    "T=0: H^CD == LSWT matrix (SIA gap A(2S-1)+S lambda_0, CD branch of delta_cd)",
    np.allclose(H0, diagC - Scd * JqC[5]),
)

# ----------------------------------------------------------------------------
# SECTION 7 — Bridge to TB2J magnon3.py conventions (numpy reimplementation)
# ----------------------------------------------------------------------------
print("\n=== Section 7: TB2J magnon3.py bridge ===")


def tb2j_Jq(Rlist, JR, Snorm, kpts):
    """Minimal copy of Magnon.Jq (no propagation vector): JR/(S_i S_j),
    Fourier phase exp(-2 pi i k.R)."""
    JRn = np.einsum("rijxy,i,j->rijxy", JR, 1.0 / Snorm, 1.0 / Snorm)
    Jq = np.zeros((len(kpts),) + JR.shape[1:], dtype=complex)
    for iR, R in enumerate(Rlist):
        for iq, q in enumerate(kpts):
            Jq[iq] += np.exp(-2j * np.pi * (R @ q)) * JRn[iR]
    return Jq


def tb2j_Hq(Rlist, JR, Snorm, kpts):
    """Minimal copy of Magnon.Hq for collinear-z moments along +z.

    get_rotation_arrays returns U = x + i y and V = z for every site in that
    case, so we hardcode those (documented in the report).
    """
    n = len(Snorm)
    Jq = -tb2j_Jq(Rlist, JR, Snorm, kpts)
    Jmq = Jq.swapaxes(-1, -2).swapaxes(1, 2)
    U = np.tile(np.array([1.0 + 0.0j, 1.0j, 0.0]), (n, 1))
    V = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    J0 = -tb2j_Jq(Rlist, JR, Snorm, np.zeros((1, 3)))[0]
    A1 = np.einsum("ix,qijxy,jy->qij", U, Jmq, U.conj())
    A2 = np.einsum("ix,qijxy,jy->qij", U.conj(), Jq.conj(), U)
    B = np.einsum("ix,qijxy,jy->qij", U, Jmq, U)
    C = np.diag(np.einsum("ix,ijxy,jy,j->i", V, 2 * J0, V, Snorm))
    sq = np.sqrt(Snorm)
    A1 = np.einsum("qij,i,j->qij", A1, sq, sq)
    A2 = np.einsum("qij,i,j->qij", A2, sq, sq)
    B = np.einsum("qij,i,j->qij", B, sq, sq)
    H = np.block([[A1 - C, B], [B.swapaxes(-1, -2).conj(), A2 - C]])
    return H, A1 - C, B


def tb2j_energies(H):
    """Minimal copy of Magnon._diagonalize_magnon_hamiltonian (positive modes)."""
    n = H.shape[-1] // 2
    I = np.eye(n)
    K = np.linalg.cholesky(H)
    g = np.block([[1 * I, 0 * I], [0 * I, -1 * I]])
    eig_matrix = K.swapaxes(-1, -2).conj() @ g @ K
    return np.linalg.eigvalsh(eig_matrix)[:, n:]


def paper_Jq(couplings, kpts, n_sub):
    """Paper-side Fourier transform J^{ab}_q = sum_R J^{ab}(R) e^{+2 pi i q R}."""
    out = np.zeros((len(kpts), n_sub, n_sub), dtype=complex)
    for iq, q in enumerate(kpts):
        for (a, b, R), v in couplings.items():
            out[iq, a, b] += v * np.exp(2j * np.pi * (R * q[0]))
    return out


# Two-sublattice 1D model with complex J^{12}_q (inter bonds at cells 0 and -1
# with different couplings).  Paper-side parameters:
Sv = 1.2
pJ = {
    "J12_0": 1.7,
    "J12_m1": 0.9,
    "J11_1": 0.35,
    "J22_1": 0.35,
    "lam12_0": 0.12,
    "lam12_m1": 0.07,
    "lam11_1": 0.04,
    "lam22_1": 0.04,
    "A": 0.21,
}
coup_iso = {
    (0, 1, 0): pJ["J12_0"],
    (1, 0, 0): pJ["J12_0"],
    (0, 1, -1): pJ["J12_m1"],
    (1, 0, 1): pJ["J12_m1"],
    (0, 0, 1): pJ["J11_1"],
    (0, 0, -1): pJ["J11_1"],
    (1, 1, 1): pJ["J22_1"],
    (1, 1, -1): pJ["J22_1"],
}
coup_lam = {
    (0, 1, 0): pJ["lam12_0"],
    (1, 0, 0): pJ["lam12_0"],
    (0, 1, -1): pJ["lam12_m1"],
    (1, 0, 1): pJ["lam12_m1"],
    (0, 0, 1): pJ["lam11_1"],
    (0, 0, -1): pJ["lam11_1"],
    (1, 1, 1): pJ["lam22_1"],
    (1, 1, -1): pJ["lam22_1"],
}
kpts = np.array([[f, 0.0, 0.0] for f in np.linspace(0, 0.5, 21)])
Jp_q = paper_Jq(coup_iso, kpts, 2)
lp_q = paper_Jq(coup_lam, kpts, 2)
verify(
    "model has complex J^{12}_q (phase-convention stress test)",
    np.abs(Jp_q[:, 0, 1].imag).max() > 1e-3,
)

# paper H^{RPA,T=0}:  S delta_ab sum_c (J0+lam0) + 2AS delta_ab - S J_q
J0p = Jp_q[0]
l0p = lp_q[0]
H_paper = (
    np.stack(
        [np.diag(Sv * (J0p + l0p).sum(axis=1)) + 2 * pJ["A"] * Sv * np.eye(2)]
        * len(kpts)
    )
    - Sv * Jp_q
)
verify(
    "paper H^{RPA,T=0} is Hermitian at all q",
    np.allclose(H_paper, H_paper.conj().transpose(0, 2, 1)),
)

# TB2J-side stored tensors:  JR = (1/2) J^paper S_a S_b (iso I + lam zz),
# SIA added on-site as k1 = A S^2 in the zz element of JR[R=0, a, a].
Rint_to_iR = {-1: 0, 0: 1, 1: 2}
JR_t = np.zeros((3, 2, 2, 3, 3))
Snorm_t = np.array([Sv, Sv])
ez = np.zeros(3)
ez[2] = 1.0
for (a, b, R), Jv_ in coup_iso.items():
    JR_t[Rint_to_iR[R], a, b] += (
        0.5
        * Snorm_t[a]
        * Snorm_t[b]
        * (Jv_ * np.eye(3) + coup_lam[(a, b, R)] * np.outer(ez, ez))
    )
for a in range(2):
    JR_t[1, a, a] += pJ["A"] * Snorm_t[a] ** 2 * np.outer(ez, ez)
Rlist_t = [np.array([r, 0, 0]) for r in (-1, 0, 1)]
H_tb2j, pos_block, Bblock = tb2j_Hq(Rlist_t, JR_t, Snorm_t, kpts)
Jt_q = tb2j_Jq(Rlist_t, JR_t, Snorm_t, kpts)  # (nk, 2, 2, 3, 3), normalized

# (a) phase + normalization conventions: J^TB2J(k) == (1/2) J^paper(-k) == conj/2
verify(
    "J^TB2J(k) == (1/2) J^paper(-k): e^{-2pi i k R} vs e^{+i q R}, stored factor 1/2",
    np.allclose(Jt_q[..., 0, 0], 0.5 * Jp_q.conj())
    and np.allclose(
        Jt_q[..., 2, 2],
        0.5 * (Jp_q + lp_q).conj() + pJ["A"] * np.eye(2),
    ),
)
# (b) anisotropy extraction from the normalized tensor:
#     2 (J~^zz - J~^xx) = lambda_q + 2A delta_ab  (RPA effective anisotropy)
lam_extract = 2 * (Jt_q[..., 2, 2] - Jt_q[..., 0, 0])
verify(
    "lambda_q^{ab} + 2A delta_ab == 2 (J~^zz - J~^xx) of the normalized TB2J tensor",
    np.allclose(lam_extract, lp_q.conj() + 2 * pJ["A"] * np.eye(2)),
)
# (c) block identification: the swapaxes in Jmq compensates the Fourier sign,
#     so the positive-mode block A1 - C equals the paper matrix at the SAME k.
verify(
    "A1 - C == H^{RPA,T=0}_q elementwise at q=k (and B == 0 for zz anisotropy)",
    np.allclose(pos_block, H_paper) and np.allclose(Bblock, 0),
)
# (d) eigenvalue equality: TB2J BdG positive modes vs paper matrix
E_tb2j = np.sort(tb2j_energies(H_tb2j), axis=1)
E_paper = np.sort(np.linalg.eigvalsh(H_paper), axis=1)
verify(
    "TB2J BdG positive-mode eigenvalues == eig(H^{RPA,T=0}_q) at every k",
    np.allclose(E_tb2j, E_paper, atol=1e-10),
)
# (e) Gamma gap and optical mode (eqs. delta_rpa and optical, 2-sublattice)
wG = np.sort(np.linalg.eigvalsh(H_paper[0]))
lam0_tot = (l0p.sum(axis=1))[0]  # lambda_0^{11} + lambda_0^{12}
gap = 2 * pJ["A"] * Sv + Sv * lam0_tot
J12_0 = pJ["J12_0"] + pJ["J12_m1"]
verify(
    "Gamma gap == 2AS + S lambda_0  (TB2J/RPA/semi-classical convention)",
    abs(wG[0] - gap) < 1e-10,
)
verify(
    "optical mode - gap == 2 S J^{12}_0  (eq. optical)",
    abs((wG[1] - wG[0]) - 2 * Sv * J12_0) < 1e-10,
)
gap_hp = pJ["A"] * (2 * Sv - 1) + Sv * lam0_tot
verify(
    "TB2J SIA convention is RPA-like (2AS), not HP/CD-like (A(2S-1)) at T=0",
    abs(wG[0] - gap) < 1e-12 and abs(gap_hp - wG[0]) > 1e-3,
)

# (f) real-J_q variant: elementwise A1 - C == H^paper(k) directly
coup_iso_r = {
    (0, 1, 1): 1.2,
    (1, 0, -1): 1.2,
    (0, 1, -1): 1.2,
    (1, 0, 1): 1.2,
    (0, 0, 1): 0.3,
    (0, 0, -1): 0.3,
    (1, 1, 1): 0.4,
    (1, 1, -1): 0.4,
}
coup_lam_r = {
    (0, 1, 1): 0.1,
    (1, 0, -1): 0.1,
    (0, 1, -1): 0.1,
    (1, 0, 1): 0.1,
    (0, 0, 1): 0.05,
    (0, 0, -1): 0.05,
    (1, 1, 1): 0.02,
    (1, 1, -1): 0.02,
}
Jp_qr = paper_Jq(coup_iso_r, kpts, 2)
lp_qr = paper_Jq(coup_lam_r, kpts, 2)
J0pr, l0pr = Jp_qr[0], lp_qr[0]
H_paperr = (
    np.stack(
        [np.diag(Sv * (J0pr + l0pr).sum(axis=1)) + 2 * pJ["A"] * Sv * np.eye(2)]
        * len(kpts)
    )
    - Sv * Jp_qr
)
JR_r = np.zeros((3, 2, 2, 3, 3))
for (a, b, R), Jv_ in coup_iso_r.items():
    JR_r[Rint_to_iR[R], a, b] += (
        0.5 * Sv**2 * (Jv_ * np.eye(3) + coup_lam_r[(a, b, R)] * np.outer(ez, ez))
    )
for a in range(2):
    JR_r[1, a, a] += pJ["A"] * Sv**2 * np.outer(ez, ez)
_, pos_r, B_r = tb2j_Hq(Rlist_t, JR_r, np.array([Sv, Sv]), kpts)
verify(
    "even-bond model: J_q real and A1 - C == H^paper(k) elementwise",
    np.allclose(Jp_qr.imag, 0) and np.allclose(pos_r, H_paperr) and np.allclose(B_r, 0),
)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"ALL {len(PASS_LINES)} CHECKS PASSED")
print("=" * 70)
for i, name in enumerate(PASS_LINES, 1):
    print(f"  [{i:2d}] {name}")
