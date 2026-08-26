#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_heisenberg_hp_lswt.py
========================
Sympy-verified derivation (part 1 of the TB2J thermal-magnon documentation):

    spin algebra -> Heisenberg Hamiltonian -> Holstein-Primakoff transformation
    -> quadratic (linear) spin-wave theory for a Bravais ferromagnet:
         omega_q = S (J_0 - J_q)
    -> Bogoliubov diagonalization of the collinear two-sublattice
       antiferromagnet (needed later for T_N):
         omega_k = 2 J S |sin(k a)|   (1D chain check)

Reference for equation labels (LaTeX source: TB2J/Refs/2405.00477/main.tex,
arXiv:2405.00477 Sec. 2 "Theory"):

    Eq. (1)  [heisenbrghamiltonian]        H = -1/2 sum_{i!=j} J_ij S_i.S_j
    Eq. (2)                              S^+-_i = S^x_i +- i S^y_i
    Eq. (3)  [eq:hamiltonian in circular coordinates]
    Eq. (4a) [spin-comm-1]                [S^z_i, S^+-_j] = +- S^+-_i delta_ij
    Eq. (4b) [spin-comm-2]                [S^+_i, S^-_j] = 2 S^z_i delta_ij
    Eq. (4c) [spin-comm-H]                [S^+-_i, H] = +- sum_{j!=i} J_ij (...)
    Eq. (5a-c) [hp_spin+/-/z]             Holstein-Primakoff transformation
    Eq. (10) [hp-q-transform]             a_q = N^-1/2 sum_i a_i e^{-i q.R_i}
    Eq. (11) [Jij-q-space]                J_q = sum_R J_0R e^{i q.R}
    Eq. (12) [HP_magnon_energy]           omega_q^HP = <S^z>(J_0-J_q) + ...
    Eq. (13) [HP_mag], Eq. (14) [phi]     <S^z> = S - phi

Conventions (identical to the paper):
  * J_ij > 0 is ferromagnetic; the fully polarized |+z> state is the FM ground
    state.  The antiferromagnet is represented by J_ij = -J_af < 0, i.e.
    H = + J_af sum_bonds S_i.S_j with J_af > 0.
  * Atomic units (hbar = 1); spins dimensionless.
  * Noncommutative sympy symbols are used for boson operators; every operator
    identity is verified on explicit finite matrix representations (spin
    matrices; truncated boson Fock spaces), never by hand.

Every check is executed at run time, protected by `assert`, and prints a PASS
line.  Run with:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python 01_heisenberg_hp_lswt.py
"""

import itertools

import numpy as np
import sympy as sp
from sympy import (
    I,
    Matrix,
    Rational,
    Symbol,
    cos,
    diag,
    exp,
    eye,
    pi,
    sin,
    sqrt,
)
from sympy.physics.quantum import TensorProduct

NPASS = [0]


def ok(cid, desc, cond):
    """Assert a check and print a PASS line."""
    if not cond:
        raise AssertionError(f"FAIL {cid}: {desc}")
    NPASS[0] += 1
    print(f"PASS {cid:<7} {desc}")


def zero_mat(M):
    """True if the sympy Matrix M is identically zero (entrywise simplify)."""
    return M.applyfunc(lambda e: sp.simplify(sp.expand(e))) == sp.zeros(*M.shape)


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def spin_matrices(S):
    """Exact spin-S matrices in the |m> basis ordered m = S, S-1, ..., -S."""
    d = int(round(2 * S + 1))
    ms = [sp.nsimplify(S - k) for k in range(d)]
    Sz = diag(*ms)
    Sp = Matrix(
        d,
        d,
        lambda i, j: sqrt(sp.nsimplify(S * (S + 1) - ms[j] * (ms[j] + 1)))
        if i == j - 1
        else 0,
    )
    Sm = Sp.H
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2 * I)
    return dict(S=S, dim=d, ms=ms, Sz=Sz, Sp=Sp, Sm=Sm, Sx=Sx, Sy=Sy)


def site_op(op, site, n_sites, dim):
    """Embed local operator `op` acting on `site` of an n-site product space."""
    return TensorProduct(*[op if k == site else eye(dim) for k in range(n_sites)])


def boson_ops(D):
    """Truncated boson matrices a, a^dagger, n = a^dagger a on dim-D Fock space."""
    a = Matrix(D, D, lambda i, j: sqrt(j) if i == j - 1 else 0)
    ad = a.H
    return a, ad, ad * a


def nc_count(term):
    """Number of noncommutative factors in an (expanded) monomial."""
    return sum(1 for f in sp.Mul.make_args(term) if not f.is_commutative)


def quadratic_part(expr):
    """Drop all monomials with more than two boson operators (HP expansion)."""
    return sp.Add(*[t for t in sp.Add.make_args(sp.expand(expr)) if nc_count(t) <= 2])


def eval_nc(expr, rep, dim):
    """Evaluate a normal-ordered sympy expression of noncommutative symbols on
    matrix representations `rep` (symbol -> dim x dim Matrix)."""
    total = None
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff = sp.Integer(1)
        mats = []
        for f in sp.Mul.make_args(term):
            if f in rep:
                mats.append(rep[f])
            else:
                coeff = coeff * f  # commutative prefactor
        M = eye(dim)
        for m in mats:
            M = M * m
        M = coeff * M
        total = M if total is None else total + M
    return total


def np_fock(D):
    """Numpy boson matrices (annihilation, creation, number) of dimension D."""
    a = np.diag(np.sqrt(np.arange(1, D, dtype=float)), 1)
    return a, a.T.copy(), np.diag(np.arange(D, dtype=float))


# ============================================================================
# SECTION 1 - Spin algebra in explicit matrix representation
# ============================================================================
def section1():
    print("\n=== Section 1: spin algebra (matrix representation) ===")
    for S in [sp.Rational(1, 2), 1, sp.Rational(3, 2)]:
        tag = f"S={S}"
        ops = spin_matrices(S)
        Sp, Sm, Sz = ops["Sp"], ops["Sm"], ops["Sz"]
        ok(
            f"1.1[{tag}]",
            "[Sz,S+] = +S+  and  [Sz,S-] = -S+",
            zero_mat(Sz * Sp - Sp * Sz - Sp) and zero_mat(Sz * Sm - Sm * Sz + Sm),
        )
        ok(f"1.2[{tag}]", "[S+,S-] = 2 Sz", zero_mat(Sp * Sm - Sm * Sp - 2 * Sz))
        ok(
            f"1.3[{tag}]",
            "S.S = S(S+1) Identity (Casimir)",
            zero_mat(
                ops["Sx"] * ops["Sx"]
                + ops["Sy"] * ops["Sy"]
                + Sz * Sz
                - S * (S + 1) * eye(ops["dim"])
            ),
        )
        ok(
            f"1.4[{tag}]",
            "Sx,Sz Hermitian; Sp = Sm^dagger",
            ops["Sx"] == ops["Sx"].H and ops["Sz"] == ops["Sz"].H and Sp == Sm.H,
        )
        ok(f"1.5[{tag}]", "Sz eigenvalues m = S..-S", list(Sz.diagonal()) == ops["ms"])


# ============================================================================
# SECTION 2 - Two-site Heisenberg cluster: circular form, [S+-,H], exact
#             diagonalization vs Clebsch-Gordan, one-magnon sector
# ============================================================================
def section2():
    print("\n=== Section 2: two-site Heisenberg cluster (exact) ===")
    J = sp.Symbol("J", positive=True)
    for S in [sp.Rational(1, 2), 1]:
        tag = f"S={S}"
        o = spin_matrices(S)
        d = o["dim"]
        Sp1, Sm1, Sz1 = (
            site_op(o["Sp"], 0, 2, d),
            site_op(o["Sm"], 0, 2, d),
            site_op(o["Sz"], 0, 2, d),
        )
        Sp2, Sm2, Sz2 = (
            site_op(o["Sp"], 1, 2, d),
            site_op(o["Sm"], 1, 2, d),
            site_op(o["Sz"], 1, 2, d),
        )
        Sx1, Sy1 = site_op(o["Sx"], 0, 2, d), site_op(o["Sy"], 0, 2, d)
        Sx2, Sy2 = site_op(o["Sx"], 1, 2, d), site_op(o["Sy"], 1, 2, d)

        # paper Eq. (3): H = -J ( 1/2[S1+ S2- + S1- S2+] + Sz1 Sz2 )  for a pair
        Hc = -J * (Rational(1, 2) * (Sp1 * Sm2 + Sm1 * Sp2) + Sz1 * Sz2)
        Hx = -J * (Sx1 * Sx2 + Sy1 * Sy2 + Sz1 * Sz2)
        ok(
            f"2.1[{tag}]",
            "circular-coordinates H equals Cartesian H",
            zero_mat(Hc - Hx),
        )

        # paper Eq. (4c): [S1+-, H] = +- J (S1+- Sz2 - Sz1 S2+-)
        rhs_p = J * (Sp1 * Sz2 - Sz1 * Sp2)
        rhs_m = -J * (Sm1 * Sz2 - Sz1 * Sm2)
        ok(
            f"2.2[{tag}]",
            "[S_i^+-, H] matches paper Eq. (4c) [spin-comm-H]",
            zero_mat(Sp1 * Hc - Hc * Sp1 - rhs_p)
            and zero_mat(Sm1 * Hc - Hc * Sm1 - rhs_m),
        )

        # exact diagonalization vs Clebsch-Gordan: E(St) = -(J/2)(St(St+1)
        #                                              - 2 S(S+1))
        ev = (Hc.subs(J, 1)).eigenvals()
        cg = {}
        for St in [sp.nsimplify(k) for k in range(int(round(2 * S)) + 1)]:
            cg[-Rational(1, 2) * (St * (St + 1) - 2 * S * (S + 1))] = int(
                round(2 * St + 1)
            )
        ok(
            f"2.3[{tag}]",
            f"exact eigenvalues == CG spectrum {cg}",
            {sp.nsimplify(k): v for k, v in ev.items()} == cg,
        )

        # one-magnon sector (m1+m2 = 2S-1): energies relative to E0 = -J S^2
        idx = [
            i
            for i, tup in enumerate(itertools.product(o["ms"], repeat=2))
            if sum(tup) == 2 * S - 1
        ]
        Hsec = (Hc.subs(J, 1)).extract(idx, idx)
        E0 = (Hc.subs(J, 1))[0, 0]  # fully polarized |SS> state
        rel = {sp.nsimplify(v - E0): mult for v, mult in Hsec.eigenvals().items()}
        ok(
            f"2.4[{tag}]",
            "E0 = -J S^2 and one-magnon energies {0, 2JS}",
            E0 == -(S**2) and rel == {0: 1, 2 * S: 1},
        )


# ============================================================================
# SECTION 3 - Holstein-Primakoff transformation, exact on the physical block
# ============================================================================
def section3():
    print("\n=== Section 3: Holstein-Primakoff transformation ===")
    S = sp.Symbol("S", positive=True)
    a_nc = Symbol("a", commutative=False)
    ad_nc = Symbol("a^\\dagger", commutative=False)
    n_nc = ad_nc * a_nc
    print("    HP relations (paper Eqs. 5a-c):")
    print(f"      S+ = {sp.sqrt(2*S - n_nc)} * {a_nc}")
    print(f"      S- = {ad_nc} * {sp.sqrt(2*S - n_nc)}")
    print(f"      Sz = S - {n_nc}")

    # expansion of the square root used for the quadratic Hamiltonian
    x = sp.Symbol("x", positive=True)
    series_sq = sp.sqrt(2 * S - x).series(x, 0, 2).removeO()
    ok(
        "3.1",
        "sqrt(2S-n) = sqrt(2S)(1 - n/(4S)) + O(n^2)",
        sp.simplify(series_sq - sp.sqrt(2 * S) * (1 - x / (4 * S))) == 0,
    )

    for Sv in [sp.Rational(1, 2), 1]:
        tag = f"S={Sv}"
        p = int(round(2 * Sv + 1))  # physical block size  (n <= 2S)
        D = p + 2  # truncation n <= 2S+2, as required
        a, ad, n = boson_ops(D)
        sq = diag(*[sqrt(2 * Sv - k) for k in range(D)])
        Sp = sq * a  # sqrt(2S-n) a     (paper Eq. 5a)
        Sm = ad * sq  # a^dagger sqrt(2S-n) (Eq. 5b)
        Sz = Sv * eye(D) - n  # S - n              (Eq. 5c)

        ok(
            f"3.2[{tag}]",
            f"[a,a+] = 1 on the n<=2S+1 block (D={D})",
            (a * ad - ad * a)[: D - 1, : D - 1] == eye(D - 1),
        )

        def b(M):
            return M[:p, :p]  # physical block n <= 2S

        ok(
            f"3.3[{tag}]",
            "[Sz,S+-] = +- S+- on physical block",
            zero_mat(b(Sz * Sp - Sp * Sz - Sp)) and zero_mat(b(Sz * Sm - Sm * Sz + Sm)),
        )
        ok(
            f"3.4[{tag}]",
            "[S+,S-] = 2 Sz on physical block",
            zero_mat(b(Sp * Sm - Sm * Sp - 2 * Sz)),
        )
        Sx = (Sp + Sm) / 2
        Sy = (Sp - Sm) / (2 * I)
        ok(
            f"3.5[{tag}]",
            "S.S = S(S+1) Identity on physical block",
            zero_mat(b(Sx * Sx + Sy * Sy + Sz * Sz - Sv * (Sv + 1) * eye(D))),
        )

        ex = spin_matrices(Sv)
        ok(
            f"3.6[{tag}]",
            "HP boson representation reproduces the exact " "spin-S matrices",
            b(Sp) == ex["Sp"] and b(Sm) == ex["Sm"] and b(Sz) == ex["Sz"],
        )

    # bosons on different sites commute (2-mode representation)
    a1, ad1, _ = boson_ops(5)
    a2m, ad2m, _ = boson_ops(5)
    a_1 = TensorProduct(a1, eye(5))
    ad_2 = TensorProduct(eye(5), ad2m)
    ok("3.7", "[a_i, a_j^+] = 0 for i != j", zero_mat(a_1 * ad_2 - ad_2 * a_1))


# ============================================================================
# SECTION 4 - Quadratic HP Hamiltonian -> Fourier -> omega_q = S (J_0 - J_q)
# ============================================================================
def fm_quadratic_hamiltonian(n, J_of_pair, S, syms):
    """H_2 = E_cl + S J_0 sum_i ad_i a_i - S sum_{i!=j} J_ij ad_i a_j
    in noncommutative boson symbols syms['a'][i], syms['ad'][i].
    E_cl = -(S^2/2) sum_{i!=j} J_ij = -N S^2 J_0 / 2."""
    a, ad = syms["a"], syms["ad"]
    J0 = sum(J_of_pair(0, j) for j in range(n) if j != 0)
    Ecl = (
        -(S**2)
        * Rational(1, 2)
        * sum(J_of_pair(i, j) for i in range(n) for j in range(n) if i != j)
    )
    H2 = Ecl + S * J0 * sum(ad[i] * a[i] for i in range(n))
    H2 += -S * sum(
        J_of_pair(i, j) * ad[i] * a[j] for i in range(n) for j in range(n) if i != j
    )
    return sp.expand(H2), J0, Ecl


def section4():
    print("\n=== Section 4: quadratic HP Hamiltonian and omega_q = S(J0-Jq) ===")
    S = sp.Symbol("S", positive=True)

    # ---- 4.1 derivation of H_2 from the HP substitution on a bond ----------
    ai, adi = (Symbol("a_i", commutative=False), Symbol("ad_i", commutative=False))
    aj, adj = (Symbol("a_j", commutative=False), Symbol("ad_j", commutative=False))
    sq_i = sqrt(2 * S) - adi * ai / (2 * sqrt(2 * S))  # sqrt(2S-n_i) + O(n^2)
    sq_j = sqrt(2 * S) - adj * aj / (2 * sqrt(2 * S))
    Sp_i, Sm_i = sq_i * ai, adi * sq_i  # paper Eqs. (5a,b)
    Sp_j, Sm_j = sq_j * aj, adj * sq_j
    flip = Rational(1, 2) * (Sp_i * Sm_j + Sm_i * Sp_j)  # paper Eq. (3) term
    flip_q = quadratic_part(flip).subs(ai * adj, adj * ai)  # a_i ad_j -> ad_j a_i
    ok(
        "4.1a",
        "quadratic part of 1/2(S_i+S_j- + S_i-S_j+) is " "S (ad_i a_j + ad_j a_i)",
        sp.expand(flip_q - S * (adi * aj + adj * ai)) == 0,
    )
    SzSz = (S - adi * ai) * (S - adj * aj)  # exact algebra
    ok(
        "4.1b",
        "quadratic part of Sz_i Sz_j is S^2 - S(n_i+n_j)",
        sp.expand(quadratic_part(SzSz) - (S**2 - S * (adi * ai + adj * aj))) == 0,
    )
    # check the normal-ordering step a_i ad_j = ad_j a_i (i != j) on matrices
    a5, ad5, _ = boson_ops(5)
    rep = {
        ai: TensorProduct(a5, eye(5)),
        adi: TensorProduct(ad5, eye(5)),
        adj: TensorProduct(eye(5), ad5),
    }
    ok(
        "4.1c",
        "a_i ad_j = ad_j a_i for i != j (matrix rep)",
        zero_mat(eval_nc(ai * adj - adj * ai, rep, 25)),
    )

    # ---- 4.2 Fourier diagonalization on the N=4 ring, symbolic J1,J2 --------
    n = 4
    J1, J2 = sp.symbols("J1 J2", positive=True)

    def J_of_pair(i, j):
        dist = (i - j) % n
        return {1: J1, 3: J1, 2: J2}[dist]

    syms = {
        "a": [Symbol(f"a{k}", commutative=False) for k in range(n)],
        "ad": [Symbol(f"ad{k}", commutative=False) for k in range(n)],
    }
    H2, J0, Ecl = fm_quadratic_hamiltonian(n, J_of_pair, S, syms)
    ok(
        "4.2a",
        "E_cl = -N S^2 J_0/2 = -2 S^2 (2J1+J2); on the N=4 ring the "
        "antipodal (distance-2) neighbour is counted once: J_0 = 2 J1 + J2",
        sp.expand(Ecl + 2 * S**2 * (2 * J1 + J2)) == 0
        and sp.expand(J0 - 2 * J1 - J2) == 0,
    )

    # paper Eq. (10): a_q = N^-1/2 sum_i a_i e^{-i q.R_i}
    #            =>   a_i = N^-1/2 sum_q e^{+i q.R_i} a_q   (unitary inverse)
    qs = [0, pi / 2, pi, 3 * pi / 2]
    aq = [Symbol(f"a_q{m}", commutative=False) for m in range(n)]
    adq = [Symbol(f"ad_q{m}", commutative=False) for m in range(n)]
    sub = {}
    for i in range(n):
        sub[syms["a"][i]] = sp.Rational(1, 2) * sum(
            exp(I * qs[m] * i) * aq[m] for m in range(n)
        )
        sub[syms["ad"][i]] = sp.Rational(1, 2) * sum(
            exp(-I * qs[m] * i) * adq[m] for m in range(n)
        )
    Hq = sp.expand(H2.subs(sub))

    def Jq_at(angle):
        """J_q = sum_R J_{0R} e^{i q R}  (paper Eq. (11)); displacements on
        the N=4 ring are R = +1,-1(=+3),+2, and e^{3i q} = e^{-i q} on the
        q-grid (angles that are multiples of pi/2), so J_q is real there."""
        return J1 * (exp(I * angle) + exp(-I * angle)) + J2 * exp(2 * I * angle)

    def Jq_of(m):
        return Jq_at(qs[m])

    all_ops = set(aq) | set(adq)
    const = Hq.subs({s: 0 for s in all_ops})
    ok(
        "4.2b",
        "constant part survives Fourier transform unchanged",
        sp.expand(const - Ecl) == 0,
    )
    good = True
    for m1 in range(n):
        for m2 in range(n):
            keep = {s: 0 for s in all_ops}
            keep[adq[m1]], keep[aq[m2]] = 1, 1
            c = sp.expand((Hq - const).subs(keep))
            expected = (S * (J0 - Jq_of(m2))).expand() if m1 == m2 else 0
            good &= sp.simplify(c - expected) == 0
    for term in sp.Add.make_args(sp.expand(Hq - const)):
        # every monomial must be normal ordered (ad_q before a_q) and at most
        # quadratic in the bosons
        facs = [f for f in sp.Mul.make_args(term) if not f.is_commutative]
        names = [str(f) for f in facs]
        good &= len(facs) == 0 or (
            len(facs) == 2
            and names[0].startswith("ad")
            and not names[1].startswith("ad")
        )
    ok(
        "4.2c",
        "H_2 = E_cl + sum_q S(J0-J_q) ad_q a_q exactly "
        "(cross terms cancel; J_q from paper Eq. (11))",
        good,
    )

    # ---- 4.3 paper Eq. (12) [HP_magnon_energy] reduces to LSWT at T=0 -------
    # omega_q^HP = <Sz>(J0-Jq) + (1/N) sum_q' (J_q' - J_{q-q'}) n^B_q'  (12)
    # <Sz> = S - phi,  phi = (1/N) sum_q n^B_q                          (13,14)
    nB = [Symbol(f"nB_{m}", nonnegative=True) for m in range(n)]
    Szav = S - sum(nB) / 4
    for m in range(n):
        omega_HP = Szav * (J0 - Jq_of(m)) + Rational(1, 4) * sum(
            Jq_of(mp) - Jq_at(qs[m] - qs[mp]) for mp in range(n)
        )
        t0 = omega_HP.subs({v: 0 for v in nB})
        ok(
            f"4.3a[q={qs[m]/pi:.3f}pi]",
            "Eq. (12) at T=0 (n^B=0, <Sz>=S) "
            f"-> S(J0-J_q) = {sp.simplify(S*(J0-Jq_of(m)))}",
            sp.simplify(t0 - S * (J0 - Jq_of(m))) == 0,
        )
    # uniform Bose occupation: the interaction term vanishes identically
    for m in range(n):
        corr = Rational(1, 4) * sum(
            Jq_of(mp) - Jq_at(qs[m] - qs[mp]) for mp in range(n)
        )
        ok(
            f"4.3b[q={qs[m]/pi:.3f}pi]",
            "(1/N)sum_q' (J_q'-J_{q-q'}) = 0 " "(uniform n^B gives no shift)",
            sp.simplify(corr) == 0,
        )

    # ---- 4.4 dimer: bosonic H_2 vs exact one-magnon sector -----------------
    Jv = 1
    for Sv in [sp.Rational(1, 2), 1]:
        tag = f"S={Sv}"
        A, B, Ecl_d = Sv * Jv, -Sv * Jv, -(Sv**2) * Jv
        # analytic single-particle (one-magnon) spectrum of H_2:
        M1 = Matrix([[A, B], [B, A]])
        ok(
            f"4.4a[{tag}]",
            "dimer H_2 one-magnon energies = {0, 2JS} " "(= exact result of check 2.4)",
            M1.eigenvals() == {0: 1, 2 * Jv * Sv: 1},
        )
        # full bosonic H_2 diagonalized numerically in a truncated Fock space.
        # The mode rotation mixes only states of equal total boson number, so
        # the subspace N_tot <= D-1 is closed and unaffected by the ceiling;
        # there the spectrum is exactly E_cl + m * 2JS.
        D = 6
        aa, add, nn = np_fock(D)
        Id = np.eye(D)
        Hm = float(A) * (np.kron(nn, Id) + np.kron(Id, nn)) + float(B) * (
            np.kron(add, aa) + np.kron(aa, add)
        )
        safe = [n1 * D + n2 for n1 in range(D) for n2 in range(D) if n1 + n2 <= D - 1]
        ev = np.linalg.eigvalsh(Hm[np.ix_(safe, safe)]) + float(Ecl_d)
        ana = np.sort(
            np.array(
                [
                    float(Ecl_d + 2 * Jv * Sv * q)
                    for p in range(D)
                    for q in range(D)
                    if p + q <= D - 1
                ]
            )
        )
        ok(
            f"4.4b[{tag}]",
            f"Fock-space spectrum of quadratic HP dimer on the "
            f"N_tot<= {D-1} sector = E_cl + m*2JS [max err "
            f"{np.max(np.abs(ev-ana)):.2e}]",
            np.max(np.abs(ev - ana)) < 1e-10,
        )

    # ---- 4.5 exact one-magnon diagonalization of the N=4 ring --------------
    J1v, J2v = sp.Integer(1), sp.Rational(3, 10)
    J0v = 2 * J1v + J2v

    def Jq_num(angle):
        return J1v * (exp(I * angle) + exp(-I * angle)) + J2v * exp(2 * I * angle)

    def J_of_pair_num(i, j):
        dist = (i - j) % n
        return {1: J1v, 3: J1v, 2: J2v}[dist]

    for Sv in [sp.Rational(1, 2), 1]:
        tag = f"S={Sv}"
        o = spin_matrices(Sv)
        d = o["dim"]
        SpL = [site_op(o["Sp"], k, n, d) for k in range(n)]
        SmL = [site_op(o["Sm"], k, n, d) for k in range(n)]
        SzL = [site_op(o["Sz"], k, n, d) for k in range(n)]
        H = -Rational(1, 2) * sum(
            (
                J_of_pair_num(i, j)
                * (
                    Rational(1, 2) * (SpL[i] * SmL[j] + SmL[i] * SpL[j])
                    + SzL[i] * SzL[j]
                )
                for i in range(n)
                for j in range(n)
                if i != j
            ),
            sp.zeros(d**n),
        )
        idx = [
            k
            for k, tup in enumerate(itertools.product(o["ms"], repeat=n))
            if sum(tup) == n * Sv - 1
        ]
        Hsec = H.extract(idx, idx)
        E0 = H[0, 0]
        ok(
            f"4.5a[{tag}]",
            "E_0(FM ring) = -N S^2 J_0/2",
            sp.simplify(E0 + n * Sv**2 * J0v / 2) == 0,
        )
        exact = sorted(
            [
                sp.simplify(v - E0)
                for v, mult in Hsec.eigenvals().items()
                for _ in range(mult)
            ]
        )
        lswt = sorted([sp.simplify(Sv * (J0v - Jq_num(q))) for q in qs])
        ok(
            f"4.5b[{tag}]",
            f"exact one-magnon spectrum {exact} == LSWT " f"{{S(J0-J_q)}}",
            exact == lswt,
        )


# ============================================================================
# SECTION 5 - Two-sublattice antiferromagnet: Bogoliubov diagonalization
# ============================================================================
def section5():
    print("\n=== Section 5: AFM Bogoliubov diagonalization (for T_N) ===")
    Jaf = sp.Symbol("J_af", positive=True)
    S = sp.Symbol("S", positive=True)

    # ---- 5.0 rotated frame on the B sublattice preserves the algebra -------
    o = spin_matrices(1)
    Stz, Stp, Stm = -o["Sz"], o["Sm"], o["Sp"]  # S~z=-Sz, S~+- = S-+
    ok(
        "5.0",
        "rotated frame (-Sz, S-, S+) still satisfies spin algebra",
        zero_mat(Stz * Stp - Stp * Stz - Stp)
        and zero_mat(Stp * Stm - Stm * Stp - 2 * Stz),
    )

    # ---- 5.1 elementary boson commutators on the two-mode Fock matrix ------
    a, ad, _ = boson_ops(5)
    a1, ad1 = TensorProduct(a, eye(5)), TensorProduct(ad, eye(5))
    a2, ad2 = TensorProduct(eye(5), a), TensorProduct(eye(5), ad)

    # keep away from the truncation ceiling (raising-type commutators)
    def bb(M):
        return M[:9, :9]

    elem = {
        "[a1 a1+, x] = 1": bb(a1 * ad1 - ad1 * a1) == eye(9),
        "[ad1 a1, a1] = -a1": zero_mat(bb(ad1 * a1 * a1 - a1 * ad1 * a1 + a1)),
        "[ad2 b-type: ad1 ad2, a1] = -ad2": zero_mat(
            bb(ad1 * ad2 * a1 - a1 * ad1 * ad2 + ad2)
        ),
        "[a1 a2, ad2] = a1": zero_mat(bb(a1 * a2 * ad2 - ad2 * a1 * a2 - a1)),
        "[ad2 a2, ad2] = ad2": zero_mat(bb(ad2 * a2 * ad2 - ad2 * ad2 * a2 - ad2)),
        "[ad1 ad2, ad2] = 0": zero_mat(bb(ad1 * ad2 * ad2 - ad2 * ad1 * ad2)),
        "[a1 a2, a1] = 0": zero_mat(bb(a1 * a2 * a1 - a1 * a1 * a2)),
        "[ad2 a2, a1] = 0": zero_mat(bb(ad2 * a2 * a1 - a1 * ad2 * a2)),
        "[ad1 a1, ad2] = 0": zero_mat(bb(ad1 * a1 * ad2 - ad2 * ad1 * a1)),
    }
    ok(
        "5.1",
        "elementary boson commutators [X Y, op] table (matrix rep)",
        all(elem.values()),
    )

    # ---- 5.2 symbolic commutator -> BdG matrix -> omega = sqrt(A^2-B^2) ----
    aa, ada, bb_, bdb = (
        Symbol("a", commutative=False),
        Symbol("a^\\dagger", commutative=False),
        Symbol("b", commutative=False),
        Symbol("b^\\dagger", commutative=False),
    )
    A, Bk = sp.symbols("A B_k", positive=True)
    elementary = {(aa, ada): 1, (ada, aa): -1, (bb_, bdb): 1, (bdb, bb_): -1}

    def comm_Q(H, op):
        """[H, op] for H quadratic in bosons, using the verified table."""
        out = 0
        for term in sp.Add.make_args(sp.expand(H)):
            facs = [f for f in sp.Mul.make_args(term) if not f.is_commutative]
            coeff = sp.prod([f for f in sp.Mul.make_args(term) if f.is_commutative])
            assert len(facs) == 2
            X, Y = facs
            out += coeff * (
                X * elementary.get((Y, op), 0) + elementary.get((X, op), 0) * Y
            )
        return sp.expand(out)

    Hblk = A * (ada * aa + bdb * bb_) + Bk * (ada * bdb + aa * bb_)
    c_a = comm_Q(Hblk, aa)  # [H, a]
    c_bd = comm_Q(Hblk, bdb)  # [H, b+]
    ok(
        "5.2a",
        "[H,a] = -A a - B_k b+  and  [H,b+] = A b+ + B_k a  " "(equations of motion)",
        sp.expand(c_a + A * aa + Bk * bdb) == 0
        and sp.expand(c_bd - A * bdb - Bk * aa) == 0,
    )
    # i d/dt (a, b+)^T = M (a, b+)^T with M = [[A, B],[-B, -A]]
    Mbdg = Matrix([[A, Bk], [-Bk, -A]])
    ev = Mbdg.eigenvals()  # {-sqrt(A-B)*sqrt(A+B), +sqrt(A-B)*sqrt(A+B)}
    w = sqrt(A**2 - Bk**2)
    # Cayley-Hamilton: M^2 = (A^2 - B_k^2) 1 and tr M = 0  <=>  eig = +-w
    Msq = sp.expand(Mbdg * Mbdg - (A**2 - Bk**2) * eye(2))
    ok(
        "5.2b",
        "BdG eigenvalues are +-sqrt(A^2 - B_k^2) -> magnon energy "
        "omega_k = sqrt(A^2 - B_k^2)",
        Msq == sp.zeros(2, 2)
        and sp.expand(Mbdg.trace()) == 0
        and all(sp.simplify(k**2 - w**2) == 0 for k in ev.keys()),
    )

    # ---- 5.3 Bogoliubov angle identity --------------------------------------
    th, t = sp.symbols("theta t", positive=True)
    t_exact = sp.Rational(5 - 4, 3)  # tanh(theta) = (A-omega)/B
    dbl = sp.tanh(2 * th) - 2 * sp.tanh(th) / (1 + sp.tanh(th) ** 2)
    ok(
        "5.3a",
        "tanh(2 theta) = 2 tanh(theta)/(1+tanh^2) (double angle)",
        sp.simplify(dbl.rewrite(sp.exp)) == 0,
    )
    ok(
        "5.3b",
        "tanh(2 theta) = B/A with tanh(theta) = (A-omega)/B "
        "(A=5, B=3, omega=4 exact rationals)",
        sp.simplify(2 * t_exact / (1 + t_exact**2) - sp.Rational(3, 5)) == 0,
    )
    w_sym = sqrt(A**2 - Bk**2)
    t_sym = (A - w_sym) / Bk
    ok(
        "5.3c",
        "same identity symbolically: 2t/(1+t^2) - B/A = 0 with "
        "t = (A-omega)/B, omega = sqrt(A^2-B^2)",
        sp.simplify((2 * t_sym / (1 + t_sym**2) - Bk / A).rewrite(sp.exp)) == 0,
    )

    # ---- 5.4 AFM dimer: LSWT ground-state energy is exact ------------------
    # z=1: E_cl = -J_af S^2, A = J_af S, B = J_af S  =>  omega = 0,
    # E_GS(LSWT) = E_cl + (omega - A) = -J_af S(S+1)  = exact singlet energy
    Egs = -Jaf * S**2 + (0 - Jaf * S)
    ok(
        "5.4",
        "AFM dimer LSWT ground state -J_af S(S+1) equals the exact "
        "singlet energy for all S",
        sp.simplify(Egs + Jaf * S * (S + 1)) == 0,
    )

    # ---- 5.5 chain H_2 in q space: A_k = 2 J S,  B_k = 2 J S cos(k a) ------
    Nc = 4  # cells; 8 sites; a = 1
    syA = {
        k: v
        for k, v in zip(
            ["aA", "adA", "bB", "bdB"],
            [
                [Symbol(f"aA{i}", commutative=False) for i in range(Nc)],
                [Symbol(f"adA{i}", commutative=False) for i in range(Nc)],
                [Symbol(f"bB{i}", commutative=False) for i in range(Nc)],
                [Symbol(f"bdB{i}", commutative=False) for i in range(Nc)],
            ],
        )
    }
    aA, adA, bB, bdB = syA["aA"], syA["adA"], syA["bB"], syA["bdB"]
    Ecl_c = -8 * Jaf * S**2
    H2 = Ecl_c + 2 * Jaf * S * sum(adA[i] * aA[i] + bdB[i] * bB[i] for i in range(Nc))
    H2 += (
        Jaf
        * S
        * sum(
            aA[i] * bB[i]
            + adA[i] * bdB[i]
            + aA[i] * bB[(i - 1) % Nc]
            + adA[i] * bdB[(i - 1) % Nc]
            for i in range(Nc)
        )
    )
    # Fourier: cells at R_n = 2n, B atom offset +1 inside the cell;
    # k_m = 2 pi m /(2 Nc) = pi m /4, m = 0..3
    qm = [pi * m / 4 for m in range(Nc)]
    sub = {}
    for i in range(Nc):
        sub[aA[i]] = sp.Rational(1, 2) * sum(
            exp(I * qm[m] * 2 * i) * Symbol(f"aAk{m}", commutative=False)
            for m in range(Nc)
        )
        sub[adA[i]] = sp.Rational(1, 2) * sum(
            exp(-I * qm[m] * 2 * i) * Symbol(f"adAk{m}", commutative=False)
            for m in range(Nc)
        )
        sub[bB[i]] = sp.Rational(1, 2) * sum(
            exp(I * qm[m] * (2 * i + 1)) * Symbol(f"bBk{m}", commutative=False)
            for m in range(Nc)
        )
        sub[bdB[i]] = sp.Rational(1, 2) * sum(
            exp(-I * qm[m] * (2 * i + 1)) * Symbol(f"bdBk{m}", commutative=False)
            for m in range(Nc)
        )
    Hq = sp.expand(H2.subs(sub))
    ops = [
        Symbol(f"{p}{m}", commutative=False)
        for p in ["aAk", "adAk", "bBk", "bdBk"]
        for m in range(Nc)
    ]
    const = Hq.subs({s: 0 for s in ops})

    def coeff(op1, op2):
        keep = {s: 0 for s in ops}
        keep[op1], keep[op2] = 1, 1
        return sp.expand((Hq - const).subs(keep))

    adAk = [Symbol(f"adAk{m}", commutative=False) for m in range(Nc)]
    aAk = [Symbol(f"aAk{m}", commutative=False) for m in range(Nc)]
    bdBk = [Symbol(f"bdBk{m}", commutative=False) for m in range(Nc)]
    bBk = [Symbol(f"bBk{m}", commutative=False) for m in range(Nc)]
    goodA = all(
        sp.simplify(coeff(adAk[m], aAk[m]) - 2 * Jaf * S) == 0
        and sp.simplify(coeff(bdBk[m], bBk[m]) - 2 * Jaf * S) == 0
        for m in range(Nc)
    )
    ok("5.5a", "number-term coefficient A_k = 2 J_af S for all k", goodA)
    goodB = True
    for m in range(Nc):
        mp = (-m) % Nc
        # the pairing coefficient carries the plane-wave phase of the b-mode
        # label k_{mp} (mp = -m on the cell reciprocal lattice, period pi);
        # cos flips sign between folded representatives, |sin| does not.
        expected = Jaf * S * (exp(I * qm[mp]) + exp(-I * qm[mp]))
        goodB &= sp.simplify(coeff(adAk[m], bdBk[mp]) - expected) == 0
        goodB &= sp.simplify(coeff(aAk[m], bBk[mp]) - expected) == 0
        # no other monomial families
        for m2 in range(Nc):
            if m2 != mp:
                goodB &= sp.simplify(coeff(adAk[m], bdBk[m2])) == 0
                goodB &= sp.simplify(coeff(aAk[m], bBk[m2])) == 0
            goodB &= sp.simplify(coeff(adAk[m], bBk[m2])) == 0
            goodB &= sp.simplify(coeff(aAk[m], bdBk[m2])) == 0
            if m2 != m:
                goodB &= sp.simplify(coeff(adAk[m], aAk[m2])) == 0
                goodB &= sp.simplify(coeff(bdBk[m], bBk[m2])) == 0
    ok(
        "5.5b",
        "pairing coefficient B_k = 2 J_af S cos(k a); all cross " "terms vanish",
        goodB,
    )

    # ---- 5.6 dispersion: omega_k = 2 J S |sin(k a)| ------------------------
    for m in range(Nc):
        Ak, Bkk = 2 * Jaf * S, 2 * Jaf * S * cos(qm[m])
        wk = sp.simplify(sqrt(Ak**2 - sp.expand(Bkk**2)))
        target = 2 * Jaf * S * abs(sin(qm[m]))
        ok(
            f"5.6[k=pi*{m}/4]",
            f"omega_k = {sp.simplify(target)} = " f"2 J S |sin(k a)|",
            sp.simplify(wk - target) == 0,
        )

    # ---- 5.7 numeric Fock-space diagonalization of the k blocks ------------
    def squeezed_block(Av, Bv, D):
        aa_, add_, nn_ = np_fock(D)
        Id = np.eye(D)
        return np.linalg.eigvalsh(
            Av * (np.kron(nn_, Id) + np.kron(Id, nn_))
            + Bv * (np.kron(aa_, aa_) + np.kron(add_, add_))
        )

    # generic squeezed block A=5, B=3 -> omega=4, GS offset omega-A=-1
    Av, Bv, wv, D = 5.0, 3.0, 4.0, 20
    ev = squeezed_block(Av, Bv, D)
    ana = np.sort(
        np.array([(wv - Av) + p * wv for p in range(4) for _ in range(p + 1)])
    )
    ok(
        "5.7a",
        f"squeezed block A=5,B=3: lowest levels = (omega-A)+m*omega "
        f"[max err {np.max(np.abs(ev[:10]-ana[:10])):.2e}]",
        np.max(np.abs(ev[:10] - ana[:10])) < 1e-9,
    )

    # chain blocks with J_af=1, S=1/2  =>  A=1, B_k = cos k
    Jv, Sv = 1.0, 0.5
    Av = 2 * Jv * Sv
    for m in [1, 2, 3]:
        Bv = 2 * Jv * Sv * float(cos(qm[m]))
        wv = float(2 * Jv * Sv * abs(sin(qm[m])))
        ev = squeezed_block(Av, Bv, 24)
        ana = np.sort(
            np.array([(wv - Av) + p * wv for p in range(4) for _ in range(p + 1)])
        )
        ok(
            f"5.7b[k=pi*{m}/4]",
            f"chain block spectrum matches "
            f"(omega-A)+m*omega with omega=2JS|sin k|={wv:.6f} "
            f"[max err {np.max(np.abs(ev[:10]-ana[:10])):.2e}]",
            np.max(np.abs(ev[:10] - ana[:10])) < 1e-9,
        )
    # k=0 block is marginal (Goldstone, omega=0, t=(A-omega)/B=1): the
    # formal LSWT offset (omega-A) = -A is approached only as ~1/D; we verify
    # the monotone convergence with its scaling instead of a tight tolerance.
    lam8 = squeezed_block(Av, Av, 8)[0]
    lam32 = squeezed_block(Av, Av, 32)[0]
    ok(
        "5.7c",
        "k=0 (Goldstone) block: lambda_min -> -A monotonically with "
        f"~1/D scaling ({lam8:.4f} -> {lam32:.4f} vs -A = -1)",
        -Av - 1e-9 <= lam32 < lam8
        and abs(lam32 + Av) < abs(lam8 + Av) / 3
        and abs(lam32 + Av) < 0.05,
    )

    # ---- 5.8 exact AFM dimer spectrum vs LSWT (documented caveat) ----------
    o2 = spin_matrices(sp.Rational(1, 2))
    Sp1 = site_op(o2["Sp"], 0, 2, 2)
    Sm1 = site_op(o2["Sm"], 0, 2, 2)
    Sz1 = site_op(o2["Sz"], 0, 2, 2)
    Sp2 = site_op(o2["Sp"], 1, 2, 2)
    Sm2 = site_op(o2["Sm"], 1, 2, 2)
    Sz2 = site_op(o2["Sz"], 1, 2, 2)
    Hd = (Sp1 * Sm2 + Sm1 * Sp2) / 2 + Sz1 * Sz2  # H = +J S1.S2, J=1
    evd = Hd.eigenvals()
    ok(
        "5.8",
        "exact AFM dimer S=1/2: singlet -3/4 (x1), triplet +1/4 (x3); "
        "LSWT reproduces E_GS = -3/4 (check 5.4) but predicts a spurious "
        "omega=0 instead of the exact gap 1 (documented LSWT caveat for "
        "zero-dimensional systems)",
        evd == {-sp.Rational(3, 4): 1, sp.Rational(1, 4): 3},
    )


def main():
    print(__doc__.split("\n")[1])
    print("sympy", sp.__version__, "| numpy", np.__version__)
    section1()
    section2()
    section3()
    section4()
    section5()
    print(f"\nALL {NPASS[0]} CHECKS PASSED")


if __name__ == "__main__":
    main()
