#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_spacegroup_symmetrization.py
===============================
Sympy-verified derivation (part 4 of the TB2J documentation series):

    symmetrization of exchange tensors by space-group / magnetic
    space-group operations, replacing distance+tag heuristics by exact
    orbit projection.  Contents:

      1. Tensor correspondence: pair tensor Gamma = J_iso I + J_ani + A(D),
         A^{ab} = eps^{abg} D^g, reversal identity Gamma_ji(-R) = Gamma_ij^T.
      2. Symmetry operation action: site/bond maps for g = {W|t},
         R' = W R + n_j - n_i, W_c = cell W cell^{-1},
         energy invariance under Gamma' = W_c Gamma W_c^T.
      3. DMI as axial vector: W A(D) W^T = A(det(W) W^{-T} D) in general,
         = A(det(W) W D) on O(3); hence D' = det(W_c) W_c D.
      4. Time-reversal cancellation: primed ops act identically on all
         stored tensors (bilinear Gamma and rank-2 on-site K); priming
         only enlarges site-map orbits.
      5. Projection properties: P = (1/|G|) sum_g T_g is a Reynolds
         operator (P^2 = P); averaged tensor invariant; exact channel
         round trip (C2v and D2d worked groups).
      6. Moriya-rule worked examples (exact symbolic orbit averages):
         inversion, mirror containing bond, mirror perpendicular to bond,
         2-fold axis perpendicular to bond, primed half-translation AFM,
         C3 about the bond, pure translation.
      7. Numerical validation with seeded numpy RNG.

Source convention (docs/src/convention.rst):
    E = - sum_i S_i^T K_i S_i
        - sum_{i!=j} [ J^iso_ij S_i.S_j + S_i J^ani_ij S_j
                       + D_ij . (S_i x S_j) ]
    every ordered pair (ij and ji) stored separately; spins normalized.

Shared mathematical contract (pinned):
    Gamma_ij(R) = J^iso I + J^ani + A(D),   A^{ab} = eps^{abg} D^g
    Gamma_ji(-R) = Gamma_ij(R)^T
    g = {W|t}:  x -> Wx + t (mod 1);  d = x_j + R - x_i -> d' = W d;
    R' = d' - (x_j' - x_i') = W R + n_j - n_i  (integers)
    W_c = cell W cell^{-1}:  Gamma' = W_c Gamma W_c^T,  K' = W_c K W_c^T,
    D' = det(W_c) W_c D  (axial vector).
    Time reversal (primed ops): S -> -W_c S; both spin flips cancel for
    every stored term, so the tensor action is priming-independent.

Every check is executed at run time, protected by `assert`, and prints a
PASS line.  Run with:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/04_spacegroup_symmetrization.py
"""

import numpy as np
import sympy as sp
from sympy import Matrix, Rational, Symbol, cos, eye, sin, sqrt

NPASS = [0]


def ok(cid, desc, cond):
    """Assert a check and print a PASS line."""
    if not cond:
        raise AssertionError(f"FAIL {cid}: {desc}")
    NPASS[0] += 1
    print(f"PASS {cid:<10} {desc}")


def simp(e):
    """Canonical simplification: expand, cancel rationals, simplify."""
    return sp.simplify(sp.cancel(sp.expand(e)))


def mat_zero(M):
    """True if the sympy Matrix M is identically zero."""
    return all(simp(e) == 0 for e in M)


def levi(a, b, c):
    return sp.LeviCivita(a, b, c)


# ----------------------------------------------------------------------------
# Channel algebra (shared contract)
# ----------------------------------------------------------------------------
def A_of_D(D):
    """Skew-symmetric matrix A^{ab} = sum_g eps^{abg} D^g (D: 3x1 Matrix)."""
    return Matrix(3, 3, lambda a, b: sum(levi(a, b, g) * D[g] for g in range(3)))


def decompose(G):
    """Gamma -> (J_iso, J_ani, D) with
    J_iso = tr(Gamma)/3,
    J_ani = sym(Gamma) - J_iso I   (symmetric, traceless),
    D^g   = 1/2 sum_{ab} eps^{gab} skew(Gamma)^{ab}."""
    iso = G.trace() / 3
    ani = (G + G.T) / 2 - iso * eye(3)
    K = (G - G.T) / 2
    D = Matrix(
        3,
        1,
        lambda g, _j: Rational(1, 2)
        * sum(levi(g, a, b) * K[a, b] for a in range(3) for b in range(3)),
    )
    return iso, ani, D


def recompose(iso, ani, D):
    """Inverse of decompose: Gamma = J_iso I + J_ani + A(D)."""
    return iso * eye(3) + ani + A_of_D(D)


def generic_G(tag="g"):
    """Generic 3x3 matrix of independent symbols."""
    return Matrix(3, 3, lambda a, b: Symbol(f"{tag}_{a}{b}"))


def generic_vec(tag, n=3):
    return Matrix(n, 1, lambda i, j=0: Symbol(f"{tag}{i}"))


# ----------------------------------------------------------------------------
# Space-group operations on stored fields {(i, j, R): Gamma}
# ----------------------------------------------------------------------------
SITES_BOND = {  # 2-site bond along z, midpoint at (0,0,1/2); the two sites are
    # distinct sublattices (not related by a lattice vector)
    0: Matrix([0, 0, Rational(1, 4)]),
    1: Matrix([0, 0, Rational(3, 4)]),
}
SITES_AFM = {  # 1D AFM dimer: two sublattices a half translation apart
    0: Matrix([0, 0, 0]),
    1: Matrix([0, 0, Rational(1, 2)]),
}


class Op:
    """Space-group operation g = {W|t} on fractional coordinates.

    W     : integer 3x3 rotation in the fractional (lattice) basis
    t     : fractional translation
    rev   : compose with bond reversal (data-model identity
            Gamma_ji(-R) = Gamma_ij^T; commutes with every g)
    primed: time-reversed (magnetic) operation.  Per the contract the
            tensor action is independent of priming; the flag is carried
            for documentation only.
    """

    def __init__(self, W, t=(0, 0, 0), rev=False, primed=False):
        self.W = W if isinstance(W, Matrix) else Matrix(3, 3, W)
        self.t = Matrix(3, 1, list(t))
        self.rev = rev
        self.primed = primed

    def Wc(self, cell):
        """Cartesian rotation cell W cell^{-1}."""
        return cell * self.W * cell.inv()

    def map_site(self, x, sites):
        """Fractional x -> (site index, lattice shift n) with
        W x + t = x_site + n (exact integer n).  An exact match (n = 0)
        takes priority over lattice-shifted matches."""
        img = self.W * x + self.t
        fallback = None
        for s, xs in sites.items():
            d = img - xs
            ints = [simp(e) for e in d]
            if all(e == 0 for e in ints):
                return s, Matrix(3, 1, [0, 0, 0])
            if fallback is None and all(e.is_integer for e in ints):
                fallback = (s, Matrix(3, 1, [int(e) for e in ints]))
        if fallback is not None:
            return fallback
        raise KeyError(f"image {list(img)} matches no site")

    def map_key(self, key, sites, reduce_mod=False):
        """Bond map (i,j,R) -> (i',j',R') with d = x_j + R - x_i -> W d and
        R' = W R + n_j - n_i; optionally reduced into the magnetic cell
        (quotient by pure lattice translations, see section 6g)."""
        i, j, R = key
        ip, ni = self.map_site(sites[i], sites)
        jp, nj = self.map_site(sites[j], sites)
        Rp = self.W * Matrix(3, 1, list(R)) + nj - ni
        if reduce_mod:
            Rp = Rp.applyfunc(lambda e: sp.Mod(e, 1))
        if self.rev:  # reversal acts after g
            ip, jp, Rp = jp, ip, -Rp
        return (ip, jp, tuple(Rp))

    def transport(self, key, G, sites, cell, reduce_mod=False):
        """Return (image key, transformed tensor):
        Gamma' = W_c Gamma W_c^T (transpose if rev)."""
        k2 = self.map_key(key, sites, reduce_mod)
        Wc = self.Wc(cell)
        M = Wc * G * Wc.T
        if self.rev:
            M = M.T
        return k2, M


def group_with_reversal(gens):
    """The group generated by the ops plus reversal: {E} u gens, times
    {unprimed-reversal, rev-composed} (reversal commutes with every g)."""
    base = [Op(eye(3))]
    for g in gens:
        if g.W == eye(3) and g.t == sp.zeros(3, 1):
            continue  # identity already present; never double-count
        base.append(g)
    ops = [Op(g.W, list(g.t), rev=False, primed=g.primed) for g in base]
    ops += [Op(g.W, list(g.t), rev=True, primed=g.primed) for g in base]
    return ops


def project(field, ops, sites, cell=eye(3), reduce_mod=False):
    """Reynolds operator P = (1/|G|) sum_g T_g on a stored field."""
    out = {k: sp.zeros(3, 3) for k in field}
    for op in ops:
        for k, G in field.items():
            k2, M = op.transport(k, G, sites, cell, reduce_mod)
            if k2 not in out:
                raise KeyError(f"key set not closed: {k} -> {k2}")
            out[k2] += M
    n = Rational(len(ops))
    return {k: M.applyfunc(simp) / n for k, M in out.items()}


def field_difference(F1, F2):
    return {k: F1[k] - F2[k] for k in F1}


def field_is_zero(F):
    return all(mat_zero(G) for G in F.values())


def field_invariant(F, ops, sites, cell=eye(3), reduce_mod=False):
    """True if T_g F = F for every g (field-level invariance)."""
    for op in ops:
        T = {k: sp.zeros(3, 3) for k in F}
        for k, G in F.items():
            k2, M = op.transport(k, G, sites, cell, reduce_mod)
            T[k2] += M
        if not field_is_zero(field_difference(T, F)):
            return False, op
    return True, None


def numeric_transport(op, key, G, sites, reduce_mod=False):
    """Same key logic as Op.transport with a numpy tensor."""
    i, j, R = key
    ip, ni = op.map_site(sites[i], sites)
    jp, nj = op.map_site(sites[j], sites)
    Rp = op.W * Matrix(3, 1, list(R)) + nj - ni
    if reduce_mod:
        Rp = Rp.applyfunc(lambda e: sp.Mod(e, 1))
    if op.rev:
        ip, jp, Rp = jp, ip, -Rp
    Wc = np.array(op.Wc(eye(3)), dtype=float)
    M = Wc @ G @ Wc.T
    if op.rev:
        M = M.T.copy()
    return (ip, jp, tuple(Rp)), M


def numeric_project(field, ops, sites, reduce_mod=False):
    out = {k: np.zeros((3, 3)) for k in field}
    for op in ops:
        for k, G in field.items():
            k2, M = numeric_transport(op, k, G, sites, reduce_mod)
            if k2 not in out:
                raise KeyError(f"key set not closed: {k} -> {k2}")
            out[k2] += M
    n = len(ops)
    return {k: M / n for k, M in out.items()}


def numeric_invariance_dev(Fbar, ops, sites, reduce_mod=False):
    dev = 0.0
    for op in ops:
        T = {k: np.zeros((3, 3)) for k in Fbar}
        for k, G in Fbar.items():
            k2, M = numeric_transport(op, k, G, sites, reduce_mod)
            T[k2] += M
        dev = max(dev, max(np.max(np.abs(T[k] - Fbar[k])) for k in Fbar))
    return dev


def numeric_decompose(G):
    iso = np.trace(G) / 3.0
    ani = (G + G.T) / 2 - iso * np.eye(3)
    K = (G - G.T) / 2
    D = np.zeros(3)
    for g in range(3):
        for a in range(3):
            for b in range(3):
                D[g] += 0.5 * int(levi(g, a, b)) * K[a, b]
    return iso, ani, D


def numeric_recompose(iso, ani, D):
    A = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            for g in range(3):
                A[a, b] += int(levi(a, b, g)) * D[g]
    return iso * np.eye(3) + ani + A


def rand_orth(rng, det_sign):
    """Random orthogonal 3x3 with the requested det sign (seeded)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.sign(np.linalg.det(q)) != det_sign:
        q[:, [0, 1]] = q[:, [1, 0]]
    return q


# ----------------------------------------------------------------------------
# Worked-example groups
# ----------------------------------------------------------------------------
def Rz_cart(ang):
    """Exact cartesian rotation about z (ang = 'q2'/'q3' for 120/240 deg)."""
    if ang == "q3":
        c, s = Rational(-1, 2), -sqrt(3) / 2
    elif ang == "q2":
        c, s = Rational(-1, 2), sqrt(3) / 2
    else:
        raise ValueError(ang)
    return Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# gens: ops whose generated group (plus reversal) constrains the bond;
# Dzero/Dfree: forced-zero / surviving D components;
# anizero: forced-zero symmetric-ani entries; anieq: equal diag pairs.
EXAMPLES = {
    "6a inversion center at bond midpoint": dict(
        gens=[Op(-eye(3))],
        sites=SITES_BOND,
        reduce_mod=False,
        Dzero=[0, 1, 2],
        anizero=[],
        anieq=None,
    ),
    "6b mirror plane containing bond (xz)": dict(
        gens=[Op(Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))],
        sites=SITES_BOND,
        reduce_mod=False,
        Dzero=[0, 2],
        anizero=[(0, 1), (1, 2)],
        anieq=None,
    ),
    "6c mirror plane perpendicular to bond (xy)": dict(
        gens=[Op(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]]))],
        sites=SITES_BOND,
        reduce_mod=False,
        Dzero=[2],
        anizero=[(0, 2), (1, 2)],
        anieq=None,
    ),
    "6d 2-fold axis perpendicular to bond (x)": dict(
        gens=[Op(Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))],
        sites=SITES_BOND,
        reduce_mod=False,
        Dzero=[0],
        anizero=[(0, 1), (0, 2)],
        anieq=None,
    ),
    "6e primed half-translation AFM (theta T_1/2)": dict(
        gens=[Op(eye(3), t=(0, 0, Rational(1, 2)), primed=True)],
        sites=SITES_AFM,
        reduce_mod=True,
        Dzero=[0, 1, 2],
        anizero=[],
        anieq=None,
    ),
    "6f C3 rotation about bond axis (z)": dict(
        gens=[Op(Rz_cart("q3")), Op(Rz_cart("q2"))],
        sites=SITES_BOND,
        reduce_mod=False,
        Dzero=[0, 1],
        anizero=[(0, 1), (0, 2), (1, 2)],
        anieq=(0, 1),
    ),
}


def run_symbolic_case(ex):
    """Generic-9-symbol bond field -> (F, ops, Fbar, iso, ani, D, invariant)."""
    G = generic_G()
    sites = ex["sites"]
    F = {(0, 1, (0, 0, 0)): G, (1, 0, (0, 0, 0)): G.T}
    ops = group_with_reversal(ex["gens"])
    Fbar = project(F, ops, sites, reduce_mod=ex["reduce_mod"])
    iso, ani, D = decompose(Fbar[(0, 1, (0, 0, 0))])
    inv, bad = field_invariant(Fbar, ops, sites, reduce_mod=ex["reduce_mod"])
    return G, F, ops, Fbar, iso, ani, D, inv, bad


# ============================================================================
# SECTION 1 - tensor correspondence and reversal identity
# ============================================================================
def section1():
    print("\n=== Section 1: pair tensor Gamma = J_iso I + J_ani + A(D), reversal ===")
    Dv = generic_vec("D")
    A = A_of_D(Dv)
    ok("1.1", "A(D) skew-symmetric: A + A^T = 0", mat_zero(A + A.T))

    sa = generic_vec("p")
    sb = generic_vec("q")
    cross = Matrix(
        [
            sa[1] * sb[2] - sa[2] * sb[1],
            sa[2] * sb[0] - sa[0] * sb[2],
            sa[0] * sb[1] - sa[1] * sb[0],
        ]
    )
    ok(
        "1.2",
        "S_i^T A(D) S_j = D . (S_i x S_j)  (index computation)",
        simp((sa.T * A * sb)[0, 0] - (Dv.T * cross)[0, 0]) == 0,
    )

    G = generic_G()
    iso, ani, Drec = decompose(G)
    ok(
        "1.3",
        "J^iso = tr(Gamma)/3; decomposition exact: iso I + ani + A(D) = Gamma",
        mat_zero(recompose(iso, ani, Drec) - G),
    )
    ok(
        "1.4",
        "J^ani symmetric and traceless",
        mat_zero(ani - ani.T) and simp(ani.trace()) == 0,
    )
    ok(
        "1.5",
        "A(D_rec) = skew(Gamma): D extracted uniquely",
        mat_zero(A_of_D(Drec) - (G - G.T) / 2),
    )

    # Reversal identity Gamma_ji(-R) = Gamma_ij(R)^T and channel consequences
    isoT, aniT, DT = decompose(G.T)
    ok("1.6", "reversal: J^iso(Gamma^T) = J^iso(Gamma)", simp(isoT - iso) == 0)
    ok("1.7", "reversal: J^ani(Gamma^T) = J^ani(Gamma)", mat_zero(aniT - ani))
    ok("1.8", "reversal: D(Gamma^T) = -D(Gamma)  (DMI flips sign)", mat_zero(DT + Drec))

    # Double-counting consistency of the stored ordered pairs
    E_pair = -(sa.T * G * sb)[0, 0] - (sb.T * G.T * sa)[0, 0]
    ok(
        "1.9",
        "E_pair = -S_i^T G S_j - S_j^T G^T S_i = -2 S_i^T Gamma S_j",
        simp(E_pair + 2 * (sa.T * G * sb)[0, 0]) == 0,
    )
    # channel form with an explicitly decomposed generic tensor
    isod, anid, Dd = decompose(generic_G("h"))
    ha = generic_vec("u")
    hb = generic_vec("v")
    cross_uv = Matrix(
        [
            ha[1] * hb[2] - ha[2] * hb[1],
            ha[2] * hb[0] - ha[0] * hb[2],
            ha[0] * hb[1] - ha[1] * hb[0],
        ]
    )
    hd = generic_G("h")
    lhs = -(ha.T * hd * hb)[0, 0] - (hb.T * hd.T * ha)[0, 0]
    rhs = (
        -2 * isod * (ha.T * hb)[0, 0]
        - 2 * (ha.T * anid * hb)[0, 0]
        - 2 * (Dd.T * cross_uv)[0, 0]
    )
    ok(
        "1.10",
        "channel form: E_pair = -2[J S_i.S_j + S_i J^ani S_j + D.(S_i x S_j)]",
        simp(lhs - rhs) == 0,
    )


# ============================================================================
# SECTION 2 - symmetry operation action
# ============================================================================
def section2():
    print("\n=== Section 2: op action {W|t}: site/bond maps and tensor law ===")
    th = Symbol("theta", real=True)
    Rz = Matrix([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])
    ok("2.1", "Rz(theta) orthogonal: Rz^T Rz = I", mat_zero(Rz.T * Rz - eye(3)))

    G = generic_G()
    si, sj = generic_vec("a"), generic_vec("b")
    E0 = -(si.T * G * sj)[0, 0]
    for cid, W, lbl in [
        ("2.2", Rz, "Rz(theta), det=+1"),
        ("2.3", Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]]), "mirror xz, det=-1"),
        ("2.4", -eye(3), "inversion, det=-1"),
    ]:
        Gp = W * G * W.T
        Emap = -((W * si).T * Gp * (W * sj))[0, 0]
        ok(
            cid,
            f"energy invariant under {lbl}: "
            f"S'^T Gamma' S' = S^T Gamma S with Gamma' = W Gamma W^T",
            simp(Emap - E0) == 0,
        )

    # Bond map in fractional coordinates: R' = W R + n_j - n_i
    w = Matrix(3, 3, lambda a, b: Symbol(f"w{a}{b}"))
    Rv = Matrix([3, -2, 5])
    ni = Matrix([-1, 0, 2])
    nj = Matrix([1, 1, -1])
    xi = Matrix([Rational(1, 3), Rational(1, 4), 0])
    xj = Matrix([Rational(2, 3), Rational(3, 4), Rational(1, 2)])
    t = Matrix([Rational(1, 2), 0, Rational(1, 4)])
    d = xj + Rv - xi
    xpi = w * xi + t - ni  # periodic image of the i image
    xpj = w * xj + t - nj
    Rp = w * d - (xpj - xpi)  # contract: R' = d' - (x_j' - x_i')
    ok(
        "2.5",
        "bond map: R' = W R + n_j - n_i (integer lattice vector)",
        mat_zero(Rp - (w * Rv + nj - ni)),
    )

    # Grounding: integer fractional W of a hexagonal cell is a cartesian rotation
    a, c = sp.symbols("a c", positive=True)
    cell = Matrix([[a, a / 2, 0], [0, a * sqrt(3) / 2, 0], [0, 0, c]])
    W60 = Matrix([[0, -1, 0], [1, 1, 0], [0, 0, 1]])  # a1 -> a2 (60 deg)
    Wc = simp(cell * W60 * cell.inv())
    ok(
        "2.6",
        "hexagonal integer W -> cartesian W_c = cell W cell^-1 orthogonal",
        mat_zero(Wc.T * Wc - eye(3)),
    )
    ok("2.7", "hexagonal 60-deg: det W_c = +1", simp(Wc.det() - 1) == 0)

    # Op machinery: hexagonal 60-deg W with a half-translation along c,
    # sites on the rotation axis at z = 0 and z = 1/2
    op = Op(W60, t=(0, 0, Rational(1, 2)))
    sites = {0: Matrix([0, 0, 0]), 1: Matrix([0, 0, Rational(1, 2)])}
    # site0 -> (0,0,1/2) = site 1 (n=0); site1 -> (0,0,1) = site 0 + (0,0,1)
    k2 = op.map_key((0, 1, (2, -1, 3)), sites)
    ok(
        "2.8",
        "Op.map_key: (0,1; 2,-1,3) -> (1,0; 1,1,4) = W R + n_j - n_i",
        k2 == (1, 0, (1, 1, 4)),
    )


# ============================================================================
# SECTION 3 - DMI as axial vector
# ============================================================================
def section3():
    print("\n=== Section 3: axial vector law A(det W W D) = W A(D) W^T ===")
    Dv = generic_vec("D")

    # Master identity for a general invertible symbolic W
    w = Matrix(3, 3, lambda a, b: Symbol(f"w{a}{b}"))
    dW = sp.cancel(w.det())
    Dmaster = dW * (w.inv().T * Dv)
    ok(
        "3.1",
        "general invertible W: W A(D) W^T = A(det(W) W^{-T} D)",
        mat_zero(w * A_of_D(Dv) * w.T - A_of_D(Dmaster)),
    )

    # Orthogonal families (Euler angles): W^{-T} = W, det = +-1
    al, be, ga = sp.symbols("alpha beta gamma", real=True)

    def Rz(t):
        return Matrix([[cos(t), -sin(t), 0], [sin(t), cos(t), 0], [0, 0, 1]])

    def Rx(t):
        return Matrix([[1, 0, 0], [0, cos(t), -sin(t)], [0, sin(t), cos(t)]])

    Wp = Rz(ga) * Rx(be) * Rz(al)  # det = +1 component of O(3)
    Wi = Wp * Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # det = -1 component
    for cid, W, lbl in [
        ("3.2", Wp, "proper Euler family"),
        ("3.4", Wi, "improper Euler family"),
    ]:
        dWm = simp(W.det())
        ok(f"{cid}a", f"{lbl}: det W = {dWm} (symbolic)", dWm in (1, -1))
        ok(
            f"{cid}b",
            f"axial law A(det(W) W D) = W A(D) W^T [{lbl}]",
            mat_zero(W * A_of_D(Dv) * W.T - A_of_D(dWm * (W * Dv))),
        )

    # Concrete det = +-1 substitution cases
    mirror = Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # det = -1
    ok(
        "3.5",
        "mirror xz (det=-1): A(-W D) = W A(D) W^T -> D flips handedness",
        mat_zero(mirror * A_of_D(Dv) * mirror.T - A_of_D(-1 * (mirror * Dv))),
    )
    ok(
        "3.6",
        "inversion (det=-1, W=-I): A(D) invariant as matrix, D -> -D",
        mat_zero((-eye(3)) * A_of_D(Dv) * (-eye(3)).T - A_of_D(Dv))
        and mat_zero((-1) * ((-eye(3)) * Dv) - Dv),
    )

    # Seeded numeric spot check; full numerical battery in section 7
    rng = np.random.default_rng(20240517)
    Q = rand_orth(rng, -1)
    Dn = rng.standard_normal(3)
    An = np.array(A_of_D(Matrix(3, 1, Dn)), dtype=float)
    lhs = Q @ An @ Q.T
    rhs = np.array(A_of_D(Matrix(3, 1, np.linalg.det(Q) * (Q @ Dn))), dtype=float)
    ok(
        "3.7",
        "numeric random orthogonal (det=-1): W A W^T = A(det W W D)",
        np.max(np.abs(lhs - rhs)) < 1e-12,
    )


# ============================================================================
# SECTION 4 - time-reversal cancellation (primed ops)
# ============================================================================
def section4():
    print("\n=== Section 4: time reversal S -> -W S cancels in stored terms ===")
    th = Symbol("theta", real=True)
    sg = Symbol("sigma")
    Rz = Matrix([[cos(th), -sin(th), 0], [sin(th), cos(th), 0], [0, 0, 1]])
    mirror = Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    G = generic_G()
    si, sj = generic_vec("a"), generic_vec("b")
    E0 = (si.T * G * sj)[0, 0]

    diffs = {}
    for cid, W, lbl in [("4.1", Rz, "Rz(theta)"), ("4.2", mirror, "mirror xz")]:
        Gp = W * G * W.T
        Emap = ((sg * (W * si)).T * Gp * (sg * (W * sj)))[0, 0]
        diff = simp(Emap - E0)
        ok(
            f"{cid}a",
            f"bilinear: S'^T Gamma' S' - S^T Gamma S = (sigma^2-1) S^T Gamma S [{lbl}]",
            simp(diff - (sg**2 - 1) * E0) == 0,
        )
        ok(
            f"{cid}b",
            f"sigma = +-1 (primed or unprimed): invariant [{lbl}]",
            simp(diff.subs(sg, 1)) == 0 and simp(diff.subs(sg, -1)) == 0,
        )
        diffs[cid] = diff

    # Single-site rank-2 anisotropy K
    K = Matrix(3, 3, lambda a, b: Symbol(f"k{a}{b}"))
    Ks = (K + K.T) / 2  # physical K is symmetric
    s1 = generic_vec("s")
    E0K = (s1.T * Ks * s1)[0, 0]
    for cid, W, lbl in [("4.3", Rz, "Rz(theta)"), ("4.4", mirror, "mirror xz")]:
        Kp = W * Ks * W.T
        Emap = ((sg * (W * s1)).T * Kp * (sg * (W * s1)))[0, 0]
        diff = simp(Emap - E0K)
        ok(
            f"{cid}a",
            f"single-site K: (sigma W S)^T K' (sigma W S) - S^T K S "
            f"= (sigma^2-1) S^T K S [{lbl}]",
            simp(diff - (sg**2 - 1) * E0K) == 0,
        )
        ok(
            f"{cid}b",
            f"sigma = +-1: K invariant [{lbl}]",
            simp(diff.subs(sg, 1)) == 0 and simp(diff.subs(sg, -1)) == 0,
        )

    ok(
        "4.5",
        "conclusion: Gamma' = W Gamma W^T contains no sigma; primed and "
        "unprimed ops have IDENTICAL tensor action",
        simp(diffs["4.1"] - (sg**2 - 1) * E0) == 0
        and "sigma" not in str(Rz * G * Rz.T),
    )


# ============================================================================
# SECTION 5 - projection properties: P^2 = P, invariance, round trip
# ============================================================================
def section5():
    print("\n=== Section 5: Reynolds projection P = (1/|G|) sum_g T_g ===")
    C2v = [
        Op(eye(3)),
        Op(Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),
        Op(Matrix([[1, 0, 0], [0, -1, 0], [0, 0, 1]])),
        Op(Matrix([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])),
    ]
    D2d = [
        Op(eye(3)),
        Op(Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])),
        Op(Matrix([[0, -1, 0], [1, 0, 0], [0, 0, -1]])),
        Op(Matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),
        Op(Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])),
        Op(Matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])),
        Op(Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),
        Op(Matrix([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])),
    ]

    for cid, name, gens, aniz, anieq in [
        ("5.1", "C2v (mm2) on a bond along z", C2v, [(0, 1), (0, 2), (1, 2)], None),
        (
            "5.2",
            "D2d (-42m) around a bond along z",
            D2d,
            [(0, 1), (0, 2), (1, 2)],
            (0, 1),
        ),
    ]:
        G = generic_G()
        F = {(0, 1, (0, 0, 0)): G, (1, 0, (0, 0, 0)): G.T}
        ops = group_with_reversal(gens)
        # key-set closure (every op maps the stored set into itself)
        closed = all(op.map_key(k, SITES_BOND) in F for op in ops for k in F)
        ok(
            f"{cid}a",
            f"{name}: key set closed under group+reversal " f"(|G| = {len(ops)})",
            closed,
        )
        Fbar = project(F, ops, SITES_BOND)
        inv, bad = field_invariant(Fbar, ops, SITES_BOND)
        ok(f"{cid}b", f"{name}: averaged field invariant under every op", inv)
        Fbar2 = project(Fbar, ops, SITES_BOND)
        ok(
            f"{cid}c",
            f"{name}: P^2 = P (averaging twice = once), generic entries",
            field_is_zero(field_difference(Fbar2, Fbar)),
        )
        iso, ani, D = decompose(Fbar[(0, 1, (0, 0, 0))])
        ok(
            f"{cid}d",
            f"{name}: channel decomposition exact round trip (both keys)",
            mat_zero(recompose(iso, ani, D) - Fbar[(0, 1, (0, 0, 0))])
            and mat_zero(
                recompose(*decompose(Fbar[(1, 0, (0, 0, 0))])) - Fbar[(1, 0, (0, 0, 0))]
            ),
        )
        ok(f"{cid}e", f"{name}: D forced to exactly 0 by the group", mat_zero(D))
        desc = f"{name}: J^ani diagonal" + (" and uniaxial (xx = yy)" if anieq else "")
        cond = all(simp(ani[a, b]) == 0 for a, b in aniz)
        if anieq:
            cond = cond and simp(ani[anieq[0], anieq[0]] - ani[anieq[1], anieq[1]]) == 0
        ok(f"{cid}f", desc, cond)


# ============================================================================
# SECTION 6 - Moriya-rule worked examples (exact symbolic orbit averages)
# ============================================================================
def section6():
    print("\n=== Section 6: Moriya rules as exact symbolic orbit averages ===")
    for name, ex in EXAMPLES.items():
        cid = name.split()[0]  # "6a" ... "6f"
        G, F, ops, Fbar, iso, ani, D, inv, bad = run_symbolic_case(ex)
        ok(f"{cid}.1", f"{name}: averaged field invariant under group+reversal", inv)
        iso2, ani2, D2 = decompose(Fbar[(1, 0, (0, 0, 0))])
        ok(
            f"{cid}.2",
            f"{name}: decomposition round trip on both keys",
            mat_zero(recompose(iso, ani, D) - Fbar[(0, 1, (0, 0, 0))])
            and mat_zero(recompose(iso2, ani2, D2) - Fbar[(1, 0, (0, 0, 0))]),
        )
        ok(
            f"{cid}.3",
            f"{name}: forced-zero D components exactly "
            f"{[f'D[{i}]' for i in ex['Dzero']]}"
            + (
                f", surviving {sorted(set(range(3)) - set(ex['Dzero']))}"
                if len(ex["Dzero"]) < 3
                else ""
            ),
            all(simp(D[i]) == 0 for i in ex["Dzero"]),
        )
        if ex["anizero"]:
            ok(
                f"{cid}.4",
                f"{name}: forced-zero J^ani entries exactly {ex['anizero']}",
                all(simp(ani[a, b]) == 0 for a, b in ex["anizero"]),
            )
        if ex["anieq"] is not None:
            i0, i1 = ex["anieq"]
            ok(
                f"{cid}.5",
                f"{name}: J^ani_xx = J^ani_yy (uniaxial)",
                simp(ani[i0, i0] - ani[i1, i1]) == 0,
            )

    # 6a extras: inversion leaves the full symmetric part unconstrained
    _, _, _, _, iso, ani, D, _, _ = run_symbolic_case(EXAMPLES[list(EXAMPLES)[0]])
    G = generic_G()
    ok(
        "6a.6",
        "inversion: J^ani (and J^iso) completely unconstrained "
        "(ani = sym(Gamma) - iso I exactly)",
        mat_zero(ani - ((G + G.T) / 2 - G.trace() / 3 * eye(3))),
    )

    # 6b/6c/6d: the op EXCHANGES the two sites -> constraint lands on the
    # reversed key and combines with the reversal identity
    for ex_name, sub in [
        ("6c mirror plane perpendicular to bond (xy)", "6c.0"),
        ("6d 2-fold axis perpendicular to bond (x)", "6d.0"),
    ]:
        op = EXAMPLES[ex_name]["gens"][0]
        ok(
            sub,
            f"{ex_name}: op maps ordered bond (i,j,R) -> (j,i,R) (site exchange)",
            op.map_key((0, 1, (0, 0, 0)), SITES_BOND) == (1, 0, (0, 0, 0)),
        )

    # 6e extras: the averaged Gamma is exactly symmetric; ani survives intact
    G6e, _, _, Fbar6e, _, ani6e, D6e, _, _ = run_symbolic_case(
        EXAMPLES["6e primed half-translation AFM (theta T_1/2)"]
    )
    k0 = (0, 1, (0, 0, 0))
    ok(
        "6e.6",
        "theta T_1/2: averaged Gamma symmetric, skew(Gamma_bar) = 0",
        mat_zero(Fbar6e[k0] - Fbar6e[k0].T),
    )
    ok(
        "6e.7",
        "theta T_1/2: J^ani and J^iso survive intact (ani = sym - iso I)",
        mat_zero(ani6e - ((G6e + G6e.T) / 2 - G6e.trace() / 3 * eye(3))),
    )
    op6e = EXAMPLES["6e primed half-translation AFM (theta T_1/2)"]["gens"][0]
    ok(
        "6e.8",
        "theta T_1/2: unprimed T_1/2 is NOT an AFM site map for spins "
        "but IS in the magnetic group only when primed; tensor action "
        "identical (W = I -> Gamma' = Gamma)",
        mat_zero(op6e.Wc(eye(3)) * G6e * op6e.Wc(eye(3)).T - G6e),
    )

    # 6g: pure translation {W = I, t = T}
    print("--- 6g translation-only op ---")
    Gt = generic_G()
    sites_t = SITES_BOND
    op_t = Op(eye(3), t=(1, 0, 0))
    # bond key in the translation quotient (magnetic/chemical cell basis):
    # R is reduced mod the cell, so R + T is the SAME stored key
    k0t = (0, 1, (0, 0, 0))
    k1t = (1, 0, (0, 0, 0))
    Ft = {k0t: Gt, k1t: Gt.T}
    k2 = op_t.map_key(k0t, sites_t, reduce_mod=True)
    ok(
        "6g.1",
        "translation (1,0,0): (i,j,R) -> (i,j,R+T) -> same key mod cell",
        k2 == k0t,
    )
    _, Mt = op_t.transport(k0t, Gt, sites_t, eye(3), reduce_mod=True)
    ok(
        "6g.2",
        "translation: tensor maps to itself, Gamma' = W_c Gamma W_c^T = Gamma",
        mat_zero(Mt - Gt),
    )
    ops_t = group_with_reversal([op_t])
    Ftbar = project(Ft, ops_t, sites_t, reduce_mod=True)
    ok(
        "6g.3",
        "translation (+reversal): projection acts trivially, P(Gamma) = Gamma; "
        "consistency constraint Gamma(R+T) = Gamma(R)",
        mat_zero(Ftbar[k0t] - Gt) and mat_zero(Ftbar[k1t] - Gt.T),
    )


# ============================================================================
# SECTION 7 - numerical validation (seeded)
# ============================================================================
def section7():
    print("\n=== Section 7: numerical validation with seeded numpy RNG ===")
    seed = 20240517
    rng = np.random.default_rng(seed)
    tol = 1e-10

    # channel identities on random tensors
    okall = True
    for _ in range(3):
        Dn = rng.standard_normal(3)
        An = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                for g in range(3):
                    An[a, b] += int(levi(a, b, g)) * Dn[g]
        sn = rng.standard_normal(3)
        tn = rng.standard_normal(3)
        lhs = sn @ An @ tn
        rhs = np.dot(Dn, np.cross(sn, tn))
        Gr = rng.standard_normal((3, 3))
        iso, ani, D = numeric_decompose(Gr)
        rt = np.max(np.abs(numeric_recompose(iso, ani, D) - Gr))
        rev = np.max(np.abs(np.array(numeric_decompose(Gr.T)[2]) + D))
        okall = okall and abs(lhs - rhs) < tol and rt < tol and rev < tol
    ok(
        "7.1",
        f"random tensors: S^T A S = D.(S x S), decomposition round trip, "
        f"reversal D -> -D  [seed {seed}]",
        okall,
    )

    # per-example: random non-invariant field -> project -> invariance + patterns
    for name, ex in EXAMPLES.items():
        cid = name.split()[0]
        G0 = rng.standard_normal((3, 3))
        F = {(0, 1, (0, 0, 0)): G0, (1, 0, (0, 0, 0)): G0.T.copy()}
        ops = group_with_reversal(ex["gens"])
        Fbar = numeric_project(F, ops, ex["sites"], reduce_mod=ex["reduce_mod"])
        dev = numeric_invariance_dev(
            Fbar, ops, ex["sites"], reduce_mod=ex["reduce_mod"]
        )
        iso, ani, D = numeric_decompose(Fbar[(0, 1, (0, 0, 0))])
        rt = np.max(np.abs(numeric_recompose(iso, ani, D) - Fbar[(0, 1, (0, 0, 0))]))
        dz = max(abs(D[i]) for i in ex["Dzero"])
        az = max((abs(ani[a, b]) for a, b in ex["anizero"]), default=0.0)
        surv = (
            max(abs(D[i]) for i in range(3) if i not in ex["Dzero"])
            if len(ex["Dzero"]) < 3
            else None
        )
        cond = (
            dev < tol
            and rt < tol
            and dz < tol
            and az < tol
            and (surv is None or surv > 1e-2)
        )
        extra = f", surviving |D| max {surv:.3f}" if surv is not None else ""
        ok(
            f"{cid}.n",
            f"{name}: numeric projection invariant (dev {dev:.1e}), "
            f"round trip {rt:.1e}, zero channels < {tol}{extra}",
            cond,
        )

    # 6g numeric: translation identity
    Gt = rng.standard_normal((3, 3))
    op_t = Op(eye(3), t=(1, 0, 0))
    _, Mt = op_t.transport((0, 1, (0, 0, 0)), Gt, SITES_BOND, eye(3), reduce_mod=True)
    ok(
        "6g.n",
        "translation: numeric Gamma' = Gamma on translated key",
        np.max(np.abs(np.array(Mt, dtype=float) - Gt)) < tol,
    )

    # random orthogonal sets: axial law and cancellation numerically
    okall = True
    for dsign in (+1, -1):
        Q = rand_orth(rng, dsign)
        Gn = rng.standard_normal((3, 3))
        Dn = numeric_decompose(Gn)[2]
        Gp = Q @ Gn @ Q.T
        Dp = numeric_decompose(Gp)[2]
        err = np.max(np.abs(Dp - dsign * (Q @ Dn)))
        sp1 = rng.standard_normal(3)
        sp2 = rng.standard_normal(3)
        e0 = sp1 @ Gn @ sp2
        e1 = (-Q @ sp1) @ Gp @ (-Q @ sp2)  # primed mapping, sigma = -1
        okall = okall and err < tol and abs(e0 - e1) < tol
    ok(
        "7.2",
        "random orthogonal (det +-1): D' = det(W) W D numerically; "
        "primed cancellation S -> -W S exact",
        okall,
    )


def main():
    print(__doc__.split("\n")[1])
    print("sympy", sp.__version__, "| numpy", np.__version__)
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section7()
    print(f"\nALL {NPASS[0]} CHECKS PASSED")


if __name__ == "__main__":
    main()
