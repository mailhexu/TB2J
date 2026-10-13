"""Collinear-limit projection: transverse curvature -> Heisenberg J (story 011).

Convention authority for the anchor comparison (AC-006):

* The physical pair-once Heisenberg energy is
  E = -sum_{a<b} J_pair_ab (e_a . e_b), positive J = ferromagnetic.
  TB2J's exchange_Jdict stores both ordered directions, each carrying
  J_stored_ab = J_pair_ab / 2 (exchange_spiral.py:490-497).
* The frozen-response core (story 009) reports the raw transverse Hessian
  K_ab = d^2 E / (d theta_a d theta_b) in eV/rad^2 for site tilts
  theta_a about declared transverse axes t_a (u_a = t_a x n).

This script asserts symbolically:

  IDENTITY-1 (transverse scalar product): for a collinear reference with all
  moments along a unit axis n, tilting site a by theta_a about a transverse
  axis t_a (t_a . n = 0) gives
      d^2 (e_a . e_b) / (d theta_a d theta_b) |_0 = u_a . u_b,
  with u_a = t_a x n a unit vector.

  IDENTITY-2 (projection): with E = -sum_{a<b} J_pair_ab (e_a . e_b),
      K_aa = sum_{b != a} J_pair_ab,
      K_ab = -J_pair_ab (u_a . u_b)   (a != b).
  The TB2J ordered-pair value for a common transverse tilt direction is
      J_stored_ab = -K_ab / 2; each direction carries this half value.
  A pure pairwise Heisenberg model obeys K_aa = -sum_{b != a} K_ab;
  a frozen band energy need not satisfy this model closure.

  IDENTITY-3 (AFM sign): with e_b = -n the pair term becomes
      E_ab = +J_pair_ab cos(theta_a - theta_b),
      K_ab = +J_pair_ab (u_a . u_b);
  the stored value is -K_ab / (2 sign(e_a.e_b) (u_a.u_b)).

  IDENTITY-4 (numeric): the projection reproduces finite differences of the
  classical Heisenberg energy for random J and per-site transverse axes.

Run in the mydev environment; any failed assertion exits nonzero.
"""

import numpy as np
import sympy as sp


def rodrigues_matrix(axis, angle):
    axis = sp.Matrix(axis)
    n = axis.norm()
    K = sp.Matrix(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return sp.eye(3) + sp.sin(angle) * K / n + (1 - sp.cos(angle)) * (K / n) ** 2


def _orthonormalize(vec, ref):
    """Component of vec orthogonal to unit ref, normalized."""
    ref = sp.Matrix(ref)
    v = sp.Matrix(vec) - ref * sp.Matrix(vec).dot(ref)
    return sp.simplify(v / v.norm())


def test_transverse_scalar_product():
    n = sp.Matrix([sp.Rational(1, 5), -1, sp.Rational(7, 5)])
    n = n / n.norm()
    ta = _orthonormalize([sp.Rational(1, 2), sp.Rational(1, 3), -sp.Rational(1, 7)], n)
    tb = _orthonormalize([-sp.Rational(2, 5), sp.Rational(3, 5), sp.Rational(1, 9)], n)
    th_a, th_b = sp.symbols("theta_a theta_b", real=True)
    ea = rodrigues_matrix(ta, th_a) * n
    eb = rodrigues_matrix(tb, th_b) * n
    dot = sp.simplify(ea.dot(eb))
    mixed = sp.simplify(sp.diff(dot, th_a, th_b).subs({th_a: 0, th_b: 0}))
    ua = sp.simplify(ta.cross(n))
    ub = sp.simplify(tb.cross(n))
    expected = sp.simplify(ua.dot(ub))
    assert sp.simplify(mixed - expected) == 0, "IDENTITY-1 mixed derivative != u_a.u_b"
    assert sp.simplify(ua.norm() - 1) == 0 and sp.simplify(ub.norm() - 1) == 0
    print("IDENTITY-1 PASS: d2(ea.eb)/dth_a dth_b = u_a . u_b (unit u)")


def test_projection_and_sum_rule():
    n = sp.Matrix([1, 0, 0])
    J12, J13, J23 = sp.symbols("J12 J13 J23", positive=True)
    Js = {(1, 2): J12, (1, 3): J13, (2, 3): J23}
    th1, th2, th3 = sp.symbols("theta_1 theta_2 theta_3", real=True)
    thetas = {1: th1, 2: th2, 3: th3}
    # one transverse axis per site, not mutually parallel
    t = {
        1: sp.Matrix([0, 1, 0]),
        2: sp.Matrix([0, 0, 1]),
        3: sp.Matrix([0, 1, 1]) / sp.sqrt(2),
    }
    e = {a: rodrigues_matrix(t[a], thetas[a]) * n for a in t}
    E = -sum(J * e[a].dot(e[b]) for (a, b), J in Js.items())
    zero = {th1: 0, th2: 0, th3: 0}
    u = {a: sp.simplify(t[a].cross(n)) for a in t}
    for (a, b), J in Js.items():
        Kab = sp.simplify(sp.diff(E, thetas[a], thetas[b]).subs(zero))
        assert sp.simplify(Kab + J * u[a].dot(u[b])) == 0, f"IDENTITY-2 pair {a}{b}"
        if u[a].dot(u[b]) != 0:
            assert sp.simplify(-Kab / (2 * u[a].dot(u[b])) - J / 2) == 0
        else:
            assert Kab == 0  # orthogonal transverse probes carry no pair projection
    for a in (1, 2, 3):
        Kaa = sp.simplify(sp.diff(E, thetas[a], 2).subs(zero))
        Jsum = sum(J for (c, d), J in Js.items() if a in (c, d))
        assert sp.simplify(Kaa - Jsum) == 0, f"IDENTITY-2 diagonal site {a}"
    # common-axis sum rule: all u equal => K_aa = -sum_b K_ab
    tc = {a: sp.Matrix([0, 1, 0]) for a in t}
    ec = {a: rodrigues_matrix(tc[a], thetas[a]) * n for a in t}
    Ec = -sum(J * ec[a].dot(ec[b]) for (a, b), J in Js.items())
    for a in (1, 2, 3):
        Kaa = sp.diff(Ec, thetas[a], 2).subs(zero)
        Kab_sum = sum(
            sp.diff(Ec, thetas[a], thetas[b]).subs(zero) for b in (1, 2, 3) if b != a
        )
        assert sp.simplify(Kaa + Kab_sum) == 0, f"IDENTITY-2 sum rule site {a}"
    print("IDENTITY-2 PASS: K_aa = sum_b J_ab, K_ab = -J_ab (u_a.u_b); sum rule holds")


def test_afm_sign():
    n = sp.Matrix([1, 0, 0])
    J = sp.Symbol("J", positive=True)
    th_a, th_b = sp.symbols("theta_a theta_b", real=True)
    ta = sp.Matrix([0, 1, 0])
    tb = sp.Matrix([0, 1, 1]) / sp.sqrt(2)  # non-orthogonal transverse axes
    ea = rodrigues_matrix(ta, th_a) * n
    eb = -rodrigues_matrix(tb, th_b) * n  # antiparallel reference
    E = -J * ea.dot(eb)
    Kab = sp.diff(E, th_a, th_b).subs({th_a: 0, th_b: 0})
    ua = ta.cross(n)
    ub = tb.cross(n)
    assert sp.simplify(-Kab / (-2 * ua.dot(ub)) - J / 2) == 0
    print(
        "IDENTITY-3 PASS: AFM K_ab = +J_pair (u_a.u_b); "
        "ordered-pair J_stored = -K_ab/(2 sign(e_a.e_b) (u_a.u_b))"
    )


def test_numeric_projection():
    rng = np.random.default_rng(11)
    n = np.array([1.0, 0.0, 0.0])
    J = {(0, 1): 0.037, (0, 2): -0.012, (1, 2): 0.006}
    axes = {}
    for a in range(3):
        v = rng.normal(size=3)
        v -= n * v.dot(n)
        axes[a] = v / np.linalg.norm(v)
    u = {a: np.cross(axes[a], n) for a in range(3)}

    def energy(thetas):
        e = [n * np.cos(th) + u[a] * np.sin(th) for a, th in enumerate(thetas)]
        return -sum(
            J[(a, b)] * np.dot(e[a], e[b]) for a in range(3) for b in range(a + 1, 3)
        )

    h = 1e-4
    K = np.zeros((3, 3))
    for a in range(3):
        tp = np.zeros(3)
        tp[a] = h
        tm = np.zeros(3)
        tm[a] = -h
        K[a, a] = (energy(tp) - 2 * energy(np.zeros(3)) + energy(tm)) / h**2
    for a in range(3):
        for b in range(a + 1, 3):
            pp, pm, mp, mm = (np.zeros(3) for _ in range(4))
            pp[a] = pm[a] = h
            mp[a] = mm[a] = -h
            pp[b] = mp[b] = h
            pm[b] = mm[b] = -h
            K[a, b] = K[b, a] = (energy(pp) - energy(pm) - energy(mp) + energy(mm)) / (
                4 * h**2
            )
    for (a, b), Jref in J.items():
        Jstored = -K[a, b] / (2 * np.dot(u[a], u[b]))
        assert (
            abs(Jstored - Jref / 2) < 1e-9
        ), f"IDENTITY-4 pair {a}{b}: {Jstored} vs stored {Jref / 2}"
    print("IDENTITY-4 PASS: finite-difference Hessian projects to ordered-pair J")


def main():
    test_transverse_scalar_product()
    test_projection_and_sum_rule()
    test_afm_sign()
    test_numeric_projection()
    print("collinear_anchor_projection: ALL ASSERTIONS GREEN")
    Kab, alignment, transverse_dot = sp.symbols(
        "K_ab alignment transverse_dot", real=True, nonzero=True
    )
    stored_j = -Kab / (2 * alignment * transverse_dot)
    assert sp.simplify(2 * alignment * transverse_dot * stored_j + Kab) == 0
    return {"stored_ordered_pair_j": stored_j}


if __name__ == "__main__":
    main()
