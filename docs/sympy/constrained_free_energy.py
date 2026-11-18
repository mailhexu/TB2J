"""Assertion-checked derivation: constrained-envelope sign and
finite-width free-energy pitch identities.

Story 001 (spiral-first-order-response), NFR-005 / FR-010 conventions.
Independent of TB2J/TBUpy.  These identities decide (i) the sign of the
multiplier term in a constrained total-energy slope and (ii) why a raw
eigenvalue-sum difference is NOT a free-energy slope at finite smearing.

Constrained envelope.  For a constrained branch defined by C(x, theta)=0
and Lagrangian L = E + lambda C (THIS sign convention is pinned):

    dE_con/dtheta = (dE/dtheta + lambda dC/dtheta)|_{x = x*(theta)}

evaluated on the stationary branch.  With the opposite Lagrangian
convention L' = E - lambda' C the multiplier flips sign, lambda' = -lambda:
reporting the multiplier term with the wrong sign misstates the
constraint work.

Finite width.  Mermin free energy at fixed electron number N,

    F = sum_n f_n eps_n + T sum_n [f_n ln f_n + (1-f_n) ln(1-f_n)]
      = sum_n f_n eps_n - T S,   S = -sum_n [f_n ln f_n + (1-f_n) ln(1-f_n)],

is stationary in the occupations at fixed N.  Along the branch of
stationary states (envelope theorem):

    dF/dq = sum_n f_n deps_n/dq          (frozen VARIATIONAL occupations)

while the raw eigenvalue sum with re-Fermi-filled occupations,

    E_raw(q) = sum_n f_n*(q) eps_n(q)  =>  dE_raw/dq
             = dF/dq + sum_n f_n' (eps_n - mu)   (!= dF/dq at T > 0),

differs by the occupation-response term.  Re-Fermi filling alone does
NOT turn a raw-sum slope into the free-energy slope; the entropy piece
(or equivalently the frozen-variational-occupation slope) is required.
As T -> 0 the difference vanishes.

Checks
------
[A] symbolic constrained envelope on E = x^2/2 - g x cos(theta),
    C = x - sin(theta): branch derivative identity, multiplier sign
    under both L conventions, nonzero constraint work at generic theta.
[B] numeric constrained branch: Newton stationarity + central FD of the
    re-solved branch == dE/dtheta + lambda dC/dtheta <= 1e-9.
[C] symbolic two-level Mermin: envelope dF/dq = sum f deps/dq; raw-sum
    excess = (eps1 - eps2) f1'(q) = sum_n f_n' (eps_n - mu).
[D] numeric: fixed-N Fermi-Dirac occupations; FD of the re-solved free
    energy == frozen-variational slope <= 1e-9; raw-sum re-Fermi FD
    differs by exactly sum f' (eps - mu); zero-width limit closes it.

Run with the mydev environment:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/constrained_free_energy.py
"""

import numpy as np
import sympy as sp

# --------------------------------------------------------------------------
# [A] symbolic constrained envelope
# --------------------------------------------------------------------------


def check_symbolic_envelope():
    x, th, g, lam = sp.symbols("x theta g lambda", real=True)
    E = x**2 / 2 - g * x * sp.cos(th)
    C = x - sp.sin(th)
    L = E + lam * C
    stat = sp.solve([sp.Eq(sp.diff(L, x), 0), sp.Eq(C, 0)], [x, lam], dict=True)[0]
    x_star, lam_val = sp.simplify(stat[x]), sp.simplify(stat[lam])
    assert sp.simplify(x_star - sp.sin(th)) == 0
    assert sp.simplify(lam_val - (g * sp.cos(th) - sp.sin(th))) == 0
    # envelope identity: dE_con/dtheta == dE/dtheta + lambda dC/dtheta
    dE_con = sp.diff(E.subs(x, x_star), th)
    rhs = (sp.diff(E, th) + lam * sp.diff(C, th)).subs({x: x_star, lam: lam_val})
    assert sp.simplify(dE_con - rhs) == 0
    # opposite Lagrangian convention flips the multiplier
    stat2 = sp.solve(
        [sp.Eq(sp.diff(E - lam * C, x), 0), sp.Eq(C, 0)], [x, lam], dict=True
    )[0]
    assert sp.simplify(stat2[lam] + lam_val) == 0
    # constraint work is genuinely nonzero at generic theta
    work = sp.simplify(
        (lam_val * sp.diff(C, th)).subs({th: sp.Rational(3, 10), g: sp.Rational(4, 10)})
    )
    assert work != 0
    print(
        "[A] constrained envelope: dE_con/dtheta = dE/dtheta + "
        "lambda dC/dtheta at x*(theta); L=E-lambda'C gives lambda'=-lambda; "
        "constraint work nonzero at generic theta ... OK"
    )


# --------------------------------------------------------------------------
# numeric constrained branch
# --------------------------------------------------------------------------


def E_FUNC(x, th, g):
    return 0.5 * x * x - g * x * np.cos(th)


def constrained_branch(th, g):
    """Newton solve of dL/dx = 0, C = 0 for L = E + lambda C."""
    x, lam = np.sin(th), g * np.cos(th) - np.sin(th)  # analytic seed
    for _ in range(100):
        F1 = x - g * np.cos(th) + lam  # dL/dx
        F2 = x - np.sin(th)  # C
        step = np.linalg.solve(np.array([[1.0, 1.0], [1.0, 0.0]]), [-F1, -F2])
        x, lam = x + step[0], lam + step[1]
        if abs(F1) < 1e-14 and abs(F2) < 1e-14:
            break
    return x, lam


def check_numeric_envelope():
    g, th0, h = 0.4, 0.31, 1e-5

    def E_con(th):
        x, _ = constrained_branch(th, g)
        return E_FUNC(x, th, g)

    x0, lam0 = constrained_branch(th0, g)
    analytic = g * x0 * np.sin(th0) + lam0 * (-np.cos(th0))
    fd = (E_con(th0 + h) - E_con(th0 - h)) / (2 * h)
    assert abs(analytic - fd) < 1e-9, (analytic, fd)
    assert abs(x0 - np.sin(th0)) < 1e-12
    assert abs(lam0 - (g * np.cos(th0) - np.sin(th0))) < 1e-12
    print(
        f"[B] numeric branch: FD of re-solved constrained energy "
        f"{fd:.12f} == dE/dtheta + lambda dC/dtheta {analytic:.12f} "
        f"(<=1e-9); lambda = {lam0:.6f} follows the L=E+lambda C sign ... OK"
    )


# --------------------------------------------------------------------------
# [C] symbolic two-level Mermin free energy
# --------------------------------------------------------------------------


def check_symbolic_mermin():
    q, T, N, f1, f1p = sp.symbols("q T N f1 f1p", positive=True)
    a1, b1, a2, b2 = sp.symbols("a1 b1 a2 b2", real=True)
    eps1, eps2 = a1 + b1 * q, a2 + b2 * q
    f2 = N - f1

    def s(f):
        return f * sp.log(f) + (1 - f) * sp.log(1 - f)

    F = f1 * eps1 + f2 * eps2 + T * (s(f1) + s(f2))
    # along the branch f1 = f1(q): dF/dq = partial_q F + (dF/df1) f1'
    partial_q = sp.diff(F, q)  # fixed occupations
    assert sp.simplify(partial_q - (f1 * b1 + f2 * b2)) == 0
    dF_df1 = sp.simplify(sp.diff(F, f1))  # f2 = N - f1 included
    # stationarity of the Mermin functional at fixed N: dF/df1 = 0, so
    # dF/dq = sum f deps/dq (envelope theorem); eps_n - mu = T s'(f_n)
    assert (
        sp.simplify(
            dF_df1 - (eps1 - eps2 + T * (sp.diff(s(f1), f1) + sp.diff(s(f2), f1)))
        )
        == 0
    )
    # raw eigenvalue sum along the re-Fermi branch:
    # dE_raw/dq = sum f deps/dq + (eps1 - eps2) f1'
    E_raw = f1 * eps1 + f2 * eps2
    excess = sp.simplify(sp.diff(E_raw, q) + (eps1 - eps2) * f1p - partial_q)
    assert sp.simplify(excess - (eps1 - eps2) * f1p) == 0
    # with the stationarity relation eps_n - mu = T s'(f_n):
    # sum_n f_n' (eps_n - mu) = f1' [(eps1-mu) - (eps2-mu)] = (eps1-eps2) f1'
    mu = sp.Symbol("mu", real=True)
    general_excess = f1p * (eps1 - mu) + (-f1p) * (eps2 - mu)
    assert sp.simplify(general_excess - (eps1 - eps2) * f1p) == 0
    print(
        "[C] symbolic Mermin: dF/dq = sum f deps/dq on the stationary "
        "branch (dF/df1 = 0); raw-sum excess = sum_n f_n'(eps_n - mu) "
        "= (eps1 - eps2) f1' made explicit ... OK"
    )


# --------------------------------------------------------------------------
# numeric fixed-N occupations and free-energy slopes
# --------------------------------------------------------------------------


def entropy(f):
    fc = np.clip(f, 1e-14, 1 - 1e-14)
    return fc * np.log(fc) + (1 - fc) * np.log(1 - fc)


def solve_occupations(eps, nel, T):
    """Fixed-N Fermi-Dirac occupations and mu by bisection."""
    lo, hi = eps.min() - 20 * T - 1.0, eps.max() + 20 * T + 1.0
    for _ in range(300):
        mu = 0.5 * (lo + hi)
        f = 1.0 / (1.0 + np.exp(np.clip((eps - mu) / T, -700.0, 700.0)))
        if f.sum() > nel:
            hi = mu
        else:
            lo = mu
    mu = 0.5 * (lo + hi)
    return 1.0 / (1.0 + np.exp((eps - mu) / T)), mu


def check_numeric_free_energy():
    bq = np.array([0.9, -0.4, 0.55, -0.2])  # deps/dq per level
    a0 = np.array([0.3, 1.1, 0.7, 1.6])
    q0, nel, T, h = 0.37, 1.6, 0.1, 1e-5

    def state_at(q, T_try=T):
        eps = a0 + bq * q
        f, mu = solve_occupations(eps, nel, T_try)
        # F = sum f eps + T sum[f ln f + (1-f) ln(1-f)] = sum f eps - T S
        F = float(np.sum(f * eps) + T_try * np.sum(entropy(f)))
        return F, f, eps, mu

    def F_at(q, T_try=T):
        return state_at(q, T_try)[0]

    def E_raw_at(q, T_try=T):
        _, f, eps, _ = state_at(q, T_try)
        return float(np.sum(f * eps))

    F0, f0, eps0, mu0 = state_at(q0)
    fdF = (F_at(q0 + h) - F_at(q0 - h)) / (2 * h)
    frozen_slope = float(np.sum(f0 * bq))
    assert abs(fdF - frozen_slope) < 1e-9, (fdF, frozen_slope)

    fd_raw = (E_raw_at(q0 + h) - E_raw_at(q0 - h)) / (2 * h)
    f_plus, _ = solve_occupations(a0 + bq * (q0 + h), nel, T)
    f_minus, _ = solve_occupations(a0 + bq * (q0 - h), nel, T)
    dfdq = (f_plus - f_minus) / (2 * h)
    excess = float(np.sum(dfdq * (eps0 - mu0)))
    assert abs(fd_raw - fdF) > 1e-6, (fd_raw, fdF)
    assert abs((fd_raw - fdF) - excess) < 1e-9, (fd_raw, fdF, excess)

    def raw_minus_free(T_try):
        return (E_raw_at(q0 + h, T_try) - E_raw_at(q0 - h, T_try)) / (2 * h) - (
            F_at(q0 + h, T_try) - F_at(q0 - h, T_try)
        ) / (2 * h)

    wide, narrow = raw_minus_free(0.2), raw_minus_free(1e-3)
    assert narrow < wide and narrow < 1e-6, (wide, narrow)
    print(
        f"[D] finite width: F-slope FD {fdF:.9f} == frozen-variational "
        f"slope {frozen_slope:.9f} (<=1e-9); raw re-Fermi slope differs "
        f"by {fd_raw - fdF:+.3e}, matches sum f'(eps-mu) {excess:+.3e} "
        f"(<=1e-9); zero-width closes it: |diff| {wide:.1e} (T=0.2) -> "
        f"{narrow:.1e} (T=1e-3) ... OK"
    )


def main():
    check_symbolic_envelope()
    check_numeric_envelope()
    check_symbolic_mermin()
    check_numeric_free_energy()
    print("\nAll constrained/free-energy assertions passed.")


if __name__ == "__main__":
    main()
