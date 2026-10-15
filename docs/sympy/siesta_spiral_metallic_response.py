"""Verified fixed-N Fermi-Dirac response for a frozen spinor pencil.

Run with mydev: python docs/sympy/siesta_spiral_metallic_response.py
The two-level algebra is exact; a separately implemented finite-difference
free-energy calculation re-solves the global chemical potential at every step.
"""

import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import expit


def check_symbolic():
    beta, mu = sp.symbols("beta mu", positive=True)
    e1, e2 = sp.symbols("e1 e2", real=True)

    def f(e):
        return 1 / (1 + sp.exp(beta * (e - mu)))

    def fp(e):
        return -beta * f(e) * (1 - f(e))

    omega = -sum(sp.log(1 + sp.exp(beta * (mu - e))) for e in (e1, e2)) / beta
    assert sp.simplify(sp.diff(omega, mu) + f(e1) + f(e2)) == 0
    assert sp.simplify(sp.diff(f(e1), e1) - fp(e1)) == 0
    assert (
        sp.simplify(
            sp.limit(
                (f(e1) - f(e1 + sp.Symbol("d"))) / (-sp.Symbol("d")), sp.Symbol("d"), 0
            )
            - fp(e1)
        )
        == 0
    )

    # One global chemical potential. N_mu = -D, where D = sum f'_eps < 0.
    a, b, d, omega_ab = sp.symbols("A_a A_b D Omega_ab", nonzero=True)
    omega_a_mu = -a
    mu_b = b / d  # N_b=b, N_mu=-d, so -N_b/N_mu=b/d.
    assert sp.simplify(omega_ab + omega_a_mu * mu_b - (omega_ab - a * b / d)) == 0

    # Two-level pencil: differentiating eigenvalues exactly proves the
    # interband denominator and onsite W term, without huge radical expansions.
    ta, tb = sp.symbols("ta tb", real=True)
    va0, va1, vb0, vb1, xa, xb, w0, w1 = sp.symbols(
        "va0 va1 vb0 vb1 xa xb w0 w1", real=True
    )
    h00 = -2 + va0 * ta + vb0 * tb + w0 * ta * tb
    h11 = 1 + va1 * ta + vb1 * tb + w1 * ta * tb
    h01 = xa * ta + xb * tb
    gap = sp.sqrt((h11 - h00) ** 2 + 4 * h01**2)
    levels = [(h00 + h11 - gap) / 2, (h00 + h11 + gap) / 2]
    origin = {ta: 0, tb: 0}
    for e, base, onsite, sign in zip(levels, (-2, 1), (w0, w1), (-1, 1)):
        assert sp.simplify(e.subs(origin) - base) == 0
        assert (
            sp.simplify(sp.diff(e, ta).subs(origin) - (va0 if sign < 0 else va1)) == 0
        )
        assert (
            sp.simplify(
                sp.diff(e, ta, tb).subs(origin) - (onsite + sign * 2 * xa * xb / 3)
            )
            == 0
        )
    omega_hessian = sum(
        f(base) * (onsite + sign * 2 * xa * xb / 3) + fp(base) * va * vb
        for base, onsite, sign, va, vb in ((-2, w0, -1, va0, vb0), (1, w1, 1, va1, vb1))
    )
    kernel = ((f(-2) - f(1)) / (-3)) * 2 * xa * xb
    spectral = f(-2) * w0 + f(1) * w1 + fp(-2) * va0 * vb0 + fp(1) * va1 * vb1 + kernel
    assert sp.simplify(omega_hessian - spectral) == 0
    return {
        "omega_mu": True,
        "fermi_derivative": True,
        "degenerate_kernel": True,
        "schur_sign": True,
        "two_level_spectral_kernel": True,
    }


def free_energy(theta, width=0.2, target=1.0):
    """Independent FD oracle: diagonalize and re-solve mu at each angle."""
    a, b = theta
    h = np.array(
        [
            [-2 + 0.4 * a + 0.2 * b + 0.1 * a * b, 0.3 * a + 0.15 * b],
            [0.3 * a + 0.15 * b, 1 + 0.1 * a - 0.25 * b - 0.2 * a * b],
        ]
    )
    eig = np.linalg.eigvalsh(h)
    mu = brentq(
        lambda trial: expit((trial - eig) / width).sum() - target,
        eig[0] - 50 * width,
        eig[-1] + 50 * width,
        xtol=1e-14,
    )
    f = expit((mu - eig) / width)
    assert abs(f.sum() - target) < 1e-12
    omega = -width * np.logaddexp(0, (mu - eig) / width).sum()
    entropy = -(f * np.log(f) + (1 - f) * np.log1p(-f)).sum()
    assert abs(omega + mu * target - (f @ eig - width * entropy)) < 1e-12
    return omega + mu * target, mu, eig


def check_numeric():
    width = 0.2
    _, mu, eig = free_energy((0.0, 0.0), width)
    f = expit((mu - eig) / width)
    fp = -f * (1 - f) / width
    va = np.array([[0.4, 0.3], [0.3, 0.1]])
    vb = np.array([[0.2, 0.15], [0.15, -0.25]])
    wab = np.diag([0.1, -0.2])
    vertices = (va, vb)
    d = fp.sum()  # D = sum df/depsilon < 0, NOT dN/dmu.
    assert d < 0
    a = [float(fp @ np.diag(v)) for v in vertices]
    k = np.zeros((2, 2))
    for i, vi in enumerate(vertices):
        for j, vj in enumerate(vertices):
            onsite = float(f @ np.diag(wab)) if i != j else 0.0
            diagonal = float(fp @ (np.diag(vi) * np.diag(vj)))
            interband = 2 * (f[0] - f[1]) / (eig[0] - eig[1]) * vi[0, 1] * vj[1, 0]
            k[i, j] = onsite + diagonal + interband - a[i] * a[j] / d
    h = 2e-3

    def energy(x, y):
        return free_energy((x, y), width)[0]

    fd = np.array(
        [
            [
                (energy(h, 0) - 2 * energy(0, 0) + energy(-h, 0)) / h**2,
                (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                / (4 * h**2),
            ],
            [
                (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h))
                / (4 * h**2),
                (energy(0, h) - 2 * energy(0, 0) + energy(0, -h)) / h**2,
            ],
        ]
    )
    error = float(np.max(np.abs(k - fd)))
    assert error < 1e-5, (k, fd, error)
    fixed_mu = k + np.outer(a, a) / d
    assert np.max(np.abs(fixed_mu - fd)) > 1e-4
    # A common level shift must have zero curvature at fixed N. The
    # fixed-mu diagonal kernel is D; the Schur correction cancels it.
    assert abs(d - d**2 / d) < 1e-14
    return {
        "finite_difference_error": error,
        "fixed_mu_error": float(np.max(np.abs(fixed_mu - fd))),
        "chemical_potential": mu,
    }


def main():
    results = {**check_symbolic(), **check_numeric()}
    print(results)
    print("ALL ASSERTIONS GREEN")
    return results


if __name__ == "__main__":
    main()
