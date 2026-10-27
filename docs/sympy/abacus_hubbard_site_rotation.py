#!/usr/bin/env python3
"""Exact finite projector and spin-rotation checks for native PW DFT+U.

Run with mydev: python docs/sympy/abacus_hubbard_site_rotation.py
Paired report: abacus_hubbard_site_rotation.md.
MathDoc: tb2j-dev/abacus-hubbard-site-rotation

These assertions concern a finite Hilbert space and a prescribed co-rotation
of a separable site operator. They do not prove an ABACUS noncollinear SCF
response, general periodic Fourier convention or completeness of a truncated basis.
"""

import sympy as s


def run():
    # Generic complex 2-by-2 cross-overlap and site matrix: check projection
    # of a separable operator using independent explicit matrix products.
    x = s.Matrix(2, 2, lambda i, j: s.Symbol(f"x{i}{j}", complex=True))
    d = s.Matrix(2, 2, lambda i, j: s.Symbol(f"d{i}{j}", complex=True))
    direct = s.Matrix(
        2,
        2,
        lambda i, j: sum(
            x[i, a] * d[a, b] * s.conjugate(x[j, b]) for a in range(2) for b in range(2)
        ),
    )
    assert (direct - x * d * x.H).applyfunc(s.expand) == s.zeros(2)

    # Symbolic entries, rather than one numerical choice, check the trace
    # factorization for two-dimensional ambient and local spaces.
    def matrix_symbols(prefix):
        return s.Matrix(2, 2, lambda i, j: s.Symbol(f"{prefix}{i}{j}", complex=True))

    aa, bb = matrix_symbols("a"), matrix_symbols("b")
    dd, ee = matrix_symbols("d"), matrix_symbols("e")
    gg, hh = matrix_symbols("g"), matrix_symbols("h")
    expanded_full = s.trace(aa * dd * aa.H * gg * bb * ee * bb.H * hh).expand()
    expanded_reduced = s.trace(dd * aa.H * gg * bb * ee * bb.H * hh * aa).expand()
    assert s.expand(expanded_full - expanded_reduced) == 0

    # Rectangular factors: verify two different local vertices and two
    # complex Green matrices without assuming orthogonal channel groups.
    # This checks the finite factorized trace, not a periodic Fourier sum.
    A = s.Matrix([[1, s.I], [2, 1], [s.I, -1]])
    B = s.Matrix([[1], [1 - s.I], [2]])
    D = s.Matrix([[2, s.I], [-s.I, 1]])
    E = s.Matrix([[3]])
    Gu = s.Matrix([[1, s.I, 2], [-s.I, 2, 1], [2, 1, 3]])
    Gd = s.Matrix([[2, 1, s.I], [1, 3, 2], [-s.I, 2, 1]])
    full_factor_trace = s.trace((A * D * A.H) * Gu * (B * E * B.H) * Gd)
    reduced_factor_trace = s.trace(D * (A.H * Gu * B) * E * (B.H * Gd * A))
    assert s.expand(full_factor_trace - reduced_factor_trace) == 0
    assert s.expand(full_factor_trace) != 0
    assert s.trace((A * (D - D) * A.H) * Gu * (B * E * B.H) * Gd) == 0
    assert (
        s.expand(
            s.trace((A * (2 * D) * A.H) * Gu * (B * E * B.H) * Gd)
            - 2 * full_factor_trace
        )
        == 0
    )

    # Exact full-rank complex nonorthogonal two-center example, including
    # direct physical and projected spin-dependent resolvents.
    I = s.I
    a = s.Matrix([1, 0])
    b = s.Matrix([I / 2, s.sqrt(3) / 2])
    phi = a.row_join(b)
    metric = phi.H * phi
    assert metric.det() == s.Rational(3, 4)
    W = a * a.H
    vertex = phi.H * W * phi
    expected = s.Matrix([[1, I / 2], [-I / 2, s.Rational(1, 4)]])
    assert s.simplify(vertex - expected) == s.zeros(2)
    site_only = s.diag(1, 0)
    assert vertex != site_only
    Gup = (2 * s.eye(2) - W).inv()
    Gdn = (2 * s.eye(2)).inv()
    dual_up = metric.inv() * phi.H * Gup * phi * metric.inv()
    dual_down = metric.inv() * phi.H * Gdn * phi * metric.inv()
    full = s.simplify(s.trace(vertex * dual_up * vertex * dual_down))
    physical = s.simplify(s.trace(W * Gup * W * Gdn))
    local = s.simplify(s.trace(site_only * dual_up * site_only * dual_down))
    assert full == physical == s.Rational(1, 2)
    assert local == s.Rational(7, 9)

    # A native separable U factor is primal, whereas the existing projected
    # XC matrix acts with dual factors. Keep their cross terms explicitly.
    dual = phi * metric.inv()
    factors_i = phi[:, 0].row_join(dual[:, 0])
    factors_j = phi[:, 1].row_join(dual[:, 1])
    coeff_i = s.diag(2, 3)
    coeff_j = s.diag(5, 7)
    total_i = factors_i * coeff_i * factors_i.H
    total_j = factors_j * coeff_j * factors_j.H
    g1 = s.Matrix([[1, s.I], [-s.I, 2]])
    g2 = s.Matrix([[2, 1], [1, 3]])
    mixed_full = s.trace(total_i * g1 * total_j * g2)
    mixed_factorized = s.trace(
        coeff_i * factors_i.H * g1 * factors_j * coeff_j * factors_j.H * g2 * factors_i
    )
    u_only = s.trace(
        (2 * phi[:, 0] * phi[:, 0].H) * g1 * (5 * phi[:, 1] * phi[:, 1].H) * g2
    )
    xc_only = s.trace(
        (3 * dual[:, 0] * dual[:, 0].H) * g1 * (7 * dual[:, 1] * dual[:, 1].H) * g2
    )
    assert s.simplify(mixed_full - mixed_factorized) == 0
    assert s.simplify(mixed_full - u_only - xc_only) != 0

    # Rotate the prescribed spin-dependent site operator. Spin-last ordering.
    theta = s.symbols("theta", real=True)
    sy = s.Matrix([[0, -I], [I, 0]])
    sx = s.Matrix([[0, 1], [1, 0]])
    sz = s.diag(1, -1)
    u = s.cos(theta / 2) * s.eye(2) - I * s.sin(theta / 2) * sy
    rotation = s.kronecker_product(s.eye(2), u)
    M = s.kronecker_product(W, sz)
    dM = (rotation * M * rotation.H).diff(theta).subs(theta, 0)
    phi_spin = s.kronecker_product(phi, s.eye(2))
    dM_cov = s.simplify(phi_spin.H * dM * phi_spin)
    assert dM_cov == s.kronecker_product(vertex, sx)
    assert dM_cov != s.kronecker_product(site_only, sx)
    # Native scalar potential: U * (1/2 delta - occupation transpose).
    # Evaluate both physical spin channels independently before swapping them.
    U = s.symbols("U", real=True)
    nup = s.Matrix(2, 2, lambda i, j: s.Symbol(f"nu{i}{j}", real=True))
    ndn = s.Matrix(2, 2, lambda i, j: s.Symbol(f"nd{i}{j}", real=True))

    def potential(occupation):
        return U * (s.eye(2) / 2 - occupation.T)

    split = potential(nup) - potential(ndn)
    swapped = potential(ndn) - potential(nup)
    assert s.simplify(split - U * (ndn - nup).T) == s.zeros(2)
    assert s.simplify(swapped + split) == s.zeros(2)
    assert split.subs(dict(zip(ndn, nup)), simultaneous=True) == s.zeros(2)

    # Two-cell finite periodic model: a local site perturbation couples
    # distinct k points. Compare direct-space trace with explicit double-k
    # blocks after unitary Fourier transformation, with nonorthogonal
    # projector metric dressing performed before block contraction.
    fcell = s.Matrix([[1, 1], [1, -1]]) / s.sqrt(2)
    F = s.kronecker_product(fcell, s.eye(2))
    P = s.diag(phi, phi)
    Sm = P.H * P
    Wi = s.diag(W, s.zeros(2))
    Wj = s.diag(s.zeros(2), W)
    h0 = s.kronecker_product(
        s.Matrix([[0, s.Rational(1, 3)], [s.Rational(1, 3), 0]]), s.eye(2)
    )
    G1 = (3 * s.eye(4) - h0 - Wi - Wj).inv()
    G2 = (3 * s.eye(4) - h0).inv()
    cov_i, cov_j = P.H * Wi * P, P.H * Wj * P
    gd1 = Sm.inv() * P.H * G1 * P * Sm.inv()
    gd2 = Sm.inv() * P.H * G2 * P * Sm.inv()
    physical_periodic = s.simplify(s.trace(Wi * G1 * Wj * G2))
    projected_periodic = s.simplify(s.trace(cov_i * gd1 * cov_j * gd2))
    assert physical_periodic == projected_periodic
    assert physical_periodic != 0
    Ki, Kj = F * cov_i * F.H, F * cov_j * F.H
    K1, K2 = F * gd1 * F.H, F * gd2 * F.H
    assert K1[:2, 2:] == s.zeros(2)
    assert K2[:2, 2:] == s.zeros(2)
    double_k = sum(
        s.trace(
            Ki[2 * k : 2 * k + 2, 2 * q : 2 * q + 2]
            * K1[2 * q : 2 * q + 2, 2 * q : 2 * q + 2]
            * Kj[2 * q : 2 * q + 2, 2 * k : 2 * k + 2]
            * K2[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
        )
        for k in range(2)
        for q in range(2)
    )
    assert s.simplify(double_k - physical_periodic) == 0
    assert Ki[:2, 2:] != s.zeros(2)
    diagonal_only = sum(
        s.trace(
            Ki[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            * K1[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            * Kj[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
            * K2[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
        )
        for k in range(2)
    )
    assert s.simplify(diagonal_only - physical_periodic) != 0
    print("PASS: two-cell periodic trace requires distinct-k rotation blocks")
    print("PASS: generic separable projection and prescribed spin rotation")
    print("PASS: nonorthogonal physical/covariant trace 1/2; site block 7/9")
    print("LIMIT: finite toy periodicity, not native SCF or general periodic proof")


if __name__ == "__main__":
    run()
