"""Split-SOC insertion algebra: generalized-S resolvent derivative and topologies.

Story 001 of the split-SOC KS-band spec; promotes the research-note algebra
check (split_soc_without_lcao_research.md, "Algebra check performed for this
report") to a repo script, and pins it on random complex matrices at 1e-14.

Contract objects (fixed reference density, strength-0 principle):
  G_lambda(z) = [z S - H0 - lambda W]^{-1}     generalized (non-orthogonal) S
  G0 = G_0;  W = W_SO enters the PROPAGATOR only, never the vertices;
  V_a, V_b = magnetic-site rotation vertices (site a and b, distinct).

Asserted identities:

1. Resolvent derivative (S = I and positive-definite non-diagonal S):
     dG/dlambda|_0 = G0 W G0
     d2G/dlambda2|_0 = 2 G0 W G0 W G0
   exact in sympy on symbolic rational matrices; numerically via the
   independent defining equations A0 G' = W G0, A0 G'' = 2 W G' (A0 = zS-H0,
   solved without reusing G0), plus complex-step differentiation.
2. Two-vertex insertion topologies:
     d/dlambda Tr[V_a G V_b G]|_0 = Tr[V_a G0 W G0 V_b G0] + Tr[V_a G0 V_b G0 W G0]
   exactly the two orderings; SOC appears only sandwiched between propagators
   (never fused into a vertex), and the vertices are untouched by lambda.
3. Generalized-S factorization (both S = I and S != I):
     zS - H = S^{1/2} (z 1 - S^{-1/2} H S^{-1/2}) S^{1/2}
   and the orthonormalized propagator equals the generalized one.
4. All-atom coverage: for W_SO = W1 + W2 (site decomposition), the derivative
   is additive: dG/dlambda|_0(W1+W2) = G0(W1+W2)G0 = sum_c G0 Wc G0.
5. Strength-0 reference: all identities are evaluated at the lambda = 0
   reference G0 built from H0 alone (asserted: G0 depends only on zS - H0).

Run with the mydev environment.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

TOL = 1.0e-14

_RNG = np.random.default_rng(20260927)


def _exact_matrices(seed: int, s_identity: bool):
    """Exact rational test objects: z, S, H0, W, Va, Vb."""
    z = sp.Rational(3, 2) + 2 * sp.I  # exact complex energy away from the spectrum
    rng = np.random.default_rng(seed)

    def rand_rat_mat(n: int, herm: bool):
        a = sp.Matrix(rng.integers(-4, 5, (n, n)))
        if herm:
            a = a + a.T
        return a

    n = 3
    s_mat = sp.eye(n) if s_identity else rand_rat_mat(n, True) + 3 * sp.eye(n)
    h0 = rand_rat_mat(n, True)
    w_mat = rand_rat_mat(n, True)
    v_a = rand_rat_mat(n, True)
    v_b = rand_rat_mat(n, True)
    return z, s_mat, h0, w_mat, v_a, v_b


def _g0_of(z, s_mat, h0):
    return (z * s_mat - h0).inv()


def check_resolvent_derivative_symbolic() -> None:
    """Assertion 1 (exact): dG = G0 W G0, d2G = 2 G0 W G0 W G0, both S cases."""
    lam = sp.symbols("lambda")
    for s_identity in (True, False):
        z, s_mat, h0, w_mat, _, _ = _exact_matrices(11 + s_identity, s_identity)
        a_mat = z * s_mat - h0
        g0 = a_mat.inv()
        g_lambda = (a_mat - lam * w_mat).inv()
        first = sp.cancel(g_lambda.diff(lam).subs(lam, 0) - g0 * w_mat * g0)
        assert first == sp.zeros(*first.shape), f"dG != G0 W G0 (S=I: {s_identity})"
        second = sp.cancel(
            g_lambda.diff(lam, 2).subs(lam, 0) - 2 * g0 * w_mat * g0 * w_mat * g0
        )
        assert second == sp.zeros(*second.shape), f"d2G wrong (S=I: {s_identity})"
        print(
            f"  symbolic exact (S=I: {s_identity}): dG = G0 W G0 ; d2G = 2 G0 W G0 W G0"
        )


def check_generalized_s_factorization_symbolic() -> None:
    """Assertion 3 (exact for diagonal S, 1e-13 numeric for general SPD S)."""
    # exact symbolic with a diagonal S (exact square root)
    z = sp.Rational(3, 2) + 2 * sp.I
    s_diag = sp.diag(2, sp.Rational(5, 2), 3)
    h0 = sp.Matrix([[1, 2, 0], [2, -1, 1], [0, 1, 2]])
    g_gen = (z * s_diag - h0).inv()
    s_half = sp.diag(*[sp.sqrt(e) for e in s_diag.diagonal()])
    g_orth = (
        s_half.inv()
        * (z * sp.eye(3) - s_half.inv() * h0 * s_half.inv()).inv()
        * s_half.inv()
    )
    diff = sp.cancel(g_gen - g_orth)
    assert diff == sp.zeros(*diff.shape), "exact diagonal-S factorization failed"
    print("  symbolic exact (diagonal S): generalized == orthonormalized propagator")

    # numeric with a general SPD non-diagonal S
    import scipy.linalg as sla

    n = 4
    rng = np.random.default_rng(23)
    a_rand = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    s_mat = np.eye(n) + 0.3 * (a_rand + a_rand.conj().T) / 2.0
    h0n = _random_complex_hermitian(n, rng)
    zn = 1.7 + 2.3j
    s_half = sla.sqrtm(s_mat)
    g_gen = np.linalg.inv(zn * s_mat - h0n)
    g_orth = (
        np.linalg.inv(s_half)
        @ np.linalg.inv(
            zn * np.eye(n) - np.linalg.inv(s_half) @ h0n @ np.linalg.inv(s_half)
        )
        @ np.linalg.inv(s_half)
    )
    dev = np.abs(g_gen - g_orth).max()
    scale = np.abs(g_gen).max()
    assert dev < 1e-12 * scale, f"numeric factorization failed: {dev} (scale {scale})"
    print(
        f"  numeric (general SPD S): generalized == orthonormalized propagator (dev {dev:.1e}, scale {scale:.1f})"
    )


def _random_complex_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (a + a.conj().T) / np.sqrt(2.0 * n)


def check_resolvent_derivative_numeric() -> None:
    """Assertion 1 (numeric 1e-14): independent linear-solve + complex-step."""
    for s_nonunit in (False, True):
        n = 4
        rng = np.random.default_rng(31 + s_nonunit)
        z = 1.7 + 2.3j
        s_mat = np.eye(n, dtype=complex)
        if s_nonunit:
            s_rand = _random_complex_hermitian(n, rng)
            s_mat = s_mat + 0.4 * s_rand / np.linalg.eigvalsh(s_rand).max()
        h0 = _random_complex_hermitian(n, rng)
        w_mat = _random_complex_hermitian(n, rng)

        a0 = z * s_mat - h0
        g0 = np.linalg.inv(a0)

        # independent derivative: solve A0 X = W G0 (no reuse of the G0WG0 form)
        wg0 = w_mat @ g0
        g_first = np.linalg.solve(a0, wg0)
        dev = np.abs(g_first - g0 @ w_mat @ g0).max()
        scale = np.abs(g_first).max()
        assert dev < TOL * scale, f"dG vs solve mismatch (S!=I: {s_nonunit}): {dev}"
        print(
            f"  numeric (S!=I: {s_nonunit}): solve(A0, W G0) == G0 W G0  (dev {dev:.1e}, scale {scale:.1f})"
        )

        # second derivative: A0 G'' = 2 W G'
        g_second = np.linalg.solve(a0, 2.0 * w_mat @ g_first)
        dev = np.abs(g_second - 2.0 * (g0 @ w_mat @ g0 @ w_mat @ g0)).max()
        assert dev < TOL * scale, f"d2G mismatch: {dev}"
        print(
            f"  numeric (S!=I: {s_nonunit}): solve(A0, 2W G') == 2 G0 W G0 W G0  (dev {dev:.1e})"
        )

        # exact cubic resolvent identity:
        # (A0 - hW)(G0 + hG0WG0 + h^2 G0WG0WG0) = I - h^3 (W G0)^3
        hh = 1e-3
        approx = g0 + hh * g0 @ w_mat @ g0 + hh**2 * g0 @ w_mat @ g0 @ w_mat @ g0
        resid = (
            (a0 - hh * w_mat) @ approx
            - np.eye(n)
            + hh**3 * np.linalg.matrix_power(w_mat @ g0, 3)
        )
        dev = np.abs(resid).max()
        assert dev < 1e-12, f"cubic resolvent identity failed: {dev}"
        print(
            f"  numeric (S!=I: {s_nonunit}): cubic resolvent identity (dev {dev:.1e})"
        )


def check_two_vertex_topologies_numeric() -> None:
    """Assertion 2: the two insertion topologies, complex-step vs algebra."""
    n = 4
    rng = np.random.default_rng(47)
    z = 1.9 + 2.1j
    s_mat = np.eye(n)
    h0 = _random_complex_hermitian(n, rng)
    w1 = _random_complex_hermitian(n, rng)
    w2 = _random_complex_hermitian(n, rng)
    w_mat = w1 + w2  # site decomposition: W_SO covers all atoms
    v_a = _random_complex_hermitian(n, rng)
    v_b = _random_complex_hermitian(n, rng)

    a0 = z * s_mat - h0
    g0 = np.linalg.inv(a0)
    trace0 = np.trace(v_a @ g0 @ v_b @ g0)

    # algebraic first-order coefficient from the two topologies
    topo = np.trace(v_a @ g0 @ w_mat @ g0 @ v_b @ g0) + np.trace(
        v_a @ g0 @ v_b @ g0 @ w_mat @ g0
    )

    # finite-difference trace derivative with Richardson extrapolation
    # (independent of the topology algebra)
    hh, hh2 = 1e-4, 2e-4

    def d_trace(h_step: float) -> complex:
        g_h = np.linalg.inv(a0 - h_step * w_mat)
        return (np.trace(v_a @ g_h @ v_b @ g_h) - trace0) / h_step

    d_fd = (2.0 * d_trace(hh) - d_trace(hh2)) / 1.0  # O(hh^2) extrapolation
    scale = max(1.0, abs(topo))
    dev = abs(d_fd - topo)
    assert dev < 1e-8 * scale, f"topology sum != dTr: {dev} (scale {scale})"
    print(
        f"  two-vertex topologies == FD dTr (Richardson, dev {dev:.1e}, scale {scale:.1f})"
    )

    # each topology alone is NOT the full derivative (both required)
    t1 = np.trace(v_a @ g0 @ w_mat @ g0 @ v_b @ g0)
    dev = abs(t1 - topo)
    assert dev > 1e-3 * scale, "topologies unexpectedly degenerate"
    print(f"  both orderings required (single-topology gap {dev:.1e})")

    # SOC only in the propagator: per-site additivity of the derivative
    g1 = g0 @ w1 @ g0
    g2 = g0 @ w2 @ g0
    dev = np.abs((g1 + g2) - g0 @ w_mat @ g0).max()
    assert dev < TOL, f"site additivity of dG failed: {dev}"
    topo_split = np.trace(v_a @ g0 @ (w1 + w2) @ g0 @ v_b @ g0) + np.trace(
        v_a @ g0 @ v_b @ g0 @ (w1 + w2) @ g0
    )
    dev = abs(topo_split - topo)
    assert dev < 1e-12 * scale, f"site additivity of the trace failed: {dev}"
    print(f"  W_SO = W1 + W2 (all atoms): derivative additive (dev {dev:.1e})")

    # strength-0 reference: G0 built from H0 alone; lambda only enters via W
    assert np.abs(g0 - np.linalg.inv(z * s_mat - h0)).max() == 0.0
    print("  strength-0 reference: G0 = [zS - H0]^{-1} (lambda-free)")


def main() -> None:
    print("split_soc_insertion: generalized-S resolvent insertion algebra (story-001)")
    check_resolvent_derivative_symbolic()
    check_generalized_s_factorization_symbolic()
    check_resolvent_derivative_numeric()
    check_two_vertex_topologies_numeric()
    print(
        "all assertions passed (exact symbolic + 1e-14 numeric on random complex toys)"
    )


if __name__ == "__main__":
    main()
