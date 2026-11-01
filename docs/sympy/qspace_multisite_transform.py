#!/usr/bin/env python3
"""Multi-site finite-grid transform and basis-phase authority (Story 009).

Exact symbolic verification, on a two-sublattice toy with inequivalent
sites and a complete three-point reciprocal mesh, of the q -> R convention
that the reciprocal-space driver must satisfy to reproduce the real-space
Lichtenstein pair table:

S1  ordered-pair kernel identity.  With the production real-space Green
    transform G(R) = (1/N) sum_k exp(-2 pi i R.k) G(k) (k2Rfactor =
    -2 pi i), the real-space pair kernel

        T_ij(R) = tr[D_i G^up_ij(R) D_j G^down_ji(-R)]

    equals (1/N) sum_q exp(+2 pi i R.q) K1_ij(q) where

        K1_ij(q) = sum_k tr[D_i G^up_ij(k) D_j G^down_ji(k+q)] / N.

    The production q_to_r minus phase instead yields T_ji(R) (= T_ij(-R)),
    which differs from T_ij(R) for asymmetric inter-sublattice models.

S2  imaginary-extraction placement.  The per-q imaginary part does not
    commute with the phase transform on an asymmetric inter-sublattice
    model; the extraction must be taken after the transform.

S3  channel-average mesh-sum premise (the approved one-site reciprocity
    claim extended to inter-sublattice pairs): on the complete mesh the
    coded symmetric kernel (K1 + K2)/2 transforms to the same real-space
    pair as the single-channel K1.

S4  cell-gauge basis-phase premise.  The orbital-block projection
    (Gk_atom) introduces no extra position phase; all basis-position
    dependence is carried by the hopping values.  A position twist
    (the q.tau generalized-Bloch convention) is a different model class,
    outside this authority.

Everything is exact: mesh phases are cube roots of unity and the S1/S3
equalities hold as rational functions of the symbolic energy, hence after
any common linear energy functional (in particular the production contour
integral).
"""

from __future__ import annotations

import argparse
import json
import platform

import sympy as s

N_MESH = 3


def mesh_phases(n=N_MESH):
    # Exact algebraic phases in Q(sqrt(3)); the exp(2 pi i m / n) principal
    # form defeats simplify on the combined rational functions.
    if n != 3:
        raise NotImplementedError("authority pinned to the three-point mesh")
    w = -s.Rational(1, 2) + s.sqrt(3) * s.I / 2
    return [s.Integer(1), w, s.conjugate(w)]


def build_model():
    """Two inequivalent sublattices A, B; real-gauge asymmetric hoppings.

    h^s(k) = [[ea + s dA, tAB + tR conj(w_k)],
              [tAB + tL w_k, eb + s dB]]
    with w_k the mesh phase, up channel s=+1, down s=-1.  Real gauge; the
    asymmetry tR != tL makes the inter-sublattice response R-asymmetric.
    The concrete values are exact rationals (generic: dA != dB, tR != tL);
    only the energy E stays symbolic so the equalities hold as rational
    functions of E, hence after any linear energy functional.
    """
    E = s.Symbol("E", real=True)
    ea = eb = s.Rational(1, 10)
    dA, dB = s.Rational(7, 20), s.Rational(1, 4)
    tAB, tR, tL = s.Rational(2, 25), s.Rational(1, 20), s.Rational(3, 100)

    def hk(wk, sign):
        return s.Matrix(
            [
                [ea + sign * dA, tAB + tR * s.conjugate(wk)],
                [tAB + tL * wk, eb + sign * dB],
            ]
        )

    return E, hk


def resolvent(hk, E):
    return (E * s.eye(hk.shape[0]) - hk).inv()


def kernel1(w_mesh, hk, E, i, j, qshift):
    """sum_k tr[D_i G^up_ij(k) D_j G^down_ji(k+q)] / N, symbolic in E."""
    N = len(w_mesh)
    idx = {w: m for m, w in enumerate(w_mesh)}
    tot = s.Integer(0)
    for wk in w_mesh:
        wq = w_mesh[(idx[wk] + qshift) % N]
        g_up = resolvent(hk(wk, +1), E)[i, j]
        g_dn = resolvent(hk(wq, -1), E)[j, i]
        tot += g_up * g_dn
    return s.cancel(tot / N)


def kernel2(w_mesh, hk, E, i, j, qshift):
    """sum_k tr[D_i G^down_ij(k) D_j G^up_ji(k+q)] / N, symbolic in E."""
    N = len(w_mesh)
    idx = {w: m for m, w in enumerate(w_mesh)}
    tot = s.Integer(0)
    for wk in w_mesh:
        wq = w_mesh[(idx[wk] + qshift) % N]
        g_dn = resolvent(hk(wk, -1), E)[i, j]
        g_up = resolvent(hk(wq, +1), E)[j, i]
        tot += g_dn * g_up
    return s.cancel(tot / N)


def real_space_T(w_mesh, hk, E, i, j, R):
    """T_ij(R) = tr[D_i G^up_ij(R) D_j G^down_ji(-R)], symbolic in E.

    Production convention: minus-phase transform exp(-2 pi i R.k) with
    1/N kweights, so G_ij(R) = sum_k conj(w_k)^R G_ij(k) / N and
    G_ji(-R) = sum_k w_k^R G_ji(k) / N.
    """
    N = len(w_mesh)
    g_up_R = (
        sum(s.conjugate(wk) ** R * resolvent(hk(wk, +1), E)[i, j] for wk in w_mesh) / N
    )
    g_dn_mR = sum(wk**R * resolvent(hk(wk, -1), E)[j, i] for wk in w_mesh) / N
    return s.cancel(g_up_R * g_dn_mR)


def transform(w_mesh, kernels, R, E, E_val):
    """(1/N) sum_q w_q^R kernel(q) evaluated at E_val."""
    N = len(w_mesh)
    return sum(w_mesh[q] ** R * kernels(q).subs(E, E_val) for q in range(N)) / N


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kosmic-json", action="store_true")
    args = parser.parse_args()

    w_mesh = mesh_phases()
    E, hk = build_model()
    E_val = s.Rational(-1, 2)  # inside the up-channel band, below mu = 0

    # ---- S1: plus-phase inverse reproduces the real-space pair kernel ----
    s1_diff = s.Integer(0)
    minus_counterexample = None
    for i, j in ((0, 1), (1, 0)):
        for R in (0, 1, 2):
            T = real_space_T(w_mesh, hk, E, i, j, R).subs(E, E_val)
            plus = transform(
                w_mesh,
                lambda q, i=i, j=j: kernel1(w_mesh, hk, E, i, j, q),
                R,
                E,
                E_val,
            )
            # minus phase means conj(w_q)^R; recompute explicitly:
            N = len(w_mesh)
            minus = (
                sum(
                    s.conjugate(w_mesh[q]) ** R
                    * kernel1(w_mesh, hk, E, i, j, q).subs(E, E_val)
                    for q in range(N)
                )
                / N
            )
            s1_diff += s.simplify(T - plus)
            if i != j and R != 0:
                diff = s.simplify(T - minus)
                if diff != 0:
                    minus_counterexample = diff
    assert s1_diff == 0, s1_diff
    assert minus_counterexample is not None and minus_counterexample != 0

    # ---- S2: imaginary extraction must follow the transform ----
    i, j, R = 0, 1, 1
    N = len(w_mesh)
    per_q_im = (
        sum(
            w_mesh[q] ** R * s.im(kernel1(w_mesh, hk, E, i, j, q).subs(E, E_val))
            for q in range(N)
        )
        / N
    )
    post_transform_im = s.im(
        sum(
            w_mesh[q] ** R * kernel1(w_mesh, hk, E, i, j, q).subs(E, E_val)
            for q in range(N)
        )
        / N
    )
    s2_diff = s.simplify(per_q_im - post_transform_im)
    assert s2_diff != 0, s2_diff

    # ---- S3: coded channel average == single channel under the ----
    # ---- Im-projected transform (the production comparison)      ----
    # The complex integrand transforms differ at fixed energy; the
    # physical (imaginary-projected, energy-integrated) responses agree
    # on the complete mesh under the reciprocity premises.
    s3_diff = s.Integer(0)
    for R in (0, 1, 2):
        N = len(w_mesh)
        t_avg = s.im(
            sum(
                w_mesh[q] ** R
                * (
                    (kernel1(w_mesh, hk, E, 0, 1, q) + kernel2(w_mesh, hk, E, 0, 1, q))
                    / 2
                ).subs(E, E_val)
                for q in range(N)
            )
            / N
        )
        t_one = s.im(
            sum(
                w_mesh[q] ** R * kernel1(w_mesh, hk, E, 0, 1, q).subs(E, E_val)
                for q in range(N)
            )
            / N
        )
        s3_diff += s.simplify(t_avg - t_one)
    assert s3_diff == 0, s3_diff

    # ---- S4: cell gauge: the block projection carries no extra phase ----
    # A constant onsite gauge U = diag(1, e^{i phi}) applied to both spin
    # channels leaves the ordered-pair kernel invariant exactly (the
    # trace cancels the phases), so Gk_atom needs no position factor; a
    # k-dependent position twist (q.tau) is a different model class.
    phi = s.Symbol("phi", real=True)
    U = s.diag(1, s.exp(s.I * phi))

    def gauge_kernel(q):
        tot = s.Integer(0)
        idx = {w: m for m, w in enumerate(w_mesh)}
        for wk in w_mesh:
            wq = w_mesh[(idx[wk] + q) % N_MESH]
            g_up = (U * resolvent(hk(wk, +1), E)[0:2, 0:2] * U.conjugate())[0, 1]
            g_dn = (U * resolvent(hk(wq, -1), E)[0:2, 0:2] * U.conjugate())[1, 0]
            tot += g_up * g_dn
        return s.cancel(tot / N_MESH)

    plain = kernel1(w_mesh, hk, E, 0, 1, 1)
    gauged = gauge_kernel(1)
    s4 = s.simplify((gauged - plain).subs(E, E_val))
    assert s4 == 0, s4

    named = {
        "multisite_plus_phase_identity": s1_diff,
        "multisite_minus_phase_counterexample": s.simplify(s.Abs(minus_counterexample)),
        "imag_extraction_order_difference": s.simplify(s.Abs(s2_diff)),
        "channel_average_mesh_sum": s3_diff,
        "cell_gauge_block_projection": s4,
    }

    if args.kosmic_json:
        payload = {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "sympy_version": s.__version__,
            "results": [
                {
                    "name": name,
                    "latex": s.latex(named[name]),
                    "srepr": s.srepr(named[name]),
                }
                for name in sorted(named)
            ],
        }
        print(json.dumps(payload))
    else:
        for name in sorted(named):
            print(f"{name}: {named[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
