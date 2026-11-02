#!/usr/bin/env python3
"""Assertion-verified q-phased nonorthogonal Green and exchange response."""

import argparse
import json
import platform

import sympy as sp


def derive():
    tau, z, ef = sp.symbols("tau z E_F_ref")
    ak = sp.Matrix([[z - 1, -sp.I / 3], [sp.I / 3, 2 * z + 1]])
    aq = sp.Matrix([[2 * z - 2, sp.Rational(1, 4)], [sp.Rational(1, 4), z + 3]])
    dh = sp.Matrix([[sp.Rational(2, 3), sp.I], [-2 * sp.I, sp.Rational(1, 5)]])
    ds = sp.Matrix([[sp.Rational(1, 7), 0], [sp.Rational(1, 9), sp.Rational(1, 11)]])
    shifted_dh = dh - ef * ds
    vertex = shifted_dh - z * ds
    assert sp.simplify(vertex - (dh - (z + ef) * ds)) == sp.zeros(2)
    # Independent differentiation of a block pencil, not differentiation of GVG.
    zero = sp.zeros(2)
    pencil = sp.BlockMatrix([[ak, -tau * vertex], [zero, aq]]).as_explicit()
    response = pencil.inv().diff(tau).subs(tau, 0)[:2, 2:]
    assert sp.simplify(response - ak.inv() * vertex * aq.inv()) == zero

    k, q = sp.symbols("k q", real=True)
    R = sp.symbols("R", integer=True)
    forward = sp.exp(-2 * sp.pi * sp.I * (k + q) * R)
    reverse = sp.exp(2 * sp.pi * sp.I * k * R)
    stored_negative = sp.exp(2 * sp.pi * sp.I * (k + q) * R)
    assert (
        sp.simplify(reverse / stored_negative - sp.exp(-2 * sp.pi * sp.I * q * R)) == 0
    )
    onsite_phase = sp.exp(-2 * sp.pi * sp.I * q * R)
    Pi, Pj, Gij, Gji, dPi, dPj, dGij, dGji = sp.symbols(
        "Pi Pj Gij Gji dPi dPj dGij dGji", commutative=False
    )
    product = (
        (Pi + tau * dPi)
        * (Gij + tau * dGij)
        * (Pj + tau * onsite_phase * dPj)
        * (Gji + tau * dGji)
    )
    response_product = sp.expand(product.diff(tau).subs(tau, 0))
    expected = (
        dPi * Gij * Pj * Gji
        + Pi * dGij * Pj * Gji
        + onsite_phase * Pi * Gij * dPj * Gji
        + Pi * Gij * Pj * dGji
    )
    assert sp.expand(response_product - expected) == 0

    ar, ai, br, bi = sp.symbols("ar ai br bi", real=True)
    cosine, sine = ar + sp.I * ai, br + sp.I * bi
    cq, cm = cosine - sp.I * sine, cosine + sp.I * sine
    physical = sp.expand((cq - sp.conjugate(cm)) / (2 * sp.I))
    assert sp.simplify(physical - (ai - sp.I * bi)) == 0
    assert sp.simplify(sp.im(cq) - physical) != 0

    # Equilibrium copies, unlike localized derivatives, carry a 1/N average.
    h0, tr, ti = sp.symbols("h0 tr ti", real=True)
    hopping = tr + sp.I * ti
    omega = -sp.Rational(1, 2) + sp.I * sp.sqrt(3) / 2
    hsc = sp.Matrix(
        [
            [h0, hopping, sp.conjugate(hopping)],
            [sp.conjugate(hopping), h0, hopping],
            [hopping, sp.conjugate(hopping), h0],
        ]
    )
    bloch = sp.Matrix(3, 3, lambda row, col: omega ** (row * col) / sp.sqrt(3))
    assert sp.simplify(bloch.conjugate().T * bloch) == sp.eye(3)
    diagonal = sp.diag(
        *[
            h0 + hopping * omega**j + sp.conjugate(hopping) * omega ** (-j)
            for j in range(3)
        ]
    )
    assert sp.simplify(bloch.conjugate().T * hsc * bloch - diagonal) == sp.zeros(3)
    copy_average = sum(hsc[row, (row + 1) % 3] for row in range(3)) / 3
    assert sp.simplify(copy_average - hopping) == 0
    results = {
        "fixed_gauge_pencil_vertex": vertex,
        "mixed_momentum_inverse_derivative": response,
        "oriented_forward_bond_phase": forward,
        "oriented_reverse_bond_phase": reverse,
        "onsite_endpoint_phase": onsite_phase,
        "four_term_exchange_product_rule": expected,
        "complex_linear_contour_reduction": physical,
        "primitive_reference_copy_average": copy_average,
    }
    return [
        {"name": name, "srepr": sp.srepr(value), "latex": sp.latex(value)}
        for name, value in sorted(results.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kosmic-json", action="store_true")
    args = parser.parse_args()
    results = derive()
    if args.kosmic_json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "python_version": platform.python_version(),
                    "sympy_version": sp.__version__,
                    "results": results,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"PASS: {len(results)} q-phased exchange identities")


if __name__ == "__main__":
    main()
