"""Exact vector/orbital-offset identities for the SIESTA spiral importer.

Run with mydev Python. Complements siesta_spiral_pencil.py [A, C, D].
"""

from pathlib import Path

import numpy as np
import sympy as sp

# Rows are direct lattice vectors. A skew exact cell prevents a transpose
# error from being hidden by the orthogonal producer fixture.
A = sp.Matrix([[4, 1, 0], [0, 5, 1], [1, 0, 6]])
B = 2 * sp.pi * A.inv().T
assert sp.simplify(A * B.T - 2 * sp.pi * sp.eye(3)) == sp.zeros(3)
q = sp.Matrix(1, 3, sp.symbols("q1 q2 q3", real=True))
R = sp.Matrix(1, 3, sp.symbols("R1 R2 R3", integer=True))
ti = sp.Matrix(1, 3, sp.symbols("ti1 ti2 ti3", real=True))
tj = sp.Matrix(1, 3, sp.symbols("tj1 tj2 tj3", real=True))
xij = (R + tj - ti) * A
qcart = q * B
assert sp.simplify((qcart * xij.T)[0] - 2 * sp.pi * (q * (R + tj - ti).T)[0]) == 0
assert sp.simplify((-R + ti - tj) * A + xij) == sp.zeros(1, 3)

# At q=0 the intra-cell Bloch phase is a unitary gauge change, hence a
# q=0 eigenvalue test cannot detect omitted orbital positions.
k, ri, rj = sp.symbols("k ri rj", real=True)
assert (
    sp.simplify(
        sp.exp(-sp.I * k * (rj - ri)) - sp.exp(sp.I * k * ri) * sp.exp(-sp.I * k * rj)
    )
    == 0
)

# Stored mixing partner relation, source-exact setup_k (same-block H12/H21).
kx, qx, h3, h4 = sp.symbols("kx qx h3 h4", real=True)
h12 = h3 - sp.I * h4
h21 = sp.conjugate(h12)
h12_partner = sp.exp(-sp.I * qx) * h12
h21_partner = sp.conjugate(h12_partner)
Bij = sp.Matrix(
    [[0, h12 * sp.exp(-sp.I * (kx + qx / 2))], [h21 * sp.exp(-sp.I * (kx - qx / 2)), 0]]
)
Bji = sp.Matrix(
    [
        [0, h12_partner * sp.exp(sp.I * (kx + qx / 2))],
        [h21_partner * sp.exp(sp.I * (kx - qx / 2)), 0],
    ]
)
assert sp.simplify(Bji - Bij.H) == sp.zeros(2)
print(
    "PASS: skew-cell reciprocal dual, full orbital displacement, q=0 gauge, textured adjoint"
)

# Independently certify the stored partner relation on the recorded producer.
fixture = (
    Path(__file__).resolve().parents[2]
    / "tests/siesta_spiral_fixtures/siesta_spiral_fe3.npz"
)
with np.load(fixture, allow_pickle=False) as f:
    entries = {
        (int(i), int(j), tuple(f["sc_off"][c])): h
        for i, j, c, h in zip(f["mu"], f["nu"], f["isc"], f["h"], strict=True)
    }
    max_defect = 0.0
    for (i, j, R), h in entries.items():
        partner = entries[(j, i, tuple(-v for v in R))]
        x = (
            np.asarray(R) @ f["cell"]
            + f["orbital_positions"][j]
            - f["orbital_positions"][i]
        )
        predicted = np.exp(-1j * f["q_cart"] @ x) * complex(h[2], -h[3])
        max_defect = max(max_defect, abs(complex(partner[2], -partner[3]) - predicted))
    assert max_defect < 1e-8, max_defect
    print(f"PASS: actual producer mixing partners, maximum defect {max_defect:.3e} eV")
