"""Assert the two-level split-SOC contour second-order MAE normalization."""

import numpy as np
import sympy as sp
from ase.units import kB

from TB2J.mycfr import CFR

lam, strength = sp.symbols("lambda strength", real=True)
h = sp.Matrix([[-1, lam * strength], [lam * strength, 1]])
low = -sp.sqrt(1 + (lam * strength) ** 2)
assert sp.simplify(sp.diff(low, lam, 2).subs(lam, 0) / 2 + strength**2 / 2) == 0
assert sp.simplify(sp.trace(h)) == 0
assert sp.simplify(h.det() + 1 + (lam * strength) ** 2) == 0

energies = np.array([-1.0, 1.0])
w = np.array([[0.0, 0.1], [0.1, 0.0]])
for nz, tolerance in ((12, 1e-7), (30, 1e-9)):
    contour = CFR(nz=nz, T=0.05 / kB)
    trace = []
    for z in contour.path:
        g = 1.0 / (z - energies)
        trace.append(np.sum(abs(w) ** 2 * g[:, None] * g[None, :]))
    second_order = -np.imag(contour.integrate_values(np.array(trace))) / (2 * np.pi)
    assert abs(second_order + 0.005) < tolerance, (nz, second_order)
print("two-level SOC band curvature and CFR contour factor verified")
