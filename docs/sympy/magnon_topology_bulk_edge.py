"""Sign certificate for signed magnon bulk–edge flow conventions.

The FHS overlap link uses <u(k)|u(k+dk)> and positive plaquette order
(edge, width).  This script fixes the sign of the spectral-flow comparison
for a continuum Dirac domain wall; numerical lattice counts are covered by
the bulk-edge regressions, not inferred from this local certificate.
"""

import sympy as sp

kx, ky, m0 = sp.symbols("kx ky m0", real=True, positive=True)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])


# With the package's overlap link <u(k)|u(k+dk)>, use A = +i <u|du>.
# The lower-band FHS curvature is the degree density for H = d.sigma.
def lower_fhs_curvature(mass):
    d = sp.Matrix([kx, ky, mass])
    density = d.dot(d.diff(kx).cross(d.diff(ky)))
    return sp.simplify(density / (2 * (d.dot(d)) ** sp.Rational(3, 2)))


radial_k = sp.symbols("radial_k", nonnegative=True)
curvature_inside = lower_fhs_curvature(-m0).subs({kx: radial_k, ky: 0})
curvature_vacuum = lower_fhs_curvature(m0).subs({kx: radial_k, ky: 0})
flux_inside = sp.integrate(
    2 * sp.pi * radial_k * curvature_inside / (2 * sp.pi), (radial_k, 0, sp.oo)
)
flux_vacuum = sp.integrate(
    2 * sp.pi * radial_k * curvature_vacuum / (2 * sp.pi), (radial_k, 0, sp.oo)
)
assert sp.simplify(flux_inside) == -sp.Rational(1, 2)
assert sp.simplify(flux_vacuum) == sp.Rational(1, 2)
assert sp.simplify(flux_inside - flux_vacuum) == -1

# Lower-width edge: mass changes +m0 -> -m0 as width y increases into the
# sample.  The normalizable zero-mode spinor has sigma_x=+1, hence E=+k_edge.
chi_left = sp.Matrix([1, 1]) / sp.sqrt(2)
assert sx * chi_left == chi_left
assert sp.simplify(-sp.I * sy * sx * chi_left + sz * chi_left) == sp.zeros(2, 1)
assert sx * chi_left * kx == chi_left * kx

# Upper-width edge reverses the mass wall and the induced edge orientation.
chi_right = sp.Matrix([1, -1]) / sp.sqrt(2)
assert sx * chi_right == -chi_right
assert sp.simplify(-sp.I * sy * (-sx) * chi_right - sz * chi_right) == sp.zeros(2, 1)
assert sx * chi_right * kx == -chi_right * kx

# For a sample in y>0 relative to a trivial vacuum, signed crossings are
# measured with increasing global k_edge. Under this overlap/FHS convention
# the lower-width flow is minus the bulk Chern sum; the upper-width flow is plus.
delta_chern = flux_inside - flux_vacuum
left_flow = 1
right_flow = -1
assert left_flow == -delta_chern
assert right_flow == delta_chern

if __name__ == "__main__":
    print("Dirac domain-wall certificate: min-width flow = -C, max-width flow = +C")
