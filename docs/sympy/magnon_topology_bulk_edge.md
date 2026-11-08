# Signed bulk–edge orientation certificate

`magnon_topology_bulk_edge.py` pins the sign convention used by
`compare_bulk_edge` before the numerical benchmark is interpreted. The bulk
FHS plaquette follows increasing `(k_edge, k_width)` and the link is
`<u(k)|u(k+dk)>/|...|`; its corresponding connection is
`A = +i <u|du>`. The certificate evaluates the lower-band curvature of a
continuum Dirac mass wall and the normalizable edge spinor directly.

For a strip whose interior is at increasing width, the min-width edge has
`E(k_edge) = +k_edge` in the positive-Chern wall convention; the max-width
edge has `E(k_edge) = -k_edge`. With crossings signed by the slope as
`k_edge` increases, the convention is therefore

- min-width (`left`) edge: `signed_flow = -sum(Chern below gap)`;
- max-width (`right`) edge: `signed_flow = +sum(Chern below gap)`.

The label left/right is geometric, not a drawing-dependent screen direction:
left is the strip boundary at the minimum declared width coordinate, and
right is the maximum-width boundary. Reversing the declared edge translation
reverses the reciprocal coordinate and both signs consistently. The embedding
basis, its handedness, and the fixed normal momentum are recorded in the
comparison output.

This certificate fixes the local sign calibration; it is not a numerical
lattice benchmark or a proof of finite-width convergence. The executable
honeycomb strip regressions exercise both edges, width/k sampling, a trivial
control, and failure cases using `calculate_bulk`, `calculate_strip`, and
`compare_bulk_edge` themselves.
