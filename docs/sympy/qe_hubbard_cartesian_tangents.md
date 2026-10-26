# QE native transverse-to-Cartesian tangent map

Executable authority: `qe_hubbard_cartesian_tangents.py`. Executed with the
configured `mydev` SymPy environment; all assertions passed. This is an
implementation pin extending the existing native frozen-source contract, not
a migration or edit of the vault MathDoc.

## Assumptions and units

- The source frame supplies a validated unit direction and two orthonormal
  transverse unit directions. Native record22 retains those directions and
  its two analytic derivatives.
- Spatial projectors, scalar charge/potential, generalized overlap and fixed
  ionic/SOC terms stay frozen, as in `qe_hubbard_native_contract.py`.
- Native analytic derivatives are physical `dV/dtheta` in Ry, not derivatives
  of the doubled magnetic splitting. Convert them to eV once; do not add a
  factor2.

## Executed checks

The script verifies the transverse projector identity for arbitrary spherical
angles, including the arbitrary but valid basis at a pole. It verifies that
projecting each Cartesian target onto that transverse basis gives the same
rotation generator as the direct target. An exact tilted native x-reference
basis then verifies this map on a general symbolic Pauli potential and checks
the zero longitudinal contraction. These checks justify the coefficient
weights taken from `rotation.tangent_directions[axis, :]` in the existing QE
reader; they do not replace native source/action or finite-difference gates.

The independent complex coefficient/filter and redundant joint trace pins
remain in `qe_hubbard_joint_frame.py`. Actual source records and mapped
occupation fallback still require native validation. No exchange tensor,
material, convergence, FR-NC or mixed-family certification follows from these
symbolic checks.
