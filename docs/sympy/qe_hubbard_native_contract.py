"""Exact SymPy pins for QE's native on-site Hubbard conventions.

Run with the configured ``mydev`` Python.  These are algebraic contract
checks only; they do not establish QE action, occupation, or rotation parity.
"""

from __future__ import annotations

import sympy as sp

I2 = sp.eye(2)
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SZ = sp.diag(1, -1)
PAULI = (SX, SY, SZ)


def _projection_fixture():
    """Return a small exact S-applied and orthogonalized projector fixture."""
    phi = sp.Matrix(
        [
            [sp.Rational(1, 2), 0],
            [2, 2 * sp.I],
            [0, 1],
        ]
    )
    overlap_operator = sp.diag(4, 1, 1)
    overlap = phi.H * overlap_operator * phi
    root = sp.Matrix([[2, sp.I], [-sp.I, 2]])
    assert overlap == root * root
    assert root == root.H
    assert root.eigenvals() == {sp.Integer(1): 1, sp.Integer(3): 1}

    root_inverse = root.inv()
    wfc_s = overlap_operator * phi
    wfc_ortho = wfc_s * root_inverse
    wrong_transposed_orthogonalization = wfc_s * root_inverse.T
    assert wfc_ortho != wrong_transposed_orthogonalization
    return phi, overlap_operator, overlap, root_inverse, wfc_s, wfc_ortho


def _native_nc_fixture():
    """Build QE's (uu, ud, du, dd) complex occupation and potential blocks."""
    ldim = 2
    u_eff = sp.Rational(3, 2)
    proj = sp.Matrix([1 + sp.I, 2 - sp.I, 1 - 2 * sp.I, -2 + sp.I])
    # new_ns_nc stores conjugate(proj[m1,s1]) * proj[m2,s2].
    occupation = (proj.conjugate() * proj.T).applyfunc(sp.expand)
    assert occupation == occupation.H

    spin_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    occupation_blocks = []
    for spin1, spin2 in spin_pairs:
        rows = [spin1 * ldim + m for m in range(ldim)]
        cols = [spin2 * ldim + m for m in range(ldim)]
        occupation_blocks.append(occupation.extract(rows, cols))

    # v_hubbard_nc swaps only the spin-flip component index and uses
    # ns(m2,m1,is1); alpha, J0, and beta are zero in this Dudarev fixture.
    potential_blocks = []
    for component, (spin1, spin2) in enumerate(spin_pairs):
        partner_component = 2 * spin2 + spin1
        block = -u_eff * occupation_blocks[partner_component].T
        if spin1 == spin2:
            block += (u_eff / 2) * sp.eye(ldim)
        potential_blocks.append(block.applyfunc(sp.expand))

    potential = sp.zeros(2 * ldim)
    for component, (spin1, spin2) in enumerate(spin_pairs):
        row0 = spin1 * ldim
        col0 = spin2 * ldim
        potential[row0 : row0 + ldim, col0 : col0 + ldim] = potential_blocks[component]

    return proj, occupation, u_eff, occupation_blocks, potential_blocks, potential


def _cross_matrix(vector):
    x, y, z = vector
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _rotation_matrix(axis, theta):
    generator = _cross_matrix(axis)
    return sp.eye(3) + sp.sin(theta) * generator + (1 - sp.cos(theta)) * generator**2


def _pauli_vector(vector):
    return sum((vector[i] * PAULI[i] for i in range(3)), sp.zeros(2))


def _simplify_matrix(matrix, trigonometric=False):
    if trigonometric:
        return matrix.applyfunc(lambda value: sp.trigsimp(sp.simplify(value)))
    return matrix.applyfunc(sp.simplify)


def _differentiate_at_zero(matrix, variable):
    return matrix.applyfunc(
        lambda value: sp.simplify(sp.diff(value, variable).subs(variable, 0))
    )


# MathDoc: tb2j/qe-hubbard-native-contract
# QE acts with W_U V_U W_U^dagger, where W_U=S Phi or S Phi O^(-1/2).
def check_wfcU_action():
    phi, overlap_operator, overlap, root_inverse, wfc_s, wfc_ortho = (
        _projection_fixture()
    )
    assert overlap == phi.H * overlap_operator * phi
    assert root_inverse * overlap * root_inverse == sp.eye(2)

    potential = sp.Matrix([[1, 2 + sp.I], [2 - sp.I, 3]])
    state = sp.Matrix([1 + sp.I, 2 - sp.I, 3 + 2 * sp.I])
    raw_action = wfc_s * potential * wfc_s.H * state
    ortho_action = wfc_ortho * potential * wfc_ortho.H * state

    assert _simplify_matrix(raw_action - wfc_s * (potential * (wfc_s.H * state))) == sp.zeros(3, 1)
    assert _simplify_matrix(ortho_action - wfc_ortho * (potential * (wfc_ortho.H * state))) == sp.zeros(3, 1)
    raw_operator = wfc_s * potential * wfc_s.H
    ortho_operator = wfc_ortho * potential * wfc_ortho.H
    wrong_ortho_wfc = wfc_s * root_inverse.T
    assert _simplify_matrix(raw_operator - phi * potential * phi.H) != sp.zeros(3)
    assert _simplify_matrix(raw_operator - wfc_s * potential * wfc_s.T) != sp.zeros(3)
    assert _simplify_matrix(raw_operator - wfc_s * potential.conjugate() * wfc_s.H) != sp.zeros(3)
    assert _simplify_matrix(raw_operator - ortho_operator) != sp.zeros(3)
    assert _simplify_matrix(ortho_operator - wrong_ortho_wfc * potential * wrong_ortho_wfc.H) != sp.zeros(3)
    assert wfc_ortho != wrong_ortho_wfc
    return {
        "wfcu_s_applied_action_probe": sp.expand(raw_action[0]),
        "wfcu_orthogonalized_action_probe": sp.expand(ortho_action[0]),
        "wfcu_inverse_overlap_root_imaginary_entry": root_inverse[0, 1],
    }


# MathDoc: tb2j/qe-hubbard-native-contract
# In spin-major/orbital-inner order, v_hub=U/2 I-U n^T; vhpsi_U applies
# that stored matrix directly, without another transpose or conjugation.
def check_native_spinor_blocks():
    _, occupation, u_eff, _, potential_blocks, potential = _native_nc_fixture()
    ldim = 2
    expected = (u_eff / 2) * sp.eye(2 * ldim) - u_eff * occupation.T
    wrong_no_transpose = (u_eff / 2) * sp.eye(2 * ldim) - u_eff * occupation
    wrong_component_order = sp.zeros(2 * ldim)
    wrong_sources = (0, 2, 3, 1)  # mistaken uu,dd,ud,du reading

    for component, (spin1, spin2) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        row0 = spin1 * ldim
        col0 = spin2 * ldim
        wrong_component_order[
            row0 : row0 + ldim, col0 : col0 + ldim
        ] = potential_blocks[wrong_sources[component]]

    assert potential == expected
    assert potential == potential.H
    assert potential != wrong_no_transpose
    assert potential != potential.conjugate()
    assert potential != wrong_component_order
    assert sp.expand(occupation[0, 1]) == 1 - 3 * sp.I
    assert sp.expand(potential[0, 1]) == -sp.Rational(3, 2) - sp.Rational(9, 2) * sp.I
    assert sp.expand(potential[0, 2]) == sp.Rational(3, 2) - sp.Rational(9, 2) * sp.I
    assert sp.expand(potential[2, 0] - sp.conjugate(potential[0, 2])) == 0

    # The native column action is W_U V_U W_U^dagger psi, with indices
    # (spin1-1)*ldim+m1 and component 2*(spin1-1)+spin2.
    _, _, _, _, wfc_s, wfc_ortho = _projection_fixture()
    wfcU_s = sp.kronecker_product(I2, wfc_s)
    wfcU_ortho = sp.kronecker_product(I2, wfc_ortho)
    pw_state = sp.Matrix(
        [1 + sp.I, 2 - sp.I, 3, 1 - 2 * sp.I, 2 + 3 * sp.I, -1 + sp.I]
    )
    action_operator = wfcU_s * potential * wfcU_s.H
    ortho_action_operator = wfcU_ortho * potential * wfcU_ortho.H
    action = action_operator * pw_state
    assert _simplify_matrix(action - wfcU_s * (potential * (wfcU_s.H * pw_state))) == sp.zeros(6, 1)
    assert _simplify_matrix(action_operator - wfcU_s * potential.conjugate() * wfcU_s.H) != sp.zeros(6)
    assert _simplify_matrix(action_operator - wfcU_s * potential * wfcU_s.T) != sp.zeros(6)
    assert _simplify_matrix(action_operator - ortho_action_operator) != sp.zeros(6)

    return {
        "native_potential_uu_complex_offdiagonal": potential[0, 1],
        "native_potential_ud_complex_offdiagonal": potential[0, 2],
        "native_potential_conjugation_defect": sp.expand(
            potential[0, 1] - sp.conjugate(potential[0, 1])
        ),
        "native_wfcu_action_probe": sp.expand(action[0]),
    }


# MathDoc: tb2j/qe-hubbard-native-contract
# Collinear main and background shells occupy disjoint acted column blocks;
# the full spinor main matrix embeds only the collinear main up/down blocks.
def check_collinear_shell_embedding():
    u_eff = sp.Rational(3, 2)
    u_back = sp.Rational(5, 4)
    n_up = sp.Matrix(
        [[sp.Rational(1, 3), sp.Rational(1, 5)], [sp.Rational(1, 5), sp.Rational(1, 4)]]
    )
    n_down = sp.Matrix(
        [[sp.Rational(1, 6), sp.Rational(1, 7)], [sp.Rational(1, 7), sp.Rational(1, 8)]]
    )

    def dudarev_potential(occupation, parameter):
        return (parameter / 2) * sp.eye(occupation.rows) - parameter * occupation.T

    v_up = dudarev_potential(n_up, u_eff)
    v_down = dudarev_potential(n_down, u_eff)
    collinear_main = sp.diag(v_up, v_down)
    nc_main_occupation = sp.diag(n_up, n_down)
    full_spinor_main = (u_eff / 2) * sp.eye(4) - u_eff * nc_main_occupation.T
    assert collinear_main == full_spinor_main

    # One main background block and the optional backall block per spin are
    # kept in separate column slices, matching offsetU_back/back1 selection.
    background_occupations = (
        sp.Rational(1, 3),
        sp.Rational(2, 5),
        sp.Rational(1, 5),
        sp.Rational(1, 4),
    )
    background_potentials = tuple(
        dudarev_potential(sp.Matrix([[value]]), u_back)
        for value in background_occupations
    )
    collinear_shell_columns = sp.diag(
        v_up,
        background_potentials[0],
        background_potentials[1],
        v_down,
        background_potentials[2],
        background_potentials[3],
    )
    main_indices = [0, 1, 4, 5]
    background_indices = [2, 3, 6, 7]
    assert collinear_shell_columns.extract(main_indices, main_indices) == collinear_main
    assert collinear_shell_columns.extract(
        main_indices, background_indices
    ) == sp.zeros(4, 4)
    assert collinear_shell_columns.extract(
        background_indices, background_indices
    ) != sp.zeros(4, 4)

    return {
        "collinear_main_spinor_embedding_entry": full_spinor_main[0, 0],
        "collinear_background_main_block_entry": background_potentials[0][0, 0],
        "collinear_main_background_cross_block": sp.Integer(0),
    }


# MathDoc: tb2j/qe-hubbard-native-contract
# The scalar charge/potential, S-applied projectors, S, and fixed SOC stay
# frozen; the native spin-column magnetic blocks alone carry the tangent.
def check_frozen_source_rotation_and_tangent():
    _, occupation, u_eff, _, _, potential = _native_nc_fixture()
    ldim = 2
    uu = potential.extract([0, 1], [0, 1])
    ud = potential.extract([0, 1], [2, 3])
    du = potential.extract([2, 3], [0, 1])
    dd = potential.extract([2, 3], [2, 3])

    scalar = _simplify_matrix((uu + dd) / 2)
    magnetic = (
        _simplify_matrix((ud + du) / 2),
        _simplify_matrix((du - ud) / (2 * sp.I)),
        _simplify_matrix((uu - dd) / 2),
    )
    occupation_charge = (
        occupation.extract([0, 1], [0, 1])
        + occupation.extract([2, 3], [2, 3])
    ) / 2
    assert scalar == (u_eff / 2) * sp.eye(ldim) - u_eff * occupation_charge.T
    assert all(block == block.H for block in (scalar, *magnetic))

    reconstructed = sp.kronecker_product(I2, scalar) + sum(
        (sp.kronecker_product(PAULI[i], magnetic[i]) for i in range(3)),
        sp.zeros(4),
    )
    assert _simplify_matrix(reconstructed - potential) == sp.zeros(4)

    theta = sp.symbols("theta", real=True)
    b = sp.symbols("b", real=True)
    reference = sp.Matrix([sp.Rational(3, 5), 0, sp.Rational(4, 5)])
    tangent_direction = sp.Matrix([0, 1, 0])
    axis = reference.cross(tangent_direction)
    assert reference.dot(reference) == 1
    assert tangent_direction.dot(tangent_direction) == 1
    assert reference.dot(tangent_direction) == 0
    assert axis.dot(axis) == 1

    generator = _cross_matrix(axis)
    rotation = _rotation_matrix(axis, theta)
    magnetic_blocks = list(magnetic)
    rotated_magnetic = [
        sum(
            (rotation[i, j] * magnetic_blocks[j] for j in range(3)),
            sp.zeros(ldim),
        )
        for i in range(3)
    ]
    rotated_potential = sp.kronecker_product(I2, scalar) + sum(
        (
            sp.kronecker_product(PAULI[i], rotated_magnetic[i])
            for i in range(3)
        ),
        sp.zeros(4),
    )
    tangent_blocks = [
        sum(
            (generator[i, j] * magnetic_blocks[j] for j in range(3)),
            sp.zeros(ldim),
        )
        for i in range(3)
    ]
    expected_tangent = sum(
        (sp.kronecker_product(PAULI[i], tangent_blocks[i]) for i in range(3)),
        sp.zeros(4),
    )
    potential_tangent = _differentiate_at_zero(rotated_potential, theta)
    assert rotated_potential.subs(theta, 0) == potential
    assert _simplify_matrix(potential_tangent - expected_tangent) == sp.zeros(4)
    assert _differentiate_at_zero(sp.kronecker_product(I2, scalar), theta) == sp.zeros(4)

    # The rational tilted axis gives a finite exact SU(2) realization as well
    # as the vector tangent dB/dtheta=(n x t) x B.
    orbital_source = sp.Matrix([[2, 1 + sp.I], [1 - sp.I, 3]])
    spin_rotation = sp.cos(theta / 2) * I2 - sp.I * sp.sin(theta / 2) * _pauli_vector(axis)
    spin_orbital_rotation = sp.kronecker_product(spin_rotation, I2)
    tilted_source = b * sp.kronecker_product(_pauli_vector(reference), orbital_source)
    rotated_tilted_source = (
        spin_orbital_rotation * tilted_source * spin_orbital_rotation.H
    )
    rotated_reference = rotation * reference
    expected_tilted_source = b * sp.kronecker_product(
        _pauli_vector(rotated_reference), orbital_source
    )
    assert _simplify_matrix(
        rotated_tilted_source - expected_tilted_source, trigonometric=True
    ) == sp.zeros(4)

    rotated_field = b * rotation * reference
    field_tangent = _differentiate_at_zero(rotated_field, theta)
    assert field_tangent == b * tangent_direction
    assert field_tangent == axis.cross(b * reference)

    # Action-level tangent keeps the same projectors and metric and adds a
    # fixed, Hermitian SOC-like operator that is not part of the rotated source.
    phi, overlap_operator, _, _, _, wfc_ortho = _projection_fixture()
    wfcU_ortho = sp.kronecker_product(I2, wfc_ortho)
    spatial_projectors = sp.kronecker_product(I2, phi)
    spin_metric = sp.kronecker_product(I2, overlap_operator)
    fixed_soc = sp.zeros(6)
    fixed_soc[0, 4] = sp.I / 3
    fixed_soc[4, 0] = -sp.I / 3
    fixed_soc[1, 3] = sp.Rational(2, 5)
    fixed_soc[3, 1] = sp.Rational(2, 5)
    assert fixed_soc == fixed_soc.H

    action_with_frozen_terms = fixed_soc + wfcU_ortho * rotated_potential * wfcU_ortho.H
    action_tangent = _differentiate_at_zero(action_with_frozen_terms, theta)
    assert _simplify_matrix(
        action_tangent - wfcU_ortho * expected_tangent * wfcU_ortho.H
    ) == sp.zeros(6)
    assert _differentiate_at_zero(fixed_soc, theta) == sp.zeros(6)
    assert _differentiate_at_zero(wfcU_ortho, theta) == sp.zeros(6, 4)
    assert _differentiate_at_zero(spatial_projectors, theta) == sp.zeros(6, 4)
    assert _differentiate_at_zero(spin_metric, theta) == sp.zeros(6)

    return {
        "native_scalar_charge_potential_probe": scalar[0, 0],
        "native_tilted_source_tangent_y": field_tangent[1],
        "native_fixed_soc_tangent_probe": sp.diff(fixed_soc[0, 4], theta),
        "native_frozen_action_tangent_probe": action_tangent[0, 0],
    }


# MathDoc: tb2j/qe-hubbard-native-contract
# A fixed spin-orbital map need not intertwine source-column and output-spin
# rotations, so a full-Hamiltonian commutator is not a native-source proof.
def check_fixed_spin_orbital_map_non_covariance():
    spin_rotation = (I2 - sp.I * SY) / sp.sqrt(2)
    spin_orbital_rotation = sp.kronecker_product(spin_rotation, I2)
    fixed_map = sp.diag(1, 1, 1, -1)  # controlled phase in spin/orbital order
    source = sp.kronecker_product(SZ, I2)

    source_rotated_first = spin_orbital_rotation * source * spin_orbital_rotation.H
    map_after_source_rotation = fixed_map * source_rotated_first * fixed_map.H
    map_before_output_rotation = (
        spin_orbital_rotation * (fixed_map * source * fixed_map.H)
        * spin_orbital_rotation.H
    )
    covariance_defect = _simplify_matrix(
        map_after_source_rotation - map_before_output_rotation
    )
    assert spin_rotation * SZ * spin_rotation.H == SX
    assert covariance_defect != sp.zeros(4)
    assert covariance_defect[1, 3] == -2

    return {"fixed_spin_orbital_map_covariance_defect": covariance_defect[1, 3]}


# MathDoc: tb2j/qe-hubbard-native-contract
# The declared frame comes from the original XC source, or an explicitly
# mapped U direction when XC vanishes; it is never inferred from XC+U.
def check_declared_reference_frame_independence():
    b = sp.symbols("b", positive=True)
    u_scale = sp.symbols("u_scale", real=True)
    original_xc_source = sp.Matrix([0, 0, b])
    total_source = original_xc_source * (1 + u_scale)
    xc_reference = original_xc_source / sp.sqrt(original_xc_source.dot(original_xc_source))
    cancelled_total = total_source.subs(u_scale, -1)
    assert cancelled_total == sp.zeros(3, 1)
    assert xc_reference.dot(xc_reference) == 1
    # The direction from the original XC source survives cancellation and
    # does not flip when the combined magnetic source reverses.
    reversed_total = total_source.subs(u_scale, -2)
    reversed_direction = reversed_total / sp.sqrt(reversed_total.dot(reversed_total))
    assert _simplify_matrix(reversed_direction + xc_reference) == sp.zeros(3, 1)
    assert _differentiate_at_zero(xc_reference, u_scale) == sp.zeros(3, 1)

    # A nontrivial declared column-to-reference map is necessary for the
    # zero-XC fallback. This toy proper rotation is not a measured QE map.
    column_u_source = sp.Matrix([b, 0, 0])
    column_to_reference = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert column_to_reference.T * column_to_reference == sp.eye(3)
    assert column_to_reference.det() == 1
    mapped_u_source = column_to_reference * column_u_source
    mapped_u_reference = mapped_u_source / sp.sqrt(mapped_u_source.dot(mapped_u_source))
    assert mapped_u_reference.dot(mapped_u_reference) == 1
    assert mapped_u_reference != column_u_source / b

    return {
        "cancelled_xc_u_total_z": cancelled_total[2],
        "independent_xc_reference_z": xc_reference[2],
        "explicit_u_only_reference_y": mapped_u_reference[1],
    }


CHECKS = (
    ("native_wfcU_action", check_wfcU_action),
    ("native_spinor_blocks_and_conjugation", check_native_spinor_blocks),
    ("collinear_main_and_background_embeddings", check_collinear_shell_embedding),
    ("frozen_source_rotation_and_tangent", check_frozen_source_rotation_and_tangent),
    ("fixed_spin_orbital_non_covariance_control", check_fixed_spin_orbital_map_non_covariance),
    ("declared_reference_frame_independence", check_declared_reference_frame_independence),
)


def main() -> dict[str, sp.Expr]:
    """Run exact assertions and return named expressions for the adapter."""
    results: dict[str, sp.Expr] = {}
    for _, check in CHECKS:
        results.update(check())
    return results


if __name__ == "__main__":
    main()
    for check_name, _ in CHECKS:
        print(f"PASS: {check_name}")
