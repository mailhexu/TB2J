"""Assertion authority for story 008: frozen total-H magnetic response.

Run in mydev: python docs/sympy/siesta_spiral_frozen_response.py
Paired assumptions, signs, metric rules and derivation: same-name .md file.
No production code is imported. Exact assertions never fall back to probes.

Revision 3 (review STD-1/STD-2 plus the declared joint-action convention):
the two-site model uses the story-005 incident partitions with the
TWO-SIDED SAME-U declaration applied per incident site. A single pass p
rotates every p-incident block two-sided, W -> U_p W U_p^dag; the joint
perturbation composes the passes in the declared site order (A first, then
B), so a shared AB/BA block receives the ordered composition
W -> U_B U_A W U_A^dag U_B^dag. This is a per-block conjugation (Hermitian
for arbitrary per-site end rotations), it is NOT the block-diagonal
site rotation diag(U_A,U_B) (whose per-side sandwich is not even
Hermitian; asserted), and on shared blocks it differs from the
single-site actions of either end. First vertices, one-site kernels and
the one-site torque are re-derived on this path; the mixed second vertices
carry the composition. The one-site/two-site hierarchy is built from
independent single-site and joint constructions with its conditional
equality assumption stated and asserted; a spin-textured overlap
counter-model breaks the identification.
"""

import hashlib

import numpy as np
import scipy.linalg as la
import sympy as sp

I = sp.I
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -I], [I, 0]])
SZ = sp.diag(1, -1)
PAULI = (SX, SY, SZ)
HALF = sp.Rational(1, 2)


def zero(value):
    """Require exact symbolic equality, entrywise for a matrix."""
    entries = list(value) if isinstance(value, sp.MatrixBase) else [value]
    for entry in entries:
        reduced = (
            sp.trigsimp(sp.expand(entry))
            if entry.has(sp.sin, sp.cos)
            else sp.cancel(entry)
        )
        assert reduced == 0, reduced


def any_nonzero(matrix):
    return any(sp.simplify(entry) != 0 for entry in matrix)


def numeric(matrix):
    return np.array(matrix.evalf(), dtype=complex)


def residue_sum(expression, poles, variable):
    expression = sp.cancel(expression)
    return sp.simplify(sum(sp.residue(expression, variable, pole) for pole in poles))


def rotation(axis, angle):
    return sp.cos(angle / 2) * sp.eye(2) - I * sp.sin(angle / 2) * axis


def commutator(a, b):
    return a * b - b * a


def decompose(matrix):
    """q=0 shadow: scalar spin trace and traceless total-H magnetic part."""
    scalar = sp.zeros(4)
    for a in range(2):
        for b in range(2):
            block = matrix[2 * a : 2 * a + 2, 2 * b : 2 * b + 2]
            scalar[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] = (
                sp.trace(block) / 2 * sp.eye(2)
            )
    return scalar, matrix - scalar


def incident(site, magnetic):
    """Story-005 incidence: a block belongs to `site` when a side is on it."""
    out = sp.zeros(4)
    for r in range(4):
        for c in range(4):
            if r // 2 == site or c // 2 == site:
                out[r, c] = magnetic[r, c]
    return out


def incident_blocks(site):
    return [(i, j) for i in range(2) for j in range(2) if i == site or j == site]


def site_generator(axis, site):
    """Half-Pauli generator acting on `site` only (identity elsewhere)."""
    blocks = [sp.zeros(2), sp.zeros(2)]
    blocks[site] = axis / 2
    return sp.Matrix(sp.diag(*blocks))


def pass_two_sided(site, matrix, u):
    """Story-005 pass: two-sided same-U on every `site`-incident block.

    Non-incident blocks are passed through untouched.
    """
    out = sp.Matrix(matrix)
    for i, j in incident_blocks(site):
        blk = out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = u * blk * u.H
    return out


def phi_pass(site, half_axis, matrix):
    """First-order generator of the two-sided same-U pass (blockwise)."""
    out = sp.zeros(4)
    for i, j in incident_blocks(site):
        blk = matrix[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = -I * (
            half_axis * blk - blk * half_axis
        )
    return out


def comm_pass(site, half_axis, matrix):
    """Blockwise commutator [t, .] on every `site`-incident block."""
    out = sp.zeros(4)
    for i, j in incident_blocks(site):
        blk = matrix[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
        out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = half_axis * blk - blk * half_axis
    return out


def vertex_of(site, axis, inc):
    """V = dH/dtheta on site `site`, blockwise two-sided on shared bonds."""
    return phi_pass(site, axis / 2, inc)


def seagull_same_site(site, axis_a, axis_b, inc):
    """Second vertex of one site's exponential chart (two-sided passes)."""
    ta, tb = axis_a / 2, axis_b / 2
    return -HALF * (
        comm_pass(site, ta, comm_pass(site, tb, inc))
        + comm_pass(site, tb, comm_pass(site, ta, inc))
    )


def seagull_mixed_ordered(site_a, axis_a, site_b, axis_b, inc_a, magnetic_full):
    """Mixed second vertex of the ordered joint action (A pass, then B).

    d2H/dtheta_a dtheta_b = phi_B( V_A restricted to the B-incident
    blocks ) = -[t_b,[t_a,W]] on the shared AB/BA blocks; it is a single
    object (Clairaut), so the (b,a) Hessian entry reuses this matrix.
    """
    v_a = vertex_of(site_a, axis_a, inc_a)
    return phi_pass(site_b, axis_b / 2, v_a)


def seagull_mixed_symmetric(site_a, axis_a, site_b, axis_b, magnetic_full):
    """Unordered (symmetrized) reference form, for contrast assertions."""
    ta, tb = axis_a / 2, axis_b / 2
    return -HALF * (
        comm_pass(site_a, ta, comm_pass(site_b, tb, magnetic_full))
        + comm_pass(site_b, tb, comm_pass(site_a, ta, magnetic_full))
    )


def main():
    print(f"SymPy {sp.__version__}; frozen response, exact algebra + numerical checks")

    # [A] Differentiate actual SU(2) rotations, not a transcribed vertex.
    theta = sp.symbols("theta", real=True)
    bx, by, bz = sp.symbols("b_x b_y b_z", real=True)
    magnetic = bx * SX + by * SY + bz * SZ
    for sigma in PAULI:
        u = rotation(sigma, theta)
        rotated = u * magnetic * u.H
        v = -I * commutator(sigma / 2, magnetic)
        w = -commutator(sigma / 2, commutator(sigma / 2, magnetic))
        zero(rotated.diff(theta).subs(theta, 0) - v)
        zero(rotated.diff(theta, 2).subs(theta, 0) - w)
        zero(sp.trace(rotated) - sp.trace(magnetic))
        zero(u * u.H - sp.eye(2))
    zero(-I * commutator(SY / 2, bz * SZ) - bz * SX)
    zero(-commutator(SY / 2, commutator(SY / 2, bz * SZ)) + bz * SZ)

    # Exponential chart U=exp[-i(x sigma_x+y sigma_y)/2], through order two.
    x, y = sp.symbols("x y", real=True)
    generator = (x * SX + y * SY) / 2
    u2 = sp.eye(2) - I * generator - generator**2 / 2
    rotated2 = u2 * magnetic * u2.H
    wxy = -HALF * (
        commutator(SX / 2, commutator(SY / 2, magnetic))
        + commutator(SY / 2, commutator(SX / 2, magnetic))
    )
    zero(rotated2.diff(x, y).subs({x: 0, y: 0}) - wxy)
    zero(
        rotated2.diff(x, 2).subs({x: 0, y: 0})
        + commutator(SX / 2, commutator(SX / 2, magnetic))
    )
    print("[A] SU(2) first, second and mixed derivatives; positive y sends +z to +x")

    # [B] Independent analytic eigenvalue check of perturbation denominators.
    a = sp.symbols("a", real=True)
    v11, v22, wr11 = sp.symbols("v11 v22 w11", real=True)
    vr, vi, t = sp.symbols("vr vi t", real=True)
    gap = sp.symbols("gap", positive=True)
    h2 = sp.Matrix(
        [
            [a + t * v11 + t * t * wr11 / 2, t * (vr - I * vi)],
            [t * (vr + I * vi), a + gap + t * v22],
        ]
    )
    lower = (
        sp.trace(h2) - sp.sqrt((h2[0, 0] - h2[1, 1]) ** 2 + 4 * h2[0, 1] * h2[1, 0])
    ) / 2
    zero(lower.diff(t).subs(t, 0) - v11)
    zero(lower.diff(t, 2).subs(t, 0) - (wr11 - 2 * (vr**2 + vi**2) / gap))
    # Characteristic-polynomial differentiation is a separate rearrangement.
    energy = sp.Function("energy")(t)
    determinant = (h2 - energy * sp.eye(2)).det()
    substitutions = {
        energy.subs(t, 0): a,
        sp.diff(energy, t).subs(t, 0): v11,
        sp.diff(energy, t, 2).subs(t, 0): wr11 - 2 * (vr**2 + vi**2) / gap,
    }
    zero(determinant.diff(t).subs(t, 0).subs(substitutions))
    zero(determinant.diff(t, 2).subs(t, 0).subs(substitutions))
    # Assert the central and Richardson convergence orders symbolically.
    derivatives = sp.symbols("e0:7", real=True)
    polynomial = sum(derivatives[n] * t**n / sp.factorial(n) for n in range(7))
    h = sp.symbols("h", positive=True)
    central_gradient = (polynomial.subs(t, h) - polynomial.subs(t, -h)) / (2 * h)
    central_hessian = (
        polynomial.subs(t, h) - 2 * polynomial.subs(t, 0) + polynomial.subs(t, -h)
    ) / h**2
    zero(
        central_gradient
        - derivatives[1]
        - derivatives[3] * h**2 / 6
        - derivatives[5] * h**4 / 120
    )
    zero(
        central_hessian
        - derivatives[2]
        - derivatives[4] * h**2 / 12
        - derivatives[6] * h**4 / 360
    )
    richardson_g = (4 * central_gradient.subs(h, h / 2) - central_gradient) / 3
    richardson_k = (4 * central_hessian.subs(h, h / 2) - central_hessian) / 3
    zero(richardson_g - derivatives[1] + derivatives[5] * h**4 / 480)
    zero(richardson_k - derivatives[2] + derivatives[6] * h**4 / 1440)
    print(
        "[B] Eigenvalue and characteristic-polynomial derivatives agree, including factor two"
    )

    # [C1] The declared joint action: ordered two-sided same-U passes.
    # A single pass rotates every incident block two-sided with the same U
    # (story-005 certificate); the joint perturbation composes the passes in
    # the declared site order, A first then B. Quantified on shared blocks:
    z_a, y_b = sp.symbols("z_A y_B", real=True)
    # Commuting axis: rotating A about z leaves b_z sigma_z unchanged.
    zero(rotation(SZ, z_a) * (bz * SZ) * rotation(SZ, z_a).H - bz * SZ)
    # Non-commuting example W = sigma_x: the single-site A action is the
    # vector rotation R_z; the ordered joint action is R_y R_z composed; the
    # per-side sandwich is neither and is not even Hermitian.
    single_a = rotation(SZ, z_a) * SX * rotation(SZ, z_a).H
    zero(single_a - (sp.cos(z_a) * SX + sp.sin(z_a) * SY))
    ordered_block = (
        rotation(SY, y_b)
        * rotation(SZ, z_a)
        * SX
        * rotation(SZ, z_a).H
        * rotation(SY, y_b).H
    )
    zero(
        ordered_block
        - (
            sp.cos(z_a) * sp.cos(y_b) * SX
            + sp.sin(z_a) * SY
            - sp.cos(z_a) * sp.sin(y_b) * SZ
        )
    )
    per_side_block = rotation(SZ, z_a) * SX * rotation(SY, y_b).H
    assert any_nonzero(per_side_block - per_side_block.H)
    assert any_nonzero(ordered_block - per_side_block)
    assert any_nonzero(single_a - per_side_block)
    print(
        "[C1] Ordered two-sided joint action: Hermitian conjugation per block; "
        "per-side sandwich is not (half-angle, non-Hermitian)"
    )

    # [C2] Exact two-site, one spinor orbital per site, gapped reference.
    # The whitening factor is site-mixed (A and B orbitals couple with
    # different amplitudes) but spin-scalar, matching the assembled lab
    # overlap of the pinned q=0 shadow; exact eigenvectors stay rational.
    orbital_l = sp.Matrix([[1, sp.Rational(1, 4)], [sp.Rational(1, 6), 1]])
    l = sp.kronecker_product(orbital_l, sp.eye(2))
    overlap = l.T * l
    householder_vector = sp.Matrix([1, 2, 3, 4])
    q = sp.eye(4) - 2 * householder_vector * householder_vector.T / 30
    eigenvalues = (-3, -1, 2, 4)
    diagonal = sp.diag(*eigenvalues)
    coefficients = l.inv() * q
    h0 = l.T * q * diagonal * q.T * l
    scalar, full_magnetic = decompose(h0)
    b_inc = [incident(0, full_magnetic), incident(1, full_magnetic)]
    zero(q.T * q - sp.eye(4))
    zero(coefficients.H * overlap * coefficients - sp.eye(4))
    zero(h0 * coefficients - overlap * coefficients * diagonal)
    zero(h0 - scalar - full_magnetic)
    assert all(minor > 0 for minor in [overlap[:n, :n].det() for n in range(1, 5)])
    assert eigenvalues[1] < 0 < eigenvalues[2]
    # Incidence inventory (review STD-1): shared AB/BA blocks in BOTH sets.
    zero(b_inc[0][2:4, 2:4])
    zero(b_inc[1][:2, :2])
    shared = b_inc[0] + b_inc[1] - full_magnetic
    zero(shared[:2, :2])
    zero(shared[2:4, 2:4])
    assert any_nonzero(shared[:2, 2:4])
    assert any_nonzero(b_inc[1][:2, 2:4])

    # Coordinate order: Ax, Ay, Bx, By. Vertices and seagulls are derivatives
    # of the ordered two-sided production path; S and H_scalar are frozen.
    vertices, seagulls = [], {}
    for site in range(2):
        for axis in (SX, SY):
            vertices.append(vertex_of(site, axis, b_inc[site]))
    for v in vertices[:2]:
        zero(v[2:4, 2:4])  # site-A vertex has no BB support
    for v in vertices[2:]:
        zero(v[:2, :2])  # site-B vertex has no AA support
    for v in vertices:
        zero(v - v.H)
    for aindex in range(4):
        for bindex in range(4):
            site_a, site_b = aindex // 2, bindex // 2
            axis_a, axis_b = (SX, SY)[aindex % 2], (SX, SY)[bindex % 2]
            if site_a == site_b:
                w = seagull_same_site(site_a, axis_a, axis_b, b_inc[site_a])
            elif site_a == 0:
                w = seagull_mixed_ordered(
                    site_a, axis_a, site_b, axis_b, b_inc[site_a], full_magnetic
                )
            else:
                # Clairaut: one mixed object per axis pair; the (b,a) entry
                # reuses the (a,b) vertex computed with the declared order.
                w = seagulls[bindex, aindex]
            seagulls[aindex, bindex] = sp.expand(w)
    for aindex in range(4):
        for bindex in range(4):
            zero(seagulls[aindex, bindex] - seagulls[aindex, bindex].H)
            zero(seagulls[aindex, bindex] - seagulls[bindex, aindex])
    # Shared-bond mixed seagulls: nonzero; equal to the symmetrized form
    # exactly when the end generators commute (same axis); the mixed-axis
    # ordered vertex differs from the symmetrized reference form.
    assert any_nonzero(seagulls[1, 3])
    assert any_nonzero(seagulls[0, 3])
    zero(seagulls[1, 3] - seagull_mixed_symmetric(0, SY, 1, SY, full_magnetic))
    zero(seagulls[0, 2] - seagull_mixed_symmetric(0, SX, 1, SX, full_magnetic))
    sym03 = seagull_mixed_symmetric(0, SX, 1, SY, full_magnetic)
    assert any_nonzero(sym03 - seagulls[0, 3])

    # Single-site paths (theta_other = 0) reproduce V and W_pp.
    for site in range(2):
        u = rotation(SY, theta)
        path = pass_two_sided(site, scalar + full_magnetic, u)
        zero(path.diff(theta).subs(theta, 0) - vertices[2 * site + 1])
        zero(path.diff(theta, 2).subs(theta, 0) - seagulls[2 * site + 1, 2 * site + 1])
        zero(path - path.H)
        zero(pass_two_sided(site, scalar, u) - scalar)
    # Joint ordered path reproduces every vertex kind through order two.
    t_a, t_b = sp.symbols("t_A t_B", real=True)
    joint_ordered = pass_two_sided(
        1,
        pass_two_sided(0, scalar + full_magnetic, rotation(SY, t_a)),
        rotation(SY, t_b),
    )
    zero(joint_ordered.diff(t_a).subs({t_a: 0, t_b: 0}) - vertices[1])
    zero(joint_ordered.diff(t_b).subs({t_a: 0, t_b: 0}) - vertices[3])
    zero(joint_ordered.diff(t_a, 2).subs({t_a: 0, t_b: 0}) - seagulls[1, 1])
    zero(joint_ordered.diff(t_b, 2).subs({t_a: 0, t_b: 0}) - seagulls[3, 3])
    zero(joint_ordered.diff(t_a, t_b).subs({t_a: 0, t_b: 0}) - seagulls[1, 3])
    zero(joint_ordered - joint_ordered.H)
    # The ordered composition is NOT the block-diagonal site rotation.
    joint_diag = (
        scalar
        + sp.diag(rotation(SY, t_a), rotation(SY, t_b))
        * full_magnetic
        * sp.diag(rotation(SY, t_a), rotation(SY, t_b)).H
    )
    assert any_nonzero(joint_ordered - joint_diag)
    print(
        "[C] Ordered two-sided passes on incident partitions; vertices, seagulls, "
        "single-site and joint paths asserted"
    )

    # [D] Pencil identities including an assembled spin-dependent metric probe.
    z = sp.symbols("z")
    green = (
        coefficients * (z * sp.eye(4) - diagonal).inv() * coefficients.H
    ).applyfunc(sp.cancel)
    mixed_resolvent = (green * overlap).applyfunc(sp.cancel)
    zero((z * overlap - h0) * green - sp.eye(4))
    zero(green * (z * overlap - h0) - sp.eye(4))
    zero((z * sp.eye(4) - overlap.inv() * h0) * mixed_resolvent - sp.eye(4))
    zero(green.diff(z) + green * overlap * green)
    zero(sp.trace(overlap * green) - sum(1 / (z - e) for e in eigenvalues))
    # General metric, not necessarily a scalar in spin space, via congruence.
    transform = sp.Matrix(
        [
            [1, sp.Rational(1, 5), 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, sp.Rational(1, 7)],
            [0, 0, 0, 1],
        ]
    )
    spin_s = transform.H * overlap * transform
    spin_h = transform.H * h0 * transform
    spin_g = transform.inv() * green * transform.H.inv()
    zero((z * spin_s - spin_h) * spin_g - sp.eye(4))
    zero(sp.trace(spin_s * spin_g) - sp.trace(overlap * green))
    for v in vertices:
        zero(sp.trace(mixed_resolvent * overlap.inv() * v) - sp.trace(green * v))
        spin_v = transform.H * v * transform
        zero(sp.trace(spin_g * spin_v) - sp.trace(green * v))
    zero(
        sp.trace(green.diff(z) * vertices[1])
        + sp.trace(overlap * green * vertices[1] * green)
    )
    print(
        "[D] S-weighted spectral trace, right-hand S^-1, covariant insertions and spin-dependent metric"
    )

    # [E] Differentiate the inverse pencil explicitly with a symbolic path.
    v, w = vertices[1], seagulls[1, 1]
    gv = [(green * vertex).applyfunc(sp.cancel) for vertex in vertices]
    g1 = (gv[1] * green).applyfunc(sp.cancel)
    g2 = (
        (green * w).applyfunc(sp.cancel) * green
        + 2 * (gv[1] * gv[1]).applyfunc(sp.cancel) * green
    ).applyfunc(sp.cancel)
    # Derivatives of the identity A(t)G(t)=I yield both response terms.
    pencil = z * overlap - h0 - t * v - t * t * w / 2
    gseries = green + t * g1 + t * t * g2 / 2
    product = pencil * gseries
    zero(product.subs(t, 0) - sp.eye(4))
    zero(product.diff(t).subs(t, 0))
    zero(product.diff(t, 2).subs(t, 0))
    # A two-parameter inverse series independently asserts the mixed terms.
    s = sp.symbols("s", real=True)
    va, vb, wab = vertices[0], vertices[1], seagulls[0, 1]
    ga = (gv[0] * green).applyfunc(sp.cancel)
    mixed_g = (
        (green * wab).applyfunc(sp.cancel) * green
        + (gv[0] * gv[1] + gv[1] * gv[0]).applyfunc(sp.cancel) * green
    ).applyfunc(sp.cancel)
    mixed_product = (z * overlap - h0 - t * va - s * vb - t * s * wab) * (
        green + t * ga + s * g1 + t * s * mixed_g
    )
    zero(mixed_product.diff(t, s).subs({t: 0, s: 0}))
    # Integration-by-parts derivatives prove the band and grand-energy signs.
    mu = sp.symbols("mu", real=True)
    contraction = sp.cancel(sp.trace(green * w + gv[1] * gv[1]))
    zero(sp.diff(sp.trace(gv[1]), z) + sp.trace(overlap * g1))
    zero(sp.diff(contraction, z) + sp.trace(overlap * g2))
    occupied = eigenvalues[:2]
    for weight in (z, z - mu):
        zero(
            residue_sum(weight * sp.trace(overlap * g1), occupied, z)
            - residue_sum(sp.trace(gv[1]), occupied, z)
        )
        zero(
            residue_sum(weight * sp.trace(overlap * g2), occupied, z)
            - residue_sum(contraction, occupied, z)
        )
    print("[E] Frozen band/grand-energy derivatives and exact resolvent response signs")

    # [F] Residue kernel equals the gapped divided difference, including repeated poles.
    occupations = (1, 1, 0, 0)
    kernel = sp.zeros(4)
    for n in range(4):
        for m in range(4):
            contour = residue_sum(
                1 / ((z - eigenvalues[n]) * (z - eigenvalues[m])), occupied, z
            )
            expected = (
                0
                if n == m
                else sp.Rational(
                    occupations[n] - occupations[m], eigenvalues[n] - eigenvalues[m]
                )
            )
            zero(contour - expected)
            kernel[n, m] = expected
    eigen_v = [coefficients.H * v * coefficients for v in vertices]
    gradient = sp.Matrix(
        [sum(occupations[n] * v[n, n] for n in range(4)) for v in eigen_v]
    )
    onsite, response = sp.zeros(4), sp.zeros(4)
    for aindex in range(4):
        zero(
            residue_sum(sp.trace(green * vertices[aindex]), occupied, z)
            - gradient[aindex]
        )
        for bindex in range(4):
            eigen_w = coefficients.H * seagulls[aindex, bindex] * coefficients
            onsite[aindex, bindex] = sum(
                occupations[n] * eigen_w[n, n] for n in range(4)
            )
            response[aindex, bindex] = sp.simplify(
                sum(
                    kernel[n, m] * eigen_v[aindex][n, m] * eigen_v[bindex][m, n]
                    for n in range(4)
                    for m in range(4)
                )
            )
            zero(
                residue_sum(sp.trace(gv[aindex] * gv[bindex]), occupied, z)
                - response[aindex, bindex]
            )
            zero(
                residue_sum(sp.trace(green * seagulls[aindex, bindex]), occupied, z)
                - onsite[aindex, bindex]
            )
    curvature = onsite + response
    zero(curvature - curvature.T)
    assert sp.simplify(gradient[1]) != 0
    assert sp.simplify(gradient[3]) != 0
    assert sp.simplify(onsite[1, 1]) != 0
    assert sp.simplify(response[1, 1]) != 0
    assert sp.simplify(onsite[1, 3]) != 0
    assert sp.simplify(response[1, 3]) != 0
    assert sp.simplify(onsite[3, 3]) != 0
    print(f"[F] gradient = {gradient.T}, residual checks follow in [G2]")
    print(f"[F] K = {curvature}")
    print(
        "[F] Incident-partition torques and full 16-entry Hessian: exact contour/eigenbasis equivalence"
    )

    # [G1] Nonstationary counterexample under the two-sided convention:
    # ordered local x,z rotations on A versus the declared exponential chart.
    x, zz = sp.symbols("x zz", real=True)
    ux = rotation(SX, x)
    uz = rotation(SZ, zz)
    ordered = scalar + pass_two_sided(0, pass_two_sided(0, full_magnetic, uz), ux)
    ordered_w = ordered.diff(x, zz).subs({x: 0, zz: 0})
    generator_xz2 = x * SX / 2 + zz * SZ / 2
    symmetric_u2 = sp.eye(2) - I * generator_xz2 - generator_xz2**2 / 2
    symmetric_path2 = scalar + pass_two_sided(0, full_magnetic, symmetric_u2)
    symmetric_w = symmetric_path2.diff(x, zz).subs({x: 0, zz: 0})
    for coordinate in (x, zz):
        zero(
            ordered.diff(coordinate).subs({x: 0, zz: 0})
            - symmetric_path2.diff(coordinate).subs({x: 0, zz: 0})
        )
    zero(ordered_w - symmetric_w + vertices[1] / 2)
    density = coefficients[:, :2] * coefficients[:, :2].H
    zero(sp.trace(density * (ordered_w - symmetric_w)) + gradient[1] / 2)
    assert sp.trace(density * (ordered_w - symmetric_w)) != 0
    print(
        f"[G1] Nonstationary g_Ay={gradient[1]}, torque=-g_Ay; chart correction={-gradient[1]/2}"
    )

    # [G2] One-site/two-site hierarchy (review STD-2) on the ordered path.
    # One-site kernels come from the single-site paths; the cross kernel from
    # the joint ordered action. Under the ordered two-sided convention the
    # common-angle shift is NOT a conjugation of the joint path (asserted),
    # so no symmetry forces kappa_A = -K_AB = kappa_B; the identification is
    # a conditional property requiring the shared-bond composition to act as
    # a single rotation, which the declared two-sided composition does not.
    # All three misfits are computed nonzeros, never dropped.
    shift = sp.Symbol("shift", real=True)
    u_shift = sp.diag(rotation(SY, shift), rotation(SY, shift))
    t_p, t_q = sp.symbols("t_p t_q", real=True)
    joint_y = pass_two_sided(
        1, pass_two_sided(0, full_magnetic, rotation(SY, t_p)), rotation(SY, t_q)
    )
    joint_y_shifted = pass_two_sided(
        1,
        pass_two_sided(0, full_magnetic, rotation(SY, t_p + shift)),
        rotation(SY, t_q + shift),
    )
    assert any_nonzero(u_shift * joint_y * u_shift.H - joint_y_shifted)
    kappa_a, kappa_b, cross_k = curvature[1, 1], curvature[3, 3], curvature[1, 3]
    sum_residual = sp.simplify(kappa_a + 2 * cross_k + kappa_b)
    assert sp.simplify(kappa_a + cross_k) != 0
    assert sp.simplify(kappa_b + cross_k) != 0
    assert sum_residual != 0
    assert sp.simplify(kappa_a - kappa_b) != 0
    print(
        f"[G2] primary: kappa_A={kappa_a}, K_AB={cross_k}, kappa_B={kappa_b}, "
        f"sum-rule residual={sum_residual} (one-site identification fails)"
    )

    def spin_block_scalar(matrix):
        out = sp.zeros(4)
        for a in range(2):
            for b in range(2):
                blk = matrix[2 * a : 2 * a + 2, 2 * b : 2 * b + 2]
                out[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] = (
                    sp.trace(blk) / 2 * sp.eye(2)
                )
        return out

    zero(overlap - spin_block_scalar(overlap))
    # Counter-model: an assembled spin-textured frozen overlap. The
    # identification fails; the one-site misfit is a computed nonzero.
    orbital_t = sp.Matrix(
        [
            [1, sp.Rational(1, 4), 0, 0],
            [0, 1, 0, sp.Rational(1, 5)],
            [0, 0, 1, sp.Rational(1, 6)],
            [0, sp.Rational(1, 7), 0, 1],
        ]
    )
    overlap_t = orbital_t.T * orbital_t
    coefficients_t = orbital_t.inv() * q
    h0_t = orbital_t.T * q * diagonal * q.T * orbital_t
    scalar_t, magnetic_t = decompose(h0_t)
    zero(coefficients_t.H * overlap_t * coefficients_t - sp.eye(4))
    assert any_nonzero(overlap_t - spin_block_scalar(overlap_t))
    green_t = (
        coefficients_t * (z * sp.eye(4) - diagonal).inv() * coefficients_t.H
    ).applyfunc(sp.cancel)
    inc_t = [incident(0, magnetic_t), incident(1, magnetic_t)]
    v_ay_t = vertex_of(0, SY, inc_t[0])
    v_by_t = vertex_of(1, SY, inc_t[1])
    g_ay_t = residue_sum(sp.trace(green_t * v_ay_t), occupied, z)
    g_by_t = residue_sum(sp.trace(green_t * v_by_t), occupied, z)
    assert sp.simplify(g_ay_t + g_by_t) != 0
    w_ay_t = seagull_same_site(0, SY, SY, inc_t[0])
    kappa_a_t = residue_sum(sp.trace(green_t * w_ay_t), occupied, z) + residue_sum(
        sp.trace(green_t * v_ay_t * green_t * v_ay_t), occupied, z
    )
    w_by_t = seagull_same_site(1, SY, SY, inc_t[1])
    kappa_b_t = residue_sum(sp.trace(green_t * w_by_t), occupied, z) + residue_sum(
        sp.trace(green_t * v_by_t * green_t * v_by_t), occupied, z
    )
    w_cross_t = seagull_mixed_ordered(0, SY, 1, SY, inc_t[0], magnetic_t)
    cross_t = residue_sum(sp.trace(green_t * w_cross_t), occupied, z) + residue_sum(
        sp.trace(green_t * v_ay_t * green_t * v_by_t), occupied, z
    )
    assert sp.simplify(kappa_a_t - kappa_b_t) != 0
    assert sp.simplify(kappa_a_t + 2 * cross_t + kappa_b_t) != 0
    assert sp.simplify(cross_t + kappa_a_t) != 0
    assert sp.simplify(cross_t + kappa_b_t) != 0
    # Conditional assumption: the mixed seagull vanishes exactly when the
    # shared-bond splitting is zero, while the response coupling survives.
    variant = sp.Matrix(full_magnetic)
    variant[:2, 2:] = sp.zeros(2)
    variant[2:, :2] = sp.zeros(2)
    variant_inc = [incident(0, variant), incident(1, variant)]
    zero(phi_pass(1, SY / 2, phi_pass(0, SY / 2, variant)))
    variant_num = numeric(scalar + variant)
    evals_v, c_v = la.eigh(variant_num, numeric(overlap))
    # T=0 gapped ensemble: the two lowest eigenstates are occupied and are
    # separated from the empty ones (the split need not sit at zero energy).
    assert evals_v[1] < evals_v[2]
    vav = c_v.conj().T @ numeric(vertex_of(0, SY, variant_inc[0])) @ c_v
    vbv = c_v.conj().T @ numeric(vertex_of(1, SY, variant_inc[1])) @ c_v
    response_v = sum(
        (occupations[n] - occupations[m])
        / (evals_v[n] - evals_v[m])
        * (vav[n, m] * vbv[m, n]).real
        for n in range(4)
        for m in range(4)
        if n != m
    )
    assert abs(response_v) > 1e-9
    # Stationary collinear control: zero torque and a valid one-site kernel.
    stationary_h = sp.diag(-3, -1, 2, 4)
    stationary_scalar, stationary_b = decompose(stationary_h)
    stationary_inc = [incident(0, stationary_b), incident(1, stationary_b)]

    def divided(n, m):
        return (
            0
            if n == m
            else sp.Rational(
                occupations[n] - occupations[m], eigenvalues[n] - eigenvalues[m]
            )
        )

    def stationary_curvature(site_a, axis_a, site_b, axis_b):
        va = vertex_of(site_a, axis_a, stationary_inc[site_a])
        vb = vertex_of(site_b, axis_b, stationary_inc[site_b])
        if site_a == site_b:
            wa = seagull_same_site(site_a, axis_a, axis_b, stationary_inc[site_a])
        else:
            wa = phi_pass(
                site_b, axis_b / 2, phi_pass(site_a, axis_a / 2, stationary_b)
            )
        return sp.simplify(
            sum(occupations[n] * wa[n, n] for n in range(4))
            + sum(
                divided(n, m) * va[n, m] * vb[m, n] for n in range(4) for m in range(4)
            )
        )

    stat_kappa_a = stationary_curvature(0, SY, 0, SY)
    stat_kappa_b = stationary_curvature(1, SY, 1, SY)
    stat_cross = stationary_curvature(0, SY, 1, SY)
    stationary_v = vertex_of(0, SY, stationary_inc[0])
    zero(sp.trace(sp.diag(1, 1, 0, 0) * stationary_v))
    zero(stat_kappa_a + 2 * stat_cross + stat_kappa_b)
    zero(stat_kappa_a - stat_kappa_b)
    zero(stat_cross + stat_kappa_a)
    print(
        f"[G2] textured-S: g_sum={sp.simplify(g_ay_t+g_by_t)}, "
        f"kappa_A={kappa_a_t}, K_AB={cross_t}, kappa_B={kappa_b_t}"
    )

    # [H] Direct numerical generalized eigensolves, independent of contour sums.
    sn, hn = numeric(overlap), numeric(h0)
    cn = numeric(coefficients)
    scalar_n, magnetic_n = numeric(scalar), numeric(full_magnetic)
    e, c = la.eigh(hn, sn)
    np.testing.assert_allclose(e, eigenvalues, atol=2e-14)
    zn = 0.31 + 0.73j
    gn = la.inv(zn * sn - hn)
    np.testing.assert_allclose(gn, cn @ np.diag(1 / (zn - e)) @ cn.conj().T, atol=2e-14)
    np.testing.assert_allclose(np.trace(sn @ gn), sum(1 / (zn - e)), atol=2e-14)
    np.testing.assert_allclose(
        np.trace(gn @ numeric(v)),
        np.trace((c.conj().T @ numeric(v) @ c) @ np.diag(1 / (zn - e))),
        atol=2e-14,
    )
    vx = np.array([c.conj().T @ numeric(vv) @ c for vv in vertices])
    gnum = np.real(np.trace(vx[:, :2, :2], axis1=1, axis2=2))
    np.testing.assert_allclose(gnum, numeric(gradient).ravel().real, atol=2e-14)
    knum = np.zeros((4, 4))
    for ai in range(4):
        for bi in range(4):
            wn = c.conj().T @ numeric(seagulls[ai, bi]) @ c
            knum[ai, bi] = np.trace(wn[:2, :2]).real + sum(
                (occupations[n] - occupations[m])
                / (e[n] - e[m])
                * (vx[ai, n, m] * vx[bi, m, n]).real
                for n in range(4)
                for m in range(4)
                if n != m
            )
    np.testing.assert_allclose(knum, numeric(curvature).real, atol=3e-14)
    # Numeric congruence probe independently checks spin-dependent S.
    spinn, spinhn = numeric(spin_s), numeric(spin_h)
    spinvn = numeric(transform.H * v * transform)
    es, cs = la.eigh(spinhn, spinn)
    np.testing.assert_allclose(es, e, atol=2e-14)
    np.testing.assert_allclose(
        np.trace(cs[:, :2].conj().T @ spinvn @ cs[:, :2]).real,
        float(gradient[1]),
        atol=2e-14,
    )

    sx_n, sy_n = numeric(SX), numeric(SY)

    def pass_numeric(matrix, site, u):
        out = np.array(matrix, copy=True)
        for i, j in incident_blocks(site):
            blk = out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            out[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = u @ blk @ u.conj().T
        return out

    def band_energy(parameters):
        """Production path: ordered two-sided passes, S and scalar frozen."""
        ua = la.expm(-1j * (parameters[0] * sx_n + parameters[1] * sy_n) / 2)
        ub = la.expm(-1j * (parameters[2] * sx_n + parameters[3] * sy_n) / 2)
        total = pass_numeric(pass_numeric(scalar_n + magnetic_n, 0, ua), 1, ub)
        values = la.eigvalsh(total, sn)
        assert values[1] < 0 < values[2], values
        return values[:2].sum()

    reference_energy = band_energy(np.zeros(4))
    np.testing.assert_allclose(reference_energy, -4, atol=2e-14)
    # Every Hessian element is independently checked by finite rotations.
    step = 2e-4
    for ai in range(4):
        da = np.eye(4)[ai] * step
        central_g = (band_energy(da) - band_energy(-da)) / (2 * step)
        np.testing.assert_allclose(central_g, gnum[ai], atol=2e-8)
        for bi in range(4):
            db = np.eye(4)[bi] * step
            central_k = (
                band_energy(da + db)
                - band_energy(da - db)
                - band_energy(-da + db)
                + band_energy(-da - db)
            ) / (4 * step * step)
            np.testing.assert_allclose(central_k, knum[ai, bi], atol=2e-7)

    # Independent one-site and two-site extractions (review STD-2), taken
    # from separate finite-rotation paths. The exact contour values remain
    # the precision authority; raw central differences at a small step keep
    # both truncation and round-off below the asserted tolerances.
    def central(energy_function, amplitude):
        plus, minus = energy_function(amplitude), energy_function(-amplitude)
        return np.array(
            [
                (plus - minus) / (2 * amplitude),
                (plus - 2 * reference_energy + minus) / amplitude**2,
            ]
        )

    def central_cross(energy_function, amplitude):
        return (
            energy_function(np.array([0.0, amplitude, 0.0, amplitude]))
            - energy_function(np.array([0.0, amplitude, 0.0, -amplitude]))
            - energy_function(np.array([0.0, -amplitude, 0.0, amplitude]))
            + energy_function(np.array([0.0, -amplitude, 0.0, -amplitude]))
        ) / (4 * amplitude**2)

    def ay_family(amplitude):
        return band_energy(np.array([0.0, amplitude, 0.0, 0.0]))

    def by_family(amplitude):
        return band_energy(np.array([0.0, 0.0, 0.0, amplitude]))

    hierarchy_step = 1e-3
    kappa_a_n = central(ay_family, hierarchy_step)[1]
    kappa_b_n = central(by_family, hierarchy_step)[1]
    cross_n = central_cross(band_energy, hierarchy_step)
    np.testing.assert_allclose(kappa_a_n, knum[1, 1], atol=1e-6)
    np.testing.assert_allclose(kappa_b_n, knum[3, 3], atol=1e-6)
    np.testing.assert_allclose(cross_n, knum[1, 3], atol=1e-6)
    np.testing.assert_allclose(
        kappa_a_n + 2 * cross_n + kappa_b_n,
        float(kappa_a + 2 * cross_k + kappa_b),
        atol=1e-6,
    )
    assert abs(kappa_a_n + 2 * cross_n + kappa_b_n) > 1e-4
    assert abs(kappa_a_n + cross_n) > 1e-4
    assert abs(kappa_b_n + cross_n) > 1e-4
    assert abs(kappa_a_n - kappa_b_n) > 1e-4
    # The textured-S counter-model numerically: identification fails.
    overlap_tn = numeric(overlap_t)
    scalar_tn, magnetic_tn = numeric(scalar_t), numeric(magnetic_t)

    def band_energy_textured(parameters):
        ua = la.expm(-1j * (parameters[0] * sx_n + parameters[1] * sy_n) / 2)
        ub = la.expm(-1j * (parameters[2] * sx_n + parameters[3] * sy_n) / 2)
        total = pass_numeric(pass_numeric(scalar_tn + magnetic_tn, 0, ua), 1, ub)
        values = la.eigvalsh(total, overlap_tn)
        assert values[1] < 0 < values[2], values
        return values[:2].sum()

    def ay_family_textured(amplitude):
        return band_energy_textured(np.array([0.0, amplitude, 0.0, 0.0]))

    def by_family_textured(amplitude):
        return band_energy_textured(np.array([0.0, 0.0, 0.0, amplitude]))

    g_ay_tn = (ay_family_textured(1e-3) - ay_family_textured(-1e-3)) / (2e-3)
    g_by_tn = (by_family_textured(1e-3) - by_family_textured(-1e-3)) / (2e-3)
    kappa_a_tn = central(ay_family_textured, hierarchy_step)[1]
    kappa_b_tn = central(by_family_textured, hierarchy_step)[1]
    cross_tn = central_cross(band_energy_textured, hierarchy_step)
    np.testing.assert_allclose(g_ay_tn + g_by_tn, float(g_ay_t + g_by_t), atol=1e-5)
    np.testing.assert_allclose(kappa_a_tn, float(kappa_a_t), atol=1e-6)
    np.testing.assert_allclose(kappa_b_tn, float(kappa_b_t), atol=1e-6)
    np.testing.assert_allclose(cross_tn, float(cross_t), atol=1e-6)
    assert abs(kappa_a_tn - kappa_b_tn) > 1e-4
    assert abs(kappa_a_tn + 2 * cross_tn + kappa_b_tn) > 1e-4
    print(
        f"[H] independent extractions: kappa_A={kappa_a_n:.12g}, kappa_B={kappa_b_n:.12g}, "
        f"K_AB={cross_n:.12g}, sum rule residual={kappa_a_n+2*cross_n+kappa_b_n:.3g}"
    )
    print(
        f"[H] textured-S numerics: g_sum={g_ay_tn+g_by_tn:.6g}, "
        f"kappa_A={kappa_a_tn:.6g}, kappa_B={kappa_b_tn:.6g}, "
        f"sum residual={kappa_a_tn+2*cross_tn+kappa_b_tn:.6g}"
    )

    def ay_energy(amplitude):
        return band_energy(np.array([0.0, amplitude, 0.0, 0.0]))

    target = np.array([float(gradient[1]), float(curvature[1, 1])])
    steps = np.array([0.2, 0.1, 0.05])
    central_errors, richardson_errors = [], []
    for amplitude in steps:
        full, half = central(ay_energy, amplitude), central(ay_energy, amplitude / 2)
        second_error = np.abs(full - target)
        fourth_error = np.abs((4 * half - full) / 3 - target)
        central_errors.append(second_error)
        richardson_errors.append(fourth_error)
        print(
            f"[H] h={amplitude:.3f}: O(h^2) errors {second_error}; O(h^4) errors {fourth_error}"
        )
    ratios2 = np.array(central_errors[:-1]) / np.array(central_errors[1:])
    ratios4 = np.array(richardson_errors[:-1]) / np.array(richardson_errors[1:])
    assert np.all((ratios2 > 3.7) & (ratios2 < 4.3)), ratios2
    assert np.all((ratios4 > 13) & (ratios4 < 19)), ratios4
    # Counterexample curvature difference also measured by finite rotations.
    amplitude = 3e-4

    def site_a_energy(px, pz, ordered_chart):
        u_pz = la.expm(-1j * pz * numeric(SZ) / 2)
        u_px = la.expm(-1j * px * sx_n / 2)
        u = (
            u_px @ u_pz
            if ordered_chart
            else la.expm(-1j * (px * sx_n + pz * numeric(SZ)) / 2)
        )
        total = scalar_n + pass_numeric(magnetic_n, 0, u)
        values = la.eigvalsh(total, sn)
        assert values[1] < 0 < values[2], values
        return values[:2].sum()

    ordered_cross = sum(
        sx * sz * site_a_energy(sx * amplitude, sz * amplitude, True)
        for sx in (-1, 1)
        for sz in (-1, 1)
    ) / (4 * amplitude * amplitude)
    symmetric_cross = sum(
        sx * sz * site_a_energy(sx * amplitude, sz * amplitude, False)
        for sx in (-1, 1)
        for sz in (-1, 1)
    ) / (4 * amplitude * amplitude)
    np.testing.assert_allclose(
        ordered_cross - symmetric_cross, -float(gradient[1]) / 2, atol=8e-8
    )
    print(
        f"[H] g_Ay={target[0]:.12g}, tau_Ay={-target[0]:.12g}, g_Bx={gnum[2]:.12g}, "
        f"g_By={gnum[3]:.12g}, K_AyAy={target[1]:.12g} "
        f"(onsite={float(onsite[1, 1]):.12g}, response={float(response[1, 1]):.12g}), "
        f"K_AyBy={float(curvature[1, 3]):.12g} "
        f"(onsite={float(onsite[1, 3]):.12g}, response={float(response[1, 3]):.12g})"
    )
    print(
        "[H] Generalized eigensolves, all finite-rotation Hessian entries and nonstationary chart probe"
    )

    for path in (
        "docs/sympy/siesta_spiral_frozen_response.py",
        "docs/sympy/siesta_spiral_frozen_response.md",
    ):
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"[H] sha256 {path}: {digest}")
    print("ALL ASSERTIONS GREEN")


if __name__ == "__main__":
    main()
