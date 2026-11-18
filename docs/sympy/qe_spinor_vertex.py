"""Exact QE KB spinor magnetic-vertex and native-block convention checks.

The scalar-relativistic collinear limit is an identity. A fully relativistic
UPF's fcoef mapping is NOT declared spin-rotation covariant by this script;
its native fixed-source action and rotation must be checked on real data.
Run with the mydev environment: python docs/sympy/qe_spinor_vertex.py.
"""

import numpy as np
import sympy as sp

I2 = sp.eye(2)
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SZ = sp.diag(1, -1)
SIGMA = (SX, SY, SZ)


def check_qe_native_order():
    """newd_nc_acc and add_vuspsi_nc use (uu,ud,du,dd), not (uu,dd,ud,du)."""
    c, bx, by, bz = sp.symbols("c bx by bz", real=True)
    native = sp.Matrix([[c + bz, bx - sp.I * by], [bx + sp.I * by, c - bz]])
    assert native == c * I2 + bx * SX + by * SY + bz * SZ
    assert (native[0, 0], native[0, 1], native[1, 0], native[1, 1]) == (
        c + bz,
        bx - sp.I * by,
        bx + sp.I * by,
        c - bz,
    )
    assert native == native.H
    return native


def check_complex_field():
    """A real-space real field produces complex Hermitian channel blocks."""
    beta = sp.Matrix([[1, sp.I]])
    projected = beta.H * sp.diag(2) * beta
    assert projected == sp.Matrix([[2, 2 * sp.I], [-2 * sp.I, 2]])
    assert projected == projected.H
    assert projected != projected.applyfunc(sp.re)


def check_soc_subtraction_and_covariance_boundary():
    """A same-source difference cancels SOC; a generic fcoef need not commute with spin."""
    q0 = sp.Matrix([[3, sp.I], [-sp.I, 5]])
    qx = sp.Matrix([[0, 1], [1, 0]])
    soc_orb = sp.Matrix([[1, sp.I], [-sp.I, 2]])
    fixed_soc = sp.kronecker_product(soc_orb, SZ)
    # CNOT is an exact unitary spin-orbital entangler, used as a NEGATIVE
    # CONTROL, not as a claim about QE fcoef Clebsch-Gordan coefficients.
    fcoef = sp.eye(4)
    fcoef[2, 2] = 0
    fcoef[3, 3] = 0
    fcoef[2, 3] = 1
    fcoef[3, 2] = 1
    assert fcoef.H * fcoef == sp.eye(4)
    charge = sp.kronecker_product(q0, I2)
    magnetic = sp.kronecker_product(qx, SX)
    total = fixed_soc + fcoef.H * (charge + magnetic) * fcoef
    same_source_zero_field = fixed_soc + fcoef.H * charge * fcoef
    separated = total - same_source_zero_field
    assert separated == fcoef.H * magnetic * fcoef
    assert separated == separated.H
    assert (total - sp.trace(total) / 4 * sp.eye(4)) != separated
    # Nontrivial spin-orbital mixing can map a magnetic source into a
    # spin-scalar channel block. Do not discard its spin trace: NC1.1
    # uses native source responses, not a spin-only commutator.
    assert separated == sp.kronecker_product(qx, I2)
    assert separated * sp.kronecker_product(I2, SZ) == (
        sp.kronecker_product(I2, SZ) * separated
    )
    rotated_source = sp.kronecker_product(qx, SY)
    assert fcoef.H * rotated_source * fcoef != separated
    return separated


def check_paw_magnetic_source_separation():
    """PAW D1 magnetic XC rotates with B; its scalar part and SOC do not."""
    q0, p0, smooth_bz, paw_bz, fixed_so = sp.symbols(
        "q0 p0 smooth_bz paw_bz fixed_so", real=True
    )
    total = (q0 + p0) * I2 + (smooth_bz + paw_bz) * SZ + fixed_so * SX
    zero_magnetic_source = (q0 + p0) * I2 + fixed_so * SX
    separated = total - zero_magnetic_source
    assert separated == (smooth_bz + paw_bz) * SZ
    assert separated[0, 0] - separated[1, 1] == 2 * (smooth_bz + paw_bz)
    assert total - ((q0 + p0) * I2 + paw_bz * SZ + fixed_so * SX) != separated
    uy = (I2 - sp.I * SY) / sp.sqrt(2)
    rotated_source = (q0 + p0) * I2 + (smooth_bz + paw_bz) * SX + fixed_so * SX
    assert rotated_source - zero_magnetic_source == uy * separated * uy.H


def check_source_rotation_signs():
    """Native frozen-source vector rotation must match SU(2) conjugation."""
    ux = (I2 - sp.I * SX) / sp.sqrt(2)
    uy = (I2 - sp.I * SY) / sp.sqrt(2)
    assert sp.simplify(ux * SZ * ux.H + SY) == sp.zeros(2)
    assert sp.simplify(ux * SY * ux.H - SZ) == sp.zeros(2)
    assert sp.simplify(uy * SZ * uy.H - SX) == sp.zeros(2)
    assert sp.simplify(uy * SX * uy.H + SZ) == sp.zeros(2)
    bx, by, bz = sp.symbols("bx by bz", real=True)
    field = bx * SX + by * SY + bz * SZ
    assert sp.simplify(ux * field * ux.H - (bx * SX - bz * SY + by * SZ)) == sp.zeros(2)
    assert sp.simplify(uy * field * uy.H - (bz * SX + by * SY - bx * SZ)) == sp.zeros(2)


def check_non_covariant_source_tangent():
    """A linear native magnetic map can require an explicit source tangent."""
    # Abstract linear map: not a model of QE's measured fcoef, but a
    # counterexample to replacing native source rotation by a commutator.
    lam, theta = sp.symbols("lambda theta", real=True)
    dz = lam * SZ
    dx = 2 * lam * SX
    rotated_source = sp.sin(theta) * dx + sp.cos(theta) * dz
    physical_source_x = sp.diff(rotated_source, theta).subs(theta, 0)
    commutator_x = -sp.I / 4 * (SY * (2 * dz) - (2 * dz) * SY)
    assert physical_source_x == 2 * lam * SX
    assert commutator_x == lam * SX
    assert physical_source_x != commutator_x
    # Pauli-covariant scalar-relativistic mapping instead has dx=lam*sigma_x.
    scalar_source = sp.diff(sp.sin(theta) * lam * SX + sp.cos(theta) * dz, theta).subs(
        theta, 0
    )
    assert scalar_source == commutator_x
    epsilon = sp.Rational(1, 100000)
    finite_difference = (
        (rotated_source.subs(theta, epsilon) - rotated_source.subs(theta, -epsilon))
        / (2 * epsilon)
    ).subs(lam, sp.Rational(7, 10))
    assert (
        max(
            abs(complex(sp.N(value)))
            for value in finite_difference
            - physical_source_x.subs(lam, sp.Rational(7, 10))
        )
        < 1e-9
    )


def check_source_response_tensor():
    """Nine source/target blocks reconstruct a native frozen-field tangent."""
    theta = sp.symbols("theta", real=True)
    n = sp.Matrix([sp.Rational(3, 5), 0, sp.Rational(4, 5)])
    t = sp.Matrix([0, 1, 0])
    axis = n.cross(t)
    assert axis.dot(axis) == 1
    generator = sp.Matrix(
        [
            [sum(sp.LeviCivita(i, k, j) * axis[k] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]
    )
    assert generator.T == -generator
    rotation = (
        sp.eye(3) + sp.sin(theta) * generator + (1 - sp.cos(theta)) * generator**2
    )
    response = [
        [
            sp.Rational(i + j + 1, 17) * I2
            + sp.Rational((i + 1) * (j + 2), 7) * SX
            + sp.Rational(i - j, 11) * SY
            + sp.Rational(2 * i + j + 1, 13) * SZ
            for j in range(3)
        ]
        for i in range(3)
    ]
    assert all(block == block.H for row in response for block in row)
    mapped = sum(
        (rotation[i, j] * response[i][j] for i in range(3) for j in range(3)),
        sp.zeros(2),
    )
    tangent = sum(
        (generator[i, j] * response[i][j] for i in range(3) for j in range(3)),
        sp.zeros(2),
    )
    assert sp.simplify(sp.diff(mapped, theta).subs(theta, 0) - tangent) == sp.zeros(2)
    epsilon = sp.Rational(1, 10000)
    central = (mapped.subs(theta, epsilon) - mapped.subs(theta, -epsilon)) / (
        2 * epsilon
    )
    assert max(abs(complex(sp.N(value))) for value in central - tangent) < 1e-8

    # A scalar-relativistic z source has Bz only; native response to
    # placing it along any output Pauli axis is b*sigma_i.
    b = sp.symbols("b", real=True)
    scalar = [[sp.zeros(2) for _ in range(3)] for _ in range(3)]
    for i in range(3):
        scalar[i][2] = b * SIGMA[i]
    z_axis = sp.Matrix([0, 0, 1]).cross(sp.Matrix([1, 0, 0]))
    z_generator = sp.Matrix(
        [
            [
                sum(sp.LeviCivita(i, k, j) * z_axis[k] for k in range(3))
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    source_x = sum(
        (z_generator[i, j] * scalar[i][j] for i in range(3) for j in range(3)),
        sp.zeros(2),
    )
    existing_x = -sp.I / 4 * (SY * (2 * b * SZ) - (2 * b * SZ) * SY)
    assert source_x == existing_x == b * SX


def check_collinear_limit_and_tangent():
    """The z-leg smooth+augmentation vertex equals QE v1.2 on its uu block."""
    gram = sp.Matrix([[3, sp.I], [-sp.I, 4]])
    smooth_bz = sp.Matrix([[1, 2 * sp.I], [-2 * sp.I, 3]])
    aug_bz = sp.Matrix([[2, sp.I], [-sp.I, 5]])
    assert gram == gram.H and gram.det() != 0
    assert smooth_bz == smooth_bz.H and aug_bz == aug_bz.H
    field = gram.inv() * smooth_bz * gram.inv() + aug_bz
    vertex = 2 * sp.kronecker_product(field, SZ)
    dbeta_xc = 2 * smooth_bz
    deeq_up_minus_down = 2 * aug_bz
    collinear_v12 = gram.inv() * dbeta_xc * gram.inv() + deeq_up_minus_down
    uu = vertex.extract((0, 2), (0, 2))
    dd = vertex.extract((1, 3), (1, 3))
    assert uu == collinear_v12
    assert dd == -collinear_v12
    assert vertex.extract((0, 2), (1, 3)) == sp.zeros(2)
    # Orbital-first channel order: (p,spin). For n=z, t=x, n cross t=y.
    generator = sp.kronecker_product(sp.eye(2), SY)
    tangent_x = -sp.I / 4 * (generator * vertex - vertex * generator)
    assert tangent_x == sp.kronecker_product(field, SX)
    # Reversing the magnetic splitting also reverses the inferred moment
    # direction, so the rotation generator and splitting each change sign.
    reversed_tangent = -sp.I / 4 * ((-generator) * (-vertex) - (-vertex) * (-generator))
    assert reversed_tangent == tangent_x


def check_nc_smooth_only_source_and_tangent():
    """ADR-002: an NC source/tangent is smooth-only with measured-zero D.

    For a norm-conserving site with Gram ``M``, projected smooth Pauli
    field ``B_z`` and *measured-zero* native responses ``D_{c<-j} = 0``,
    the operator and tangent are exactly::

        Delta = 2E kron(M^+ B_z M^+, sigma_z),   V_x = E kron(M^+ B_z M^+, sigma_x)

    the explicit source-resolved vertex equals the commutator tangent of
    the same operator, the longitudinal vertex vanishes, and rotating
    the smooth source (and moment) by a Pauli quarter-turn conjugates
    the vertex exactly. This is the algebraic identity the reader
    asserts numerically on has_so=F species.
    """
    E = sp.symbols("E", real=True)
    gram = sp.Matrix([[3, sp.I], [-sp.I, 4]])
    smooth_bz = sp.Matrix(
        [[sp.Rational(1, 50), 2 * sp.I], [-2 * sp.I, sp.Rational(3, 100)]]
    )
    assert gram == gram.H and smooth_bz == smooth_bz.H
    q = gram.inv() * smooth_bz * gram.inv()

    # measured-zero D: the smooth-only operator and explicit tangent.
    delta = 2 * E * sp.kronecker_product(q, SZ)
    # n = z, t = x -> generator axis n x t = y; weights K_{cj}=eps_{cdj} g_d
    # select (c=x, j=z) with weight +1.
    vertex_x = E * sp.kronecker_product(q, SX)
    generator = sp.kronecker_product(I2, SY)
    assert sp.simplify(
        -sp.I / 4 * (generator * delta - delta * generator) - vertex_x
    ) == sp.zeros(4)
    # longitudinal vertex: n x n = 0.
    generator_z = sp.kronecker_product(I2, SZ)
    assert -sp.I / 4 * (generator_z * delta - delta * generator_z) == sp.zeros(4)
    # a nonzero D_{c<-j} breaks the commutator equality (the SO-bearing case).
    d_contamination = E * sp.kronecker_product(q, SY)
    explicit_with_d = vertex_x + d_contamination
    assert explicit_with_d != -sp.I / 4 * (generator * delta - delta * generator)

    # Pauli covariance: conjugating the smooth source by U_y
    # (SZ -> SX, SX -> -SZ, SY -> SY) conjugates the operator exactly and
    # rotates the transverse tangent legs with it (z leg -> x-config z leg
    # equals the conjugated y leg since R_y(-pi/2) fixes y).
    uy = (I2 - sp.I * SY) / sp.sqrt(2)
    pair = sp.kronecker_product(I2, uy)
    delta_x = 2 * E * sp.kronecker_product(q, SX)
    assert sp.simplify(delta_x - pair * delta * pair.H) == sp.zeros(4)
    vertex_y = E * sp.kronecker_product(q, SY)
    gen_z = sp.kronecker_product(I2, SZ)
    tangent_z_of_rotated = -sp.I / 4 * (gen_z * delta_x - delta_x * gen_z)
    assert sp.simplify(tangent_z_of_rotated - pair * vertex_y * pair.H) == sp.zeros(4)
    # longitudinal leg of the rotated (x-moment) config vanishes
    gen_x = sp.kronecker_product(I2, SX)
    assert -sp.I / 4 * (gen_x * delta_x - delta_x * gen_x) == sp.zeros(4)
    return vertex_x


def check_gram_retained_subspace_reconstruction():
    """Exact dual reconstruction holds only on the Gram-retained subspace.

    With the rcond-truncated transform ``M^+_tau`` (eigenmode below the
    cutoff replaced by zero) the dual identity ``M^+ M = I`` degrades to
    the retained projector ``P``: a smooth field supported on the
    retained subspace is reconstructed exactly, while any component on
    a discarded mode is annihilated instead of dressed. This is the
    principal-subspace restriction behind ADR-003.
    """
    m1 = sp.symbols("m1", positive=True)
    M = sp.diag(m1, 2 * m1)
    assert sp.simplify(M.inv() * M - sp.eye(2)) == sp.zeros(2)
    # retained subspace = span(e1); truncated transform drops mode e2
    M_trunc = sp.diag(1 / m1, 0)
    P = sp.diag(1, 0)
    assert sp.simplify(M_trunc * M - P) == sp.zeros(2)
    # supported on the retained subspace: exact reconstruction
    a = sp.symbols("a", real=True)
    A_retained = a * P
    assert sp.simplify(M_trunc * A_retained * M_trunc - (a / m1**2) * P) == sp.zeros(2)
    # a component coupling into the discarded mode is annihilated by the
    # truncated transform, not dressed
    A_coupling = sp.Rational(3, 7) * sp.Matrix([[0, 1], [1, 0]])
    assert M_trunc * A_coupling * M_trunc == sp.zeros(2)
    full_dressed = M.inv() * A_coupling * M.inv()
    assert sp.simplify(full_dressed - A_coupling / (2 * m1**2)) == sp.zeros(2)
    return P


def check_normalized_antisymmetric_counterexample():
    """Normalized near-null antisymmetric mode: contraction exactly 1 vs 0.

    For ``M = [[1, s], [s, 1]]`` (eigenvalues 1+s retained, 1-s
    discarded), the antisymmetric mode ``u_- = (1, -1)/sqrt(2)`` and the
    constant field ``A = M``, the primal state
    ``x = u_-/sqrt(1-s)`` (norm squared ``1/(1-s)`` diverging as
    ``s -> 1``) has the dressed quadratic
    contraction ``<M x| M^+ A M^+ |M x>`` equal to **exactly 1** with the
    full inverse and **exactly 0** with the rcond-truncated transform,
    for every ``0 < s < 1``. The prior universal ``O(1-s)`` error bound
    silently assumed bounded, unnormalized primal coefficients and is
    disproved; nothing here re-pins it. A clean eigengap prevents
    numerical ambiguity, not loss of normalized spectral states, and
    the (j, m_j) spinor projectors are not claimed as a square
    transform of the QE channels.
    """
    s = sp.symbols("s", positive=True)
    pp = (I2 + SX) / 2
    pm = (I2 - SX) / 2
    M = (1 + s) * pp + (1 - s) * pm
    assert M == sp.Matrix([[1, s], [s, 1]])
    # exact check for 0 < s < 1: parameterize s = 1 - t**2 so
    # sqrt(1-s) = t is a positive symbol and conjugates collapse.
    t = sp.symbols("t", positive=True)
    Mt = M.subs(s, 1 - t**2)
    u = sp.Matrix([1, -1]) / sp.sqrt(2)
    x = u / t
    # the primal state norm diverges as s -> 1: ||x||^2 = 1/(1-s). This
    # unbounded normalization is what defeats any universal O(1-s) bound.
    assert sp.simplify((x.H * x)[0, 0] - 1 / (1 - (1 - t**2))) == 0
    A = Mt
    full = Mt.inv() * A * Mt.inv()
    truncated = pp.subs(s, 1 - t**2) / (2 - t**2)  # retained-mode-only transform
    Mx = sp.simplify(Mt * x)
    contraction_full = sp.simplify((Mx.H * full * Mx)[0, 0])
    contraction_truncated = sp.simplify((Mx.H * truncated * A * truncated * Mx)[0, 0])
    assert contraction_full == 1
    assert contraction_truncated == 0
    # numerical spot check near degeneracy; 1-s below the rcond cutoff so
    # the antisymmetric mode is genuinely truncated
    s_num = sp.Float("0.99999999", 15)
    Mn = np.array(M.subs(s, s_num).tolist(), dtype=float)
    evals, vecs = np.linalg.eigh(Mn)
    trunc_inv = (
        vecs @ np.diag(np.where(evals > 1e-6 * evals.max(), 1 / evals, 0.0)) @ vecs.T
    )
    un = np.array(u, dtype=float).ravel()
    xn = un / np.sqrt(float(1 - s_num))
    Mxn = Mn @ xn
    full_num = np.array(full.subs(t, sp.sqrt(1 - s_num)).tolist(), dtype=float)
    assert abs(Mxn @ full_num @ Mxn - 1) < 1e-9
    assert abs(Mxn @ (trunc_inv @ Mn @ trunc_inv) @ Mxn) < 1e-12
    return contraction_full


def main():
    check_qe_native_order()
    check_complex_field()
    check_soc_subtraction_and_covariance_boundary()
    check_collinear_limit_and_tangent()
    check_source_rotation_signs()
    check_paw_magnetic_source_separation()
    check_non_covariant_source_tangent()
    check_source_response_tensor()
    check_nc_smooth_only_source_and_tangent()
    check_gram_retained_subspace_reconstruction()
    check_normalized_antisymmetric_counterexample()
    print(
        "PASS: QE packing, complex field, PAW/SOC separation, SU2 signs, native "
        "source tangents, NC smooth-only vertex, retained-subspace reconstruction, "
        "normalized antisymmetric counterexample"
    )


if __name__ == "__main__":
    main()
