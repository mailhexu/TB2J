"""SymPy certificate for EMC FM and AFM angle/boson conventions.

Run with the mydev environment: python docs/sympy/emc_magnon_normalization.py
The electron spinor vertex and q-phase certificate lives in the EMC package.
"""

import argparse
import json
import platform

import numpy as np
import sympy as sp


def assert_fm_and_cholesky():
    S, D = sp.symbols("S D", positive=True, real=True)
    u, b, bd = sp.symbols("u b bdag", complex=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sm = (sx - sp.I * sy) / 2
    sp_ = (sx + sp.I * sy) / 2
    # n_+ = sqrt(2/S) u b, n_- = sqrt(2/S) u* bdag;
    # n_x=(n_++n_-)/2 and n_y=(n_+-n_-)/(2i).
    plus = sp.sqrt(2 / S) * u * b
    minus = sp.sqrt(2 / S) * sp.conjugate(u) * bd
    perturbation = D * ((plus + minus) * sx / 2 + (plus - minus) * sy / (2 * sp.I))
    assert sp.simplify(perturbation.diff(b) - D * sp.sqrt(2 / S) * u * sm) == sp.zeros(
        2
    )
    assert sp.simplify(
        perturbation.diff(bd) - D * sp.sqrt(2 / S) * sp.conjugate(u) * sp_
    ) == sp.zeros(2)
    theta = sp.symbols("theta", real=True)
    assert sp.limit((sp.sin(theta) - theta) / theta**3, theta, 0) == -sp.Rational(1, 6)

    # TB2J returns the Euclidean-unit eigenvector v of A=K^dag g K.
    # This generic invertible lower triangular K proves the inverse identity
    # used to recover psi=sqrt(omega) K^{-dag}v and its metric norm.
    a, c = sp.symbols("a c", positive=True, real=True)
    t = sp.symbols("t", real=True)
    K = sp.Matrix([[a, 0], [t, c]])
    g = sp.diag(1, -1)
    A = K.H * g * K
    assert sp.simplify(A.inv() - K.inv() * g * K.H.inv()) == sp.zeros(2)
    omega = sp.symbols("omega", positive=True, real=True)
    v1, v2 = sp.symbols("v1 v2", complex=True)
    v = sp.Matrix([v1, v2])
    psi = sp.sqrt(omega) * K.H.inv() * v
    # The first equality reduces using A v=omega v; the second follows
    # from omega v^dag A^-1 v=v^dag v for a unit Euclidean eigenvector.
    residual_identity = (
        K * K.H * psi
        - omega * g * psi
        - sp.sqrt(omega) * K * A.inv() * (A * v - omega * v)
    )
    assert residual_identity.applyfunc(sp.simplify) == sp.zeros(2, 1)
    assert sp.simplify((psi.H * g * psi)[0] - omega * (v.H * A.inv() * v)[0]) == 0
    assert sp.simplify((sp.sqrt(omega) * K.H.inv() * v).subs(omega, 0)) == sp.zeros(
        2, 1
    )
    # Kept silent so --kosmic-json can emit exactly one protocol payload.


def named_expressions():
    """Return the named symbolic expressions exported to the math registry."""
    S = sp.symbols("S", positive=True, real=True)
    u, v = sp.symbols("u v", complex=True)
    eta = sp.symbols("eta", real=True)
    annihilation, creation = sp.symbols("a a_dag", complex=True)

    local_x = (annihilation + creation) / sp.sqrt(2 * S)
    local_y = (annihilation - creation) / (sp.I * sp.sqrt(2 * S))
    x_q = (u + v) / sp.sqrt(2 * S)
    y_q = (u - v) / (sp.I * sp.sqrt(2 * S))

    # Native (Option-B local-axis) route: per-site unscaled lab quadratures
    # consumed by the interleaved-spinor vertex, the local-axis half-splitting
    # extracted from global channels, the frozen rotation-generator pair, and
    # the site boson annihilation combination.
    d_z = sp.symbols("D_z", real=True)
    x_native = u + v
    y_native = eta * (u - v)
    expressions = {
        "afm_local_frame_lab_y_annihilation": sp.simplify(eta * y_q),
        "afm_local_frame_plus_nplus": sp.sqrt(2 / S) * annihilation,
        "afm_local_frame_minus_nplus": sp.sqrt(2 / S) * creation,
        "afm_nambu_creation_reverse_x": sp.conjugate(x_q),
        "afm_nambu_creation_reverse_y": sp.conjugate(y_q),
        "afm_nambu_local_x_annihilation": sp.simplify(x_q),
        "afm_nambu_local_y_annihilation": sp.simplify(y_q),
        "afm_nambu_operator_local_x": local_x,
        "afm_nambu_operator_local_y": local_y,
        "afm_opposite_frame_lab_y_creation_reverse": sp.conjugate(eta * y_q),
        "fm_hp_nplus_annihilation": sp.sqrt(2 / S) * u,
        "native_route_site_x_amplitude": x_native,
        "native_route_site_y_amplitude": y_native,
        "native_route_local_axis_half_splitting": eta * d_z,
        "native_route_omega_x": -(u - v),
        "native_route_omega_y": eta * (u + v),
        "native_route_boson_annihilation": sp.simplify(
            (x_native - sp.I * y_native) / sp.sqrt(2 * S)
        ),
        "native_route_reverse_creation_x": sp.conjugate(x_native),
        "native_route_reverse_creation_y": sp.conjugate(y_native),
    }
    return expressions


def derive():
    """Return protocol-ready, named SymPy results in lexical order."""
    expressions = named_expressions()
    return [
        {
            "name": name,
            "srepr": sp.srepr(expressions[name]),
            "latex": sp.latex(expressions[name]),
        }
        for name in sorted(expressions)
    ]


def _assert_symbolic_derivation():
    S = sp.symbols("S", positive=True, real=True)
    u, v, beta = sp.symbols("u v beta", complex=True)
    eta = sp.symbols("eta", real=True)
    aq, amq_dag = sp.symbols("a_q a_minus_q_dag", complex=True)

    # Local HP spin operators are expanded independently in x and y.
    local_x = (aq + amq_dag) / sp.sqrt(2 * S)
    local_y = (aq - amq_dag) / (sp.I * sp.sqrt(2 * S))
    lab_nplus = local_x + sp.I * eta * local_y
    lab_nminus = local_x - sp.I * eta * local_y
    assert sp.simplify(lab_nplus.subs(eta, 1) - sp.sqrt(2 / S) * aq) == 0
    assert sp.simplify(lab_nplus.subs(eta, -1) - sp.sqrt(2 / S) * amq_dag) == 0
    assert sp.simplify(lab_nminus.subs(eta, 1) - sp.sqrt(2 / S) * amq_dag) == 0
    assert sp.simplify(lab_nminus.subs(eta, -1) - sp.sqrt(2 / S) * aq) == 0

    # TB2J's ordered local Nambu vector is (a_q, a^dagger_-q).
    x_operator = (aq + amq_dag) / sp.sqrt(2 * S)
    y_operator = (aq - amq_dag) / (sp.I * sp.sqrt(2 * S))
    x_mode = sp.simplify(x_operator.subs({aq: u * beta, amq_dag: v * beta}))
    y_mode = sp.simplify(y_operator.subs({aq: u * beta, amq_dag: v * beta}))
    assert sp.simplify(sp.diff(x_mode, beta) - (u + v) / sp.sqrt(2 * S)) == 0
    assert sp.simplify(sp.diff(y_mode, beta) - (u - v) / (sp.I * sp.sqrt(2 * S))) == 0
    assert sp.simplify(sp.conjugate(sp.diff(x_mode, beta))) == sp.conjugate(
        (u + v) / sp.sqrt(2 * S)
    )
    assert sp.simplify(sp.conjugate(sp.diff(y_mode, beta))) == sp.conjugate(
        (u - v) / (sp.I * sp.sqrt(2 * S))
    )

    # The two source local frames have U=e1+i e2=(1,+/-i,0).
    U_plus = sp.Matrix([1, sp.I, 0])
    U_minus = sp.Matrix([1, -sp.I, 0])
    assert U_minus == sp.conjugate(U_plus)


def _assert_native_route_derivation():
    """Certify the Option-B local-axis vertex against the frozen generator."""
    u, v, d = sp.symbols("u v d", complex=True)
    D_z = sp.symbols("D_z", real=True)
    sigma_x = sp.Matrix([[0, 1], [1, 0]])
    sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_z = sp.Matrix([[1, 0], [0, -1]])

    for eta_value in (1, -1):
        eta = sp.Integer(eta_value)
        x = u + v
        y = eta * (u - v)
        omega_x = -(u - v)
        omega_y = eta * (u + v)
        # Local-axis positive half-splitting d on frame (x, eta*y, eta*z)
        # enters the lab Hamiltonian as eta*d*sigma_z.
        lab_field = d * eta * sigma_z
        generator = omega_x * sigma_x + omega_y * sigma_y
        bridge = d * (x * sigma_x + y * sigma_y)
        rotation = -sp.I / 2 * (generator * lab_field - lab_field * generator)
        assert sp.simplify(bridge - rotation) == sp.zeros(2)

        # The half-splitting extracted from global spin channels is D_z;
        # the local-axis value is eta*D_z, whose lab coefficient is D_z.
        local_d = eta * D_z
        assert sp.simplify(local_d * eta - D_z) == 0

        # Negative control: conjugating the hole amplitude changes both
        # quadratures for generic complex u, v.
        wrong_x = u + sp.conjugate(v)
        wrong_y = eta * (u - sp.conjugate(v))
        assert sp.simplify(x - wrong_x) != 0
        assert sp.simplify(y - wrong_y) != 0

        # Negative control: flipping eta flips only the y quadrature, and
        # the difference is generically nonzero (free u, v symbols remain).
        flipped_y = -eta * (u - v)
        difference = sp.simplify(y - flipped_y)
        assert difference != 0
        assert difference.free_symbols == {u, v}

    # The eta dependence itself is certified: substituting eta = -1 into the
    # named y amplitude differs from eta = +1 for generic u, v.
    eta = sp.symbols("eta", real=True)
    y_named = eta * (u - v)
    assert sp.simplify(y_named.subs(eta, -1) - y_named.subs(eta, 1)) != 0


def _require_positive_semidefinite(hessian, tolerance=1e-10):
    if not np.allclose(hessian, hessian.conj().T, atol=tolerance, rtol=0.0):
        raise ValueError("energy Hessian is not Hermitian")
    smallest = float(np.linalg.eigvalsh(hessian)[0])
    if smallest < -tolerance:
        raise ValueError("energy Hessian is indefinite")
    return smallest


def _rotate_moment(reference, tangent, angle):
    """Rotate a unit reference moment by angle*|tangent| toward its tangent."""
    magnitude = float(np.linalg.norm(tangent))
    if magnitude == 0.0:
        return reference.copy()
    # The axis is reference x tangent; its cross product with reference is
    # tangent, so this is an independent Rodrigues rotation of the moment.
    direction = reference * np.cos(angle * magnitude)
    direction += (tangent / magnitude) * np.sin(angle * magnitude)
    return direction


def _finite_angle_supercell_oracle(u, v):
    """Independently rotate coherent AFM HP fields in a four-cell supercell."""
    spin_lengths = np.array([1.0, 2.0])
    eta = np.array([1.0, -1.0])
    ncell = 4
    q = 2.0 * np.pi / ncell
    cell_phase = np.exp(1j * q * np.arange(ncell))
    x_mode = (u + v) / np.sqrt(2.0 * spin_lengths)
    y_mode = (u - v) / (1j * np.sqrt(2.0 * spin_lengths))
    eps = 1e-5
    quadrature_errors = {}
    reverse_pair_error = 0.0
    response_by_quadrature = {}

    def local_spin_from_hp_amplitude(alpha, spin):
        """Use the exact polar-angle map for a coherent local HP amplitude."""
        half_sine = abs(alpha) / np.sqrt(2.0 * spin)
        if half_sine >= 1.0:
            raise ValueError("HP amplitude is outside the finite-spin polar map")
        theta = 2.0 * np.arcsin(half_sine)
        sine_theta = np.sin(theta)
        cosine_theta = np.cos(theta)
        if alpha == 0:
            return np.array([0.0, 0.0, 1.0])
        return np.array(
            [
                sine_theta * alpha.real / abs(alpha),
                sine_theta * alpha.imag / abs(alpha),
                cosine_theta,
            ]
        )

    for label, beta in (("cosine", 1.0 + 0.0j), ("sine", 0.0 + 1.0j)):

        def supercell_moments(angle):
            moments = np.empty((ncell, 2, 3), dtype=float)
            for cell in range(ncell):
                for site in range(2):
                    # Build the boson coherent field and its adjoint partner
                    # directly; no candidate x/y response enters this producer.
                    alpha = angle * (
                        u[site] * beta * cell_phase[cell]
                        + np.conjugate(v[site] * beta) * cell_phase[cell].conj()
                    )
                    local_spin = local_spin_from_hp_amplitude(alpha, spin_lengths[site])
                    # TB2J's collinear frames are (x, eta*y, eta*z).
                    lab_spin = local_spin * np.array([1.0, eta[site], eta[site]])
                    moments[cell, site] = spin_lengths[site] * lab_spin
            return moments

        plus = supercell_moments(eps)
        minus = supercell_moments(-eps)
        moment_derivative = (plus - minus) / (2.0 * eps)
        unit_spin_derivative = moment_derivative / spin_lengths[None, :, None]

        # These candidate derivatives are comparison targets only; finite
        # moments above came independently from alpha and the exact polar map.
        target = np.zeros((ncell, 2, 3), dtype=float)
        target[:, :, 0] = 2.0 * np.real(beta * x_mode[None, :] * cell_phase[:, None])
        target[:, :, 1] = (
            eta[None, :] * 2.0 * np.real(beta * y_mode[None, :] * cell_phase[:, None])
        )
        finite_error = float(np.max(np.abs(unit_spin_derivative - target)))
        assert finite_error < 2e-8
        assert np.allclose(
            np.linalg.norm(plus, axis=-1),
            np.broadcast_to(spin_lengths, (ncell, 2)),
            atol=1e-12,
            rtol=0.0,
        )
        quadrature_errors[label] = finite_error

        # The inverse transform uses exp(-i q R); its -q partner is the
        # conjugate Fourier coefficient of these real, finite-angle moments.
        forward_x = (
            np.sum(unit_spin_derivative[:, :, 0] * cell_phase.conj()[:, None], axis=0)
            / ncell
        )
        forward_y = (
            np.sum(unit_spin_derivative[:, :, 1] * cell_phase.conj()[:, None], axis=0)
            / ncell
        )
        reverse_x = (
            np.sum(unit_spin_derivative[:, :, 0] * cell_phase[:, None], axis=0) / ncell
        )
        reverse_y = (
            np.sum(unit_spin_derivative[:, :, 1] * cell_phase[:, None], axis=0) / ncell
        )
        expected_x = beta * x_mode
        expected_y = eta * beta * y_mode
        assert np.allclose(forward_x, expected_x, atol=2e-8, rtol=2e-8)
        assert np.allclose(forward_y, expected_y, atol=2e-8, rtol=2e-8)
        assert np.allclose(reverse_x, forward_x.conj(), atol=2e-8, rtol=2e-8)
        assert np.allclose(reverse_y, forward_y.conj(), atol=2e-8, rtol=2e-8)
        reverse_pair_error = max(
            reverse_pair_error,
            float(np.max(np.abs(reverse_x - forward_x.conj()))),
            float(np.max(np.abs(reverse_y - forward_y.conj()))),
        )
        response_by_quadrature[label] = (forward_x, forward_y)

    # Deliberately conjugate the hole component at q. The independently
    # generated finite-angle Fourier response must reject this wrong map.
    wrong_x = (u + v.conj()) / np.sqrt(2.0 * spin_lengths)
    wrong_y = (u - v.conj()) / (1j * np.sqrt(2.0 * spin_lengths))
    observed_x, observed_y = response_by_quadrature["cosine"]
    wrong_component_mismatch = max(
        float(np.max(np.abs(observed_x - wrong_x))),
        float(np.max(np.abs(observed_y - eta * wrong_y))),
    )
    assert wrong_component_mismatch > 1e-3

    # Deliberately flip the second sublattice eta. The finite-angle Fourier
    # response must reject the wrong lab-frame y target. Compare at the
    # cosine quadrature phase (beta = 1), not the leaked loop phase.
    eta_wrong = eta.copy()
    eta_wrong[1] = -eta_wrong[1]
    observed_y_cosine = response_by_quadrature["cosine"][1]
    wrong_eta_mismatch = float(np.max(np.abs(observed_y_cosine - eta_wrong * y_mode)))
    assert wrong_eta_mismatch > 1e-3
    # The correct eta leaves only the finite-angle tolerance, proving the
    # mismatch above isolates the sign error rather than a tolerance miss.
    correct_eta_residual = float(np.max(np.abs(observed_y_cosine - eta * y_mode)))
    assert correct_eta_residual < 1e-8
    return {
        "finite_angle_max_abs_error_by_quadrature": quadrature_errors,
        "q_minus_q_reverse_pair_max_abs_error": reverse_pair_error,
        "wrong_hole_conjugation_mismatch": wrong_component_mismatch,
        "native_wrong_eta_mismatch": wrong_eta_mismatch,
        "native_correct_eta_residual": correct_eta_residual,
        "spin_lengths": spin_lengths.tolist(),
    }


def run_checks():
    """Run all symbolic and numerical certificates and return observations."""
    assert_fm_and_cholesky()
    _assert_symbolic_derivation()
    _assert_native_route_derivation()

    # This is a two-sublattice AFM Nambu Hessian in the TB2J block order
    # (a_A(q), a_B(q), a_A^dagger(-q), a_B^dagger(-q)). Its complex symmetric
    # pairing block ensures that the hole component has a nontrivial phase.
    pairing = 0.5 * np.exp(0.4j)
    A1 = np.diag([2.0, 3.0]).astype(complex)
    A2 = np.diag([2.5, 1.5]).astype(complex)
    B = np.array([[0.0, pairing], [pairing, 0.0]], dtype=complex)
    hessian = np.block([[A1, B], [B.conj().T, A2]])
    n = 2
    metric = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
    h_min = _require_positive_semidefinite(hessian)
    assert h_min > 1e-10

    K = np.linalg.cholesky(hessian)
    cholesky_pencil = K.conj().T @ metric @ K
    energies, raw_vectors = np.linalg.eigh(cholesky_pencil)
    signed_gH = np.linalg.eigvals(metric @ hessian)
    assert np.max(np.abs(signed_gH.imag)) < 1e-10
    assert np.count_nonzero(energies > 1e-10) == n
    assert np.count_nonzero(energies < -1e-10) == n
    assert np.allclose(
        np.sort(energies),
        np.sort(signed_gH.real),
        atol=1e-10,
        rtol=1e-10,
    )

    positive_indices = np.flatnonzero(energies > 1e-10)
    positive_modes = []
    for index in positive_indices:
        energy = float(energies[index])
        raw = raw_vectors[:, index]
        psi = np.sqrt(energy) * np.linalg.solve(K.conj().T, raw)
        norm = float(np.vdot(psi, metric @ psi).real)
        residual = float(np.linalg.norm(hessian @ psi - energy * metric @ psi))
        assert np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10)
        assert residual < 1e-10
        positive_modes.append((energy, psi, norm))
    energy, psi, metric_norm = min(positive_modes, key=lambda item: item[0])
    u = psi[:n]
    v = psi[n:]
    assert abs(u[0]) > 1e-6
    assert abs(v[1]) > 1e-6
    assert np.max(np.abs(v)) > 1e-3

    oracle = _finite_angle_supercell_oracle(u, v)

    # The physical stability gate rejects an indefinite energy Hessian even
    # if a signed dynamical pencil could be diagonalized.
    unstable = np.diag([1.0, -0.25])
    unstable_rejected = False
    try:
        _require_positive_semidefinite(unstable)
    except ValueError as error:
        unstable_rejected = "indefinite" in str(error)
    assert unstable_rejected

    # A singular AFM global-angle direction is evaluated without a Cholesky
    # inverse or a boson normalization.
    singular_spin_lengths = np.array([1.0, 2.0])
    singular_eta = np.array([1.0, -1.0])
    singular_vector = np.concatenate(
        (np.sqrt(singular_spin_lengths), -np.sqrt(singular_spin_lengths))
    )
    singular_projector = (
        np.outer(singular_vector, singular_vector)
        / np.vdot(singular_vector, singular_vector).real
    )
    singular_hessian = np.eye(4) - singular_projector
    singular_metric = np.diag([1.0, 1.0, -1.0, -1.0])
    assert abs(_require_positive_semidefinite(singular_hessian)) < 1e-14
    assert np.allclose(singular_hessian @ singular_vector, 0.0, atol=1e-14)
    assert np.isclose(
        np.vdot(singular_vector, singular_hessian @ singular_vector).real, 0.0
    )
    assert np.isclose(
        np.vdot(singular_vector, singular_metric @ singular_vector).real, 0.0
    )
    singular_u = singular_vector[:2]
    singular_v = singular_vector[2:]
    singular_x = (singular_u + singular_v) / np.sqrt(2.0 * singular_spin_lengths)
    singular_y = (singular_u - singular_v) / (1j * np.sqrt(2.0 * singular_spin_lengths))
    assert np.allclose(singular_x, 0.0, atol=1e-14)
    singular_beta = -1j / (2.0 * np.sqrt(2.0))
    singular_tangent = np.zeros((2, 3), dtype=float)
    singular_tangent[:, 1] = 2.0 * np.real(singular_beta * singular_y) * singular_eta
    singular_reference = np.zeros((2, 3), dtype=float)
    singular_reference[:, 2] = singular_eta
    singular_eps = 1e-5

    def rotated_singular_moments(angle):
        return np.array(
            [
                singular_spin_lengths[site]
                * _rotate_moment(
                    singular_reference[site], singular_tangent[site], angle
                )
                for site in range(2)
            ]
        )

    singular_plus = rotated_singular_moments(singular_eps)
    singular_minus = rotated_singular_moments(-singular_eps)
    singular_fd = (
        (singular_plus - singular_minus)
        / (2.0 * singular_eps)
        / singular_spin_lengths[:, None]
    )
    singular_angle_error = float(np.max(np.abs(singular_fd - singular_tangent)))
    assert singular_angle_error < 2e-8
    singular_angle_response_magnitude = float(
        np.max(np.linalg.norm(singular_tangent, axis=-1))
    )
    assert np.isclose(singular_angle_response_magnitude, 1.0)
    delta = sp.symbols("delta", positive=True, real=True)
    reduced_metric = sp.diag(1, -1)
    reduced_hessian = sp.Matrix([[1 + delta, 1], [1, 1 + delta]])
    reduced_dynamical = reduced_metric * reduced_hessian
    reduced_characteristic_coefficients = [
        sp.simplify(coefficient)
        for coefficient in reduced_dynamical.charpoly().all_coeffs()
    ]
    expected_characteristic_coefficients = [
        sp.S.One,
        sp.S.Zero,
        -delta * (2 + delta),
    ]
    assert len(reduced_characteristic_coefficients) == len(
        expected_characteristic_coefficients
    )
    assert all(
        sp.simplify(actual - expected) == 0
        for actual, expected in zip(
            reduced_characteristic_coefficients,
            expected_characteristic_coefficients,
        )
    )
    gap_limit = sp.limit(sp.sqrt(delta * (2 + delta)), delta, 0, dir="+")
    assert gap_limit == 0

    # Native Option-B numeric bridge: per-site perturbation-operator form
    # versus the frozen rotation-generator commutator, both sublattice signs,
    # with the conjugated-hole negative control on the operator amplitudes.
    rng = np.random.default_rng(20261101)
    u_num = rng.normal(size=2) + 1j * rng.normal(size=2)
    v_num = rng.normal(size=2) + 1j * rng.normal(size=2)
    d_num = float(rng.uniform(0.5, 1.5))
    sigma_x_num = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y_num = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z_num = np.diag([1.0, -1.0]).astype(complex)
    native_bridge_error = 0.0
    native_wrong_hole_mismatch = 0.0
    for eta_value in (1.0, -1.0):
        eta_num = np.array([eta_value, -eta_value])
        x_num = u_num + v_num
        y_num = eta_num * (u_num - v_num)
        omega_num = np.stack((-(u_num - v_num), eta_num * (u_num + v_num)))
        for site in range(2):
            field_num = d_num * eta_num[site] * sigma_z_num
            generator_num = (
                omega_num[0, site] * sigma_x_num + omega_num[1, site] * sigma_y_num
            )
            rotation_num = -0.5j * (
                generator_num @ field_num - field_num @ generator_num
            )
            bridge_num = d_num * (x_num[site] * sigma_x_num + y_num[site] * sigma_y_num)
            native_bridge_error = max(
                native_bridge_error,
                float(np.max(np.abs(bridge_num - rotation_num))),
            )
            wrong_x_num = u_num[site] + np.conjugate(v_num[site])
            wrong_y_num = eta_num[site] * (u_num[site] - np.conjugate(v_num[site]))
            wrong_bridge_num = d_num * (
                wrong_x_num * sigma_x_num + wrong_y_num * sigma_y_num
            )
            native_wrong_hole_mismatch = max(
                native_wrong_hole_mismatch,
                float(np.max(np.abs(bridge_num - wrong_bridge_num))),
            )
    assert native_bridge_error < 1e-14
    assert native_wrong_hole_mismatch > 1e-3
    return {
        "afm_positive_mode_energy_eV": energy,
        "afm_positive_mode_metric_norm": metric_norm,
        "afm_nonzero_hole_amplitude": float(np.linalg.norm(v)),
        "energy_hessian_min_eigenvalue_eV": h_min,
        "signed_gH_eigenvalues_eV": sorted(float(value.real) for value in signed_gH),
        "unstable_hessian_rejected": unstable_rejected,
        "singular_angle_response_magnitude": singular_angle_response_magnitude,
        "singular_finite_angle_max_abs_error": singular_angle_error,
        "singular_gap_limit_eV": float(gap_limit),
        "singular_boson_normalization_assigned": False,
        "native_bridge_max_abs_error": native_bridge_error,
        "native_wrong_hole_operator_mismatch": native_wrong_hole_mismatch,
        **oracle,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kosmic-json", action="store_true")
    args = parser.parse_args()
    observations = run_checks()
    if args.kosmic_json:
        payload = {
            "schema_version": 1,
            "python_version": platform.python_version(),
            "sympy_version": sp.__version__,
            "results": derive(),
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    print("FM HP, Cholesky metric, and AFM local-frame certificates: OK")
    for name, value in sorted(observations.items()):
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
