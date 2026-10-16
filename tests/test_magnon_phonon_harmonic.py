"""Story 011: harmonic mixed spin/lattice Hessian solver (ADR-005).

TEST-001 controlled two-boson benchmark against independently derived
energies across a detuning sweep.
TEST-002 stability threshold crossing flagged, never regularized.
TEST-003 forbidden scalar-collinear control: rejected input, and a typed
zero-coupling Hessian returns bare modes with zero mixing.
TEST-004 magnetic/displacement weight transfer at resonance matches the
derived expressions.
"""

import numpy as np
import pytest

from TB2J.magnon_phonon.harmonic import (
    ControlledTwoBosonModel,
    HarmonicMixedModel,
    MixedSpinDisplacementHessian,
    hybrid_modes,
)

POINTS = np.array([[0.0, 0.0, 0.0]])


def _derived_energies(a, b, g):
    disc = np.sqrt(((a**2 - b**2) / 2) ** 2 + 4 * a * b * g**2)
    return np.sort(np.sqrt([(a**2 + b**2) / 2 + disc, (a**2 + b**2) / 2 - disc]))


def _derived_share(a, b, g, E):
    r = (E**2 - a**2) / (2 * g * b)
    Wm = 1 + (a / E) ** 2
    Wp = r**2 * (1 + (b / E) ** 2)
    return Wm / (Wm + Wp)


def test_controlled_model_detuning_sweep():
    """TEST-001: solver energies equal the derived closed form across detuning."""
    beta = 2.0
    for ratio in [0.4, 0.7, 1.0, 1.6, 2.5, 4.0]:
        alpha = ratio * beta
        for gscale in [0.05, 0.2, 0.45]:
            g = gscale * np.sqrt(alpha * beta) * 0.9
            model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
            res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
            derived = _derived_energies(alpha, beta, g)
            assert np.allclose(np.sort(res.energies[0]), derived, atol=1e-10)
            assert res.is_stable[0]


def test_stability_threshold_flagged_not_regularized():
    """TEST-002: 4g^2 >= alpha beta is flagged with NaN energies and zero shift."""
    alpha = beta = 2.0
    g_above = np.sqrt(alpha * beta / 4) * 1.05
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g_above)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert not res.is_stable[0]
    assert res.diagnostics.stability_flag[0]
    assert np.all(np.isnan(res.energies[0]))
    assert res.diagnostics.shift[0] == 0.0
    g_below = np.sqrt(alpha * beta / 4) * 0.95
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g_below)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert res.is_stable[0]
    assert not res.diagnostics.stability_flag[0]


def test_forbidden_scalar_collinear_input_rejected():
    """TEST-003a: a scalar-collinear-derived Hessian is rejected at construction."""
    with pytest.raises(ValueError, match="scalar"):
        MixedSpinDisplacementHessian(
            n_mag_sites=1,
            displacement_labels=((0, 2),),
            producer="scalar_collinear_exchange_striction",
            magnon_bare=lambda pts: np.zeros((len(pts), 2, 2)),
            phonon_bare=lambda pts: np.ones((len(pts), 1, 1)),
            mixed_derivative=lambda pts: np.ones((len(pts), 1, 1)),
            spin_lengths=(2.0,),
            provenance={"source": "single collinear transverse measurement"},
        )


def test_scalar_collinear_forbidden_control_returns_bare_modes():
    """TEST-003b: scalar-collinear control produces bare modes, never mixing."""
    alpha, beta = 2.0, 1.0

    def magnon_bare(points):
        result = np.zeros((len(points), 2, 2))
        result[:, 0, 0] = alpha
        result[:, 1, 1] = alpha
        return result

    mixed = MixedSpinDisplacementHessian.scalar_collinear_forbidden_control(
        n_mag_sites=1,
        displacement_labels=((0, 2),),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.full((len(pts), 1, 1), beta**2),
        spin_lengths=(2.0,),
        provenance={"source": "scalar-collinear exchange input"},
    )
    res = hybrid_modes(POINTS, mixed)
    energies = np.sort(res.energies[0])
    assert np.allclose(energies, np.sort([alpha, beta]), atol=1e-12)
    # each bare mode carries unit weight on its own species
    order = np.argsort(res.energies[0])
    assert np.allclose(res.displacement_weight[0][order], [1.0, 0.0], atol=1e-12)
    assert np.allclose(res.magnetic_weight[0][order], [0.0, 1.0], atol=1e-12)
    assert np.allclose(
        res.magnetic_weight[0] + res.displacement_weight[0], 1.0, atol=1e-12
    )


def test_weight_transfer_at_resonance():
    """TEST-004: weight shares match the derived expressions on and off resonance."""
    beta = 1.5
    for ratio in [0.6, 1.0, 2.2]:
        alpha = ratio * beta
        for gscale in [0.1, 0.35]:
            g = gscale * np.sqrt(alpha * beta)
            model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
            res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
            derived = _derived_energies(alpha, beta, g)
            for mode in range(2):
                share = _derived_share(alpha, beta, g, derived[mode])
                t = res.paraunitary[0][:, mode]
                wm = abs(t[0]) ** 2 + abs(t[2]) ** 2
                wp = abs(t[1]) ** 2 + abs(t[3]) ** 2
                assert abs(res.magnetic_weight[0][mode] - share) < 1e-9
                assert abs(wm / (wm + wp) - share) < 1e-9
                assert abs(res.displacement_weight[0][mode] - (1 - share)) < 1e-9
    # exact resonance: both branches share weights equally
    alpha = beta = 2.0
    g = 0.4
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
    res = hybrid_modes(POINTS, ControlledTwoBosonModel.to_mixed_hessian(model))
    assert np.allclose(res.magnetic_weight[0], 0.5, atol=1e-12)
    assert np.allclose(res.displacement_weight[0], 0.5, atol=1e-12)


def test_hybrid_cli_manifest_result():
    """The hybrid CLI calculation accepts the controlled JSON-manifest shape."""
    from TB2J.magnon_phonon.cli import _run_hybrid

    result = _run_hybrid(
        {
            "calculation": "hybrid",
            "model": {
                "type": "controlled_two_boson",
                "alpha_eV": 2.0,
                "beta_eV": 1.0,
                "g_eV": 0.1,
            },
            "points": [[0.0, 0.0, 0.0]],
        }
    )
    assert result["calculation"] == "hybrid"
    assert result["is_stable"] == [True]
    assert len(result["energies_eV"][0]) == 2


def test_hybrid_cli_accepts_typed_npz_hessian(tmp_path):
    """The CLI loads pre-supplied typed Hessian blocks outside the kernel."""
    from TB2J.magnon_phonon.cli import _run_hybrid

    path = tmp_path / "mixed.npz"
    np.savez(
        path,
        points=POINTS,
        magnon_bare=np.array([[[2.0, 0.0], [0.0, 2.0]]]),
        phonon_bare=np.array([[[1.0]]]),
        coupling=np.array([[[0.1 * np.sqrt(2 * 1.0)]]]),  # physical G for g=0.1, S=2
    )
    result = _run_hybrid(
        {
            "calculation": "hybrid",
            "model": {
                "type": "mixed_hessian_npz",
                "npz_path": str(path),
                "n_mag_sites": 1,
                "spin_coordinate_labels": [[0, "local-transverse-complex"]],
                "displacement_labels": [[0, 2]],
                "spin_lengths": [2.0],
                "displacement_coordinate": "mass_weighted",
                "producer": "controlled_mixed_torque",
                "provenance": {"source": "test controlled Hessian"},
            },
            "points": POINTS.tolist(),
        }
    )
    assert result["is_stable"] == [True]
    assert np.allclose(np.sort(result["energies_eV"][0]), _derived_energies(2, 1, 0.1))


def test_model_hybrid_modes_facade():
    """The model facade exposes the requested hybrid_modes(points, Hessian) API."""
    model = ControlledTwoBosonModel(alpha_eV=2.0, beta_eV=1.0, g_eV=0.15)
    mixed = model.to_mixed_hessian()
    result = HarmonicMixedModel(mixed).hybrid_modes(POINTS, mixed)
    assert np.allclose(result.energies[0], _derived_energies(2.0, 1.0, 0.15))


def test_hybrid_cli_writes_strict_json_for_unstable_modes(monkeypatch, capsys):
    """Flagged NaN numerical outputs are emitted as standards-compliant JSON null."""
    import io
    import json
    import sys

    from TB2J.magnon_phonon.cli import main

    manifest = {
        "calculation": "hybrid",
        "model": {
            "type": "controlled_two_boson",
            "alpha_eV": 2.0,
            "beta_eV": 2.0,
            "g_eV": 1.05,
        },
        "points": [[0.0, 0.0, 0.0]],
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(manifest)))
    main(["hybrid", "-"])
    result = json.loads(capsys.readouterr().out)
    assert result["is_stable"] == [False]
    assert result["energies_eV"] == [[None, None]]
    assert result["diagnostics"]["shift"] == [0.0]


def _nonreciprocal_hessian(j0, j1, beta_sq):
    """Hand-built nonreciprocal one-site FM Hessian: J(k) = j0 + j1 sin(2pi k).

    The Nambu block is [[J(k), 0], [0, J(-k)]] with real J: its lower
    block is the opposite-k partner, NOT the conjugate of the upper block.
    """

    def magnon_bare(points):
        out = np.zeros((len(points), 2, 2), dtype=complex)
        for i, k in enumerate(points):
            jk = j0 + j1 * np.sin(2 * np.pi * k[0])
            jmk = j0 + j1 * np.sin(-2 * np.pi * k[0])
            out[i, 0, 0] = jk
            out[i, 1, 1] = jmk
        return out

    return MixedSpinDisplacementHessian.scalar_collinear_forbidden_control(
        n_mag_sites=1,
        displacement_labels=((0, 2),),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.full((len(pts), 1, 1), beta_sq),
        spin_lengths=(2.0,),
        provenance={"source": "nonreciprocal zero-mixing control"},
    )


def test_zero_mixing_preserves_nonreciprocal_bare_magnons():
    """TEST-005: hybrid zero-mixing energies equal the supplied Hq at k and -k."""
    j0, j1, beta = 2.0, 0.4, 1.3
    kpts = np.array([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]])
    mixed = _nonreciprocal_hessian(j0, j1, beta**2)
    res = hybrid_modes(kpts, mixed)
    for i, k in enumerate(kpts):
        jk = j0 + j1 * np.sin(2 * np.pi * k[0])
        expected = np.sort([jk, beta])
        assert np.allclose(np.sort(res.energies[i]), expected, atol=1e-12), (
            k,
            res.energies[i],
        )


def test_physical_hessian_conversion_spin_and_zeropoint():
    """TEST-006: physical mixed derivative converts to bosonic coupling once."""
    alpha, beta, g = 2.0, 1.0, 0.12
    # controlled model via its physical mixed derivative; spin length S = 2
    # gives sqrt(S/2) = 1 and the single mass-weighted mode carries
    # e = 1, omega = beta, so G = g sqrt(2 beta).
    model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
    res = hybrid_modes(POINTS, model.to_mixed_hessian())
    assert np.allclose(
        np.sort(res.energies[0]), _derived_energies(alpha, beta, g), atol=1e-10
    )

    # same physical coupling expressed with spin length S = 1: the Hessian
    # must scale by sqrt(2/S); the solver must undo it exactly once.
    mixed_s1 = MixedSpinDisplacementHessian(
        n_mag_sites=1,
        displacement_labels=((0, 0),),
        spin_lengths=(1.0,),
        displacement_coordinate="mass_weighted",
        magnon_bare=lambda pts: np.array(
            [[[alpha, 0.0], [0.0, alpha]]] * len(pts), dtype=complex
        ),
        phonon_bare=lambda pts: np.full((len(pts), 1, 1), beta**2),
        mixed_derivative=lambda pts: np.full(
            (len(pts), 1, 1), g * np.sqrt(2 * beta) / np.sqrt(0.5)
        ),
        producer="controlled_two_boson_s1",
        provenance={"source": "S=1 conversion check"},
    )
    res_s1 = hybrid_modes(POINTS, mixed_s1)
    assert np.allclose(
        np.sort(res_s1.energies[0]), _derived_energies(alpha, beta, g), atol=1e-10
    )

    # Cartesian-basis variant: masses convert the derivative exactly once.
    mass = 4.0  # amu
    mixed_cart = MixedSpinDisplacementHessian(
        n_mag_sites=1,
        displacement_labels=((0, 0),),
        spin_lengths=(2.0,),
        displacement_coordinate="cartesian_angstrom",
        masses_amu=(mass,),
        magnon_bare=lambda pts: np.array(
            [[[alpha, 0.0], [0.0, alpha]]] * len(pts), dtype=complex
        ),
        phonon_bare=lambda pts: np.full((len(pts), 1, 1), beta**2),
        mixed_derivative=lambda pts: np.full(
            (len(pts), 1, 1), g * np.sqrt(2 * beta) * np.sqrt(mass)
        ),
        producer="controlled_two_boson_cartesian",
        provenance={"source": "Cartesian conversion check"},
    )
    res_cart = hybrid_modes(POINTS, mixed_cart)
    assert np.allclose(
        np.sort(res_cart.energies[0]), _derived_energies(alpha, beta, g), atol=1e-10
    )


def test_observable_strengths_match_operator_matrix_elements():
    """TEST-007: physical strengths equal derived operator residues."""
    spin_length = 2.0
    beta = 1.5
    for ratio in [0.5, 1.0, 2.4]:
        alpha = ratio * beta
        g = 0.3 * np.sqrt(alpha * beta)
        model = ControlledTwoBosonModel(alpha_eV=alpha, beta_eV=beta, g_eV=g)
        res = hybrid_modes(POINTS, model.to_mixed_hessian())
        for mode in range(2):
            t = res.paraunitary[0][:, mode]
            tb = t[0]
            ta = t[1]
            tah = t[3]
            mag_expect = 2 * spin_length * abs(tb) ** 2
            dis_expect = abs(ta + np.conj(tah)) ** 2 / (2 * beta)
            assert abs(res.magnetic_strength[0, mode] - mag_expect) < 1e-10
            assert abs(res.displacement_strength[0, mode] - dis_expect) < 1e-10
    # bare limits: g -> 0 gives pure magnon M = 2S and pure phonon D = 1/(2 beta)
    model = ControlledTwoBosonModel(alpha_eV=2.0, beta_eV=1.0, g_eV=1e-12)
    res = hybrid_modes(POINTS, model.to_mixed_hessian())
    order = np.argsort(res.energies[0])  # [phonon-like 1.0, magnon-like 2.0]
    assert abs(res.magnetic_strength[0][order[0]] - 0.0) < 1e-9
    assert abs(res.displacement_strength[0][order[0]] - 1.0 / (2 * 1.0)) < 1e-9
    assert abs(res.magnetic_strength[0][order[1]] - 2 * spin_length) < 1e-9
    assert abs(res.displacement_strength[0][order[1]] - 0.0) < 1e-9


def test_multimode_mixed_eigenvectors_lehmann():
    """TEST-008: mixed unequal-frequency phonons match truncated-Fock Lehmann.

    Two displacement coordinates whose mass-weighted dynamical matrix has
    rotated eigenvectors and distinct energies exercise the per-mode
    zero-point ordering; the exact one-quantum matrix elements come from
    independent sparse diagonalization of the same bosonic Hamiltonian.
    """
    from TB2J.magnon_phonon.harmonic import MixedSpinDisplacementHessian

    alpha, omega1, omega2 = 2.0, 0.9, 2.3
    spin_length = 2.0
    g = 0.15 * np.sqrt(alpha * omega1)  # visible hole admixture, stable
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dyn = rot @ np.diag([omega1**2, omega2**2]) @ rot.T

    def magnon_bare(pts):
        out = np.zeros((len(pts), 2, 2), dtype=complex)
        out[:, 0, 0] = alpha
        out[:, 1, 1] = alpha
        return out

    mixed = MixedSpinDisplacementHessian(
        n_mag_sites=1,
        displacement_labels=((0, 0), (0, 1)),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.array([dyn] * len(pts)),
        mixed_derivative=lambda pts: np.full((len(pts), 1, 2), g * np.sqrt(2 * omega1)),
        spin_lengths=(spin_length,),
        producer="multimode_control",
        provenance={"source": "mixed unequal-omega Lehmann fixture"},
    )
    res = hybrid_modes(POINTS, mixed)

    # independent truncated-Fock reference: H and w_beta as sparse matrices
    nb = 7  # quanta cutoff per boson; truncated-Fock reference converges
    # to the harmonic BdG values within ~1e-3 at this coupling
    dim = nb**3

    def state(ib, i1, i2):
        return (ib * nb + i1) * nb + i2

    def op(kind, which):
        mat = np.zeros((dim, dim))
        for ib in range(nb):
            for i1 in range(nb):
                for i2 in range(nb):
                    s = state(ib, i1, i2)
                    occ = (ib, i1, i2)[which]
                    if kind == "n":
                        mat[s, s] = occ
                    elif kind == "p" and occ + 1 < nb:  # a^+
                        t = state(
                            ib + (which == 0),
                            i1 + (which == 1),
                            i2 + (which == 2),
                        )
                        mat[t, s] = np.sqrt(occ + 1)
                    elif kind == "m" and occ > 0:  # a
                        t = state(
                            ib - (which == 0),
                            i1 - (which == 1),
                            i2 - (which == 2),
                        )
                        mat[t, s] = np.sqrt(occ)
        return mat

    b, a1, a2 = op("m", 0), op("m", 1), op("m", 2)
    b_d, a1_d, a2_d = op("p", 0), op("p", 1), op("p", 2)
    b_n, a1_n, a2_n = op("n", 0), op("n", 1), op("n", 2)
    ham = (alpha * b_n + omega1 * a1_n + omega2 * a2_n).astype(complex)
    # H_int = sum_nu Gamma_nu (b+b^+)(a_nu+a_nu^+); the Gamma_nu are taken
    # from the same conversion routine the solver uses.
    from TB2J.magnon_phonon.harmonic import _bosonic_coupling

    gamma_block = _bosonic_coupling(
        mixed,
        np.zeros(3),
        np.array([omega1, omega2]),
        rot,
    )[0]
    a_ops = [a1 + a1_d, a2 + a2_d]
    for nu, a_pair in enumerate(a_ops):
        ham += gamma_block[nu] * (b + b_d) @ a_pair
    evals, evecs = np.linalg.eigh(ham)
    ground = evecs[:, 0]
    # identify the three one-quantum-like states by total-quanta expectation
    # (multi-quanta states interleave the sorted spectrum)
    n_op = b_n + a1_n + a2_n
    quanta = np.abs(np.conj(evecs).T @ (n_op @ evecs)).diagonal()
    one_q = sorted(np.argsort(np.abs(quanta - 1.0))[:3])

    def strength(op_beta):
        elem = np.array([ground @ (op_beta @ evecs[:, idx]) for idx in one_q])
        return np.abs(elem) ** 2

    # physical displacement: per-mode zero point before the eigenvector mix
    w0 = rot[0, 0] * (a1 + a1_d) / np.sqrt(2 * omega1) + rot[0, 1] * (
        a2 + a2_d
    ) / np.sqrt(2 * omega2)
    w1 = rot[1, 0] * (a1 + a1_d) / np.sqrt(2 * omega1) + rot[1, 1] * (
        a2 + a2_d
    ) / np.sqrt(2 * omega2)
    lehmann_dis = strength(w0) + strength(w1)
    lehmann_mag = strength(np.sqrt(2 * spin_length) * b)

    order = np.argsort(res.energies[0])
    for rank, mode in enumerate(order):
        lehmann_idx = one_q[rank]
        # solver energies are excitations relative to the ground state
        assert abs(res.energies[0][mode] - (evals[lehmann_idx] - evals[0])) < 5e-3
        assert abs(res.displacement_strength[0][mode] - lehmann_dis[rank]) < 5e-3
        assert abs(res.magnetic_strength[0][mode] - lehmann_mag[rank]) < 5e-3


def test_complex_coupling_lehmann_raw_nambu_sum():
    """TEST-009: complex coupling distinguishes raw-sum from conjugated hole.

    The Fock reference builds its coupling independently from the fixture
    G, spin length and mode energies through the derived expression
    Gamma_nu = G_beta e_{beta nu} sqrt(S/2)/sqrt(2 omega_nu).
    """
    from TB2J.magnon_phonon.harmonic import MixedSpinDisplacementHessian

    alpha, omega1, omega2 = 2.0, 1.1, 2.1
    spin_length = 2.0
    theta = 0.9
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dyn = rot @ np.diag([omega1**2, omega2**2]) @ rot.T
    phase = np.exp(0.8j)
    g_row = np.array([[0.10, 0.10 * 0.5 * phase]]) * np.sqrt(2 * omega1)

    def magnon_bare(pts):
        out = np.zeros((len(pts), 2, 2), dtype=complex)
        out[:, 0, 0] = alpha
        out[:, 1, 1] = alpha
        return out

    mixed = MixedSpinDisplacementHessian(
        n_mag_sites=1,
        displacement_labels=((0, 0), (0, 1)),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.array([dyn] * len(pts)),
        mixed_derivative=lambda pts: np.array([g_row] * len(pts)),
        spin_lengths=(spin_length,),
        producer="complex_coupling_control",
        provenance={"source": "complex coupling raw-sum check"},
    )
    res = hybrid_modes(POINTS, mixed)

    # independent coupling arithmetic (no solver conversion routine)
    gammas = np.array(
        [
            (g_row @ rot)[0, nu]
            * np.sqrt(spin_length / 2.0)
            / np.sqrt(2 * (omega1 if nu == 0 else omega2))
            for nu in range(2)
        ]
    )

    nb = 7
    dim = nb**3

    def state(ib, i1, i2):
        return (ib * nb + i1) * nb + i2

    def op(kind, which):
        mat = np.zeros((dim, dim))
        for ib in range(nb):
            for i1 in range(nb):
                for i2 in range(nb):
                    s = state(ib, i1, i2)
                    occ = (ib, i1, i2)[which]
                    if kind == "n":
                        mat[s, s] = occ
                    elif kind == "p" and occ + 1 < nb:
                        mat[
                            state(
                                ib + (which == 0), i1 + (which == 1), i2 + (which == 2)
                            ),
                            s,
                        ] = np.sqrt(occ + 1)
                    elif kind == "m" and occ > 0:
                        mat[
                            state(
                                ib - (which == 0), i1 - (which == 1), i2 - (which == 2)
                            ),
                            s,
                        ] = np.sqrt(occ)
        return mat

    b, a1, a2 = op("m", 0), op("m", 1), op("m", 2)
    b_d, a1_d, a2_d = op("p", 0), op("p", 1), op("p", 2)
    n_op = op("n", 0) + op("n", 1) + op("n", 2)
    ham = (alpha * op("n", 0) + omega1 * op("n", 1) + omega2 * op("n", 2)).astype(
        complex
    )
    for gamma_nu, a_pair in zip(gammas, [a1 + a1_d, a2 + a2_d]):
        ham = ham + gamma_nu * (b + b_d) @ a_pair
    evals, evecs = np.linalg.eigh(ham)
    ground = evecs[:, 0]
    quanta = np.real(np.einsum("ij,jk,ki->i", evecs.conj().T, n_op, evecs))
    one_q = sorted(np.argsort(np.abs(quanta - 1.0))[:3])

    def strength(op_beta):
        return np.abs([ground @ (op_beta @ evecs[:, idx]) for idx in one_q]) ** 2

    w0 = rot[0, 0] * (a1 + a1_d) / np.sqrt(2 * omega1) + rot[0, 1] * (
        a2 + a2_d
    ) / np.sqrt(2 * omega2)
    w1 = rot[1, 0] * (a1 + a1_d) / np.sqrt(2 * omega1) + rot[1, 1] * (
        a2 + a2_d
    ) / np.sqrt(2 * omega2)
    lehmann_dis = strength(w0) + strength(w1)
    lehmann_mag = strength(np.sqrt(2 * spin_length) * b)

    order = np.argsort(res.energies[0])
    for rank, mode in enumerate(order):
        idx = one_q[rank]
        assert abs(res.energies[0][mode] - (evals[idx] - evals[0])) < 5e-3
        assert abs(res.displacement_strength[0][mode] - lehmann_dis[rank]) < 5e-3
        assert abs(res.magnetic_strength[0][mode] - lehmann_mag[rank]) < 5e-3


def test_conversion_from_declared_energy_finite_difference():
    """TEST-010: independently differentiated Cartesian energy, mixed
    spins/masses/frequencies, matches a separate Fock-space oracle."""
    from TB2J.magnon_phonon.harmonic import MixedSpinDisplacementHessian

    alphas = np.array([1.8, 2.4])
    omegas = np.array([0.8, 2.1])
    spins = np.array([1.0, 3.0])
    masses = np.array([4.0, 9.0])
    coeff = np.array([[0.4, 0.2], [0.25, 0.35]])  # declared E = sum C_ij s_i u_j

    def magnon_bare(pts):
        out = np.zeros((len(pts), 4, 4), dtype=complex)
        out[:, :2, :2] = np.diag(alphas)
        out[:, 2:, 2:] = np.diag(alphas)
        return out

    # Independently finite-difference the declared physical energy function.
    h = 1e-4

    def energy(spin_vec, displacement):
        return float(spin_vec @ coeff @ displacement)

    G_cart = np.zeros_like(coeff)
    for i in range(2):
        for j in range(2):
            si_p, si_m = np.zeros(2), np.zeros(2)
            uj_p, uj_m = np.zeros(2), np.zeros(2)
            si_p[i], si_m[i] = h, -h
            uj_p[j], uj_m[j] = h, -h
            G_cart[i, j] = (
                energy(si_p, uj_p)
                - energy(si_m, uj_p)
                - energy(si_p, uj_m)
                + energy(si_m, uj_m)
            ) / (4 * h * h)
    assert np.allclose(G_cart, coeff, atol=1e-9)

    mixed = MixedSpinDisplacementHessian(
        n_mag_sites=2,
        displacement_labels=((0, 2), (1, 2)),
        magnon_bare=magnon_bare,
        phonon_bare=lambda pts: np.array([np.diag(omegas**2)] * len(pts)),
        mixed_derivative=lambda pts: np.array([G_cart] * len(pts)),
        spin_lengths=tuple(spins),
        displacement_coordinate="cartesian_angstrom",
        masses_amu=tuple(masses),
        producer="declared_energy_control",
        provenance={"source": "finite-differenced E(s_i,u_j)"},
    )
    result = hybrid_modes(POINTS, mixed)

    # Independent conversion from the declared energy derivatives:
    # Cartesian u -> mass-weighted w, then HP spin and phonon zero-point.
    expected_gamma = (
        G_cart
        / np.sqrt(masses)[None, :]
        * np.sqrt(spins / 2.0)[:, None]
        / np.sqrt(2.0 * omegas)[None, :]
    )

    # Independent truncated-Fock Lehmann oracle for four unit-commutator
    # bosons; the Fock coefficients above use expected_gamma directly.
    nb = 4
    dim = nb**4

    def state(n0, n1, n2, n3):
        return ((n0 * nb + n1) * nb + n2) * nb + n3

    def boson_op(kind, which):
        mat = np.zeros((dim, dim))
        for occ in np.ndindex((nb, nb, nb, nb)):
            col = state(*occ)
            n = occ[which]
            if kind == "n":
                mat[col, col] = n
            elif kind == "p" and n + 1 < nb:
                target = list(occ)
                target[which] += 1
                mat[state(*target), col] = np.sqrt(n + 1)
            elif kind == "m" and n > 0:
                target = list(occ)
                target[which] -= 1
                mat[state(*target), col] = np.sqrt(n)
        return mat

    annih = [boson_op("m", i) for i in range(4)]
    create = [boson_op("p", i) for i in range(4)]
    number = [boson_op("n", i) for i in range(4)]
    ham = sum(alphas[i] * number[i] for i in range(2))
    ham = ham + sum(omegas[j] * number[2 + j] for j in range(2))
    for i in range(2):
        for j in range(2):
            ham = ham + expected_gamma[i, j] * (annih[i] + create[i]) @ (
                annih[2 + j] + create[2 + j]
            )
    evals, evecs = np.linalg.eigh(ham)
    ground = evecs[:, 0]
    n_total = sum(number)
    n_expect = np.real(np.einsum("ij,jk,ki->i", evecs.conj().T, n_total, evecs))
    one_q = sorted(np.argsort(np.abs(n_expect - 1.0))[:4])

    def lehmann_strength(operator):
        return np.abs([ground @ (operator @ evecs[:, idx]) for idx in one_q]) ** 2

    mag_strength = sum(
        lehmann_strength(np.sqrt(2 * spins[i]) * annih[i]) for i in range(2)
    )
    disp_strength = sum(
        lehmann_strength(
            (annih[2 + j] + create[2 + j]) / np.sqrt(2 * omegas[j] * masses[j])
        )
        for j in range(2)
    )
    for rank, mode in enumerate(np.argsort(result.energies[0])):
        idx = one_q[rank]
        assert abs(result.energies[0, mode] - (evals[idx] - evals[0])) < 1e-2
        assert abs(result.magnetic_strength[0, mode] - mag_strength[rank]) < 1e-2
        assert abs(result.displacement_strength[0, mode] - disp_strength[rank]) < 1e-2
