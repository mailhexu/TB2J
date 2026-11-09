"""Anomalous bosonic modes and validity diagnostics (ticket 03).

The anomalous benchmark is the honeycomb FM of ticket 02 plus a small
symmetric anisotropic NN exchange (Jani) delta*diag(1,-1,0): the certified
pairing contraction b(J) = J_xx - J_yy = 2*delta makes the model genuinely
anomalous (B != 0) while the spectrum stays stable. Conventions are pinned by
``docs/sympy/magnon_topology_bosonic_modes.py``.
"""

import json
import sys

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.magnon.topology import BulkRequest, calculate_bulk, prepare_topology_model
from TB2J.magnon.topology.bosonic import solve_bosonic_modes

J1 = 0.01
J2 = 0.002
K_SIA = 0.001
DELTA = 0.0003


def honeycomb_spinio(D=0.0005, delta=0.0):
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]])
    atoms = Atoms(
        "Cr2",
        scaled_positions=[[0.0, 0.0, 0.0], [2 / 3, -1 / 3, 0.0]],
        cell=basis,
        pbc=True,
    )
    magmoms = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    exc, dmi, jani = {}, {}, {}

    def rev(R):
        return tuple(-np.array(R))

    for R in [(0, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        exc[(R, 0, 1)] = J1
        exc[(rev(R), 1, 0)] = J1
        if delta:
            t = delta * np.diag([1.0, -1.0, 0.0])
            jani[(R, 0, 1)] = t
            jani[(rev(R), 1, 0)] = t.T
    for R in [(1, 0, 0), (-1, 1, 0), (0, -1, 0)]:
        exc[(R, 0, 0)] = J2
        exc[(rev(R), 0, 0)] = J2
        exc[(R, 1, 1)] = J2
        exc[(rev(R), 1, 1)] = J2
        dmi[(R, 0, 0)] = np.array([0.0, 0.0, D])
        dmi[(rev(R), 0, 0)] = np.array([0.0, 0.0, -D])
        dmi[(R, 1, 1)] = np.array([0.0, 0.0, -D])
        dmi[(rev(R), 1, 1)] = np.array([0.0, 0.0, D])
    sia = {0: np.diag([0.0, 0.0, K_SIA]), 1: np.diag([0.0, 0.0, K_SIA])}
    spinio = SpinIO(
        atoms=atoms,
        spinat=magmoms,
        charges=np.zeros(2),
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        dmi_ddict=dmi if D else None,
        Jani_dict=jani if delta else None,
        sia_tensor=sia,
    )
    spinio._build_Rlist()
    return spinio


@pytest.fixture(scope="module")
def anomalous_model():
    return prepare_topology_model(honeycomb_spinio(delta=DELTA), source_path="bench")


@pytest.fixture(scope="module")
def baseline_model():
    return prepare_topology_model(honeycomb_spinio(delta=0.0), source_path="bench")


def test_anomalous_hole_block_is_the_conjugate_negative_momentum_block(
    anomalous_model,
):
    from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian

    kpoint = np.array([[0.11, 0.23, 0.0]])
    hamiltonian = local_frame_bosonic_hamiltonian(
        anomalous_model.JR,
        anomalous_model.Rlist,
        anomalous_model.magmoms,
        kpoint,
    )[0]
    reverse = local_frame_bosonic_hamiltonian(
        anomalous_model.JR,
        anomalous_model.Rlist,
        anomalous_model.magmoms,
        -kpoint,
    )[0]
    particle = hamiltonian[:2, :2]
    hole = hamiltonian[2:, 2:]
    np.testing.assert_allclose(particle, particle.conj().T, atol=1e-12)
    assert np.max(np.abs(particle - particle.conj())) > 1e-6
    np.testing.assert_allclose(hole, reverse[:2, :2].conj(), atol=1e-12)

    real_bond_model = prepare_topology_model(
        honeycomb_spinio(D=0.0, delta=DELTA), source_path="real-bond"
    )
    real_bond_h = local_frame_bosonic_hamiltonian(
        real_bond_model.JR,
        real_bond_model.Rlist,
        real_bond_model.magmoms,
        kpoint,
    )[0]
    np.testing.assert_allclose(real_bond_h[2:, 2:], real_bond_h[:2, :2], atol=1e-12)


def test_anomalous_pairing_present_and_stable(anomalous_model):
    magnon = anomalous_model.magnon()
    H = magnon.Hq(np.array([[0.11, 0.23, 0.0]]))[0]
    assert np.max(np.abs(H[:2, 2:])) > 1e-6  # B != 0: genuinely anomalous
    m = solve_bosonic_modes(H)
    assert m.classification == "positive_definite"
    assert m.residuals["max_metric_norm_error"] < 1e-8
    assert m.residuals["max_generalized_residual"] < 1e-10


def test_anomalous_bulk_matches_baseline_chern(anomalous_model, baseline_model):
    # small delta opens no gap closing: same pinned band Chern as ticket 02
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(12, 12), bands=[0]))
    ref = calculate_bulk(baseline_model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert res.valid
    assert res.chern == ref.chern == 1
    assert res.diagnostics["solver"]["max_pairing_residual"] < 1e-8


def test_anomalous_cli_end_to_end(tmp_path):
    from TB2J.magnon.magnon_cli import main

    exc = honeycomb_spinio(delta=DELTA)
    root = tmp_path / "res_anomalous"
    exc.write_pickle(str(root))
    out = tmp_path / "topo.json"
    argv = sys.argv
    sys.argv = [
        "TB2J_magnon",
        "-p",
        str(root),
        "--topology",
        "--topology-mesh",
        "12",
        "12",
        "--topology-bands",
        "0",
        "--topology-out",
        str(out),
    ]
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        sys.argv = argv
    assert exit_info.value.code == 0
    data = json.loads(out.read_text())
    assert data["valid"] is True and data["chern"] == 1


def test_semidefinite_goldstone_reports_hessian_nullspace_without_singular_solve():
    H = np.ones((2, 2), dtype=complex)

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.null_space.shape == (2, 1)
    np.testing.assert_allclose(H @ modes.null_space, 0.0, atol=1e-12)
    assert modes.residuals["null_space_residual"] < 1e-12


def _semidefinite_goldstone_hessian():
    # legacy doubled form [[A, 0], [0, A*]] with a rank-1 FM Goldstone block:
    # spectrum {+0.016, +0.016, dust, dust} like the SIESTA CrI3 collinear
    # Gamma point. The |dust| < zero_tol negative mode makes the Cholesky
    # failure deterministic while staying inside the semidefinite gate, so
    # the singular-square-root branch is pinned.
    p = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    q = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    a = 0.016 * p - 1e-12 * q
    return np.block([[a, np.zeros((2, 2))], [np.zeros((2, 2)), a.conj()]])


def test_semidefinite_hessian_with_positive_modes_classifies_zero_mode():
    # negative roundoff dust below zero_tol makes the Cholesky failure
    # deterministic, pinning the singular-square-root branch.
    H = _semidefinite_goldstone_hessian()

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.null_space.shape == (4, 2)
    np.testing.assert_allclose(H @ modes.null_space, 0.0, atol=1e-12)
    assert modes.residuals["null_space_residual"] < 1e-12
    assert modes.residuals["n_zero_modes"] == 1
    # positive mode is still returned, normalized and consistent with
    # H v = omega g v (Colpa convention, sqrt(omega)-rescaled to unit norm).
    assert modes.vectors.shape == (4, 1)
    n = 2
    g = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))
    norm = np.real(modes.vectors[:, 0].conj() @ g @ modes.vectors[:, 0])
    assert abs(norm - 1.0) < 1e-12
    omega = modes.energies[modes.positive_indices[0]]
    resid = H @ modes.vectors[:, 0] - omega * (g @ modes.vectors[:, 0])
    assert np.max(np.abs(resid)) < 1e-12
    assert modes.residuals["max_generalized_residual"] < 1e-12
    assert modes.residuals["max_metric_norm_error"] < 1e-12


def test_singular_hessian_with_pairing_channel_exact_pencil_solution():
    # nonzero symmetric-anisotropy pairing on a singular Goldstone Hessian:
    # the generalized eigenvector leaves range(H) so that g v stays in
    # range(H); the reduced pencil must return exact identities (a plain
    # range-restricted back-transform misses the null-space admixture by
    # ~4e-3 in the generalized residual).
    s2 = np.sqrt(0.5)
    rot = np.array([[s2, 0, s2, 0], [0, s2, 0, s2], [s2, 0, -s2, 0], [0, s2, 0, -s2]])
    p = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])
    q = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    x, b, dust = 0.016, 0.003, -1e-12
    diag_blocks = np.block(
        [
            [(x + b) * p + dust * q, np.zeros((2, 2))],
            [np.zeros((2, 2)), (x - b) * p + dust * q],
        ]
    )
    H = rot.T @ diag_blocks @ rot  # [[A, B], [B, A]] physical doubled form
    assert np.max(np.abs(H[:2, 2:])) > 1e-6  # genuinely paired

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    omega = modes.energies[modes.positive_indices[0]]
    # true generalized eigenvalue of the exactly-singular clamped problem
    np.testing.assert_allclose(omega, np.sqrt(x**2 - b**2), rtol=1e-9)
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    v = modes.vectors[:, 0]
    resid = H @ v - omega * (g @ v)
    assert np.max(np.abs(resid)) < 1e-12
    vu = v / np.sqrt(omega)
    assert abs(omega * np.real(vu.conj() @ g @ vu) - 1.0) < 1e-12
    assert modes.residuals["max_metric_norm_error"] < 1e-12
    assert modes.residuals["null_space_residual"] < 1e-12


def test_metric_degenerate_jordan_null_keeps_positive_mode_exact():
    # a g-neutral (Jordan) Goldstone null vector makes N = psi^dag g psi
    # singular, yet a decoupled positive mode survives: the pencil must be
    # solved on the nondegenerate quotient with the constraint enforced by
    # pseudoinverse and per-mode solvability validation -- never a
    # singular-matrix crash.
    beta = 0.5
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
    H[0, 2] = H[2, 0] = -1.0
    H[1, 3] = H[3, 1] = beta
    psi = np.array([1, 0, 1, 0]) / np.sqrt(2)
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    assert abs(psi.conj() @ (g @ psi)) < 1e-14  # g-neutral null vector
    assert H @ psi == pytest.approx(0.0, abs=1e-14)

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    omega = modes.energies[modes.positive_indices[0]]
    assert omega == pytest.approx(np.sqrt(1.0 - beta**2), rel=1e-10)
    v = modes.vectors[:, 0]
    resid = H @ v - omega * (g @ v)
    assert np.max(np.abs(resid)) < 1e-12
    vu = v / np.sqrt(omega)
    assert abs(omega * np.real(vu.conj() @ (g @ vu)) - 1.0) < 1e-12
    assert modes.residuals["max_metric_norm_error"] < 1e-12
    assert modes.residuals["null_space_residual"] < 1e-12


def test_paraunitary_mixed_jordan_positive_mode_preserved():
    # regression (Main-reported): a g-unitary paraunitary mode mix on a PSD
    # singular Hessian lifts the exact Jordan zeros into a complex dust pair
    # (+-1.09e-8 +- 3.2e-8j above zero_tol) while a genuine positive mode
    # survives. The solver must classify zero_mode and preserve the positive
    # mode (metric-norm dust recognition), never raise or refuse.
    h = np.array(
        [[1, 0, 1, 0], [0, 2, 0, 0.3], [1, 0, 1, 0], [0, 0.3, 0, 2]], dtype=float
    )
    t = np.array([[0.2, 0.25], [0.25, -0.1]])
    from scipy.linalg import expm

    mix = expm(np.block([[np.zeros((2, 2)), t], [t, np.zeros((2, 2))]]))
    H = mix.T @ h @ mix
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    assert abs(float(np.max(np.abs(mix.T @ g @ mix - g)))) < 1e-14

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.residuals["n_jordan_dust_modes"] == 2
    omega = modes.energies[modes.positive_indices[0]]
    assert omega == pytest.approx(np.sqrt(3.91), rel=1e-10)
    v = modes.vectors[:, 0]
    resid = H @ v - omega * (g @ v)
    assert np.max(np.abs(resid)) < 1e-12
    vu = v / np.sqrt(omega)
    assert abs(omega * np.real(vu.conj() @ (g @ vu)) - 1.0) < 1e-12
    assert modes.residuals["max_metric_norm_error"] < 1e-12
    assert modes.residuals["null_space_residual"] < 1e-12


def test_squeezed_small_metric_mode_is_genuine_not_jordan_dust():
    # regression (Main-reported): a strongly squeezed mode has a tiny metric
    # norm (|v^dag g v| ~ 4.7e-7 for the Euclidean-normalized pencil vector),
    # far below any absolute metric-norm cut, yet its spectral weight
    # omega*|v^dag g v| sits well above the rounding floor of ||H||. The
    # solver must keep it as a genuine positive mode -- never discard it as
    # Jordan dust.
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = H[2, 2] = 1.0
    H[1, 1] = H[3, 3] = 1e5
    H[1, 3] = H[3, 1] = 1e5 - 1.1e-8
    g = np.diag([1.0, 1.0, -1.0, -1.0])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "positive_definite"
    assert modes.energies.size == 2
    # +-sqrt(2.2e-3) exactly by the characteristic polynomial; eigensolvers
    # are limited to ~1e-3 relative accuracy by cond(H) ~ 2e13, so the
    # regression asserts preservation and order of magnitude, not last digits
    assert modes.energies[0] == pytest.approx(np.sqrt(2.2e-3), rel=2e-3)
    assert modes.energies[1] == pytest.approx(1.0, rel=1e-12)
    for k, omega in enumerate(modes.energies):
        v = modes.vectors[:, k]
        resid = H @ v - omega * (g @ v)
        assert np.max(np.abs(resid)) < 1e-6  # absolute; backward error ~1e-16
        assert abs(np.real(v.conj() @ (g @ v)) - 1.0) < 1e-9
    assert modes.residuals["max_metric_norm_error"] < 1e-8


def test_exact_jordan_zero_with_squeezed_positive_mode_recovered_not_zero():
    # regression (Main-reported, exact fixture): H[0,0]=H[2,2]=H[0,2]=1 is an
    # exact Goldstone Jordan zero, and the 1e5 block split by 1.1e-8 carries a
    # squeezed positive mode. The Hessian is PSD, so dynamical instability is
    # impossible: the solver must recover the squeezed mode and classify the
    # point zero_mode, never dynamically_unstable and never labeling the
    # squeezed positive mode as zero.
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = H[2, 2] = 1.0
    H[0, 2] = H[2, 0] = 1.0
    H[1, 1] = H[3, 3] = 1e5
    H[1, 3] = H[3, 1] = 1e5 - 1.1e-8
    g = np.diag([1.0, 1.0, -1.0, -1.0])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.energies[0] == 0.0
    assert modes.residuals["n_jordan_dust_modes"] == 2
    # exact pencil root of the STORED float64 matrix: the float64 grid
    # quantizes delta = 1e5 - fl(1e5 - 1.1e-8), so the symbolic 1.1e-8 root
    # sqrt(2.2e-3) is not attainable; the certified parity-reduced spectrum
    # is exact for the stored matrix
    delta = 1e5 - (1e5 - 1.1e-8)
    assert delta > 0.0
    exact = float(np.sqrt(delta * (2e5 - delta)))
    omega = modes.energies[modes.positive_indices[0]]
    assert omega == pytest.approx(exact, rel=1e-10)
    v = modes.vectors[:, 0]
    assert abs(np.real(v.conj() @ (g @ v)) - 1.0) < 1e-9
    # absolute residual is scaled by cond(H) ~ 2e13; the backward error is
    # the meaningful validation and is reported by the solver
    assert np.max(np.abs(H @ v - omega * (g @ v))) < 1e-6
    assert modes.residuals["max_generalized_residual"] < 1e-8
    assert modes.residuals["null_space_residual"] < 1e-12


def test_positive_definite_nonreciprocal_same_k_asymmetry_is_not_instability():
    # H(k) = diag(2, 3) with g = diag(1, -1): the metric spectrum is {2, -3},
    # so the same-k +- pairing residual is 1.0, yet H is positive definite and
    # the only physical mode is omega = 2. Bosonic particle-hole pairing
    # relates H(k) to H(-k); same-k asymmetry is diagnostic only. Classifying
    # this as dynamically unstable is the reported false refusal.
    H = np.diag([2.0, 3.0]).astype(complex)
    g = np.diag([1.0, -1.0])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "positive_definite"
    np.testing.assert_allclose(modes.energies, [2.0], atol=1e-12)
    assert modes.vectors.shape == (2, 1)
    v = modes.vectors[:, 0]
    assert abs(np.real(v.conj() @ (g @ v)) - 1.0) < 1e-12
    assert np.max(np.abs(H @ v - 2.0 * (g @ v))) < 1e-12
    # retained diagnostic; it is deliberately NOT used as a stability proof
    assert modes.residuals["pairing_residual"] == pytest.approx(1.0, abs=1e-12)


def test_positive_definite_nonreciprocal_block_modes_are_normalized():
    # legacy doubled form with different particle/hole blocks (nonreciprocal):
    # H = diag(1, 2, 3, 4) with g = diag(1, 1, -1, -1). The metric spectrum is
    # {1, 2, -3, -4}, so same-k +- pairing fails, but H is positive definite
    # and the physical modes are the positive generalized eigenvalues 1 and 2.
    H = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    g = np.diag([1.0, 1.0, -1.0, -1.0])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "positive_definite"
    np.testing.assert_allclose(modes.energies, [1.0, 2.0], atol=1e-12)
    assert modes.vectors.shape == (4, 2)
    np.testing.assert_allclose(
        modes.vectors.conj().T @ (g @ modes.vectors), np.eye(2), atol=1e-12
    )
    for k, omega in enumerate(modes.energies[modes.positive_indices]):
        v = modes.vectors[:, k]
        assert np.max(np.abs(H @ v - omega * (g @ v))) < 1e-12
    assert modes.residuals["pairing_residual"] == pytest.approx(2.0, abs=1e-12)


def test_nonreciprocal_coupled_positive_definite_modes_normalized():
    # genuine off-diagonal nonreciprocity: different particle/hole diagonal
    # blocks plus a Hermitian pairing block. The metric spectrum
    # {-3.98, -2.98, 0.98, 1.98} is not +- paired, yet H is positive definite.
    A1 = np.diag([1.0, 2.0])
    A2 = np.diag([3.0, 4.0])
    B = 0.3 * np.array([[0.0, 1.0], [1.0, 0.0]])
    H = np.block([[A1, B], [B.T, A2]]).astype(complex)
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    assert np.linalg.eigvalsh(H)[0] > 0.0

    modes = solve_bosonic_modes(H)

    assert modes.classification == "positive_definite"
    assert modes.energies.size == 2 and np.all(modes.energies > 0.0)
    np.testing.assert_allclose(
        modes.vectors.conj().T @ (g @ modes.vectors), np.eye(2), atol=1e-12
    )
    for k, omega in enumerate(modes.energies[modes.positive_indices]):
        v = modes.vectors[:, k]
        assert np.max(np.abs(H @ v - omega * (g @ v))) < 1e-12
    assert modes.residuals["pairing_residual"] == pytest.approx(2.0, abs=1e-12)


def test_semidefinite_nonreciprocal_hessian_keeps_normalized_positive_modes():
    # H = diag(1, 2, 0, 3) is positive semidefinite and singular, with
    # different particle/hole blocks (nonreciprocal) and metric spectrum
    # {1, 2, 0, -3}. The positive modes have no same-k negative partners, but
    # H PSD guarantees real frequencies: the solver must classify zero_mode,
    # attach the Hessian null space and keep both normalized positive modes.
    H = np.diag([1.0, 2.0, 0.0, 3.0]).astype(complex)
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    # positive (1, 2), negative (-3) and Goldstone (0) finite modes coexist
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvals(g @ H).real), [-3.0, 0.0, 1.0, 2.0], atol=1e-12
    )

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    np.testing.assert_allclose(modes.energies, [1.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(H @ modes.null_space, 0.0, atol=1e-12)
    # the same-k pairing diagnostic is nonzero (no -1/-2 partners) but it is
    # not a validity gate
    assert modes.residuals["pairing_residual"] > 0.0
    assert modes.vectors.shape == (4, 2)
    np.testing.assert_allclose(
        modes.vectors.conj().T @ (g @ modes.vectors), np.eye(2), atol=1e-12
    )
    for k, omega in enumerate(modes.energies[modes.positive_indices]):
        v = modes.vectors[:, k]
        assert np.max(np.abs(H @ v - omega * (g @ v))) < 1e-12


def test_nonreciprocal_singular_squeezed_mode_uses_general_pencil():
    # A2 = A1 + 1e-9 breaks the exact standard form while leaving a
    # near-Goldstone zero mode (below zero_tol) and a strongly squeezed
    # positive mode. The parity reduction must not be applied; the general
    # pencil route must recover the squeezed mode.
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = H[2, 2] = 1.0
    H[0, 2] = H[2, 0] = 1.0
    H[1, 1] = H[3, 3] = 1e5
    H[1, 3] = H[3, 1] = 1e5 - 1.1e-8
    H[2, 2] += 1e-9
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    assert not np.array_equal(H[2:, 2:], H[:2, :2])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.null_space is not None
    assert modes.residuals["null_space_residual"] < 1e-8
    assert modes.residuals["n_jordan_dust_modes"] == 2
    omega = modes.energies[modes.positive_indices[0]]
    delta = 1e5 - (1e5 - 1.1e-8)
    exact = float(np.sqrt(delta * (2e5 - delta)))
    assert omega == pytest.approx(exact, rel=5e-3)
    v = modes.vectors[:, 0]
    assert abs(np.real(v.conj() @ (g @ v)) - 1.0) < 1e-8
    # squeezed-mode vectors have large Euclidean norm; the solver's relative
    # generalized residual is the meaningful validation
    assert modes.residuals["max_generalized_residual"] < 1e-8


def test_nonreciprocal_negative_direction_still_energetically_unstable():
    # The validity gate is the Hessian signature, not same-k pairing: a
    # Hermitian Hessian with a resolved negative direction stays refused even
    # when its metric frequencies are real.
    modes = solve_bosonic_modes(np.diag([1.0, -2.0]).astype(complex))

    assert modes.classification == "energetically_unstable"
    assert modes.vectors is None


def test_block_mismatch_solved_as_actual_positive_definite_matrix():
    # guard: A2 = A1 + 1e-6 changes the Hamiltonian, so the certified parity
    # reduction (which assumes the exact standard form) must not be applied.
    # The Hessian remains positive definite, so the correct consumer-visible
    # result is a valid solve of the ACTUAL matrix -- never the idealized
    # parity zero mode and never a false same-k-pairing dynamic instability.
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = H[2, 2] = 1.0
    H[0, 2] = H[2, 0] = 1.0
    H[1, 1] = H[3, 3] = 1e5
    H[1, 3] = H[3, 1] = 1e5 - 1.1e-8
    H[2, 2] += 1e-6
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    assert not np.array_equal(H[2:, 2:], H[:2, :2])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "positive_definite"
    assert modes.vectors is not None
    assert modes.h_min_eigenvalue > 0.0
    # the lifted Goldstone is a real positive mode, not a fabricated zero
    assert modes.energies[0] > 0.0
    np.testing.assert_allclose(
        modes.vectors.conj().T @ (g @ modes.vectors), np.eye(2), atol=1e-8
    )
    for k, omega in enumerate(modes.energies[modes.positive_indices]):
        v = modes.vectors[:, k]
        assert np.max(np.abs(H @ v - omega * (g @ v))) < 1e-6


def test_degenerate_positive_pair_g_orthonormal_and_exact():
    # doubled form of A = diag(1, 2, 2), B = diag(1, 0.3, 0.3): the pencil has
    # two exactly degenerate positive modes omega = sqrt(3.91) and two zeros.
    # The emitted basis must be g-orthonormal (V^dag g V = I) even inside the
    # degenerate subspace, and both modes must satisfy the generalized
    # eigenproblem exactly.
    A = np.diag([1.0, 2.0, 2.0])
    B = np.diag([1.0, 0.3, 0.3])
    H = np.block([[A, B], [B, A]])
    g = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    modes = solve_bosonic_modes(H)

    assert modes.classification == "zero_mode"
    assert modes.energies[0] == 0.0
    np.testing.assert_allclose(modes.energies[1:], np.sqrt(3.91), rtol=1e-10)
    assert modes.vectors.shape == (6, 2)
    gram = modes.vectors.conj().T @ (g @ modes.vectors)
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)
    for k, omega in enumerate(modes.energies[modes.positive_indices]):
        resid = H @ modes.vectors[:, k] - omega * (g @ modes.vectors[:, k])
        assert np.max(np.abs(resid)) < 1e-10
    assert modes.residuals["max_generalized_residual"] < 1e-10
    assert modes.residuals["n_zero_modes"] == 1


def test_zero_mode_group_touching_zero_refused():
    # drop the SIA: Goldstone touches band 0 on the mesh
    exc = honeycomb_spinio(delta=0.0)
    exc.sia_tensor = {0: np.zeros((3, 3)), 1: np.zeros((3, 3))}
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert any("touches stable zero modes" in f for f in res.diagnostics["failures"])


def test_separated_band_group_proceeds_with_zero_modes():
    # band 1 stays separated from the Goldstone: validated invariant proceeds
    exc = honeycomb_spinio(delta=0.0)
    exc.sia_tensor = {0: np.zeros((3, 3)), 1: np.zeros((3, 3))}
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(12, 12), bands=[1]))
    assert res.valid, res.diagnostics["failures"]
    assert res.chern == -1
    assert res.diagnostics["solver"]["n_mesh_points_with_zero_modes"] > 0


def test_canted_torque_detected_by_assembly():
    # canted moments: nonzero transverse torque; the Hessian matrix at Gamma
    # is positive definite, so only the assembly diagnostic can refuse.
    exc = honeycomb_spinio(delta=0.0)
    exc.spinat = np.array([[1.0, 1.0, np.sqrt(2.0)], [1.0, 1.0, np.sqrt(2.0)]])
    model = prepare_topology_model(exc, source_path="bench")
    magnon = model.magnon()
    H_gamma = magnon.Hq(np.array([[0.0, 0.0, 0.0]]))[0]
    # the matrix alone looks solvable at Gamma
    np.linalg.cholesky(H_gamma)
    from TB2J.magnon.topology.assembly import classical_transverse_gradient

    stat = classical_transverse_gradient(model.JR, model.Rlist, model.magmoms)
    assert not stat.stationary
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert any("nonstationary" in f for f in res.diagnostics["failures"])


def test_unequal_spin_lengths_preserve_noncollinear_stationarity():
    from TB2J.magnon.topology.assembly import classical_transverse_gradient
    from TB2J.magnon.topology.strip import _local_fields

    directions = np.array(
        [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3.0) / 2, 0.0], [-0.5, -np.sqrt(3.0) / 2, 0.0]]
    )
    moments = directions * np.array([2.0, 4.0, 6.0])[:, None]
    exchange = np.zeros((1, 3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                exchange[0, i, j] = -np.eye(3)
    rlist = np.zeros((1, 3), dtype=int)

    stat = classical_transverse_gradient(exchange, rlist, moments)
    assert stat.stationary
    np.testing.assert_allclose(stat.max_transverse, 0.0, atol=1e-14)
    fields = _local_fields(exchange, rlist, moments)
    np.testing.assert_allclose(
        fields - np.einsum("ij,ij->i", fields, directions)[:, None] * directions,
        0.0,
        atol=1e-14,
    )


def test_negative_hessian_real_frequency_refused_with_character(tmp_path):
    # easy-plane overcritical SIA: torque-free saddle with real dynamical
    # frequencies but a negative Hessian direction.
    basis = 10.0 * np.eye(3)
    atoms = Atoms(
        "Cr2", positions=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], cell=basis, pbc=True
    )
    exc = {}
    for R in [(1, 0, 0)]:
        rev = tuple(-np.array(R))
        exc[(R, 0, 1)] = 0.01
        exc[(rev, 1, 0)] = 0.01
    sia = {0: np.diag([0.02, 0.02, 0.001]), 1: np.diag([0.02, 0.02, 0.001])}
    io = SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        charges=[0.0, 0.0],
        index_spin=[0, 1],
        colinear=False,
        exchange_Jdict=exc,
        sia_tensor=sia,
    )
    io._build_Rlist()
    model = prepare_topology_model(io, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    assert res.chern is None
    assert res.diagnostics["solver"]["unstable_frequency_character"] == ["real"]

    # the legacy path still silently shifts and returns energies (baseline)
    magnon = model.magnon()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = magnon._magnon_energies(np.array([[0.0, 0.0, 0.0]]))
    assert len(caught) > 0
    assert np.all(np.isfinite(np.real(legacy)))


def test_complex_frequency_instability_reports_character():
    exc = honeycomb_spinio(delta=0.0)
    exc.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    model = prepare_topology_model(exc, source_path="bench")
    res = calculate_bulk(model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert not res.valid
    chars = res.diagnostics["solver"]["unstable_frequency_character"]
    assert "complex" in chars


def test_near_zero_resolved_mode_stays_positive(anomalous_model):
    # a small but resolved positive energy stays positive_definite; the
    # explicit margins are carried in the diagnostics rather than hidden.
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(6, 6), bands=[0]))
    assert res.valid
    assert res.diagnostics["solver"]["min_physical_energy"] > 0
    assert res.diagnostics["solver"]["classifications"] == {"positive_definite": 36}


def test_json_carries_validity_classes(tmp_path, anomalous_model):
    res = calculate_bulk(anomalous_model, BulkRequest(mesh=(6, 6), bands=[0]))
    data = json.loads(json.dumps(res.to_json_dict()))
    solver = data["diagnostics"]["solver"]
    assert solver["classifications"] == {"positive_definite": 36}
    assert data["conventions"]["certificate"] == (
        "docs/sympy/magnon_topology_lswt_ferromagnet.py"
    )
