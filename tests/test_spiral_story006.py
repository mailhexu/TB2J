"""Story 006: consolidated toy-model suite - gaps, gates, Toth-Lake.

Adds the missing coverage on top of the existing spiral suite (stories
003-005) without duplicating it.  Existing coverage intentionally NOT
re-asserted here: ``G_q`` blocks vs explicit supercell resolvent at
1e-10 (``test_spiral_green``), doubly degenerate / nonorthogonal
pencils, the mapping ``diag_consistency`` gate at 1e-6 and the
contour-vs-eigenbasis probe class (``test_exchange_spiral``).

New in this module:

* gate behaviour with documented diagnostics: the torque gate hard-fails
  with a complete diagnostic report on the non-torque-free ``q*``
  reference, ``gate_override`` runs to completion and *records* the
  override, and violations are flagged in the report (never silent);
* the dimer fixture (``N = 2`` ring) end-to-end with overlap and
  rotating-frame ``V_U``: contract rebuild identity, folded-vs-
  supercell spectrum, unfolded blocks vs the explicit resolvent;
* the V_U rotation-covariance identity at 1e-10 through the tbupy
  helper (``tbupy.spiral_state.check_V_U_rotation_covariance``);
* the Toth-Lake local-frame flat-screw construction from known-J
  chains (normative: ``docs/sympy/spiral_state_mft.py``,
  ``check_toth_lake_flat_screw``): textbook J1 chains, the J1-J2
  spiral pitch, and the story-005 fixture with ``J`` recovered from
  the MFT curvature - the analytic preparation for story 007.
"""

from __future__ import annotations

import numpy as np
import pytest
from spiral_fixtures import (
    dimer_classes,
    j1j2_chain,
    make_ring_state,
    pencil_eigenvalues,
    q_star_ring,
    random_hermitian,
    rebuild_hq_sq,
    ring_hopping_classes,
    ring_kmesh,
    sharp_filling,
    supercell_spiral,
    toth_lake_dynamical,
    twist_unitary,
)

from TB2J.exchange_spiral import ExchangeSpiral
from TB2J.spiral_green import SpiralGreen
from TB2J.spiral_kernels import SpiralGateError, eigenbasis_kernels_dense, torque_gate

Q = 1.0 / 6.0


# ---------------------------------------------------------------------------
# gate behaviour: hard fail, override, flagged-not-silent diagnostics
# ---------------------------------------------------------------------------
def test_torque_gate_hard_fail_diagnostics():
    """Non-torque-free reference: hard fail with a complete diagnostic."""
    state = q_star_ring()  # torque_norms within tolerance: violations raise
    curv = eigenbasis_kernels_dense(state, 6)
    with pytest.raises(SpiralGateError) as exc:
        torque_gate(curv, state, 6, tol=1e-6)
    rep = exc.value.report
    assert rep["gate"] == "torque"
    assert rep["passed"] is False
    assert rep["flagged"] is False  # bundle does not record the torque
    assert rep["max_residual"] > rep["tol"]
    # documented diagnostics: residuals, scale, tolerance, recorded norms
    for key in (
        "residuals_cos",
        "residuals_sin",
        "scale",
        "tol",
        "torque_norms",
        "allow_violation",
    ):
        assert key in rep, key
    worst = max(
        float(np.max(np.abs(rep["residuals_cos"]))),
        float(np.max(np.abs(rep["residuals_sin"]))),
    )
    assert rep["max_residual"] == pytest.approx(worst)
    assert "torque-balanced" in str(exc.value)


def test_torque_gate_override_flagged_path():
    """A bundle whose torque_norms record the violation is flagged, not raised."""
    state = q_star_ring(torqued=0.1)
    curv = eigenbasis_kernels_dense(state, 6)
    report = torque_gate(curv, state, 6, tol=1e-6)  # no raise
    assert report["passed"] is False
    assert report["flagged"] is True
    assert report["torque_norms"] is not None
    assert float(np.max(report["torque_norms"])) > report["tol"]
    assert report["max_residual"] > report["tol"]


def test_exchange_gate_override_records_not_silent():
    """gate_override runs to completion and records the violation."""
    torqued = q_star_ring(torqued=0.1)
    calc = ExchangeSpiral.from_spiral_state(torqued, ncell=6, gate_override=True)
    report = calc.run()
    assert report["gate_override"] is True
    gate = report["gates"]["torque"]
    assert gate["passed"] is False
    assert gate["flagged"] is True
    assert gate["max_residual"] > gate["tol"]
    assert calc.exchange_Jdict  # mapping ran to completion despite the gate

    # un-torqued bundle with override: still visible in the report numbers
    plain = q_star_ring()
    calc2 = ExchangeSpiral.from_spiral_state(plain, ncell=6, gate_override=True)
    report2 = calc2.run()
    gate2 = report2["gates"]["torque"]
    assert gate2["passed"] is False
    assert gate2["flagged"] is False
    assert gate2["max_residual"] > gate2["tol"]


# ---------------------------------------------------------------------------
# implementation-level gap: dimer end-to-end with overlap + V_U
# ---------------------------------------------------------------------------
def test_dimer_green_blocks_and_spectrum():
    """Two-site dimer: rebuild identity, spectrum identity, G_q blocks."""
    ncell = 2
    classes, eps = dimer_classes(t=0.55, eps0=0.1)
    v_u = random_hermitian(2, 17)
    state = make_ring_state(
        ncell, classes, eps, q=0.5, b_local=(0.6,), overlap=True, v_u=v_u
    )
    green = SpiralGreen(state, ring_kmesh(ncell))

    # contract rebuild identity through the tbupy assembler
    for ik, k in enumerate(green.kmesh):
        href, sref = rebuild_hq_sq(state, k)
        assert np.max(np.abs(green.Hq[ik] - href)) < 1e-12
        assert np.max(np.abs(green.Sq[ik] - sref)) < 1e-12

    # folded spectrum == independent docs-convention supercell spectrum
    hs, ss = supercell_spiral(
        ncell,
        classes,
        eps,
        q_frac=state.q_frac,
        taus=state.taus,
        phis=state.phis,
        b_local=state.B_local,
        overlap=True,
    )
    w = twist_unitary(ncell, state.norb, 0.5)
    hs = w @ hs @ w.conj().T + np.kron(np.eye(ncell), v_u)
    ss = w @ ss @ w.conj().T
    folded = np.sort(green.evals.ravel())
    sc = np.sort(pencil_eigenvalues(hs, ss))
    assert np.max(np.abs(folded - sc)) < 1e-9

    # unfolded blocks vs the twisted supercell resolvent
    energy = 1.1 - 0.4j
    gs = np.linalg.inv(energy * ss - hs)
    g_untw = w.conj().T @ gs @ w
    at = green.at(energy)
    for r in (0, 1, 2, 3):
        block = at.unfold(0, [(0, r)])[0]
        ref = g_untw[0:2, 2 * (r % ncell) : 2 * (r % ncell) + 2]
        assert np.max(np.abs(block - ref)) < 1e-10


# ---------------------------------------------------------------------------
# implementation-level gap: V_U rotation covariance at 1e-10 (tbupy helper)
# ---------------------------------------------------------------------------
def test_vu_rotation_covariance_tbupy_helper():
    """V_U(rotated rho) == rotated V_U(rho) via the tbupy-side helper."""
    basis_mod = pytest.importorskip("tbupy.basis")
    hub = pytest.importorskip("tbupy.hubbard")
    ss_mod = pytest.importorskip("tbupy.spiral_state")

    from types import SimpleNamespace

    orbs = [SimpleNamespace(iatom=0, l=0), SimpleNamespace(iatom=1, l=0)]
    atoms = SimpleNamespace(get_chemical_symbols=lambda: ["Fe", "Fe"])
    basis = basis_mod.BasisSet(orbs, atoms, nspin=2)
    term = hub.SpinorDudarevTerm(
        basis, {"Fe": {"U": 2.0, "J": 0.3, "L": 0}}, dc_type="FLL-ns"
    )

    rng = np.random.default_rng(11)
    g = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    rho = g @ g.conj().T
    axis = rng.normal(size=(2, 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    site_u = np.zeros((2, 2, 2), dtype=complex)
    for i in range(2):
        n = axis[i]
        sigma_n = n[0] * sx + n[1] * sy + n[2] * sz
        site_u[i] = np.eye(2) * np.cos(0.35) + 1j * sigma_n * np.sin(0.35)

    report = ss_mod.check_V_U_rotation_covariance(term, rho, site_u, atol=1e-10)
    assert report["passed"], report
    assert report["max_residual"] <= 1e-10


# ---------------------------------------------------------------------------
# Toth-Lake flat-screw construction on known-J chains
# ---------------------------------------------------------------------------
def test_toth_lake_j1_chains_textbook():
    """J1 chains: AFM pitch pi reproduces omega = 2J|sin k|; FM q=0 limit."""
    kgrid = np.arange(16) / 16.0

    # AFM J1 (script convention J1 < 0): spiral pitch q = 1/2, max of J~
    j_r = j1j2_chain(j1=-1.0, j2=0.0)
    tl = toth_lake_dynamical(j_r, kgrid, q=0.5, s=1.0)
    assert tl["radicand"].real.min() > -1e-12  # stable
    np.testing.assert_allclose(
        tl["omega"].real, 2.0 * np.abs(np.sin(2 * np.pi * kgrid)), atol=1e-12
    )
    np.testing.assert_allclose(tl["omega"].imag, 0.0, atol=1e-12)
    # Goldstone zeros at k = 0 and k = +-1/2 (indices 0 and 8)
    assert np.max(np.abs(tl["omega"][[0, 8]])) < 1e-12

    # FM control (script convention J > 0) at q = 0: omega = J~(0) - J~(k)
    tl_fm = toth_lake_dynamical(j1j2_chain(j1=1.0, j2=0.0), kgrid, q=0.0, s=1.0)
    np.testing.assert_allclose(tl_fm["omega"].real, tl_fm["C"].real, atol=1e-12)
    assert tl_fm["radicand"].real.min() > -1e-12


def test_toth_lake_j1j2_spiral_pitch():
    """J1-J2 chain at the classical spiral pitch: stability + Goldstone zeros."""
    j_r = j1j2_chain(j1=1.0, j2=-1.0)
    # classical pitch: cos(2 pi q*) = -J1 / (4 J2)
    q_star = float(np.arccos(0.25) / (2.0 * np.pi))
    kgrid = np.arange(2000) / 2000.0
    tl = toth_lake_dynamical(j_r, kgrid, q=q_star, s=1.0)
    assert tl["C"].real.min() > 0.0  # J~(q*) is the global maximum
    assert tl["radicand"].real.min() > -1e-10  # real spectrum on the mesh
    assert np.max(np.abs(tl["omega"].imag)) < 1e-10

    # Goldstone zeros at k = 0 and k = +-q* (exact identities)
    focus = toth_lake_dynamical(
        j_r, np.array([0.0, q_star, 1.0 - q_star]), q_star, s=1.0
    )
    assert np.max(np.abs(focus["omega"])) < 1e-9

    # the bosonic dynamical matrix reproduces +-omega from the h/gamma blocks
    eig = np.linalg.eigvals(tl["dynamical"])
    np.testing.assert_allclose(
        np.sort(np.abs(eig), axis=1),
        np.column_stack([np.abs(tl["omega"]), np.abs(tl["omega"])]),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(eig.sum(axis=1), 0.0, atol=1e-10)


def test_toth_lake_extracted_j_story005_fixture():
    """Story-005 known-J fixture: recovered J feeds the Toth-Lake zeros."""
    classes, eps = ring_hopping_classes(6, 71)
    state = sharp_filling(
        make_ring_state(6, classes, eps, q=Q, b_local=(1.5,), width=0.2), 6
    )
    calc = ExchangeSpiral.from_spiral_state(state)
    report = calc.run()
    assert report["gates"]["diag_consistency"]["max_residual"] <= 1e-6

    j_r = {
        (R[0],): float(val)
        for (R, i, j), val in calc.exchange_Jdict.items()
        if (i, j) == (0, 0)
    }
    kgrid = np.arange(6) / 6.0
    tl = toth_lake_dynamical(j_r, kgrid, Q, s=1.0)
    scale = max(np.max(np.abs(tl["A"])), np.max(np.abs(tl["C"])), 1e-12)

    # Goldstone zeros at k = 0 and k = +-q: exact identities of the
    # extracted J (parity of J(R) and the reference pitch).  The blocks
    # vanish at the fp parity floor; omega = sqrt(A C) amplifies that
    # floor by the square root, so assert the blocks tightly and omega
    # at the amplified tolerance.
    a0 = tl["A"][0]
    c_pm = np.array([tl["C"][1], tl["C"][5]])
    assert abs(a0) <= 1e-12 * scale
    assert np.max(np.abs(c_pm)) <= 1e-12 * scale
    zeros = np.array([tl["omega"][0], tl["omega"][1], tl["omega"][5]])
    assert np.max(np.abs(zeros)) <= 1e-7

    # dynamical-matrix spectrum reproduces +-sqrt(A_k C_k)
    eig = np.linalg.eigvals(tl["dynamical"])
    np.testing.assert_allclose(
        np.sort(np.abs(eig), axis=1),
        np.column_stack([np.abs(tl["omega"]), np.abs(tl["omega"])]),
        rtol=1e-10,
        atol=1e-12,
    )
