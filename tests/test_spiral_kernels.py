"""Story 004: two-channel spiral MFT curvature kernels and gates.

Checks the :mod:`TB2J.spiral_kernels` path against independent
references on toy rings (fixtures in :mod:`spiral_fixtures`):

* planar-gauge folded spectrum == explicit lab-frame supercell spectrum
  (half-shifted mesh absorbing the ring spinor flux);
* twist-aware unfolding of the folded resolvents == dense resolvent
  blocks at complex Matsubara energies;
* contour/Matsubara pair kernels == degeneracy-safe eigenbasis
  reference (multi-seed, both channels, overlap, doubly degenerate
  spectra, sublattice taus/phis);
* both == the lab-frame FD curvature of the explicit supercell (the
  probe identity);
* gates: Goldstone ``C^bb . 1 = 0`` at every tested q, torque zero
  modes at the torque-free q (flagged at the grid-minimizing
  non-torque-free q*), and the q=0 LKAG anchor ``C^dd = C^bb = M``,
  ``C^db = 0``.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from spiral_fixtures import (
    degenerate_ring_classes,
    make_ring_state,
    ring_hopping_classes,
)

from TB2J.spiral_kernels import (
    SpiralGateError,
    contour_kernels_dense,
    eigenbasis_kernels_dense,
    goldstone_gate,
    inplane_response_fd,
    lab_supercell,
    planar_flux_shift,
    planar_spiral_green,
    q0_anchor_check,
    torque_gate,
)

Q = 1.0 / 6.0
WIDTH = 5e-3  # sharp frozen filling: fp = f'(eps) terms at e^(-gap/w)/w << 1e-9


def _sharp_filling(state, ncell, width=WIDTH):
    """Centre ``efermi`` in the largest gap near the median, narrow width.

    The force-theorem protocol of the normative derivation script: the
    frozen occupations are sharp (no partially filled levels), so the
    masked trace, the resolvent contour, and the exact FD coincide.
    Uses the explicit supercell spectrum (valid at any q).
    """
    H, S = lab_supercell(state, ncell)
    if S is None:
        ev = np.linalg.eigvalsh(H)
    else:
        L = np.linalg.cholesky(S)
        Li = np.linalg.inv(L)
        A = Li @ H @ Li.conj().T
        ev = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
    ev = np.sort(ev)
    gaps = ev[1:] - ev[:-1]
    cand = [i for i, g in enumerate(gaps) if g > 0.2]
    i0 = max(cand or range(len(gaps)), key=lambda i: gaps[i])
    mu = float(0.5 * (ev[i0] + ev[i0 + 1]))
    meta = json.loads(state.metadata_json)
    meta["width"] = width
    return dataclasses.replace(state, efermi=mu, metadata_json=json.dumps(meta))


def _ring(N, seed, q=Q, overlap=False, degenerate=False):
    if degenerate:
        classes, eps = degenerate_ring_classes()
    else:
        classes, eps = ring_hopping_classes(N, seed)
    state = make_ring_state(
        N, classes, eps, q=q, b_local=(1.5,), overlap=overlap, width=0.2
    )
    return _sharp_filling(state, N)


def _lab_fd_curvature(state, ncell, h=1e-4):
    """4-point FD curvature of the frozen band sum (probe identity)."""
    H, S = lab_supercell(state, ncell)
    norb = state.norb
    size = 2 * norb * ncell
    Bf = 0.5 * np.asarray(state.B_local, dtype=float)
    q = float(np.asarray(state.q_frac, dtype=float)[0])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)

    def energy(c):
        Hp = H.copy()
        for a in range(ncell):
            for mu_i in range(norb):
                th = 2 * np.pi * q * a
                s0 = 2 * (a * norb + mu_i)
                Hp[s0 : s0 + 2, s0 : s0 + 2] += (
                    c[a * norb + mu_i] * Bf[mu_i] * sy
                    + c[ncell * norb + a * norb + mu_i]
                    * Bf[mu_i]
                    * (-np.sin(th) * sz + np.cos(th) * sx)
                    - 0.5
                    * (c[a * norb + mu_i] ** 2 + c[ncell * norb + a * norb + mu_i] ** 2)
                    * Bf[mu_i]
                    * (np.cos(th) * sz + np.sin(th) * sx)
                )
        if S is None:
            ev = np.linalg.eigvalsh(Hp)
        else:
            L = np.linalg.cholesky(S)
            Li = np.linalg.inv(L)
            A = Li @ Hp @ Li.conj().T
            ev = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
        f = 1.0 / (1.0 + np.exp((ev - state.efermi) / state.width))
        return float(np.sum(f * ev))

    Cfd = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):

            def mixed(hh):
                ei = np.zeros(size)
                ej = np.zeros(size)
                ei[i] = hh
                ej[j] = hh
                return (
                    energy(ei + ej)
                    - energy(ei - ej)
                    - energy(-ei + ej)
                    + energy(-ei - ej)
                ) / (4 * hh * hh)

            # Richardson: kill the O(h^2) truncation of the 4-point rule
            val = (4.0 * mixed(h / 2.0) - mixed(h)) / 3.0
            Cfd[i, j] = Cfd[j, i] = val
    return 0.5 * (Cfd + Cfd.T)


# ---------------------------------------------------------------------------
# folded spectrum and unfolding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ncell,seed,overlap,degenerate",
    [
        (6, 71, False, False),
        (6, 72, True, False),
        (12, 73, False, False),
        (6, 0, False, True),
    ],
)
def test_planar_folded_spectrum_matches_lab_supercell(ncell, seed, overlap, degenerate):
    state = _ring(ncell, seed, overlap=overlap, degenerate=degenerate)
    green = planar_spiral_green(state, ncell)
    ev_fold = np.sort(green.evals.ravel())
    H, S = lab_supercell(state, ncell)
    if S is None:
        ev_lab = np.linalg.eigvalsh(H)
    else:
        L = np.linalg.cholesky(S)
        Li = np.linalg.inv(L)
        A = Li @ H @ Li.conj().T
        ev_lab = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
    assert np.max(np.abs(ev_fold - np.sort(ev_lab))) < 1e-10


def test_planar_flux_shift_rules():
    state = _ring(6, 71)
    assert planar_flux_shift(state, 6) == 0.5
    state0 = _ring(6, 71, q=0.0)
    assert planar_flux_shift(state0, 6) == 0.0
    state3 = _ring(6, 71, q=1.0 / 3.0)
    assert planar_flux_shift(state3, 6) == 0.0
    state_inc = _ring(6, 71, q=0.19)
    with pytest.raises(ValueError, match="commensurate"):
        planar_flux_shift(state_inc, 6)


# ---------------------------------------------------------------------------
# contour vs eigenbasis reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ncell,seed,overlap,degenerate,taus",
    [
        (6, 71, False, False, False),
        (6, 72, True, False, False),
        (12, 73, False, False, False),
        (6, 0, False, True, False),
        (6, 31, False, False, True),
    ],
)
def test_contour_matches_eigenbasis_reference(ncell, seed, overlap, degenerate, taus):
    if taus:
        classes, eps = degenerate_ring_classes()
        taus_arr = np.array([[0.0, 0, 0], [0.25, 0, 0]])
        phis_arr = np.array([0.0, np.pi / 3])
        state = make_ring_state(
            ncell,
            classes,
            eps,
            q=Q,
            b_local=(1.5, 1.5),
            taus=taus_arr,
            phis=phis_arr,
            width=0.2,
        )
        state = _sharp_filling(state, ncell)
    else:
        state = _ring(ncell, seed, overlap=overlap, degenerate=degenerate)
    ref = eigenbasis_kernels_dense(state, ncell)
    con = contour_kernels_dense(state, ncell)
    for pair in (("d", "d"), ("b", "b"), ("d", "b"), ("b", "d")):
        diff = np.max(np.abs(ref[pair] - con[pair]))
        scale = max(np.max(np.abs(ref[pair])), 1e-12)
        assert diff <= 1e-5 * max(1.0, scale), (pair, diff, scale)


# ---------------------------------------------------------------------------
# probe identity: kernel == lab-frame FD of the explicit supercell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ncell,seed,overlap,degenerate",
    [
        (6, 71, False, False),
        (6, 72, True, False),
        (6, 0, False, True),
    ],
)
def test_kernel_matches_lab_frame_fd(ncell, seed, overlap, degenerate):
    state = _ring(ncell, seed, overlap=overlap, degenerate=degenerate)
    ref = eigenbasis_kernels_dense(state, ncell)
    Cfd = _lab_fd_curvature(state, ncell)
    half = ncell * state.norb
    scale = max(np.max(np.abs(Cfd)), 1e-12)
    assert np.max(np.abs(ref[("d", "d")] - Cfd[:half, :half])) <= 1e-5 * max(1.0, scale)
    assert np.max(np.abs(ref[("b", "b")] - Cfd[half:, half:])) <= 1e-5 * max(1.0, scale)
    assert np.max(np.abs(ref[("d", "b")] - Cfd[:half, half:])) <= 1e-5 * max(1.0, scale)
    # contour path agrees with the same FD class
    con = contour_kernels_dense(state, ncell)
    assert np.max(np.abs(con[("d", "d")] - Cfd[:half, :half])) <= 1e-5 * max(1.0, scale)
    assert np.max(np.abs(con[("b", "b")] - Cfd[half:, half:])) <= 1e-5 * max(1.0, scale)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [Q, 1.0 / 3.0, 0.0])
def test_goldstone_gate_commensurate(q):
    ncell = 6
    state = _ring(ncell, 71, q=q)
    curv = contour_kernels_dense(state, ncell)
    report = goldstone_gate(curv, tol=1e-7)
    assert report["passed"]


def test_goldstone_gate_dense_incommensurate():
    ncell = 6
    state = _ring(ncell, 72, q=0.19)
    curv = eigenbasis_kernels_dense(state, ncell)
    report = goldstone_gate(curv, tol=1e-7)
    assert report["passed"]
    curv_c = contour_kernels_dense(state, ncell)
    report_c = goldstone_gate(curv_c, tol=1e-7)
    assert report_c["passed"]


def test_torque_gate_passes_at_torque_free_q():
    ncell = 6
    state = _ring(ncell, 71, q=Q)
    curv = contour_kernels_dense(state, ncell)
    report = torque_gate(curv, state, ncell, tol=1e-6)
    assert report["passed"] and not report["flagged"]


def _q_star_state(ncell=6, seed=71):
    """Grid-minimizing (non-torque-free) pitch of the derivation script."""
    classes, eps = ring_hopping_classes(ncell, seed)
    base = make_ring_state(ncell, classes, eps, q=Q, b_local=(1.5,), width=0.2)

    def E_of_q(qv):
        st = dataclasses.replace(base, q_frac=np.array([qv, 0.0, 0.0]))
        H, _ = lab_supercell(st, ncell)
        ev = np.linalg.eigvalsh(H)
        mu = float(np.quantile(ev, 0.5))
        f = 1.0 / (1.0 + np.exp((ev - mu) / 0.05))
        return float(np.sum(ev * f))

    grid = np.linspace(0.0, 1.0, 241, endpoint=False)
    q_star = float(grid[int(np.argmin([E_of_q(qv) for qv in grid]))])
    state = make_ring_state(ncell, classes, eps, q=q_star, b_local=(1.5,), width=0.2)
    return _sharp_filling(state, ncell)


def test_torque_gate_flags_non_torque_free_q_star():
    ncell = 6
    state = _q_star_state(ncell)
    # the bundle records the (non-zero) local torque of this reference
    state = dataclasses.replace(state, torque_norms=np.full(state.norb, 0.1))
    curv = eigenbasis_kernels_dense(state, ncell)
    report = torque_gate(curv, state, ncell, tol=1e-6)
    assert not report["passed"]
    assert report["flagged"]
    scale = report["scale"]
    assert 1e-3 * scale <= report["max_residual"] <= 1.0 * scale


def test_torque_gate_raises_for_torque_free_bundle_on_violation():
    ncell = 6
    state = _q_star_state(ncell)
    curv = eigenbasis_kernels_dense(state, ncell)
    with pytest.raises(SpiralGateError) as exc:
        torque_gate(curv, state, ncell, tol=1e-6)
    assert "torque-balanced" in str(exc.value)
    assert exc.value.report["gate"] == "torque"


# ---------------------------------------------------------------------------
# q=0 LKAG anchor
# ---------------------------------------------------------------------------


def test_q0_anchor_and_collinear_M():
    ncell = 6
    state = _ring(ncell, 72, q=0.0)
    curv = contour_kernels_dense(state, ncell)
    M = inplane_response_fd(state, ncell)
    report = q0_anchor_check(curv, M, tol=1e-5)
    assert report["passed"]
    assert report["max_db"] <= 1e-9
    assert report["max_dd_minus_bb"] <= 1e-9
