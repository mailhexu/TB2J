"""Story 003: folded resolvent, pole sums, and unfolding of spiral G_q.

Checks the :mod:`TB2J.spiral_green` path against independent
references on toy rings (fixtures in :mod:`spiral_fixtures`):

* pole-sum resolvent identities per k (S-weighted trace, spectral
  representation of the resolvent);
* unfolded real-space spin blocks ``G_q[(0, i), (R, j)](E)`` versus the
  explicit supercell Green function (direct inverse and eigenbasis
  spectral sum), including exactly doubly degenerate spectra;
* pairwise block Hermiticity at real E;
* trace/sum rules tying unfolded blocks to folded spectral sums;
* the contract rebuild rule (tbupy assembler) and the folded-vs-
  supercell spectrum identity.
"""

from __future__ import annotations

import numpy as np
import pytest
from spiral_fixtures import (
    degenerate_ring_classes,
    make_ring_state,
    pencil_eigenvalues,
    random_hermitian,
    rebuild_hq_sq,
    ring_hopping_classes,
    ring_kmesh,
    supercell_spiral,
    twist_unitary,
)

from TB2J.spiral_green import SpiralGreen

Q = 1.0 / 6.0


def _random_ring(N, seed, overlap=False, v_u=None):
    classes, eps = ring_hopping_classes(N, seed)
    state = make_ring_state(
        N, classes, eps, q=Q, b_local=(0.4,), overlap=overlap, v_u=v_u
    )
    return state, classes, eps


def _supercell(N, state, classes, eps, overlap=False):
    H, S = supercell_spiral(
        N,
        classes,
        eps,
        q_frac=state.q_frac,
        taus=state.taus,
        phis=state.phis,
        b_local=state.B_local,
        overlap=overlap,
    )
    if S is None:
        S = np.eye(2 * state.norb * N)
    return H, S


def _eigenbasis_green(H, S, E):
    """Explicit eigenbasis supercell Green function (full spectral sum)."""
    L = np.linalg.cholesky(S)
    Li = np.linalg.inv(L)
    A = Li @ H @ Li.conj().T
    w, U = np.linalg.eigh(0.5 * (A + A.conj().T))
    C = Li.conj().T @ U
    return (C * (1.0 / (E - w))) @ C.conj().T


@pytest.mark.parametrize(
    "N,seed,overlap", [(6, 31, False), (6, 32, True), (12, 33, False)]
)
def test_pole_sum_and_unfold_vs_supercell(N, seed, overlap):
    state, classes, eps = _random_ring(N, seed, overlap=overlap)
    green = SpiralGreen(state, ring_kmesh(N))
    E = 1.1 - 0.4j
    at = green.at(E)

    # S-weighted pole sum == sum_n 1/(E - eps_n), per k
    assert np.max(np.abs(at.pole_sum_trace(green.Sq) - at.pole_sum())) < 1e-10
    # resolvent == spectral sum over the standardized eigenbasis
    Gspec = np.einsum(
        "kin,kjn,kn->kij", at.evecs, at.evecs.conj(), 1.0 / (E - at.evals)
    )
    assert np.max(np.abs(at.Gq - Gspec)) < 1e-10

    # unfolded real-space blocks vs explicit supercell Green function
    Hs, Ss = _supercell(N, state, classes, eps, overlap=overlap)
    Gs = np.linalg.inv(E * Ss - Hs)
    Gs_eig = _eigenbasis_green(Hs, Ss, E)
    assert np.max(np.abs(Gs - Gs_eig)) < 1e-9
    for R in (0, 1, 2, 3, N - 2, N - 1, N):
        block = at.unfold(0, [(0, R)])[0]
        ref = Gs[0:2, 2 * (R % N) : 2 * (R % N) + 2]
        assert np.max(np.abs(block - ref)) < 1e-10


def test_degenerate_local_field_ring():
    """Two-copy ring: known analytic folded spectrum, doubly degenerate."""
    N, B, t, eps0 = 6, 0.5, 0.35, 0.2
    classes, eps = degenerate_ring_classes(t=t, eps0=eps0)
    state = make_ring_state(N, classes, eps, q=Q, b_local=(B, B))
    green = SpiralGreen(state, ring_kmesh(N))
    for ik, k in enumerate(green.kmesh):
        up = eps0 + B / 2 + 2 * t * np.cos(2 * np.pi * (k[0] - Q / 2))
        dn = eps0 - B / 2 + 2 * t * np.cos(2 * np.pi * (k[0] + Q / 2))
        ref = np.sort(np.array([up, up, dn, dn]))
        ev = np.sort(green.evals[ik])
        assert np.max(np.abs(ev - ref)) < 1e-10
        # exact pairwise (double) degeneracy
        assert np.max(np.abs(ev - ev[[1, 0, 3, 2]])) < 1e-12

    E = 1.1 - 0.4j
    at = green.at(E)
    assert np.max(np.abs(at.pole_sum_trace(green.Sq) - at.pole_sum())) < 1e-10
    Hs, Ss = _supercell(N, state, classes, eps)
    Gs = np.linalg.inv(E * Ss - Hs)
    Gs_eig = _eigenbasis_green(Hs, Ss, E)
    assert np.max(np.abs(Gs - Gs_eig)) < 1e-9
    for R in (0, 1, 2, 4, 5):
        block = at.unfold(0, [(0, R)])[0]
        ref = Gs[0:2, 2 * state.norb * R : 2 * state.norb * R + 2]
        assert np.max(np.abs(block - ref)) < 1e-10


@pytest.mark.parametrize("kind", ["random_overlap", "degenerate"])
def test_unfold_with_taus_and_phis(kind):
    """Unfold factors stay exact for per-orbital taus/phis (assembler)."""
    N = 6
    taus = np.array([[0.1, 0.0, 0.0]])
    phis = np.array([np.pi / 5])
    if kind == "random_overlap":
        classes, eps = ring_hopping_classes(N, 35)
        state = make_ring_state(
            N, classes, eps, q=Q, b_local=(0.3,), overlap=True, taus=taus, phis=phis
        )
        overlap = True
    else:
        classes, eps = degenerate_ring_classes()
        taus = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        state = make_ring_state(
            N,
            classes,
            eps,
            q=Q,
            b_local=(0.3, 0.3),
            taus=taus,
            phis=np.full(2, np.pi / 5),
        )
        overlap = False
    green = SpiralGreen(state, ring_kmesh(N))
    E = 1.1 - 0.4j
    at = green.at(E)
    Hs, Ss = _supercell(N, state, classes, eps, overlap=overlap)
    Gs = np.linalg.inv(E * Ss - Hs)
    for R in (0, 1, 3, 5):
        block = at.unfold(0, [(0, R)])[0]
        ref = Gs[0:2, 2 * state.norb * R : 2 * state.norb * R + 2]
        assert np.max(np.abs(block - ref)) < 1e-10


@pytest.mark.parametrize("kind", ["random_overlap", "degenerate"])
def test_block_hermiticity(kind):
    N = 6
    if kind == "random_overlap":
        state, classes, eps = _random_ring(N, 32, overlap=True)
    else:
        classes, eps = degenerate_ring_classes()
        state = make_ring_state(N, classes, eps, q=Q, b_local=(0.5, 0.5))
    green = SpiralGreen(state, ring_kmesh(N))
    E = float(np.max(green.evals)) + 2.0
    at = green.at(E)
    for mu in range(state.norb):
        for nu in range(state.norb):
            for R in (-2, -1, 1, 2):
                fwd = at.unfold(mu, [(nu, R)])[0]
                bwd = at.unfold(nu, [(mu, -R)])[0]
                assert np.max(np.abs(fwd - bwd.conj().T)) < 1e-10


@pytest.mark.parametrize("kind", ["random_overlap", "degenerate"])
def test_trace_sum_rules(kind):
    N = 6
    if kind == "random_overlap":
        state, classes, eps = _random_ring(N, 32, overlap=True)
    else:
        classes, eps = degenerate_ring_classes()
        state = make_ring_state(N, classes, eps, q=Q, b_local=(0.5, 0.5))
    green = SpiralGreen(state, ring_kmesh(N))
    E = 0.9 + 0.35j
    at = green.at(E)
    norb = state.norb
    # (a) cell-0 row sum over all cells/orbitals/spin == Tr Gq(k_0)
    row = sum(
        np.trace(at.unfold(mu, [(nu, R)])[0])
        for mu in range(norb)
        for nu in range(norb)
        for R in range(N)
    )
    assert abs(row - at.pole_sum_trace()[0]) < 1e-8
    # (b) full supercell trace == sum over the mesh of Tr Gq(k)
    onsite = sum(np.trace(at.unfold(mu, [(mu, 0)])[0]) for mu in range(norb))
    assert abs(N * onsite - np.sum(at.pole_sum_trace())) < 1e-8


@pytest.mark.parametrize(
    "N,seed,overlap,with_vu",
    [
        (6, 31, False, False),
        (6, 32, True, False),
        (12, 33, False, True),
        (6, 34, True, True),
    ],
)
def test_rebuild_rule_and_supercell_spectrum(N, seed, overlap, with_vu):
    v_u = random_hermitian(2, seed + 100) if with_vu else None
    state, classes, eps = _random_ring(N, seed, overlap=overlap, v_u=v_u)
    green = SpiralGreen(state, ring_kmesh(N))

    # contract rebuild identity against the tbupy assembler
    for ik, k in enumerate(green.kmesh):
        Href, Sref = rebuild_hq_sq(state, k)
        assert np.max(np.abs(green.Hq[ik] - Href)) < 1e-12
        assert np.max(np.abs(green.Sq[ik] - Sref)) < 1e-12

    # folded spectrum == independent docs-convention supercell spectrum
    Hs, Ss = _supercell(N, state, classes, eps, overlap=overlap)
    if with_vu:
        W = twist_unitary(N, state.norb, Q)
        Hs = W @ Hs @ W.conj().T + np.kron(np.eye(N), v_u)
        Ss = W @ Ss @ W.conj().T
    folded = np.sort(green.evals.ravel())
    sc = np.sort(pencil_eigenvalues(Hs, Ss))
    assert np.max(np.abs(folded - sc)) < 1e-9

    if with_vu:
        # unfolded blocks vs the twisted-basis supercell resolvent
        E = 1.1 - 0.4j
        Gs = np.linalg.inv(E * Ss - Hs)
        G_untw = W.conj().T @ Gs @ W
        at = green.at(E)
        for R in (0, 1, 2):
            block = at.unfold(0, [(0, R)])[0]
            ref = G_untw[0:2, 2 * R : 2 * R + 2]
            assert np.max(np.abs(block - ref)) < 1e-10
