"""Toy-ring SpiralState bundles for the spiral Green-function tests.

Story 003 fixtures.  The builders are independent of tbupy: the
contract arrays are constructed directly (single-sublattice rings,
interleaved spin, ``q_frac = 1/6``), following the model conventions
of ``docs/sympy/spiral_state_mft.py`` and ``spiral_green_function_mft.py``:
spiral angles ``theta[a, mu] = 2 pi q (a + tau_mu) + phi_mu`` and
supercell spin-interleaved index ``2 (a norb + mu) + s``.

Two bundle families:

* random hermitian rings (:func:`make_ring_state` with
  :func:`ring_hopping_classes`): real f8 collinear channels, optionally
  nonorthogonal (FPLO-style rotated overlap) and optionally with a
  rotating-frame ``V_U``;
* local-field ring of two identical uncoupled orbital copies
  (:func:`degenerate_ring_classes`): every folded level is exactly
  doubly degenerate with a known analytic dispersion.

:func:`rebuild_hq_sq` applies the contract rebuild rule through the
tbupy multi-sublattice assembler (single source of truth) plus the
interleaved local field and ``V_U``; :func:`supercell_spiral` is the
independent docs-convention supercell reference used by the tests.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

from TB2J.spiral_green import SpiralGreen, SpiralState

__all__ = [
    "ring_kmesh",
    "ring_hopping_classes",
    "degenerate_ring_classes",
    "make_ring_state",
    "random_hermitian",
    "supercell_spiral",
    "twist_unitary",
    "rebuild_hq_sq",
    "pencil_eigenvalues",
]


def ring_kmesh(N: int) -> np.ndarray:
    """Uniform commensurate mesh ``k_m = (m/N, 0, 0)``, m = 0..N-1."""
    return np.column_stack((np.arange(N) / N, np.zeros(N), np.zeros(N)))


def _overlap(mu: int, nu: int, R: int) -> float:
    """Docs-convention overlap amplitude of a stored class."""
    return 0.3 * np.exp(-0.7 * (abs(mu - nu) + abs(R))) + 0.1


def ring_hopping_classes(N: int, seed: int):
    """Random hermitian hopping classes of a single-orbital ring.

    Same shell structure as ``hermitian_classes`` in the derivation
    scripts, restricted to one sublattice with real (f8) amplitudes:
    symmetric pairs at ``+-R`` for ``R = 1..(N-1)//2`` plus the
    self-conjugate ``R = N/2`` shell for even N; on-site amplitude 0
    (the on-site energy is stored separately).
    """
    rng = np.random.default_rng(seed)
    reps = list(range(1, (N - 1) // 2 + 1))
    if N % 2 == 0:
        reps.append(N // 2)
    classes = []
    for R in reps:
        amp = round(float(rng.normal()), 6)
        classes.append((0, 0, R, amp))
        classes.append((0, 0, -R, amp))
    eps0 = round(float(rng.normal()), 6)
    return classes, [eps0]


def degenerate_ring_classes(t: float = 0.35, eps0: float = 0.2):
    """Nearest-neighbour ring of two identical uncoupled orbital copies.

    With a uniform ``B_local`` the folded spectrum is known analytically
    and every level is exactly doubly degenerate (copy symmetry).
    """
    classes = []
    for copy in (0, 1):
        classes.append((copy, copy, 1, t))
        classes.append((copy, copy, -1, t))
    return classes, [eps0, eps0]


def _arrays_from_classes(classes, eps_list, overlap: bool):
    """Bundle arrays ``(Rlist, HR, SR)`` from stored hopping classes."""
    rs = sorted({R for (_, _, R, _) in classes} | {0})
    norb = len(eps_list)
    HR = np.zeros((len(rs), norb, norb))
    SR = np.zeros((len(rs), norb, norb))
    SR[rs.index(0)] = np.eye(norb)
    for mu, nu, R, amp in classes:
        pos = rs.index(R)
        HR[pos, mu, nu] += amp
        if overlap:
            SR[pos, mu, nu] += _overlap(mu, nu, R)
    for mu, eps in enumerate(eps_list):
        HR[rs.index(0), mu, mu] += eps
    rlist = np.zeros((len(rs), 3), dtype=np.int64)
    rlist[:, 0] = rs
    return rlist, HR, SR


def make_ring_state(
    N: int,
    classes,
    eps_list,
    q: float = 1.0 / 6.0,
    b_local=(0.4,),
    overlap: bool = False,
    taus=None,
    phis=None,
    v_u=None,
    width: float = 0.2,
):
    """Build a frozen ``SpiralState`` toy bundle on an N-cell ring.

    ``efermi`` is the 55% quantile of the folded spectrum and ``rho`` the
    k-averaged occupied twisted-gauge density matrix
    ``sum_k w_k C_k f_k C_k^dag`` - both from direct diagonalization of
    the folded pencils (including B_local and V_U).
    """
    norb = len(eps_list)
    taus = np.zeros((norb, 3)) if taus is None else np.asarray(taus, dtype=float)
    phis = np.zeros(norb) if phis is None else np.asarray(phis, dtype=float)
    b_local = np.asarray(b_local, dtype=float)
    if b_local.shape != (norb,):
        b_local = np.full(norb, float(b_local[0]))
    v_u = None if v_u is None else np.asarray(v_u, dtype=complex)
    rlist, HR, SR = _arrays_from_classes(classes, eps_list, overlap)
    state = SpiralState(
        HR_up=HR,
        HR_dn=HR.copy(),
        SR=SR,
        Rlist=rlist,
        q_frac=np.array([q, 0.0, 0.0]),
        taus=taus,
        phis=phis,
        B_local=b_local,
        rho=None,
        V_U=v_u,
        efermi=0.0,
        torque_norms=np.zeros(norb),
        metadata_json="{}",
    )
    green = SpiralGreen(state, ring_kmesh(N))
    efermi = float(np.quantile(green.evals, 0.55))
    f = 1.0 / (1.0 + np.exp((green.evals - efermi) / width))
    # k-averaged occupied twisted-gauge density matrix: sum_k w_k C f C^dag
    rho = np.einsum(
        "k,kin,kn,kjn->ij", green.kweights, green.evecs, f, green.evecs.conj()
    )
    meta = {
        "dc_type": "spiral_frozen_state",
        "hubbard_type": None,
        "width": width,
        "nel": float(np.sum(green.kweights[:, None] * f)),
        "kmesh": [N, 1, 1],
        "kweights": [1.0 / N],
        "guards": {"torque_max": 0.0, "converged": True},
        "q_even_diag": {"q_frac": [q, 0.0, 0.0], "commensurate": True},
    }
    state = dataclasses.replace(
        state,
        efermi=efermi,
        rho=rho,
        metadata_json=json.dumps(meta),
    )
    return state


def random_hermitian(n: int, seed: int, scale: float = 0.1) -> np.ndarray:
    """Random hermitian ``(n, n)`` matrix (rotating-frame V_U fixture)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return scale * 0.5 * (A + np.conj(A.T))


def supercell_spiral(
    N: int,
    classes,
    eps_list,
    q_frac,
    taus,
    phis,
    b_local,
    overlap: bool = False,
):
    """Docs-convention supercell spiral ``(H_s, S_s)`` (untwisted basis).

    Independent reference builder: hopping classes dressed with the
    half-twist phases ``exp(-i sigma dtheta/2)`` of
    ``theta[a, mu] = 2 pi q_frac . (a e_1 + tau_mu) + phi_mu`` (i.e.
    ``dtheta = 2 pi q_frac . (R_vec + tau_nu - tau_mu) + phi_nu -
    phi_mu``), on-site ``eps_mu + (1/2 - s) B_mu`` (contract
    normalization: B_local is the full up/down splitting),
    spin-interleaved index ``2 (a norb + mu) + s``.
    """
    norb = len(eps_list)
    q_frac = np.asarray(q_frac, dtype=float)
    taus = np.asarray(taus, dtype=float)
    phis = np.asarray(phis, dtype=float)
    b_local = np.asarray(b_local, dtype=float)
    size = 2 * norb * N
    H = np.zeros((size, size), dtype=complex)
    S = np.eye(size, dtype=complex) if overlap else None

    def idx(a, mu, s):
        return 2 * (a % N) * norb + 2 * mu + s

    by_res = {}
    for mu, nu, R, amp in classes:
        by_res.setdefault(R % N, []).append((mu, nu, R, amp))
    for a in range(N):
        for rep, lst in by_res.items():
            b = (a + rep) % N
            for mu, nu, R, amp in lst:
                for s in (0, 1):
                    sigma = 1.0 if s == 0 else -1.0
                    dtau = np.array([R, 0.0, 0.0]) + taus[nu] - taus[mu]
                    dth = 2 * np.pi * (q_frac @ dtau) + phis[nu] - phis[mu]
                    ph = np.exp(-0.5j * sigma * dth)
                    H[idx(a, mu, s), idx(b, nu, s)] += amp * ph
                    if overlap:
                        ov = _overlap(mu, nu, R)
                        S[idx(a, mu, s), idx(b, nu, s)] += ov * ph
    for a in range(N):
        for mu in range(norb):
            for s in (0, 1):
                H[idx(a, mu, s), idx(a, mu, s)] += (
                    eps_list[mu] + (0.5 - s) * b_local[mu]
                )
    return H, S


def twist_unitary(N: int, norb: int, q: float) -> np.ndarray:
    """Twisted-Bloch unitary (derivation script, Eq. section 2).

    ``W[(a mu s), (k mu s)] = N^{-1/2} e^{2 pi i k a} e^{+i sigma_s pi q a}``:
    the basis carries the cell half-twist only; the sublattice phases
    live in the folded Hamiltonian.
    """
    W = np.zeros((2 * norb * N, 2 * norb * N), dtype=complex)
    for a in range(N):
        for n in range(N):
            k = n / N
            for mu in range(norb):
                for s in (0, 1):
                    sigma = 1.0 if s == 0 else -1.0
                    W[2 * a * norb + 2 * mu + s, 2 * norb * n + 2 * mu + s] = (
                        np.exp(-2j * np.pi * k * a)
                        / np.sqrt(N)
                        * np.exp(0.5j * sigma * 2 * np.pi * q * a)
                    )
    return W


def rebuild_hq_sq(state: SpiralState, k_frac):
    """Contract rebuild rule via the tbupy assembler (source of truth).

    ``Hq(k) = assemble_multisublattice_hq_sq(...) + interleave(B/2
    sigma_z) + V_U``, ``Sq(k)`` from the same assembler.
    """
    from tbupy.generalized_bloch import (
        MultiSublatticeSpiralConfig,
        assemble_multisublattice_hq_sq,
    )

    config = MultiSublatticeSpiralConfig(state.q_frac, state.taus, state.phis)
    Hq, Sq = assemble_multisublattice_hq_sq(
        state.HR_up, state.HR_dn, state.SR, state.Rlist, config, k_frac
    )
    B = np.asarray(state.B_local, dtype=float)
    Hq[0::2, 0::2] += np.diag(0.5 * B)
    Hq[1::2, 1::2] -= np.diag(0.5 * B)
    if state.V_U is not None:
        Hq = Hq + np.asarray(state.V_U, dtype=complex)
    return Hq, Sq


def pencil_eigenvalues(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Hermitian-definite pencil eigenvalues ``(H, S)``, ascending."""
    L = np.linalg.cholesky(S)
    Li = np.linalg.inv(L)
    A = Li @ H @ Li.conj().T
    return np.linalg.eigvalsh(0.5 * (A + A.conj().T))
