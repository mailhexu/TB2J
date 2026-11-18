"""Story 004 (spiral-first-order-response): signed local frozen-band gradients.

Checks :func:`TB2J.spiral_first_order.frozen_band_gradient` against
independent frozen-occupation angle finite differences on toy references:

* nonstationary two-sublattice lab ring: signed nonzero beta gradient ==
  central angle FD of the field-only perturbation; equal-phase control
  is zero (TEST-001);
* planar delta channel evaluated and FD-consistent at zero by planar
  spin symmetry (TEST-001);
* planar local-frame pencil (``V1^beta = Bf sx``, ``V1^delta = Bf sy``,
  ``dS = 0`` for the field-only rotation) at an incommensurate q on a
  primitive k mesh: both channels for both atoms match FD;
* multi-orbital nonorthogonal reference with a co-rotating spinor
  frame: gradients aggregate by atom, the moving-basis overlap
  response ``-eps dS`` is required to match FD, and a hard-rotated
  stored rho is refused instead of silently replacing the occupied
  projector (TEST-002);
* provenance and eigenpair normalization refusal paths.

All FD oracles keep the reference occupations frozen; no re-Fermi.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from scipy.linalg import expm

from TB2J.spiral_first_order import (
    FrozenBandDensityMismatchError,
    FrozenBandEigenpairError,
    FrozenBandProvenanceError,
    frozen_band_gradient,
)

FD_STEP = 1.0e-5
FD_TOL = 1.0e-8

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# ---------------------------------------------------------------------------
# shared toy-model helpers (independent of the implementation under test)
# ---------------------------------------------------------------------------


def _site_operator(nsite, i, block):
    """Embed a 2x2 spin block on the interleaved spinor site ``i``."""
    out = np.zeros((2 * nsite, 2 * nsite), dtype=complex)
    out[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] += block
    return out


def _spin_field(nsite, i, bvec):
    """On-site exchange field ``b . sigma`` (``bvec`` = (x, y, z), eV)."""
    return _site_operator(
        nsite, i, bvec[0] * SIGMA_X + bvec[1] * SIGMA_Y + bvec[2] * SIGMA_Z
    )


def _sharp_occupations(eps_rows):
    """Frozen sharp occupations: fill every level below the largest gap.

    Force-theorem protocol: no partially filled level, so the frozen
    occupations are unambiguous and the analytic/FD paths agree without
    smearing conventions.
    """
    ev = np.sort(np.asarray(eps_rows, dtype=float).ravel())
    gaps = ev[1:] - ev[:-1]
    i0 = int(np.argmax(gaps))
    mu = 0.5 * (ev[i0] + ev[i0 + 1])
    return (np.asarray(eps_rows, dtype=float) < mu).astype(float)


def _spectral_rho(c_rows, f_rows, wk):
    """Independent band density ``sum_k w_k sum_n f_nk c_nk c_nk^dag``."""
    n = c_rows[0].shape[0]
    rho = np.zeros((n, n), dtype=complex)
    for ck, fk, w in zip(c_rows, f_rows, wk):
        rho = rho + w * (ck * fk[None, :]) @ ck.conj().T
    return rho


def _provenance(nelec, constraint="unconstrained"):
    return {
        "schema_version": "tbupy_spiral_state_v2",
        "gauge": "planar_y",
        "q_frac": (0.0, 0.0, 0.0),
        "electron_count": float(nelec),
        "occupation_rule": "fixed: sharp filling below frozen gap centre",
        "constraint": constraint,
        "field_role": "intrinsic_exchange",
        "field_rotation_policy": "co_rotating with the planar reference",
    }


def _diagonalize_reference(H_list):
    """Eigenpairs of each H(k) (ascending eigenvalues, S-normalized c)."""
    eps_rows, c_rows = [], []
    for Hk in H_list:
        ev, vec = np.linalg.eigh(Hk)
        eps_rows.append(ev)
        c_rows.append(vec)
    return np.array(eps_rows), np.array(c_rows)


def _band_energy(ref_rows, f_rows, wk):
    """Frozen-occupation band energy; rows are H(k) or (H(k), S(k))."""
    total = 0.0
    for item, f, w in zip(ref_rows, f_rows, wk):
        if isinstance(item, tuple):
            ev = scipy.linalg.eigh(item[0], item[1], eigvals_only=True)
        else:
            ev = np.linalg.eigvalsh(item)
        total += w * float(np.dot(f, np.sort(ev)))
    return total


def _fd_gradient(make_reference, f_rows, wk):
    """Central frozen-occupation difference of ``make_reference(+/-h)``."""
    plus = _band_energy(make_reference(+FD_STEP), f_rows, wk)
    minus = _band_energy(make_reference(-FD_STEP), f_rows, wk)
    return (plus - minus) / (2.0 * FD_STEP)


# ---------------------------------------------------------------------------
# lab-frame planar ring (orthogonal basis, field-only angle perturbation)
# ---------------------------------------------------------------------------

B_RING = 1.5


def _ring_hopping(nsite, t_even=1.0, t_odd=0.6):
    """Dimerized periodic chain (alternating hopping), spin scalar."""
    n = 2 * nsite
    out = np.zeros((n, n), dtype=complex)
    for i in range(nsite):
        j = (i + 1) % nsite
        t = t_even if i % 2 == 0 else t_odd
        for s in (0, 1):
            out[2 * i + s, 2 * j + s] += -t
            out[2 * j + s, 2 * i + s] += -t
    return out


def _ring_hamiltonian(theta, delta):
    """Field on site i along ``n(th, d) = (sin th cos d, sin d, cos th cos d)``.

    The beta/delta channels act ONLY on the on-site field operators.
    """
    nsite = len(theta)
    H = _ring_hopping(nsite)
    for i in range(nsite):
        nvec = (
            np.sin(theta[i]) * np.cos(delta[i]),
            np.sin(delta[i]),
            np.cos(theta[i]) * np.cos(delta[i]),
        )
        H = H + _spin_field(nsite, i, B_RING * np.asarray(nvec))
    return H


def _ring_v1(theta):
    """Analytic first-rotation operators (derivation-oracle pin):
    ``V1^beta = Bf (cos th sx - sin th sz)``, ``V1^delta = Bf sy``."""
    nsite = len(theta)
    v1_beta, v1_delta = [], []
    for i in range(nsite):
        v1_beta.append(
            _site_operator(
                nsite,
                i,
                B_RING * (np.cos(theta[i]) * SIGMA_X - np.sin(theta[i]) * SIGMA_Z),
            )
        )
        v1_delta.append(_site_operator(nsite, i, B_RING * SIGMA_Y))
    return np.array(v1_beta), np.array(v1_delta)


def _ring_fixture(nsite=4):
    """Deliberately off-equilibrium two-sublattice pattern.

    Angles follow an A/B (0 / ~pi) pattern with unequal deviations, and
    the hopping is dimerized, so no gauge-uniform twist makes the
    configuration stationary: the beta gradient is genuinely nonzero.
    """
    theta = np.array([0.9, np.pi - 0.4, 0.2, np.pi + 0.6])[:nsite]
    H = _ring_hamiltonian(theta, np.zeros(nsite))
    eps, c = _diagonalize_reference([H])
    f = _sharp_occupations(eps)
    return theta, eps, c, f


def _ring_evaluator_args(theta, eps, c, f, **overrides):
    nsite = len(theta)
    v1_beta, v1_delta = _ring_v1(theta)
    args = dict(
        eigenvalues=eps,
        eigenvectors=c,
        kweights=np.array([1.0]),
        occupations=f,
        overlap=None,
        orbital_to_atom=np.arange(nsite),
        v1_beta=v1_beta,
        v1_delta=v1_delta,
        rho_scf=_spectral_rho(c, f, [1.0]),
        provenance=_provenance(float(np.sum(f))),
    )
    args.update(overrides)
    return args


def test_ring_beta_gradient_matches_frozen_angle_fd_nonstationary():
    """Signed nonzero beta gradient == central frozen-occupation angle FD."""
    theta, eps, c, f = _ring_fixture()
    nsite = len(theta)
    res = frozen_band_gradient(**_ring_evaluator_args(theta, eps, c, f))

    fd = np.array(
        [
            _fd_gradient(
                lambda h, i=i: [
                    _ring_hamiltonian(
                        theta + h * (np.arange(nsite) == i), np.zeros(nsite)
                    )
                ],
                f,
                [1.0],
            )
            for i in range(nsite)
        ]
    )
    assert np.max(np.abs(res.g_beta - fd)) < FD_TOL
    # nonstationary reference: signed and clearly nonzero
    assert np.max(np.abs(res.g_beta)) > 1.0e-3
    # report key consumed by the response API/CLI layer
    assert "frozen_band_gradient" in res.as_dict()
    assert res.as_dict()["frozen_band_gradient"]["units"] == "eV/rad"
    # frozen-band torque components are the exact negatives
    assert np.array_equal(res.torque_beta, -res.g_beta)
    assert res.density_discrepancy < 1e-10
    # block-diagonal local operators: the v2 spatial map expands over the
    # interleaved spinor basis and the owner-binned orbital decomposition
    # reproduces the atom gradients exactly (consumer-visible aggregation)
    spinor_map = np.repeat(np.arange(nsite), 2)
    assert np.allclose(
        np.bincount(spinor_map, weights=res.g_beta_orbital, minlength=nsite),
        res.g_beta,
        atol=1e-12,
    )
    assert np.allclose(
        np.bincount(spinor_map, weights=res.g_delta_orbital, minlength=nsite),
        res.g_delta,
        atol=1e-12,
    )


def test_ring_equal_phase_control_beta_below_tolerance():
    """Equal-phase stationary control: every beta gradient below 1e-8."""
    nsite = 6
    theta = np.full(nsite, 0.7)
    H = _ring_hamiltonian(theta, np.zeros(nsite))
    eps, c = _diagonalize_reference([H])
    f = _sharp_occupations(eps)
    res = frozen_band_gradient(**_ring_evaluator_args(theta, eps, c, f))
    assert np.max(np.abs(res.g_beta)) < FD_TOL


def test_ring_delta_channel_evaluated_and_fd_consistent_at_zero():
    """Planar spin symmetry: delta evaluated, ~0, and FD-consistent."""
    theta, eps, c, f = _ring_fixture()
    nsite = len(theta)
    res = frozen_band_gradient(**_ring_evaluator_args(theta, eps, c, f))

    fd_delta = np.array(
        [
            _fd_gradient(
                lambda h, i=i: [_ring_hamiltonian(theta, h * (np.arange(nsite) == i))],
                f,
                [1.0],
            )
            for i in range(nsite)
        ]
    )
    assert np.max(np.abs(res.g_delta - fd_delta)) < FD_TOL
    # evaluated and vanishing by planar symmetry, not merely skipped
    assert np.max(np.abs(res.g_delta)) < FD_TOL


# ---------------------------------------------------------------------------
# planar local-frame pencil at incommensurate q on a primitive k mesh
# ---------------------------------------------------------------------------

Q_OFFGRID = 0.37  # incommensurate pitch
BF_PLANAR = 1.2


def _planar_y_rotation(d):
    return expm(-0.5j * d * SIGMA_Y)


def _planar_pencil(k_frac, beta=0.0, delta=0.0, atom=0, t1=-1.0, t2=-0.4):
    """Planar local-frame folded pencil of a two-sublattice chain.

    Spin-scalar hopping dressed by the co-rotating SU(2) phases
    ``U(D) = exp(-i D sy / 2)`` with relative twist ``D = 2 pi q``;
    fields sit along local z on both sublattices. ``beta``/``delta``
    tilt ONLY the ``atom``-sublattice on-site field (physical
    perturbation; hopping and overlap untouched, so dS = 0 for both
    channels). Basis order: (A up, A down, B up, B down).
    """
    D = 2.0 * np.pi * Q_OFFGRID
    n = 4
    H = np.zeros((n, n), dtype=complex)
    hop_ab = t1 + t2 * np.exp(-2.0j * np.pi * k_frac)
    H[0:2, 2:4] += hop_ab * _planar_y_rotation(D)
    H[2:4, 0:2] += hop_ab.conjugate() * _planar_y_rotation(-D)
    tilt = BF_PLANAR * (
        np.sin(beta) * np.cos(delta) * SIGMA_X
        + np.sin(delta) * SIGMA_Y
        + np.cos(beta) * np.cos(delta) * SIGMA_Z
    )
    H = H + _site_operator(2, atom, tilt)
    other = 1 - atom
    H = H + _site_operator(2, other, BF_PLANAR * SIGMA_Z)
    return H


def test_offgrid_q_primitive_mesh_both_channels_match_fd():
    """Both channels at incommensurate q on a primitive k mesh == FD."""
    kmesh = np.arange(6) / 6.0
    wk = np.full(6, 1.0 / 6.0)
    eps, c = _diagonalize_reference([_planar_pencil(k) for k in kmesh])
    f = _sharp_occupations(eps)

    v1_beta = np.array(
        [
            _site_operator(2, 0, BF_PLANAR * SIGMA_X),
            _site_operator(2, 1, BF_PLANAR * SIGMA_X),
        ]
    )
    v1_delta = np.array(
        [
            _site_operator(2, 0, BF_PLANAR * SIGMA_Y),
            _site_operator(2, 1, BF_PLANAR * SIGMA_Y),
        ]
    )
    nelec = float(np.sum(f * wk[:, None]))
    res = frozen_band_gradient(
        eigenvalues=eps,
        eigenvectors=c,
        kweights=wk,
        occupations=f,
        overlap=None,
        orbital_to_atom=np.arange(2),
        v1_beta=v1_beta,
        v1_delta=v1_delta,
        rho_scf=_spectral_rho(c, f, wk),
        provenance=_provenance(nelec, constraint={"kind": "none"}),
    )

    for atom in (0, 1):
        fd_beta = _fd_gradient(
            lambda h, a=atom: [_planar_pencil(k, beta=h, atom=a) for k in kmesh],
            f,
            wk,
        )
        fd_delta = _fd_gradient(
            lambda h, a=atom: [_planar_pencil(k, delta=h, atom=a) for k in kmesh],
            f,
            wk,
        )
        assert abs(res.g_beta[atom] - fd_beta) < FD_TOL
        assert abs(res.g_delta[atom] - fd_delta) < FD_TOL
    # off-equilibrium relative twist: signed nonzero beta
    assert np.max(np.abs(res.g_beta)) > 1.0e-3


# ---------------------------------------------------------------------------
# multi-orbital nonorthogonal reference with a co-rotating spinor frame
# ---------------------------------------------------------------------------

MO_NATOM, MO_NORB = 2, 2
MO_N = MO_NATOM * MO_NORB * 2  # interleaved spin per orbital
MO_BF = np.array([1.1, 0.75])  # unequal field magnitudes per atom
MO_ONSITE = np.array([0.35, -0.2])
MO_BASE = {0: 0.0, 1: 0.8}  # nonstationary reference (relative angle 0.8)
MO_NFILL = 5  # odd sharp filling of 8 levels, in a real gap at every k


def _mo_index(atom, orbital, spin):
    return atom * (2 * MO_NORB) + orbital * 2 + spin


def _mo_mixing(atom):
    """Fixed nonorthogonal per-atom mixing of the (orbital, spin) block."""
    rng = np.random.default_rng(11 + atom)
    scale = 0.12 / np.sqrt(2.0)
    return np.eye(2 * MO_NORB) + scale * (
        rng.standard_normal((2 * MO_NORB, 2 * MO_NORB))
        + 1j * rng.standard_normal((2 * MO_NORB, 2 * MO_NORB))
    )


def _mo_frame_generator(channel, atom):
    """Anti-Hermitian generator of the co-rotating local spinor frame."""
    gen = {"beta": -0.5j * SIGMA_Y, "delta": -0.5j * SIGMA_X}[channel]
    return (0.25 + 0.1 * atom) * np.kron(np.eye(MO_NORB), gen)


def _mo_W(pert=None):
    """Block-diagonal basis transform ``W_a(eta) = M_a exp(eta G_a)``.

    ``pert = (atom, channel, eta)`` co-rotates atom ``atom``'s frame
    along the perturbation's own channel generator.
    """
    W = np.eye(MO_N, dtype=complex)
    channel = pert[1] if pert is not None else "beta"
    for a in range(MO_NATOM):
        eta = pert[2] if (pert is not None and pert[0] == a) else 0.0
        blk = slice(a * 2 * MO_NORB, (a + 1) * 2 * MO_NORB)
        W[blk, blk] = _mo_mixing(a) @ expm(eta * _mo_frame_generator(channel, a))
    return W


def _mo_field_angle(a, pert):
    """In-plane base angle of atom a plus an optional beta perturbation."""
    eta = pert[2] if (pert is not None and pert[0] == a and pert[1] == "beta") else 0.0
    return MO_BASE[a] + eta


def _mo_h0(k, pert=None):
    """Underlying orthonormal multi-orbital model: hopping + local fields.

    Atom a carries ``Bf_a (cos th sx-in-plane rotated field)`` on BOTH
    its orbitals with in-plane angle ``th = MO_BASE[a] + eta``; the
    delta channel tilts the field out of the plane by ``eta``:
    ``n = (sin th cos d, sin d, cos th cos d)``.
    """
    H = np.zeros((MO_N, MO_N), dtype=complex)
    t = -(1.0 + 0.2 * np.cos(2.0 * np.pi * k))
    for o in range(MO_NORB):
        for s in range(2):
            i, j = _mo_index(0, o, s), _mo_index(1, o, s)
            H[i, j] += t
            H[j, i] += t
    for a in range(MO_NATOM):
        base = a * 2 * MO_NORB
        for s in range(2):
            H[base + s, base + 2 + s] += 0.3
            H[base + 2 + s, base + s] += 0.3
        d = (
            pert[2]
            if (pert is not None and pert[0] == a and pert[1] == "delta")
            else 0.0
        )
        th = _mo_field_angle(a, pert)
        for o in range(MO_NORB):
            for s in (0, 1):
                H[_mo_index(a, o, s), _mo_index(a, o, s)] += MO_ONSITE[a] + MO_BF[
                    a
                ] * np.cos(th) * np.cos(d) * (1 if s == 0 else -1)
            blk = slice(_mo_index(a, o, 0), _mo_index(a, o, 0) + 2)
            H[blk, blk] += MO_BF[a] * (
                np.sin(th) * np.cos(d) * SIGMA_X + np.sin(d) * SIGMA_Y
            )
    return 0.5 * (H + H.conj().T)


def _mo_representation(k, pert=None):
    W = _mo_W(pert)
    return W.conj().T @ _mo_h0(k, pert) @ W, W.conj().T @ W


def _mo_analytic_operators(kmesh, wk):
    """Per-k, per-atom dH (both channels) and dS at the reference.

    ``dW_a(ch)`` carries ONLY atom a's block (``M_a G_a(ch)``); the lab
    first-rotation operators follow the derivation-oracle pin:
    ``V1^beta = Bf (-sin th sz + cos th sx)``, ``V1^delta = Bf sy`` at
    the base angle ``th``. ``dH = dW^dag H0 W + W^dag V1 W + W^dag H0
    dW``; ``dS = dW^dag W + W^dag dW``.
    """
    W0 = _mo_W(None)
    nk = len(kmesh)
    ops = {
        "v1_beta": np.zeros((nk, MO_NATOM, MO_N, MO_N), dtype=complex),
        "v1_delta": np.zeros((nk, MO_NATOM, MO_N, MO_N), dtype=complex),
        "ds_beta": np.zeros((nk, MO_NATOM, MO_N, MO_N), dtype=complex),
        "ds_delta": np.zeros((nk, MO_NATOM, MO_N, MO_N), dtype=complex),
    }
    for ik, k in enumerate(kmesh):
        H0 = _mo_h0(k)
        for a in range(MO_NATOM):
            blk = slice(a * 2 * MO_NORB, (a + 1) * 2 * MO_NORB)
            th = MO_BASE[a]
            v1_lab = {
                "beta": np.zeros((MO_N, MO_N), dtype=complex),
                "delta": np.zeros((MO_N, MO_N), dtype=complex),
            }
            for o in range(MO_NORB):
                i0 = _mo_index(a, o, 0)
                v1_lab["beta"][i0 : i0 + 2, i0 : i0 + 2] += MO_BF[a] * (
                    -np.sin(th) * SIGMA_Z + np.cos(th) * SIGMA_X
                )
                v1_lab["delta"][i0 : i0 + 2, i0 : i0 + 2] += MO_BF[a] * SIGMA_Y
            for ch in ("beta", "delta"):
                dW = np.zeros((MO_N, MO_N), dtype=complex)
                dW[blk, blk] = _mo_mixing(a) @ _mo_frame_generator(ch, a)
                ops[f"v1_{ch}"][ik, a] += (
                    dW.conj().T @ H0 @ W0
                    + W0.conj().T @ v1_lab[ch] @ W0
                    + W0.conj().T @ H0 @ dW
                )
                ops[f"ds_{ch}"][ik, a] += dW.conj().T @ W0 + W0.conj().T @ dW
    return ops["v1_beta"], ops["v1_delta"], ops["ds_beta"], ops["ds_delta"]


def test_multi_orbital_nonorthogonal_matches_fd_and_aggregates_by_atom():
    """Moving-basis overlap response + atom aggregation, spatial orb map."""
    kmesh = np.array([0.0, 0.5])
    wk = np.array([0.5, 0.5])
    S_rows, c_rows, eps_rows = [], [], []
    for k in kmesh:
        Hk, Sk = _mo_representation(k)
        ev, vec = scipy.linalg.eigh(Hk, Sk)
        S_rows.append(Sk)
        eps_rows.append(ev)
        c_rows.append(vec)
    eps_rows = np.array(eps_rows)
    c_rows = np.array(c_rows)
    # odd sharp filling in a real gap at every k (largest-gap autotiller
    # would land on a particle-hole-symmetric filling with zero signal)
    f_rows = np.zeros_like(eps_rows)
    f_rows[:, :MO_NFILL] = 1.0
    v1_beta, v1_delta, ds_beta, ds_delta = _mo_analytic_operators(kmesh, wk)

    # consistent stored rho via the lab-frame projector and inverse congruence:
    # psi_lab = W0 c  =>  rho_rep = W0^{-1} (sum w f psi psi^dag) W0^{-dag}
    W0 = _mo_W(None)
    P_lab = np.zeros((MO_N, MO_N), dtype=complex)
    for ck, fk, w in zip(c_rows, f_rows, wk):
        psi = W0 @ ck
        P_lab += w * (psi * fk[None, :]) @ psi.conj().T
    W0_inv = np.linalg.inv(W0)
    rho_scf = W0_inv @ P_lab @ W0_inv.conj().T

    nelec = float(np.sum(f_rows * wk[:, None]))
    res = frozen_band_gradient(
        eigenvalues=eps_rows,
        eigenvectors=c_rows,
        kweights=wk,
        occupations=f_rows,
        overlap=np.array(S_rows),
        orbital_to_atom=np.repeat(np.arange(MO_NATOM), MO_NORB),  # spatial (norb,)
        v1_beta=v1_beta,
        v1_delta=v1_delta,
        ds_beta=ds_beta,
        ds_delta=ds_delta,
        rho_scf=rho_scf,
        provenance=_provenance(nelec),
    )

    for a in range(MO_NATOM):
        fd_beta = _fd_gradient(
            lambda h, at=a: [_mo_representation(k, (at, "beta", h)) for k in kmesh],
            f_rows,
            wk,
        )
        fd_delta = _fd_gradient(
            lambda h, at=a: [_mo_representation(k, (at, "delta", h)) for k in kmesh],
            f_rows,
            wk,
        )
        assert abs(res.g_beta[a] - fd_beta) < FD_TOL
        assert abs(res.g_delta[a] - fd_delta) < FD_TOL
    # nonstationary relative angle: the beta channel carries real signal
    assert np.max(np.abs(res.g_beta)) > 1.0e-2

    # orbital decomposition: per-basis-orbital diagonal of the summed
    # products; the consumer-visible atomic aggregation itself is the
    # g_beta/g_delta arrays (each atom's FULL trace), verified against FD
    # above. The decomposition obeys the same total-sum identity.
    assert abs(float(np.sum(res.g_beta_orbital)) - float(np.sum(res.g_beta))) < 1e-10
    assert abs(float(np.sum(res.g_delta_orbital)) - float(np.sum(res.g_delta))) < 1e-10

    # sensitivity: dropping the overlap response visibly breaks FD agreement
    res_nods = frozen_band_gradient(
        eigenvalues=eps_rows,
        eigenvectors=c_rows,
        kweights=wk,
        occupations=f_rows,
        overlap=np.array(S_rows),
        orbital_to_atom=np.repeat(np.arange(MO_NATOM), MO_NORB),
        v1_beta=v1_beta,
        v1_delta=v1_delta,
        ds_beta=None,
        ds_delta=None,
        rho_scf=rho_scf,
        provenance=_provenance(nelec),
    )
    fd0 = _fd_gradient(
        lambda h: [_mo_representation(k, (0, "beta", h)) for k in kmesh],
        f_rows,
        wk,
    )
    assert abs(res_nods.g_beta[0] - fd0) > 1.0e-6


# ---------------------------------------------------------------------------
# refusal paths
# ---------------------------------------------------------------------------


def test_hard_rotated_stored_density_is_refused():
    """A hard-rotated stored rho raises instead of silently replacing rho_spec."""
    theta, eps, c, f = _ring_fixture()
    rho_ok = _spectral_rho(c, f, [1.0])
    U = expm(-0.05j * SIGMA_Y)  # extra rotation absent from H
    rho_bad = rho_ok.copy()
    rho_bad[0:2, 0:2] = U @ rho_ok[0:2, 0:2] @ U.conj().T

    kwargs = _ring_evaluator_args(theta, eps, c, f)
    kwargs.pop("rho_scf")
    kwargs.pop("provenance")
    res = frozen_band_gradient(
        rho_scf=rho_ok, provenance=_provenance(float(f.sum())), **kwargs
    )
    assert res.density_discrepancy < 1e-10
    with pytest.raises(FrozenBandDensityMismatchError) as exc:
        frozen_band_gradient(
            rho_scf=rho_bad, provenance=_provenance(float(f.sum())), **kwargs
        )
    assert "density" in str(exc.value).lower()


def test_missing_provenance_refused():
    theta, eps, c, f = _ring_fixture()
    kwargs = _ring_evaluator_args(theta, eps, c, f)
    kwargs.pop("provenance")
    with pytest.raises(FrozenBandProvenanceError):
        frozen_band_gradient(provenance=None, **kwargs)
    for key in (
        "schema_version",
        "gauge",
        "q_frac",
        "electron_count",
        "occupation_rule",
        "constraint",
        "field_role",
        "field_rotation_policy",
    ):
        prov = _provenance(float(f.sum()))
        prov.pop(key)
        with pytest.raises(FrozenBandProvenanceError):
            frozen_band_gradient(provenance=prov, **kwargs)
    # inconsistent electron count vs the frozen occupations
    with pytest.raises(FrozenBandProvenanceError):
        frozen_band_gradient(provenance=_provenance(float(f.sum()) + 1.0), **kwargs)
    # empty constraint record (v2 always stores an explicit constraint)
    with pytest.raises(FrozenBandProvenanceError):
        frozen_band_gradient(
            provenance=_provenance(float(f.sum()), constraint=""), **kwargs
        )


def test_unnormalized_eigenvectors_refused():
    theta, eps, c, f = _ring_fixture()
    c_bad = c.copy()
    c_bad[0][:, 0] *= 1.001
    kwargs = _ring_evaluator_args(theta, eps, c, f)
    kwargs["eigenvectors"] = c_bad
    with pytest.raises(FrozenBandEigenpairError):
        frozen_band_gradient(**kwargs)
