"""Consumer-facing tests for Born scattering spectra (Story 010, ADR-004).

Acceptance criteria:
- TEST-001: truncated-Fock Lehmann/perturbative reference on a controlled
  ring model matches both Born kernels (magnon and phonon), with
  truncation convergence;
- TEST-002: g -> 0 recovers bare spectra exactly, with quadratic coupling
  scaling;
- TEST-003: symmetry-forbidden controls produce no spurious damping, and
  the full matrix structure is kept at degeneracies;
- TEST-004: T = 0 limit (emission-only) and finite-T Bose occupations;
- TEST-005: artificial broadening / grid metadata reported separately from
  calculated linewidths.
"""

import numpy as np
import pytest

from TB2J.magnon_phonon.scattering import (
    ScatteringRequest,
    bose_occupation,
    magnon_self_energy,
)
from TB2J.magnon_phonon.vertex import MagnonPhononModel
from tests.test_magnon_phonon_vertex import (
    _bonds_container,
    _pattern,
    _pattern_rows,
    _phonon_input,
    _spring_magnon,
)

KB_EV_K = 8.617333262e-5

# ---------------------------------------------------------------------------
# controlled ring model for the truncated-Fock Lehmann reference
# ---------------------------------------------------------------------------
NMOM = 4
DELTA_EV = 0.08
T_HOP_EV = 0.03
OMEGA0_EV = 0.05
OMEGA_DISP_EV = 0.02
TEMP_K = 150.0
G_SCALE = 0.0015
#: larger scale for the phonon bubble (its signal is n1 - n2 suppressed)
G_SCALE_PH = 0.006


def _magnon_band(k):
    # the sin term lifts the k <-> -k degeneracy: the one-shot Born kernels
    # are nondegenerate-perturbative statements
    return (
        DELTA_EV
        + 2 * T_HOP_EV * (1 - np.cos(2 * np.pi * k / NMOM))
        + 0.015 * np.sin(2 * np.pi * k / NMOM)
    )


#: resonance-free phonon energies: |Omega_q - deps| and |Omega_q - eps_k|
#: are all >= 0.035 eV, so no one-magnon level hybridizes with a
#: magnon-plus-phonon state and the O(g^2) comparison stays controlled
_PHONON_LEVELS_EV = (0.245, 0.265, 0.29, 0.31)


def _phonon_band(q):
    return _PHONON_LEVELS_EV[int(round(q * NMOM)) % NMOM]


def _couplings(seed=5):
    """Ring couplings satisfying the lattice Hermiticity identity
    g_{k+q,-q} = g*_{kq}, so that H_int = sum g a^dag_{k+q} a_k (b_q +
    b^dag_{-q}) is Hermitian as a single sum (declared vertex form)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(NMOM, NMOM)) + 1j * rng.normal(size=(NMOM, NMOM))
    out = np.zeros((NMOM, NMOM), dtype=complex)
    out[:, 0] = g[:, 0].real  # q = 0 is self-conjugate: real
    # q = 2 is self-conjugate (-q = 2) with k -> k+2: real and k-paired
    out[:, 2] = g[:, 2].real
    out[2, 2], out[3, 2] = out[0, 2], out[1, 2]
    # q = 1 and q = 3 = -1 form the conjugate pair: g_{k+1, 3} = g*_{k, 1}
    out[:, 1] = g[:, 1]
    for k in range(NMOM):
        out[(k + 1) % NMOM, 3] = np.conj(out[k, 1])
    return out


def _enum_states(prefix, budget, n_modes, out):
    if len(prefix) == n_modes:
        out.append(prefix)
        return
    for n in range(budget + 1):
        _enum_states(prefix + (n,), budget - n, n_modes, out)


def _ring_hamiltonian(couplings, n_mag_max=2, n_ph_max=2, coupling_scale=G_SCALE):
    """Full H on the truncated Fock space of the ring model.

    Magnon occupations n_k and phonon occupations m_q, each sector with a
    bounded total number.  Interaction (the exact ring analogue of the
    declared first-order Hamiltonian):

        H_int = sum_{kq} g_{kq} c^dag_{k+q} c_k (b_q + b^dag_{-q}),
    """
    mag_states = []
    _enum_states((), n_mag_max, NMOM, mag_states)
    ph_states = []
    _enum_states((), n_ph_max, NMOM, ph_states)
    mag_index = {s: i for i, s in enumerate(mag_states)}
    ph_index = {s: i for i, s in enumerate(ph_states)}
    n_ph = len(ph_states)
    dim = len(mag_states) * n_ph

    def idx(mag_i, ph_i):
        return mag_i * n_ph + ph_i

    H = np.zeros((dim, dim), dtype=complex)
    for mag_i, mag in enumerate(mag_states):
        em = sum(n * _magnon_band(k) for k, n in enumerate(mag))
        for ph_i, ph in enumerate(ph_states):
            ep = sum(m * _phonon_band(q) for q, m in enumerate(ph))
            H[idx(mag_i, ph_i), idx(mag_i, ph_i)] = em + ep
    for k in range(NMOM):
        for q in range(NMOM):
            g = coupling_scale * couplings[k, q]
            qm = (-q) % NMOM
            for mag_i, mag in enumerate(mag_states):
                if mag[k] == 0:
                    continue
                tgt = list(mag)
                tgt[k] -= 1
                tgt[(k + q) % NMOM] += 1
                if tgt[(k + q) % NMOM] > n_mag_max:
                    continue
                amp_mag = np.sqrt(mag[k] * tgt[(k + q) % NMOM])
                for ph_i, ph in enumerate(ph_states):
                    for sign, qph in ((+1, q), (-1, qm)):
                        if sign == +1:
                            if ph[qph] == 0:
                                continue
                            pht = list(ph)
                            pht[qph] -= 1
                            amp_ph = np.sqrt(ph[qph])
                        else:
                            if ph[qph] + 1 > n_ph_max:
                                continue
                            pht = list(ph)
                            pht[qph] += 1
                            if sum(pht) > n_ph_max:
                                continue
                            amp_ph = np.sqrt(pht[qph])
                        row = idx(mag_index[tuple(tgt)], ph_index[tuple(pht)])
                        col = idx(mag_i, ph_i)
                        H[row, col] += g * amp_mag * amp_ph
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)  # declared form
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs, (mag_states, ph_states, mag_index, ph_index)


def _ring_greens(
    evals, evecs, sector, species, index, omega, eta, temperature_K, bare=False
):
    """Thermal commutator Green function of c_index (magnon) or b_index."""
    if bare:
        eps0 = _magnon_band(index) if species == "magnon" else _phonon_band(index)
        return 1.0 / (omega - eps0 + 1j * eta)
    mag_states, ph_states, mag_index, ph_index = sector
    states, idx_map = (
        (mag_states, mag_index) if species == "magnon" else (ph_states, ph_index)
    )
    p = np.exp(-evals / (KB_EV_K * temperature_K))
    p /= p.sum()
    mag_dim, ph_dim = len(mag_states), len(ph_states)
    dim = len(evals)
    # operator a_index (annihilation; the commutator Lehmann formula uses
    # <m|A|n> with A the annihilator) embedded on the Fock product space
    op = np.zeros((dim, dim))
    for s in states:
        t = list(s)
        t[index] += 1
        t = tuple(t)
        if t not in idx_map:
            continue
        amp = np.sqrt(t[index])
        if species == "magnon":
            for other in range(ph_dim):
                op[idx_map[s] * ph_dim + other, idx_map[t] * ph_dim + other] = amp
        else:
            for other in range(mag_dim):
                op[other * ph_dim + idx_map[s], other * ph_dim + idx_map[t]] = amp
    # full Lehmann sum over eigenstate pairs (gauge invariant under any
    # eigenvector convention within degenerate subspaces):
    # G(w) = sum_mn (p_m - p_n) |<m| op |n>|^2 / (w - (E_n - E_m) + i eta)
    mat = evecs.conj().T @ op @ evecs
    gap = evals[None, :] - evals[:, None]  # E_n - E_m for [m, n]
    weight = p[:, None] - p[None, :]
    denom = omega[:, None, None] - gap[None, :, :] + 1j * eta
    contrib = weight[None, :, :] * np.abs(mat)[None, :, :] ** 2 / denom
    return contrib.sum(axis=(1, 2))


def _truncated_bare(couplings, species, index, omega, eta, n_ph_max=2):
    """Zero-coupling Green function in the SAME truncated Fock space (the
    commutator identity is not closed by the truncation, so the analytic
    free form is not the correct reference)."""
    evals, evecs, sector = _ring_hamiltonian(
        np.zeros((NMOM, NMOM), dtype=complex), n_ph_max=n_ph_max
    )
    return _ring_greens(evals, evecs, sector, species, index, omega, eta, TEMP_K)


def _coordinate_operator(sector, q_ext, om_q):
    """x_q = (b_q + b^dag_q)/sqrt(2 Omega_q) on the truncated product space."""
    mag_states, ph_states, mag_index, ph_index = sector
    dim = len(mag_states) * len(ph_states)
    op = np.zeros((dim, dim))
    for s in ph_states:
        up = list(s)
        up[q_ext] += 1
        up = tuple(up)
        if up in ph_index:
            amp = np.sqrt(up[q_ext])
            for other in range(len(mag_states)):
                src = other * len(ph_states) + mag_index[s]
                dst = other * len(ph_states) + ph_index[up]
                op[dst, src] += amp  # b^dag part
        dn = list(s)
        if dn[q_ext] > 0:
            dn[q_ext] -= 1
            dn = tuple(dn)
            amp = np.sqrt(s[q_ext])
            for other in range(len(mag_states)):
                src = other * len(ph_states) + mag_index[s]
                dst = other * len(ph_states) + ph_index[dn]
                op[dst, src] += amp  # b part
    return op / np.sqrt(2 * om_q)


def _ring_greens_operator(evals, evecs, op, omega, eta, temperature_K):
    """Commutator Green function of an explicit operator on the product
    space (full eigenbasis Lehmann sum)."""
    p = np.exp(-evals / (KB_EV_K * temperature_K))
    p /= p.sum()
    mat = evecs.conj().T @ op @ evecs
    gap = evals[None, :] - evals[:, None]
    weight = p[:, None] - p[None, :]
    contrib = (
        weight[None, :, :]
        * np.abs(mat)[None, :, :] ** 2
        / (omega[:, None, None] - gap[None, :, :] + 1j * eta)
    )
    return contrib.sum(axis=(1, 2))


def _safe_omega(evals, candidates, temperature_K=1.0, min_dist=0.01, p_min=1e-3):
    """Grid points at least ``min_dist`` from any weighted transition of the
    interacting reference (Boltzmann weight above ``p_min``): the one-shot
    Born comparison is a controlled off-resonance statement."""
    p = np.exp(-evals / (KB_EV_K * temperature_K))
    p /= p.sum()
    relevant = evals[p > p_min]
    gaps = np.unique(np.round(relevant - relevant.min(), 12))
    keep = [w for w in candidates if np.min(np.abs(gaps - w)) > min_dist and w > 0.0]
    assert len(keep) >= 4, "not enough off-resonance grid points"
    return np.array(keep)


def _extract_sigma(g_minus_g0, omega, eps0, eta):
    """One-shot self-energy from the O(g^2) Green-function difference:
    Sigma = (G0^R)^{-1} (G - G0^R) (G0^R)^{-1} with the retarded bare
    inverse (omega - eps0 + i eta), so the broadening bias cancels."""
    g0inv = omega - eps0 + 1j * eta
    return g0inv * g_minus_g0 * g0inv


def _kernel_magnon_input(k_ext, couplings, eta_):
    g = np.zeros((NMOM, 1, 1, 1), dtype=complex)
    eps_kq = np.zeros((NMOM, 1))
    om_q = np.zeros((NMOM, 1))
    for q in range(NMOM):
        # ring H carries no 1/sqrt(N_cell); the declared-vertex amplitude is
        # sqrt(NMOM) x the ring coupling
        g[q, 0, 0, 0] = np.sqrt(NMOM) * G_SCALE * couplings[k_ext, q]
        eps_kq[q, 0] = _magnon_band((k_ext + q) % NMOM)
        om_q[q, 0] = _phonon_band(q)
    return g, eps_kq, om_q


def _minimal_dp_weight(g_ring, e1, e2, om_q, temperature_K, cutoff=3, ladder=12):
    """Second-order double-pole weight at the bare +-Omega_q poles of the
    coordinate Green function, computed from the per-channel MINIMAL Fock
    model (modes c_k, c_{k+q}, b_q) with the oracle's OWN interaction
    V = g (c2^dag c1 + c1^dag c2)(b + b^dag) and its own per-configuration
    energy shifts: never the implementation's closed-form S.  The
    construction is the script-certified single-dressing Lehmann sum
    (docs/sympy/born_scattering_spectra,
    check_phonon_hot_band_prefactor, part (d)); the returned weight is
    attached identically at +Omega_q and -Omega_q (exactly symmetric S
    pair)."""
    states = [
        (a, b, m)
        for a in range(cutoff + 1)
        for b in range(cutoff + 1)
        for m in range(ladder + 1)
    ]
    index = {s: i for i, s in enumerate(states)}
    dim = len(states)

    def _ann(which):
        op = np.zeros((dim, dim))
        for s in states:
            if s[which] > 0:
                t = list(s)
                t[which] -= 1
                op[index[tuple(t)], index[s]] = np.sqrt(s[which])
        return op

    a1, a2, b = _ann(0), _ann(1), _ann(2)
    x = (b + b.conj().T) / np.sqrt(2.0 * om_q)
    x0mask = x != 0
    V = g_ring * a2.conj().T @ a1 @ (b + b.conj().T) + np.conj(
        g_ring
    ) * a1.conj().T @ a2 @ (b + b.conj().T)
    E = np.array([a * e1 + b2 * e2 + m * om_q for a, b2, m in states])
    p = np.exp(-E / (KB_EV_K * temperature_K))
    p /= p.sum()
    dmat = E[:, None] - E[None, :]
    safe = np.where(dmat == 0.0, 1.0, dmat)
    dinv = np.where(dmat == 0.0, 0.0, 1.0 / safe)
    np.fill_diagonal(dinv, 0.0)
    dE = np.einsum("ca,ac->a", V.T * dinv, V)
    dp_weight = 0.0
    for a in range(dim):
        for b3 in range(dim):
            gap = round(E[b3] - E[a], 9)
            if x0mask[a, b3] and abs(gap - om_q) < 1e-9:
                dp_weight += (p[a] - p[b3]) * abs(x[a, b3]) ** 2 * dE[a].real
                dp_weight -= (p[a] - p[b3]) * abs(x[a, b3]) ** 2 * dE[b3].real
    return dp_weight


def _phonon_bare_pole_oracle(couplings, q_ext, n_ph_max, omega, eta, eps_k, eps_kq):
    """Independent truncated-Fock Lehmann oracle for the phonon
    coordinate-channel hot-band kernel on the ring model.

    Builds the bare product Fock space (magnon cutoff 2, phonon cutoff
    ``n_ph_max``), the ring interaction
    ``V = sum_{kq} g_kq c^dag_{k+q} c_k (b_q + b^dag_{-q})`` at scale
    ``G_SCALE_PH``, and evaluates the second-order Lehmann correction of
    the coordinate Green function x_q as a pole sum over bare
    energy differences:

    - double poles at +-Omega_q (bare x-connected pairs) with the
      per-configuration single-dressing energy shifts dE of the oracle's
      OWN Fock-space interaction V (never the implementation's closed
      form);
    - simple poles at +- deps_k (hop k -> k+q, same phonon content) with
      the single-dressing coordinate amplitudes xtilde.

    Returns the complex pole sum evaluated on ``omega``.
    """
    mag_states = []
    _enum_states((), 2, NMOM, mag_states)
    ph_states = []
    _enum_states((), n_ph_max, NMOM, ph_states)
    nm, nph = len(mag_states), len(ph_states)
    dim = nm * nph
    mag_index = {s: i for i, s in enumerate(mag_states)}
    ph_index = {s: i for i, s in enumerate(ph_states)}

    emb_ph = np.zeros((dim, dim))
    for ai, ma in enumerate(mag_states):
        for pi, pa in enumerate(ph_states):
            col = ai * nph + pi
            for aj, mb in enumerate(mag_states):
                for pj, pb in enumerate(ph_states):
                    if ma == mb and _one_phonon(pa, pb, q_ext):
                        emb_ph[col, aj * nph + ph_index[pb]] += 1.0
    E = np.zeros(dim)
    for ai, ma in enumerate(mag_states):
        em = sum(n * _magnon_band(k) for k, n in enumerate(ma))
        for pi, pa in enumerate(ph_states):
            ep = sum(m * _phonon_band(q) for q, m in enumerate(pa))
            E[ai * nph + pi] = em + ep
    p = np.exp(-E / (KB_EV_K * TEMP_K))
    p /= p.sum()
    # interaction in the ring convention, restricted to the DECLARED
    # external channel: the bubble attaches to the propagating branch at
    # q_ext only (hops k -> k+q_ext with the q_ext phonon operator), which
    # is exactly the channel the closed kernel sums over.  Lehmann terms
    # with internal phonon momenta q' != q_ext (other-branch density
    # fluctuations) are outside the declared one-shot object.
    V = np.zeros((dim, dim), dtype=complex)
    q = q_ext
    qm = (-q) % NMOM
    for k in range(NMOM):
        gkq = G_SCALE_PH * couplings[k, q]
        for col, (ma, pa) in enumerate((m, ph) for m in mag_states for ph in ph_states):
            if ma[k] == 0:
                continue
            tgt = list(ma)
            tgt[k] -= 1
            tgt[(k + q) % NMOM] += 1
            if tgt[(k + q) % NMOM] > 2:
                continue
            amp_mag = np.sqrt(ma[k] * tgt[(k + q) % NMOM])
            for sign, qph in ((+1, q), (-1, qm)):
                if sign == +1:
                    if pa[qph] == 0:
                        continue
                    pht = list(pa)
                    pht[qph] -= 1
                    amp_ph = np.sqrt(pa[qph])
                else:
                    if pa[qph] + 1 > n_ph_max:
                        continue
                    pht = list(pa)
                    pht[qph] += 1
                    if sum(pht) > n_ph_max:
                        continue
                    amp_ph = np.sqrt(pht[qph])
                row = mag_index[tuple(tgt)] * nph + ph_index[tuple(pht)]
                V[row, col] += gkq * amp_mag * amp_ph
    # dressed coordinate operator pieces
    x = emb_ph / np.sqrt(2.0 * _phonon_band(q_ext))
    dmat = E[:, None] - E[None, :]
    safe = np.where(dmat == 0.0, 1.0, dmat)
    dinv = np.where(dmat == 0.0, 0.0, 1.0 / safe)
    np.fill_diagonal(dinv, 0.0)
    xtil = (V.T * dinv) @ x + x @ (V * dinv).T
    x0mask = x != 0
    # simple poles from the oracle's own dressed coordinate amplitudes
    pole_w = {}
    for a in range(dim):
        for b in range(dim):
            gap = round(E[b] - E[a], 9)
            if x0mask[a, b]:
                continue
            wgt = (p[a] - p[b]) * abs(xtil[a, b]) ** 2
            if abs(wgt) > 1e-30:
                pole_w[("sp", gap)] = pole_w.get(("sp", gap), 0.0) + wgt.real
    # double poles: the per-channel minimal Fock construction (own
    # interaction and energy shifts, converged ladder cutoffs), summed
    # over internal momenta with the ring couplings
    s_tot = 0.0
    for k in range(NMOM):
        s_tot += _minimal_dp_weight(
            G_SCALE_PH * couplings[k, q_ext],
            eps_k[k, 0],
            eps_kq[k, 0],
            _phonon_band(q_ext),
            TEMP_K,
        )
    pole_w[("dp", _phonon_band(q_ext))] = s_tot
    pole_w[("dp", -_phonon_band(q_ext))] = s_tot
    out = np.zeros(omega.size, dtype=complex)
    for (kind, gap), wgt in pole_w.items():
        if kind == "dp":
            out += wgt / (omega - gap + 1j * eta) ** 2
        else:
            out += wgt / (omega - gap + 1j * eta)
    return out


def _one_phonon(pa, pb, q_ext):
    diff = tuple(b - a for a, b in zip(pa, pb))
    return diff == tuple(
        1 if k == q_ext else 0 for k in range(len(pa))
    ) or diff == tuple(-1 if k == q_ext else 0 for k in range(len(pa)))


def _one_hop(ma, mb):
    total = sum(b - a for a, b in zip(ma, mb))
    nz = [(i, b - a) for i, (a, b) in enumerate(zip(ma, mb)) if a != b]
    return total == 0 and len(nz) == 2 and sorted(d for _, d in nz) == [-1, 1]


class TestLehmannReference:
    def test_magnon_kernel_matches_truncated_fock(self):
        couplings = _couplings()
        k_ext = 1
        evals, _, _ = _ring_hamiltonian(couplings)
        omega = _safe_omega(evals, [0.03, 0.12, 0.14, 0.19, 0.28, 0.34, 0.44])
        eta = 1e-3
        g, eps_kq, om_q = _kernel_magnon_input(k_ext, couplings, eta)
        sigma = magnon_self_energy(
            g,
            eps_kq,
            om_q,
            bose_occupation(om_q, TEMP_K),
            np.full(NMOM, 1.0 / NMOM),
            omega,
            eta,
        )
        evals, evecs, sector = _ring_hamiltonian(couplings)
        g_ref = _ring_greens(evals, evecs, sector, "magnon", k_ext, omega, eta, TEMP_K)
        g0 = _truncated_bare(couplings, "magnon", k_ext, omega, eta)
        sigma_ref = _extract_sigma(g_ref - g0, omega, _magnon_band(k_ext), eta)
        np.testing.assert_allclose(sigma[:, 0, 0], sigma_ref, rtol=2e-1, atol=5e-6)

    def test_truncation_convergence(self):
        couplings = _couplings(7)
        evals, _, _ = _ring_hamiltonian(couplings)
        omega = _safe_omega(evals, [0.03, 0.12, 0.14, 0.19, 0.28, 0.34])
        eta = 1e-3
        k_ext = 2
        g, eps_kq, om_q = _kernel_magnon_input(k_ext, couplings, eta)
        sigma = magnon_self_energy(
            g,
            eps_kq,
            om_q,
            bose_occupation(om_q, TEMP_K),
            np.full(NMOM, 1.0 / NMOM),
            omega,
            eta,
        )
        refs = []
        for n_ph_max in (2, 3):
            evals, evecs, sector = _ring_hamiltonian(
                couplings, n_mag_max=2, n_ph_max=n_ph_max
            )
            gf = _ring_greens(evals, evecs, sector, "magnon", k_ext, omega, eta, TEMP_K)
            g0 = _truncated_bare(
                couplings, "magnon", k_ext, omega, eta, n_ph_max=n_ph_max
            )
            refs.append(_extract_sigma(gf - g0, omega, _magnon_band(k_ext), eta))
        np.testing.assert_allclose(refs[0], refs[1], atol=5e-6)
        np.testing.assert_allclose(sigma[:, 0, 0], refs[1], rtol=2e-1, atol=5e-6)

    def test_phonon_kernel_matches_truncated_fock(self):
        """Seam phonon hot-band kernel == truncated-Fock Lehmann
        intermediate-state sum (bare-pole one-shot Born) on the ring.

        The oracle evaluates, on the truncated product Fock space of the
        ring model at finite T, the second-order Lehmann correction of the
        coordinate Green function x_q = (b_q + b^dag_q)/sqrt(2 Omega_q):

        - double-pole restructuring of the bare +-Omega_q pairs with the
          per-configuration energy shifts dE (single-dressing);
        - new simple poles at omega = +- deps_k (hop k -> k+q) with the
          dressed coordinate amplitudes xtilde (single-dressing).

        The seam kernel must reproduce this oracle exactly (up to Fock
        truncation of the phonon/magnon ladders).
        """
        couplings = _couplings(3)
        q_ext = 2
        eta = 1e-4
        omega = np.array([0.02, 0.05, 0.12, 0.18, 0.26, 0.33, 0.40])
        g = np.zeros((NMOM, 1, 1, 1), dtype=complex)
        eps_k = np.zeros((NMOM, 1))
        eps_kq = np.zeros((NMOM, 1))
        for k in range(NMOM):
            g[k, 0, 0, 0] = np.sqrt(NMOM) * G_SCALE_PH * couplings[k, q_ext]
            eps_k[k, 0] = _magnon_band(k)
            eps_kq[k, 0] = _magnon_band((k + q_ext) % NMOM)
        om_q = np.array([_phonon_band(q_ext)])

        from TB2J.magnon_phonon.scattering import phonon_hot_band_kernel

        delta_d, _ = phonon_hot_band_kernel(
            g,
            eps_k,
            eps_kq,
            bose_occupation(eps_k, TEMP_K),
            bose_occupation(eps_kq, TEMP_K),
            om_q,
            np.full(NMOM, 1.0 / NMOM),
            omega,
            eta,
        )
        oracle = _phonon_bare_pole_oracle(
            couplings, q_ext, 2, omega, eta, eps_k, eps_kq
        )
        seam = delta_d[:, 0]
        # the oracle pole sum is the exact same object the seam encodes
        np.testing.assert_allclose(seam, oracle, rtol=1e-5, atol=1e-6)

    def test_phonon_oracle_truncation_convergence(self):
        """The phonon Lehmann oracle is converged in the phonon ladder
        cutoff, and the seam matches the converged reference."""
        couplings = _couplings(3)
        q_ext = 2
        eta = 1e-4
        omega = np.array([0.02, 0.05, 0.12, 0.18, 0.26, 0.33, 0.40])
        g = np.zeros((NMOM, 1, 1, 1), dtype=complex)
        eps_k = np.zeros((NMOM, 1))
        eps_kq = np.zeros((NMOM, 1))
        for k in range(NMOM):
            g[k, 0, 0, 0] = np.sqrt(NMOM) * G_SCALE_PH * couplings[k, q_ext]
            eps_k[k, 0] = _magnon_band(k)
            eps_kq[k, 0] = _magnon_band((k + q_ext) % NMOM)
        refs = [
            _phonon_bare_pole_oracle(couplings, q_ext, n_ph, omega, eta, eps_k, eps_kq)
            for n_ph in (2, 3)
        ]
        np.testing.assert_allclose(refs[0], refs[1], rtol=1e-5, atol=1e-6)
        om_q = np.array([_phonon_band(q_ext)])

        from TB2J.magnon_phonon.scattering import phonon_hot_band_kernel

        delta_d, _ = phonon_hot_band_kernel(
            g,
            eps_k,
            eps_kq,
            bose_occupation(eps_k, TEMP_K),
            bose_occupation(eps_kq, TEMP_K),
            om_q,
            np.full(NMOM, 1.0 / NMOM),
            omega,
            eta,
        )
        np.testing.assert_allclose(delta_d[:, 0], refs[1], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# seam-level tests on the spring model (Story 009 fixture)
# ---------------------------------------------------------------------------
def _spring_model(lambda_scale=1.0, q=1 / 3):
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    rows = _pattern_rows(q, vec, zpa) * lambda_scale
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    phonon = _phonon_input((q, 0.0, 0.0), vec)
    return model, phonon, q


def _spring_request(phonon, q, species=None):
    vec, masses, _ = _pattern()
    mesh_q = [q]
    mesh_phonons = (
        _phonon_input(
            (q, 0.0, 0.0), vec, frequency_thz=float(phonon.frequencies_thz[0])
        ),
    )
    return ScatteringRequest(
        omega_eV=np.linspace(1e-4, 0.12, 61),
        temperature_K=100.0,
        broadening_eV=2e-3,
        magnon_kpoints=np.array([[0.2, 0.0, 0.0], [0.35, 0.0, 0.0]]),
        phonon_qpoints=np.array([[q, 0.0, 0.0]]),
        phonon_modes=(phonon,),
        magnon_mesh_qpoints=np.array([[mq, 0.0, 0.0] for mq in mesh_q]),
        magnon_mesh_weights=np.full(1, 1.0),
        magnon_phonon_modes=mesh_phonons,
        phonon_mesh_kpoints=np.array([[0.125, 0.0, 0.0], [0.375, 0.0, 0.0]]),
        phonon_mesh_weights=np.full(2, 0.5),
        species=species,
    )


class TestZeroCouplingRecovery:
    def test_g_zero_exact_bare_recovery(self):
        model, phonon, q = _spring_model(lambda_scale=0.0)
        result = model.scattering_spectra(_spring_request(phonon, q))
        np.testing.assert_allclose(result.magnon.sigma, 0.0, atol=0.0)
        np.testing.assert_allclose(result.phonon.sigma, 0.0, atol=0.0)
        w = _spring_request(phonon, q).omega_eV
        d0 = 1.0 / (
            (w + 1j * result.broadening_eV) ** 2
            - result.phonon.bare_energies[0, 0] ** 2
        )
        np.testing.assert_allclose(result.phonon.green[0, :, 0, 0], d0)
        omega = _spring_request(phonon, q).omega_eV
        eps = result.magnon.bare_energies[0, 0]
        g_bare = 1.0 / (omega + 1j * result.broadening_eV - eps)
        np.testing.assert_allclose(result.magnon.green[0, :, 0, 0], g_bare)
        np.testing.assert_allclose(result.magnon.spectral[0, :, 0, 0], -2 * g_bare.imag)
        np.testing.assert_allclose(result.magnon.linewidths, 0.0, atol=0.0)

    def test_quadratic_coupling_scaling(self):
        model1, phonon, q = _spring_model(lambda_scale=1e-2)
        model2, _, _ = _spring_model(lambda_scale=2e-2)
        request = _spring_request(phonon, q)
        r1 = model1.scattering_spectra(request)
        r2 = model2.scattering_spectra(request)
        mask = ~np.isclose(r1.magnon.sigma, 0.0, atol=1e-18)
        ratio = r2.magnon.sigma[mask] / r1.magnon.sigma[mask]
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-9)


class TestForbiddenControls:
    def test_symmetry_zero_vertex_no_damping(self):
        # symmetry-forbidden coupling (vanishing projected rows): spectra
        # equal the bare ones, no spurious linewidth anywhere
        model, phonon, q = _spring_model(lambda_scale=0.0)
        result = model.scattering_spectra(_spring_request(phonon, q))
        np.testing.assert_allclose(result.magnon.linewidths, 0.0, atol=0.0)
        np.testing.assert_allclose(result.phonon.linewidths, 0.0, atol=0.0)

    def test_below_threshold_im_is_broadening_only(self):
        # at T = 0 and omega below all eps_{k+q} + Omega thresholds, the
        # absorptive part scales linearly with eta: no physical damping
        model, phonon, q = _spring_model(lambda_scale=1.0)
        omega = np.linspace(1e-4, 0.02, 40)  # well below thresholds
        results = []
        for eta in (2e-3, 1e-3):
            request = ScatteringRequest(
                omega_eV=omega,
                temperature_K=0.0,
                broadening_eV=eta,
                magnon_kpoints=np.array([[0.2, 0.0, 0.0]]),
                phonon_qpoints=np.array([[q, 0.0, 0.0]]),
                phonon_modes=(phonon,),
                magnon_mesh_qpoints=np.array([[q, 0.0, 0.0]]),
                magnon_mesh_weights=np.full(1, 1.0),
                magnon_phonon_modes=(phonon,),
                phonon_mesh_kpoints=np.array([[0.125, 0.0, 0.0]]),
                phonon_mesh_weights=np.full(1, 1.0),
                species=("magnon",),
            )
            results.append(model.scattering_spectra(request))
        im1 = results[0].magnon.sigma[0, :, 0, 0].imag
        im2 = results[1].magnon.sigma[0, :, 0, 0].imag
        np.testing.assert_allclose(im1 / im2, 2.0, rtol=1e-6)

    def test_degenerate_matrix_structure_kept(self):
        """Two degenerate phonon branches: the one-shot correction is
        branch-diagonal (each self-energy attaches to a single propagating
        line) and each diagonal entry equals the single-branch kernel; no
        spurious off-diagonal damping appears in the degenerate subspace."""
        eps_k = np.array([[0.08], [0.155], [0.2]])
        eps_kq = eps_k + np.array([[0.12], [0.03], [0.06]])
        om_q = np.array([0.245, 0.245])  # degenerate pair
        g = np.zeros((3, 1, 1, 2), dtype=complex)
        g[:, 0, 0, 0] = np.sqrt(NMOM) * G_SCALE_PH * 0.4
        g[:, 0, 0, 1] = np.sqrt(NMOM) * G_SCALE_PH * 0.4
        eta = 1e-4
        omega = np.array([0.05, 0.12, 0.2, 0.33])

        from TB2J.magnon_phonon.scattering import phonon_hot_band_kernel

        n_in = bose_occupation(eps_k, TEMP_K)
        n_out = bose_occupation(eps_kq, TEMP_K)
        dd, _ = phonon_hot_band_kernel(
            g,
            eps_k,
            eps_kq,
            n_in,
            n_out,
            om_q,
            np.full(3, 1.0 / 3),
            omega,
            eta,
        )
        single, _ = phonon_hot_band_kernel(
            g[:, :, :, :1],
            eps_k,
            eps_kq,
            n_in,
            n_out,
            om_q[:1],
            np.full(3, 1.0 / 3),
            omega,
            eta,
        )
        assert dd.shape == (omega.size, 2)
        np.testing.assert_allclose(dd[:, 0], dd[:, 1], rtol=1e-12)
        np.testing.assert_allclose(dd[:, 0], single[:, 0], rtol=1e-12)


class TestTemperatureLimits:
    def test_t_zero_emission_only(self):
        g = np.array([[[[0.05 + 0.02j]]]])
        eps_kq = np.array([[0.09]])
        om_q = np.array([[0.03]])
        omega = np.array([0.1, 0.15])
        s0 = magnon_self_energy(
            g,
            eps_kq,
            om_q,
            bose_occupation(om_q, 0.0),
            np.array([1.0]),
            omega,
            1e-3,
        )
        expected = np.abs(g[0, 0, 0, 0]) ** 2 / (omega - 0.09 - 0.03 + 1j * 1e-3)
        np.testing.assert_allclose(s0[:, 0, 0], expected, rtol=1e-12)

    def test_finite_t_bose_weights(self):
        g = np.array([[[[0.05 + 0.02j]]]])
        eps_kq = np.array([[0.09]])
        om_q = np.array([[0.03]])
        omega = np.array([0.1, 0.15])
        temperature = 400.0
        n_bose = 1.0 / (np.exp(om_q[0, 0] / (KB_EV_K * temperature)) - 1.0)
        s = magnon_self_energy(
            g,
            eps_kq,
            om_q,
            bose_occupation(om_q, temperature),
            np.array([1.0]),
            omega,
            1e-3,
        )
        g2 = np.abs(g[0, 0, 0, 0]) ** 2
        expected = g2 * (
            (n_bose + 1) / (omega - 0.09 - 0.03 + 1j * 1e-3)
            + n_bose / (omega - 0.09 + 0.03 + 1j * 1e-3)
        )
        np.testing.assert_allclose(s[:, 0, 0], expected, rtol=1e-12)

    def test_negative_temperature_rejected(self):
        with pytest.raises(ValueError):
            bose_occupation(np.array([0.05]), -1.0)


class TestMetadataAndLinewidths:
    def test_broadening_and_grid_reported_separately(self):
        model, phonon, q = _spring_model(lambda_scale=1.0)
        request = _spring_request(phonon, q)
        result = model.scattering_spectra(request)
        meta = result.metadata
        assert meta["broadening_eV"] == request.broadening_eV
        assert meta["omega_grid_eV"].shape == request.omega_eV.shape
        assert np.all(np.diff(meta["omega_grid_eV"]) > 0)
        np.testing.assert_allclose(meta["magnon_mesh_weights"].sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(meta["phonon_mesh_weights"].sum(), 1.0, atol=1e-12)
        # linewidths are a separate array: -Im Sigma at the bare pole
        for i_k in range(result.magnon.sigma.shape[0]):
            for alpha in range(result.magnon.sigma.shape[-1]):
                eps = result.magnon.bare_energies[i_k, alpha]
                i_w = int(np.argmin(np.abs(request.omega_eV - eps)))
                expected = -result.magnon.sigma[i_k, i_w, alpha, alpha].imag
                np.testing.assert_allclose(
                    result.magnon.linewidths[i_k, alpha], expected
                )
        assert meta["static_tadpole_excluded"] is True
        assert "one-shot O(g^2) Born" in meta["approximation"]


# ---------------------------------------------------------------------------
# round-1 review fixes: contraction, seam availability, self-energy and
# grid-error reporting, CLI
# ---------------------------------------------------------------------------
class TestMultibranchKernelContraction:
    def test_multibranch_grid_length_differs_from_branch_count(self):
        """The magnon kernel contracts the phonon BRANCH index between
        both g operands while the frequency stays a distinct free output
        axis; the internal-mesh length (3) differs from the branch count
        (2) and the omega grid (5 points) differs from both, so the old
        contraction (which shared one einsum label between the frequency
        axis and the conjugated branch index) cannot even evaluate."""
        rng = np.random.default_rng(21)
        nq, nu = 3, 2
        g = rng.normal(size=(nq, 2, 2, nu)) + 1j * rng.normal(size=(nq, 2, 2, nu))
        eps_kq = np.array([[0.11, 0.19], [0.13, 0.17], [0.12, 0.18]])
        om_q = np.array([[0.05, 0.07], [0.06, 0.08], [0.045, 0.075]])
        n_bose = np.full((nq, nu), 0.3)
        weights = np.full(nq, 1.0 / nq)
        omega = np.linspace(0.05, 0.3, 5)
        eta = 1e-3
        sigma = magnon_self_energy(g, eps_kq, om_q, n_bose, weights, omega, eta)
        assert sigma.shape == (omega.size, 2, 2)
        # explicit loop reference
        kernel = (n_bose[:, None, :, None] + 1.0) / (
            omega[None, None, None, :]
            - eps_kq[:, :, None, None]
            - om_q[:, None, :, None]
            + 1j * eta
        ) + n_bose[:, None, :, None] / (
            omega[None, None, None, :]
            - eps_kq[:, :, None, None]
            + om_q[:, None, :, None]
            + 1j * eta
        )  # (nq, m, nu, nw)
        ref = np.zeros((omega.size, 2, 2), dtype=complex)
        for iq in range(nq):
            for im in range(2):
                for inu in range(nu):
                    amp = weights[iq] * kernel[iq, im, inu, :]
                    ref += amp[:, None, None] * np.outer(
                        g[iq, im, :, inu], np.conj(g[iq, im, :, inu])
                    )
        np.testing.assert_allclose(sigma, ref, rtol=1e-12)

    def test_branch_contribution_is_not_dropped(self):
        """With two inequivalent branches the kernel equals the SUM over
        branches; dropping either branch (a separate-summation failure
        mode) breaks the comparison."""
        g = np.zeros((1, 1, 1, 2), dtype=complex)
        g[0, 0, 0, 0] = 0.1
        g[0, 0, 0, 1] = 0.2
        eps_kq = np.array([[0.1]])
        om_q = np.array([[0.04, 0.06]])
        n_bose = np.array([[0.0, 0.0]])
        omega = np.array([0.2, 0.3])
        sigma = magnon_self_energy(
            g, eps_kq, om_q, n_bose, np.array([1.0]), omega, 1e-3
        )
        expected = 0.1**2 / (omega - 0.1 - 0.04 + 1j * 1e-3) + 0.2**2 / (
            omega - 0.1 - 0.06 + 1j * 1e-3
        )
        np.testing.assert_allclose(sigma[:, 0, 0], expected, rtol=1e-12)


class TestSeamAndReporting:
    def test_seam_method_exists_on_bare_import(self):
        """STD-005: importing the package alone binds the scattering seam;
        the method always exists on MagnonPhononModel."""
        import importlib

        import TB2J.magnon_phonon as pkg

        importlib.reload(pkg)
        from TB2J.magnon_phonon.vertex import MagnonPhononModel

        assert callable(getattr(MagnonPhononModel, "scattering_spectra", None))

    def test_phonon_sigma_is_dyson_self_energy_not_delta_d(self):
        """SPEC-002: sigma is the Dyson-partitioned self-energy
        D0^-1 deltaD D0^-1; green carries the additive delta D on D0;
        linewidths come from sigma."""
        model, phonon, q = _spring_model(lambda_scale=1.0)
        request = _spring_request(phonon, q)
        result = model.scattering_spectra(request)
        spec = result.phonon
        omega = request.omega_eV
        eta = request.broadening_eV
        om0 = spec.bare_energies[0, 0]
        # D0^-1 = (omega + i eta)^2 - om0^2, written from the definition of
        # the bare oscillator propagator, not by inverting d0
        d0_inv = (omega + 1j * eta) ** 2 - om0**2
        delta_d = spec.green[0, :, 0, 0] - 1.0 / d0_inv
        np.testing.assert_allclose(
            spec.sigma[0, :, 0, 0], d0_inv * delta_d * d0_inv, rtol=1e-5, atol=1e-9
        )
        i_w = int(np.argmin(np.abs(omega - om0)))
        np.testing.assert_allclose(
            spec.linewidths[0, 0], -spec.sigma[0, i_w, 0, 0].imag, rtol=1e-12
        )
        # the additive correction is not what the linewidths field reports
        assert not np.allclose(spec.linewidths[0, 0], -delta_d[i_w].imag)

    def test_grid_error_reported_separately_from_linewidths(self):
        """SPEC-004: a real nearest-grid sampling offset is reported per
        on-shell evaluation, separate from the linewidths, and mirrored in
        metadata."""
        model, phonon, q = _spring_model(lambda_scale=1.0)
        request = _spring_request(phonon, q)
        result = model.scattering_spectra(request)
        omega = request.omega_eV
        for spec in (result.magnon, result.phonon):
            n_ext, n_modes = spec.bare_energies.shape
            assert spec.grid_error.shape == (n_ext, n_modes)
            for i in range(n_ext):
                for a in range(n_modes):
                    eps = spec.bare_energies[i, a]
                    expected = abs(eps - omega[int(np.argmin(np.abs(omega - eps)))])
                    np.testing.assert_allclose(spec.grid_error[i, a], expected)
            assert np.all(spec.grid_error >= 0.0)
        # interior poles sample the grid to at most one grid step; exterior
        # poles are allowed to sit outside the reported window
        omega = request.omega_eV
        spacing = np.max(np.diff(omega))
        for spec in (result.magnon, result.phonon):
            inside = (spec.bare_energies >= omega.min()) & (
                spec.bare_energies <= omega.max()
            )
            assert np.all(spec.grid_error[inside] <= spacing)
        grid_meta = result.metadata["grid_error"]
        np.testing.assert_allclose(
            grid_meta["magnon_max_offset_eV"], np.max(result.magnon.grid_error)
        )
        np.testing.assert_allclose(
            grid_meta["phonon_max_offset_eV"], np.max(result.phonon.grid_error)
        )
        assert "sampling offset" in grid_meta["convention"]


class TestCLI:
    @staticmethod
    def _manifest(path):
        import json

        manifest = {
            "calculation": "scattering",
            "omega_eV": [0.1, 0.2, 0.3],
            "broadening_eV": 1e-3,
            "temperature_K": 150.0,
            "magnon": {
                "g": [[[[0.05]]]],
                "eps_kq": [[0.09]],
                "omega_q_ev": [[0.03]],
                "weights": [1.0],
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        return manifest

    def test_help_lists_scattering(self, capsys):
        from TB2J.magnon_phonon import cli

        try:
            cli.main(["--help"])
        except SystemExit:
            pass
        assert "scattering" in capsys.readouterr().out

    def test_scattering_manifest_run(self, tmp_path):
        import json

        from TB2J.magnon_phonon import cli

        path = tmp_path / "manifest.json"
        self._manifest(path)
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli.main(["scattering", str(path)])
        result = json.loads(buffer.getvalue())
        assert result["calculation"] == "scattering"
        sigma = result["magnon"]["sigma"]
        assert sigma["real"][0][0][0] != 0.0 or sigma["imag"][0][0][0] != 0.0
        assert result["metadata"]["approximation"]

    def test_scattering_cli_module_help(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "TB2J.magnon_phonon.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0
        assert "scattering" in proc.stdout
