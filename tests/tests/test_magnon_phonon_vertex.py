"""Consumer-facing tests for the explicit magnon-phonon vertex (Story 009).

Acceptance criteria:
- TEST-001: finite spin/lattice model derivative comparison in the
  weak-displacement limit (mechanical real-space reference, independent code);
- TEST-002: forbidden scalar-collinear bilinear mixing is zero;
- TEST-003: vertex Hermiticity across sampled k, q (including anomalous v≠0);
- TEST-004: degenerate-subspace rephasing invariance of contracted quantities.
"""

import hashlib
import json

import numpy as np
import pytest

from TB2J.magnon.eigenstates import SiteBosonDiagnostics, SiteBosonModes
from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon_phonon.loaders import as_phonon_mode_input, load_matched_rows
from TB2J.magnon_phonon.vertex import (
    MagnonPhononModel,
    ModeProjectedDerivatives,
    OrientedBondDerivatives,
    PhononModeInput,
    _projected_mode_zero_point,
    _zero_point_amplitudes,
)

HBAR = 1.054571817e-34
AMU = 1.66053906660e-27


def _colpa(H):
    """Independent bosonic Colpa diagonalization (test-local)."""
    n = H.shape[0] // 2
    K = np.linalg.cholesky(H)
    g = np.diag(np.r_[np.ones(n), -np.ones(n)])
    evals, evecs = np.linalg.eigh(K.conj().T @ g @ K)
    physical = np.linalg.solve(K.conj().T, evecs) * np.sqrt(np.abs(evals))[None, :]
    T = np.concatenate((physical[:, n:], physical[:, :n][:, ::-1]), axis=1)
    return T, evals[n:]


# --------------------------------------------------------------------------
# finite classical spin/lattice model: 1D chain, two atoms per cell,
# exponential bond springs (nonlinear J(tau)), one inert on-site bond.
# --------------------------------------------------------------------------
A_CELL = 2.4
XCELL = np.array([0.0, 1.15])
SPINS = np.array([1.2, 0.9])
BONDS = [
    # (i, j, R, j0, beta); J(L) = j0 exp(-beta (len(L) - len0)); beta 0: inert
    (0, 1, 0, 0.9, 1.7),
    (1, 0, 0, 0.9, 1.7),
    (0, 1, 1, 0.55, 1.1),
    (1, 0, -1, 0.55, 1.1),
    (0, 0, 0, 0.30, 0.0),
    (1, 1, 0, 0.20, 0.0),
]
NSITES = 2


def _len0(bi):
    i, j, R, j0, beta = BONDS[bi]
    if beta == 0.0:
        return 1.0
    return abs((XCELL[j] + R * A_CELL) - XCELL[i])


def _spring_J(bi, tau_i, tau_j):
    i, j, R, j0, beta = BONDS[bi]
    if beta == 0.0:
        return j0
    length = abs((XCELL[j] + R * A_CELL + tau_j) - (XCELL[i] + tau_i))
    return j0 * np.exp(-beta * (length - _len0(bi)))


def _dJ_endpoint(bi):
    """Unit-direction dJ/dtau (x axis) at the i and j endpoints."""
    i, j, R, j0, beta = BONDS[bi]
    if beta == 0.0:
        return 0.0, 0.0
    dhat = ((XCELL[j] + R * A_CELL) - XCELL[i]) / _len0(bi)
    return +j0 * beta * dhat, -j0 * beta * dhat


def _spring_magnon():
    """The spring model as a TB2J Magnon (full-moment JR convention)."""
    Rlist = np.array([(b[2], 0, 0) for b in BONDS], dtype=int)
    JR = np.zeros((len(BONDS), NSITES, NSITES, 3, 3))
    for ir, (i, j, R, j0, beta) in enumerate(BONDS):
        JR[ir, i, j] = np.eye(3) * j0 * SPINS[i] * SPINS[j]
    mags = np.zeros((NSITES, 3))
    mags[:, 2] = 2.0 * SPINS
    magnon = Magnon(
        nspin=NSITES,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([A_CELL, 10.0, 10.0]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


def _pattern(frequency_thz=5.3, seed=9):
    rng = np.random.default_rng(seed)
    vec = np.linalg.qr(rng.normal(size=(NSITES, NSITES)))[0][:, 0]
    masses = np.array([1.3, 0.7])
    omega = 2 * np.pi * frequency_thz * 1e12
    zpa = np.sqrt(HBAR / (2 * masses * AMU * omega)) * 1e10
    return vec, masses, zpa


def _phonon_input(q, vec, frequency_thz=5.3, masses=None, nmodes=1):
    masses = np.array([1.3, 0.7]) if masses is None else masses
    eig = np.zeros((3 * NSITES, nmodes), dtype=complex)
    eig[0::3, 0] = vec
    return PhononModeInput(
        q=np.asarray(q, dtype=float),
        frequencies_thz=np.full(nmodes, frequency_thz),
        masses_amu=masses,
        eigenvectors=eig,
    )


def _pattern_rows(q, vec, zpa, nmodes=1):
    """Pattern-projected full-moment rows (eV) at momentum q, from the springs."""
    rows = np.zeros((len(BONDS), nmodes), dtype=complex)
    for bi, (i, j, R, j0, beta) in enumerate(BONDS):
        di, dj = _dJ_endpoint(bi)
        unit = zpa[i] * di * vec[i] + np.exp(2j * np.pi * q * R) * zpa[j] * dj * vec[j]
        rows[bi, 0] = unit * SPINS[i] * SPINS[j]  # full-moment convention
    return rows


def _bonds_container(q, rows, convention="pattern_projected"):
    return OrientedBondDerivatives(
        i=np.array([b[0] for b in BONDS], dtype=int),
        j=np.array([b[1] for b in BONDS], dtype=int),
        R=np.array([(b[2], 0, 0) for b in BONDS], dtype=int),
        dJ_dtau=rows,
        convention=convention,
        source_q=np.array([q, 0.0, 0.0]) if convention == "pattern_projected" else None,
    )


# --------------------------------------------------------------------------
# independent mechanical reference (real-space assembly + two-momentum Bloch)
# --------------------------------------------------------------------------
class MechanicalReference:
    """Mechanical real-space reference (mirrors the asserted derivation-script
    structure: full real-space bilinear, two-momentum Bloch sums with the
    1/N of the Bloch convention, independent Colpa, u/w projection)."""

    NCELL = 6

    def __init__(self):
        Uv = np.array([1.0, 1.0j, 0.0])
        self.t = [
            np.sqrt(SPINS[i] * SPINS[j]) / 2 * np.sum(Uv * np.conj(Uv))
            for i, j, R, j0, beta in BONDS
        ]
        self.s = [
            np.sqrt(SPINS[i] * SPINS[j]) / 2 * np.sum(np.conj(Uv) * np.conj(Uv))
            for i, j, R, j0, beta in BONDS
        ]
        assert max(abs(s) for s in self.s) < 1e-14

    def static_hessian(self, k):
        A = np.zeros((NSITES, NSITES), dtype=complex)
        B = np.zeros((NSITES, NSITES), dtype=complex)
        for bi, (i, j, R, j0, beta) in enumerate(BONDS):
            Jb = _spring_J(bi, 0.0, 0.0)
            ph = np.exp(2j * np.pi * k * R)
            A[i, j] += -Jb * self.t[bi] * ph
            A[j, i] += -Jb * np.conj(self.t[bi]) * np.conj(ph)
            A[i, i] += Jb * SPINS[j]
            A[j, j] += Jb * SPINS[i]
            B[i, j] += -Jb * self.s[bi] * ph
        return np.block([[A, B], [B.conj(), A.conj()]])

    def modulated_M1(self, q, disp, h, field):
        """Full real-space a^dag a matrix for a REAL cos/sin displacement field."""
        trig = np.cos if field == "cos" else np.sin
        M1 = np.zeros((self.NCELL * NSITES, self.NCELL * NSITES), dtype=complex)
        for L in range(self.NCELL):
            for bi, (i, j, R, j0, beta) in enumerate(BONDS):
                tau_i = h * disp[i] * trig(2 * np.pi * q * L)
                tau_j = h * disp[j] * trig(2 * np.pi * q * (L + R))
                Jb = _spring_J(bi, tau_i, tau_j)
                x = L * NSITES + i
                y = ((L + R) % self.NCELL) * NSITES + j
                M1[x, y] += -Jb * self.t[bi]
                M1[y, x] += -Jb * np.conj(self.t[bi])
                M1[x, x] += Jb * SPINS[j]
                M1[y, y] += Jb * SPINS[i]
        return M1

    def bloch2(self, M, p1, p2):
        out = np.zeros((NSITES, NSITES), dtype=complex)
        for Lx in range(self.NCELL):
            for Ly in range(self.NCELL):
                ph = np.exp(-2j * np.pi * p1 * Lx) * np.exp(2j * np.pi * p2 * Ly)
                out += (
                    ph
                    * M[
                        Lx * NSITES : (Lx + 1) * NSITES,
                        Ly * NSITES : (Ly + 1) * NSITES,
                    ]
                )
        return out / self.NCELL

    def _delta_m1(self, q, disp, h):
        dcos = (
            self.modulated_M1(q, disp, +h, "cos")
            - self.modulated_M1(q, disp, -h, "cos")
        ) / (2 * h)
        dsin = (
            self.modulated_M1(q, disp, +h, "sin")
            - self.modulated_M1(q, disp, -h, "sin")
        ) / (2 * h)
        return dcos + 1j * dsin

    def vertex(self, k, q, disp, h):
        kp, mk, mkp = (k + q) % 1, (-k) % 1, (-(k + q)) % 1
        dM1 = self._delta_m1(q, disp, h)
        M_kp_k = self.bloch2(dM1, kp, k)
        M_mk_mkp = self.bloch2(dM1, mk, mkp)
        T = {p: _colpa(self.static_hessian(p))[0] for p in (kp, k, mk, mkp)}
        u_kp, u_k = T[kp][:NSITES, :NSITES].T, T[k][:NSITES, :NSITES].T
        w_mk, w_mkp = T[mk][NSITES:, :NSITES].T, T[mkp][NSITES:, :NSITES].T
        g = np.zeros((NSITES, NSITES), dtype=complex)
        for mu in range(NSITES):
            for nu in range(NSITES):
                val = 0j
                for ci in range(NSITES):
                    for cj in range(NSITES):
                        val += np.conj(u_kp[mu, ci]) * M_kp_k[ci, cj] * u_k[nu, cj]
                        val += np.conj(w_mk[nu, ci]) * M_mk_mkp[ci, cj] * w_mkp[mu, cj]
                g[mu, nu] = val
        return g

    def mixed_blocks(self, k, q, disp, h):
        """Independent projections of the modulated bilinear at (k+q, -(k+q)).

        Returns ``(g_allowed, cross_uw, cross_wu)``: the number-conserving
        ``gamma^dag_{k+q} gamma_k`` projection (identical to :meth:`vertex`)
        and the u-w / w-u cross contractions at the conjugate pairing
        ``(k+q, -(k+q))``, i.e. the would-be bilinear mixing block that the
        scalar collinear coupling must not generate.
        """
        kp, mkp = (k + q) % 1, (-(k + q)) % 1
        dM1 = self._delta_m1(q, disp, h)
        M = self.bloch2(dM1, kp, mkp)
        T_kp = _colpa(self.static_hessian(kp))[0]
        T_mkp = _colpa(self.static_hessian(mkp))[0]
        u_kp, w_kp = T_kp[:NSITES, :NSITES].T, T_kp[NSITES:, :NSITES].T
        u_mkp, w_mkp = T_mkp[:NSITES, :NSITES].T, T_mkp[NSITES:, :NSITES].T
        allowed = self.vertex(k, q, disp, h)
        cross_uw = np.conj(u_kp) @ M @ np.conj(w_mkp).T
        cross_wu = np.conj(w_kp) @ M @ np.conj(u_mkp).T
        return allowed, cross_uw, cross_wu


# --------------------------------------------------------------------------
# TEST-001: finite-model derivative comparison
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "k,q", [(1 / 6, 1 / 3), (1 / 3, 1 / 6), (1 / 6, 1 / 6), (1 / 6, 2 / 3)]
)
def test_finite_model_derivative_comparison(k, q):
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    rows = _pattern_rows(q, vec, zpa)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    phonon = _phonon_input((q, 0.0, 0.0), vec)
    result = model.vertex((k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon)
    reference = MechanicalReference().vertex(k, q, vec * zpa, 1e-4)
    np.testing.assert_allclose(result.g[:, :, 0], reference, atol=1e-8)


def test_weak_displacement_limit_amplitude_independence():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rows = _pattern_rows(q, vec, zpa)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    phonon = _phonon_input((q, 0.0, 0.0), vec)
    g = model.vertex((1 / 6, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon).g
    # the projected vertex is the exact linear coefficient: independent of the
    # reference FD amplitude used by the mechanical comparison
    reference_small = MechanicalReference().vertex(1 / 6, q, vec * zpa, 1e-5)
    reference_large = MechanicalReference().vertex(1 / 6, q, vec * zpa, 1e-4)
    np.testing.assert_allclose(reference_small, reference_large, atol=1e-9)
    np.testing.assert_allclose(g[:, :, 0], reference_small, atol=1e-8)


def test_on_site_bond_rows_have_identically_zero_vertex():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 0.0
    rows = _pattern_rows(q, vec, zpa)
    # keep only the inert on-site bonds' rows
    on_site = np.zeros_like(rows)
    for bi, (i, j, R, j0, beta) in enumerate(BONDS):
        if i == j:
            on_site[bi] = rows[bi]
    model = MagnonPhononModel(magnon, _bonds_container(0.0, on_site))
    result = model.vertex(
        (1 / 6, 0.0, 0.0), (0.0, 0.0, 0.0), phonon_modes=_phonon_input((0, 0, 0), vec)
    )
    np.testing.assert_allclose(result.g, 0.0, atol=1e-14)


# --------------------------------------------------------------------------
# TEST-002: forbidden scalar-collinear bilinear mixing
# --------------------------------------------------------------------------
def test_noncollinear_reference_is_rejected():
    magnon = _spring_magnon()
    mags = np.array(
        [
            [0.0, 0.0, 2.0 * SPINS[0]],
            [2.0 * SPINS[1] * np.sin(0.3), 0.0, 2.0 * SPINS[1] * np.cos(0.3)],
        ]
    )
    magnon.set_reference(
        np.zeros(3),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.0, 1.0, 0.0]),
        magmoms=mags,
    )
    vec, masses, zpa = _pattern()
    q = 0.0
    rows = _pattern_rows(q, vec, zpa)
    container = _bonds_container(q, rows, convention="home_cell")
    model = MagnonPhononModel(magnon, container)
    with pytest.raises(ValueError, match="[Cc]ollinear"):
        model.vertex(
            (0.2, 0.0, 0.0), (0.0, 0.0, 0.0), phonon_modes=_phonon_input((0, 0, 0), vec)
        )


# --------------------------------------------------------------------------
# TEST-003: Hermiticity across sampled k, q
# --------------------------------------------------------------------------
def _synthetic_rows(rng, n_modes):
    rows = rng.normal(size=(len(BONDS), n_modes)) + 1j * rng.normal(
        size=(len(BONDS), n_modes)
    )
    return rows * SPINS[0]  # full-moment scale; arbitrary classical derivative


def _anomalous_magnon():
    """Two-site FM model with on-site anisotropy -> v != 0 magnon branches."""
    Rlist = np.array([(0, 0, 0), (1, 0, 0), (-1, 0, 0)], dtype=int)
    JR = np.zeros((3, 2, 2, 3, 3))
    JR[0, 0, 0] = np.diag([-2.0, -1.0, 0.0]) * SPINS[0] ** 2
    JR[0, 1, 1] = np.diag([-2.0, -1.0, 0.0]) * SPINS[1] ** 2
    JR[1, 0, 1] = np.eye(3) * 0.4 * SPINS[0] * SPINS[1]
    JR[2, 1, 0] = np.eye(3) * 0.4 * SPINS[0] * SPINS[1]
    mags = np.zeros((2, 3))
    mags[:, 2] = 2.0 * SPINS
    magnon = Magnon(
        nspin=2,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([2.4, 10, 10]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


@pytest.mark.parametrize("magnon_name", ["spring", "anomalous"])
@pytest.mark.parametrize(
    "k,q", [(1 / 6, 1 / 3), (1 / 4, 1 / 4), (1 / 3, 1 / 6), (0.2, 0.35)]
)
def test_vertex_hermiticity_across_sampled_kq(magnon_name, k, q):
    magnon = _spring_magnon() if magnon_name == "spring" else _anomalous_magnon()
    vec, masses, zpa = _pattern()
    n_modes = 2
    rng = np.random.default_rng(7)
    rows_q = _synthetic_rows(rng, n_modes)
    model = MagnonPhononModel(
        magnon,
        _bonds_container(q, rows_q),
    )
    phonon_q = _phonon_input((q, 0, 0), vec, nmodes=n_modes)
    phonon_mq = _phonon_input((-q, 0, 0), vec, nmodes=n_modes)
    container_mq = _bonds_container(-q, np.conj(rows_q))
    model_mq = MagnonPhononModel(magnon, container_mq)
    kp = (k + q) % 1
    if any(abs(p) < 1e-9 for p in (k, kp, -k % 1, (-kp) % 1)):
        pytest.skip("Gamma point sampling is out of scope for the vertex")
    g_plus = model.vertex((k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon_q).g
    g_minus = model_mq.vertex((kp, 0.0, 0.0), (-q, 0.0, 0.0), phonon_modes=phonon_mq).g
    np.testing.assert_allclose(g_plus, np.conj(np.swapaxes(g_minus, 0, 1)), atol=1e-10)


def test_anomalous_model_has_nonzero_hole_amplitudes():
    magnon = _anomalous_magnon()
    modes = magnon.get_site_boson_modes([[0.2, 0.0, 0.0]])
    assert np.max(np.abs(modes.v)) > 0.05


# --------------------------------------------------------------------------
# TEST-004: degenerate-subspace rephasing invariance
# --------------------------------------------------------------------------
def _degenerate_magnon():
    """Two decoupled identical sublattices: exactly doubly degenerate bands."""
    Rlist = np.array([(1, 0, 0), (-1, 0, 0)], dtype=int)
    JR = np.zeros((2, 2, 2, 3, 3))
    JR[0, 0, 0] = np.eye(3) * 0.8 * SPINS[0] ** 2
    JR[1, 0, 0] = np.eye(3) * 0.8 * SPINS[0] ** 2
    JR[0, 1, 1] = np.eye(3) * 0.8 * SPINS[0] ** 2
    JR[1, 1, 1] = np.eye(3) * 0.8 * SPINS[0] ** 2
    mags = np.zeros((2, 3))
    mags[:, 2] = 2.0 * SPINS[0]
    magnon = Magnon(
        nspin=2,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([2.4, 10, 10]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


def _rephase_modes(modes, thetas):
    """u(p) -> u(p) Theta_p^dag, v(p) -> v(p) Theta_{-p}^T (fixed site bosons)."""
    u = modes.u.copy()
    v = modes.v.copy()
    for ik in range(len(u)):
        u[ik] = thetas[ik].conj() @ u[ik]
    # partner momentum index: 0=k+q, 1=k, 2=-k, 3=-(k+q)
    partner = {0: 3, 1: 2, 2: 1, 3: 0}
    for ik in range(len(v)):
        v[ik] = thetas[partner[ik]] @ v[ik]
    from TB2J.magnon.eigenstates import SiteBosonModes

    return SiteBosonModes(
        kpoints=modes.kpoints,
        energies=modes.energies,
        u=u,
        v=v,
        paraunitary=modes.paraunitary,
        spin_lengths=modes.spin_lengths,
        U=modes.U,
        V=modes.V,
        diagnostics=modes.diagnostics,
        is_zero_mode=modes.is_zero_mode,
    )


@pytest.mark.parametrize("k,q", [(1 / 6, 1 / 3), (1 / 4, 1 / 4)])
def test_degenerate_subspace_rephasing_invariance(k, q):
    magnon = _degenerate_magnon()
    vec, masses, zpa = _pattern()
    qv = q
    rng = np.random.default_rng(5)
    rows = _synthetic_rows(rng, 1)
    model = MagnonPhononModel(magnon, _bonds_container(qv, rows))
    phonon = _phonon_input((qv, 0, 0), vec)
    folded = np.array(
        [
            [(k + qv) % 1, 0, 0],
            [k % 1, 0, 0],
            [(-(k + qv)) % 1, 0, 0],
            [(-k) % 1, 0, 0],
        ]
    )
    modes = magnon.get_site_boson_modes(folded)
    energies = modes.energies
    assert np.allclose(energies[:, 0], energies[:, 1]), "branches must be degenerate"
    result = model.vertex(
        (k, 0.0, 0.0), (qv, 0.0, 0.0), phonon_modes=phonon, site_boson_modes=modes
    )
    g0 = result.g

    rng = np.random.default_rng(11)
    thetas = []
    for _ in range(4):
        raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        qmat, _ = np.linalg.qr(raw)
        thetas.append(qmat)
    rephased = _rephase_modes(modes, thetas)
    result2 = model.vertex(
        (k, 0.0, 0.0), (qv, 0.0, 0.0), phonon_modes=phonon, site_boson_modes=rephased
    )
    g1 = result2.g

    g0 = g0[:, :, 0]
    g1 = g1[:, :, 0]
    res0 = np.linalg.eigvalsh(g0 @ g0.conj().T)
    res1 = np.linalg.eigvalsh(g1 @ g1.conj().T)
    np.testing.assert_allclose(res1, res0, atol=1e-10)
    # covariance: g' = Theta_{k+q} g Theta_k^dag
    target = thetas[0] @ g0 @ thetas[1].conj().T
    np.testing.assert_allclose(g1, target, atol=1e-10)


def test_phase_rephasing_preserves_moduli_and_residue():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rng = np.random.default_rng(3)
    rows = _synthetic_rows(rng, 1)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    phonon = _phonon_input((q, 0, 0), vec)
    k = 1 / 6
    folded = np.array(
        [
            [(k + q) % 1, 0, 0],
            [k % 1, 0, 0],
            [(-(k + q)) % 1, 0, 0],
            [(-k) % 1, 0, 0],
        ]
    )
    modes = magnon.get_site_boson_modes(folded)
    n = modes.u.shape[1]
    phases = np.exp(1j * np.array([0.6, -1.1]))[:n]
    thetas = [np.diag(phases) for _ in range(4)]
    g0 = model.vertex(
        (k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon, site_boson_modes=modes
    ).g
    g1 = model.vertex(
        (k, 0.0, 0.0),
        (q, 0.0, 0.0),
        phonon_modes=phonon,
        site_boson_modes=_rephase_modes(modes, thetas),
    ).g
    g0 = g0[:, :, 0]
    g1 = g1[:, :, 0]
    np.testing.assert_allclose(np.abs(g1), np.abs(g0), atol=1e-10)
    res0 = np.linalg.eigvalsh(g0 @ g0.conj().T)
    res1 = np.linalg.eigvalsh(g1 @ g1.conj().T)
    np.testing.assert_allclose(res1, res0, atol=1e-10)


# --------------------------------------------------------------------------
# loaders and validation
# --------------------------------------------------------------------------
def _matched_rows_document(axis, atom):
    return {
        "schema": "matched-periodic-J-v1",
        "campaign": "fixture",
        "h_Ang": 0.005,
        "settings": {},
        "provenance": {
            "geometry": {
                "cartesian_axis_one_based": axis,
                "displaced_atom_zero_based": atom,
                "half_step_A": 0.005,
            },
            "units": {"derivative": "meV/Angstrom"},
        },
        "rows": [
            {"i": 0, "j": 1, "R": [1, 0, 0], "FD_meV_A": -1.5},
            {"i": 1, "j": 0, "R": [-1, 0, 0], "FD_meV_A": 0.25},
        ],
    }


def test_load_matched_rows_rejects_missing_cartesian_axis_at_vertex(tmp_path):
    paths = []
    for axis in (1, 2):
        path = tmp_path / f"rows-axis{axis}.json"
        path.write_text(json.dumps(_matched_rows_document(axis, atom=0)))
        paths.append(path)
    with pytest.raises(ValueError, match="all three Cartesian axes"):
        load_matched_rows(paths)

    # The vertex seam also refuses a manually constructed partial Cartesian
    # row tensor; an absent axis is not silently treated as a measured zero.
    magnon = _spring_magnon()
    partial = OrientedBondDerivatives(
        i=np.array([0, 1]),
        j=np.array([1, 0]),
        R=np.array([[1, 0, 0], [-1, 0, 0]]),
        dJ_dtau=np.zeros((2, 1, 2)),
        convention="home_cell",
    )
    phonon = PhononModeInput(
        q=np.zeros(3),
        frequencies_thz=np.array([5.3]),
        masses_amu=np.array([1.3]),
        eigenvectors=np.ones((3, 1), dtype=complex) / np.sqrt(3),
    )
    with pytest.raises(ValueError, match="three Cartesian axes"):
        MagnonPhononModel(magnon, partial).vertex(
            (1 / 6, 0.0, 0.0), (0.0, 0.0, 0.0), phonon_modes=phonon
        )


def test_load_matched_rows_converts_units_with_complete_axis_coverage(tmp_path):
    paths = []
    for axis in (1, 2, 3):
        path = tmp_path / f"rows-axis{axis}.json"
        path.write_text(json.dumps(_matched_rows_document(axis, atom=0)))
        paths.append(path)
    bonds, meta = load_matched_rows(paths)
    assert bonds.convention == "home_cell"
    assert bonds.units == "eV/Angstrom"
    assert bonds.dJ_dtau.shape == (2, 1, 3)
    np.testing.assert_allclose(bonds.dJ_dtau[0, 0, 0], -1.5e-3)
    np.testing.assert_allclose(bonds.dJ_dtau[1, 0, 1], 0.25e-3)
    np.testing.assert_allclose(bonds.dJ_dtau[0, 0, 2], -1.5e-3)
    assert meta["schema"] == "matched-periodic-J-v1"


def test_load_matched_rows_rejects_inconsistent_bond_sets(tmp_path):
    good = _matched_rows_document(1, atom=0)
    bad = _matched_rows_document(2, atom=0)
    bad["rows"] = [dict(good["rows"][0])]
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(good))
    p2.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="inconsistent"):
        load_matched_rows([p1, p2])


def _s002_report(q, mode, polarization, bonds):
    rows = []
    for i, j, R, value in bonds:
        rows.append(
            {
                "iatom": i,
                "jatom": j,
                "R": list(R),
                "O_real_meV_per_Ang": value.real * 1e3,
                "O_imag_meV_per_Ang": value.imag * 1e3,
                "FD_real_meV_per_Ang": (value * 0.99).real * 1e3,
                "FD_imag_meV_per_Ang": (value * 0.99).imag * 1e3,
            }
        )
    return {
        "schema_version": 1,
        "campaigns": {
            "K": {
                "campaign": {"campaign_id": "fixture-K"},
                "reports": [
                    {
                        "mode": mode,
                        "q": list(q),
                        "polarization_real": np.asarray(polarization).real.tolist(),
                        "polarization_imag": np.asarray(polarization).imag.tolist(),
                        "rows": rows,
                    }
                ],
            }
        },
    }


def test_s002_complex_report_loads_into_vertex_and_preserves_identity(tmp_path):
    from TB2J.magnon_phonon.loaders import load_s002_report
    from TB2J.magnon_phonon.vertex import ModeProjectedDerivatives

    q = (1 / 3, 1 / 3, 0.0)
    polarization = np.array([[0.8 + 0.1j, 0.2j, 0.0], [0.4, 0.1, 0.0]], dtype=complex)
    polarization /= np.linalg.norm(polarization)
    report_bonds = [
        (0, 1, (-1, 0, 0), 0.12 + 0.07j),
        (1, 0, (1, 0, 0), -0.04 + 0.03j),
    ]
    path = tmp_path / "s002.json"
    mode_evidence_path, _ = _write_s002_fixture(
        path, q, "acoustic-like", polarization, report_bonds, mode_index=0
    )
    derivatives = load_s002_report(
        path,
        campaign="K",
        mode="acoustic-like",
        expected_bonds=[(i, j, R) for i, j, R, _ in report_bonds],
        mode_evidence_path=mode_evidence_path,
    )
    assert isinstance(derivatives, ModeProjectedDerivatives)
    assert derivatives.mode == "acoustic-like"
    np.testing.assert_allclose(derivatives.q, q)
    np.testing.assert_allclose(derivatives.polarization, polarization)
    np.testing.assert_allclose(derivatives.O_pattern, [0.12 + 0.07j, -0.04 + 0.03j])
    np.testing.assert_allclose(
        derivatives.FD_pattern, [0.1188 + 0.0693j, -0.0396 + 0.0297j]
    )

    magnon = _spring_magnon()
    masses = np.array([1.3, 0.7])
    eig = np.sqrt(masses[:, None]) * polarization
    eig = eig.reshape(-1, 1)
    eig /= np.linalg.norm(eig)
    phonon = PhononModeInput(
        q=np.asarray(q),
        frequencies_thz=np.array([5.3]),
        masses_amu=masses,
        eigenvectors=eig,
        provenance={
            "mode_evidence_sha256": derivatives.mode_evidence_sha256,
            "mode": derivatives.mode,
        },
    )
    model = MagnonPhononModel(magnon, derivatives)
    with pytest.raises(ValueError, match="mode index"):
        model.vertex(
            (1 / 6, 1 / 6, 0),
            q,
            phonon_modes=phonon,
            mode_index=1,
        )
    phonon_without_join = PhononModeInput(
        q=np.asarray(q),
        frequencies_thz=phonon.frequencies_thz,
        masses_amu=masses,
        eigenvectors=eig,
    )
    with pytest.raises(ValueError, match="provenance"):
        model.vertex(
            (1 / 6, 1 / 6, 0),
            q,
            phonon_modes=phonon_without_join,
            mode_index=0,
        )
    block = model.vertex((1 / 6, 1 / 6, 0), q, phonon_modes=phonon, mode_index=0)
    assert block.g.shape == (2, 2, 1)
    assert block.conventions["mode_identity"] == "acoustic-like"

    # A conjugate report at -q is the Hermiticity partner; the imported report
    # O retains its complex phase and directed-bond rows.
    partner_path = tmp_path / "s002-minus.json"
    qminus = tuple(-np.asarray(q))
    partner_mode_path, _ = _write_s002_fixture(
        partner_path,
        qminus,
        "acoustic-like",
        polarization.conj(),
        [(i, j, R, value.conjugate()) for i, j, R, value in report_bonds],
        mode_index=0,
    )
    partner = load_s002_report(
        partner_path,
        campaign="K",
        mode="acoustic-like",
        expected_bonds=[(i, j, R) for i, j, R, _ in report_bonds],
        mode_evidence_path=partner_mode_path,
    )
    phonon_minus = PhononModeInput(
        q=np.asarray(qminus),
        frequencies_thz=np.array([5.3]),
        masses_amu=masses,
        eigenvectors=eig.conj(),
        provenance={
            "mode_evidence_sha256": partner.mode_evidence_sha256,
            "mode": partner.mode,
        },
    )
    reverse = MagnonPhononModel(magnon, partner).vertex(
        (1 / 6 + q[0], 1 / 6 + q[1], 0),
        qminus,
        phonon_modes=phonon_minus,
        mode_index=0,
    )
    np.testing.assert_allclose(
        block.g[:, :, 0], np.conj(reverse.g[:, :, 0].T), atol=1e-10
    )


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("wrong_q", "q"),
        ("missing_component", "O_imag"),
        ("missing_bond", "bond set"),
        ("missing_polarization", "missing polarization"),
        ("wrong_mode", "mode"),
    ],
)
def test_s002_report_rejects_invalid_identity_or_incomplete_rows(
    tmp_path, mutation, match
):
    from TB2J.magnon_phonon.loaders import load_s002_report

    q = (1 / 3, 1 / 3, 0.0)
    polarization = np.ones((2, 3), dtype=complex)
    polarization /= np.linalg.norm(polarization)
    bonds = [
        (0, 1, (-1, 0, 0), 0.12 + 0.07j),
        (1, 0, (1, 0, 0), -0.04 + 0.03j),
    ]
    mode = "acoustic-like"
    expected = [(i, j, R) for i, j, R, _ in bonds]
    path = tmp_path / f"{mutation}.json"
    mode_evidence_path, mode_evidence = _write_s002_fixture(
        path, q, mode, polarization, bonds
    )
    data = json.loads(path.read_text())
    report = data["campaigns"]["K"]["reports"][0]
    if mutation == "wrong_q":
        report["q"] = [0.25, 0.25, 0.0]
        path.write_text(json.dumps(data))
    elif mutation == "missing_component":
        del report["rows"][0]["O_imag_meV_per_Ang"]
        path.write_text(json.dumps(data))
    elif mutation == "missing_bond":
        report["rows"].pop()
        path.write_text(json.dumps(data))
    elif mutation == "missing_polarization":
        mode_evidence.pop("polarization_real")
        evidence_bytes = json.dumps(mode_evidence, sort_keys=True).encode()
        mode_evidence_path.write_bytes(evidence_bytes)
        report["source_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
        path.write_text(json.dumps(data))
    elif mutation == "wrong_mode":
        mode = "Cr-dominated-optical"
    with pytest.raises(ValueError, match=match):
        load_s002_report(
            path,
            campaign="K",
            mode=mode,
            expected_bonds=expected,
            mode_evidence_path=mode_evidence_path,
        )


def _write_s002_fixture(path, q, mode, polarization, bonds, mode_index=0):
    data = _s002_report(q, mode, polarization, bonds)
    report = data["campaigns"]["K"]["reports"][0]
    report.pop("polarization_real")
    report.pop("polarization_imag")
    report["source_evidence"] = f"{path.stem}-mode.json"
    evidence = {
        "mode": mode,
        "mode_eigenvector_index": mode_index,
        "q": list(q),
        "polarization_real": np.asarray(polarization).real.tolist(),
        "polarization_imag": np.asarray(polarization).imag.tolist(),
    }
    evidence_path = path.parent / report["source_evidence"]
    evidence_bytes = json.dumps(evidence, sort_keys=True).encode()
    evidence_path.write_bytes(evidence_bytes)
    report["source_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    path.write_text(json.dumps(data))
    return evidence_path, evidence


def test_as_phonon_mode_input_duck_typing():
    vec, masses, zpa = _pattern()
    phonon = _phonon_input((0.3, 0, 0), vec)

    class _Producer:
        q = phonon.q
        frequencies_thz = phonon.frequencies_thz
        masses_amu = phonon.masses_amu
        eigenvectors = phonon.eigenvectors

    adapted = as_phonon_mode_input(_Producer())
    np.testing.assert_allclose(adapted.q, [0.3, 0, 0])
    assert adapted.eigenvectors.shape == (6, 1)

    class _Broken:
        q = phonon.q

    with pytest.raises(ValueError, match="lacks attribute"):
        as_phonon_mode_input(_Broken())

    class _BadRows:
        q = phonon.q
        frequencies_thz = phonon.frequencies_thz
        masses_amu = phonon.masses_amu
        eigenvectors = phonon.eigenvectors[:3]

    with pytest.raises(ValueError, match="rows"):
        as_phonon_mode_input(_BadRows())


def test_home_cell_rows_rejected_at_finite_q():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rows = _pattern_rows(q, vec, zpa)
    container = _bonds_container(0.0, rows, convention="home_cell")
    model = MagnonPhononModel(magnon, container)
    with pytest.raises(ValueError, match="home_cell"):
        model.vertex(
            (1 / 6, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=_phonon_input((q, 0, 0), vec)
        )


def test_pattern_rows_rejected_on_momentum_mismatch():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rows = _pattern_rows(q, vec, zpa)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    with pytest.raises(ValueError, match="momentum"):
        model.vertex(
            (1 / 6, 0.0, 0.0),
            (q, 0.0, 0.0),
            phonon_modes=_phonon_input((1 / 4, 0, 0), vec),
        )


def test_unstable_phonon_frequencies_are_rejected():
    vec, masses, zpa = _pattern()
    with pytest.raises(ValueError, match="positive phonon frequencies"):
        _zero_point_amplitudes(np.array([5.3, -1.0]), masses)


def test_bond_site_out_of_range_is_rejected():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rows = _pattern_rows(q, vec, zpa)
    container = OrientedBondDerivatives(
        i=np.array([0, 1, 0, 1, 0, 5], dtype=int),
        j=np.array([b[1] for b in BONDS], dtype=int),
        R=np.array([(b[2], 0, 0) for b in BONDS], dtype=int),
        dJ_dtau=rows,
        convention="pattern_projected",
        source_q=np.array([q, 0.0, 0.0]),
    )
    model = MagnonPhononModel(magnon, container)
    with pytest.raises(ValueError, match="out of range"):
        model.vertex(
            (1 / 6, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=_phonon_input((q, 0, 0), vec)
        )


def test_injected_modes_must_match_queried_points():
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    rows = _pattern_rows(q, vec, zpa)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    modes = magnon.get_site_boson_modes([[0.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="four folded points"):
        model.vertex(
            (1 / 6, 0.0, 0.0),
            (q, 0.0, 0.0),
            phonon_modes=_phonon_input((q, 0, 0), vec),
            site_boson_modes=modes,
        )


# --------------------------------------------------------------------------
# Round-1 review fixes
# --------------------------------------------------------------------------
def _unit_cartesian(phonon, mode_index=0):
    masses = np.asarray(phonon.masses_amu, dtype=float)
    cart = (
        np.asarray(phonon.eigenvectors[:, mode_index], dtype=complex).reshape(
            len(masses), 3
        )
        / np.sqrt(masses)[:, None]
    )
    return cart / np.linalg.norm(cart)


def _mode_projected_container(q, phonon, phase=0.0):
    """ModeProjectedDerivatives whose report polarization is e^{i phase} * cart."""
    cart = _unit_cartesian(phonon)
    polarization = np.exp(1j * phase) * cart
    # O_pattern is the fully pattern-projected derivative (Bloch phase on the
    # j endpoint) along the report polarization, per Angstrom; the zero-point
    # factor is applied exactly once by the vertex module.
    O = np.zeros(len(BONDS), dtype=complex)
    for bi, (i, j, R, j0, beta) in enumerate(BONDS):
        di, dj = _dJ_endpoint(bi)
        O[bi] = (
            (
                di * polarization[i, 0]
                + np.exp(2j * np.pi * q * R) * dj * polarization[j, 0]
            )
            * SPINS[i]
            * SPINS[j]
        )
    return ModeProjectedDerivatives(
        q=np.array([q, 0.0, 0.0]),
        mode="m0",
        mode_index=0,
        polarization=polarization,
        i=np.array([b[0] for b in BONDS], dtype=int),
        j=np.array([b[1] for b in BONDS], dtype=int),
        R=np.array([(b[2], 0, 0) for b in BONDS], dtype=int),
        O_pattern=O,
        FD_pattern=O,
        mode_evidence_sha256="fixture-sha",
    )


def _provenanced_phonon(q, vec):
    phonon = _phonon_input((q, 0.0, 0.0), vec)
    return PhononModeInput(
        q=phonon.q,
        frequencies_thz=phonon.frequencies_thz,
        masses_amu=phonon.masses_amu,
        eigenvectors=phonon.eigenvectors,
        provenance={"mode": "m0", "mode_evidence_sha256": "fixture-sha"},
    )


def test_mode_projected_join_applies_report_polarization_phase():
    # STD-001: a report polarization that equals the phonon Cartesian mode up
    # to a global phase must join with the measured relative phase applied to
    # O_pattern; the vertex must equal the phase-free report's vertex.
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    k = 1 / 6
    phonon = _provenanced_phonon(q, vec)
    g_plain = (
        MagnonPhononModel(magnon, _mode_projected_container(q, phonon, 0.0))
        .vertex((k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon, mode_index=0)
        .g
    )
    g_rotated = (
        MagnonPhononModel(magnon, _mode_projected_container(q, phonon, phase=0.7))
        .vertex((k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon, mode_index=0)
        .g
    )
    np.testing.assert_allclose(g_rotated, g_plain, atol=1e-12)
    # the joined rows are physically correct: the same springs contracted with
    # the module's own Cartesian pattern (Bloch phase on the j endpoint)
    # reproduce the joined vertex, and match the mechanical reference driven
    # by the same displacement field
    cart, zpa_eff = _projected_mode_zero_point(phonon, 0)
    rows_expected = np.zeros(len(BONDS), dtype=complex)
    for bi, (i, j, R, j0, beta) in enumerate(BONDS):
        di, dj = _dJ_endpoint(bi)
        rows_expected[bi] = (
            zpa_eff
            * (di * cart[i, 0] + np.exp(2j * np.pi * q * R) * dj * cart[j, 0])
            * SPINS[i]
            * SPINS[j]
        )
    g_expected = (
        MagnonPhononModel(magnon, _bonds_container(q, rows_expected[:, None]))
        .vertex((k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon)
        .g
    )
    np.testing.assert_allclose(g_plain, g_expected, atol=1e-10)
    reference = MechanicalReference().vertex(k, q, cart[:, 0] * zpa_eff, 1e-4)
    np.testing.assert_allclose(g_expected[:, :, 0], reference, atol=1e-8)
    assert np.max(np.abs(g_plain)) > 1e-9


def test_scalar_collinear_forbidden_block_is_zero_in_reference():
    # SPEC-001: independently assert the forbidden bilinear mixing block of the
    # mechanical reference is zero — distinct from re-comparing the allowed
    # a^dag a projection, which must itself be nonzero for the control to bite.
    magnon = _spring_magnon()
    vec, masses, zpa = _pattern()
    q = 1 / 3
    k = 1 / 6
    rows = _pattern_rows(q, vec, zpa)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    result = model.vertex(
        (k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=_phonon_input((q, 0, 0), vec)
    )
    reference = MechanicalReference()
    g_allowed, cross_uw, cross_wu = reference.mixed_blocks(k, q, vec * zpa, 1e-4)
    np.testing.assert_allclose(result.g[:, :, 0], g_allowed, atol=1e-8)
    # the allowed a^dag a projection is nonvanishing (nonvacuous control) ...
    assert np.max(np.abs(g_allowed)) > 1e-9
    # ... while the u-w / w-u cross contractions at the conjugate pairing
    # (k+q, -(k+q)) — the would-be bilinear mixing block — vanish identically
    # for the scalar collinear coupling.
    np.testing.assert_allclose(cross_uw, 0.0, atol=1e-12)
    np.testing.assert_allclose(cross_wu, 0.0, atol=1e-12)
    # structurally: the collinear spring reference has no hole amplitudes
    folded = np.array(
        [[(k + q) % 1, 0, 0], [k % 1, 0, 0], [(q - k) % 1, 0, 0], [(-k) % 1, 0, 0]]
    )
    modes = magnon.get_site_boson_modes(folded)
    assert np.max(np.abs(modes.v)) < 1e-12


def _degenerate_anomalous_magnon():
    """Two decoupled copies of the anisotropic two-site chain: degenerate
    bands with nonzero hole (v) amplitudes."""
    Rlist = np.array(
        [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (-1, 0, 0)], dtype=int
    )
    JR = np.zeros((6, 4, 4, 3, 3))
    for ir, off in ((0, 0), (3, 2)):
        i, j = off, off + 1
        JR[ir, i, i] = np.diag([-2.0, -1.0, 0.0]) * SPINS[0] ** 2
        JR[ir, j, j] = np.diag([-2.0, -1.0, 0.0]) * SPINS[1] ** 2
        JR[ir + 1, i, j] = np.eye(3) * 0.4 * SPINS[0] * SPINS[1]
        JR[ir + 2, j, i] = np.eye(3) * 0.4 * SPINS[0] * SPINS[1]
    mags = np.zeros((4, 3))
    mags[0, 2] = mags[2, 2] = 2.0 * SPINS[0]
    mags[1, 2] = mags[3, 2] = 2.0 * SPINS[1]
    magnon = Magnon(
        nspin=4,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.diag([2.4, 10, 10]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


@pytest.mark.parametrize("k,q", [(1 / 6, 1 / 3), (1 / 4, 1 / 4)])
def test_degenerate_subspace_with_hole_amplitudes_rephasing(k, q):
    # SPEC-002: degenerate-magnon-subspace covariance/invariance must also hold
    # when the hole (v) amplitudes are nonzero.
    magnon = _degenerate_anomalous_magnon()
    vec, masses, zpa = _pattern()
    rng = np.random.default_rng(5)
    rows = _synthetic_rows(rng, 1)
    model = MagnonPhononModel(magnon, _bonds_container(q, rows))
    phonon = _phonon_input((q, 0, 0), vec)
    folded = np.array(
        [
            [(k + q) % 1, 0, 0],
            [k % 1, 0, 0],
            [(-(k + q)) % 1, 0, 0],
            [(-k) % 1, 0, 0],
        ]
    )
    modes = magnon.get_site_boson_modes(folded)
    sorted_e = np.sort(modes.energies, axis=1)
    assert np.allclose(sorted_e[:, 0], sorted_e[:, 1])
    assert np.allclose(sorted_e[:, 2], sorted_e[:, 3])
    assert np.max(np.abs(modes.v)) > 0.05, "hole amplitudes must be nonzero"
    g0 = model.vertex(
        (k, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon, site_boson_modes=modes
    ).g[:, :, 0]

    rng = np.random.default_rng(11)
    thetas = []
    for _ in range(4):
        blocks = []
        for _ in range(2):
            raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            qmat, _ = np.linalg.qr(raw)
            blocks.append(qmat)
        thetas.append(
            np.block([[blocks[0], np.zeros((2, 2))], [np.zeros((2, 2)), blocks[1]]])
        )
    # rephase with the partner pairing matched to the folded order
    # [k+q, k, -(k+q), -k]: v(p) -> Theta_{-p} v(p)
    u = modes.u.copy()
    v = modes.v.copy()
    partner = {0: 2, 1: 3, 2: 0, 3: 1}
    for ik in range(4):
        u[ik] = thetas[ik].conj() @ u[ik]
        v[ik] = thetas[partner[ik]] @ v[ik]
    rephased = SiteBosonModes(
        kpoints=modes.kpoints,
        energies=modes.energies,
        u=u,
        v=v,
        paraunitary=modes.paraunitary,
        spin_lengths=modes.spin_lengths,
        U=modes.U,
        V=modes.V,
        diagnostics=modes.diagnostics,
        is_zero_mode=modes.is_zero_mode,
    )
    g1 = model.vertex(
        (k, 0.0, 0.0),
        (q, 0.0, 0.0),
        phonon_modes=phonon,
        site_boson_modes=rephased,
    ).g[:, :, 0]
    np.testing.assert_allclose(g1, thetas[0] @ g0 @ thetas[1].conj().T, atol=1e-10)
    res0 = np.linalg.eigvalsh(g0 @ g0.conj().T)
    res1 = np.linalg.eigvalsh(g1 @ g1.conj().T)
    np.testing.assert_allclose(res1, res0, atol=1e-10)


def test_records_and_vertex_do_not_mutate_caller_arrays():
    # STD-002: read-only records must own their arrays; the finite-q join must
    # normalize a copy of the report polarization.
    magnon = _spring_magnon()
    folded = np.array([[0.5, 0, 0], [1 / 6, 0, 0], [0.5, 0, 0], [1 / 6, 0, 0]])
    base = magnon.get_site_boson_modes(folded)
    kwargs = dict(
        kpoints=base.kpoints.copy(),
        energies=base.energies.copy(),
        u=base.u.copy(),
        v=base.v.copy(),
        paraunitary=base.paraunitary.copy(),
        spin_lengths=base.spin_lengths.copy(),
        U=base.U.copy(),
        V=base.V.copy(),
        diagnostics=SiteBosonDiagnostics(
            min_eig=np.ones(4),
            stability_tol=np.ones(4) * 1e-8,
            symmetry_residual=np.zeros(4),
            shift=np.zeros(4),
            cholesky_tier=np.zeros(4, dtype=int),
        ),
        is_zero_mode=base.is_zero_mode.copy(),
    )
    diag_kwargs = kwargs.pop("diagnostics")
    sentinels = {name: val.copy() for name, val in kwargs.items()}
    record = SiteBosonModes(diagnostics=diag_kwargs, **kwargs)
    for name, arr in kwargs.items():
        assert arr.flags.writeable, f"caller array {name} was made read-only"
        np.testing.assert_array_equal(arr, sentinels[name])
        assert not getattr(record, name).flags.writeable

    diag = record.diagnostics
    for name in ("min_eig", "stability_tol", "symmetry_residual", "shift"):
        assert not getattr(diag, name).flags.writeable

    # vertex join: complex report polarization must not be normalized in place
    vec, masses, zpa = _pattern()
    q = 1 / 3
    phonon = _provenanced_phonon(q, vec)
    container = _mode_projected_container(q, phonon, phase=0.4)
    polarization_before = container.polarization.copy()
    assert container.polarization.flags.writeable
    MagnonPhononModel(magnon, container).vertex(
        (1 / 6, 0.0, 0.0), (q, 0.0, 0.0), phonon_modes=phonon, mode_index=0
    )
    assert container.polarization.flags.writeable
    np.testing.assert_array_equal(container.polarization, polarization_before)
