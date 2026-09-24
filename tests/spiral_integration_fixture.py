"""Deterministic synthetic integration fixture for the spiral path (story 007).

A two-sublattice spin-independent tight-binding chain with a local exchange
field, driven through the *real* TBUpy rotating-frame spinor SCF
(:func:`tbupy.spiral_state.run_spiral_scf` with the
:class:`~tbupy.spiral_state.MultiSublatticeSpiralProvider`) at a commensurate
torque-free ``q`` and exported as a ``tbupy_spiral_state`` v1 sidecar
(``*.spiral.nc``).

Model (hand-picked amplitudes, no RNG -> bitwise deterministic):

* primitive cell = one dimer along ``e_1`` (lattice constant 1 A, 12 A vacuum
  in ``e_2``/``e_3``), sublattices A at ``tau = 0`` and B displaced
  transversally at ``tau = (0, 1/2, 0)``;
* spin-scalar real hopping classes: A-A and B-B at ``R = +-1`` (``t1``),
  A-B at ``R = 0`` (``t2``) and ``R = +-1`` (``t3``), on-site ``eps_A/eps_B``;
* orthogonal basis (``SR = 1``), local exchange field ``B_local`` per orbital
  (saturated-moment reference: the up channel is completely filled at
  ``nel = 2``, so every commensurate ``q`` is exactly torque-free and the
  reference is a band insulator);

The saturated class is the derivation-script reference case (docs/sympy/
spiral_green_function_mft.py section 4c): the torque/goldstone/diagonal
gates hold exactly and the q=0 LKAG anchor identity is exact up to the FD
and contour discretization of the two independent evaluation routes.

Two protocol facts pinned here (story 007, verified numerically):

* the transverse displacement keeps ``q . tau_mu`` sublattice-independent.
  With an intra-cell phase ``q . (tau_B - tau_A) != 0`` the TB2J-side
  frozen reference (``lab_supercell``/planar gauge) is *not* exactly
  torque-balanced: its frozen local moments cant by O((t/B)^2) and the
  torque gate fails at the 1e-3 level.  ``q . tau_mu`` independent of
  ``mu`` (here 0 for both sublattices) restores the exact stationary
  reference (transverse torque at machine zero);
* the sidecar ``efermi`` is the *frozen* Fermi level of the
  force-theorem protocol: after the real SCF (whose converged density
  supplies the bundle ``rho``/``V_U``/torque gate quantities) the value
  is re-centred in the largest gap of the TB2J-side reference
  Hamiltonian, exactly like the story-005 toy fixtures.  The TBUpy SCF
  and the TB2J kernels evaluate gauge-inequivalent folded pencils (the
  tbupy assembler twists with sigma_z phases, the TB2J reference with
  sigma_y rotations), so the SCF Fermi level is not used as the frozen
  value; only its converged density and gate diagnostics are.

The generated sidecars are cached under ``tests/_spiral_integration_cache/``
keyed by a fingerprint of every model/SCF parameter, so the test suite is
fast and deterministic while still exercising the real SCF.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# model parameters (all model/scf knobs live here; they key the cache)
# ---------------------------------------------------------------------------
CELL_A = 1.0  # lattice constant along the chain (Ang)
VAC = 12.0  # transverse vacuum (Ang)
T1 = 0.35  # A-A / B-B hopping, R = +-1
T2 = 0.25  # A-B hopping, R = 0
T3 = 0.15  # A-B hopping, R = +-1
EPS = (0.05, -0.05)  # on-site energies (A, B)
B_LOCAL = (3.0, 3.0)  # local exchange field (full up/dn splitting), eV
NEL = 2.0  # electrons per cell (saturated up channel)
WIDTH = 0.05  # SCF / frozen-occupation smearing width, eV
SYMBOLS = ("Cr", "Cr")

CACHE_DIR = Path(__file__).resolve().parent / "_spiral_integration_cache"


def cell_vectors() -> np.ndarray:
    """Row-major primitive cell of the dimer chain."""
    return np.diag([CELL_A, VAC, VAC])


def taus_fractional() -> np.ndarray:
    """Per-orbital fractional positions ``(2, 3)``.

    B is displaced transversally so that ``q . tau_mu = 0`` for every
    sublattice (see the module docstring: an intra-cell ``q . tau``
    phase cants the TB2J-side frozen reference and breaks the torque
    gate).
    """
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])


def _reference_arrays():
    """Spin-scalar reference ``(Rlist, HR, SR)`` of the dimer chain."""
    classes = [
        (0, 0, 1, T1),
        (0, 0, -1, T1),
        (1, 1, 1, T1),
        (1, 1, -1, T1),
        (0, 1, 0, T2),
        (1, 0, 0, T2),
        (0, 1, 1, T3),
        (0, 1, -1, T3),
        (1, 0, 1, T3),
        (1, 0, -1, T3),
    ]
    rs = sorted({R for (_, _, R, _) in classes} | {0})
    norb = 2
    HR = np.zeros((len(rs), norb, norb))
    SR = np.zeros((len(rs), norb, norb))
    SR[rs.index(0)] = np.eye(norb)
    for mu, nu, R, amp in classes:
        HR[rs.index(R), mu, nu] += amp
    for mu, eps in enumerate(EPS):
        HR[rs.index(0), mu, mu] += eps
    rlist = np.zeros((len(rs), 3), dtype=np.int64)
    rlist[:, 0] = rs
    return rlist, HR, SR


def scf_kmesh(q: float, ncell: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-shifted commensurate mesh of the twisted-gauge ring.

    The sigma_z twisted gauge carries one flux quantum through the wrap when
    ``q * ncell`` is odd, i.e. the folded mesh is ``k_m = (m + 1/2)/N``;
    even/zero ``q * ncell`` uses the uniform mesh.  (Same parity rule as
    ``TB2J.spiral_kernels.planar_flux_shift``.)
    """
    qn_r = round(q * ncell)
    shift = 0.5 if int(qn_r) % 2 == 1 else 0.0
    kpts = np.column_stack(
        ((np.arange(ncell) + shift) / ncell, np.zeros(ncell), np.zeros(ncell))
    )
    return kpts, np.full(ncell, 1.0 / ncell)


def _provider(q: np.ndarray | list):
    """Build the real :class:`MultiSublatticeSpiralProvider` at ``q``."""
    from ase import Atoms
    from tbupy.generalized_bloch import MultiSublatticeSpiralConfig
    from tbupy.spiral_state import MultiSublatticeSpiralProvider
    from tbupy.util import SimpleOrbital

    rlist, HR, SR = _reference_arrays()
    taus = taus_fractional()
    atoms = Atoms(
        symbols=list(SYMBOLS),
        positions=taus @ cell_vectors(),
        cell=cell_vectors(),
        pbc=True,
    )
    config = MultiSublatticeSpiralConfig(
        np.asarray(q, dtype=float), taus, np.zeros(len(taus))
    )
    return MultiSublatticeSpiralProvider(
        HR,
        HR.copy(),
        SR,
        rlist,
        config,
        np.asarray(B_LOCAL, dtype=float),
        orbs=[SimpleOrbital(iatom=i, l=0, m=0, label=f"s{i}") for i in range(2)],
        atoms=atoms,
        nel=NEL,
        is_orthogonal=True,
    )


def _fingerprint(q: float, ncell: int, torque_tol: float) -> str:
    payload = {
        "cell": cell_vectors().tolist(),
        "taus": taus_fractional().tolist(),
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "EPS": list(EPS),
        "B_local": list(B_LOCAL),
        "nel": NEL,
        "width": WIDTH,
        "symbols": list(SYMBOLS),
        "q": float(q),
        "ncell": int(ncell),
        "torque_tol": torque_tol,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def frozen_efermi(state, ncell: int) -> float:
    """Centre of the largest gap of the TB2J-side reference Hamiltonian.

    The frozen force-theorem protocol evaluates the curvature with a Fermi
    level inside a gap; the value is taken from the reference object the
    TB2J kernels actually build (:func:`TB2J.spiral_kernels.lab_supercell`).
    """
    from TB2J.spiral_green import SpiralState as TB2JSpiralState
    from TB2J.spiral_kernels import lab_supercell

    mirror = TB2JSpiralState(
        **{f.name: getattr(state, f.name) for f in dataclasses.fields(TB2JSpiralState)}
    )
    H, S = lab_supercell(mirror, ncell)
    if S is None:
        ev = np.linalg.eigvalsh(H)
    else:
        L = np.linalg.cholesky(S)
        Li = np.linalg.inv(L)
        A = Li @ H @ Li.conj().T
        ev = np.linalg.eigvalsh(0.5 * (A + A.conj().T))
    ev = np.sort(ev)
    gaps = ev[1:] - ev[:-1]
    i0 = int(np.argmax(gaps))
    return float(0.5 * (ev[i0] + ev[i0 + 1]))


def build_spiral_bundle(q: float, ncell: int, torque_tol: float = 1e-6):
    """Run the real rotating-frame spinor SCF and return the tbupy bundle.

    ``spiral_state_from_scf`` enforces the local-torque gate on the SCF
    density (raising on violation), records the per-orbital norms and the
    q-evenness diagnostic in the sidecar metadata.  The returned bundle's
    ``efermi`` is the *frozen* value of the force-theorem protocol
    (largest-gap centre of the TB2J-side reference, see module docstring);
    ``rho``/``V_U``/``torque_norms`` are the SCF's own converged results.
    """
    from tbupy.spiral_state import run_spiral_scf, spiral_state_from_scf

    provider = _provider([q, 0.0, 0.0])
    kpts, kweights = scf_kmesh(q, ncell)
    scf = run_spiral_scf(
        provider,
        kpts,
        kweights,
        nel=NEL,
        width=WIDTH,
        tol_energy=1e-8,
        tol_rho=1e-10,
        max_iter=100,
    )
    state = spiral_state_from_scf(
        provider,
        scf,
        kmesh=kpts,
        kweights=kweights,
        width=WIDTH,
        torque_tolerance=torque_tol,
    )
    meta = json.loads(state.metadata_json)
    meta["frozen_efermi"] = "lab_supercell_largest_gap_center"
    meta["scf_efermi"] = float(state.efermi)
    return dataclasses.replace(
        state,
        efermi=frozen_efermi(state, ncell),
        metadata_json=json.dumps(meta, sort_keys=True),
    )


def cached_bundle(q: float, ncell: int, torque_tol: float = 1e-6) -> Path:
    """Sidecar path of the ``q`` bundle, generated on first use.

    The cache key is the fingerprint of every model/SCF parameter, so the
    sidecar regenerates automatically when the fixture definition changes.
    """
    from tbupy.spiral_state import save_spiral_state

    fp = _fingerprint(q, ncell, torque_tol)
    tag = f"q{q:.6f}_n{ncell}_{fp}".replace(".", "p").replace("-", "m")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"spiral_fixture_{tag}.spiral.nc"
    if not path.exists():
        save_spiral_state(path, build_spiral_bundle(q, ncell, torque_tol))
    return path


# ---------------------------------------------------------------------------
# classical Heisenberg LSWT on the ring (Toth-Lake cone gate, story 007)
# ---------------------------------------------------------------------------


def site_matrix_from_jdict(exchange_Jdict, ncell: int, norb: int) -> np.ndarray:
    """Site-ordered real symmetric matrix of the written Heisenberg tensors.

    ``(R, i, j)`` keys are expanded over the ring and symmetrized (the
    written tensors are inversion-symmetric, ``J(R,i,j) = J(-R,j,i)``).
    """
    J = np.zeros((ncell * norb, ncell * norb))
    for key, val in exchange_Jdict.items():
        head = key[0]
        if isinstance(head, (tuple, list, np.ndarray)):
            R = int(np.asarray(head).ravel()[0])
            i, j = int(key[1]), int(key[2])
        else:
            R, i, j = int(head), 0, 0
        for a in range(ncell):
            b = (a + R) % ncell
            J[a * norb + i, b * norb + j] += float(np.real(val))
    return 0.5 * (J + J.T)


def spiral_frames(ncell, norb, q, phis=None, cone=np.pi / 2):
    """Local frames of the flat screw / conical reference.

    ``Theta_(a,mu) = 2 pi q a + phi_mu``; ``cone`` tilts the moments from
    the spiral plane toward +y (the spiral normal), so the flat screw is
    ``cone = pi/2`` and the field-aligned limit is ``cone = 0``.
    """
    phis = np.zeros(norb) if phis is None else np.asarray(phis, float)
    theta = (2 * np.pi * q * np.arange(ncell)[:, None] + phis[None, :]).ravel()
    st, ct = np.sin(theta), np.cos(theta)
    sc, cc = np.sin(cone), np.cos(cone)
    one = np.ones_like(ct)
    nv = np.stack([sc * ct, cc * one, sc * st], axis=1)
    xv = np.stack([-st, np.zeros_like(ct), ct], axis=1)
    yv = np.cross(nv, xv)
    return nv, xv, yv


def classical_energy(J_site, ncell, norb, q, cone, field=0.0):
    """Classical Heisenberg + Zeeman energy of the conical spiral state."""
    Jm = np.asarray(J_site, float)
    nv, _, _ = spiral_frames(ncell, norb, q, None, cone)
    ex = sum(
        0.5 * Jm[i, j] * float(nv[i] @ nv[j])
        for i in range(ncell * norb)
        for j in range(i + 1, ncell * norb)
    )
    return 2.0 * ex - field * float(np.sum(nv[:, 1]))


def minimize_cone_angle(J_site, ncell, norb, q, field, n_iter=200):
    """Golden-section minimization of the classical energy over the cone angle."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a_, b_ = 1e-9, np.pi - 1e-9

    def e(t):
        return classical_energy(J_site, ncell, norb, q, t, field)

    for _ in range(n_iter):
        c_ = b_ - gr * (b_ - a_)
        d_ = a_ + gr * (b_ - a_)
        if e(c_) < e(d_):
            b_ = d_
        else:
            a_ = c_
    return 0.5 * (a_ + b_)


def toth_lake_multiband(
    J_site, ncell, norb, q, phis=None, cone=np.pi / 2, field=0.0, s=1.0
):
    """Real-space bosonic LSWT about the (conical) single-Q reference.

    General non-collinear Holstein-Primakown quadratic form in the local
    frames (Toth-Lake construction; scalar case validated against
    ``spiral_fixtures.toth_lake_dynamical`` to 2e-12 on the Néel ring).
    Returns ``(omega, labels)``: positive-mode energies sorted ascending
    and their cell-momentum mesh labels (FFT peak of the local-frame
    amplitude; flat-screw zeros appear at labels ``0`` and ``+- q*ncell``).
    """
    n_site = ncell * norb
    Jm = np.asarray(J_site, float)
    nv, xv, yv = spiral_frames(ncell, norb, q, phis, cone)
    h0 = np.zeros((n_site, n_site), dtype=complex)
    g0 = np.zeros((n_site, n_site), dtype=complex)
    for i in range(n_site):
        for j in range(i + 1, n_site):
            Wxx = float(xv[i] @ xv[j])
            Wyy = float(yv[i] @ yv[j])
            Wxy = float(xv[i] @ yv[j])
            Wyx = float(yv[i] @ xv[j])
            hop = (s / 2) * Jm[i, j] * (Wxx + Wyy + 1j * (Wyx - Wxy))
            pr = (s / 2) * Jm[i, j] * (Wxx - Wyy - 1j * (Wxy + Wyx))
            h0[i, j] += hop
            h0[j, i] += np.conj(hop)
            g0[i, j] += pr
            g0[j, i] += pr
            lng = s * Jm[i, j] * float(nv[i] @ nv[j])
            h0[i, i] -= lng
            h0[j, j] -= lng
    if field:
        for i in range(n_site):
            h0[i, i] += field * float(nv[i, 1])
    D = np.zeros((2 * n_site, 2 * n_site), dtype=complex)
    D[:n_site, :n_site] = h0
    D[:n_site, n_site:] = g0
    D[n_site:, :n_site] = -g0.conj()
    D[n_site:, n_site:] = -h0.conj()
    ev, vecs = np.linalg.eig(D)
    sel = np.argsort(ev.real)[n_site:]
    omega = ev.real[sel]
    theta = (2 * np.pi * q * np.arange(ncell)[:, None]).repeat(norb, 1).ravel()
    gauge = np.exp(-1j * theta)
    labels = []
    for m in sel:
        u = vecs[n_site:, m] * gauge
        ft = np.abs(np.fft.fft(u.reshape(ncell, norb).mean(axis=1)))
        labels.append(int(np.argmax(ft)))
    return omega, labels


def classical_torques(J_site, ncell, norb, q, phis=None, cone=np.pi / 2):
    """Classical exchange torque ``n_i x b_i`` decomposed in the local frame.

    Returns ``(torque_inplane, torque_normal)``: the magnitudes of the
    transverse-tangent (within-spiral-plane) and normal-to-plane torque
    components per site.  A torque-free planar reference has zero in-plane
    torque; the normal component drives the conical instability.
    """
    Jm = np.asarray(J_site, float)
    nv, xv, yv = spiral_frames(ncell, norb, q, phis, cone)
    n_site = ncell * norb
    t_in = np.zeros(n_site)
    t_norm = np.zeros(n_site)
    for i in range(n_site):
        b = np.zeros(3)
        for j in range(n_site):
            b += Jm[i, j] * nv[j]
        torque = np.cross(nv[i], b)
        t_in[i] = float(torque @ xv[i])
        t_norm[i] = float(torque @ yv[i])
    return t_in, t_norm
