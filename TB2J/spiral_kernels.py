"""Two-channel MFT curvature kernels and gates for spiral references (story 004).

Magnetic force theorem about a frozen spiral reference, following the
normative identities of ``docs/sympy/spiral_state_mft.py``: the
curvature is evaluated about the *constrained-planar* spiral at lab
angles ``Theta_{a mu} = 2 pi q.(a + tau_mu) + phi_mu`` (field in the
(z, x) plane), with local transverse perturbations

* out-of-plane ``delta_a``:  ``V1 = Bf * sigma_y`` (site independent:
  the tilt direction is the spiral normal, invariant under the gauge
  rotation), and
* in-plane ``beta_a``:  ``V1 = Bf * d_theta field = Bf * (-sin(Theta)
  sigma_z + cos(Theta) sigma_x)``,
* ``V2 = -1/2 (delta^2 + beta^2) * Bf * n̂·sigma`` (the local field
  itself; ``Bf sigma_z`` in the local frame), and

curvature ``E2 = Tr[V2 rho] + (1/2) sum' (f_n - f_m) |V1_nm|^2 /
(eps_n - eps_m)`` evaluated per site pair.  ``Bf = B_local / 2`` is the
exchange-field amplitude implied by the bundle's ``+-B/2`` splitting.

Pinned convention (why): the contract's first-guess folded forms
(``B(cos alpha sigma_y + sin alpha sigma_x)`` etc. in the sigma_z-twist
gauge) describe the transverse response of the flat-screw (conical
theta_0 = 0) reference produced by the story-002/003 rebuild rule.
About that object the Goldstone and torque gate vectors are *not* zero
modes (the flat screw carries no moment texture; verified numerically).
The normative gate and anchor identities live on the constrained-planar
reference above, whose local frame is the sigma_y-rotation gauge.  The
equivalence test ``kernel(folded path) == kernel(explicit lab-frame
supercell path)`` pinned the signs and rotations to exactly the forms
stated here; the contract's alpha_mu-dependent operator shapes are
superseded (the sublattice angles enter the pencil phases, not the
operator shapes).

Folded path: the planar-gauge pencil dresses each stored hopping class
with ``exp(-(i/2) (2 pi q.R + alpha_nu - alpha_mu) sigma_y)`` and adds
the flat local field ``Bf sigma_z``; on the half-shifted mesh ``k_m =
(m + s)/N`` (``s = 1/2`` iff ``round(qN)`` is odd) its spectrum equals
the explicit supercell spectrum exactly.  The ring carries one spinor
flux quantum through the wrap bonds (a sigma_y rotation, not a scalar
sign), so the folded blocks admit *no* phase-form site unfolding; the
pair kernels are therefore evaluated on the explicit supercell
resolvents (:func:`contour_kernels_dense`), which is exact at O(N) per
Matsubara point and matches the folded spectrum identity tests below.

Gates (NFR-004) hard-fail with diagnostics; a bundle whose recorded
``torque_norms`` already exceed tolerance is flagged instead of raised.
"""

from __future__ import annotations

import numpy as np

from TB2J.spiral_green import SpiralGreen, SpiralState

__all__ = [
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "SpiralGateError",
    "SpiralCurvature",
    "local_field_amplitude",
    "spiral_angles",
    "planar_flux_shift",
    "planar_kmesh",
    "assemble_planar_pencil",
    "planar_spiral_green",
    "lab_supercell",
    "eigenbasis_kernels_dense",
    "contour_kernels_dense",
    "goldstone_gate",
    "torque_gate",
    "q0_anchor_check",
    "inplane_response_fd",
]

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]])
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

#: real rotation generator ``-i sigma_y`` so that ``rot_y(t) = exp(-i t sy)``
_SY = np.array([[0.0, -1.0], [1.0, 0.0]])


def _rot_y(theta: float) -> np.ndarray:
    """``exp(-i theta sigma_y)`` as a real 2x2 matrix."""
    return np.cos(theta) * np.eye(2) + np.sin(theta) * _SY


class SpiralGateError(ValueError):
    """A spiral curvature gate was violated.

    Carries the full diagnostic report (residuals, tolerances, scale)
    so callers can surface gate failures instead of silent ones.
    """

    def __init__(self, message: str, report: dict) -> None:
        super().__init__(message)
        self.report = report


def local_field_amplitude(state: SpiralState) -> np.ndarray:
    """Exchange-field amplitude ``Bf = B_local / 2`` per orbital.

    The bundle stores the up/down splitting ``B_local`` (the rebuild
    rule adds ``+-B_local/2`` diagonals), so the field operator is
    ``Bf * n̂·sigma`` with ``Bf = B_local / 2``.
    """
    return 0.5 * np.asarray(state.B_local, dtype=float)


def spiral_angles(state: SpiralState, ncell: int) -> np.ndarray:
    """Lab-frame spiral angles ``Theta[a, mu]`` for ``a = 0..ncell-1``."""
    q = np.asarray(state.q_frac, dtype=float)
    taus = np.asarray(state.taus, dtype=float)
    phis = np.asarray(state.phis, dtype=float)
    cells = np.zeros((ncell, 3))
    cells[:, 0] = np.arange(ncell)
    cell_twist = 2.0 * np.pi * (cells @ q)  # (ncell,)
    sub_twist = 2.0 * np.pi * (taus @ q) + phis  # (norb,)
    return cell_twist[:, None] + sub_twist[None, :]  # (ncell, norb)


def planar_flux_shift(state: SpiralState, ncell: int) -> float:
    """Half-shift ``s`` of the folded mesh absorbing the ring spinor flux.

    A planar ring at commensurate ``q`` carries the flux
    ``exp(-i pi qN sigma_y)``: the folded mesh must satisfy
    ``exp(2 pi i k N) = -1`` when ``round(qN)`` is odd.  Raises for
    incommensurate ``qN`` (no folded path exists; use the dense path).
    """
    qn = float(np.asarray(state.q_frac, dtype=float)[0]) * ncell
    qn_r = round(qn)
    if abs(qn - qn_r) > 1e-9:
        raise ValueError(
            f"planar-gauge folding needs qN commensurate, got qN = {qn!r}; "
            "use eigenbasis_kernels_dense for the explicit supercell path"
        )
    return 0.5 if int(qn_r) % 2 == 1 else 0.0


def planar_kmesh(state: SpiralState, ncell: int) -> np.ndarray:
    """Half-shifted commensurate mesh for the planar-gauge folding."""
    shift = planar_flux_shift(state, ncell)
    ks = (np.arange(ncell) + shift) / ncell
    return np.column_stack((ks, np.zeros(ncell), np.zeros(ncell)))


def assemble_planar_pencil(state: SpiralState, k_frac) -> tuple[np.ndarray, np.ndarray]:
    """Planar-gauge (local-frame) folded pencil ``(Hq, Sq)`` at one ``k``.

    ``Hq[k]_{mu nu} = sum_R e^{2 pi i k.R} HR_{mu nu}(R)
    exp(-(i/2)(2 pi q.R + alpha_nu - alpha_mu) sigma_y)`` with
    ``alpha_mu = 2 pi q.tau_mu + phi_mu``: each stored class is dressed
    with the sigma_y half-rotation connecting the local frames of the
    two ends.  The flat local field ``Bf sigma_z`` is added per orbital
    and ``V_U`` cell-periodically (toy bundles only: a rotating-frame
    ``V_U`` of the sigma_z-twist gauge transfers into this frame only up
    to that gauge difference).  The overlap ``SR`` is rotated with the
    same site unitaries (FPLO convention).
    """
    HR = np.asarray(state.HR_up, dtype=complex)
    HRdn = np.asarray(state.HR_dn, dtype=complex)
    if not np.allclose(HR, HRdn):
        raise ValueError(
            "the planar reference needs spin-scalar hopping classes "
            "(HR_up == HR_dn); SOC is out of scope for the spiral path"
        )
    SR = np.asarray(state.SR, dtype=complex)
    Rlist = np.asarray(state.Rlist)
    norb = state.norb
    q = np.asarray(state.q_frac, dtype=float)
    alpha = 2.0 * np.pi * (np.asarray(state.taus, dtype=float) @ q) + np.asarray(
        state.phis, dtype=float
    )
    Bf = local_field_amplitude(state)
    k = np.asarray(k_frac, dtype=float)

    Hq = np.zeros((2 * norb, 2 * norb), dtype=complex)
    Sq = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for iR, R in enumerate(Rlist):
        rf = float(R[0])
        bloch = np.exp(-2.0j * np.pi * float(k @ R.astype(float)))
        for mu in range(norb):
            for nu in range(norb):
                dth = 2.0 * np.pi * q[0] * rf + alpha[nu] - alpha[mu]
                rot = _rot_y(-0.5 * dth)
                sl = (slice(2 * mu, 2 * mu + 2), slice(2 * nu, 2 * nu + 2))
                Hq[sl] += bloch * HR[iR, mu, nu] * rot
                Sq[sl] += bloch * SR[iR, mu, nu] * rot
    for mu in range(norb):
        Hq[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] += Bf[mu] * SIGMA_Z
    if state.V_U is not None:
        Hq = Hq + np.asarray(state.V_U, dtype=complex)
    return Hq, Sq


def planar_spiral_green(state: SpiralState, ncell: int) -> SpiralGreen:
    """Folded planar-gauge resolvent on the half-shifted mesh."""
    return SpiralGreen(
        state,
        planar_kmesh(state, ncell),
        assembler=assemble_planar_pencil,
    )


def lab_supercell(state: SpiralState, ncell: int):
    """Explicit lab-frame planar supercell ``(H, S)`` (untwisted basis).

    Spin-scalar stored classes plus the rotating local fields
    ``Bf (cos Theta sigma_z + sin Theta sigma_x)`` per site; ``S`` is
    the phase-free expanded overlap, or ``None`` for orthogonal bundles.
    """
    HR = np.asarray(state.HR_up, dtype=complex)
    HRdn = np.asarray(state.HR_dn, dtype=complex)
    if not np.allclose(HR, HRdn):
        raise ValueError(
            "the planar reference needs spin-scalar hopping classes "
            "(HR_up == HR_dn); SOC is out of scope for the spiral path"
        )
    SR = np.asarray(state.SR, dtype=complex)
    Rlist = np.asarray(state.Rlist)
    norb = state.norb
    Bf = local_field_amplitude(state)
    thetas = spiral_angles(state, ncell)
    size = 2 * norb * ncell
    H = np.zeros((size, size), dtype=complex)
    S = np.zeros((size, size), dtype=complex)

    def idx(a: int, mu: int, s: int) -> int:
        return 2 * ((a % ncell) * norb + mu) + s

    for a in range(ncell):
        for iR, R in enumerate(Rlist):
            for mu in range(norb):
                for nu in range(norb):
                    for s in (0, 1):
                        H[idx(a, mu, s), idx(a + int(R[0]), nu, s)] += HR[iR, mu, nu]
                        S[idx(a, mu, s), idx(a + int(R[0]), nu, s)] += SR[iR, mu, nu]
        for mu in range(norb):
            th = thetas[a, mu]
            s0 = idx(a, mu, 0)
            H[s0 : s0 + 2, s0 : s0 + 2] += Bf[mu] * (
                np.cos(th) * SIGMA_Z + np.sin(th) * SIGMA_X
            )
    if np.allclose(S, np.eye(size), atol=1e-12):
        S = None
    return H, S


class SpiralCurvature:
    """Assembled two-channel curvature matrices over ring sites.

    ``C[(ca, cb)]`` is the real symmetric ``(ncell*norb, ncell*norb)``
    matrix of second derivatives ``d^2 E2 / dc_i dc_j`` for the
    coefficient pairs of channels ``ca, cb in {'d', 'b'}`` (site order
    ``a * norb + mu``).  The ``V2`` diagonal is folded in.
    """

    def __init__(self, blocks: dict, ncell: int, norb: int):
        self.blocks = blocks
        self.ncell = int(ncell)
        self.norb = int(norb)

    @property
    def nsite(self) -> int:
        return self.ncell * self.norb

    def __getitem__(self, pair):
        return self.blocks[pair]

    def scaled_residual(self, vec) -> float:
        vec = np.asarray(vec, dtype=float)
        scale = max(max(np.max(np.abs(b)) for b in self.blocks.values()), 1e-12)
        return float(np.max(np.abs(vec)) / scale)


def _v2_diagonal_dense(
    evals: np.ndarray,
    evecs: np.ndarray,
    f: np.ndarray,
    thetas: np.ndarray,
    norb: int,
    Bf: np.ndarray,
    ncell: int,
) -> np.ndarray:
    """Dense-path ``V2`` diagonal ``-Bf_mu (m_a . n̂_a)`` per site."""
    out = np.zeros(ncell * norb)
    for a in range(ncell):
        for mu in range(norb):
            rows = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            block = evecs[rows, :]
            rho = (block * f[None, :]) @ block.conj().T
            mx = np.real(np.trace(rho @ SIGMA_X))
            my = np.real(np.trace(rho @ SIGMA_Y))
            mz = np.real(np.trace(rho @ SIGMA_Z))
            th = thetas[a, mu]
            nhat = np.array([np.sin(th), 0.0, np.cos(th)])
            out[a * norb + mu] = -Bf[mu] * float(np.dot([mx, my, mz], nhat))
    return out


def _wn_weights(evals: np.ndarray, f: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Degeneracy-safe antisymmetrized weights ``(f_n - f_m)/(eps_n - eps_m)``."""
    den = evals[:, None] - evals[None, :]
    df = f[:, None] - f[None, :]
    good = np.abs(den) > tol
    return np.where(good, df / np.where(good, den, 1.0), 0.0)


def eigenbasis_kernels_dense(state: SpiralState, ncell: int) -> SpiralCurvature:
    """Degeneracy-safe eigenbasis reference on the explicit supercell.

    One dense (generalized) eigendecomposition of the lab-frame
    supercell; the pair kernel is
    ``C^{ab}[i, j] = sum' (f_n - f_m)/(eps_n - eps_m) Re[(O_i)_nm (O_j)_mn]``
    with intra-degenerate-cluster terms masked, plus the ``V2``
    diagonal.  Works at any ``q`` (no commensurability requirement).
    """
    H, S = lab_supercell(state, ncell)
    norb = state.norb
    size = 2 * norb * ncell
    if S is None:
        evals, evecs = np.linalg.eigh(H)
    else:
        L = np.linalg.cholesky(S)
        Li = np.linalg.inv(L)
        A = Li @ H @ Li.conj().T
        evals, U = np.linalg.eigh(0.5 * (A + A.conj().T))
        evecs = Li.conj().T @ U
    mu_f = float(state.efermi)
    f = 1.0 / (1.0 + np.exp((evals - mu_f) / state.width))
    wn = _wn_weights(evals, f)
    thetas = spiral_angles(state, ncell)
    Bf = local_field_amplitude(state)

    ops = {}
    for a in range(ncell):
        for mu_i in range(norb):
            th = thetas[a, mu_i]
            s0 = 2 * (a * norb + mu_i)
            E = np.zeros((size, size), dtype=complex)
            E[s0 : s0 + 2, s0 : s0 + 2] = Bf[mu_i] * SIGMA_Y
            ops[("d", a, mu_i)] = E
            E = np.zeros((size, size), dtype=complex)
            E[s0 : s0 + 2, s0 : s0 + 2] = Bf[mu_i] * (
                -np.sin(th) * SIGMA_Z + np.cos(th) * SIGMA_X
            )
            ops[("b", a, mu_i)] = E
    Eops = {key: evecs.conj().T @ val @ evecs for key, val in ops.items()}

    nsite = norb * ncell
    blocks = {}
    for ca in ("d", "b"):
        for cb in ("d", "b"):
            M = np.zeros((nsite, nsite))
            for (cka, a, mua), E1 in Eops.items():
                if cka != ca:
                    continue
                for (ckb, b, mub), E2 in Eops.items():
                    if ckb != cb:
                        continue
                    M[a * norb + mua, b * norb + mub] = np.real(np.sum(wn * E1 * E2.T))
            if ca == cb:
                v2 = _v2_diagonal_dense(evals, evecs, f, thetas, norb, Bf, ncell)
                M = M + np.diag(v2)
            blocks[(ca, cb)] = 0.5 * (M + M.T)
    return SpiralCurvature(blocks, ncell, norb)


def contour_kernels_dense(
    state: SpiralState,
    ncell: int,
    n_matsubara: int = 3000,
) -> SpiralCurvature:
    """Contour/GF pair kernels on the explicit supercell resolvents.

    Matsubara form of the contour curvature at the pair level: with
    ``z_n = efermi + i pi width (2n + 1)`` and the real-space blocks of
    ``G(z) = (z S - H_lab)^{-1}``,
    ``C^{ab}[i, j] = width * sum_n Re Tr2[O_i G_ij(R; z_n) O_j
    G_ji(-R; z_n)]`` plus the analytic ``A/z^2`` tail of the sum.  The
    ``f'(eps)`` double-pole residue terms vanish for the sharp
    frozen-force-theorem protocol (mu inside a gap); the eigenbasis
    reference applies the identical intra-cluster masking.  The ``V2``
    diagonal is folded in.  Valid at any ``q``.
    """
    H, S = lab_supercell(state, ncell)
    norb = state.norb
    size = 2 * norb * ncell
    thetas = spiral_angles(state, ncell)
    Bf = local_field_amplitude(state)
    width = float(state.width)
    mu = float(state.efermi)
    ns = np.arange(-n_matsubara, n_matsubara + 1)
    zs = float(state.efermi) + 1j * np.pi * width * (2 * ns + 1.0)

    ops = {}
    for a in range(ncell):
        for mu_i in range(norb):
            th = thetas[a, mu_i]
            s0 = 2 * (a * norb + mu_i)
            E = np.zeros((size, size), dtype=complex)
            E[s0 : s0 + 2, s0 : s0 + 2] = Bf[mu_i] * SIGMA_Y
            ops[("d", a, mu_i)] = E
            E = np.zeros((size, size), dtype=complex)
            E[s0 : s0 + 2, s0 : s0 + 2] = Bf[mu_i] * (
                -np.sin(th) * SIGMA_Z + np.cos(th) * SIGMA_X
            )
            ops[("b", a, mu_i)] = E

    blocks = {}
    evals, evecs = _dense_ed(H, S)
    f = _frozen_f(state, evals=evals)
    Eops = {key: evecs.conj().T @ val @ evecs for key, val in ops.items()}
    # degenerate clusters of the frozen spectrum
    clusters = []
    done = set()
    for n in range(size):
        if n in done:
            continue
        cl = [m for m in range(size) if abs(evals[m] - evals[n]) < 1e-9]
        done.update(cl)
        if len(cl) > 1:
            clusters.append(cl)
    in_cluster = np.zeros(size, dtype=bool)
    for cl in clusters:
        in_cluster[cl] = True
    fp = -(1.0 / width) * f * (1.0 - f)
    G_chunks = []
    for z0 in range(0, len(zs), 512):
        zc = zs[z0 : z0 + 512]
        G_chunks.append(
            (
                zc,
                np.linalg.inv(zc[:, None, None] * _eye_or_S(S, size)[None] - H[None]),
            )
        )
    # site-block slices once: site_blocks[k][(i, j)] = G[z, 2i:2i+2, 2j:2j+2]
    nsite = ncell * norb
    site_blocks = [
        {
            (i, j): G[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            for i in range(nsite)
            for j in range(nsite)
        }
        for _, G in G_chunks
    ]
    # Asymptotic subtraction: the Matsubara tail of the sandwich converges
    # only like 1/N.  Expand G(z) = P/z + PB/z^2 + PB^2/z^3 + ... with
    # P = S^-1, B = PH, subtract the explicit A2/z^2 + A3/z^3 + A4/z^4
    # coefficients of the pair sandwich from the summed integrand (the
    # remainder decays as 1/z^5), and add the closed-form Matsubara sums
    # of the subtracted terms back analytically.
    if S is None:
        P = np.eye(size)
    else:
        P = np.linalg.inv(S)
    B = P @ H
    B2 = B @ B
    S2 = _matsubara_zsum(mu, width, 2)
    S3 = _matsubara_zsum(mu, width, 3)
    S4 = _matsubara_zsum(mu, width, 4)
    for ca in ("d", "b"):
        for cb in ("d", "b"):
            acc = np.zeros((nsite, nsite))
            for (cka, a, mua), O1 in ops.items():
                if cka != ca:
                    continue
                E1 = Eops[(cka, a, mua)]
                for (ckb, b, mub), O2 in ops.items():
                    if ckb != cb:
                        continue
                    E2 = Eops[(ckb, b, mub)]
                    i = a * norb + mua
                    j = b * norb + mub
                    o1 = O1[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
                    o2 = O2[2 * j : 2 * j + 2, 2 * j : 2 * j + 2]
                    p_ij = P[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    p_ji = P[2 * j : 2 * j + 2, 2 * i : 2 * i + 2]
                    b_ij = B[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    b_ji = B[2 * j : 2 * j + 2, 2 * i : 2 * i + 2]
                    b2_ij = B2[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                    b2_ji = B2[2 * j : 2 * j + 2, 2 * i : 2 * i + 2]
                    a2 = np.trace(o1 @ p_ij @ o2 @ p_ji)
                    a3 = np.trace(o1 @ b_ij @ o2 @ p_ji) + np.trace(
                        o1 @ p_ij @ o2 @ b_ji
                    )
                    a4 = (
                        np.trace(o1 @ b2_ij @ o2 @ p_ji)
                        + np.trace(o1 @ p_ij @ o2 @ b2_ji)
                        + np.trace(o1 @ b_ij @ o2 @ b_ji)
                    )
                    tot = 0.0
                    for (zc, _), blk in zip(G_chunks, site_blocks):
                        h = np.einsum(
                            "ab,zbc,cd,zda->z",
                            o1,
                            blk[(i, j)],
                            o2,
                            blk[(j, i)],
                        )
                        h -= a2 / zc**2 + a3 / zc**3 + a4 / zc**4
                        tot += width * np.real(np.sum(h))
                    tot += width * float(np.real(a2 * S2 + a3 * S3 + a4 * S4))
                    # f' residue correction: Z = C_masked + J => C = Re(Z) - Re(J).
                    # Cluster members' diagonal fp terms are already inside
                    # the cluster block sums, so the plain-diagonal piece
                    # covers only non-cluster states.
                    not_cl = ~in_cluster
                    J = complex(
                        np.sum(
                            fp[not_cl] * E1.diagonal()[not_cl] * E2.diagonal()[not_cl]
                        )
                    )
                    for cl in clusters:
                        J = J + fp[cl[0]] * np.sum(
                            E1[np.ix_(cl, cl)] * E2[np.ix_(cl, cl)].T
                        )
                    tot -= float(np.real(J))
                    acc[i, j] = tot
            if ca == cb:
                v2 = _v2_diagonal_dense(evals, evecs, f, thetas, norb, Bf, ncell)
                acc = acc + np.diag(v2)
            blocks[(ca, cb)] = 0.5 * (acc + acc.T)
    return SpiralCurvature(blocks, ncell, norb)


def _matsubara_zsum(mu: float, width: float, k: int) -> float:
    """Closed-form ``sum_n (mu + i pi width (2n+1))^-k`` for k = 2, 3, 4.

    From ``sum_n 1/(x + i pi w (2n+1)) = tanh(x/(2w))/(2w)`` by repeated
    differentiation; verified against brute-force sums to ~1e-14 (k = 3,
    4) and through the 1/N asymptote (k = 2).
    """
    t = np.tanh(mu / (2.0 * width))
    sh = 1.0 / np.cosh(mu / (2.0 * width)) ** 2
    if k == 2:
        return float(-sh / (4.0 * width**2))
    if k == 3:
        return float(-sh * t / (8.0 * width**3))
    if k == 4:
        return float(-sh * (2.0 * t**2 - sh) / (48.0 * width**4))
    raise ValueError(f"k must be 2, 3 or 4, got {k}")


def _eye_or_S(S, size):
    if S is None:
        return np.eye(size)
    return S


def _dense_ed(H, S):
    if S is None:
        return np.linalg.eigh(H)
    L = np.linalg.cholesky(S)
    Li = np.linalg.inv(L)
    A = Li @ H @ Li.conj().T
    evals, U = np.linalg.eigh(0.5 * (A + A.conj().T))
    return evals, Li.conj().T @ U


def _frozen_f(state, evals=None, H=None, S=None):
    if evals is None:
        evals, _ = _dense_ed(H, S)
    return 1.0 / (1.0 + np.exp((evals - state.efermi) / state.width))


def goldstone_gate(
    curv: SpiralCurvature,
    tol: float = 1e-7,
    allow_violation: bool = False,
) -> dict:
    """Goldstone gate: ``C^bb . 1 = 0`` (uniform in-plane rotation).

    Returns the diagnostic report; raises :class:`SpiralGateError` on
    violation unless ``allow_violation`` is set.
    """
    C = curv[("b", "b")]
    resid = C.sum(axis=1)
    scale = max(np.max(np.abs(C)), 1e-12)
    report = {
        "gate": "goldstone",
        "residuals": resid,
        "max_residual": float(np.max(np.abs(resid))),
        "scale": float(scale),
        "tol": float(tol),
        "passed": bool(np.max(np.abs(resid)) <= tol),
        "allow_violation": bool(allow_violation),
    }
    if not report["passed"] and not allow_violation:
        raise SpiralGateError(
            "Goldstone gate violated: C^bb . 1 = 0 requires a uniform "
            f"in-plane rotation zero mode (max residual "
            f"{report['max_residual']:.3e} > tol {tol:g}).",
            report,
        )
    return report


def torque_gate(
    curv: SpiralCurvature,
    state: SpiralState,
    ncell: int,
    tol: float = 1e-6,
    allow_violation: bool = False,
) -> dict:
    """Torque gate: the ``C^dd`` cos/sin zero modes.

    Global R_x / R_z rotations are zero modes of the out-of-plane block
    iff the reference is torque-balanced.  A bundle whose recorded
    ``torque_norms`` already exceed ``tol`` is *flagged* (report with
    ``passed=False, flagged=True``) instead of raised.
    """
    C = curv[("d", "d")]
    thetas = spiral_angles(state, ncell)
    ang = np.repeat(thetas[:, 0], curv.norb)
    r_cos = C @ np.cos(ang)
    r_sin = C @ np.sin(ang)
    scale = max(np.max(np.abs(C)), 1e-12)
    worst = max(float(np.max(np.abs(r_cos))), float(np.max(np.abs(r_sin))))
    norms = state.torque_norms
    bundle_torqued = norms is not None and bool(
        np.max(np.asarray(norms, dtype=float)) > tol
    )
    passed = bool(worst <= tol)
    flagged = bool((not passed) and bundle_torqued)
    report = {
        "gate": "torque",
        "residuals_cos": r_cos,
        "residuals_sin": r_sin,
        "max_residual": worst,
        "scale": float(scale),
        "tol": float(tol),
        "torque_norms": None if norms is None else np.asarray(norms, dtype=float),
        "passed": passed,
        "flagged": flagged,
        "allow_violation": bool(allow_violation),
    }
    if not passed and not flagged and not allow_violation:
        raise SpiralGateError(
            "Torque gate violated: C^dd cos/sin zero modes require a "
            f"torque-balanced reference (max residual {worst:.3e} > tol "
            f"{tol:g} with bundle torque_norms within tolerance).",
            report,
        )
    return report


def q0_anchor_check(
    curv: SpiralCurvature,
    M: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """q=0 LKAG anchor: ``C^dd = C^bb = M`` and ``C^db = 0``."""
    Cdd = curv[("d", "d")]
    Cbb = curv[("b", "b")]
    Cdb = curv[("d", "b")]
    report = {
        "gate": "q0_anchor",
        "max_dd_minus_bb": float(np.max(np.abs(Cdd - Cbb))),
        "max_bb_minus_M": float(np.max(np.abs(Cbb - M))),
        "max_db": float(np.max(np.abs(Cdb))),
        "tol": float(tol),
        "passed": bool(
            np.max(np.abs(Cdd - Cbb)) <= tol
            and np.max(np.abs(Cbb - M)) <= tol
            and np.max(np.abs(Cdb)) <= tol
        ),
    }
    if not report["passed"]:
        raise SpiralGateError(
            "q=0 LKAG anchor violated: C^dd = C^bb = M and C^db = 0 "
            f"required (|Cdd-Cbb| = {report['max_dd_minus_bb']:.3e}, "
            f"|Cbb-M| = {report['max_bb_minus_M']:.3e}, "
            f"|Cdb| = {report['max_db']:.3e}).",
            report,
        )
    return report


def inplane_response_fd(
    state: SpiralState,
    ncell: int,
    h: float = 1e-4,
) -> np.ndarray:
    """Collinear in-plane response matrix ``M`` by angle-FD.

    The derivation-script protocol: frozen-occupation band energy of the
    explicit supercell under per-site in-plane angle shifts
    ``beta_a`` — the exact rotated field, i.e. the linear
    ``V1 = sum_a beta_a Bf(-sin(Theta) sigma_z + cos(Theta) sigma_x)``
    plus the second-order ``V2 = -1/2 sum_a beta_a^2 Bf(cos(Theta)
    sigma_z + sin(Theta) sigma_x)`` tip-back — 4-point mixed second
    differences with Richardson extrapolation.
    """
    H, S = lab_supercell(state, ncell)
    norb = state.norb
    thetas = spiral_angles(state, ncell)
    Bf = local_field_amplitude(state)
    nsite = norb * ncell

    def energy(beta):
        Hp = H.copy()
        for a in range(ncell):
            for mu_i in range(norb):
                th = thetas[a, mu_i]
                s0 = 2 * (a * norb + mu_i)
                b = beta[a * norb + mu_i]
                Hp[s0 : s0 + 2, s0 : s0 + 2] += b * Bf[mu_i] * (
                    -np.sin(th) * SIGMA_Z + np.cos(th) * SIGMA_X
                ) - 0.5 * b**2 * Bf[mu_i] * (
                    np.cos(th) * SIGMA_Z + np.sin(th) * SIGMA_X
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

    M = np.zeros((nsite, nsite))
    for i in range(nsite):
        for j in range(i, nsite):

            def mixed(hh):
                ei = np.zeros(nsite)
                ej = np.zeros(nsite)
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
            M[i, j] = M[j, i] = val
    return 0.5 * (M + M.T)
