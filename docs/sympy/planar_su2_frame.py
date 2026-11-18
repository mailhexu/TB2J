"""Assertion-checked derivation: planar SU(2) frame for spin-scalar
hopping and nonorthogonal overlap.

Story 001 (spiral-first-order-response), NFR-001/NFR-005.  This pins the
gauge, phase and Fourier conventions that ADR-R1 production code must
match, BEFORE production code exists.  The scripts import neither TB2J
nor TBUpy: every builder here is an independent oracle.

Conventions (pinned here, mirrored by ``TBUpy/tbupy/planar_spiral.py``):

* Interleaved spinor layout ``2*(cell*norb + mu) + s``; folded primitive
  blocks indexed ``2*mu + s``.
* Site unitary about the spiral normal y::

      U(theta) = exp(-i theta sigma_y / 2)          (real 2x2 rotation)

  so the lab-frame planar field at angle ``Theta`` is
  ``Bf (cos(Theta) sz + sin(Theta) sx) = U(Theta) sz U(theta)^dag`` with
  ``Bf = B_local / 2`` (the stored up/down splitting is ``2*Bf``).
* Lab spiral angles ``Theta[a, mu] = 2 pi q.(R_a + tau_mu) + phi_mu``
  with sublattice phase ``alpha_mu = 2 pi q.tau_mu + phi_mu``.
* Gauge transform of a spin-scalar class (BOTH rotation vertices: the
  left vertex ``mu`` and right vertex ``nu`` enter through
  ``Delta = Theta[R nu] - Theta[0 mu] = 2 pi q.R + alpha_nu - alpha_mu``)::

      Hq(k)[mu,nu]  = sum_R e^{-2 pi i k.R} h(mu,nu,R)  U(Delta_{mu nu}(R))
      Sq(k)[mu,nu]  = sum_R e^{-2 pi i k.R} s(mu,nu,R)  U(Delta_{mu nu}(R))

  with the Hermitian pairing ``h(nu,mu,-R) = h(mu,nu,R)^*`` (overlap
  alike) and the flat local-frame field ``Bf sz`` added on site.  The
  pair ``(e^{-2 pi i k.R}, U(+Delta))`` is pinned; the mirrored pair
  ``(e^{+2 pi i k.R}, U(-Delta))`` is its ``k -> -k`` equivalent and is
  asserted to give the same lab spectrum too.
* Folded mesh on ``N`` cells: ``k_m = (m + s)/N`` with the half flux
  shift ``s = 1/2`` iff ``round(q_x N)`` is odd, else ``s = 0``
  (``planar_flux_shift``).  Verified numerically for even and odd
  numerators; a wrong shift yields a plausible but WRONG spectrum.

Checks
------
[A] SU(2) basics: unitarity, composition ``U(a)U(b) = U(a+b)``, active
    rotation of the moment, lab field form.
[B] Dimer gauge transform (symbolic, both vertices):
    ``G^dag H_lab G = H_loc`` and ``G^dag S_lab G = S_loc`` with
    ``G = diag(U(theta_1), U(theta_2))`` - the left and right vertex
    rotations of every bond appear exactly once, in ``U(Delta)``.
[C] Explicit commensurate lab ring vs folded primitive pencil
    (numeric, multi-sublattice, NONORTHOGONAL overlap): identical
    generalized spectra on the half-shifted mesh for even and odd
    ``round(qN)``, and the mapped lab eigenvectors
    ``c_lab[(a,mu,s)] = e^{-2 pi i k_m a} c_fold[(mu,s)]`` satisfy the
    lab pencil equation (gauge transform verified, not just spectra).
[D] Mirrored Fourier/dressing pair ``(e^{+2 pi i k.R}, U(-Delta))``
    reproduces the same lab spectrum (``k -> -k`` equivalence).
[E] Pure-gauge invariance: conjugating the whole lab ring by fixed site
    unitaries leaves every eigenvalue unchanged - a whole-basis local
    unitary is NOT a physical perturbation (Main invariant: local
    gradient operators act on the magnetic field only).

Run with the mydev environment:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/planar_su2_frame.py
"""

import numpy as np
import sympy as sp

# --------------------------------------------------------------------------
# symbolic helpers
# --------------------------------------------------------------------------

SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
SZ = sp.Matrix([[1, 0], [0, -1]])


def U(angle):
    """SU(2) rotation exp(-i angle sigma_y / 2) as a real 2x2 matrix.

    exp(-i a sy/2) = cos(a/2) I - i sin(a/2) sy with -i sy = [[0,-1],[1,0]].
    """
    c, s = sp.cos(angle / 2), sp.sin(angle / 2)
    return sp.Matrix([[c, -s], [s, c]])


def rot_real(angle):
    """Real SO(3)-like generator form exp(-i angle sigma_y)."""
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[c, s], [-s, c]])


def check_su2_basics():
    """[A] Unitarity, composition, active moment rotation, lab field."""
    th = sp.Symbol("theta", real=True)
    assert sp.simplify(U(th).H * U(th) - sp.eye(2)) == sp.zeros(2)
    a, b = sp.symbols("a b", real=True)
    assert sp.simplify(U(a) * U(b) - U(a + b)) == sp.zeros(2)
    # active rotation of the moment direction (y rotation in the z-x plane)
    rho = (sp.eye(2) + SZ) / 2
    rot = sp.simplify(U(th) * rho * U(th).H)
    expect = (sp.eye(2) + sp.cos(th) * SZ + sp.sin(th) * SX) / 2
    assert sp.simplify(rot - expect) == sp.zeros(2)
    # lab field = U(Theta) Bf sz U(Theta)^dag
    Bf, T = sp.symbols("Bf T", real=True)
    lab = sp.simplify(U(T) * (Bf * SZ) * U(T).H)
    assert sp.simplify(lab - Bf * (sp.cos(T) * SZ + sp.sin(T) * SX)) == sp.zeros(2)
    print(
        "[A] SU(2) basics: U unitary, U(a)U(b)=U(a+b), lab field "
        "Bf(cosT sz + sinT sx) = U(T) sz U(T)^dag ... OK"
    )


def check_dimer_gauge_transform():
    """[B] Symbolic dimer: G^dag H_lab G and G^dag S_lab G, both vertices."""
    B1, B2, h, t1, t2, s12 = sp.symbols("B1 B2 h t1 t2 s12", real=True)
    field1 = B1 * (sp.cos(t1) * SZ + sp.sin(t1) * SX)
    field2 = B2 * (sp.cos(t2) * SZ + sp.sin(t2) * SX)
    H_lab = sp.Matrix([[field1, h * sp.eye(2)], [h * sp.eye(2), field2]])
    S_lab = sp.Matrix([[sp.eye(2), s12 * sp.eye(2)], [s12 * sp.eye(2), sp.eye(2)]])
    G = sp.diag(U(t1), U(t2))
    H_loc = sp.Matrix([[B1 * SZ, h * U(t2 - t1)], [h * U(t1 - t2), B2 * SZ]])
    S_loc = sp.Matrix([[sp.eye(2), s12 * U(t2 - t1)], [s12 * U(t1 - t2), sp.eye(2)]])
    assert sp.simplify(G.H * H_lab * G - H_loc) == sp.zeros(4)
    assert sp.simplify(G.H * S_lab * G - S_loc) == sp.zeros(4)
    # vertex decomposition of Delta = twist + right vertex - left vertex
    q, R, tau1, tau2, p1, p2 = sp.symbols("q R tau1 tau2 phi1 phi2", real=True)
    Delta = (
        2 * sp.pi * q * R + (2 * sp.pi * q * tau2 + p2) - (2 * sp.pi * q * tau1 + p1)
    )
    assert sp.expand(Delta - (2 * sp.pi * q * (R + tau2 - tau1) + p2 - p1)) == 0
    # both vertices appear with unit strength and opposite sign
    assert sp.diff(Delta, p1) == -1 and sp.diff(Delta, p2) == 1
    assert sp.simplify(sp.diff(Delta, q) - 2 * sp.pi * (R + tau2 - tau1)) == 0
    print(
        "[B] dimer gauge transform: G^dag H_lab G = H_loc, G^dag S_lab G "
        "= S_loc; Delta carries left (-) and right (+) vertices ... OK"
    )


# --------------------------------------------------------------------------
# numeric builders (numpy only; independent of TB2J/TBUpy)
# --------------------------------------------------------------------------


def rot2(theta):
    """exp(-i theta sigma_y / 2) numeric real 2x2: [[c,-s],[s,c]]."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])


def make_classes(rng, norb, max_rep, hop=0.4, ov=0.08, real=False):
    """Random Hermitian spin-scalar classes + nonorthogonal overlap.

    Returns (classes, taus, phis, B_local): classes is a list of
    (mu, nu, R_signed(3,), amplitude) containing both Hermitian
    partners; overlap classes share the pairing with identity-dominant
    amplitudes (SPD).  ``real=True`` draws real amplitudes only, i.e. a
    time-reversal-symmetric model on which the planar reflection
    symmetry forces the delta channel to vanish.
    """
    classes = []
    reps = [r + 1 for r in range(max_rep)]

    def draw(scale):
        if real:
            return complex(round(rng.normal(), 6), 0.0) * scale / hop
        return complex(round(rng.normal(), 6), round(rng.normal(), 6)) * scale

    for mu in range(norb):
        for nu in range(norb):
            for R in reps:
                vec = np.array([R, 0, 0])
                amp = draw(hop)
                if mu == nu:
                    amp = complex(amp.real, 0.0)
                classes.append((mu, nu, vec, amp))
                classes.append((nu, mu, -vec, amp.conjugate()))
    sc = ov

    def draw_ov():
        if real:
            return complex(round(rng.uniform(-1, 1), 6), 0.0) * sc
        return complex(round(rng.uniform(-1, 1), 6), round(rng.uniform(-1, 1), 6)) * sc

    for mu in range(norb):
        for nu in range(norb):
            for R in reps[: 1 + (mu + nu) % max_rep]:
                vec = np.array([R, 0, 0])
                amp = draw_ov()
                if mu == nu:
                    amp = complex(amp.real, 0.0)
                classes.append((mu, nu, vec, amp, True))  # overlap flag
                classes.append((nu, mu, -vec, amp.conjugate(), True))
    taus = np.round(rng.random((norb, 3)) * 0.4, 6)
    phis = np.round(rng.normal(size=norb), 6)
    B_local = np.round(rng.uniform(0.8, 1.6, size=norb), 6)
    return classes, taus, phis, B_local


def angles(taus, phis, q, ncell):
    """Theta[a, mu] = 2 pi q.(R_a + tau_mu) + phi_mu."""
    cells = np.zeros((ncell, 3))
    cells[:, 0] = np.arange(ncell)
    return 2 * np.pi * (cells[:, None, :] + taus[None, :, :]) @ q + phis[None, :]


def split_classes(classes):
    hop, ovl = [], []
    for c in classes:
        (ovl if len(c) == 5 else hop).append(c)
    return hop, ovl


def assemble_lab(classes, taus, phis, q, B_local, ncell):
    """Explicit lab ring (H, S): spin-scalar classes + rotating fields."""
    hop, ovl = split_classes(classes)
    norb = taus.shape[0]
    size = 2 * norb * ncell
    H = np.zeros((size, size), dtype=complex)
    S = np.eye(size, dtype=complex)
    Bf = 0.5 * B_local
    Th = angles(taus, phis, q, ncell)

    def idx(a, mu, s):
        return 2 * (a * norb + mu) + s

    for cls in hop:
        mu, nu, R, amp = cls[0], cls[1], cls[2], cls[3]
        for a in range(ncell):
            b = (a + int(R[0])) % ncell
            for s in (0, 1):
                H[idx(a, mu, s), idx(b, nu, s)] += amp
    for cls in ovl:
        mu, nu, R, amp = cls[0], cls[1], cls[2], cls[3]
        for a in range(ncell):
            b = (a + int(R[0])) % ncell
            for s in (0, 1):
                S[idx(a, mu, s), idx(b, nu, s)] += amp
    for a in range(ncell):
        for mu in range(norb):
            th = Th[a, mu]
            blk = Bf[mu] * (
                np.cos(th) * np.diag([1.0, -1.0])
                + np.sin(th) * np.array([[0, 1], [1, 0]], dtype=complex)
            )
            H[idx(a, mu, 0) : idx(a, mu, 0) + 2, idx(a, mu, 0) : idx(a, mu, 0) + 2] += (
                blk
            )
    return H, S


def assemble_folded(
    classes, taus, phis, q, B_local, k, bloch_sign=-1.0, dressing_sign=+1.0
):
    """Folded primitive pencil (Hq, Sq) at fractional k (3,)."""
    hop, ovl = split_classes(classes)
    norb = taus.shape[0]
    alpha = 2 * np.pi * (taus @ q) + phis
    Bf = 0.5 * B_local
    Hq = np.zeros((2 * norb, 2 * norb), dtype=complex)
    Sq = np.eye(2 * norb, dtype=complex)
    for cls_h, target in [(c, Hq) for c in hop] + [(c, Sq) for c in ovl]:
        mu, nu, R, amp = cls_h[0], cls_h[1], cls_h[2], cls_h[3]
        Delta = 2 * np.pi * float(q @ R) + alpha[nu] - alpha[mu]
        bloch = np.exp(bloch_sign * 2j * np.pi * float(k @ R.astype(float)))
        block = bloch * amp * rot2(dressing_sign * Delta)
        target[2 * mu : 2 * mu + 2, 2 * nu : 2 * nu + 2] += block
    for mu in range(norb):
        Hq[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] += Bf[mu] * np.diag([1.0, -1.0])
    return Hq, Sq


def flux_shift(qx, ncell):
    """s = 1/2 iff round(qx*N) odd (planar_flux_shift convention)."""
    qn = qx * ncell
    assert abs(qn - round(qn)) < 1e-9, f"commensurate qN required, got {qn}"
    return 0.5 if int(round(qn)) % 2 == 1 else 0.0


def gen_eigh(H, S):
    """Generalized eigenvalues via Cholesky reduction (numpy only)."""
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    A = Linv @ H @ Linv.conj().T
    A = 0.5 * (A + A.conj().T)
    return np.linalg.eigvalsh(A)


def folded_spectrum(classes, taus, phis, q, B_local, ncell, **kw):
    s = flux_shift(q[0], ncell)
    vals = []
    for m in range(ncell):
        k = np.array([(m + s) / ncell, 0.0, 0.0])
        Hq, Sq = assemble_folded(classes, taus, phis, q, B_local, k, **kw)
        vals.append(gen_eigh(Hq, Sq))
    return np.sort(np.concatenate(vals))


def check_lab_vs_folded():
    """[C] Lab ring == folded pencil: spectra + eigenvector gauge map."""
    for seed, (num, den) in zip((11, 12), ((3, 8), (2, 8))):
        q = np.array([num / den, 0.0, 0.0])
        ncell, norb = den, 2
        rng = np.random.default_rng(seed)
        classes, taus, phis, B_local = make_classes(rng, norb, ncell // 2)
        H_lab, S_lab = assemble_lab(classes, taus, phis, q, B_local, ncell)
        # SPD of the expanded overlap
        wS = np.linalg.eigvalsh(0.5 * (S_lab + S_lab.conj().T))
        assert wS.min() > 0.1, f"overlap not SPD: {wS.min()}"
        lab_vals = gen_eigh(H_lab, S_lab)
        fold_vals = folded_spectrum(classes, taus, phis, q, B_local, ncell)
        dev = np.max(np.abs(lab_vals - fold_vals)) / max(1.0, np.max(np.abs(lab_vals)))
        assert dev < 1e-10, f"seed {seed} q={q[0]}: spectrum dev {dev}"

        # eigenvector gauge map: c_lab[(a,mu,s)] = e^{-2 pi i k_m a} c_fold
        s_shift = flux_shift(q[0], ncell)
        residual = 0.0
        for m in range(ncell):
            k = np.array([(m + s_shift) / ncell, 0.0, 0.0])
            Hq, Sq = assemble_folded(classes, taus, phis, q, B_local, k)
            L = np.linalg.cholesky(Sq)
            Linv = np.linalg.inv(L)
            A = Linv @ Hq @ Linv.conj().T
            A = 0.5 * (A + A.conj().T)
            w, u = np.linalg.eigh(A)
            c_fold = Linv.conj().T @ u  # generalized eigenvectors
            phases = np.exp(-2j * np.pi * k[0] * np.arange(ncell))
            Th = angles(taus, phis, q, ncell)
            for col in range(2 * norb):
                c_lab = np.zeros(2 * norb * ncell, dtype=complex)
                for a in range(ncell):
                    for mu in range(norb):
                        # local-frame Bloch wave, then site unitary to lab
                        blk = phases[a] * (
                            rot2(Th[a, mu]) @ c_fold[2 * mu : 2 * mu + 2, col]
                        )
                        c_lab[2 * (a * norb + mu) : 2 * (a * norb + mu) + 2] = blk
                r = H_lab @ c_lab - w[col] * (S_lab @ c_lab)
                residual = max(residual, np.max(np.abs(r)))
        assert residual < 1e-9, f"seed {seed}: mapped-vector residual {residual}"
        print(
            f"[C] seed {seed} q={num}/{den}: lab vs folded spectrum dev "
            f"{dev:8.1e}, mapped eigenvectors satisfy lab pencil "
            f"(max residual {residual:8.1e})  ... OK"
        )


def check_mirrored_pair():
    """[D] Mirrored (bloch,dressing) sign pair is the k -> -k equivalent."""
    q = np.array([3.0 / 8.0, 0.0, 0.0])
    ncell, norb = 8, 2
    rng = np.random.default_rng(13)
    classes, taus, phis, B_local = make_classes(rng, norb, ncell // 2)
    H_lab, S_lab = assemble_lab(classes, taus, phis, q, B_local, ncell)
    lab_vals = gen_eigh(H_lab, S_lab)
    fold_vals = folded_spectrum(
        classes, taus, phis, q, B_local, ncell, bloch_sign=+1.0, dressing_sign=-1.0
    )
    dev = np.max(np.abs(lab_vals - fold_vals)) / max(1.0, np.max(np.abs(lab_vals)))
    assert dev < 1e-10, dev
    print(
        f"[D] mirrored pair (e^{{+2 pi i k.R}}, U(-Delta)) spectrum dev "
        f"{dev:8.1e} ... OK"
    )


def check_pure_gauge_invariance():
    """[E] Whole-basis site unitaries: pure gauge, spectrum invariant."""
    q = np.array([3.0 / 8.0, 0.0, 0.0])
    ncell, norb = 8, 2
    rng = np.random.default_rng(14)
    classes, taus, phis, B_local = make_classes(rng, norb, ncell // 2)
    H_lab, S_lab = assemble_lab(classes, taus, phis, q, B_local, ncell)
    vals0 = gen_eigh(H_lab, S_lab)
    Th = angles(taus, phis, q, ncell)
    norb_ = norb
    G = np.eye(2 * norb_ * ncell, dtype=complex)
    for a in range(ncell):
        for mu in range(norb_):
            rot = rot2(Th[a, mu])
            sl = slice(2 * (a * norb_ + mu), 2 * (a * norb_ + mu) + 2)
            G[sl, sl] = rot
    vals1 = gen_eigh(G @ H_lab @ G.conj().T, G @ S_lab @ G.conj().T)
    dev = np.max(np.abs(vals0 - vals1))
    assert dev < 1e-10, dev
    print(
        f"[E] pure-gauge whole-basis rotation: eigenvalue drift {dev:8.1e} "
        "(local gradients must act on fields only)  ... OK"
    )


def main():
    check_su2_basics()
    check_dimer_gauge_transform()
    check_lab_vs_folded()
    check_mirrored_pair()
    check_pure_gauge_invariance()
    print("\nAll planar SU(2) frame assertions passed.")


if __name__ == "__main__":
    main()
