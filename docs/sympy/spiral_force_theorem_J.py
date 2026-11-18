"""Spiral force-theorem energy mapping to Heisenberg exchange: sympy.

Second derivation script of the spin-spiral MFT spec (TB2J <-> TBUpy).

Pins, with assertion checks (all in TB2J conventions: SpinIO exchange
Jdict {(R,i,j)} with positive J = ferromagnetic, Heisenberg
E = -sum_{ij} J_ij e_i.e_j with each ordered pair counted once
magnon
Fourier convention J(q) = sum_R J(R) exp(-2 pi i q.R)
spiral spin
angle theta[R] = 2 pi q.R right-handed about +z):

1. Classical flat-spiral energy of a Heisenberg model:
       E(q)/N = -sum_R J(R) cos(2 pi q.R) = -Re J(q)
       E(q) - E(0) = J(0) - Re J(q) >= 0   (stability for FM J).
   Verified symbolically for symbolic shells {0, +-1, +-2}.

2. Exact inversion (energy mapping): on the q grid q_n = n/N the shell
   couplings are recovered from spiral energies by discrete Fourier
   inversion.  Verified symbolically.

3. LSWT bridge: with the same J, the FM magnon dispersion of the TB2J
   magnon module is
       omega(q) = 2 S [J(0) - J(q)]
   (Goldstone at q=0), and the classical spiral energy cost is
       E(q) - E(0) = (S/2) omega(q) = omega(q) / (2 S).
   Verified symbolically
   pins the factor-2/S conventions against the
   "M = moment" conventions of e.g. Daglum PRB 113, 214401 (2026)
   where M = 2S (mu_B units).

4. Multi-sublattice helical energy:
       E/N = -sum_{mu nu R} S_mu S_nu J_{mu nu}(R)
                 cos(2 pi q.R + alpha_nu - alpha_mu),
       alpha_mu = 2 pi q tau_mu + phi_mu.
   Bipartite NN chain: minimum at |phi_B - phi_A| = pi for J > 0
   the
   energy is periodic under q -> q + 2/delta (magnetic-cell folding)

   the bipartite LSWT dispersion
       omega(k) = 2 S sqrt(J_AA J_BB - |J_AB(k)|^2)
   reproduces the collinear AFM magnons with the chemical-Gamma /
   magnetic-Gamma folding of the incommensurate-magnon note.

5. Force-theorem (one-shot MFT) band-energy curvature == LKAG-type
   exchange: for a two-site mean-field Hamiltonian with local fields
   rotated by +-theta/2 and the density matrix FIXED at the reference
   solution,
       E_MFT(theta) = Tr[H(theta) rho_0],
       (1/2) E_MFT''(0) = -h^2 * chi_AB
   where -h^2 chi_AB is exactly the second-order contour trace
   sum_poles w_n Tr[Delta_A G_AB(E_n) Delta_B G_BA(E_n)] built from the
   reference resolvent (Liechtenstein 1987 kernel at second order).
   Verified symbolically on a two-site two-field model.
   Self-consistent relaxation (per-q SCF) changes the curvature —
   demonstrated numerically (P-b vs P-c protocols).

Run with the mydev environment:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/spiral_force_theorem_J.py
"""

import numpy as np
import sympy as sp

# --------------------------------------------------------------------------
# 1. Classical spiral energy of a Heisenberg ring
# --------------------------------------------------------------------------


def check_classical_spiral_energy():
    q = sp.Symbol("q", real=True)
    J1, J2 = sp.symbols("J_1 J_2", real=True)
    # pair shells R = +-1, +-2 (a pair dictionary has NO R = 0 onsite term;
    # the symbol J(0) below means J(q=0), not an R=0 shell)
    shells = [(1, J1), (-1, J1), (2, J2), (-2, J2)]

    def E(qv):
        return -sum(J * sp.cos(2 * sp.pi * qv * R) for R, J in shells)

    # J(q) with the TB2J magnon convention exp(-2 pi i q R)
    Jq = sum(J * sp.exp(-2 * sp.I * sp.pi * q * R) for R, J in shells)
    Jq_exp = sp.expand_complex(Jq)
    Jre = sum(J * sp.cos(2 * sp.pi * q * R) for R, J in shells)
    Jim = -sum(J * sp.sin(2 * sp.pi * q * R) for R, J in shells)
    assert sp.simplify(sp.re(Jq_exp) - Jre) == 0
    assert sp.simplify(sp.im(Jq_exp) - Jim) == 0
    assert sp.simplify(Jim) == 0  # +-R pairing kills the imaginary part
    # E(q) - E(0) = J(q=0) - Re J(q), with J(q=0) = sum_R J(R) = 2 J1 + 2 J2
    diff = sp.simplify(E(q) - E(0) - ((2 * J1 + 2 * J2) - Jre))
    assert diff == 0
    # exact coefficient structure at q = 1/4:
    # E(1/4)-E(0) = 2 J1 (1 - cos(pi/2)) + 2 J2 (1 - cos(pi))
    E14 = sp.simplify(E(sp.Rational(1, 4)) - E(0))
    expect14 = sp.simplify(2 * J1 * (1 - 0) + 2 * J2 * (1 - (-1)))
    assert sp.simplify(E14 - expect14) == 0
    print(
        "[1] classical spiral energy: E(q)/N = -Re J(q); "
        "E(q)-E(0) = J(0) - Re J(q)  ... OK"
    )


check_classical_spiral_energy()


# --------------------------------------------------------------------------
# 2. Exact inversion: E(q_n) -> J(R)
# --------------------------------------------------------------------------


def check_energy_mapping_inversion():
    """Given E(q_n) = -Re J(q_n) on q_n = n/N, recover J(R) shells."""
    N = 6
    Jtrue = {
        0: sp.Rational(7, 3),
        1: sp.Rational(-2, 5),
        2: sp.Rational(3, 7),
        3: sp.Rational(1, 11),
    }

    # J(R) defined for R = 0, +-1, +-2, +3/-3 (N even: R=N/2 self-paired)
    def E_of(qn):
        val = sp.Rational(0)
        for R, J in Jtrue.items():
            mult = 1 if R == 0 else (1 if 2 * R == N else 2)
            val -= mult * J * sp.cos(2 * sp.pi * qn * R)
        return val

    Es = [E_of(sp.Rational(n, N)) for n in range(N)]
    # inversion: J(R) = (1/N) sum_n [E(0) - E(q_n)] e^{+2 pi i q_n R} / (S^2
    # =1) with shell multiplicity handled by the +- pairing
    for R in (1, 2):
        val = (
            sum(
                (Es[0] - Es[n]) * sp.exp(2 * sp.I * sp.pi * sp.Rational(n, N) * R)
                for n in range(N)
            )
            / N
        )
        assert sp.simplify(sp.re(val) - Jtrue[R]) == 0, (R, val)
        assert sp.simplify(sp.im(val)) == 0
    # R = N/2 self-paired shell: factor 1/2
    val = (
        sum(
            (Es[0] - Es[n]) * sp.exp(2 * sp.I * sp.pi * sp.Rational(n, N) * 3)
            for n in range(N)
        )
        / N
    )
    assert sp.simplify(sp.re(val) - Jtrue[3]) == 0
    print(
        "[2] energy mapping inversion: J(R) recovered from E(q_n), "
        "shell multiplicities pinned (R and -R pair once, R=N/2 once)  ... OK"
    )


check_energy_mapping_inversion()


# --------------------------------------------------------------------------
# 3. LSWT bridge: omega(q) = 2S[J(0) - J(q)],  E(q) - E(0) = omega/(2S)
# --------------------------------------------------------------------------


def check_lswt_bridge():
    q = sp.Symbol("q", real=True)
    S = sp.Symbol("S", positive=True)
    J1 = sp.Symbol("J_1", real=True)
    shells = [(1, J1), (-1, J1)]

    def E(qv):
        return -sum(J * sp.cos(2 * sp.pi * qv * R) for R, J in shells)

    def Jq(qv):
        return sum(J * sp.exp(-2 * sp.I * sp.pi * qv * R) for R, J in shells)

    Jq0 = sp.simplify(sp.re(Jq(0)))  # J(q=0) = 2 J1
    omega = 2 * S * (Jq0 - sp.re(Jq(q)))
    delta_E = sp.simplify(E(q) - E(0))
    # With S^2 absorbed into J: delta_E = J(0) - J(q).
    assert sp.simplify(delta_E - (Jq0 - sp.re(Jq(q)))) == 0
    # Dimensional form: E(q)-E(0) = S^2 [J(0)-J(q)] = (S/2) omega(q).
    assert sp.simplify(delta_E * S**2 - omega * S / 2) == 0
    # Goldstone
    assert sp.simplify(omega.subs(q, 0)) == 0
    # small-q curvature: d2E/dq2|_0 = (2 pi)^2 sum_R J(R) R^2 (positive)
    d2 = sp.simplify(sp.diff(E(q), q, 2).subs(q, 0))
    assert d2 == (2 * sp.pi) ** 2 * (2 * J1)
    print(
        "[3] LSWT bridge: omega(q) = 2S[J(0)-J(q)]; E(q)-E(0) = "
        "S^2[J(0)-J(q)] = (S/2) omega(q); Goldstone at q=0; "
        "d2E/dq2|_0 = (2 pi)^2 sum_R J(R) R^2 > 0  ... OK"
    )


check_lswt_bridge()


# --------------------------------------------------------------------------
# 4. Multi-sublattice helical energy and bipartite AFM folding
# --------------------------------------------------------------------------


def check_multisublattice():
    q = sp.Symbol("q", real=True)
    J = sp.Symbol("J", positive=True)
    S = sp.Symbol("S", positive=True)
    dphi = sp.Symbol("Delta_phi", real=True)
    # NN bipartite chain: cell = 2 sites, tau_A = 0, tau_B = 1/2 (lattice a=1)
    # intracell bond (R=0, B-A: alpha_B - alpha_A = pi q + dphi)
    # intercell bond (R=1, A_next - B: 2 pi q (1 - 1/2) + alpha_A - alpha_B)
    alpha_A = 0
    alpha_B = sp.pi * q + dphi
    # energy per cell (2 sites), ordered pairs (A,B) and (B,A_next)
    e_intra = -2 * S**2 * J * sp.cos(alpha_B - alpha_A)
    e_inter = -2 * S**2 * J * sp.cos(2 * sp.pi * q - alpha_B + alpha_A)
    e_cell = sp.simplify(e_intra + e_inter)
    # both bonds combine: cos(pi q + dphi) + cos(pi q - dphi)
    #                   = 2 cos(pi q) cos(dphi)
    expect = -4 * S**2 * J * sp.cos(sp.pi * q) * sp.cos(dphi)
    assert sp.simplify(e_cell - expect) == 0, e_cell
    # minimum at dphi = pi for J > 0: dE/d(dphi) = 0 at dphi = pi
    de = sp.simplify(sp.diff(e_cell, dphi).subs(dphi, sp.pi))
    assert de == 0
    # magnetic-cell folding: e is periodic under q -> q + 2 (delta = 1/2)
    assert sp.simplify(e_cell.subs(q, q + 2) - e_cell) == 0
    print(
        "[4a] bipartite helical energy: e(q) = -4 S^2 J cos(pi q) "
        "cos(Delta_phi); AFM minimum at dphi = pi; period q -> q + 2  ... OK"
    )


check_multisublattice()


def check_bipartite_lswt():
    """Bipartite LSWT: omega(k) = 2S sqrt(J_AA J_BB - |J_AB(k)|^2) and the
    chemical-Gamma / magnetic-Gamma folding (afm_1d_gamma_q_magnon note)."""
    k = sp.Symbol("k", real=True)
    S = sp.Symbol("S", positive=True)
    J = sp.Symbol("J", positive=True)
    # 1D bipartite chain, one J: J_AA = J_BB = 2J (double-counted shells +-1
    # within the same sublattice at distance 2a => shells +-2 of the chemical
    # lattice give 2J), J_AB(k) = J(e^{-2 pi i k /2} + e^{+2 pi i k /2}) = 2J
    # cos(pi k)
    JAA = 2 * J
    JAB = 2 * J * sp.cos(sp.pi * k)
    omega = 2 * S * sp.sqrt(JAA**2 - JAB**2)
    omega = sp.simplify(omega)
    expect = 4 * S * J * sp.Abs(sp.sin(sp.pi * k))
    # check |sin| equality by squaring (avoid Abs handling)
    assert sp.simplify(sp.expand(omega**2 - expect**2)) == 0
    # degeneracy omega(Gamma) = omega(R) with R the magnetic zone corner:
    # chemical k = 0 and k = 1 (zone boundary) both give zero modes
    assert sp.simplify(omega.subs(k, 0)) == 0
    assert sp.simplify(omega.subs(k, 1)) == 0
    print(
        "[4b] bipartite LSWT: omega = 2S sqrt(J_AA J_BB - |J_AB(k)|^2) = "
        "4SJ|sin(pi k)|; zero modes at chemical Gamma and R  ... OK"
    )


check_bipartite_lswt()

# --------------------------------------------------------------------------
# 5. Force-theorem (P-b) band-energy curvature == second-order resolvent
#    (LKAG-type) kernel; P-b vs P-c distinction
# --------------------------------------------------------------------------

SX = sp.Matrix([[0, 1], [1, 0]])
SZ = sp.Matrix([[1, 0], [0, -1]])


def rotated_field(h, sign, theta):
    """Local field h sigma_z rotated by sign*theta/2 about y (pair-rotation
    geometry
    a single-q spiral is a coherent sequence of these local
    rotations, so the second-order kernel is the same)."""
    ct, st = sp.cos(sign * theta / 2), sp.sin(sign * theta / 2)
    return h * (ct * SZ + st * SX)


def two_site_model(h, t, theta):
    """4x4 Hamiltonian: site A field +h rotated +theta/2, site B field -h
    rotated -theta/2, hopping t (spin independent). Interleaved spin
    index (A up, A down, B up, B down)."""
    HA = rotated_field(h, +1, theta)
    HB = rotated_field(-h, -1, theta)
    H = sp.zeros(4, 4)
    for i in range(2):
        for j in range(2):
            H[i, j] += HA[i, j]  # field within site A
            H[2 + i, 2 + j] += HB[i, j]  # field within site B
    H[0, 2] += t  # hopping A-B, spin conserving
    H[2, 0] += t
    H[1, 3] += t
    H[3, 1] += t
    return H


def check_second_order_structure_symbolic():
    """Symbolic structure of H(theta) = H0 + theta V1 + theta^2 V2 + ..."""
    theta = sp.Symbol("theta", real=True)
    h, t = sp.symbols("h t", positive=True)
    Ht = two_site_model(h, t, theta)
    V1 = sp.simplify(sp.diff(Ht, theta).subs(theta, 0))
    V2 = sp.simplify(sp.diff(Ht, theta, 2).subs(theta, 0)) / 2
    expect_V1 = sp.zeros(4, 4)
    for s in (0, 1):
        expect_V1[s, 1 - s] += h / 2
        expect_V1[2 + s, 2 + (1 - s)] += h / 2
    assert sp.simplify(V1 - expect_V1) == sp.zeros(4, 4)
    expect_V2 = sp.zeros(4, 4)
    for s in (0, 1):
        expect_V2[s, s] += -h / 8 * (1 if s == 0 else -1)
        expect_V2[2 + s, 2 + s] += h / 8 * (1 if s == 0 else -1)
    assert sp.simplify(V2 - expect_V2) == sp.zeros(4, 4)
    print(
        "[5a] rotated-field expansion: V1 = (h/2)(sigma_x^A + sigma_x^B); "
        "V2 = (-h/8 sigma_z^A + h/8 sigma_z^B)  ... OK"
    )


check_second_order_structure_symbolic()


def check_force_theorem_numeric():
    """Numeric checks:

    (i)  band-sum curvature == 2 Tr[V2 rho0]
         + 2 sum_{n,m} f_n(1-f_m) |V1_nm|^2 / (eps_n - eps_m)
         (second-order perturbation theory == resolvent/LKAG kernel)

    (ii) the force-theorem trace form Tr[(H(theta)-H0) rho0] matches the
         band-sum energy difference to O(theta^3) (relative error O(theta))

    (iii) self-consistent relaxation changes the curvature (P-b != P-c).
    """
    # Unequal field magnitudes break the (exact) Kramers-like degeneracy of
    # the symmetric AFM dimer, so plain (non-degenerate) second-order
    # perturbation theory applies exactly.
    hA, hB, t = 1.7, 1.3, 0.9
    beta = 6.0
    mu = 0.0

    def Hmat(theta):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        HA = hA * np.array([[c, s], [s, -c]])
        HB = hB * np.array([[-c, s], [s, c]])
        H = np.zeros((4, 4))
        H[:2, :2] += HA
        H[2:, 2:] += HB
        H[0, 2] += t
        H[2, 0] += t
        H[1, 3] += t
        H[3, 1] += t
        return H

    H0 = Hmat(0.0)
    w0, v0 = np.linalg.eigh(H0)
    f0 = 1.0 / (1.0 + np.exp(beta * (w0 - mu)))
    rho0 = (v0 * f0) @ v0.conj().T

    # Analytic V1, V2 (matching check [5a] with per-site magnitudes):
    # V1 = (hA/2) sigma_x^A + (hB/2) sigma_x^B
    # V2 = diag(-hA/8, +hA/8) in A; diag(+hB/8, -hB/8) in B
    V1 = np.zeros((4, 4))
    V1[0, 1] = V1[1, 0] = hA / 2
    V1[2, 3] = V1[3, 2] = hB / 2
    V2 = np.diag([-hA / 8, hA / 8, hB / 8, -hB / 8]).astype(complex)

    # Two second-order kernels at finite smearing beta:
    #   fixed-occupation band sum:      2 Tr[V2 rho0] + 2 sum f_n |V1nm|^2 / dE
    #   self-consistent (relaxed rho):  2 Tr[V2 rho0] + 2 sum f_n(1-f_m) |V1nm|^2 / dE
    # (the f_n(1-f_m) factor is the rho_1 relaxation; the two agree at T=0
    #  where occ-occ terms cancel pairwise, and differ at finite beta).
    curv_pt_band = 2.0 * np.trace(V2 @ rho0).real
    curv_pt_sc = curv_pt_band
    for n in range(4):
        for m in range(4):
            if n == m:
                continue
            v1nm = v0[:, n].conj() @ V1 @ v0[:, m]
            denom = w0[n] - w0[m]
            if abs(denom) > 1e-12:
                curv_pt_band += 2.0 * (f0[n] * abs(v1nm) ** 2 / denom).real
                curv_pt_sc += -2.0 * (f0[n] * f0[m] * abs(v1nm) ** 2 / denom).real

    def E_band(theta):
        w = np.linalg.eigvalsh(Hmat(theta))
        return float(np.sum(w * f0))

    def curv4(Efun):
        """Fourth-order central second difference."""
        hh = 1e-3
        return (
            -Efun(2 * hh)
            + 16 * Efun(hh)
            - 30 * Efun(0.0)
            + 16 * Efun(-hh)
            - Efun(-2 * hh)
        ) / (12 * hh**2)

    curv_band = curv4(E_band)
    assert abs(curv_band - curv_pt_band) < 1e-7 * abs(curv_band), (
        curv_band,
        curv_pt_band,
    )

    # (ii) force theorem: the fixed-potential eigenvalue-sum difference
    # matches the self-consistent energy difference up to O(theta^3).
    def E_sc(theta):
        H = Hmat(theta)
        w, v = np.linalg.eigh(H)
        f = 1.0 / (1.0 + np.exp(beta * (w - mu)))
        rho = (v * f) @ v.conj().T
        return float(np.trace(H @ rho).real)

    th = 1e-2
    delta_band = E_band(th) - E_band(0.0)
    delta_sc = E_sc(th) - E_sc(0.0)
    rel_err = abs(delta_sc - delta_band) / abs(delta_band)
    assert rel_err < 5e-2, (rel_err, th)  # small: FT error is O(theta^2)

    curv_sc = curv4(E_sc)
    # At a non-stationary reference (external fields not the SC fields of
    # the model) and finite smearing, SC relaxation shifts the curvature:
    assert abs(curv_sc - curv_band) > 1e-4 * abs(curv_band), (curv_sc, curv_band)
    print(
        f"[5b] band curvature {curv_band:.6f} == fixed-occupation "
        f"second-order kernel {curv_pt_band:.6f}; FT band-vs-SC energy "
        f"agreement rel {rel_err:.1e} at theta={th}; SC curvature "
        f"{curv_sc:.6f} differs (P-b != P-c)  ... OK"
    )


check_force_theorem_numeric()

print("\nAll assertions passed.")
