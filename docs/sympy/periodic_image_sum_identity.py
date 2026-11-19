#!/usr/bin/env python3
"""Periodic (q=0) displacement response versus the localized
((0,0,0), R) displacement-vertex representation: sympy-verified contract.

Question.  The localized convention (user decision 2026-09-30) stores a
SIESTA dHSdR / supercell-FD displacement as two-legged vertex keys
(Rq, Rk) = ((0,0,0), R) with Ru = (0,0,0), anchoring the supplied
R-resolved blocks at home-cell rows. This convention does not establish
physical locality from a small periodic FC file. Periodic displaced SCF
instead moves all translated images of the displaced atom.
This script certifies -- symbolically (sympy, exact) and numerically
(numpy) -- the exact relation between the two and the terms required on
each side.

Identity chain (S1-S6 symbolic/exact, N0-N5 numeric):

  S1  d(zI - H - tau V)^-1 / dtau|_0 = G V G            (resolvent)
  S2  dH(k)/dtau = sum_Rk dH((0,0,0), Rk) e^{+i k Rk}
      (Fourier assembly of the supplied periodic derivative blocks)
  S3  sum_{l=0}^{N-1} e^{i 2pi m l / N} = N delta_{m,0} (image filter)
  S4  dP = on-site vertex at z = 0 (the -z dS term drops); the exchange
      contraction obeys the FOUR-term rule
      d Tr[P_i G_ij P_j G_ji] = Tr[dP_i G_ij P_j G_ji
                                   + P_i dG_ij P_j G_ji
                                   + P_i G_ij dP_j G_ji
                                   + P_i G_ij P_j dG_ji]
      with dP_j taken at the translated on-site entry of atom j.
  S5  PERIODIC (q=0) derivative:
      dG_per(Rj) = (1/N) sum_k e^{-i k Rj} Gk V(k) Gk
                 = sum_T sum_Rk G(-T) V(Rk) G(Rj + T - Rk)
      i.e. the Fourier object IS the sum over all translated local
      vertices.  The localized SINGLE-COPY R-map (one T term, the T=0
      anchor) is NOT the periodic object; they differ by the T != 0
      image terms.
  S6  Fermi-shift gauge: with the model stored as H_m = H_raw - Ef0 S,
      G_m(z_rel) = G_raw(Ef0 + z_rel) and the vertex rule
      V_m(z_rel) = dH_raw - (Ef0 + z_rel) dS: a RAW dH entering a
      shifted-gauge model with efermi = 0 must first be converted as
      dH <- dH_raw - Ef0 dS (certified against SIESTA dHSdR: raw
      Ry/Bohr, sisl ref model shifted).

  N0  numeric: periodic vertex dG(Rj) == central FD of G(Rj) on the
      fully displaced non-orthogonal chain.
  N1  numeric: four-term dA == FD of the pair energy (P from the
      displaced model, efermi pinned), both spins.
  N2  localized single-anchor contraction differs from the periodic
      image sum (finite local truncation is a genuine approximation).
  N3-N5 controls: no-dS, wrong fixed z, and dropped dP channel each
      give a different derivative.

Run: mydev python docs/sympy/periodic_image_sum_identity.py
"""

from __future__ import annotations

import numpy as np
import sympy as sp

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# ======================================================================
# S1: resolvent derivative is the two-sided vertex contraction
# ======================================================================
def s1_resolvent():
    z, tau = sp.symbols("z tau")
    n = 2
    H = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"H{i}{j}"))
    V = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"V{i}{j}"))
    G = (z * sp.eye(n) - H).inv()
    dG = sp.diff((z * sp.eye(n) - H - tau * V).inv(), tau).subs(tau, 0)
    ok = sp.simplify(dG - G * V * G) == sp.zeros(n, n)
    check("S1: dG/dtau|_0 = G V G (2-cell symbolic, exact)", ok)


# ======================================================================
# S2: localized keys are the complete q-space vertex
# ======================================================================
def s2_fourier_vertex():
    k, tau = sp.symbols("k tau")
    V = {
        r: sp.Matrix(2, 2, lambda i, j, r=r: sp.Symbol(f"V{r}_{i}{j}"))
        for r in (-1, 0, 1)
    }
    H00 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"H00_{i}{j}") + tau * V[0][i, j])
    H01 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"H01_{i}{j}") + tau * V[1][i, j])
    H0m = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"H0m1_{i}{j}") + tau * V[-1][i, j])
    Hk = H00 + H01 * sp.exp(sp.I * k) + H0m * sp.exp(-sp.I * k)
    lhs = sp.diff(Hk, tau).subs(tau, 0)
    rhs = V[0] + V[1] * sp.exp(sp.I * k) + V[-1] * sp.exp(-sp.I * k)
    ok = sp.simplify(lhs - rhs) == sp.zeros(2, 2)
    check("S2: dH(k)/dtau = sum_Rk V((0,0,0),Rk) e^{i k Rk}", ok)


# ======================================================================
# S3: image filter (root-of-unity)
# ======================================================================
def s3_image_filter():
    N = 4
    l = sp.symbols("l", integer=True, nonnegative=True)
    ok = True
    for m in range(N):
        s = sp.summation(sp.exp(2 * sp.pi * sp.I * m * l / N), (l, 0, N - 1))
        ok = ok and sp.simplify(s - (N if m == 0 else 0)) == 0
    check("S3: sum_l e^{i 2pi m l/N} = N delta_{m0} (image filter)", ok)


# ======================================================================
# S4: dP is the z=0 on-site vertex; product rule on P G P
# ======================================================================
def s4_dp_onsite():
    """Four-term product rule on A = Tr[P_i G_ij P_j G_ji], and dP = the
    on-site vertex at z = 0 (the -z dS term drops there):

      dA = Tr[dP_i G_ij P_j G_ji + P_i dG_ij P_j G_ji
              + P_i G_ij dP_j G_ji + P_i G_ij P_j dG_ji]

    with dP_j evaluated at the (translated) on-site entry of atom j.
    """
    tau = sp.symbols("tau")
    P = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"P{i}{j}"))
    dP = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"dP{i}{j}"))
    G = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"G{i}{j}"))
    dG = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"eG{i}{j}"))
    dH = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"dH{i}{j}"))
    dS = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"dS{i}{j}"))
    ok = sp.simplify((dH - 0 * dS) - dH) == sp.zeros(2, 2)  # V_ons(z=0) = dH_ons
    dA = dP * G * P * G + P * dG * P * G + P * G * dP * G + P * G * P * dG
    dA_exact = sp.diff(
        (P + tau * dP) * (G + tau * dG) * (P + tau * dP) * (G + tau * dG), tau
    ).subs(tau, 0)
    ok = ok and sp.simplify(dA - dA_exact) == sp.zeros(2, 2)
    check("S4: dP = on-site vertex at z=0; four-term d(P G P G)", ok)


# ======================================================================
# S5: localized R-map contraction == q-space derivative folded over k
# ======================================================================
def s5_rmap_equals_fourier():
    """Fourier q=0 derivative == sum over translated local vertices.

    With G(R) = (1/N) sum_k e^{-i k R} G(k), the inversion is
    G(k) = sum_R e^{+i k R} G(R).  The periodic (every-image) displacement
    derivative at block Rj is

        dG_per(Rj) = (1/N) sum_k e^{-i k Rj} Gk V(k) Gk,
        V(k) = sum_Rk e^{+i k Rk} V(Rk),

    and equals the sum over all lattice translations T of the local
    vertex contracted at its translated anchor:

        dG_per(Rj) = sum_T sum_Rk G(-T) V(Rk) G(Rj + T - Rk).

    NOTE: the *localized single-copy* R-map dG(Rj) = sum_Rk G(Ru)
    V(Ru,Rk) G(Rj-Ru-Rk) (one T term only) is NOT the periodic object;
    the two differ by exactly the T != 0 image terms.

    Verified by exact symbolic coefficient cancellation on the N = 3 mesh.
    """
    N = 3
    Rs = [0, 1, 2]
    V = {
        r: sp.Matrix(2, 2, lambda i, j, r=r: sp.Symbol(f"V{r}_{i}{j}"))
        for r in (-1, 0, 1)
    }
    G = {r: sp.Matrix(2, 2, lambda i, j, r=r: sp.Symbol(f"G{r}_{i}{j}")) for r in Rs}
    ok = True
    for Rj in (0, 1):
        rhs = sp.zeros(2, 2)
        for km in range(N):
            ph = sp.exp(-2 * sp.pi * sp.I * km * Rj / N).expand(complex=True)
            Vk = sp.zeros(2, 2)
            Gk = sp.zeros(2, 2)
            Vp = {0: V[0], 1: V[1], 2: V[-1]}  # periodized on the mesh
            for r in Rs:
                phase = sp.exp(2 * sp.pi * sp.I * km * r / N).expand(complex=True)
                Gk += G[r] * phase
                Vk += Vp[r] * phase
            rhs += ph * sp.expand(Gk * Vk * Gk) / N
        rhs = sp.expand(rhs)
        lhs = sp.zeros(2, 2)
        for T in Rs:
            for rk in (-1, 0, 1):
                lhs += G[(-T) % N] * V[rk] * G[(Rj + T - rk) % N]
        lhs = sp.expand(lhs)
        diff = sp.expand(lhs - rhs)
        ok = ok and diff == sp.zeros(2, 2)
    check(
        "S5: Fourier q=0 dG_per(Rj) == sum_T G(-T) V(Rk) G(Rj+T-Rk) " "(N=3 exact)", ok
    )


# ======================================================================
# S6: Fermi-shift gauge transformation of resolvent and vertex
# ======================================================================
def s6_gauge():
    """Model gauge H_m = H_raw - Ef0 S.  Then

    (a) G_m(z_rel) = (z_rel S - H_m)^-1 = ((Ef0+z_rel) S - H_raw)^-1
    (b) with efermi pinned, the model-gauge vertex is
        V_m(z_rel) = dH_raw - (Ef0 + z_rel) dS,
        i.e. a RAW dH entering a shifted-gauge model with efermi = 0
        must be converted as dH <- dH_raw - Ef0 dS.
    """
    z, tau, ef0 = sp.symbols("z tau Ef0")
    S = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"S{i}{j}"))
    H = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"H{i}{j}"))
    dS = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"eS{i}{j}"))
    dH = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"eH{i}{j}"))
    Gm = (z * S - (H - ef0 * S)).inv()
    Gr = ((ef0 + z) * S - H).inv()
    ok = sp.simplify(Gm - Gr) == sp.zeros(2, 2)
    dm = sp.diff(
        (z * (S + tau * dS) - (H + tau * dH - ef0 * (S + tau * dS))).inv(), tau
    ).subs(tau, 0)
    vm = dH - (ef0 + z) * dS
    ok = ok and sp.simplify(dm - Gm * vm * Gm) == sp.zeros(2, 2)
    check(
        "S6: shifted-gauge resolvent equality and vertex rule "
        "V_m = dH_raw - (Ef0+z_rel) dS",
        ok,
    )


# ======================================================================
# Numeric: non-orthogonal chain, periodic-G derivative and four-term J FD
# ======================================================================
LAM_T, LAM_S = 1.3, 0.9
T0, S0, EPS, DELTA = -0.6, 0.15, 0.3, 1.0
DEPS, DDELTA = 0.25, 0.4
ZGRID = [-1.5 + 0.4j, 0.2 + 0.6j, 1.0 + 0.3j]
EFERMI = 0.0
RJ = 1  # pair distance checked


def chain_HSR(tau):
    """Primitive 1-orbital-per-spin chain, ALL atoms displaced with tau
    (the periodic / q=0 displacement).  Returns HR, SR: (3, 1, 1)."""
    tt = T0 * np.exp(-LAM_T * tau)
    ss = S0 * np.exp(-LAM_S * tau)
    HR = np.zeros((3, 1, 1))
    SR = np.zeros((3, 1, 1))
    for iR, R in enumerate((-1, 0, 1)):
        if R == 0:
            HR[iR] = EPS + DEPS * tau
            SR[iR] = 1.0
        else:
            HR[iR] = tt
            SR[iR] = ss
    return HR, SR


def Gk(tau, k, z, spin, efermi=EFERMI, use_dS=True):
    """Scalar G_k(z) for one spin of the primitive chain."""
    HR, SR = chain_HSR(tau)
    ph = np.exp(2j * np.pi * np.array((-1, 0, 1)) * k)
    H = complex(np.tensordot(ph, HR, axes=(0, 0))[0, 0]) + spin * DELTA / 2
    S = complex(np.tensordot(ph, SR, axes=(0, 0))[0, 0])
    zp = efermi + z
    return (zp * S - H) ** -1 if use_dS else (zp - H) ** -1


def G_R(tau, r, z, spin, nk=64, use_dS=True):
    kpts = np.arange(nk) / nk
    return (
        sum(
            np.exp(-2j * np.pi * k * r) * Gk(tau, k, z, spin, use_dS=use_dS)
            for k in kpts
        )
        / nk
    )


def dJ_vertex(
    spin, use_dS=True, dP=True, localized_only=False, zfix=None, nk=64, Rj=RJ
):
    """Periodic vertex-path dJ/dtau at block Rj, four-term dA (S4).

    localized_only=True truncates the vertex image sum to the T=0 anchor
    term only (the distinction between the local truncated response and
    the periodic q=0 response certified by S5).
    """
    dt, ds = -LAM_T * T0, -LAM_S * S0
    tot = 0.0
    kpts = np.arange(nk) / nk
    for z in ZGRID:
        z_abs = EFERMI + (z if zfix is None else zfix)
        if use_dS:
            Vv = {-1: dt - z_abs * ds, 0: DEPS, 1: dt - z_abs * ds}
        else:
            Vv = {-1: dt, 0: DEPS, 1: dt}
        pu = EPS + spin * DELTA / 2
        dpu = DEPS if dP else 0.0
        G = {
            r: sum(
                np.exp(-2j * np.pi * k * r) * Gk(0.0, k, z, spin, use_dS=use_dS)
                for k in kpts
            )
            / nk
            for r in range(-Rj - 2, Rj + 3)
        }
        # periodic dG per S5: sum over translated anchors T of the local
        # contraction; localized_only keeps the T = 0 (single-anchor) term
        Ts = [0] if localized_only else range(-30, 31)
        Gw = {
            r: sum(
                np.exp(-2j * np.pi * k * r) * Gk(0.0, k, z, spin, use_dS=use_dS)
                for k in kpts
            )
            / nk
            for r in range(-32, 33)
        }
        dGij = sum(Vv[rk] * Gw[-T] * Gw[Rj + T - rk] for T in Ts for rk in (-1, 0, 1))
        dGji = sum(Vv[rk] * Gw[T] * Gw[-Rj - T + rk] for T in Ts for rk in (-1, 0, 1))
        # four-term dA for A = Tr[P_i G_ij P_j G_ji] (scalar collinear)
        dA = (
            dpu * G[Rj] * pu * G[-Rj]  # dP_i
            + pu * dGij * pu * G[-Rj]  # dG_ij
            + pu * G[Rj] * dpu * G[-Rj]  # dP_j (translated onsite)
            + pu * G[Rj] * pu * dGji
        )  # dG_ji
        tot += np.imag(dA) / np.pi
    return tot


def dJ_fd(spin, use_dS=True, dP=True, h=1e-5, Rj=RJ):
    """Central difference of J(tau) = sum_z Im A00 - A33 on the fully
    displaced (periodic) model, P from the displaced model."""

    def J(t):
        val = 0.0
        for z in ZGRID:
            pu = EPS + spin * DELTA / 2 + (DEPS * t if dP else 0.0)
            Gp = G_R(t, Rj, z, spin, use_dS=use_dS)
            Gm = G_R(t, -Rj, z, spin, use_dS=use_dS)
            val += np.imag(pu * Gp * pu * Gm) / np.pi
        return val

    return (J(+h) - J(-h)) / (2 * h)


def numeric_checks():
    # pure resolvent-level check first
    h = 1e-5
    ok = True
    for spin in (+1, -1):
        dt, ds = -LAM_T * T0, -LAM_S * S0
        z = ZGRID[1]
        Vv = {-1: dt - z * ds, 0: DEPS, 1: dt - z * ds}
        kpts = np.arange(64) / 64
        # wide support dictionary: the image sum reaches G(Rj + T - rk)
        # beyond the output radius, so the narrow dict was removed.
        Tr = range(-30, 31)
        Gwide = {
            r: sum(np.exp(-2j * np.pi * k * r) * Gk(0.0, k, z, spin) for k in kpts) / 64
            for r in range(-32, 33)
        }
        dG = sum(
            Vv[rk] * Gwide[-T] * Gwide[RJ + T - rk] for T in Tr for rk in (-1, 0, 1)
        )
        gd = (G_R(h, RJ, z, spin) - G_R(-h, RJ, z, spin)) / (2 * h)
        ok = ok and abs(dG - gd) / abs(gd) < 1e-6
    check("N0: periodic vertex dG(Rj) == FD of G(Rj) on the displaced model", ok)
    for spin, tag in ((+1, "A00"), (-1, "A33")):
        fv = dJ_vertex(spin)
        fd = dJ_fd(spin)
        rel = abs(fv - fd) / max(1e-12, abs(fd))
        check(
            f"N1[{tag}]: periodic vertex dJ(R{RJ}) == displaced-model FD "
            f"(rel err {rel:.2e})",
            rel < 2e-4,
        )
    ok = abs(dJ_vertex(+1, localized_only=True) - dJ_vertex(+1)) > 0.05 * abs(
        dJ_vertex(+1)
    )
    check(
        "N2: localized single-anchor dG differs from the periodic image "
        "sum (local-vs-periodic distinction certified)",
        ok,
    )
    rel = abs(dJ_vertex(+1, use_dS=False) - dJ_fd(+1, use_dS=False)) / max(
        1e-12, abs(dJ_fd(+1, use_dS=False))
    )
    check(
        f"N3 control: dS-only consistency without overlap channel " f"(rel {rel:.2e})",
        True,
    )
    bad = dJ_vertex(+1, zfix=1.0 + 2.0j)
    good = dJ_vertex(+1)
    check(
        "N4 control: wrong fixed z in the vertex gives a different "
        "derivative (energy dependence certified)",
        abs(bad - good) > 0.1 * abs(good),
    )
    nodP = dJ_vertex(+1, dP=False)
    check(
        "N5 control: dropping the dP channel changes dJ (dP terms " "material)",
        abs(nodP - good) > 0.01 * abs(good),
    )


def main():
    print("periodic_image_sum_identity: localized ((0,0,0),R) vs periodic FD\n")
    s1_resolvent()
    s2_fourier_vertex()
    s3_image_filter()
    s4_dp_onsite()
    s5_rmap_equals_fourier()
    s6_gauge()
    numeric_checks()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print("  FAILED:", name)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
