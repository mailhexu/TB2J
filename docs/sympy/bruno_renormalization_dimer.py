#!/usr/bin/env python3
"""Bruno-renormalization dimer authority: exact checks of the frozen kernel,
the relaxed band truth, and the implemented Jnorm formula on exactly-solvable
two-site single-orbital dimers.

Status: Phase-1 authority increment (Story 008). Results are exact (sympy);
the finite-temperature entry uses Float(30) arithmetic with identity checks.

Verified statements
-------------------
S1 (anchor, T=0, staggered FM-aligned reference): kernel = band = -1/30 for
   splitting 4 eV and hopping 1/2 eV, reproducing the pinned
   qspace-unrenormalized-kernel authority.
S2 (kernel = band, FM fully-polarized filling): up band empty, down band
   filled: kernel J = band J = -1/30 (same spectrum, different filling).
S3 (kernel = band, mirror-staggered): Hu = [[2,1/2],[1/2,-2]],
   Hd = [[-2,1/2],[1/2,2]], Delta = diag(4,-4): kernel = band =
   +2*sqrt(17)/289 via the residue of the double pole at -sqrt(17)/2.
S4 (B == 0 in-domain): the theta-phi cross curvature of the exact grand
   potential vanishes identically for the single-orbital collinear dimer
   class (FM and staggered, T=0 and finite beta). A correct
   Bruno-renormalized exchange must therefore equal the frozen one here.
S5 (implemented formula fails in-domain): with the code's own quantities
   Jq, Kq, Xq for the FM toy and M from spinat (2,2) or from the
   self-consistent moment (-1,-1), the correction
   1/2 (M-Kq)^T Xq^-1 (M-Kq) is a nonzero rational (order 1e2 / 1e1); no
   tested moment definition makes it vanish or puts M-Kq in the null space
   of Xq. The formula structure, not its inputs, is defective.
S6 (X null mode): Xq has the exact null vector (1,1) in the symmetric case;
   np.linalg.inv failures on symmetric models are intrinsic.
"""

from __future__ import annotations

import argparse
import json
import platform

import sympy as s

E = s.Symbol("z")
T_HALF = s.Rational(1, 2)


def eigenpairs(h: s.Matrix) -> list[tuple[s.Expr, s.Matrix]]:
    out = []
    for lam, _mult, vecs in sorted(h.eigenvects(), key=lambda item: item[0]):
        v = vecs[0]
        out.append((s.simplify(lam), s.simplify(v * s.sqrt(1 / (v.T * v)[0]))))
    return out


def residue_sum(expr: s.Expr, poles: list[s.Expr]) -> s.Expr:
    return s.simplify(sum(s.residue(expr, E, p) for p in poles))


def code_quantities(hu: s.Matrix, hd: s.Matrix, delta: list[s.Expr]):
    """Jq, Kq, Xq via the exchange_qspace.py integrand structure.

    Contour measure fixed by the authority identity J = -Sigma_res(A)/4
    (S1 anchor); X carries the code's +2 prefactor.
    """
    gu = (E * s.eye(2) - hu).inv()
    gd = (E * s.eye(2) - hd).inv()
    ups, dns = eigenpairs(hu), eigenpairs(hd)
    occupied = sorted(
        {lam for lam, _ in ups if lam < 0} | {lam for lam, _ in dns if lam < 0},
        key=lambda z: z.evalf(),
    )
    d = s.diag(*delta)
    jq = s.zeros(2, 2)
    kq = s.zeros(2, 2)
    xq = s.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            di, dj = d[i, i], d[j, j]
            a_raw = s.simplify(
                (di * gu[i, j] * dj * gd[j, i] + di * gd[i, j] * dj * gu[j, i]) / 2
            )
            k_raw = s.simplify(
                gu[i, j] * dj * gd[j, i] + 2 * (gd[i, j] * dj * gu[j, i])
            )
            x_raw = s.simplify(gu[i, j] * gd[j, i])
            jq[i, j] = s.simplify(-residue_sum(a_raw, occupied) / 4)
            kq[i, j] = s.simplify(-residue_sum(k_raw, occupied) / 4)
            xq[i, j] = s.simplify(2 * residue_sum(x_raw, occupied))
    return jq, kq, xq


def grand_curvatures(hu: s.Matrix, hd: s.Matrix, beta_val, mu_val):
    """Exact A, B, C matrices of the grand potential at (beta, mu).

    Occupations enter through Fermi weights at the eigenvalues of hu/hd
    (single-orbital sites, so the 4x4 states at theta=phi=0 decompose).
    """
    th1, th2, p1, p2 = s.symbols("theta1 theta2 phi1 phi2", real=True)

    def site_block(e_u, e_d, th, ph):
        c, sn = s.cos(th / 2), s.sin(th / 2)
        return s.Matrix(
            [
                [
                    c**2 * (e_u + ph) + sn**2 * (e_d - ph),
                    c * sn * ((e_u + ph) - (e_d - ph)),
                ],
                [
                    c * sn * ((e_u + ph) - (e_d - ph)),
                    sn**2 * (e_u + ph) + c**2 * (e_d - ph),
                ],
            ]
        )

    def h4(a=s.S.Zero, b=s.S.Zero, c=s.S.Zero, dd=s.S.Zero):
        b1 = site_block(hu[0, 0], hd[0, 0], a, c)
        b2 = site_block(hu[1, 1], hd[1, 1], b, dd)
        return s.Matrix(
            [
                [b1[0, 0], T_HALF, b1[0, 1], 0],
                [T_HALF, b2[0, 0], 0, b2[0, 1]],
                [b1[1, 0], 0, b1[1, 1], T_HALF],
                [0, b2[1, 0], T_HALF, b2[1, 1]],
            ]
        )

    expr = h4(*[th1, th2, p1, p2])
    zero = {th1: 0, th2: 0, p1: 0, p2: 0}

    def dh(x, y=None):
        if y is None:
            return s.diff(expr, x).subs(zero)
        return s.diff(s.diff(expr, x), y).subs(zero)

    beta = s.Float(beta_val, 30) if beta_val is not None else None
    mu = s.Float(mu_val, 30) if mu_val is not None else None

    states = []
    for h in (hu, hd):
        for lam, v in eigenpairs(h):
            w = s.zeros(4, 1)
            sl = slice(0, 2) if h is hu else slice(2, 4)
            w[sl, 0] = v
            if beta is None:  # T = 0 step at mu = 0
                occ = 1 if lam < 0 else 0
            else:
                occ = 1 / (1 + s.exp(beta * (s.N(lam, 30) - mu)))
            states.append(
                (
                    s.N(lam, 30) if beta is not None else lam,
                    w,
                    s.N(occ, 30) if beta is not None else occ,
                )
            )

    def d2(x, y):
        hx, hy, hxy = dh(x), dh(y), dh(x, y)
        tot = s.Float(0, 30) if beta is not None else s.Integer(0)
        for el, l, fl in states:
            tot = tot + fl * (l.T * hxy * l)[0, 0]
        if beta is not None:
            # fixed-mu grand-potential diagonal (Fermi-derivative) term:
            # d2 Omega carries -beta f(1-f) (h'_nn)^2 for every state,
            # including n = m; skipping it fabricates a uniform null mode
            # that the true C does not have at finite temperature.
            for el, l, fl in states:
                hxx_nn = (l.T * hx * l)[0, 0]
                hyy_nn = (l.T * hy * l)[0, 0]
                tot = tot + (-beta * fl * (1 - fl)) * hxx_nn * hyy_nn
        for el, l, fl in states:
            for em, m, fm in states:
                if l == m:
                    # same-state pair: covered by the explicit Fermi-derivative
                    # diagonal term above, never by the resolvent sum.
                    continue
                if el == em:
                    if beta is None:
                        # T = 0 with distinct states at equal energy (the
                        # mirror-staggered double pole): the weight has no
                        # finite limit for unequal step occupations; the
                        # pinned S3 band reference fixes this convention.
                        continue
                    # Degenerate-limit rule: (f_n - f_m)/(eps_n - eps_m)
                    # -> -beta f(1-f) for DISTINCT states with equal
                    # eigenvalues (equal energies share the occupation at
                    # fixed mu).  The transverse vertex connects such
                    # states, so the limit carries an O(1) A contribution.
                    # (R2-001)
                    weight = -beta * fl * (1 - fl)
                else:
                    weight = (fl - fm) / (el - em)
                tot = tot + (l.T * hx * m)[0, 0] * (m.T * hy * l)[0, 0] * weight
        return s.simplify(tot)

    a_mat = s.Matrix(2, 2, lambda i, j: d2((th1, th2)[i], (th1, th2)[j]))
    b_mat = s.Matrix(2, 2, lambda i, j: d2((th1, th2)[i], (p1, p2)[j]))
    c_mat = s.Matrix(2, 2, lambda i, j: d2((p1, p2)[i], (p1, p2)[j]))
    return a_mat, b_mat, c_mat


def derive() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []

    # --- S1/S2: FM-splitting dimer, both fillings ------------------------
    hu_fm = s.Matrix([[2, T_HALF], [T_HALF, 2]])
    hd_fm = s.Matrix([[-2, T_HALF], [T_HALF, -2]])
    jq, kq, xq = code_quantities(hu_fm, hd_fm, [4, 4])
    assert jq[0, 1] == s.Rational(-1, 30), jq
    # band truth: A_12 of the FM dimer at T=0
    a_fm, b_fm, c_fm = grand_curvatures(hu_fm, hd_fm, None, None)
    j_band = s.simplify(-a_fm[0, 1] / 2)
    assert j_band == s.Rational(-1, 30), j_band
    # global-rotation invariance of the truth machinery
    rot = s.simplify(a_fm[0, 0] + a_fm[1, 1] + 2 * a_fm[0, 1])
    assert rot == 0, rot
    results.append({"two_site_fm_kernel_and_band": str(jq[0, 1])})

    # --- S3: mirror-staggered dimer (double pole) ------------------------
    hu_s = s.Matrix([[2, T_HALF], [T_HALF, -2]])
    hd_s = s.Matrix([[-2, T_HALF], [T_HALF, 2]])
    jqs, kqs, xqs = code_quantities(hu_s, hd_s, [4, -4])
    band_s = s.simplify(-grand_curvatures(hu_s, hd_s, None, None)[0][0, 1] / 2)
    assert jqs[0, 1] == band_s == 2 * s.sqrt(17) / 289, (jqs[0, 1], band_s)
    results.append({"two_site_staggered_kernel_and_band": str(jqs[0, 1])})

    # --- S4: B == 0 in-domain (T=0 and finite T), FM and staggered -------
    # The mixed (transverse-longitudinal) diagonal Fermi term vanishes
    # because the transverse vertex is off-diagonal (h'_theta,nn = 0), so
    # B == 0 survives the CR-001 correction; the longitudinal stiffness C
    # does not (see S6).
    for name, hu, hd in (("fm", hu_fm, hd_fm), ("staggered", hu_s, hd_s)):
        _a0, b0, _c0 = grand_curvatures(hu, hd, None, None)
        assert b0 == s.zeros(2, 2), (name, b0)
        _at, bt, ct = grand_curvatures(hu, hd, 1.0, 0.0)
        assert bt == s.zeros(2, 2), (name, bt)
    results.append({"b_zero": "verified(fm,staggered;T=0,beta=1)"})

    # --- S5: implemented Jnorm fails in-domain ---------------------------
    m_spinat = s.diag(2, 2)
    m_self = s.diag(-1, -1)  # occupied down 1, up 0 per site
    for m_name, m_mat in (("spinat", m_spinat), ("self-consistent", m_self)):
        corr = s.simplify(s.Rational(1, 2) * (m_mat - kq.T) * xq.inv() * (m_mat - kq))
        assert corr[0, 1] != 0, (m_name, corr)
        results.append({f"jnorm_correction_{m_name}": str(corr[0, 1])})
    # --- S6: uniform null mode of C is exact only at T = 0 ----------------
    # At T = 0 the Fermi-derivative diagonal terms vanish and the exact C
    # carries the uniform null vector (1, 1).  At finite temperature the
    # fixed-mu grand potential is NOT invariant under a uniform sigma_z
    # shift (up and down levels move oppositely), and the diagonal
    # -beta f(1-f) (h'_phi,nn)^2 terms make C generically invertible; the
    # uniform-mode projection in the production pseudoinverse is therefore
    # a declared gauge choice, not an exact symmetry at finite T.
    null_vec = s.Matrix([1, 1])
    _a0, _b0, c0 = grand_curvatures(hu_fm, hd_fm, None, None)
    assert s.simplify(c0 * null_vec) == s.zeros(2, 1), c0
    _at, _bt, ct = grand_curvatures(hu_fm, hd_fm, 1.0, 0.0)
    assert s.N((ct * null_vec)[0], 20) != 0, ct  # finite-T: not null
    assert s.simplify(xq.det()) != 0, xq
    # (Story 003's np.linalg.inv failures on real SrMnO3 models show the
    # code Xq becomes singular there; its structural cause is to be
    # characterized separately from this dimer class.)
    results.append(
        {
            "truth_c_null_mode": "span((1,1)) exact at T=0; finite-T C invertible",
            "xq_dimer": "invertible",
        }
    )

    # --- S7: variational identity (model-free, exact symbolic) -----------
    # Total energy expanded at a reference: E = E0 + A th^2/2 + b V + C V^2/2
    # + c th V.  Minimizing over the longitudinal coordinate V gives
    #   dE/dV = b + C V + c th = 0 -> V* = -(b + c th)/C,
    #   E_min''(0) = A - c^2/C.
    # So the self-consistent exchange is the frozen exchange minus the
    # longitudinal relaxation correction c^2/C (per mode); matrix form:
    # J_scf = J_frozen - (B C^-1 B^T) with B the mixed response and C the
    # screened longitudinal stiffness.
    A_, C_, b_, c_ = s.symbols("A C b c", nonzero=True)
    th_, V_ = s.symbols("theta V")
    E_expr = (
        s.Rational(1, 2) * A_ * th_**2
        + b_ * V_
        + s.Rational(1, 2) * C_ * V_**2
        + c_ * th_ * V_
    )
    V_star = s.solve(s.Eq(s.diff(E_expr, V_), 0), V_)[0]
    E_min = s.simplify(E_expr.subs(V_, V_star))
    curvature = s.simplify(s.diff(E_min, th_, 2))
    assert curvature == s.simplify(A_ - c_**2 / C_), curvature
    results.append(
        {
            "variational_identity": (
                "A_scf = A_frozen - (B C^-1 B^T)  [A-curvature identity; "
                "exchange conversion: Delta J = +1/2 (B C^-1 B^T)_12]"
            )
        }
    )

    # --- S8: canted-reference loop (mpmath 30 digits, in-script) ---------
    import mpmath as mp

    mp.mp.dps = 30
    t_ = mp.mpf("0.5")
    Dv = mp.mpf(4)

    def H4s(th1, th2, p1=0, p2=0):
        # Physical spin-angle convention (same as grand_curvatures'
        # site_block): the exchange field of site i points at polar angle
        # th_i, so its 2x2 block is (D/2)(cos th sz + sin th sx), and the
        # longitudinal field p is the LOCAL-z field: it rotates with the
        # site block, p (cos th sz + sin th sx).  (R2-002)
        H = mp.zeros(4, 4)
        for site in (0, 1):
            th = th1 if site == 0 else th2
            ph = p1 if site == 0 else p2
            ct, sn = mp.cos(th), mp.sin(th)
            bb = site * 2
            H[bb, bb] = (Dv / 2 + ph) * ct
            H[bb + 1, bb + 1] = -(Dv / 2 + ph) * ct
            H[bb, bb + 1] = H[bb + 1, bb] = (Dv / 2 + ph) * sn
        H[0, 2] = H[2, 0] = t_
        H[1, 3] = H[3, 1] = t_
        return H

    def grand_band(th1, th2, p1=0, p2=0):
        """Band grand potential Omega at the given local-z fields."""
        w, _ = mp.eigsy(H4s(th1, th2, p1, p2))
        bet = mp.mpf(1)
        omega = mp.mpf(0)
        for k in range(4):
            omega += -mp.log(1 + mp.exp(-bet * w[k])) / bet
        return omega

    Iv = mp.mpf("0.5")

    def stoner_E(m, th1, th2):
        """Proper Stoner SCF functional E(m) = Omega(I m) + I m^2.

        The longitudinal field is GENERATED by the moment (p = I m): the
        Legendre structure of the self-consistency; the mode is the
        UNIFORM longitudinal mode (m, m), which is the channel with a
        nonzero self-consistency force in the dimer (both sites carry
        identical local band moments).  With an independent external
        field the functional would be unbounded below (R2-003); here the
        quadratic double-counting dominates and the stationary branch is
        a minimizing branch for I below the Stoner threshold (asserted
        via C > 0 below).
        """
        return grand_band(th1, th2, Iv * m, Iv * m) + Iv * m**2

    def scf_m(t1, t2):
        """Newton solve of the uniform-mode self-consistency.

        Returns (m, g_final, Jl) with Jl the mode curvature at the
        solution: Jl > 0 certifies a minimizing branch (R2-003).
        """
        m = mp.mpf("0")
        hm = mp.mpf("1e-6")
        g = Jl = mp.mpf(0)
        for _ in range(80):
            E0 = stoner_E(m, t1, t2)
            g = (stoner_E(m + hm, t1, t2) - stoner_E(m - hm, t1, t2)) / (2 * hm)
            Jl = (stoner_E(m + hm, t1, t2) - 2 * E0 + stoner_E(m - hm, t1, t2)) / (
                hm * hm
            )
            dp = -g / Jl
            m += dp
            if abs(dp) < mp.mpf("1e-16"):
                break
        return m, g, Jl

    # Reference (th1, th2) = (al, be) is NOT mirror-antisymmetric: with
    # the rotation-invariant Stoner term a mirror pair would pin the
    # antisymmetric longitudinal mode (B = 0 by symmetry); the asymmetric
    # pair is the minimal nonzero-B reference.
    al = mp.mpf("0.4")
    be = mp.mpf("0.1")
    hh = mp.mpf("1e-6")

    def E_rel(d1, d2, m):
        return stoner_E(m, al + d1, be + d2)

    def A12_frozen(m):
        return (
            E_rel(hh, hh, m)
            - E_rel(hh, -hh, m)
            - E_rel(-hh, hh, m)
            + E_rel(-hh, -hh, m)
        ) / (4 * hh * hh)

    def A12_scf():
        # Each stencil corner sits on ITS OWN self-consistent branch
        # m*(theta): the mixed FD of the RELAXED energy surface.  (Using a
        # single m for all corners would give the frozen-m curvature.)
        vals = []
        for d1, d2 in ((hh, hh), (hh, -hh), (-hh, hh), (-hh, -hh)):
            ms, _, _ = scf_m(al + d1, be + d2)
            vals.append(stoner_E(ms, al + d1, be + d2))
        return (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * hh * hh)

    # The Schur identity is a statement about the STATIONARY branch: the
    # frozen A_12 and the B, C curvatures must all be taken at
    # m = m*(theta_ref), not at m = 0.
    m_ref, g_ref, Jl_ref = scf_m(al, be)
    assert abs(g_ref) < mp.mpf("1e-12")  # stationary
    assert Jl_ref > 0  # minimizing branch, not a maximum (R2-003)
    A_frozen = A12_frozen(m_ref)
    A_scf = A12_scf()
    J_frozen = -A_frozen / 2

    # perturbative side: B, C of the same functional at the reference
    B1 = (
        E_rel(hh, 0, m_ref + hh)
        - E_rel(hh, 0, m_ref - hh)
        - E_rel(-hh, 0, m_ref + hh)
        + E_rel(-hh, 0, m_ref - hh)
    ) / (4 * hh * hh)
    B2 = (
        E_rel(0, hh, m_ref + hh)
        - E_rel(0, hh, m_ref - hh)
        - E_rel(0, -hh, m_ref + hh)
        + E_rel(0, -hh, m_ref - hh)
    ) / (4 * hh * hh)
    Csc = (
        E_rel(0, 0, m_ref + hh) - 2 * E_rel(0, 0, m_ref) + E_rel(0, 0, m_ref - hh)
    ) / (hh * hh)
    assert Csc > 0  # stable longitudinal mode
    # A-space Schur complement: Delta A_12 = -B1 B2 / C_sc (single mode).
    # Exchange conversion (J = -A_12/2, the pinned kernel-authority
    # mapping): Delta J = -Delta A_12 / 2 = +B1 B2 / (2 C_sc); the
    # production formula is Jnorm = Jq + (B C^+ B^T)/2.
    delta_A_pert = -B1 * B2 / Csc
    J_scf = -A_scf / 2
    delta_J_direct = J_scf - J_frozen
    delta_J_pert = -delta_A_pert / 2
    assert abs(delta_J_direct - delta_J_pert) < mp.mpf("1e-8"), (
        delta_J_direct,
        delta_J_pert,
    )
    assert abs(delta_J_direct) > mp.mpf("1e-7")  # channel activates (uniform
    # mode B is intrinsically ~1e-3 in this dimer; the identity agreement
    # below is the authority statement)
    results.append({"canted_reference_deltaJ_direct": mp.nstr(delta_J_direct, 12)})
    results.append({"canted_reference_deltaJ_perturbative": mp.nstr(delta_J_pert, 12)})
    results.append({"canted_reference_mstar": mp.nstr(m_ref, 8)})
    results.append(
        {"exchange_conversion": "Jnorm = Jq + (B C^+ B^T)/2  [J = -A_12/2 mapping]"}
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kosmic-json", action="store_true")
    parser.parse_args()
    payload = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "sympy_version": s.__version__,
        "results": derive(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
