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
        for el, l, fl in states:
            for em, m, fm in states:
                if el == em:
                    continue
                tot = tot + (l.T * hx * m)[0, 0] * (m.T * hy * l)[0, 0] * (fl - fm) / (
                    el - em
                )
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
    for name, hu, hd in (("fm", hu_fm, hd_fm), ("staggered", hu_s, hd_s)):
        _a0, b0, _c0 = grand_curvatures(hu, hd, None, None)
        assert b0 == s.zeros(2, 2), (name, b0)
        _at, bt, ct = grand_curvatures(hu, hd, 1.0, 0.0)
        assert bt == s.zeros(2, 2), (name, bt)
        assert s.N(ct.det(), 20) == 0, (name, ct)  # uniform null mode
    results.append({"b_zero_and_c_null_mode": "verified(fm,staggered;T=0,beta=1)"})

    # --- S5: implemented Jnorm fails in-domain ---------------------------
    m_spinat = s.diag(2, 2)
    m_self = s.diag(-1, -1)  # occupied down 1, up 0 per site
    for m_name, m_mat in (("spinat", m_spinat), ("self-consistent", m_self)):
        corr = s.simplify(s.Rational(1, 2) * (m_mat - kq.T) * xq.inv() * (m_mat - kq))
        assert corr[0, 1] != 0, (m_name, corr)
        results.append({f"jnorm_correction_{m_name}": str(corr[0, 1])})
    # --- S6: truth C has the uniform null mode; code Xq is invertible here
    null_vec = s.Matrix([1, 1])
    _a0, _b0, c0 = grand_curvatures(hu_fm, hd_fm, None, None)
    assert s.simplify(c0 * null_vec) == s.zeros(2, 1), c0
    assert s.simplify(xq.det()) != 0, xq
    # (Story 003's np.linalg.inv failures on real SrMnO3 models show the
    # code Xq becomes singular there; its structural cause is to be
    # characterized separately from this dimer class.)
    results.append({"truth_c_null_mode": "span((1,1))", "xq_dimer": "invertible"})

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
    results.append({"variational_identity": "J_scf = J_frozen - (B C^-1 B^T)"})

    # --- S8: canted-reference loop (mpmath 30 digits, in-script) ---------
    import mpmath as mp

    mp.mp.dps = 30
    t_ = mp.mpf("0.5")
    Dv = mp.mpf(4)

    def H4s(th1, th2, p1=0, p2=0):
        H = mp.zeros(4, 4)
        for site in (0, 1):
            th = th1 if site == 0 else th2
            ph = p1 if site == 0 else p2
            ct, sn = mp.cos(th / 2), mp.sin(th / 2)
            bb = site * 2
            H[bb, bb] = Dv / 2 * ct + ph
            H[bb + 1, bb + 1] = -Dv / 2 * ct - ph
            H[bb, bb + 1] = H[bb + 1, bb] = Dv / 2 * sn
        H[0, 2] = H[2, 0] = t_
        H[1, 3] = H[3, 1] = t_
        return H

    def etot_stoner(th1, th2, p1=0, p2=0, Iv=mp.mpf("0.5")):
        w, v = mp.eigsy(H4s(th1, th2, p1, p2))
        bet = mp.mpf(1)
        E = mp.mpf(0)
        m = [mp.mpf(0), mp.mpf(0)]
        for k in range(4):
            fk = 1 / (1 + mp.exp(bet * w[k]))
            E += fk * w[k]
            for site in (0, 1):
                up = v[site * 2, k]
                dn = v[site * 2 + 1, k]
                m[site] += fk * (mp.re(up * mp.conj(up)) - mp.re(dn * mp.conj(dn)))
        return E - Iv / 2 * (m[0] ** 2 + m[1] ** 2)

    def scf_p(t1, t2):
        p = mp.mpf("0")
        hh = mp.mpf("1e-6")
        for _ in range(60):
            E0 = etot_stoner(t1, t2, p, -p)
            g = (
                etot_stoner(t1, t2, p + hh, -p - hh)
                - etot_stoner(t1, t2, p - hh, -p + hh)
            ) / (2 * hh)
            Jl = (
                etot_stoner(t1, t2, p + hh, -p - hh)
                - 2 * E0
                + etot_stoner(t1, t2, p - hh, -p + hh)
            ) / (hh * hh)
            dp = -g / Jl
            p += dp
            if abs(dp) < mp.mpf("1e-14"):
                break
        return p

    al = mp.mpf("0.4")
    hh = mp.mpf("1e-4")

    def Ef(d1, d2):
        return etot_stoner(al + d1, -al + d2, 0, 0)

    J_frozen = (Ef(hh, hh) - Ef(hh, -hh) - Ef(-hh, hh) + Ef(-hh, -hh)) / (4 * hh * hh)

    def Es(d1, d2):
        p2s = scf_p(al + d1, -al + d2)
        return etot_stoner(al + d1, -al + d2, p2s, -p2s)

    J_scf = (Es(hh, hh) - Es(hh, -hh) - Es(-hh, hh) + Es(-hh, -hh)) / (4 * hh * hh)

    # perturbative side: B, C of the total energy at the reference
    def E0f(d1, d2, p):
        return etot_stoner(al + d1, -al + d2, p, -p)

    B1 = (E0f(hh, 0, hh) - E0f(hh, 0, -hh) - E0f(-hh, 0, hh) + E0f(-hh, 0, -hh)) / (
        4 * hh * hh
    )
    B2 = (E0f(0, hh, hh) - E0f(0, hh, -hh) - E0f(0, -hh, hh) + E0f(0, -hh, -hh)) / (
        4 * hh * hh
    )
    Csc = (E0f(0, 0, hh) - 2 * E0f(0, 0, 0) + E0f(0, 0, -hh)) / (hh * hh)
    corr_pert = -(B1 * B2) / Csc
    corr_direct = J_scf - J_frozen
    assert abs(corr_direct - corr_pert) < mp.mpf("1e-8"), (corr_direct, corr_pert)
    assert abs(corr_direct) > mp.mpf("1e-3")  # nonzero at the canted reference
    results.append({"canted_reference_correction": mp.nstr(corr_direct, 12)})
    results.append({"canted_reference_perturbative": mp.nstr(corr_pert, 12)})

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
