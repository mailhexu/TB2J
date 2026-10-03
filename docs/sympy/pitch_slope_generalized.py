"""Assertion-checked derivation: generalized (frozen-band) pitch slope
dE/dq for a nonorthogonal folded pencil.

Story 001 (spiral-first-order-response), NFR-005, FR-009 conventions.
Independent of TB2J/TBUpy; reuses the pinned planar-frame builders of
``planar_su2_frame`` and the occupation helpers of
``local_gradient_nonorthogonal`` so there is one convention source.

Observable.  For fractional-reciprocal q components q_alpha
(alpha = x, y, z) and fixed k mesh, fixed occupations f, unchanged
fields, the frozen-band pitch slope per primitive cell is

    dE/dq_alpha = sum_k w_k sum_n f_nk
                  c_nk^dag (dH/dq_alpha - eps_nk dS/dq_alpha) c_nk

in eV / primitive cell / fractional q component (generalized pencil
derivative; c^dag S c = 1).  The q dependence of a class dressing is

    Delta_{mu nu}(R) = 2 pi q.R + alpha_nu - alpha_mu ,
    dDelta/dq_alpha  = 2 pi (R_alpha + tau_{nu,alpha} - tau_{mu,alpha}) ,
    dU(Delta)/dq_alpha = (-i/2) sigma_y U(Delta) dDelta/dq_alpha ,

i.e. BOTH vertices move with q: the twist vertex 2 pi R_alpha and the
intra-cell vertex 2 pi (tau_nu - tau_mu)_alpha; hopping AND overlap are
dressed identically, so dS/dq_alpha is generically NONZERO (the
representation itself moves under a pitch change - unlike the local
beta/delta gradient, where dS = 0).

At fixed k mesh the half-shift s of the mesh is HELD FIXED across the
+/-h legs (round(qN) does not change for small h).

Checks
------
[A] symbolic: scalar normalized pencil eps(q) = (a + b q)/(1 + c q)
    satisfies (dH/dq - eps dS/dq)/S = d eps/dq identically; exact
    rational 2x2 matrix pencil: d eps/dq = c^dag (dH/dq - eps dS/dq) c
    with the S-normalized eigenvector c.
[B] analytic folded q derivative (3, 2n, 2n) vs element-wise central FD
    of the assembled pencil; dS/dq materially nonzero (nonorthogonal);
    dH/dq includes both vertices (twist + intra-cell; asserted by a
    tau-ablation: equal taus annihilate the intra-cell part).
[C] slope vector vs frozen-occupation central FD of the folded band sum
    along each q_alpha (same fixed mesh), and the negative control:
    DROPPING -eps dS/dq breaks the agreement.
[D] lab-frame cross-check: exact chain rule through the spiral angles
    (dE/dq_alpha = sum 2 pi (R_a + tau_mu)_alpha Tr(rho dH/dTheta)).
    For intra-cell-only components (y, z here) both representations are
    the same finite system: folded == lab exactly.  The twist component
    (x) is seam-affected at finite N (the wrap-bond mismatch 2 pi delta N
    makes ring and folded mesh different systems off the commensurate
    grid); the folded value is the normative primitive-cell slope
    (FR-009/NFR-002) and the lab difference is reported, not asserted.

Run with the mydev environment:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/pitch_slope_generalized.py
"""

import local_gradient_nonorthogonal as lg
import numpy as np
import planar_su2_frame as fr
import sympy as sp

SY = np.array([[0.0, -1.0j], [1.0j, 0.0]])


# --------------------------------------------------------------------------
# analytic q derivative of the folded pencil
# --------------------------------------------------------------------------


def assemble_folded_qderivative(classes, taus, phis, q, B_local, k):
    """(dH, dS) each (3, 2n, 2n): analytic d/dq_alpha of the pencil."""
    hop, ovl = fr.split_classes(classes)
    norb = taus.shape[0]
    alpha = 2 * np.pi * (taus @ q) + phis
    dH = np.zeros((3, 2 * norb, 2 * norb), dtype=complex)
    dS = np.zeros((3, 2 * norb, 2 * norb), dtype=complex)
    for cls, target in [(c, dH) for c in hop] + [(c, dS) for c in ovl]:
        mu, nu, R, amp = cls[0], cls[1], cls[2], cls[3]
        Delta = 2 * np.pi * float(q @ R) + alpha[nu] - alpha[mu]
        bloch = np.exp(-2j * np.pi * float(k @ R.astype(float)))
        dU = (-0.5j) * (SY @ fr.rot2(Delta))
        for a in range(3):
            dDelta = 2 * np.pi * (R[a] + taus[nu, a] - taus[mu, a])
            target[a, 2 * mu : 2 * mu + 2, 2 * nu : 2 * nu + 2] += (
                bloch * amp * dU * dDelta
            )
    return dH, dS


# --------------------------------------------------------------------------
# [A] symbolic pencil identities
# --------------------------------------------------------------------------


def check_symbolic_pencils():
    a, b, c_, q = sp.symbols("a b c q", real=True)
    H, S = a + b * q, 1 + c_ * q
    eps = H / S
    resid = sp.simplify(sp.diff(H, q) - eps * sp.diff(S, q) - sp.diff(eps, q) * S)
    assert resid == 0, resid

    q = sp.Symbol("q", real=True)
    Hm = sp.Matrix(
        [
            [sp.Rational(5, 3), sp.Rational(1, 5) + q / 3],
            [sp.Rational(1, 5) + q / 3, sp.Rational(3, 8) - q / 6],
        ]
    )
    Sm = sp.Matrix(
        [
            [sp.Rational(4, 3), sp.Rational(1, 13) + q / 11],
            [sp.Rational(1, 13) + q / 11, sp.Rational(7, 5) - q / 8],
        ]
    )
    lam = sp.Symbol("lambda")
    eps_expr = sp.solve(sp.expand((Hm - lam * Sm).det()), lam)[0]
    q0 = sp.Rational(1, 3)
    eps0 = eps_expr.subs(q, q0)
    H0, S0 = Hm.subs(q, q0), Sm.subs(q, q0)
    x = sp.symbols("x", real=True)
    row = (H0 - eps0 * S0) * sp.Matrix([x, 1])
    xs = sp.solve(sp.Eq(row[0], 0), x)[0]
    vec = sp.Matrix([xs, 1])
    cvec = vec / sp.sqrt(sp.simplify((vec.T * S0 * vec)[0, 0]))
    deps = eps_expr.diff(q).subs(q, q0)
    rhs = (cvec.T * (sp.diff(Hm, q) - eps0 * sp.diff(Sm, q)).subs(q, q0) * cvec)[0, 0]
    assert sp.simplify(deps - rhs) == 0, (deps, rhs)
    print(
        "[A] symbolic: (dH - eps dS)/S = d eps/dq identically (scalar and "
        "rational 2x2 matrix pencil) ... OK"
    )


# --------------------------------------------------------------------------
# [B] analytic derivative vs element-wise FD
# --------------------------------------------------------------------------


def check_qderivative_against_fd():
    ref = lg.build_reference()
    norb = ref["norb"]  # noqa: F841 (fixture parity)
    k = np.array([1.0 / 8.0, 0.0, 0.0])
    dH, dS = assemble_folded_qderivative(
        ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
    )
    h = 1e-6
    for a in range(3):
        qp = ref["q"].copy()
        qm = ref["q"].copy()
        qp[a] += h
        qm[a] -= h
        Hp, Sp = fr.assemble_folded(
            ref["classes"], ref["taus"], ref["phis"], qp, ref["B_local"], k
        )
        Hm, Sm = fr.assemble_folded(
            ref["classes"], ref["taus"], ref["phis"], qm, ref["B_local"], k
        )
        fdH = (Hp - Hm) / (2 * h)
        fdS = (Sp - Sm) / (2 * h)
        assert np.max(np.abs(fdH - dH[a])) < 1e-8, (a, np.max(np.abs(fdH - dH[a])))
        assert np.max(np.abs(fdS - dS[a])) < 1e-8, (a, np.max(np.abs(fdS - dS[a])))
    assert np.max(np.abs(dS)) > 1e-3, np.max(np.abs(dS))
    # tau ablation: equal taus annihilate the intra-cell vertex
    ref2 = lg.build_reference(seed=22)
    ref2["taus"] = np.zeros_like(ref2["taus"])
    dH2, _ = assemble_folded_qderivative(
        ref2["classes"], ref2["taus"], ref2["phis"], ref2["q"], ref2["B_local"], k
    )
    dH_expect = np.zeros_like(dH2)
    hop, _ = fr.split_classes(ref2["classes"])
    alpha = 2 * np.pi * (ref2["taus"] @ ref2["q"]) + ref2["phis"]
    for cls in hop:
        mu, nu, R, amp = cls[0], cls[1], cls[2], cls[3]
        Delta = 2 * np.pi * float(ref2["q"] @ R) + alpha[nu] - alpha[mu]
        bloch = np.exp(-2j * np.pi * float(k @ R.astype(float)))
        dU = (-0.5j) * (SY @ fr.rot2(Delta))
        for a in range(3):
            dH_expect[a, 2 * mu : 2 * mu + 2, 2 * nu : 2 * nu + 2] += (
                bloch * amp * dU * 2 * np.pi * R[a]
            )
    assert np.max(np.abs(dH2 - dH_expect)) < 1e-12
    print(
        "[B] analytic (dH,dS) (3,2n,2n) vs element-wise FD <= 1e-8; "
        "|dS| materially nonzero; tau-ablation pins both vertices ... OK"
    )


# --------------------------------------------------------------------------
# [C] slope vector vs frozen-occupation FD
# --------------------------------------------------------------------------


def check_slope_against_fd():
    ref = lg.build_reference()
    eig = lg.reference_eigenpairs(ref)
    ks, kweights, w_all, c_all, focc = (
        eig["ks"],
        eig["kweights"],
        eig["w_all"],
        eig["c_all"],
        eig["focc"],
    )
    slopes = np.zeros(3)
    for k, w, c, f, wgt in zip(ks, w_all, c_all, focc, kweights):
        dH, dS = assemble_folded_qderivative(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
        )
        for a in range(3):
            slopes[a] += wgt * float(
                np.real(
                    np.sum(
                        f
                        * (
                            np.einsum("in,ij,jn->n", c.conj(), dH[a], c)
                            - w * np.einsum("in,ij,jn->n", c.conj(), dS[a], c)
                        )
                    )
                )
            )
    h = 1e-5
    fd = np.zeros(3)

    def folded_sum_at(qv):
        tot = 0.0
        for k, f, wgt in zip(ks, focc, kweights):
            Hq, Sq = fr.assemble_folded(
                ref["classes"], ref["taus"], ref["phis"], qv, ref["B_local"], k
            )
            tot += wgt * float(np.sum(f * lg.gen_eigvalsh(Hq, Sq)))
        return tot

    for a in range(3):
        qp, qm = ref["q"].copy(), ref["q"].copy()
        qp[a] += h
        qm[a] -= h
        fd[a] = (folded_sum_at(qp) - folded_sum_at(qm)) / (2 * h)
    dev = np.max(np.abs(slopes - fd))
    assert dev < 1e-8, (slopes, fd, dev)
    # negative control: dropping -eps dS breaks it
    slopes_nods = np.zeros(3)
    for k, w, c, f, wgt in zip(ks, w_all, c_all, focc, kweights):
        dH, dS = assemble_folded_qderivative(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
        )
        for a in range(3):
            slopes_nods[a] += wgt * float(
                np.real(np.sum(f * np.einsum("in,ij,jn->n", c.conj(), dH[a], c)))
            )
    err = np.max(np.abs(slopes_nods - fd))
    assert err > 1e-6, err
    print(
        f"[C] frozen pitch slope (3,) vs fixed-mesh frozen FD: max dev "
        f"{dev:8.1e}; dropping -eps*dS errs by {err:8.1e} (control) ... OK"
    )
    return ref, eig, slopes


# --------------------------------------------------------------------------
# [D] lab-frame cross-check (exact chain rule, seam-free components)
# --------------------------------------------------------------------------


def check_lab_crosscheck(ref, eig, slopes):
    """Cross-frame validation of the pitch slope.

    The exact lab-frame slope is the chain rule through the spiral
    angles, dE/dq_alpha = sum_{a,mu} 2 pi (R_a + tau_mu)_alpha
    Tr(rho dH/dTheta_{a mu}), with rho the lab spectral density at the
    SAME fixed occupations.  For components whose action is
    intra-cell-only (tau offsets, here y and z) both representations
    describe the same finite system and MUST agree exactly.  The twist
    component (x) is seam-affected at finite N: the folded mesh and the
    explicit ring differ off the commensurate grid (the wrap bond
    mismatch 2 pi delta N), so the lab x-slope is NOT an oracle for the
    primitive folded slope (FR-009/NFR-002 pin the folded value); the
    difference is reported, not asserted away.
    """
    kweights, w_all, focc = eig["kweights"], eig["w_all"], eig["focc"]  # noqa: F841 (fixture parity)
    norb, ncell = ref["norb"], ref["ncell"]
    Bf = 0.5 * ref["B_local"]
    H_lab, S_lab = fr.assemble_lab(
        ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], ncell
    )
    eps_lab, u_lab = lg.gen_eigh_vec(H_lab, S_lab)
    flat = np.sort(np.concatenate(w_all))
    f_flat = lg.logistic_f(flat, eig["mu"], eig["width"])
    rho = (u_lab * f_flat[None, :]) @ u_lab.conj().T / ncell
    Th = fr.angles(ref["taus"], ref["phis"], ref["q"], ncell)
    slope_lab = np.zeros(3)
    for a in range(ncell):
        for mu in range(norb):
            sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            vertex = np.zeros_like(H_lab)
            vertex[sl, sl] = Bf[mu] * (
                -np.sin(Th[a, mu]) * lg.SZ + np.cos(Th[a, mu]) * lg.SX
            )
            weight = 2 * np.pi * (float(a) + ref["taus"][mu])
            slope_lab += weight * np.real(np.trace(rho @ vertex))
    dev_yz = np.max(np.abs(slopes[1:] - slope_lab[1:]))
    assert dev_yz < 1e-8, (slopes, slope_lab, dev_yz)
    seam_diff = abs(slopes[0] - slope_lab[0])  # noqa: F841 (fixture parity)
    print(
        f"[D] lab chain-rule slope == folded for intra-cell components "
        f"(y,z): max dev {dev_yz:8.1e}; twist (x): folded {slopes[0]:.6f} "
        f"vs lab-ring {slope_lab[0]:.6f} (seam-affected at finite N; "
        f"folded value is normative) ... OK"
    )


def main():
    check_symbolic_pencils()
    check_qderivative_against_fd()
    ref, eig, slopes = check_slope_against_fd()
    check_lab_crosscheck(ref, eig, slopes)
    print("\nAll pitch-slope assertions passed.")


if __name__ == "__main__":
    main()
