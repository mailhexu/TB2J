"""Assertion-checked derivation: magnetic force theorem ABOUT a spiral state.

Reference: ring with hopping classes + local fields at lab angles
Theta_a = 2 pi q a in the (z,x)-plane:  B (cosT sigma_z + sinT sigma_x).
Local transverse frame at site a (n_a = field direction):
  - out-of-plane channel d_a: tilt about t_a = y x n_a
    ->  n_a -> n_a cos d_a + y sin d_a        (perturbation  B sigma_y)
  - in-plane channel b_a: rotation about y (spiral-plane rotation)
    ->  theta_a -> theta_a + b_a   (perturbation  d(field)/dtheta)

Central identity (LKAG about a spiral state, eigenbasis form):

    E^(2) = Tr[V2 rho0] + (1/2) sum_{n != m} (f_n - f_m) |V1_nm|^2 / (eps_n - eps_m)

with V1 = sum_a [ d_a B sigma_y + b_a dfield/dtheta ]_a,
     V2 = -(1/2) sum_a [d_a^2 + b_a^2] field_a(0),
frozen occupations f_n from the spiral reference (force theorem).
The antisymmetrized (f_n - f_m)/2 form is identical algebraically and
immune to exact degeneracies: f_n = f_m there, and the intra-pair
in-cluster splitting is first order, canceled by the 4-point FD.

Checks:
  (A) full 2N x 2N local-frame curvature C from exact FD of the
      frozen-occupation band sum == C from the second-order trace.
  (B) exact global-rotation zero mode R_y: uniform in-plane rotation
      is a global unitary -> C_yy @ 1 = 0 exactly, at any q.
  (C) torque structure: the out-of-plane block C_dd has the cos(T_a)
      and sin(T_a) zero modes (global R_x, R_z) when the spiral is
      torque-balanced.  The commensurate q = 1/6 spiral is torque-free
      by symmetry (asserted)
      the grid-minimizing q* is not (reported).
"""

import itertools

import numpy as np
import sympy as sp


def hermitian_classes(N, L, seed):
    rng = np.random.default_rng(seed)
    classes = []
    reps = list(range(1, (N - 1) // 2 + 1))
    if N % 2 == 0:
        reps.append(N // 2)
    for mu, nu in itertools.combinations_with_replacement(range(L), 2):
        for R in reps:
            amp = round(float(rng.normal()), 6) + 0j
            classes.append((mu, nu, R, amp))
            classes.append((nu, mu, -R, amp.conjugate()))
        if mu != nu:
            amp = round(float(rng.normal()), 6) + 0j
            classes.append((mu, nu, 0, amp))
            classes.append((nu, mu, 0, amp.conjugate()))
    for mu in range(L):
        classes.append((mu, mu, 0, 0j))
    eps = [round(rng.normal(), 6) for _ in range(L)]
    Bf = [round(rng.normal(), 6) for _ in range(L)]
    return classes, eps, Bf


def build_field_ring(N, classes, eps, Bf, angles):
    size = 2 * N
    H = np.zeros((size, size), dtype=complex)
    by_res = {}
    for mu, nu, R, amp in classes:
        by_res.setdefault(R % N, []).append((mu, nu, R, amp))
    for a in range(N):
        for rep, entries in by_res.items():
            b = (a + rep) % N
            for _mu, _nu, _R, amp in entries:
                H[2 * a, 2 * b] += amp
                H[2 * a + 1, 2 * b + 1] += amp
    for a in range(N):
        c, s = np.cos(angles[a]), np.sin(angles[a])
        H[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] += Bf[0] * np.array([[c, s], [s, -c]])
    return H


SIG_Y = np.array([[0, -1j], [1j, 0]])


def field_block(B, theta):
    c, s = np.cos(theta), np.sin(theta)
    return B * np.array([[c, s], [s, -c]])


def dfield_dtheta(B, theta):
    c, s = np.cos(theta), np.sin(theta)
    return B * np.array([[-s, c], [c, s]])


def build_perturbed_ring(N, classes, eps, Bf, angles, d_tilt, b_rot):
    """Ring with local-frame transverse rotations (d out-of-plane,
    b in-plane) about the spiral reference at angles[]."""
    H = build_field_ring(N, classes, eps, Bf, angles)
    for a in range(N):
        th = angles[a] + b_rot[a]
        blk = np.cos(d_tilt[a]) * field_block(Bf[0], th).astype(complex)
        blk += Bf[0] * np.sin(d_tilt[a]) * SIG_Y
        H[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] -= field_block(Bf[0], angles[a])
        H[2 * a : 2 * a + 2, 2 * a : 2 * a + 2] += blk
    return H


def check_numeric_spiral_curvature():
    N = 6
    for seed in (71, 72, 73):
        classes, eps, Bf = hermitian_classes(N, 1, seed)
        Bf = [1.5]

        # ---- torque diagnostic at the grid-minimizing q --------------
        def E_of_q(qv):
            ang = 2 * np.pi * qv * np.arange(N)
            w = np.linalg.eigvalsh(build_field_ring(N, classes, eps, Bf, ang))
            mu = float(np.quantile(w, 0.5))
            f = 1.0 / (1.0 + np.exp((w - mu) / 0.05))
            return float(np.sum(w * f))

        qgrid = np.linspace(0, 1, 241, endpoint=False)
        q_star = qgrid[int(np.argmin([E_of_q(q) for q in qgrid]))]

        for q, tag in ((1.0 / 6.0, "commensurate q=1/6"), (q_star, f"q*={q_star:.4f}")):
            angles = 2 * np.pi * q * np.arange(N)
            H0 = build_field_ring(N, classes, eps, Bf, angles)
            w0, c0 = np.linalg.eigh(H0)
            mu0 = float(np.quantile(w0, 0.5))
            f0 = 1.0 / (1.0 + np.exp((w0 - mu0) / 0.05))

            # ---- (A) FD curvature of the frozen-occupation band sum ---
            d = 1e-4

            def E_mft(dv, bv):
                H = build_perturbed_ring(N, classes, eps, Bf, angles, dv, bv)
                return float(np.sum(f0 * np.linalg.eigvalsh(H)))

            def evec(channel, a):
                v = np.zeros((2, N))
                v[channel, a] = 1.0
                return v

            def fd_curvature(vec1, vec2):
                return (
                    E_mft(d * vec1[0] + d * vec2[0], d * vec1[1] + d * vec2[1])
                    - E_mft(d * vec1[0] - d * vec2[0], d * vec1[1] - d * vec2[1])
                    - E_mft(-d * vec1[0] + d * vec2[0], -d * vec1[1] + d * vec2[1])
                    + E_mft(-d * vec1[0] - d * vec2[0], -d * vec1[1] - d * vec2[1])
                ) / (4 * d * d)

            C_fd = np.zeros((2 * N, 2 * N))
            for i in range(2 * N):
                for j in range(i, 2 * N):
                    v = fd_curvature(evec(i // N, i % N), evec(j // N, j % N))
                    C_fd[i, j] = C_fd[j, i] = v
            C_fd = 0.5 * (C_fd + C_fd.T)

            # ---- second-order trace in the spiral eigenbasis ----------
            def E2_trace(dv, bv):
                V1 = np.zeros((2 * N, 2 * N), dtype=complex)
                V2 = np.zeros((2 * N, 2 * N), dtype=complex)
                for a in range(N):
                    sl = slice(2 * a, 2 * a + 2)
                    V1[sl, sl] += dv[a] * Bf[0] * SIG_Y
                    V1[sl, sl] += bv[a] * dfield_dtheta(Bf[0], angles[a])
                    V2[sl, sl] -= (
                        0.5 * (dv[a] ** 2 + bv[a] ** 2) * field_block(Bf[0], angles[a])
                    )
                V1e = c0.conj().T @ V1 @ c0
                V2e = c0.conj().T @ V2 @ c0
                e2 = float(np.sum(f0 * np.real(np.diag(V2e))))
                good = ~np.eye(len(w0), dtype=bool) & (
                    np.abs(w0[:, None] - w0[None, :]) > 1e-10
                )
                den = np.where(good, w0[:, None] - w0[None, :], 1.0)
                df = f0[:, None] - f0[None, :]
                e2 += 0.5 * float(
                    np.sum(np.where(good, df / den, 0.0) * np.abs(V1e) ** 2)
                )
                return e2

            C_tr = np.zeros((2 * N, 2 * N))
            for i in range(2 * N):
                for j in range(i, 2 * N):
                    ei, ej = evec(i // N, i % N), evec(j // N, j % N)
                    v = (
                        E2_trace(ei[0] + ej[0], ei[1] + ej[1])
                        - E2_trace(ei[0] - ej[0], ei[1] - ej[1])
                        - E2_trace(-ei[0] + ej[0], -ei[1] + ej[1])
                        + E2_trace(-ei[0] - ej[0], -ei[1] - ej[1])
                    ) / 4
                    C_tr[i, j] = C_tr[j, i] = v
            C_tr = 0.5 * (C_tr + C_tr.T)

            rel = np.max(np.abs(C_fd - C_tr)) / max(np.max(np.abs(C_fd)), 1e-12)
            # ---- (B) exact R_y zero mode ------------------------------
            scale = max(np.max(np.abs(C_fd)), 1e-12)
            z_y = np.max(np.abs(C_fd[N:, N:].sum(axis=1)))
            # ---- (C) torque structure of the out-of-plane block -------
            D = C_fd[:N, :N]
            z_cos = np.max(np.abs(D @ np.cos(angles)))
            z_sin = np.max(np.abs(D @ np.sin(angles)))
            print(
                f"seed {seed} {tag:22s}  FD-vs-trace rel = {rel:9.2e}   "
                f"|C| = {scale:.4f}  R_y zero = {z_y:7.1e}  "
                f"cos/sin zero = {z_cos / scale:7.1e} {z_sin / scale:7.1e}"
            )
            assert rel < 5e-6, rel
            assert z_y < 1e-6 * max(scale, 1.0), z_y
            if q == 1.0 / 6.0:
                # torque-balanced commensurate spiral: all three global
                # rotations are zero modes of the local-frame curvature
                assert z_cos < 1e-5 * scale and z_sin < 1e-5 * scale, (z_cos, z_sin)
    print(
        "PROBE OK: spiral-state MFT curvature (both local transverse "
        "channels) == second-order trace; global-rotation zero modes "
        "at the torque-balanced reference."
    )


def check_symbolic_local_frame_and_heisenberg():
    """Exact Pauli expansion and pairwise-Heisenberg Hessian signs."""
    B, theta, delta, beta = sp.symbols("B theta delta beta", real=True)
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    field = B * (
        (sp.cos(theta + beta) * sz + sp.sin(theta + beta) * sx) * sp.cos(delta)
        + sy * sp.sin(delta)
    )
    zero = {delta: 0, beta: 0}
    assert sp.simplify(
        field.subs(zero) - B * (sp.cos(theta) * sz + sp.sin(theta) * sx)
    ) == sp.zeros(2)
    assert sp.simplify(sp.diff(field, delta).subs(zero) - B * sy) == sp.zeros(2)
    assert sp.simplify(
        sp.diff(field, beta).subs(zero) - B * (-sp.sin(theta) * sz + sp.cos(theta) * sx)
    ) == sp.zeros(2)
    assert sp.simplify(
        sp.diff(field, delta, 2).subs(zero)
        + B * (sp.cos(theta) * sz + sp.sin(theta) * sx)
    ) == sp.zeros(2)
    assert sp.simplify(
        sp.diff(field, beta, 2).subs(zero)
        + B * (sp.cos(theta) * sz + sp.sin(theta) * sx)
    ) == sp.zeros(2)

    ta, tb, da, db, ba, bb, J = sp.symbols("ta tb da db ba bb J", real=True)
    ea = sp.Matrix(
        [sp.sin(ta + ba) * sp.cos(da), sp.sin(da), sp.cos(ta + ba) * sp.cos(da)]
    )
    eb = sp.Matrix(
        [sp.sin(tb + bb) * sp.cos(db), sp.sin(db), sp.cos(tb + bb) * sp.cos(db)]
    )
    energy = -J * (ea.dot(eb))
    at0 = {da: 0, db: 0, ba: 0, bb: 0}
    # Off-diagonal local-frame Hessian: both channels carry the minus sign.
    assert sp.simplify(sp.diff(energy, da, db).subs(at0) + J) == 0
    assert sp.simplify(sp.diff(energy, ba, bb).subs(at0) + J * sp.cos(ta - tb)) == 0
    assert sp.simplify(sp.diff(energy, da, bb).subs(at0)) == 0
    print("[1] symbolic local-frame V1/V2 and Heisenberg Hessian signs ... OK")


def _fd_block(N, classes, eps, Bf, angles, f0, channel_a, channel_b, step=1e-4):
    def energy(dv, bv):
        H = build_perturbed_ring(N, classes, eps, Bf, angles, dv, bv)
        return float(np.sum(f0 * np.linalg.eigvalsh(H)))

    def basis(channel, atom):
        v = np.zeros((2, N))
        v[channel, atom] = 1.0
        return v

    block = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            x, y = basis(channel_a, a), basis(channel_b, b)
            block[a, b] = (
                energy(step * (x[0] + y[0]), step * (x[1] + y[1]))
                - energy(step * (x[0] - y[0]), step * (x[1] - y[1]))
                - energy(step * (-x[0] + y[0]), step * (-x[1] + y[1]))
                + energy(-step * (x[0] + y[0]), -step * (x[1] + y[1]))
            ) / (4 * step * step)
    return 0.5 * (block + block.T) if channel_a == channel_b else block


def check_q0_anchor_and_mapping():
    N = 6
    classes, eps, Bf = hermitian_classes(N, 1, 72)
    Bf = [1.5]
    angles = np.zeros(N)
    H0 = build_field_ring(N, classes, eps, Bf, angles)
    w0 = np.linalg.eigvalsh(H0)
    mu = float(np.quantile(w0, 0.5))
    f0 = 1 / (1 + np.exp((w0 - mu) / 0.05))
    Cdd = _fd_block(N, classes, eps, Bf, angles, f0, 0, 0)
    Cbb = _fd_block(N, classes, eps, Bf, angles, f0, 1, 1)
    Cdb = _fd_block(N, classes, eps, Bf, angles, f0, 0, 1)
    assert np.max(np.abs(Cdd - Cbb)) < 1e-6
    assert np.max(np.abs(Cdb)) < 1e-6
    # M of the collinear derivation is exactly this in-plane block.
    Jpair = -Cdd.copy()
    np.fill_diagonal(Jpair, 0.0)
    assert np.max(np.abs(np.diag(Cdd) - np.sum(Jpair, axis=1))) < 2e-6
    print("[3] q=0 anchor: Cdd=Cbb=M, Cdb=0; J_ab=-Cdd_ab ... OK")


def check_toth_lake_flat_screw():
    J0, Jk, Jq, Jkp, Jkm, S = sp.symbols("J0 Jk Jq Jkp Jkm S", real=True)
    A = Jq - (Jkp + Jkm) / 2
    C = Jq - Jk
    omega2 = sp.expand(S**2 * A * C)
    # k=0: J(k+q)=J(k-q)=J(q); k=q: J(k)=J(q).
    assert sp.simplify(omega2.subs({Jkp: Jq, Jkm: Jq, Jk: J0})) == 0
    assert sp.simplify(omega2.subs({Jk: Jq})) == 0
    # q=0 and inversion-even J: A=C=J(0)-J(k).
    fm = sp.simplify(omega2.subs({Jq: J0, Jkp: Jk, Jkm: Jk}) - S**2 * (J0 - Jk) ** 2)
    assert fm == 0
    print("[4] Toth-Lake flat-screw zeros at k=0,+/-q and FM limit ... OK")


def main():
    check_symbolic_local_frame_and_heisenberg()
    check_numeric_spiral_curvature()
    check_q0_anchor_and_mapping()
    check_toth_lake_flat_screw()
    print("\nAll spiral-state MFT assertions passed.")


if __name__ == "__main__":
    main()
