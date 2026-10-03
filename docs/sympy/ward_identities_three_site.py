"""Assertion-checked derivation: field-amplitude/sign conventions and
nonstationary Ward identities on a three-site pair-model oracle.

Story 001 (spiral-first-order-response), TEST-002 / FR-006 / FR-007.
Independent of TB2J/TBUpy; reuses the pinned builders of
``planar_su2_frame`` and the reference helpers of
``local_gradient_nonorthogonal`` so there is one convention source.

Pair model (classical, pair-once).  With planar moments

    e_i(beta_i, delta_i; Theta_i) =
        (sin(Theta_i + beta_i) cos(delta_i), sin(delta_i),
         cos(Theta_i + beta_i) cos(delta_i)),

E = -sum_{i<j} J_ij e_i . e_j obeys, at beta = delta = 0:

    g_i^beta  ==  sum_{j != i} J_ij sin(Theta_i - Theta_j),
    g_i^delta ==  0,
    C^{dd}_{ij} == -J_ij,   C^{bb}_{ij} == -J_ij cos(Theta_i-Theta_j),
    C^{db}_{ij} == 0                       (i != j; diagonals from cos),

the nonstationary (gradient-corrected) Ward identities

    C^{bb} 1 == 0,
    C^{dd} sin(Theta) == cos(Theta) .* g^beta,
    C^{dd} cos(Theta) == -sin(Theta) .* g^beta,

the rank deficiency of the first-gradient J fit (three gradients, two
free couplings: sum_i g_i^beta = 0 fixes the null direction 1), and the
ordered-pair conversion of TB2J's SpinIO.exchange_Jdict (each ordered
pair carries HALF the pair-once J):

    g_i^beta == -2 sum_{j != i} J^ord_ij sin(Theta_i - Theta_j).

An inversion-symmetric J1-J2 chain has ZERO local torque at every pitch
while dE/dq can be nonzero: a global boundary twist is not a local
rotation.

Electronic oracle.  A three-site electronic tight-binding ring with
spin-scalar hopping and co-rotating local fields Bf (cosT sz + sinT sx)
is globally SU(2) invariant, which yields the functional-general
gradient-corrected scalar Ward identities (V2 tip-back included in the
frozen FD curvature):

    R_y:  C^{bb} 1 == 0,
    R_x:  cosT^T C^{dd} cosT + sum_j g_j^beta sinT_j cosT_j == 0,
    R_z:  sinT^T C^{dd} sinT - sum_j g_j^beta sinT_j cosT_j == 0,

with the generator decompositions v(R_x)_j = -cosT_j delta_j,
a(R_x)_j = sinT_j cosT_j beta_j; v(R_z)_j = sinT_j delta_j,
a(R_z)_j = -sinT_j cosT_j beta_j.  Only at a stationary reference
(g = 0) do these reduce to plain Hessian zero modes.

Checks
------
[A] symbolic three-site pair model: gradient, Hessian blocks, Ward
    identities, J-fit rank 2, ordered-pair factor 2, flat-chain
    zero-torque / nonzero dE/dq.
[B] numeric finite differences of the same classical model (independent
    numeric oracle): FD gradient and FD curvature match the symbolic
    C blocks and satisfy the Ward identities.
[C] electronic three-site ring: frozen-band g (analytic) + frozen FD
    curvature; Cbb.1 = 0 and the R_x / R_z scalar Ward identities;
    field-amplitude convention (splitting of Bf n.sigma = B_local).

Run with the mydev environment:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/ward_identities_three_site.py
"""

import local_gradient_nonorthogonal as lg
import numpy as np
import planar_su2_frame as fr
import sympy as sp

SX = lg.SX
SY = lg.SY
SZ = lg.SZ


# --------------------------------------------------------------------------
# [A] symbolic three-site pair model
# --------------------------------------------------------------------------


def symbolic_pair_model():
    t = sp.symbols("t0:3", real=True)
    b = sp.symbols("b0:3", real=True)
    d = sp.symbols("d0:3", real=True)
    j = sp.symbols("j01 j02 j12", real=True)
    e = [
        sp.Matrix(
            [
                sp.sin(t[i] + b[i]) * sp.cos(d[i]),
                sp.sin(d[i]),
                sp.cos(t[i] + b[i]) * sp.cos(d[i]),
            ]
        )
        for i in range(3)
    ]
    pairs = [(0, 1), (0, 2), (1, 2)]
    energy = -sum(j[p] * e[i].dot(e[k]) for p, (i, k) in enumerate(pairs))
    zero = {**{bi: 0 for bi in b}, **{di: 0 for di in d}}
    g = sp.Matrix([sp.trigsimp(sp.diff(energy, b[i]).subs(zero)) for i in range(3)])
    g_expected = sp.Matrix(
        [
            sum(
                j[pairs.index((min(i, k), max(i, k)))] * sp.sin(t[i] - t[k])
                for k in range(3)
                if k != i
            )
            for i in range(3)
        ]
    )
    assert all(sp.trigsimp(v) == 0 for v in g - g_expected)
    assert all(sp.simplify(sp.diff(energy, d[i]).subs(zero)) == 0 for i in range(3))
    hdd = sp.Matrix(
        3, 3, lambda i, k: sp.trigsimp(sp.diff(energy, d[i], d[k]).subs(zero))
    )
    hbb = sp.Matrix(
        3, 3, lambda i, k: sp.trigsimp(sp.diff(energy, b[i], b[k]).subs(zero))
    )
    hdb = sp.Matrix(
        3, 3, lambda i, k: sp.trigsimp(sp.diff(energy, d[i], b[k]).subs(zero))
    )
    s_vec = sp.Matrix([sp.sin(x) for x in t])
    c_vec = sp.Matrix([sp.cos(x) for x in t])
    assert all(sp.trigsimp(v) == 0 for v in hbb * sp.ones(3, 1))
    assert all(sp.trigsimp(v) == 0 for v in hdb.reshape(9, 1))
    for i, k in pairs:
        ji = j[pairs.index((i, k))]
        assert sp.trigsimp(hdd[i, k] + ji) == 0
        assert sp.trigsimp(hbb[i, k] + ji * sp.cos(t[i] - t[k])) == 0
    assert all(sp.trigsimp(v) == 0 for v in hdd * s_vec - sp.diag(*c_vec) * g)
    assert all(sp.trigsimp(v) == 0 for v in hdd * c_vec + sp.diag(*s_vec) * g)
    # rank of the J -> g map: 3 gradients but rank 2
    jac = g.jacobian(sp.Matrix(j)).subs({t[0]: 0, t[1]: sp.pi / 3, t[2]: sp.pi / 2})
    assert jac.rank() == 2 and len(jac.nullspace()) == 1
    # ordered-pair conversion: J_pair = 2 J_ord
    jo = sp.symbols("jo01 jo02 jo12", real=True)
    g_ord = sp.Matrix(
        [
            sum(
                2 * jo[pairs.index((min(i, k), max(i, k)))] * sp.sin(t[i] - t[k])
                for k in range(3)
                if k != i
            )
            for i in range(3)
        ]
    )
    assert all(
        sp.trigsimp(v) == 0
        for v in g.subs({j[0]: 2 * jo[0], j[1]: 2 * jo[1], j[2]: 2 * jo[2]}) - g_ord
    )
    # inversion-symmetric J1-J2 chain: zero local torque, nonzero dE/dq
    q, J1, J2 = sp.symbols("q J1 J2", real=True)
    torque = J1 * (sp.sin(-q) + sp.sin(q)) + J2 * (sp.sin(-2 * q) + sp.sin(2 * q))
    assert sp.trigsimp(torque) == 0
    e_per_cell = -J1 * sp.cos(q) - J2 * sp.cos(2 * q)
    assert (
        sp.simplify(
            sp.diff(e_per_cell, q).subs({q: sp.pi / 3, J1: 1, J2: 0}) - sp.sqrt(3) / 2
        )
        == 0
    )
    print(
        "[A] three-site pair model: g_beta formula, g_delta=0, C blocks, "
        "Ward identities (gradient-corrected), J-fit rank 2, ordered-pair "
        "factor 2, flat-chain zero torque with nonzero dE/dq ... OK"
    )
    return t, j, pairs, g, hdd, hbb


def numeric_pair_oracle():
    """FD of the classical model confirms the symbolic blocks."""
    t, j, pairs, g, hdd, hbb = symbolic_pair_model()
    subs = {
        **{t[0]: 0.31, t[1]: 1.12, t[2]: 2.05},
        **{j[0]: 0.8, j[1]: -0.45, j[2]: 0.6},
    }
    th = np.array([float(t[i].subs(subs)) for i in range(3)])
    J = np.array([float(j[k].subs(subs)) for k in range(3)])
    g_exact = np.array([float(g[i].subs(subs)) for i in range(3)])
    hdd_exact = np.array(
        [[float(hdd[i, k].subs(subs)) for k in range(3)] for i in range(3)]
    )
    hbb_exact = np.array(
        [[float(hbb[i, k].subs(subs)) for k in range(3)] for i in range(3)]
    )

    def moment(i, beta_i, delta_i):
        return np.array(
            [
                np.sin(th[i] + beta_i) * np.cos(delta_i),
                np.sin(delta_i),
                np.cos(th[i] + beta_i) * np.cos(delta_i),
            ]
        )

    def energy(betas, deltas):
        val = 0.0
        for (i, k), ji in zip(pairs, J):
            val -= ji * float(
                moment(i, betas[i], deltas[i]) @ moment(k, betas[k], deltas[k])
            )
        return val

    z = np.zeros(3)
    h = 1e-4
    fd_g = np.array(
        [
            (energy(bump(z, i, h), z) - energy(bump(z, i, -h), z)) / (2 * h)
            for i in range(3)
        ]
    )
    assert np.max(np.abs(fd_g - g_exact)) < 1e-8, (fd_g, g_exact)

    def fd_block(ch_a, ch_b):
        mat = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):

                def corner(sa, sb):
                    betas = np.zeros(3)
                    deltas = np.zeros(3)
                    if ch_a == "b":
                        betas[i] += sa
                    else:
                        deltas[i] += sa
                    if ch_b == "b":
                        betas[k] += sb
                    else:
                        deltas[k] += sb
                    return energy(betas, deltas)

                mat[i, k] = (
                    corner(h, h) - corner(h, -h) - corner(-h, h) + corner(-h, -h)
                ) / (4 * h * h)
        return 0.5 * (mat + mat.T)

    fd_hdd = fd_block("d", "d")
    fd_hbb = fd_block("b", "b")
    assert np.max(np.abs(fd_hdd - hdd_exact)) < 1e-6
    assert np.max(np.abs(fd_hbb - hbb_exact)) < 1e-6
    cosT, sinT = np.cos(th), np.sin(th)
    assert np.max(np.abs(fd_hbb @ np.ones(3))) < 1e-8
    assert np.max(np.abs(fd_hdd @ sinT - cosT * g_exact)) < 1e-8
    assert np.max(np.abs(fd_hdd @ cosT + sinT * g_exact)) < 1e-8
    print(
        f"[B] classical FD oracle: FD g (max dev "
        f"{np.max(np.abs(fd_g - g_exact)):.1e}), FD C blocks (<=1e-6), "
        f"Ward residuals <=1e-8 ... OK"
    )


def bump(vec, i, h):
    out = vec.copy()
    out[i] += h
    return out


# --------------------------------------------------------------------------
# [C] electronic three-site ring oracle
# --------------------------------------------------------------------------


def electronic_ring_reference(seed=31):
    ref = lg.build_reference(seed=seed, qnum=(1, 3), norb=3, ncell=3, natom=3)
    ref["atom_of_orb"] = np.arange(3)
    eig = lg.reference_eigenpairs(ref)
    return ref, eig


def fd_curvature(ref, eig, channel, h=1e-3):
    """(natom, natom) frozen FD Hessian block from the folded pencil."""
    ks, kweights, focc = eig["ks"], eig["kweights"], eig["focc"]
    norb = ref["norb"]
    natom = ref["natom"]
    mat = np.zeros((natom, natom))
    for i in range(natom):
        for k in range(natom):

            def energy(b_i, d_i, b_k, d_k):
                beta = np.zeros(norb)
                delta = np.zeros(norb)
                beta[ref["atom_of_orb"] == i] += b_i
                delta[ref["atom_of_orb"] == i] += d_i
                beta[ref["atom_of_orb"] == k] += b_k
                delta[ref["atom_of_orb"] == k] += d_k
                return lg.folded_band_sum(ref, ks, kweights, focc, beta, delta)

            if channel == "bb":
                mat[i, k] = (
                    energy(h, 0, h, 0)
                    - energy(h, 0, -h, 0)
                    - energy(-h, 0, h, 0)
                    + energy(-h, 0, -h, 0)
                ) / (4 * h * h)
            else:
                mat[i, k] = (
                    energy(0, h, 0, h)
                    - energy(0, h, 0, -h)
                    - energy(0, -h, 0, h)
                    + energy(0, -h, 0, -h)
                ) / (4 * h * h)
    return 0.5 * (mat + mat.T)


def lab_field_matrix_sites(ref, beta_s, delta_s):
    """Lab field-only matrix for per-SITE (a, mu) angle arrays."""
    norb, ncell = ref["norb"], ref["ncell"]
    Bf = 0.5 * ref["B_local"]
    Th = fr.angles(ref["taus"], ref["phis"], ref["q"], ncell)
    op = np.zeros((2 * norb * ncell, 2 * norb * ncell), dtype=complex)
    for a in range(ncell):
        for mu in range(norb):
            sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            op[sl, sl] = Bf[mu] * (
                (
                    np.cos(Th[a, mu] + beta_s[a, mu]) * SZ
                    + np.sin(Th[a, mu] + beta_s[a, mu]) * SX
                )
                * np.cos(delta_s[a, mu])
                + SY * np.sin(delta_s[a, mu])
            )
    return op


def lab_density(ref, eig):
    """Lab spectral density at the SAME frozen occupations."""
    w_all, focc = eig["w_all"], eig["focc"]  # noqa: F841 (fixture parity)
    norb, ncell = ref["norb"], ref["ncell"]  # noqa: F841 (fixture parity)
    H_lab, S_lab = fr.assemble_lab(
        ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], ncell
    )
    eps_lab, u_lab = lg.gen_eigh_vec(H_lab, S_lab)
    flat = np.sort(np.concatenate(w_all))
    f_flat = lg.logistic_f(flat, eig["mu"], eig["width"])
    rho = (u_lab * f_flat[None, :]) @ u_lab.conj().T / ncell
    Th = fr.angles(ref["taus"], ref["phis"], ref["q"], ncell)
    return H_lab, S_lab, rho, Th


def check_electronic_ward():
    """Ward identities in the per-site (a, mu) lab coordinates.

    The global R_x / R_z generators have per-site amplitudes
    (-cosT_j, sinT_j) that vary within each orbital family, so the
    identities live in the per-site lab coordinates, NOT in the folded
    per-orbital coordinates (whose coordinates are shared by every cell
    replica of an orbital)."""
    ref, eig = electronic_ring_reference()
    norb, ncell = ref["norb"], ref["ncell"]
    ns = norb * ncell
    Bf = 0.5 * ref["B_local"]
    H_lab, S_lab, rho, Th = lab_density(ref, eig)
    zero_b = np.zeros((ncell, norb))
    cache = {"H0": H_lab - lab_field_matrix_sites(ref, zero_b, zero_b), "S": S_lab}

    def band_sum(beta_s, delta_s):
        H = cache["H0"] + lab_field_matrix_sites(ref, beta_s, delta_s)
        eps = np.sort(lg.gen_eigvalsh(H, cache["S"]))
        f_flat = lg.logistic_f(
            np.sort(np.concatenate(eig["w_all"])), eig["mu"], eig["width"]
        )
        return float(np.sum(f_flat * eps)) / ncell

    # per-site gradients (analytic trace)
    g_beta = np.zeros(ns)
    g_delta = np.zeros(ns)
    for a in range(ncell):
        for mu in range(norb):
            sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            op = np.zeros_like(H_lab)
            op[sl, sl] = Bf[mu] * (-np.sin(Th[a, mu]) * SZ + np.cos(Th[a, mu]) * SX)
            g_beta[a * norb + mu] = np.real(np.trace(rho @ op))
            op2 = np.zeros_like(H_lab)
            op2[sl, sl] = Bf[mu] * SY
            g_delta[a * norb + mu] = np.real(np.trace(rho @ op2))
    # per-site frozen FD curvature blocks
    h = 1e-3

    def site_block(channel):
        mat = np.zeros((ns, ns))

        def idx(s):
            return (s // norb, s % norb)

        for i in range(ns):
            ai, mui = idx(i)
            for k in range(i, ns):
                ak, muk = idx(k)

                def corner(sa, sb):
                    bs = np.zeros((ncell, norb))
                    ds = np.zeros((ncell, norb))
                    if channel == "bb":
                        bs[ai, mui] += sa
                        bs[ak, muk] += sb
                    else:
                        ds[ai, mui] += sa
                        ds[ak, muk] += sb
                    return band_sum(bs, ds)

                mat[i, k] = (
                    corner(h, h) - corner(h, -h) - corner(-h, h) + corner(-h, -h)
                ) / (4 * h * h)
                mat[k, i] = mat[i, k]
        return mat

    Cbb = site_block("bb")
    Cdd = site_block("dd")
    scale = max(np.max(np.abs(Cbb)), np.max(np.abs(Cdd)), 1e-12)
    # FD floor: second-difference truncation + roundoff at h = 1e-3
    tol = 1e-5 * scale + 1e-7
    cosT = np.array([np.cos(Th[a, m]) for a in range(ncell) for m in range(norb)])
    sinT = np.array([np.sin(Th[a, m]) for a in range(ncell) for m in range(norb)])
    res_bb = np.max(np.abs(Cbb @ np.ones(ns)))
    res_rx1 = abs(float(np.sum(g_delta * cosT)))  # R_x first order (delta)
    rx = float(cosT @ Cdd @ cosT + np.sum(g_beta * sinT * cosT))
    rz = float(sinT @ Cdd @ sinT - np.sum(g_beta * sinT * cosT))
    assert res_bb < tol, res_bb
    assert res_rx1 < 1e-5 * max(np.max(np.abs(g_delta)), 1e-12) + 1e-9, res_rx1
    assert abs(rx) < tol, rx
    assert abs(rz) < tol, rz
    # field-amplitude convention: field vertex eigen-splitting = B_local
    for a in range(ncell):
        for mu in range(norb):
            sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            # no on-site hopping classes: base ring has empty site blocks
            assert np.max(np.abs(cache["H0"][sl, sl])) < 1e-12
            field = Bf[mu] * (np.cos(Th[a, mu]) * SZ + np.sin(Th[a, mu]) * SX)
            w = np.linalg.eigvalsh(field)
            assert abs((w[1] - w[0]) - ref["B_local"][mu]) < 1e-12
    print(
        f"[C] electronic ring (per-site lab coords): Cbb.1 residual "
        f"{res_bb:8.1e}; R_x first-order (delta) Ward {res_rx1:8.1e}; "
        f"R_x/R_z scalar Ward {abs(rx):8.1e}/{abs(rz):8.1e} "
        f"(<=1e-5|C|); splitting = B_local exact ... OK"
    )


def main():
    numeric_pair_oracle()
    check_electronic_ward()
    print("\nAll Ward-identity assertions passed.")


if __name__ == "__main__":
    main()
