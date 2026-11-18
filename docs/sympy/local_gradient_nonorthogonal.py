"""Assertion-checked derivation: signed local beta/delta frozen-band
gradient on normalized NONORTHOGONAL eigenpairs.

Story 001 (spiral-first-order-response), NFR-001/NFR-005, FR-002/FR-003
conventions.  Independent of TB2J/TBUpy; reuses the pinned planar-frame
builders of ``planar_su2_frame`` so there is one convention source.

Physics invariant (binding): a LOCAL spin rotation perturbs the
magnetic on-site field (plus declared co-rotating local operators),
NOT the whole basis.  In the fixed planar local frame the flat field is
``Bf sz`` and

    V1^beta = Bf sx,     V1^delta = Bf sy,     dS = 0.

Rotating the whole H and S by site unitaries is pure gauge and has
EXACTLY zero band-energy derivative; the ``dH`` artefact alone is
nonzero and is cancelled by the ``-eps dS`` term - asserted explicitly
below as the required proof that a nonzero dS marks a representation
change, not a physical local gradient.

Central identity.  For a generalized eigenpair ``H c = eps S c`` with
``c^dag S c = 1``:

    d eps = c^dag (dH - eps dS) c ,
    g_i^a = sum_k w_k sum_n f_nk c_nk^dag (V1_i^a - eps_nk dS_i^a) c_nk

in eV/rad, atom-summed over the orbitals owned by atom ``i`` (orbital
contributions sum before atom reporting).  With ``dS = 0`` (fixed
basis, the local-gradient case) this is ``Tr(rho V1)``.

Checks
------
[A] symbolic rotation vertices: lab ``V1^beta = Bf(-sinT sz + cosT sx)``,
    ``V1^delta = Bf sy``, ``V2 = -Bf n.sigma``; local-frame flat forms
    ``Bf sx`` / ``Bf sy``; covariance ``V1_lab = U(Theta) V1_local
    U(Theta)^dag``; splitting of ``Bf n.sigma`` equals ``B_local``.
[B] symbolic generalized-pencil derivative: exact rational pencil;
    ``d eps = c^dag(dH - eps dS)c`` exactly, and the expression WITHOUT
    ``-eps dS`` is provably wrong on the same pencil.
[C] numeric: nonorthogonal multi-sublattice folded toy at generic q:
    analytic local-frame gradient == analytic lab-frame gradient ==
    frozen-occupation central differences (folded pencil AND lab ring),
    beta and delta channels, multi-orbital atoms.
    Controls: deliberately off-equilibrium phase -> ``max|g^beta|``
    materially nonzero; parallel-field reference -> all gradients zero;
    ``g^delta == 0`` by planar spin symmetry (analytic AND FD, not
    skipped); global sum rule ``sum_i g_i^beta = 0``.
[D] pure-gauge negative control: along ``H -> e^{-ieK} H e^{ieK}``,
    ``S -> e^{-ieK} S e^{ieK}`` with K a site-diagonal basis-rotation
    generator, the eigenpair one-form ``sum w f c^dag(dH - eps dS)c``
    vanishes to machine precision while ``sum w f c^dag dH c`` alone is
    material: the artefact is real and cancelled only with the overlap
    term.

Run with the mydev environment:

    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/local_gradient_nonorthogonal.py
"""

import numpy as np
import planar_su2_frame as fr
import sympy as sp

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

SIGMA = {"beta": SX, "delta": SY}


# --------------------------------------------------------------------------
# [A] symbolic rotation vertices
# --------------------------------------------------------------------------


def check_symbolic_vertices():
    B, T, dlt, bta = sp.symbols("Bf Theta delta beta", real=True)
    sx, sy, sz = fr.SX, fr.SY, fr.SZ
    field = B * (
        (sp.cos(T + bta) * sz + sp.sin(T + bta) * sx) * sp.cos(dlt) + sy * sp.sin(dlt)
    )
    zero = {dlt: 0, bta: 0}
    assert sp.simplify(
        field.subs(zero) - B * (sp.cos(T) * sz + sp.sin(T) * sx)
    ) == sp.zeros(2)
    v1b = sp.simplify(sp.diff(field, bta).subs(zero))
    assert sp.simplify(v1b - B * (-sp.sin(T) * sz + sp.cos(T) * sx)) == sp.zeros(2)
    v1d = sp.simplify(sp.diff(field, dlt).subs(zero))
    assert sp.simplify(v1d - B * sy) == sp.zeros(2)
    for angle in (bta, dlt):
        assert sp.simplify(
            sp.diff(field, angle, 2).subs(zero) + B * (sp.cos(T) * sz + sp.sin(T) * sx)
        ) == sp.zeros(2)
    # local frame (Theta = 0): flat field Bf sz, vertices Bf sx / Bf sy
    assert sp.simplify(field.subs({T: 0, **zero}) - B * sz) == sp.zeros(2)
    assert sp.simplify(sp.diff(field, bta).subs({T: 0, **zero}) - B * sx) == sp.zeros(2)
    assert sp.simplify(sp.diff(field, dlt).subs({T: 0, **zero}) - B * sy) == sp.zeros(2)
    # covariance: lab vertex = U(Theta) local vertex U(Theta)^dag
    U = fr.U(T)
    assert sp.simplify(U * (B * sx) * U.H - v1b) == sp.zeros(2)
    assert sp.simplify(U * (B * sy) * U.H - v1d) == sp.zeros(2)
    # amplitude: spectrum of Bf n.sigma is {+Bf, -Bf} -> splitting 2 Bf
    x = sp.Symbol("x")
    assert sp.simplify((B * sz).charpoly(x).as_expr() - (x**2 - B**2)) == 0
    print(
        "[A] rotation vertices: V1_beta=Bf(-sinT sz+cosT sx), "
        "V1_delta=Bf sy; local frame Bf sx/Bf sy; U-covariance; "
        "splitting 2*Bf = B_local ... OK"
    )


# --------------------------------------------------------------------------
# [B] symbolic generalized-pencil derivative identity
# --------------------------------------------------------------------------


def check_symbolic_pencil_identity():
    """Exact rational 2x2 pencil affine in q: d eps = c^dag(dH - eps dS)c."""
    q = sp.Symbol("q", real=True)
    H = sp.Matrix(
        [
            [sp.Rational(3, 2), sp.Rational(1, 3) + q / 5],
            [sp.Rational(1, 3) + q / 5, sp.Rational(2, 7) - q / 4],
        ]
    )
    S = sp.Matrix(
        [
            [2, sp.Rational(1, 11) + q / 7],
            [sp.Rational(1, 11) + q / 7, sp.Rational(3, 2) + q / 9],
        ]
    )
    lam = sp.Symbol("lambda")
    eps_expr = sp.solve(sp.expand((H - lam * S).det()), lam)[0]
    q0 = sp.Rational(2, 5)
    eps0 = eps_expr.subs(q, q0)
    H0, S0 = H.subs(q, q0), S.subs(q, q0)
    x = sp.symbols("x", real=True)
    vec_un = sp.Matrix([x, 1])
    row = (H0 - eps0 * S0) * vec_un
    x_sol = sp.solve(sp.Eq(row[0], 0), x)[0]
    vec_un = sp.Matrix([x_sol, 1])
    assert sp.simplify(row.subs(x, x_sol)[1]) == 0
    norm2 = sp.simplify((vec_un.T * S0 * vec_un)[0, 0])
    c = vec_un / sp.sqrt(norm2)
    assert sp.simplify((c.T * S0 * c)[0, 0] - 1) == 0
    assert sp.simplify((H0 - eps0 * S0) * c) == sp.zeros(2, 1)
    dH, dS = sp.diff(H, q).subs(q, q0), sp.diff(S, q).subs(q, q0)
    deps = eps_expr.diff(q).subs(q, q0)
    rhs = sp.simplify((c.T * (dH - eps0 * dS) * c)[0, 0])
    assert sp.simplify(deps - rhs) == 0, (deps, rhs)
    rhs_wrong = sp.simplify((c.T * dH * c)[0, 0])
    assert sp.simplify(deps - rhs_wrong) != 0
    print(
        "[B] generalized pencil: d eps = c^dag(dH - eps dS)c exactly "
        "(rational 2x2); dropping -eps*dS provably wrong ... OK"
    )


# --------------------------------------------------------------------------
# numeric helpers (reuse pinned builders from planar_su2_frame)
# --------------------------------------------------------------------------


def gen_eigh_vec(H, S):
    """Generalized eigendecomposition (ascending), c^dag S c = I."""
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    A = Linv @ H @ Linv.conj().T
    A = 0.5 * (A + A.conj().T)
    w, u = np.linalg.eigh(A)
    return w, Linv.conj().T @ u


def gen_eigvalsh(H, S):
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    A = Linv @ H @ Linv.conj().T
    return np.linalg.eigvalsh(0.5 * (A + A.conj().T))


def fermi_mu(w_all, nel, width):
    """Fixed-N Fermi-Dirac chemical potential by bisection."""
    flat = np.concatenate(w_all)
    lo, hi = flat.min() - 10 * width, flat.max() + 10 * width
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(np.sum(1.0 / (1.0 + np.exp((flat - mid) / width)))) > nel:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def logistic_f(w, mu, width):
    return 1.0 / (1.0 + np.exp((w - mu) / width))


def fermi_occ(w_all, nel, width):
    """Per-k fixed-N Fermi-Dirac occupations (frozen reference)."""
    mu = fermi_mu(w_all, nel, width)
    return [logistic_f(w, mu, width) for w in w_all], mu


def build_reference(
    seed=21, qnum=(3, 8), phis=None, taus=None, norb=4, ncell=8, natom=2, real=False
):
    rng = np.random.default_rng(seed)
    classes, taus_, phis_, B_local = fr.make_classes(rng, norb, ncell // 2, real=real)
    if taus is not None:
        taus_ = np.array(taus, dtype=float)
    if phis is not None:
        phis_ = np.array(phis, dtype=float)
    q = np.array([qnum[0] / qnum[1], 0.0, 0.0])
    return dict(
        classes=classes,
        taus=taus_,
        phis=phis_,
        B_local=B_local,
        q=q,
        ncell=ncell,
        norb=norb,
        natom=natom,
        atom_of_orb=np.arange(norb) % natom,
    )


def folded_mesh(ref):
    s = fr.flux_shift(ref["q"][0], ref["ncell"])
    ks = [np.array([(m + s) / ref["ncell"], 0.0, 0.0]) for m in range(ref["ncell"])]
    return ks


def reference_eigenpairs(ref, nel_fill=0.7, width=0.05):
    ks = folded_mesh(ref)
    w_all, c_all = [], []
    for k in ks:
        Hq, Sq = fr.assemble_folded(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
        )
        w, c = gen_eigh_vec(Hq, Sq)
        w_all.append(w)
        c_all.append(c)
    focc, mu = fermi_occ(w_all, nel=nel_fill * len(np.concatenate(w_all)), width=width)
    kweights = np.full(ref["ncell"], 1.0 / ref["ncell"])
    return dict(
        ks=ks,
        kweights=kweights,
        w_all=w_all,
        c_all=c_all,
        focc=focc,
        mu=mu,
        width=width,
    )


def local_field_matrix(norb, Bf, beta, delta):
    """Full (2n,2n) field contribution for per-orbital (beta, delta)."""
    op = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for mu in range(norb):
        op[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] = Bf[mu] * (
            (np.cos(beta[mu]) * SZ + np.sin(beta[mu]) * SX) * np.cos(delta[mu])
            + SY * np.sin(delta[mu])
        )
    return op


def rotate_atom(vec, atom_of_orb, atom, h):
    vec = np.array(vec, dtype=float)
    vec[atom_of_orb == atom] += h
    return vec


def folded_band_sum(ref, ks, kweights, focc, beta, delta):
    norb = ref["norb"]
    Bf = 0.5 * ref["B_local"]
    flat_fields = local_field_matrix(norb, Bf, np.zeros(norb), np.zeros(norb))
    tot = 0.0
    for k, f, wgt in zip(ks, focc, kweights):
        Hq, Sq = fr.assemble_folded(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
        )
        Hq += local_field_matrix(norb, Bf, beta, delta) - flat_fields
        tot += wgt * float(np.sum(f * gen_eigvalsh(Hq, Sq)))
    return tot


def lab_field_matrix(ref, ncell, beta, delta):
    """Lab-ring field-only contribution for per-orbital (beta, delta)."""
    norb = ref["norb"]
    Bf = 0.5 * ref["B_local"]
    Th = fr.angles(ref["taus"], ref["phis"], ref["q"], ncell)
    op = np.zeros((2 * norb * ncell, 2 * norb * ncell), dtype=complex)
    for a in range(ncell):
        for mu in range(norb):
            sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
            op[sl, sl] = Bf[mu] * (
                (np.cos(Th[a, mu] + beta[mu]) * SZ + np.sin(Th[a, mu] + beta[mu]) * SX)
                * np.cos(delta[mu])
                + SY * np.sin(delta[mu])
            )
    return op


def lab_band_sum(ref, eig, beta, delta, cache={}):
    """Whole-ring frozen-occupation band sum (per primitive cell)."""
    norb, ncell = ref["norb"], ref["ncell"]
    key = id(ref)
    if key not in cache:
        H_lab, S_lab = fr.assemble_lab(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], ncell
        )
        zero_b = np.zeros(norb)
        cache[key] = (H_lab - lab_field_matrix(ref, ncell, zero_b, zero_b), S_lab)
    H0, S_lab = cache[key]
    H = H0 + lab_field_matrix(ref, ncell, beta, delta)
    eps = np.sort(gen_eigvalsh(H, S_lab))
    f_flat = logistic_f(np.sort(np.concatenate(eig["w_all"])), eig["mu"], eig["width"])
    return float(np.sum(f_flat * eps)) / ncell


# --------------------------------------------------------------------------
# [C] numeric local gradient vs FD, both frames
# --------------------------------------------------------------------------


def g_analytic_folded(ref, eig, channel):
    """Atom-resolved local-frame gradient Tr(rho V1), dS = 0."""
    kweights, w_all, c_all, focc = (
        eig["kweights"],
        eig["w_all"],
        eig["c_all"],
        eig["focc"],
    )
    norb = ref["norb"]
    Bf = 0.5 * ref["B_local"]
    out = {}
    for atom in sorted(set(ref["atom_of_orb"])):
        op = np.zeros((2 * norb, 2 * norb), dtype=complex)
        for mu in np.flatnonzero(ref["atom_of_orb"] == atom):
            op[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] = Bf[mu] * SIGMA[channel]
        acc = 0.0
        for w, c, f, wgt in zip(w_all, c_all, focc, kweights):
            per_band = np.einsum("in,ij,jn->n", c.conj(), op, c)
            acc += wgt * float(np.sum(f * per_band.real))
        out[atom] = acc
    return out


def g_analytic_lab(ref, eig, channel):
    """Same gradient in the lab frame: direct S-orthonormal lab eigenpairs
    carrying the SAME fixed occupations (spectrum equality makes the
    ascending lists identical, so f transfers exactly)."""
    w_all, focc = eig["w_all"], eig["focc"]  # noqa: F841 (fixture parity)
    norb, ncell = ref["norb"], ref["ncell"]
    Bf = 0.5 * ref["B_local"]
    H_lab, S_lab = fr.assemble_lab(
        ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], ncell
    )
    eps_lab, u_lab = gen_eigh_vec(H_lab, S_lab)
    flat_sorted = np.sort(np.concatenate(w_all))
    assert np.max(np.abs(np.sort(eps_lab) - flat_sorted)) < 1e-10
    f_flat = logistic_f(flat_sorted, eig["mu"], eig["width"])
    rho = (u_lab * f_flat[None, :]) @ u_lab.conj().T / ncell
    Th = fr.angles(ref["taus"], ref["phis"], ref["q"], ncell)
    out = {}
    for atom in sorted(set(ref["atom_of_orb"])):
        op = np.zeros_like(H_lab)
        for a in range(ncell):
            for mu in np.flatnonzero(ref["atom_of_orb"] == atom):
                sl = slice(2 * (a * norb + mu), 2 * (a * norb + mu) + 2)
                if channel == "beta":
                    op[sl, sl] = Bf[mu] * (
                        -np.sin(Th[a, mu]) * SZ + np.cos(Th[a, mu]) * SX
                    )
                else:
                    op[sl, sl] = Bf[mu] * SY
        out[atom] = float(np.real(np.trace(rho @ op)))
    return out


def fd_gradient(ref, eig, kind, atom, channel, h=1e-4):
    """Frozen-occupation central difference, folded pencil or lab ring."""
    ks, kweights, focc = eig["ks"], eig["kweights"], eig["focc"]
    norb = ref["norb"]
    zero = np.zeros(norb)

    def energy(beta, delta):
        if kind == "folded":
            return folded_band_sum(ref, ks, kweights, focc, beta, delta)
        return lab_band_sum(ref, eig, beta, delta)  # noqa: F841 (fixture parity)

    def bumped(base, sign):
        vec = base.copy()
        vec[ref["atom_of_orb"] == atom] += sign * h
        return (vec, zero) if channel == "beta" else (zero, vec)

    ep = energy(*bumped(zero, +1))
    em = energy(*bumped(zero, -1))
    return (ep - em) / (2 * h)


def check_numeric_local_gradient():
    for tag, override, delta_zero in (
        ("off-equilibrium (complex classes)", {}, False),
        ("off-equilibrium (real classes, delta control)", {"real": True}, True),
        (
            "uniform-FM control (q=0)",
            {"phis": [0.0] * 4, "taus": np.zeros((4, 3)), "qnum": (0, 1)},
            True,
        ),
    ):
        ref = build_reference(**override)
        eig = reference_eigenpairs(ref)
        g = {ch: g_analytic_folded(ref, eig, ch) for ch in ("beta", "delta")}
        g_lab = {ch: g_analytic_lab(ref, eig, ch) for ch in ("beta", "delta")}
        for ch in ("beta", "delta"):
            for atom in g[ch]:
                assert abs(g[ch][atom] - g_lab[ch][atom]) < 1e-9, (
                    tag,
                    ch,
                    atom,
                    g[ch][atom],
                    g_lab[ch][atom],
                )
        # frozen-occupation central FD on BOTH representations
        for kind in ("folded", "lab"):
            for ch in ("beta", "delta"):
                for atom in g[ch]:
                    fd = fd_gradient(ref, eig, kind, atom, ch)
                    assert abs(g[ch][atom] - fd) < 1e-7, (
                        tag,
                        kind,
                        ch,
                        atom,
                        g[ch][atom],
                        fd,
                    )
        assert abs(sum(g["beta"].values())) < 1e-9, g["beta"]
        max_beta = max(abs(v) for v in g["beta"].values())
        max_delta = max(abs(v) for v in g["delta"].values())
        if tag.startswith("off-equilibrium"):
            assert max_beta > 1e-3, g["beta"]
        else:
            assert max_beta < 1e-10, g["beta"]
        if delta_zero:
            assert max_delta < 1e-9, g["delta"]
        print(
            f"[C] {tag}: analytic==FD both frames; max|g_beta|={max_beta:.6f},"
            f" |sum g_beta|={abs(sum(g['beta'].values())):.1e}, "
            f"max|g_delta|={max_delta:.1e}"
            + (
                " (vanishes by planar symmetry, analytic AND FD)"
                if delta_zero
                else " (nonzero: complex hoppings break the planar reflection"
                " symmetry; physics, not gauge)"
            )
            + "  ... OK"
        )


# --------------------------------------------------------------------------
# [D] pure-gauge negative control
# --------------------------------------------------------------------------


def check_pure_gauge_cancellation():
    ref = build_reference()
    eig = reference_eigenpairs(ref)
    ks, kweights, w_all, c_all, focc = (
        eig["ks"],
        eig["kweights"],
        eig["w_all"],
        eig["c_all"],
        eig["focc"],
    )
    norb = ref["norb"]
    # K restricted to ONE atom's orbitals: a global generator would make
    # the dH artefact cancel by the sum_i g_i^beta = 0 Ward identity.
    K = np.zeros((2 * norb, 2 * norb), dtype=complex)
    for mu in np.flatnonzero(ref["atom_of_orb"] == 0):
        K[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2] = SY / 2  # U(eps) generator
    oneform = 0.0
    artefact = 0.0
    for k, w, c, f, wgt in zip(ks, w_all, c_all, focc, kweights):
        Hq, Sq = fr.assemble_folded(
            ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
        )
        dH = 1j * (K @ Hq - Hq @ K)  # d/de U^dag(e) H U(e) at e = 0
        dS = 1j * (K @ Sq - Sq @ K)
        oneform += wgt * float(
            np.real(
                np.sum(
                    f
                    * (
                        np.einsum("in,ij,jn->n", c.conj(), dH, c)
                        - w * np.einsum("in,ij,jn->n", c.conj(), dS, c)
                    )
                )
            )
        )
        artefact += wgt * float(
            np.real(np.sum(f * np.einsum("in,ij,jn->n", c.conj(), dH, c)))
        )
    assert abs(oneform) < 1e-10, oneform
    assert abs(artefact) > 1e-4, artefact

    def e_of(eps):
        Ue = np.eye(2 * norb, dtype=complex) * np.cos(eps) - 1j * np.sin(eps) * (2 * K)
        tot = 0.0
        for wgt, f, k in zip(kweights, focc, ks):
            Hq, Sq = fr.assemble_folded(
                ref["classes"], ref["taus"], ref["phis"], ref["q"], ref["B_local"], k
            )
            tot += wgt * float(
                np.sum(f * gen_eigvalsh(Ue.conj().T @ Hq @ Ue, Ue.conj().T @ Sq @ Ue))
            )
        return tot

    h = 1e-4
    drift = abs(e_of(h) - e_of(-h)) / (2 * h)
    assert drift < 1e-9, drift
    print(
        f"[D] pure gauge: one-form {oneform:8.1e} == 0 while dH-only "
        f"artefact {artefact:8.1e} is material; band-energy drift along "
        f"the gauge direction {drift:8.1e}  ... OK"
    )


def main():
    check_symbolic_vertices()
    check_symbolic_pencil_identity()
    check_numeric_local_gradient()
    check_pure_gauge_cancellation()
    print("\nAll local-gradient assertions passed.")


if __name__ == "__main__":
    main()
