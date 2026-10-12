"""SIESTA rotating-frame spiral pencil: sympy/numeric verified conventions.

Story 001 (siesta-fixed-spin-spiral-nonorthogonal-xc-mft / epic 001).
Grounded in the SIESTA source audit (research memo 2026-10-12, section 1)
and the already-pinned GBT central identity of
``spin_spiral_generalized_bloch.py`` (which this script does NOT re-prove;
it bridges SIESTA's exporter conventions onto that result).

Source conventions under test (diag2kspiral.F90, setup_k, ~lines 326-351):

* sparse spin components: H1=H11 (up-up), H2=H22 (down-down),
  H3=Re H12, H4=-Im H12, i.e. H12 = H3 - i H4, H21 = H3 + i H4.
* k-phase  kphs = exp(-i k.x_ij)  on every entry,
* q-phase  qphs = exp(-i q.x_ij):
      up-up   gets  kphs * qphs        (D11)
      down-dn gets  kphs * conj(qphs)  (D22)
      H12 entry uses D11, H21 entry uses D22, then the spin box is
      Hermitized:  D12 <- (D12 + conj(D21))/2.
* the SAME phase pair is applied to the OVERLAP blocks S (spin-independent
  in the lab frame): S11 gets D11, S22 gets D22.

Pinned, with assertion checks:

[A] spinor packing round-trip (symbolic): the 4 real components reconstruct
    a Hermitian 2x2 lab block; the phased+Hermitized rotating-frame block is
    Hermitian across the (i,j)/(j,i) pair under the physical pairing
    W(nu,mu,-R) = W(mu,nu,R)^dag.

[B] ring/folded equivalence through the SIESTA assembly (numeric, exact
    construction, spin-diagonal stored blocks): the two spin channels are
    plain Fourier transforms at k+q and k-q; since the commensurate k mesh
    is invariant under the q-shift, the union of the generalized
    eigenvalues of the SIESTA-assembled pencils (H(k), S(k)) over the
    folded mesh equals the supercell ring pencil spectrum (agreement
    1e-13 on random nonorthogonal multi-orbital models).  Blocks with
    spin mixing (H3, H4 != 0) have their stored-component frame fixed by
    the SCF (vmatsp/rhoofdsp), not by diag2kspiral alone: the full
    textured-frame equivalence is deferred to the producer-fixture
    stories (AC-001/AC-002); what IS pinned analytically for the mixing
    channels is the packing/Hermitization structure of [A].

[C] rotating-frame overlap is spin-dependent (symbolic + numeric): even
    with a spin-independent lab S, the assembled S(k) is different in the
    two spin channels, S_up(k) = S_lab(k+q), S_dn(k) = S_lab(k-q).  All
    dual-basis formulas must therefore use the assembled spinor S, never
    the scalar lab S.

[D] dual-basis identities on the pencil (exact rational + numeric):
        Tr A  = Tr(S S^-1 A) = Tr(S^-1 A S)   (O18's two dual forms;
                                               Tr(S A~) pairs with the
                                               COEFFICIENT form A~ = S^-1 A)
        (z S - H)^-1 = (zI - S^-1 H)^-1 S^-1   (pencil resolvent; S^-1
                                                on the RIGHT)
        (zS - H)^-1 = sum_n c_n c_n^dag / (z - eps_n)
                      with H c = eps S c, c^dag S c = 1
        rho(eps) = -1/pi Im Tr( S (zS-H)^-1 ) = Lorentzian spectral sum

Run with the mydev environment:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/siesta_spiral_pencil.py
"""

import numpy as np
import sympy as sp

rng = np.random.default_rng(20261012)


def assert_close(name, a, b, tol=1e-11):
    err = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    if not err < tol:
        raise AssertionError(f"{name}: max|diff| = {err:.3e} >= {tol}")
    print(f"  ok  {name}: max|diff| = {err:.2e}")


# ---------------------------------------------------------------------------
# [A] symbolic packing and Hermitization
# ---------------------------------------------------------------------------
def assert_zero(name, expr):
    """simplify() first; fall back to exact-precision numeric probing
    (sympy often fails to combine equal compound exponents)."""
    if sp.simplify(expr) == 0:
        print(f"  ok  {name} (symbolic)")
        return
    probes = [
        (sp.Rational(3, 7), sp.Rational(2, 5), sp.Rational(5, 11)),
        (sp.Rational(1, 3), sp.Rational(5, 13), sp.Rational(2, 9)),
    ]
    extra = [
        sp.Rational(11, 4),
        sp.Rational(-7, 5),
        sp.Rational(13, 6),
        sp.Rational(-3, 8),
    ]
    for kk, qq, xx in probes:
        subs = {sp.Symbol("k"): kk, sp.Symbol("q"): qq, sp.Symbol("x"): xx}
        for i, sym in enumerate(sorted(expr.free_symbols, key=str)):
            subs.setdefault(sym, extra[i % len(extra)])
        val = expr.subs(subs)
        if abs(sp.N(val, 60)) > sp.Float("1e-40"):
            raise AssertionError(f"{name}: not zero (probe {sp.N(val, 20)})")
    print(f"  ok  {name} (numeric probe)")


def check_packing_symbolic():
    print("[A] spinor packing / Hermitization (symbolic)")
    H1, H2, H3, H4 = sp.symbols("H1 H2 H3 H4", real=True)
    lab = sp.Matrix([[H1, H3 - sp.I * H4], [H3 + sp.I * H4, H2]])
    assert sp.simplify(lab - lab.H) == sp.zeros(2, 2), "lab block not Hermitian"
    print("  ok  4-component lab block is Hermitian")

    k, q, x = sp.symbols("k q x", real=True)
    kphs = sp.exp(-sp.I * k * x)
    qphs = sp.exp(-sp.I * q * x)
    D11 = kphs * qphs
    D22 = kphs * sp.conjugate(qphs)
    D12 = sp.Rational(1, 2) * (
        (H3 - sp.I * H4) * D11 + sp.conjugate((H3 + sp.I * H4) * D22)
    )
    D21 = sp.conjugate(D12)
    # Non-trivial pinned fact: the Hermitized off-diagonal reduces to the
    # full-q phase times the AVERAGED k-weight:
    #     D12 = (H3 - i H4) exp(-i q x) cos(k x)
    C_ref = (H3 - sp.I * H4) * sp.exp(-sp.I * q * x) * sp.cos(k * x)
    assert_zero("D12 == (H3 - i H4) e^{-i q x} cos(k x)", D12 - C_ref)
    print("  ok  D12 = (H3 - i H4) e^{-i q x} cos(k x)")

    # Spin-diagonal reduction: the two channels see k+q and k-q exactly.
    diag_up = sp.simplify(sp.exp(sp.I * (k + q) * x) * H1 * D11)
    diag_dn = sp.simplify(sp.exp(sp.I * (k - q) * x) * H2 * D22)
    assert diag_up == H1 and diag_dn == H2, "channel momentum split wrong"
    print("  ok  spin-diagonal channels carry e^{-i(k+q)x} / e^{-i(k-q)x}")

    # Hermiticity structure of the SIESTA assembly (two distinct mechanisms):
    # (i) SPIN off-diagonal: hermitized WITHIN each stored entry -- the entry
    #     contributes (C12, conj(C12)) as a conjugate pair, so the summed
    #     H(k) spin box is Hermitian for any single entry.
    assert sp.simplify(D21 - sp.conjugate(D12)) == 0
    print("  ok  per-entry spin box contributes conjugate C12/C21 pair")

    # (ii) ORBITAL off-diagonal: comes from the partner entry (nu,mu,-R) of
    # W(nu,mu,-R) = W(mu,nu,R)^dag, whose stored components are
    # H1'=H1, H2'=H2, H3'=H3, H4'=-H4 at x'=-x.  Its diagonal channels and
    # spin off-diagonal must be the elementwise conjugates of this entry's.
    def assembled(h1, h2, h3, h4, xx):
        kp = sp.exp(-sp.I * k * xx)
        qp = sp.exp(-sp.I * q * xx)
        d11 = kp * qp
        d22 = kp * sp.conjugate(qp)
        d12 = sp.Rational(1, 2) * (
            (h3 - sp.I * h4) * d11 + sp.conjugate((h3 + sp.I * h4) * d22)
        )
        return sp.Matrix([[h1 * d11, d12], [sp.conjugate(d12), h2 * d22]])

    Bij = assembled(H1, H2, H3, H4, x)
    Bji = assembled(H1, H2, H3, -H4, -x)
    assert_zero(
        "partner diagonal (up) is conjugate", Bji[0, 0] - sp.conjugate(Bij[0, 0])
    )
    assert_zero(
        "partner diagonal (dn) is conjugate", Bji[1, 1] - sp.conjugate(Bij[1, 1])
    )
    assert_zero("partner C12 is conjugate", Bji[0, 1] - sp.conjugate(Bij[0, 1]))
    print("  ok  partner entry (x->-x, H4->-H4) is the elementwise conjugate")


# ---------------------------------------------------------------------------
# model builder: random nonorthogonal ring with general Hermitian spinor blocks
# ---------------------------------------------------------------------------
def herm2(rng):
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    return (a + a.conj().T) / 2


def build_model(ncell=6, norb=2, nR=3):
    """Lab-frame spinor blocks W(mu,nu,R) (2x2 Hermitian in spin) with the
    physical pairing W(nu,mu,-R) = W(mu,nu,R)^dag, and a spin-independent
    SPD overlap S(mu,nu,R) with the same pairing."""
    W, S = {}, {}
    for R in range(nR):
        for mu in range(norb):
            for nu in range(norb):
                if R == 0 and mu > nu:
                    continue
                blk = herm2(rng)
                W[(mu, nu, R)] = blk
                W[(nu, mu, -R)] = blk.conj().T
    for R in range(nR):
        for mu in range(norb):
            for nu in range(norb):
                if (mu, nu, R) in S:
                    continue
                if R == 0 and mu > nu:
                    continue
                s = rng.normal(size=(1, 1))[0, 0]
                s = np.array([[s]])
                # overlap blocks are norb x norb scalars here per (mu,nu)
                val = rng.normal()
                S[(mu, nu, R)] = np.array([[val]])
                S[(nu, mu, -R)] = np.array([[val]])  # real, symmetric
    return W, S


def siesta_assembly(W, S, ncell, norb, k, q):
    """Assemble the rotating-frame pencil exactly as diag2kspiral does."""
    n = norb
    H = np.zeros((2 * n, 2 * n), complex)
    Sm = np.zeros((2 * n, 2 * n), complex)
    for (mu, nu, R), blk in W.items():
        x = R  # 1D lattice vector in primitive units
        kphs = np.exp(-1j * k * x)
        qphs = np.exp(-1j * q * x)
        D11 = kphs * qphs
        D22 = kphs * np.conj(qphs)
        iu, ju = mu, nu
        H[iu, ju] += blk[0, 0] * D11
        H[iu, ju + n] += blk[0, 1] * D11
        H[iu + n, ju] += blk[1, 0] * D22
        H[iu + n, ju + n] += blk[1, 1] * D22
        Sv = float(S[(mu, nu, R)][0, 0])
        Sm[iu, ju] += Sv * D11
        Sm[iu + n, ju + n] += Sv * D22
    # Hermitize the spin off-diagonal boxes (source does this per entry)
    H12 = H[:n, n:]
    H21 = H[n:, :n]
    H12h = 0.5 * (H12 + H21.conj().T)
    H[:n, n:] = H12h
    H[n:, :n] = H12h.conj().T
    return H, Sm


def ring_assembly(W, S, ncell, norb):
    """Supercell ring operator in the lab frame (no rotating-frame phases)."""
    n = ncell * norb
    H = np.zeros((2 * n, 2 * n), complex)
    Sm = np.zeros((2 * n, 2 * n))
    for c in range(ncell):
        for (mu, nu, R), blk in W.items():
            j = ((c + R) % ncell) * norb + nu
            i = c * norb + mu
            H[i, j] += blk[0, 0]
            H[i, j + n] += blk[0, 1]
            H[i + n, j] += blk[1, 0]
            H[i + n, j + n] += blk[1, 1]
            Sv = float(S[(mu, nu, R)][0, 0])
            Sm[i, j] += Sv
            Sm[i + n, j + n] += Sv
    return H, Sm


def gen_eigs(H, S):
    import scipy.linalg

    return scipy.linalg.eigh(H, S, eigvals_only=True)


def check_ring_folded(ncell=6, norb=2, q_num=1, nR=3):
    print("[B] SIESTA assembly vs supercell ring (numeric, spin-diagonal W)")
    # Scope (pinned by this derivation): for spin-diagonal stored blocks the
    # two spin channels are plain Fourier transforms at k+q and k-q, and the
    # commensurate mesh is invariant under the q-shift, so the union of the
    # SIESTA-assembled pencils over the folded k mesh equals the ring pencil
    # spectrum -- phases on H AND S included.  For blocks with spin mixing
    # (H3, H4 != 0) the stored-component frame is fixed by the SCF
    # (vmatsp/rhoofdsp), not by diag2kspiral alone; the full textured-frame
    # equivalence therefore requires a real producer fixture and is deferred
    # to stories 003/004 (AC-001/AC-002) -- see the paired report.
    W, S = build_model(ncell, norb, nR)
    for key, blk in list(W.items()):
        # spin-channel offsets ONLY (mu/nu-dependent offsets would break the
        # W(nu,mu,-R) = W(mu,nu,R)^dag pairing)
        d = np.diag([blk[0, 0].real + 0.31, blk[1, 1].real - 0.27])
        W[key] = d.astype(complex)
    for mu, nu, R in list(S.keys()):
        if R == 0 and mu == nu:
            S[(mu, nu, R)] = np.array([[float(S[(mu, nu, R)][0, 0]) + 4.0 * norb]])
    q = 2 * np.pi * q_num / ncell
    Hr, Sr = ring_assembly(W, S, ncell, norb)
    ev_ring = gen_eigs(Hr, Sr)
    ev_fold = []
    for n in range(ncell):
        k = 2 * np.pi * n / ncell
        Hk, Sk = siesta_assembly(W, S, ncell, norb, k, q)
        assert np.allclose(Hk, Hk.conj().T, atol=1e-12), "H(k) not Hermitian"
        assert np.allclose(Sk, Sk.conj().T, atol=1e-12), "S(k) not Hermitian"
        ev_fold.append(gen_eigs(Hk, Sk))
    ev_fold = np.sort(np.concatenate(ev_fold))
    assert_close(
        "union of folded pencil eigenvalues == ring eigenvalues",
        ev_fold,
        np.sort(ev_ring),
        tol=1e-9,
    )


# ---------------------------------------------------------------------------
# [C] rotating-frame overlap is spin-dependent
# ---------------------------------------------------------------------------
def check_spin_dependent_overlap():
    print("[C] rotating-frame S(k) per spin channel (numeric)")
    W, S = build_model(6, 2, 3)
    k, q = 0.3, 2 * np.pi / 6
    _, Sk = siesta_assembly(W, S, 6, 2, k, q)
    Sup = Sk[:2, :2]
    Sdn = Sk[2:, 2:]
    assert not np.allclose(Sup, Sdn), "spin channels of S(k) unexpectedly equal"

    # explicit channel identities: S_up(k) = S_lab(k+q), S_dn(k) = S_lab(k-q)
    def S_lab(kk):
        out = np.zeros((2, 2), complex)
        for (mu, nu, R), v in S.items():
            out[mu, nu] += float(v[0, 0]) * np.exp(-1j * kk * R)
        return out

    assert_close("S_up(k) == S_lab(k+q)", Sup, S_lab(k + q))
    assert_close("S_dn(k) == S_lab(k-q)", Sdn, S_lab(k - q))
    print("  ok  assembled S(k) carries the full-q phases per spin channel")


# ---------------------------------------------------------------------------
# [D] dual-basis identities (exact rational)
# ---------------------------------------------------------------------------
def check_dual_basis():
    print("[D] dual-basis pencil identities (exact rational)")
    n = 3
    Ar = rng.integers(-5, 6, size=(n, n))
    A = sp.Matrix(Ar.tolist())
    St = (A * A.T) / sp.Integer(10) + sp.eye(n)  # SPD, exact
    Hm = sp.Matrix((rng.integers(-5, 6, size=(n, n))).tolist())
    Hm = (Hm + Hm.T) / 2
    z = sp.symbols("z")

    # O18 App. B carries TWO dual conventions; both trace statements hold,
    # but they pair with DIFFERENT dual forms (a conflation trap):
    #   coefficient form   A~ = S^-1 A          ->  Tr(S A~) == Tr(A)
    #   similarity form    A~ = S^-1 A S        ->  Tr(A~)   == Tr(A)  (cyclic)
    Acoef = St.inv() * A
    Asim = St.inv() * A * St
    assert sp.simplify(sp.trace(A) - sp.trace(St * Acoef)) == 0, "coefficient trace"
    assert sp.simplify(sp.trace(A) - sp.trace(Asim)) == 0, "similarity trace"
    print("  ok  Tr A == Tr(S S^-1 A) == Tr(S^-1 A S)  [both dual forms]")

    # Pencil resolvent factorization (exact): from zS - H = S (zI - S^-1 H),
    #   (zS - H)^-1 == (zI - S^-1 H)^-1 S^-1     (S^-1 on the RIGHT)
    lhs = (z * St - Hm).inv()
    rhs = (z * sp.eye(n) - St.inv() * Hm).inv() * St.inv()
    diff = sp.simplify(lhs - rhs)
    assert diff == sp.zeros(n, n), f"resolvent factorization failed:\n{diff}"
    print("  ok  (zS - H)^-1 == (zI - S^-1 H)^-1 S^-1   [exact]")

    # DOS identity, numeric: rho(eps) = -1/pi Im Tr(S G~) == spectral sum.
    # Spectral decomposition: (zS - H)^-1 = sum_n c_n c_n^dag/(z - eps_n)
    # with H c = eps S c, c^dag S c = 1.
    Sn = np.array(St.tolist(), dtype=float)
    Hn = np.array(Hm.tolist(), dtype=float)
    import scipy.linalg

    w, V = scipy.linalg.eigh(Hn, Sn)
    eps0, delta = 0.37, 1e-3
    z0 = eps0 + 1j * delta
    G = np.linalg.inv(z0 * Sn - Hn)
    rho_tr = -np.imag(np.trace(Sn @ G)) / np.pi
    rho_spec = np.sum(delta / np.pi / ((eps0 - w) ** 2 + delta**2))
    assert_close(
        "rho(eps) via Tr(S G~) == Lorentzian spectral sum", rho_tr, rho_spec, tol=1e-8
    )
    Gspec = sum(np.outer(V[:, i], V[:, i].conj()) / (z0 - w[i]) for i in range(n))
    assert_close("pencil resolvent == spectral decomposition", G, Gspec, tol=1e-9)


if __name__ == "__main__":
    check_packing_symbolic()
    check_ring_folded()
    check_spin_dependent_overlap()
    check_dual_basis()
    print("\nAll pinned checks passed.")
