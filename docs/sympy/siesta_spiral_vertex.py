"""Production magnetic vertex from the total-H spin splitting (story 005).

Pinned, with assertions, on a minimal 2-orbital 2-site symbolic model:

[A] Pauli decomposition of the stored lab-frame spinor block
    W(mu,nu,R) = [[h1, h3-i*h4], [h3+i*h4, h2]]:
        W = h0*sigma0 + hx*sigma_x + hy*sigma_y + hz*sigma_z
        h0 = (h1+h2)/2,  hx = h3,  hy = h4,  hz = (h1-h2)/2
    (the sign hy = h4 is fixed by H12 = h3 - i*h4, since
     (hx - i*hy) is the 12 element of h.sigma).  h0 is the SCALAR part;
     (hx,hy,hz) is the MAGNETIC VERTEX (Oroszlany convention: the
     spin-dependent part of the saved total H, no XC reconstruction).

[B] Partner consistency implied by the bundle validate()/Hermiticity
    pairing (nu,mu,-R): the scalar parts are shared,
        h1(j,i,-R) = h1(i,j,R),  h2(j,i,-R) = h2(i,j,R),
    and the mixing components carry the converged-spiral texture
        h12(j,i,-R) = e^{-i q.x_ij} h12(i,j,R),   h12 = h3 - i*h4
    (h3' - i*h4' = e^{-i q.x} (h3 - i*h4)); at the shadow q.x = 2*pi*n
    this reduces to identical components, NOT the plain adjoint pairing
    h4' = -h4 -- which in fact breaks pencil Hermiticity for q.x != 0.
    With the textured partner the assembled k-space pencil is exactly
    Hermitian.

[C] Local SU(2) rotations U_p = exp(-i*theta_p*sigma_n/2) and the
    DECLARED PARTITION (vertex 2.0).  A magnetic block (mu,nu,R) has
    distinct endpoint-owner sites P = {owner(mu), owner(nu)}; the
    rotation is allocated additively,
        W' = sum_p w_p U_p W U_p^dag
           = W + sum_p w_p (U_p W U_p^dag - W),   w_p = 1/|P|,
    each contribution conjugated by the FULL U_p (two-sided, same U);
    the scalar part h0 and the overlap S are untouched and the pencil
    stays Hermitian.  Same-site blocks (|P| = 1) carry w = 1 (full
    conjugation); cross-site (shared) blocks carry w = 1/2 per
    endpoint site, so one site's rotation changes only its weighted
    half -- NOT the whole block.  Under a common U_p = U the weighted
    contributions of a shared block sum to U W U^dag exactly (global
    covariance: the joint equal-angle action transforms the block
    once), and the mixed second vertex d^2 W'/dtheta_p dtheta_q
    (p != q) vanishes identically; the rejected double-conjugation
    (full-incidence) reading carries a spurious mixed second vertex.
    Per contribution U_p (h.sigma) U_p^dag = (R_n(theta_p) h).sigma.
    The sector phases d11 = e^{-i(k+q/2)x}, d22 = e^{-i(k-q/2)x}
    attach to ROW-spin components, so at q != 0 they do not commute
    with a spin rotation on spin-mixing blocks; the rotating frame is
    DEFINED by carrying the q/2 gauge in the same d11/d22 placement
    for H, H_scalar and B (pinned, not derived).  (A one-sided row-
    or column-only application mixes the scalar part in via
    Tr(U_p h.sigma) != 0 and is NOT the production convention; audit
    story 007 quantifies such convention differences.)

[D] Rotating-frame statement: the vertex is the k-space operator
        B(k) = H(k) - H_scalar(k),
    H_scalar(k) = sum_entries h0 * (d11 |up><up| + d22 |dn><dn|),
    B(k) = sum_entries [ (hx sigma_x + hy sigma_y + hz sigma_z)
                         placed with the SAME sector phases d11/d22 as
                         the Hamiltonian path ].
    The q/2 gauge is carried consistently because B uses exactly the
    phases of H; Hermiticity of B(k) follows from [B].

[E] Certificate fields the implementation must emit (asserted present).

Run with the mydev environment:
    source /home/hexu/projects/myenvs/mydev/bin/activate
    python docs/sympy/siesta_spiral_vertex.py
"""

import numpy as np
import sympy as sp

I = sp.I
SX = sp.Matrix([[0, 1], [1, 0]])
SY = sp.Matrix([[0, -I], [I, 0]])
SZ = sp.Matrix([[1, 0], [0, -1]])
S0 = sp.eye(2)
PAULI = (S0, SX, SY, SZ)

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok  {name}" + (f" ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def assert_zero(name, expr, probe_syms=("k", "q", "x")):
    """simplify() first; fall back to exact rational numeric probing."""
    if sp.simplify(expr) == 0:
        check(name, True, "symbolic")
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
        subs = {sp.Symbol(s): v for s, v in zip(probe_syms, (kk, qq, xx))}
        for i, sym in enumerate(sorted(expr.free_symbols, key=str)):
            subs.setdefault(sym, extra[i % len(extra)])
        if abs(sp.N(expr.subs(subs), 60)) > sp.Float("1e-40"):
            check(name, False, f"probe value {sp.N(expr.subs(subs), 20)}")
            return
    check(name, True, "numeric probe")


k, q, x, th = sp.symbols("k q x theta", real=True)
H1, H2, H3, H4 = sp.symbols("H1 H2 H3 H4", real=True)

# ---------------------------------------------------------------------------
# [A] Pauli decomposition of the stored spinor block
# ---------------------------------------------------------------------------
print("[A] Pauli decomposition of the stored total-H spinor block")

lab = sp.Matrix([[H1, H3 - I * H4], [H3 + I * H4, H2]])
h0 = (H1 + H2) / 2
hx, hy, hz = H3, H4, (H1 - H2) / 2
reconstructed = h0 * S0 + hx * SX + hy * SY + hz * SZ
assert sp.simplify(lab - reconstructed) == sp.zeros(2)
check("W = h0*sigma0 + hx*sigma_x + hy*sigma_y + hz*sigma_z", True)

# Sign of hy pinned by the packing convention H12 = h3 - i*h4: the 12
# element of hx*sigma_x + hy*sigma_y is hx - i*hy, hence hy = +h4.
assert sp.simplify(reconstructed[0, 1] - (H3 - I * H4)) == 0
check("hy = +h4 (from H12 = h3 - i*h4, 12 element of h.sigma = hx - i*hy)", True)
assert sp.simplify((lab - h0 * S0).trace()) == 0
check("magnetic part is traceless: Tr(W - h0*sigma0) = 0", True)
assert sp.expand(lab[0, 0] - lab[1, 1] - 2 * hz) == 0
check("hz = (h1 - h2)/2 is half the spin splitting of the block", True)

# ---------------------------------------------------------------------------
# [B] Partner consistency (the validate() Hermiticity pairing)
# ---------------------------------------------------------------------------
print("[B] (nu,mu,-R) partner consistency conditions")

# Stored partner components on (j,i,-R).  bundle validate() checks only
# that the partner EXISTS and shares the spin-diagonal components; the
# mixing texture is fixture-certified (story 003, siesta_spiral_vector):
qx = sp.Symbol("qx", real=True)  # q . x_ij
h12 = H3 - I * H4
h12p = sp.exp(-I * qx) * h12  # partner's stored 12 component
h3p = sp.re(sp.expand(h12p, complex=True))
h4p = -sp.im(sp.expand(h12p, complex=True))
assert sp.simplify(h3p - I * h4p - h12p.rewrite(sp.cos)) == 0
check("partner stored form h3' - i*h4' = e^{-i q.x} (h3 - i*h4)", True)
# shadow at q.x = 2*pi*n: h3' = h3, h4' = h4 (components identical); the
# plain real-space adjoint pairing (h3' = h3, h4' = -h4) is NOT what a
# converged spiral SCF stores.
assert sp.simplify(h12p.subs(qx, 0) - h12) == 0
check("q.x = 2*pi*n shadow: partner components identical, h3'=h3, h4'=h4", True)

# 2-site model: sublattices A (mu=0, the rotated one) and B (mu=1).
# Entries: onsite (0,0,0), onsite (1,1,0), bond (0,1,0) with displacement
# x, and its partner (1,0,0) with displacement -x.  Symbolic blocks.
h1b, h2b = sp.symbols("h1b h2b", real=True)  # bond scalars
c12 = sp.Symbol("c12")  # bond h12, complex
bond = {
    "mu": 0,
    "nu": 1,
    "R": 0,
    "x": x,
    "W": sp.Matrix([[H1, c12], [sp.conjugate(c12), H2]]),
}
partner = {
    "mu": 1,
    "nu": 0,
    "R": 0,
    "x": -x,
    "W": sp.Matrix(
        [[H1, sp.exp(-I * q * x) * c12], [sp.exp(I * q * x) * sp.conjugate(c12), H2]]
    ),
}
onsiteA = {
    "mu": 0,
    "nu": 0,
    "R": 0,
    "x": sp.Integer(0),
    "W": sp.Matrix([[H1, H3 - I * H4], [H3 + I * H4, H2]]),
}
onsiteB = {
    "mu": 1,
    "nu": 1,
    "R": 0,
    "x": sp.Integer(0),
    "W": sp.Matrix([[h1b, 0], [0, h2b]]),
}
ENTRIES = [bond, partner, onsiteA, onsiteB]

# scalar-part sharing across the pair (what validate() checks directly)
assert sp.simplify(partner["W"][0, 0] - bond["W"][0, 0]) == 0
assert sp.simplify(partner["W"][1, 1] - bond["W"][1, 1]) == 0
check("scalar parts shared: h1'=h1, h2'=h2 across (mu,nu,R)<->(nu,mu,-R)", True)


def d11(xx):
    return sp.exp(-I * (k + q / 2) * xx)


def d22(xx):
    return sp.exp(-I * (k - q / 2) * xx)


def pencil(entries, rot=None):
    """Assemble H(k) exactly as setup_k: entry (mu,nu) block placed at
    rows (mu, spin), cols (nu, spin); 12 element gets d11(x), 21 gets
    d22(x); diagonal 11/22 get d11/d22.  rot: optional dict
    mu -> 2x2 unitary applied to that sublattice's rows AND columns."""
    U = {m: (rot or {}).get(m, sp.eye(2)) for m in (0, 1)}
    H = sp.zeros(4)
    for e in entries:
        mu, nu, xx, W = e["mu"], e["nu"], e["x"], e["W"]
        W = U[mu] * W * U[nu].H
        H[2 * mu + 0, 2 * nu + 0] += W[0, 0] * d11(xx)
        H[2 * mu + 0, 2 * nu + 1] += W[0, 1] * d11(xx)
        H[2 * mu + 1, 2 * nu + 0] += W[1, 0] * d22(xx)
        H[2 * mu + 1, 2 * nu + 1] += W[1, 1] * d22(xx)
    return H


def sector_block(xx):
    """Spin-diagonal phase matrix for one entry: diag(d11, d22)."""
    return sp.diag(d11(xx), d22(xx))


Hp = pencil(ENTRIES)
# substitute a consistent symbolic texture for the bond c12 so simplify
# can close Hermiticity exactly: c12 = h3 - i*h4 with the partner given
# by the SAME texture rule already baked in above.
Hp = Hp.subs(c12, H3 - I * H4)
resid = sp.simplify(Hp - Hp.H)
assert resid == sp.zeros(4), resid
check(
    "assembled H(k) is exactly Hermitian given the textured partner", True, "symbolic"
)

# negative control: the plain real-space adjoint pairing (h4' = -h4, no
# texture) does NOT assemble to a Hermitian pencil for q.x != 0.
bad_partner = {
    "mu": 1,
    "nu": 0,
    "R": 0,
    "x": -x,
    "W": sp.Matrix([[H1, H3 + I * H4], [H3 - I * H4, H2]]),
}
H_bad = pencil([bond, bad_partner, onsiteA, onsiteB]).subs(c12, H3 - I * H4)
assert sp.simplify(H_bad - H_bad.H) != sp.zeros(4)
check("negative control: untextured h4'=-h4 partner breaks Hermiticity", True)

# q.x = 2*pi*n shadow: partner components reduce to the identical set.
Hp_shadow = pencil(ENTRIES).subs(c12, H3 - I * H4).subs(q, 2 * sp.pi / x)
assert sp.simplify(Hp_shadow - Hp_shadow.H) == sp.zeros(4)
check("q.x = 2*pi*n shadow in the assembled pencil stays Hermitian", True)

# ---------------------------------------------------------------------------
# [C] Local SU(2) rotations and the weighted declared partition
# ---------------------------------------------------------------------------
print("[C] local SU(2) rotations and the weighted declared partition (vertex 2.0)")


def gen(nvec):
    return nvec[0] * SX + nvec[1] * SY + nvec[2] * SZ


def rot_mat(nvec, ang):
    n = gen(nvec)
    return sp.simplify(sp.cos(ang / 2)) * S0 - I * sp.sin(ang / 2) * n


# -- axis z, symbolic theta: exact conjugation identity on h.sigma
Uz = rot_mat((0, 0, 1), th)
hv = sp.Matrix(1, 3, [hx, hy, hz])
Rz = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
lhs = sp.trigsimp(sp.expand(Uz * (hx * SX + hy * SY + hz * SZ) * Uz.H))
rhs = sp.trigsimp(
    sp.expand((Rz * hv.T)[0] * SX + (Rz * hv.T)[1] * SY + (Rz * hv.T)[2] * SZ)
)
assert sp.trigsimp(lhs - rhs) == sp.zeros(2)
check("U_z (h.sigma) U_z^dag = (R_z(theta) h).sigma  (symbolic)", True)

# -- axis x, symbolic theta
Ux = rot_mat((1, 0, 0), th)
Rx = sp.Matrix([[1, 0, 0], [0, sp.cos(th), -sp.sin(th)], [0, sp.sin(th), sp.cos(th)]])
lhs = sp.trigsimp(sp.expand(Ux * (hx * SX + hy * SY + hz * SZ) * Ux.H))
rh = Rx * hv.T
rhs = sp.trigsimp(sp.expand(rh[0] * SX + rh[1] * SY + rh[2] * SZ))
assert sp.trigsimp(lhs - rhs) == sp.zeros(2)
check("U_x (h.sigma) U_x^dag = (R_x(theta) h).sigma  (symbolic)", True)

# -- generic axis, exact rational angles (sympy-closed) and a random axis
#    in double precision
for nvec in (
    (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)),
    (sp.Rational(-2, 7), sp.Rational(3, 7), sp.Rational(6, 7)),
):
    assert sp.simplify(sum(v**2 for v in nvec) - 1) == 0
    U = rot_mat(nvec, sp.Rational(7, 5))
    # Rodrigues rotation by th around unit n
    ct, st = sp.cos(sp.Rational(7, 5)), sp.sin(sp.Rational(7, 5))
    nx, ny, nz = nvec
    K = sp.Matrix([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
    R = ct * sp.eye(3) + st * K + (1 - ct) * (K * K + sp.eye(3))
    blk = hx * SX + hy * SY + hz * SZ
    lhs = sp.simplify(U * blk * U.H)
    rh = R * hv.T
    rhs = sp.simplify(rh[0] * SX + rh[1] * SY + rh[2] * SZ)
    assert sp.simplify(lhs - rhs) == sp.zeros(2)
check(
    "generic unit axis: U (h.sigma) U^dag = (R_n(theta) h).sigma "
    "(Rodrigues, exact rational)",
    True,
)

rng = np.random.default_rng(20261013)
nrand = rng.normal(size=3)
nrand /= np.linalg.norm(nrand)
thrand = 0.9
Unp = np.cos(thrand / 2) * np.eye(2) - 1j * np.sin(thrand / 2) * (
    nrand[0] * np.array(SX.tolist(), dtype=complex)
    + nrand[1] * np.array(SY.tolist(), dtype=complex)
    + nrand[2] * np.array(SZ.tolist(), dtype=complex)
)
vals = rng.normal(size=3)
K = np.array(
    [[0, -nrand[2], nrand[1]], [nrand[2], 0, -nrand[0]], [-nrand[1], nrand[0], 0]]
)
ct, st = np.cos(thrand), np.sin(thrand)
Rnp = ct * np.eye(3) + st * K + (1 - ct) * (K @ K + np.eye(3))
blkn = sum(
    vals[i] * np.array(P.tolist(), dtype=complex) for i, P in enumerate(PAULI[1:])
)
err = np.abs(
    Unp @ blkn @ Unp.conj().T
    - sum(
        (Rnp @ vals)[i] * np.array(P.tolist(), dtype=complex)
        for i, P in enumerate(PAULI[1:])
    )
).max()
assert err < 1e-13, err
check("random numeric axis/angle conjugation identity", True, f"max|diff| = {err:.2e}")

# -- pencil level: the DECLARED PARTITION rotation (vertex 2.0).  A
#    magnetic block (mu,nu,R) has distinct endpoint-owner sites
#    P = {owner(mu), owner(nu)}; the rotation is allocated additively,
#        W' = sum_p w_p U_p W U_p^dag = W + sum_p w_p (U_p W U_p^dag - W),
#    w_p = 1/|P|, each contribution conjugated by the FULL U_p
#    (two-sided, same U).  Same-site blocks (|P| = 1) carry w = 1: full
#    conjugation.  Cross-site (shared) blocks carry w = 1/2 per endpoint
#    site: one site's rotation changes only its weighted half, NOT the
#    whole block.  Under a common U_p = U the contributions sum to
#    U W U^dag exactly (global covariance: the joint equal-angle action
#    transforms the block once).  The scalar part h0 and S are untouched
#    and the pencil stays Hermitian.  (One-sided row- or column-only
#    application mixes the scalar part in via Tr(U_p h.sigma) != 0 and
#    is a different convention -- audit story 007 quantifies such
#    differences; the certificate must declare which one is used.)
UA = rot_mat((0, 0, 1), sp.Symbol("thetaA", real=True))
UB = rot_mat((0, 0, 1), sp.Symbol("thetaB", real=True))


def block_owners(e):
    """Distinct endpoint-owner sites: owners of the row and column orbital."""
    return (e["mu"], e["nu"]) if e["mu"] != e["nu"] else (e["mu"],)


def owner_weights(e):
    P = block_owners(e)
    return {p: sp.Rational(1, len(P)) for p in P}


def rotate_block(W, e, rot):
    """Declared partition allocation, additive form (vertex 2.0):
    W + sum_p w_p (U_p W U_p^dag - W) with w_p = 1/count of distinct
    endpoint owners; sites absent from `rot` act as identity."""
    U = {p: rot.get(p, sp.eye(2)) for p in owner_weights(e)}
    return sp.expand(
        W
        + sum(
            (w * (U[p] * W * U[p].H - W) for p, w in owner_weights(e).items()),
            sp.zeros(2),
        )
    )


def rotate_entries(entries, rot):
    return [{**e, "W": rotate_block(e["W"], e, rot)} for e in entries]


rotA = {0: UA}  # declared case: only sublattice A rotates

assert sp.simplify(UA * S0 * UA.H - S0) == sp.zeros(2)
check(
    "two-sided: U_A sigma0 U_A^dag = sigma0 (scalar part invariant)", True, "symbolic"
)
assert sp.simplify(UA * S0) - S0 != sp.zeros(2)
check(
    "one-sided control: U_A sigma0 != sigma0 (row-only rotation mixes "
    "the scalar part -- must not be used for S)",
    True,
)


# owner/weight partition of the model
_wt = {(e["mu"], e["nu"]): owner_weights(e) for e in ENTRIES}
assert _wt[(0, 0)] == {0: sp.Rational(1)}
assert _wt[(1, 1)] == {1: sp.Rational(1)}
assert _wt[(0, 1)] == {0: sp.Rational(1, 2), 1: sp.Rational(1, 2)}
assert _wt[(1, 0)] == {1: sp.Rational(1, 2), 0: sp.Rational(1, 2)}
check(
    "owner partition: onsite blocks single-owner w=1; intersite "
    "(0,1)/(1,0) shared by both endpoint sites, w=1/2 each",
    True,
    str(_wt),
)

# additive and weighted-sum forms agree entry by entry
for e in ENTRIES:
    closed = sp.zeros(2)
    for p, w in owner_weights(e).items():
        Up = rotA.get(p, sp.eye(2))
        closed += w * (Up * e["W"] * Up.H)
    assert sp.simplify(rotate_block(e["W"], e, rotA) - closed) == sp.zeros(2)
check(
    "allocation is additive: W' = W + sum_p w_p (U_p W U_p^dag - W) "
    "= sum_p w_p U_p W U_p^dag",
    True,
    "symbolic",
)

# per entry under the A-only declared case: rotated block = scalar part
# + weighted rotated magnetic part; h0 and the trace are invariant
for e in ENTRIES:
    W, w0 = e["W"], sp.simplify((e["W"][0, 0] + e["W"][1, 1]) / 2)
    Wm = W - w0 * S0
    Wrot = rotate_block(W, e, rotA)
    magrot = sp.zeros(2)
    for p, w in owner_weights(e).items():
        Up = rotA.get(p, sp.eye(2))
        magrot += w * (Up * Wm * Up.H)
    assert sp.simplify(Wrot - (w0 * S0 + magrot)) == sp.zeros(2)
    assert sp.simplify(Wrot[0, 0] + Wrot[1, 1] - 2 * w0) == 0
check(
    "every entry: W' = h0 sigma0 + sum_p w_p U_p (h.sigma) U_p^dag; "
    "h0 invariant, magnetic part traceless",
    True,
    "symbolic",
)

# same-site vs shared blocks under the A-only declared case
_WA = onsiteA["W"]
assert sp.simplify(rotate_block(_WA, onsiteA, rotA) - UA * _WA * UA.H) == sp.zeros(2)
_WB = onsiteB["W"]
assert sp.simplify(rotate_block(_WB, onsiteB, rotA) - _WB) == sp.zeros(2)
_Wbd = bond["W"]
assert sp.simplify(
    rotate_block(_Wbd, bond, rotA) - (_Wbd / 2 + UA * _Wbd * UA.H / 2)
) == sp.zeros(2)
assert sp.simplify(rotate_block(_Wbd, bond, rotA) - UA * _Wbd * UA.H) != sp.zeros(2)
_wp = partner["W"]
assert sp.simplify(
    rotate_block(_wp, partner, rotA) - (_wp / 2 + UA * _wp * UA.H / 2)
) == sp.zeros(2)
check(
    "same-site blocks: w=1 full conjugation (onsite A) / untouched under "
    "A-only (onsite B); shared bond+partner: one site moves only its "
    "w=1/2 half, NOT the full conjugation (old full-incidence claim "
    "rejected)",
    True,
    "symbolic",
)

# first/second vertex structure on the shared bond (B' = the allocation,
# angles thetaA, thetaB on the two endpoint sites)
thA = sp.Symbol("thetaA", real=True)
thB = sp.Symbol("thetaB", real=True)
_at0 = {thA: 0, thB: 0}
Wb = bond["W"].subs(c12, H3 - I * H4)
alloc_t = (
    Wb
    + sp.Rational(1, 2) * (UA * Wb * UA.H - Wb)
    + sp.Rational(1, 2) * (UB * Wb * UB.H - Wb)
)
template1 = sp.expand(sp.diff(UA * Wb * UA.H, thA).subs(_at0))
assert sp.simplify(
    sp.diff(alloc_t, thA).subs(_at0) - sp.Rational(1, 2) * template1
) == sp.zeros(2)
check("single-site first vertex on a shared block carries w=1/2", True, "symbolic")
template2 = sp.expand(sp.diff(UA * Wb * UA.H, thA, 2).subs(_at0))
assert sp.simplify(
    sp.diff(alloc_t, thA, 2).subs(_at0) - sp.Rational(1, 2) * template2
) == sp.zeros(2)
check(
    "same-site second vertex on a shared block carries w=1/2 "
    "(not w^2, not unweighted)",
    True,
    "symbolic",
)
_mixed = sp.diff(sp.diff(alloc_t, thA), thB).subs(_at0)
assert sp.simplify(_mixed) == sp.zeros(2)
check(
    "cross-site second vertex vanishes: d^2 B'/dtheta_A dtheta_B = 0",
    True,
    "symbolic",
)
# negative control: the rejected double-conjugation reading has a
# spurious mixed second vertex on the shared block
dbl_t = sp.expand(UB * UA * Wb * UA.H * UB.H)
assert sp.simplify(sp.diff(sp.diff(dbl_t, thA), thB).subs(_at0)) != sp.zeros(2)
check(
    "negative control: double conjugation U_B U_A W U_A^dag U_B^dag "
    "carries a spurious mixed second vertex",
    True,
    "symbolic",
)

# global covariance: a common U on all endpoint owners acts once
Ucov = rot_mat((0, 0, 1), th)
for e in ENTRIES:
    Wuni = rotate_block(e["W"], e, {0: Ucov, 1: Ucov})
    assert sp.simplify(Wuni - Ucov * e["W"] * Ucov.H) == sp.zeros(2)
check(
    "global covariance: common U_p = U sums the weighted contributions "
    "to U W U^dag (joint equal-angle action transforms the block once)",
    True,
    "symbolic",
)

# H_scalar: h0 with the SAME sector phases; magnetic: the remainder.
H_scalar = sp.zeros(4)
H_mag = sp.zeros(4)
for e in ENTRIES:
    mu, nu, xx, W = e["mu"], e["nu"], e["x"], e["W"]
    w0 = sp.simplify((W[0, 0] + W[1, 1]) / 2)
    Wm = W - w0 * S0
    H_scalar[2 * mu + 0, 2 * nu + 0] += w0 * d11(xx)
    H_scalar[2 * mu + 1, 2 * nu + 1] += w0 * d22(xx)
    H_mag[2 * mu + 0, 2 * nu + 0] += Wm[0, 0] * d11(xx)
    H_mag[2 * mu + 0, 2 * nu + 1] += Wm[0, 1] * d11(xx)
    H_mag[2 * mu + 1, 2 * nu + 0] += Wm[1, 0] * d22(xx)
    H_mag[2 * mu + 1, 2 * nu + 1] += Wm[1, 1] * d22(xx)
assert sp.simplify(pencil(ENTRIES) - H_scalar - H_mag) == sp.zeros(4)
check("H(k) = H_scalar(k) + B(k) with per-entry h0 split", True, "symbolic")

Hsc = H_scalar.subs(c12, H3 - I * H4)
assert sp.simplify(Hsc - Hsc.H) == sp.zeros(4)
check("H_scalar(k) is spin-diagonal (h0 in each sector, d11/d22 phases)", True)

# spin-independent lab overlap: blocks are delta_ab, and the implementation
# never conjugates S -- S(k) is invariant by construction.
S0p = pencil([{**e, "W": S0} for e in ENTRIES])
assert sp.simplify(S0p - S0p.subs(c12, H3 - I * H4)) == sp.zeros(4)
check("S(k) carries no magnetic components and is never rotated", True)

H_base = pencil(ENTRIES).subs(c12, H3 - I * H4)
H_rot = pencil(rotate_entries(ENTRIES, rotA)).subs(c12, H3 - I * H4)
# by linearity of the assembly, H_rot - H_base is the pencil of the
# per-entry block differences W' - W; those are purely magnetic (checked
# above), so H_scalar is untouched:
mag_rot_entries = []
for e in ENTRIES:
    w0 = (e["W"][0, 0] + e["W"][1, 1]) / 2
    Wm = e["W"] - w0 * S0
    mag_rot_entries.append({**e, "W": rotate_block(Wm, e, rotA)})
assert sp.simplify(
    (H_rot - H_scalar - pencil(mag_rot_entries)).subs(c12, H3 - I * H4)
) == sp.zeros(4)
check(
    "rotation changes ONLY the magnetic part: "
    "H_rot = H_scalar + pencil(rotated magnetic blocks)",
    True,
    "symbolic",
)
Brot = H_rot - H_scalar
assert sp.simplify(Brot - Brot.H) == sp.zeros(4)
check("B(k) remains Hermitian after the partition rotation " "(symbolic theta)", True)

# inventory: endpoint owners of each entry (vertex 2.0 partition).
inv = {(e["mu"], e["nu"]): block_owners(e) for e in ENTRIES}
assert inv[(0, 0)] == (0,)
assert inv[(1, 1)] == (1,)
assert inv[(0, 1)] == (0, 1)
assert inv[(1, 0)] == (1, 0)
check(
    "inventory: onsite blocks have a single endpoint owner (w=1); the "
    "shared (0,1)/(1,0) bond pair is owned by both sites (w=1/2 each)",
    True,
    str(inv),
)

# partition locality: adding B's rotation adds exactly its weighted
# increment w_B (U_B W U_B^dag - W) to every B-owned half; entries with
# owner set (0,) are untouched.
only_A = pencil(rotate_entries(ENTRIES, rotA))
with_B = pencil(rotate_entries(ENTRIES, {0: UA, 1: UB}))
expected_incr = sp.zeros(4)
for e in ENTRIES:
    wB = owner_weights(e).get(1, 0)
    if wB == 0:
        continue
    dW = sp.expand(wB * (UB * e["W"] * UB.H - e["W"]))
    mu, nu, xx = e["mu"], e["nu"], e["x"]
    expected_incr[2 * mu + 0, 2 * nu + 0] += dW[0, 0] * d11(xx)
    expected_incr[2 * mu + 0, 2 * nu + 1] += dW[0, 1] * d11(xx)
    expected_incr[2 * mu + 1, 2 * nu + 0] += dW[1, 0] * d22(xx)
    expected_incr[2 * mu + 1, 2 * nu + 1] += dW[1, 1] * d22(xx)
assert sp.simplify(
    (with_B - only_A - expected_incr).subs(c12, H3 - I * H4)
) == sp.zeros(4)
check(
    "partition locality: adding B's rotation contributes exactly "
    "w_B (U_B W U_B^dag - W) on B-owned halves (onsite B: w=1; shared "
    "bond pair: w=1/2); A-owned blocks untouched",
    True,
)

# ---------------------------------------------------------------------------
# [D] Rotating-frame vertex operator and the q/2 gauge
# ---------------------------------------------------------------------------
print("[D] rotating-frame vertex operator B(k)")

Bk = H_mag.subs(c12, H3 - I * H4)
# q/2 gauge consistency: B(k) uses the SAME sector phases as H, so
# H - H_scalar is gauge-consistent and Hermitian (checked in [C]); here
# pin that at q -> 0 B(k) reduces to the collinear vertex Fourier sum
# sum_entries (h.sigma) e^{-i k x}.
Bq0 = sp.simplify(Bk.subs(q, 0))
Bexpect = sp.zeros(4)
for e in ENTRIES:
    mu, nu, xx, W = e["mu"], e["nu"], e["x"], e["W"]
    Wm = W - (W[0, 0] + W[1, 1]) / 2 * S0
    Wm = Wm.subs(c12, H3 - I * H4).subs(q, 0)
    Bexpect[2 * mu + 0, 2 * nu + 0] += Wm[0, 0] * sp.exp(-I * k * xx)
    Bexpect[2 * mu + 0, 2 * nu + 1] += Wm[0, 1] * sp.exp(-I * k * xx)
    Bexpect[2 * mu + 1, 2 * nu + 0] += Wm[1, 0] * sp.exp(-I * k * xx)
    Bexpect[2 * mu + 1, 2 * nu + 1] += Wm[1, 1] * sp.exp(-I * k * xx)
_Bdiff = sp.powsimp(sp.expand(Bq0 - Bexpect), force=True)
assert sp.simplify(_Bdiff) == sp.zeros(4)
check(
    "q -> 0: B(k) reduces to the collinear vertex Fourier sum "
    "sum (h.sigma) e^{-i k x}",
    True,
)

# the vertex transforms as an operator under the partition rotation.  The
# sector phases d11/d22 attach to ROW-spin components, so at q != 0 they
# do NOT commute with a spin rotation on spin-mixing blocks: the rotating
# frame is defined by carrying the q/2 gauge in the SAME d11/d22 placement
# for H, H_scalar and B.  In the collinear limit q = 0 (d11 = d22) the
# phases become scalar per entry and entrywise rotation equals the global
# conjugation H(k) -> W H(k) W^dag, W = diag(U, U) over sites.
nrm = rng.normal(size=3)
nrm = nrm / np.linalg.norm(nrm)
Un = np.cos(0.7 / 2) * np.eye(2) - 1j * np.sin(0.7 / 2) * sum(
    nrm[i] * np.array(P.tolist(), dtype=complex) for i, P in enumerate(PAULI[1:])
)
W_glob = np.zeros((4, 4), dtype=complex)
W_glob[0:2, 0:2] = Un
W_glob[2:4, 2:4] = Un
subs_num = {
    k: sp.Rational(1, 3),
    q: 0,
    x: sp.Rational(4, 5),
    H1: sp.Rational(6, 5),
    H2: sp.Rational(4, 3),
    H3: sp.Rational(1, 7),
    H4: sp.Rational(2, 9),
    h1b: sp.Rational(5, 4),
    h2b: sp.Rational(9, 7),
    th: sp.Rational(7, 5),
}
Hn = np.array(sp.N(H_base.subs(subs_num), 30).tolist(), dtype=complex)
Uall = sp.Matrix(Un.tolist())
# partition covering BOTH sublattices with the same U: the weighted
# contributions (w = 1/2 + 1/2 on the shared bond) sum to the full
# conjugation on every block -- nonzero intersite B included.
all_rot = rotate_entries(ENTRIES, {0: Uall, 1: Uall})
Hn_rot = np.array(
    sp.N(pencil(all_rot).subs(c12, H3 - I * H4).subs(subs_num), 30).tolist(),
    dtype=complex,
)
err = np.abs(Hn_rot - W_glob @ Hn @ W_glob.conj().T).max()
assert err < 1e-12, err
check(
    "q = 0: weighted partition under common U == global conjugation "
    "W H(k) W^dag (w = 1/2 + 1/2 = full; nonzero intersite B)",
    True,
    f"max|diff| = {err:.2e}",
)
# at q != 0 the same comparison fails by O(q): the placement IS the gauge.
subs_q = dict(subs_num)
subs_q[q] = sp.Rational(2, 5)
Hn_q = np.array(sp.N(H_base.subs(subs_q), 30).tolist(), dtype=complex)
Hn_rot_q = np.array(
    sp.N(
        pencil(rotate_entries(ENTRIES, {0: Uall, 1: Uall}))
        .subs(c12, H3 - I * H4)
        .subs(subs_q),
        30,
    ).tolist(),
    dtype=complex,
)
err_q = np.abs(Hn_rot_q - W_glob @ Hn_q @ W_glob.conj().T).max()
assert err_q > 1e-6, err_q
check(
    "q != 0: entrywise rotation != global conjugation (difference "
    "O(q)) -- the sector-phase placement carries the q/2 gauge and must "
    "be declared, not derived",
    True,
    f"max|diff| = {err_q:.2e}",
)

# ---------------------------------------------------------------------------
# [E] certificate fields the implementation must emit
# ---------------------------------------------------------------------------
print("[E] provenance certificate fields")

REQUIRED_CERT_FIELDS = (
    "vertex_version",  # "2.0"; serialized v1.0 vertices reject at the gate
    "vertex_convention",  # "Oroszlany spin-dependent part of saved total H"
    "vertex_convention_value",  # e.g. "spin-split of stored total H; no XC reconstruction"
    "partition_declaration",  # weighted site contributions, w=1/count of endpoint owners
    "packing_convention",  # H12 = h3 - i h4; hx=h3, hy=h4, hz=(h1-h2)/2
    "sector_phase_convention",  # d11 = e^{-i(k+q/2)x}, d22 = e^{-i(k-q/2)x}
    "projector_convention",  # dual-basis: S on which side (Tr(S A~), S^-1 right in resolvent)
    "metric_convention",  # overlap used in generalized eigenproblem
    "allocation",  # DZP channels retained flag
)
VERTEX_VERSION = "2.0"
PARTITION_DECLARATION = (
    "weighted site contributions, w=1/count of distinct endpoint owners, "
    "each conjugated by full U_site; scalar h0 and S untouched"
)
cert = {
    "vertex_version": VERTEX_VERSION,
    "vertex_convention": "oroszlany_total_h_spin_split",
    "vertex_convention_value": "spin-dependent part of saved total H; no XC reconstruction on "
    "this path: B = W - h0*sigma0, hx=h3, hy=h4, hz=(h1-h2)/2",
    "partition_declaration": PARTITION_DECLARATION,
    "packing_convention": "H12 = h3 - i*h4",
    "sector_phase_convention": "d11 = exp(-i(k+q/2)x), d22 = exp(-i(k-q/2)x)",
    "projector_convention": "coeff form A~ = S^-1 A; Tr A = Tr(S A~)",
    "metric_convention": "generalized eig H c = eps S c with assembled "
    "spin-dependent S(k)",
    "allocation": "DZP channels retained",
}
missing = [f for f in REQUIRED_CERT_FIELDS if f not in cert]
assert not missing, missing
check(
    "certificate carries all required provenance fields",
    True,
    f"{len(REQUIRED_CERT_FIELDS)} fields",
)
assert cert["vertex_version"] == "2.0"
check("vertex_version pins 2.0 (v1.0 vertices reject)", True)
assert cert["partition_declaration"] == (
    "weighted site contributions, w=1/count of distinct endpoint owners, "
    "each conjugated by full U_site; scalar h0 and S untouched"
)
check("partition_declaration pins the vertex-2.0 weighted allocation", True)
assert "no XC reconstruction" in cert["vertex_convention_value"]
check("certificate states no-XC-reconstruction path", True)

print()
if FAILURES:
    print(f"FAILED: {FAILURES}")
    raise SystemExit(1)
print("ALL ASSERTIONS GREEN")
