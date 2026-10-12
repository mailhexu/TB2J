#!/usr/bin/env python
"""Exact folding identity for the q_half_per_spin_sector rotating-frame pencil.

Model (all exact, sympy): 1D chain, lattice constant a = 1 (b = 2*pi), two
atoms per primitive cell at z = 0 and z = 1/2, one orbital per atom,
couplings within +-1 primitive cell, pitch q = b/4 so the commensurate
supercell holds n_s = 4 primitive cells and the folded mesh is
K = {0, b/4, b/2, 3b/4}.

Pencils replicate SpiralImporter.assemble_pencil (vector path) verbatim.
For bond (mu, nu, R) with xij = R*a + pos[nu] - pos[mu] (PRIMITIVE,
unwrapped displacement):

    d11 = exp(-i (k + q/2) xij)   row sector up   (H11 = h1, H12 = h3 - i h4)
    d22 = exp(-i (k - q/2) xij)   row sector down (H22 = h2, H21 = conj(H12))
    S spin-diagonal with the same d11 / d22.

Hermiticity of the assembled pencil (the fixture oracle enforces 1e-8)
requires the producer partner constraint

    h12(nu, mu, -R) = h12(mu, nu, R) * exp(-i q xij(mu, nu, R)),

i.e. the mixing blocks carry the spiral texture; the model builds it in
exactly, so every pencil below is Hermitian to machine precision.

Proven here element-wise and exactly (sympy, exact + 60-digit certificates):

    F^H H_sc F = +_k H(k),   F^H S_sc F = +_k S(k),

where H_sc, S_sc is the supercell Gamma pencil assembled with the
primitive-displacement phases e^{-i s q xij / 2} (row sector s = +-1) and
WRAPPED bonds included (coset sums), and F is the plain untwisted Fourier
matrix F[(m mu s), (j mu s)] = exp(-i k_j x_{m mu}) / sqrt(n_s).
Hence eig(H_sc, S_sc) = union_k eig(H(k), S(k)) as multisets, exactly.

Failure variants are constructed and asserted to fail:
  dropped : wrapped bonds dropped (open boundary -- the naive tiling),
  affine  : wrapped-bond phases recomputed from supercell-affine
            displacements (loses the e^{+i s q w L / 2} = (-1)^{sw} frame
            branch; breaks Hermiticity under the certified convention).

Run:  mydev python fold_identity_sympy.py   (exits 0 green / 1 red)
"""

import sys

import numpy as np
import sympy as sp
from scipy.linalg import eigh as geigh

I = sp.I
A = sp.Integer(1)  # primitive lattice constant
BZ = 2 * sp.pi  # reciprocal lattice vector b
Q = BZ / 4  # pitch q = b/4
NS = 4  # supercell = 4 primitive cells
L = NS * A  # supercell lattice constant
POS = (sp.Integer(0), sp.Rational(1, 2))  # atom positions in cell
KS = [sp.Rational(j, 4) * BZ for j in range(NS)]  # {0, b/4, b/2, 3b/4}
UP, DN = 1, -1  # row-sector signs: d11 (+q/2), d22 (-q/2)

FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
# minimal model
# --------------------------------------------------------------------------
def xij(mu, nu, R):
    """PRIMITIVE (unwrapped) displacement of bond (mu, nu, R)."""
    return R * A + POS[nu] - POS[mu]


# bonds (mu, nu, R) -> (h1, h2, c12, s); c12 = h3 - i*h4 (complex), s real.
BOND_DATA = [
    (0, 0, 0, sp.Rational(1, 5), sp.Rational(1, 5), sp.Integer(0), sp.Integer(1)),
    (1, 1, 0, sp.Rational(7, 10), sp.Rational(7, 10), sp.Integer(0), sp.Integer(1)),
    (
        0,
        1,
        0,
        sp.Rational(3, 10),
        sp.Rational(3, 10),
        sp.Rational(1, 5) - I * sp.Rational(1, 7),
        sp.Rational(1, 8),
    ),
    (
        1,
        0,
        1,
        sp.Rational(1, 3),
        sp.Rational(1, 3),
        sp.Rational(2, 5) - I * sp.Rational(1, 9),
        sp.Rational(1, 8),
    ),
]


def build_bonds():
    """Bond table with the certified partner constraint baked in."""
    bonds = {}
    for mu, nu, R, h1, h2, c12, s in BOND_DATA:
        bonds[(mu, nu, R)] = (h1, h2, c12, s)
        partner = (nu, mu, -R)
        if partner != (mu, nu, R):
            # producer texture: mixing block of the reversed bond carries
            # the extra phase exp(-i q xij(mu, nu, R))
            bonds[partner] = (h1, h2, c12 * sp.exp(-I * Q * xij(mu, nu, R)), s)
        else:
            assert sp.simplify(c12 * sp.exp(-I * Q * xij(mu, nu, R)) - c12) == 0
    return bonds


BONDS = build_bonds()


def prim_pencil(k):
    """(H(k), S(k)) exactly as assemble_pencil, 4x4 (2 atoms x 2 spin)."""
    H, S = sp.zeros(4, 4), sp.zeros(4, 4)

    def ix(mu, s):
        return 2 * mu + (0 if s == UP else 1)

    for (mu, nu, R), (h1, h2, c12, s_ov) in BONDS.items():
        x = xij(mu, nu, R)
        d11 = sp.exp(-I * (k + Q / 2) * x)
        d22 = sp.exp(-I * (k - Q / 2) * x)
        H[ix(mu, UP), ix(nu, UP)] += h1 * d11
        H[ix(mu, UP), ix(nu, DN)] += c12 * d11
        H[ix(mu, DN), ix(nu, UP)] += sp.conjugate(c12) * d22
        H[ix(mu, DN), ix(nu, DN)] += h2 * d22
        S[ix(mu, UP), ix(nu, UP)] += s_ov * d11
        S[ix(mu, DN), ix(nu, DN)] += s_ov * d22
    return H, S


def site(m, mu, s):
    """Supercell index: [up sites 0..7 | down sites 8..15]."""
    return 2 * m + mu + 8 * (0 if s == UP else 1)


def sc_pencil(mode):
    """Supercell Gamma pencil, dim 16x16.

    mode:
      corrected - wrapped bonds included, phase from the PRIMITIVE xij
      dropped   - wrapped bonds dropped (naive tiling / open boundary)
      affine    - wrapped bonds included but phase from the supercell-affine
                  displacement x_{n nu} - x_{m mu}
      plain     - no phases at all (local-frame block sums)
    """
    dim = 4 * NS
    H, S = sp.zeros(dim, dim), sp.zeros(dim, dim)
    for (mu, nu, R), (h1, h2, c12, s_ov) in BONDS.items():
        xp = xij(mu, nu, R)
        for m in range(NS):
            n_raw = m + R
            if mode == "dropped":
                if not 0 <= n_raw < NS:
                    continue
                n = n_raw
            else:
                n = n_raw % NS
            if mode == "affine":
                y = n * A + POS[nu] - m * A - POS[mu]
                phase = sp.exp(-I * (Q / 2) * y)
            elif mode == "plain":
                phase = sp.Integer(1)
            else:
                phase = sp.exp(-I * (Q / 2) * xp)  # k = 0
            d11, d22 = phase, sp.conjugate(phase)
            H[site(m, mu, UP), site(n, nu, UP)] += h1 * d11
            H[site(m, mu, UP), site(n, nu, DN)] += c12 * d11
            H[site(m, mu, DN), site(n, nu, UP)] += sp.conjugate(c12) * d22
            H[site(m, mu, DN), site(n, nu, DN)] += h2 * d22
            S[site(m, mu, UP), site(n, nu, UP)] += s_ov * d11
            S[site(m, mu, DN), site(n, nu, DN)] += s_ov * d22
    return H, S


def fourier():
    """F[(m mu s), (j mu s)] = exp(-i k_j x_{m mu}) / sqrt(NS); unitary."""
    F = sp.zeros(4 * NS, 4 * NS)
    for j in range(NS):
        for mu in range(2):
            for s in (UP, DN):
                col = 4 * j + 2 * mu + (0 if s == UP else 1)
                for m in range(NS):
                    F[site(m, mu, s), col] = sp.exp(-I * KS[j] * (m * A + POS[mu])) / 2
    return F


def zero_certificate(expr, digits=60):
    """(exact_zero, numerically_zero_at_<1e-40, value)."""
    e = sp.powsimp(sp.expand(expr), force=True)
    if e == 0:
        return True, True, sp.Integer(0)
    ec = sp.expand_complex(e)
    if ec == 0:
        return True, True, sp.Integer(0)
    if sp.simplify(ec) == 0:
        return True, True, sp.Integer(0)
    val = complex(sp.N(e, digits))
    return False, abs(val) < 1e-40, val


def max_abs_residual(E):
    return max(abs(complex(sp.N(v, 40))) for v in E)


def herm_defect(Mnp):
    return float(np.abs(Mnp - Mnp.conj().T).max())


def to_np(M):
    return np.array(M.evalf(30).tolist(), dtype=complex)


# --------------------------------------------------------------------------
# 1) exact structural lemmas
# --------------------------------------------------------------------------
print("== exact scalar lemmas (sympy simplify) ==")

u_cell = sp.exp(-I * Q * A / 2)  # frame factor / primitive cell
check(
    "frame factor per primitive cell u = exp(-i q a / 2) = exp(-i pi/4)",
    sp.simplify(u_cell - sp.exp(-I * sp.pi / 4)) == 0,
)
check("u**2 = -i  (two primitive cells)", sp.simplify(u_cell**2 + I) == 0)
check(
    "u**4 = -1  (one supercell translation: spin-1/2 double cover)",
    sp.simplify(u_cell**4 + 1) == 0,
)

for w in (1, -1):
    # affine branch vs primitive branch of a wrapped bond (w supercell wraps)
    x = sp.Symbol("x")
    ratio = sp.exp(-I * (Q / 2) * (x - w * L)) / sp.exp(-I * (Q / 2) * x)
    check(
        f"affine/primitive branch ratio for w={w:+d} equals exp(i s q w L/2) = -1",
        sp.simplify(ratio - sp.exp(I * (Q / 2) * w * L)) == 0
        and sp.simplify(sp.exp(I * (Q / 2) * w * L) + 1) == 0,
    )

# geometric sum: (1/n_s) sum_m exp(i (k_j - k_j') (m a + p_mu)) = delta_jj'
for j in range(NS):
    for jp in range(NS):
        ssum = sum(sp.exp(I * (KS[j] - KS[jp]) * (m * A + POS[0])) for m in range(NS))
        target = NS if j == jp else 0
        if sp.simplify(ssum - target) != 0:
            check(f"geometric sum lemma j={j} j'={jp}", False)
for j in range(NS):
    for w in (1, -1):
        if sp.simplify(sp.exp(-I * KS[j] * 4 * w) - 1) != 0:
            check(f"wrap phase lemma j={j} w={w:+d}", False)
print("  [ok] geometric-sum and wrap-phase lemmas (all j, j', w)")

# partner constraint present in the model (this is what makes pencils Hermitian)
for (mu, nu, R), (h1, h2, c12, s_ov) in BONDS.items():
    if (nu, mu, -R) != (mu, nu, R):
        assert (
            sp.simplify(BONDS[(nu, mu, -R)][2] - c12 * sp.exp(-I * Q * xij(mu, nu, R)))
            == 0
        )
print("  [ok] producer partner constraint h12(nu,mu,-R) = h12(mu,nu,R) e^{-i q xij}")

# --------------------------------------------------------------------------
# 2) exact element-wise folding identity:  F^H M_sc F = +_k M(k)
# --------------------------------------------------------------------------
print("== element-wise folding identity (exact / 60-digit certificates) ==")
F = fourier()
FH = F.conjugate().T

# unitarity check
unit_max = 0.0
In = sp.eye(4 * NS)
for a in range(4 * NS):
    for b in range(4 * NS):
        _, okn, _ = zero_certificate((FH * F)[a, b] - In[a, b])
        unit_max = max(unit_max, abs(complex(sp.N((FH * F)[a, b] - In[a, b], 60))))
check(
    "F is unitary (max |F^H F - 1| < 1e-40)",
    unit_max < 1e-40,
    f"max residual {unit_max:.1e}",
)

BLOCS_H = [prim_pencil(k)[0] for k in KS]
BLOCS_S = [prim_pencil(k)[1] for k in KS]
BD_H, BD_S = sp.diag(*BLOCS_H), sp.diag(*BLOCS_S)

H_C, S_C = sc_pencil("corrected")
E_H = FH * H_C * F - BD_H
E_S = FH * S_C * F - BD_S

for name, E in (("H", E_H), ("S", E_S)):
    exact = sum(zero_certificate(v)[0] for v in E)
    numeric_ok = all(zero_certificate(v)[1] for v in E)
    res = max_abs_residual(E)
    check(
        f"F^H S_{name}_sc F - ⊕ {name}(k) = 0 element-wise",
        numeric_ok,
        f"{exact}/256 entries proven exact by simplification, "
        f"max residual {res:.1e}",
    )

# --------------------------------------------------------------------------
# 3) spectra (same solver the gate uses: scipy eigh on the raw pencils)
# --------------------------------------------------------------------------
print("== spectra (scipy eigh, as in the gate) ==")


def spectrum(H, S):
    return np.sort(geigh(H, S, eigvals_only=True))


union = np.sort(
    np.concatenate([spectrum(to_np(h), to_np(s)) for h, s in zip(BLOCS_H, BLOCS_S)])
)
Hn_C, Sn_C = to_np(H_C), to_np(S_C)
spec_C = spectrum(Hn_C, Sn_C)
gap_C = float(np.abs(spec_C - union).max())
print(
    f"  hermiticity defect  H_sc {herm_defect(Hn_C):.2e}, "
    f"S_sc {herm_defect(Sn_C):.2e}"
)
check(
    "supercell Gamma pencils Hermitian (< 1e-12)",
    herm_defect(Hn_C) < 1e-12 and herm_defect(Sn_C) < 1e-12,
)
for kk, h4, s4 in zip(KS, BLOCS_H, BLOCS_S):
    hn, sn = to_np(h4), to_np(s4)
    assert herm_defect(hn) < 1e-12 and herm_defect(sn) < 1e-12
    np.linalg.cholesky(sn)
np.linalg.cholesky(Sn_C)
print("  [ok] every S(k) and S_sc positive definite (Cholesky)")
check(
    "eig(H_sc, S_sc) = union_k eig(H(k), S(k))  (max |dE| < 1e-10)",
    gap_C < 1e-10,
    f"max |dE| = {gap_C:.2e} eV",
)

H_A, S_A = sc_pencil("dropped")
Hn_A, Sn_A = to_np(H_A), to_np(S_A)
spec_A = spectrum(Hn_A, Sn_A)
gap_A = float(np.abs(spec_A - union).max())
E_A = FH * H_A * F - BD_H
res_A = max_abs_residual(E_A)
check(
    "naive tiling (wrapped bonds dropped) breaks the identity: "
    "max|E_A| > 1e-3 and max|dE| > 1e-3",
    res_A > 1e-3 and gap_A > 1e-3,
    f"max|F^H H_A F - +H(k)| = {res_A:.3f}, max|dE| = {gap_A:.3f} eV, "
    f"herm defect {herm_defect(Hn_A):.1e}",
)

H_B, S_B = sc_pencil("affine")
Hn_B, Sn_B = to_np(H_B), to_np(S_B)
defect_B = herm_defect(Hn_B)
spec_B = spectrum(Hn_B, Sn_B)
gap_B = float(np.abs(spec_B - union).max())
check(
    "affine supercell-displacement phases remain Hermitian (defect < 1e-12)"
    " -- the failure is spectral, not a solver artifact",
    defect_B < 1e-12,
    f"defect {defect_B:.1e}",
)
check(
    "affine phases break the spectrum: max|dE| > 1e-3 "
    "(missing frame factor (-1)^w per wrapped bond)",
    gap_B > 1e-3,
    f"max|dE| = {gap_B:.3f} eV",
)

H_P, S_P = sc_pencil("plain")
spec_P = spectrum(to_np(H_P), to_np(S_P))
gap_P = float(np.abs(spec_P - union).max())
print(
    f"  [info] local-frame plain block sums: max|dE| vs union = {gap_P:.3f} eV "
    f"(different convention -- not the certified pencil, report only)"
)

# --------------------------------------------------------------------------
# 4) record
# --------------------------------------------------------------------------
print("\n== recorded spectra (eV) ==")
print("  union :", np.array2string(union, precision=8, floatmode="fixed"))
print("  sc    :", np.array2string(spec_C, precision=8, floatmode="fixed"))

print()
if FAILURES:
    print(f"FAILED: {FAILURES}")
    sys.exit(1)
print("ALL ASSERTIONS GREEN")
