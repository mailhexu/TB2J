# Exact folding identity for the `q_half_per_spin_sector` rotating-frame pencil

Story 004, Axis A. Derived and machine-verified in `fold_identity_sympy.py`
(runs green in ~2 s with the mydev stack; every identity below is asserted).

## 1. Conventions (verbatim `SpiralImporter.assemble_pencil`, vector path)

Primitive cell: lattice vectors `cell`, orbital positions `pos[0..n-1]`.
Bond `(mu, nu, R)` carries real spin-diagonal blocks `h1, h2`, the mixing
block `h12 = h3 - i*h4`, and overlap `s`, with the partner bond
`(nu, mu, -R)` present. The displacement is

    xij(R) = R @ cell + pos[nu] - pos[mu]        (PRIMITIVE, unwrapped)

and the pencil at wave-vector k is

    d11 = exp(-i (k + q/2)·xij)   row sector up   : H11 = h1, H12 = h12
    d22 = exp(-i (k - q/2)·xij)   row sector down : H21 = conj(h12), H22 = h2
    S    spin-diagonal with the same d11 / d22.

Hermiticity of the raw pencil (the fixture oracle enforces `1e-8`) forces the
producer partner constraint

    h12(nu, mu, -R) = h12(mu, nu, R) * exp(-i q·xij(mu, nu, R)),

i.e. the mixing blocks carry the spiral texture. The symbolic model builds
this constraint in, and all pencils are Hermitian to machine precision.

## 2. Theorem (proved, element-wise exact)

Let `K = {k_j = j*b/n_s, j = 0..n_s-1}` be the folded mesh (here
`n_s = 4`, `q = b/4`, `K = {0, b/4, b/2, 3b/4}`). Assemble the supercell
Gamma pencil (k = 0) as

    H_sc[(m mu s), (n nu s')] = SUM over R with n = m + R (mod n_s) of
                                T^{ss'}(R) * exp(-i s (q·xij(R)) / 2)

with `s = +1` (up) / `-1` (down), `T^{↑↑} = h1`, `T^{↑↓} = h12`,
`T^{↓↑} = conj(h12)`, `T^{↓↓} = h2` (S analogous with `s(R)`), and — the
critical point — **wrapped bonds included, their phase still taken from the
PRIMITIVE unwrapped displacement `xij(R)`**, never from a supercell-affine
displacement. Then with the plain (untwisted) Fourier matrix

    F[(m mu s), (j mu s)] = exp(-i k_j·x_{m mu}) / sqrt(n_s),
    x_{m mu} = m @ cell + pos[mu],

    F^H H_sc F = ⊕_{k in K} H(k)   and   F^H S_sc F = ⊕_{k in K} S(k)

hold **element-wise, exactly**. F is unitary, so the pencils are congruent
and

    eig(H_sc, S_sc) = ∪_{k in K} eig(H(k), S(k))    (as multisets, exactly).

So the candidate identity in the task is **true as stated** (phases
`e^{-i q·xij/2}` on primitive displacements) — provided wrapped bonds are
included. The naive assembly that drops them is the only bug.

*Proof* (one line per ingredient, all verified exactly by sympy): insert
`n = m + R - w*n_s`; then `x_{n nu} = x_{m mu} + xij(R) - w*L` and
`exp(-i k_j w L) = 1` because `k_j L = 2 pi j` (only this is used — the
identity holds for **any** q, commensurate or not); the remaining factor
`exp(-i s(q/2)xij(R))` is bond-based, so the m-sum
`(1/n_s) Σ_m exp(i(k_j - k_j')x_{m mu}) = δ_jj'` collapses exactly.
Measured: 256/256 H entries and 256/256 S entries reduced to exact zero by
sympy simplification; spectral match 1.3e-15 eV.

## 3. The spinor frame factor that makes the identity close

The rotating frame attaches to every site `x` the sector-`s` spinor frame

    u_s(x) = exp(-i s (q·x) / 2)          (q·x/2 half the local pitch angle)

* per primitive translation `a`:  `u_s(a) = exp(∓ i π/4)`  (here `q·a = π/2`),
* per supercell translation `L = n_s a`:  `u_s(L) = (u_s(a))^{n_s} = exp(∓ i π) = -1`
  — the spin-1/2 double-cover sign (verified: `u² = -i`, `u⁴ = -1`).

Displacement phases inherit this frame:

    exp(-i s (q/2)·xij_affine) = (-1)^w · exp(-i s (q/2)·xij_primitive),

for a bond that wraps `w = ±1` supercell boundaries
(`xij_affine = xij_prim - w L`). Hence:

* **Primitive-branch phases (the theorem's assembly) are correct.**
* **Supercell-affine branches silently insert `(-1)^w` per wrapped bond.**
  Because the factor is the same for both sectors, the matrix stays
  Hermitian (defect 0.0e+00 in the model!) — the failure is purely
  spectral: max |dE| = **0.369 eV**. This is exactly the naive
  "phases on supercell displacements" formula of the task statement: it is
  wrong unless each wrapped bond is multiplied back by `(-1)^w`
  (equivalently: keep the primitive branch).
* **Dropping wrapped bonds** (open boundary, the current
  `supercell_gamma` in `gate_story004.py`) breaks the element-wise
  identity (max |ΔF^H H F| = 0.104) and the spectrum: max |dE| = **0.298 eV**.

For reference, plain un-phased local-frame block sums (a different
convention than the certified pencil) give 0.235 eV — not an option here.

## 4. Corrected assembly algorithm (drop-in fix)

```python
def supercell_gamma(bundle, n_s):
    """n_s-fold supercell Gamma pencil from primitive blocks (exact fold)."""
    # ... layout as before: site (m, mu, sector) ...
    for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
        sv = bundle.overlap[(mu, nu, R)]
        xij = np.asarray(R, float) @ cell + pos[nu] - pos[mu]   # PRIMITIVE
        qphs = np.exp(-1j * float(qvec @ xij) / 2)
        d11, d22 = qphs, np.conj(qphs)
        h12 = complex(h3, -h4)
        for n1 in range(n_s):
            n2 = (n1 + R[2]) % n_s        # <-- WRAP, never drop
            # add h1*d11, h12*d11, conj(h12)*d22, h2*d22 (and s*d11/s*d22)
            # at rows/cols of (n1, mu, up/down) x (n2, nu, up/down)
```

The fix to `gate_story004.py::supercell_gamma` is two lines: replace

```python
n2 = n1 + R[2]
if not 0 <= n2 < n_s:
    continue
```

with `n2 = (n1 + R[2]) % n_s`. Everything else (primitive `xij` phases,
element layout) is already correct.

## 5. Verification record (model: 2 atoms, 1 orbital, ±1 cell, q = b/4)

| check | result |
|---|---|
| exact element-wise `F^H H_sc F - ⊕H(k)` | 256/256 entries zero (sympy simplify), residual ≤ 2.6e-172 |
| exact element-wise `F^H S_sc F - ⊕S(k)` | 256/256 entries zero, residual ≤ 3.2e-173 |
| F unitary | max residual 2.8e-191 |
| Hermiticity of H_sc, S_sc and all S(k) | defect 0.0e+00; S positive definite (Cholesky) |
| `eig(H_sc, S_sc)` vs folded union | max |dE| = **1.33e-15 eV** |
| dropped wrapped bonds | elementwise Δ 0.104, max |dE| = **0.298 eV** (Hermitian) |
| supercell-affine phases | Hermitian, max |dE| = **0.369 eV** (missing `(-1)^w`) |
| frame factor per primitive cell / per supercell | `e^{∓iπ/4}` / `-1` (exact) |

## 6. Relation to the earlier `sisl tile` failure

The fe4 HSX is a SIESTA generalized-Bloch-theorem (GBT) output: the stored
matrix lives in the rotating frame whose primitive-cell periodicity is
*translation ⊗ spinor rotation* `u(a) = e^{-i q·a σz/2}`, antiperiodic
(`-1`) over the magnetic supercell. Plain `tile(4)` and plain sisl `Hk(k)`
ignore that sector twist, so both sides of the old comparison were wrong in
incompatible ways. In the bundle representation the twist lives entirely in
the `d11/d22` phases (and the partner constraint of §1); assembly with
primitive-branch phases and wrapped bonds included is its exact folding —
which is what Axis A now gates at solver precision.
