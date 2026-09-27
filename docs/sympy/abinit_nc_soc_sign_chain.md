# ABINIT NC SOC Sign Chain — i^l / amet(−i) / Conjugation — Derivation Report

Story 001 of the split-SOC KS-band spec. Pins the NC SOC operator convention of
`research-supporting/split-soc-abinit-nc-pypao.md` (§1.3, §2.1, §3) against the
ABINIT source, with a finite toy G-space contraction.

Script: `abinit_nc_soc_sign_chain.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-27). Random complex data, 1e-14.

## Source pins (ABINIT tree, branch `savetb2j`)

- `src/66_nonlocal/m_contract.F90::metric_so`:
  - Pauli Re/Im packing: `σ_x/2, σ_y/2, σ_z/2` (`pauli(Re/Im,s,s',n)`);
  - optional spinaxis `U(α,β)`: `S_n → U† S_n U` with
    `U = [[cb·e^{-iα/2}, −sb·e^{-iα/2}], [sb·e^{+iα/2}, cb·e^{+iα/2}]]`;
  - antisymmetric tensor `A^{(n)}_{iy1,iy2} = g(m1,iy1)g(m2,iy2) − g(m2,iy1)g(m1,iy2)`
    with `(n, m1, m2)` an even permutation of (0,1,2);
  - final Re/Im **swap** = multiplication of the complex matrix by **−i**.
- `src/66_nonlocal/m_nonlop_pl.F90` (SO pass): `amet(2)` applies to `i·gxa`
  (`temp = (−Im, +Re)`), `amet(1)` to `gxa` directly, summed over the source
  spin; `metcon_so` rank 1 is a plain 3×3 application of `amet`.
- Projector FT (pypao/abinao): `P = Σ_G i^l f(|k+G|) Y_lm(ĝ) e^{+i2π(k+G)·τ_a} c_G`,
  real tesseral harmonics per `pypao/spherical_harmonics.py` (scipy `lpmv`,
  Condon–Shortley phase, ABINIT ordering `l²+l+m`).

## metric_so internals (asserted, dev 0.0)

1. Pre-swap `amet0 = Σ_n (σ_n/2)⊗A^{(n)}`; the final Re/Im swap multiplies the
   complex matrix by **−i**: `amet = −i amet0` (dev 0.0).
2. With `gprimd = I`: `amet == L⃗⊗S⃗` (S = σ/2) in the real Cartesian-p basis,
   `L_n = −iε_{n·,·}` (dev 0.0). The metric index `iy` of the rank-1 real
   tensor is **Cartesian-ordered (x, y, z)**; the tesseral m-order is
   (m=+1,−1,0) = (x, y, z) — the permutation is part of the pinned chain.
3. `amet(α,β) = (U†⊗1) amet(0,0) (U⊗1)` (dev 5.6e-17): **spinaxis rotates the
   spin (Pauli) side only; L stays lattice-fixed** — the split-SOC frame
   separation, asserted from the code.

## Two-branch contraction (asserted, 4.4e-16)

Applying `amet(1)` to `g` and `amet(2)` to `i·g` (the m_nonlop_pl branch
structure, any `gprimd`) equals direct multiplication by the complex matrix
`−i·amet0` (dev 4.4e-16, scale 2.4).

## Finite toy G-space contraction — the pinned chain (dev 8.4e-17)

Single l = 1 channel, 6 random G, random k, weight `4π(2l+1)·eso/V`:

- ABINIT side:
  `W[(G'σ'),(Gσ)] = Σ_{iy1,iy2} t*[G',iy1]·amet[iy1,iy2,σ',σ]·t[G,iy2]`,
  `t[G,iy] = i^l f(q) Y^R_{l,m(iy)}(ĝ) e^{+i2π(k+G)τ}` (m(iy) = (+1,−1,0) for
  cart (x,y,z)).
- Python complex-Y form (pinned):
  `W = Σ_{m'm} Y_{m'}(ĝ')·[L·S]^c_{m'm}·Y*_m(ĝ)·(i^l)(−i)^l·f'f·e^{+i2π(G'−G)τ}`.

**The complex conjugation belongs to the KET (G-side) tensor.** The naive
placement — `Y*(ĝ')` on the bra (G') side, as written in research-note §2.1 —
produces the complex conjugate operator (asserted to differ; e.g. element
+0.609i vs −0.609i in an axis-aligned case). This is the CrI3
conjugation-convention incident class, now pinned with teeth.

Structure asserts: W Hermitian (dev 0.0), per-G spin trace of L·S zero (0.0),
per-site terms Hermitian and the two-site sum Hermitian ⇒ **W_SO additive over
all sites (ligands included)**.

## Negative controls (all asserted to break the match)

| control | deviation (scale 0.48) |
|---|---|
| conjugation on the bra (G') side instead of the ket (G) side | 0.35 |
| ket atomic-phase conjugation flipped (`e^{−iφ}` → `e^{+iφ}`) | 0.41 |
| ket `i^l` sign flipped (`i` → `−i`) | 0.41 |

## Consequences for implementation

1. The Python kernel (abinao/pypao, FR-031) must evaluate the complex-Y kernel
   with the conjugated tensor on the **G (ket)** side; the real-tensor path
   carries the conjugation on the G' tensor. The two agree (asserted here).
2. `amet` semantics: apply `amet(1)` to `g`, `amet(2)` to `i·g`; equivalently
   apply the complex matrix `−i·amet0`. The Re/Im swap in `metric_so` is exactly
   the `−i` factor.
3. The metric index of rank-1 real tensors is Cartesian-ordered; tesseral
   m-order is (y, z, x) — keep the permutation explicit in any kernel.
4. `spinaxis` rotates the spin side only (S → U†SU): the W_SO operator is
   lattice-fixed while the magnetic frame rotates — the split-SOC contract.
