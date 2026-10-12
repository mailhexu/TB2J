# Periodic (q=0) displacement response vs the localized ((0,0,0), R) vertex

Authoritative derivation: `periodic_image_sum_identity.py` (run with the
mydev python; exits nonzero on any failed check). Scope: mathematical
contract only — no production code changed.

## Contract

1. **Localized convention (preserved, user decision 2026-09-30).**
   SIESTA `dHSdR.nc` / supercell-FD displacement data are stored as
   two-legged vertex keys `(Rq, Rk) = ((0,0,0), R)` with `Ru = (0,0,0)`:
   the adopted representation anchors the supplied R-resolved blocks at
   home-cell rows. S2 verifies their Fourier assembly, not that a small
   periodic FC file uniquely identifies an isolated bulk displacement.

2. **Localized single-copy contraction (what `get_dGR` computes with
   `Rq = {(0,0,0)}`).**
   `dG_loc(Rj; z) = sum_Rk G(Ru) V(Ru, Rk; z) G(Rj - Ru - Rk)`, with
   `V(z) = dH - z_abs dS`, `z_abs = efermi + z`. This differentiates
   one copy of the supplied operator; physical locality is an input
   provenance question. It is **not** the periodic response, which
   additionally contains all translated-anchor terms (S5, N2).

3. **Periodic (q=0) response = image sum of translated local vertices.**
   With `G(k) = sum_R e^{+i k R} G(R)` (inverse of
   `G(R) = (1/N) sum_k e^{-i k R} Gk`):
   `dG_per(Rj) = (1/N) sum_k e^{-i k Rj} Gk V(k) Gk
              = sum_T sum_Rk G(-T) V(Rk) G(Rj + T - Rk)`.
   The required contraction for comparison with a periodic displaced-SCF
   finite difference at q=0 is therefore the **anchor-summed** form: the
   local vertex contracted at every translated anchor `T`, then combined
   in the four-term exchange contraction.

4. **Exchange contraction and dP terms.**
   `A = Tr[P_i G_ij P_j G_ji]`; its displacement derivative obeys the
   four-term rule
   `dA = Tr[dP_i G_ij P_j G_ji + P_i dG_ij P_j G_ji + P_i G_ij dP_j G_ji
            + P_i G_ij P_j dG_ji]`.
   `dP` is the on-site vertex entry at `z = 0` (the `-z dS` term drops
   there); `dP_j` must be evaluated at the **translated** on-site entry
   of atom j (cell of j), not reused from i.

5. **Fermi gauge (decisive for SIESTA).**
   `dHSdR.nc` values are the **raw** derivative (Ry/Bohr); the sisl-read
   reference model is stored Fermi-shifted (`H_m = H_raw - Ef0 S`). With
   efermi pinned to the reference value, the model-gauge vertex is
   `V_m(z_rel) = dH_raw - (Ef0 + z_rel) dS`; passing `efermi = 0` with a
   shifted model therefore requires converting
   `dH <- dH_raw - Ef0_ref dS` first (S6). Omitting this term is a
   first-order error of magnitude `|Ef0| |dS|`.

6. **efermi pinned.** Both sides must use the reference-configuration
   Fermi level. A displaced-SCF FD whose own `Ef(tau)` drifts measures an
   extra uniform `-dEf S` vertex (a rigid z shift); either pin the FD or
   the comparison carries that systematic error.

## What was certified (checks in the script)

| Check | Statement | Result |
|---|---|---|
| S1 | `dG/dtau|_0 = G V G` (2-cell symbolic) | exact |
| S2 | `dH(k)/dtau = sum_Rk dH((0,0,0),Rk) e^{ikRk}` | exact |
| S3 | root-of-unity image filter `N delta_{m0}` | exact |
| S4 | dP = on-site vertex at z=0; four-term rule | exact |
| S5 | `dG_per(Rj) = sum_T sum_Rk G(-T) V(Rk) G(Rj+T-Rk)` (N=3) | exact |
| S6 | shifted-gauge resolvent + vertex rule | exact |
| N0 | periodic vertex dG == FD of G on displaced chain | ~1e-6 rel |
| N1 | four-term dA == FD pair energy, both spins | ~5e-6 rel |
| N2 | single-anchor (local truncated) != periodic image sum | certified |
| N3-5 | no-dS / wrong fixed z / no-dP each change dJ | certified |

## Local vs periodic: physical limitation of the truncated local path

The localized single-anchor contraction is the exact derivative for a
*single* displaced atom only when the Green functions entering it are the
exact periodic ones AND the displaced atom's perturbation is truly
single-copy. In a periodic displaced-SCF run at q=0 every image moves,
so the exact comparison object is the anchor-summed contraction (S5).
Truncating the anchor sum (local response with finite image range) is a
controlled approximation whose error decays with the product of Green
functions at far image distances; it is NOT an identity, and the
periodic driver must therefore perform the explicit `Rk`-keyed image sum
(as now implemented in `ExchangePert2Sl` / the `response="periodic"`
path).

## Edge cases

- **Gauge**: raw dH + shifted model without `-Ef0 dS` conversion breaks
  the identity at `O(|Ef0||dS|)` (the ±239 meV/A vs ±4.5 symptom).
- **efermi drift**: unpinned FD Fermi level = extra `-dEf S` vertex.
- **Fixed z**: assembling V at a wrong/constant z fails (N4); V must be
  assembled at each contour energy `z_abs = efermi + z`.
- **Mesh commensurability**: supercell folded models are exact only on
  k-meshes with `nk` dividing the supercell multiplicity.
- **dPj placement**: `dP_j` must be read at atom j's own (translated)
  on-site entry; sharing `P_i`/`dP_i` for both ends is wrong (fixed in
  the local-oracle tests).
- **Contour convergence**: with the full-spectrum Green the valence-contour
  protocol (or very deep contours) is required; truncated-nz runs
  converge non-monotonically (separate investigation, not certified here).
