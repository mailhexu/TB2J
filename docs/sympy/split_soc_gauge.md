# Split-SOC Gauge Conventions — Sympy/Numeric Derivation Report

Story 001 of the split-SOC KS-band spec
(`Projects/TB2J/specs/split-soc-ks/stories/story-001-sympy-conventions.md`).

Script: `split_soc_gauge.py` (assertion-checked; run in `mydev`).
Status: **all assertions pass** (2026-09-27). All identities asserted on random
complex matrices at 1e-14 against O(1)-scaled toys (deviations printed).

## Pinned objects

- `C(θ,φ)` — GPAW `soc_eigenstates` spinor-basis matrix (`gpaw/spinorbit.py:380-383`,
  26.7.0). Verified equal to the standard active rotation `exp(-iφs_z/2)exp(-iθs_y/2)`
  (dev 0.0 with the verbatim formula).
- `U_TB2J(θ,φ)` — `TB2J/mathutils/rotate_spin.py::rotation_matrix` (verbatim formula).
- `O` — `O_wv = Tr[σ_w (C σ_v C†)]/2` (SO(3), `O e_z = n(θ,φ)`).

## Axis-map equivalence (asserted)

- `C σ_z C† = n(θ,φ)·σ⃗` (dev ≤ 1.6e-16 over z/x/y/generic legs).
- `U_TB2J† σ_z U_TB2J = n(θ,φ)·σ⃗` (dev 0.0).
- The two Wigner-D matrices are **distinct** (|U_TB2J − C| = 0.82 at (0.9, 1.3)):
  only the axis map `(θ,φ) ↦ n` is shared; never mix the two D-matrices.

## `add_soc` tensordot chain — corrected pin (asserted)

The verbatim GPAW chain (`spinorbit.py:83-85`)

```
H = tensordot(C, H, (0,1)); H = tensordot(C.T.conj(), H, (1,1))
```

computes **`H ← C† H Cᵀ`** (dev 5e-16) — *not* `C† H C†`:

| leg | C | verbatim chain equals |
|---|---|---|
| z | I | `C†(σ·L)C` (identical) |
| x (real C) | real | `C† H C†` — differs from `C†HC` by gap 2.9 |
| generic | complex | differs from `C†HC` by gap 1.7 |

Consequence: the gauge theorem below uses the analytic form `C†(σ·L)C`;
an adapter that must reproduce GPAW bit-for-bit has to use `C† H Cᵀ`
(flagged for the FR-010/FR-011 GPAW adapter story). The research-slice
assertion (c) ("chain equals C†(σ·L)C") holds only at C = I.

## Gauge theorem ψ/χ (asserted, dev 6.7e-16)

With

```
H_ψ = h0⊗1 + Δ⊗σ_z + Σ_v W_v ⊗ C†σ_vC      (GPAW ψ picture)
H_χ = h0⊗1 + Δ⊗n̂·σ⃗ + Σ_v W_v ⊗ σ_v          (SIESTA χ picture)
U   = 1 ⊗ C
```

- `H_χ = U H_ψ U†` (dev 6.7e-16, scale 5.1);
- spectra identical (dev 2.7e-15; min gap 0.94 in the toy);
- eigenvectors `X_χ = U X_ψ · diag(e^{iφ_m})` with
  `e^{iφ_m} = conj(⟨x_χ,m|U x_ψ,m⟩)` (dev 1.3e-15; non-degenerate spectrum required).

## Projections (asserted)

- `P^χ = C P^ψ` on the m_j index (`einsum('st,mti->msi', C, P_psi)`); per-band
  `|P|²` preserved (dev ≤ 1.8e-15).
- For n̂ = x̂, C mixes the two S_z components; **no** diagonal sign/permutation map
  reproduces it (min residual 1.76): it is a general Wigner-D on m_j.
- Caution: the embedded vertex `B† v B` is *not* covariant under the component map
  alone (which slots transform is convention-dependent); no such identity is asserted.

## Exchange-tensor gauge map (asserted, relative 5.6e-16 at scale 179)

With lattice Cartesian probes in both pictures and everything-else conjugated
(`V^χ = UV^ψU†`, `G^χ = UG^ψU†`), for general unstructured Green data:

```
T_χ = O T_ψ Oᵀ          (dev 1.0e-13 at scale 179 → relative 5.6e-16)
Oᵀ T_χ O = T_ψ          (inverse labeling, dev 1.2e-13)
```

`O ∈ SO(3)` (dev 1.1e-16, det +1), `O e_z = n̂` (dev 1.1e-16). The real-part
convention commutes with the map (dev 5.7e-14).

**Adjudication:** the research report and ADR-4 write the compact form
`T_χ = Oᵀ T_ψ O`. For general (unstructured) data the direction is
`T_χ = O T_ψ Oᵀ`; the report's form is the same law with the ψ/χ label roles
exchanged (leg-frame ↔ lattice). The pinned invariants are: `O ∈ SO(3)`,
`O e_z = n̂`, and two-sided conjugation of the tensor by the leg rotation.
**Downstream (FR-010/FR-011 GPAW adapter): converting the leg-frame kernel
output to lattice components must use `T_lattice = O_d T_leg O_dᵀ`.**

## Verification summary

- C unitarity/axis identity: 4 legs, dev ≤ 1.6e-16.
- TB2J axis map: dev 0.0; D-matrix distinctness asserted (0.82).
- `add_soc` chain: dev ≤ 5e-16 (C†HCᵀ form); x/z-leg special forms asserted.
- Gauge theorem: 6.7e-16; spectra 2.7e-15; eigenvectors 1.3e-15.
- Wigner-D projections: ≤ 1.8e-15; sign-map exclusion 1.76.
- Tensor gauge map: relative 5.6e-16; inverse map and real-part form asserted.

## Consequences for implementation

1. GPAW adapter exports ψ-gauge data; tensors rotate per leg with `O_d` as above
   (direction adjudicated here — see ADR-4 note).
2. `spinat` per leg = leg axis `n̂_d` (io_merge transverse-plane requirement).
3. `rotation_matrix` (TB2J) and GPAW `C` must never be mixed as D-matrices;
   only the axis map is shared.
4. GPAW-verbatim SOC rotation is `C†HCᵀ`; SIESTA-equivalence proofs must use the
   analytic `C†(σ·L)C` and will show a residual against raw GPAW internals for
   non-real C legs.
