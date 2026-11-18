# Split-SOC Gauge Conventions — Numeric Derivation Report

Story 001 of the split-SOC KS-band spec
(`Projects/TB2J/specs/split-soc-ks/stories/story-001-sympy-conventions.md`).

Script: `split_soc_gauge.py` (assertion-checked, numpy; run in `mydev`).
Status: **all assertions pass** (2026-09-27, rev. 2). All identities asserted on
random complex matrices at 1e-14 against O(1)-scaled toys (deviations printed).

## Pinned objects

- `C(θ,φ)` — GPAW `soc_eigenstates` spinor-basis matrix (`gpaw/spinorbit.py:380-383`,
  26.7.0). Asserted equal to the standard active rotation
  `expm(-iφs_z/2) expm(-iθs_y/2)` (dev < 1e-15; scipy `expm`, in-script).
- `U_TB2J(θ,φ)` — `TB2J/mathutils/rotate_spin.py::rotation_matrix` (verbatim formula).
- `O` — `O_wv = Tr[σ_w (C σ_v C†)]/2` (SO(3), `O e_z = n(θ,φ)`).

## Axis-map equivalence (asserted)

- `C σ_z C† = n(θ,φ)·σ⃗` (dev ≤ 1.6e-16 over z/x/y/generic legs).
- `U_TB2J† σ_z U_TB2J = n(θ,φ)·σ⃗` (dev 0.0).
- The two Wigner-D matrices are **distinct** (|U_TB2J − C| = 0.82 at (0.9, 1.3)):
  only the axis map `(θ,φ) ↦ n` is shared; never mix the two D-matrices.

## `add_soc` tensordot chain — standard form confirmed (asserted)

The verbatim GPAW chain (`spinorbit.py:74-75`)

```
H = tensordot(C, H, (0, 1)); H = tensordot(C.T.conj(), H, (1, 1))
```

computes the **standard `H ← C† (σ·L) C`** — asserted at every leg
(z/x/y/generic) on both the packed `σ·L` toy and general non-packed spin-leg
data (per-(i,j) plain matrix products as targets; dev < 1e-15). Negative
control: `C† H Cᵀ` differs (gap 1.72).

Revision note: an earlier draft of this script claimed the chain equals
`C† H Cᵀ`. That was an **einsum operand-label trap** — the comparison target
`einsum('as,stij,bt', C†, H, C.T)` is exactly `C† H C`, because the operand
`C.T` with labels `(b, t)` contributes `C[t, b]` (the operand transpose
cancels). The trap is documented in the script docstring so it is not
repeated. No adapter correction is needed; the research-slice assertion (c)
stands.

## Gauge theorem ψ/χ (asserted, dev 9.0e-16)

With

```
H_ψ = h0⊗1 + Δ⊗σ_z + Σ_v W_v ⊗ C†σ_vC      (GPAW ψ picture)
H_χ = h0⊗1 + Δ⊗n̂·σ⃗ + Σ_v W_v ⊗ σ_v          (SIESTA χ picture)
U   = 1 ⊗ C
```

- `H_χ = U H_ψ U†` (dev 9.0e-16, scale 4.8);
- spectra identical (dev 2.8e-15; min gap 0.62 in the toy);
- eigenvectors `X_χ = U X_ψ · diag(e^{iφ_m})` with
  `e^{iφ_m} = conj(⟨x_χ,m|U x_ψ,m⟩)` (dev 1.7e-15; non-degenerate spectrum required).

## Projections (asserted)

- `P^χ = C P^ψ` on the m_j index (`einsum('st,mti->msi', C, P_psi)`); per-band
  `|P|²` preserved (dev ≤ 3.6e-15).
- For n̂ = x̂, C mixes the two S_z components; **no** diagonal sign/permutation map
  reproduces it (min residual 2.0): it is a general Wigner-D on m_j.
- Caution: the embedded vertex `B† v B` is *not* covariant under the component map
  alone (which slots transform is convention-dependent); no such identity is asserted.

## Exchange-tensor gauge map (asserted, relative 3.6e-16 at scale 166)

With lattice Cartesian probes in both pictures and everything-else conjugated
(`V^χ = UV^ψU†`, `G^χ = UG^ψU†`), for general unstructured Green data:

```
T_χ = O T_ψ Oᵀ          (dev 5.9e-14 at scale 166 → relative 3.6e-16)
Oᵀ T_χ O = T_ψ          (inverse labeling, dev 1.5e-13)
```

`O ∈ SO(3)` (dev 1.1e-16, det +1), `O e_z = n̂` (dev 1.1e-16). The real-part
convention commutes with the map (dev 4.3e-14).

**Adjudication:** the research report and ADR-4 write the compact form
`T_χ = Oᵀ T_ψ O`. For general (unstructured) data the direction is
`T_χ = O T_ψ Oᵀ`; the report's form is the same law with the ψ/χ label roles
exchanged (leg-frame ↔ lattice). The pinned invariants are: `O ∈ SO(3)`,
`O e_z = n̂`, and two-sided conjugation of the tensor by the leg rotation.
**Downstream (FR-010/FR-011 GPAW adapter): converting the leg-frame kernel
output to lattice components must use `T_lattice = O_d T_leg O_dᵀ`.**
(SPEC-001: the AC/ADR-4/FR-011/note formula text is being amended centrally.)

## Symbolic layer (exact sympy, added rev. 3)

`check_symbolic_frame_law` — exact symbolic equalities with real symbols
θ, φ and symbolic H entries (no numeric substitution):

- `C` unitary; `C σ_z C† == n̂(θ,φ)·σ⃗`;
- `O == ½Tr[σ_w C σ_v C†]` satisfies `O Oᵀ == I` (trigsimp-fu), `det O == 1`,
  `O e_z == n̂`;
- `U_TB2J† σ_z U_TB2J == n̂·σ⃗`;
- the verbatim tensordot **index algebra** (replicated term-by-term) equals
  `C† H C` for symbolic H entries — the exact-algebra settlement of SPEC-002.

## Verification summary

- `C` == expm rotation: dev < 1e-15 (three legs); unitary/axis: dev ≤ 1.6e-16.
- TB2J axis map: dev 0.0; D-matrix distinctness asserted (0.82).
- `add_soc` chain == `C†(σ·L)C`: dev < 1e-15, all legs, packed + general data;
  `C†HCᵀ` negative control gap 1.72.
- Gauge theorem: 9.0e-16; spectra 2.8e-15; eigenvectors 1.7e-15.
- Wigner-D projections: ≤ 3.6e-15; sign-map exclusion 2.0.
- Tensor gauge map: relative 3.6e-16; inverse map and real-part form asserted.

## Consequences for implementation

1. GPAW adapter exports ψ-gauge data; tensors rotate per leg with `O_d` as above
   (direction adjudicated here — see ADR-4 note).
2. `spinat` per leg = leg axis `n̂_d` (io_merge transverse-plane requirement).
3. `rotation_matrix` (TB2J) and GPAW `C` must never be mixed as D-matrices;
   only the axis map is shared.
4. GPAW's `add_soc` rotation is the standard `C†(σ·L)C` — verbatim fidelity and
   the SIESTA-equivalence theorem agree; no adapter correction needed.
