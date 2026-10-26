# QE native Hubbard action and source-frame contract

## Scope and evidence status

This report pairs with `qe_hubbard_native_contract.py`. It pins exact algebraic
conventions for the QE native on-site Hubbard channel; it is **not** a native
QE action, occupation, or rotation proof and does not certify any pseudopotential
family. The source observations below are from QE 8.0dev revision
`301742e16546b927cfe35fb2fcf4d9338f3e294a`.

The SymPy script uses exact integers, Gaussian integers, rationals, and
symbolic trigonometric functions. It contains representative two-orbital and
three-component Hilbert-space checks, not an arbitrary-dimension proof. On
2026-10-26 its ordinary mydev execution completed all six named checks.
No QE run or native numerical check is claimed.

## Pinned native source facts

| Contract | QE source map and observed behavior |
|---|---|
| Actual projector columns | `PW/src/orthoUwfc.f90:88-106` creates the atomic columns (`atomic_wfc_nc_updown` for noncollinear mode), applies `s_psi`, optionally calls `ortho_swfc` on all `natomwfc` columns, then selects the U columns into `wfcU`. `ortho_swfc:261-269,303-319,334-368,430-459` defines the full overlap `O_ij=<phi_i|S|phi_j>`, constructs `O^(-1/2)`, and, for the S-applied path, transforms the `S phi` columns. In column-matrix notation the action columns are `W_U=S Phi` or, when orthogonalized, `W_U=S Phi O^(-1/2)`. This is a global atomic overlap before correlated-column selection, not a site-only inverse. `PW/src/ldaU.f90:694-745` selects main and background columns; `noncolin` doubles the main-column count, while the background slices have separate offsets. |
| Native spinor action and layout | `PW/src/vhpsi_U.f90:367-372` forms `proj=W_U^dagger psi`. At `391-400`, `vns(m1+ldim*(is1-1),m2+ldim*(is2-1))` is assigned directly from `v%ns_nc(m1,m2,npol*(is1-1)+is2)`. With `npol=2`, this is spin-major/orbital-inner `uu,ud,du,dd`, and the orbital count is not itself spin-expanded. `403-412` applies `vns*proj` and then `wfcU*ctemp`; neither GEMM adds a transpose or conjugation to the stored potential. Thus the implemented separable action is `W_U V_U W_U^dagger psi`. |
| Occupation conjugation and component storage | `PW/src/new_ns.f90:430-448,470-486` stores complex noncollinear occupations and accumulates `conjg(proj(m1,s1))*proj(m2,s2)`. At `595-609`, the combined component index is `npol*(is1-1)+is2`; `614-639` imposes `ns(m1,m2,is)=conjg(ns(m2,m1,partner_spin_index))`. The source layout is therefore not inferred from an assumed matrix transpose. |
| Native noncollinear potential | `PW/src/v_hubbard.f90:224-227` declares complex `ns` and `v_hub`. `242-251` swaps the spin-flip partner indices 2 and 3. For the scalar Dudarev branch with `Hubbard_alpha=J0=beta=0`, `253-289` adds `U/2` on the spin/orbital diagonal and subtracts `U*ns(m2,m1,partner_spin_component)`. In full spin-major/orbital-inner matrix notation this is `V_U=(U/2) I-U N^T`, where the transpose swaps both spin and orbital index pairs. The script uses asymmetric complex off-diagonal data and checks that the native block assignment, transpose convention, block order, and extra-conjugation negative controls differ exactly. |
| Collinear main and background action | `PW/src/v_hubbard.f90:145-162` gives the collinear main Dudarev coefficient (`effU=U` when `J0=0`, and the same `ns(m2,m1)` orientation); `PW/src/v_hubbard.f90:344-363` gives the background coefficient from `Hubbard_U2` and `Hubbard_alpha_back`. In `PW/src/vhpsi_U.f90:81-116,127-167,213-245,256-294`, main, first background, and optional `backall` second-background blocks act through `offsetU`, `offsetU_back`, and `offsetU_back1` as distinct column slices. For a collinear occupation with no spin-flip blocks, the main up/down direct sum equals the full spinor main matrix. The script represents the acted background subblocks separately; it does not assert a spinor background value. |
| Spinor background boundary | The audited spinor action `PW/src/vhpsi_U.f90:324-432` constructs and applies main-shell `v%ns_nc` blocks through `offsetU`; it has no `is_hubbard_back` action branch. The paired assertions do not model that missing branch as a zero potential. Architecture ADR-001 explicitly treats noncollinear/SOC background U as a native limitation to reject. |
| Frozen source and declared frame | The approved Architecture ADR-003 (`specs/architecture.md:147-173`) requires fixed charge, spatial Hubbard projectors, S/orthogonalization, scalar U, and lattice-fixed SOC while rotating only the magnetic source in its native spin-column convention, with private occupation copies and native rebuild/action checks. ADR-004 (`:175-198`) defines the rotation reference independently of source selection: use the original unextended XC source frame, or a separately validated mapped U direction when XC is zero; do not infer it from a cancelled XC+U total. These are approved future contracts, not claims that QE already implements the exporter operation. |

The source formulas above are observations, not runtime support certification. In
particular, source comments and the executable SymPy representation do not prove
that an exporter can reconstruct the full native columns or reproduce action
on arbitrary plane-wave states.

## What the executable assertions pin

The script's `MathDoc: tb2j/qe-hubbard-native-contract` comment tags are the
stable links from algebra to source-contract sections.

1. **S-applied and orthogonalized `wfcU` action.** A positive Hermitian toy
   overlap `S`, complex `Phi`, and exact Hermitian square root of
   `O=Phi^dagger S Phi` give both `W_U=S Phi` and
   `W_U=S Phi O^(-1/2)`. The script checks `W_U V_U W_U^dagger psi`, including
   negative controls for omitting `S`, omitting `O^(-1/2)`, transposing instead
   of adjointing a complex column matrix, and conjugating the coefficient
   matrix again.
2. **Spinor potential packing and conjugation.** An exact complex occupation
   is assembled in `uu,ud,du,dd` order from `conjg(proj_i)*proj_j`. The
   potential blocks are formed with QE's spin-flip partner and `ns(m2,m1)`
   rule, then checked against `U/2 I-U N^T` and the direct `vhpsi_U` action.
   The deliberately asymmetric entries distinguish the transposed
   occupation, a wrongly reordered block sequence, and an additional
   potential conjugation.
3. **Collinear embeddings.** Exact symmetric real up/down occupations give
   the same main-shell coefficient as the block-diagonal restriction of the
   full spinor main matrix. Separate nonzero background and optional second
   background slices are kept disjoint from that main embedding; they are not
   promoted to spinor support.
4. **Frozen magnetic source and tangent.** The complex Hermitian spinor
   potential is decomposed into a spin-scalar orbital matrix (including the
   fixed scalar-charge part) and three magnetic orbital matrices. Only the
   latter rotate. The check uses the exact tilted reference
   `n=(3/5,0,4/5)`, transverse `t=(0,1,0)`, and axis `n x t`; it checks the
   SU(2) source-column rotation and derivative. An action-level derivative
   keeps the same S-applied/orthogonalized columns, metric, and a fixed
   Hermitian SOC-like term.
5. **Noncovariance negative control after a fixed map.** An exact fixed
   spin-orbital controlled-phase map does not intertwine a source-column spin
   rotation with an output-spin rotation. The resulting nonzero defect is a
   counterexample to inferring a native source tangent from a commutator of
   the fully mapped operator. The map is an algebraic control, not QE's
   relativistic projector or a QE covariance measurement.
6. **Independent reference frame.** A declared XC direction remains defined
   when an equal-and-opposite U source cancels it in the total; for zero XC,
   the separate U-only case uses an explicit mapped direction. No direction
   is computed by normalizing the combined source.

## Assumptions, units, and limitations

- The represented functional is scalar effective-U Dudarev: `U` is real and
  orbital-independent, and `alpha`, `J0`, and `beta` are zero in these test
  fixtures. The native routines contain additional parameter branches; those
  branches are outside this contract's admitted scope. Background fixtures
  use scalar `U2` and zero `alpha_back`.
- QE's Hubbard potential/energy coefficients are in Ry (the native energy
  printouts identify Ry at `v_hubbard.f90:297-299` and `:376-379`). The exact
  tests encode no units; `S`/occupations are treated in the normalized
  wavefunction convention, and the rotation angle is in radians.
- The matrices are exact representatives only. The action check does not
  claim arbitrary-dimensional closure, compact site support, agreement with a
  finite beta span, or a physical-family result. In particular, the native
  orthogonalizer works on all atom-wavefunction columns before selection;
  downstream code must retain that complete construction.
- The frozen SOC-like matrix is a fixed Hermitian control used solely to
  assert zero source derivative. It is not an extracted QE SOC operator.
  Likewise, the symbolic rotation and source tangent are algebraic contracts,
  not independently measured native occupation rebuilds or finite-angle
  action parity.
- The new project MathDoc is a hash-pinned protocol adapter, not a second
  equation authority. `main()` returns the named SymPy expressions.
  Ordinary script execution prints six named PASS lines only after all
  assertions complete.

## Downstream gates not satisfied by this report

Before an exporter can be admitted, independent QE checks still need to
reconstruct the actual per-k `wfcU` and its full all-atom S/orthogonalization
map; compare native action on independent plane-wave trial states, including
out-of-beta-span directions; verify occupation contractions and all active
shell offsets; rebuild rotated potentials from private occupation copies;
measure finite rotations and derivatives with step convergence; and prove
complete QE state restoration. The spinor reference frame, shell map, and
background refusal must be carried explicitly. Source inspection and this
symbolic contract do not clear any of those gates.

Story 001 records the downstream budgets: relative action residual `<=1e-10`,
analytically-zero absolute residual `<=1e-12 Ry`, and derivative residual
`<=1e-6` with step convergence. None was evaluated here, and no existing
NC1.1 gate is relaxed.

## Execution command and status

From the authorized source writer root:

```sh
/home/hexu/projects/myenvs/mydev/bin/python docs/sympy/qe_hubbard_native_contract.py
```

**Status:** ordinary mydev execution on 2026-10-26 passed all six named
symbolic checks. The hash-pinned adapter also passed its update and
read-only protocol checks. No native QE action, occupation, rotation,
or physical-family certification was performed or inferred.
