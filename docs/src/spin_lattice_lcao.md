# Spin-lattice coupling from non-orthogonal LCAO models

`TB2J.spin_lattice_lcao` computes the spin-lattice parameter
$O_{IJa}=\partial J_{IJ}^{\mathrm{iso}}/\partial\tau_a$ for a **localized**
displacement $\tau_a$ of one atom, in a **non-orthogonal** LCAO basis where
the displacement changes both the Hamiltonian ($\partial_aH$) and overlap
($\partial_aS$). The Green-function derivative uses the energy-dependent
pencil vertex

$$
\partial_{\tau_a}G(z)=G(z)\,\bigl(\partial_{\tau_a}H-z\,\partial_{\tau_a}S\bigr)\,G(z),
$$

assertion-checked in `docs/sympy/nonorthogonal_spin_lattice.py`. The A-matrix
derivative includes both on-site magnetic-vertex terms $\partial_{\tau_a}P$
(the exchange splitting drifts with the displaced configuration).

## Inputs

- **Reference model**: a merged spinor (up/down) `LCAOHamiltonian`-style
  model with `HR`, `SR`, `Rlist` (SIESTA/ABACUS via HamiltonIO). Pin
  `efermi` from the undisplaced reference in the same energy convention as
  the model eigenvalues.
- **Localized displacement derivative** (`LocalizedDisplacementDerivative`):
  - `from_abacus_dh(parser)`: ABACUS direct derivatives
    (`out_mat_dh` / `out_mat_ds` CSR files, frozen-potential provenance,
    multi-k required; atom-restricted output gives localized data without
    unfolding). Units Ry/Bohr are converted to eV/Å.
  - `from_siesta_dhsdr(block)`: SIESTA `FC.Save.dHS T` output
    (`SystemLabel.dHSdR.nc`, upstream `dev`/`rel-5.4`), self-consistent
    central differences. Blocks are R-resolved over the auxiliary
    supercell; under the adopted convention (user decision 2026-09-30)
    they are treated as the localized derivative of the displaced home-cell
    atom: each R-block maps to the vertex key (Rq, Rk) = ((0,0,0), R).
  - `from_supercell_runs(ref, plus, minus, orb_map, atom_orbs, delta)`:
    central differences of ± displaced **supercell** runs
    (self-consistent provenance), unfolded to primitive two-legged form.
    `orb_map` labels each supercell orbital with `(primitive_orbital,
    cell)`; use a complete cell labeling and a supercell larger than the
    hopping range so displaced-image supports do not overlap.

## SIESTA dHSdR workflow

Run SIESTA with `FC.Save.dHS T` during a force-constant run and build the
localized derivative directly:

```fdf
MD.TypeOfRun       FC
FC.Save.dHS        T
FC.Displacement    0.02 Bohr
```

```python
from HamiltonIO.siesta.dhsdr import read_dHSdR
from TB2J.spin_lattice_lcao import LocalizedDisplacementDerivative

block = next(read_dHSdR("siesta.dHSdR.nc", atoms=[1],
                        displacements=[(1, 1)]))
disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(block)
```

SIESTA `ia` and `ixyz` indices are 1-based. `read_dHSdR` returns raw
derivative values in Ry/Bohr (`dH`) and 1/Bohr (`dS`); the adapter converts
them to eV/Å and 1/Å via `block_to_R_lists`. The reader handles the writer's
netCDF dimension order (`isc_off(n_s,xyz)`, `dH(spin,nnzs)` as seen from
Python), SXC column encoding (`cell=(list_col-1)//no_u`,
`orbital=(list_col-1)%no_u`), independent dH/dS sparse patterns, and
rejects duplicate supercell offsets.

A one-atom bcc primitive cell is uninformative: displacing its sole atom
translates the whole lattice, so dH/dR and dS/dR vanish by translational
invariance. Use at least the two-atom conventional bcc cell for bcc Fe.

Note: the raw FC blocks are produced by displacing the periodic image
sublattice in phase; the adopted convention (user decision 2026-09-30)
treats them as the localized home-cell derivative. Cross-validation against
explicit ± localized supercell runs is planned once the current SIESTA-path
absolute-J discrepancy (see below) is resolved.

## Localized-vertex usage

```python
from TB2J.spin_lattice_lcao import (
    LocalizedDisplacementDerivative, gen_exchange_spin_lattice_lcao)

disp = LocalizedDisplacementDerivative.from_abacus_dh(parser, direction=(1, 0, 0))
sio = gen_exchange_spin_lattice_lcao(
    tbmodel=spinor_model,          # merged up/dn LCAO model
    atoms=atoms,
    disp=disp,
    basis=basis,                   # one name per spinor basis function
    efermi=efermi,                 # pinned from the undisplaced reference
    magnetic_elements=["Mn"],
    kmesh=(5, 5, 5),
    output_path="TB2J_results",
)
# sio.exchange_Jdict: J_iso(R); sio.dJdx[(R, i, j)]: O_IJa in eV/Å
```

`Jonly=True` computes exchange only. Derivative provenance is recorded in
the output description.

## Validation

- Localized path: `tests/test_spin_lattice_fd.py` compares production $O$ to
  the central difference of production $J$ on a matched-discretization
  supercell oracle (64 k-points ↔ 64-cell Γ-only supercell); no-dS and
  fixed-z negative controls disagree.
- The SIESTA reader has synthetic netCDF fixture tests for sparse decoding,
  schema dimensions, filtering, and unit conversion. A real 2-atom bccFe FC
  file was parsed during development; dH/dR and dS/dR are nonzero and their
  maximum norms agree across cubic axes. This is a parser smoke only, not a
  spin-lattice coupling validation.
- The bccFe absolute-J smoke comparison is blocked by an unresolved
  discrepancy: the stored tutorial nearest-neighbor exchange is 18.8 meV,
  while current SIESTA/TB2J runs in this environment return about 0.01 meV.
  Do not treat an absolute bccFe spin-lattice result as validated until the
  J discrepancy is resolved; see the `nonorth-spin-lattice` story-005 log
  dated 2026-09-30.
- The double-k-weight real-space folding is correct for the localized vertex
  (independent k,k' sums); do not change it to a single weight.
- Caveat: `TBGreen` truncates the eigen window at `e_f + 5.1` eV; keep the
  model spectrum (and contour) inside the retained window.
