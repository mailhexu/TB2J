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
  - `from_supercell_runs(ref, plus, minus, orb_map, atom_orbs, delta)`:
    central differences of ± displaced **supercell** runs
    (self-consistent provenance), unfolded to primitive two-legged form.
    `orb_map` labels each supercell orbital with `(primitive_orbital,
    cell)`; use a complete cell labeling and a supercell larger than the
    hopping range so displaced-image supports do not overlap.

## SIESTA force-constant derivatives: periodic q=0 response

Upstream SIESTA `dev` and `rel-5.4` provide `FC.Save.dHS T` during an
`MD.TypeOfRun FC` calculation. The output `SystemLabel.dHSdR.nc` contains
central differences of H and S for a periodic displacement of each unit-cell
atom and Cartesian axis (a q=0 sublattice displacement). `HamiltonIO` can
read these blocks:

```fdf
MD.TypeOfRun       FC
FC.Save.dHS        T
FC.Displacement    0.02 Bohr
```

```python
from HamiltonIO.siesta.dhsdr import read_dHSdR

blocks = list(read_dHSdR("siesta.dHSdR.nc", atoms=[1],
                         displacements=[(1, 1)]))
block = blocks[0]  # raw periodic q=0 dH/dR and dS/dR data
```

SIESTA `ia` and `ixyz` indices are 1-based. The reader handles the writer's
netCDF dimension order (`isc_off(n_s,xyz)`, `dH(spin,nnzs)` as seen from
Python), the SIESTA SXC column encoding (`cell=(list_col-1)//no_u`,
`orbital=(list_col-1)%no_u`), and independent dH/dS sparse patterns. It
converts Ry/Bohr to eV/Å and 1/Bohr to 1/Å.

**Do not pass this block to the localized `from_supercell_runs`/`ABACUS`
vertex path.** `FC.Save.dHS` is a periodic q=0 sublattice response: all
periodic images of the selected unit-cell atom move together. It is not, in
general, the derivative for displacing only the home-cell atom. Recovering a
localized two-legged vertex would require separating image contributions and
handling ambiguous same-sublattice terms. Story 005 is being revised to add a
dedicated q=0 spin-lattice contraction; that API is not implemented yet.
A one-atom bcc primitive cell is especially uninformative: moving its sole
atom translates the entire lattice, so dH/dR and dS/dR vanish by translational
invariance. Use at least a two-atom conventional bcc cell when inspecting
nonzero sublattice response.

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
