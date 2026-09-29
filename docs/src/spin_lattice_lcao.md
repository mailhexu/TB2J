# Spin-lattice coupling from non-orthogonal LCAO models

`TB2J.spin_lattice_lcao` computes the spin-lattice (spin-phonon) parameter
$O_{IJa}=\partial J_{IJ}^{\mathrm{iso}}/\partial\tau_a$ for a **localized**
displacement $\tau_a$ of one atom, in a **non-orthogonal** LCAO basis where
the displacement changes both the Hamiltonian ($\partial_aH$) and the overlap
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
  `efermi` from the undisplaced reference.
- **Displacement derivative** (`LocalizedDisplacementDerivative`), two
  producers:
  - `from_abacus_dh(parser)`: ABACUS direct derivatives
    (`out_mat_dh` / `out_mat_ds` CSR files, frozen-potential provenance,
    multi-k required; atom-restricted output gives the localized data with
    no unfolding). Units Ry/Bohr are converted to eV/Å.
  - `from_supercell_runs(ref, plus, minus, orb_map, atom_orbs, delta)`:
    central differences of ± displaced **supercell** runs
    (self-consistent provenance), unfolded to primitive two-legged form.
    `orb_map` labels each supercell orbital with `(primitive_orbital,
    cell)`; use the symmetric cell range (e.g. −2,−1,0,1 for N=4) so
    boundary bonds fold to the shortest image. The supercell must exceed
    the hopping range so the displaced atom does not see its images.

## Usage

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
# sio.exchange_Jdict: J_iso(R); sio.dJdx[(R, i, j)]: O_{IJa} in eV/Å
```

`Jonly=True` computes exchange only. The derivative provenance
(frozen-potential vs self-consistent FD) is recorded in the output
description.

## Validation

- Unit: vertex assembly, (Rq,Rk) folding vs the reference contraction,
  producer outputs vs analytic chain derivatives.
- Integration (`tests/test_spin_lattice_fd.py`): the production $O$ equals
  the central difference of the production $J$ on a matched-discretization
  supercell oracle (64 k-points ⟺ 64-cell Γ-only supercell), with the
  $\partial S$ and splitting-drift channels active; no-`dS` and fixed-`z`
  negative controls disagree.
- The double-$k$-weight real-space folding is correct for the localized
  vertex (independent $k,k'$ sums); do not "fix" it to a single weight.
- Caveat: `TBGreen` truncates the eigen window at `emax = e_f + 5.1` eV;
  keep the model spectrum (and contour) inside the retained window.
