# Two-index unfold-epc consumer

`unfold-epc` is an optional standalone producer. Importing TB2J, its Green
functions, or `TB2J.interfaces.unfold_epc` does not import that dependency.
The consumer calls load the versioned file contract lazily. Missing optional
packages and unknown netCDF schema versions raise explicitly.

Use a HamiltonIO checkout/version with TB2J's existing SIESTA spin-lattice
model-adapter support. The reference helper reuses that adapter; it does not
require a separate primitive SCF or a primitive `dHSdR.nc` calculation.

```python
from TB2J.interfaces.unfold_epc import (
    load_unfold_epc, load_unfold_epc_reference,
    gen_exchange_spin_lattice_unfold_epc,
)

store = load_unfold_epc("epc.nc")
model = load_unfold_epc_reference("campaign/ref/siesta.fdf", "epc.nc")
Oq = gen_exchange_spin_lattice_unfold_epc(
    model, model.atoms, "epc.nc", q=[1/3, 1/3, 0],
    polarization=mode_cartesian_vector,
    basis=model.orbs, magnetic_elements=["Cr"],
    kmesh=(9, 9, 1), emin=-85, emax=0, nz=320, Rcut=5,
)
```

`Oq` maps `(primitive_cell_R, primitive_atom_i, primitive_atom_j)` to a
**complex** exchange derivative in eV/Ang. These are primitive atom labels,
not the magnetic-spin indices used by serialized supercell `SpinIO` results.
Multiply by 1000 for meV/Ang. All directed reported bonds are retained.

The input model must have matching primitive orbital order, a cell-periodic
Bloch gauge (`convention=2`), interleaved collinear spins, and the fixed
reference Hamiltonian `Hphysical-Ef_ref*S`. The helper reconstructs this
model by averaging equilibrium translated copies of the **same** shared
reference and verifies its netCDF hash against the store. It constructs
primitive basis labels from the reference orbitals. No endpoint filter is
applied to derivative blocks.

`polarization[natom,3]` is an explicit Cartesian pattern. The consumer does
not mass-weight or conjugate it implicitly. If `basis` is omitted, model
`orbs` supplies it. If `magnetic_elements` is omitted, all atomic species
are selected, matching the existing SIESTA wrapper; explicitly select Cr
for a Cr-only exchange acceptance campaign.

## Pencil, phases, and endpoint terms

The store contains raw physical `dH,dS`. Contour energy `z` is relative to
the fixed reference Fermi zero. The vertex is
`dH-(Ef_ref+z)*dS`, and the oriented response is
`G(k) V(k,q) G(k+q)`. The displacement is `exp(-2pi*i*q.L)`, with bra `k`
and ket `k+q`. The forward bond phase is `exp(-2pi*i*(k+q).R)`; the reverse
phase is `exp(+2pi*i*k.R)`. The two onsite splitting derivatives remain in
the product rule; the second endpoint carries `exp(-2pi*i*q.R)`.

Complex modes require a complex-linear contour reduction. The implementation
integrates q and the conjugate-polarization partner at -q, then returns
`(Cq-conj(Cminusq))/(2i)`. Taking `imag(Cq)` alone would mix the cosine and
sine responses incorrectly. All bands are retained and an endpoint-free
fixed semicircle contour is used.

[Assertion script](../sympy/q_phased_exchange_contraction.py) and
[derivation report](../sympy/q_phased_exchange_contraction.md) verify the
pencil, both bond orientations, both endpoint terms, complex-linear
reduction, and equilibrium `1/N` copy normalization.

## Discrete comparison with independent frozen SCFs

For an exact comparison at a finite supercell electronic mesh, pass explicit
primitive `kpts,kweights` obtained by unfolding every supercell point and
all its commensurate primitive images. The acceptance example does this;
it does not compare unrelated primitive and supercell quadratures.
`kmesh` still bounds the real-space bond enumeration. Electronic-mesh
convergence remains a separate question, not an automatic guarantee.

The committed unfold-epc example `examples/cri3_exchange_acceptance.py`
computes one preselected Cr-dominated optical K eigenmode, every directed Cr
nearest-neighbor component, and independent ± cosine/sine **SCF** J values.
It reports the elementwise limit `0.1 + 0.01*abs(FD)` meV/Ang. Its
`drop_overlap=True` control removes the whole physical overlap channel,
including the reference-Fermi correction, while preserving the equilibrium
nonorthogonal reference. It must fail that same gate to certify the control.

Finite steps approximate derivatives. A localized central-difference stencil
and an independently displaced mode need not superpose exactly at finite h.
Keep the declared tolerances fixed and converge the displacement/SCF
accuracy if the matrix gate fails. Do not suppress errors or reinterpret
finite-q acoustic strain as an acoustic null.
