# Spin-lattice coupling from non-orthogonal LCAO models

`TB2J.spin_lattice_lcao` computes the displacement derivative of isotropic exchange in a non-orthogonal basis. Both Hamiltonian and overlap derivatives enter the energy-dependent pencil vertex:

$$
\partial_{\tau_a}G(z)=G(z)\bigl(\partial_{\tau_a}H-z\partial_{\tau_a}S\bigr)G(z).
$$

The verified derivation is `docs/sympy/nonorthogonal_spin_lattice.py`. The exchange derivative includes **all four product-rule terms**: the two Green-function insertions and the two magnetic on-site vertex derivatives. For a localized perturbation, the second on-site derivative is evaluated at the translated bond endpoint, not at the first endpoint's cell.

## Localized and periodic response

`LocalizedDisplacementDerivative` stores two-legged real-space data keyed by `(Rq, Rk)`, with `Ru` identifying the displaced cell. Supported producers:

- `from_abacus_dh(parser)`: frozen-potential ABACUS `out_mat_dh/out_mat_ds` data.
- `from_supercell_runs(ref, plus, minus, orb_map, atom_orbs, delta)`: self-consistent supercell central differences with explicit orbital/cell ownership.
- `from_siesta_dhsdr(block, reference_fermi=...)`: R-resolved SIESTA FC data in the adopted home-cell convention `((0,0,0), R)`.

The driver accepts `response="localized"` (default) or `response="periodic"`. Periodic mode sums translations of the home-cell vertex; in reciprocal space it evaluates the diagonal-k response. **A periodic displaced-cell SCF finite difference must be compared with this image sum, not an unsummed localized result.** The adopted SIESTA home-cell representation is retained; a small periodic FC file alone does not independently identify the isolated-displacement response of infinite bulk.

`Rcut` selects output exchange pairs only. The localized contraction computes the additional internal Green-function lattice vectors required by the displacement support, even outside that radius. No fixed four-cell derivative cutoff is applied in this driver.

## SIESTA calculation

Use upstream SIESTA with netCDF4 support and:

```fdf
MD.TypeOfRun       FC
FC.Save.dHS        T
FC.Displacement    0.005 Ang
FC.First           1
FC.Last            1
```

Run a separate **undisplaced SCF** with the same pseudopotential, basis, electronic temperature, grid and k-mesh. Do not use the final displaced FC Hamiltonian as the reference. A one-atom primitive bcc cell only undergoes a rigid translation; use at least the two-atom conventional cell.

```python
from TB2J.interfaces.siesta_spin_lattice import gen_exchange_spin_lattice_siesta

sio = gen_exchange_spin_lattice_siesta(
    "reference/siesta.fdf", "fc/siesta.dHSdR.nc",
    atom=1, axis=1,                 # SIESTA indices are 1-based
    response="periodic",           # matches displaced-cell SCF differences
    magnetic_elements=["Fe"],
    kmesh=(9, 9, 9), nz=320,         # converge for the material; not universal defaults
    smearing=0.025852,              # k_B T in eV, approximately 300 K
    Rcut=3.0,
    output_path="spin_lattice_results",
)
```

The parser handles SXC cell/orbital indexing, independent dH/dS sparse patterns, netCDF dimension orientation, and duplicate offset rejection. Raw dH is **Ry/Bohr** and dS is **1/Bohr**; the adapter converts to eV/Å and 1/Å. The `Ry` dH attribute omits the displacement denominator; it does not imply that the stored derivative is eV/Bohr.

### Fixed reference energy gauge

SIESTA FC stores the derivative of the **raw** KS Hamiltonian, while sisl loads `H - E_F(file) S`. The SIESTA entrypoint reads the undisplaced Fermi energy and converts the vertex to `dH - E_F(reference) dS`. Do not pass the raw Fermi energy as the exchange driver's `efermi`: it is **zero** in this shifted gauge.

For independent displaced models:

```python
from TB2J.interfaces.siesta_spin_lattice import load_siesta_spin_lattice_model

reference, raw_ef = load_siesta_spin_lattice_model("reference/siesta.fdf")
plus, _ = load_siesta_spin_lattice_model("plus/siesta.fdf", reference_fermi=raw_ef)
minus, _ = load_siesta_spin_lattice_model("minus/siesta.fdf", reference_fermi=raw_ef)
```

The loader undoes the displacement-dependent part of sisl's Fermi shift. All three calculations then use the same zero energy and chemical potential. Differentiating a moving Fermi shift measures a different observable.

## Finite-difference comparison and convergence

The executable example validates an FC mode against **independent displaced SCFs** and fails if the requested tolerance is exceeded:

```bash
python examples/siesta_spin_lattice.py \
  --fdf reference/siesta.fdf --dhsdr fc/siesta.dHSdR.nc \
  --plus-fdf plus/siesta.fdf --minus-fdf minus/siesta.fdf --delta 0.005 \
  --atom 1 --axis 1 --elements Fe --kmesh 9 9 9 --nz 320 \
  --output comparison
```

`comparison/comparison.json` records each bond, analytic derivative, central difference, error, settings and acceptance thresholds. Defaults are 1% relative plus 0.1 meV/Å absolute error; near-zero derivatives use the absolute term. Displaced geometry, cell and pair ownership are checked. Run success is not a convergence certificate: also vary displacement, SCF and exchange k-meshes, real-space grid, and contour points until a plateau is observed.

Spin-lattice calculations retain the **full spectrum**. Applying `G(dH-z dS)G` to a truncated spectral Green function does not differentiate the full pencil inverse. Virtual high-energy states are particularly important in non-orthogonal bases. Small CFR orders can have severe high-energy-tail errors; do not assume 20–50 poles are adequate. The valence-window Legendre contour and finite-temperature CFR integrate different occupied-state protocols; compare like with like.

For collinear SIESTA models, the implementation stores exact scalar-spin eigensystems and reconstructs full matrices only when requested through Green accessors. Contour contractions stream all k-points in blocks, including the final partial block; this bounds temporary G/dG storage without discarding bands or sampling points. Independent FD sweeps release each exchange object after extracting its per-bond results, rather than retaining three eigensystems concurrently.

### Reproducible bccFe convergence workflow

`examples/siesta_spin_lattice_bccfe_convergence.py` creates a two-atom cubic Fe cell with PBE/DZP, tight SCF tolerances and the supplied Fe PSML. It runs reference, independent plus/minus SCFs and FC cases, then sweeps CFR order, exchange k-mesh, displacement, DFT k-mesh and real-space grid. Run in the environment containing TB2J, HamiltonIO, sisl and netCDF4:

```bash
python examples/siesta_spin_lattice_bccfe_convergence.py \
  --siesta /path/to/siesta --pseudo /path/to/Fe.1.psml \
  --output bccfe-convergence
```

The resulting `convergence_certificate.json` contains every bond, finite-difference error, requested tolerances and selected plateau settings. Individual measured rows are also saved under `measured_rows/`. A full certificate requires contour, exchange mesh, displacement and DFT/grid stages, followed by final-setting checks with larger CFR order, larger exchange mesh and a second displacement. These checks prevent common-mode integration errors from passing a matched FD comparison. Partial-stage runs are explicitly marked partial. `spin_lattice_results/` is written only after all required gates pass. Extend the requested lists when the workflow reports no plateau or lacks a larger final-setting probe; a failure is not evidence of convergence.

`--reuse-existing DIR` reuses physical SCF/FC cases only after checking geometry, basis, pseudopotential, grid, SCF tolerances, temperature and clean convergence. Existing unvalidated outputs are never overwritten or relabeled. The certified observable is the **periodic image-summed response** at fixed reference chemical potential and the selected pair radius; it is not a certificate of an isolated infinite-bulk displacement.

Cutoff rows record the **actual** `InitMesh` FFT dimensions and effective cutoff read from the SCF output. Different requested `MeshCutoff` values can produce the same grid; such a pair cannot certify grid convergence. Extend the cutoff list until distinct physical grids agree.

### Rough convergence for development

During development, use a declared coarse convergence target rather than claiming precision convergence. The convergence example accepts `--conv-atol 0.5 --conv-rtol 0.05` (meV/Angstrom and fractional relative error). Keep the independent-FD correctness gate at its default `--fd-atol 0.1 --fd-rtol 0.01`; a looser integration target is not permission to loosen the derivative check. Completed contour checks may retain the tighter contour tolerance.

Record the selected settings, all compared bonds, actual FFT grids and tolerances, and label the result **development-level convergence**. Keep failed precision attempts separate. Check against already completed finer references when available; do not rerun a large precision campaign merely to finish a development smoke test.

The tested two-atom bccFe development case uses PBE/DZP, the supplied Fe.1 PSML, a2.848 Ang cubic cell, DFT9^3, exchange17^3, requested400 Ry (actual36^3 FFT grid/441.598 Ry), 300 K, h0.005 Ang, CFR nz320 and Rcut3 Ang. All28 off-site bonds pass the independent-SCF FD gate; the maximum error is2.13e-5 meV/Ang. The kernel evaluates30 internal pairs, including two self terms that are not serialized as exchange bonds.

| Development check | Maximum change (meV/Ang) |
|---|---:|
| CFR320/640 at the selected physical case | 4.65e-7 |
| Exchange17/19 at DFT9/grid400 | 1.96791 |
| Exchange17/37 finer reference, same physical case | 1.30736 |
| DFT7/9 at exchange17/grid400 | 0.34917 |
| Actual FFT30^3/36^3 at DFT9/exchange17 | 0.10800 |
| h0.005/0.01, analytic derivative | 0.03502 |
| h0.005/0.01, independent FD derivative | 0.03091 |

These pass the declared development threshold0.5 meV/Ang +5% of the largest compared response; the contour already meets its tighter threshold. Cubic x/y and x/z checks have maximum errors1.16e-6 and7.71e-5 meV/Ang. This is **rough convergence for development**, not a claim of1% precision convergence. The retained research artifact is `.tmp/bccfe-development-validation/development_certificate.json`, with `validation_level="development"` and `precision_convergence_claimed=false`.

## Generic localized driver

```python
from TB2J.spin_lattice_lcao import gen_exchange_spin_lattice_lcao

sio = gen_exchange_spin_lattice_lcao(
    tbmodel=spinor_model, atoms=atoms, disp=disp, basis=basis,
    efermi=reference_fermi_in_model_gauge,
    magnetic_elements=["Fe"], response="localized",
    kmesh=(9, 9, 9), nz=320, Rcut=3.0,
    output_path="TB2J_results",
)
```

`Jonly=True` requires no derivative data. `SpinIO.exchange_Jdict` is in **eV** and `SpinIO.dJdx` in **eV/Å**. Multiply by 1000 to report meV or meV/Å. The response convention and derivative provenance are recorded in the output description.

## Regression evidence

- `test_spin_lattice_driver.py` compares the public localized driver's serialized derivative with independently perturbed supercell exchange, including the untranslated neighbour's own splitting and internal Green-function support beyond `Rcut`.
- `test_spin_lattice_periodic.py` compares the periodic response with independent perturbed pencils containing overlap response, splitting drift, and high-energy hybridization.
- `test_siesta_dhsdr_adapter.py` checks the fixed-gauge resolvent derivative and parser boundary behavior.
- HamiltonIO's `test_siesta_collinear_merge.py` verifies both spin spectra and resolvents through a real SIESTA file.

The earlier tutorial absolute-J discrepancy is resolved: raw pickle eV values were mistaken for meV, and the tutorial used `Fe_3d` rather than full-basis exchange. Its 18.8040 meV reference is reproduced with the same orbital selection and untruncated Green function. It is not a blocker for SIESTA spin-lattice validation.
