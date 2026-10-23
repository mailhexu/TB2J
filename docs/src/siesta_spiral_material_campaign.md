# Story 022: CrI3 SIESTA material-response campaign

This page records how to reproduce the Story-022 CrI3 campaign. It defines the producer calculations, independent response-convergence axes, derivative evidence, and the separation between SIESTA self-consistent calculations and the frozen-reference TB2J CLI. It records the method and available producer provenance; it does **not** assert convergence or an AC-005/AC-007/AC-009 pass.

The companion evidence directory is `memnotes/Projects/TB2J/evidence/2026-10-16-siesta-cri3-material-campaign/`. It contains `campaign_plan.json`, `run_campaign.py`, and, after the CLI certification phase, the four manifests and Story-006 partition report named below. The driver requires those certified inputs; it does not synthesize replacements.

## Material, producer, and baseline

The producer is SIESTA **5.4.2-23-g549d9de29**, source revision `549d9de2985008c2571350029b5d81cd94a3e0e1`. The recorded executable is `/home/hexu/projects/siesta_git/siesta_spinor/_build/build_fresh/Src/siesta`, SHA-256 `3b5f8078fe5113438f51f4c243dbe35ce80a1b200e323fdcabf144ac33adcf6d`. The producer source-file digests are:

| SIESTA source | SHA-256 |
| --- | --- |
| `Src/diag2kspiral.F90` | `03915969991a9088e2b1e78743bf269f7bf59e84beb802390bd59b20c626dabd` |
| `Src/io_hsx.F90` | `919c49ae7669813e5c543b128a40a0ee952a81013b34af460319a71dec4cf730` |
| `Src/final_H_f_stress.F` | `f1a9067e949581de816d3f44beef99e2e649f9189dd4a845fabbe704d2932545` |
| `Src/bands.F` | `d7946f47521c217b65c6ff860ef4c7153680e47155c46bfa17468c50d203113d` |

Both pseudopotentials are scalar-relativistic PSML 1.1 files generated on 2026-10-10 by **ONCVPSP-3.3.1+psml-3.3.1-75**:

| Element | PSML SHA-256 | Generator input identity |
| --- | --- | --- |
| Cr | `dfad34498fd6d0a8582118cc4e948fc240657754fa2a116436c8fe762e25b506` | ONCVPSP Cr, Z=24, 14 valence electrons, scalar-relativistic |
| I | `6d48b5c64c2b62d15dfe2bb5dd3be87fd9d87d3185dc48e640b2c52dbf06ef25` | ONCVPSP I, Z=53, 7 valence electrons, scalar-relativistic |

The 8-atom monolayer cell contains Cr2I6, uses lattice vectors `(6.867,0,0)`, `(-3.4335,5.946178895,0)`, `(0,0,20)` Angstrom, PBE, `ElectronicTemperature 300 K`, and 70 electrons. The anchor is Cr/I DZP with `PAO.EnergyShift 0.05 eV`, `MeshCutoff 200 Ry`, and a `4×4×1` Monkhorst-Pack grid. Its sampled gap is **1.10929829 eV**. The producer moments are about **+3.49 μB per Cr** and **−0.164 μB per I** along the reference spin directions. The listed SIESTA `DM.InitSpin` values use `(magnitude, theta, phi)`; `+3.0 0.0 0.0` seeds each Cr along +z.

All producer inputs use native noncollinear generalized Bloch theorem (GBT): `Spin non-collinear`, `Spin.Spiral.Scale ReciprocalLatticeVectors`, and `Spin.Spiral=(0.25,0,0)`, i.e. q=`(0.25,0,0)` in reciprocal-lattice units. The common SCF setup is `SCF.MustConverge T`, `MaxSCFIterations 300`, `SCF.H.Converge F`, `SCF.DM.Converge T`, `DM.Tolerance 1.d-5`, `DM.MixingWeight 0.02`, `DM.NumberPulay 8`, and `DM.UseSaveDM F`; SIESTA writes the Hamiltonian with `SaveHS T`.

The producer ladder inputs are **independent SIESTA SCFs** with their own basis/grid/k-point settings. They are not response-rung calculations of the TB2J CLI. The CLI instead evaluates a **frozen-reference MFT response** for each producer reference; it does not run SIESTA or self-consistently relax the reference.

## Independent response ladders

Run one axis at a time against the anchor, retaining the same q, operator/partition, ensemble declaration, and other declared settings. Use these exact manifests:

| Case | Manifest | Change from anchor |
| --- | --- | --- |
| Anchor | `anchor.json` | DZP, 200 Ry, `4×4×1`; derivative evidence source |
| Basis | `tzp_g200_k4.json` | Cr/I TZP; 200 Ry, `4×4×1` |
| Grid | `dzp_g300_k4.json` | DZP, 300 Ry, `4×4×1` |
| k mesh | `dzp_g200_k8.json` | DZP, 200 Ry, `8×8×1` |

The plan budgets are relative curvature change ≤5% for basis, ≤2% for grid, and ≤2% for k mesh. The anchor also declares contour quadrature-node rungs `[512,1024]`; compare their **absolute** curvature change against `1e-4 eV/rad²`. The response-node rung comparison is not a fourth SIESTA SCF and does not alter the producer k mesh. Relative changes use the campaign helper's denominator `max(1e-12, max(abs(refined curvature components)))`; absolute changes are maximum componentwise differences in `eV/rad²`.

These are independent axes. The basis, grid, and k-mesh comparisons change only their named producer setting relative to the anchor; do not combine or attribute mixed-axis changes to one factor. The report preserves each rung's complete curvature mapping and stores each ladder only under `convergence_ladders`. No result is pre-declared as passing.

## Reproduce the CLI campaign

From an installed, importable TB2J environment, after placing the certified inputs beside the driver in the evidence directory, choose a fresh output root and invoke:

```bash
python /path/to/evidence/2026-10-16-siesta-cri3-material-campaign/run_campaign.py \
  --output-root /path/to/new-cri3-campaign-run
```

`--output-root` is required. The directory may be newly created or already exist, but the four case directories and associated stdout/stderr files must not already exist. The driver refuses collisions rather than overwriting. For each case it executes only the documented command, from the manifest directory:

```bash
tb2j-siesta-spiral MANIFEST --output NEW_DIR
```

The CLI itself requires a new output directory and publishes a validated directory containing `TB2J.pickle`, `provenance.json`, and `run.json`. The driver additionally preserves each command's exact output streams as `<case>.stdout.txt` and `<case>.stderr.txt` under the output root and writes `campaign_report.json` only after all cases succeed and the provenance reports validate. CLI failures, missing outputs, invalid JSON, missing report fields, or absent source inputs fail closed; there is no fallback report or fabricated result.

The aggregate report retains each manifest, producer declaration, input digest, settings digest, code revision and content digest, and CLI `reproducibility_sha256`. It builds numerical ladders using `TB2J.interfaces.siesta_spiral_campaign`; raw mapping residual records have a separate `mapping_residuals` section. Anchor AC-005 and AC-009 verdicts are grouped under `derivative_agreement`, outside the convergence ladders. Run outputs are immutable evidence; reproduce into a different output root to compare the CLI timestamp-independent scientific reproducibility hashes.

## Derivative evidence versus numerical convergence

AC-005 is the response's finite-rotation cross-check against the analytic response. AC-009 is the declared central displacement/vertex-derivative validation. The campaign keeps both under the anchor's `derivative_agreement` section, with gate records retained as emitted by the CLI. Their agreement criteria and units are not basis, grid, k-mesh, or contour-convergence tolerances; one does not imply another. The displaced producer references move only atom 3 (I) in z by ±0.05 Angstrom (= ±0.094486306231288508 Bohr = ±0.0025 fractional z in the 20 Angstrom cell). The CLI records the manifest-declared `periodic-image-sum` gauge. This page makes no AC-005 or AC-009 pass claim.

The response core's producer and rotated H(k) pencil checks use the H-only absolute tolerance `1e-5 eV`. Overlap S(k) remains at an absolute `1e-8` dimensionless limit, and response vertices/matrices remain at absolute `1e-8 eV`. Separately, the independent fixture eigenvalue-oracle pencil gate (`PENCIL_HERMITICITY_REL_TOL`) retains the scale-aware bound `1e-8 * max(1, max(abs(matrix)))`; this relative oracle gate does not govern response-core checks. No matrix is symmetrized to satisfy a gate; a violation is refused. These checks remain independent of campaign convergence budgets and derivative agreement.

The contour/eigenbasis identity budget is the user-approved absolute `1e-6 eV`. The accepted raw-H asymmetry produces a small, node-independent representation mismatch between the eigenbasis and resolvent evaluations (measured `1.0e-7 eV/rad^2` on the anchor); it is recorded in every response as `contour_vs_band_max_abs` and refused only above `1e-6 eV`. The campaign manifests declare `margin_ev: 0.5` (below gap/2 in every case); the original 0.1 eV margin produced genuine flat-ellipse quadrature error (`1.86e-02 eV`) that is preserved as refusal diagnostics, not reinterpreted.

## Story-006 partition sensitivity; Story-007 limitation

`production_vertex_partition_sensitivity.json` is the separate Story-006 sensitivity source. It compares spatial partitions using a common transverse-x probe rotation at 0.35 rad, with identical unrotated blocks. Preserve its per-variant magnetic-operator changes and pairwise spreads as a sensitivity record, not as an axis of the response convergence analysis:

| Partition comparison | Variants' magnetic-operator changes | Maximum pairwise spread |
| --- | --- | ---: |
| `Cr1_per_atom_vs_Cr1_all_I_corotating` | `per_atom_sites`: 2.7283717482798315; `Cr1_plus_all_I_corotating`: 2.757389770317133 | 0.39898113819829817 |
| `I1_per_atom_vs_grouped_ligands` | `per_atom_ligands`: 0.08678934250520073; `grouped_ligands`: 0.23366871700093303 | 0.2169531731327686 |

These values document sensitivity to the declared spatial partition; they are neither response convergence errors nor evidence that the audit vertex is valid. The partition report states that its mapping residual is not computed there: AC-007 residuals belong to the CLI run and are retained separately as raw mapping diagnostics. The **Story-007 audit-vertex check was not invoked because Story 007 was user-skipped/unimplemented**. Do not infer vertex-audit availability, AC-007 success, material validation, or convergence from this report.

## Source records

Producer SIESTA version, executable/source hashes, anchor input and input digests are recorded in `producer.json` in the Story-022 producer run. Each certified CLI manifest records digest-pinned bundle, structure, FDF, PSML, and displacement inputs. The four exact manifest basenames and sibling Story-006 report are the campaign driver's complete evidence inputs; retain them unchanged when archiving a reproduction.
