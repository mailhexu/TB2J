# bcc Fe three-reference QE KB spinor input

These PBE inputs use primitive bcc Fe (`celldm(1)=5.42` bohr), a fully relativistic 16-electron Fe ultrasoft UPF, 80/800 Ry cutoffs, independent x/y/z SCF reference densities on an 8³ mesh, and full-BZ 12³ NSCF bands (`nbnd=48`). The x/y/z legs have the same geometry and band window. They are **not** three postprocessed rotations of one electronic state. QE's magnetization direction is selected by `angle1/angle2`; its lattice-fixed spin–orbit operator must not be rotated with the magnetic vertex.

1. Use the instrumented `pw.x` from the QE `TB2J` branch. Download the official [`Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF`](https://pseudopotentials.quantum-espresso.org/upf_files/Fe.rel-pbe-spn-rrkjus_psl.1.0.0.UPF) into `pseudo/`; its SHA-256 is `5687f06ea9041819a7b2c8403e2f557ca795d94808cee4f97e1018ed4a8e3233`. Create `tmp/` and run from this directory.
2. For each `axis` in `x y z`, run `pw.x -in fe_${axis}_scf.in` **without** `TB2J_DUMP`; require SCF convergence. Then run `TB2J_PROJECTORS=kb TB2J_DUMP=fe_${axis}.bin pw.x -in fe_${axis}_nscf.in` using its matching density. Do not enable pool parallelism or use gamma-only wavefunctions. The NSCF decks request the complete BZ via `nosym` and `noinv`.
3. The writer's NC1.1 native ``add_vuspsi`` action, nine source responses,
   rotated PAW magnetic sources and finite-difference gates must pass before
   reading. For the tested PBE US full-BZ family, the physically distinct
   nearest-neighbor shell passes the repeated-row consistency gate:

   ```bash
   qe2J.py --input_x fe_x.bin --input_y fe_y.bin --input_z fe_z.bin \
     --elements Fe --Rcut 2.6 --output_path TB2J_results_fe_spinor
   ```

   This is a **nearest-neighbor-only** rank-nine tensor, not a full
   six-angstrom tensor. The same US family at ``--Rcut 6`` fails its
   far-shell repeated-row gate (``2.586256e-5 eV`` versus ``5e-8 eV``);
   leave the gate intact. Onsite self-pairs are excluded before merging
   because they are not exchange interactions.

A single dump is only a transverse-leg diagnostic. Do **not** pass
``--overlap_mode`` for dual KB coefficients. The magnetic nonlocal
response excludes fixed ionic/SOC terms; ``deeq_nc_total`` is
diagnostic only. The three-leg merge rejects missing directions and
inconsistent physical exchange rows. For the scalar-relativistic
z/collinear anchor, use the matching pseudopotential, band window
and k mesh; compare its transverse leg rather than treating it as
a full tensor.

A fully relativistic PAW analogue replaces the UPF with
`Fe.rel-pbe-spn-kjpaw_psl.1.0.0.UPF` (SHA-256
`b1ac78c1c8500118b7c393a0de775f2ae040e55b9d021ba1cfb5a6796a989c1b`)
in **all six** decks and changes `conv_thr` to `1.0d-12`
in **each of the three SCF** inputs before producing the matching
NSCF densities. The original `1.0d-8` PAW SCF threshold leaves
an unacceptable first-shell repeated-row spread
`4.117504e-6 eV`; the tightened x/y/z family passes at
`2.086799e-8 eV` and gives `Jiso=14.7775026 meV`
for the same `--Rcut 2.6` command. Its `--Rcut 6`
far shell still fails (`1.558e-5 eV` at the first
inconsistent pair). US/PAW are separate projector bases;
never merge their reference dumps.

## Scalar norm-conserving (NC) beta control

The four `nc_fe_*.in` decks use the **same** 16-electron SG15
`Fe_ONCV_PBE-1.0.upf` (SHA-256
`724ca3ae588400ca6936057e0dcf9f5966ea40d09a781a21c5732798c03dfb17`),
80/320 Ry cutoffs, a primitive bcc cell and the complete 12³
Brillouin zone. Place that UPF in `pseudo/` and create `tmp_nc/`
and `tmp_nc_col/` before running. The collinear NC v1.2 control
uses 24 bands **per spin**; its spinor NC1.1 z leg uses **48
total spinor bands**, not the 24-band spinor smoke input. A
24-versus-48-total comparison is not a matched band window.

```bash
pw.x -in nc_fe_col_scf.in > nc_fe_col_scf.out
TB2J_PROJECTORS=kb TB2J_DUMP=nc_fe_col.bin pw.x -in nc_fe_col_nscf.in > nc_fe_col_nscf.out
pw.x -in nc_fe_z_scf.in > nc_fe_z_scf.out
TB2J_PROJECTORS=kb TB2J_DUMP=nc_fe_z.bin pw.x -in nc_fe_z_nscf.in > nc_fe_z_nscf.out
qe2J.py --input nc_fe_col.bin --elements Fe --Rcut 2.6 --nz 30 --smearing 0.05 --output_path nc_fe_col_exchange
```

For the spinor z **transverse leg** use the public reader and raw-leg
API; a single reference cannot create a rank-nine tensor:

```python
from TB2J.interfaces.qe_projector import read_qe_spinor_dump
from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange

data = read_qe_spinor_dump("nc_fe_z.bin", legacy_no_u=True)
leg = compute_spinor_projector_exchange(
    data, Rpts=[[1, 0, 0], [-1, 0, 0]], nz=30, smearing_eV=0.05,
)
print("R=(1,0,0) transverse Jxx (meV):",
      1000 * leg[((1, 0, 0), 0, 0)]["J_leg"][0, 0])
```

For a legacy NC1.1 file, `legacy_no_u=True` is an explicit user attestation
that the QE calculation did not use +U; the file cannot prove that absence.
New NC1.2 files with a validated absent-U extension need no assertion.

With the pinned UPF, 12³ mesh, 30 poles and 0.05 eV contour,
the matched collinear first-neighbor coupling is 14.7932825 meV;
the NC1.1 z transverse leg is 14.7910404 meV at
`R=(1,0,0)` and its symmetry partners. The maximum
absolute difference over six first-neighbor vectors is
0.0023220 meV, below the 0.01 meV G3 gate.

NC NSCF uses `diagonalization='cg'`: on this UPF
(`number_of_wfc=0`), random-wavefunction Davidson NSCF
failed with `cdiaghg` even though the corresponding SCF
converged. The writer requires a **dressed smooth-source**
presence check, native projected-Pauli action, complex
Gram/field parity, source quarter-turn and tilted rotations,
and a `1e-3 rad` central difference. A nonmagnetic NC site
with zero dressed source is refused; a measured-zero nonlocal
magnetic source is expected for NC, not a missing vertex.

An SO-bearing NC control uses the *different* official
[PseudoDojo fully relativistic Fe NC UPF](https://www.pseudo-dojo.org/pseudos/nc-fr-04_pbe_standard/Fe.upf.gz)
(SHA-256 of decompressed `Fe.upf`:
`b9b7671a2c667a6f87a7228d30fcc205e339b69f5568fe3f3dba567fdce59a19`).
Independent x/y/z SCF densities and full-BZ NSCF dumps are
required, not rotations of one reference. Setting
`TB2J_NC_GRAM_DIAG` during each NSCF writes a separate
**normalized per-k beta-mode** diagnostic; use a new filename
distinct from `TB2J_DUMP`. On the 2³ FR Fe pilot, the ten
discarded modes carry 20.19% of the beta-projected occupied
spectral weight and 42.80% in the
`|E-E_F|<=0.5 eV` window. Those coarse-mesh weights and
the strongly threshold-sensitive dressed operator do **not**
certify a FR tensor. `qe2J.py --input_x/y/z` refuses SO-bearing
or Gram-truncated NC until per-k physical weight and
exchange-threshold sensitivity justify its error budget,
including the same FR UPF with `lspinorb=F` (QE clears
`has_so` but does not average its j-split beta radials).

The independent 12³ full-BZ x/y/z references with that FR UPF
(48 spinor bands each) discard ten of 34 beta modes at **every**
k point at `rcond=1e-6`. Those modes carry 20.03% of the
beta-projected occupied weight and 43.14% of the
`|E-E_F|<=0.5 eV` weight in each reference. At
`rcond=1e-5` and `1e-3`, 13 and 16 modes carry
24.22% and 32.42% of the occupied weight.
Reconditioning the same raw x leg at `rcond=1e-6`,
`1e-5`, `1e-3` changes `Jyy(R=(1,0,0))` from
9.6503205 to 9.6675615 to 9.6776549 meV;
across all x/y/z raw legs and six first neighbors, the
largest component shift over the full bracket is
0.0273347 meV. Neither this shift nor the clean eigengap
is a bound on the loss of normalized physical states.
The actual three-reference production CLI refused before
creating a tensor directory. A separate real Fe NC + O US
FeO dump reads and passes per-site source gates, but O
is induced magnetic (`DMAX=0.1074555 Ry`); no converged
real zero-source US ligand was validated. The public
three-reference CLI explicitly refuses mixed NC+US/PAW
spinor families too, while the strict reader remains
available for research diagnostics.

The same FR UPF with `lspinorb=F` was also run using
independent x/y/z 12³ SCFs and full-BZ 48-band NSCFs.
Its weighted scalar `dion` leaves all 34 j-split
beta channels and 43.1286% of Fermi-window
beta-projected weight in the ten modes discarded
at `rcond=1e-6`; the production three-leg CLI
refuses this SOC-off control too. The matched raw
transverse symmetric off-diagonal legs at
`R=(1,0,0)` (x:yz, y:xz, z:xy) are
`(+2.539e-9,-1.777e-9,+0.904e-9) meV` SOC-off
against `(0.0001784051,0.0001779155,0.0001776593) meV`
SOC-on. This is a research-only leg contrast:
the substantial discarded physical weight forbids
a certified anisotropy or SOC-only quantitative claim.
