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
