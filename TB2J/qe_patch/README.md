# TB2J Quantum ESPRESSO patch: projector-Green dump (`qe_patch`)

This directory documents the instrumented Quantum ESPRESSO build that exports
projector data for TB2J's projector-Green exchange workflow. The
instrumentation does **not** ship as a diff to apply: it lives on branch
`TB2J` of the user QE fork `git@gitlab.com:mailhexu/q-e.git`
(base: upstream `develop`, reports as `PWSCF v.8.0dev`). You clone the fork,
check out the branch, and build `pw.x` as usual.

The exporter adds one file, `PW/src/becp_dump.f90`, plus small hooks in
`electrons.f90` and `non_scf.f90` (and the corresponding source-list
registrations). It writes no physics changes: at the end of a converged run it
recomputes the ultrasoft/PAW projector coefficients
`becp%k(nkb, nbnd)` for every k-point from the stored wavefunctions (the
`pw2wannier90` pattern: `get_buffer` → `init_us_2` → explicit
`P_ni = sum_G vkb_i^* evc_n`) and dumps them with all metadata needed by TB2J
into one sequential-unformatted binary file.

## 1. Obtaining and building

```bash
git clone git@gitlab.com:mailhexu/q-e.git
cd q-e
git checkout TB2J
./configure --disable-parallel   # or your preferred configuration
make pw
```

Any configuration that builds `pw.x` with gfortran works; the dump is
sequential-unformatted Fortran with 4-byte record markers (see the format
section below), so use the gfortran-built binary that `make pw` produces.

## 2. Running: scf + nscf with `TB2J_DUMP`

The dump is enabled purely through the environment variable `TB2J_DUMP` — no
change to the QE input files is required. When the variable is unset the hook
is a no-op.

Constraints enforced at dump time (`errore` aborts):

* **`npool = 1`**: do not use `-nk`/`-npools`; a pool-parallel dump is
  rejected (`nkstot != nks` would make the k lists incomplete).
* **Collinear LSDA only**: `nspin = 2` (`noncolin`/`lspinorb` are rejected).
* **No gamma-only runs**: a Γ-only calculation is rejected; use a regular
  k-point mesh.
* The run must be converged; the dump is written once, on the ionode, after
  the electronic loop finishes.
* US **or** PAW species must be present (see the family support table in
  `docs/src/qe_projector.rst`; pure NC pseudopotentials cannot be exported —
  the separable beta spin vertex vanishes identically).

Recommended workflow — scf with the production mesh, then nscf on a denser
mesh, each with its **own** dump filename:

```bash
# --- scf: converged density, collinear LSDA
export TB2J_DUMP=scf_dump.bin
pw.x -i scf.in > scf.out

# --- nscf: dense k mesh for the exchange integral, full BZ
export TB2J_DUMP=nscf_dump.bin
pw.x -i nscf.in > nscf.out
```

with the relevant input flags:

```text
# scf.in  (namelist &system)
  nspin = 2
  starting_magnetization = ...     # collinear LSDA setup

# nscf.in  (namelist &system)
  nspin = 2
  nosym = .true.
  noinv = .true.                   # full-BZ k list: TB2J needs every k of the
                                   # mesh, not the irreducible wedge
# nscf.in  (namelist &electrons + cards)
  nbnd = ...                       # include the empty bands you want in G(E)
  K_POINTS automatic
  nk1 nk2 nk3 0 0 0                # dense mesh
```

`nosym = .true., noinv = .true.` matters: the dump stores the k-list of the
run as-is (`xk`, `wk`), and TB2J's Green-function reconstruction requires the
full Brillouin zone. With symmetry on, QE generates only the irreducible
wedge.

TB2J consumes the **nscf** dump (dense mesh, the `nscf_dump.bin` above). The
scf dump is kept for population diagnostics — see the `becsum` caveat below.

## 3. Dump format v1.2

Sequential unformatted Fortran (gfortran: 4-byte **little-endian** record
markers). All arrays are Fortran/column-major. The authoritative record list
for v1.2 (v1.1 = v1.2 minus `dbeta_xc`/`ddd_paw`; v1.0 = v1.1 minus `becsum`/`rho%bec`):

| # | Content | Layout (dtype × shape, Fortran order) |
|---|---------|----------------------------------------|
| 0 | magic + dims | `S16` magic (`'TB2JQEDUMPV1.2 '`, `'TB2JQEDUMPV1.1 '`, or `'TB2JQEDUMPV1   '`), then `i4 × 9`: `nspin, nks, nkstot, nbnd, nat, nsp, nhm, lmaxkb, nkb` |
| 1 | Fermi/smearing | `f8 × 5`: `nelec, ef, ef_up, ef_dw, degauss`; then `i4 × 3`: `ngauss, ltetra, lgauss` (logicals as 0/1) |
| 2 | cell | `f8`: `at(3,3), bg(3,3), alat, omega`; then `i4`: `ibrav` |
| 3 | species labels | `S3 × nsp` (`atm`) |
| 4 | ions | `i4 × nat` (`ityp`), `f8(3,nat)` (`tau`, cartesian, units of alat) |
| 5 | projector metadata | `i4 × nsp` (`nh`), `i4 × nsp` (tvanp flags), `i4 × nsp` (tpawp flags) |
| 6 | `deeq` | `f8(nhm,nhm,nat,nspin)` |
| 7 | `dvan` | `f8(nhm,nhm,nsp)` |
| 8 | `qq_at` | `f8(nhm,nhm,nat)` |
| 9 | beta Gram diagnostic | `c16(nhm,nhm,nat)` — diagnostics ONLY, never a channel metric |
| 10 | k data | `f8(3,nks)` (`xk`, cartesian 2π/alat), `f8(nks)` (`wk`), `i4(nks)` (`isk`, 1/2) |
| 11 | bands | `f8(nbnd,nks)` (`et`), `f8(nbnd,nks)` (`wg`) |
| 12 … 12+nks−1 | per-k `becp` | `c16(nkb,nbnd)` — projector × band for that k |
| 12+nks | `becsum` | `f8(nhm*(nhm+1)/2, nat, nspin)` — packed triangular, v1.1+ |
| 13+nks | `rho%bec` | same shape, PAW runs only, v1.1+ |
| 14+nks | `dbeta_xc` | `f8(nhm,nhm,nat)` — <beta_i\|V_xc^up − V_xc^dn\|beta_j> (Ry), v1.2 only |
| 15+nks | `ddd_paw` | `f8(nhm*(nhm+1)/2, nat, nspin)` — packed one-center D^1, PAW only, v1.2 only |

v1.0 ends after the per-k records. v1.2 `dbeta_xc` is computed on the dense FFT
grid with a dump-time real-space/G-space beta-Gram parity check; the TB2J site
vertex is `M^-1 dbeta_xc M^-1 + (deeq_up − deeq_dn)` (M = beta Gram record:
joint metric transformation; the deeq term already contains the PAW `ddd_paw`
splitting). The deeq-only vertex was falsified by the bccFe G3 gate.
The TB2J reader rejects unknown magic/version strings, non-LSDA dumps
(`nspin != 2`), pool-parallel dumps (`nkstot != nks`), dumps without any US or
PAW species, and any record-length/shape mismatch.

### Semantics and caveats

* **Channel layout.** `becp(nkb, nbnd)` per k holds
  `P_ni = <beta_n|psi_i>` with beta channels concatenated over atoms in QE
  ion order; atom `na` (1-based) of species `nt = ityp(na)` owns channels
  `ofsbeta(na)+1 … ofsbeta(na)+nh(nt)`, where `ofsbeta` is the cumulative
  offset over atoms (the TB2J reader rebuilds it from `ityp` + `nh`).
* **Coefficients are already dual** relative to the implicit basis dual to
  beta, and `deeq` is the matching covariant separable-operator coefficient
  (`V_NL = beta D beta^dagger`). The TB2J contract therefore keeps
  `overlap_k = None`: the `qq_at`/Gram records are **diagnostics only** and
  are **never** used as a channel metric.
* **`qq_at` is not a Gram matrix.** `qq_at` belongs to the wavefunction
  overlap operator `S_psi = I + beta q_at beta^dagger`, not to the TB2J
  projector channel. Do not treat `qq_at` (or the record-9 Gram diagnostic)
  as a metric to dress the Green function; the dual coefficients already
  produce the dual-dual Green matrix.
* **Operator and units.** The spin vertex is `hij = deeq(up) - deeq(down)`
  per atom block; `deeq`, `dvan`, `et` and `ef` are in Ry and are converted
  to eV by the reader with `RYTOEV = 13.605693122994`. Coefficients are
  dimensionless.
* **Spin selection.** LSDA (`nspin = 2`): every k belongs to one spin channel
  via `isk` (1 = up, 2 = down).
* **`becsum` scf-vs-nscf caveat.** In a v1.1 dump, scf dumps hold the
  converged occupations of the final `sum_band`, while nscf dumps hold the
  restart occupations from `hinit1` (i.e. the **scf** mesh), *not* dense-mesh
  weights. Any occupation-based parity check (G2) must therefore use **scf**
  dumps; that is why the workflow above keeps `scf_dump.bin` around.

## 4. Reading the dump with TB2J

The Python side is documented in `docs/src/qe_projector.rst` (user guide) and
implemented in `TB2J/interfaces/qe_projector.py` (raw parser
`parse_qe_dump` → `QEProjectorDump`) with normalization in
`TB2J.interfaces.qe_projector.read_qe_dump` → `ProjectorGreenData`. The CLI is
`TB2J/scripts/qe2J.py`.

## Known quirks

- Pure-NC nscf runs with random starting wavefunctions can abort in
  `cdiaghg` ("S matrix not positive definite") with Davidson on some
  ONCV setups; use `diagonalization = 'cg'` and
  `startingwfc = 'atomic+random'` for the nscf step (validated:
  `Fe_ONCV_PBE-1.0.upf`, bccFe).
- `becsum` is zero-filled in NC-only runs (no augmentation charges); the
  occupation-parity diagnostic applies to US/PAW dumps only.

## Family support (validated on bccFe, dump v1.2 vertex)

| Family | bccFe J1 (meV) | Status |
|--------|----------------|--------|
| PAW (`kjpaw`) | 14.76 | validated (GPAW reference 14.73) |
| NC (ONCV) | 15.22 | validated |
| US (`rrkjus`) | 16.15 (20^3) | validated (within cross-code spread) |
