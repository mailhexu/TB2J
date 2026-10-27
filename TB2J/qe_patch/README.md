# TB2J Quantum ESPRESSO patch: projector-Green dump (`qe_patch`)

This directory documents the instrumented Quantum ESPRESSO build that exports
projector data for TB2J's projector-Green exchange workflow. The
instrumentation does **not** ship as a diff to apply: it lives on branch
`TB2J` of the user QE fork `git@gitlab.com:mailhexu/q-e.git`
(base: upstream `develop`, reports as `PWSCF v.8.0dev`). You clone the fork,
check out the branch, and build `pw.x` as usual.

The exporter uses `PW/src/becp_dump.f90` and
`PW/src/becp_dump_hubbard.f90`, with hooks in `electrons.f90` and
`non_scf.f90` and the corresponding source-list registrations.
When enabled, it recomputes the selected projector
coefficients for every k-point from stored wavefunctions. The default
`TB2J_PROJECTORS=kb` exports `<beta|psi>` (the `pw2wannier90` pattern:
`get_buffer` → `init_us_2` → explicit plane-wave dot product);
`TB2J_PROJECTORS=atomic` exports `<atomic_wfc|psi>` and the full atomic
Gram matrix at every k. The dump includes the metadata needed by TB2J.

Current profiles are collinear KB `TB2JQEDUMPV1.3`, collinear atomic
`TB2JQEATWFC1.1`, and KB spinor `TB2JQEDUMPNC1.2`. Each appends typed
native Hubbard status/source evidence to its predecessor payload. The
predecessor layouts and historical **no-U** benchmarks below remain useful
format references; they are not active-U exchange certification.
The complete modern contract, `HUBBARD` versus `TB2J_PROJECTORS` choices,
API/CLI usage and measured capability limits are in
[`docs/src/qe_projector.rst`](../../docs/src/qe_projector.rst).
QE's authoritative native layout and reproducible source fixtures are
`Doc/TB2J_Hubbard_extension.md` and `test-suite/tb2j_hubbard/README.md`
in the instrumented QE checkout.

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

The dump is enabled with `TB2J_DUMP`; with this variable unset the hook is
a no-op. Select `TB2J_PROJECTORS=atomic` for UPF atomic wavefunctions;
unset it (or set `kb`) for the default KB beta channels. Use a different
dump filename for each run and mode.

Constraints enforced at dump time (`errore` aborts):

* **`npool = 1`**: do not use `-nk`/`-npools`; a pool-parallel dump is
  rejected (`nkstot != nks` would make the k lists incomplete).
* **Collinear**: `nspin = 2`; **noncollinear KB**:
  `noncolin=.true., domag=.true.` (`nspin=4`, spinors). Atomic mode
  remains collinear. SO-bearing UPFs require the NC1.1 source-response
  exporter and its native action, source-parity, and finite-difference
  gates; the legacy NC1.0 commutator is not valid for SO-bearing beta
  channels.
* **No gamma-only runs**: a Γ-only calculation is rejected; use a regular
  k-point mesh.
* The run must be converged; the dump is written once, on the ionode, after
  the electronic loop finishes.
* For atomic mode, every UPF must provide `PP_PSWFC` atomic wavefunctions;
  the exporter rejects a species with none. KB mode supports NC, US and PAW.

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
| 3 | species labels | QE `atm` fixed-capacity record (6-byte blank-padded labels in this fork; reader also accepts legacy `S3 × nsp`) |
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
(`nspin != 2`), pool-parallel dumps (`nkstot != nks`), incompatible
family/version combinations, and record-length/shape mismatches.
Pre-v1.2 KB dumps can be inspected but are refused for exchange because
the deeq-only vertex failed bccFe G3.

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
* **Operator and units.** In v1.2 the vertex is the joint-metric-dressed
  smooth-XC splitting plus `deeq(up) - deeq(down)`; v1.0/v1.1 use the
  `deeq` difference alone. Energies and operators are in Ry, converted
  to eV by the reader with `RYTOEV = 13.605693122994`.
* **Spin selection.** LSDA (`nspin = 2`): every k belongs to one spin channel
  via `isk` (1 = up, 2 = down).
* **`becsum` scf-vs-nscf caveat.** In a v1.1 dump, scf dumps hold the
  converged occupations of the final `sum_band`, while nscf dumps hold the
  restart occupations from `hinit1` (i.e. the **scf** mesh), *not* dense-mesh
  weights. Any occupation-based parity check (G2) must therefore use **scf**
  dumps; that is why the workflow above keeps `scf_dump.bin` around.

## 4. Atomic-wavefunction dump v1.0

Set `TB2J_PROJECTORS=atomic` and `TB2J_DUMP=atomic_nscf.bin` when
running `pw.x`. QE `atomic_wfc` constructs Bloch-summed pseudo-atomic
orbitals from the UPF `PP_PSWFC` block; this is a different **basis and
binary schema** from KB mode. The tested SG15 `Fe_ONCV_PBE-1.0.upf`
contains no atomic wavefunctions and is rejected; an NC UPF with
`PP_PSWFC` is required. The reader auto-detects the magic.

The magic is `TB2JQEATWFC1.0` (`S16` padded). Like the KB dump, records
use gfortran sequential unformatted 4-byte little-endian markers and
Fortran-ordered arrays. Atomic records are:

| # | Content | Layout |
|---|---------|--------|
| 0 | header | `S16` magic; `i4 × 8`: `nspin,nks,nkstot,nbnd,nat,nsp,nproj,max_nsite` |
| 1 | Fermi/smearing | `f8 × 5`: `nelec,ef,ef_up,ef_dw,degauss`; `i4 × 3`: `ngauss,ltetra,lgauss` |
| 2 | cell | `f8`: `at(3,3),bg(3,3),alat,omega`; `i4`: `ibrav` |
| 3 | species labels | QE fixed-capacity `CHARACTER(LEN=6)` labels |
| 4 | ions | `i4(nat)` `ityp`; `f8(3,nat)` `tau` |
| 5 | channels | `i4(nsp)` `nsite_species`; `i4(nproj)` each `projector_l,m_ordinal,radial_upf_index` |
| 6 | k mesh | `f8(3,nks)` `xk`, `f8(nks)` `wk`, `i4(nks)` `isk` |
| 7 | bands | `f8(nbnd,nks)` `et,wg` |
| `8+2*ik` | k-point primal coefficients | `c16(nproj,nbnd)` `C=<atomic_wfc(k)|psi(k)>`, `ik=0…nks-1` |
| `9+2*ik` | k-point full Gram | `c16(nproj,nproj)` `M=<atomic_wfc(k)|atomic_wfc(k)>` |
| `8+2*nks` | onsite smooth XC | `c16(max_nsite,max_nsite,nat)` `cov_xc` (Ry) |
| `9+2*nks` | onsite augmentation | same shape `cov_aug` (Ry) |

Channels are ordered by atom, UPF atomic radial orbital, then QE real
spherical-harmonic ordinal. `nsite_species` counts atomic orbitals only,
not KB beta channels. `M(k)` includes intersite blocks; the coefficients
are **primal**, so the TB2J projector-Green runtime applies its
k-dependent inverse (or selected regularization). The covariant onsite
vertex is `cov_xc + cov_aug`: the weighted full-BZ (up-spin copy only)
`R=0` average of `<atomic(k)|Vxc_up−Vxc_dn|atomic(k)>` and
`B(k)(deeq_up−deeq_dn)B(k)†`, with `B=<atomic(k)|beta(k)>`.
Taking only the first, often Γ, point folds neighboring periodic images
into the onsite operator. `cov_aug` is zero for NC; the format has no
`becsum` or separate `deeq` record. Both operators are converted
from Ry to eV by TB2J.

`qe2J.py --input atomic_nscf.bin --output_path TB2J_atomic --elements Fe`
uses a full inverse by default; atomic dumps alone accept
`--overlap_mode {inverse,svd,lowdin,tikhonov,plain}` and
`--overlap_rcond`. Passing these controls to a KB dump raises an error.
See `docs/src/qe_projector.rst` for the CLI and
`examples/qe/feo/` for reproducible two-atom US/PAW KB input decks.

### bccFe atomic-basis validation boundary

With a weighted full-BZ `R=0` vertex and an unregularized full `M(k)`
inverse, the optional atomic basis remains **unvalidated** quantitatively
on bccFe. Matched calculations use the same UPF within each row, 24 bands
and a full-BZ 12³ mesh; US/PAW use 60 Ry and NC uses 100 Ry:

| Fe UPF | KB `J1` (meV) | Atomic `J1` (meV) |
|---|---:|---:|
| PAW (`kjpaw`) | 14.7601 | 20.4267 |
| US (`rrkjus`) | 17.6958 | 24.3862 |
| NC (with `PP_PSWFC`) | 10.0730 | 24.5939 |

The NC UPF here (`Fe_NC_PAO.upf`) is **not** the atomic-wavefunction-free
SG15 ONCV UPF giving KB 15.22 meV below. An earlier first-Γ vertex gave
PAW 26.2341 meV: averaging removes that periodic-image artifact but
does not make the atomic and KB subspaces equivalent. These discrepancies
are not corrected with a fitted scale or overlap cutoff. The selectable
atomic mode is a research comparison surface, not a validated replacement
for the KB exchange numbers in the following table.

## 5. Noncollinear KB spinor dumps NC1.0 / NC1.1

`TB2J_PROJECTORS=kb` plus `TB2J_DUMP=spinor_nscf.bin` instruments
converged, magnetic `noncolin=.true.` NC, US, PAW and mixed-family
runs. Each independent x/y/z reference needs its own SCF density and
full-BZ NSCF with identical geometry, pseudopotential, k mesh and
spinor band window. Use `nosym=.true., noinv=.true.` on NSCF and no
pools or gamma-only mode. The 16-valence Fe US and scalar NC decks
live in `examples/qe/bccfe_spinor/`. The exporter checks complex beta
Gram/field parity, native total `add_vuspsi` action, frozen magnetic
source subtraction and source/action rotation. For NC, the measured
nonlocal magnetic source and all nine native response slabs are zero;
the per-site beta-projected **smooth** Pauli field carries its vertex.
For mixed cells the native projected-local-action oracle uses
site-local beta-built trials: a global trial contains intersite
`<beta_i|B|beta_j>` terms absent from the onsite dump. `s_psi` is
the generalized overlap, not a magnetic-action oracle. SO-bearing
native `fcoef` is not bare-spin covariant; its conjugation residual
is informational, not a gate.

The sequential-unformatted spinor ABI uses separate magics
`TB2JQEDUMPNC1.0` and `TB2JQEDUMPNC1.1`; neither overloads the
collinear v1.2 or atomic formats. Numeric arrays use Fortran order
and little-endian gfortran four-byte record markers:

| Record | Contents |
|---|---|
| 0 | `S16` magic plus 11 `i4` dimensions/flags: `nspin,nks,nkstot,nbnd,nat,nsp,nhm,lmaxkb,nkb,lspinorb,domag` |
| 1 | `f8 nelec,ef,degauss`; `i4 ngauss,ltetra,lgauss` |
| 2–4 | cell `at,bg,alat,omega,ibrav`; `S6(nsp)` species labels; `ityp(nat),tau(3,nat)` |
| 5 | contiguous `i4(nsp)` arrays `nh,tvanp,tpawp,has_so` |
| 6–8 | native total `deeq_nc_total(nhm,nhm,nat,4)`; separately magnetic `deeq_nc_magnetic` (both `c16`, `(uu,ud,du,dd)`); complex beta Gram `c16(nhm,nhm,nat)` |
| 9–10 | `xk(3,nks),wk(nks)`; `et(nbnd,nks),wg(nbnd,nks)` (all `f8`) |
| 11…10+nks | per-k `becp%nc(nkb,2,nbnd)` (`c16`) |
| 11+nks | complex Hermitian `<beta|B_x,y,z|beta>` (`c16(nhm,nhm,nat,3)`) |
| 12+nks | deterministic zero reserved `f8(nhm*(nhm+1)/2,nat,4)`; QE's NSCF `becsum` is uninitialized, **not** an occupation oracle |
| 13+nks (NC1.1 only) | nine native source responses `D_response(nhm,nhm,nat,4,3,3)` (`c16`); axes: two beta channels, atom, native spin block `(uu,ud,du,dd)`, destination Pauli field `c`, original magnetic source `j` |

TB2J normalizes one full-BZ spinor k set (`sum(wk)=1`), takes band
occupations from `wg/wk`, and keeps already-dual beta coefficients
without applying an overlap inverse to the Green function. The
Hermitian beta-Gram pseudoinverse (`rcond=1e-6`) dresses only the
**projected smooth magnetic field**; it never changes `becp` or the
native nonlocal blocks. A numerical eigengap is necessary for stable
truncation, but does **not** bound the physical weight of a
Gram-normalized discarded mode. In the PseudoDojo SO-bearing Fe NC
pilot, `rcond=1e-6` retains modes at eigenvalue `5.4e-6` and
amplifies `max|M^+ A M^+|` to about `4535 Ry`. Neither a fitted
scale nor an eigengap certifies that FR-NC exchange vertex.

The spinor splitting is
`Delta = 2 RYTOEV [sum_i (M^+ A_i M^+) tensor sigma_i + Dmag]`;
fixed ionic/SOC terms from `deeq_nc_total` are excluded.
For NC1.1, each source `j` is placed into every destination Pauli
slot `c` at frozen density, with the PAW one-center magnetic source
rotated in tandem; `D_response[c,j]` is the total-minus-zero-source
native nonlocal matrix. Its diagonal sum reconstructs `Dmag`.
For local moment direction `n`, the tangent for rotation axis `a` is
`V_a = RYTOEV sum_cj K_cj^(a) [(M^+ A_j M^+) tensor sigma_c
+ D_response[c,j]]`, where
`K_cj^(a) = sum_d epsilon_cdj (n cross e_a)_d`.
This is `dHmag/dtheta = (1/2) dDelta/dtheta`; it rotates
**only the magnetic source**, not the fixed SOC Hamiltonian.

NC1.0 has no native source response and refuses SO-bearing data.
For NC1.1 pure NC, `Dmag` and all `D_response` slabs are measured
zero, and the smooth-source explicit tangent reduces exactly to
the scalar spin commutator. The writer admits a magnetic NC site
using `max|M^+ A M^+|`, checks the native smooth Pauli action on
its beta span, x/y quarter-turn and tilted rotations, a central
difference at `1e-3 rad`, and retained-subspace reconstruction.
Its printed `TB2J_CPL_OUT` is a **per-site Euclidean plane-wave
projection residual** on stored states; it includes discarded
in-span content and does not measure completeness outside the
full beta span separately. In mixed US/NC cells it is not a
generalized-overlap norm or a bound on exchange error.

A standard full tensor requires all three dumps:

```bash
qe2J.py --input_x fe_x.bin --input_y fe_y.bin --input_z fe_z.bin \
  --elements Fe --Rcut 2.6 --output_path TB2J_results_fe_spinor
```

One dump can only yield a raw transverse leg; the writer rejects a
single dump as full `exchange.out`. At first-neighbor R=(1,0,0)
(12³ full BZ, 48 spinor bands versus 24 bands per collinear spin,
30 poles, 0.05 eV contour smearing), scalar-relativistic Fe US NC1.1
gives `Jxx=Jyy=17.6997098 meV` versus matched same-UPF collinear KB
`17.6957657 meV`, difference `0.0039441 meV` below the
`0.01 meV` G3 tolerance. The PAW NC1.1 z leg gives
`Jxx=Jyy=14.7589664 meV` versus matched collinear
`14.7600550 meV`, difference `0.0010886 meV`, also within
the same bound. The earlier NC1.0 US leg was
`17.6929221 meV`, a separate legacy ABI check.

SO-bearing Fe US/PAW previously failed the **invalid** bare-spin
conjugation gate (relative residuals `0.33148`/`0.27662`).
NC1.1 keeps this as an informational diagnostic and validates
the native source-response action directly. A successful export
does **not** by itself validate a merged tensor: independent
x/y/z SCF references must pass the repeated-transverse-row gate
for **every intersite pair written within** the requested `Rcut`.
The on-site self-pair and out-of-cutoff pairs are excluded before
the gate. The PBE fully relativistic Fe US 12³ family passes the
first-neighbor-only `Rcut=2.6 Å` tensor (eight equivalent
neighbors, `Jiso=17.7174448 meV`, worst repeated-row spread
`8.367697e-9 eV` versus `5e-8 eV`). Its `Rcut=6 Å`
far shells instead fail (`2.5862561e-5 eV`).

Independent scalar-relativistic US x/y/z references at the same
mesh pass with repeated-row spread `1.012117e-8 eV` and
`|Jani_yz(R=(1,0,0))|<2.3e-15 meV`, against the
SO-bearing US value `0.000863377 meV`; this is a
nonzero-SOC anisotropy control, not a matched-UPF
isotropic SOC-shift measurement.

SO-bearing Fe PAW also passes native response/action gates,
but `conv_thr=1e-8 Ry` SCF densities **fail** first-neighbor
merging (`4.117504e-6 eV`). Tightening all three SCF
references to `conv_thr=1e-12 Ry` and regenerating the 12³
NSCF dumps yields a genuine first-neighbor rank-nine tensor:
`Jiso=14.7775026 meV`, maximum repeated-row spread
`2.086799e-8 eV`, first-shell DMI norm ≤`7.8915e-7 meV`
and tensor pair-reversal residual <`4e-18 eV`.
The tight PAW `Rcut=6 Å` family still fails (first
inconsistent far row `1.558e-5 eV`); do not relax the gate
to manufacture a six-angstrom US or PAW tensor.

The historical NC1.0/NC1.1 payload does not certify a Hubbard-source
derivative and must not be used to infer active-U support. Current NC1.2
exports the native scalar-Dudarev main-shell source and explicit frozen
rotation tangents for admitted KB spinor cases. Native spinor background U
and atomic-exchange spinor mode remain unsupported. Native source/action
success alone is not independent x/y/z production-tensor certification.

Real scalar ONCV Fe NC1.1 2³/8³/12³ spinor NSCF runs pass
the source, native-action and rotation gates. The 2³ nonmagnetic
NC control has zero dressed field and refuses to write a dump.
Scalar NC exchange is **anchored**: with matched band windows
(48 total spinor bands versus 24 per collinear spin), same
`Fe_ONCV_PBE-1.0` UPF, 12³ full BZ, 30 poles and 0.05 eV
smearing, the first-neighbor transverse legs give
`Jxx=Jyy=14.7910404 meV` versus matched collinear v1.2
`14.7932825 meV`; the maximum absolute difference over all six
first-neighbor vectors is `0.0023220 meV`, inside the 0.01 meV
G3 gate (US 0.0039, PAW 0.0011). An unmatched 24-vs-48 window
artificially inflates the discrepancy to 0.300 meV.

A real 2³ Fe NC + O US FeO run passes **per-site** writer/reader
gates, but O has a nonzero induced source (`DMAX=0.1074555 Ry`).
Neither that run nor two nonconverged compensated-Fe trials provide
the required real **nonmagnetic** augmented ligand. The production
three-leg `qe2J.py` path therefore explicitly refuses mixed NC
and augmented spinor references; the strict reader still admits their
per-site diagnostics, not a publishable mixed-family tensor.

For the PseudoDojo
16-valence fully relativistic Fe NC UPF (SHA-256
`b9b7671a2c667a6f87a7228d30fcc205e339b69f5568fe3f3dba567fdce59a19`),
a genuine SOC-on 2³ NC1.1 dump passes the writer but
**cannot certify a tensor**: across the whole 2³ Brillouin zone,
the ten Gram modes discarded at `rcond=1e-6` carry 20.19%
of the beta-projected occupied-state weight and 42.80% of the
`|E-E_F|<=0.5 eV` window weight. A single-k gap and rank-nine
test cannot bound this error. `qe2J.py` refuses SO-bearing NC and
NC with truncated modes, including the same FR UPF run with
SOC disabled (`average_pp` clears `has_so` but retains separate
beta radials); `read_qe_spinor_dump` permits research-only raw
legs with an explicit warning.

Independent x/y/z **12³ full-BZ** SCFs and 48-band NSCFs
with that same FR UPF pass the native writer gates. At every one of
1728 k points in each reference, ten of 34 beta modes fall below
`rcond=1e-6`; they carry 20.03% of beta-projected occupied
weight and 43.14% within `|E-E_F|<=0.5 eV`.
At `rcond=1e-5` and `1e-3`, 13 and 16 modes are discarded,
carrying 24.22% and 32.42% of occupied weight. On the *same
x-reference*, research-only `Jyy(R=(1,0,0))` is 9.6503205,
9.6675615 and 9.6776549 meV at those three thresholds.
Across the independent x/y/z raw legs, six first-neighbor
translations and all components, the full threshold bracket
changes at most 0.0273347 meV. The three-reference production
CLI **refuses** before rank-nine output, with no tensor directory:
the large discarded normalized-state weight has no justified
physical error budget. The modest raw-leg shift is not an
error bound and does not override that refusal.

A cutoff **below** the first measured eigengap was also
tested: `rcond=1e-10` retains all 34 modes at every k.
The untruncated smooth operator reaches
`3.4746e8 eV` (versus `8.6952e4 eV` at `1e-6`)
and violates the unchanged Hermiticity gate in all
three legs by `13.7–16.9 eV` absolute. No below-gap
exchange component is defined by the validated trace;
symmetrizing this unstable operator to manufacture a
number would conceal a failure. The reported
`1e-6`–`1e-3` raw sensitivity does **not** bracket
the first eigengap or bound the missing physical
weight; both sides of that gap now explicitly fail
certification for different reasons.

A **matched same-UPF SOC-off control** was run with
`lspinorb=F`: independent x/y/z 12³ SCFs and full-BZ
48-band NSCFs at the same 80/320 Ry cutoffs, Fermi–Dirac
0.01 Ry smearing, 30 CFR poles and 0.05 eV contour.
QE's weighted scalar `dion` clears `has_so` but retains
the 34 j-split beta channels. At every k, the ten modes
discarded at `rcond=1e-6` still carry 20.0307% of
beta-projected occupied weight and 43.1286% of the
±0.5 eV Fermi-window weight. The **three-distinct-leg
SOC-off production CLI also refuses** before output.
At `R=(1,0,0)`, the *research-only* symmetric
transverse off-diagonal raw legs (x:yz, y:xz, z:xy)
are `+0.0001784051`, `+0.0001779155`,
`+0.0001776593 meV` with SOC on, versus
`+2.539e-9`, `-1.777e-9`, `+0.904e-9 meV`
with SOC off. These are not a rank-nine anisotropy
tensor or a calibrated SOC effect: both conditioned
beta spaces lose about 43% of the normalized near-Fermi
weight, and disabling SOC changes QE's weighted
nonlocal coefficients without averaging beta radials.

For a cross-code check, independent GPAW 26.7 LDA/PW92
x/y/z spinor references (8³, 96 bands, 1100 eV) were
generated. Their initial first-shell rank-nine merge
**refuses** at `1.592e-7 eV` versus GPAW's unchanged
`1e-8 eV` repeated-row bound. Tightening all three
GPAW SCFs to energy/density `1e-8` still refuses
(first row `4.751e-8 eV`; worst first-shell spread
`5.241305e-6 eV`, transverse-mask residual
`1.750664e-6 eV`). Do not construct a GPAW full
tensor from these references. At 30 poles / 0.05 eV
the **single x leg** gives first-neighbor transverse
`Jyy=13.803708 meV`; QE PBE (12³, 48 bands, 80 Ry)
gives US `17.717442` and tight PAW `14.777512 meV`.
XC, PAW setups, beta subspaces, meshes and band
windows differ, so this is not a matched-functional
quantitative tensor comparison. GPAW's spinor PW SCF
mode in this installation rejects PBE.

## 6. Reading the dump with TB2J

The Python side is documented in `docs/src/qe_projector.rst`.
Mode-specific raw parsers `parse_qe_dump`, `parse_qe_atomic_dump`,
`parse_qe_spinor_dump` and normalized readers `read_qe_dump`,
`read_qe_atomic_dump`, `read_qe_spinor_dump` live in
`TB2J/interfaces/qe_projector.py` and return `ProjectorGreenData`.
`TB2J/scripts/qe2J.py` auto-detects the magic or accepts x/y/z
spinor inputs explicitly.

## Known quirks

- Pure-NC nscf runs with random starting wavefunctions can abort in
  `cdiaghg` ("S matrix not positive definite") with Davidson on the
  tested ONCV setup; `diagonalization='cg'` succeeds. In a mixed
  Fe NC + O US 2³ NSCF, also set `diago_cg_maxiter=200` if the
  default CG limit leaves six or more bands unconverged.
- Collinear v1.2 `becsum` is zero-filled for norm-conserving runs;
  its US/PAW SCF occupation-parity diagnostic does **not** extend
  to NC1.0/NC1.1 spinor NSCF dumps. Their reserved four-component
  record is zero for every family; use `wg/wk` for occupations.

## KB family support (validated on bccFe, dump v1.2 vertex)

| Family | bccFe J1 (meV) | Status |
|--------|----------------|--------|
| PAW (`kjpaw`) | 14.76 | validated (GPAW reference 14.73) |
| NC (ONCV) | 15.22 | validated |
| US (`rrkjus`) | 16.15 (20^3) | validated (within cross-code spread) |
