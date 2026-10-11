# Quantum ESPRESSO → TB2J: FM rocksalt FeO

This example uses the **collinear ferromagnetic** rocksalt primitive FeO cell
($a=8.19$ bohr; Fe at the origin, O at $(1/2,1/2,1/2)$ in fcc-primitive
coordinates). It exercises the QE `TB2J` fork branch's KB beta channel with
both Fe ultrasoft (`rrkjus`) and Fe PAW (`kjpaw`) pseudopotentials; O is
ultrasoft in both legs. It is not the experimentally ordered AFM FeO phase.

## Prerequisites

Build `pw.x` from `git@gitlab.com:mailhexu/q-e.git`, branch `TB2J`, as in
[`TB2J/qe_patch/README.md`](../../../TB2J/qe_patch/README.md). Use a
collinear calculation (`nspin=2`), one k pool, non-gamma-only wavefunctions,
and full BZ for the nscf step (`nosym=.true.`, `noinv=.true.`).

Place these exact pseudopotentials in `./pseudo/` (the inputs use this path):

| File | Source | SHA-256 of the validated file |
|---|---|---|
| `Fe.pbe-spn-rrkjus_psl.1.0.0.UPF` | [QE PS-library UPF](https://pseudopotentials.quantum-espresso.org/upf_files/Fe.pbe-spn-rrkjus_psl.1.0.0.UPF) | `32627f8c99bc4b2884f289ac6fba4a539e7672fa3514787ef52dd9b93bfa6cb4` |
| `Fe.pbe-spn-kjpaw_psl.1.0.0.UPF` | [QE PS-library UPF](https://pseudopotentials.quantum-espresso.org/upf_files/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF) | `a8c4ca198bc91c217b6b05670605b6d0c7865e8d36eb0d2154e9c33ba58e279a` |
| `O.pbe-rrkjus.UPF` | validated local QE O PBE RRKJ3 US UPF v0 (`~/espresso/pseudo/O.pbe-rrkjus.UPF`); the QE [archive URL](https://pseudopotentials.quantum-espresso.org/upf_files/O.pbe-rrkjus.UPF) serves a v2 conversion — verify the hash if substituting | `2b470cba6d6caf963806951ec9135ae425462160d82ce78026da3842f94b01e9` |

No pseudopotential bytes or large runtime output are checked into TB2J.

## Run

From this directory with the patched `pw.x` on `PATH`:

```sh
mkdir -p pseudo
# Install the three UPFs above into pseudo/, then run either family:
TB2J_DUMP=feo_us_scf.bin pw.x -in feo_us_scf.in > feo_us_scf.out
TB2J_DUMP=feo_us_nscf.bin pw.x -in feo_us_nscf14.in > feo_us_nscf.out
qe2J.py --input feo_us_nscf.bin --output_path FeO_US --elements Fe --Rcut 7

TB2J_DUMP=feo_paw_scf.bin pw.x -in feo_paw_scf.in > feo_paw_scf.out
TB2J_DUMP=feo_paw_nscf.bin pw.x -in feo_paw_nscf14.in > feo_paw_nscf.out
qe2J.py --input feo_paw_nscf.bin --output_path FeO_PAW --elements Fe --Rcut 7
```

Each family has its own `prefix` beneath the shared `./tmp` restart
directory; run its SCF before its NSCF. `TB2J_DUMP` is an output path,
not an input namelist variable. The SCF dump is useful for QE-native
occupation parity (G2);
exchange consumes the denser NSCF dump. Results are in `exchange.out` and
the normal TB2J/SpinIO output tree. The `qe2J.py` implementation is shared
across QE KB families.

## Measured exchange and limitations

2026-10-09, PBE, Fermi–Dirac `degauss=0.01` Ry, 24 bands, $8^3$ SCF,
$14^3$ full-BZ NSCF, 60/720 Ry Fe US + O US or 60/480 Ry Fe PAW + O US.
Fe moments are 4.11 (US) and 4.10 (PAW) $\mu_B$ per cell. Shell averages:

| Fe–Fe distance (Å) | QE US $J$ (meV) | QE PAW $J$ (meV) | Existing GPAW reference (meV) | ABINIT+UPF PAO (meV) |
|---:|---:|---:|---:|---:|
| 3.065 (first shell) | 5.7275 | 4.9822 | 5.8173 | 7.7202 |
| 4.334 (second shell) | −12.0164 | −13.0731 | −17.6106 | −15.3981 |

The $10^3\to14^3$ mesh shift was under 0.06 meV for both $J_1$ and $J_2$.
Both QE legs reproduce shell signs and roughly the first-shell scale. The
second-shell magnitude is **15–32% smaller** than the ABINIT/GPAW legs;
these pseudopotentials and representations are not matched to the reference
calculations, so this is a limitation, not a claimed cross-code equivalence.
See the `TB2J_projector` manuscript for the accompanying bcc-Fe comparison.

## Optional UPF atomic-wavefunction basis

To compare with the **same Fe/O US pseudopotentials and SCF density**,
rerun the US NSCF on the supplied 10³ full-BZ mesh with the alternate
projector selector; the default KB dump above is unaffected:

```sh
TB2J_PROJECTORS=kb TB2J_DUMP=feo_us_kb10.bin pw.x -in feo_us_nscf10.in > feo_us_kb10.out
TB2J_PROJECTORS=atomic TB2J_DUMP=feo_us_atomic10.bin pw.x -in feo_us_nscf10.in > feo_us_atomic10.out
qe2J.py --input feo_us_atomic10.bin --output_path FeO_US_atomic10 --elements Fe --Rcut 7
```

The atomic writer maps 14 orbitals into Fe [10] and O [4], writes a full
14×14 Gram matrix at every k and passes its real-space/G-space parity
check (3.03×10⁻¹³ in the validated run). On 10³, Fe–Fe first/second
shell values are approximately +7.125/−15.687 meV in atomic mode
versus +5.7257/−11.9647 meV in matched KB mode. This is an end-to-end
two-species smoke test, **not** validation that the atomic basis
reproduces KB or the independent GPAW/ABINIT references. The bccFe
atomic-basis mismatch and UPF `PP_PSWFC` requirement are documented
in [`TB2J/qe_patch/README.md`](../../../TB2J/qe_patch/README.md).
