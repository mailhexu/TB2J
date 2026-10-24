### Use TB2J with ABACUS

ABACUS has separate TB2J interfaces for LCAO Hamiltonian/overlap output and supported plane-wave UPF atomic-projector output. The existing LCAO interface is available since TB2J 0.8.0. The PW route below uses a native optional exporter and the common projector exchange engine; Wannier90 remains an alternative workflow, not a prerequisite for this atomic-projector route.

#### Plane-wave scalar-NC UPF atomic projectors

The optional native exporter is detected automatically when NetCDF C with NetCDF-4 group support and `nlohmann-json` are available; otherwise it is unavailable. `ENABLE_JSON` remains independent, and the native `source/CMakeLists.txt` build layer performs detection. There is no exporter build switch. See the native `docs/advanced/install.md` for dependency-location hints and details.

`out_tb2j` is a runtime opt-in and requires CPU complex-double PW wavefunctions. GPU and single-precision export are unsupported. The tested native target in the earlier campaign was `abacus_basic_para`.

Use a converged native PW Kohn–Sham calculation with `nspin 2`, `symmetry -1`, `out_tb2j 1`, no U/SOC/noncollinearity, and original scalar norm-conserving UPFs containing atomic wavefunctions. Only a certified generated full-BZ Gamma/MP recipe is supported. An explicit or pre-reduced list is not certified by setting `symmetry -1`. PAW/USPP and meta-GGA/hybrid XC are not supported here. Projectors are **all UPF atomic-wavefunction channels**, not beta projectors, Wannier functions, or the filtered native initial guess.

After native convergence and complete publication, run the installed dedicated command:

```bash
abacus_projector2J.py --input OUT.ABACUS/tb2j_projector.nc \
    --output_path TB2J_results --elements Fe --Rcut 10 --nz 30 --smearing 0.05
```

Adjust the output suffix and magnetic elements for the material. `--Rcut` is Angstrom; `--smearing` is contour broadening in eV, not the native SCF Gaussian width. The command fixes complete `delta_total`, projector populations and explicit inverse metric even under `TB2J_GREEN_MODE=plain`; it has no operator/overlap selector or pseudoinverse fallback. The full-Gram condition limit is `1e12`.

The common reader/engine writes the full standard output tree with truthful ABACUS NC-UPF provenance. `SpinIO.load_pickle(path=OUTPUT_DIRECTORY)` provides full-precision exchange values; rounded `exchange.out` values cannot resolve strict MPI differences. The existing `abacus2J.py` command below remains the LCAO interface and is unchanged.

Only `tb2j_projector.nc` is the final native dataset. A failed run may leave an older final or an unpublished `.partial.*` sibling after abrupt MPI abort; never pass partial files to the consumer or rename them into a final. Native streaming failures abort the world rather than silently falling back.

See {ref}`canonical consumer semantics and measured validation boundaries <abacus-plane-wave-atomic-projectors>`. The shell budgets and bccFe values describe the retained earlier fixed-build campaign; they are not new verification results for the simplified native exporter. That campaign accepts only the first three shells through 4.056 Angstrom, with distinct replay 0.01 meV/shell, cross-code 0.10 meV/shell and same-binary serial/MPI 0.000001 meV/shell budgets. Requested `Rcut=10` is not a convergence window: the common consumer also truncates to its k-mesh supercell and warns about aliased exterior images. No extended-range or uniformly converged unoccupied-spectrum claim follows.


#### LCAO collinear calculation without SOC

Let's start from the example of Fe. The example files can be found here: https://github.com/mailhexu/TB2J_examples/tree/master/Abacus/Fe_no_SOC . 

First do the ABACUS calculation. Note that the Kohn-Sham Hamiltonian and the overlap matrix is needed as the input to TB2J. We need to put 

``` 
out_mat_hs2  1 
```

in the ABACUS INPUT file, so that the Hamiltonian matrix H(R) (in Ry) and overlap matrix S(R) will be written into files in the directory `OUT.${suffix}` . In the INPUT, the line

```
sufffix Fe
```

specifies the suffix of the output. Thus the output will be in the directory OUT.Fe when the DFT calculation is finished.

 In this calculation, we set the  path to the directory of the DFT calculation, which is the current directory (". ") and the suffix to Fe. 

Now we can run the abacus2J.py command to calculate the exchange parameters:

```bash
abacus2J.py --path . --suffix Fe --elements Fe  --kmesh 7 7 7
```

This first read the atomic structures from th STRU file,  then read the Hamiltonian and the overlap matrices stored in the files named starting from "data-HR-" and "data-SR-" files.  It also read the fermi energy from the OUT.Fe/running\_scf.log file.  

With the command above, we can calculate the J with a 7x7x7 k-point grid. This allows for the calculation of exchange between spin pairs between 7x7x7 supercell.  Note: the kmesh is not dense enough for a practical calculation. For a very dense k-mesh, the --rcut option can be used to set the maximum distance of the magnetic interactions and thus reduce the computation cost. But be sure that the cutoff is not too small. 

#### Non-collinear calculation with SOC

The DMI and anisotropic exchange are result of the SOC, therefore requires the DFT calculation to be done with SOC enabled. To get the full set of exchange parameters, a "rotate and merge" procedure is needed, in which several DFT calculations with either the structure or the spin rotated are needed. 
For each of the non-collinear calcualtion, we compute the exchange parameters from the DFT calculation with the  same command as in the collienar case. 

```bash
abacus2J.py --path . --suffix Fe --elements Fe  --kmesh 7 7 7
```

And then the "TB2J_merge.py" command can be used to get the final spin interaction parameters. 


#### Split SOC workflow

TB2J supports two split-SOC modes for non-collinear exchange calculations with ABACUS, selected with the `--split_soc` flag:

**Single-step** (`--split_soc single`): requires one SOC ABACUS output directory with both

```
out_mat_hs2 1
out_mat_hs2_soc 1
```

in `INPUT`. These write `hrs1_nao.csr` (total Hamiltonian), `srs1_nao.csr` (overlap), and `hrs1-soc_nao.csr` (the exact nonlocal difference $H(\mathrm{configured\ SOC}) - H_{\mathrm{scalar}}$). No separate no-SOC calculation is needed.

```bash
abacus2J.py --path . --suffix Fe --elements Fe --kmesh 7 7 7 --split_soc single
```

**Two-step** (`--split_soc two`): requires matched no-SOC and SOC ABACUS output directories with the same suffix. The `--path` option selects the SOC directory, and `--path_nosoc` selects the no-SOC directory.

```bash
abacus2J.py --path . --suffix Fe --elements Fe --kmesh 7 7 7 --split_soc two --path_nosoc ../nosoc
```

In both modes, `--output_path` selects the directory where TB2J writes the exchange results.


#### Parameters of abacus2J.py

We can use the command 

```bash
abacus2J.py --help
```

to view the parameters and the usage of them in abacus2J.py.  

```

TB2J version 0.8.0
Copyright (C) 2018-2024  TB2J group.
This software is distributed with the 2-Clause BSD License, without any warranty. For more details, see the LICENSE file delivered with this software.


usage: abacus2J.py [-h] [--path PATH] [--suffix SUFFIX] [--elements [ELEMENTS ...]] [--rcut RCUT] [--efermi EFERMI]
                   [--kmesh [KMESH ...]] [--emin EMIN] [--use_cache] [--nz NZ] [--cutoff CUTOFF]
                   [--exclude_orbs EXCLUDE_ORBS [EXCLUDE_ORBS ...]] [--np NP] [--description DESCRIPTION]
                   [--orb_decomposition] [--fname FNAME] [--output_path OUTPUT_PATH]

abacus2J: Using magnetic force theorem to calculate exchange parameter J from ABACUS Hamiltonian in the LCAO mode

options:
  -h, --help            show this help message and exit
  --path PATH           the path of the ABACUS calculation
  --suffix SUFFIX       the label of the ABACUS calculation. There should be an output directory called OUT.suffix
  --elements [ELEMENTS ...]
                        list of elements to be considered in Heisenberg model.
  --rcut RCUT           range of R. The default is all the commesurate R to the kmesh
  --efermi EFERMI       Fermi energy in eV. For test only.
  --kmesh [KMESH ...]   kmesh in the format of kx ky kz. Monkhorst pack. If all the numbers are odd, it is Gamma
                        cenetered. (strongly recommended), Default: 5 5 5
  --emin EMIN           energy minimum below efermi, default -14 eV
  --use_cache           whether to use disk file for temporary storing wavefunctions and hamiltonian to reduce memory
                        usage. Default: False
  --nz NZ               number of integration steps. Default: 50
  --cutoff CUTOFF       The minimum of J amplitude to write, (in eV). Default: 1e-7 eV
  --exclude_orbs EXCLUDE_ORBS [EXCLUDE_ORBS ...]
                        the indices of wannier functions to be excluded from magnetic site. counting start from 0.
                        Default is none.
  --np NP               number of cpu cores to use in parallel, default: 1
  --description DESCRIPTION
                        add description of the calculatiion to the xml file. Essential information, like the xc
                        functional, U values, magnetic state should be given.
  --orb_decomposition   whether to do orbital decomposition in the non-collinear mode. Default: False.
  --fname FNAME         exchange xml file name. default: exchange.xml
  --output_path OUTPUT_PATH
                        The path of the output directory, default is TB2J_results
```

