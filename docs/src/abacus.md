### Use TB2J with ABACUS

In this tutorial we will learn how to use TB2J with ABACUS.  The TB2J-ABACUS interface is available since TB2J version 0.8.0. There are three types of basis set in ABACUS, the plane-wave (PW), the linear-combinatio of atomic orbitals (LCAO), and the LCAO-in-PW. With the LCAO basis set, TB2J can directly take the output and compute the exchange parameters. For the other type of basis set, the Wannier90 interace can be used instead.  In this tutorial we will use LCAO. 


#### Collinear calculation without SOC

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




