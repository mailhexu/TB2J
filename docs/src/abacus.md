### Use TB2J with Abacus

In this tutorial we will learn how to use TB2J with Abacus.  The TB2J-Abacus interface is available since TB2J version 0.8.0.


#### Collinear calculation without SOC

Let's start from the example of Fe. The input files used can be found in the examples/Abacus/Fe directory. 

First do the abacus calculation there. There are three types of basis set in Abacus, the plane-wave (PW), the linear-combinatio of atomic orbitals (LCAO), and the LCAO-in-PW. With the LCAO basis set, TB2J can directly take the output and compute the exchange parameters. In this tutorial we will take this approach. 

For the other type of basis set, the Wannier90 interace can be used instead. 

 In this calculation, we set the suffix to Fe. When the DFT calculation is finished. 

Now we can run the abacus2J.py command to calculate the exchange parameters:

```bash
abacus2J.py --path . --suffix Fe --elements Fe  --kmesh 7 7 7
```

This first read the siesta.fdf, the input file for Siesta. It then read the Hamiltonian and the overlap matrices, calculate the J with a 7x7x7 k-point grid. This allows for the calculation of exchange between spin pairs between 7x7x7 supercell. 



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

abacus2J: Using magnetic force theorem to calculate exchange parameter J from abacus Hamiltonian in the LCAO mode

options:
  -h, --help            show this help message and exit
  --path PATH           the path of the abacus calculation
  --suffix SUFFIX       the label of the abacus calculation. There should be an output directory called OUT.suffix
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

Non-collinear calculation
-----------------------------------------

The non-collinear calculation with TB2J-Abacus interace is currently under development. It will be available soon. 

