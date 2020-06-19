## Description

Use wannier function tight binding hamiltonian to calulate parameters of Heisenberg model. It can produce xml input for the spin dynamics in the multibinit package. 

The theoretical background for exchanges calculation is described in 
 - Initial idea of Green's function method of calculating J.
 
   [Liechtenstein et al. J.M.M.M. 67,65-74 (1987), (aka LKAG)](https://doi.org/10.1016/0304-8853\(87\)90721-9) 

 - Isotropic exchange using Maximally localized Wannier function.

   [Korotin et al. Phys. Rev. B 91, 224405 (2015)](http://link.aps.org/doi/10.1103/PhysRevB.91.224405)

 -  Full exchange tensor.

    [Antropov et al. Physica B 237-238 (1997) 336-340](https://www.sciencedirect.com/science/article/pii/S0921452697002032) 
 
 -  Biqudratic term.

    [S. Lounis, P. H. Dederichs, Phys. Rev. B 82 180404(R) (2010)](https://doi.org/10.1103/PhysRevB.82.180404) 
 
    [Szilva et al, Phys. Rev. Lett. 111, 127204](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.127204)

## Dependencies
* python (tested for ver 3.6)
* numpy 
* scipy
* ASE (atomic simulation environment) 
* matplotlib  (optional) if you want to plot magnon band structure directly. 
* sisl (optional) for Siesta interface


## Installation
pip install TB2J

## Usage

### With Wannier90

* Construct Wannier functions with DFT codes and Wannier90 for both spin channel. Note that all the band related to the exchange should be included in the basis set. For example, in transitional metal oxides, both transition metal $d$ orbitals and oxygen $p$ orbitals should be included.

* Prepare the input files, which are:

  * A file which has the positions of the atom. 

     A large set of formats are supported (thanks to ASE). The list of supported file formats can be found https://wiki.fysik.dtu.dk/ase/ase/io/io.html . However, files which uses the symmetrized wyckoff position should not be used, since it is impossible to tell the order of the atoms in the calculation. VASP POSCAR and xyz format format are recommended for their simplicity.

  * The wannier function hamiltonian files, including

      - wannier90.up(dn)_hr.dat files: is the hamiltonian file
      - wannier90.up(dn)_centres.xyz files: the positions of the wannier centers.
      - wannier90.up(dn).win : wannier90 input files.

  * ~~An additional file called basis.txt, which provides the information of the wannier functions.~~  

    (This file is no longer mandatory in the latest version. The code will try to find which atom does the wannier function "belong" to . However, if it cannot identify, that is when some WF centers are far from atom,  this file can be used to tell the code how you want them to be assigned. However, it should be checked whether this is physical!!)

    The format of the file is:

       (Atom+index of atom )|orbital| None| None       (index of wannier function in hamiltonian)

    e.g.

    ```
    	Fe1|dz2|None|None       1
    	Fe1|dxz|None|None       2
    	Fe1|dyz|None|None       3
    	...
    	O1|px|None|None 11
    	...
    ```

    where "Fe1" means first Fe atom in the structure. These None are of no use here. The first None is used when there  are for example, two s orbitals, which would become 1 and 2 instead of None. The second None is used to describe the orientation of the axis for the orbitals if they are not the same as the DFT axis.
    The allowed values for the orbitals are:
    ​    's',
    ​    'py',
    ​    'pz',
    ​    'px',
    ​    'dxy',
    ​    'dyz',
    ​    'dz2',
    ​    'dxz',
    ​    'dx2-y2',
    ​    'fz3',
    ​    'fxz2',
    ​    'fyz2',
    ​    'fxyz',
    ​    'fz(x2-y2)',
    ​    'fx(x2-3y2)',
    ​    'fy(3x2-y2)'
    (as in Wannier90)


* Run wann2J.py to get the exchange parameters. 

  The usage of this command is:

```
usage: wann2J.py [-h] [--path PATH] [--poscar POSCAR]
                 [--prefix_spinor PREFIX_SPINOR] [--prefix_up PREFIX_UP]
                 [--prefix_down PREFIX_DOWN]
                 [--elements [ELEMENTS [ELEMENTS ...]]]
                 [--rrange [RRANGE [RRANGE ...]]] [--efermi EFERMI]
                 [--kmesh [KMESH [KMESH ...]]] [--emin EMIN] [--emax EMAX]
                 [--height HEIGHT] [--nz1 NZ1] [--nz2 NZ2] [--nz3 NZ3]
                 [--inpexc INPEXC] [--cutoff CUTOFF]
                 [--exclude_orbs EXCLUDE_ORBS [EXCLUDE_ORBS ...]] [--plot]
                 [--description DESCRIPTION] [--fname FNAME] [--show]
                 [--spinor]

wann2J: Using magnetic force theorem to calculate exchange parameter J from
wannier functions

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to the wannier files
  --poscar POSCAR       name of the position file
  --prefix_spinor PREFIX_SPINOR
                        prefix to the spinor wannier files
  --prefix_up PREFIX_UP
                        prefix to the spin up wannier files
  --prefix_down PREFIX_DOWN
                        prefix to the spin down wannier files
  --elements [ELEMENTS [ELEMENTS ...]]
                        elements to be considered in Heisenberg model
  --rrange [RRANGE [RRANGE ...]]
                        range of R
  --efermi EFERMI       Fermi energy in eV
  --kmesh [KMESH [KMESH ...]]
                        kmesh in the format of kx ky kz
  --emin EMIN           energy minimum below efermi, default -12 eV
  --emax EMAX           energy maximum above efermi, default 0.0 eV
  --height HEIGHT       energy contour, a small number (often between 0.1 to
                        0.5, default 0.2)
  --nz1 NZ1             number of steps 1, default: 50
  --nz2 NZ2             number of steps 2, default: 200
  --nz3 NZ3             number of steps 3, default: 50
  --cutoff CUTOFF       The minimum of J amplitude to write, (in eV), default
                        is 1e-5 eV
  --exclude_orbs EXCLUDE_ORBS [EXCLUDE_ORBS ...]
                        the indices of wannier functions to be excluded from
                        magnetic site. counting start from 0
  --plot                Plot magnon energy band instead of calculating J. Path
                        will be automatically selected.
  --description DESCRIPTION
                        add description of the calculatiion to the xml file.
                        Essential information, like the xc functional, U
                        values, magnetic state should be given.
  --fname FNAME         exchange xml file name. default: exchange.xml
  --show                whether to show magnon band structure.
  --spinor              Whether to use spinor wannier function.

  ```

  Only the --elements and --efermi are mandatory. The default values of  other options are often fine.
  For the use of spinor wannier function, use the --spinor option and specify the --prefix_spinor.
  For the use of spin_up and spin_down wannier function, do not use --spinor, and specify --prefix_up, and --prefix_down


### With Siesta

For calculating the  parameters  of  the  Heisenberg  Hamiltonian  the localized DFT Hamiltonian  and  the  overlap  matrix  need  to  be  saved. For example,   we can  use the  options  “CDF.Save=True”,  “SaveHS=True”,  and“Write.DMHS.Netcdf=True” in Siesta to enable thesaving of these matrices.

Only a  minimal set of parameters is needed for Siesta: the filename of the input for the siesta calculation and defining the magnetic atomspecies; Most of the other information needed, including whether the calculation has SOC enabled and  the Fermi energy, are readily provided in the Siesta results.  We can get the exchange parameters, with one simple line of command,

```
siesta2J.py --input -fname=’siesta.fdf’ --element Fe 
```

### Output files
By running wann2J.py, a directory with the name TB2J_results will be generated. 
The following output files will be inside the directory:

- exchange.out: A human readable output file, which summarizes the result.
- exchange.xml: A file for the input for the spin dynamics in Multibinit. see [Multibinit turorial][https://docs.abinit.org/tutorial/lattice_model/] for more details. (Note: The spin dynamics tutorial is not yet online as in Oct, 2018. It will be released with the next version of Abinit, likely at the end of 2018.)
- exchange.ucf & exchange.exch: Input files for Tom's ASD code. 

In the exchange.out file, there are three sections:

* Cell section: the cell parameter 
* Atoms section: The positions, charge and magnetic moments of the atoms.  Here the charge and magnetic moments are only summed over the wannier functions attached to them, therefore are different from the output from DFT. Also, since the integration area is different (e.g. in DFT the integration could be inside a paw sphere, or a atomic orbital.). For localized d and f orbitals, the magnetic moment should be close to DFT results. If you find large difference, there could be something wrong, either in the construction of wannier functions, or in the parameters used in TB2J (e.g. the height, or emin & emax, or kmesh).

```
  Atoms:
  (Note: charge and magmoms only count the wannier functions.)
   Symbol_number       x          y          z      w_charge   w_magmom
  Bi1               0.2413     0.2413     0.2413     2.5446     0.0163
  Bi2               4.2060     4.2060     4.2060     2.5377     0.0190
  Fe1               2.0165     2.0165     2.0165     5.7418     4.1535
  Fe2               5.9812     5.9812     5.9812     5.7412     -4.1523
  O1                5.5238     2.1558     3.9388     5.3062     -0.0439
  O2                6.0903     5.5389     3.9539     5.3064     0.0473
  O3                3.9388     5.5238     2.1558     5.3063     -0.0439
  O4                3.9539     6.0903     5.5389     5.3064     0.0472
  O5                2.1558     3.9388     5.5238     5.3062     -0.0438
  O6                5.5389     3.9539     6.0903     5.3065     0.0471
  
```

* Exchange section: 

  As below, the exchange section has item defined as:

In the non-spinor case, the output will be
  - i , j: the label of the site. Note that only magnetic sites are indexed, i.e. they are not the indices of the atoms in the structure. This is to be consistent with the definition in exchange.xml. The corresponding atoms (e.g. Fe1) are in the braket. 

  -  R: The index of cell. 
  -  J: J  (as in Heisenberg Hamiltonian $H=-\sum_{i,j} J_{ij} \vec{S_i}\cdot\vec{S_j}$). Note the - sign and that there is no (i<j)!
  - vec: The vector between two atoms, unit is angstrom
  - distance: the norm of the vector, in angstrom.
  - orbital contributions: J can be written as the sum of the contribution from orbitals. Here the contribution of orbitals of atom i are listed. The order is the same as in the wannier functions. 

  ```
  Exchange:
    i           j           R           J(meV)          vector          distance(A)
   1(Fe2  )    0(Fe1  ) (  1,   0,   1) -22.7243 ( 0.015,  3.934,  0.015)  3.934
   0(Fe1  )    1(Fe2  ) ( -1,   0,  -1) -22.7243 (-0.015, -3.934, -0.015)  3.934
   1(Fe2  )    0(Fe1  ) (  1,   0,   0) -22.7168 (-3.934, -0.015, -0.015)  3.934
  ....
  
  ```

  The items are sorted by the amplitude of the J. Values smaller than 0.01 meV are discarded. 



## Examples

The examles are in the test folder.

## Tutorial
A tutorial from how to generate wannier function from DFT to calculation of J is in the tutorial directory. 


## Information

The code is not in production at this stage.  We encourage the testing of this code.

If you would like to be informed about the development of this code, please contact me (mailhexu(AT)gmail(DOT)com).  Serious bugs, or new functionality will be informed through email.



 

