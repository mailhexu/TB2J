### Symmetrization of the exchange parameters 

The exchange parameters from the output of TB2J do not necessarily preserve the symmetry of the crystal.
The reasons are:
 1. The magnetic state breaks the symmetry of the crystal. As TB2J perturb this magnetic state, the J values should has the symmetry of the magnetic state.

 2. There's numerical noise in the calculation, either at the DFT stage, or the Wannierization procedure, or during the post-processing procedure within TB2J. This numerical noise 
 should be very small. If it is not, the calculation procedure should be optimized. 


Sometimes, the exchange parameters should strictly have the symmetry of the crystal strucuture. We can use the script `TB2J_symmetrize.py` to symmetrize the exchange parameters.

:Warning: In the current version, the script symmetrizes the exchange parameters by taking the symmetry of the crystal into account. It does not take the magnetic moment into account.

:Warning: Only the isotropic exchange is symmetrized in this version. The symmetrization of the DMI and anisotropic exchange will be implemented in a later version. 

This script symmetrizes the exchange parameters by taking the symmetry of the crystal into account.

```
usage: TB2J_symmetrize.py [-h] [-i INPATH] [-o OUTPATH] [-s SYMPREC]

Symmetrize exchange parameters. Currently, it take the crystal symmetry into account and not the magnetic moment into account.
Also, only the isotropic exchange is symmetrized in this version. The symmetrization of the DMI and anisotropic exchange will be implemented in a later version. 


options:
  -h, --help            show this help message and exit
  -i INPATH, --inpath INPATH
                        input path to the exchange parameters
  -o OUTPATH, --outpath OUTPATH
                        output path to the symmetrized exchange parameters
  -s SYMPREC, --symprec SYMPREC
                        precision for symmetry detection. default is 1e-5 Angstrom
```

It use the crystal symmetry to find the symmetrically equivalent atom pairs, and the symmetry operator between them. Then the J values are averaged over the symmetrically equivalent atom pairs.
It can read the data from TB2J\_results, and write the symmetrized exchange parameters to a new directory. 
