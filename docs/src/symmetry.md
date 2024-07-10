### Symmetrization of the exchange parameters 

The exchange parameters obtained from the output of TB2J may not preserve the symmetry of the crystal. This can be attributed to two reasons:

1. The magnetic state breaks the symmetry of the crystal. As TB2J perturbs this magnetic state, the J values should reflect the symmetry of the magnetic state.

2. Numerical noise can be present in the calculation, either during the DFT stage, the Wannierization procedure, or the post-processing within TB2J. This numerical noise should ideally be very small. If it is not, the calculation procedure should be optimized. 

In some cases, it is necessary for the exchange parameters to strictly adhere to the symmetry of the crystal structure. To achieve this, you can use the script `TB2J_symmetrize.py` to symmetrize the exchange parameters.

:Warning: Please note that the current version of the script only considers the symmetry of the crystal and does not take the magnetic moment into account.

:Warning: Additionally, only the isotropic exchange is symmetrized in this version. The symmetrization of the DMI and anisotropic exchange will be implemented in a future version. 

The `TB2J_symmetrize.py` script utilizes the symmetry of the crystal to identify symmetrically equivalent atom pairs and the corresponding symmetry operators. It then averages the J values over these symmetrically equivalent atom pairs.

Here is the usage information for the script:

```
usage: TB2J_symmetrize.py [-h] [-i INPATH] [-o OUTPATH] [-s SYMPREC]

Symmetrize exchange parameters. Currently, it takes the crystal symmetry into account and not the magnetic moment.
Also, only the isotropic exchange is symmetrized in this version. The symmetrization of the DMI and anisotropic exchange will be implemented in a future version. 

options:
  -h, --help            show this help message and exit
  -i INPATH, --inpath INPATH
                        input path to the exchange parameters
  -o OUTPATH, --outpath OUTPATH
                        output path to the symmetrized exchange parameters
  -s SYMPREC, --symprec SYMPREC
                        precision for symmetry detection. The default value is 1e-5 Angstrom
```

An example:

```bash
TB2J_symmetrize.py -i TB2J_results -o TB2J_symmetrized -s 1e-4
```

The script can read the data from TB2J\_results and write the symmetrized exchange parameters to a new directory TB2J\_symmetrized. The precistion for detecting the symmetry is 1e-4 angstrom. 

