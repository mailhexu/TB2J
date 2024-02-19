Use TB2J with Siesta
====================

In this tutorial we will learn how to use TB2J with Siesta. First we calculate the isotropic exchange in bcc Fe. Then we calculate the isotropic exhange, anisotropic exchange and Dzyanoshinskii-Moriya interaction (DMI) parameters from a calculation with spin-orbit coupling enabled.
Before running this tutorial, please make sure that the sisl package is installed.

Collinear calculation without SOC
--------------------------------------
Let's start from the example of BCC Fe. The input files used can be found in the examples/Siesta/bccFe directory. 

First, we do a siesta self consistent calculation with the BCC Fe primitive cell of bcc Fe has only one Fe atom. In the example, we use the pseudopotential from the PseudoDojo dataset in the psml format. Note that at the moment the psml support is implemented in the master development and "Max" branches of siesta (see https://gitlab.com/siesta-project/siesta/-/wikis/Guide-to-Siesta-versions for the different versions of siesta, and https://gitlab.com/siesta-project/siesta/-/wikis/How-to-build-the-master-version-of-Siesta for how to build it.). We need to save the electronic Kohn-Sham Hamiltonian in the atomic orbital basis set with the options:

::

   SaveHS True

and this option to  use the netcdf format (if netcdf is enabled within the siesta version being used). 

::
   CDF.Save True

After that, we will have the files siesta.nc and DMHS.nc file, which contains the Hamiltonian and overlap matrix information.

Now we can run the siesta2J.py command to calculate the exchange parameters:

::

   siesta2J.py --fdf_fname siesta.fdf --elements Fe --kmesh 7 7 7

This first read the siesta.fdf, the input file for Siesta. It then read the Hamiltonian and the overlap matrices, calculate the J with a :math:`7\times 7 \times 7` k-point grid. This allows for the calculation of exchange between spin pairs between :math:`i` and :math:`j` in a :math:`7\times 7 \times 7` supercell, where :math:`i` is fixed in the center cell. Note: the kmesh is not dense enough for a practical calculation. Please check the convergence. 



Non-collinear calculation
-----------------------------------------

The anisotropic exchange and the DMI parameters can be calculated with non-collinear DFT calculation. The procedure is almost the same as in the collinear calculation except that the parameters for non-collinear calculation must be set in the Siesta input (Spin should be set to non-colinear or spin-orbit). 

