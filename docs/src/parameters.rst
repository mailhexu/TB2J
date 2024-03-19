Parameters in calculation of magnetic interaction parameters
=============================================================

List of parameters:
-------------------------------
The list of parameter can be found using:

::

   wann2J.py --help

or

::

   siesta2J.py --help

The parameter will be explained in the following text.

* kmesh: Three integers to specify the size of a Monkhorst-Pack mesh. This is the mesh of k-points used to calculate the Green's functions. The real space supercell in which the magnetic interactions are calculated, has the same size as the k-mesh. For example, a :math:`7 \times 7 \times` k-mesh is linked with a :math:`7 \times 7 \times` supercell, the atom :math:`i` resides in the center cell, whereas the :math:`j` atom can be in all the cells in the supercell. 

* efermi: The Fermi energy in eV. For insulators, it can be inside the gap. For metals, it should be the same as in the DFT calculation. Due to the different algorithms in the integration of the density, the Fermi energy could be slightly shifted from the DFT value. 

* emin, emax, and nz: During the calculation, there  is a integration :math:`\int_{emin}^{emax} d\epsilon` calculation. The emin and emax are relative values to the Fermi energy. The emax should be 0.   The emin should be low enough so that all the electronic states that affect the magnetic interactions are integrated. This can be checked with the local density of the states. Below the emin, the spin up and down density of states should be almost identical.  The nz is the number of steps in this integration.  The emax can be used to adjust the integration if we want to simulate the effect of the charge doping. This is with the approximation that there is only a rigid shift of the band structure. However, it is recommended to dope charge within the DFT then this approximation is not needed. The emax parameter will thus be deprecated soon.

* rcut: rcut is the cutoff distance between two ion pairs between which the magnetic interaction parameters are calculated. By default, all the pairs inside the supercell defined by the kmesh 

* exclude_orbs: the indeces of orbitals, whose contribution will not be counted in the magnetic interaction. It is a list of integers. The indices are zero based.
