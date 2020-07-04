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

* emin, emax, and nz: During the calculation, there  is a integration :math:`\int_{emin}^{emax} d\epsilon` calculation. The emin and emax are relative values to the Fermi energy. The emax should be close to 0. It can be used to adjust the integration if the charge . The emin should be low enough so that all the electronic states that affect the magnetic interactions are integrated. The nz is the number of steps in this integration. 

* rcut: rcut is the cutoff distance between two ion pairs between which the magnetic interaction parameters are calculated. By default, all the pairs inside the supercell defined by the kmesh 

* exclude_orbs: the indeces of orbitals, whose contribution will not be counted in the magnetic interaction. It is a list of integers. The indices are zero based.
