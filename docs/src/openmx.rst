Use TB2J with OpenMX
====================

In this tutorial we will learn how to use TB2J with OpenMX with the example of cubic SrMnO3. The example input files can be found in the examples directory of 

 The interface to OpenMX is distributed as a plugin to TB2J called TB2J OpenMX under the GPL license, which need to be installed separately, because code from OpenMX which is under the GPL license is used in the parser of OpenMX files.

Install TB2J-OpenMX
--------------------------------------

::

    pip install TB2J-OpenMX

running TB2J
--------------------------------------

In the DFT calculation, the ”HS.fileout on” options should be enabled, so that the Hamiltonian and the overlap matrices are written to a ”.scfout“ file. Then we can run the command openmx2J.py. The necessary input are the path of the calculation, the prefix of the OpenMX files, and the magnetic elements:

::

    openmx2J.py -- prefix openmx --elements Fe --kmesh 7 7 7


openmx2J.py then read the openmx.xyz and the openmx.scfout files from the OpenMX output, and output the results to TB2J_results.  Note: the kmesh is not dense enough for a practical calculation.


