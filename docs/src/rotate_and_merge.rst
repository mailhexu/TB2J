.. _amp-lable:

Averaging multiple parameters
===============================

As discussed in the previous section, the :math:`z` component of the DMI, and the :math:`xz`, :math:`yz`, :math:`zz`, :math:`zx`, :math:`zy`, components of the anisotropic exchanges are non-physical, and an :math:`xyz` average is needed to get the full set of magnetic interaction parameters if the spins are all along :math:`z`. In this case, scripts to rotate the structure and merge the results are provided, they are named TB2J\_rotate.py and TB2J\_merge.py. The TB2J\_rotate.py reads the structure file and generates three files containing the :math:`z\rightarrow x`, :math:`z\rightarrow y`,and the non-rotated structures. The output files are named atoms\_x, atoms\_y, atoms\_z. A large number of output file formats is supported thanks to the ASE library and the format of the output structure files is provided using the --format parameter. An example for using the rotate file is:

.. code-block:: shell

   TB2J_rotate.py BiFeO3.vasp --ftype vasp


.. note::
    
    Some file format does not include the cell orientation, e.g. the cif file. They should not be used as the output format.
    

The user has to perform DFT single point energy calculations for these three structures in different directories, keeping the spins along the $z$ direction, and run TB2J on each of them. After producing the TB2J results for the three rotated structures, we can merge the DMI results with the following command by providing the paths to the TB2J results of the three cases:

::

   TB2J_merge.py BiFeO3_x BiFeO3_y BiFeO3_z --type structure


Note that the whole structure are rotated w.r.t. the laboratory axis but not to the cell axis. Therefore, the k-points should not be changed in both the DFT calculation and the TB2J calculation. 

A new TB2J\_results directory is then made which contains the merged final results. 

Another method is to do the DFT calculation with spins along the :math:`x`, :math:`y` and :math:`z` axis, respectively, and then merge the result with:

::

   TB2J_merge.py BiFeO3_x BiFeO3_y BiFeO3_z --type spin

