Writting eigen values and eigenvectors of J(q)
================================================

In this section we show how to write the eigen values and eigen vectors for a given q-point mesh.
With this information, we can estimate the lowest energy spin configuration in the supercells conmensurate to the q-point mesh.

There is a script within the TB2J package: TB2J_eigen.py, which can be
write the eigen value and eigen vectors. The command should be run under the TB2J_results directory.

We can show its usage by:

::

   TB2J_eigen.py --help
   
   TB2J version 0.7.1.1
   Copyright (C) 2018-2020  TB2J group.
   This software is distributed with the 2-Clause BSD License, without any warranty. For more details, see the LICENSE file delivered with this software.
   
   
   usage: TB2J_eigen.py [-h] [--path PATH] [--qmesh [QMESH ...]] [--gamma] [--output_fname OUTPUT_FNAME]
   
   TB2J_eigen.py: Write the eigen values and eigen vectors to file.
   
   optional arguments:
     -h, --help            show this help message and exit
     --path PATH           The path of the TB2J_results file
     --qmesh [QMESH ...]   qmesh in the format of kx ky kz. Monkhorst pack or Gamma-centered.
     --gamma               whether shift the qpoint grid to Gamma-centered. Default: False
     --output_fname OUTPUT_FNAME
                           The file name of the output. Default: eigenJq.txt
   


