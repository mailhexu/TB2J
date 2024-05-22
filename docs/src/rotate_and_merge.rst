.. _amp-lable:

Averaging multiple parameters
===============================

When the spins of sites :math:`i` and :math:`j` are along the directions :math:`\hat{\mathbf{m}}_i` and :math:`\hat{\mathbf{m}}_j`, respectively, the components of :math:`\mathbf{J}^{ani}_{ij}` and :math:`\mathbf{D}_{ij}` along those directions will be unphysical. In other words, if :math:`\hat{\mathbf{u}}` is a unit vector orthogonal to both :math:`\hat{\mathbf{m}}_i` and :math:`\hat{\mathbf{m}}_j`, we can only obtain the projections :math:`\hat{\mathbf{u}}^T \mathbf{J}^{ani}_{ij} \hat{\mathbf{u}}` and :math:`\hat{\mathbf{u}}^T \mathbf{D}_{ij} \hat{\mathbf{u}}`. To obtain the other components, we need to rotate the spins or alternatively, rotate the structure while keeping the spin directions fixed. This method takes the approximation that the electronic strucuture is only slightly affected by the rotation of the spins, which will only lead to ne
gligible relative differences in the magnetic interaction parameters. This is a good approximation for systems with weak SOC (i.e. the SOC is much weaker than the exchange-correlation). But it can fail for systems with strong SOC. 

Notice that for collinear systems, there will be two orthonormal vectors :math:`\hat{\mathbf{u}}` and :math:`\hat{\mathbf{v}}` that are also orthogonal to :math:`\hat{\mathbf{m}}_i` and :math:`\hat{\mathbf{m}}_j`. 

The projection for :math:`\mathbf{J}^{ani}_{ij}` can be written as

:math:`\hat{\mathbf{u}}^T \mathbf{J}^{ani}_{ij} \hat{\mathbf{u}} = \hat{J}_{ij}^{xx} u_x^2 + \hat{J}_{ij}^{yy} u_y^2 + \hat{J}_{ij}^{zz} u_z^2 + 2\hat{J}_{ij}^{xy} u_x u_y + 2\hat{J}_{ij}^{yz} u_y u_z + 2\hat{J}_{ij}^{zx} u_z u_x,`

where we considered :math:`\mathbf{J}^{ani}_{ij}` to be symmetric. This equation gives us a way of reconstructing :math:`\mathbf{J}^{ani}_{ij}` by performing TB2J calculations on rotated spin configurations. If we perform six calculations such that :math:`\hat{\mathbf{u}}` lies along six different directions, we obtain six linear equations that can be solved for the six independent components of :math:`\mathbf{J}^{ani}_{ij}`. We can also reconstruct the :math:`\mathbf{D}_{ij}` tensor in a similar way. Moreover, if the system is collinear then only three different calculations are needed. Note that when the system is only slightly noncollinear, e.g. AFM systems with weak-ferromagnetism due to spin canting, we can still treat it as collinear and three calculations is still enough.

While rotating the spins can be done in the DFT calculation, the feature is sometimes not available for some DFT codes. In this case, an alternative is to rotate the structures while keeping the spins fixed. 
To account for this, TB2J provides scripts to rotate the structure named TB2J\_rotate.py. The TB2J\_rotate.py reads the structue file and generates three(six) files containing the rotated structures whenever the system is collinear (non-collinear). The --noncollinear parameters is used to specify whether the system is noncollinear. The output files are named atoms\_i (i = 0, ..., 5), where atoms\_0 contains the unrotated structure. A large number of file formats is supported thanks to the ASE library and the output structure files format is provided through the --ftype parameter. An example for using the rotate file with a collinear system is:

.. code-block:: shell

   TB2J_rotate.py BiFeO3.vasp --ftype vasp

If te system is noncollinear, then we run the following instead:

.. code-block:: shell

   TB2J_rotate.py BiFeO3.vasp --ftype vasp --noncollinear   

The user has to perform DFT single point energy calculations for the same structure with different spin orientations, or the generated structures in different directories, keeping the spins along the $z$ direction, and run TB2J on each of them. 

After producing the TB2J results for the rotated structures, we can merge the results with the following command by providing the paths to the TB2J results of the three cases:

::

   TB2J_merge.py BiFeO3_1 BiFeO3_2 BiFeO3_0

Here the last directory will be taken as the reference structure. Note that the whole structure are rotated w.r.t. the laboratory axis but not to the cell axis. Therefore, the k-points should not be changed in both the DFT calculation and the TB2J calculation. 


A new TB2J\_results directory is then made which contains the merged final results. 



Another method is to do the DFT calculation with spins rotated globally. That is they are rotated with respect to an axis, but their relative orientations remain the same. This can be specified in the initial magnetic moments from a DFT calculation. For calculations done with SIESTA, there is a script that rotates the density matrix file along different directions. We can then use these density matrix files to run single point calculations to obtain the required rotated magnetic configurations. An example is:

::

   TB2J_rotateDM.py --fdf_fname /location/of/the/siesta/*.fdf/file

As in the previous case, we can use the --noncollinear parameter to generate more configurations. The merging process is performed in the same way.


  
