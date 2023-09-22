Frequently asked questions.
===========================



How can I ask questions or report bugs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We recommend posting the questions to the TB2J forum, https://groups.google.com/g/tb2j . Equivalently you can send emails to  tb2j@googlegroups.com . Another option is the discussion page on github: https://github.com/mailhexu/TB2J/discussions . Before doing so, please read the documents and first try to find out if things are already there.

For reporting bugs, you can also open a issue on https://github.com/mailhexu/TB2J/issues .

If you meet with a bug, please first try to upgrade to the latest version to see if it is still there. And when reporting a bug, please post the inputs, the TB2J command, and the version of TB2J being used if possible. Should these files be kept secret, try to reproduce the bug in a simple system.

It is highly recommended to sign with your real name and affiliation. We appreciate the opportunity for us to get to know the community.

Any kind of feedback will help us to make improvement. Don't hesitate to get in contact!

Is it reasonable to do the DFT calculation in a magnetic non-ground state for the calculation of the exchange parameters?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It depends on how "Heisenberg" the material is. In a ideal Heisenberg model, the exchange parameters does not depend on the orientation of the spins. But in a real material it is only an approximation. Although it is a good approximation for many materials, there could be other cases that it fails. 

To do such computation can be very helpful when the magnetic ground state is unknown or difficult to compute with DFT, e.g. huge supercell could be needed to model some complex magnetic states. In these cases, the estimation of the exchange parameters could be useful for finding the ground state, or provide an estimation of the other magnetic properties. 


What quantities should I look into for validating the Wannier functions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, compare the band structure from the DFT results and that from the Wannier Hamiltonian. Then check the Wannier centers and Wannier spread to see if they are near the atom centers and the spread is small enough. From that you can also get some limited sense on the symmetry of the Wannier functions. Another thing to check in the collinear-spin case is the Re/Im ratio of the Wannier functions, which can be found in the Wannier90 output files. 

  
How can I improve the Wannierization?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This web page provides some nice tips on how to build high quality Wannier functions:
`<https://www.wanniertools.org/tutorials/high-quality-wfs>`

How can I speedup the calculation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TB2J can be used in parallel mode, with the --np option to specify the number of processes to be used. Note that it can
only use one computer node. So if you're using a job management system like slurm or pbs, you need to explicitly specify that
the resources should be allocated in a single node. 

Is is possible to reduce the memory usage?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TB2J need the wave functions for all k-points so it can use a lot of memory for large systems. In parallel mode, this is more of a issue as each process
will store one copy of it. However, you can use the --use-cache option so that the wave functions are saved in a shared file by all the processes. 


My exchange parameters are different from the results from total energy methods. What are the possible reasons?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The exchange parameters from TB2J and those from the total energy are not completely the same so they can be different.

* TB2J uses the magnetic force theorem and perturbation theory, which is more accurate if the spin orientation only slightly deviate from the reference state. 
    The total energy method often flip the spins to get the energies with various spin configuration, therefore is probably better at describing the interactions if the spin is more disordered. 

* The results from the total energy method depends on the model it assumes. 
    For example, for system with long-range spin ineraction, or higher order interactions, these parameters are re-normalized into the exchange parameters. Whereas TB2J does not, which makes the physically meaning more tractable. 

* The conventions should be checked when you compare the results from different sources. 
    Perhaps sometimes it is just a factor of 1/2 or whether the S is normalized to 1. 

The results seems to contradict the experimental results. Why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are many possible reasons for the discrepancies between experimental and TB2J results. Here is a incomplete list
First There are many assumption made throughout the calculations, which could be unrealistic or unsuitable for the specific material. 

* In assumptions inherited from DFT calculations.  
    The Born-Oppenheimer approximation, the mean-field approximation, the LDA/GGA/metaGGA/etc. 
* In the Heisenberg model:
    The Heisenberg model is a oversimplified model. There could be terms which are important for the specific system but are not considered in the model.
    For example, the higher order exchange interactions, and the interaction between the spin and other degrees of freedom (e.g. lattice vibration, charge transfer). 
* In the magnetic force theorem (MFT):
    * The MFT is only exact as a perturbation to the ground state, which is accurate for the related properties, eg. the magnon dispersion curve. But for properties related to large deviation from the ground state, e.g. the critical temperature, the exchange parameters from the MFT might not be a good approximation (though in many material it is surprisingly good). 
    * The rigid spin rotation assumption is invalid, for example, when the spins are strongly delocalized. 



Does TB2J work with 2D structures or molecules?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

