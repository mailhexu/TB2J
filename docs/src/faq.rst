Frequently asked questions.
==========================================


 How to cite TB2J?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 See the reference section.

How can I ask questions or report bugs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We recommend posting the questions to the TB2J forum, `https://groups.google.com/g/tb2j<https://groups.google.com/g/tb2j>`_ . Equivalently you can send emails to  `tb2j@googlegroups.com<tb2j@googlegroups.com>`_ . Another option is the discussion page on github: `https://github.com/mailhexu/TB2J/discussions<https://github.com/mailhexu/TB2J/discussions``_ . Before doing so, please read the documents and first try to find out if things are already there.

For reporting bugs, you can also open a issue on `https://github.com/mailhexu/TB2J/issues<https://github.com/mailhexu/TB2J/issues>`_ .

If you meet with a bug, please first try to upgrade to the latest version to see if it is still there. And when reporting a bug, please post the inputs, the TB2J command, and the version of TB2J being used if possible. Should these files be kept secret, try to reproduce the bug in a simple system.

It is highly recommended to sign with your real name and affiliation. We appreciate the opportunity for us to get to know the community.

Any kind of feedback will help us to make improvement. Don't hesitate to get in contact!

Is it reasonable to do the DFT calculation in a magnetic non-ground state?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


What quantities should I look into for validating the Wannier functions?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  
How can I improve the Wannierization?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

How can I speedup the calculation?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TB2J can be used in parallel mode, with the --np option to specify the number of processes to be used. Note that it can
only use one computer node. So if you're using a job management system like slurm or pbs, you need to explicitly specify that
the resources should be allocated in a single node.

Is is possible to reduce the memory usage?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TB2J need the wave functions for all k-points so it can use a lot of memory for large systems. In parallel mode, this is more of a issue as each process
will store one copy of it. However, you can use the --use-cache option so that the wave functions are saved in a shared file by all the processes. 


My exchange parameters are different from the results from total energy methods. What are the possible reasons?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The results seems to contradict the experimental results. Why?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are many possible reasons for the discrepancies between experimental and TB2J results. Here is a incomplete list
First There are many assumption made throughout the calculations, which could be unrealistic or unsuitable for the specific material. 
- In assumptions inherited from DFT calculations, like the Born-Oppenheimer approximation, the mean-field approximation, the LDA/GGA/metaGGA/etc. 
- In the Heisenberg model:
  * The Heisenberg model is a oversimplified model. There could be terms which are import for the specific system but are not considered in the model.
    For example, the higher order exchange interactions, and the interaction between the spin and other degrees of freedom (e.g. lattice vibration, charge transfer). 
- In the magnetic force theorem (MFT):
  * The MFT is only exact as a perturbation to the ground state, which is accurate for the related properties, eg. the magnon dispersion curve. But for properties related to large deviation from the ground state, e.g. the critical temperature, the exchange parameters from the MFT might not be a good approximation (though in many material it is surprisingly good). 
  * The rigid spin rotation assumption is invalid, for example, when the spins are strongly delocalized. 



Does TB2J work with 2D structures or molecules?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Yes.

