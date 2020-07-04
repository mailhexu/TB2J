.. TB2J documentation master file, created by
   sphinx-quickstart on Sun Jun 21 20:54:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TB2J's documentation!
================================

 TB2J is a open source Python package for the automatic computation of magnetic interactions (including exchange and Dzyaloshinskii-Moriya) between atoms of magnetic crystals from density functional Hamiltonians based on Wannierfunctions or linear combination of atomic orbitals.  The program is based on the Green’s function method with thelocal rigid spin rotation treated as a perturbation.  As input, the package uses the output of either Wannier90, whichis interfaced with many density functional theory packages, or of codes based on localised orbitals.  A minimal userinput is needed, which allows for easy integration into high-throughput workflows.
 The source code can be found at `here <https://github.com/mailhexu/TB2J>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   src/install.rst
   src/convention.rst
   src/tutorial.rst
   src/applications.rst
   src/extend.rst
   src/faq.rst
   src/Contributors.rst
   src/references
   src/ReleaseNotes.rst


Indices and tables
==================

* :ref:`genindex`
* :Ref:`modindex`
* :ref:`search`
