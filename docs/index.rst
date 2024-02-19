.. TB2J documentation master file, created by
   sphinx-quickstart on Sun Jun 21 20:54:23 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TB2J's documentation!
================================

 TB2J is an open-source Python package for the automatic computation of magnetic interactions (including exchange and Dzyaloshinskii-Moriya) between atoms of magnetic crystals from density functional Hamiltonians based on Wannierfunctions or linear combinations of atomic orbitals.  The program is based on Green’s function method with the local rigid spin rotation treated as a perturbation.  As input, the package uses the output of either Wannier90, whichis interfaced with many density functional theory packages, or of codes based on localized orbitals (SIESTA, OpenMX and Abacus).  A minimal user-input is needed, which allows for easy integration into high-throughput workflows.

 The TB2J project is initialized in the PhyTheMa and Nanomat teams in the University of Liege.  

 The source code can be found at `https://github.com/mailhexu/TB2J <https://github.com/mailhexu/TB2J>`_.

 For questions please use the online forum at `<https://groups.google.com/g/tb2j>`_, or send email to `<tb2j@googlegroup.com>`_.

 More TB2J examples with full DFT/Wannier data can be found at `https://github.com/mailhexu/TB2J_examples <https://github.com/mailhexu/TB2J_examples>`_ .

 There are video tutorials in TB2J channel on youtube: `<https://www.youtube.com/channel/UCPbmKE10Wz3orbo4x-g0c9A>`_ .


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   src/install.rst
   src/convention.rst
   src/tutorial.rst
   src/applications.rst
   src/extend.rst
   src/roadmap.md
   src/ecosystem.md
   src/faq.rst
   src/Contributors.rst
   src/references
   src/development.md
   src/ReleaseNotes.md


Indices and tables
==================

* :ref:`genindex`

