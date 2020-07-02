Installation
============

Dependencies
------------

-  python (>=3.6)
-  numpy
-  scipy
-  ASE (atomic simulation environment)
-  matplotlib
-  sisl (optional) for Siesta interface
-  GPAW (optional) For gpaw interface

How to install
--------------

The most easy way to install TB2J is to use pip:

::

   pip install TB2J

You can also download TB2J from the github page, and install with

::

   python setup.py install

The –user option will help if there is permission problem.

It is suggested that it being installed within a virtual environment
using e.g. pyenv or conda.

By default, TB2J only forces the non-optional dependencies to be
installed automatically. The sisl package which is used to read the
Hamiltonian from the Siesta or OpenMX output is needed, which can also
be installed with pip. The GPAW-TB2J interface is through python
directly, which of course requires the gpaw python package. The sisl and
gpaw python package can be installed via pip, too.
