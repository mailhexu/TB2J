Installation
============


The easiest way to install TB2J is to use pip:

::

   pip install TB2J

You can also download TB2J from the github page, and install with

::

   python setup.py install

The `--user` option will help if there are permission problems.

It is suggested that it being installed within a virtual environment
using e.g. pyenv, uv or conda.

You can install optional dependencies for specific interfaces or all of them:

- For the Siesta interface:

::

    pip install TB2J[siesta]

- For the lawaf interface:

::

    pip install TB2J[lawaf]

- To install all optional dependencies:


::

    pip install TB2J[all]




Dependencies
------------
TB2J is a python package which requires python version higher than 3.8 to work.
It depends on the following packages.

-  numpy<2.0
-  scipy
-  matplotlib
-  ase>=3.19
-  tqdm
-  pathos
-  packaging>=20.0
-  HamiltonIO>=0.2.4
-  pre-commit
-  sympair>0.1.0
-  tomli>=2.0.0
-  tomli-w>=1.0.0


If you use pip to install, they will be automatically installed, so there is no need to 
install them manually before installing TB2J. 

There are some optional dependencies, which you need to install if needed.

-  sisl>0.10.0 (optional) for Siesta interface
-  netcdf4 (optional) for Siesta interface
-  lawaf==0.2.3 (optional) for lawaf interface
-  GPAW (optional) For gpaw interface (not yet fully operational).


By default, TB2J only forces the non-optional dependencies to be
installed automatically. The `sisl` and `netcdf4` packages, which are
used to read the Hamiltonian from the Siesta output, are optional and
can be installed with the `[siesta]` extra. For example:

::

    pip install TB2J[siesta]

installed automatically. The sisl package (with netcdf4) which is used to read the
Hamiltonian from the Siesta or OpenMX output is needed, which can also
be installed with pip. The GPAW-TB2J interface is through python
directly, which of course requires the gpaw python package. The sisl and
gpaw python package can be installed via pip, too. For example:

::

    pip3 install sisl netcdf4


How to install in a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is recommended to install TB2J in a virtual environment (venv), which is like a
world parallel to the main Python environment where other packages 
are installed. With this, conflicts between library versions can be avoided. 
For example, you may have other packages that only work with an old version of numpy, 
whereas TB2J requires a newer version. Within this venv, you can have the new version 
without worrying about conflicts. 

One way to build a python venv is to use `venv <https://docs.python.org/3/library/venv.html>`_  library built in python. We can make a new 
python virtual environment named TB2J like this:

::

    python3 -m venv <your path>/TB2J

where you can replace <your path> to the path where you want to put the files for the venv. 

Then you can activate the venv by using

::

    source <your path>/TB2J/bin/activate

The within this venv you can install the python packages. 
And this venv should be activated when you use TB2J. 

There are other ways to build virtual environments, for example, with `conda <https://docs.conda.io/>`_ .
