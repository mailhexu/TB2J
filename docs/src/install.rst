Installation
============

Dependencies
------------
TB2J is a python package which requires python version higher than 3.6 to work.
It depends on the following packages.

-  numpy>1.16.5
-  scipy
-  matplotlib
-  ase>=3.19
-  tqdm>=4.42.0
-  p_tqdm
-  pathos
-  packaging


If you use pip to intall, they will be automatically installed so there is no need to 
install them manually before installing TB2J. 

There are some optional dependencies, which you need to install if needed.

-  sisl>0.10.0 (optional) for Siesta interface
-  GPAW (optional) For gpaw interface (not yet fully operational).

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
gpaw python package can be installed via pip, too. For example:

::

    pip3 install sisl


How to install in a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is recommend to install TB2J in virtual enviroment (venv), which is like a
world parallel to the main python environment where the other packages 
are installed. With this, conflictions between library versions can be avoided. 
For example, may be you have other packages only work with a old version of numpy, 
whereas TB2J requires a newer version. Then within this venv you can have the new version 
without needing to worry about the conflictions. 

One way to build a python venv is to use `venv <https://docs.python.org/3/library/venv.html>`_  library built in python. We can make a new 
python virtual environment named TB2J like this:

::

    python3 -m venv <your path>/TB2J

where you can replace <your path> to the path where you want to put the files for the venv. 

Then you can activate the venv py using

::

    source <your path>/TB2J/bin/activate

The within this venv you can install the python packages. 
And this venv should be activated when you use TB2J. 

There are other ways to build virtual environments, for example, with `conda <https://docs.conda.io/>`_ . 
