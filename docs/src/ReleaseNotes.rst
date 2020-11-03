Release Notes
=============

v0.3.5 November 3, 2020
-------------------------
Add --groupby option in wann2J.py to specify the order of the basis set in the hamiltonian.


v0.3.3 September 12, 2020
-------------------------
- Use collinear exchange calculator for siesta-collinear calculation, which is faster.

v0.3.2 September 12, 2020
-------------------------
- add --use_cache option to reduce the memory usage by storing the Hamiltonian 
    and eigenvectors on disk using memory map.


v0.3.1 September 3, 2020
-------------------------
- A bug in the sign of the magnetization along y in Wannier and OpenMX mode is fixed.


v0.3 August 31, 2020
------------------------
- A bug in calculation of anisotropic exchange is fixed.
- add TB2J_merge.py for merging DMI and anisotropic exchange from calculations 
  with different spin orientation or structure rotation.
- Improvement on output txt file.
- An interface to OpenMX (TB2J_OpenMX) is added in a separate github under GPLv3.
  at https://github.com/mailhexu/TB2J-OpenMX
- Many improvement and bugfixes


v0.2 2020
---------

-  Moved to github
-  DMI and anisotropic exchange
-  Magnon band structure (For FM and single magnetic specie)
-  Siesta Input
-  Documentation on readthedocs

v0.1 2018
---------

-  Initial version on gitlab.abinit.org
-  Isotropic exchange
-  Wannier function as input
-  Interface with Multibinit, Tom’s ASD, and Vampire
