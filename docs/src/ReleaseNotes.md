## Release Notes
------------------------------------------------------------------------

#### Current development version (v1.0.0-alpha) 
These are the new features and changes  not yet included in the official release.

- Computing MAE and single-ion anisotropy is now possible with the ABACUS and SIESTA interfaces. 
This currently requires an non-official SIESTA branch which can seperate the spin-orbit coupling and the exchange-correlation Hamiltonian. (see this MR: https://gitlab.com/siesta-project/siesta/-/merge\_requests/309)

- The full implementation of the magnon band structure from the linear spin wave theory. (Thanks to Andres Tellez Mora and Aldo Romero!) 

- An improved method of the downfolding method which implements the ligand correction to the exchange based on the Wannier function method. This requires the updated version of LaWaF (https://github.com/mailhexu/lawaf) and the updated version of the TB2J\_downfold.py script.

- Added a new parameter (--index_magnetic_atoms) to the wann2J.py, siesta2J.py, and abacus2J.py scripts, so that a more detailed selection of the magnetic atoms than the one in --element is possible. This is useful when we need to select a subset of a specific element in a system.

- There is a major refactoring of the interface to the DFT codes. The parsing of the electron Hamiltonian from the DFT codes are now in a separate python package called HamiltonIO (github.com/mailhexu/HamiltonIO). This package is used by TB2J but is made general to be used with other packages too.


#### v0.11.0 October 10, 2024
- Allowing to symmetrize the exchange according to the crystal symmetry with the TB2J\_symmetrize.py script. Note that the spin order is not considered in this symmetrization process. 

#### v0.10.0 September 1, 2024
- Improved orbital-decomposition to the ABACUS interface. 
- Allow computing anisotropic J and DMI without three or more calculations within ABACUS and SIESTA interfaces. 


#### v0.9.0 March 22, 2024
Improved merge method for anisotropic exchange and DMI. (thanks to Andres Tellez Mora!)


#### v0.8.2 March 4, 2024
TB2J can now read the "tb.dat" file instead of the "hr.dat"+"centers.xyz" files. 

(>=0.8.2.2) Allow atom symbols+number format (e.g. Fe1, Fe2) in Wannier .win file, and in the --magnetic\_elements option. ([issue46](https://github.com/mailhexu/TB2J/issues/46#issue-2167665710))
Allow synthetic atom in siesta (>=0.8.2.4).
Print actual emin in non-collinear mode.  (0.8.2.5)
Reduce memory usage by not computing density matrix from Green's function. (0.8.2.6)

#### v0.8.1 Febrary 25, 2024
Interface with ABACUS for non-collinaer spin calculations is implemented. 


#### v0.8.0 Febrary 18, 2024
Add a new DFT code interface to ABACUS! Thanks to Zhen-Xiong Shen and Gan Jin from the ABACUS 
 team for providing the coding for parsing the ABACUS output files. 
In this version the colliear spin is implemented and the non-collinear will be soon added. 


#### v0.7.7 October 11, 2023
Added script: TB2J\_magnon\_dos.py for plotting the magnon density of states. 
See https://tb2j.readthedocs.io/en/latest/src/magnon\_band.html


#### v0.7.6 May 10, 2023
TB2J\_magnon.py now writes the band structure infomation into a json file. 
A script to read the json file and plot the band structure is in the same directory. 

#### v0.7.3.1 July 24, 2022

Improve error message for wannier functions badly localized to atomic
centers.

#### v0.7.3 June 8, 2022

The Vampire output include the DMI and anisotropic exchange. The
TB2J\_downfold.py is extended to DMI and anisotropic exchange (for early
test only).


#### v0.7.2.1 April 20, 2022

Fix compatibility issue with Python3.10

#### v0.7.2 March 01, 2022

Add TB2J\_eigen.py script to write the eigen values and eigenvectors of
the J(q) in a qpoint mesh. Remove J\' and B from the output, which are
often not useful and confusing.

#### v0.7.1 January 04, 2022

Bug fix: convention in Vampire output (tensor-\>tensorial, and a factor
of 2 added to the exchange values). Some documentation about the Vampire
format added. (Contributions from Jun Gyu Lee.)

#### v0.7.0 October 05, 2021

Allow to do orbital decompositions to isotropic/anisotropic exchange and
DMI with \--orb\_decomposition.

#### v0.6.10 September 29, 2021

Bug fix: wrong matrix alignment in orbital decomposition for atom pairs
with different species.

#### v0.6.9 September 15, 2021

Better xticks in magnon bands.


#### v0.6.8 September 15, 2021

Bug fix: downfolding with non-magnetic atoms now works. Bug fix: Error
reading from user specified structural input file.


#### v0.6.7 September 10, 2021

Use tqdm & p\_tqdm instead of progressbar. Fix progressbar in parallel
mode.


#### v0.6.6 September 1, 2021

Output the figure of J vs distance. Fix a bug when the orbitals are not
grouped by atoms.

#### v0.6.4 August 9, 2021

Documentation of orbital decomposition. More concise orbital
decompostion to the exchange with siesta2J.py. Allow to specify the
orbitals used in the decomposition in siesta2J.py \--element option.

#### v0.6.3 July 3, 2021

Revert qsolver to old version.

#### v0.6.2 May 27, 2021

Enable reading structures from wannier .win file so \--posfile is no more necessray.

:   \--posfile option can still be used.

#### v0.6.1

Change the internal order of orbitals in noncollinear Tight-binding.

A script for building docker image has been added (Nikolas Garofil).

#### v0.6.0

Add TB2J\_downfold.py script to deal with ligand spin contribution.

#### v0.5.0

Add Wannier input from banddownfolder package using
\--wannier\_type=banddownfolder. Currently only collinear calculation
supported.

#### v0.4.4 March 16, 2021

Allow parallel over k in tight binding eigen solver.

#### v0.4.3 March 11, 2021

Add Reference to TB2J paper.

#### v0.4.2 March 11, 2021

Fix a bug that the atoms scaled positions get wrapped. Fix a bug with
consecutive parallel run in python mode.

#### v0.4.1 February 2, 2021

Use a Legendre path for the integration which is more stable and
requires less poles(\--nz). Memory optimization.

#### v0.4.0 February 1, 2021

Add \--np option to specify number of cpu cores in parallel. Dependency
on pathos is added.

#### v0.3.8 December 29, 2020

Add \--output\_path option to specify the output path.

#### v0.3.6 December 7, 2020

Use Simpson\'s rule instead of Euler for integration.

#### v0.3.5 November 3, 2020

Add \--groupby option in wann2J.py to specify the order of the basis set
in the hamiltonian.

#### v0.3.3 September 12, 2020

-   Use collinear exchange calculator for siesta-collinear calculation,
    which is faster.

#### v0.3.2 September 12, 2020


    add \--use\_cache option to reduce the memory usage by storing the Hamiltonian
    
    :   and eigenvectors on disk using memory map.

#### v0.3.1 September 3, 2020

-   A bug in the sign of the magnetization along y in Wannier and OpenMX
    mode is fixed.

#### v0.3 August 31, 2020

-   A bug in calculation of anisotropic exchange is fixed.
-   add TB2J\_merge.py for merging DMI and anisotropic exchange from
    calculations with different spin orientation or structure rotation.
-   Improvement on output txt file.
-   An interface to OpenMX (TB2J\_OpenMX) is added in a separate github
    under GPLv3. at <https://github.com/mailhexu/TB2J-OpenMX>
-   Many improvement and bugfixes

#### v0.2 2020

-   Moved to github
-   DMI and anisotropic exchange
-   Magnon band structure (For FM and single magnetic specie)
-   Siesta Input
-   Documentation on readthedocs

#### v0.1 2018

-   Initial version on gitlab.abinit.org
-   Isotropic exchange
-   Wannier function as input
-   Interface with Multibinit, Tom's ASD, and Vampire
