### Ecosystem



#### Input to TB2J

 TB2J starts from electron tight-binding-like Hamiltonian with localized basis set. Currently, this includes the Wannier-function Hamiltonian built with Wannier90, and the pseudo-atomic-orbital (PAO) based codes (SIESTA and OpenMX). 

- [WANNIER90](https://wannier.org/):  Wannier90 is an open-source code (released under [GPLv2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.html)) for generating maximally-localized Wannier functions and using them to compute advanced electronic properties of materials with high efficiency and accuracy. Many [electronic structure codes](https://wannier.org/download/#es-codes) have an interface to Wannier90, including [Quantum ESPRESSO](http://www.quantum-espresso.org/), [Abinit](http://www.abinit.org/), [VASP](https://www.vasp.at/), [Siesta](http://www.icmab.es/siesta), [Wien2k](http://www.wien2k.at/), [Fleur](http://www.flapw.de/), [OpenMX](http://www.openmx-square.org/) and [GPAW](https://wiki.fysik.dtu.dk/gpaw/). 
- [SIESTA](https://siesta-project.org/siesta/): SIESTA is both a method and its computer program implementation, to perform efficient electronic structure calculations and ab initio molecular dynamics simulations of molecules and solids. SIESTA's efficiency stems from the use of a basis set of strictly-localized atomic orbitals. A very important feature of the code is that its accuracy and cost can be tuned in a wide range, from quick exploratory calculations to highly accurate simulations matching the quality of other approaches, such as plane-wave methods. The parsing of the SIESTA output files is through [sisl](https://github.com/zerothi/sisl). 
- [OpenMX](https://www.openmx-square.org/): OpenMX (Open source package for Material eXplorer) is a software package for nano-scale material simulations based on density functional theories (DFT), norm-conserving pseudopotentials, and pseudo-atomic localized basis functions. The methods and algorithms used in OpenMX and their implementation are carefully designed for the realization of large-scale *ab initio* electronic structure calculations on parallel computers based on the MPI or MPI/OpenMP hybrid parallelism. The TB2J-OpenMX interface is packaged in [TB2J-OpenMX](https://github.com/mailhexu/TB2J-OpenMX) under the GPLv3 license. 

#### Spin dynamics code interfaced with TB2J

TB2J can provide the input files containing the parameters for Heisenberg models to be used in spin-dynamics code.  Currently, TB2J is interfaced to MULTIBINIT and Vampire. 

* [MULTIBINIT](https://www.abinit.org/): MULTIBINIT is a framework for the "second-principles" method. It is deployed in the [ABINIT](https://www.abinit.org/) package. It aims at automatic mapping first-principles model to effective models which reproduce the first-principles precision but with much lower computational cost. Dynamics with multiple degrees of freedom, including lattice distortion, spin, and electron can be included in the model.  The spin part of MULTIBINIT implements the atomistic spin dynamics from Heisenberg model and Landau-Lifshitz-Gilbert equations.  TB2J was initially built to provide the parameters for spin model in MULTIBINIT. The documenation of spin dynamics can be found [here](https://docs.abinit.org/tutorial/spin_model/).

* [Vampire](https://vampire.york.ac.uk/):Vampire is a high performance general purpose code for the atomistic simulation of magnetic materials. Using a variety of common simulation methods it can calculate the equilibrium and dynamic magnetic properties of a wide variety of magnetic materials and phenomena, including ferro, ferri and antiferromagnets, core-shell nanoparticles, ultrafast spin dynamics, magnetic recording media, heat assisted magnetic recording, exchange bias, magnetic multilayer films and complete devices.

#### Codes for Linear Spin Wave method and magnon band structure

* [RAD-tools](https://rad-tools.org/): RAD-tools is a python package for the spin Hamiltonian analysis (with built-in notation changes) and magnon band structure calculation. It is interfaced directly with the TB2J .txt output ("exchange.out") and can compute the magnon band structure via the [linear spin wave theory](https://rad-tools.org/en/stable/user-guide/library/magnon-dispersion.html) for **ferromagnetic**, **antiferromagnetic** and **spiral** magnetic structures. Documentation of the usage can be found on the package website: if you know python -  [use as library](https://rad-tools.org/en/stable/user-guide/module/magnons/index.html) or if you do not know python - [use console interface](https://rad-tools.org/en/stable/user-guide/scripts/rad-plot-tb2j-magnons.html).


  

#### Related software without already-built interface with TB2J

There are many other tools which can be used together with TB2J, but the interface is not yet built (or made publicly available). 

* [UppASD](https://www.physics.uu.se/forskning/materialteori/pagaende-forskning/uppasd/): The UppASD package is a simulation tool for atomistic spin dynamics at finite temperatures. The program evolves in time the equations of motion for atomic magnetic moments in a solid. The equations take the form of the Landau-Lifshitz-Gilbert (LLG) equation. For most of the applications done so far, the magnetic exchange parameters of a classical Heisenberg Hamiltonian have been used in ASD simulations. The parameters are extracted from ab-initio DFT codes.

* [Spirit](https://spirit-code.github.io/): Spirit is a modern cross-platform framework for spin dynamics. Its features includes: atomistic spin lattice Heisenberg model including also DMI and dipole-dipole, Spin Dynamics simulations obeying the Landau-Lifschitz-Gilbert equation, direct energy minimization with different solvers, Minimum Energy Path calculations for transitions between different spin configurations, using the GNEB method. It provides a python package making complex simulation workflows easy, desktop UI with powerful, live 3D visualisations and direct control of most system parameters, and Modular backends including parallelization on GPU (CUDA) and CPU (OpenMP).

* [SpinW](https://spinw.org/): SpinW is a MATLAB library that can plot and numerically simulate magnetic structures and excitations of given spin Hamiltonian using classical Monte Carlo simulation and linear spin wave theory.

(If you know other software that can be used together with TB2J, or if you can help with interfacing with these codes,  please contact us.)



