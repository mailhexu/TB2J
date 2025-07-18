# TB2J Product Overview

TB2J is a Python package for calculating magnetic interaction parameters in Heisenberg models from DFT (Density Functional Theory) calculations. It uses the magnetic force theorem with local rigid spin rotation as a perturbation in the Green's function method.

## Key Features
- Calculates Heisenberg model parameters: isotropic exchange, anisotropic exchange, Dzyaloshinskii-Moriya interaction
- Supports multiple DFT codes via Wannier90: Abinit, Quantum Espresso, Siesta, VASP
- Supports numerical orbital codes: Siesta, OpenMX, ABACUS
- Calculates magnon band structures from Heisenberg Hamiltonians
- Generates input for spin dynamics/Monte Carlo codes (MULTIBINIT)
- Requires only ground state DFT calculations (no supercells needed)
- Minimal user input for black-box workflows

## Target Users
- Computational physicists and materials scientists
- Researchers working with magnetic materials and spin interactions
- Users of DFT codes who need magnetic exchange parameters