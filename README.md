[![Python application](https://github.com/mailhexu/TB2J/actions/workflows/python-app.yml/badge.svg)](https://github.com/mailhexu/TB2J/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/tb2j/badge/?version=latest)](https://tb2j.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://app.travis-ci.com/mailhexu/TB2J.svg?branch=master)](https://app.travis-ci.com/mailhexu/TB2J)
[![Downloads](https://pepy.tech/badge/tb2j)](https://pepy.tech/project/tb2j)

## Description

TB2J is a open source python package for calculating the magnetic interaction parameters in Heisenberg models from DFT. It use the magnetic force theorem and take the local rigid spin rotation as a perturbation in the Green's function method. 

The TB2J project is initialized in the PhyTheMa and Nanomat teams in the University of Liege.

The features include:
 - Calculates  parameters in Heisenberg model, including isotropic exchange, anisotropic exchange, Dyzanoshinskii-Moriya interaction.
 - Can use the input from many DFT codes with Wannier90, e.g. Abinit, Quantum Espresso, Siesta, VASP, etc.
 - Can use input from DFT codes with numerical orbitals from Siesta, OpenMX and ABACUS.
 - Calculate magnon band structure from the Heisenberg Hamiltonian.
 - Generate input for spin dynamics/Monte Carlo codes MULTIBINIT.
 - Require only ground state DFT calculation.
 - No need for supercells.
 - Calculate magnetic interaction up to large distance. 
 - Minimal user input, which allows for a black-box like experience and automatic workflows.
 - Versatile API on both the input (DFT Hamiltonian) and the output (Heisenberg model) sides.

For more information, see the documentation on
 <https://tb2j.readthedocs.io/en/latest/>

## Dependencies
* python (tested for ver 3.6)
* numpy 
* scipy
* ASE (atomic simulation environment) 
* matplotlib  (optional) if you want to plot magnon band structure directly. 
* sisl (optional) for Siesta interface

## Installation
pip install TB2J

## Message:
- We welcome contributions. If you would like to add the interface to other codes, or extend the capability of TB2J, please contact us! <mailhexu_AT_gmail_DOT_com>

