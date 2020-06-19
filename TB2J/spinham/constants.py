"""
Units and constants.
"""
import numpy as np

ps = 1e-15
fs = 1e-12

bohr_mag = 9.27400995e-24
gyromagnetic_ratio = 1.76e11
mu_B = bohr_mag

#from scipy.constants import (Boltzmann, hbar, mu_0, epsilon_0,

Boltzmann, hbar, mu_0, epsilon_0, elementary_charge, pi = (
    1.38064852e-23, 1.0545718001391127e-34, 1.2566370614359173e-06,
    8.854187817620389e-12, 1.6021766208e-19, np.pi)

kb=Boltzmann

eV = elementary_charge * 1
meV = 0.001 * eV

# Pauli Matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Creation/anihination operators
creation_op = np.array([[0, 0], [1, 0]])
annihilation_op = np.array([[0, 1], [0, 0]])

number_op = np.array([[0, 0], [0, 1]])
