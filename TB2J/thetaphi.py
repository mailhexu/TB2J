import numpy as np


def theta_phi_even_spaced(n):
    """
    Return n evenly spaced theta and phi values in the ranges [0, pi] and [0, 2*pi] respectively.
    """
    phis = []
    thetas = []
    for i in range(n):
        phi = 2 * np.pi * i / n
        phis.append(phi)
        r = np.sin(np.pi * i / n)
        theta = np.arccos(1 - 2 * r)
        thetas.append(theta)
    return thetas, phis
