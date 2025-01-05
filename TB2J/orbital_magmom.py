"""
utilities to handel orbital magnetic moments
"""

import numpy as np


def complex_spherical_harmonic_to_real_spherical_harmonic(l=1):
    """
    matrix to convert the complex spherical harmonics to real spherical harmonics
    """
    MCR = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex128)
    MCR[0 + l, 0 + l] = 1.0
    for m in range(1, l + 1):
        mi = m + l
        mpi = -m + l
        MCR[mi, mi] = (-1) ** m / 2
        MCR[mi, mpi] = 1.0 / 2
        MCR[mpi, mi] = -1j * (-1) ** m / 2
        MCR[mpi, mpi] = 1j / 2

    return MCR


def test_complex_spherical_harmonic_to_real_spherical_harmonic():
    """
    test the conversion matrix
    """
    l = 3
    MCR = complex_spherical_harmonic_to_real_spherical_harmonic(l)
    # print(MCR*np.sqrt(2))
    print(MCR * 2)


if __name__ == "__main__":
    test_complex_spherical_harmonic_to_real_spherical_harmonic()
