"""
This module implements the methods to extract J, DMI, and Jani from the A matrices
"""
import numpy as np

# A0 indices: u, v, iorb, jorb, where u, v=0,1, 2, 3 (I, x, y, z)


def A0A1_to_J1(A00, A11):
    J1 = -2 * np.imag(
        A00[0, 0]
        - A11[0, 0]  # + np.sum(np.diag(A00)) - np.sum(np.diag(A11)))
        + A00[1, 1]
        + A00[2, 2]
        + A00[3, 3]
        - A11[1, 1]
        - A11[2, 2]
        - A11[3, 3]
    )


def A0A1_to_DMI1(A00, A01, A10, A11):
    D = -4 * np.array((np.real(A01[0, i] - A10[0, i]) for i in range(3)))
    return D


def A0A1_to_Jani1(A00, A11):
    Jani = -4 * np.array(np.imag(A11[1:, 1:] - A00[1:, 1:]))
    return Jani


def A0A1_to_J2(A00, A11):
    J2 = -2 * np.imag(
        A00[0, 0]
        + A11[0, 0]  # + np.sum(np.diag(A00)) - np.sum(np.diag(A11)))
        - A00[1, 1]
        - A00[2, 2]
        - A00[3, 3]
        - A11[1, 1]
        - A11[2, 2]
        - A11[3, 3]
    )


def A0A1_to_DMI2(A01, A10):
    D = -4 * np.array((np.real(A01[0, i] + A10[0, i]) for i in range(3)))


def A0A1_to_Jani2(A00, A11):
    Jani = -4 * np.array(np.imag(A11[1:, 1:] + A00[1:, 1:]))
    return Jani


def test():
    A = np.ones((2, 2, 3), dtype=int)
    print(A)
    print(np.trace(A))


test()
