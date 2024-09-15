#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mycfr.py
# Time: 2017/12/10 19:29:43
"""
Continued fraction representation.
"""

import numpy as np
from scipy.linalg import eig

kb = 8.61733e-5  # Boltzmann constant in eV


class CFR:
    """
    Integration with the continued fraction representation.
    """

    def __init__(self, cutoff=0, T=300):
        self.cutoff = cutoff
        self.T = T
        self.beta = 1 / (kb * self.T)
        self.Rinf = 1e10
        if cutoff <= 0:
            raise ValueError("cutoff should be larger than 0.")
        else:
            self.prepare_poles()

    def prepare_poles(self):
        ##b_mat = [1 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)) / (kb * self.#T)) for j in range(0, self.cutoff - 1)]
        jmat = np.arange(0, self.cutoff - 1)
        b_mat = 1 / (2.0 * np.sqrt((2 * (jmat + 1) - 1) * (2 * (jmat + 1) + 1)))
        b_mat = np.diag(b_mat, -1) + np.diag(b_mat, 1)

        self.poles, residues = eig(b_mat)
        residules = 0.25 * np.abs(residues[0, :]) ** 2 / self.poles**2

        self.weights = np.where(
            np.real(self.poles) > 0, 4.0j / self.beta * residules, 0.0
        )

        # add a point to the poles: 1e10j
        self.path = 1j / self.poles * kb * self.T
        self.path = np.concatenate((self.path, [self.Rinf * 1j]))
        self.npoles = len(self.poles)

        # add weights for the point 1e10j
        self.weights = np.concatenate((self.weights, [-self.Rinf]))
        # self.weights = np.concatenate((self.weights, [00.0]))
        # zeros moment is 1j * R * test_gf(1j * R), but the real part of it will be taken. In contrast to the other part, where the imaginary part is taken.

    def integrate_values(self, gf_vals):
        return np.imag(gf_vals @ self.weights)

    def integrate(self, gf, ef=0):
        """
        Integration with the continued fraction representation.

        :param gf: Green's function
        :param ef: Fermi energy
        :return: integration result
        """
        path = self.path
        gf_vals = gf(path, ef=ef)
        return self.integrate_values(gf_vals)


def test_cfr():
    cfr = CFR(cutoff=100)

    def test_gf(z, ef=0.1):
        return 1 / (z - 3)

    print(cfr.integrate(test_gf, ef=0))


if __name__ == "__main__":
    test_cfr()
