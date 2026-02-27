# -*- coding: utf-8 -*-
# File: mycfr.py
# Time: 2017/12/10 19:29:43
"""
Continued fraction representation.
"""

import numpy as np
from ase.units import kB
from scipy.linalg import eig


class CFR:
    """Integration with the continued fraction representation."""

    def __init__(self, nz: int = 50, T: float = 200.0):
        self.nz = nz
        self.T = float(T)
        self.beta = 1.0 / (kB * self.T)
        self.Rinf = 1e10
        if nz <= 0:
            raise ValueError("nz should be larger than 0.")
        else:
            self.prepare_poles()

    def prepare_poles(self):
        ##b_mat = [1 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)) / (kB * self.#T)) for j in range(0, self.nz- 1)]
        jmat = np.arange(0, self.nz - 1)
        b_mat = 1 / (2.0 * np.sqrt((2 * (jmat + 1) - 1) * (2 * (jmat + 1) + 1)))
        b_mat = np.diag(b_mat, -1) + np.diag(b_mat, 1)
        self.poles, residules = eig(b_mat)

        residules = 0.25 * np.abs(residules[0, :]) ** 2 / self.poles**2
        # residules = 0.25 * np.diag(residules) ** 2 / self.poles**2

        # self.weights = np.where(
        #        #np.real(self.poles) > 0, 4.0j / self.beta * residules, 0.0
        #        #np.real(self.poles) > 0, 2.0j / self.beta * residules, 2.0j / self.beta * residules
        #    np.real(self.poles) > 0, 2.0j / self.beta * residules, 0.0
        # )

        # self.path = 1j / self.poles * kB * self.T

        self.path = []
        self.weights = []
        for p, r in zip(self.poles, residules):
            if p > 0:
                self.path.append(1j / p * kB * self.T)
                w = 2.0j / self.beta * r
                self.weights.append(w)
                self.path.append(-1j / p * kB * self.T)
                self.weights.append(w)

        self.path = np.array(self.path)
        self.weights = np.array(self.weights)

        # from J. Phys. Soc. Jpn. 88, 114706 (2019)
        # A_mat =  -1/2 *np.diag(1, -1) + np.diag(1, 1)
        # B_mat = np.diag([2*i-1 for i in range(1, self.nz)])
        # eigp, eigv = eig(A_mat, B_mat)
        # zp = 1j / eigp * kB * self.T
        # Rp = 0.25 * np.diag(eigv)**2 * zp **2

        # print the poles and the weights
        # for i in range(len(self.poles)):
        #    print("Pole: ", self.poles[i], "Weight: ", self.weights[i])

        # add a point to the poles: 1e10j
        self.path = np.concatenate((self.path, [self.Rinf * 1j]))
        # self.path = np.concatenate((self.path, [0.0]))
        self.npoles = len(self.poles)

        # add weights for the point 1e10j
        # self.weights = np.concatenate((self.weights, [-self.Rinf]))
        self.weights = np.concatenate((self.weights, [00.0]))
        # zeros moment is 1j * R * test_gf(1j * R), but the real part of it will be taken. In contrast to the other part, where the imaginary part is taken.

    def integrate_values(self, gf_vals, imag=False):
        ret = 0
        for i in range(self.npoles + 1):
            ret += self.weights[i] * gf_vals[i]
        ret *= (
            -np.pi / 2
        )  # This is to be compatible with the integration of contour, where /-np.pi is used after the integration. And the factor of 2 is because the Fermi function here is 2-fold degenerate.
        if imag:
            return np.imag(ret)
        return ret

    def integrate_func(self, gf, ef=0):
        """
        Integration with the continued fraction representation.

        :param gf: Green's function
        :param ef: Fermi energy
        :return: integration result
        """
        path = self.path
        gf_vals = gf(path, ef=ef)
        return self.integrate_values(gf_vals)


class CFR2(CFR):
    """
    Alternative integration with the continued fraction representation.
    Follows Ozaki, PRB 75, 035123 (2007) and Eq (c1-26).
    It includes the 1/2*mu0 term explicitly.
    """

    def prepare_poles(self):
        # Call parent to set up poles, residues, path
        super().prepare_poles()

        # Recompute weights for CFR2 formulation
        # CFR2: rho = 0.5 * mu0 + (1/beta) * sum (G+ + G-) * Rp
        # We want integrate_values to return res such that Im[-1/pi * res] = rho
        # So res = -pi * j * rho = -pi*j * [0.5*mu0 + (1/beta)*sum]

        # For the mu0 term (last element in path):
        # mu0 = iR * G(iR), and we want contribution: -pi*j * 0.5 * mu0
        # = -pi*j * 0.5 * iR * G(iR) = 0.5*pi*R * G(iR)
        self.weights[-1] = 0.5 * np.pi * self.Rinf

        # For the pole summation terms:
        # Original CFR weights: 2j/beta * Rp (for each of +/- pair)
        # CFR2 needs: -pi*j * (1/beta) * Rp for each pole
        # But we have pairs (G+ and G-), so each gets the same weight
        # Total contribution from pair: -pi*j * (1/beta) * Rp * (G+ + G-)
        # So each individual weight should be: -pi*j * (1/beta) * Rp

        # Recompute residues to get correct weights
        jmat = np.arange(0, self.nz - 1)
        b_mat = 1 / (2.0 * np.sqrt((2 * (jmat + 1) - 1) * (2 * (jmat + 1) + 1)))
        b_mat = np.diag(b_mat, -1) + np.diag(b_mat, 1)
        poles, eigenvectors = eig(b_mat)
        residules = 0.25 * np.abs(eigenvectors[0, :]) ** 2 / poles**2

        # Update weights for pole pairs
        idx = 0
        for p, r in zip(poles, residules):
            if p > 0:
                w = -np.pi * 1j * r / self.beta
                self.weights[idx] = w  # G(alpha+)
                self.weights[idx + 1] = w  # G(alpha-)
                idx += 2

    def integrate_values(self, gf_vals, imag=False):
        # gf_vals can be:
        # - scalar: shape (n_points,)
        # - matrix: shape (n_points, nbasis, nbasis)
        # weights: shape (n_points,)

        # Use einsum to handle both cases
        if gf_vals[0].ndim == 0:
            # Scalar case
            ret = np.sum(self.weights * gf_vals)
        else:
            # Matrix case: einsum over the first axis
            ret = np.einsum("i,i...->...", self.weights, gf_vals)

        if imag:
            return np.imag(ret)
        return ret


def test_cfr():
    cfr = CFR(nz=100)

    def test_gf(z, ef=0.1):
        return 1 / (z - 3 + ef)

    r = cfr.integrate_func(test_gf, ef=2)
    r = -np.imag(r) / np.pi * 2
    if not np.close(r, -1.0):
        raise AssertionError(
            f"Test failed, the integration should be close to -1.0 but get {r}"
        )
    return r


if __name__ == "__main__":
    test_cfr()
