# -*- coding: utf-8 -*-
# File: mycfr.py
# Time: 2017/12/10 19:29:43
"""
Continued fraction representation.
"""

import numpy as np
from ase.units import kB
from scipy.linalg import eig


class CFR2:
    """Normalized continued-fraction integral with its high-frequency moment.

    ``nz`` is an even integer >= 2; ``T`` is the positive temperature in K.
    Energies and weights use eV. Samples alternate positive/negative imaginary
    poles, followed by the large-frequency moment sample. Integrate along the
    leading sample axis; scalar, matrix and batched complex values are supported.

    The result contains quadrature normalization, not an observable prefactor.
    For a scalar resolvent, occupation is ``-result.imag / pi``. Full complex
    density matrices instead require anti-Hermitian extraction. No spin
    degeneracy is supplied. See Ozaki, Phys. Rev. B 75, 035123 (2007).
    """

    def __init__(self, nz: int = 50, T: float = 200.0):
        if (
            isinstance(nz, (bool, np.bool_))
            or not isinstance(nz, (int, np.integer))
            or nz < 2
            or nz % 2
        ):
            raise ValueError("nz must be an even integer >= 2.")
        self.nz = int(nz)
        self.T = float(T)
        self.beta = 1.0 / (kB * self.T)
        self.Rinf = 1e10
        self.prepare_poles()

    def prepare_poles(self):
        # MathDoc: tb2j/cfr2-integration-contract#finite_pole_normalization_ratio
        j = np.arange(1, self.nz)
        off_diagonal = 1 / (2 * np.sqrt((2 * j - 1) * (2 * j + 1)))
        matrix = np.diag(off_diagonal, -1) + np.diag(off_diagonal, 1)
        self.poles, eigenvectors = eig(matrix)
        residues = 0.25 * np.abs(eigenvectors[0]) ** 2 / self.poles**2
        self.path = np.empty(self.nz + 1, dtype=complex)
        self.weights = np.empty(self.nz + 1, dtype=complex)
        index = 0
        for pole, residue in zip(self.poles, residues):
            if pole.real > 0:
                z = 1j / pole / self.beta
                self.path[index : index + 2] = (z, -z)
                self.weights[index : index + 2] = -np.pi * 1j * residue / self.beta
                index += 2
        self.npoles = self.nz
        # MathDoc: tb2j/cfr2-integration-contract#moment_sample_contribution
        self.path[-1] = 1j * self.Rinf
        self.weights[-1] = np.pi * self.Rinf / 2

    def integrate_values(self, gf_vals, imag=False):
        """Contract stored weights with values on the leading sample axis."""
        ret = np.einsum("i,i...->...", self.weights, gf_vals)
        return np.imag(ret) if imag else ret

    def integrate_func(self, gf, ef=0):
        """Integrate ``gf(self.path, ef=ef)`` without shifting the path again."""
        return self.integrate_values(gf(self.path, ef=ef))


if __name__ == "__main__":
    contour = CFR2(nz=80, T=600)
    energy = 0.2
    occupation = -contour.integrate_values(1 / (contour.path - energy)).imag / np.pi
    expected = 1 / (1 + np.exp(energy / (kB * contour.T)))
    np.testing.assert_allclose(occupation, expected, atol=1e-10, rtol=0)
    print(f"CFR2 occupation at {energy} eV, {contour.T} K: {occupation:.12g}")
