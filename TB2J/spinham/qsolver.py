#!/usr/bin/env python
import math
import numpy as np
import scipy.linalg as linalg
from TB2J.kpoints import monkhorst_pack
from ase.dft.dos import DOS
import matplotlib.pyplot as plt
from ase.units import eV, J


class QSolver(object):
    def __init__(self, hamiltonian):
        self.ham = hamiltonian
        self.nspin = self.ham.nspin
        if len(self.ham.spin.shape) == 2:
            M = linalg.norm(self.ham.spin, axis=1)
        else:
            M = self.ham.spin

        self.M_mat = np.kron(np.sqrt(np.einsum("i,j->ij", M, M)), np.ones((3, 3)))
        self.Eref = None

    def get_Eref(self):
        mat = np.zeros((3 * self.nspin, 3 * self.nspin), dtype=complex)
        for key, val in self.ham.get_total_hessian_ijR().items():
            i, j, R = key
            mat[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] -= val
        mat = mat * 4.0 / self.M_mat
        SZ = np.zeros(self.nspin * 3, dtype=float)
        SZ[2::3] = 1.0
        SZ /= np.linalg.norm(SZ)
        self.Eref = np.dot(SZ, np.dot(mat, SZ)).real
        return self.Eref

    def solve_k(self, kpt, eigen_vectors=True, Jq=False):
        if self.Eref is None and not Jq:
            self.get_Eref()
        mat = np.zeros((3 * self.nspin, 3 * self.nspin), dtype=complex)
        for key, val in self.ham.get_total_hessian_ijR().items():
            i, j, R = key
            mat[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3] -= val * np.exp(
                2.0j * math.pi * np.dot(kpt, R)
            )
        if not Jq:
            mat = mat * 4.0 / self.M_mat
        if eigen_vectors:
            evals, evecs = linalg.eigh(mat)
            evals = np.real(evals)
            if not Jq:
                evals -= self.Eref
            return evals, evecs
        else:
            evals = np.linalg.eigvalsh(mat)
            if not Jq:
                evals -= self.Eref
            return evals

    def solve_all(self, kpts, eigen_vectors=True, Jq=False):
        eval_list = []
        evec_list = []
        if eigen_vectors:
            for kpt in kpts:
                evals, evecs = self.solve_k(kpt, eigen_vectors=True, Jq=Jq)
                eval_list.append(evals)
                evec_list.append(evecs)
            return np.array(eval_list), np.array(evec_list)
        else:
            for kpt in kpts:
                evals = self.solve_k(kpt, eigen_vectors=False, Jq=Jq)
                eval_list.append(evals)
            return np.array(eval_list)


class QSolverASEWrapper(QSolver):
    def set(self, atoms=None, kmesh=[9, 9, 9], gamma=True, Jq=False, **kwargs):
        self.atoms = atoms
        self.dos_args = {
            "kmesh": kmesh,
            "gamma": gamma,
            "Jq": Jq,
        }
        self.dos_kwargs = kwargs
        self.kpts = monkhorst_pack(
            self.dos_args["kmesh"], gamma_center=self.dos_args["gamma"]
        )
        self.weights = np.ones(len(self.kpts)) / len(self.kpts)

    def get_k_points_and_weights(self):
        return self.kpts, self.weights

    def get_k_point_weights(self):
        return self.weights

    def get_number_of_spins(self):
        return 1

    def get_eigenvalues(self, kpt, spin=0):
        """
        return the eigenvalues at a given k-point. The energy unit is eV
        args:
            kpt: k-point index.
            spin: spin index.
        """
        kpoint = self.kpts[kpt]
        evals = self.solve_k(kpoint, eigen_vectors=False, Jq=self.dos_args["Jq"])
        evals = evals * J / eV
        return evals

    def get_fermi_level(self):
        return 0.0

    def get_bz_k_points(self):
        return self.kpts

    def get_dos(self, width=0.1, window=None, npts=401):
        dos = DOS(self, width=width, window=window, npts=npts)
        energies = dos.get_energies()
        tdos = dos.get_dos()
        return energies, tdos

    def plot_dos(
        self,
        smearing_width=0.0001,
        window=None,
        npts=401,
        output="dos.pdf",
        ax=None,
        show=True,
        dos_filename="magnon_dos.txt",
    ):
        """
        plot total DOS.
        :param width: width of Gaussian smearing
        :param window: energy window
        :param npts: number of points
        :param output: output filename
        :param ax: matplotlib axis
        :return: ax
        """
        if ax is None:
            _fig, ax = plt.subplots()
        energies, tdos = self.get_dos(width=smearing_width, window=window, npts=npts)
        energies = energies * 1000
        tdos = tdos / 1000
        if dos_filename is not None:
            np.savetxt(
                dos_filename,
                np.array([energies, tdos]).T,
                header="Energy(meV) DOS(state/meV)",
            )
        ax.plot(energies, tdos)
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("DOS (states/meV)")
        ax.set_title("Total DOS")
        if output is not None:
            plt.savefig(output)
        if show:
            plt.show()
        return ax
