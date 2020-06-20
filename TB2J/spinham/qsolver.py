#!/usr/bin/env python
import math
import numpy as np
import scipy.linalg as linalg

class QSolver(object):
    def __init__(self, hamiltonian):
        self.ham = hamiltonian
        self.nspin = self.ham.nspin
        M = linalg.norm(self.ham.spinat, axis=1)
        self.M_mat=np.einsum('i,j->ij', M, M)

    def solve_k(self, kpt, eigen_vectors=True):
        mat = np.zeros((3 * self.nspin, 3 * self.nspin), dtype=complex)
        for key, val in self.ham.get_total_hessian_ijR().items():
            i, j, R = key
            mat[i * 3:i * 3 + 3, j * 3:j * 3 + 3] -= val * np.exp(
               2.0j * math.pi * np.dot(kpt, R))
        mat=mat*8.0/self.M_mat
        if eigen_vectors:
            evals, evecs = linalg.eigh(mat)
            return evals, evecs
        else:
            evals = np.linalg.eigvalsh(mat)
            return evals

    def solve_all(self, kpts, eigen_vectors=True):
        eval_list = []
        evec_list = []
        if eigen_vectors:
            for kpt in kpts:
                evals, evecs = self.solve_k(kpt, eigen_vectors=True)
                eval_list.append(evals)
                evec_list.append(evecs)
            return np.array(eval_list), np.array(evec_list)
        else:
            for kpt in kpts:
                evals = self.solve_k(kpt, eigen_vectors=False)
                eval_list.append(evals)
            return np.array(eval_list)
