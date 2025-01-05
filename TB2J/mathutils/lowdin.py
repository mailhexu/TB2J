import numpy as np
from scipy.linalg import eigh


def Lowdin(S):
    """
    Calculate S^(-1/2).
    Which is used in lowind's symmetric orthonormalization.
    psi_prime = S^(-1/2) psi
    """
    eigval, eigvec = eigh(S)
    S_half = eigvec @ np.diag(np.sqrt(1.0 / eigval)) @ (eigvec.T.conj())
    return S_half


def Lowdin_symmetric_orthonormalization(H, S):
    """
    Lowdin's symmetric orthonormalization.
    """
    S_half = Lowdin(S)
    H_prime = S_half @ H @ S_half
    return H_prime
