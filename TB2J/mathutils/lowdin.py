import numpy as np
from scipy.linalg import inv, eigh


def Lowdin(S):
    """
    Calculate S^(-1/2).
    Which is used in lowind's symmetric orthonormalization.
    psi_prime = S^(-1/2) psi
    """
    eigval, eigvec = eigh(S)
    return eigvec @ np.diag(np.sqrt(1.0 / eigval)) @ (eigvec.T.conj())
