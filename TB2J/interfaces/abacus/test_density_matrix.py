import numpy as np
from scipy.linalg import eigh


def gen_random_hermitean_matrix(n):
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    return A + A.conj().T


def gen_overlap_matrix(n):
    A = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    return np.dot(A, A.conj().T)


def fermi_function(x, ef, beta):
    return 1.0 / (np.exp(beta * (x - ef)) + 1)


def test():
    n = 10
    A = gen_random_hermitean_matrix(n)
    S = gen_overlap_matrix(n)
    beta = 0.1
    ef = 0

    evals, evecs = eigh(A, S)

    etot = np.sum(evals * fermi_function(evals, ef, beta))

    rho = np.einsum("ib,b,jb->ij", evecs, fermi_function(evals, ef, beta), evecs.conj())

    etot2 = np.trace(np.dot(A, rho))

    print(etot, etot2)


if __name__ == "__main__":
    test()
