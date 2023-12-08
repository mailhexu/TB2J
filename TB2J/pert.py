import numpy as np
import scipy.linalg as sl


def eigen_to_G(evals, evecs, efermi, energy):
    """calculate green's function from eigenvalue/eigenvector for energy(e-ef): G(e-ef).
    :param evals:  eigen values
    :param evecs:  eigen vectors
    :param efermi: fermi energy
    :param energy: energy
    :returns: Green's function G,
    :rtype:  Matrix with same shape of the Hamiltonian (and eigenvector)
    """
    return evecs.dot(np.diag(1.0 / (-evals + energy + efermi))).dot(evecs.conj().T)


def H2G(H, energy):
    return np.linalg.inv(energy * np.eye(len(H)) - H)


def Gpert(H0, dH, e):
    """
    Show Dyson
    """
    evals, evecs = sl.eigh(H0)
    G0 = eigen_to_G(evals, evecs, 0.0, e)

    G01 = H2G(H0, e)
    print(G0 - G01)

    evals, evecs = sl.eigh(H0 + dH)
    G = eigen_to_G(evals, evecs, 0.0, e)

    print(G - (G0 @ dH @ G + G0))
    print(G - (G0 @ dH @ G0 + G0))
    # print(G0@dH@G0+G0@dH@G0@dH@G0-G+G0)
    # print(G0@dH@G0+G0@dH@G0@dH@G0+G0@dH@G0@dH@G0@dH@G0-G+G0)


def test():
    H0 = np.random.rand(4, 4)
    H0 = H0 + H0.T.conj()

    dH = np.random.rand(4, 4)
    dH = H0 + H0.T.conj()

    dH = dH * 0.1

    Gpert(H0, dH, 1.3 + 0.00j)


# test()
