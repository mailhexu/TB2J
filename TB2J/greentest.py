import numpy as np
from scipy.linalg import eigh, inv


def eigen_to_G(evals, evecs, efermi, energy):
    """ calculate green's function from eigenvalue/eigenvector for energy(e-ef): G(e-ef).
    :param evals:  eigen values
    :param evecs:  eigen vectors
    :param efermi: fermi energy
    :param energy: energy
    :returns: Green's function G,
    :rtype:  Matrix with same shape of the Hamiltonian (and eigenvector)
    """
    G=evecs.dot(np.diag(1.0 / (-evals + (energy + efermi)))).dot(
        evecs.conj().T)
    return G

def green_H(H, energy, S=np.eye(3)):
    return inv(S*energy - H)

def green_H_eig(H,energy):
    evals, evecs = eigh(H)
    return eigen_to_G(evals, evecs, 0.0, energy)


def test_eigh():
    H=np.random.random((3,3))
    H=H+H.T.conj()
    evals ,evecs=eigh(H)
    #print(f"VT@V: {evecs.T.conj()@evecs}")
    green_H(H, 1)
    green_H_eig(H, 1)
    print(f"H: {H}")

    S=np.random.random((3,3))+np.random.random((3,3))*1j
    S=S+S.T.conj()
    S=np.eye(3)*0.4+S
    evals, evecs=eigh(H, S)

    print(f"VT@V: {evecs.T.conj()@evecs}")
    print(f"VT@S@V: {evecs.T.conj()@S@evecs}")  # I
    print(f"V@S@VT: {evecs@S@evecs.T.conj()}")  # Not I
    print(f"S@VT@evals@V: {S@evecs.T.conj()@np.diag(evals)@evecs}")
    print(f"V@evals@VT: {evecs@np.diag(evals)@evecs.T.conj()}")
    print(f"VT@evals@V: {evecs.T.conj()@np.diag(evals)@evecs}")

    G1=green_H(H, 0.3, S=S)
    #print("G1=", G1)

    evals, evecs= eigh(H, S)
    G2= eigen_to_G(evals, evecs, 0.3, 0)
    G2=green_H(H, 0.3, S=S)
    print(f"G1-G2={G1-G2}")




test_eigh()
