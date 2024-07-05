"""
Module about Pauli (and I) matrices.
"""

import numpy as np
from numpy import zeros_like
from scipy.linalg import svd

s0 = np.array([[1, 0], [0, 1]])
s1 = np.array([[0, 1], [1, 0]])
s2 = np.array([[0, -1j], [1j, 0]])
s3 = np.array([[1, 0], [0, -1]])

s0T = s0.T
s1T = s1.T
s2T = s2.T
s3T = s3.T

pauli_dict = {0: s0, 1: s1, 2: s2, 3: s3}


def pauli_mat(nbasis, i):
    """
    nbasis: size of the matrix. should be multiple of 2.
    i: index of pauli dictionary.
    """
    N = nbasis // 2
    assert N * 2 == nbasis
    M = np.ones((N, N), dtype="complex")
    spm = pauli_dict[i]
    return np.block([[M * spm[0, 0], M * spm[0, 1]], [M * spm[1, 0], M * spm[1, 1]]])


def pauli_decomp(M):
    """Given a 2*2 matrix, get the I, x, y, z component.
    :param M: 2*2 matrix
    :returns:  (I, x, y, z) are four scalars.
    :rtype: same as dtype of M
    """
    return (
        np.trace(s0.dot(M)) / 2,
        np.trace(s1.dot(M)) / 2,
        np.trace(s2.dot(M)) / 2,
        np.trace(s3.dot(M)) / 2,
    )


def pauli_decomp2(M):
    """Given a 2*2 matrix, get the I, x, y, z component. (method2)
    :param M: 2*2 matrix
    :returns:  (I, x, y, z) are four scalars.
    :rtype: same as dtype of M
    """
    return (
        np.sum(M * s0T) / 2,
        np.sum(M * s1T) / 2,
        np.sum(M * s2T) / 2,
        np.sum(M * s3T) / 2,
    )


def pauli_sigma_norm(M):
    MI, Mx, My, Mz = pauli_decomp2(M)
    return np.linalg.norm([Mx, My, Mz])


def pauli_block_I(M, norb):
    """
    I compoenent of a matrix, see pauli_block
    """
    ret = zeros_like(M)
    tmp = (M[::2, ::2] + M[1::2, 1::2]) / 2
    ret[::2, ::2] = ret[1::2, 1::2] = tmp
    return ret


def pauli_block_x(M, norb):
    """
    x compoenent of a matrix, see pauli_block
    """
    ret = zeros_like(M)
    tmp = (M[::2, 1::2] + M[1::2, ::2]) / 2
    ret[::2, 1::2] = ret[1::2, ::2] = tmp
    return ret


def pauli_block_y(M, norb):
    """
    y compoenent of a matrix, see pauli_block
    """
    ret = zeros_like(M)
    tmp = (M[::2, 1::2] * (-1j) + M[1::2, ::2] * (1j)) / 2
    ret[::2, 1::2] = tmp * (-1j)
    ret[1::2, ::2] = tmp * 1j
    return tmp, ret


def pauli_block_z(M, norb):
    """z compoenent of a matrix, see pauli_block
    :param M:
    :param norb:
    :returns:
    :rtype:
    """
    ret = zeros_like(M)
    tmp = (M[::2, ::2] - M[1::2, 1::2]) / 2
    ret[::2, ::2] = tmp
    ret[1::2, 1::2] = -tmp
    return tmp, ret


def pauli_block(M, idim):
    """Get the I, x, y, z component of a matrix.
    :param M: The input matrix,  aranged in four blocks:
    [[upup, updn], [dnup, dndn]]. let norb be number of orbitals in
    each block. (so M has dim 2norb1*2norb2)
    :param idim: 0, 1,2, 3 for I , x, y, z.
    :returns:  the idim(th) component of M
    :rtype: a matrix with shape of M.shape//2
    """
    # ret = zeros_like(M)
    if idim == 0:
        tmp = (M[::2, ::2] + M[1::2, 1::2]) / 2.0
    elif idim == 1:
        tmp = (M[::2, 1::2] + M[1::2, ::2]) / 2.0
    elif idim == 2:
        # Note that this is not element wise product with sigma_y but dot product
        # sigma_y=[[0, -1j],[1j, 0]]
        tmp = (M[::2, 1::2] * (1.0j) + M[1::2, ::2] * (-1.0j)) / 2.0
    elif idim == 3:
        tmp = (M[::2, ::2] - M[1::2, 1::2]) / 2.0
    else:
        raise NotImplementedError()
    return tmp


def pauli_block_all(M):
    MI = (M[::2, ::2] + M[1::2, 1::2]) / 2.0
    Mx = (M[::2, 1::2] + M[1::2, ::2]) / 2.0
    # Note that this is not element wise product with sigma_y but dot product
    My = (M[::2, 1::2] - M[1::2, ::2]) * 0.5j
    Mz = (M[::2, ::2] - M[1::2, 1::2]) / 2.0
    return MI, Mx, My, Mz


def gather_pauli_blocks(MI, Mx, My, Mz, coeffs=[1.0, 1.0, 1.0, 1.0]):
    """
    Gather the I, x, y, z component of a matrix.
    """
    cI, cx, cy, cz = coeffs
    return (
        cI * np.kron(MI, s0)
        + cx * np.kron(Mx, s1)
        + cy * np.kron(My, s2)
        + cz * np.kron(Mz, s3)
    )


def pauli_part(M, coeffs=[1.0, 1.0, 1.0, 1.0]):
    """
    Get the I, x, y, z part of a matrix.
    """
    MI, Mx, My, Mz = pauli_block_all(M)
    return gather_pauli_blocks(MI, Mx, My, Mz, coeffs=coeffs)


def chargepart(M):
    """
    Get the charge part of a matrix.
    """
    MI = (M[::2, ::2] + M[1::2, 1::2]) / 2.0
    Mcopy = np.zeros_like(M)
    Mcopy[::2, ::2] = MI
    Mcopy[1::2, 1::2] = MI
    return Mcopy


def spinpart(M):
    """
    Get the spin part of a matrix.
    """
    MI = (M[::2, ::2] + M[1::2, 1::2]) / 2.0
    Mcopy = M.copy()
    Mcopy[::2, ::2] -= MI
    Mcopy[1::2, 1::2] -= MI
    return Mcopy


def test_gather_pauli_blocks():
    M = np.random.rand(4, 4)
    MI, Mx, My, Mz = pauli_block_all(M)
    M2 = gather_pauli_blocks(MI, Mx, My, Mz)
    assert np.allclose(M, M2)


def op_norm(M):
    """
    Return the operator norm of a matrix.
    """
    return max(svd(M)[1])


def pauli_block_sigma_norm(M):
    """
    M= MI * I + \vec{P} dot \vec{sigma}
    = MI*I + p * \vec{e} dot \vec{sigma}
    where p is the norm of P.
    """
    MI, Mx, My, Mz = pauli_block_all(M)
    ex, ey, ez = np.trace(Mx), np.trace(My), np.trace(Mz)
    # ex,ey,ez = op_norm(Mx), op_norm(My), op_norm(Mz)
    evec = np.array((ex, ey, ez))
    evec = evec / np.linalg.norm(evec)
    return Mx * evec[0] + My * evec[1] + Mz * evec[2]
