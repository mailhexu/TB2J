import numpy as np
from scipy.sparse import eye_array, kron

from TB2J.pauli import gather_pauli_blocks, pauli_block_all


def rotate_Matrix_from_z_to_axis(M, axis, normalize=True):
    """
    Given a spinor matrix M, rotate it from z-axis to axis.
    The spinor matrix M is a 2x2 matrix, which can be decomposed as I, x, y, z components using Pauli matrices.
    """
    MI, Mx, My, Mz = pauli_block_all(M)
    axis = axis / np.linalg.norm(axis)
    # M_new = s0* MI +  Mz * (axis[0] * s1 + axis[1] * s2 + axis[2] * s3) *2
    M_new = gather_pauli_blocks(MI, Mz * axis[0], Mz * axis[1], Mz * axis[2])
    return M_new


def spherical_to_cartesian(theta, phi, normalize=True):
    """
    Convert spherical coordinates to cartesian
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    vec = np.array([x, y, z])
    if normalize:
        vec = vec / np.linalg.norm(vec)
    return vec


def rotation_matrix(theta, phi):
    """
    The unitray operator U, that U^dagger * s3 * U is the rotated s3 by theta and phi
    """
    U = np.array(
        [
            [np.cos(theta / 2), np.exp(-1j * phi) * np.sin(theta / 2)],
            [-np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    )
    return U


def rotate_spinor_single_block(M, theta, phi):
    """
    Rotate the spinor matrix M by theta and phi
    """
    U = rotation_matrix(theta, phi)
    Uinv = np.linalg.inv(U)
    return Uinv @ M @ U


def rotate_spinor_matrix(M, theta, phi, method="einsum"):
    """
    Rotate the spinor matrix M by theta and phi,
    """
    if method == "plain":
        return rotate_spinor_matrix_plain(M, theta, phi)
    elif method == "einsum":
        return rotate_spinor_matrix_einsum(M, theta, phi)
    elif method == "reshape":
        return rotate_spinor_matrix_reshape(M, theta, phi)
    elif method == "kron":
        return rotate_spinor_matrix_kron(M, theta, phi)
    elif method == "spkron":
        return rotate_spinor_matrix_spkron(M, theta, phi)
    else:
        raise ValueError(f"Unknown method: {method}")


def rotate_spinor_matrix_plain(M, theta, phi):
    """
    M is a matrix with shape (2N, 2N), where N is the number of sites, and each site has a 2x2 matrix
    rotate each 2x2 block by theta and phi
    """
    Mnew = np.zeros_like(M)
    U = rotation_matrix(theta, phi)
    UT = U.conj().T
    tmp = np.zeros((2, 2), dtype=np.complex128)
    for i in range(M.shape[0] // 2):
        for j in range(M.shape[0] // 2):
            for k in range(2):
                for l in range(2):
                    tmp[k, l] = M[2 * i + k, 2 * j + l]
            # tmp[:, :]=M[2*i:2*i+2, 2*j:2*j+2]
            Mnew[2 * i : 2 * i + 2, 2 * j : 2 * j + 2] = UT @ tmp @ U
    return Mnew


def rotate_spinor_matrix_einsum(M, theta, phi):
    """
    Rotate the spinor matrix M by theta and phi,
    """
    shape = M.shape
    n1 = np.product(shape[:-1]) // 2
    n2 = M.shape[-1] // 2
    Mnew = np.reshape(M, (n1, 2, n2, 2))  # .swapaxes(1, 2)
    # print("Mnew:", Mnew)
    U = rotation_matrix(theta, phi)
    UT = U.conj().T
    Mnew = np.einsum(
        "ij, rjsk, kl -> risl", UT, Mnew, U, optimize=True, dtype=np.complex128
    )
    Mnew = Mnew.reshape(shape)
    return Mnew


def rotate_spinor_matrix_einsum_R(M, theta, phi):
    """
    Rotate the spinor matrix M by theta and phi,
    """
    nR = M.shape[0]
    N = M.shape[1] // 2
    Mnew = np.reshape(M, (nR, N, 2, N, 2))  # .swapaxes(1, 2)
    U = rotation_matrix(theta, phi)
    UT = U.conj().T
    Mnew = np.einsum(
        "ij, nrjsk, kl -> nrisl", UT, Mnew, U, optimize=True, dtype=np.complex128
    )
    Mnew = Mnew.reshape(nR, 2 * N, 2 * N)
    return Mnew


def rotate_spinor_Matrix_R(M, theta, phi):
    return rotate_spinor_matrix_einsum_R(M, theta, phi)


def rotate_spinor_matrix_reshape(M, theta, phi):
    """
    Rotate the spinor matrix M by theta and phi,
    """
    N = M.shape[0] // 2
    Mnew = np.reshape(M, (N, 2, N, 2)).swapaxes(1, 2)
    # print("Mnew:", Mnew)
    U = rotation_matrix(theta, phi)
    UT = U.conj().T
    Mnew = UT @ Mnew @ U
    Mnew = Mnew.swapaxes(1, 2).reshape(2 * N, 2 * N)
    return Mnew


def rotate_spinor_matrix_kron(M, theta, phi):
    """ """
    U = rotation_matrix(theta, phi)
    # U = np.kron( U, np.eye(M.shape[0]//2))
    # U = kron(eye_array(M.shape[0]//2), U)
    U = np.kron(np.eye(M.shape[0] // 2), U)
    M = U.conj().T @ M @ U
    return M


def rotate_spinor_matrix_spkron(M, theta, phi):
    """ """
    U = rotation_matrix(theta, phi)
    # U = np.kron( U, np.eye(M.shape[0]//2))
    U = kron(eye_array(M.shape[0] // 2), U)
    # U = np.kron(np.eye(M.shape[0]//2), U)
    M = U.conj().T @ M @ U
    return M


def test_rotate_spinor_M():
    N = 2
    M_re = np.random.rand(N * 2, N * 2)
    M_im = np.random.rand(N * 2, N * 2)
    M = M_re + 1j * M_im
    M = M + M.T.conj()
    # M=np.array([[1, 0], [0, 1]], dtype=np.complex128)
    print(f"Original M: {M}")

    import timeit

    print("Time for rotate_spinor_matrix")
    print(
        timeit.timeit(lambda: rotate_spinor_matrix(M, np.pi / 2, np.pi / 2), number=10)
    )
    print("Time for rotate_spinor_matrix_einsum")
    print(
        timeit.timeit(
            lambda: rotate_spinor_matrix_einsum(M, np.pi / 2, np.pi / 2), number=10
        )
    )
    print("Time for rotate_spinor_matrix_reshape")
    print(
        timeit.timeit(
            lambda: rotate_spinor_matrix_reshape(M, np.pi / 2, np.pi / 2), number=10
        )
    )
    print("Time for rotate_spinor_matrix_kron")
    print(
        timeit.timeit(
            lambda: rotate_spinor_matrix_kron(M, np.pi / 2, np.pi / 2), number=10
        )
    )
    print("Time for rotate_spinor_matrix_spkron")
    print(
        timeit.timeit(
            lambda: rotate_spinor_matrix_spkron(M, np.pi / 2, np.pi / 2), number=10
        )
    )

    Mrot1 = rotate_spinor_matrix(M, np.pi / 2, np.pi / 2)
    Mrot2 = rotate_spinor_matrix_einsum(M, np.pi / 2, np.pi / 2)
    Mrot3 = rotate_spinor_matrix_reshape(M, np.pi / 2, np.pi / 2)
    Mrot4 = rotate_spinor_matrix_kron(M, np.pi / 2, np.pi / 2)
    Mrot5 = rotate_spinor_matrix_spkron(M, np.pi / 2, np.pi / 2)
    print(f"Rotated M with jit:\n {Mrot1}")
    print(f"Rotated M with einsum:\n {Mrot2-Mrot1}")
    print(f"Rotated M with reshape:\n {Mrot3-Mrot1}")
    print(f"Rotated M with kron:\n {Mrot4-Mrot1}")
    print(f"Rotated M with spkron:\n {Mrot5-Mrot1}")

    M_rot00 = rotate_spinor_matrix(M, 0, 0)
    M_rot00_sph = rotate_Matrix_from_z_to_spherical(M, 0, 0)
    print(f"Rotated M with theta=0, phi=0 compared with M:\n {M_rot00-M}")
    print(f"Rotated M with theta=0, phi=0 compared with M:\n {M_rot00_sph-M}")


def test_rotate_spinor_oneblock():
    M = np.array([[1.1, 0], [0, 0.9]])
    print(np.array(pauli_block_all(M)).ravel())
    print("Rotate by pi/2, pi/2 (z to y)")
    Mnew = rotate_spinor_matrix_einsum(M, np.pi / 2, np.pi / 2)
    print(np.array(pauli_block_all(Mnew)).ravel())

    print("Rotate by pi/2, 0 (z to x)")
    Mnew = rotate_spinor_matrix_kron(M, np.pi / 2, 0)

    print(np.array(pauli_block_all(Mnew)).ravel())

    print(Mnew)


def rotate_Matrix_from_z_to_spherical(M, theta, phi, normalize=True):
    """
    Given a spinor matrix M, rotate it from z-axis to spherical coordinates
    """
    # axis = spherical_to_cartesian(theta, phi, normalize)
    # return rotate_Matrix_from_z_to_axis(M, axis, normalize)
    return rotate_spinor_matrix_einsum(M, theta, phi)


def test_rotate_Matrix_from_z_to_spherical():
    M_re = np.random.rand(2, 2)
    M_im = np.random.rand(2, 2)
    M = M_re + 1j * M_im
    print(M)
    M_rot = rotate_Matrix_from_z_to_spherical(M, 0, 0)
    print(M_rot)


def test_rotate_Matrix_from_z_to_axis():
    M = np.array([[1.1, 0], [0, 0.9]])
    print(pauli_block_all(M))
    Mnew = rotate_Matrix_from_z_to_axis(M, [1, 1, 1])
    print(pauli_block_all(Mnew))
    print(Mnew)

    M = np.array(
        [
            [-9.90532976e-06 + 0.0j, 0.00000000e00 + 0.0j],
            [0.00000000e00 + 0.0j, -9.88431291e-06 + 0.0j],
        ]
    )
    print(M)
    print(rotate_Matrix_from_z_to_axis(M, [0, 0, 1]))


if __name__ == "__main__":
    # test_rotate_Matrix_from_z_to_axis()
    # test_rotate_Matrix_from_z_to_spherical()
    test_rotate_spinor_M()
    # test_rotate_spinor_oneblock()
    pass
