import numpy as np
from TB2J.pauli import pauli_block_all, s0, s1, s2, s3, gather_pauli_blocks


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
    test_rotate_Matrix_from_z_to_axis()
