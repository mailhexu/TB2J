import numpy as np
from aiida_tb2j.data import ExchangeData
from aiida_tb2j.data.exchange import get_rotation_arrays
from itertools import combinations_with_replacement

ux, uy, uz = np.eye(3).reshape((3, 1, 3))


def combine_arrays(u, v):
    """ """
    return np.concatenate(
        [u * v, np.roll(u, -1, axis=-1) * v, np.roll(v, -1, axis=-1) * u], axis=-1
    )


def get_coefficients(magmoms, indices):
    i, j = indices

    U, V = zip(*[get_rotation_arrays(magmoms, u=u) for u in [ux, uy, uz]])
    U = np.stack(U).swapaxes(0, 1)
    V = np.stack(V).swapaxes(0, 1)

    uc = combine_arrays(U[i], U[j].conj())
    ur = combine_arrays(U[i], U[j])
    uc2 = combine_arrays(U[i].conj(), U[j])
    u = np.concatenate([uc, ur, uc2], axis=1)

    return u, V


def get_C(H0, u, V):
    n = int(H0.shape[-1] / 2)
    upi = np.triu_indices(n)
    dig = np.diag_indices(n)

    i, j = upi
    AB0 = H0[:, [i, i, i + n], [j, j + n, j + n]]
    AB0 = np.swapaxes(AB0, 0, 2).reshape(len(i), 9)
    J0_flat = np.linalg.solve(u, AB0)
    J0 = np.empty((n, n, 3, 3), dtype=complex)
    # FIXME: syntax check failed. Is the following line correct?
    J0[*upi] = J0_flat[:, [0, 6, 5, 3, 1, 7, 8, 4, 2]].reshape(-1, 3, 3)
    # J0[upi] = J0_flat[:, [0, 6, 5, 3, 1, 7, 8, 4, 2]].reshape(-1, 3, 3)
    J0 += J0.swapaxes(0, 1)
    # FIXME: syntax check failed. Is the following line correct?
    J0[*dig] = 0.0
    J0[dig] = 0.0
    C = np.array([np.diag(a) for a in np.einsum("imx,ijxy,jmy->mi", V, 2 * J0, V)])
    return C


def get_J(H, kpoints, exchange):
    n = int(H.shape[-1] / 2)
    upi = np.triu_indices(n)
    dig = np.diag_indices(n)
    i, j = upi

    magmoms = exchange.magmoms()[np.unique(exchange.pairs)]
    magmoms /= np.linalg.norm(magmoms, axis=-1).reshape(-1, 1)
    u, V = get_coefficients(magmoms, indices=upi)

    H0 = np.stack(
        [
            1000
            * exchange._H_matrix(
                kpoints=np.zeros((1, 3)), with_DMI=True, with_Jani=True, u=u
            )
            for u in [ux, uy, uz]
        ]
    )[:, 0, :, :]
    C = get_C(H0, u, V)
    H[:, :, :n, :n] += C.reshape(3, 1, n, n)
    H[:, :, n:, n:] += C.reshape(3, 1, n, n)

    AB = H[:, :, [i, i, i + n], [j, j + n, j + n]]
    AB[:, :, 2, :] = AB[:, ::-1, 2, :]
    AB = np.moveaxis(AB, [2, 3], [1, 0]).reshape(len(i), 9, -1)

    vectors = exchange.get_vectors()
    exp_summand = np.exp(-2j * np.pi * vectors @ kpoints.T)
    nAB = np.einsum("nik,ndk->nid", AB, exp_summand) / len(kpoints)

    ii = np.where(i == j)
    i0 = np.where(np.linalg.norm(vectors, axis=-1) == 0.0)
    J = np.linalg.solve(u, nAB).swapaxes(1, 2)
    J = J[:, :, [0, 6, 5, 3, 1, 7, 8, 4, 2]].reshape(len(i), -1, 3, 3)
    J *= -1
    J[ii] *= 2
    J[i0] *= 0

    return J
