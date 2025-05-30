import numpy as np

__all__ = ["generate_grid", "get_rotation_arrays", "round_to_precision", "uz", "I"]

I = np.eye(3)
uz = np.array([[0.0, 0.0, 1.0]])


def generate_grid(kmesh, sort=True):
    half_grid = [int(n / 2) for n in kmesh]
    grid = np.stack(
        np.meshgrid(*[np.arange(-n, n + 1) for n in half_grid]), axis=-1
    ).reshape(-1, 3)

    if sort:
        idx = np.linalg.norm(grid, axis=-1).argsort()
        grid = grid[idx]

    return grid


def get_rotation_arrays(magmoms, u=uz):
    dim = magmoms.shape[0]
    v = magmoms
    n = np.cross(u, v)
    n /= np.linalg.norm(n, axis=-1).reshape(dim, 1)
    z = np.repeat(u, dim, axis=0)
    A = np.stack([z, np.cross(n, z), n], axis=1)
    B = np.stack([v, np.cross(n, v), n], axis=1)
    R = np.einsum("nki,nkj->nij", A, B)

    Rnan = np.isnan(R)
    if Rnan.any():
        nanidx = np.where(Rnan)[0]
        R[nanidx] = I
        R[nanidx, 2] = v[nanidx]

    U = R[:, 0] + 1j * R[:, 1]
    V = R[:, 2]

    return U, V


def round_to_precision(array, precision):
    return precision * np.round(array / precision)
