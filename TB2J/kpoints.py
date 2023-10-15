import numpy as np


def monkhorst_pack(size, gamma_center=False):
    """Construct a uniform sampling of k-space of given size.
    Modified from ase.dft.kpoints with gamma_center option added"""
    if np.less_equal(size, 0).any():
        raise ValueError("Illegal size: %s" % list(size))
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    asize = np.array(size)
    shift = 0.5 * ((asize + 1) % 2) / asize
    mkpts = (kpts + 0.5) / size - 0.5
    if gamma_center:
        mkpts += shift

    return mkpts
