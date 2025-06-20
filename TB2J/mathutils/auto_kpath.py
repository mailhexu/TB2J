#!/usr/bin/env python
from __future__ import division

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.cell import Cell
from ase.dft.kpoints import bandpath

# from minimulti.spin.hamiltonian import SpinHamiltonian
# from minimulti.spin.mover import SpinMover


def group_band_path(bp, eps=1e-8, shift=0.15):
    """Groups band paths by separating segments with small distances between points.

    Parameters
    ----------
    bp : ASE.cell.BandPath
        The band path object containing k-points and special points
    eps : float, optional
        The threshold distance below which points are considered to be in the same group.
        Default is 1e-8.
    shift : float, optional
        The shift distance to apply between different segments for visualization.
        Default is 0.15.

    Returns
    -------
    tuple
        Contains:
        - xlist : list of arrays
            The x-coordinates for each segment, shifted for visualization
        - kptlist : list of arrays
            The k-points for each segment
        - Xs : array
            The x-coordinates of special points, shifted for visualization
        - knames : list
            The names of special k-points
    """
    xs, Xs, knames = bp.get_linear_kpoint_axis()
    kpts = bp.kpts

    m = xs[1:] - xs[:-1] < eps
    segments = [0] + list(np.where(m)[0] + 1) + [len(xs)]

    # split Xlist
    xlist, kptlist = [], []
    for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
        kptlist.append(kpts[start:end])
        xlist.append(xs[start:end] + i * shift)

    m = Xs[1:] - Xs[:-1] < eps

    s = np.where(m)[0] + 1

    for i in s:
        Xs[i:] += shift

    return xlist, kptlist, Xs, knames


def test_group_band_path():
    """Visualize the band path grouping functionality.

    This function demonstrates how group_band_path works by:
    1. Creating a simple test case (H atom with rectangular cell)
    2. Generating a band path with 50 points
    3. Grouping the path segments using group_band_path
    4. Plotting each segment to visualize how segments are shifted

    The resulting plot shows:
    - Each path segment plotted separately
    - Special k-points labeled on x-axis
    - Segments shifted relative to each other for clarity

    Note: This is a visualization function rather than a unit test.
    It helps understand the band path grouping behavior visually.
    """
    atoms = ase.Atoms("H", cell=[1, 1, 2])
    bp = atoms.cell.bandpath(npoints=50)
    xlist, kptlist, Xs, knames, spk = group_band_path(bp)

    for x, k in zip(xlist, kptlist):
        plt.plot(x, x)

    plt.xticks(Xs, knames)
    plt.show()


def auto_kpath(cell, knames, kvectors=None, npoints=100, supercell_matrix=None):
    """Generates an automatic k-path for band structure calculations.

    Parameters
    ----------
    cell : array_like
        The unit cell vectors
    knames : list or None
        Names of the high-symmetry k-points. If None and kvectors is None,
        automatically determines the path. If kvectors is provided, these names
        correspond to the provided k-vectors.
    kvectors : array_like, optional
        Explicit k-vectors for the path. If None, vectors are determined from
        knames or automatically. Default is None.
    npoints : int, optional
        Number of k-points along the path. Default is 100.
    supercell_matrix : array_like, optional
        Transformation matrix for supercell calculations. If provided, k-points
        are transformed accordingly. Default is None.

    Returns
    -------
    tuple
        Contains:
        - xlist : list of arrays
            The x-coordinates for each path segment
        - kptlist : list of arrays
            The k-points for each segment
        - Xs : array
            The x-coordinates of special points
        - knames : list
            The names of special k-points
        - spk : dict
            Dictionary mapping k-point names to their coordinates
    """
    if knames is None and kvectors is None:
        # fully automatic k-path
        bp = Cell(cell).bandpath(npoints=npoints)
        spk = bp.special_points
        xlist, kptlist, Xs, knames = group_band_path(bp)
    elif knames is not None and kvectors is None:
        # user specified kpath by name
        bp = Cell(cell).bandpath(knames, npoints=npoints)
        spk = bp.special_points
        kpts = bp.kpts
        xlist, kptlist, Xs, knames = group_band_path(bp)
    else:
        # user spcified kpath and kvector.
        kpts, x, Xs = bandpath(kvectors, cell, npoints)
        spk = dict(zip(knames, kvectors))
        xlist = [x]
        kptlist = [kpts]

    if supercell_matrix is not None:
        kptlist = [np.dot(k, supercell_matrix) for k in kptlist]
    print("High symmetry k-points:")
    for name, k in spk.items():
        if name == "G":
            name = "Gamma"
        print(f"{name}: {k}")
    return xlist, kptlist, Xs, knames, spk
