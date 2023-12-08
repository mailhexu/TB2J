#!/usr/bin/env python
from __future__ import division
import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.dft.kpoints import get_special_points, parse_path_string

# from minimulti.spin.hamiltonian import SpinHamiltonian
# from minimulti.spin.mover import SpinMover


def group_band_path(bp, eps=1e-8, shift=0.15):
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
    atoms = ase.Atoms("H", cell=[1, 1, 2])
    bp = atoms.cell.bandpath(npoints=50)
    xlist, kptlist, Xs, knames = group_band_path(bp)

    for x, k in zip(xlist, kptlist):
        plt.plot(x, x)

    plt.xticks(Xs, knames)
    plt.show()


def plot_3d_vector(positions, vectors, length=0.1):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    ax.quiver(
        x, y, z, u, v, w, length=length, normalize=False, pivot="middle", cmap="seismic"
    )
    plt.show()


def plot_2d_vector(positions, vectors, show_z=True, length=0.1, ylimit=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = positions.shape[1]
    x, y, z = positions.T
    u, v, w = vectors.T
    # ax.streamplot(x, y, u, v,  linewidth=1, cmap=plt.cm.inferno,
    #    density=2, arrowstyle='->', arrowsize=1.5)
    ax.scatter(x, y, s=50, color="r")
    if show_z:
        ax.quiver(
            x,
            y,
            u,
            v,
            w,
            length=length,
            units="width",
            pivot="middle",
            cmap="seismic",
        )
    else:
        ax.quiver(x, y, u, v, units="width", pivot="middle", cmap="seismic")
    if ylimit is not None:
        ax.set_ylim(ylimit[0], ylimit[1])
    # plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_supercell(
    ham, supercell_matrix=np.diag([30, 1, 1]), plot_type="2d", length=0.1, ylimit=None
):
    sc_ham = ham.make_supercell(supercell_matrix)
    sc_ham.s = np.random.rand(*sc_ham.s.shape) - 0.5
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(time_step=3e-4, temperature=0, total_time=6, save_all_spin=False)

    mover.run(write_step=20)
    pos = np.dot(sc_ham.pos, sc_ham.cell)
    if plot_type == "2d":
        plot_2d_vector(pos, mover.s, show_z=False, length=length, ylimit=ylimit)
    elif plot_type == "3d":
        plot_3d_vector(pos, mover.s, length=length)


# exchange_1d()

from ase.dft.kpoints import *


def mybandpath(path, cell, npoints=50, eps=1e-3):
    """Make a list of kpoints defining the path between the given points.

    path: list or str
        Can be:

        * a string that parse_path_string() understands: 'GXL'
        * a list of BZ points: [(0, 0, 0), (0.5, 0, 0)]
        * or several lists of BZ points if the the path is not continuous.
    cell: 3x3
        Unit cell of the atoms.
    npoints: int
        Length of the output kpts list.

    Return list of k-points, list of x-coordinates and list of
    x-coordinates of special points."""

    if isinstance(path, str):
        special = get_special_points(cell, eps=eps)
        paths = []
        for names in parse_path_string(path):
            paths.append([special[name] for name in names])
    elif np.array(path[0]).ndim == 1:
        paths = [path]
    else:
        paths = path

    points = np.concatenate(paths)
    dists = points[1:] - points[:-1]
    lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]

    i = 0
    for path in paths[:-1]:
        i += len(path)
        lengths[i - 1] = 0

    length = sum(lengths)
    kpts = []
    x0 = 0
    x = []
    X = [0]
    for P, d, L in zip(points[:-1], dists, lengths):
        n = max(2, int(round(L * (npoints - len(x)) / (length - x0))))

        for t in np.linspace(0, 1, n)[:-1]:
            kpts.append(P + t * d)
            x.append(x0 + t * L)
        x0 += L
        X.append(x0)
    kpts.append(points[-1])
    x.append(x0)

    return np.array(kpts), np.array(x), np.array(X)


def fix_cell(cell, eps=5e-3):
    cellpar = cell_to_cellpar(cell)
    for i in [3, 4, 5]:
        if abs(cellpar[i] / 90.0 * np.pi / 2 - np.pi / 2) < eps:
            cellpar[i] = 90.0
    return cellpar_to_cell(cellpar)


def plot_M_vs_time(ham, supercell_matrix=np.eye(3), temperature=0):
    sc_ham = ham.make_supercell(supercell_matrix)
    mover = SpinMover(hamiltonian=sc_ham)
    mover.set(time_step=1e-5, temperature=temperature, total_time=1, save_all_spin=True)

    mover.run(write_step=10)

    hist = mover.get_hist()
    hspin = np.array(hist["spin"])
    time = np.array(hist["time"])
    tspin = np.array(hist["total_spin"])

    Ms = np.linalg.det(supercell_matrix)
    plt.figure()
    plt.plot(time, np.linalg.norm(tspin, axis=1) / Ms, label="total", color="black")
    plt.plot(time, tspin[:, 0] / Ms, label="x")
    plt.plot(time, tspin[:, 1] / Ms, label="y")
    plt.plot(time, tspin[:, 2] / Ms, label="z")
    plt.xlabel("time (s)")
    plt.ylabel("magnetic moment ($\mu_B$)")
    plt.legend()
    # plt.show()
    # avg_total_m = np.average((np.linalg.norm(tspin, axis=1)/Ms)[:])
    plt.show()


def plot_M_vs_T(ham, supercell_matrix=np.eye(3), Tlist=np.arange(0.0, 110, 20)):
    Mlist = []
    for temperature in Tlist:
        sc_ham = ham.make_supercell(supercell_matrix)
        mover = SpinMover(hamiltonian=sc_ham)
        mover.set(
            time_step=2e-4,
            # damping_factor=0.1,
            temperature=temperature,
            total_time=1,
            save_all_spin=True,
        )

        mover.run(write_step=10)

        hist = mover.get_hist()
        hspin = np.array(hist["spin"])
        time = np.array(hist["time"])
        tspin = np.array(hist["total_spin"])

        Ms = np.linalg.det(supercell_matrix)
        avg_total_m = np.average((np.linalg.norm(tspin, axis=1) / Ms)[300:])
        print("T: %s   M: %s" % (temperature, avg_total_m))
        Mlist.append(avg_total_m)

    plt.plot(Tlist, Mlist)
    plt.ylim(-0.01, 1.01)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Average magnetization (Ms)")
    plt.show()
