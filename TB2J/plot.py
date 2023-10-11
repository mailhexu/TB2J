import matplotlib.pyplot as plt
from TB2J.spinham.spin_api import SpinModel
from TB2J.io_exchange.io_exchange import SpinIO
import numpy as np
from TB2J import __version__
from TB2J.spinham.qsolver import QSolverASEWrapper
import argparse
import os


def write_eigen(qmesh, gamma=True, path="./", output_fname="EigenJq.txt", **kwargs):
    m = SpinIO.load_pickle(path)
    m.write_Jq(kmesh=qmesh, path=path, gamma=gamma, output_fname=output_fname, **kwargs)


def plot_magnon_band(
    fname="exchange.xml",
    path="./",
    npoints=301,
    show=True,
    kvectors=None,
    knames=None,
    figfname="magnon_band.pdf",
    supercell_matrix=np.eye(3),
    Jq=False,
    kpath_fname="exchange_kpth.txt",
    ax=None,
    **kwargs
):
    m = SpinModel(fname=fname, sc_matrix=None)
    m.set_ham(**kwargs)
    m.plot_magnon_band(
        kvectors=kvectors,
        knames=knames,
        npoints=npoints,
        kpath_fname=kpath_fname,
        Jq=Jq,
        supercell_matrix=supercell_matrix,
        ax=ax,
    )

    plt.savefig(figfname)
    if show:
        plt.show()


def plot_magnon_dos(
    fname="exchange.xml",
    path="./",
    npoints=301,
    window=None,
    smearing_width=0.1,
    kmesh=[9, 9, 9],
    gamma=True,
    show=True,
    figfname="magnon_dos.pdf",
    txt_filename="magnon_dos.txt",
    Jq=False,
    ax=None,
    **kwargs
):
    ffname = os.path.join(path, "exchange.xml")
    if not (os.path.exists(ffname) and os.path.isfile(ffname)):
        ffname = os.path.join(path, "Multibinit", "exchange.xml")
    model = SpinModel(ffname)
    solver = QSolverASEWrapper(model.ham)
    solver.set(kmesh=kmesh, gamma=gamma, Jq=Jq)
    solver.plot_dos(
        npts=npoints,
        window=window,
        output=figfname,
        smearing_width=smearing_width,
        ax=ax,
        show=show,
        dos_filename=txt_filename,
    )


def command_line_plot_magnon_dos():
    """
    map plot_magnon_dos to a shell command using argparse
    """
    parser = argparse.ArgumentParser(description="Plot magnon DOS")
    # parser.add_argument("fname", type=str, help="exchange.xml")
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./",
        help="path to exchange.xml, often the TB2J_results directory, or the Multibinit directory inside it.",
    )
    parser.add_argument(
        "-n", "--npoints", type=int, default=301, help="number of points in the energy"
    )
    parser.add_argument(
        "-w",
        "--window",
        type=float,
        nargs=2,
        default=None,
        help="energy window for the dos, two numbers giving the lower and upper bound",
    )
    parser.add_argument(
        "-k", "--kmesh", type=int, nargs=3, default=[9, 9, 9], help="k mesh"
    )

    parser.add_argument(
        "-s",
        "--smearing_width",
        type=float,
        default=10,
        help="Gauss smearing width in meV.",
    )

    parser.add_argument(
        "-g", "--gamma", action="store_true", help="Use gamma centered k mesh."
    )
    parser.add_argument(
        "-Jq",
        action="store_true",
        help="Plot the eigenvalues of J(q) instead of the magnon frequency.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the figure. By default it is not shown on the screen.",
    )
    parser.add_argument(
        "-f",
        "--fig_filename",
        type=str,
        default="magnon_dos.pdf",
        help="output filename for figure. Default: magnon_dos.pdf",
    )
    parser.add_argument(
        "-t",
        "--txt_filename",
        type=str,
        default="magnon_dos.txt",
        help="output filename of the data for the magnon DOS. Default: magnond_dos.txt",
    )
    args = parser.parse_args()
    plot_magnon_dos(
        # fname=args.fname,
        path=args.path,
        npoints=args.npoints,
        kmesh=args.kmesh,
        gamma=args.gamma,
        smearing_width=args.smearing_width / 1000,
        Jq=args.Jq,
        figfname=args.fig_filename,
        txt_filename=args.txt_filename,
        show=args.show,
    )
