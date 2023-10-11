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
    model = SpinModel(fname=os.path.join(path, "Multibinit", "exchange.xml"))
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
        "-p", "--path", type=str, default="./", help="path to exchange.xml"
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
        default=0.01,
        help="Gauss smearing width in eV.",
    )

    parser.add_argument(
        "-g", "--gamma", action="store_true", help="gamma centered k mesh"
    )
    parser.add_argument("-Jq", action="store_true", help="use Jq")
    parser.add_argument("--show", action="store_true", help="show the figure")
    parser.add_argument(
        "-f",
        "--fig_filename",
        type=str,
        default="magnon_dos.pdf",
        help="output filename for figure.",
    )
    parser.add_argument(
        "-t",
        "--txt_filename",
        type=str,
        default="magnon_dos.txt",
        help="output filename of the data for the magnon DOS.",
    )
    args = parser.parse_args()
    plot_magnon_dos(
        # fname=args.fname,
        path=args.path,
        npoints=args.npoints,
        kmesh=args.kmesh,
        gamma=args.gamma,
        smearing_width=args.smearing_width,
        Jq=args.Jq,
        figfname=args.fig_filename,
        txt_filename=args.txt_filename,
        show=args.show,
    )
