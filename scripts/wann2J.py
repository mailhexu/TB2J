#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from TB2J.manager import gen_exchange
from TB2J.plot import plot_magnon_band
from TB2J.versioninfo import print_license




def run_wann2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=
        "wann2J: Using magnetic force theorem to calculate exchange parameter J from wannier functions"
    )
    parser.add_argument(
        '--path', help="path to the wannier files", default='./', type=str)
    parser.add_argument(
        '--posfile',
        help="name of the position file",
        default='POSCAR',
        type=str)
    parser.add_argument(
        '--prefix_spinor',
        help="prefix to the spinor wannier files",
        default='wannier90',
        type=str)
    parser.add_argument(
        '--prefix_up',
        help="prefix to the spin up wannier files",
        default='wannier90.up',
        type=str)
    parser.add_argument(
        '--prefix_down',
        help="prefix to the spin down wannier files",
        default='wannier90.dn',
        type=str)
    parser.add_argument(
        '--elements',
        help="elements to be considered in Heisenberg model",
        default=None,
        type=str,
        nargs='*')
    parser.add_argument(
        '--rcut', help='cutoff of spin pair distance. The default is to calculate all commensurate R point to the k mesh.', default=None, type=float)
    parser.add_argument(
        '--efermi', help='Fermi energy in eV', default=None, type=float)
    parser.add_argument(
        '--ne',
        help=
        'number of total electrons(optional). If given, the integeration will stop when the number of electrons reaches this value.',
        type=float,
        default=None)

    parser.add_argument(
        '--kmesh',
        help='kmesh in the format of kx ky kz',
        type=int,
        nargs='*',
        default=[5, 5, 5])
    parser.add_argument(
        '--emin',
        help='energy minimum below efermi, default -14 eV',
        type=float,
        default=-14.0)
    parser.add_argument(
        '--emax',
        help='energy maximum above efermi, default 0.0 eV',
        type=float,
        default=0.0)
    parser.add_argument(
        '--nz', help='number of steps for semicircle contour, default: 100', default=100, type=int)
    parser.add_argument(
        '--height',
        help=
        'energy contour, a small number (often between 0.1 to 0.5, default 0.1)',
        type=float,
        default=0.1)
    parser.add_argument(
        '--nz1', help='number of steps 1, default: 50', default=50, type=int)
    parser.add_argument(
        '--nz2', help='number of steps 2, default: 200', default=200, type=int)
    parser.add_argument(
        '--nz3', help='number of steps 3, default: 50', default=50, type=int)
    parser.add_argument(
        '--cutoff',
        help="The minimum of J amplitude to write, (in eV), default is 1e-5 eV",
        default=1e-5,
        type=float)
    parser.add_argument(
        '--exclude_orbs',
        help=
        "the indices of wannier functions to be excluded from magnetic site. counting start from 0",
        default=[],
        type=int,
        nargs='+')

    parser.add_argument(
        "--plot",
        action="store_true",
        help=
        "Plot magnon energy band instead of calculating J. Path will be automatically selected."
    )

    parser.add_argument(
        "--description",
        help=
        "add description of the calculatiion to the xml file. Essential information, like the xc functional, U values, magnetic state should be given.",
        type=str,
        default="Calculated with TB2J.")

    parser.add_argument(
        "--fname",
        default='exchange.xml',
        type=str,
        help='exchange xml file name. default: exchange.xml')
    parser.add_argument(
        "--show",
        action="store_true",
        help="whether to show magnon band structure.",
        default=False)

    parser.add_argument(
        "--spinor",
        action="store_true",
        help="Whether to use spinor wannier function.",
        default=False)

    args = parser.parse_args()

    if args.plot:
       plot_magnon_band(fname=args.fname, npoints=301, show=args.show)
    else:
        if args.efermi is None:
            print("Please input fermi energy using --efermi ")
            exit()
        if args.elements is None:
            print("Please input the magnetic elements, e.g. --elements Fe Ni")
            exit()
        gen_exchange(
            path=args.path,
            colinear=(not args.spinor),
            posfile=args.posfile,
            efermi=args.efermi,
            ne=args.ne,
            kmesh=args.kmesh,
            magnetic_elements=args.elements,
            Rcut=args.rcut,
            prefix_SOC=args.prefix_spinor,
            prefix_up=args.prefix_up,
            prefix_dn=args.prefix_down,
            emin=args.emin,
            emax=args.emax,
            nz=args.nz,
            height=args.height,
            nz1=args.nz1,
            nz2=args.nz2,
            nz3=args.nz3,
            description=args.description,
            exclude_orbs=args.exclude_orbs)


if __name__ == "__main__":
    run_wann2J()
