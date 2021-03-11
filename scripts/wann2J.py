#!/usr/bin/env python3
import argparse
import sys
from TB2J.manager import gen_exchange
from TB2J.versioninfo import print_license


def run_wann2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=
        "wann2J: Using magnetic force theorem to calculate exchange parameter J from wannier functions"
    )
    parser.add_argument('--path',
                        help="path to the wannier files",
                        default='./',
                        type=str)
    parser.add_argument('--posfile',
                        help="name of the position file",
                        default='POSCAR',
                        type=str)
    parser.add_argument('--prefix_spinor',
                        help="prefix to the spinor wannier files",
                        default='wannier90',
                        type=str)
    parser.add_argument('--prefix_up',
                        help="prefix to the spin up wannier files",
                        default='wannier90.up',
                        type=str)
    parser.add_argument('--prefix_down',
                        help="prefix to the spin down wannier files",
                        default='wannier90.dn',
                        type=str)
    parser.add_argument('--elements',
                        help="elements to be considered in Heisenberg model",
                        default=None,
                        type=str,
                        nargs='*')
    parser.add_argument(
        '--groupby',
        help=
        "In the spinor case, the order of the orbitals have two conventions: 1: group by spin (orb1_up, orb2_up,... orb1_down, ...), 2,group by orbital (orb1_up, orb1_down, orb2_up, ...,). Use 'spin' in the former case and 'orbital' in the latter case. The default is spin.",
        default='spin',
        type=str,
    )

    parser.add_argument(
        '--rcut',
        help=
        'cutoff of spin pair distance. The default is to calculate all commensurate R point to the k mesh.',
        default=None,
        type=float)
    parser.add_argument('--efermi',
                        help='Fermi energy in eV',
                        default=None,
                        type=float)
    parser.add_argument('--kmesh',
                        help='kmesh in the format of kx ky kz',
                        type=int,
                        nargs='*',
                        default=[5, 5, 5])
    parser.add_argument('--emin',
                        help='energy minimum below efermi, default -14 eV',
                        type=float,
                        default=-14.0)
    parser.add_argument('--emax',
                        help='energy maximum above efermi, default 0.0 eV',
                        type=float,
                        default=0.0)
    parser.add_argument(
        '--nz',
        help='number of steps for semicircle contour, default: 100',
        default=100,
        type=int)
    parser.add_argument(
        '--height',
        help=
        'energy contour, a small number (often between 0.1 to 0.5, default 0.1)',
        type=float,
        default=0.1)
    parser.add_argument('--nz1',
                        help='number of steps 1, default: 50',
                        default=50,
                        type=int)
    parser.add_argument('--nz2',
                        help='number of steps 2, default: 200',
                        default=200,
                        type=int)
    parser.add_argument('--nz3',
                        help='number of steps 3, default: 50',
                        default=50,
                        type=int)
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

    parser.add_argument('--np',
                        help='number of cpu cores to use in parallel, default: 1',
                        default=1,
                        type=int)


    parser.add_argument(
        '--use_cache',
        help=
        "whether to use disk file for temporary storing wavefunctions and hamiltonian to reduce memory usage. Default: False",
        action='store_true',
        default=False)

    parser.add_argument(
        "--description",
        help=
        "add description of the calculatiion to the xml file. Essential information, like the xc functional, U values, magnetic state should be given.",
        type=str,
        default="Calculated with TB2J.")

    parser.add_argument("--spinor",
                        action="store_true",
                        help="Whether to use spinor wannier function.",
                        default=False)
    parser.add_argument(
        "--output_path",
        help="The path of the output directory, default is TB2J_results",
        type=str,
        default="TB2J_results")

    args = parser.parse_args()

    if args.efermi is None:
        print("Please input fermi energy using --efermi ")
        sys.exit()
    if args.elements is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        sys.exit()

    gen_exchange(path=args.path,
                 colinear=(not args.spinor),
                 groupby=args.groupby,
                 posfile=args.posfile,
                 efermi=args.efermi,
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
                 use_cache=args.use_cache,
                 np=args.np,
                 description=args.description,
                 output_path=args.output_path,
                 exclude_orbs=args.exclude_orbs)


if __name__ == "__main__":
    run_wann2J()
