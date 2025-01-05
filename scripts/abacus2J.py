#!/usr/bin/env python3
import argparse
import sys

from TB2J.interfaces import gen_exchange_abacus
from TB2J.versioninfo import print_license


def run_abacus2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="abacus2J: Using magnetic force theorem to calculate exchange parameter J from abacus Hamiltonian in the LCAO mode"
    )

    parser.add_argument(
        "--path", help="the path of the abacus calculation", default="./", type=str
    )

    parser.add_argument(
        "--suffix",
        help="the label of the abacus calculation. There should be an output directory called OUT.suffix",
        default="abacus",
        type=str,
    )

    parser.add_argument(
        "--elements",
        help="list of elements to be considered in Heisenberg model.",
        # ,  For each element, a postfixes can be used to specify the orbitals(Only with Siesta backend), eg. Fe_3d, or Fe_3d_4s ",
        default=None,
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--rcut",
        help="range of R. The default is all the commesurate R to the kmesh",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--efermi", help="Fermi energy in eV. For test only. ", default=None, type=float
    )
    parser.add_argument(
        "--kmesh",
        help="kmesh in the format of kx ky kz. Monkhorst pack. If all the numbers are odd, it is Gamma cenetered. (strongly recommended), Default: 5 5 5",
        type=int,
        nargs="*",
        default=[5, 5, 5],
    )
    parser.add_argument(
        "--emin",
        help="energy minimum below efermi, default -14 eV",
        type=float,
        default=-14.0,
    )

    parser.add_argument(
        "--use_cache",
        help="whether to use disk file for temporary storing wavefunctions and hamiltonian to reduce memory usage. Default: False",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--nz", help="number of integration steps. Default: 50", default=50, type=int
    )

    parser.add_argument(
        "--cutoff",
        help="The minimum of J amplitude to write, (in eV). Default: 1e-7 eV",
        default=1e-7,
        type=float,
    )

    parser.add_argument(
        "--exclude_orbs",
        help="the indices of wannier functions to be excluded from magnetic site. counting start from 0. Default is none.",
        default=[],
        type=int,
        nargs="+",
    )

    parser.add_argument(
        "--nproc",
        "--np",
        help="number of cpu cores to use in parallel, default: 1",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--description",
        help="add description of the calculatiion to the xml file. Essential information, like the xc functional, U values, magnetic state should be given.",
        type=str,
        default="Calculated with TB2J.",
    )

    parser.add_argument(
        "--orb_decomposition",
        default=False,
        action="store_true",
        help="whether to do orbital decomposition in the non-collinear mode. Default: False.",
    )

    parser.add_argument(
        "--fname",
        default="exchange.xml",
        type=str,
        help="exchange xml file name. default: exchange.xml",
    )

    parser.add_argument(
        "--output_path",
        help="The path of the output directory, default is TB2J_results",
        type=str,
        default="TB2J_results",
    )

    args = parser.parse_args()

    if args.elements is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        sys.exit()

    # include_orbs = {}

    gen_exchange_abacus(
        path=args.path,
        suffix=args.suffix,
        kmesh=args.kmesh,
        magnetic_elements=args.elements,
        include_orbs={},
        Rcut=args.rcut,
        emin=args.emin,
        nz=args.nz,
        description=args.description,
        output_path=args.output_path,
        use_cache=args.use_cache,
        nproc=args.nproc,
        exclude_orbs=args.exclude_orbs,
        orb_decomposition=args.orb_decomposition,
    )


if __name__ == "__main__":
    run_abacus2J()
