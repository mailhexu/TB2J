#!/usr/bin/env python3
import argparse
import sys

from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces import gen_exchange_siesta
from TB2J.versioninfo import print_license


def run_siesta2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="siesta2J: Using magnetic force theorem to calculate exchange parameter J from siesta Hamiltonian"
    )
    # Add siesta specific arguments
    parser.add_argument(
        "--fdf_fname", help="path of the input fdf file", default="./", type=str
    )
    parser.add_argument(
        "--fname",
        default="exchange.xml",
        type=str,
        help="exchange xml file name. default: exchange.xml",
    )
    parser.add_argument(
        "--split_soc",
        help="whether the SOC part of the Hamiltonian can be read from the output of siesta. Default: False",
        action="store_true",
        default=False,
    )

    # Add common exchange arguments
    parser = add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    if args.elements is None:
        print("Please input the magnetic elements, e.g. --elements Fe Ni")
        sys.exit()

    # include_orbs = {}
    # for element in args.elements:
    #    if "_" in element:
    #        elem = element.split("_")[0]
    #        orb = element.split("_")[1:]
    #        include_orbs[elem] = orb
    #    else:
    #        include_orbs[element] = None

    index_magnetic_atoms = args.index_magnetic_atoms
    if index_magnetic_atoms is not None:
        index_magnetic_atoms = [i - 1 for i in index_magnetic_atoms]

    gen_exchange_siesta(
        fdf_fname=args.fdf_fname,
        kmesh=args.kmesh,
        # magnetic_elements=list(include_orbs.keys()),
        # include_orbs=include_orbs,
        magnetic_elements=args.elements,
        include_orbs={},
        Rcut=args.rcut,
        emin=args.emin,
        emax=args.emax,
        nz=args.nz,
        description=args.description,
        output_path=args.output_path,
        use_cache=args.use_cache,
        nproc=args.np,
        exclude_orbs=args.exclude_orbs,
        orb_decomposition=args.orb_decomposition,
        read_H_soc=args.split_soc,
        orth=args.orth,
        index_magnetic_atoms=index_magnetic_atoms,
    )


if __name__ == "__main__":
    run_siesta2J()
