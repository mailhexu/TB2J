#!/usr/bin/env python3
import argparse
import sys

from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces import gen_exchange_abacus
from TB2J.versioninfo import print_license


def run_abacus2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="abacus2J: Using magnetic force theorem to calculate exchange parameter J from abacus Hamiltonian in the LCAO mode"
    )
    # Add ABACUS specific arguments
    parser.add_argument(
        "--path", help="the path of the abacus calculation", default="./", type=str
    )
    parser.add_argument(
        "--suffix",
        help="the label of the abacus calculation. There should be an output directory called OUT.suffix",
        default="abacus",
        type=str,
    )

    # Add common exchange arguments
    parser = add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    index_magnetic_atoms = args.index_magnetic_atoms
    if index_magnetic_atoms is not None:
        index_magnetic_atoms = [i - 1 for i in index_magnetic_atoms]

    if args.elements is None and index_magnetic_atoms is None:
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
        nproc=args.np,
        exclude_orbs=args.exclude_orbs,
        orb_decomposition=args.orb_decomposition,
        index_magnetic_atoms=index_magnetic_atoms,
    )


if __name__ == "__main__":
    run_abacus2J()
