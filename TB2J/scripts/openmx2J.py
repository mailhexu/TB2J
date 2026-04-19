#!/usr/bin/env python3
import argparse
import sys

from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces import gen_exchange_openmx
from TB2J.versioninfo import print_license


def run_openmx2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="openmx2J: Using magnetic force theorem to calculate exchange parameter J from OpenMX Hamiltonian"
    )
    # Add OpenMX specific arguments
    parser.add_argument(
        "--path",
        help="the path of the OpenMX calculation",
        default="./",
        type=str,
    )
    parser.add_argument(
        "--prefix",
        help="the prefix of the OpenMX scfout and xyz files",
        default="openmx",
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

    gen_exchange_openmx(
        path=args.path,
        prefix=args.prefix,
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
    run_openmx2J()
