#!/usr/bin/env python3
import argparse

from TB2J.exchange_dmft import ExchangeDMFTNCL
from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces.dmft import DMFTManager
from TB2J.versioninfo import print_license


def run_dmft2J():
    print_license()
    parser = argparse.ArgumentParser(
        description="dmft2J: Calculate exchange parameters using DMFT self-energies"
    )
    parser.add_argument(
        "--path", help="path to Wannier90 files", default="./", type=str
    )
    parser.add_argument(
        "--posfile", help="name of the position file", default="POSCAR", type=str
    )
    parser.add_argument(
        "--prefix",
        help="prefix to Wannier90 files",
        default="wannier90",
        type=str,
    )
    parser.add_argument(
        "--dmft_file",
        help="path to DMFT self-energy HDF5 file",
        required=True,
        type=str,
    )

    # Add standard exchange arguments
    parser = add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    # Initialize DMFT manager
    dmft_manager = DMFTManager(
        path=args.path,
        prefix=args.prefix,
        atoms=None,  # Will be read from Wannier files
        dmft_file=args.dmft_file,
        output_path=args.output,
        **vars(args),
    )

    # Get model, basis, description
    dmft_model, basis, description = dmft_manager()

    # Run exchange calculation
    print("Starting DMFT exchange calculation...")
    if args.nspin == 1:
        from TB2J.exchange import ExchangeCLDMFT

        ExchangeClass = ExchangeCLDMFT
    else:
        ExchangeClass = ExchangeDMFTNCL

    exchange = ExchangeClass(
        tbmodels=dmft_model,
        atoms=dmft_model.atoms,
        basis=basis,
        description=description,
        **vars(args),
    )

    exchange.run(path=args.output)


if __name__ == "__main__":
    run_dmft2J()
