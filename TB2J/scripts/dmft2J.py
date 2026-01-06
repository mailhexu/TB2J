#!/usr/bin/env python3
import argparse

from TB2J.exchange import ExchangeDMFT
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
        "--prefix_SOC",
        help="prefix to Wannier90 non-colinear files",
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
    # Note: DMFTManager handles reading Wannier90 and wrapping with DMFT data
    dmft_manager = DMFTManager(
        path=args.path,
        prefix_SOC=args.prefix_SOC,
        atoms=None,  # Will be read from Wannier files
        dmft_file=args.dmft_file,
        output_path=args.output,
        **vars(args),  # Pass other args like kmesh, nspin, etc.
    )

    # Get model, basis, description
    dmft_model, basis, description = dmft_manager()

    # Run exchange calculation
    # We select ExchangeDMFT explicitly (not through Manager selection logic)
    print("Starting DMFT exchange calculation...")
    exchange = ExchangeDMFT(
        tbmodels=dmft_model,
        atoms=None,
        basis=basis,
        description=description,
        **vars(args),
    )

    exchange.run(path=args.output)


if __name__ == "__main__":
    run_dmft2J()
