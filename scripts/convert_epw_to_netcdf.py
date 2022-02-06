#!/usr/bin/env python3
from TB2J.epwparser import save_epmat_to_nc
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="convert epwdata to netcdf."
    )
    parser.add_argument('--path',
                        help="path of the epw data",
                        default='./',
                        type=str)
    parser.add_argument('--prefix',
                        help="prefix of the epw data",
                        type=str)

    parser.add_argument('--ncfile',
                        help="name of the output netcdf file",
                        default='epmat.nc',
                        type=str)

    args = parser.parse_args()
    save_epmat_to_nc(args.path, args.prefix, args.ncfile)


if __name__ == "__main__":
    main()
