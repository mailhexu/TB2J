#!/usr/bin/env python3
import argparse

from TB2J.rotate_atoms import rotate_xyz


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "fname", help="name of the file containing the atomic structure", type=str
    )
    parser.add_argument(
        "--ftype",
        help="type of the output files, e.g.  xyz. Please use the format which contains the full cell matrix. (e.g. .cif file should not be used) ",
        default="xyz",
        type=str,
    )
    parser.add_argument(
        "--noncollinear",
        action="store_true",
        help="If present, six different configurations will be generated. These are required for non-collinear systems.",
    )

    args = parser.parse_args()
    rotate_xyz(args.fname, ftype=args.ftype, noncollinear=args.noncollinear)


if __name__ == "__main__":
    main()
