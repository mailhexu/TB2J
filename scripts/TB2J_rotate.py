#!/usr/bin/env python3
import argparse
from TB2J.rotate_atoms import rotate_atom_xyz, rotate_xyz, check_ftype


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('fname',
                        help="name of the file containing the atomic structure",
                        type=str)
    parser.add_argument('--ftype',
                        help="type of the output files, e.g.  xyz. Please use the ",
                        default='xyz',
                        type=str)

    args=parser.parse_args()
    rotate_xyz(args.fname, ftype=args.ftype)


if __name__ == "__main__":
    main()
