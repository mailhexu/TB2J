#!/usr/bin/env python3
import argparse
import os
import sys
from TB2J.io_merge import merge


def main():
    parser = argparse.ArgumentParser(
        description=
        "TB2J_merge: merge the exchange parameters calculated from different directions. It can accept two types of calculations, first, the lattice structures are rotated from the original structure, from z to x, y, and z, respectively. The structures could be generated using the TB2J_rotate.py command. The second type is the three calculations have the spins along the x, y, and z axis, respectively."
    )
    parser.add_argument(
        'directories',
        type=str,
        nargs='+',
        help=
        'The three directories the TB2J calculations are done. Inside each of them, there should be a TB2J_results directory which contains the magnetic interaction parameter files. e.g. Fe_x Fe_y Fe_z'
    )
    parser.add_argument(
        '--type',
        '-T',
        type=str,
        help=
        'The type of calculations, either structure of spin, meaning that the three calculations are done by rotating the structure/spin. '
    )
    #merge(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], save=True)
    args = parser.parse_args()
    merge(*(args.directories), args.type.strip().lower())


main()
