#!/usr/bin/env python3
import argparse

from TB2J.io_merge import merge
from TB2J.versioninfo import print_license


def main():
    parser = argparse.ArgumentParser(
        description="TB2J_merge: merge the exchange parameters calculated from different directions. It can accept two types of calculations, first, the lattice structures are rotated from the original structure, from z to x, y, and z, respectively. The structures could be generated using the TB2J_rotate.py command. The second type is the three calculations have the spins along the x, y, and z axis, respectively."
    )
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="The three directories the TB2J calculations are done. Inside each of them, there should be a TB2J_results directory which contains the magnetic interaction parameter files. e.g. Fe_x Fe_y Fe_z. Alternatively, it could  be the TB2J results directories.",
    )

    parser.add_argument(
        "--type",
        "-T",
        type=str,
        help="The type of calculations, either structure of spin, meaning that the three calculations are done by rotating the structure/spin. ",
    )
    parser.add_argument(
        "--output_path",
        help="The path of the output directory, default is TB2J_results",
        type=str,
        default="TB2J_results",
    )
    parser.add_argument(
        "--main_path",
        help="The path containning the reference structure.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    # merge(*(args.directories), args.type.strip().lower(), path=args.output_path)
    # merge(*(args.directories), method=args.type.strip().lower(), path=args.output_path)
    # merge2(args.directories, args.type.strip().lower(), path=args.output_path)
    print_license()
    print("Merging the TB2J results from the following directories: ", args.directories)
    merge(*args.directories, main_path=args.main_path, write_path=args.output_path)
    print("Merging completed. The results are saved in:", args.output_path)


if __name__ == "__main__":
    main()
