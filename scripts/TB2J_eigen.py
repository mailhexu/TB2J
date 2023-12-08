#!/usr/bin/env python3
from TB2J.plot import write_eigen
from TB2J.versioninfo import print_license
import argparse

"""
The script to plot the magnon band structure.
"""


def write_eigen_info():
    print_license()
    parser = argparse.ArgumentParser(
        description="TB2J_eigen.py: Write the eigen values and eigen vectors to file."
    )
    parser.add_argument(
        "--path", default="./", type=str, help="The path of the TB2J_results file"
    )

    parser.add_argument(
        "--qmesh",
        help="qmesh in the format of kx ky kz. Monkhorst pack or Gamma-centered.",
        type=int,
        nargs="*",
        default=[8, 8, 8],
    )

    parser.add_argument(
        "--gamma",
        help="whether shift the qpoint grid to  Gamma-centered. Default: False",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--output_fname",
        type=str,
        help="The file name of the output. Default: eigenJq.txt",
        default="eigenJq.txt",
    )

    args = parser.parse_args()

    write_eigen(args.qmesh, args.gamma, output_fname=args.output_fname)


write_eigen_info()
