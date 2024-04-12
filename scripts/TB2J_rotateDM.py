#!/usr/bin/env python3
import argparse
from TB2J.rotate_siestaDM import rotate_DM

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--fdf_fname", help="Name of the *.fdf siesta file."
    )
    parser.add_argument(
        "--noncollinear",
        action="store_true",
        help="If present, six different configurations will be generated. These are required for non-collinear systems."
    )

    args = parser.parse_args()
    rotate_DM(args.fdf_fname, noncollinear=args.noncollinear)


if __name__ == "__main__":
    main()
