#!/usr/bin/env python3
import argparse
import os

from TB2J.io_exchange.io_exchange import SpinIO


def main():
    parser = argparse.ArgumentParser(
        description="Plot exchange parameters vs distance grouped by species."
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="./",
        help="Path to TB2J results directory or TB2J.pickle file.",
    )
    parser.add_argument(
        "-f",
        "--fname",
        type=str,
        default="JvsR_species.pdf",
        help="Output figure filename.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        pickle_path = os.path.join(args.path, "TB2J.pickle")
    else:
        pickle_path = args.path

    if not os.path.exists(pickle_path):
        print(f"Error: {pickle_path} not found.")
        return

    spinio = SpinIO.load_pickle(
        os.path.dirname(pickle_path), os.path.basename(pickle_path)
    )
    spinio.plot_JvsR_by_species(fname=args.fname, show=args.show)


if __name__ == "__main__":
    main()
