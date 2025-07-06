#!/usr/bin/env python3
"""Command-line tool for plotting magnon DOS from TB2J results."""

import argparse
from pathlib import Path

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.magnon_dos import plot_magnon_dos


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot magnon DOS from TB2J results"
    )
    parser.add_argument(
        "--path",
        default="TB2J_results",
        help="Path to TB2J results directory (default: TB2J_results)",
    )
    parser.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=[20, 20, 20],
        metavar=("nx", "ny", "nz"),
        help="k-point mesh dimensions (default: 20, 20, 20)",
    )
    parser.add_argument(
        "--no-gamma",
        action="store_false",
        dest="gamma",
        help="Exclude Gamma point from k-mesh",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0.001,
        help="Gaussian smearing width in eV (default: 0.001)",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("emin", "emax"),
        help="Energy window in meV (optional)",
    )
    parser.add_argument(
        "--npts",
        type=int,
        default=401,
        help="Number of energy points (default: 401)",
    )
    parser.add_argument(
        "--output",
        default="magnon_dos.png",
        help="Output filename for plot (default: magnon_dos.png)",
    )
    parser.add_argument(
        "-show",
        action="store_true",
        dest="show",
        help="Show plot window",
    )

    args = parser.parse_args()

    # Check if TB2J results exist
    if not Path(args.path).exists():
        raise FileNotFoundError(f"TB2J results not found at {args.path}")

    # Load magnon calculator
    print(f"Loading exchange parameters from {args.path}...")
    magnon = Magnon.from_TB2J_results(path=args.path)

    # Convert window from meV to eV if provided
    window = None
    if args.window is not None:
        window = (args.window[0] / 1000, args.window[1] / 1000)

    # Calculate and plot DOS
    print("\nCalculating magnon DOS...")
    _dos = plot_magnon_dos(
        magnon,
        kmesh=args.kmesh,
        gamma=args.gamma,
        width=args.width,
        window=window,
        npts=args.npts,
        filename=args.output,
        show=args.show,
    )

    print(f"\nPlot saved to {args.output}")
    data_file = Path(args.output).with_suffix(".json")
    print(f"DOS data saved to {data_file}")


if __name__ == "__main__":
    main()
