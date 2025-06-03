#!/usr/bin/env python3
from TB2J.magnon import plot_tb2j_magnon_bands


def plot_tb2j_magnon_bands_cli():
    """Command line interface for plotting magnon band structures from TB2J data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot magnon band structure from TB2J data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-f", "--file", default="TB2J.pickle", help="Path to TB2J pickle file"
    )

    parser.add_argument(
        "-p",
        "--path",
        help="High-symmetry k-point path (e.g., 'GXMG' for square lattice)",
    )

    parser.add_argument(
        "-n",
        "--npoints",
        type=int,
        default=300,
        help="Number of k-points for band structure calculation",
    )

    parser.add_argument(
        "-a",
        "--anisotropic",
        action="store_true",
        help="Include anisotropic interactions",
    )

    parser.add_argument(
        "-q",
        "--quadratic",
        action="store_true",
        help="Include biquadratic interactions",
    )

    parser.add_argument("-o", "--output", help="Output filename for the plot")

    parser.add_argument(
        "--no-pbc",
        nargs="+",
        type=int,
        choices=[0, 1, 2],
        help="Disable periodic boundary conditions in specified directions (0=x, 1=y, 2=z)",
    )

    args = parser.parse_args()

    # Set up periodic boundary conditions
    pbc = [True, True, True]
    if args.no_pbc:
        for direction in args.no_pbc:
            pbc[direction] = False
    pbc = tuple(pbc)

    # Plot the bands
    plot_tb2j_magnon_bands(
        pickle_file=args.file,
        path=args.path,
        npoints=args.npoints,
        anisotropic=args.anisotropic,
        quadratic=args.quadratic,
        pbc=pbc,
        filename=args.output,
    )


if __name__ == "__main__":
    plot_tb2j_magnon_bands_cli()
