#!/usr/bin/env python3
"""Command-line tool for plotting magnon DOS from TB2J results."""

import argparse
from pathlib import Path

import numpy as np

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.magnon_dos import plot_magnon_dos


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot magnon DOS from TB2J results"
    )
    parser.add_argument(
        "-p",
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
        "-o",
        "--output",
        default="magnon_dos.png",
        help="Output filename for plot (default: magnon_dos.png)",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        dest="show",
        help="Show plot window",
    )

    # Exchange interaction options (same as in magnon bands)
    parser.add_argument(
        "-j",
        "--Jiso",
        action="store_true",
        default=True,
        help="Include isotropic exchange interactions (default: True)",
    )
    parser.add_argument(
        "--no-Jiso",
        action="store_false",
        dest="Jiso",
        help="Exclude isotropic exchange interactions",
    )
    parser.add_argument(
        "-a",
        "--Jani",
        action="store_true",
        default=False,
        help="Include anisotropic exchange interactions (default: False)",
    )
    parser.add_argument(
        "-d",
        "--DMI",
        action="store_true",
        default=False,
        help="Include Dzyaloshinskii-Moriya interactions (default: False)",
    )

    # Reference vector options (same as in magnon bands)
    parser.add_argument(
        "-q",
        "--Q",
        nargs=3,
        type=float,
        metavar=("Qx", "Qy", "Qz"),
        help="Propagation vector [Qx, Qy, Qz] (default: [0, 0, 0])",
    )
    parser.add_argument(
        "-u",
        "--uz-file",
        type=str,
        help="Path to file containing quantization axes for each spin (nspin×3 array)",
    )
    parser.add_argument(
        "-c",
        "--spin-conf-file",
        type=str,
        help="Path to file containing magnetic moments for each spin (nspin×3 array)",
    )
    parser.add_argument(
        "-v",
        "--n",
        nargs=3,
        type=float,
        metavar=("nx", "ny", "nz"),
        help="Normal vector for rotation [nx, ny, nz] (default: [0, 0, 1])",
    )

    args = parser.parse_args()

    # Check if TB2J results exist
    if not Path(args.path).exists():
        raise FileNotFoundError(f"TB2J results not found at {args.path}")

    # Load magnon calculator with exchange interaction options
    print(f"Loading exchange parameters from {args.path}...")
    magnon = Magnon.from_TB2J_results(
        path=args.path, Jiso=args.Jiso, Jani=args.Jani, DMI=args.DMI
    )

    # Set reference vectors if provided (same logic as in magnon bands)
    Q = [0, 0, 0] if args.Q is None else args.Q
    n = [0, 0, 1] if args.n is None else args.n

    # Handle quantization axes
    if args.uz_file is not None:
        # Make path relative to TB2J results if not absolute
        uz_file = args.uz_file
        if not Path(uz_file).is_absolute():
            uz_file = str(Path(args.path) / uz_file)

        uz = np.loadtxt(uz_file)
        if uz.shape[1] != 3:
            raise ValueError(
                f"Quantization axes file should contain a nspin×3 array. Got shape {uz.shape}"
            )
        if uz.shape[0] != magnon.nspin:
            raise ValueError(
                f"Number of spins in uz file ({uz.shape[0]}) does not match the system ({magnon.nspin})"
            )
    else:
        # Default: [0, 0, 1] for all spins
        uz = np.array([[0, 0, 1]], dtype=float)

    # Handle spin configuration
    if args.spin_conf_file is not None:
        # Make path relative to TB2J results if not absolute
        spin_conf_file = args.spin_conf_file
        if not Path(spin_conf_file).is_absolute():
            spin_conf_file = str(Path(args.path) / spin_conf_file)

        magmoms = np.loadtxt(spin_conf_file)
        if magmoms.shape[1] != 3:
            raise ValueError(
                f"Spin configuration file should contain a nspin×3 array. Got shape {magmoms.shape}"
            )
        if magmoms.shape[0] != magnon.nspin:
            raise ValueError(
                f"Number of spins in spin configuration file ({magmoms.shape[0]}) does not match the system ({magnon.nspin})"
            )
    else:
        magmoms = None

    # Set reference configuration
    magnon.set_reference(Q, uz, n, magmoms)

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
