"""Unified CLI for magnon band and DOS calculations."""

import argparse
import warnings

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J
from TB2J.magnon.magnon_parameters import (
    MagnonParameters,
    add_common_magnon_args,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the unified magnon CLI."""
    parser = argparse.ArgumentParser(
        description="Calculate and plot magnon band structure and/or DOS from TB2J results"
    )

    parser.add_argument(
        "--bands",
        action="store_true",
        help="Plot magnon band structure",
    )
    parser.add_argument(
        "--dos",
        action="store_true",
        help="Plot magnon density of states",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--config",
        type=str,
        help="Path to TOML configuration file",
    )
    group.add_argument(
        "--save-config",
        type=str,
        help="Save default configuration to specified TOML file",
    )

    add_common_magnon_args(parser)

    band_group = parser.add_argument_group("Band structure options")
    add_band_specific_args_to_group(band_group)

    dos_group = parser.add_argument_group("DOS options")
    add_dos_specific_args_to_group(dos_group)

    return parser


def add_band_specific_args_to_group(group) -> None:
    """Add band-specific arguments to an argument group."""
    group.add_argument(
        "-k",
        "--kpath",
        default=None,
        help="k-path specification (default: auto-detected from cell type)",
    )
    group.add_argument(
        "--npoints",
        type=int,
        default=300,
        help="Number of k-points along the path (default: 300)",
    )
    group.add_argument(
        "--band-output",
        default="magnon_bands.png",
        help="Output file name for band structure (default: magnon_bands.png)",
    )


def add_dos_specific_args_to_group(group) -> None:
    """Add DOS-specific arguments to an argument group."""
    group.add_argument(
        "--kmesh",
        type=int,
        nargs=3,
        default=[20, 20, 20],
        metavar=("nx", "ny", "nz"),
        help="k-point mesh dimensions (default: 20, 20, 20)",
    )
    group.add_argument(
        "--no-gamma",
        action="store_false",
        dest="gamma",
        help="Exclude Gamma point from k-mesh",
    )
    group.add_argument(
        "--width",
        type=float,
        default=0.001,
        help="Gaussian smearing width in eV (default: 0.001)",
    )
    group.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("emin", "emax"),
        help="Energy window in meV (optional)",
    )
    group.add_argument(
        "--npts",
        type=int,
        default=401,
        help="Number of energy points (default: 401)",
    )
    group.add_argument(
        "--dos-output",
        default="magnon_dos.png",
        help="Output filename for DOS plot (default: magnon_dos.png)",
    )


def main():
    """Main entry point for the unified magnon CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.save_config:
        params = MagnonParameters()
        params.to_toml(args.save_config)
        print(f"Saved default configuration to {args.save_config}")
        return

    if not args.bands and not args.dos:
        parser.error("Please specify at least one of --bands or --dos")

    if args.config:
        params = MagnonParameters.from_toml(args.config)
    else:
        window = None
        if args.window is not None:
            window = tuple(args.window)
        params = MagnonParameters(
            path=args.path,
            Jiso=args.Jiso,
            Jani=args.Jani,
            SIA=args.SIA,
            DMI=args.DMI,
            spin_conf_file=args.spin_conf_file,
            show=args.show,
        )

    if args.bands:
        warnings.warn(
            """
            # !!!!!!!!!!!!!!!!!! WARNING: =============================
            # 
            # This functionality is under development and should not be used in production.
            # It is provided for testing and development purposes only.
            # Please use with caution and report any issues to the developers.
            #
            # This warning will be removed in future releases.
            # =====================================
            """,
            UserWarning,
            stacklevel=2,
        )
        band_params = MagnonParameters(
            path=params.path,
            filename=args.band_output,
            Jiso=params.Jiso,
            Jani=params.Jani,
            SIA=params.SIA,
            DMI=params.DMI,
            Q=params.Q,
            uz_file=params.uz_file,
            n=params.n,
            spin_conf_file=params.spin_conf_file,
            show=params.show,
            kpath=args.kpath,
            npoints=args.npoints,
        )
        plot_magnon_bands_from_TB2J(band_params)

    if args.dos:
        dos_params = MagnonParameters(
            path=params.path,
            filename=args.dos_output,
            Jiso=params.Jiso,
            Jani=params.Jani,
            SIA=params.SIA,
            DMI=params.DMI,
            Q=params.Q,
            uz_file=params.uz_file,
            n=params.n,
            spin_conf_file=params.spin_conf_file,
            show=params.show,
            kmesh=args.kmesh,
            gamma=args.gamma,
            width=args.width,
            window=window,
            npts=args.npts,
        )
        plot_magnon_dos_from_TB2J(dos_params)


if __name__ == "__main__":
    main()
