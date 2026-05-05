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
    parser.add_argument(
        "--animate",
        type=str,
        help="Load an exported magnon eigenstate file and build/render an animation scene",
    )
    parser.add_argument(
        "--export-format",
        nargs="+",
        choices=["json", "netcdf"],
        default=["json"],
        help="Data export format(s) for band/DOS calculations",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default=None,
        help="Prefix for exported magnon data files",
    )
    parser.add_argument(
        "--save-wavefunctions",
        action="store_true",
        help="Include wavefunctions in exported magnon data",
    )
    parser.add_argument(
        "--scene-output",
        type=str,
        default=None,
        help="Output JSON file for Three.js scene data",
    )
    parser.add_argument(
        "--k-index", type=int, default=0, help="Animation k-point index"
    )
    parser.add_argument(
        "--band-index", type=int, default=0, help="Animation band index"
    )
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="Animation amplitude"
    )
    parser.add_argument(
        "--frames", type=int, default=40, help="Number of animation frames"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Render animation scene with Streamlit instead of only writing scene JSON",
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
        "--qpoints",
        type=str,
        default=None,
        metavar="NAME:x,y,z",
        help=(
            "Custom q-points as comma-separated name:coord pairs. "
            "Format: 'G:0,0,0,X:0.5,0,0,M:0.5,0.5,0'. "
            "Overrides ASE default special points. "
            "Coordinates in fractional reciprocal lattice units."
        ),
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

    if args.animate:
        from TB2J.magnon.streamlit_viewer import build_scene_from_file, render_scene

        scene = build_scene_from_file(
            args.animate,
            kpoint_index=args.k_index,
            band_index=args.band_index,
            amplitude=args.amplitude,
            nframes=args.frames,
        )
        if args.scene_output:
            import json

            with open(args.scene_output, "w") as f:
                json.dump(scene, f, indent=2)
            print(f"Scene data saved to {args.scene_output}")
        if args.streamlit:
            render_scene(scene)
        return

    if not args.bands and not args.dos:
        parser.error("Please specify at least one of --bands or --dos")

    window = None
    if args.config:
        params = MagnonParameters.from_toml(args.config)
        if params.window is not None:
            window = params.window
    else:
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
            export_formats=args.export_format,
            export_prefix=args.export_prefix,
            save_wavefunctions=args.save_wavefunctions,
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
        qpoints = None
        if hasattr(args, "qpoints") and args.qpoints:
            from TB2J.magnon.magnon_parameters import parse_qpoints_string

            qpoints = parse_qpoints_string(args.qpoints)

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
            qpoints=qpoints,
            export_formats=params.export_formats,
            export_prefix=params.export_prefix,
            save_wavefunctions=params.save_wavefunctions,
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
            export_formats=params.export_formats,
            export_prefix=params.export_prefix,
            save_wavefunctions=params.save_wavefunctions,
        )
        plot_magnon_dos_from_TB2J(dos_params)


if __name__ == "__main__":
    main()
