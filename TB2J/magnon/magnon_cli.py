"""Unified CLI for magnon band and DOS calculations."""

import argparse
import warnings

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J
from TB2J.magnon.magnon_parameters import (
    MagnonParameters,
    add_common_magnon_args,
    parse_common_args,
)
from TB2J.magnon.thermal_parameters import add_thermal_args


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

    add_thermal_args(parser)
    parser.add_argument(
        "--thermal-output",
        type=str,
        default=None,
        help="Output JSON file for the thermal-magnon result "
        "(default: <export-prefix>.thermal.json)",
    )

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
    group.add_argument(
        "--use-primitive-kpath",
        action="store_true",
        default=False,
        dest="use_primitive_kpath",
        help=(
            "Generate the high-symmetry k-path in the primitive-cell BZ and "
            "fold k-points into the supercell reciprocal lattice. Requires "
            "primitive_cell/supercell_matrix stored in the TB2J pickle "
            "(set by TB2J_edit supercell / make_supercell)."
        ),
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


def _thermal_band_kpoints(magnon, args):
    """Band k-points for explicit-temperature thermal runs.

    Reuses the band-path CLI options exactly like the ``--bands`` path:
    named ``--qpoints`` take precedence, then ``--kpath`` (ASE bandpath),
    then the auto-detected high-symmetry path.  With
    ``--use-primitive-kpath`` and a supercell-backed model, the path is
    generated in the primitive BZ and folded with the same
    ``k_sc = k_prim @ S.T`` convention as ``Magnon.get_magnon_bands``.
    """
    import numpy as np

    if getattr(args, "qpoints", None):
        from TB2J.magnon.magnon_parameters import parse_qpoints_string

        qpoints = parse_qpoints_string(args.qpoints)
        return np.array(list(qpoints.values()), dtype=float)

    npoints = getattr(args, "npoints", None) or 300
    path = getattr(args, "kpath", None)
    use_primitive = getattr(args, "use_primitive_kpath", False)
    if (
        use_primitive
        and magnon.primitive_cell is not None
        and magnon.supercell_matrix is not None
    ):
        from ase.cell import Cell as AseCell

        from TB2J.mathutils.auto_kpath import auto_kpath

        fold_matrix = np.array(magnon.supercell_matrix, dtype=float).T
        prim_cell_array = np.array(magnon.primitive_cell.get_cell())
        if path is None:
            _, kptlist, _, _, _ = auto_kpath(
                prim_cell_array,
                None,
                npoints=npoints,
                supercell_matrix=fold_matrix,
            )
            return np.concatenate(kptlist)
        k_prim = (
            AseCell(prim_cell_array)
            .bandpath(path=path, npoints=npoints, pbc=[True, True, True])
            .kpts
        )
        return k_prim @ fold_matrix
    if path is not None:
        return np.array(
            magnon.cell.bandpath(
                path=path, npoints=npoints, pbc=[True, True, True]
            ).kpts
        )
    from TB2J.mathutils.auto_kpath import auto_kpath

    _, kptlist, _, _, _ = auto_kpath(magnon.cell, None, npoints=npoints)
    return np.concatenate(kptlist)


def run_thermal_calculation(args, params):
    """Run the thermal-magnon solver and export its versioned JSON result.

    Reuses the band/DOS model preparation (TB2J results path, reference
    configuration), runs ``ThermalMagnonSolver`` for the selected
    ``--thermal-*`` method, and serializes the ``tb2j.magnon.thermal``
    result alongside the band/DOS outputs.
    """
    from TB2J.magnon.magnon_parameters import prepare_magnon_from_params
    from TB2J.magnon.thermal_parameters import thermal_parameters_from_args
    from TB2J.magnon.thermal_solver import ThermalMagnonSolver

    thermal_params = thermal_parameters_from_args(args)
    magnon = prepare_magnon_from_params(params)
    print(f"\nRunning thermal calculation (method: {thermal_params.thermal_method})...")
    solver = ThermalMagnonSolver(magnon, thermal_params)
    temperatures = list(thermal_params.thermal_temperatures)
    if thermal_params.thermal_method != "mfa" and temperatures:
        # Spectral methods emit temperature blocks only when band k-points
        # accompany the temperatures; reuse the band-path options (ADR 0006).
        result = solver.calculate(
            temperatures_K=temperatures,
            band_kpoints=_thermal_band_kpoints(magnon, args),
        )
    else:
        result = solver.calculate()
    filename = args.thermal_output
    if filename is None:
        prefix = params.export_prefix or "TB2J_magnon"
        filename = f"{prefix}.thermal.json"
    result.save_json(filename)
    transition = result.transition
    print(f"Thermal result written to {filename}")
    if transition is not None:
        print(
            f"  status: {result.status}, {transition.kind}: "
            f"{transition.temperature_K:.2f} K (converged: "
            f"{transition.converged}, method validity: "
            f"{transition.method_validity})"
        )
    return result


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

    thermal_requested = any(
        value is not None
        for name, value in vars(args).items()
        if name.startswith("thermal_")
    )
    if not args.bands and not args.dos and not thermal_requested:
        parser.error(
            "Please specify at least one of --bands, --dos, or a --thermal-* option"
        )

    window = None
    if args.config:
        params = MagnonParameters.from_toml(args.config)
        if params.window is not None:
            window = params.window
    else:
        if args.window is not None:
            window = tuple(args.window)
        params = parse_common_args(args)

    if thermal_requested:
        run_thermal_calculation(args, params)

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
            spin_conf=params.spin_conf,
            show=params.show,
            kpath=args.kpath,
            npoints=args.npoints,
            qpoints=qpoints,
            export_formats=params.export_formats,
            export_prefix=params.export_prefix,
            save_wavefunctions=params.save_wavefunctions,
            use_primitive_kpath=getattr(args, "use_primitive_kpath", False),
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
            spin_conf=params.spin_conf,
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
