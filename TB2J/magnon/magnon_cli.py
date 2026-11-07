"""Unified CLI for magnon band and DOS calculations."""

import argparse
import sys
import warnings

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J
from TB2J.magnon.magnon_parameters import (
    MagnonParameters,
    add_common_magnon_args,
    parse_common_args,
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

    topo_group = parser.add_argument_group("Topology options (opt-in, draft)")
    topo_group.add_argument(
        "--topology",
        action="store_true",
        default=False,
        help=(
            "Run the bulk magnon topology calculation (Berry-curvature flux "
            "and band Chern numbers) instead of bands/DOS. Requires a "
            "topology-ready model; unsupported interactions are refused."
        ),
    )
    topo_group.add_argument(
        "--topology-mesh",
        type=int,
        nargs=2,
        default=[12, 12],
        metavar=("NU", "NV"),
        help="Endpoint-exclusive reciprocal mesh for --topology (default: 12 12).",
    )
    topo_group.add_argument(
        "--topology-bands",
        type=str,
        default=None,
        help=(
            "Comma-separated zero-based band indices for --topology "
            "(default: all positive bands as one group)."
        ),
    )
    topo_group.add_argument(
        "--topology-groups",
        type=str,
        default=None,
        help=(
            "Semicolon-separated band groups for --topology, e.g. "
            "'0;1' requests two independent groups. Mutually exclusive "
            "with --topology-bands."
        ),
    )
    topo_group.add_argument(
        "--topology-refine",
        action="store_true",
        default=False,
        help=(
            "Add a mesh-refinement report (coarse vs factor-2 refined mesh) "
            "to the topology JSON under 'refinement'."
        ),
    )
    topo_group.add_argument(
        "--topology-q",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("Q1", "Q2", "Q3"),
        help="Single-Q ordering vector in input-cell reciprocal coordinates.",
    )
    topo_group.add_argument(
        "--topology-normal-k",
        type=float,
        default=0.0,
        help="Fixed normal reciprocal coordinate for the bulk slice.",
    )
    topo_group.add_argument(
        "--topology-out",
        type=str,
        default="topology_bulk.json",
        help="Output JSON path for --topology (default: topology_bulk.json).",
    )

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


def run_topology(args, params: MagnonParameters) -> int:
    """Opt-in bulk topology mode. Returns a process exit code."""
    import json

    from TB2J.io_exchange import SpinIO
    from TB2J.magnon.topology.bulk import BulkRequest, calculate_bulk
    from TB2J.magnon.topology.embedding import LatticeEmbedding
    from TB2J.magnon.topology.magnetic_cell import expand_magnetic_cell
    from TB2J.magnon.topology.preparation import (
        TopologyPreparationError,
        prepare_topology_model,
    )

    exc = SpinIO.load_pickle(path=params.path)
    include = {
        "Jiso": params.Jiso,
        "Jani": params.Jani,
        "DMI": params.DMI,
        "SIA": params.SIA,
    }
    try:
        model = prepare_topology_model(exc, include=include, source_path=params.path)
        if any(float(q) != 0.0 for q in args.topology_q):
            model = expand_magnetic_cell(model, args.topology_q)
    except TopologyPreparationError as error:
        print(f"Topology preparation refused the input: {error}")
        return 2
    bands = None
    groups = None
    if args.topology_groups:
        groups = [
            [int(b) for b in part.split(",")]
            for part in args.topology_groups.split(";")
        ]
    elif args.topology_bands:
        bands = [int(b) for b in args.topology_bands.split(",")]
    embedding = LatticeEmbedding(
        basis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        normal_k=args.topology_normal_k,
    )
    request = BulkRequest(
        mesh=tuple(args.topology_mesh),
        bands=bands,
        groups=groups,
        embedding=embedding,
    )
    result = calculate_bulk(model, request)
    record = result.to_json_dict()
    if args.topology_refine:
        from TB2J.magnon.topology.bulk import refine_bulk

        record["refinement"] = refine_bulk(model, request)
    with open(args.topology_out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"Topology result written to {args.topology_out}")
    diag = result.diagnostics
    if result.valid:
        for g in result.groups or []:
            print(f"Chern number for bands {g['bands']}: {g['chern']}")
            gap = g["min_external_gap"]
            if gap is not None:
                print(f"  minimum external gap: {gap:.6e} eV")
        return 0
    print("Topology result is INVALID; no Chern value emitted. Reasons:")
    for reason in diag.get("failures", []):
        print(f"  - {reason}")
    return 1


def main():
    """Main entry point for the unified magnon CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.save_config:
        params = MagnonParameters()
        params.to_toml(args.save_config)
        print(f"Saved default configuration to {args.save_config}")
        return

    if not args.bands and not args.dos and not args.topology:
        parser.error("Please specify at least one of --bands, --dos or --topology")

    window = None
    if args.config:
        params = MagnonParameters.from_toml(args.config)
        if params.window is not None:
            window = params.window
    else:
        if args.window is not None:
            window = tuple(args.window)
        params = parse_common_args(args)

    if args.topology:
        sys.exit(run_topology(args, params))

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
        )
        plot_magnon_dos_from_TB2J(dos_params)


if __name__ == "__main__":
    main()
