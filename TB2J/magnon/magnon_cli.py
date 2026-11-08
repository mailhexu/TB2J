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

    topo_group = parser.add_argument_group("Topology options (opt-in)")
    topo_group.add_argument(
        "--topology",
        action="store_true",
        default=False,
        help=(
            "Run bulk topology or, with --topology-strip-width, an open-strip "
            "magnon spectrum instead of bands/DOS."
        ),
    )
    topo_group.add_argument(
        "--topology-mesh",
        type=int,
        nargs=2,
        default=[12, 12],
        metavar=("NU", "NV"),
        help="Endpoint-exclusive reciprocal mesh for bulk topology (default: 12 12).",
    )
    topo_group.add_argument(
        "--topology-bands",
        type=str,
        default=None,
        help=(
            "Comma-separated zero-based band indices for bulk topology "
            "(default: all positive bands as one group)."
        ),
    )
    topo_group.add_argument(
        "--topology-groups",
        type=str,
        default=None,
        help=(
            "Semicolon-separated band groups for bulk topology, e.g. "
            "'0;1' requests two independent groups. Mutually exclusive "
            "with --topology-bands."
        ),
    )
    topo_group.add_argument(
        "--topology-strip-width",
        type=int,
        default=None,
        metavar="N",
        help="Calculate a physical open strip N embedding-width cells wide.",
    )
    topo_group.add_argument(
        "--topology-strip-kpoints",
        type=int,
        default=None,
        metavar="NK",
        help="Endpoint-exclusive edge-periodic strip momentum count (default: 81).",
    )
    topo_group.add_argument(
        "--topology-strip-termination",
        type=float,
        nargs=2,
        default=None,
        metavar=("LEFT", "RIGHT"),
        help=(
            "Fractional left/right physical cut offsets in the width cell "
            "(default: 0 0)."
        ),
    )
    topo_group.add_argument(
        "--topology-strip-edge-layers",
        type=int,
        default=None,
        metavar="N",
        help="Number of width-cell layers in each physical edge region (default: 1).",
    )
    topo_group.add_argument(
        "--topology-strip-reference",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON array of corrected (nsite,3) moments in "
            "strip_site_map order."
        ),
    )
    topo_group.add_argument(
        "--topology-strip-out",
        type=str,
        default=None,
        help="Separate strip JSON output path (used by paired bulk/strip workflows).",
    )
    topo_group.add_argument(
        "--topology-compare",
        action="store_true",
        default=False,
        help="Run paired bulk and open-strip calculations, then compare one gap.",
    )
    topo_group.add_argument(
        "--topology-gap",
        type=int,
        default=None,
        metavar="N",
        help="Band count below the selected bulk gap for --topology-compare.",
    )
    topo_group.add_argument(
        "--topology-edge",
        choices=("left", "right"),
        default=None,
        help="Strip boundary for signed flow (left=min width; right=max width).",
    )
    topo_group.add_argument(
        "--topology-comparison-out",
        type=str,
        default=None,
        help="Versioned comparison JSON output path (default: topology_comparison.json).",
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
        help="Fixed normal reciprocal coordinate for both bulk and strip.",
    )
    topo_group.add_argument(
        "--topology-basis",
        type=int,
        nargs=9,
        default=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        metavar=("B11", "B12", "B13", "B21", "B22", "B23", "B31", "B32", "B33"),
        help="Ordered edge/width/normal direct-basis rows in input lattice coordinates.",
    )
    topo_group.add_argument(
        "--topology-out",
        type=str,
        default=None,
        help="Output JSON path (default: topology_bulk.json for bulk, topology_strip.json for strip).",
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


def topology_strip_request_from_args(args, embedding):
    """Resolve strip-only CLI inputs into the public Python request."""
    import json

    import numpy as np

    from TB2J.magnon.topology.strip import StripRequest

    reference = None
    if args.topology_strip_reference:
        with open(args.topology_strip_reference) as stream:
            reference = np.asarray(json.load(stream), dtype=float)

    termination = (
        (0.0, 0.0)
        if args.topology_strip_termination is None
        else tuple(args.topology_strip_termination)
    )
    return StripRequest(
        width=args.topology_strip_width,
        kpoints=(
            81 if args.topology_strip_kpoints is None else args.topology_strip_kpoints
        ),
        embedding=embedding,
        termination=termination,
        edge_layers=(
            1
            if args.topology_strip_edge_layers is None
            else args.topology_strip_edge_layers
        ),
        reference_magmoms=reference,
    )


def write_topology_json(path, record, valid):
    """Write a result; never truncate an existing destination for invalid data."""
    import json

    mode = "w" if valid else "x"
    try:
        with open(path, mode) as stream:
            json.dump(record, stream, indent=1)
    except FileExistsError:
        return False
    return True


def run_topology_strip(args, model, embedding) -> int:
    """Run strip topology from a prepared model, returning the CLI status."""
    from TB2J.magnon.topology.strip import calculate_strip

    try:
        request = topology_strip_request_from_args(args, embedding)
        result = calculate_strip(model, request)
    except (OSError, TypeError, ValueError) as error:
        print(f"Topology strip request refused: {error}")
        return 2
    output = args.topology_strip_out or args.topology_out or "topology_strip.json"
    try:
        written = write_topology_json(output, result.to_json_dict(), result.valid)
    except OSError as error:
        print(f"Could not write strip result {output}: {error}")
        return 2
    if written:
        print(f"Strip result written to {output}")
    else:
        print(
            "Invalid strip result not written; existing output was preserved: "
            f"{output}"
        )
    if result.valid:
        print(
            f"Valid open-strip spectrum: width={request.width}, "
            f"sites={len(result.site_map)}, edge k-points={request.kpoints}"
        )
        return 0
    print("Open-strip spectrum is INVALID. Reasons:")
    for reason in result.diagnostics.get("failures", []):
        print(f"  - {reason}")
    return 1


def run_topology_comparison(args, model, request, embedding) -> int:
    """Run paired calculations and preserve both standalone records."""
    from pathlib import Path

    from TB2J.magnon.topology.bulk import BulkRequest, calculate_bulk
    from TB2J.magnon.topology.bulk_edge import compare_bulk_edge
    from TB2J.magnon.topology.strip import calculate_strip

    if args.topology_strip_width is None:
        print("--topology-compare requires --topology-strip-width.")
        return 2
    if args.topology_gap is None or args.topology_gap < 1:
        print("--topology-compare requires a positive --topology-gap.")
        return 2
    if request.bands is None and request.groups is None:
        request = BulkRequest(
            mesh=request.mesh,
            bands=list(range(args.topology_gap)),
            embedding=embedding,
            gap_tol=request.gap_tol,
            link_tol=request.link_tol,
            stationarity_rtol=request.stationarity_rtol,
        )
    bulk_output = args.topology_out or "topology_bulk.json"
    strip_output = args.topology_strip_out or "topology_strip.json"
    comparison_output = args.topology_comparison_out or "topology_comparison.json"
    edge = args.topology_edge or "left"
    if (
        len(
            {
                Path(bulk_output).resolve(),
                Path(strip_output).resolve(),
                Path(comparison_output).resolve(),
            }
        )
        != 3
    ):
        print("Bulk, strip and comparison outputs must use distinct paths.")
        return 2
    try:
        strip_request = topology_strip_request_from_args(args, embedding)
    except (OSError, TypeError, ValueError) as error:
        print(f"Topology strip request refused: {error}")
        return 2
    try:
        bulk_result = calculate_bulk(model, request)
    except (TypeError, ValueError) as error:
        print(f"Bulk topology request refused: {error}")
        return 2
    bulk_record = bulk_result.to_json_dict()
    if args.topology_refine:
        from TB2J.magnon.topology.bulk import refine_bulk

        bulk_record["refinement"] = refine_bulk(model, request)
    try:
        bulk_written = write_topology_json(bulk_output, bulk_record, bulk_result.valid)
    except OSError as error:
        print(f"Could not write bulk result {bulk_output}: {error}")
        return 2
    if bulk_written:
        print(f"Bulk result written to {bulk_output}")
    else:
        print(
            "Invalid bulk result not written; existing output was preserved: "
            f"{bulk_output}"
        )
    try:
        strip_result = calculate_strip(model, strip_request)
    except (OSError, TypeError, ValueError) as error:
        print(f"Topology strip request refused: {error}")
        return 2
    try:
        strip_written = write_topology_json(
            strip_output, strip_result.to_json_dict(), strip_result.valid
        )
    except OSError as error:
        print(f"Could not write strip result {strip_output}: {error}")
        return 2
    if strip_written:
        print(f"Strip result written to {strip_output}")
    else:
        print(
            "Invalid strip result not written; existing output was preserved: "
            f"{strip_output}"
        )

    comparison = compare_bulk_edge(bulk_result, strip_result, args.topology_gap, edge)
    try:
        comparison_written = write_topology_json(
            comparison_output, comparison.to_json_dict(), comparison.valid
        )
    except OSError as error:
        print(f"Could not write comparison result {comparison_output}: {error}")
        return 2
    if comparison_written:
        print(f"Bulk-edge comparison written to {comparison_output}")
    else:
        print(
            "Invalid comparison not written; existing output was preserved: "
            f"{comparison_output}"
        )
    if comparison.valid:
        print(
            f"Signed {edge} edge flow {comparison.signed_crossings} "
            f"matches the bulk Chern sum {comparison.chern_below_gap}."
        )
        return 0
    print("Bulk-edge comparison is INVALID. Reasons:")
    for reason in comparison.diagnostics.get("failures", []):
        print(f"  - {reason}")
    return 1


def run_topology(args, params: MagnonParameters) -> int:
    """Opt-in bulk topology mode. Returns a process exit code."""

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
    try:
        embedding = LatticeEmbedding(
            basis=[
                args.topology_basis[0:3],
                args.topology_basis[3:6],
                args.topology_basis[6:9],
            ],
            normal_k=args.topology_normal_k,
        )
    except ValueError as error:
        print(f"Topology geometry refused the input: {error}")
        return 2
    if args.topology_strip_width is not None and not getattr(
        args, "topology_compare", False
    ):
        if args.topology_bands or args.topology_groups or args.topology_refine:
            print(
                "Bulk band selection/refinement options cannot be combined with "
                "a standalone open-strip spectrum."
            )
            return 2
        return run_topology_strip(args, model, embedding)
    try:
        request = BulkRequest(
            mesh=tuple(args.topology_mesh),
            bands=bands,
            groups=groups,
            embedding=embedding,
        )
    except (TypeError, ValueError) as error:
        print(f"Bulk topology request refused: {error}")
        return 2
    if getattr(args, "topology_compare", False):
        return run_topology_comparison(args, model, request, embedding)
    result = calculate_bulk(model, request)
    record = result.to_json_dict()
    if args.topology_refine:
        from TB2J.magnon.topology.bulk import refine_bulk

        record["refinement"] = refine_bulk(model, request)
    output = args.topology_out or "topology_bulk.json"
    try:
        written = write_topology_json(output, record, result.valid)
    except OSError as error:
        print(f"Could not write bulk result {output}: {error}")
        return 2
    if written:
        print(f"Topology result written to {output}")
    else:
        print(
            "Invalid topology result not written; existing output was preserved: "
            f"{output}"
        )
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


def validate_topology_cli_options(args, parser) -> None:
    """Reject topology options that would otherwise be silently ignored."""
    comparison_options = (
        args.topology_gap is not None
        or args.topology_edge is not None
        or args.topology_comparison_out is not None
    )
    if comparison_options and not args.topology_compare:
        parser.error(
            "--topology-gap, --topology-edge and --topology-comparison-out "
            "require --topology-compare."
        )

    strip_options = (
        args.topology_strip_kpoints is not None
        or args.topology_strip_termination is not None
        or args.topology_strip_edge_layers is not None
        or args.topology_strip_reference is not None
        or args.topology_strip_out is not None
    )
    if args.topology_strip_width is not None and not (
        args.topology or args.topology_compare
    ):
        parser.error(
            "--topology-strip-width requires --topology or --topology-compare."
        )
    if strip_options and not (
        args.topology_strip_width is not None or args.topology_compare
    ):
        parser.error(
            "strip-specific options require --topology-strip-width or "
            "--topology-compare."
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
    validate_topology_cli_options(args, parser)

    if (
        not args.bands
        and not args.dos
        and not args.topology
        and not args.topology_compare
    ):
        parser.error(
            "Please specify at least one of --bands, --dos, --topology or "
            "--topology-compare"
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

    if args.topology or args.topology_compare:
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
