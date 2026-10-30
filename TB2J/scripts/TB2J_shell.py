"""Generate and export editable symmetry-shell exchange documents.

``TB2J_shell generate`` writes an annotated ``exchange.shells.toml`` from a
lattice structure or a trusted TB2J pickle; pickle input also writes a
``.context.npz`` companion that must travel with the shell file. After editing
shell values, ``TB2J_shell export`` writes the standard TB2J results tree.
"""

import argparse
import sys
from pathlib import Path
from pickle import UnpicklingError

import numpy as np
import tomli

from TB2J.io_exchange.shell_workflow import export_shells, generate_shells
from TB2J.versioninfo import print_license


class _DefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        if "default:" in (action.help or "").lower():
            return action.help
        return super()._get_help_string(action)


def _moments(path):
    if path is None:
        return None
    with path.open("rb") as stream:
        data = tomli.load(stream)
    if set(data) != {"moments"} or not isinstance(data["moments"], list):
        raise ValueError(
            "--reference-spins requires a TOML moments = [...] array in full atom order"
        )
    return data["moments"]


def _parser():
    parser = argparse.ArgumentParser(prog="TB2J_shell", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate",
        help="Generate an editable shell document from a structure or trusted pickle",
        formatter_class=_DefaultsFormatter,
    )
    generate.add_argument("source", type=Path)
    selection = generate.add_mutually_exclusive_group()
    selection.add_argument(
        "-s",
        "--magnetic-species",
        nargs="+",
        metavar="SYMBOL",
        help="Chemical symbols selecting every matching atom (structure input; required unless -a)",
    )
    selection.add_argument(
        "-a",
        "--magnetic-atoms",
        nargs="+",
        type=int,
        metavar="INDEX",
        help="One-based full-structure atom indices (structure input; required unless -s)",
    )
    generate.add_argument(
        "-c",
        "--cutoff",
        type=float,
        help="Neighbor cutoff in the selected distance unit (required for structure input)",
    )
    generate.add_argument(
        "-d",
        "--distance-unit",
        choices=("Angstrom", "Bohr", "nm"),
        default="Angstrom",
        help="Unit for cutoff and shell distances",
    )
    generate.add_argument(
        "-m",
        "--mode",
        choices=("isotropic", "tensor"),
        help="Exchange mode (default: isotropic for structures; infer from pickle)",
    )
    generate.add_argument(
        "-y",
        "--symmetry",
        choices=("sg", "msg"),
        default="sg",
        help="Symmetry type",
    )
    generate.add_argument(
        "-r",
        "--reference-spins",
        type=Path,
        help="TOML moments = [...] array in full atom order (default: none)",
    )
    generate.add_argument(
        "-p",
        "--symprec",
        type=float,
        default=1e-5,
        help="Spatial symmetry tolerance in Angstrom",
    )
    generate.add_argument(
        "-t",
        "--magnetic-tol",
        type=float,
        default=1e-5,
        help="Magnetic symmetry tolerance",
    )
    generate.add_argument(
        "-P",
        "--project",
        action="store_true",
        default=None,
        help="Symmetrize native exchange (default: automatic for pickle input)",
    )
    generate.add_argument(
        "-n",
        "--no-equivalent-comments",
        action="store_true",
        help="Omit equivalent-pair comments",
    )
    generate.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("exchange.shells.toml"),
        help="Output shell document",
    )

    export = commands.add_parser(
        "export",
        help="Expand an edited shell document into standard TB2J results",
        formatter_class=_DefaultsFormatter,
    )
    export.add_argument(
        "source", type=Path, help="Shell TOML document, with its context companion"
    )
    export.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("TB2J_results"),
        help="Output results directory",
    )
    return parser


def _generate(args):
    print(f"Source: {args.source}\nOutput: {args.output.resolve()}")
    pickle_input = args.source.suffix.lower() in {".pickle", ".pkl"}
    action = "symmetrizing imported exchange" if pickle_input else "building template"
    print(f"Generating shells using {args.symmetry} symmetry ({action}).", flush=True)
    reference_spins = _moments(args.reference_spins)
    kwargs = {}
    for name in ("magnetic_species", "magnetic_atoms", "cutoff", "mode"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if reference_spins is not None:
        kwargs["reference_spins"] = reference_spins
    if args.symprec is not None:
        kwargs["symprec"] = args.symprec
    if args.magnetic_tol is not None:
        kwargs["magnetic_tol"] = args.magnetic_tol
    result = generate_shells(
        args.source,
        args.output,
        symmetry=args.symmetry,
        project=args.project,
        distance_unit=args.distance_unit,
        equivalent_comments=not args.no_equivalent_comments,
        **kwargs,
    )
    if pickle_input:
        maximum = max(
            (float(np.max(np.abs(row[4]))) for row in result.report.residuals),
            default=0.0,
        )
        print(
            f"Symmetrized exchange; maximum absolute component change: {maximum:.8g} eV"
        )
    print(
        f"Exchange mode: {result.document.exchange_mode}; "
        f"symmetry: {result.document.symmetry_mode}"
    )
    print(f"Wrote {len(result.document.shells)} shells to {args.output}")
    if result.document.context_sha256 is not None:
        print(
            f"Wrote context companion {args.output.with_suffix('.context.npz')}; "
            f"move it together with {args.output}"
        )


def _export(args):
    result = export_shells(args.source, args.output)
    for target, reason in result.results.omitted:
        print(f"omitted {target}: {reason}", file=sys.stderr)
    print(f"Wrote standard TB2J results to {args.output}")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    print_license()
    try:
        if args.command == "generate":
            _generate(args)
        else:
            _export(args)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        EOFError,
        UnpicklingError,
        AssertionError,
    ) as error:
        print(f"TB2J_shell: {error or type(error).__name__}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
