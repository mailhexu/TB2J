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

import tomli

from TB2J.io_exchange.shell_workflow import export_shells, generate_shells


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
    )
    generate.add_argument("source", type=Path)
    selection = generate.add_mutually_exclusive_group()
    selection.add_argument(
        "--magnetic-species",
        nargs="+",
        metavar="SYMBOL",
        help="Chemical symbols selecting every matching atom (structure input)",
    )
    selection.add_argument(
        "--magnetic-atoms",
        nargs="+",
        type=int,
        metavar="INDEX",
        help="One-based full-structure atom indices (structure input)",
    )
    generate.add_argument(
        "--cutoff",
        type=float,
        help="Neighbor cutoff in Angstrom (structure input)",
    )
    generate.add_argument("--mode", choices=("isotropic", "tensor"))
    generate.add_argument("--symmetry", choices=("sg", "msg"), default="sg")
    generate.add_argument(
        "--reference-spins",
        type=Path,
        help="TOML moments = [...] array in full atom order (structure input)",
    )
    generate.add_argument("--symprec", type=float)
    generate.add_argument("--magnetic-tol", type=float)
    generate.add_argument(
        "--project",
        action="store_true",
        help="Project inconsistent native exchange instead of failing (pickle input)",
    )
    generate.add_argument("--no-equivalent-comments", action="store_true")
    generate.add_argument(
        "-o", "--output", type=Path, default=Path("exchange.shells.toml")
    )

    export = commands.add_parser(
        "export", help="Expand an edited shell document into standard TB2J results"
    )
    export.add_argument(
        "source", type=Path, help="Shell TOML document, with its context companion"
    )
    export.add_argument("-o", "--output", type=Path, default=Path("TB2J_results"))
    return parser


def _generate(args):
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
        equivalent_comments=not args.no_equivalent_comments,
        **kwargs,
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
