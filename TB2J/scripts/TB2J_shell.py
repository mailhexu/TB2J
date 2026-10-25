"""Editable symmetry-shell workflows over the public native exchange API."""

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from pathlib import Path
from pickle import UnpicklingError

import tomli
from ase.io import read

from TB2J.io_exchange import shells

_LOSSES = (
    "dmi",
    "jani",
    "sia",
    "sia_tensor",
    "biquadratic",
    "njt",
    "orbital",
    "djdx",
    "damping",
    "gyromag",
    "charges",
)
# Expansion governs ancillary base data; pair-channel losses belong to native IO.
_EXPANSION_LOSSES = {
    "sia",
    "biquadratic",
    "njt",
    "orbital",
    "djdx",
    "damping",
    "gyromag",
    "charges",
}
_HEADER_FLAGS = (
    "symmetry",
    "mode",
    "magnetic_atoms",
    "reference_spins",
    "symprec",
    "magnetic_tol",
    "energy_unit",
    "zero_tol",
    "value_atol",
    "value_rtol",
)
_CONTEXT_FLAGS = {"magnetic_atoms", "reference_spins"}


def _header_arguments(parser):
    parser.add_argument("--symmetry", choices=("sg", "msg"))
    parser.add_argument("--mode", choices=("isotropic", "tensor"))
    parser.add_argument(
        "--magnetic-atoms", nargs="+", type=int, help="One-based atom indices"
    )
    parser.add_argument(
        "--reference-spins", type=Path, help="TOML moments array in full atom order"
    )
    parser.add_argument("--symprec", type=float)
    parser.add_argument("--magnetic-tol", type=float)
    parser.add_argument("--energy-unit", choices=("meV", "eV"))
    parser.add_argument("--zero-tol", type=float)
    parser.add_argument("--value-atol", type=float)
    parser.add_argument("--value-rtol", type=float)


def _parser():
    parser = argparse.ArgumentParser(prog="TB2J_shell", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="Generate unfilled symmetry shells")
    template.add_argument("structure", type=Path)
    template.add_argument(
        "--cutoff", type=float, required=True, help="Neighbor cutoff in Angstrom"
    )
    _header_arguments(template)
    template.add_argument(
        "--reference-spins-model",
        choices=("unit-along-z",),
        help="Explicit unit-z model on every atom, SG only; not calculated moments",
    )
    template.add_argument(
        "-o", "--output", type=Path, default=Path("exchange.shells.toml")
    )
    template.add_argument("--no-equivalent-comments", action="store_true")
    convert = commands.add_parser(
        "convert", help="Convert shells and native exchange formats"
    )
    convert.add_argument("input", type=Path)
    convert.add_argument(
        "--from",
        dest="source_format",
        choices=("auto", "shells", "pickle", "text", "xml"),
        default="auto",
        help="Pickle must be selected explicitly and must be trusted",
    )
    convert.add_argument(
        "--to",
        dest="target",
        choices=("shells", "pickle", "text", "xml", "tb2j"),
        required=True,
    )
    convert.add_argument("-o", "--output", type=Path, required=True)
    convert.add_argument(
        "--base",
        type=Path,
        help="Trusted native pickle for shell-to-native base updates",
    )
    convert.add_argument("--project", action="store_true")
    convert.add_argument("--allow-loss", nargs="+", choices=_LOSSES, default=())
    convert.add_argument("--complete-orbits", action="store_true")
    convert.add_argument("--snap-zero", action="store_true")
    convert.add_argument("--structure", type=Path)
    convert.add_argument("--report", type=Path)
    convert.add_argument("--no-equivalent-comments", action="store_true")
    _header_arguments(convert)
    validate = commands.add_parser(
        "validate", help="Validate shells without replacing symmetry"
    )
    validate.add_argument("input", type=Path)
    validate.add_argument("--for-export", action="store_true")
    validate.add_argument("--check-producer-ops", action="store_true")
    return parser


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


def _kwargs(args):
    names = (
        "symmetry",
        "mode",
        "symprec",
        "magnetic_tol",
        "energy_unit",
        "zero_tol",
        "value_atol",
        "value_rtol",
    )
    result = {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }
    result.setdefault("mode", "isotropic")
    return result


def _reject_flags(args, names, route):
    supplied = []
    for name in names:
        value = getattr(args, name, None)
        if value is not None and value is not False and value != ():
            supplied.append("--" + name.replace("_", "-"))
    if supplied:
        raise ValueError(f"{', '.join(supplied)} not applicable to {route}")


def _detect_format(path):
    # Closed-world signatures only; unrecognized binary data NEVER reaches pickle.
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as error:
        raise ValueError(
            "unknown input format; select --from explicitly; trusted pickle requires --from pickle"
        ) from error
    if re.match(r"\s*(?:<\?xml[^>]*\?>\s*)?<System_definition(?:\s|>)", text):
        return "xml"
    if re.match(r"\s*(?:=+[ \t]*\r?\n\s*)*Information:[ \t]*(?:\r?\n|$)", text):
        return "text"
    try:
        header = tomli.loads(text)
    except tomli.TOMLDecodeError:
        header = {}
    if "schema_version" in header:
        return "shells"
    raise ValueError(
        "unknown input format; use --from shells for headerless hand files or select an explicit supported native format"
    )


def _json_value(value):
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    return value


def _reports(reports, path=None):
    for stage, report in reports.items():
        print(f"{stage}: ok")
        profile = getattr(report, "precision_profile", None)
        if profile:
            print(f"  precision: {profile}; {len(report.quanta)} field bounds")
        for name in (
            "residuals",
            "created_pair_keys",
            "created_channel_keys",
            "snapped_values",
            "encoding_changes",
        ):
            rows = getattr(report, name, ())
            if rows:
                print(f"  {name}: {len(rows)} (full values in --report)")
        for name in ("losses", "metadata_actions", "messages"):
            for row in getattr(report, name, ()):
                print(f"  {name}: {row}")
        if getattr(report, "producer_drift", None) is not None:
            print(f"  producer_drift: {report.producer_drift}")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_value(reports), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _template(args):
    if args.magnetic_atoms is None:
        raise ValueError(
            "template requires --magnetic-atoms with the full magnetic-site selection"
        )
    if args.reference_spins is not None and args.reference_spins_model is not None:
        raise ValueError(
            "select --reference-spins or --reference-spins-model, not both"
        )
    if args.reference_spins_model is not None and args.symmetry == "msg":
        raise ValueError(
            "msg requires actual complete reference moments, not --reference-spins-model"
        )
    atoms = read(args.structure)
    moments = _moments(args.reference_spins)
    if args.reference_spins_model is not None:
        moments = [1.0] * len(atoms)
    document = shells.template(
        atoms,
        args.magnetic_atoms,
        args.cutoff,
        reference_spins=moments,
        **_kwargs(args),
    )
    if args.reference_spins_model is not None:
        document = replace(document, moment_source="model:unit-along-z")
    report = shells.validate_shells(document)
    shells.write_shells(
        document, args.output, equivalent_comments=not args.no_equivalent_comments
    )
    print(f"Wrote {len(document.shells)} unfilled shells to {args.output}")
    _reports({"template": report})


def _check_report_path(args):
    if args.report is None:
        return
    protected = {
        path.resolve()
        for path in (
            args.input,
            args.output,
            args.base,
            args.structure,
            args.reference_spins,
        )
        if path is not None
    }
    native_names = {
        "pickle": ("TB2J.pickle",),
        "text": ("exchange.out",),
        "xml": ("exchange.xml",),
        "tb2j": ("TB2J.pickle", "exchange.out", "exchange.xml", "structure.vasp"),
    }
    protected.update(
        (args.output / name).resolve() for name in native_names.get(args.target, ())
    )
    if args.report.resolve() in protected or (
        args.report.exists()
        and any(path.exists() and args.report.samefile(path) for path in protected)
    ):
        raise ValueError(
            "--report must not replace an input, context file or generated exchange artifact"
        )


def _convert(args):
    source_format = (
        _detect_format(args.input)
        if args.source_format == "auto"
        else args.source_format
    )
    compressing = source_format != "shells" and args.target == "shells"
    expanding = source_format == "shells" and args.target != "shells"
    if source_format == "shells":
        _reject_flags(args, _HEADER_FLAGS, "shell input")
        _reject_flags(args, ("structure",), "shell input")
    elif not compressing:
        _reject_flags(
            args,
            tuple(name for name in _HEADER_FLAGS if name not in _CONTEXT_FLAGS),
            "native-to-native conversion",
        )
    if not compressing:
        _reject_flags(
            args, ("snap_zero", "project"), "this route (native-to-shells only)"
        )
    if not expanding:
        _reject_flags(
            args, ("base", "complete_orbits"), "this route (shells-to-native only)"
        )
    if args.target != "shells":
        _reject_flags(args, ("no_equivalent_comments",), "non-shell output")
    if source_format == "shells" and args.target == "shells":
        _reject_flags(args, ("allow_loss",), "shell rewrite")
    _check_report_path(args)
    reports = {}
    if source_format == "shells":
        document = shells.read_shells(args.input)
        reports["validate"] = shells.validate_shells(document, for_export=expanding)
        if expanding:
            base = None
            if args.base is not None:
                loaded = shells.read_exchange(args.base, "pickle")
                base = loaded.spinio
                reports["base"] = loaded.report
            expanded = shells.expand(
                document,
                base=base,
                complete_orbits=args.complete_orbits,
                allow_loss=tuple(
                    name for name in args.allow_loss if name in _EXPANSION_LOSSES
                ),
            )
            native = expanded.spinio
            reports["expand"] = expanded.report
    else:
        moments = _moments(args.reference_spins)
        context = shells.NativeContext(
            atoms=args.structure,
            magnetic_atoms=args.magnetic_atoms,
            reference_spins=moments,
        )
        loaded = shells.read_exchange(
            args.input, source_format, context=context, allow_loss=args.allow_loss
        )
        native = loaded.spinio
        reports["read"] = loaded.report
        if compressing:
            result = shells.compress(
                native,
                reference_spins=moments,
                on_inconsistent="project" if args.project else "error",
                snap_zero=args.snap_zero,
                **_kwargs(args),
            )
            document = result.document
            reports["compress"] = result.report
    if args.target == "shells":
        shells.write_shells(
            document, args.output, equivalent_comments=not args.no_equivalent_comments
        )
    else:
        reports["write"] = shells.write_exchange(
            native, args.output, args.target, allow_loss=args.allow_loss
        )
    _reports(reports, args.report)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            _template(args)
        elif args.command == "convert":
            _convert(args)
        else:
            document = shells.read_shells(args.input)
            report = shells.validate_shells(
                document,
                for_export=args.for_export,
                check_producer_ops=args.check_producer_ops,
            )
            _reports({"validate": report})
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
