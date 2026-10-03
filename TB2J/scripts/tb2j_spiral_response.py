"""CLI for the frozen spiral-state response of a persisted v2 bundle.

Runs the additive first-order/pitch/curvature response of a
``tbupy_spiral_state`` v2 sidecar and writes the JSON report
(:func:`TB2J.spiral_response.response_report`). Gates (density
spectral-projector check, field symmetry, provenance) fail closed:
on any :class:`SpiralResponseError` the CLI prints the reason to
stderr, exits nonzero, and writes no output.

Usage::

    python -m TB2J.scripts.tb2j_spiral_response \
        --spiral-state model.spiral.nc --output response.json \
        [--matched-q-record certified-matched-q.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen spiral-state response report (schema v2 bundles)",
    )
    parser.add_argument(
        "--spiral-state",
        required=True,
        help="path to a tbupy_spiral_state v2 sidecar (.spiral.nc)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output JSON report path",
    )
    parser.add_argument(
        "--matched-q-record",
        default=None,
        help="optional certified matched-q pitch record (TBUpy SCF legs)",
    )
    parser.add_argument(
        "--density-tol",
        type=float,
        default=None,
        help="override the spectral density gate tolerance",
    )
    parser.add_argument(
        "--field-tol",
        type=float,
        default=None,
        help="override the field-symmetry gate tolerance",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from TB2J.spiral_response import (
        SpiralResponseError,
        compute_frozen_response,
        response_report,
    )

    kwargs = {}
    if args.density_tol is not None:
        kwargs["density_tol"] = args.density_tol
    if args.field_tol is not None:
        kwargs["field_tol"] = args.field_tol
    if args.matched_q_record is not None:
        kwargs["matched_q_record"] = args.matched_q_record
    try:
        result = compute_frozen_response(args.spiral_state, **kwargs)
        report = response_report(result)
    except SpiralResponseError as exc:
        print(f"spiral-state response gate failed: {exc}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
