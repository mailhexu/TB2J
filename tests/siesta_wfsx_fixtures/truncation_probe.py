#!/usr/bin/env python3
"""Opt-in native band-filter refusal and checked incompleteness evidence.

MathDoc: tb2j/siesta-wfsx-spectral#incomplete_completion_difference
The authoritative script executes the missing-pole and unobserved-operator
counterexamples; this command cites its actual result rather than re-deriving
or offering a partial-band exchange approximation.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parents[1]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))


def run_probe(work_root, math_script):
    from HamiltonIO.siesta.wfsx import IncompleteSpectrum, SiestaWFSXParser
    from test_siesta_wfsx_green import rewrite_test_records, source_model

    inputs = work_root / "input"
    inputs.mkdir()
    model, metadata, fixture, _ = source_model(inputs, "scalar_nspin1")
    basis = fixture["states"]["no_u"]
    groups = fixture["states"]["nk"]

    def omit_last_band(records):
        # Same valid native Fortran-record surgery as HamiltonIO's existing
        # test_band_filtered_export_is_rejected_before_streaming regression.
        filtered = records[:4]
        cursor = 4
        for _ in range(groups):
            header = records[cursor : cursor + 3]
            if [row.nbytes for row in header] != [36, 4, 4]:
                raise RuntimeError("unexpected native scalar record layout")
            if int(header[2].view(np.int32)[0]) != basis:
                raise RuntimeError("source fixture is not complete")
            header[2].view(np.int32)[0] = basis - 1
            filtered.extend(header)
            filtered.extend(records[cursor + 3 : cursor + 3 + 3 * (basis - 1)])
            cursor += 3 + 3 * basis
        if cursor != len(records):
            raise RuntimeError("unconsumed source records")
        records[:] = filtered

    source = model.eigenstate_source.wfsx_path
    rewrite_test_records(source, omit_last_band)
    refusal = None
    try:
        with SiestaWFSXParser(source, metadata).open():
            raise RuntimeError("band-filtered non-square spectrum was accepted")
    except IncompleteSpectrum as exc:
        refusal = {
            "category": f"{type(exc).__module__}.{type(exc).__name__}",
            "message": str(exc),
        }

    command = [sys.executable, str(math_script), "--kosmic-json"]
    checked = subprocess.run(command, capture_output=True, text=True, check=True)
    math = json.loads(checked.stdout)
    named = {row["name"]: row for row in math["results"]}
    ambiguity = named["incomplete_completion_difference"]
    # Executing derive() also asserts the different full/truncated resolvents
    # and different unobserved spin vertices for the SAME supplied eigenpairs.
    return {
        "pass": True,
        "native_filtered_input": {
            "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "basis_size": basis,
            "retained_states_per_record": basis - 1,
            "records": groups,
            "refusal": refusal,
            "producer_run_id": fixture["producer_run_id"],
            "mutation": "omit last band in every native record and correct each state count; no H/S fallback",
        },
        "checked_incompleteness_authority": {
            "script": str(math_script),
            "sha256": hashlib.sha256(math_script.read_bytes()).hexdigest(),
            "command": command,
            "returncode": checked.returncode,
            "python_version": math["python_version"],
            "sympy_version": math["sympy_version"],
            "result": ambiguity,
            "executed_assertions": "derive(): identical supplied partial generalized eigenpairs permit different full Hamiltonians and spin vertices; the truncated Green function differs from the full resolvent because a pole is omitted",
            "limit": "A non-square coefficient spectrum cannot identify the missing operators. No partial-band exchange mode is offered.",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--math-script",
        type=Path,
        required=True,
        help="trusted Kosmic spectral authority script",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    args.work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="truncation-", dir=args.work_root))
    try:
        report = run_probe(work, args.math_script.resolve())
    except Exception:
        report = {
            "pass": False,
            "error": traceback.format_exc(),
            "work_root": str(work),
        }
    for output in (args.output_json, args.output_md):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["pass"]:
        native = report["native_filtered_input"]
        authority = report["checked_incompleteness_authority"]
        text = (
            "# WFSX incomplete-spectrum evidence\n\n"
            f"Native `{native['retained_states_per_record']}`-state records for the "
            f"`{native['basis_size']}`-orbital basis refuse as "
            f"`{native['refusal']['category']}` before streaming.\n\n"
            f"{authority['executed_assertions']}.\n\n"
            f"Executed authority SHA-256: `{authority['sha256']}`; "
            f"Python {authority['python_version']}, SymPy {authority['sympy_version']}. "
            "Result handle: `tb2j/siesta-wfsx-spectral#incomplete_completion_difference`.\n\n"
            f"{authority['limit']}\n"
        )
    else:
        text = (
            "# WFSX incomplete-spectrum evidence: FAILED\n\n```text\n"
            + report["error"]
            + "```\n"
        )
    args.output_md.write_text(text)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "work_root": str(work),
                "json": str(args.output_json),
            }
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
