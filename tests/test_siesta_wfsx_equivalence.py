#!/usr/bin/env python3
"""Opt-in story-008 full-array equivalence report command (no pytest session).

Invocation:

    python tests/test_siesta_wfsx_equivalence.py --work-root DIR \
        --output-json FILE --output-md FILE [--native-trace]

Run it with the story worktrees importable through the normal environment,
e.g. PYTHONPATH=<siesta-wfsx-tb2j worktree>:<siesta-wfsx-hamiltonio worktree>
and the shared scientific interpreter. The command is an opt-in report: it
performs no permanent test rows, changes no production source, threshold,
fixture or golden data, and writes only to the caller-provided work root and
the explicitly given output paths.

What it actually measures, always by running the real code paths:

- Identifier-aligned full-array comparisons between real TB2J consumers driven
  by complete SIESTA states (states.WFSX + flat metadata JSON) and the
  same-run ordinary H/S oracle (``ordinary_exchange`` with
  ``complete_spectrum=True``), for the magnetic collinear representative
  (full grid / internal fold / representative file) and the full nspin=4
  spinor case, under both ``TB2J_SIESTA_GREEN_MODE`` settings.
- Compared quantities, each mapped to its checked MathDoc handle in
  ``memnotes/Projects/TB2J/mathdocs/siesta-wfsx-spectral.md`` (never
  re-derived here):
    * H(k), S(k), mesh H0, Delta/HR0 ... #complete_hamiltonian,
      #complete_metric, #orbital_lattice_gauge, #energy_reference; H0 is the
      mesh-periodized average only (#mesh_onsite_alias).
    * G(k,z), G(R,z) .................... #complete_resolvent; the inverse
      Fourier phase in get_GR (#fourier_inverse_sign).
    * contravariant Green mode .......... #metric_dressed_green and
      #complete_metric_inverse (S^-1 = C C^dagger).
    * populations rho, rho(R), charges, spins ... #population_operator,
      #population_trace, #covariant_density_trace.
    * self-inverse counting ............. #scalar_partner_transpose and
      #scalar_density_conjugate: a self-inverse point is symmetric and
      counted once.
  No eigenvector or partial-spectrum comparison; no new formula or threshold.
- Controlled quadrature variants: shuffled WFSX file records, permuted
  requested order, reciprocal-lattice wrapping, Gamma-only and non-closed
  shifted subsets, with explicit kpts/kweights on BOTH the oracle and the
  WFSX side and no added k.
- Every full exchange tensor via actual SpinIO write/read-back, key-aligned
  by R and site (not only isotropic values, norms or eigenvectors).
- The coordinate-only half-grid self-inverse boundary reused from the
  inherited HamiltonIO regression (HW tests/test_wfsx.py::
  test_nonzero_self_inverse_coordinate_is_not_doubled) as a structural check;
  explicitly NOT an operator-equivalence claim at the synthetic coordinate.
- The named refusal of scalar reduction for full spinor exchange
  (``UnsupportedSpinMode``) with a measured zero factorization count.
- Finally the real story-007 entry trace matrix from
  ``tests/test_siesta_wfsx_no_hs_execution.py::trace_entry_cases`` run under
  a child of the work root; its rows are appended verbatim. Native file-open
  tracing happens only with ``--native-trace`` (requires /bin/strace and the
  installed siesta2J.py console script; the trace module fails clearly when
  they are unavailable).

Exit status is nonzero when any numerical threshold, structural guard, key
set, or trace-row check is unmet; measured failure evidence is preserved in
the reports instead of being widened or dropped.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

STORY = (
    "tb2j/siesta-wavefunction-backed-exchange-without-electronic-diagonali/story/008"
)
MATHDOC = "memnotes/Projects/TB2J/mathdocs/siesta-wfsx-spectral.md"
REPORT_SCHEMA = "tb2j/siesta-wfsx-equivalence-report/1"

COLLINEAR_FULL = "collinear_nspin2_full"
COLLINEAR_REDUCED = "collinear_nspin2_reduced"
SPINOR_FULL = "noncollinear_nspin4_full"

# Shared, unchanged Architecture thresholds.
TOLERANCES = {
    "H_S_H0_Delta_HR0": {
        "atol": 2e-5,
        "rtol": 1e-5,
        "unit": "eV (H), dimensionless (S)",
    },
    "green_functions": {"atol": 2e-5, "rtol": 1e-4, "unit": "1/eV"},
    "populations_charges_spins": {"atol": 1e-4, "rtol": 1e-5, "unit": "electrons"},
    "exchange_tensors": {"atol": 1e-6, "rtol": 1e-4, "unit": "eV"},
}

Z_POINTS = (complex(-20, 0.7), complex(-5, 1), complex(0.3, 2))
R_POINTS = ((0, 0, 0), (0, 0, 1), (0, 0, -1))
GREEN_MODES = ("default", "contravariant")

PERMUTATION = [3, 0, 4, 2, 1]
REQUEST_ORDER = [2, 4, 0, 3, 1]
WRAPPING = (1, -2, 3)
NONCLOSED_INDICES = [3, 1, 4]
NONCLOSED_SHIFT = (1, -2, 3)

EXPECTED_TRACE_ROWS = 8
TRACE_FIXTURES = ("collinear_nspin2_reduced", "noncollinear_nspin4_full")
TRACE_ENTRIES = ("console", "python")

QUANTITY_HANDLES = {
    "H(k)": ["#complete_hamiltonian", "#orbital_lattice_gauge"],
    "S(k)": ["#complete_metric", "#orbital_lattice_gauge"],
    "H0": ["#mesh_onsite_alias"],
    "Delta": ["#complete_hamiltonian", "#energy_reference"],
    "HR0": ["#complete_hamiltonian", "#mesh_onsite_alias"],
    "rho": ["#population_operator", "#population_trace"],
    "rho(R)": ["#covariant_density_trace", "#population_operator"],
    "G(k,z)": ["#complete_resolvent", "#energy_reference"],
    "G(R,z)": ["#complete_resolvent", "#fourier_inverse_sign"],
    "population trace": ["#population_trace"],
    "charges": [],
    "spinat": [],
    "exchange_Jdict": [],
    "Jani": [],
    "DMI": [],
    "full 3x3 tensors": [],
    "green_mode contravariant": ["#metric_dressed_green", "#complete_metric_inverse"],
    "self-inverse counting": ["#scalar_partner_transpose", "#scalar_density_conjugate"],
}

COLLINEAR_TENSOR_FLAGS = {"Jiso": True, "Jani": False, "DMI": False, "SIA": False}
COLLINEAR_TENSOR_NOTE = (
    "collinear SpinIO output carries the isotropic exchange_Jdict only; the "
    "full 3x3 tensor is J*I3 per site pair, compared elementwise per R/site"
)
SPINOR_TENSOR_FLAGS = {"Jiso": True, "Jani": True, "DMI": True, "SIA": True}

SCOPE_LIMITS = [
    "Only Green functions and operators are compared across independent "
    "diagonalizations; eigenvectors and partial spectra are never compared.",
    "H0 is the requested-mesh arithmetic (mesh-periodized) average; no "
    f"continuum H(R=0) identification is claimed [{MATHDOC}#mesh_onsite_alias].",
    "float32 WFSX source coefficients bound the recoverable accuracy "
    f"[{MATHDOC}#float32_conditioning_report]; the LU reciprocal-condition "
    "guard (refusal below 1e-10) can reject a fixture.",
    "The ordinary oracle is the story-002 same-run H/S parser (SislParser, "
    "complete_spectrum=True); the recorded 23a9903 HSX-offset question "
    "remains a governance item outside this report and no ordinary-path "
    "code was changed here.",
    "collinear/half_boundary mutates only k-point headers on a copy "
    "(coordinate-only synthetic boundary inherited from HamiltonIO "
    "tests/test_wfsx.py::test_nonzero_self_inverse_coordinate_is_not_doubled); "
    "no H/S operator equivalence is claimed at the synthetic coordinate.",
    "default and contravariant Green modes are each compared against the "
    "same-mode oracle only; they are never asserted mutually equal.",
    "Gamma-only and non-closed shifted subsets are exercised on the magnetic "
    "collinear representative; the nspin=4 spinor case is exercised on the "
    "full grid and the shuffled/permuted/wrapped request. Scalar reduction of "
    "spinor exchange must be refused (UnsupportedSpinMode) and is checked as "
    "a refusal, never attempted.",
    "This report driver is uncommitted test infrastructure; git HEADs and the "
    "driver content hash are recorded and are not claimed as commits.",
    "The trace matrix is produced by "
    "tests/test_siesta_wfsx_no_hs_execution.py::trace_entry_cases; native "
    "open tracing only with --native-trace (requires /bin/strace and the "
    "installed siesta2J.py console script); permanent tests never require "
    "ptrace.",
    "No production source, threshold, fixture or golden data was modified by "
    "this command; fixtures are consumed read-only and all mutations happen "
    "on copies under the caller-provided work root.",
]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_capture(root, *args):
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def rounded(value, digits=6):
    try:
        return float(f"{float(value):.{digits}g}")
    except (TypeError, ValueError):
        return value


def key_str(key):
    if isinstance(key, dict):
        return json.dumps(jsonable(key), sort_keys=True)
    if isinstance(key, (tuple, list, np.ndarray)):
        return "(" + ", ".join(key_str(item) for item in key) + ")"
    if isinstance(key, np.integer):
        return str(int(key))
    if isinstance(key, np.floating):
        return repr(float(key))
    return str(key)


def jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, (complex, np.complexfloating)):
        return {
            "real": jsonable(complex(value).real),
            "imag": jsonable(complex(value).imag),
        }
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key_str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return str(value)


def z_label(z):
    return f"{z.real:g}{z.imag:+g}j"


# --------------------------------------------------------------------------- #
# report primitives
# --------------------------------------------------------------------------- #
def new_entry(case, mode, fixture, folder, **extra):
    entry = {
        "case": case,
        "green_mode": mode,
        "fixture": fixture,
        "folder": str(folder),
        "inputs": {},
        "requested": {},
        "notes": [],
        "guards": [],
        "arrays": [],
        "keysets": {},
        "errors": [],
        "seconds": {},
    }
    entry.update(extra)
    return entry


def record_inputs(entry, folder, fixture_name, mutation=None):
    from test_siesta_wfsx_green import fixture_case

    case_dir, manifest, _ = fixture_case(fixture_name)
    inputs = {
        "fixture": fixture_name,
        "fixture_dir": str(case_dir),
        "fixture_json_sha256": sha256_file(case_dir / "fixture.json"),
        "producer_run_id": manifest.get("producer_run_id"),
        "states_WFSX_sha256": sha256_file(folder / "states.WFSX"),
        "metadata_json_sha256": sha256_file(folder / "metadata.json"),
        "consumed_read_only": True,
    }
    if mutation:
        inputs["mutation_on_copy"] = mutation
    entry["inputs"] = inputs


def compare_array(entry, quantity, key, actual, expected, atol, rtol):
    """Element-aligned threshold row; NaN never passes; shapes must match."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    record = {
        "quantity": quantity,
        "key": key_str(key) if not isinstance(key, str) else key,
        "atol": atol,
        "rtol": rtol,
        "cited_handles": QUANTITY_HANDLES.get(quantity, []),
    }
    if actual.shape != expected.shape:
        record.update(
            {
                "shape": [list(actual.shape), list(expected.shape)],
                "elements": int(actual.size),
                "pass": False,
                "reason": "shape mismatch",
            }
        )
        entry["arrays"].append(record)
        return False
    difference = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    ok = bool(difference.size == 0 or np.all(difference <= bound))
    worst = float((difference / bound).max()) if difference.size else 0.0
    record.update(
        {
            "shape": list(actual.shape),
            "elements": int(actual.size),
            "max_abs_error": rounded(np.max(difference) if difference.size else 0.0),
            "worst_tolerance_ratio": rounded(worst, 4),
            "pass": ok,
        }
    )
    entry["arrays"].append(record)
    return ok


def guard_row(entry, name, observed, expected, note=None):
    row = {
        "guard": name,
        "observed": jsonable(observed),
        "expected": jsonable(expected),
        "pass": None if expected is None else jsonable(observed) == jsonable(expected),
    }
    if note:
        row["note"] = note
    entry["guards"].append(row)
    return row


def bound_guard_row(entry, name, observed, bound, note=None):
    row = {
        "guard": name,
        "observed": jsonable(observed),
        "expected": f"<= {bound}",
        "pass": bool(observed <= bound),
    }
    if note:
        row["note"] = note
    entry["guards"].append(row)
    return row


def info_row(entry, name, observed, note=None):
    guard_row(entry, name, observed, None, note=note)


def compare_keyset(entry, name, actual_keys, expected_keys):
    actual = {key_str(key) for key in (actual_keys or [])}
    expected = {key_str(key) for key in (expected_keys or [])}
    match = actual == expected
    entry["keysets"][name] = {
        "match": match,
        "count": len(expected),
        "missing_from_actual": sorted(expected - actual)[:16],
        "unexpected_in_actual": sorted(actual - expected)[:16],
    }
    return match


def _stacked(mapping, keys, template):
    complex_template = np.iscomplexobj(template)
    fill = np.full(template.size, np.nan, dtype=complex if complex_template else float)
    rows = []
    for key in keys:
        value = (mapping or {}).get(key)
        if value is None:
            rows.append(fill)
        else:
            rows.append(np.asarray(value).reshape(-1))
    return np.concatenate(rows) if rows else np.zeros(0)


def compare_value_dict(entry, name, actual, expected, atol, rtol):
    """Exact key-set equality plus key-aligned element comparison of values."""
    keys = sorted(expected or {}, key=key_str)
    ok = compare_keyset(entry, f"{name}:keys", (actual or {}).keys(), keys)
    if not keys:
        return ok
    template = np.asarray(next(iter(expected.values())))
    actual_values = _stacked(actual, keys, template)
    expected_values = _stacked(expected, keys, template)
    ok = (
        compare_array(
            entry,
            name,
            f"{len(keys)} key-aligned values",
            actual_values,
            expected_values,
            atol,
            rtol,
        )
        and ok
    )
    return ok


def tensor_dict(spinio, flags):
    jmat = spinio.get_full_Jtensor_for_Rlist(order="ij33", **flags)
    return {
        tuple(int(value) for value in R): jmat[iR]
        for iR, R in enumerate(np.asarray(spinio.Rlist))
    }


def compare_spinio_pair(entry, actual, expected, flags, tensor_note):
    """Full-tensor comparison of actual SpinIO write/read-back pickles."""
    ok = guard_row(
        entry,
        "spinio.Rlist identical",
        np.asarray(actual.Rlist).tolist(),
        np.asarray(expected.Rlist).tolist(),
    )["pass"]
    entry["notes"].append(tensor_note)
    actual_tensors = tensor_dict(actual, flags)
    expected_tensors = tensor_dict(expected, flags)
    ok = (
        compare_keyset(
            entry,
            "full 3x3 tensors:R keys",
            actual_tensors.keys(),
            expected_tensors.keys(),
        )
        and ok
    )
    for r_key in sorted(expected_tensors, key=key_str):
        present = actual_tensors.get(r_key)
        missing = np.full(np.shape(expected_tensors[r_key]), np.nan)
        ok = (
            compare_array(
                entry,
                "full 3x3 tensors",
                f"R={key_str(r_key)} (i,j,3x3)",
                present if present is not None else missing,
                expected_tensors[r_key],
                TOLERANCES["exchange_tensors"]["atol"],
                TOLERANCES["exchange_tensors"]["rtol"],
            )
            and ok
        )
    ok = (
        compare_array(
            entry,
            "charges",
            "SpinIO pickle atoms",
            actual.charges,
            expected.charges,
            TOLERANCES["populations_charges_spins"]["atol"],
            TOLERANCES["populations_charges_spins"]["rtol"],
        )
        and ok
    )
    ok = (
        compare_array(
            entry,
            "spinat",
            "SpinIO pickle atoms x 3",
            actual.spinat,
            expected.spinat,
            TOLERANCES["populations_charges_spins"]["atol"],
            TOLERANCES["populations_charges_spins"]["rtol"],
        )
        and ok
    )
    return ok


def compare_green_pair(entry, actual, expected, channel, note=None):
    """Compare a WFSX-backed TBGreen with the same-mode ordinary TBGreen."""
    tol_hs = TOLERANCES["H_S_H0_Delta_HR0"]
    tol_g = TOLERANCES["green_functions"]
    tol_rho = TOLERANCES["populations_charges_spins"]
    ok = True
    if note:
        entry["notes"].append(note)
    oracle_model = getattr(expected, "tbmodel", None)
    if expected._spectral_source is None and oracle_model is None:
        # ExchangeCL2 releases its model references after preparation. Read the
        # same-run oracle separately; never reconstruct H/S from WFSX here.
        from HamiltonIO.siesta.sisl_wrapper import SislParser
        from test_siesta_wfsx_green import fixture_case

        case, manifest, _ = fixture_case(COLLINEAR_FULL)
        models = SislParser(str(case / manifest["files"]["fdf"])).get_model()
        oracle_model = models[0 if channel == "up" else 1]
    for ik in range(len(actual.kpts)):
        klabel = np.round(np.asarray(actual.kpts[ik], dtype=float), 8).tolist()
        if expected._spectral_source is None:
            expected_h, expected_s = oracle_model.gen_ham(expected.kpts[ik])
        else:
            expected_h, expected_s = expected.get_Hk(ik), expected.get_Sk(ik)
        ok = (
            compare_array(
                entry,
                "H(k)",
                f"{channel} k#{ik} {klabel}",
                actual.get_Hk(ik),
                expected_h,
                tol_hs["atol"],
                tol_hs["rtol"],
            )
            and ok
        )
        ok = (
            compare_array(
                entry,
                "S(k)",
                f"{channel} k#{ik} {klabel}",
                actual.get_Sk(ik),
                expected_s,
                tol_hs["atol"],
                tol_hs["rtol"],
            )
            and ok
        )
    ok = (
        compare_array(
            entry, "H0", channel, actual.H0, expected.H0, tol_hs["atol"], tol_hs["rtol"]
        )
        and ok
    )
    ok = (
        compare_array(
            entry,
            "rho",
            channel,
            actual.get_density_matrix(),
            expected.get_density_matrix(),
            tol_rho["atol"],
            tol_rho["rtol"],
        )
        and ok
    )
    ok = (
        compare_array(
            entry,
            "rho(R)",
            f"{channel} R set {list(R_POINTS)}",
            actual.get_rho_R(R_POINTS),
            expected.get_rho_R(R_POINTS),
            tol_rho["atol"],
            tol_rho["rtol"],
        )
        and ok
    )
    for z in Z_POINTS:
        ok = (
            compare_array(
                entry,
                "G(k,z)",
                f"{channel} z={z_label(z)}",
                actual.get_Gk_all(z),
                expected.get_Gk_all(z),
                tol_g["atol"],
                tol_g["rtol"],
            )
            and ok
        )
        ok = (
            compare_array(
                entry,
                "G(R,z)",
                f"{channel} z={z_label(z)} R set {list(R_POINTS)}",
                actual.get_GR(R_POINTS, z),
                expected.get_GR(R_POINTS, z),
                tol_g["atol"],
                tol_g["rtol"],
            )
            and ok
        )
    return ok


def snapshot_guards(
    entry, exchange, channels, expected_reps, expected_full, partner_zero
):
    """Guard counters of the WFSX-backed greens (LU/reconstruction evidence)."""
    for name, green in channels:
        stats = green.spectral_stats
        guard_row(
            entry, f"{name}.factorizations", stats["factorizations"], expected_reps
        )
        guard_row(
            entry,
            f"{name}.representative_kpoints",
            stats["representative_kpoints"],
            expected_reps,
        )
        guard_row(entry, f"{name}.full_kpoints", stats["full_kpoints"], expected_full)
        guard_row(
            entry,
            f"{name}.scalar_partner_expansions",
            stats["scalar_partner_expansions"],
            0 if partner_zero else None,
            note=None
            if partner_zero
            else "measured; folded representative sets expand -k partners",
        )
        info_row(entry, f"{name}.energy_evaluations", stats["energy_evaluations"])
        info_row(entry, f"{name}.import_seconds", rounded(stats["import_seconds"], 4))
        info_row(
            entry,
            f"{name}.preparation_seconds",
            rounded(stats["preparation_seconds"], 4),
        )
        info_row(entry, f"{name}.source_precision", stats["source_precision"])


def contour_guards(entry, exchange, oracle):
    guard_row(
        entry,
        "contour.path identical to oracle",
        np.asarray(exchange.contour.path),
        np.asarray(oracle.contour.path),
    )
    info_row(entry, "contour.kind", type(exchange.contour).__name__)
    info_row(entry, "contour.nz", len(np.asarray(exchange.contour.path)))
    info_row(entry, "contour.path", np.asarray(exchange.contour.path))
    for attribute in ("integration_method", "emin", "emax", "smearing"):
        if hasattr(exchange, attribute):
            info_row(entry, f"exchange.{attribute}", getattr(exchange, attribute))


def quadrature_guards(entry, green, expected_kpts, expected_weights, shifted):
    expected_kpts = np.asarray(expected_kpts, dtype=float)
    expected_weights = np.asarray(expected_weights, dtype=float)
    if shifted:
        difference = np.abs(np.asarray(green.kpts, dtype=float) - expected_kpts)
        wrapped = np.abs(difference - np.round(difference))
        bound_guard_row(
            entry,
            "requested kpoints resolve within 1e-8 wrapped",
            float(wrapped.max(initial=0.0)),
            1e-8,
            note="reciprocal wrapping: wrapped absolute tolerance 1e-8, zero relative",
        )
        info_row(
            entry,
            "requested_kpoints_wrapped_max_deviation",
            rounded(float(wrapped.max(initial=0.0))),
        )
    else:
        deviation = float(
            np.abs(np.asarray(green.kpts, dtype=float) - expected_kpts).max(initial=0.0)
        )
        bound_guard_row(
            entry,
            "requested kpoints preserved (<=1e-8)",
            deviation,
            1e-8,
            note="canonical wrapped matching tolerance; exact for explicitly requested points",
        )
    weight_deviation = float(
        np.abs(np.asarray(green.kweights, dtype=float) - expected_weights).max(
            initial=0.0
        )
    )
    bound_guard_row(
        entry,
        "requested kweights preserved (<=1e-8)",
        weight_deviation,
        1e-8,
    )
    info_row(entry, "requested_kpoints", expected_kpts)
    info_row(entry, "requested_kweights", expected_weights)


def case_pass(entry):
    if entry["errors"]:
        return False
    if any(row.get("pass") is False for row in entry["arrays"]):
        return False
    if any(row.get("pass") is False for row in entry["guards"]):
        return False
    if any(not info.get("match", True) for info in entry["keysets"].values()):
        return False
    return True


# --------------------------------------------------------------------------- #
# run-context helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def green_mode(mode):
    previous = os.environ.get("TB2J_SIESTA_GREEN_MODE")
    if mode == "default":
        os.environ.pop("TB2J_SIESTA_GREEN_MODE", None)
    else:
        os.environ["TB2J_SIESTA_GREEN_MODE"] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TB2J_SIESTA_GREEN_MODE", None)
        else:
            os.environ["TB2J_SIESTA_GREEN_MODE"] = previous


@contextlib.contextmanager
def case_output_log(path):
    """Keep verbose library stdout in the work root as raw run evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream, contextlib.redirect_stdout(stream):
        yield


def finalize_exchange(exchange):
    if exchange is not None:
        with contextlib.suppress(Exception):
            exchange.finalize()


# --------------------------------------------------------------------------- #
# case runners
# --------------------------------------------------------------------------- #
def run_collinear_baselines(report, ctx, params):
    """collinear/full, collinear/internal_fold, collinear/representative.

    All three compare against one same-mode ordinary ExchangeCL2 oracle with
    complete_spectrum=True, exactly like the permanent story-006 tests.
    """
    from test_siesta_wfsx_exchange import ordinary_exchange
    from test_siesta_wfsx_green import source_model

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx
    from TB2J.io_exchange import SpinIO

    mode_dir = ctx["mode_dir"]
    base = mode_dir / "collinear"
    try:
        oracle = ordinary_exchange(COLLINEAR_FULL, **params)
        oracle.calculate_all()
        oracle.write_output(path=base / "oracle_output")
        expected = SpinIO.load_pickle(path=base / "oracle_output")
    except Exception:
        entry = new_entry("collinear/oracle", ctx["mode"], COLLINEAR_FULL, base)
        entry["errors"].append(traceback.format_exc())
        entry["pass"] = False
        report["cases"].append(entry)
        print(
            f"[story008] mode={ctx['mode']} case=collinear/oracle: "
            f"ordinary oracle failed; collinear baselines skipped, evidence preserved",
            flush=True,
        )
        return
    tol_hs = TOLERANCES["H_S_H0_Delta_HR0"]
    tol_rho = TOLERANCES["populations_charges_spins"]
    unreduced = None
    unreduced_output = None
    try:
        for case_id, fixture, fold, reps in (
            ("collinear/full", COLLINEAR_FULL, False, 5),
            ("collinear/internal_fold", COLLINEAR_FULL, True, 3),
            ("collinear/representative", COLLINEAR_REDUCED, False, 3),
        ):
            slug = case_id.split("/", 1)[1]
            folder = base / slug
            folder.mkdir(parents=True, exist_ok=True)
            entry = new_entry(case_id, ctx["mode"], fixture, folder)
            started = time.perf_counter()
            exchange = None
            try:
                with case_output_log(folder / "run_log.txt"):
                    model, metadata, manifest, _ = source_model(folder, fixture)
                    record_inputs(entry, folder, fixture)
                    entry["requested"] = {
                        "kpoints": "source scf mesh (default request)",
                        "weights": "uniform 1/n",
                        "reduce_scalar": fold,
                    }
                    exchange = prepare_exchange_siesta_wfsx(
                        model.eigenstate_source,
                        reduce_scalar=fold,
                        cache_path=folder / "cache",
                        **params,
                    )
                    compare_green_pair(entry, exchange.Gup, oracle.Gup, "up")
                    compare_green_pair(entry, exchange.Gdn, oracle.Gdn, "dn")
                    compare_array(
                        entry,
                        "Delta",
                        "HR0 up - HR0 dn",
                        exchange.Delta,
                        oracle.Delta,
                        tol_hs["atol"],
                        tol_hs["rtol"],
                    )
                    contour_guards(entry, exchange, oracle)
                    # The default request is the TB2J mesh, not native file order.
                    original_points = np.asarray(oracle.Gup.kpts, dtype=float)
                    for channel, green in (("up", exchange.Gup), ("dn", exchange.Gdn)):
                        quadrature_guards(
                            entry,
                            green,
                            original_points,
                            np.full(len(original_points), 1.0 / len(original_points)),
                            shifted=False,
                        )
                    snapshot_guards(
                        entry,
                        exchange,
                        (("up", exchange.Gup), ("dn", exchange.Gdn)),
                        expected_reps=reps,
                        expected_full=len(original_points),
                        partner_zero=(reps == 5),
                    )
                    exchange.calculate_all()
                    snapshot_guards(
                        entry,
                        exchange,
                        (("up", exchange.Gup), ("dn", exchange.Gdn)),
                        expected_reps=reps,
                        expected_full=len(original_points),
                        partner_zero=(reps == 5),
                    )
                    compare_array(
                        entry,
                        "charges",
                        "exchange atoms",
                        exchange.charges,
                        oracle.charges,
                        tol_rho["atol"],
                        tol_rho["rtol"],
                    )
                    compare_array(
                        entry,
                        "spinat",
                        "exchange atoms x 3",
                        exchange.spinat,
                        oracle.spinat,
                        tol_rho["atol"],
                        tol_rho["rtol"],
                    )
                    compare_value_dict(
                        entry,
                        "exchange_Jdict",
                        exchange.exchange_Jdict,
                        oracle.exchange_Jdict,
                        TOLERANCES["exchange_tensors"]["atol"],
                        TOLERANCES["exchange_tensors"]["rtol"],
                    )
                    exchange.write_output(path=folder / "wfsx_output")
                    actual = SpinIO.load_pickle(path=folder / "wfsx_output")
                    compare_spinio_pair(
                        entry,
                        actual,
                        expected,
                        COLLINEAR_TENSOR_FLAGS,
                        COLLINEAR_TENSOR_NOTE,
                    )
                    if reps == 5:
                        unreduced, unreduced_output = exchange, actual
                    else:
                        if unreduced is None:
                            raise RuntimeError("unreduced WFSX reference failed")
                        first_direct = len(entry["arrays"])
                        compare_green_pair(entry, exchange.Gup, unreduced.Gup, "up")
                        compare_green_pair(entry, exchange.Gdn, unreduced.Gdn, "dn")
                        for quantity in ("Delta", "charges", "spinat"):
                            tolerance = tol_hs if quantity == "Delta" else tol_rho
                            compare_array(
                                entry,
                                quantity,
                                "unreduced WFSX reference",
                                getattr(exchange, quantity),
                                getattr(unreduced, quantity),
                                tolerance["atol"],
                                tolerance["rtol"],
                            )
                        compare_spinio_pair(
                            entry,
                            actual,
                            unreduced_output,
                            COLLINEAR_TENSOR_FLAGS,
                            COLLINEAR_TENSOR_NOTE,
                        )
                        for row in entry["arrays"][first_direct:]:
                            row["comparison_reference"] = "unreduced WFSX"
            except Exception:
                entry["errors"].append(traceback.format_exc())
            finally:
                if exchange is not unreduced:
                    finalize_exchange(exchange)
            entry["seconds"]["wall"] = rounded(time.perf_counter() - started, 3)
            _close_case(report, entry)
    finally:
        finalize_exchange(oracle)
        finalize_exchange(unreduced)


COLLINEAR_VARIANTS = (
    (
        "collinear/shuffle_permute_wrap",
        {
            "permute_file": PERMUTATION,
            "indices": REQUEST_ORDER,
            "shift": WRAPPING,
            "reps": 3,
            "partner_zero": False,
        },
    ),
    (
        "collinear/gamma_only",
        {
            "permute_file": None,
            "indices": None,  # Select Gamma by coordinate, never native file order.
            "shift": None,
            "reps": 1,
            "partner_zero": True,
        },
    ),
    (
        "collinear/nonclosed_shifted",
        {
            "permute_file": None,
            "indices": NONCLOSED_INDICES,
            "shift": NONCLOSED_SHIFT,
            "reps": 3,
            "partner_zero": True,
        },
    ),
)


def run_collinear_variant(report, ctx, params, case_id, spec):
    """Shuffled file / permuted order / wrapping / explicit-subset variants.

    Both the oracle and the WFSX consumer receive the identical explicit
    kpts/kweights; nothing adds a k point. The controlled file shuffle and
    request permutation reuse the permanent mutation helpers on copies.
    """
    from test_siesta_wfsx_exchange import ordinary_exchange
    from test_siesta_wfsx_green import permute_k_records, source_model

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx
    from TB2J.io_exchange import SpinIO

    mode_dir = ctx["mode_dir"]
    folder = mode_dir / "collinear_variants" / case_id.split("/", 1)[1]
    folder.mkdir(parents=True, exist_ok=True)
    entry = new_entry(case_id, ctx["mode"], COLLINEAR_FULL, folder)
    started = time.perf_counter()
    oracle = None
    exchange = None
    tol_hs = TOLERANCES["H_S_H0_Delta_HR0"]
    tol_rho = TOLERANCES["populations_charges_spins"]
    try:
        with case_output_log(folder / "run_log.txt"):
            model, metadata, manifest, base_points = source_model(
                folder, COLLINEAR_FULL
            )
            mutation = None
            if spec["permute_file"]:
                permute_k_records(
                    model.eigenstate_source.wfsx_path, manifest, spec["permute_file"]
                )
                mutation = f"permute_k_records order={spec['permute_file']} on the copy"
            record_inputs(entry, folder, COLLINEAR_FULL, mutation=mutation)
            indices = spec["indices"]
            if case_id == "collinear/gamma_only":
                wrapped = np.asarray(base_points) - np.rint(base_points)
                indices = np.flatnonzero(np.max(np.abs(wrapped), axis=1) <= 1e-8)
                if len(indices) != 1:
                    raise RuntimeError(
                        f"expected exactly one Gamma coordinate, got {indices}"
                    )
                guard_row(
                    entry,
                    "Gamma selected by wrapped coordinate",
                    bool(np.all(np.abs(wrapped[indices]) <= 1e-8)),
                    True,
                )
            requested = np.asarray(base_points)[indices]
            if spec["shift"]:
                requested = requested + np.asarray(spec["shift"], dtype=float)
            weights = np.full(len(requested), 1.0 / len(requested))
            entry["requested"] = {
                "kpoints": np.round(requested, 10).tolist(),
                "weights": weights.tolist(),
                "indices": np.asarray(indices, dtype=int).tolist(),
                "reciprocal_shift": list(spec["shift"]) if spec["shift"] else None,
                "file_permutation": list(spec["permute_file"])
                if spec["permute_file"]
                else None,
                "reduce_scalar": True,
            }
            variant_params = {**params, "kpts": requested, "kweights": weights}
            oracle = ordinary_exchange(COLLINEAR_FULL, **variant_params)
            oracle.calculate_all()
            oracle.write_output(path=folder / "oracle_output")
            expected = SpinIO.load_pickle(path=folder / "oracle_output")
            exchange = prepare_exchange_siesta_wfsx(
                model.eigenstate_source,
                reduce_scalar=True,
                cache_path=folder / "cache",
                **variant_params,
            )
            compare_green_pair(entry, exchange.Gup, oracle.Gup, "up")
            compare_green_pair(entry, exchange.Gdn, oracle.Gdn, "dn")
            compare_array(
                entry,
                "Delta",
                "HR0 up - HR0 dn",
                exchange.Delta,
                oracle.Delta,
                tol_hs["atol"],
                tol_hs["rtol"],
            )
            contour_guards(entry, exchange, oracle)
            shifted = bool(spec["shift"])
            for channel, green in (("up", exchange.Gup), ("dn", exchange.Gdn)):
                quadrature_guards(entry, green, requested, weights, shifted=shifted)
            snapshot_guards(
                entry,
                exchange,
                (("up", exchange.Gup), ("dn", exchange.Gdn)),
                expected_reps=spec["reps"],
                expected_full=len(requested),
                partner_zero=spec["partner_zero"],
            )
            exchange.calculate_all()
            snapshot_guards(
                entry,
                exchange,
                (("up", exchange.Gup), ("dn", exchange.Gdn)),
                expected_reps=spec["reps"],
                expected_full=len(requested),
                partner_zero=spec["partner_zero"],
            )
            compare_array(
                entry,
                "charges",
                "exchange atoms",
                exchange.charges,
                oracle.charges,
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            compare_array(
                entry,
                "spinat",
                "exchange atoms x 3",
                exchange.spinat,
                oracle.spinat,
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            compare_value_dict(
                entry,
                "exchange_Jdict",
                exchange.exchange_Jdict,
                oracle.exchange_Jdict,
                TOLERANCES["exchange_tensors"]["atol"],
                TOLERANCES["exchange_tensors"]["rtol"],
            )
            exchange.write_output(path=folder / "wfsx_output")
            actual = SpinIO.load_pickle(path=folder / "wfsx_output")
            compare_spinio_pair(
                entry, actual, expected, COLLINEAR_TENSOR_FLAGS, COLLINEAR_TENSOR_NOTE
            )
    except Exception:
        entry["errors"].append(traceback.format_exc())
    finally:
        finalize_exchange(exchange)
        finalize_exchange(oracle)
    entry["seconds"]["wall"] = rounded(time.perf_counter() - started, 3)
    _close_case(report, entry)


def run_nc4_case(report, ctx, params, case_id, spec):
    """Full nspin=4 spinor exchange against the ordinary ExchangeNCL oracle."""
    from test_siesta_wfsx_exchange import ordinary_exchange
    from test_siesta_wfsx_green import permute_k_records, source_model

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx
    from TB2J.io_exchange import SpinIO

    mode_dir = ctx["mode_dir"]
    slug = case_id.split("/", 1)[1]
    folder = mode_dir / "nc4" / slug
    folder.mkdir(parents=True, exist_ok=True)
    entry = new_entry(case_id, ctx["mode"], SPINOR_FULL, folder)
    started = time.perf_counter()
    oracle = None
    exchange = None
    tol_hs = TOLERANCES["H_S_H0_Delta_HR0"]
    tol_rho = TOLERANCES["populations_charges_spins"]
    try:
        with case_output_log(folder / "run_log.txt"):
            model, metadata, manifest, base_points = source_model(folder, SPINOR_FULL)
            mutation = None
            if spec.get("permute_file"):
                permute_k_records(
                    model.eigenstate_source.wfsx_path, manifest, spec["permute_file"]
                )
                mutation = f"permute_k_records order={spec['permute_file']} on the copy"
            record_inputs(entry, folder, SPINOR_FULL, mutation=mutation)
            if spec.get("indices") is not None:
                requested = np.asarray(base_points)[spec["indices"]]
                if spec.get("shift"):
                    requested = requested + np.asarray(spec["shift"], dtype=float)
                weights = np.full(len(requested), 1.0 / len(requested))
                entry["requested"] = {
                    "kpoints": np.round(requested, 10).tolist(),
                    "weights": weights.tolist(),
                    "file_permutation": list(spec["permute_file"])
                    if spec.get("permute_file")
                    else None,
                    "reciprocal_shift": list(spec["shift"])
                    if spec.get("shift")
                    else None,
                    "reduce_scalar": False,
                }
                variant_params = {**params, "kpts": requested, "kweights": weights}
                shifted = bool(spec.get("shift"))
            else:
                requested = np.asarray(metadata["scf_mesh"]["kpoints"], dtype=float)
                weights = np.full(len(requested), 1.0 / len(requested))
                entry["requested"] = {
                    "kpoints": "source scf mesh (default request)",
                    "reduce_scalar": False,
                }
                variant_params = params
                shifted = False
            oracle = ordinary_exchange(SPINOR_FULL, **variant_params)
            oracle.calculate_all()
            oracle.write_output(path=folder / "oracle_output")
            expected = SpinIO.load_pickle(path=folder / "oracle_output")
            exchange = prepare_exchange_siesta_wfsx(
                model.eigenstate_source,
                cache_path=folder / "cache",
                **variant_params,
            )
            compare_green_pair(entry, exchange.G, oracle.G, "spinor")
            compare_array(
                entry,
                "HR0",
                "R0 onsite",
                exchange.HR0,
                oracle.HR0,
                tol_hs["atol"],
                tol_hs["rtol"],
            )
            compare_array(
                entry,
                "rho",
                "mixed spinor population",
                exchange.rho,
                oracle.rho,
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            contour_guards(entry, exchange, oracle)
            quadrature_guards(entry, exchange.G, requested, weights, shifted=shifted)
            snapshot_guards(
                entry,
                exchange,
                (("spinor", exchange.G),),
                expected_reps=spec.get("reps", 5),
                expected_full=len(requested),
                partner_zero=True,
            )
            exchange.calculate_all()
            snapshot_guards(
                entry,
                exchange,
                (("spinor", exchange.G),),
                expected_reps=spec.get("reps", 5),
                expected_full=len(requested),
                partner_zero=True,
            )
            compare_array(
                entry,
                "charges",
                "exchange atoms",
                exchange.charges,
                oracle.charges,
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            compare_array(
                entry,
                "spinat",
                "exchange atoms x 3",
                exchange.spinat,
                oracle.spinat,
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            tol_j = TOLERANCES["exchange_tensors"]
            compare_value_dict(
                entry,
                "exchange_Jdict",
                exchange.exchange_Jdict,
                oracle.exchange_Jdict,
                tol_j["atol"],
                tol_j["rtol"],
            )
            compare_value_dict(
                entry, "Jani", exchange.Jani, oracle.Jani, tol_j["atol"], tol_j["rtol"]
            )
            compare_value_dict(
                entry, "DMI", exchange.DMI, oracle.DMI, tol_j["atol"], tol_j["rtol"]
            )
            exchange.write_output(path=folder / "wfsx_output")
            actual = SpinIO.load_pickle(path=folder / "wfsx_output")
            compare_spinio_pair(
                entry,
                actual,
                expected,
                SPINOR_TENSOR_FLAGS,
                "full spinor tensors: Jiso+Jani+DMI+SIA enabled",
            )
    except Exception:
        entry["errors"].append(traceback.format_exc())
    finally:
        finalize_exchange(exchange)
        finalize_exchange(oracle)
    entry["seconds"]["wall"] = rounded(time.perf_counter() - started, 3)
    _close_case(report, entry)


def run_half_boundary_case(report, ctx, params):
    """Coordinate-only nonzero-half self-inverse boundary (structural only).

    Reuses the inherited HamiltonIO regression
    ``tests/test_wfsx.py::test_nonzero_self_inverse_coordinate_is_not_doubled``
    on a copy: only the k-point headers of one record move to half the
    reciprocal z coordinate. The state coefficients stay those of the original
    record, so this case asserts consumer counting/finiteness only and never
    claims operator equivalence at the synthetic coordinate.
    """
    from test_siesta_wfsx_green import rewrite_test_records, source_model

    from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

    mode_dir = ctx["mode_dir"]
    folder = mode_dir / "half_boundary"
    folder.mkdir(parents=True, exist_ok=True)
    entry = new_entry(
        "collinear/half_boundary",
        ctx["mode"],
        COLLINEAR_FULL,
        folder,
        limits_note=(
            "coordinate-only synthetic self-inverse boundary; the state "
            "coefficients remain those of the original k record, so no H/S "
            "operator equivalence is claimed at the synthetic coordinate"
        ),
    )
    started = time.perf_counter()
    exchange = None
    try:
        with case_output_log(folder / "run_log.txt"):
            from sisl.unit.siesta import unit_convert

            model, metadata, manifest, _ = source_model(folder, COLLINEAR_FULL)
            cart_z = (
                0.5
                * 2
                * np.pi
                / (metadata["cell_angstrom"][2][2] * unit_convert("Ang", "Bohr"))
            )
            stride = 3 + 3 * manifest["states"]["no_u"]

            def half_coordinate(records):
                for spin in (0, 1):
                    records[4 + (4 + spin) * stride][4:].view(np.float64)[:3] = [
                        0,
                        0,
                        cart_z,
                    ]

            rewrite_test_records(model.eigenstate_source.wfsx_path, half_coordinate)
            record_inputs(
                entry,
                folder,
                COLLINEAR_FULL,
                mutation="k-point headers of record ik=4 set to [0,0,0.5c_z] (coordinate-only)",
            )
            requested = np.array([[0.0, 0.0, -0.5]])
            weights = np.array([1.0])
            entry["requested"] = {
                "kpoints": requested.tolist(),
                "weights": weights.tolist(),
                "reduce_scalar": True,
                "kind": "coordinate-only self-inverse boundary",
            }
            variant_params = {**params, "kpts": requested, "kweights": weights}
            exchange = prepare_exchange_siesta_wfsx(
                model.eigenstate_source,
                reduce_scalar=True,
                cache_path=folder / "cache",
                **variant_params,
            )
            tol_rho = TOLERANCES["populations_charges_spins"]
            for channel, green in (("up", exchange.Gup), ("dn", exchange.Gdn)):
                spectral_map = green._spectral_map
                guard_row(
                    entry,
                    f"{channel}.map.record_indices",
                    spectral_map.record_indices.tolist(),
                    [[4, 5]],
                )
                guard_row(
                    entry,
                    f"{channel}.map.representative_count",
                    len(spectral_map.representative_kpoints),
                    1,
                )
                guard_row(
                    entry,
                    f"{channel}.map.representative_inverted_any",
                    bool(np.any(spectral_map.representative_inverted)),
                    False,
                )
                guard_row(
                    entry,
                    f"{channel}.map.requires_operator_validation",
                    bool(spectral_map.requires_operator_validation),
                    False,
                )
                guard_row(
                    entry,
                    f"{channel}.map.weights",
                    np.asarray(spectral_map.mesh.weights, dtype=float),
                    weights,
                )
            snapshot_guards(
                entry,
                exchange,
                (("up", exchange.Gup), ("dn", exchange.Gdn)),
                expected_reps=1,
                expected_full=1,
                partner_zero=True,
            )
            quadrature_guards(entry, exchange.Gup, requested, weights, shifted=False)
            finite = True
            for z in Z_POINTS:
                finite = (
                    guard_row(
                        entry,
                        f"G(k,z) finite at half-boundary z={z_label(z)}",
                        bool(
                            np.isfinite(np.asarray(exchange.Gup.get_Gk_all(z))).all()
                            and np.isfinite(
                                np.asarray(exchange.Gdn.get_Gk_all(z))
                            ).all()
                        ),
                        True,
                    )["pass"]
                    and finite
                )
            finite = (
                guard_row(
                    entry,
                    "rho(R) finite at R0+-z",
                    bool(
                        np.isfinite(np.asarray(exchange.Gup.get_rho_R(R_POINTS))).all()
                        and np.isfinite(
                            np.asarray(exchange.Gdn.get_rho_R(R_POINTS))
                        ).all()
                    ),
                    True,
                )["pass"]
                and finite
            )
            total_population = np.trace(np.asarray(exchange.rho_up)) + np.trace(
                np.asarray(exchange.rho_dn)
            )
            compare_array(
                entry,
                "population trace",
                "trace(rho_up) + trace(rho_dn) vs declared electrons",
                np.array([total_population]),
                np.array([metadata["electron_count"]]),
                tol_rho["atol"],
                tol_rho["rtol"],
            )
            info_row(
                entry,
                "self-inverse counting",
                "one representative slot, no inverted partner, counted once",
                note=f"[{MATHDOC}#scalar_partner_transpose]",
            )
    except Exception:
        entry["errors"].append(traceback.format_exc())
    finally:
        finalize_exchange(exchange)
    entry["seconds"]["wall"] = rounded(time.perf_counter() - started, 3)
    _close_case(report, entry)


def run_spinor_refusal_case(report, ctx, params):
    """Scalar reduction of full spinor exchange must be refused by name."""
    from test_siesta_wfsx_green import source_model

    mode_dir = ctx["mode_dir"]
    folder = mode_dir / "spinor_reduction_refused"
    folder.mkdir(parents=True, exist_ok=True)
    entry = new_entry("nc4/spinor_reduction_refused", ctx["mode"], SPINOR_FULL, folder)
    started = time.perf_counter()
    try:
        with case_output_log(folder / "run_log.txt"):
            import scipy.linalg
            from HamiltonIO.siesta.sisl_wrapper import SislParser
            from HamiltonIO.siesta.wfsx import UnsupportedSpinMode

            from TB2J.interfaces.siesta_interface import prepare_exchange_siesta_wfsx

            model, _, _, _ = source_model(folder, SPINOR_FULL)
            record_inputs(entry, folder, SPINOR_FULL)
            counters = {"lu_factor": 0, "SislParser.__init__": 0}
            real_lu = scipy.linalg.lu_factor
            real_init = SislParser.__init__

            def counting_lu(*args, **kwargs):
                counters["lu_factor"] += 1
                return real_lu(*args, **kwargs)

            def counting_init(inner_self, *args, **kwargs):
                counters["SislParser.__init__"] += 1
                return real_init(inner_self, *args, **kwargs)

            scipy.linalg.lu_factor = counting_lu
            SislParser.__init__ = counting_init
            refused = None
            unexpected = None
            try:
                try:
                    unexpected = prepare_exchange_siesta_wfsx(
                        model.eigenstate_source,
                        reduce_scalar=True,
                        cache_path=folder / "cache",
                        **params,
                    )
                except UnsupportedSpinMode as exc:
                    refused = f"{type(exc).__module__}.{type(exc).__name__}"
                except Exception as exc:  # wrong category is failure evidence
                    refused = f"WRONG CATEGORY {type(exc).__module__}.{type(exc).__name__}: {exc}"
            finally:
                finalize_exchange(unexpected)
                scipy.linalg.lu_factor = real_lu
                SislParser.__init__ = real_init
            entry["requested"] = {"reduce_scalar": True, "kind": "must be refused"}
            guard_row(
                entry,
                "refusal category",
                refused,
                "HamiltonIO.siesta.wfsx.UnsupportedSpinMode",
            )
            guard_row(entry, "no exchange constructed", unexpected is None, True)
            guard_row(
                entry, "measured scipy.linalg.lu_factor calls", counters["lu_factor"], 0
            )
            guard_row(
                entry,
                "measured SislParser constructions",
                counters["SislParser.__init__"],
                0,
            )
            info_row(entry, "refusal message", refused)
    except Exception:
        entry["errors"].append(traceback.format_exc())
    entry["seconds"]["wall"] = rounded(time.perf_counter() - started, 3)
    _close_case(report, entry)


def _close_case(report, entry):
    entry["pass"] = case_pass(entry)
    report["cases"].append(entry)
    worst = None
    ratios = [
        row.get("worst_tolerance_ratio")
        for row in entry["arrays"]
        if row.get("worst_tolerance_ratio") is not None
    ]
    if ratios:
        worst = max(ratios)
    print(
        f"[story008] mode={entry['green_mode']} case={entry['case']}: "
        f"arrays={len(entry['arrays'])} worst_ratio={worst} "
        f"pass={entry['pass']} ({entry['seconds'].get('wall')}s)",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# trace matrix (story 007 entries)
# --------------------------------------------------------------------------- #
def evaluate_trace_row(row):
    """Mirror the agreed row-evidence contract; return failed-check strings."""
    checks = []

    def require(condition, message):
        try:
            ok = bool(condition)
        except Exception:
            ok = False
        if not ok:
            checks.append(message)

    fixture = row.get("fixture")
    collinear = fixture == "collinear_nspin2_reduced"
    summary = row.get("run_summary") or {}
    guard = row.get("guard") or {}
    require(row.get("returncode") == 0, "returncode != 0")
    eigensolver = summary.get("electronic_eigensolver") or {}
    require(eigensolver.get("calls") == 0, "measured electronic eigensolver calls != 0")
    require(
        eigensolver.get("seconds") == 0, "measured electronic eigensolver seconds != 0"
    )
    require(
        summary.get("green_mode") == row.get("green_mode"),
        "run_summary green_mode mismatch",
    )
    require(
        (summary.get("kpoints") or {}).get("requested") == 5, "requested kpoints != 5"
    )
    require(
        summary.get("spin_mode") == (2 if collinear else 4),
        "run_summary spin_mode mismatch",
    )
    require(
        (summary.get("matrix_sizes") or {}).get("basis") == (18 if collinear else 36),
        "run_summary basis size mismatch",
    )
    require(guard.get("electronic_guard_hits") == {}, "electronic guard hits present")
    require(
        guard.get("cfr_auxiliary_pole_eig_calls", 0) >= 1,
        "CFR auxiliary pole eigensolve not observed",
    )
    get_sile = guard.get("sisl_get_sile") or {}
    require(get_sile.get("calls", 0) >= 1, "ordinary sisl.get_sile not observed")
    require(get_sile.get("unexpected") == [], "unexpected sisl.get_sile paths")
    paths = get_sile.get("paths") or []
    require(
        len(set(paths)) == 1 and paths and paths[0].endswith("states.WFSX"),
        "get_sile did not open only states.WFSX",
    )
    memmaps = guard.get("memmaps") or {}
    require(memmaps.get("created", 0) >= 1, "no memmap lifecycle observed")
    require(
        memmaps.get("closed") == memmaps.get("created"), "memmap close count mismatch"
    )
    require(memmaps.get("open_at_exit") == [], "memmaps left open at exit")
    require(
        (guard.get("reader_close") or {}).get("events", 0) >= 1,
        "native reader close not observed",
    )
    require(
        (guard.get("audit_opens") or {}).get("forbidden") == [],
        "forbidden audit opens present",
    )
    inputs = row.get("inputs") or {}
    require(inputs.get("preserved") is True, "input files not byte-preserved")
    require(
        set(inputs.get("names") or []) == {"metadata.json", "states.WFSX"},
        "unexpected input file set",
    )
    summary_inputs = summary.get("inputs") or []
    require(
        bool(summary_inputs)
        and summary_inputs[0].get("sha256")
        == (inputs.get("sha256") or {}).get("states.WFSX"),
        "run_summary WFSX hash mismatch",
    )
    cache = row.get("cache") or {}
    require(cache.get("listing") == ["user.txt"], "cache listing != [user.txt]")
    require(cache.get("marker_preserved") is True, "user cache marker not preserved")
    require(
        cache.get("owned_children_removed") is True, "owned cache children not removed"
    )
    require(
        (row.get("output") or {}).get("tb2j_pickle") is True,
        "TB2J.pickle output missing",
    )
    native = row.get("native")
    if native is not None:
        from test_siesta_wfsx_no_hs_execution import TRACE_METHOD

        require(native.get("method") == TRACE_METHOD, "unexpected native trace method")
        syscalls = native.get("syscalls") or {}
        require(
            set(syscalls) == {"open", "openat", "openat2"},
            "unexpected native syscall set",
        )
        require(
            native.get("opens_total") == sum(syscalls.values())
            and native.get("opens_total", 0) > 0,
            "native open counts inconsistent",
        )
        require(
            native.get("forbidden_h_s_opens") == [],
            "native forbidden H/S opens present",
        )
        log = native.get("log")
        require(
            bool(log) and (Path(row.get("row_root", "")) / log).is_file(),
            "native strace log missing",
        )
    if row.get("entry") == "console":
        require(
            row.get("entry_impl") in {"installed-script", "sys-exit-wrapper"},
            "unexpected console entry implementation",
        )
    return checks


def run_trace_matrix(work_root, native_trace):
    section = {
        "module": "test_siesta_wfsx_no_hs_execution.trace_entry_cases",
        "work_root": str(work_root / "trace_matrix"),
        "native_trace": bool(native_trace),
        "expected_rows": EXPECTED_TRACE_ROWS,
        "rows": [],
        "row_checks": {},
        "set_errors": [],
        "errors": [],
        "pass": False,
    }
    started = time.perf_counter()
    try:
        from test_siesta_wfsx_no_hs_execution import TRACE_METHOD, trace_entry_cases

        section["native_trace_method"] = TRACE_METHOD
        rows = trace_entry_cases(
            work_root=work_root / "trace_matrix", native_trace=native_trace
        )
        section["rows"] = jsonable(rows)
        failed = []
        for row in rows:
            checks = evaluate_trace_row(row)
            row_id = f"{row.get('entry')}-{row.get('fixture')}-{row.get('green_mode')}"
            section["row_checks"][row_id] = checks
            if checks:
                failed.append(row_id)
        observed = {
            (row.get("entry"), row.get("fixture"), row.get("green_mode"))
            for row in rows
        }
        expected = {
            (entry, fixture, mode)
            for fixture in TRACE_FIXTURES
            for entry in TRACE_ENTRIES
            for mode in GREEN_MODES
        }
        if observed != expected:
            section["set_errors"].append(
                f"observed entry set {sorted(map(str, observed))} != expected {sorted(map(str, expected))}"
            )
        section["rows_total"] = len(rows)
        section["rows_failed"] = failed
        section["pass"] = (
            not failed
            and not section["set_errors"]
            and len(rows) == EXPECTED_TRACE_ROWS
        )
    except Exception:
        section["errors"].append(traceback.format_exc())
    section["seconds"] = rounded(time.perf_counter() - started, 3)
    print(
        f"[story008] trace matrix: rows={section.get('rows_total', 0)} "
        f"native_trace={bool(native_trace)} pass={section['pass']} "
        f"({section['seconds']}s)",
        flush=True,
    )
    return section


# --------------------------------------------------------------------------- #
# provenance and reports
# --------------------------------------------------------------------------- #
def collect_provenance(hamiltonio_root):
    driver = Path(__file__).resolve()
    tb2j_root = TESTS_DIR.parent
    driver_status = git_capture(tb2j_root, "status", "--porcelain", "--", str(driver))
    tracked = git_capture(tb2j_root, "ls-files", "--", str(driver))
    provenance = {
        "story": STORY,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": sys.version.split()[0],
        "tb2j_worktree": {
            "root": str(tb2j_root),
            "git_head": git_capture(tb2j_root, "rev-parse", "HEAD"),
            "assignment_baseline": "e246fdb",
        },
        "hamiltonio_worktree": {
            "root": str(hamiltonio_root),
            "git_head": git_capture(hamiltonio_root, "rev-parse", "HEAD"),
            "assignment_baseline": "da9881f",
        },
        "report_driver": {
            "path": str(driver),
            "sha256": sha256_file(driver),
            "tracked_in_git": bool((tracked or "").strip()),
            "clean_in_git": driver_status == "",
            "note": "uncommitted report driver is expected; the content hash and "
            "worktree HEADs are recorded and never claimed to be commits",
        },
        "mathdoc": MATHDOC,
        "mathdoc_source_sha256": "b4a62424de04bf3e5d0f281b8d52b0953b1058e20bc9165f2a698dc3f9a8c8d1",
    }
    for section, key in (
        ("tb2j_worktree", "tb2j"),
        ("hamiltonio_worktree", "hamiltonio"),
    ):
        head = provenance[section]["git_head"]
        baseline = provenance[section]["assignment_baseline"]
        if head is None:
            provenance[section]["baseline_match"] = None
        else:
            provenance[section]["baseline_match"] = head.startswith(baseline)
    return provenance


def build_report(args, provenance, cases, trace_section, wall_seconds):
    failed_arrays = []
    failed_guards = []
    failed_keysets = []
    failed_cases = []
    total_arrays = 0
    for case in cases:
        total_arrays += len(case["arrays"])
        if not case["pass"]:
            failed_cases.append(case["case"])
        for row in case["arrays"]:
            if row.get("pass") is False:
                failed_arrays.append(
                    f"{case['green_mode']}/{case['case']}:{row['quantity']}[{row['key']}]"
                )
        for row in case["guards"]:
            if row.get("pass") is False:
                failed_guards.append(
                    f"{case['green_mode']}/{case['case']}:{row['guard']} observed={row['observed']} expected={row['expected']}"
                )
        for name, info in case["keysets"].items():
            if not info.get("match", True):
                failed_keysets.append(f"{case['green_mode']}/{case['case']}:{name}")
    warnings = []
    for section in ("tb2j_worktree", "hamiltonio_worktree"):
        if provenance[section]["baseline_match"] is False:
            warnings.append(
                f"{section} git head {provenance[section]['git_head']} does not start with the "
                f"pinned assignment baseline {provenance[section]['assignment_baseline']}"
            )
    if not provenance["report_driver"]["tracked_in_git"]:
        warnings.append(
            "report driver is not tracked in git yet; recorded by content hash only (expected)"
        )
    trace_ok = bool(trace_section.get("pass"))
    return {
        "schema": REPORT_SCHEMA,
        "story": STORY,
        "command": {"argv": sys.argv[1:], "native_trace": bool(args.native_trace)},
        "provenance": provenance,
        "thresholds": {
            name: {**values, "mathdoc": MATHDOC} for name, values in TOLERANCES.items()
        },
        "quantity_handles": QUANTITY_HANDLES,
        "cases": cases,
        "trace_matrix": trace_section,
        "scope_limits": SCOPE_LIMITS,
        "warnings": warnings,
        "summary": {
            "green_modes": list(GREEN_MODES),
            "cases_total": len(cases),
            "cases_failed": failed_cases,
            "arrays_total": total_arrays,
            "arrays_failed": failed_arrays,
            "guards_failed": failed_guards,
            "keysets_failed": failed_keysets,
            "trace_rows_total": trace_section.get("rows_total", 0),
            "trace_rows_failed": trace_section.get("rows_failed", []),
            "overall_pass": not failed_cases
            and not failed_arrays
            and not failed_guards
            and not failed_keysets
            and trace_ok,
            "wall_seconds": rounded(wall_seconds, 3),
        },
    }


def render_markdown(report):
    lines = []
    summary = report["summary"]
    provenance = report["provenance"]
    lines.append("# SIESTA WFSX full-array equivalence report (story 008)")
    lines.append("")
    lines.append(f"- Story: `{report['story']}`")
    lines.append(
        f"- Generated: {provenance['generated_utc']} UTC "
        f"(wall {summary['wall_seconds']} s)"
    )
    lines.append(
        f"- Command: `python tests/test_siesta_wfsx_equivalence.py "
        f"{' '.join(report['command']['argv'])}`"
    )
    lines.append(
        f"- TB2J worktree: `{provenance['tb2j_worktree']['root']}` head "
        f"`{provenance['tb2j_worktree']['git_head']}` (baseline "
        f"{provenance['tb2j_worktree']['assignment_baseline']}, match: "
        f"{provenance['tb2j_worktree']['baseline_match']})"
    )
    lines.append(
        f"- HamiltonIO worktree: `{provenance['hamiltonio_worktree']['root']}` head "
        f"`{provenance['hamiltonio_worktree']['git_head']}` (baseline "
        f"{provenance['hamiltonio_worktree']['assignment_baseline']}, match: "
        f"{provenance['hamiltonio_worktree']['baseline_match']})"
    )
    lines.append(
        f"- Report driver: `{provenance['report_driver']['path']}` sha256 "
        f"`{provenance['report_driver']['sha256']}` (tracked: "
        f"{provenance['report_driver']['tracked_in_git']}, clean: "
        f"{provenance['report_driver']['clean_in_git']}) - uncommitted report "
        "infrastructure, not a commit"
    )
    lines.append(f"- Identity citations: `{provenance['mathdoc']}`")
    lines.append("")
    lines.append(
        f"**Overall: {'PASS' if summary['overall_pass'] else 'FAIL'}** - "
        f"{summary['cases_total']} cases, {summary['arrays_total']} array rows, "
        f"{len(summary['arrays_failed'])} array failures, "
        f"{len(summary['guards_failed'])} guard failures, "
        f"{len(summary['keysets_failed'])} key-set failures, "
        f"trace rows {summary['trace_rows_total']} "
        f"({len(summary['trace_rows_failed'])} failed)."
    )
    if report["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Thresholds (unchanged Architecture table)")
    lines.append("")
    lines.append("| quantity group | atol | rtol | unit |")
    lines.append("| --- | --- | --- | --- |")
    for name, values in report["thresholds"].items():
        lines.append(
            f"| {name} | {values['atol']} | {values['rtol']} | {values['unit']} |"
        )
    lines.append("")
    lines.append("## Case results")
    lines.append("")
    lines.append(
        "| case | green mode | arrays | worst ratio | array fails | guard fails | errors | result |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in report["cases"]:
        ratios = [
            row["worst_tolerance_ratio"]
            for row in case["arrays"]
            if row.get("worst_tolerance_ratio") is not None
        ]
        worst = f"{max(ratios):.4g}" if ratios else "-"
        array_fails = sum(1 for row in case["arrays"] if row.get("pass") is False)
        guard_fails = sum(1 for row in case["guards"] if row.get("pass") is False)
        lines.append(
            f"| {case['case']} | {case['green_mode']} | {len(case['arrays'])} | {worst} "
            f"| {array_fails} | {guard_fails} | {len(case['errors'])} "
            f"| {'PASS' if case['pass'] else 'FAIL'} |"
        )
    lines.append("")
    failures = [line for line in _failure_lines(report)]
    if failures:
        lines.append("## Measured failures (evidence preserved, nothing widened)")
        lines.append("")
        lines.extend(failures)
    lines.append("")
    lines.append("## Entry trace matrix")
    lines.append("")
    trace = report["trace_matrix"]
    lines.append(
        f"- Source: `{trace['module']}` under `{trace['work_root']}` "
        f"(native_trace={trace['native_trace']}"
        + (
            f", method `{trace.get('native_trace_method')}`"
            if trace.get("native_trace")
            else ""
        )
        + f"), rows {trace.get('rows_total', 0)}/{trace['expected_rows']}"
    )
    lines.append("")
    lines.append(
        "| entry | fixture | green mode | impl | native | failed checks | result |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in trace.get("rows", []):
        row_id = f"{row.get('entry')}-{row.get('fixture')}-{row.get('green_mode')}"
        checks = trace["row_checks"].get(row_id, [])
        native = row.get("native")
        native_text = (
            f"opens={native.get('opens_total')}, forbidden={len(native.get('forbidden_h_s_opens') or [])}"
            if native
            else "-"
        )
        lines.append(
            f"| {row.get('entry')} | {row.get('fixture')} | {row.get('green_mode')} "
            f"| {row.get('entry_impl')} | {native_text} | {len(checks)} "
            f"| {'PASS' if not checks else 'FAIL'} |"
        )
    for set_error in trace.get("set_errors", []):
        lines.append(f"- entry-set mismatch: {set_error}")
    for error in trace.get("errors", []):
        first = error.strip().splitlines()[-1] if error.strip() else "empty traceback"
        lines.append(f"- trace matrix error: {first}")
    lines.append("")
    lines.append("## Scope and provenance limits")
    lines.append("")
    for limit in report["scope_limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    return "\n".join(lines)


def _failure_lines(report):
    for case in report["cases"]:
        prefix = f"{case['green_mode']}/{case['case']}"
        for row in case["arrays"]:
            if row.get("pass") is False:
                lines = []
                if "reason" in row:
                    lines.append(
                        f"- {prefix}: {row['quantity']} [{row['key']}] - {row['reason']} "
                        f"(shapes {row['shape']})"
                    )
                else:
                    lines.append(
                        f"- {prefix}: {row['quantity']} [{row['key']}] max_abs_error="
                        f"{row['max_abs_error']} atol={row['atol']} rtol={row['rtol']} "
                        f"worst_ratio={row['worst_tolerance_ratio']} elements={row['elements']}"
                    )
                yield from lines
        for row in case["guards"]:
            if row.get("pass") is False:
                yield (
                    f"- {prefix}: guard {row['guard']} observed={row['observed']} "
                    f"expected={row['expected']}"
                )
        for name, info in case["keysets"].items():
            if not info.get("match", True):
                yield (
                    f"- {prefix}: key set {name} mismatch: missing="
                    f"{info['missing_from_actual']} unexpected={info['unexpected_in_actual']}"
                )
        for error in case["errors"]:
            first = (
                error.strip().splitlines()[-1] if error.strip() else "empty traceback"
            )
            yield f"- {prefix}: case error: {first} (full traceback in the JSON report)"


def write_reports(report, args):
    output_json = args.output_json.resolve()
    output_md = args.output_md.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(jsonable(report), indent=1, allow_nan=False))
    output_md.write_text(render_markdown(report))
    return output_json, output_md


def write_failure_reports(args, headline, detail):
    """Honest minimal reports when the command cannot even start measuring."""
    driver = Path(__file__).resolve()
    provenance = {
        "story": STORY,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": sys.version.split()[0],
        "tb2j_worktree": {
            "root": str(TESTS_DIR.parent),
            "git_head": git_capture(TESTS_DIR.parent, "rev-parse", "HEAD"),
            "assignment_baseline": "e246fdb",
            "baseline_match": None,
        },
        "hamiltonio_worktree": {
            "root": None,
            "git_head": None,
            "assignment_baseline": "da9881f",
            "baseline_match": None,
        },
        "report_driver": {
            "path": str(driver),
            "sha256": sha256_file(driver),
            "tracked_in_git": bool(
                (
                    git_capture(TESTS_DIR.parent, "ls-files", "--", str(driver)) or ""
                ).strip()
            ),
            "clean_in_git": git_capture(
                TESTS_DIR.parent, "status", "--porcelain", "--", str(driver)
            )
            == "",
            "note": "uncommitted report driver is expected; content hash recorded, never claimed as a commit",
        },
        "mathdoc": MATHDOC,
        "mathdoc_source_sha256": "b4a62424de04bf3e5d0f281b8d52b0953b1058e20bc9165f2a698dc3f9a8c8d1",
    }
    trace_section = {
        "module": "test_siesta_wfsx_no_hs_execution.trace_entry_cases",
        "work_root": None,
        "native_trace": bool(args.native_trace),
        "expected_rows": EXPECTED_TRACE_ROWS,
        "rows": [],
        "row_checks": {},
        "set_errors": [],
        "errors": [detail],
        "pass": False,
    }
    report = build_report(args, provenance, [], trace_section, 0.0)
    report["summary"]["startup_error"] = headline
    report["summary"]["startup_detail"] = detail
    report["warnings"].append(headline)
    return write_reports(report, args)


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="test_siesta_wfsx_equivalence.py",
        description="Opt-in story-008 SIESTA WFSX full-array equivalence report "
        "(no permanent test rows; run with the story worktrees importable via "
        "PYTHONPATH).",
        epilog="Example: PYTHONPATH=<tb2j worktree>:<hamiltonio worktree> python "
        "tests/test_siesta_wfsx_equivalence.py --work-root /tmp/story008-run "
        "--output-json reports/equivalence.json --output-md reports/equivalence.md",
    )
    parser.add_argument(
        "--work-root",
        required=True,
        type=Path,
        help="directory for all run inputs, caches and trace artifacts",
    )
    parser.add_argument(
        "--output-json", required=True, type=Path, help="machine-readable report path"
    )
    parser.add_argument(
        "--output-md", required=True, type=Path, help="concise Markdown report path"
    )
    parser.add_argument(
        "--native-trace",
        action="store_true",
        default=False,
        help="wrap the real entry children in /bin/strace to capture native "
        "open/openat/openat2 calls (requires /bin/strace and the installed "
        "siesta2J.py console script)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    # Retain prior evidence; each invocation owns a fresh complete namespace.
    work_root = Path(tempfile.mkdtemp(prefix="equivalence-", dir=work_root))
    print(f"[story008] work root: {work_root}", flush=True)
    try:
        import HamiltonIO

        from TB2J.interfaces.siesta_interface import (
            prepare_exchange_siesta_wfsx,  # noqa: F401
        )
    except Exception as exc:
        headline = (
            "TB2J/HamiltonIO story worktrees are not importable; run with "
            "PYTHONPATH=<siesta-wfsx-tb2j worktree>:<siesta-wfsx-hamiltonio worktree>"
        )
        print(f"[story008] FAIL: {headline}: {exc}", flush=True)
        write_failure_reports(args, headline, traceback.format_exc())
        return 2
    try:
        from test_siesta_wfsx_exchange import exchange_params
    except Exception as exc:
        headline = "existing story helper tests are not importable from tests/"
        print(f"[story008] FAIL: {headline}: {exc}", flush=True)
        write_failure_reports(args, headline, traceback.format_exc())
        return 2

    hamiltonio_root = Path(HamiltonIO.__file__).resolve().parents[1]
    provenance = collect_provenance(hamiltonio_root)
    print(
        f"[story008] TB2J head {provenance['tb2j_worktree']['git_head']} "
        f"HamiltonIO head {provenance['hamiltonio_worktree']['git_head']}",
        flush=True,
    )
    report = {"cases": []}

    def safe_step(step, *args):
        """Never let one runner abort the remaining measurements."""
        try:
            step(*args)
        except Exception:
            report["cases"].append(
                {
                    "case": f"runner/{step.__name__}",
                    "green_mode": os.environ.get("TB2J_SIESTA_GREEN_MODE") or "default",
                    "fixture": None,
                    "folder": str(work_root),
                    "inputs": {},
                    "requested": {},
                    "notes": [],
                    "guards": [],
                    "arrays": [],
                    "keysets": {},
                    "errors": [traceback.format_exc()],
                    "seconds": {},
                    "pass": False,
                }
            )
            print(
                f"[story008] FAIL runner {step.__name__}: evidence preserved",
                flush=True,
            )

    for mode in GREEN_MODES:
        with green_mode(mode):
            mode_dir = work_root / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            ctx = {"mode": mode, "mode_dir": mode_dir}
            try:
                params = exchange_params(mode_dir / "outputs")
            except Exception:
                safe_step(run_collinear_baselines, report, ctx, {})
                continue
            safe_step(run_collinear_baselines, report, ctx, params)
            for case_id, spec in COLLINEAR_VARIANTS:
                safe_step(run_collinear_variant, report, ctx, params, case_id, spec)
            safe_step(
                run_nc4_case,
                report,
                ctx,
                params,
                "nc4/full",
                {"reps": 5, "indices": None},
            )
            safe_step(
                run_nc4_case,
                report,
                ctx,
                params,
                "nc4/shuffle_permute_wrap",
                {
                    "reps": 5,
                    "indices": REQUEST_ORDER,
                    "shift": WRAPPING,
                    "permute_file": PERMUTATION,
                },
            )
            safe_step(run_half_boundary_case, report, ctx, params)
            safe_step(run_spinor_refusal_case, report, ctx, params)
    cases = report["cases"]
    print(
        f"[story008] array matrix complete: {len(cases)} cases, "
        f"{sum(len(case['arrays']) for case in cases)} array rows",
        flush=True,
    )
    trace_section = run_trace_matrix(work_root, args.native_trace)
    full_report = build_report(
        args, provenance, cases, trace_section, time.perf_counter() - started
    )
    output_json, output_md = write_reports(full_report, args)
    overall = full_report["summary"]["overall_pass"]
    print(
        f"[story008] overall {'PASS' if overall else 'FAIL'}; "
        f"json={output_json} md={output_md}",
        flush=True,
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
