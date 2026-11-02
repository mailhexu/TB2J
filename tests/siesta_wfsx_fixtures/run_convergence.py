"""N/2N/3N mesh-convergence characterization for SIESTA WFSX exchange.

Opt-in scientific command (Story 009).  It regenerates the unchanged primitive
O2 collinear physical template through the existing Story 002 producer
(``measurement_producer.generate_measurement_case``, the bounded measurement
extension of the SHA256SUMS-pinned ``fixture_driver`` workflow) at N^3, (2N)^3
and (3N)^3 integration meshes, measures the matched ordinary/WFSX exchange pair
at each mesh through the shared benchmark driver
(``run_benchmarks.run_matched_pair``), and judges successive refinement against
the Architecture thresholds.

Convergence protocol (approved Architecture, quoted unchanged): maximum
tensor-element change <= 0.01 meV and <= 1 % relative change for elements
exceeding 0.1 meV, required on BOTH successive refinements (N->2N and 2N->3N).
Changes are computed per step; monotonic improvement is never assumed.  An
unmet threshold stays unconverged and is flagged as needing an Architecture
amendment; thresholds, meshes and tolerances are never widened and inputs are
never changed to manufacture a pass.  Zero elements above the 0.1 meV
reference floor make the relative criterion explicitly vacuous (recorded
count 0), never a NaN pass; any non-finite tensor element fails its step.

The per-mesh same-mesh ordinary/WFSX equivalence proof recorded by the runner
is carried through separately and is not part of the refinement verdict.

Tensor values are consumed exactly as serialized by ``run_matched_pair`` in
meV; the runner publishes the unit in ``tensors_meta`` and the campaign
verifies it before computing statistics.  All R/site keys, including zero
blocks, are preserved verbatim.  The requested (explicit, not odd-coerced)
quadrature mesh is reported separately from the odd-coerced R-list mesh used
by the exchange machinery.

All generated cases, worker outputs and reports live under the caller's
work-root in a distinct fresh invocation child, so repeated invocations
preserve prior runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
TW_ROOT = TESTS_DIR.parent.parent

SCHEMA = "tb2j.siesta_wfsx_convergence/1"

# Approved Architecture convergence thresholds, quoted unchanged.  Never
# widened here; an unmet threshold escalates, it is not relaxed.
MAX_ABS_CHANGE_MEV = 0.01
MAX_RELATIVE_CHANGE = 0.01
RELATIVE_FLOOR_MEV = 0.1

# Explicit choices for the shared pair runner; both are recorded in the report.
GREEN_MODE = "default"
CACHE_POLICY = "warm"

# run_matched_pair serializes tensors in meV and publishes the unit in its
# tensors_meta block; the campaign verifies that unit before computing any
# statistic, and no conversion is ever applied on trust.
TENSOR_UNIT_EXPECTED = "meV"

REFINEMENT_FACTORS = (1, 2, 3)

KEY_COMPONENT_TOL = 1e-6


class ConvergenceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# small honest helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Actual git identity of a checkout, or an explicit unavailability note."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return {
            "root": str(root),
            "git_commit": None,
            "git_dirty": None,
            "note": "git metadata unavailable for this path",
        }
    try:
        dirty = bool(git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"root": str(root), "git_commit": commit, "git_dirty": dirty}


def _package_provenance() -> dict[str, Any]:
    import importlib.metadata
    import importlib.util

    provenance: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
    }
    for distribution in ("TB2J", "HamiltonIO", "sisl", "numpy", "scipy"):
        try:
            provenance["packages"][distribution] = {
                "version": importlib.metadata.version(distribution)
            }
        except Exception:
            provenance["packages"][distribution] = {
                "version": None,
                "note": "distribution metadata unavailable; imported from source tree",
            }
    provenance["tb2j_checkout"] = _git_snapshot(TW_ROOT)
    spec = importlib.util.find_spec("HamiltonIO")
    if spec is not None and spec.origin:
        hamiltonio_root = Path(spec.origin).resolve().parent.parent
        provenance["hamiltonio_checkout"] = _git_snapshot(hamiltonio_root)
        provenance["hamiltonio_checkout"]["module_file"] = spec.origin
    else:
        provenance["hamiltonio_checkout"] = {
            "note": "HamiltonIO not importable from the parent environment"
        }
    return provenance


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_jsonable(payload), indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _write_failure(
    json_path: Path,
    md_path: Path,
    code: str,
    detail: str,
    stage: str,
    partial: dict[str, Any],
    write_json: bool = True,
) -> None:
    """Record actual failed evidence; never a fabricated verdict."""

    payload = {
        "schema": SCHEMA,
        "kind": "convergence-failure",
        "generated_at_utc": _utc_now(),
        "stage": stage,
        "code": code,
        "detail": detail,
        "partial_evidence": partial,
    }
    if write_json:
        try:
            _write_json(json_path, payload)
        except OSError:
            pass
    try:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            "# SIESTA WFSX mesh convergence - FAILED\n\n"
            f"Stage: `{stage}`\n\nCode: `{code}`\n\n"
            f"Detail: {detail}\n\n"
            "No convergence verdict exists for this invocation; see the JSON "
            "failure record for partial evidence.\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _parse_sha_manifest(case_dir: Path) -> dict[str, str]:
    manifest = case_dir / "SHA256SUMS"
    if not manifest.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and len(fields[0]) == 64:
            entries[fields[1].lstrip("*").strip()] = fields[0].lower()
    return entries


# ---------------------------------------------------------------------------
# tensor alignment and refinement statistics
# ---------------------------------------------------------------------------


def _canonical_key(record: dict[str, Any]) -> tuple[tuple, bool]:
    """Canonical (R, i, j) alignment key from one tensor record.

    R components are lattice translations and are integral up to float
    serialization noise; non-integral components round-trip at 1e-9 and are
    flagged so the report never silently regrids keys.
    """

    r = [float(value) for value in record["R"]]
    components: list[int] = []
    for value in r:
        nearest = round(value)
        if abs(value - nearest) > KEY_COMPONENT_TOL:
            return tuple(round(item, 9) for item in r), False
        components.append(int(nearest))
    return (tuple(components), int(record["i"]), int(record["j"])), True


def _tensor_map(records: list[dict[str, Any]]) -> tuple[dict[tuple, np.ndarray], bool]:
    mapping: dict[tuple, np.ndarray] = {}
    integral = True
    for record in records:
        key, key_integral = _canonical_key(record)
        integral = integral and key_integral
        if key in mapping:
            raise ConvergenceError(
                "DUPLICATE_TENSOR_KEY",
                f"tensor records contain duplicate key {key}",
            )
        value = np.asarray(record["value"], dtype=float)
        if value.shape != (3, 3):
            raise ConvergenceError(
                "TENSOR_SHAPE",
                f"tensor record for {key} has shape {value.shape}, expected (3, 3)",
            )
        mapping[key] = value
    return mapping, integral


def _key_list(keys: list[tuple]) -> list[list[Any]]:
    return [list(key) for key in keys]


def _refinement_step(
    coarse_records: list[dict[str, Any]],
    fine_records: list[dict[str, Any]],
    coarse_mesh: list[int],
    fine_mesh: list[int],
) -> dict[str, Any]:
    """Statistics for one successive refinement step (coarse -> fine).

    The comparison requires the same full-tensor R/site domain on both sides.
    A domain shift refuses the comparison, rather than dropping unaligned
    interactions. Values enter and are judged in serialized meV.
    """

    coarse_map, coarse_integral = _tensor_map(coarse_records)
    fine_map, fine_integral = _tensor_map(fine_records)
    coarse_keys = set(coarse_map)
    fine_keys = set(fine_map)
    aligned = sorted(coarse_keys & fine_keys)
    only_coarse = sorted(coarse_keys - fine_keys)
    only_fine = sorted(fine_keys - coarse_keys)
    if only_coarse or only_fine:
        raise ConvergenceError(
            "TENSOR_DOMAIN_CHANGED",
            f"full tensor domain changed between {coarse_mesh} and {fine_mesh}: "
            f"{len(only_coarse)} coarse-only and {len(only_fine)} fine-only keys; "
            "no intersection-only convergence verdict is permitted",
        )

    step: dict[str, Any] = {
        "coarse_mesh": coarse_mesh,
        "fine_mesh": fine_mesh,
        "coarse_site_pair_count": len(coarse_keys),
        "fine_site_pair_count": len(fine_keys),
        "aligned_site_pair_count": len(aligned),
        "aligned_element_count": 9 * len(aligned),
        "only_in_coarse": {"count": len(only_coarse), "keys": _key_list(only_coarse)},
        "only_in_fine": {"count": len(only_fine), "keys": _key_list(only_fine)},
        "integer_key_components": {
            "coarse": bool(coarse_integral),
            "fine": bool(fine_integral),
        },
        "nonfinite": {"encountered": False, "keys": []},
    }

    if not aligned:
        raise ConvergenceError(
            "EMPTY_ALIGNMENT",
            f"no aligned R/site keys between mesh {coarse_mesh} and {fine_mesh}; "
            f"coarse has {len(coarse_keys)} keys, fine has {len(fine_keys)} keys",
        )

    numeric_keys: list[tuple] = []
    for key in aligned:
        if np.all(np.isfinite(coarse_map[key])) and np.all(np.isfinite(fine_map[key])):
            numeric_keys.append(key)
        else:
            step["nonfinite"]["encountered"] = True
            step["nonfinite"]["keys"].append(list(key))

    if not numeric_keys:
        step["max_abs_change_mev"] = None
        step["abs_active_count"] = 0
        step["relative"] = {
            "reference_floor_mev": RELATIVE_FLOOR_MEV,
            "active_count": 0,
            "max_relative_change": None,
            "vacuous": True,
            "note": "no finite aligned elements; criterion not evaluable",
        }
        step["pass"] = False
        step["failure_reasons"] = [
            "no finite aligned tensor elements; no verdict can be computed"
        ]
        return step

    # Values arrive in meV (unit verified against tensors_meta); thresholds
    # are evaluated directly in serialized meV with no conversion.
    differences_mev = np.array(
        [fine_map[key] - coarse_map[key] for key in numeric_keys]
    )
    references_mev = np.array([coarse_map[key] for key in numeric_keys])

    max_abs = float(np.max(np.abs(differences_mev)))
    rel_mask = np.abs(references_mev) > RELATIVE_FLOOR_MEV
    rel_active = int(rel_mask.sum())
    if rel_active:
        max_rel = float(
            np.max(np.abs(differences_mev[rel_mask] / references_mev[rel_mask]))
        )
        vacuous = False
        vacuous_note = None
    else:
        max_rel = None
        vacuous = True
        vacuous_note = (
            "no aligned element has |reference| > "
            f"{RELATIVE_FLOOR_MEV} meV; the relative criterion is explicitly "
            "vacuous (active count 0), not a numeric pass"
        )

    abs_pass = max_abs <= MAX_ABS_CHANGE_MEV
    rel_pass = True if vacuous else max_rel <= MAX_RELATIVE_CHANGE

    reasons: list[str] = []
    if not abs_pass:
        reasons.append(
            f"maximum |change| {max_abs:.6g} meV exceeds {MAX_ABS_CHANGE_MEV} meV"
        )
    if not vacuous and not rel_pass:
        reasons.append(
            f"maximum relative change {max_rel:.6g} exceeds {MAX_RELATIVE_CHANGE}"
        )
    if step["nonfinite"]["encountered"]:
        reasons.append(
            f"{len(step['nonfinite']['keys'])} aligned keys carry non-finite values"
        )

    step["max_abs_change_mev"] = max_abs
    step["abs_active_count"] = int(differences_mev.size)
    step["relative"] = {
        "reference_floor_mev": RELATIVE_FLOOR_MEV,
        "active_count": rel_active,
        "max_relative_change": max_rel,
        "vacuous": vacuous,
        "note": vacuous_note,
    }
    step["thresholds"] = {
        "max_abs_change_mev": MAX_ABS_CHANGE_MEV,
        "max_relative_change": MAX_RELATIVE_CHANGE,
    }
    step["pass"] = bool(abs_pass and rel_pass and not step["nonfinite"]["encountered"])
    step["failure_reasons"] = reasons
    return step


# ---------------------------------------------------------------------------
# mesh generation and measurement
# ---------------------------------------------------------------------------


def _fresh_invocation_dir(work_root: Path) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = work_root / f"convergence-{stamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = work_root / f"convergence-{stamp}-{suffix}"
    candidate.mkdir()
    return candidate


def _extract_rlist_mesh(result: dict[str, Any]) -> dict[str, Any]:
    """Odd-coerced R-list mesh per run, reported separately from the request."""

    extracted: dict[str, Any] = {}
    for mode in ("hs", "wfsx"):
        run = (result.get("runs") or {}).get(mode) or {}
        kpoints = run.get("kpoints") or {}
        value = kpoints.get("rlist_mesh")
        if value:
            extracted[mode] = [int(item) for item in value]
        else:
            meta = (result.get("tensors_meta") or {}).get(mode) or {}
            value = meta.get("rlist_mesh")
            extracted[mode] = [int(item) for item in value] if value else None
    return extracted


def _verify_tensor_units(record: dict[str, Any], label: str) -> None:
    """Refuse to judge thresholds against an unverified tensor unit."""

    meta = record.get("tensors_meta") or {}
    for mode in ("hs", "wfsx"):
        unit = (meta.get(mode) or {}).get("unit")
        if unit != TENSOR_UNIT_EXPECTED:
            raise ConvergenceError(
                "TENSOR_UNIT_UNSUPPORTED",
                f"mesh {label}: tensors_meta[{mode}].unit is {unit!r}, expected "
                f"{TENSOR_UNIT_EXPECTED!r}; thresholds are evaluated in meV and "
                "no unverified conversion is applied",
            )


def _slim_run_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop per-point echoes the mesh record already carries.

    The requested quadrature is stored once per mesh as
    ``full_requested_fractional``, and the fixture's ``measurement`` block is
    summarized by the surrounding mesh fields, so the committed report stays
    small. Full values remain in each pair's scratch results.json.
    """
    slim = dict(result)
    slim["provenance"] = {
        k: v for k, v in result.get("provenance", {}).items() if k != "fixture"
    }
    slim["runs"] = {}
    for mode, run in result.get("runs", {}).items():
        run = dict(run)
        kpoints = run.get("kpoints")
        if isinstance(kpoints, dict):
            kpoints = dict(kpoints)
            kpoints["requested_point_count"] = len(
                kpoints.pop("requested_fractional", [])
            )
            kpoints["weight_count"] = len(kpoints.pop("weights", []))
            run["kpoints"] = kpoints
        slim["runs"][mode] = run
    return slim


def _mesh_record(
    mesh: tuple[int, int, int],
    case_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    fixture = json.loads((case_dir / "fixture.json").read_text(encoding="utf-8"))
    measurement = fixture["measurement"]
    return {
        "requested_mesh": [int(item) for item in mesh],
        "measurement_name": measurement["name"],
        "case_dir": str(case_dir),
        "repeat_z": measurement["repeat_z"],
        "requested_kpoint_count": len(measurement["full_requested_fractional"]),
        # Explicit actual quadrature points (never odd-coerced).
        "full_requested_fractional": measurement["full_requested_fractional"],
        # Odd-coerced R-list mesh from the exchange machinery, separate axis.
        "rlist_mesh": _extract_rlist_mesh(result),
        "physical_settings": measurement["physical_settings"],
        "changed_template_fields": measurement["changed_template_fields"],
        "template_sha256": measurement["template_sha256"],
        "pseudopotential_sha256": measurement["pseudopotential_sha256"],
        "producer": fixture["producer"],
        "scf": fixture["scf"],
        "electrons": fixture["electrons"],
        "fixture_files": fixture["files"],
        "manifest_sha256": _parse_sha_manifest(case_dir),
        # Runner result with runs, same_mesh, tensors; per-point echoes above
        # are kept once at mesh level and full values stay in scratch.
        "run_result": _slim_run_result(result),
    }


def _load_runner():
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    try:
        from run_benchmarks import run_matched_pair
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        raise ConvergenceError(
            "RUNNER_UNAVAILABLE",
            "cannot import run_matched_pair from "
            f"{TESTS_DIR / 'run_benchmarks.py'}: {exc!r}",
        ) from exc
    return run_matched_pair


def _run_campaign(args: argparse.Namespace, stage: dict[str, Any]) -> dict[str, Any]:
    generate_measurement_case = _load_producer()
    run_matched_pair = _load_runner()

    siesta = Path(args.siesta).resolve()
    source_root = Path(args.source_root).resolve()
    base = int(args.base_mesh)
    if base < 1:
        raise ConvergenceError(
            "INVALID_BASE_MESH", f"--base-mesh must be a positive integer, got {base}"
        )
    meshes = [
        (base * factor, base * factor, base * factor) for factor in REFINEMENT_FACTORS
    ]
    invocation = _fresh_invocation_dir(Path(args.work_root).resolve())
    stage["invocation_dir"] = str(invocation)
    stage["completed_meshes"] = []
    print(f"[convergence] invocation root {invocation}")

    mesh_records: list[dict[str, Any]] = []
    for mesh in meshes:
        label = "x".join(str(item) for item in mesh)
        stage["current_stage"] = f"generating measurement case {label}"
        case_dir = invocation / f"mesh-{label}"
        print(f"[convergence] generating measurement case {label} ...")
        generate_measurement_case(
            siesta=siesta,
            source_root=source_root,
            output=case_dir,
            mesh=mesh,
            repeat_z=1,
        )
        stage["current_stage"] = f"measuring matched pair at {label}"
        print(f"[convergence] measuring matched pair at {label} ...")
        result = run_matched_pair(
            case_dir,
            invocation,
            green_mode=GREEN_MODE,
            cache_policy=CACHE_POLICY,
        )
        _verify_tensor_units(result, label)
        mesh_records.append(_mesh_record(mesh, case_dir, result))
        stage["completed_meshes"].append(label)
    stage["current_stage"] = "evaluating successive refinements"

    steps = [
        _refinement_step(
            mesh_records[index]["run_result"]["tensors"]["wfsx"],
            mesh_records[index + 1]["run_result"]["tensors"]["wfsx"],
            mesh_records[index]["requested_mesh"],
            mesh_records[index + 1]["requested_mesh"],
        )
        for index in range(len(mesh_records) - 1)
    ]

    converged = all(step["pass"] for step in steps)
    failed = [
        f"{'x'.join(map(str, step['coarse_mesh']))}"
        f"->{'x'.join(map(str, step['fine_mesh']))}: "
        + "; ".join(step["failure_reasons"])
        for step in steps
        if not step["pass"]
    ]
    verdict = {
        "converged": bool(converged),
        "label": "converged" if converged else "unconverged",
        "needs_architecture_amendment": not converged,
        "failed_steps": failed,
        "statement": (
            "Both successive refinements meet the approved thresholds "
            f"(max |change| <= {MAX_ABS_CHANGE_MEV} meV and max relative "
            f"change <= {MAX_RELATIVE_CHANGE} where |reference| > "
            f"{RELATIVE_FLOOR_MEV} meV)."
            if converged
            else "The approved thresholds are not met on every successive "
            "refinement. The result stays unconverged; this escalates for an "
            "Architecture amendment. Meshes, thresholds, tolerances and "
            "inputs were not changed to manufacture a pass."
        ),
    }

    siesta_sha = _sha256_file(siesta)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": "mesh-convergence",
        "generated_at_utc": _utc_now(),
        "command": {
            "argv": list(sys.argv),
            "siesta_executable": {"path": str(siesta), "sha256": siesta_sha},
            "siesta_source_root": str(source_root),
            "base_mesh": base,
            "meshes": [list(mesh) for mesh in meshes],
            "work_root": str(Path(args.work_root).resolve()),
            "invocation_dir": str(invocation),
            "output_json": str(Path(args.output_json).resolve()),
            "output_md": str(Path(args.output_md).resolve()),
        },
        "provenance": {
            "tb2j": _git_snapshot(TW_ROOT),
            "environment": _package_provenance(),
            "note": (
                "the runner additionally records per-run source hashes and "
                "revisions verbatim in each mesh record's run_result.provenance"
            ),
        },
        "method": {
            "producer": (
                "tests/siesta_wfsx_fixtures/measurement_producer.py:"
                "generate_measurement_case (bounded measurement extension of "
                "the SHA256SUMS-pinned Story 002 producer driver, reusing the "
                "existing fixture_driver helpers; unchanged O2 SZP/PBE/100K "
                "physical template, repeat_z=1; only the k-grid and the exact "
                "template fields named in changed_template_fields differ)"
            ),
            "runner": (
                "tests/siesta_wfsx_fixtures/run_benchmarks.py:run_matched_pair "
                "(shared benchmark slice; no duplicated measurement pipeline)"
            ),
            "green_mode": GREEN_MODE,
            "cache_policy": CACHE_POLICY,
            "cache_policy_note": (
                "passed explicitly to run_matched_pair; the runner owns the "
                "cold/warm mechanics (per-input POSIX_FADV_DONTNEED cold "
                "advice where available; warm = immediately repeated fresh "
                "process on the same inputs) and records them per run"
            ),
            "integration": (
                "run_matched_pair defaults: magnetic O, Rcut 4.3 A, nz 8, "
                "nproc 1, smearing 100 kB, complete_spectrum True, metadata "
                "EF shift 0; kmesh from the measurement mesh; explicit full "
                "requested fractional points with uniform weights"
            ),
            "tensor_unit": {
                "runner_serialization": "meV, verified against "
                "tensors_meta.<mode>.unit before any statistic is computed",
                "report_conversion": "none; thresholds are evaluated directly "
                "in the serialized meV values",
                "note": (
                    "an unexpected unit aborts the campaign instead of applying "
                    "an unverified conversion"
                ),
            },
            "alignment": (
                "successive refinements are compared on the exactly aligned "
                "full-tensor R/site domain (all keys, including zero blocks); "
                "unequal full R/site key sets refuse the comparison"
            ),
            "rules": [
                "converged iff BOTH successive refinements meet BOTH thresholds",
                "each step is judged on its own measured change; monotonic "
                "decrease is never assumed",
                "the relative criterion counts only elements with "
                f"|reference| > {RELATIVE_FLOOR_MEV} meV; an active count of "
                "zero is an explicit vacuous determination, never a NaN pass",
                "any non-finite tensor element fails its step",
                "the same-mesh ordinary/WFSX equivalence proof is carried "
                "separately and does not enter the refinement verdict",
            ],
            "thresholds": {
                "max_abs_change_mev": MAX_ABS_CHANGE_MEV,
                "max_relative_change": MAX_RELATIVE_CHANGE,
                "relative_reference_floor_mev": RELATIVE_FLOOR_MEV,
            },
        },
        "meshes": mesh_records,
        "refinement_steps": steps,
        "verdict": verdict,
        "same_mesh_equivalence": {
            "role": (
                "per-mesh ordinary/WFSX same-mesh equivalence proof from "
                "run_matched_pair; reported separately and not part of the "
                "refinement convergence verdict"
            ),
            "per_mesh": [
                {
                    "requested_mesh": record["requested_mesh"],
                    "same_mesh": record["run_result"].get("same_mesh"),
                }
                for record in mesh_records
            ],
        },
    }
    return report


def _load_producer():
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    try:
        from measurement_producer import generate_measurement_case
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        raise ConvergenceError(
            "PRODUCER_UNAVAILABLE",
            "cannot import generate_measurement_case from "
            f"{TESTS_DIR / 'measurement_producer.py'}: {exc!r}",
        ) from exc
    return generate_measurement_case


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _fmt(value: Any, spec: str = ".6g") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def _render_markdown(report: dict[str, Any]) -> str:
    command = report["command"]
    provenance = report["provenance"]
    tb2j = provenance["tb2j"]
    method = report["method"]
    lines: list[str] = []
    lines.append("# SIESTA WFSX mesh convergence (N/2N/3N)")
    lines.append("")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append("")
    commit = tb2j.get("git_commit")
    commit_text = (
        f"{commit} (dirty: {tb2j.get('git_dirty')})" if commit else "unavailable"
    )
    lines.append(
        f"- TB2J: `{_md_escape(commit_text)}` at `{_md_escape(tb2j.get('root'))}`"
    )
    lines.append(
        f"- SIESTA executable: `{_md_escape(command['siesta_executable']['path'])}` "
        f"(sha256 `{command['siesta_executable']['sha256'][:12]}`)"
    )
    lines.append(
        f"- SIESTA source root: `{_md_escape(command['siesta_source_root'])}`; "
        f"base mesh {command['base_mesh']}; meshes "
        + ", ".join("`" + "x".join(map(str, mesh)) + "`" for mesh in command["meshes"])
    )
    lines.append(
        f"- Invocation root: `{_md_escape(command['invocation_dir'])}` "
        "(fresh child per invocation; prior evidence preserved)"
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Producer: {_md_escape(method['producer'])}")
    lines.append(f"- Pair runner: {_md_escape(method['runner'])}")
    lines.append(
        f"- green_mode `{method['green_mode']}`; cache_policy "
        f"`{method['cache_policy']}` (explicit). "
        f"{_md_escape(method['cache_policy_note'])}"
    )
    lines.append(f"- Integration: {_md_escape(method['integration'])}")
    lines.append(f"- Alignment: {_md_escape(method['alignment'])}")
    thresholds = method["thresholds"]
    lines.append(
        f"- Thresholds (approved Architecture, unchanged): max |change| <= "
        f"{thresholds['max_abs_change_mev']} meV AND max relative change <= "
        f"{thresholds['max_relative_change']} where |reference| > "
        f"{thresholds['relative_reference_floor_mev']} meV, on BOTH successive "
        "refinements."
    )
    lines.append(
        f"- Tensor unit: {_md_escape(method['tensor_unit']['runner_serialization'])}; "
        f"{_md_escape(method['tensor_unit']['report_conversion'])}."
    )
    for rule in method["rules"]:
        lines.append(f"- {_md_escape(rule)}")
    lines.append("")

    lines.append("## Meshes")
    lines.append("")
    lines.append(
        "| requested k-mesh | requested k-points | R-list mesh (odd-coerced, hs/wfsx) "
        "| SCF iterations | Fermi (eV) | template sha | pseudo sha |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for record in report["meshes"]:
        rlist = record["rlist_mesh"]
        rlist_text = " / ".join(
            "x".join(map(str, rlist[mode])) if rlist[mode] else "n/a"
            for mode in ("hs", "wfsx")
        )
        scf = record["scf"]
        lines.append(
            "| "
            + "x".join(map(str, record["requested_mesh"]))
            + f" | {record['requested_kpoint_count']}"
            + f" | {rlist_text}"
            + f" | {scf.get('iteration_count', 'n/a')}"
            + f" | {_fmt(record['electrons'].get('fermi_energy_ev'))}"
            + f" | `{record['template_sha256'][:12]}`"
            + f" | `{record['pseudopotential_sha256'][:12]}` |"
        )
    lines.append("")
    lines.append(
        "Requested quadrature points are the explicit full requested fractional "
        "list from each measurement manifest (never odd-coerced); the odd R-list "
        "mesh is the exchange machinery's separate axis and is reported per run."
    )
    lines.append("")

    lines.append("## Successive refinement")
    lines.append("")
    lines.append(
        "| step | aligned pairs (elements) | keys only coarse/fine "
        "| max abs change (meV) | max relative change | relative active count "
        "| verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for step in report["refinement_steps"]:
        relative = step["relative"]
        if relative["vacuous"]:
            relative_text = "vacuous (0 elements above floor)"
        else:
            relative_text = _fmt(relative["max_relative_change"])
        lines.append(
            "| "
            + "x".join(map(str, step["coarse_mesh"]))
            + " -> "
            + "x".join(map(str, step["fine_mesh"]))
            + f" | {step['aligned_site_pair_count']} ({step['aligned_element_count']})"
            + f" | {step['only_in_coarse']['count']}/{step['only_in_fine']['count']}"
            + f" | {_fmt(step['max_abs_change_mev'])}"
            + f" | {relative_text}"
            + f" | {relative['active_count']}"
            + f" | {'PASS' if step['pass'] else 'FAIL'} |"
        )
    lines.append("")
    for step in report["refinement_steps"]:
        if not step["pass"]:
            lines.append(
                "Step "
                + "x".join(map(str, step["coarse_mesh"]))
                + " -> "
                + "x".join(map(str, step["fine_mesh"]))
                + " failed: "
                + "; ".join(_md_escape(reason) for reason in step["failure_reasons"])
            )
            lines.append("")
    lines.append(
        "Each step is judged on its own measured change; no monotonic "
        "improvement is assumed."
    )
    lines.append("")

    lines.append("## Same-mesh equivalence (separate proof)")
    lines.append("")
    lines.append(
        "Ordinary/WFSX equivalence at each fixed mesh, recorded by "
        "run_matched_pair; it does not enter the refinement verdict above."
    )
    lines.append("")
    lines.append("| requested k-mesh | same_mesh.pass | observations |")
    lines.append("|---|---|---|")
    for entry in report["same_mesh_equivalence"]["per_mesh"]:
        same_mesh = entry["same_mesh"]
        if same_mesh is None:
            lines.append(
                "| "
                + "x".join(map(str, entry["requested_mesh"]))
                + " | n/a (runner recorded no same_mesh section) | |"
            )
            continue
        observations = {
            key: value for key, value in same_mesh.items() if key not in ("pass",)
        }
        rendered = json.dumps(_to_jsonable(observations), allow_nan=False)
        if len(rendered) > 300:
            rendered = rendered[:297] + "..."
        lines.append(
            "| "
            + "x".join(map(str, entry["requested_mesh"]))
            + f" | {'PASS' if same_mesh.get('pass') else 'FAIL'}"
            + f" | `{_md_escape(rendered)}` |"
        )
    lines.append("")

    verdict = report["verdict"]
    lines.append("## Verdict")
    lines.append("")
    if verdict["converged"]:
        lines.append("**CONVERGED.** " + _md_escape(verdict["statement"]))
    else:
        lines.append(
            "**UNCONVERGED - needs Architecture amendment.** "
            + _md_escape(verdict["statement"])
        )
        for failure in verdict["failed_steps"]:
            lines.append(f"- {_md_escape(failure)}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "N/2N/3N mesh-convergence characterization through the existing "
            "Story 002 producer and the shared run_matched_pair driver."
        )
    )
    parser.add_argument("--siesta", type=Path, required=True, help="SIESTA executable")
    parser.add_argument(
        "--source-root", type=Path, required=True, help="SIESTA source checkout"
    )
    parser.add_argument(
        "--work-root", type=Path, required=True, help="caller-governed work root"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--base-mesh",
        type=int,
        default=5,
        help="base N for the N^3, (2N)^3, (3N)^3 refinement meshes (default 5)",
    )
    args = parser.parse_args(argv)

    stage: dict[str, Any] = {}
    try:
        report = _run_campaign(args, stage)
    except ConvergenceError as exc:
        _write_failure(
            args.output_json,
            args.output_md,
            exc.code,
            exc.message,
            stage.get("current_stage", "startup"),
            stage,
        )
        print(f"[convergence] FAILED {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - actual failed evidence, nonzero exit
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _write_failure(
            args.output_json,
            args.output_md,
            "UNEXPECTED_FAILURE",
            detail,
            stage.get("current_stage", "startup"),
            stage,
        )
        print(f"[convergence] FAILED: {detail}", file=sys.stderr)
        return 1

    try:
        _write_json(args.output_json, report)
    except Exception as exc:  # noqa: BLE001 - actual failed evidence, nonzero exit
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _write_failure(
            args.output_json,
            args.output_md,
            "REPORT_JSON_FAILED",
            detail,
            "writing JSON report",
            stage,
        )
        print(f"[convergence] FAILED writing JSON report: {detail}", file=sys.stderr)
        return 1
    try:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_render_markdown(report), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - actual failed evidence, nonzero exit
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        # The JSON report is already on disk and stays authoritative; only the
        # Markdown side records the failure.
        _write_failure(
            args.output_json,
            args.output_md,
            "REPORT_MD_FAILED",
            detail,
            "writing Markdown report",
            stage,
            write_json=False,
        )
        print(
            f"[convergence] FAILED writing Markdown report: {detail}", file=sys.stderr
        )
        return 1

    verdict = report["verdict"]
    print(
        f"[convergence] verdict: {verdict['label']}"
        + (
            " (needs architecture amendment)"
            if verdict["needs_architecture_amendment"]
            else ""
        )
    )
    for step in report["refinement_steps"]:
        print(
            f"[convergence] step "
            f"{'x'.join(map(str, step['coarse_mesh']))} -> "
            f"{'x'.join(map(str, step['fine_mesh']))}: "
            f"max |change| {_fmt(step['max_abs_change_mev'])} meV, "
            f"pass={step['pass']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
