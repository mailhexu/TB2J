#!/usr/bin/env python3
"""Story009 opt-in benchmark: fresh-process matched ordinary complete-HS vs public WFSX.

For each ``--case-dir`` (an existing Story002 fixture directory or a parent-generated
measurement case produced by the Story002 producer workflow), this command runs the
ordinary complete-HS exchange and the public WFSX exchange
(``gen_exchange_siesta(input_mode="wfsx")``) on the same physical inputs, quadrature
and contour, each in its own fresh isolated worker process (one process per input
mode, single process/thread), under an explicit cold/warm page-cache policy, and
reports the ADR-007 measurement surface honestly: measured wall times, measured
eigensolver calls/time, process-lifetime peak RSS with method and units, spectral
disk usage, input hashes and provenance.

Same-mesh equivalence is reported per pair: full exchange tensors over the
actual SpinIO operator R/site domain (including zero blocks), populations
(charges/spin), with fixed tolerances (tensor abs 1e-6 eV — serialized as
1e-3 meV after the documented x1000 conversion — / rel 1e-4; populations
abs 1e-4 / rel 1e-5). Failures are never widened and always exit nonzero
with the actual evidence.

This is a plain scientific command, not a benchmark framework; it is not collected
by pytest. Runtime data live under the caller-provided work root; the WFSX spectral
factor cache lives in an owned unique system-temporary parent, never under the vault.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Same-mesh tensor acceptance is abs 1e-6 eV / rel 1e-4. Serialized meV values
# convert SpinIO's stored eV by x1000, so the equivalent absolute bound is 1e-3 meV.
TENSOR_ATOL = 1e-3
TENSOR_RTOL = 1e-4
POPULATIONS_ATOL = 1e-4
POPULATIONS_RTOL = 1e-5
RELATIVE_DENOMINATOR_FLOOR = 1e-12

# Fixed explicit integration fields (contract defaults, same exchange_params as the
# existing WFSX exchange tests). smearing is 100 * ase.units.kB; the numeric value is
# resolved inside the worker against the actual ase in the measured environment.
DEFAULT_INTEGRATION = {
    "magnetic_elements": ["O"],
    "Rcut": 4.3,
    "nz": 8,
    "nproc": 1,
    "efermi_ev": 0.0,
    "complete_spectrum": True,
}
INTEGRATION_OVERRIDE_KEYS = {
    "magnetic_elements",
    "Rcut",
    "nz",
    "nproc",
    "smearing_ev",
    "efermi_ev",
    "complete_spectrum",
    "kmesh",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
INPUT_FILE_ROLES = ("fdf", "pseudo", "wfsx", "hsx", "kp", "eig", "orb_indx")
WORST_EXAMPLES_KEPT = 5


class BenchmarkError(RuntimeError):
    """A measurement failed; the message carries actual evidence locations."""


# ---------------------------------------------------------------------------
# small shared helpers (parent process; JSON-native only, no scientific imports)
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def _case_manifest(case_dir: Path) -> dict:
    manifest_path = case_dir / "fixture.json"
    if not manifest_path.is_file():
        raise BenchmarkError(f"case {case_dir}: missing fixture.json manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _case_full_points(manifest: dict):
    """Explicit full requested fractional points, exactly as source_model derives them."""
    fold_map = manifest["kpoints"]["fold_map"]
    if fold_map:
        return [row["selected_fractional"] for row in fold_map]
    return [row["fractional"] for row in manifest["kpoints"]["points"]]


def _case_name(case_dir: Path, manifest: dict) -> str:
    measurement = manifest.get("measurement") or {}
    return str(measurement.get("name") or manifest.get("fixture_id") or case_dir.name)


def _case_kmesh(case_dir: Path, manifest: dict, integration: dict) -> list:
    """Requested k mesh from the measurement block, the fixture mesh, or override."""
    if integration.get("kmesh") is not None:
        kmesh = [int(value) for value in integration["kmesh"]]
    else:
        measurement = manifest.get("measurement") or {}
        if "mesh" in measurement:
            kmesh = [int(value) for value in measurement["mesh"]]
        else:
            kmesh = [int(value) for value in manifest["mesh"]["dimensions"]]
    if len(kmesh) != 3 or min(kmesh) < 1:
        raise BenchmarkError(f"case {case_dir}: invalid requested kmesh {kmesh}")
    return kmesh


def _case_inputs(case_dir: Path, manifest: dict) -> dict:
    """Resolve the case input files (root-relative in the manifest)."""
    inputs = {}
    files = manifest["files"]
    for role in INPUT_FILE_ROLES:
        if files.get(role):
            path = (case_dir / files[role]).resolve()
            if not path.is_file():
                raise BenchmarkError(f"case {case_dir}: input {role} missing: {path}")
            inputs[role] = path
    return inputs


def _hash_inputs(inputs: dict) -> list:
    hashed = []
    for role, path in sorted(inputs.items()):
        hashed.append(
            {
                "role": role,
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return hashed


def _fadvise_dontneed(paths) -> list:
    """Best-effort per-input POSIX_FADV_DONTNEED; advisory only, never global."""
    observations = []
    advised = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    for path in paths:
        entry = {"path": str(path), "advised": False, "method": None, "error": None}
        if not advised:
            entry["error"] = "os.posix_fadvise unavailable on this platform"
            observations.append(entry)
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                entry["advised"] = True
                entry["method"] = "posix_fadvise(POSIX_FADV_DONTNEED) whole file"
            finally:
                os.close(fd)
        except OSError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        observations.append(entry)
    return observations


def _git_head(worktree: Path):
    try:
        out = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"git invocation failed: {type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return None, (out.stderr or "git rev-parse failed").strip()
    return out.stdout.strip(), None


def _tw_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hamiltonio_location():
    """Locate the HamiltonIO actually imported in this environment, without importing it."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("HamiltonIO")
    except (ImportError, ValueError) as exc:
        return None, f"importlib lookup failed: {exc}"
    if spec is None or spec.origin is None:
        return None, "HamiltonIO not importable in parent environment"
    origin = Path(spec.origin)
    return str(origin.parent.parent), None


def _source_content_hashes(root: str | Path | None, relative_paths: tuple) -> dict:
    if root is None:
        return {}
    base = Path(root)
    hashes = {}
    for relative in relative_paths:
        path = base / relative
        if path.is_file():
            hashes[str(relative)] = _sha256_file(path)
    return hashes


TB2J_SOURCE_FILES = (
    "TB2J/green.py",
    "TB2J/interfaces/siesta_interface.py",
    "TB2J/exchange.py",
    "TB2J/exchangeCL2.py",
    "TB2J/exchange_params.py",
    "TB2J/io_exchange/io_exchange.py",
    "tests/test_siesta_wfsx_green.py",
)
HAMILTONIO_SOURCE_FILES = (
    "HamiltonIO/siesta/wfsx.py",
    "HamiltonIO/siesta/sisl_wrapper.py",
    "HamiltonIO/lcao_hamiltonian.py",
)


def _campaign_provenance() -> dict:
    tw = _tw_root()
    tb2j_head, tb2j_error = _git_head(tw)
    hw_path, hw_locate_error = _hamiltonio_location()
    hamiltonio_head = None
    hamiltonio_error = hw_locate_error
    if hw_path:
        hamiltonio_head, hamiltonio_error = _git_head(Path(hw_path))
    return {
        "tb2j": {
            "worktree": str(tw),
            "git_head": tb2j_head,
            "git_head_error": tb2j_error,
        },
        "hamiltonio": {
            "package_path": hw_path,
            "git_head": hamiltonio_head,
            "git_head_error": hamiltonio_error,
        },
        "source_content_sha256": {
            "tb2j": _source_content_hashes(tw, TB2J_SOURCE_FILES),
            "hamiltonio": _source_content_hashes(hw_path, HAMILTONIO_SOURCE_FILES),
        },
        "run_benchmarks": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "kernel": platform.release(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "load_average_1_5_15": (
                list(os.getloadavg()) if hasattr(os, "getloadavg") else None
            ),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# worker-side measurement (runs in a fresh isolated process per input mode)
# ---------------------------------------------------------------------------


def _worker_environment_prologue() -> None:
    """Guarantee the CFR auxiliary pole eig is preloaded before electronic monitoring.

    ``TB2J.mycfr`` binds ``scipy.linalg.eig`` at import time. Importing it before the
    ``_measure_wfsx_eigensolvers`` monitor is installed keeps the auxiliary CFR pole
    eigensolve (a Lanczos-matrix diagonalization with no electronic content) outside
    the measured electronic solver figures in both input modes.
    """
    import TB2J.mycfr  # noqa: F401  (import-order contract, see docstring)


def _cfr_pole_preloaded() -> bool:
    import sys as _sys

    return "TB2J.mycfr" in _sys.modules


def _worker_package_versions() -> dict:
    import HamiltonIO
    import numpy
    import scipy

    import TB2J

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "tb2j_file": str(Path(TB2J.__file__).resolve()),
        "hamiltonio_file": str(Path(HamiltonIO.__file__).resolve()),
    }
    try:
        import sisl

        versions["sisl"] = sisl.__version__
    except Exception as exc:  # pragma: no cover - recorded honestly if absent
        versions["sisl"] = f"unavailable: {type(exc).__name__}: {exc}"
    for module, key in ((TB2J, "tb2j_version"), (HamiltonIO, "hamiltonio_version")):
        version = getattr(module, "__version__", None)
        if version:
            versions[key] = version
    return versions


def _worker_process_block(
    mode: str, green_mode: str, cache_policy: str, nproc: int
) -> dict:
    return {
        "isolated_fresh_process": True,
        "mode": mode,
        "nproc": int(nproc),
        "pid": os.getpid(),
        "interpreter": sys.executable,
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV},
        "green_mode_env": os.environ.get("TB2J_SIESTA_GREEN_MODE"),
        "requested_green_mode": green_mode,
        "requested_cache_policy": cache_policy,
        "green_mode_read_from_env": "TB2J_SIESTA_GREEN_MODE (read at green runtime, not import time)",
    }


def _worker_cold_evict(paths, cache_policy: str) -> list:
    if cache_policy != "cold":
        return [
            {
                "path": str(path),
                "advised": False,
                "method": None,
                "error": "not requested (warm policy)",
            }
            for path in paths
        ]
    return _fadvise_dontneed(paths)


def _peak_rss_block() -> dict:
    import resource
    import sys as _sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "value": int(rss * (1 if _sys.platform == "darwin" else 1024)),
        "unit": "bytes",
        "method": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss; process-lifetime high-water; "
            "macOS bytes, Linux KiB converted; child processes excluded"
        ),
    }


def _finite(value) -> bool:
    if value is None:
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def _records_finite(records) -> str | None:
    for record in records:
        for row in record["value"]:
            for value in row:
                if not _finite(value):
                    return (
                        f"non-finite exchange tensor element at R={record['R']} "
                        f"i={record['i']} j={record['j']}"
                    )
    return None


def _extract_result_tensors(output_dir: Path, kmesh_odd: list) -> dict:
    """Read the complete written exchange via SpinIO on the actual operator R list.

    ``SpinIO.get_full_Jtensor_for_Rlist(order="ij33")`` defines the full tensor on the
    R list rebuilt from the stored keyed exchange (io_exchange ``_build_Rlist``). The
    records enumerate that ACTUAL operator domain — every R in it, every ordered site
    pair including zero 3x3 blocks — and never manufacture extra R outside the Rcut.
    """
    from TB2J.io_exchange import SpinIO

    result = SpinIO.load_pickle(path=str(output_dir))
    matrix = result.get_full_Jtensor_for_Rlist(order="ij33")
    # SpinIO stores exchange tensors in eV (its plotters multiply by 1e3 for meV);
    # serialize the actual x1000 conversion so meV labels are physical values.
    records = []
    for ir, rvec in enumerate(result.Rlist):
        rlist = [int(rvec[0]), int(rvec[1]), int(rvec[2])]
        for i in range(result.nspin):
            for j in range(result.nspin):
                records.append(
                    {
                        "R": rlist,
                        "i": int(i),
                        "j": int(j),
                        "value": [
                            [float(x) * 1e3 for x in row] for row in matrix[ir][i][j]
                        ],
                    }
                )
    problem = _records_finite(records)
    if problem is not None:
        raise BenchmarkError(
            f"{output_dir}: {problem}; refusing to publish non-finite tensors"
        )
    populations = {
        "charges": [float(x) for x in result.charges],
        "spinat": [[float(x) for x in row] for row in result.spinat],
    }
    for name, values in (
        ("charges", populations["charges"]),
        ("spinat", [x for row in populations["spinat"] for x in row]),
    ):
        if not all(_finite(x) for x in values):
            raise BenchmarkError(
                f"{output_dir}: non-finite {name}; refusing to publish populations"
            )
    meta = {
        "unit": "meV",
        "unit_basis": (
            "SpinIO exchange tensors are stored in eV; these records carry the actual "
            "x1000 conversion to meV (the same conversion TB2J plotters apply), so "
            "meV thresholds apply directly to the serialized values"
        ),
        "stored_unit": "eV",
        "conversion": "value * 1e3",
        "population_units": {"charges": "e", "spinat": "mu_B"},
        "nR": len(result.Rlist),
        "nspin": int(result.nspin),
        "rlist_mesh": kmesh_odd,
        "rlist_mesh_note": (
            "requested odd-coerced k-mesh numerics; the serialized tensor records use "
            "the ACTUAL SpinIO operator R list reported in nR, not this mesh cube"
        ),
        "record_count": len(records),
        "source": (
            "SpinIO.load_pickle(TB2J.pickle).get_full_Jtensor_for_Rlist(order='ij33') "
            "on the actual operator R list; every R and ordered site pair including "
            "zero 3x3 blocks; no R manufactured outside the Rcut domain"
        ),
    }
    return {"tensors": records, "tensors_meta": meta, "populations": populations}


def _resident_inventory_ordinary(greens, exchange) -> list:
    """Measured resident matrix inventory from the live ordinary TBGreen objects."""
    inventory = []
    for index, green in enumerate(greens):
        for name in ("H0", "evals", "evecs", "H", "S"):
            array = getattr(green, name, None)
            if array is None:
                continue
            inventory.append(
                {
                    "green": index,
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "bytes": int(array.nbytes),
                }
            )
    delta = getattr(exchange, "Delta", None)
    if delta is not None:
        inventory.append(
            {
                "green": None,
                "name": "Delta",
                "shape": list(delta.shape),
                "dtype": str(delta.dtype),
                "bytes": int(delta.nbytes),
            }
        )
    return inventory


def _run_hs_worker(payload: dict) -> dict:
    """Ordinary complete-HS measurement: import measured separately from preparation."""
    from ase.units import kB
    from HamiltonIO.siesta.sisl_wrapper import SislParser

    from TB2J.exchange import ExchangeNCL
    from TB2J.exchangeCL2 import ExchangeCL2
    from TB2J.interfaces.siesta_interface import _measure_wfsx_eigensolvers

    case_dir = Path(payload["case_dir"])
    manifest = json.loads(Path(payload["manifest_path"]).read_text(encoding="utf-8"))
    integration = payload["integration"]
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _case_inputs(case_dir, manifest)
    fdf = inputs["fdf"]

    points = _case_full_points(manifest)
    kweights = [1.0 / len(points)] * len(points)
    kmesh = [int(value) for value in payload["kmesh"]]
    kmesh_odd = [value // 2 * 2 + 1 for value in kmesh]
    smearing_ev = payload["smearing_ev"]
    nspin = int(manifest["spin"]["nspin"])
    channels = 1 if nspin in (4, 8) else 2

    worker_input_eviction = _worker_cold_evict(
        [inputs[role] for role in sorted(inputs) if role != "wfsx"],
        payload["cache_policy"],
    )

    started = time.perf_counter()
    parser = SislParser(str(fdf))
    model = parser.get_model()
    import_seconds = time.perf_counter() - started
    first = model[0] if isinstance(model, tuple) else model
    basis = dict(zip(first.orbs, range(first.nbasis)))
    cls = ExchangeCL2 if isinstance(model, tuple) else ExchangeNCL

    exchange_params = dict(
        kmesh=kmesh,
        kpts=[list(point) for point in points],
        kweights=kweights,
        use_cache=False,
        nproc=int(integration["nproc"]),
        smearing=smearing_ev,
        Rcut=float(integration["Rcut"]),
        nz=int(integration["nz"]),
        efermi=float(integration["efermi_ev"]),
        magnetic_elements=list(integration["magnetic_elements"]),
        complete_spectrum=bool(integration["complete_spectrum"]),
        output_path=str(output_dir),
    )

    with _measure_wfsx_eigensolvers() as solvers:
        prepare_started = time.perf_counter()
        exchange = cls(
            tbmodels=model,
            atoms=first.atoms,
            basis=basis,
            **exchange_params,
        )
        preparation_seconds = time.perf_counter() - prepare_started
        greens = (
            [exchange.G] if hasattr(exchange, "G") else [exchange.Gup, exchange.Gdn]
        )
        resident_matrices = _resident_inventory_ordinary(greens, exchange)
        contour = exchange.contour
        contour_record = {
            "method": type(contour).__name__,
            "nz": int(integration["nz"]),
            "smearing_ev": float(smearing_ev),
            "temperature_K": float(smearing_ev / kB),
            "path_points": int(len(contour.path)),
        }
        integration_started = time.perf_counter()
        exchange.calculate_all()
        integration_seconds = time.perf_counter() - integration_started
        output_started = time.perf_counter()
        exchange.write_output(path=str(output_dir))
        output_seconds = time.perf_counter() - output_started
        peak_rss = _peak_rss_block()
        solver_snapshot = {
            "calls": int(solvers["calls"]),
            "seconds": float(solvers["seconds"]),
            "method": solvers["method"],
        }
    exchange.finalize()
    total_seconds = time.perf_counter() - started

    tensors = _extract_result_tensors(output_dir, kmesh_odd)
    electrons = manifest["electrons"]
    run = {
        "schema_version": SCHEMA_VERSION,
        "input_mode": "hs",
        "timing_seconds": {
            "import": import_seconds,
            "preparation": preparation_seconds,
            "integration": integration_seconds,
            "output": output_seconds,
            "total": total_seconds,
        },
        "electronic_eigensolver": solver_snapshot,
        "cfr_pole_eig_preload": {
            "executed": _cfr_pole_preloaded(),
            "method": (
                "TB2J.mycfr imported before the electronic monitor was installed; its "
                "module-level scipy.linalg.eig binding predates instrumentation, so the "
                "auxiliary CFR pole eigensolve is excluded from solver figures"
            ),
        },
        "inputs": _hash_inputs(
            {role: path for role, path in inputs.items() if role != "wfsx"}
        ),
        "fixture_manifest": {
            "path": str(case_dir / "fixture.json"),
            "sha256": _sha256_file(case_dir / "fixture.json"),
            "bytes": (case_dir / "fixture.json").stat().st_size,
        },
        "metadata": {
            "source_provenance": manifest.get("producer_run_id"),
            "fermi_source": electrons.get("fermi_energy_source"),
            "electron_source": electrons.get("total_electrons_source"),
            "electron_count": electrons.get("total_electrons"),
            "occupation_convention": electrons.get("occupation_convention"),
            "electronic_temperature_k": electrons.get("electronic_temperature_k"),
            "weight_semantics": manifest["kpoints"].get("weight_semantics"),
            "fixture_id": manifest.get("fixture_id"),
        },
        "spin_mode": nspin,
        "source_precision": {
            "hamiltonian_dtype": str(first.HR.dtype),
            "overlap_dtype": str(first.SR.dtype),
            "source": "SislParser SiestaHamiltonian HR/SR arrays (measured dtypes)",
        },
        "matrix_sizes": {
            "basis": int(first.nbasis),
            "orbitals": int(first.norb),
            "channels": channels,
            "atoms": int(len(first.atoms)),
        },
        "states": {
            "supplied": int(len(points) * channels * first.nbasis),
            "requested": int(len(points) * channels * first.nbasis),
            "per_record": int(first.nbasis),
        },
        "kpoints": {
            "supplied": len(points),
            "requested": len(points),
            "representatives_per_channel": [len(points)] * channels,
            "requested_fractional": [list(point) for point in points],
            "weights": kweights,
            "rlist_mesh": kmesh_odd,
            "requested_mesh": kmesh,
            "requested_mesh_note": (
                "explicit full requested points + uniform weights; kmesh only fixes the "
                "R list via the ordinary odd coercion"
            ),
        },
        "energy_reference": {
            "shifted_fermi_ev": float(integration["efermi_ev"]),
            "source_absolute_fermi_ev": electrons.get("fermi_energy_ev"),
        },
        "integration_width_ev": float(smearing_ev),
        "green_mode": os.environ.get("TB2J_SIESTA_GREEN_MODE") or "default",
        "cache": {
            "mode": "in-memory-eigenbasis (use_cache=False)",
            "spectral_disk_bytes": 0,
            "policy_note": (
                "ordinary default keeps the eigenbasis in memory and stages nothing on "
                "disk; WFSX mode stages spectral factors on disk by mandatory policy, "
                "so the two storage behaviours are not identical by design"
            ),
        },
        "contour": contour_record,
        "spectrum_complete": {
            "complete_spectrum": bool(integration["complete_spectrum"]),
            "note": (
                "full eigenbasis at every requested k; complete-resolvent mode retains "
                "every pole instead of an energy-window subset"
            ),
        },
        "resident_matrices": resident_matrices,
        "peak_rss": peak_rss,
        "worker_process": _worker_process_block(
            "hs",
            payload["green_mode"],
            payload["cache_policy"],
            int(integration["nproc"]),
        ),
        "packages": _worker_package_versions(),
        "worker_input_eviction": worker_input_eviction,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": "hs",
        "run": run,
        "output_dir": str(output_dir),
    }
    result.update(tensors)
    return result


def _run_wfsx_worker(payload: dict) -> dict:
    """Public-entry WFSX measurement; the production summary is reused verbatim."""
    import numpy as np
    from ase.units import kB

    from TB2J.interfaces import gen_exchange_siesta

    tests_dir = payload["tests_dir"]
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_siesta_wfsx_green import source_model

    case_dir = Path(payload["case_dir"])
    integration = payload["integration"]
    output_dir = Path(payload["output_dir"])
    worker_input = Path(payload["worker_input_dir"])
    worker_input.mkdir(parents=True, exist_ok=True)

    kmesh = [int(value) for value in payload["kmesh"]]
    kmesh_odd = [value // 2 * 2 + 1 for value in kmesh]
    smearing_ev = payload["smearing_ev"]

    factor_parent = Path(payload["factor_cache_parent"])
    factor_parent.mkdir(parents=True, exist_ok=True)
    cache_path = factor_parent / "cache"

    # Existing helper: copies only states.WFSX + metadata.json and returns the
    # explicit full requested points with uniform weights.
    model, metadata, _manifest, points = source_model(
        worker_input,
        name=case_dir.name,
        spin=0,
        reduce_scalar=False,
        fixture_root=case_dir.parent,
    )
    points_list = [list(point) for point in np.asarray(points).tolist()]
    kweights = [1.0 / len(points_list)] * len(points_list)

    worker_input_eviction = _worker_cold_evict(
        [worker_input / "states.WFSX", worker_input / "metadata.json"],
        payload["cache_policy"],
    )

    exchange_params = dict(
        kmesh=kmesh,
        kpts=points_list,
        kweights=kweights,
        nproc=int(integration["nproc"]),
        smearing=smearing_ev,
        Rcut=float(integration["Rcut"]),
        nz=int(integration["nz"]),
        efermi=float(integration["efermi_ev"]),
        magnetic_elements=list(integration["magnetic_elements"]),
        cache_path=str(cache_path),
        output_path=str(output_dir),
    )

    summary = gen_exchange_siesta(
        None,
        input_mode="wfsx",
        wfsx_file=worker_input / "states.WFSX",
        wfsx_metadata=metadata,
        **exchange_params,
    )

    tensors = _extract_result_tensors(output_dir, kmesh_odd)
    resident_note = (
        "the public production summary exposes actual matrix_sizes, states, symmetry "
        "events and spectral_disk_bytes but not a per-array resident-memory inventory; "
        "actual process peak RSS is reported separately, with no universal all-memory claim"
    )
    run = dict(summary)
    run["worker_process"] = _worker_process_block(
        "wfsx",
        payload["green_mode"],
        payload["cache_policy"],
        int(integration["nproc"]),
    )
    run["cfr_pole_eig_preload"] = {
        "executed": _cfr_pole_preloaded(),
        "method": (
            "TB2J.mycfr imported before the public entry installed the electronic "
            "monitor; the auxiliary CFR pole eigensolve stays outside solver figures"
        ),
    }
    run["contour"] = {
        "method": "CFR",
        "nz": int(integration["nz"]),
        "smearing_ev": float(smearing_ev),
        "temperature_K": float(smearing_ev / kB),
        "path_points": None,
        "path_points_note": (
            "production finalizes the contour internally; contour parameters are "
            "identical to the ordinary run by construction (CFR, same nz, same smearing)"
        ),
    }
    run["spectrum_complete"] = {
        "complete": True,
        "note": (
            "complete selected spectrum; producer completeness checks and the reader's "
            "IncompleteSpectrum refusal cover the full supplied band set"
        ),
    }
    run["fixture_manifest"] = {
        "path": str(case_dir / "fixture.json"),
        "sha256": _sha256_file(case_dir / "fixture.json"),
        "bytes": (case_dir / "fixture.json").stat().st_size,
    }
    run["factor_cache"] = {
        "parent": str(factor_parent),
        "location_policy": (
            "owned unique temporary parent in the system temp directory, outside the "
            "vault; TBGreen stages spectral factors inside it and removes its owned "
            "children on finalize; parent cleanup is owned by the caller"
        ),
        "owned_children_removed": bool(summary["cache"]["owned_children_removed"]),
    }
    run["resident_inventory"] = {
        "method": resident_note,
        "matrix_sizes": summary["matrix_sizes"],
        "states": summary["states"],
        "spectral_disk_bytes": int(summary["cache"]["spectral_disk_bytes"]),
        "symmetry_partner_expansion_events": [
            int(item["partner_expansion_events"]) for item in summary["symmetry"]
        ],
    }
    run["worker_input_eviction"] = worker_input_eviction
    run["packages"] = _worker_package_versions()

    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": "wfsx",
        "run": run,
        "output_dir": str(output_dir),
    }
    result.update(tensors)
    return result


def _worker_main(payload_path: str) -> int:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    results_path = Path(payload["results_path"])
    stage = "startup"
    try:
        _worker_environment_prologue()
        stage = "measurement"
        if payload["mode"] == "hs":
            result = _run_hs_worker(payload)
        elif payload["mode"] == "wfsx":
            result = _run_wfsx_worker(payload)
        else:
            raise BenchmarkError(f"unknown worker mode {payload['mode']!r}")
        stage = "write-results"
        _json_write(results_path, result)
        return 0
    except BaseException as exc:  # evidence first, nonzero exit always
        import traceback

        error = {
            "schema_version": SCHEMA_VERSION,
            "mode": payload.get("mode"),
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        try:
            _json_write(
                results_path.parent / "error.json",
                error,
            )
        except OSError:
            pass
        print(json.dumps(error, indent=2), file=sys.stderr)
        return 3


# ---------------------------------------------------------------------------
# parent side: fresh worker spawning, cold policy, same-mesh comparison
# ---------------------------------------------------------------------------


def _effective_integration(integration: dict | None, case_dir: Path) -> dict:
    effective = dict(DEFAULT_INTEGRATION)
    if integration is not None:
        if not isinstance(integration, dict):
            raise BenchmarkError(
                f"case {case_dir}: integration override must be a dict"
            )
        unknown = set(integration) - INTEGRATION_OVERRIDE_KEYS
        if unknown:
            raise BenchmarkError(
                f"case {case_dir}: unsupported integration overrides {sorted(unknown)}; "
                f"allowed keys are {sorted(INTEGRATION_OVERRIDE_KEYS)}"
            )
        effective.update(integration)
    if effective.get("nproc") != 1:
        raise BenchmarkError(
            f"case {case_dir}: matched benchmark requires nproc=1 for both modes"
        )
    if effective.get("complete_spectrum") is not True:
        raise BenchmarkError(
            f"case {case_dir}: complete_spectrum=True is required for complete-HS comparison"
        )
    return effective


def _spawn_worker(payload: dict, pair_dir: Path, mode: str) -> tuple:
    payload_path = pair_dir / f"worker-{mode}.payload.json"
    _json_write(payload_path, payload)
    env = dict(os.environ)
    env.update(THREAD_ENV)
    env["TB2J_SIESTA_GREEN_MODE"] = payload["green_mode"]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_tw_root()), existing]))
    stdout_path = pair_dir / f"worker-{mode}.stdout.log"
    stderr_path = pair_dir / f"worker-{mode}.stderr.log"
    started = time.perf_counter()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--internal-worker",
                str(payload_path),
            ],
            stdout=out,
            stderr=err,
            env=env,
            cwd=str(pair_dir),
        )
    wall = time.perf_counter() - started
    return completed.returncode, wall, stdout_path, stderr_path


def _load_worker_result(pair_dir: Path, mode: str) -> dict:
    return json.loads((pair_dir / mode / "results.json").read_text(encoding="utf-8"))


def _compare_numeric_series(reference, comparison, atol: float, rtol: float) -> dict:
    compared = 0
    failing = 0
    max_abs = None
    max_abs_key = None
    max_rel = None
    max_rel_key = None
    nonfinite = 0
    examples = []
    for index, (ref, act) in enumerate(zip(reference, comparison)):
        if not (_finite(ref) and _finite(act)):
            nonfinite += 1
            failing += 1
            continue
        diff = abs(act - ref)
        compared += 1
        if max_abs is None or diff > max_abs:
            max_abs, max_abs_key = diff, index
        rel = diff / max(abs(ref), RELATIVE_DENOMINATOR_FLOOR)
        if max_rel is None or rel > max_rel:
            max_rel, max_rel_key = rel, index
        if diff > atol + rtol * abs(ref):
            failing += 1
            if len(examples) < WORST_EXAMPLES_KEPT:
                examples.append(
                    {
                        "index": index,
                        "reference": ref,
                        "actual": act,
                        "abs": diff,
                        "rel": rel,
                    }
                )
    length_mismatch = len(reference) != len(comparison)
    return {
        "abs_tolerance": atol,
        "rel_tolerance": rtol,
        "reference_entries": len(reference),
        "comparison_entries": len(comparison),
        "length_mismatch": length_mismatch,
        "compared_elements": compared,
        "nonfinite_elements": nonfinite,
        "failing_elements": failing,
        "max_abs_deviation": max_abs,
        "max_abs_index": max_abs_key,
        "max_relative_deviation": max_rel,
        "max_relative_index": max_rel_key,
        "relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR,
        "worst_failing_examples": examples,
        "pass": bool(reference)
        and bool(comparison)
        and (not length_mismatch)
        and failing == 0,
    }


def _tensor_key(record) -> tuple:
    return (tuple(record["R"]), record["i"], record["j"])


def _compare_tensors(reference: list, comparison: list) -> dict:
    ref_key_counts = {}
    act_key_counts = {}
    for record in reference:
        key = _tensor_key(record)
        ref_key_counts[key] = ref_key_counts.get(key, 0) + 1
    for record in comparison:
        key = _tensor_key(record)
        act_key_counts[key] = act_key_counts.get(key, 0) + 1
    ref_by_key = {_tensor_key(record): record for record in reference}
    act_by_key = {_tensor_key(record): record for record in comparison}
    duplicate_reference = sorted(
        str(key) for key, count in ref_key_counts.items() if count > 1
    )
    duplicate_comparison = sorted(
        str(key) for key, count in act_key_counts.items() if count > 1
    )
    missing = sorted(str(key) for key in ref_by_key if key not in act_by_key)
    extra = sorted(str(key) for key in act_by_key if key not in ref_by_key)
    compared = 0
    failing = 0
    max_abs = None
    max_abs_key = None
    max_rel = None
    max_rel_key = None
    nonfinite = 0
    examples = []
    for key, ref_record in ref_by_key.items():
        act_record = act_by_key.get(key)
        if act_record is None:
            continue
        for ii in range(3):
            for jj in range(3):
                ref = ref_record["value"][ii][jj]
                act = act_record["value"][ii][jj]
                if not (_finite(ref) and _finite(act)):
                    nonfinite += 1
                    failing += 1
                    continue
                diff = abs(act - ref)
                compared += 1
                label = f"{key}[{ii}][{jj}]"
                if max_abs is None or diff > max_abs:
                    max_abs, max_abs_key = diff, label
                rel = diff / max(abs(ref), RELATIVE_DENOMINATOR_FLOOR)
                if max_rel is None or rel > max_rel:
                    max_rel, max_rel_key = rel, label
                if diff > TENSOR_ATOL + TENSOR_RTOL * abs(ref):
                    failing += 1
                    if len(examples) < WORST_EXAMPLES_KEPT:
                        examples.append(
                            {
                                "key": label,
                                "reference_meV": ref,
                                "actual_meV": act,
                                "abs": diff,
                                "rel": rel,
                            }
                        )
    return {
        "abs_tolerance": TENSOR_ATOL,
        "rel_tolerance": TENSOR_RTOL,
        "unit": "meV",
        "reference_records": len(reference),
        "comparison_records": len(comparison),
        "missing_keys": missing[:20],
        "missing_key_count": len(missing),
        "extra_keys": extra[:20],
        "extra_key_count": len(extra),
        "duplicate_reference_keys": duplicate_reference[:20],
        "duplicate_reference_key_count": len(duplicate_reference),
        "duplicate_comparison_keys": duplicate_comparison[:20],
        "duplicate_comparison_key_count": len(duplicate_comparison),
        "compared_elements": compared,
        "nonfinite_elements": nonfinite,
        "failing_elements": failing,
        "max_abs_deviation": max_abs,
        "max_abs_key": max_abs_key,
        "max_relative_deviation": max_rel,
        "max_relative_key": max_rel_key,
        "relative_denominator_floor_meV": RELATIVE_DENOMINATOR_FLOOR,
        "worst_failing_examples": examples,
        "pass": (
            bool(reference)
            and bool(comparison)
            and not missing
            and not extra
            and not duplicate_reference
            and not duplicate_comparison
            and failing == 0
        ),
    }


def _compare_pair(hs: dict, wfsx: dict, green_mode: str, cache_policy: str) -> dict:
    tensor = _compare_tensors(hs["tensors"], wfsx["tensors"])
    hs_pop = hs["populations"]
    wfsx_pop = wfsx["populations"]
    charges = _compare_numeric_series(
        hs_pop["charges"], wfsx_pop["charges"], POPULATIONS_ATOL, POPULATIONS_RTOL
    )
    spinat_ref = [x for row in hs_pop["spinat"] for x in row]
    spinat_act = [x for row in wfsx_pop["spinat"] for x in row]
    spinat = _compare_numeric_series(
        spinat_ref, spinat_act, POPULATIONS_ATOL, POPULATIONS_RTOL
    )
    failures = []
    if not tensor["pass"]:
        failures.append(
            "tensor: {} failing element(s), {} missing keys, {} extra keys; worst abs "
            "{} meV at {}".format(
                tensor["failing_elements"],
                tensor["missing_key_count"],
                tensor["extra_key_count"],
                tensor["max_abs_deviation"],
                tensor["max_abs_key"],
            )
        )
    for name, observation in (("charges", charges), ("spinat", spinat)):
        if not observation["pass"]:
            failures.append(
                "{}: {} failing entr(ies), max abs {} at index {}".format(
                    name,
                    observation["failing_elements"],
                    observation["max_abs_deviation"],
                    observation["max_relative_index"],
                )
            )
    return {
        "pass": tensor["pass"] and charges["pass"] and spinat["pass"],
        "green_mode": green_mode,
        "cache_policy": cache_policy,
        "tensor": tensor,
        "populations": {
            "charges": dict(charges, unit="e"),
            "spinat": dict(spinat, unit="mu_B"),
        },
        "failures": failures,
    }


def run_matched_pair(
    case_dir: Path,
    work_root: Path,
    *,
    green_mode: str = "default",
    cache_policy: str = "warm",
    integration: dict | None = None,
) -> dict:
    """Run one matched ordinary-HS vs public-WFSX pair on one case.

    A fresh isolated worker process per input mode; a newly named child directory
    inside ``work_root`` per call (exact reruns never collide). Returns the JSON-native
    measurement dict: provenance, ``runs.hs``/``runs.wfsx`` ADR-007-style summaries,
    ``same_mesh`` equivalence observations and complete ``tensors`` for both modes.
    """
    if green_mode not in ("default", "contravariant"):
        raise BenchmarkError(f"unknown green_mode {green_mode!r}")
    if cache_policy not in ("cold", "warm"):
        raise BenchmarkError(f"unknown cache_policy {cache_policy!r}")
    case_dir = Path(case_dir).resolve()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    manifest = _case_manifest(case_dir)
    name = _case_name(case_dir, manifest)
    effective = _effective_integration(integration, case_dir)
    kmesh = _case_kmesh(case_dir, manifest, effective)
    inputs = _case_inputs(case_dir, manifest)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    pair_dir = (
        work_root / f"{name}.{green_mode}.{cache_policy}.{stamp}.{uuid.uuid4().hex[:8]}"
    )
    pair_dir.mkdir(parents=True)

    base_payload = {
        "case_dir": str(case_dir),
        "manifest_path": str(case_dir / "fixture.json"),
        "tests_dir": str(Path(__file__).resolve().parent.parent),
        "green_mode": green_mode,
        "cache_policy": cache_policy,
        "integration": effective,
        "kmesh": kmesh,
        "smearing_ev": float(effective.get("smearing_ev", 100.0 * _ase_kb())),
        "tw_root": str(_tw_root()),
    }

    cold_evictions = {}
    spawn = {}
    results = {}
    factor_parent_path = None
    try:
        for mode in ("hs", "wfsx"):
            child = pair_dir / mode
            child.mkdir()
            if cache_policy == "cold":
                if mode == "wfsx":
                    cold_paths = [inputs["wfsx"], case_dir / "fixture.json"]
                else:
                    cold_paths = [
                        inputs[role] for role in sorted(inputs) if role != "wfsx"
                    ]
                cold_evictions[mode] = {
                    "inputs": _fadvise_dontneed(cold_paths),
                    "guarantee": (
                        "advisory per-input POSIX_FADV_DONTNEED only; NOT a guaranteed "
                        "global page-cache cold state"
                    ),
                }
            payload = dict(base_payload)
            payload["mode"] = mode
            payload["output_dir"] = str(child / "TB2J_results")
            payload["results_path"] = str(child / "results.json")
            payload["worker_input_dir"] = str(child / "inputs")
            if mode == "wfsx":
                factor_parent_path = Path(
                    tempfile.mkdtemp(prefix="wfsx-bench-factors-")
                )
                payload["factor_cache_parent"] = str(factor_parent_path)
            returncode, wall, stdout_log, stderr_log = _spawn_worker(
                payload, pair_dir, mode
            )
            spawn[mode] = {
                "launch_wall_seconds": wall,
                "returncode": returncode,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "note": (
                    "launch wall includes interpreter startup and imports; it is "
                    "reported separately from the worker-measured phases"
                ),
            }
            if returncode != 0:
                error_path = child / "error.json"
                detail = ""
                if error_path.is_file():
                    detail = error_path.read_text(encoding="utf-8")[:4000]
                raise BenchmarkError(
                    f"case {name}: {mode} worker failed with exit {returncode}; "
                    f"stderr log {stderr_log}; evidence: {detail or 'see logs'}"
                )
            results[mode] = _load_worker_result(pair_dir, mode)
            results[mode]["run"]["worker_process"]["launch_wall_seconds"] = wall
    finally:
        if factor_parent_path is not None and factor_parent_path.exists():
            # Owned unique temporary parent in the system temp dir; only our own
            # spectral-factor children can ever live inside it.
            shutil.rmtree(factor_parent_path, ignore_errors=True)

    if factor_parent_path is not None and "wfsx" in results:
        results["wfsx"]["run"]["factor_cache"][
            "parent_directory_removed"
        ] = not factor_parent_path.exists()
    same_mesh = _compare_pair(results["hs"], results["wfsx"], green_mode, cache_policy)
    return {
        "schema_version": SCHEMA_VERSION,
        "case": str(case_dir),
        "name": name,
        "green_mode": green_mode,
        "cache_policy": cache_policy,
        "invocation_child": str(pair_dir),
        "provenance": {
            **_campaign_provenance(),
            "fixture": {
                "fixture_id": manifest.get("fixture_id"),
                "producer_run_id": manifest.get("producer_run_id"),
                "measurement_block": manifest.get("measurement"),
                "manifest_path": str(case_dir / "fixture.json"),
                "manifest_sha256": _sha256_file(case_dir / "fixture.json"),
                "source_case_inputs": _hash_inputs(inputs),
            },
        },
        "integration": _serialized_integration(effective, base_payload["smearing_ev"]),
        "requested_kmesh": kmesh,
        "runs": {
            "hs": results["hs"]["run"],
            "wfsx": results["wfsx"]["run"],
        },
        "same_mesh": same_mesh,
        "tensors": {
            "hs": results["hs"]["tensors"],
            "wfsx": results["wfsx"]["tensors"],
        },
        "tensors_meta": {
            "hs": results["hs"]["tensors_meta"],
            "wfsx": results["wfsx"]["tensors_meta"],
        },
        "populations": {
            "hs": results["hs"]["populations"],
            "wfsx": results["wfsx"]["populations"],
        },
        "cold_policy": {
            "requested": cache_policy,
            "input_eviction": cold_evictions,
            "guarantee": (
                "cold = per-input POSIX_FADV_DONTNEED advice where supported; NOT a "
                "guaranteed global page-cache cold state; warm = immediate repeat of a "
                "fresh process on the same inputs"
            ),
        },
        "worker_spawns": spawn,
        "notes": [
            "each input mode ran in its own fresh process with one thread per BLAS pool",
            "both modes were compared against the same-mode complete-HS measurement "
            "domain: identical explicit full requested points, uniform weights, "
            "identical CFR contour parameters and identical odd-coerced R list",
        ],
    }


def _ase_kb() -> float:
    """Resolve the actual ase.units.kB; a missing dependency is a real error."""
    from ase.units import kB

    return float(kB)


def _serialized_integration(effective: dict, smearing_ev: float) -> dict:
    serialized = dict(effective)
    serialized["smearing_ev"] = float(smearing_ev)
    serialized["smearing_basis"] = (
        "explicit integration.smearing_ev override"
        if "smearing_ev" in effective
        else "100 * ase.units.kB"
    )
    return serialized


# ---------------------------------------------------------------------------
# CLI campaign
# ---------------------------------------------------------------------------


def _ratio_label(hs_total: float, wfsx_total: float) -> str:
    if not (_finite(hs_total) and _finite(wfsx_total)) or hs_total <= 0:
        return "n/a"
    return f"{wfsx_total / hs_total:.3f}"


def _markdown_report(campaign: dict) -> str:
    provenance = campaign["provenance"]
    lines = []
    lines.append("# WFSX benchmark run (Story009 opt-in command)")
    lines.append("")
    lines.append(
        "Measured on {node} ({cpus} CPUs), {kernel}, Python {python}, generated {utc}.".format(
            **{
                "node": provenance["host"]["node"],
                "cpus": provenance["host"]["cpus"],
                "kernel": provenance["host"]["kernel"],
                "python": provenance["host"]["python"],
                "utc": provenance["generated_utc"],
            }
        )
    )
    tb2j_head = (
        provenance["tb2j"]["git_head"]
        or f"unavailable ({provenance['tb2j']['git_head_error']})"
    )
    hw_head = (
        provenance["hamiltonio"]["git_head"]
        or f"unavailable ({provenance['hamiltonio']['git_head_error']})"
    )
    lines.append("")
    lines.append(
        f"- TB2J worktree HEAD: `{tb2j_head}` ({provenance['tb2j']['worktree']})"
    )
    lines.append(
        f"- HamiltonIO HEAD: `{hw_head}` ({provenance['hamiltonio']['package_path']})"
    )
    lines.append(
        f"- Driver: run_benchmarks.py sha256 `{provenance['run_benchmarks']['sha256']}`"
    )
    lines.append("")
    lines.append(
        "Method: one fresh isolated process per input mode (ordinary complete-HS vs "
        "public WFSX entry), single process and one thread per BLAS pool; identical "
        "explicit full requested k points with uniform weights, identical CFR contour "
        "parameters (nz and smearing recorded per run) and identical odd-coerced R list "
        "in both modes. Cold means per-input POSIX_FADV_DONTNEED advice only (never a "
        "guaranteed global cold page cache); warm means the immediately repeated fresh "
        "process on the same inputs. Peak RSS is resource.getrusage(RUSAGE_SELF) "
        "process-lifetime high-water in bytes, children excluded; launch wall time is "
        "reported separately from the worker-measured phases. WFSX spectral factors "
        "are staged in an owned unique system-temporary parent and removed afterwards; "
        "the ordinary mode keeps the eigenbasis in memory (use_cache=False). All "
        "numbers below are measured on the named runs only; no speed target, no "
        "projected universal benefit."
    )
    lines.append("")
    for case in campaign["cases"]:
        lines.append(f"## {case['name']} (`{case['case']}`)")
        lines.append("")
        lines.append(
            "| policy | mode | cache mode | import s | preparation s | integration s | output s | "
            "total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode, run in case["runs"].items():
            timing = run["timing_seconds"]
            rss_mib = run["peak_rss"]["value"] / (1024 * 1024)
            disk_kib = run["cache"]["spectral_disk_bytes"] / 1024
            lines.append(
                "| {policy} | {mode} | {cache} | {imp:.3f} | {prep:.3f} | {integ:.3f} | {out:.3f} | "
                "{total:.3f} | {calls} | {esec:.3f} | {rss:.1f} | {disk:.1f} |".format(
                    policy=case["cache_policy"],
                    mode=mode,
                    cache=run["cache"]["mode"],
                    imp=timing["import"],
                    prep=timing["preparation"],
                    integ=timing["integration"],
                    out=timing["output"],
                    total=timing["total"],
                    calls=run["electronic_eigensolver"]["calls"],
                    esec=run["electronic_eigensolver"]["seconds"],
                    rss=rss_mib,
                    disk=disk_kib,
                )
            )
        same = case["same_mesh"]
        tensor = same["tensor"]
        lines.append("")
        verdict = "PASS" if same["pass"] else "FAIL"
        lines.append(
            "Same-mesh equivalence ({}): **{}** — {} records/run, max |ΔJ| {} meV "
            "(tolerance abs {} / rel {}), max rel {}, populations charges max |Δ| {} e / "
            "spin max |Δ| {} mu_B (tolerance abs {} / rel {}).".format(
                case["green_mode"],
                verdict,
                tensor["reference_records"],
                tensor["max_abs_deviation"],
                tensor["abs_tolerance"],
                tensor["rel_tolerance"],
                tensor["max_relative_deviation"],
                same["populations"]["charges"]["max_abs_deviation"],
                same["populations"]["spinat"]["max_abs_deviation"],
                POPULATIONS_ATOL,
                POPULATIONS_RTOL,
            )
        )
        if not same["pass"]:
            for failure in same["failures"]:
                lines.append(f"- FAILURE: {failure}")
        hs_total = case["runs"]["hs"]["timing_seconds"]["total"]
        wfsx_total = case["runs"]["wfsx"]["timing_seconds"]["total"]
        lines.append(
            "Measured total-wall ratio wfsx/hs on this run: {} (cold policy: {}).".format(
                _ratio_label(hs_total, wfsx_total), case["cache_policy"]
            )
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--internal-worker":
        return _worker_main(argv[1])
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        action="append",
        required=True,
        help="fixture or generated measurement case directory (repeatable)",
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--green-mode",
        choices=("both", "default", "contravariant"),
        default="both",
        help="Green modes to measure (default: both; each matched to same-mode H/S)",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("cold", "warm"),
        nargs="+",
        default=["cold", "warm"],
        help="cache policies to run, in order (default: cold then warm)",
    )
    parser.add_argument(
        "--integration",
        type=str,
        default=None,
        help="JSON object overriding fixed explicit exchange fields "
        f"(allowed keys: {sorted(INTEGRATION_OVERRIDE_KEYS)})",
    )
    args = parser.parse_args(argv)

    integration = None
    if args.integration:
        try:
            integration = json.loads(args.integration)
        except ValueError as exc:
            print(f"--integration is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(integration, dict):
            print("--integration must be a JSON object", file=sys.stderr)
            return 2
    green_modes = (
        ("default", "contravariant")
        if args.green_mode == "both"
        else (args.green_mode,)
    )

    campaign = {
        "schema_version": SCHEMA_VERSION,
        "command": {
            "case_dirs": [str(Path(case).resolve()) for case in args.case_dir],
            "work_root": str(args.work_root.resolve()),
            "green_modes": list(green_modes),
            "cache_policies": list(args.cache_policy),
        },
        "provenance": _campaign_provenance(),
        "cases": [],
    }
    try:
        for case_dir in args.case_dir:
            for green_mode in green_modes:
                for policy in args.cache_policy:
                    campaign["cases"].append(
                        run_matched_pair(
                            case_dir,
                            args.work_root,
                            green_mode=green_mode,
                            cache_policy=policy,
                            integration=integration,
                        )
                    )
    except BenchmarkError as exc:
        campaign["status"] = "failed"
        campaign["error"] = str(exc)
        try:
            _json_write(args.output_json, campaign)
        except (OSError, ValueError):
            pass
        print(f"BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        import traceback

        campaign["status"] = "failed"
        campaign["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _json_write(args.output_json, campaign)
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(
                "# WFSX benchmark failed\n\n"
                f"```text\n{campaign['error']['traceback']}```\n",
                encoding="utf-8",
            )
        except (OSError, ValueError):
            pass
        print(f"BENCHMARK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    failures = [case for case in campaign["cases"] if not case["same_mesh"]["pass"]]
    campaign["status"] = "ok" if not failures else "same_mesh_failed"
    # The committed JSON stays a small durable report: full-precision exchange
    # tensors live only in each pair's scratch results.json, and the explicit
    # requested quadrature is summarized instead of echoed per run.
    compact_cases = []
    for case in campaign["cases"]:
        slim = {k: v for k, v in case.items() if k != "tensors"}
        slim["tensor_record_counts"] = {m: len(t) for m, t in case["tensors"].items()}
        for run in slim["runs"].values():
            kpoints = run.get("kpoints")
            if isinstance(kpoints, dict):
                kpoints["requested_point_count"] = len(
                    kpoints.pop("requested_fractional", [])
                )
                kpoints["weight_count"] = len(kpoints.pop("weights", []))
        compact_cases.append(slim)
    campaign["cases"] = compact_cases
    _json_write(args.output_json, campaign)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown_report(campaign), encoding="utf-8")
    if failures:
        print(
            "SAME-MESH EQUIVALENCE FAILED for: "
            + ", ".join(
                f"{case['name']} [{case['cache_policy']}]" for case in failures
            ),
            file=sys.stderr,
        )
        return 1
    print(f"Benchmark OK: {len(campaign['cases'])} matched pair run(s).")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
