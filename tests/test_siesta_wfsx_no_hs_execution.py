"""Traced proof that public WFSX entries run without H/S or electronic eigensolves.

``trace_entry_cases(work_root, native_trace=False)`` executes the real public
entries end to end, eight rows: entry ``console`` (the installed ``siesta2J.py``
console script when available, otherwise the portable
``python -c "...; sys.exit(run_siesta2J())"`` wrapper) and entry ``python``
(a fresh interpreter calling ``gen_exchange_siesta`` directly), each crossed
with the real magnetic collinear representative ``collinear_nspin2_reduced``
and the full spinor ``noncollinear_nspin4_full`` fixtures under both Green
modes (``default``/``contravariant``).

Every child runs with a generated ``sitecustomize.py`` that installs the guard
from :func:`install_trace` before any CLI import.  ``TB2J.mycfr`` is preloaded
first and its auxiliary pole eigensolve is bound separately: it is not an
electronic eigensolve and stays allowed while every real electronic entry is
forbidden (the eight NumPy/SciPy entries, the HamiltonIO
``lcao_hamiltonian``/SIESTA-wrapper ``eigh`` aliases, the inherited model
``solve``/``solve_all``/``HSE_k``/``HS_and_eigen`` methods, and the ordinary
``SislParser``).  The guard also observes ordinary ``sisl.get_sile`` use, the
native reader close path via ``_fortran_is_open``, every ``np.memmap`` handle
lifecycle, and records file opens through a Python audit hook.  The child
writes a JSON guard report under ``work_root``.

``native_trace=True`` requires the installed console script and ``/bin/strace``
and wraps every child in ``strace -f -s 200 -e trace=open,openat,openat2``;
the raw log stays under ``work_root`` and the parsed row rejects any open of
an ordinary H/S input.  Permanent tests always run with ``native_trace=False``
so the suite has no ptrace dependency; the traced layer is exercised opt-in
via ``TB2J_SIESTA_WFSX_NATIVE_TRACE=1`` or by the scientific report driver.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("sisl")

FIXTURES = Path(__file__).resolve().parent / "siesta_wfsx_fixtures"
INSTALLED_CONSOLE = Path(sys.executable).with_name("siesta2J.py")
STRACE = Path("/bin/strace")
TRACE_METHOD = f"{STRACE} -f -s 200 -e trace=open,openat,openat2"

_COLLILINEAR = "collinear_nspin2_reduced"
_SPINOR = "noncollinear_nspin4_full"
_ENTRIES = ("console", "python")
_MODES = ("default", "contravariant")
_CHILD_TIMEOUT = 900

_CONSOLE_BOOT = (
    "import sys; from TB2J.scripts.siesta2J import run_siesta2J; "
    "sys.exit(run_siesta2J())"
)
_PYTHON_BOOT = """import sys
from ase.units import kB
from TB2J.interfaces import gen_exchange_siesta

gen_exchange_siesta(
    input_mode="wfsx",
    wfsx_file=sys.argv[1],
    wfsx_metadata=sys.argv[2],
    kmesh=[1, 1, 5],
    magnetic_elements=["O"],
    Rcut=4.3,
    nz=8,
    nproc=1,
    smearing=100 * kB,
    cache_path=sys.argv[4],
    output_path=sys.argv[3],
)
"""

_SITECUSTOMIZE = '''"""Generated test-only child guard bootstrap."""
import json
import os
import traceback

if os.environ.get("TB2J_WFSX_GUARD_REPORT"):
    try:
        from test_siesta_wfsx_no_hs_execution import install_trace

        install_trace()
    except BaseException:
        with open(os.environ["TB2J_WFSX_GUARD_REPORT"], "w") as stream:
            json.dump({"install_error": traceback.format_exc()}, stream)
        raise
'''

_OPEN_CALL = re.compile(r"^(?:\d+\s+)?(openat2|openat|open)\(")
_QUOTED = re.compile(r'"([^"]*)"')
_ORDINARY_INPUT_SUFFIXES = frozenset(
    {".fdf", ".hsx", ".nc", ".tbt", ".kp", ".eig", ".orb_indx", ".psf", ".psml", ".ion"}
)


def _module_roots():
    """Resolve the code roots the parent process actually tests."""
    import HamiltonIO.siesta.wfsx as wfsx_module

    import TB2J.interfaces as interfaces_module

    return (
        str(Path(interfaces_module.__file__).resolve().parents[2]),
        str(Path(wfsx_module.__file__).resolve().parents[2]),
    )


def install_trace():
    """Install the child-only no-H/S guard; called from the generated bootstrap.

    Runs before any CLI import.  ``TB2J.mycfr`` is preloaded first and its
    auxiliary pole eigensolve is bound separately, so the CFR pole solve stays
    allowed and is never reported as electronic.
    """
    import atexit
    import weakref
    from collections import Counter

    report_path = Path(os.environ["TB2J_WFSX_GUARD_REPORT"])
    states_path = os.path.realpath(os.environ["TB2J_WFSX_GUARD_STATES"])
    metadata_path = os.path.realpath(os.environ["TB2J_WFSX_GUARD_METADATA"])
    fixtures_root = os.path.realpath(os.environ["TB2J_WFSX_GUARD_FIXTURES"])

    report = {
        "electronic_guard_hits": Counter(),
        "cfr_auxiliary_pole_eig_calls": 0,
        "sisl_get_sile": {"calls": 0, "paths": [], "unexpected": []},
        "reader_close": {"events": 0, "fortran_open_at_close": 0},
        "memmaps": {"created": 0, "closed": 0, "open_at_exit": []},
        "audit_opens": {"categories": Counter(), "forbidden": []},
    }
    memmaps = []

    def forbid(label):
        def forbidden(*args, **kwargs):
            report["electronic_guard_hits"][label] += 1
            raise AssertionError(f"no-HS execution reached electronic entry {label}")

        forbidden.__name__ = label.rsplit(".", 1)[-1]
        return forbidden

    import numpy as np
    import scipy.linalg

    import TB2J.mycfr  # noqa: F401  preload before guards: binds the original scipy eig

    original_cfr_eig = TB2J.mycfr.eig

    def cfr_auxiliary_pole_eig(*args, **kwargs):
        report["cfr_auxiliary_pole_eig_calls"] += 1
        return original_cfr_eig(*args, **kwargs)

    TB2J.mycfr.eig = cfr_auxiliary_pole_eig

    original_memmap = np.memmap

    def tracked_memmap(*args, **kwargs):
        array = original_memmap(*args, **kwargs)
        handle = getattr(array, "_mmap", None)
        if handle is not None:
            memmaps.append((weakref.ref(array), handle))
        return array

    np.memmap = tracked_memmap

    for module, prefix in (
        (np.linalg, "numpy.linalg"),
        (scipy.linalg, "scipy.linalg"),
    ):
        for name in ("eig", "eigh", "eigvals", "eigvalsh"):
            setattr(module, name, forbid(f"{prefix}.{name}"))

    from HamiltonIO import lcao_hamiltonian
    from HamiltonIO.siesta import sisl_wrapper
    from HamiltonIO.siesta import wfsx as wfsx_module

    setattr(lcao_hamiltonian, "eigh", forbid("hamiltonio.lcao_hamiltonian.eigh"))
    setattr(sisl_wrapper, "eigh", forbid("hamiltonio.siesta.sisl_wrapper.eigh"))
    for cls, prefix in (
        (lcao_hamiltonian.LCAOHamiltonian, "lcao_hamiltonian"),
        (sisl_wrapper.SiestaHamiltonian, "siesta_wrapper"),
    ):
        for name in ("solve", "solve_all", "HSE_k", "HS_and_eigen"):
            setattr(cls, name, forbid(f"{prefix}.{name}"))
    setattr(
        sisl_wrapper.SislParser, "__init__", forbid("ordinary.sisl_parser.__init__")
    )

    original_close_reader = wfsx_module._close_reader

    def observed_close_reader(reader):
        report["reader_close"]["events"] += 1
        if reader._fortran_is_open():
            report["reader_close"]["fortran_open_at_close"] += 1
        return original_close_reader(reader)

    wfsx_module._close_reader = observed_close_reader

    import sisl

    original_get_sile = sisl.get_sile

    def guarded_get_sile(path, *args, **kwargs):
        observed = report["sisl_get_sile"]
        observed["calls"] += 1
        try:
            resolved = os.path.realpath(os.fspath(path))
        except TypeError:
            resolved = str(path)
        observed["paths"].append(resolved)
        if resolved != states_path:
            observed["unexpected"].append(resolved)
            raise AssertionError(
                f"ordinary sisl.get_sile opened {resolved}; only the states "
                f"WFSX {states_path} is allowed"
            )
        return original_get_sile(path, *args, **kwargs)

    sisl.get_sile = guarded_get_sile

    categories = report["audit_opens"]["categories"]
    forbidden_opens = report["audit_opens"]["forbidden"]

    def audit(event, args):
        if event != "open" or not args:
            return
        raw = args[0]
        try:
            text = os.fsdecode(raw) if isinstance(raw, (str, bytes)) else str(raw)
            resolved = os.path.realpath(text)
        except Exception:
            return
        suffix = os.path.splitext(resolved)[1].lower()
        if resolved.startswith(fixtures_root + os.sep) or (
            suffix in _ORDINARY_INPUT_SUFFIXES
        ):
            categories["forbidden-h-s-input"] += 1
            if len(forbidden_opens) < 16:
                forbidden_opens.append(resolved)
        elif resolved == states_path:
            categories["input-states"] += 1
        elif resolved == metadata_path:
            categories["input-metadata"] += 1
        else:
            categories["other"] += 1

    sys.addaudithook(audit)

    written = []

    def write_report():
        if written:
            return
        written.append(True)
        payload = {
            "child": {
                "executable": sys.executable,
                "python": ".".join(str(part) for part in sys.version_info[:3]),
            },
            "electronic_guard_hits": dict(report["electronic_guard_hits"]),
            "cfr_auxiliary_pole_eig_calls": report["cfr_auxiliary_pole_eig_calls"],
            "sisl_get_sile": report["sisl_get_sile"],
            "reader_close": report["reader_close"],
            "memmaps": {
                "created": len(memmaps),
                "closed": sum(1 for _, handle in memmaps if handle.closed),
                "open_at_exit": [
                    index
                    for index, (_, handle) in enumerate(memmaps)
                    if not handle.closed
                ],
            },
            "audit_opens": {
                "categories": dict(report["audit_opens"]["categories"]),
                "forbidden": report["audit_opens"]["forbidden"],
            },
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    atexit.register(write_report)
    previous_hook = sys.excepthook

    def excepthook(kind, value, traceback):
        write_report()
        previous_hook(kind, value, traceback)

    sys.excepthook = excepthook


def _write_sitecustomize(work_root):
    path = work_root / "sitecustomize.py"
    if not path.is_file():
        path.write_text(_SITECUSTOMIZE)
    return path


def _child_environment(work_root, inputs, metadata_path, report_path, green_mode):
    tb2j_root, hio_root = _module_roots()
    entries = [
        str(work_root),
        str(Path(__file__).resolve().parent),
        tb2j_root,
        hio_root,
    ]
    if os.environ.get("PYTHONPATH"):
        entries.extend(os.environ["PYTHONPATH"].split(os.pathsep))
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(entries),
        "TB2J_WFSX_GUARD_REPORT": str(report_path),
        "TB2J_WFSX_GUARD_STATES": str(inputs / "states.WFSX"),
        "TB2J_WFSX_GUARD_METADATA": str(metadata_path),
        "TB2J_WFSX_GUARD_FIXTURES": str(FIXTURES),
    }
    if green_mode == "contravariant":
        env["TB2J_SIESTA_GREEN_MODE"] = "contravariant"
    else:
        env.pop("TB2J_SIESTA_GREEN_MODE", None)
    return env


def _cli_arguments(model, metadata_path, output):
    return [
        "--input-mode",
        "wfsx",
        "--wfsx-file",
        os.fspath(model.eigenstate_source.wfsx_path),
        "--wfsx-metadata",
        os.fspath(metadata_path),
        "--elements",
        "O",
        "--kmesh",
        "1",
        "1",
        "5",
        "--nz",
        "8",
        "--rcut",
        "4.3",
        "--smearing",
        "100K",
        "--output_path",
        os.fspath(output),
    ]


def _console_command(arguments, log_path, native_trace):
    if native_trace:
        return [
            STRACE,
            "-f",
            "-s",
            "200",
            "-e",
            "trace=open,openat,openat2",
            "-o",
            log_path,
            INSTALLED_CONSOLE,
            *arguments,
        ], "installed-script"
    if INSTALLED_CONSOLE.is_file() and os.access(INSTALLED_CONSOLE, os.X_OK):
        return [INSTALLED_CONSOLE, *arguments], "installed-script"
    return [sys.executable, "-c", _CONSOLE_BOOT, *arguments], "sys-exit-wrapper"


def _python_command(arguments, log_path, native_trace):
    body = [sys.executable, "-c", _PYTHON_BOOT, *arguments]
    if native_trace:
        return [
            STRACE,
            "-f",
            "-s",
            "200",
            "-e",
            "trace=open,openat,openat2",
            "-o",
            log_path,
            *body,
        ], "gen_exchange_siesta"
    return body, "gen_exchange_siesta"


def _load_guard_report(row_id, report_path):
    if not report_path.is_file():
        raise RuntimeError(f"{row_id}: child guard report missing at {report_path}")
    text = report_path.read_text()
    try:
        guard = json.loads(text)
    except ValueError:
        raise RuntimeError(
            f"{row_id}: child guard installation failed:\n{text}"
        ) from None
    if not isinstance(guard, dict):
        raise RuntimeError(f"{row_id}: unexpected guard report:\n{text}")
    if "install_error" in guard:
        raise RuntimeError(f"{row_id}: child guard failed:\n{guard['install_error']}")
    return guard


def _parse_strace_log(log_path, cwd):
    counts = {"open": 0, "openat": 0, "openat2": 0}
    forbidden = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = _OPEN_CALL.match(line)
        if match is None:
            continue
        syscall = match.group(1)
        counts[syscall] += 1
        quoted = _QUOTED.search(line)
        if quoted is None:
            continue
        raw = quoted.group(1)
        path = raw if os.path.isabs(raw) else os.path.join(os.fspath(cwd), raw)
        normalized = os.path.normpath(path)
        suffix = os.path.splitext(normalized)[1].lower()
        if normalized.startswith(str(FIXTURES) + os.sep) or (
            suffix in _ORDINARY_INPUT_SUFFIXES
        ):
            if len(forbidden) < 16:
                forbidden.append({"syscall": syscall, "path": raw})
    return {
        "method": TRACE_METHOD,
        "syscalls": counts,
        "opens_total": sum(counts.values()),
        "forbidden_h_s_opens": forbidden,
        "log": log_path.name,
    }


def _cache_evidence(cache_parent, marker, summary):
    return {
        "parent": str(cache_parent),
        "listing": sorted(path.name for path in cache_parent.iterdir()),
        "marker_preserved": marker.is_file() and marker.read_text() == "keep",
        "owned_children_removed": bool(summary["cache"]["owned_children_removed"]),
    }


def _run_entry_case(work_root, source_model, entry, fixture, green_mode, native_trace):
    row_id = f"{entry}-{fixture}-{green_mode}"
    row_root = work_root / row_id
    inputs = row_root / "input"
    inputs.mkdir(parents=True)
    output = row_root / "TB2J_results"
    cache_parent = row_root / "cache"
    cache_parent.mkdir()
    marker = cache_parent / "user.txt"
    marker.write_text("keep")
    model, _, _, _ = source_model(inputs, fixture)
    states = inputs / "states.WFSX"
    metadata_path = inputs / "metadata.json"
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs.iterdir())
    }
    report_path = row_root / "guard_report.json"
    env = _child_environment(work_root, inputs, metadata_path, report_path, green_mode)
    # Both public entries must stage under the observed, caller-owned parent.
    env["TMPDIR"] = str(cache_parent)
    log_path = row_root / "native_open.log"
    if entry == "console":
        command, impl = _console_command(
            _cli_arguments(model, metadata_path, output), log_path, native_trace
        )
    else:
        command, impl = _python_command(
            [states, metadata_path, output, cache_parent], log_path, native_trace
        )
    started = time.perf_counter()
    run = subprocess.run(
        [os.fspath(part) for part in command],
        cwd=inputs,
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT,
    )
    elapsed = time.perf_counter() - started
    if run.returncode != 0:
        raise RuntimeError(
            f"{row_id}: public entry failed ({run.returncode})\n"
            f"{run.stdout}\n{run.stderr}"
        )
    summary_path = output / "wfsx_run.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"{row_id}: missing wfsx_run.json\n{run.stdout}\n{run.stderr}"
        )
    summary = json.loads(summary_path.read_text())
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs.iterdir())
    }
    row = {
        "entry": entry,
        "entry_impl": impl,
        "fixture": fixture,
        "green_mode": green_mode,
        "command": [os.fspath(part) for part in command],
        "cwd": str(inputs),
        "row_root": str(row_root),
        "returncode": run.returncode,
        "elapsed_seconds": elapsed,
        "run_summary": summary,
        "guard": _load_guard_report(row_id, report_path),
        "inputs": {
            "names": sorted(after),
            "preserved": before == after,
            "sha256": before,
        },
        "cache": _cache_evidence(cache_parent, marker, summary),
        "output": {
            "path": str(output),
            "tb2j_pickle": (output / "TB2J.pickle").is_file(),
        },
        "native": None,
    }
    if native_trace:
        row["native"] = _parse_strace_log(log_path, inputs)
    return row


def trace_entry_cases(work_root: Path, native_trace: bool = False) -> list[dict]:
    """Run the eight public-entry rows under ``work_root`` and return evidence.

    Each JSON-native row carries the exact command, fixture, Green mode, entry
    kind, the child guard counters, the published ``wfsx_run.json`` summary,
    input/cache preservation evidence and, for ``native_trace=True``, the
    parsed native open counts with the strace method and retained log name.
    """
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if native_trace:
        missing = [
            label
            for label, path in (
                ("/bin/strace", STRACE),
                (str(INSTALLED_CONSOLE), INSTALLED_CONSOLE),
            )
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "native_trace=True requires: " + ", ".join(missing) + "; "
                "install them or call with native_trace=False"
            )
    from test_siesta_wfsx_green import source_model

    _write_sitecustomize(work_root)
    return [
        _run_entry_case(
            work_root, source_model, entry, fixture, green_mode, native_trace
        )
        for fixture in (_COLLILINEAR, _SPINOR)
        for entry in _ENTRIES
        for green_mode in _MODES
    ]


def _assert_row_evidence(row):
    fixture = row["fixture"]
    assert row["returncode"] == 0
    summary = row["run_summary"]
    assert summary["electronic_eigensolver"]["calls"] == 0
    assert summary["electronic_eigensolver"]["seconds"] == 0
    assert summary["green_mode"] == row["green_mode"]
    assert summary["kpoints"]["requested"] == 5
    assert summary["spin_mode"] == (2 if fixture == _COLLILINEAR else 4)
    assert summary["matrix_sizes"]["basis"] == (18 if fixture == _COLLILINEAR else 36)
    guard = row["guard"]
    assert guard["electronic_guard_hits"] == {}
    assert guard["cfr_auxiliary_pole_eig_calls"] >= 1
    get_sile = guard["sisl_get_sile"]
    assert get_sile["calls"] >= 1
    assert get_sile["unexpected"] == []
    assert len(set(get_sile["paths"])) == 1
    assert get_sile["paths"][0].endswith("states.WFSX")
    memmaps = guard["memmaps"]
    assert memmaps["created"] >= 1
    assert memmaps["closed"] == memmaps["created"]
    assert memmaps["open_at_exit"] == []
    assert guard["reader_close"]["events"] >= 1
    assert guard["audit_opens"]["forbidden"] == []
    assert row["inputs"]["preserved"] is True
    assert set(row["inputs"]["names"]) == {"metadata.json", "states.WFSX"}
    assert summary["inputs"][0]["sha256"] == row["inputs"]["sha256"]["states.WFSX"]
    cache = row["cache"]
    assert cache["listing"] == ["user.txt"]
    assert cache["marker_preserved"] is True
    assert cache["owned_children_removed"] is True
    assert row["output"]["tb2j_pickle"] is True


def test_trace_matrix_public_entries_run_without_hs_or_eigensolve(tmp_path):
    rows = trace_entry_cases(tmp_path)
    assert {(row["entry"], row["fixture"], row["green_mode"]) for row in rows} == {
        (entry, fixture, mode)
        for fixture in (_COLLILINEAR, _SPINOR)
        for entry in _ENTRIES
        for mode in _MODES
    }
    for row in rows:
        _assert_row_evidence(row)
    assert {row["entry_impl"] for row in rows if row["entry"] == "console"} <= {
        "installed-script",
        "sys-exit-wrapper",
    }


@pytest.mark.skipif(
    not os.environ.get("TB2J_SIESTA_WFSX_NATIVE_TRACE"),
    reason="opt-in strace layer; set TB2J_SIESTA_WFSX_NATIVE_TRACE=1",
)
def test_native_traced_entries_reject_ordinary_input_opens(tmp_path):
    rows = trace_entry_cases(tmp_path, native_trace=True)
    assert len(rows) == 8
    for row in rows:
        _assert_row_evidence(row)
        native = row["native"]
        assert native is not None
        assert native["method"] == TRACE_METHOD
        assert native["opens_total"] == sum(native["syscalls"].values()) > 0
        assert set(native["syscalls"]) == {"open", "openat", "openat2"}
        assert native["forbidden_h_s_opens"] == []
        assert (Path(row["row_root"]) / native["log"]).is_file()
