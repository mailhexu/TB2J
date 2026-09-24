"""Story 006 / ADR-S2: derivation scripts run green inside the suite.

The normative assertion-checked derivations under ``docs/sympy`` are the
single source of truth for the spiral MFT conventions (conventions,
rebuild rule, gates, Toth-Lake flat-screw identities).  Each script is
executed as a subprocess with the suite's interpreter from the repo
root; the assertion checks pass silently on success, so exit status 0
plus emitted progress output is the contract.  Total runtime of the
four scripts is a few seconds (no slow marker needed - the default
pytest profile runs them).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SYMPY_DIR = REPO_ROOT / "docs" / "sympy"
TIMEOUT_S = 600.0

#: normative assertion-checked derivation scripts (story-002..005 set)
SCRIPTS = [
    "spin_spiral_generalized_bloch",
    "spiral_force_theorem_J",
    "spiral_green_function_mft",
    "spiral_state_mft",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_derivation_script_runs(name: str):
    """Each derivation script exits 0 with all internal assertions green."""
    script = SYMPY_DIR / f"{name}.py"
    assert script.is_file(), f"missing normative derivation script: {script}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    assert (
        proc.returncode == 0
    ), f"{name} failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    # assertion-checked scripts print their check progress; empty output
    # would mean the checks never ran
    assert proc.stdout.strip(), f"{name} produced no output"
