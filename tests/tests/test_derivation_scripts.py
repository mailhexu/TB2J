"""Story 001 (split-soc-ks): derivation scripts run green inside the suite.

The assertion-checked derivations under ``docs/sympy`` are the single source of
truth for the split-SOC gauge/sign conventions pinned by story 001
(sympy-pinned gauge and sign chains before any cross-backend implementation):

- ``split_soc_gauge``            GPAW psi/chi gauge, Wigner-D projections, tensor map
- ``abinit_nc_soc_sign_chain``   ABINIT NC ``i^l`` / ``amet(-i)`` / conjugation chain
- ``split_soc_insertion``        generalized-S resolvent derivative and topologies

Each script is executed as a subprocess with the suite's interpreter from the
repo root; the assertion checks pass silently on success, so exit status 0
plus emitted progress output is the contract (SPIRAL pattern, commit 7091d66).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYMPY_DIR = REPO_ROOT / "docs" / "sympy"
TIMEOUT_S = 600.0

#: normative assertion-checked derivation scripts (story-001 set)
SCRIPTS = [
    "split_soc_gauge",
    "abinit_nc_soc_sign_chain",
    "split_soc_insertion",
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
