"""Reusable runners for E2E validation tests.

These invoke a TB2J public entry point (an installed CLI module) in a subprocess
on governed stored input, then load the canonical ``SpinIO`` result. A test that
needs the result calls :func:`run_tb2j_module`; the comparison itself is done
with the helpers in :mod:`utils.spinio_checks` (ADR-001/ADR-002).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from TB2J.io_exchange.io_exchange import SpinIO

__all__ = ["run_tb2j_module", "Tb2jCliError"]


class Tb2jCliError(RuntimeError):
    """Raised when a TB2J CLI module exits non-zero in an E2E test."""


def run_tb2j_module(
    module: str,
    args: list[str],
    output_dir: Path | str,
    *,
    pickle_name: str = "TB2J.pickle",
) -> SpinIO:
    """Run ``python -m <module> <args>`` and return the loaded ``SpinIO``.

    ``output_dir`` is passed to the CLI via ``--output_path`` and is where
    ``TB2J.pickle`` is expected. Raises :class:`Tb2jCliError` on non-zero exit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", module, *args, "--output_path", str(output_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Tb2jCliError(
            f"{module} exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return SpinIO.load_pickle(str(output_dir), fname=pickle_name)
