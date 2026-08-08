"""Shared pytest fixtures and helpers for TB2J validation tests (story 010-2).

Provides:
- ``require_input``: resolve a curated input under ``tests/data`` and skip with a
  named reason when it is absent (ADR-006), so missing coverage stays visible.
- ``tests_data``: the absolute path to the ``tests/data`` submodule root.

These are test-only utilities; they ship nothing (only the ``TB2J`` package is
installed). ``tests/`` is placed on ``sys.path`` via ``pythonpath`` in
``pyproject.toml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Root of the governed test-data submodule (TB2J_test_data).
TESTS_DATA = Path(__file__).resolve().parent / "data"


def require_input(rel_path: str, capability: str, description: str = "") -> Path:
    """Return the absolute path to a curated input under ``tests/data``.

    Skips the test with a named reason when the input is absent, so a missing
    dataset produces an explicit, actionable skip rather than a silent pass or an
    opaque failure. ``capability`` names the Feature Inventory row affected.
    """
    p = TESTS_DATA / rel_path
    if not p.exists():
        suffix = f" ({description})" if description else ""
        pytest.skip(
            f"missing curated input for '{capability}': tests/data/{rel_path}{suffix}"
        )
    return p


def resolve_example(rel_path: str, capability: str, description: str = "") -> Path:
    """Resolve a governed input from ``tests/data`` or the examples root.

    Tries the curated ``tests/data`` copy first, then ``$TB2J_EXAMPLES_DIR``
    (default ``~/projects/TB2J_examples``) so workflow tests can run locally
    against the examples tree before the data is curated into the submodule.
    Skips with a named reason when the input is available in neither place.
    """
    candidates = [TESTS_DATA / rel_path]
    examples_root = os.environ.get(
        "TB2J_EXAMPLES_DIR", str(Path.home() / "projects" / "TB2J_examples")
    )
    candidates.append(Path(examples_root) / rel_path)
    for p in candidates:
        if p.exists():
            return p
    suffix = f" ({description})" if description else ""
    pytest.skip(
        f"missing governed input for '{capability}': tried tests/data/{rel_path} "
        f"and {examples_root}/{rel_path}{suffix}"
    )
    return candidates[0]  # unreachable; for type-checkers


@pytest.fixture(scope="session")
def tests_data() -> Path:
    """Absolute path to the ``tests/data`` submodule root."""
    return TESTS_DATA
