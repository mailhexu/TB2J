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


@pytest.fixture(scope="session")
def tests_data() -> Path:
    """Absolute path to the ``tests/data`` submodule root."""
    return TESTS_DATA
