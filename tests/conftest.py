"""Shared pytest fixtures and markers for TB2J tests."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
TESTS_DATA_DIR = ROOT_DIR / "data"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tests_data_dir():
    """Path to the test data submodule root.

    Skips the test if the submodule is not initialized.
    """
    if not TESTS_DATA_DIR.is_dir() or not any(TESTS_DATA_DIR.iterdir()):
        pytest.skip(
            "Test data submodule not initialized (run ./tests/init_test_data.sh)"
        )
    return TESTS_DATA_DIR


@pytest.fixture
def e2e_tests_root(tests_data_dir):
    """Path to E2E test scenario directories under tests/data/tests/."""
    return tests_data_dir / "tests"


@pytest.fixture
def inputs_dir(tests_data_dir):
    """Path to E2E test input datasets under tests/data/inputs/."""
    return tests_data_dir / "inputs"


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: End-to-end test (needs test data submodule)"
    )
    config.addinivalue_line(
        "markers", "unit: Fast unit test (no external dependencies)"
    )
    config.addinivalue_line("markers", "slow: Test takes >30s to run")
