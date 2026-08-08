"""Tests for the pytest profile/marker configuration and conftest helpers (010-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

_REGISTERED_MARKERS = {"tier1", "tier2", "tier3", "default", "slow", "gpu", "ecosystem"}


def test_markers_registered(pytestconfig):
    """All validation markers/tiers are registered (no PytestUnknownMarkWarning)."""
    raw = pytestconfig.getini("markers")
    names = {line.split(":", 1)[0].strip() for line in raw if line.strip()}
    missing = _REGISTERED_MARKERS - names
    assert not missing, f"unregistered markers: {missing}"


def test_default_profile_deselects_optin(pytestconfig):
    """A bare `pytest` run applies the not-slow/gpu/ecosystem mark expression."""
    expr = pytestconfig.option.markexpr
    assert "slow" in expr and "gpu" in expr and "ecosystem" in expr
    # The opt-in profiles are excluded, not included.
    assert "not slow" in expr and "not gpu" in expr and "not ecosystem" in expr


def test_require_input_skips_missing():
    from conftest import require_input

    with pytest.raises(pytest.skip.Exception, match="missing curated input"):
        require_input("does/not/exist_xyz", "Test capability", "synthetic probe")


def test_require_input_returns_existing_path(tmp_path, monkeypatch):
    import conftest

    # Point TESTS_DATA at a temp tree with a known file.
    fake_root = tmp_path / "data"
    fake_root.mkdir()
    (fake_root / "foo" / "bar").mkdir(parents=True)
    monkeypatch.setattr(conftest, "TESTS_DATA", fake_root)
    got = conftest.require_input("foo/bar", "Test capability")
    assert isinstance(got, Path)
    assert got.exists()
