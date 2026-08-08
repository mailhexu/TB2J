"""Deprecation test for the native ABINIT NC-PAO exchange CLI (Epic 013-3).

ADR-008: ``abinit_nc_pao2J`` is Deprecated. It must emit a ``DeprecationWarning``
pointing users to the maintained ABINIT + abinao handoff, while remaining
functional (the loader/projector regression coverage lives in
``test_abinit_savetb2j_loader.py`` and ``test_projector_green.py``).
"""

from __future__ import annotations

import warnings

import pytest

from TB2J.scripts.abinit_nc_pao2J import run_abinit_nc_pao2J


def test_abinit_nc_pao_emits_deprecation_warning(monkeypatch):
    """Invoking the deprecated CLI entry emits a DeprecationWarning with guidance."""
    monkeypatch.setattr("sys.argv", ["abinit_nc_pao2J", "--help"])
    with pytest.warns(DeprecationWarning, match="abinit_nc_pao2J .* is deprecated"):
        with pytest.raises(SystemExit):  # --help exits after the warning fires
            run_abinit_nc_pao2J()


def test_abinit_nc_pao_deprecation_names_the_migration_path(monkeypatch):
    """The warning names the maintained replacement (ABINIT + abinao)."""
    monkeypatch.setattr("sys.argv", ["abinit_nc_pao2J", "--help"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run_abinit_nc_pao2J()
        except SystemExit:
            pass
    msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert msgs, "no DeprecationWarning emitted"
    assert "abinao" in msgs[0]
