"""Tests for the optional TBUpy interface."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from TB2J.interfaces import tbupy_interface


class DummyModel:
    def __init__(self):
        self.orbs = []
        self.atoms = object()


class DummyResult:
    def __init__(self):
        self.atoms = object()
        self.efermi = 1.25
        self.models = (DummyModel(), DummyModel())

    def to_collinear_models(self, spinflip_tol=1e-10):
        return self.models


def test_core_interface_import_does_not_require_tbupy():
    mod = importlib.import_module("TB2J.interfaces.tbupy_interface")
    assert hasattr(mod, "prepare_tbupy_inputs")


def test_prepare_tbupy_inputs_from_live_result():
    result = DummyResult()

    atoms, models, basis, efermi = tbupy_interface.prepare_tbupy_inputs(
        tbupy_result=result
    )

    assert atoms is result.atoms
    assert models is result.models
    assert basis == []
    assert efermi == 1.25


def test_prepare_tbupy_inputs_from_file_uses_lazy_loader(monkeypatch):
    result = DummyResult()
    monkeypatch.setattr(tbupy_interface, "_load_tbupy_result", lambda path: result)

    atoms, models, _basis, efermi = tbupy_interface.prepare_tbupy_inputs(
        tbupy_result_file="mock.tbupy.nc"
    )

    assert atoms is result.atoms
    assert models is result.models
    assert efermi == 1.25


def test_tbupy_manager_delegates_to_manager(monkeypatch):
    calls = {}
    result = DummyResult()

    def fake_manager_init(self, atoms, models, basis, colinear, **kwargs):
        calls["atoms"] = atoms
        calls["models"] = models
        calls["basis"] = basis
        calls["colinear"] = colinear
        calls["kwargs"] = kwargs

    monkeypatch.setattr(tbupy_interface.Manager, "__init__", fake_manager_init)

    tbupy_interface.TBUpyManager(tbupy_result=result, kmesh=np.array([3, 3, 3]))

    assert calls["atoms"] is result.atoms
    assert calls["models"] is result.models
    assert calls["colinear"] is True
    assert calls["kwargs"]["efermi"] == 1.25


def test_prepare_tbupy_inputs_rejects_spinor_result():
    class SpinorResult(DummyResult):
        def to_collinear_models(self, spinflip_tol=1e-10):
            raise ValueError("spinor TBUpy result requires spinor handoff contract")

    with pytest.raises(ValueError, match="spinor"):
        tbupy_interface.prepare_tbupy_inputs(tbupy_result=SpinorResult())
