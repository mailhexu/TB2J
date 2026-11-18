"""Tests for the TB2J Wannier90 interface ws-scheme description (story-029).

The auto-detection itself lives in HamiltonIO (``read_from_wannier_dir``) and is
covered by ``HamiltonIO/tests/test_gen_ham_scheme2.py``. TB2J's only addition is
recording the detected scheme + the ndegen-fix migration note in
``WannierManager``'s description. These tests cover that note in isolation,
without constructing the full (heavy) exchange pipeline.
"""

from types import SimpleNamespace

from TB2J.interfaces.wannier90_interface import WannierManager


def _model(use_ws):
    return SimpleNamespace(use_ws=use_ws)


def test_ws_scheme_note_scheme2():
    """A model with use_ws=True -> note mentions per-orbital/scheme 2."""
    note = WannierManager._ws_scheme_note(_model(True))
    assert "scheme 2" in note.lower() or "per-orbital" in note.lower()


def test_ws_scheme_note_scheme1():
    """A model with use_ws=False -> note mentions scheme 1 / global ndegen."""
    note = WannierManager._ws_scheme_note(_model(False))
    assert "scheme 1" in note.lower() or "global" in note.lower()


def test_ws_scheme_note_handles_tuple():
    """Collinear path passes (up, dn); note reads the first model."""
    note = WannierManager._ws_scheme_note((_model(True), _model(False)))
    assert "scheme 2" in note.lower() or "per-orbital" in note.lower()


def test_ws_scheme_note_handles_missing_attr():
    """A model without use_ws defaults to scheme 1 (no crash)."""
    note = WannierManager._ws_scheme_note(SimpleNamespace())
    assert "scheme 1" in note.lower() or "global" in note.lower()


def test_ws_scheme_note_is_concise():
    """The note records only the scheme, not a migration essay."""
    note = WannierManager._ws_scheme_note(_model(True))
    assert "will differ from previous" not in note.lower()
