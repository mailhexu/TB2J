"""Tests for tbupy2J CLI glue."""

from __future__ import annotations

from TB2J.scripts import tbupy2J


def test_tbupy2j_cli_delegates_to_python_api(monkeypatch):
    calls = {}

    def fake_gen_exchange_tbupy(**kwargs):
        calls.update(kwargs)
        return "ok"

    monkeypatch.setattr(tbupy2J, "gen_exchange_tbupy", fake_gen_exchange_tbupy)

    result = tbupy2J.main(
        [
            "--input",
            "mock.tbupy.nc",
            "--output_path",
            "out",
            "--kmesh",
            "1",
            "1",
            "1",
            "--elements",
            "Mn",
        ]
    )

    assert result == "ok"
    assert calls["tbupy_result_file"] == "mock.tbupy.nc"
    assert calls["output_path"] == "out"
    assert calls["kmesh"] == [1, 1, 1]
    assert calls["magnetic_elements"] == ["Mn"]
