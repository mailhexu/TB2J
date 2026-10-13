from unittest.mock import MagicMock, patch


def test_gen_exchange_abacus_rejects_invalid_split_soc():
    from TB2J.interfaces.abacus.gen_exchange_abacus import gen_exchange_abacus

    try:
        gen_exchange_abacus(path="soc-run", split_soc="invalid")
    except ValueError as exc:
        assert "split_soc" in str(exc)
    else:
        raise AssertionError("invalid split_soc should be rejected")


def test_gen_exchange_abacus_two_requires_nosoc_path(monkeypatch):
    import importlib

    abacus_module = importlib.import_module(
        "TB2J.interfaces.abacus.gen_exchange_abacus"
    )
    monkeypatch.setattr(abacus_module.os.path, "exists", MagicMock(return_value=True))

    try:
        abacus_module.gen_exchange_abacus(path="soc-run", split_soc="two")
    except ValueError as exc:
        assert "path_nosoc" in str(exc)
    else:
        raise AssertionError("split_soc='two' should require path_nosoc")


def test_abacus_split_soc_parsers_use_public_hamiltonio_exports():
    import importlib

    from HamiltonIO.abacus import AbacusSingleStepSOCParser, AbacusSplitSOCParser

    from TB2J import MAEGreen as mae_module

    exchange_module = importlib.import_module(
        "TB2J.interfaces.abacus.gen_exchange_abacus"
    )

    assert exchange_module.AbacusSingleStepSOCParser is AbacusSingleStepSOCParser
    assert exchange_module.AbacusSplitSOCParser is AbacusSplitSOCParser
    assert mae_module.AbacusSingleStepSOCParser is AbacusSingleStepSOCParser
    assert mae_module.AbacusSplitSOCParser is AbacusSplitSOCParser


def test_gen_exchange_abacus_reports_custom_output_path(monkeypatch, capsys):
    import importlib

    abacus_module = importlib.import_module(
        "TB2J.interfaces.abacus.gen_exchange_abacus"
    )
    model = MagicMock()
    model.atoms = ["Fe"]
    model.basis = MagicMock()
    model.efermi = 1.0
    parser = MagicMock()
    parser.read_spin.return_value = "noncollinear"
    parser.get_models.return_value = model
    exchange = MagicMock()

    monkeypatch.setattr(abacus_module.os.path, "exists", MagicMock(return_value=True))
    monkeypatch.setattr(abacus_module, "AbacusParser", MagicMock(return_value=parser))
    monkeypatch.setattr(abacus_module, "ExchangeNCL", MagicMock(return_value=exchange))

    abacus_module.gen_exchange_abacus(
        path="legacy-run",
        suffix="ABACUS",
        magnetic_elements=["Fe"],
        output_path="custom-results",
    )

    assert "custom-results" in capsys.readouterr().out


def test_gen_exchange_abacus_default_uses_legacy_parser(monkeypatch):
    import importlib

    abacus_module = importlib.import_module(
        "TB2J.interfaces.abacus.gen_exchange_abacus"
    )

    model = MagicMock()
    model.atoms = ["Fe"]
    model.basis = MagicMock()
    model.efermi = 1.0
    parser = MagicMock()
    parser.read_spin.return_value = "noncollinear"
    parser.get_models.return_value = model
    exchange = MagicMock()

    monkeypatch.setattr(abacus_module.os.path, "exists", MagicMock(return_value=True))
    monkeypatch.setattr(abacus_module, "AbacusParser", MagicMock(return_value=parser))
    monkeypatch.setattr(abacus_module, "ExchangeNCL", MagicMock(return_value=exchange))

    abacus_module.gen_exchange_abacus(
        path="legacy-run",
        suffix="ABACUS",
        magnetic_elements=["Fe"],
        output_path="out",
    )

    abacus_module.AbacusParser.assert_called_once()
    parser.get_models.assert_called_once_with()
    exchange.run.assert_called_once_with(path="out")


def test_abacus2j_forwards_single_split_soc(monkeypatch):
    from TB2J.scripts import abacus2J

    monkeypatch.setattr(
        "sys.argv",
        [
            "abacus2J.py",
            "--path",
            "soc-run",
            "--suffix",
            "ABACUS",
            "--elements",
            "Fe",
            "--split_soc",
            "single",
        ],
    )

    with patch.object(abacus2J, "print_license"), patch.object(
        abacus2J, "gen_exchange_abacus"
    ) as mock_gen:
        abacus2J.run_abacus2J()

    kwargs = mock_gen.call_args.kwargs
    assert kwargs["split_soc"] == "single"
    assert kwargs["path_nosoc"] is None


def test_abacus2j_forwards_two_split_soc_path(monkeypatch):
    from TB2J.scripts import abacus2J

    monkeypatch.setattr(
        "sys.argv",
        [
            "abacus2J.py",
            "--path",
            "soc-run",
            "--suffix",
            "ABACUS",
            "--elements",
            "Fe",
            "--split_soc",
            "two",
            "--path_nosoc",
            "nosoc-run",
        ],
    )

    with patch.object(abacus2J, "print_license"), patch.object(
        abacus2J, "gen_exchange_abacus"
    ) as mock_gen:
        abacus2J.run_abacus2J()

    kwargs = mock_gen.call_args.kwargs
    assert kwargs["split_soc"] == "two"
    assert kwargs["path_nosoc"] == "nosoc-run"


def test_abacus_get_mae_uses_single_step_parser(monkeypatch):
    from TB2J import MAEGreen as mae_module

    model = MagicMock()
    model.atoms = ["Fe"]
    model.basis = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = model
    mock_mae = MagicMock()

    monkeypatch.setattr(
        mae_module, "AbacusSingleStepSOCParser", MagicMock(return_value=parser)
    )
    monkeypatch.setattr(mae_module, "MAEGreen", MagicMock(return_value=mock_mae))

    mae_module.abacus_get_MAE(
        path_single="single-run/OUT.ABACUS",
        kmesh=[1, 1, 1],
        thetas=[0.0],
        phis=[0.0],
        output_path="out",
    )

    mae_module.AbacusSingleStepSOCParser.assert_called_once_with(
        outpath="single-run/OUT.ABACUS", binary=False
    )
    model.set_so_strength.assert_called_once_with(0.0)
    mock_mae.run.assert_called_once_with(output_path="out", with_eigen=False)


def test_abacus_get_mae_preserves_two_step_parser(monkeypatch):
    from TB2J import MAEGreen as mae_module

    model = MagicMock()
    model.atoms = ["Fe"]
    model.basis = MagicMock()
    parser = MagicMock()
    parser.parse.return_value = model
    mock_mae = MagicMock()

    monkeypatch.setattr(
        mae_module, "AbacusSplitSOCParser", MagicMock(return_value=parser)
    )
    monkeypatch.setattr(mae_module, "MAEGreen", MagicMock(return_value=mock_mae))

    mae_module.abacus_get_MAE(
        "nosoc/OUT.ABACUS",
        "soc/OUT.ABACUS",
        [1, 1, 1],
        [0.0],
        [0.0],
    )

    mae_module.AbacusSplitSOCParser.assert_called_once_with(
        outpath_nosoc="nosoc/OUT.ABACUS",
        outpath_soc="soc/OUT.ABACUS",
        binary=False,
    )


def test_abacus_get_mae_requires_single_path():
    from TB2J.MAEGreen import abacus_get_MAE

    try:
        abacus_get_MAE(split_soc="single", kmesh=[1, 1, 1], thetas=[0.0], phis=[0.0])
    except ValueError as exc:
        assert "path_single" in str(exc)
    else:
        raise AssertionError("split_soc='single' should require a single path")
