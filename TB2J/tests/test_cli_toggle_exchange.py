from unittest.mock import MagicMock, patch

from TB2J.scripts.TB2J_edit import cmd_toggle_exchange


def test_cmd_toggle_exchange():
    args = MagicMock()
    args.input = "dummy.pickle"
    args.output = "output_dir"
    args.enable = False
    args.disable = True

    with patch("TB2J.io_exchange.edit.load") as mock_load, patch(
        "TB2J.io_exchange.edit.save"
    ) as mock_save, patch("TB2J.io_exchange.edit.toggle_exchange") as mock_toggle:
        mock_spinio = MagicMock()
        mock_spinio.has_exchange = True
        mock_load.return_value = mock_spinio

        cmd_toggle_exchange(args)

        mock_load.assert_called_with("dummy.pickle")
        mock_toggle.assert_called_with(mock_spinio, enabled=False)
        mock_save.assert_called_with(mock_spinio, "output_dir")


def test_cmd_toggle_exchange_enable():
    args = MagicMock()
    args.input = "dummy.pickle"
    args.output = "output_dir"
    args.enable = True
    args.disable = False

    with patch("TB2J.io_exchange.edit.load") as mock_load, patch(
        "TB2J.io_exchange.edit.save"
    ) as mock_save, patch("TB2J.io_exchange.edit.toggle_exchange") as mock_toggle:
        mock_spinio = MagicMock()
        mock_spinio.has_exchange = False
        mock_load.return_value = mock_spinio

        cmd_toggle_exchange(args)

        mock_load.assert_called_with("dummy.pickle")
        mock_toggle.assert_called_with(mock_spinio, enabled=True)
        mock_save.assert_called_with(mock_spinio, "output_dir")


if __name__ == "__main__":
    test_cmd_toggle_exchange()
    test_cmd_toggle_exchange_enable()
