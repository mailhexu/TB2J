from unittest.mock import MagicMock, patch

from TB2J.scripts.TB2J_edit import cmd_remove_sublattice


def test_cmd_remove_sublattice():
    # Mock args
    args = MagicMock()
    args.input = "dummy.pickle"
    args.output = "output_dir"
    args.sublattice = "Sm"

    # Mock load, save, remove_sublattice
    with patch("TB2J.io_exchange.edit.load") as mock_load, patch(
        "TB2J.io_exchange.edit.save"
    ) as mock_save, patch(
        "TB2J.io_exchange.edit.remove_sublattice"
    ) as mock_remove_sublattice:
        # Setup mock return value
        mock_spinio = MagicMock()
        mock_load.return_value = mock_spinio

        # Run command
        cmd_remove_sublattice(args)

        # Assertions
        mock_load.assert_called_with("dummy.pickle")
        mock_remove_sublattice.assert_called_with(mock_spinio, "Sm")
        mock_save.assert_called_with(mock_spinio, "output_dir")


if __name__ == "__main__":
    test_cmd_remove_sublattice()
