"""Tests for Bruno's correction feature.

Covers:
- ExchangeParams.bruno_correction default and explicit setting
- CLI argument --bruno_correction parsing
- Manager NCL validation and qspace auto-enable
- ExchangeCLQspace.get_Jdict() Bruno Jdict computation
- SpinIO exchange_Jdict_bruno storage and _write_bruno_variant
- write_exchange_section J_iso(Bruno) output line

Run from the repository root:

    pytest tests/unit/test_bruno_correction.py -v

"""

from __future__ import annotations

import argparse
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from TB2J.exchange_params import (
    ExchangeParams,
    add_exchange_args_to_parser,
    parser_argument_to_dict,
)

# ---------------------------------------------------------------------------
# 1. ExchangeParams
# ---------------------------------------------------------------------------


class TestExchangeParamsBruno:
    def test_bruno_correction_defaults_to_false(self):
        params = ExchangeParams(efermi=-5.0, magnetic_elements=["Fe"])
        assert params.bruno_correction == ""

    def test_bruno_correction_can_be_set_true(self):
        params = ExchangeParams(
            efermi=-5.0, magnetic_elements=["Fe"], bruno_correction=True
        )
        assert params.bruno_correction == "fft"


# ---------------------------------------------------------------------------
# 2. CLI argument
# ---------------------------------------------------------------------------


class TestCLIArgBruno:
    def test_bruno_correction_flag_absent(self):
        parser = argparse.ArgumentParser()
        add_exchange_args_to_parser(parser)
        args = parser.parse_args([])
        assert args.bruno_correction == ""

    def test_bruno_correction_flag_present(self):
        parser = argparse.ArgumentParser()
        add_exchange_args_to_parser(parser)
        args = parser.parse_args(["--bruno_correction"])
        assert args.bruno_correction == "fft"

    def test_parser_argument_to_dict_includes_bruno(self):
        parser = argparse.ArgumentParser()
        add_exchange_args_to_parser(parser)
        args = parser.parse_args(["--bruno_correction"])
        d = parser_argument_to_dict(args)
        assert "bruno_correction" in d
        assert d["bruno_correction"] == "fft"


# ---------------------------------------------------------------------------
# 3. Manager validation and qspace auto-enable
# ---------------------------------------------------------------------------


class TestManagerBruno:
    @patch("TB2J.interfaces.manager.ExchangeCL2")
    def test_bruno_uses_realspace_by_default(self, mock_cls):
        """When bruno_correction=True and qspace=False, Manager should
        select ExchangeCL2 (real-space Bruno)."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        from TB2J.interfaces.manager import Manager

        Manager(
            atoms=MagicMock(),
            models=MagicMock(),
            basis=[],
            colinear=True,
            qspace=False,
            bruno_correction=True,
            efermi=-5.0,
            magnetic_elements=["Fe"],
            output_path="/tmp/dummy_bruno",
        )

        mock_cls.assert_called_once()

    def test_bruno_with_ncl_raises(self):
        """bruno_correction=True + colinear=False must raise NotImplementedError."""
        from TB2J.interfaces.manager import Manager

        with pytest.raises(NotImplementedError, match="Bruno"):
            Manager(
                atoms=MagicMock(),
                models=MagicMock(),
                basis=[],
                colinear=False,
                qspace=False,
                bruno_correction=True,
                efermi=-5.0,
                magnetic_elements=["Fe"],
            )

    @patch("TB2J.interfaces.manager.ExchangeCL2")
    def test_no_bruno_uses_default_class(self, mock_cls):
        """Without bruno_correction, colinear+!qspace should use ExchangeCL2."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        from TB2J.interfaces.manager import Manager

        Manager(
            atoms=MagicMock(),
            models=MagicMock(),
            basis=[],
            colinear=True,
            qspace=False,
            bruno_correction=False,
            efermi=-5.0,
            magnetic_elements=["Fe"],
            output_path="/tmp/dummy_no_bruno",
        )

        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# 4. ExchangeCLQspace.get_Jdict() Bruno block
# ---------------------------------------------------------------------------


class TestGetJdictBruno:
    """Test that get_Jdict populates exchange_Jdict_bruno when bruno_correction=True."""

    def _make_mock_exchange(self, bruno_correction=True):
        """Create a mock ExchangeCLQspace with enough state to run get_Jdict."""
        from TB2J.exchange_qspace import ExchangeCLQspace

        exchange = MagicMock(spec=ExchangeCLQspace)
        exchange.bruno_correction = bruno_correction
        exchange.exchange_Jdict = {}
        exchange.exchange_Jdict_bruno = None

        # 2 magnetic atoms, 2 R-vectors
        exchange.Rlist = np.array([[0, 0, 0], [1, 0, 0]])
        exchange.ind_mag_atoms = [0, 1]

        exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        exchange.JR = np.array(
            [
                [[1.0, 5.0], [5.0, 1.0]],
                [[1.0, 3.0], [3.0, 1.0]],
            ]
        )
        exchange.Jnorm_R = np.array(
            [
                [[1.0, 4.5], [4.5, 1.0]],
                [[1.0, 2.7], [2.7, 1.0]],
            ]
        )

        # distance_dict must contain all non-self keys for get_Jdict to populate them
        vec01 = np.array([2.5, 0.0, 0.0])
        vec10 = np.array([-2.5, 0.0, 0.0])
        exchange.distance_dict = {
            ((0, 0, 0), 0, 1): (vec01, 2.5),
            ((0, 0, 0), 1, 0): (vec10, 2.5),
            ((1, 0, 0), 0, 1): (vec01 + np.array([3.8, 0, 0]), 4.55),
            ((1, 0, 0), 1, 0): (vec10 + np.array([3.8, 0, 0]), 4.55),
        }

        def fake_ispin(iatom):
            return iatom

        exchange.ispin = fake_ispin

        ExchangeCLQspace.get_Jdict(exchange)
        return exchange

    def test_bruno_jdict_populated_when_enabled(self):
        ex = self._make_mock_exchange(bruno_correction=True)
        assert ex.exchange_Jdict_bruno is not None
        assert len(ex.exchange_Jdict_bruno) > 0

    def test_bruno_jdict_none_when_disabled(self):
        ex = self._make_mock_exchange(bruno_correction=False)
        assert ex.exchange_Jdict_bruno is None

    def test_bruno_jdict_excludes_self_interaction(self):
        ex = self._make_mock_exchange(bruno_correction=True)
        assert (0, 0, 0, 0, 0) not in ex.exchange_Jdict_bruno
        assert (0, 0, 0, 1, 1) not in ex.exchange_Jdict_bruno

    def test_bruno_jdict_includes_cross_terms(self):
        ex = self._make_mock_exchange(bruno_correction=True)
        key_01 = ((1, 0, 0), 0, 1)
        key_10 = ((1, 0, 0), 1, 0)
        assert key_01 in ex.exchange_Jdict_bruno
        assert key_10 in ex.exchange_Jdict_bruno

    def test_bruno_jdict_values_match_jnorm_r(self):
        ex = self._make_mock_exchange(bruno_correction=True)
        key = ((1, 0, 0), 0, 1)
        expected = ex.Jnorm_R[1, 0, 1] / np.sign(np.dot(ex.spinat[0], ex.spinat[1]))
        assert ex.exchange_Jdict_bruno[key] == pytest.approx(expected)

    def test_bruno_jdict_matches_bare_jdict_keys(self):
        ex = self._make_mock_exchange(bruno_correction=True)
        assert set(ex.exchange_Jdict.keys()) == set(ex.exchange_Jdict_bruno.keys())


# ---------------------------------------------------------------------------
# 5. SpinIO exchange_Jdict_bruno storage
# ---------------------------------------------------------------------------


class TestSpinIOBruno:
    def _make_spinio(self, exchange_Jdict_bruno=None):
        from ase import Atoms

        from TB2J.io_exchange.io_exchange import SpinIO

        atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5])
        return SpinIO(
            atoms=atoms,
            spinat=np.array([[0, 0, 2.0], [0, 0, 2.0]]),
            charges=np.array([6.0, 6.0]),
            index_spin=[0, 1],
            colinear=True,
            distance_dict={((1, 0, 0), 0, 1): (np.array([2.5, 0, 0]), 2.5)},
            exchange_Jdict={((1, 0, 0), 0, 1): 0.003},
            exchange_Jdict_bruno=exchange_Jdict_bruno,
        )

    def test_bruno_stored_when_provided(self):
        bruno_dict = {((1, 0, 0), 0, 1): 0.0025}
        sio = self._make_spinio(exchange_Jdict_bruno=bruno_dict)
        assert sio.exchange_Jdict_bruno == bruno_dict

    def test_bruno_defaults_to_none(self):
        sio = self._make_spinio()
        assert sio.exchange_Jdict_bruno is None


# ---------------------------------------------------------------------------
# 6. _write_bruno_variant
# ---------------------------------------------------------------------------


class TestWriteBrunoVariant:
    def _make_spinio_with_bruno(self):
        from ase import Atoms

        from TB2J.io_exchange.io_exchange import SpinIO

        atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5])
        sio = SpinIO(
            atoms=atoms,
            spinat=np.array([[0, 0, 2.0], [0, 0, 2.0]]),
            charges=np.array([6.0, 6.0]),
            index_spin=[0, 1],
            colinear=True,
            distance_dict={((1, 0, 0), 0, 1): (np.array([2.5, 0, 0]), 2.5)},
            exchange_Jdict={((1, 0, 0), 0, 1): 0.003},
            exchange_Jdict_bruno={((1, 0, 0), 0, 1): 0.0025},
        )
        return sio

    def test_bruno_variant_writes_separate_dirs(self):
        sio = self._make_spinio_with_bruno()

        sio.write_multibinit = MagicMock()
        sio.write_vampire = MagicMock()
        sio.write_espins = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            sio._write_bruno_variant(path=tmpdir)

        sio.write_multibinit.assert_called_once()
        sio.write_vampire.assert_called_once()
        sio.write_espins.assert_called_once()

        multi_path = sio.write_multibinit.call_args[1]["path"]
        vampire_path = sio.write_vampire.call_args[1]["path"]
        espins_path = sio.write_espins.call_args[1]["path"]

        assert "Multibinit_with_bruno_correction" in multi_path
        assert "Vampire_with_bruno_correction" in vampire_path
        assert "ESPInS_with_bruno_correction" in espins_path

    def test_bruno_variant_swaps_jdict(self):
        sio = self._make_spinio_with_bruno()
        original_Jdict = sio.exchange_Jdict

        captured_jdicts = []

        def capture_multibinit(path):
            captured_jdicts.append(dict(sio.exchange_Jdict))

        sio.write_multibinit = capture_multibinit
        sio.write_vampire = MagicMock()
        sio.write_espins = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            sio._write_bruno_variant(path=tmpdir)

        assert len(captured_jdicts) == 1
        bruno_key = ((1, 0, 0), 0, 1)
        assert captured_jdicts[0][bruno_key] == pytest.approx(0.0025)

        assert sio.exchange_Jdict is original_Jdict


# ---------------------------------------------------------------------------
# 7. write_exchange_section J_iso(Bruno) line
# ---------------------------------------------------------------------------


class TestWriteExchangeSectionBruno:
    def _make_spinio_with_exchange(self, bruno_dict=None):
        from ase import Atoms

        from TB2J.io_exchange.io_exchange import SpinIO

        atoms = Atoms("Fe2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[5, 5, 5])
        return SpinIO(
            atoms=atoms,
            spinat=np.array([[0, 0, 2.0], [0, 0, 2.0]]),
            charges=np.array([6.0, 6.0]),
            index_spin=[0, 1],
            colinear=True,
            distance_dict={((1, 0, 0), 0, 1): (np.array([2.5, 0, 0]), 2.5)},
            exchange_Jdict={((1, 0, 0), 0, 1): 0.003},
            exchange_Jdict_bruno=bruno_dict,
        )

    def test_bruno_line_written_when_present(self):
        from TB2J.io_exchange.io_txt import write_exchange_section

        sio = self._make_spinio_with_exchange(bruno_dict={((1, 0, 0), 0, 1): 0.0025})
        buf = StringIO()
        write_exchange_section(sio, buf)
        output = buf.getvalue()
        assert "J_iso(Bruno):" in output

    def test_bruno_line_absent_when_none(self):
        from TB2J.io_exchange.io_txt import write_exchange_section

        sio = self._make_spinio_with_exchange(bruno_dict=None)
        buf = StringIO()
        write_exchange_section(sio, buf)
        output = buf.getvalue()
        assert "J_iso(Bruno):" not in output

    def test_bruno_line_absent_when_key_missing(self):
        """If exchange_Jdict_bruno exists but doesn't contain this key,
        no Bruno line should appear."""
        from TB2J.io_exchange.io_txt import write_exchange_section

        sio = self._make_spinio_with_exchange(bruno_dict={((2, 0, 0), 0, 1): 0.002})
        buf = StringIO()
        write_exchange_section(sio, buf)
        output = buf.getvalue()
        assert "J_iso(Bruno):" not in output

    def test_bruno_value_in_meV(self):
        from TB2J.io_exchange.io_txt import write_exchange_section

        sio = self._make_spinio_with_exchange(bruno_dict={((1, 0, 0), 0, 1): 0.0025})
        buf = StringIO()
        write_exchange_section(sio, buf)
        output = buf.getvalue()

        for line in output.splitlines():
            if "J_iso(Bruno):" in line:
                value = float(line.split(":")[1].strip())
                assert value == pytest.approx(2.5, abs=0.01)
                break
        else:
            pytest.fail("J_iso(Bruno) line not found")

    def test_bruno_line_after_j_iso_line(self):
        from TB2J.io_exchange.io_txt import write_exchange_section

        sio = self._make_spinio_with_exchange(bruno_dict={((1, 0, 0), 0, 1): 0.0025})
        buf = StringIO()
        write_exchange_section(sio, buf)
        lines = buf.getvalue().splitlines()

        j_iso_idx = None
        bruno_idx = None
        for i, line in enumerate(lines):
            if "J_iso:" in line and "Bruno" not in line:
                j_iso_idx = i
            if "J_iso(Bruno):" in line:
                bruno_idx = i

        assert j_iso_idx is not None, "J_iso line not found"
        assert bruno_idx is not None, "J_iso(Bruno) line not found"
        assert (
            bruno_idx == j_iso_idx + 1
        ), f"Bruno line at {bruno_idx} should be immediately after J_iso at {j_iso_idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
