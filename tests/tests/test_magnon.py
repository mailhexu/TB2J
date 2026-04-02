"""
Tests for magnon band structure and DOS calculations.

These tests verify that the magnon calculation functionality works correctly
with different parameter configurations.

Run from the repository root:

    pytest tests/tests/test_magnon.py -v

"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J
from TB2J.magnon.magnon_parameters import MagnonParameters

ROOT_DIR = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = (
    ROOT_DIR
    / "tests"
    / "data"
    / "tests"
    / "3_CrI3_wannier_SOC"
    / "refs"
    / "TB2J_results"
)


@pytest.fixture
def tb2j_results():
    """Path to CrI3 test data."""
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_DIR}")
    return str(TEST_DATA_DIR)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMagnonBandsDefault:
    """Test magnon band structure with default parameters."""

    def test_bands_default(self, tb2j_results, temp_output_dir):
        """Test band structure calculation with all interactions enabled."""
        output_file = Path(temp_output_dir) / "bands_default.png"

        params = MagnonParameters(
            path=tb2j_results,
            kpath="GMKG",
            npoints=50,
            filename=str(output_file),
            show=False,
        )

        magnon = plot_magnon_bands_from_TB2J(params)

        assert magnon.nspin == 2
        assert output_file.exists()

        json_file = output_file.with_suffix(".json")
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)
        assert "energies" in data
        assert len(data["energies"]) == 50


class TestMagnonBandsNoDMI:
    """Test magnon band structure without DMI and anisotropic exchange."""

    def test_bands_no_dmi_jani(self, tb2j_results, temp_output_dir):
        """Test band structure with isotropic exchange only."""
        output_file = Path(temp_output_dir) / "bands_no_dmi.png"

        params = MagnonParameters(
            path=tb2j_results,
            kpath="GMKG",
            npoints=50,
            filename=str(output_file),
            DMI=False,
            Jani=False,
            show=False,
        )

        magnon = plot_magnon_bands_from_TB2J(params)

        assert magnon.nspin == 2
        assert output_file.exists()


class TestMagnonTomlConfig:
    """Test magnon calculations using TOML configuration."""

    def test_toml_save_load(self, tb2j_results, temp_output_dir):
        """Test saving and loading parameters from TOML."""
        toml_file = Path(temp_output_dir) / "config.toml"
        output_file = Path(temp_output_dir) / "bands_toml.png"

        params = MagnonParameters(
            path=tb2j_results,
            kpath="GMKG",
            npoints=50,
            filename=str(output_file),
            DMI=False,
            Jani=False,
            show=False,
        )
        params.to_toml(str(toml_file))

        assert toml_file.exists()

        params_loaded = MagnonParameters.from_toml(str(toml_file))
        assert params_loaded.path == tb2j_results
        assert params_loaded.DMI is False
        assert params_loaded.Jani is False

        params_loaded.filename = str(output_file)
        magnon = plot_magnon_bands_from_TB2J(params_loaded)

        assert magnon.nspin == 2
        assert output_file.exists()


class TestMagnonSpinConfig:
    """Test magnon calculations with custom spin configuration."""

    def test_spin_conf_direct(self, tb2j_results, temp_output_dir):
        """Test spin configuration passed directly in params."""
        output_file = Path(temp_output_dir) / "bands_spin_conf.png"

        params = MagnonParameters(
            path=tb2j_results,
            kpath="GMKG",
            npoints=50,
            filename=str(output_file),
            spin_conf=[[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]],
            DMI=False,
            Jani=False,
            show=False,
        )

        magnon = plot_magnon_bands_from_TB2J(params)

        assert magnon.nspin == 2
        assert output_file.exists()

    def test_spin_conf_file(self, tb2j_results, temp_output_dir):
        """Test spin configuration loaded from file."""
        spin_conf_file = Path(temp_output_dir) / "spin_conf.txt"
        output_file = Path(temp_output_dir) / "bands_spin_conf_file.png"

        spin_conf = np.array(
            [
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 3.0],
            ]
        )
        np.savetxt(spin_conf_file, spin_conf)

        params = MagnonParameters(
            path=tb2j_results,
            kpath="GMKG",
            npoints=50,
            filename=str(output_file),
            spin_conf_file=str(spin_conf_file),
            DMI=False,
            Jani=False,
            show=False,
        )

        magnon = plot_magnon_bands_from_TB2J(params)

        assert magnon.nspin == 2
        assert output_file.exists()

    def test_spin_conf_toml(self, tb2j_results, temp_output_dir):
        """Test spin configuration in TOML file."""
        toml_file = Path(temp_output_dir) / "config_spin.toml"
        output_file = Path(temp_output_dir) / "bands_spin_conf_toml.png"

        toml_content = f"""
path = "{tb2j_results}"
kpath = "GMKG"
npoints = 50
filename = "{output_file}"
DMI = false
Jani = false
spin_conf = [[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]
"""
        toml_file.write_text(toml_content)

        params = MagnonParameters.from_toml(str(toml_file))
        magnon = plot_magnon_bands_from_TB2J(params)

        assert magnon.nspin == 2
        assert output_file.exists()


class TestMagnonDOS:
    """Test magnon DOS calculations."""

    def test_dos_calculation(self, tb2j_results, temp_output_dir):
        """Test DOS calculation with default parameters."""
        output_file = Path(temp_output_dir) / "dos.png"

        params = MagnonParameters(
            path=tb2j_results,
            kmesh=[8, 8, 8],
            gamma=True,
            width=0.001,
            npts=101,
            filename=str(output_file),
            DMI=False,
            Jani=False,
            show=False,
        )

        plot_magnon_dos_from_TB2J(params)

        assert output_file.exists()

        json_file = output_file.with_suffix(".json")
        assert json_file.exists()


class TestMagnonParametersValidation:
    """Test MagnonParameters validation."""

    def test_spin_conf_validation_shape(self):
        """Test that spin_conf with wrong shape raises error."""
        with pytest.raises(ValueError, match="must have 3 elements"):
            MagnonParameters(
                path="TB2J_results",
                spin_conf=[[0.0, 0.0], [0.0, 0.0]],
            )

    def test_spin_conf_mutual_exclusivity(self):
        """Test that spin_conf and spin_conf_file are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            MagnonParameters(
                path="TB2J_results",
                spin_conf=[[0.0, 0.0, 3.0]],
                spin_conf_file="spin.txt",
            )

    def test_cli_spin_conf_validation(self):
        """Test CLI spin_conf parsing with wrong number of values."""
        from TB2J.magnon.magnon_parameters import parse_common_args

        class MockArgs:
            path = "TB2J_results"
            Jiso = True
            Jani = True
            DMI = True
            SIA = True
            Q = None
            uz_file = None
            n = None
            spin_conf_file = None
            spin_conf = [0, 0, 3, 0]  # 4 values, should be 3n
            show = False

        with pytest.raises(ValueError, match="must have 3n values"):
            parse_common_args(MockArgs())
