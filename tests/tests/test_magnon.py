"""
Tests for magnon band structure and DOS calculations.

These tests verify that the magnon calculation functionality works correctly
with different parameter configurations.

Run from the repository root:

    pytest tests/tests/test_magnon.py -v

"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from TB2J.magnon.magnon3 import Magnon, plot_magnon_bands_from_TB2J, save_bands_data
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
        np.testing.assert_allclose(magnon._n, [1.0, 0.0, 0.0])
        assert output_file.exists()

        json_file = output_file.with_suffix(".json")
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)
        assert "energies" in data
        assert len(data["energies"]) == 50

    def test_primitive_kpath_folds_by_requested_supercell_matrix(self, temp_output_dir):
        """Primitive q-points fold to supercell coordinates by q_prim @ S.T."""
        supercell_matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
        primitive_cell = Atoms("H", cell=np.eye(3), pbc=True)
        output_file = Path(temp_output_dir) / "folded_bands.png"

        magnon = Magnon(
            nspin=1,
            magmom=np.array([[0.0, 0.0, 1.0]]),
            Rlist=np.zeros((1, 3), dtype=int),
            JR=np.zeros((1, 1, 1, 3, 3)),
            cell=supercell_matrix @ primitive_cell.get_cell().array,
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
            primitive_cell=primitive_cell,
            supercell_matrix=supercell_matrix,
        )
        captured = {}

        def _fake_energies(kpoints):
            captured["kpoints"] = np.array(kpoints)
            return np.zeros((len(kpoints), 1))

        magnon._magnon_energies = _fake_energies

        magnon.plot_magnon_bands(
            npoints=20,
            use_primitive_kpath=True,
            filename=str(output_file),
            show=False,
        )

        kpoints = captured["kpoints"]
        assert np.any(np.all(np.isclose(kpoints, [1.0, 0.5, 0.5]), axis=1))
        assert np.any(np.all(np.isclose(kpoints, [0.5, 0.0, 0.5]), axis=1))

    def test_gamma_label_is_latex_in_output(self):
        """Gamma points should remain LaTeX-formatted in generated labels."""
        cell = Atoms("H", cell=np.eye(3), pbc=True)
        magnon = Magnon(
            nspin=1,
            magmom=np.array([[0.0, 0.0, 1.0]]),
            Rlist=np.array([[0, 0, 0]], dtype=int),
            JR=np.zeros((1, 1, 1, 3, 3)),
            cell=cell.get_cell(),
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
        )

        magnon._magnon_energies = lambda kpoints: np.zeros((len(kpoints), 1))

        labels, _bands, _xlist = magnon.get_magnon_bands(
            path="GX",
            npoints=30,
            use_primitive_kpath=False,
        )

        label_values = [label for _, label in labels]
        assert r"$\Gamma$" in label_values
        assert "Gamma" not in label_values

    def test_save_bands_data_uses_fixed_plot_script_name(self, temp_output_dir, capsys):
        data_file = Path(temp_output_dir) / "custom_band_name.json"

        save_bands_data(
            kpoints=np.zeros((2, 3)),
            energies=np.zeros((2, 1)),
            kpath_labels=[(0, r"$\Gamma$"), (1, "X")],
            special_points={"G": np.zeros(3), "X": np.array([0.5, 0.0, 0.0])},
            xcoords=np.arange(2),
            filename=str(data_file),
        )

        output = capsys.readouterr().out
        assert data_file.exists()
        assert (Path(temp_output_dir) / "plot_magnon_band.py").exists()
        assert not (Path(temp_output_dir) / "plot_custom_band_name.py").exists()
        assert "Generated magnon band files:" in output
        assert f"data: {data_file}" in output
        assert (
            f"plotting script: {Path(temp_output_dir) / 'plot_magnon_band.py'}"
            in output
        )
        assert "Usage:" not in output


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


class TestIncommensurateReferenceCLI:
    """Test public CLI configuration of the known single-Q reference."""

    def test_parser_exposes_ordering_vector_and_rotation_axis(self):
        """The reference Q/n inputs are distinct from sampled magnon k inputs."""
        from TB2J.magnon.magnon_cli import create_parser

        args = create_parser().parse_args(
            [
                "--bands",
                "--ordering-vector",
                "0",
                "0",
                "0.1429",
                "--rotation-axis",
                "0",
                "0",
                "2",
                "--kpath",
                "GX",
            ]
        )

        assert args.ordering_vector == [0.0, 0.0, 0.1429]
        assert args.rotation_axis == [0.0, 0.0, 2.0]
        assert args.kpath == "GX"

    def test_bands_cli_passes_reference_configuration(self, monkeypatch):
        """The canonical bands CLI forwards Q/n/uz to its public params object."""
        from TB2J.magnon import magnon_cli

        captured = []
        monkeypatch.setattr(magnon_cli, "plot_magnon_bands_from_TB2J", captured.append)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon.py",
                "--bands",
                "--ordering-vector",
                "0",
                "0",
                "0.1429",
                "--rotation-axis",
                "0",
                "0",
                "2",
                "--uz-file",
                "reference_axes.dat",
                "--kpath",
                "GX",
            ],
        )

        magnon_cli.main()

        assert len(captured) == 1
        params = captured[0]
        assert params.Q == [0.0, 0.0, 0.1429]
        assert params.n == [0.0, 0.0, 2.0]
        assert params.uz_file == "reference_axes.dat"
        assert params.kpath == "GX"

    def test_omitted_rotation_axis_defaults_to_x(self, tb2j_results, tmp_path):
        """Python, TOML, and CLI omission all configure the x spiral axis."""
        from TB2J.magnon.magnon_cli import create_parser
        from TB2J.magnon.magnon_parameters import (
            parse_common_args,
            prepare_magnon_from_params,
        )

        python_params = MagnonParameters(path=tb2j_results)
        prepared = prepare_magnon_from_params(python_params)
        np.testing.assert_allclose(prepared._n, [1.0, 0.0, 0.0])

        config = tmp_path / "magnon.toml"
        python_params.to_toml(config)
        toml_params = MagnonParameters.from_toml(config)
        np.testing.assert_allclose(
            prepare_magnon_from_params(toml_params)._n, [1.0, 0.0, 0.0]
        )

        cli_params = parse_common_args(
            create_parser().parse_args(["--bands", "--path", tb2j_results])
        )
        np.testing.assert_allclose(
            prepare_magnon_from_params(cli_params)._n, [1.0, 0.0, 0.0]
        )

        class SpinIOStub:
            nspin = 1
            Rlist = np.zeros((1, 3), dtype=int)
            atoms = Atoms("H", cell=np.eye(3), pbc=True)

            def get_magnetic_moments(self):
                return np.array([[0.0, 0.0, 1.0]])

            def get_full_Jtensor_for_Rlist(self, **kwargs):
                return np.zeros((1, 1, 1, 3, 3))

        direct = Magnon.load_from_io(SpinIOStub())
        np.testing.assert_allclose(direct._n, [1.0, 0.0, 0.0])

    def test_set_reference_normalizes_axis_and_rejects_zero_axis(self):
        """A rotation-axis magnitude must not rescale the spiral phase."""
        magnon = Magnon(
            nspin=1,
            magmom=np.array([[0.0, 0.0, 2.0]]),
            Rlist=np.array([[1, 0, 0]]),
            JR=np.eye(3)[None, None, None, :, :],
            cell=np.eye(3),
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
        )

        magnon.set_reference(
            Q=[0.25, 0.0, 0.0],
            uz=[[0.0, 0.0, 1.0]],
            n=[0.0, 0.0, 2.0],
        )

        np.testing.assert_allclose(magnon._n, [0.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="non-zero"):
            magnon.set_reference(
                Q=[0.25, 0.0, 0.0],
                uz=[[0.0, 0.0, 1.0]],
                n=[0.0, 0.0, 0.0],
            )

    def test_parameter_vectors_reject_non_finite_values(self):
        """Known reference vectors must be finite before calculation begins."""
        with pytest.raises(ValueError, match="finite"):
            MagnonParameters(Q=[0.0, np.nan, 0.0])
        with pytest.raises(ValueError, match="non-zero"):
            MagnonParameters(n=[0.0, 0.0, 0.0])


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
