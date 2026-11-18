from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
REF_DIR = ROOT_DIR.parent / "Refs" / "SPRKKR_RuO2"
STRUCTURE_FILE = REF_DIR / "RuO2.str"
EXCHANGE_FILE = REF_DIR / "RuO2_JXC_Jij.dat"


def test_sprkkr2magnon_parser_requires_band_or_conversion_mode():
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "--structure",
                str(STRUCTURE_FILE),
                "--exchange",
                str(EXCHANGE_FILE),
                "--magnetic-species",
                "Ru",
                "--moment",
                "0.5674",
            ]
        )


def test_sprkkr2magnon_parser_requires_moment():
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "--structure",
                str(STRUCTURE_FILE),
                "--exchange",
                str(EXCHANGE_FILE),
                "--magnetic-species",
                "Ru",
                "--bands",
            ]
        )


def test_sprkkr2magnon_cli_reports_missing_structure_file(tmp_path):
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "--structure",
                str(tmp_path / "missing.str"),
                "--exchange",
                str(EXCHANGE_FILE),
                "--magnetic-species",
                "Ru",
                "--moment",
                "0.5674",
                "--bands",
            ]
        )


def test_sprkkr2magnon_cli_rejects_unsupported_tensor_policy():
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "--structure",
                str(STRUCTURE_FILE),
                "--exchange",
                str(EXCHANGE_FILE),
                "--magnetic-species",
                "Ru",
                "--moment",
                "0.5674",
                "--tensor-policy",
                "full-tensor",
                "--bands",
            ]
        )


def test_sprkkr2magnon_cli_writes_band_outputs(tmp_path):
    from TB2J.scripts.sprkkr2magnon import main

    output_file = tmp_path / "ruo2_sprkkr_magnon.png"

    main(
        [
            "--structure",
            str(STRUCTURE_FILE),
            "--exchange",
            str(EXCHANGE_FILE),
            "--magnetic-species",
            "Ru",
            "--moment",
            "0.5674",
            "--tensor-policy",
            "isotropic",
            "--bands",
            "--kpath",
            "GX",
            "--npoints",
            "4",
            "--output",
            str(output_file),
        ]
    )

    assert output_file.exists()
    data_file = output_file.with_suffix(".json")
    assert data_file.exists()
    with data_file.open() as handle:
        data = json.load(handle)
    assert len(data["energies"]) == 4
    assert len(data["kpoints"]) == 4


def test_sprkkr2magnon_cli_short_options_write_band_outputs(tmp_path):
    from TB2J.scripts.sprkkr2magnon import main

    output_file = tmp_path / "ruo2_sprkkr_magnon_short.png"

    main(
        [
            "-s",
            str(STRUCTURE_FILE),
            "-e",
            str(EXCHANGE_FILE),
            "-S",
            "Ru",
            "-m",
            "0.5674",
            "-t",
            "isotropic",
            "-b",
            "-k",
            "GX",
            "-n",
            "4",
            "-o",
            str(output_file),
        ]
    )

    assert output_file.exists()
    assert output_file.with_suffix(".json").exists()


def test_sprkkr2magnon_cli_accepts_custom_qpoints(tmp_path):
    from TB2J.scripts.sprkkr2magnon import main

    output_file = tmp_path / "ruo2_sprkkr_custom_qpoints.png"

    main(
        [
            "-s",
            str(STRUCTURE_FILE),
            "-e",
            str(EXCHANGE_FILE),
            "-S",
            "Ru",
            "-m",
            "0.5674",
            "-t",
            "isotropic",
            "-b",
            "-k",
            "GX",
            "-n",
            "4",
            "--qpoints",
            "G:0,0,0,X:0.5,0,0",
            "-o",
            str(output_file),
        ]
    )

    assert output_file.exists()
    data_file = output_file.with_suffix(".json")
    assert data_file.exists()
    with data_file.open() as handle:
        data = json.load(handle)
    assert data["special_points"]["Gamma"] == pytest.approx([0.0, 0.0, 0.0])
    assert data["special_points"]["X"] == pytest.approx([0.5, 0.0, 0.0])


def test_sprkkr2magnon_cli_rejects_bad_qpoints(tmp_path):
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "-s",
                str(STRUCTURE_FILE),
                "-e",
                str(EXCHANGE_FILE),
                "-S",
                "Ru",
                "-m",
                "0.5674",
                "-b",
                "-k",
                "GX",
                "--qpoints",
                "G:0,0",
                "-o",
                str(tmp_path / "bad_qpoints.png"),
            ]
        )


def test_sprkkr2magnon_cli_writes_tb2j_pickle(tmp_path):
    from TB2J.io_exchange import SpinIO
    from TB2J.scripts.sprkkr2magnon import main

    output_dir = tmp_path / "TB2J_sprkkr_results"

    main(
        [
            "--structure",
            str(STRUCTURE_FILE),
            "--exchange",
            str(EXCHANGE_FILE),
            "--magnetic-species",
            "Ru",
            "--moment",
            "0.5674",
            "--write-tb2j-results",
            str(output_dir),
        ]
    )

    assert (output_dir / "TB2J.pickle").exists()
    assert (output_dir / "exchange.out").exists()
    assert (output_dir / "structure.vasp").exists()
    assert (output_dir / "Multibinit" / "exchange.xml").exists()
    spinio = SpinIO.load_pickle(path=str(output_dir))
    assert spinio.nspin == 2


def test_sprkkr2magnon_cli_short_options_write_tb2j_pickle(tmp_path):
    from TB2J.io_exchange import SpinIO
    from TB2J.scripts.sprkkr2magnon import main

    output_dir = tmp_path / "TB2J_sprkkr_results_short"

    main(
        [
            "-s",
            str(STRUCTURE_FILE),
            "-e",
            str(EXCHANGE_FILE),
            "-S",
            "Ru",
            "-m",
            "0.5674",
            "-w",
            str(output_dir),
        ]
    )

    assert (output_dir / "TB2J.pickle").exists()
    assert (output_dir / "exchange.out").exists()
    assert (output_dir / "structure.vasp").exists()
    assert (output_dir / "Multibinit" / "exchange.xml").exists()
    spinio = SpinIO.load_pickle(path=str(output_dir))
    assert spinio.nspin == 2


def test_sprkkr2magnon_cli_accepts_signed_site_moments(tmp_path):
    from TB2J.io_exchange import SpinIO
    from TB2J.scripts.sprkkr2magnon import main

    output_dir = tmp_path / "TB2J_sprkkr_signed"

    main(
        [
            "-s",
            str(STRUCTURE_FILE),
            "-e",
            str(EXCHANGE_FILE),
            "-S",
            "Ru",
            "-m",
            "0.5674",
            "-0.5674",
            "-w",
            str(output_dir),
        ]
    )

    spinio = SpinIO.load_pickle(path=str(output_dir))
    assert spinio.get_magnetic_moments()[:, 2].tolist() == pytest.approx(
        [0.5674, -0.5674]
    )


def test_sprkkr2magnon_cli_accepts_vector_moments(tmp_path):
    from TB2J.io_exchange import SpinIO
    from TB2J.scripts.sprkkr2magnon import main

    output_dir = tmp_path / "TB2J_sprkkr_vectors"

    main(
        [
            "-s",
            str(STRUCTURE_FILE),
            "-e",
            str(EXCHANGE_FILE),
            "-S",
            "Ru",
            "-m",
            "0.0",
            "0.0",
            "0.5674",
            "0.0",
            "0.0",
            "-0.5674",
            "-w",
            str(output_dir),
        ]
    )

    spinio = SpinIO.load_pickle(path=str(output_dir))
    assert np.allclose(
        spinio.get_magnetic_moments(), [[0.0, 0.0, 0.5674], [0.0, 0.0, -0.5674]]
    )


def test_sprkkr2magnon_cli_rejects_invalid_moment_length():
    from TB2J.scripts.sprkkr2magnon import main

    with pytest.raises(SystemExit):
        main(
            [
                "-s",
                str(STRUCTURE_FILE),
                "-e",
                str(EXCHANGE_FILE),
                "-S",
                "Ru",
                "-m",
                "0.1",
                "0.2",
                "0.3",
                "-w",
                "TB2J_sprkkr_bad_moments",
            ]
        )
