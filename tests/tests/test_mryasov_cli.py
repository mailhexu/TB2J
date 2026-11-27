import os
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO


def test_mryasov_cli_round_trips_periodic_bonds_and_isolated_metal(tmp_path):
    atoms = Atoms(
        "PtFe3",
        positions=[[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [2, 0, 0]],
        cell=[3, 4, 5],
        pbc=True,
    )
    source = tmp_path / "source"
    output = tmp_path / "output"
    SpinIO(
        atoms,
        spinat=np.array([[0.3, 0, 0], [3.0, 0, 0], [3.0, 0, 0], [3.0, 0, 0]]),
        charges=np.zeros(4),
        index_spin=[3, 0, 1, 2],
        exchange_Jdict={
            ((-1, 0, 0), 3, 1): 0.002,
            ((1, 0, 0), 1, 3): 0.002,
            ((0, 0, 0), 3, 2): 0.004,
            ((0, 0, 0), 2, 3): 0.004,
            ((1, 0, 0), 3, 1): 0.006,
            ((-1, 0, 0), 1, 3): 0.006,
            ((0, 0, 0), 3, 0): 0.0,
            ((0, 0, 0), 0, 3): 0.0,
        },
    ).write_pickle(path=source)
    command = [
        sys.executable,
        "-m",
        "TB2J.scripts.TB2J_downfold",
        "--method",
        "mryasov",
        "--dressing",
        "1.5",
        "--metals",
        "Fe",
        "--ligands",
        "Pt",
        "--inpath",
        str(source),
        "--outpath",
    ]
    completed = subprocess.run(
        [*command, str(output)],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = SpinIO.load_pickle(path=output)
    assert result.index_spin == [-1, 0, 1, 2]
    assert result.nspin == 3
    assert {k: j for k, j in result.exchange_Jdict.items() if j} == pytest.approx(
        {
            ((-2, 0, 0), 1, 1): 0.0015,
            ((2, 0, 0), 1, 1): 0.0015,
            ((1, 0, 0), 1, 2): 0.001,
            ((-1, 0, 0), 2, 1): 0.001,
            ((-1, 0, 0), 1, 2): 0.003,
            ((1, 0, 0), 2, 1): 0.003,
        }
    )
    assert result.distance_dict[((2, 0, 0), 1, 1)][1] == pytest.approx(6.0)
    assert "lambda=1.5" in result.description
    tom = (output / "TomASD" / "exchange.exch").read_text()
    assert tom.count("exchange_1_2:") == 1
    assert "Num_Interactions = 2;" in tom.split("exchange_1_2:", 1)[1].split("};", 1)[0]
    header = (output / "exchange.out").read_text().split("Exchange:", 1)[0]
    assert "M(x)" in header
    assert any(
        line.split()[:1] == ["Fe2"]
        and line.split()[-3:] == ["3.0000", "0.0000", "0.0000"]
        for line in header.splitlines()
    )

    # A hard-linked destination must not overwrite the original bonds.
    alias = tmp_path / "alias"
    alias.mkdir()
    original = source / "TB2J.pickle"
    original_bytes = original.read_bytes()
    os.link(original, alias / original.name)
    refused = subprocess.run(
        [*command, str(alias)], cwd=tmp_path, capture_output=True, text=True
    )
    assert refused.returncode != 0 and "new or empty directory" in refused.stderr
    assert original.read_bytes() == original_bytes

    command[command.index("--dressing") + 1] = "1e308"
    overflow = tmp_path / "overflow"
    refused = subprocess.run(
        [*command, str(overflow)], cwd=tmp_path, capture_output=True, text=True
    )
    assert refused.returncode != 0 and "not representable in meV" in refused.stderr
    assert not overflow.exists()
