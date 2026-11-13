import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange.edit import (
    load,
    make_supercell,
    open_boundary,
    run_pipeline,
    save,
)
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict

REPO_ROOT = Path(__file__).parents[2]


def _chain_spinio(nnn=True):
    """Single-spin FM Heisenberg chain: onsite + NN (+ 2nd NN) along a."""
    atoms = Atoms("Fe", cell=np.eye(3), scaled_positions=[[0, 0, 0]], pbc=True)
    terms = {((0, 0, 0), 0, 0): -0.1, ((1, 0, 0), 0, 0): 2.0}
    if nnn:
        terms[((2, 0, 0), 0, 0)] = 0.5
    return SpinIO(
        atoms=atoms,
        spinat=np.array([[0.0, 0.0, 3.0]]),
        charges=np.array([6.0]),
        index_spin=[0],
        distance_dict=gen_distance_dict(
            [0], atoms, sorted({R for (R, _, _) in terms})
        ),
        exchange_Jdict=terms,
        dmi_ddict=None,
        Jani_dict=None,
        sia_tensor=None,
    )


def test_open_boundary_empty_directions_is_noop():
    spinio = _chain_spinio()
    before = dict(spinio.exchange_Jdict)
    result = open_boundary(spinio, [])
    assert result is spinio
    assert spinio.exchange_Jdict == before
    assert all(spinio.atoms.get_pbc())


def test_open_boundary_drops_wrapping_bonds_keeps_interior():
    sc_spinio = make_supercell(_chain_spinio(nnn=False), [2, 1, 1])
    # replicated: onsite x2, NN mapped to interior + wrapping pair
    assert set(sc_spinio.exchange_Jdict) == {
        ((0, 0, 0), 0, 0),
        ((0, 0, 0), 1, 1),
        ((0, 0, 0), 0, 1),
        ((1, 0, 0), 1, 0),
    }
    open_boundary(sc_spinio, ["a"])
    assert set(sc_spinio.exchange_Jdict) == {
        ((0, 0, 0), 0, 0),
        ((0, 0, 0), 1, 1),
        ((0, 0, 0), 0, 1),
    }
    assert list(sc_spinio.atoms.get_pbc()) == [False, True, True]
    assert set(sc_spinio.distance_dict) == set(sc_spinio.exchange_Jdict)


def test_open_boundary_exact_bond_loss_count():
    sc_spinio = make_supercell(_chain_spinio(nnn=True), [6, 1, 1])
    assert len(sc_spinio.exchange_Jdict) == 6 * 3
    open_boundary(sc_spinio, "a")
    # onsite kept 6, NN kept 6-1=5, 2nd-NN kept 6-2=4
    assert len(sc_spinio.exchange_Jdict) == 6 + 5 + 4
    assert all(R[0] == 0 for (R, _, _) in sc_spinio.exchange_Jdict)
    assert len(sc_spinio.Rlist) == 1


def test_open_boundary_invalid_direction_raises():
    with pytest.raises(ValueError):
        open_boundary(_chain_spinio(), ["x"])


def _jq_true_phases(spinio, k):
    """Hermitian isotropic J(k) with full Bloch phases k.(R + r_j - r_i).

    Pair terms are stored once per bond direction, so the transpose-conjugate
    is added here. magnon3's Jq uses k.R only, which is wrong for replicated
    supercells (measured 2026-11-13: flat non-Goldstone bands); these oracles
    therefore verify the edit mapping directly at the J level.
    """
    cell = np.array(spinio.atoms.get_cell())
    pos = spinio.atoms.get_positions() @ np.linalg.inv(cell)
    nspin = max(int(i) for i in spinio.index_spin if i >= 0) + 1
    spin_to_atom = {
        int(s): a for a, s in enumerate(spinio.index_spin) if s >= 0
    }
    mat = np.zeros((nspin, nspin), dtype=complex)
    for (R, i, j), value in spinio.exchange_Jdict.items():
        phase = 2 * np.pi * np.dot(
            np.array(R) + pos[spin_to_atom[j]] - pos[spin_to_atom[i]], k
        )
        mat[i, j] += value * np.exp(1j * phase)
        mat[j, i] += np.conj(value * np.exp(1j * phase))
    return mat


@pytest.mark.parametrize("q", [0.0, 0.25])
def test_replicated_jq_folds_primitive_jq(q):
    prim = _chain_spinio()
    sc = make_supercell(_chain_spinio(), [6, 1, 1])
    # supercell fractional k for primitive q: k_sc = q * 6 mod 1
    k_sc = (q * 6.0) % 1.0
    eigen_sc = np.linalg.eigvalsh(_jq_true_phases(sc, np.array([k_sc, 0.0, 0.0])))
    # unfolded reference: primitive 1x1 J(q + G/6) for G = 0..5
    reference = sorted(
        np.linalg.eigvalsh(
            _jq_true_phases(prim, np.array([q + G / 6.0, 0.0, 0.0]))
        )[0]
        for G in range(6)
    )
    assert np.allclose(sorted(eigen_sc), reference, atol=1e-10)


def test_cut_supercell_equals_finite_cluster():
    sc = make_supercell(_chain_spinio(), [6, 1, 1])
    open_boundary(sc, ["a"])
    # strict truncation: the cut supercell must be exactly the 6-site finite
    # chain with onsite -0.1, NN J=2.0 (5 bonds) and 2nd-NN J=0.5 (4 bonds),
    # each bond stored once (upper triangle)
    expected = np.diag([-0.1] * 6)
    for m in range(6):
        for d, J in [(1, 2.0), (2, 0.5)]:
            if m + d < 6:
                expected[m, m + d] += J
    # surviving terms all have R = 0: the Jdict *is* the cluster adjacency
    adjacency = np.zeros((6, 6))
    for (R, i, j), value in sc.exchange_Jdict.items():
        assert R == (0, 0, 0)
        adjacency[i, j] += value
    assert np.allclose(adjacency, expected)


def _pipeline_toml(tmp_path, name, extra):
    prim_dir = tmp_path / "prim"
    save(_chain_spinio(), prim_dir)
    config = f'input = "{prim_dir}"\noutput = "{tmp_path / name}"\n' + extra
    toml_path = tmp_path / f"{name}.toml"
    toml_path.write_text(config)
    return toml_path, prim_dir


def test_pipeline_equals_chained_calls(tmp_path):
    toml_path, prim_dir = _pipeline_toml(
        tmp_path,
        "pipe_out",
        'supercell_matrix = [2, 1, 1]\nopen_directions = ["a"]\n',
    )
    run_pipeline(toml_path)
    via_pipeline = load(str(tmp_path / "pipe_out"))

    chained = open_boundary(
        make_supercell(load(str(prim_dir)), [2, 1, 1]), ["a"]
    )
    assert via_pipeline.exchange_Jdict == chained.exchange_Jdict
    assert list(via_pipeline.index_spin) == list(chained.index_spin)
    assert len(via_pipeline.atoms) == len(chained.atoms)
    assert list(via_pipeline.atoms.get_pbc()) == list(chained.atoms.get_pbc())


def test_pipeline_rejects_unknown_keys(tmp_path):
    toml_path, _ = _pipeline_toml(
        tmp_path, "bad_out", 'bogus_key = 1\nopen_directions = ["a"]\n'
    )
    with pytest.raises(ValueError, match="bogus_key"):
        run_pipeline(toml_path)


def test_pipeline_cli_smoke(tmp_path):
    toml_path, _ = _pipeline_toml(
        tmp_path,
        "cli_out",
        'supercell_matrix = [4, 1, 1]\nopen_directions = ["a"]\n',
    )
    result = subprocess.run(
        [sys.executable, "-m", "TB2J.scripts.TB2J_edit", "pipeline", str(toml_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli_out" / "TB2J.pickle").exists()
