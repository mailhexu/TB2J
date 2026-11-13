"""Regression tests for the spin-wave stiffness of the magnon code.

The stiffness is the coefficient of the long-wavelength expansion of the
acoustic magnon branch,

    omega(k) = omega0 + k^T A k     (quadratic branch, meV A^2)
    omega(k) = omega0 + v . k       (linear branch, meV A)

with the magnon convention of `TB2J.magnon.magnon3`
(H = -sum_<ij> J S_i.S_j, unordered bonds once, J in meV, |S| = |magmom|/2).

The reference values are the textbook ones and are pinned symbolically in
`docs/sympy/magnon_spin_stiffness.py`:

    FM chain      A_xx = |J S a^2|
    FM honeycomb  A_xx = A_yy = |J S a^2| / 4, A_xy = 0
    AFM chain     linear branch, speed |v| = J S a_cell
"""

from pathlib import Path

import numpy as np
import pytest

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.magnon_stiffness import stiffness_sum_rule

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


def _fm_chain(S, jiso, a=1.0):
    """FM chain, one site per cell, bonds to the neighbouring cells (jiso in eV).

    `Rlist` is in lattice coordinates (the cell carries the length a).
    """
    magmom = 2 * S
    Rlist = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((2, 1, 1, 3, 3))
    JR[0, 0, 0] = np.eye(3) * jiso
    JR[1, 0, 0] = np.eye(3) * jiso
    M = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0, magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3) * a,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom]]),
    )
    return M


def _fm_honeycomb(S, jiso, a=1.0):
    """FM honeycomb, two sites per cell, graphene nearest-neighbour bonds."""
    cellv = np.array([[a, 0, 0], [a / 2, np.sqrt(3) * a / 2, 0], [0, 0, 10.0]])
    nspin = 2
    magmom = 2 * S
    Rs, JR = [], []
    n = 2
    for r1 in range(-n, n + 1):
        for r2 in range(-n, n + 1):
            Rs.append(np.array([r1, r2, 0.0]))
            JR.append(np.zeros((nspin, nspin, 3, 3)))
            for i in range(nspin):
                for jj in range(nspin):
                    if i == jj:
                        continue
                    if (i, jj) == (0, 1) and (r1, r2) in [(0, 0), (-1, 0), (0, -1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
                    if (i, jj) == (1, 0) and (r1, r2) in [(0, 0), (1, 0), (0, 1)]:
                        JR[-1][i, jj] = np.eye(3) * jiso
    M = Magnon(
        nspin=nspin,
        magmom=np.array([[0.0, 0, magmom]] * nspin),
        Rlist=np.array(Rs),
        JR=np.array(JR),
        cell=cellv,
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom]] * nspin),
    )
    return M


def _afm_chain(S, jiso, a=1.0):
    """AFM chain, two-site Neel cell (magnetic cell 2a)."""
    magmom = 2 * S
    Rlist = np.array([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]])
    JR = np.zeros((3, 2, 2, 3, 3))
    JR[0, 0, 1] = np.eye(3) * jiso
    JR[0, 1, 0] = np.eye(3) * jiso
    JR[1, 1, 0] = np.eye(3) * jiso
    JR[2, 0, 1] = np.eye(3) * jiso
    M = Magnon(
        nspin=2,
        magmom=np.array([[0.0, 0, magmom], [0.0, 0, -magmom]]),
        Rlist=Rlist,
        JR=JR,
        cell=np.array([[2 * a, 0, 0], [0, 2 * a, 0], [0, 0, 10.0]]),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0, 1.0]]),
        _n=np.array([1.0, 0, 0]),
    )
    M.set_reference(
        np.zeros(3),
        np.array([[0.0, 0, 1.0]]),
        np.array([1.0, 0, 0]),
        magmoms=np.array([[0.0, 0, magmom], [0.0, 0, -magmom]]),
    )
    return M


def test_fm_chain_stiffness():
    """A_xx = |J S a^2| for the FM chain."""
    S, jiso, a = 2.0, -1.0e-3, 1.0
    st = _fm_chain(S, jiso, a).get_spin_stiffness()
    assert st.order == 2
    assert abs(abs(st.A[0, 0]) - abs(S * jiso * 1e3 * a**2)) < 1e-3
    # no exchange in the transverse directions
    assert abs(st.A[1, 1]) < 1e-8
    assert abs(st.A[2, 2]) < 1e-8
    assert abs(st.A[0, 1]) < 1e-8


def test_fm_chain_stiffness_scales_with_lattice_constant():
    """The stiffness scales with a^2."""
    S, jiso = 2.0, -1.0e-3
    a1 = _fm_chain(S, jiso, 1.0).get_spin_stiffness().A[0, 0]
    a2 = _fm_chain(S, jiso, 2.0).get_spin_stiffness().A[0, 0]
    assert abs(a2 / a1 - 4.0) < 1e-3


def test_fm_honeycomb_stiffness_is_isotropic():
    """A_xx = A_yy = |J S a^2| / 4 and A_xy = 0 for the FM honeycomb."""
    S, jiso, a = 1.5, -1.0e-3, 1.0
    st = _fm_honeycomb(S, jiso, a).get_spin_stiffness()
    theory = abs(S * jiso * 1e3 * a**2) / 4
    assert st.order == 2
    assert abs(abs(st.A[0, 0]) - theory) / theory < 1e-3
    assert abs(abs(st.A[1, 1]) - theory) / theory < 1e-3
    assert abs(st.A[0, 1]) < 1e-3 * theory


def test_afm_chain_reports_linear_branch_and_speed():
    """The AFM chain has no quadratic stiffness but a finite spin-wave speed."""
    S, jiso, a = 1.5, 1.0e-3, 1.0
    st = _afm_chain(S, jiso, a).get_spin_stiffness()
    theory = S * jiso * 1e3 * 2 * a  # J S a_cell, a_cell = 2a
    assert st.order == 1
    assert st.A is None
    assert abs(np.linalg.norm(st.velocity) - theory) / theory < 1e-3


def test_sum_rule_matches_dispersion_for_bravais_chain():
    """The real-space sum rule reproduces the dispersion route for a Bravais lattice."""
    S, jiso, a = 2.0, -1.0e-3, 1.0
    st = _fm_chain(S, jiso, a).get_spin_stiffness(
        J_iso=[jiso * 1e3, jiso * 1e3],
        bond_vectors=[[a, 0, 0], [-a, 0, 0]],
    )
    assert st.sum_rule is not None
    assert abs(abs(st.sum_rule[0, 0]) - abs(st.A[0, 0])) < 1e-3


def test_sum_rule_is_not_the_multisublattice_answer():
    """The bond sum is not the exact LSWT stiffness of a multi-sublattice lattice."""
    S, jiso, a = 1.5, -1.0e-3, 1.0
    d01 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    a1v = np.array([a, 0.0, 0.0])
    a2v = np.array([a / 2, np.sqrt(3) * a / 2, 0.0])
    A_sum = stiffness_sum_rule(
        J_iso=[jiso * 1e3] * 3,
        S=S,
        bond_vectors=[d01, d01 - a1v, d01 - a2v],
    )
    st = _fm_honeycomb(S, jiso, a).get_spin_stiffness()
    assert not np.allclose(A_sum, st.A, rtol=1e-3, atol=1e-6)


def test_stiffness_rejects_non_positive_step():
    with pytest.raises(ValueError):
        _fm_chain(2.0, -1.0e-3).get_spin_stiffness(step=0.0)


def test_cli_stiffness_prints_block(monkeypatch, capsys):
    """The CLI loads TB2J results and prints the stiffness block."""
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data not found at {TEST_DATA_DIR}")
    from TB2J.magnon.magnon_cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["TB2J_magnon.py", "--stiffness", "-p", str(TEST_DATA_DIR)],
    )
    main()
    out = capsys.readouterr().out
    assert "Spin-wave stiffness" in out
    assert "stiffness tensor A (meV A^2)" in out or "spin-wave speeds" in out
