"""Regression tests for the magnon3 convention patch (ticket 05).

These are the tests proposed to accompany the patch (they fail on the current
code: the frequencies are (2/S^2) times the textbook values).  They assert
external behaviour only -- the magnon frequencies against closed textbook
formulas -- never internal matrix layouts.

Textbook reference (Toth & Lake, J. Phys.: Condens. Matter 27, 166002 (2015),
Eq. 25/26; convention H = -sum_<ij> J S_i.S_j, unordered bonds once, J in meV,
|S| explicit):

    FM chain      omega(k) = 2 J S (1 - cos ka)
    FM honeycomb  omega_pm(k) = 3 J S (1 -/+ |gamma(k)|)

Run against the throwaway patched copy of the package:

    PYTHONPATH=<...>/scratch/lswt-audit/patched python -m pytest \
        scratch/lswt-audit/test_magnon3_convention.py -q
"""

import numpy as np
import pytest

TOL = 1e-8  # meV


def _fm_chain(S, jiso):
    """FM chain (1 site/cell, bonds +-a) with exchange jiso (meV)."""
    from TB2J.magnon.magnon3 import Magnon

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
        cell=np.eye(3),
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


def _fm_honeycomb(S, jiso):
    """FM honeycomb (2 sites/cell) with exchange jiso (meV)."""
    from TB2J.magnon.magnon3 import Magnon

    a = 1.0
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


def _exact_energies(M, kpoints):
    """Magnon energies (meV) from the exact Colpa route (eigenvalues of g H).

    The shifted-Cholesky route used by `Magnon._magnon_energies` is only
    ~1e-6 accurate and returns negative frequencies at a Goldstone point, so
    the regressions use the exact route.
    """
    H = M.Hq(kpoints)
    n = H.shape[-1] // 2
    g = np.block([[np.eye(n), np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
    ev = np.linalg.eigvals(g @ H[0])
    pos = [e.real for e in ev if e.real > -1e-9 and abs(e.imag) < 1e-8]
    return np.sort(np.array(pos)) * 1e3  # eV -> meV


@pytest.mark.parametrize("kv", [0.05, 0.1, 0.25, 0.4, 0.5])
def test_fm_chain_matches_textbook(kv):
    S, jiso = 2.0, -1.0  # meV
    w = _exact_energies(_fm_chain(S, jiso * 1e-3), np.array([[kv, 0, 0]]))
    tb = 2 * jiso * S * (1 - np.cos(2 * np.pi * kv))
    assert abs(float(w[0]) - abs(tb)) < TOL


@pytest.mark.parametrize(
    "kv", [(0.2, 0.13), (1 / 3, 2 / 3), (0.5, 0.0), (0.25, 0.25)]
)
def test_fm_honeycomb_matches_textbook(kv):
    S, jiso = 1.5, -1.0  # meV
    w = _exact_energies(_fm_honeycomb(S, jiso * 1e-3), np.array([[kv[0], kv[1], 0.0]]))
    gam = abs(1 + np.exp(2j * np.pi * kv[0]) + np.exp(2j * np.pi * kv[1])) / 3
    tb = np.sort(
        np.abs(np.array([3 * jiso * S * (1 - gam), 3 * jiso * S * (1 + gam)]))
    )
    assert np.max(np.abs(w - tb)) < TOL


def test_goldstone_mode_fm_chain():
    w = _exact_energies(_fm_chain(2.0, -1e-3), np.array([[0.0, 0, 0]]))
    assert abs(float(w[0])) < 1e-9


def test_goldstone_mode_fm_honeycomb():
    w = _exact_energies(_fm_honeycomb(1.5, -1e-3), np.array([[0.0, 0, 0]]))
    assert abs(float(w[0])) < 1e-9


def test_guard_rejects_non_uniform_reference_with_q():
    """Q != 0 with more than one magnetic site (non-uniform reference) fails
    loudly instead of silently double counting the rotating-frame exchange."""
    M = _fm_honeycomb(1.5, -1e-3)
    with pytest.raises(ValueError):
        M.set_reference(
            np.array([0.5, 0.0, 0.0]),
            np.array([[0.0, 0, 1.0]] * 2),
            np.array([1.0, 0, 0]),
            magmoms=np.array([[0.0, 0, 3.0], [0.0, 0, 1.5]]),
        )
