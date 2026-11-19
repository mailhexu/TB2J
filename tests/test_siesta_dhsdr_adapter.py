"""Adapter tests: LocalizedDisplacementDerivative.from_siesta_dhsdr (story 005)."""

from __future__ import annotations

import numpy as np
import pytest
from HamiltonIO.siesta.dhsdr import (
    BOHR_TO_ANG,
    RY_PER_BOHR_TO_EV_PER_ANG,
    DisplacementBlock,
)

from TB2J.spin_lattice_lcao import LocalizedDisplacementDerivative


def _chain_block(dt, ds):
    """DisplacementBlock with the 1D-chain pattern: 1 orbital/atom, bonds
    to cells +-1 carry dt (Ry/Bohr) and ds (1/Bohr); cell 0 is zero."""
    dH = {c: np.array([[[dt if c in (0, 2) else 0.0]]]) for c in range(3)}
    dS = {c: np.array([[ds if c in (0, 2) else 0.0]]) for c in range(3)}
    return DisplacementBlock(
        atom_index=1,
        axis=2,  # y
        nsc=(3, 1, 1),
        isc_off=np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dH=dH,
        dS=dS,
    )


def test_from_siesta_dhsdr_chain_vertex_keys():
    """The dHSdR block maps to the same fundamental-domain keys as the
    FD-validated analytic vertex: ((0,0,0), R) for R in {0, +-1}, Ru=0."""
    lam, t0, s0 = 1.3, -1.0, 0.2
    dt, ds = -lam * t0, -lam * s0
    disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(_chain_block(dt, ds))
    assert disp.Ru == (0, 0, 0)
    assert disp.direction == (0, 1, 0)
    assert disp.atom_index == 1
    assert "siesta dHSdR.nc" in disp.provenance
    # cell 0 is all-zero and dropped by block_to_R_lists
    assert set(disp.dH) == {((0, 0, 0), (-1, 0, 0)), ((0, 0, 0), (1, 0, 0))}
    # Ry/Bohr -> eV/Ang, 1/Bohr -> 1/Ang
    assert disp.dH[((0, 0, 0), (1, 0, 0))][0, 0] == pytest.approx(
        dt * RY_PER_BOHR_TO_EV_PER_ANG, rel=1e-12
    )
    assert disp.dS[((0, 0, 0), (-1, 0, 0))][0, 0] == pytest.approx(
        ds / BOHR_TO_ANG, rel=1e-12
    )


def test_from_siesta_dhsdr_spinor_interleave():
    """Collinear two-spin blocks interleave into the spinor layout with dS
    equal in both spin channels."""
    up, dn = 0.3, -0.2
    dH = {
        c: np.array([[[up if c == 2 else 0.0, dn if c == 2 else 0.0]]])
        for c in range(3)
    }
    dS = {c: np.array([[0.5 if c == 2 else 0.0]]) for c in range(3)}
    b = DisplacementBlock(
        atom_index=1,
        axis=1,
        nsc=(3, 1, 1),
        isc_off=np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dH=dH,
        dS=dS,
    )
    disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(b)
    key = ((0, 0, 0), (1, 0, 0))
    mat = disp.dH[key]
    assert mat.shape == (2, 2)
    assert mat[0, 0] == pytest.approx(up * RY_PER_BOHR_TO_EV_PER_ANG)
    assert mat[1, 1] == pytest.approx(dn * RY_PER_BOHR_TO_EV_PER_ANG)
    assert mat[0, 1] == 0 and mat[1, 0] == 0
    smat = disp.dS[key]
    np.testing.assert_allclose(smat[::2, ::2], smat[1::2, 1::2])


def test_from_siesta_dhsdr_rejects_empty_and_bad_spin():
    with pytest.raises(ValueError, match="empty"):
        LocalizedDisplacementDerivative.from_siesta_dhsdr(_chain_block(0.0, 0.0))
    dH = {c: np.zeros((1, 1, 4)) for c in range(3)}
    dH[2][0, 0, :] = 1.0
    b = DisplacementBlock(
        atom_index=1,
        axis=1,
        nsc=(3, 1, 1),
        isc_off=np.array([[-1, 0, 1], [0, 0, 0], [0, 0, 0]]),
        dH=dH,
        dS={c: np.zeros((1, 1)) for c in range(3)},
    )
    with pytest.raises(ValueError, match="spin dimension"):
        LocalizedDisplacementDerivative.from_siesta_dhsdr(b)


def test_reference_gauge_preserves_the_resolvent_derivative():
    """Raw FC and Fermi-shifted model describe the same pencil response."""
    from ase.units import Ry

    h0 = np.diag([-1.2, 0.7])
    s0 = np.eye(2) * 1.1
    dh = np.diag([0.3, -0.1])
    ds = np.eye(2) * 0.07
    block = DisplacementBlock(
        1,
        1,
        (1, 1, 1),
        np.zeros((3, 1), int),
        {0: np.array([[[0.3, -0.1]]]) * BOHR_TO_ANG / Ry},
        {0: np.array([[0.07]]) * BOHR_TO_ANG},
    )
    ef = -5.0
    disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(block, reference_fermi=ef)
    z = -0.4 + 0.8j
    g = np.linalg.inv(z * s0 - h0)
    analytic = g @ disp.to_vertex(2).vertex((0, 0, 0), (0, 0, 0), z - ef) @ g
    step = 1e-5

    def resolvent(t):
        return np.linalg.inv(z * (s0 + t * ds) - (h0 + t * dh))

    fd = (resolvent(step) - resolvent(-step)) / (2 * step)
    np.testing.assert_allclose(analytic, fd, atol=1e-9, rtol=1e-7)
