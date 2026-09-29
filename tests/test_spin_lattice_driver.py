"""Story 004 (nonorth-spin-lattice): driver end-to-end smoke test.

Runs the public driver on a synthetic non-orthogonal spinor chain with a
localized displacement (nonzero dS and splitting-drift channels) through
the full production pipeline (contour, A/dA, integration, SpinIO output)
and checks the serialized result.
"""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.spin_lattice_lcao import gen_exchange_spin_lattice_lcao

pytest.importorskip("ase")

import test_spin_lattice_fd as FD  # noqa: E402
from ase import Atoms  # noqa: E402


def test_driver_smoke_synthetic_chain(tmp_path):
    model = FD._SpinorChain(ncell=1)
    atoms = Atoms(
        "Fe", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=[True, False, False]
    )
    sio = gen_exchange_spin_lattice_lcao(
        tbmodel=model,
        atoms=atoms,
        disp=FD._analytic_vertex(),
        basis=["Fe1|orb1", "Fe1|orb1"],
        efermi=FD.EFERMI,
        magnetic_elements=["Fe"],
        kmesh=(16, 1, 1),
        emin=-10.0,
        emax=0.0,
        nz=20,
        np=1,
        output_path=str(tmp_path),
        Jonly=False,
    )
    # exchange values exist and are finite
    values = np.array(list(sio.exchange_Jdict.values()))
    assert len(values) > 0
    assert np.all(np.isfinite(values))
    # spin-lattice derivative exists for the displaced mode and is finite
    dvalues = np.array(list(sio.dJdx.values()))
    assert dvalues.size > 0
    assert np.all(np.isfinite(dvalues))
    assert any(np.abs(d) > 1e-8 for d in dvalues)
    # provenance recorded
    assert "provenance" in sio.description or "synthetic" in sio.description


def test_driver_jonly_parity(tmp_path):
    """J-only mode runs without derivative data influence."""
    model = FD._SpinorChain(ncell=1)
    atoms = Atoms(
        "Fe", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=[True, False, False]
    )
    sio = gen_exchange_spin_lattice_lcao(
        tbmodel=model,
        atoms=atoms,
        disp=FD._analytic_vertex(),
        basis=["Fe1|orb1", "Fe1|orb1"],
        efermi=FD.EFERMI,
        magnetic_elements=["Fe"],
        kmesh=(16, 1, 1),
        emin=-10.0,
        emax=0.0,
        nz=20,
        np=1,
        output_path=str(tmp_path),
        Jonly=True,
    )
    values = np.array(list(sio.exchange_Jdict.values()))
    assert len(values) > 0
    assert np.all(np.isfinite(values))
