"""Certified spinor handoff tests for the optional TBUpy interface."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh
from tbupy.io import load_scf_result
from tbupy.model import TBUpyModel
from tbupy.util import build_simple_tb

from TB2J.interfaces.tbupy_interface import prepare_tbupy_inputs
from TB2J.pauli import pauli_block_all

KPTS = np.zeros((1, 3))
KWEIGHTS = np.ones(1)


def _saved_spinor_result(tmp_path):
    ham = build_simple_tb(
        cell=np.eye(3) * 5.0,
        positions=np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]),
        symbols=["Fe", "Fe"],
        orbital_dict={"Fe": ["s"]},
        onsite={"Fe": 0.0},
        hopping=[(0.3, 0, 1, np.array([0.0, 0.0, 0.0]))],
        nspin=2,
    )
    model = TBUpyModel(
        ham,
        {"Fe": {"U": 0.1, "J": 0.0, "L": 0}},
        hubbard_type="kanamori",
        dc_type="FLL-ns",
        mixing="pulay",
        mixing_beta=0.5,
        spin_mode="spinor",
    )
    field = np.zeros((4, 4), dtype=complex)
    field[0, 1] = field[1, 0] = -0.04
    scf = model.run_scf(
        KPTS,
        KWEIGHTS,
        nel=3.0,
        width=0.05,
        max_iter=400,
        tol_energy=1e-12,
        tol_rho=1e-11,
        external_potential=field,
    )
    assert scf.converged and scf.energy_certified
    path = tmp_path / "certified_spinor.tbupy.nc"
    model.save_scf_result(path, scf)
    return load_scf_result(path), scf


def test_spinor_handoff_preserves_spectrum_pauli_and_orbital_ownership(tmp_path):
    result, scf = _saved_spinor_result(tmp_path)
    atoms, tbmodel, basis, efermi = prepare_tbupy_inputs(
        tbupy_result=result, colinear=False
    )

    assert tbmodel is result.hamiltonian
    assert atoms.get_chemical_symbols() == ["Fe", "Fe"]
    assert efermi == pytest.approx(result.efermi)
    assert basis == list(tbmodel.orbs)
    assert [orb.iatom for orb in basis] == [0, 0, 1, 1]

    Hk, Sk = tbmodel.gen_ham(KPTS[0])
    np.testing.assert_allclose(
        eigh(Hk, Sk, eigvals_only=True), result.evals[0], atol=1e-10
    )
    pauli = pauli_block_all(result.rho)
    assert abs(pauli[1, 0, 0]) > 1e-4
    assert result.metadata["energy_certified"]
    np.testing.assert_allclose(result.evecs, scf.evecs)


def test_spinor_handoff_refuses_uncertified_and_tampered_state(tmp_path):
    result, _ = _saved_spinor_result(tmp_path)
    result.metadata["energy_certified"] = False
    result.metadata["certification"]["certified"] = False
    with pytest.raises(ValueError, match="not energy-certified"):
        prepare_tbupy_inputs(tbupy_result=result, colinear=False)

    result, _ = _saved_spinor_result(tmp_path)
    result.metadata["interaction_tensor_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        prepare_tbupy_inputs(tbupy_result=result, colinear=False)


def test_spinor_handoff_refuses_unowned_basis_or_models(tmp_path):
    result, _ = _saved_spinor_result(tmp_path)
    with pytest.raises(ValueError, match="basis ownership"):
        prepare_tbupy_inputs(tbupy_result=result, basis=[], colinear=False)
    with pytest.raises(ValueError, match="certified TBUpy result"):
        prepare_tbupy_inputs(tbmodels=result.hamiltonian, colinear=False)
