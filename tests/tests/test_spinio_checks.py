"""Unit tests for the SpinIO validation helpers (story 010-1).

These build synthetic ``SpinIO`` objects directly — no DFT data — and verify
each helper passes on well-formed input and fails on a plausible bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from utils.spinio_checks import (
    check_dmi_antisymmetry,
    check_exchange_out_section,
    check_jani_hermiticity,
    check_pair_reversal,
    check_schema,
    compare_J,
)

from TB2J.io_exchange.io_exchange import SpinIO


def _make_sio(**overrides):
    """Build a minimal SpinIO and let tests override its exchange dicts."""
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[3, 3, 3], pbc=True)
    sio = SpinIO(
        atoms=atoms,
        spinat=np.array([[0, 0, 2.5], [0, 0, -2.5]]),
        charges=np.array([0.0, 0.0]),
        index_spin=[0, 1],
    )
    sio.TB2J_version = "0.9.18"
    sio.exchange_Jdict = overrides.pop("exchange_Jdict", {})
    sio.dmi_ddict = overrides.pop("dmi_ddict", {})
    sio.Jani_dict = overrides.pop("Jani_dict", {})
    return sio


# Symmetric J: J[(R,i,j)] == J[(-R,j,i)] (stored in eV here)
_SYM_J = {
    ((0, 0, 1), 0, 1): 0.0125,
    ((0, 0, -1), 1, 0): 0.0125,
    ((1, 0, 0), 0, 1): 0.0080,
    ((-1, 0, 0), 1, 0): 0.0080,
}


class TestCheckSchema:
    def test_passes_on_well_formed(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        check_schema(sio)  # no exception

    def test_fails_on_missing_version(self):
        sio = _make_sio()
        del sio.TB2J_version
        with pytest.raises(AssertionError, match="TB2J_version"):
            check_schema(sio)

    def test_fails_on_non_finite_J(self):
        sio = _make_sio(exchange_Jdict={((0, 0, 1), 0, 1): float("nan")})
        with pytest.raises(AssertionError, match="non-finite"):
            check_schema(sio)

    def test_fails_on_none_dict(self):
        sio = _make_sio()
        sio.exchange_Jdict = None
        with pytest.raises(AssertionError):
            check_schema(sio)


class TestCompareJ:
    def test_passes_within_tolerance(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        # expected given in meV; stored is eV -> 12.5 meV == 0.0125 eV
        compare_J(sio, {((0, 0, 1), 0, 1): 12.5}, tol=1e-3, unit="meV")

    def test_fails_out_of_tolerance_naming_key(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        with pytest.raises(AssertionError, match=r"\(0, 0, 1\), 0, 1"):
            compare_J(sio, {((0, 0, 1), 0, 1): 99.0}, tol=1e-3, unit="meV")

    def test_fails_on_missing_key(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        with pytest.raises(AssertionError, match="not in exchange_Jdict"):
            compare_J(sio, {((9, 9, 9), 0, 1): 1.0}, unit="meV")

    def test_eV_unit_compares_directly(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        compare_J(sio, {((0, 0, 1), 0, 1): 0.0125}, tol=1e-9, unit="eV")


class TestCheckPairReversal:
    def test_passes_symmetric(self):
        sio = _make_sio(exchange_Jdict=dict(_SYM_J))
        check_pair_reversal(sio)

    def test_fails_asymmetric(self):
        bad = dict(_SYM_J)
        bad[((0, 0, 1), 0, 1)] = 0.0200  # no longer matches (-R,j,i)
        sio = _make_sio(exchange_Jdict=bad)
        with pytest.raises(AssertionError, match="pair reversal"):
            check_pair_reversal(sio)

    def test_fails_on_missing_reverse_key(self):
        only_one = {((0, 0, 1), 0, 1): 0.0125}
        sio = _make_sio(exchange_Jdict=only_one)
        with pytest.raises(AssertionError, match="missing"):
            check_pair_reversal(sio)


class TestCheckDmiAntisymmetry:
    def test_passes_antisymmetric(self):
        dmi = {
            ((0, 0, 1), 0, 1): np.array([0.1, -0.2, 0.0]),
            ((0, 0, -1), 1, 0): np.array([-0.1, 0.2, 0.0]),
        }
        sio = _make_sio(dmi_ddict=dmi)
        check_dmi_antisymmetry(sio)

    def test_fails_symmetric(self):
        dmi = {
            ((0, 0, 1), 0, 1): np.array([0.1, -0.2, 0.0]),
            ((0, 0, -1), 1, 0): np.array([0.1, -0.2, 0.0]),
        }
        sio = _make_sio(dmi_ddict=dmi)
        with pytest.raises(AssertionError, match="DMI antisymmetry"):
            check_dmi_antisymmetry(sio)


class TestCheckJaniHermiticity:
    def test_passes_hermitian(self):
        M = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, -0.1], [0.0, -0.1, 3.0]])
        sio = _make_sio(Jani_dict={((0, 0, 1), 0, 1): M})
        check_jani_hermiticity(sio)

    def test_fails_non_hermitian(self):
        M = np.array([[1.0, 0.9, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 3.0]])
        sio = _make_sio(Jani_dict={((0, 0, 1), 0, 1): M})
        with pytest.raises(AssertionError, match="Hermitian"):
            check_jani_hermiticity(sio)

    def test_skips_when_empty(self):
        sio = _make_sio(Jani_dict={})
        check_jani_hermiticity(sio)  # SOC case with no anisotropy -> ok


class TestCheckExchangeOutSection:
    def test_passes_when_section_present(self, tmp_path):
        f = tmp_path / "exchange.out"
        f.write_text("header\nWannier90 WS interpolation: scheme 1.\nbody\n")
        check_exchange_out_section(f, "Wannier90 WS interpolation")

    def test_fails_when_absent(self, tmp_path):
        f = tmp_path / "exchange.out"
        f.write_text("header\nbody\n")
        with pytest.raises(AssertionError, match="section.*not found"):
            check_exchange_out_section(f, "Wannier90 WS interpolation")
