"""Tests for the thermal-magnon foundation (story 014).

Covers ThermalMagnonParameters validation/TOML round trip, the
``tb2j.magnon.thermal`` result schema (JSON and NetCDF round trips), and
ThermalSpinModel validation/rejection rules.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase.units import kB as KB

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.thermal_model import (
    ThermalModelValidationError,
    build_thermal_spin_model,
)
from TB2J.magnon.thermal_parameters import (
    ThermalMagnonParameters,
    add_thermal_args,
    thermal_parameters_from_args,
)
from TB2J.magnon.thermal_result import (
    MeshHistoryEntry,
    ThermalBandBlock,
    ThermalMagnonResult,
    TransitionRecord,
)

# ----------------------------------------------------------------------------
# Fixtures: synthetic magnon models in TB2J conventions
# ----------------------------------------------------------------------------


def _identity_cell(a=1.0):
    return np.eye(3) * a


def _finalize(mag):
    """Set the default collinear-z reference (sets Snorm like production use)."""
    mag.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 0.0, 1.0])
    )
    return mag


def simple_cubic_fm_magnon(J=0.05, S=1.0, sia=0.0, lam=0.0):
    """One-site simple-cubic FM: 6 nearest neighbours, J > 0 (eV).

    Storage convention (docs/sympy/03 Section 7): each directed R entry of
    JR holds one half of the physical paper coupling scaled by S^2, so the
    extracted paper quantities satisfy J0 = 6 J and lambda_0 = 6 lam.
    """
    R = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float,
    )
    nR = len(R)
    JR = np.zeros((nR, 1, 1, 3, 3))
    half = 0.5 * S * S
    for iR in range(1, nR):
        JR[iR, 0, 0] = half * np.diag([J, J, J + lam])
    if sia != 0.0:
        # on-site SIA tensor: TB2J stores k1 = A * S^2 in JR_zz(0)
        JR[0, 0, 0] = np.diag([0.0, 0.0, sia * S * S])
    magmom = np.array([[0.0, 0.0, 2.0 * S]])
    return _finalize(
        Magnon(
            nspin=1,
            magmom=magmom,
            Rlist=R,
            JR=JR,
            cell=_identity_cell(),
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
        )
    )


def two_site_fm_magnon(J=0.05, S=1.0):
    """Two-site FM chain cell with both directed entries per bond stored."""
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
    JR = np.zeros((3, 2, 2, 3, 3))
    half = 0.5 * S * S
    # bonds: A0-B0 and A0-B_{-1}, stored in both directions at half weight
    JR[0, 0, 1] = half * np.eye(3) * J
    JR[0, 1, 0] = half * np.eye(3) * J
    JR[1, 1, 0] = half * np.eye(3) * J
    JR[2, 0, 1] = half * np.eye(3) * J
    magmom = np.tile([0.0, 0.0, 2.0 * S], (2, 1))
    return _finalize(
        Magnon(
            nspin=2,
            magmom=magmom,
            Rlist=R,
            JR=JR,
            cell=_identity_cell(),
            _Q=np.zeros(3),
            _uz=np.tile([0.0, 0.0, 1.0], (2, 1)),
            _n=np.array([0.0, 0.0, 1.0]),
        )
    )


def dmi_fm_magnon(J=0.05, D=0.01, S=1.0):
    """FM with a DMI tensor entry (must be rejected by the thermal model)."""
    mag = simple_cubic_fm_magnon(J=J, S=S)
    Dten = np.array([[0.0, D, 0.0], [-D, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for iR in range(1, len(mag.Rlist)):
        mag.JR[iR, 0, 0] += Dten
    return mag


def noncollinear_magnon(S=1.0):
    """Two-site canted state (must be rejected)."""
    mag = two_site_fm_magnon(S=S)
    mag.magmom[1] = [2.0 * S, 0.0, 0.0]
    return mag


def inequivalent_spin_magnon():
    """Two-site FM with different spin lengths (must be rejected)."""
    mag = two_site_fm_magnon(S=1.0)
    mag.magmom[1] = [0.0, 0.0, 3.0]
    mag.Snorm = np.linalg.norm(mag.magmom, axis=1) / 2
    return mag


def default_params(**kwargs):
    base = dict(
        thermal_method="rpa",
        thermal_spin_regime="quantum",
        thermal_order_mode="ferromagnetic",
        thermal_dimensionality=3,
        thermal_temperatures=[100.0, 200.0],
        thermal_qmeshes=[[6, 6, 6], [8, 8, 8]],
        thermal_mesh_tolerance=1.0,
    )
    base.update(kwargs)
    return ThermalMagnonParameters(**base)


# ----------------------------------------------------------------------------
# ThermalMagnonParameters
# ----------------------------------------------------------------------------


class TestThermalMagnonParameters:
    def test_defaults_and_validation(self):
        p = default_params()
        assert p.thermal_method == "rpa"
        assert p.thermal_dimensionality == 3

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="thermal_method"):
            default_params(thermal_method="tyablikov")

    def test_invalid_regime_rejected(self):
        with pytest.raises(ValueError, match="thermal_spin_regime"):
            default_params(thermal_spin_regime="semiclassical")

    def test_invalid_order_mode_rejected(self):
        with pytest.raises(ValueError, match="thermal_order_mode"):
            default_params(thermal_order_mode="spiral")

    def test_invalid_dimensionality_rejected(self):
        with pytest.raises(ValueError, match="thermal_dimensionality"):
            default_params(thermal_dimensionality=4)
        with pytest.raises(ValueError, match="thermal_dimensionality"):
            default_params(thermal_dimensionality=0)

    def test_negative_temperature_rejected(self):
        with pytest.raises(ValueError, match="thermal_temperatures"):
            default_params(thermal_temperatures=[100.0, -5.0])

    def test_spin_override_length_checked_at_model_level(self):
        mag = two_site_fm_magnon()
        with pytest.raises(ThermalModelValidationError, match="thermal_spin"):
            build_thermal_spin_model(mag, default_params(thermal_spin=[1.0]))

    def test_spin_override_nonpositive_rejected(self):
        with pytest.raises(ValueError, match="thermal_spin"):
            default_params(thermal_spin=[1.0, -0.5])

    def test_strict_flag(self):
        p = default_params(thermal_strict=True)
        assert p.thermal_strict

    def test_solver_controls(self):
        p = default_params(
            thermal_max_iterations=500, thermal_mixing=0.3, thermal_tolerance=1e-10
        )
        assert p.thermal_max_iterations == 500
        with pytest.raises(ValueError, match="thermal_mixing"):
            default_params(thermal_mixing=1.5)

    def test_toml_roundtrip(self, tmp_path):
        p = default_params(thermal_spin=[1.0])
        fname = tmp_path / "thermal.toml"
        p.to_toml(str(fname))
        q = ThermalMagnonParameters.from_toml(str(fname))
        assert q.thermal_method == p.thermal_method
        assert q.thermal_qmeshes == p.thermal_qmeshes
        assert q.thermal_temperatures == p.thermal_temperatures
        assert q.thermal_spin == p.thermal_spin

    def test_cli_args_roundtrip(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_thermal_args(parser)
        args = parser.parse_args(
            [
                "--thermal-method",
                "callen",
                "--thermal-spin-regime",
                "classical",
                "--thermal-order-mode",
                "bipartite_afm",
                "--thermal-dimensionality",
                "2",
                "--thermal-temperatures",
                "50,100",
                "--thermal-qmeshes",
                "6x6x1,8x8x1",
                "--thermal-mesh-tolerance",
                "0.5",
                "--thermal-strict",
                "--thermal-spin",
                "1.5,1.5",
            ]
        )
        p = thermal_parameters_from_args(args)
        assert p.thermal_method == "callen"
        assert p.thermal_spin_regime == "classical"
        assert p.thermal_order_mode == "bipartite_afm"
        assert p.thermal_dimensionality == 2
        assert p.thermal_temperatures == [50.0, 100.0]
        assert p.thermal_qmeshes == [[6, 6, 1], [8, 8, 1]]
        assert p.thermal_mesh_tolerance == 0.5
        assert p.thermal_strict
        assert p.thermal_spin == [1.5, 1.5]


# ----------------------------------------------------------------------------
# ThermalSpinModel validation
# ----------------------------------------------------------------------------


class TestThermalSpinModel:
    def test_fm_single_site_builds(self):
        mag = simple_cubic_fm_magnon()
        model = build_thermal_spin_model(mag, default_params())
        assert model.order_mode == "ferromagnetic"
        assert model.dimensionality == 3
        assert model.nspin == 1
        assert np.isclose(model.S[0], 1.0)
        assert model.spin_interpretation == "physical_quantum_spin"

    def test_fm_multisite_builds(self):
        mag = two_site_fm_magnon()
        model = build_thermal_spin_model(mag, default_params())
        assert model.nspin == 2
        assert np.allclose(model.S, 1.0)

    def test_effective_spin_interpretation(self):
        mag = simple_cubic_fm_magnon(S=1.3)  # non half-integer moment
        model = build_thermal_spin_model(mag, default_params())
        assert model.spin_interpretation == "effective_quantum_spin"
        assert np.isclose(model.S[0], 1.3)

    def test_spin_override_applied(self):
        mag = simple_cubic_fm_magnon(S=1.3)
        model = build_thermal_spin_model(mag, default_params(thermal_spin=[1.5]))
        assert np.isclose(model.S[0], 1.5)
        assert model.spin_interpretation == "physical_quantum_spin"

    def test_dmi_rejected(self):
        mag = dmi_fm_magnon()
        with pytest.raises(ThermalModelValidationError, match="DMI"):
            build_thermal_spin_model(mag, default_params())

    def test_noncollinear_rejected(self):
        mag = noncollinear_magnon()
        with pytest.raises(ThermalModelValidationError, match="collinear"):
            build_thermal_spin_model(mag, default_params())

    def test_inequivalent_spins_rejected(self):
        mag = inequivalent_spin_magnon()
        with pytest.raises(ThermalModelValidationError, match="equivalent"):
            build_thermal_spin_model(mag, default_params())

    def test_afm_mode_requires_antiparallel(self):
        mag = simple_cubic_fm_magnon()
        with pytest.raises(ThermalModelValidationError, match="antiparallel"):
            build_thermal_spin_model(
                mag, default_params(thermal_order_mode="bipartite_afm")
            )

    def test_sia_extracted(self):
        A = 0.002
        mag = simple_cubic_fm_magnon(J=0.05, S=1.0, sia=A)
        model = build_thermal_spin_model(mag, default_params())
        assert np.isclose(model.A[0], A, atol=1e-12)

    def test_lambda_extracted(self):
        lam = 0.001
        mag = simple_cubic_fm_magnon(J=0.05, S=1.0, lam=lam)
        model = build_thermal_spin_model(mag, default_params())
        # lambda_q = sum_R lambda(R) e^{i q R}; at Gamma: 6 * lam
        assert np.isclose(model.lambda_q(np.zeros((1, 3)))[0, 0, 0].real, 6 * lam)

    def test_transverse_anisotropy_rejected(self):
        mag = simple_cubic_fm_magnon()
        # break xx = yy
        for iR in range(1, len(mag.Rlist)):
            mag.JR[iR, 0, 0] = np.diag([0.05, 0.06, 0.05])
        with pytest.raises(ThermalModelValidationError, match="transverse"):
            build_thermal_spin_model(mag, default_params())


# ----------------------------------------------------------------------------
# ThermalMagnonResult schema
# ----------------------------------------------------------------------------


def sample_result():
    band = ThermalBandBlock(
        temperature_K=100.0,
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        energies_eV=np.array([[0.0, 0.01], [0.02, 0.03]]),
        order_parameters=np.array([0.9, 0.9]),
        status="ordered",
    )
    transition = TransitionRecord(
        kind="curie_temperature",
        temperature_K=300.0,
        converged=True,
        method_validity="nominal",
    )
    history = [
        MeshHistoryEntry(
            qmesh=[6, 6, 6],
            estimate_K=298.0,
            residual=2.0,
            iterations=42,
            min_energy_eV=1e-5,
            status="refined",
        ),
        MeshHistoryEntry(
            qmesh=[8, 8, 8],
            estimate_K=299.0,
            residual=1.0,
            iterations=45,
            min_energy_eV=1e-5,
            status="converged",
        ),
    ]
    return ThermalMagnonResult(
        method="rpa",
        spin_regime="quantum",
        spin_interpretation="physical_quantum_spin",
        spins=[1.0],
        order_mode="ferromagnetic",
        dimensionality=3,
        status="ok",
        transition=transition,
        mesh_history=history,
        bands=[band],
    )


class TestThermalMagnonResult:
    def test_schema_name_and_version(self):
        r = sample_result()
        assert r.schema_name == "tb2j.magnon.thermal"
        assert r.schema_version == "1.0"

    def test_json_roundtrip(self, tmp_path):
        r = sample_result()
        fname = tmp_path / "thermal.json"
        r.save_json(str(fname))
        loaded = ThermalMagnonResult.load_json(str(fname))
        assert loaded.method == "rpa"
        assert loaded.transition.temperature_K == 300.0
        assert len(loaded.mesh_history) == 2
        assert loaded.mesh_history[0].qmesh == [6, 6, 6]
        assert len(loaded.bands) == 1
        assert loaded.bands[0].temperature_K == 100.0
        assert np.allclose(loaded.bands[0].energies_eV, r.bands[0].energies_eV)
        assert loaded.bands[0].status == "ordered"

    def test_json_rejects_foreign_schema(self, tmp_path):
        r = sample_result()
        fname = tmp_path / "thermal.json"
        r.save_json(str(fname))
        with open(fname) as f:
            data = json.load(f)
        data["schema_name"] = "tb2j.magnon.eigenstates"
        with open(fname, "w") as f:
            json.dump(data, f)
        with pytest.raises(ValueError, match="schema"):
            ThermalMagnonResult.load_json(str(fname))

    def test_netcdf_roundtrip(self, tmp_path):
        pytest.importorskip("netCDF4")
        r = sample_result()
        fname = tmp_path / "thermal.nc"
        r.save_netcdf(str(fname))
        loaded = ThermalMagnonResult.load_netcdf(str(fname))
        assert loaded.method == "rpa"
        assert loaded.transition.kind == "curie_temperature"
        assert loaded.bands[0].kpoints.shape == (2, 3)
        assert np.allclose(loaded.bands[0].order_parameters, [0.9, 0.9])

    def test_zero_transition_status(self):
        band = ThermalBandBlock(
            temperature_K=10.0,
            kpoints=np.zeros((1, 3)),
            energies_eV=np.zeros((1, 1)),
            order_parameters=np.array([1.0]),
            status="ordered",
        )
        r = ThermalMagnonResult(
            method="rpa",
            spin_regime="quantum",
            spin_interpretation="physical_quantum_spin",
            spins=[0.5],
            order_mode="ferromagnetic",
            dimensionality=2,
            status="zero_transition",
            transition=TransitionRecord(
                kind="curie_temperature",
                temperature_K=0.0,
                converged=True,
                method_validity="nominal",
                detail="isotropic 2D: Mermin-Wagner zero transition",
            ),
            mesh_history=[],
            bands=[band],
        )
        assert r.status == "zero_transition"

    def test_hp_breakdown_transition_kind(self):
        r = sample_result()
        r.transition = TransitionRecord(
            kind="temperature_hp_breakdown",
            temperature_K=250.0,
            converged=True,
            method_validity="nominal",
            breakdown_magnetization=0.15,
        )
        assert r.transition.kind == "temperature_hp_breakdown"
        assert r.transition.breakdown_magnetization == 0.15


# ----------------------------------------------------------------------------
# Story 015: FM RPA temperature bands and Curie temperature
# ----------------------------------------------------------------------------


def sc_chain_fm_magnon(J=0.05, S=1.0):
    """1D FM chain (isotropic): Rlist +-x only."""
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=float)
    JR = np.zeros((3, 1, 1, 3, 3))
    half = 0.5 * S * S
    JR[1, 0, 0] = half * np.eye(3) * J
    JR[2, 0, 0] = half * np.eye(3) * J
    mag = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0.0, 2.0 * S]]),
        Rlist=R,
        JR=JR,
        cell=_identity_cell(),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 0.0, 1.0]),
    )
    return _finalize(mag)


def sc_square_fm_magnon(J=0.05, S=1.0):
    """2D FM square lattice (isotropic)."""
    R = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)
    JR = np.zeros((5, 1, 1, 3, 3))
    half = 0.5 * S * S
    for iR in range(1, 5):
        JR[iR, 0, 0] = half * np.eye(3) * J
    mag = Magnon(
        nspin=1,
        magmom=np.array([[0.0, 0.0, 2.0 * S]]),
        Rlist=R,
        JR=JR,
        cell=_identity_cell(),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 0.0, 1.0]),
    )
    return _finalize(mag)


class TestFmRpaSolver:
    def test_t0_bands_match_magnon3(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon(J=0.05, S=1.0)
        solver = ThermalMagnonSolver(mag, default_params())
        result = solver.calculate(
            temperatures_K=[0.0],
            band_kpoints=np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]),
        )
        assert result.status == "ok"
        band = result.bands[0]
        ref = np.sort(mag._magnon_energies(band.kpoints), axis=1)
        assert np.allclose(band.energies_eV, ref, atol=2e-6)

    def test_temperature_softens_bands(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon(J=0.05, S=1.0)
        solver = ThermalMagnonSolver(mag, default_params())
        kp = np.array([[0.5, 0.0, 0.0]])
        r = solver.calculate(temperatures_K=[0.0, 200.0], band_kpoints=kp)
        assert r.bands[1].energies_eV[0, 0] < r.bands[0].energies_eV[0, 0]
        assert r.bands[1].order_parameters[0] < r.bands[0].order_parameters[0]
        assert np.isclose(r.bands[0].order_parameters[0], 1.0)

    def test_sc_rpa_tc_watson_reference(self):
        """Tc = S(S+1)/3 * [mean 1/(J0-Jq)]^-1 with the Watson/Joyce sum."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J = 0.05
        S = 0.5
        mag = simple_cubic_fm_magnon(J=J, S=S)
        p = default_params(
            thermal_qmeshes=[[32, 32, 32], [48, 48, 48]],
            thermal_mesh_tolerance=20.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        # continuum closed form: kBTc = S(S+1) * 1.318926 * J
        ref = S * (S + 1) * 1.318926 * J
        assert r.transition.converged
        assert abs(r.transition.temperature_K * 8.617333262e-5 - ref) / ref < 0.02

    def test_1d_zero_transition(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = sc_chain_fm_magnon()
        p = default_params(
            thermal_dimensionality=1,
            thermal_qmeshes=[[16, 1, 1], [32, 1, 1]],
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate(
            temperatures_K=[10.0], band_kpoints=np.array([[0.5, 0, 0]])
        )
        assert r.status == "zero_transition"
        assert r.transition.temperature_K == 0.0

    def test_2d_zero_transition_isotropic(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = sc_square_fm_magnon()
        p = default_params(
            thermal_dimensionality=2,
            thermal_qmeshes=[[16, 16, 1], [24, 24, 1]],
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.status == "zero_transition"
        assert r.transition.kind == "curie_temperature"

    def test_2d_gap_gives_finite_tc(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = sc_square_fm_magnon(J=0.05, S=1.0)
        A = 0.5e-3
        mag.JR[0, 0, 0] = np.diag([0.0, 0.0, A * 1.0])
        p = default_params(
            thermal_dimensionality=2,
            thermal_qmeshes=[[24, 24, 1], [32, 32, 1]],
            thermal_mesh_tolerance=2.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.status == "ok"
        assert r.transition.temperature_K > 0

    def test_self_consistent_matches_closed_form(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver, gamma_centered_mesh

        mag = simple_cubic_fm_magnon(J=0.05, S=0.5)
        p = default_params(thermal_qmeshes=[[8, 8, 8]])
        solver = ThermalMagnonSolver(mag, p)
        q = gamma_centered_mesh([8, 8, 8], 3)
        closed = solver._tc_closed_form(q)
        t1 = solver._tc_self_consistent(q, 1e-4)
        t2 = solver._tc_self_consistent(q, 2e-4)
        sc = 2.0 * t1 - t2  # Richardson extrapolation in m* (docs/sympy/02)
        assert abs(closed - sc) / closed < 5e-3

    def test_strict_mode_raises_on_unconverged(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon()
        p = default_params(
            thermal_qmeshes=[[6, 6, 6]],
            thermal_mesh_tolerance=1e-6,
            thermal_strict=True,
        )
        solver = ThermalMagnonSolver(mag, p)
        with pytest.raises(RuntimeError, match="thermal_strict"):
            solver.calculate()

    def test_flagged_estimate_without_strict(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon()
        p = default_params(
            thermal_qmeshes=[[6, 6, 6]],
            thermal_mesh_tolerance=1e-6,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.status == "ok"
        assert r.transition.converged is False
        assert len(r.mesh_history) == 1

    def test_unstable_reference_detected(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon(J=-0.05, S=1.0)  # AFM exchange, FM mode
        p = default_params()
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.status == "unstable_reference"

    def test_multisite_fm_bands_and_tc(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = two_site_fm_magnon(J=0.05, S=1.0)
        p = default_params(
            thermal_qmeshes=[[16, 1, 1], [24, 1, 1]], thermal_mesh_tolerance=5.0
        )
        # 1D two-site chain is 1D: use 3D-declared? No: keep 3D cell but chain
        # exchange only couples x -> still 3D-declared mesh is fine (y,z divisions
        # see dispersionless bands). Use 3D declaration to avoid dimensionality
        # rejection since R vectors have zero y/z.
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate(temperatures_K=[0.0], band_kpoints=np.array([[0.0, 0, 0]]))
        assert r.status == "ok"
        # optic Goldstone at Gamma: acoustic zero, optic finite
        energies = r.bands[0].energies_eV[0]
        assert abs(energies[0]) < 1e-8
        assert energies[1] > 0


# ----------------------------------------------------------------------------
# Story 016: FM Callen and HP temperature methods
# ----------------------------------------------------------------------------


class TestFmCallenHpMethods:
    def test_callen_t0_reduces_to_lswt_with_hp_gap(self):
        """CD T=0 gap uses A(2S-1), not the RPA 2AS (docs/sympy/03)."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        A = 0.5e-3
        S = 1.0
        mag = simple_cubic_fm_magnon(J=0.05, S=S, sia=A)
        p = default_params(thermal_method="callen")
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate(temperatures_K=[0.0], band_kpoints=np.array([[0.0, 0, 0]]))
        gap = r.bands[0].energies_eV[0, 0]
        assert abs(gap - A * (2 * S - 1)) < 1e-9

    def test_rpa_callen_retains_cd_sia_ordering_and_exchange_anisotropy(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        S, A, lam = 1.0, 0.5e-3, 0.25e-3
        mag = simple_cubic_fm_magnon(J=0.05, S=S, sia=A, lam=lam)
        solver = ThermalMagnonSolver(mag, default_params(thermal_method="rpa_callen"))
        q_gamma = np.array([[0.0, 0.0, 0.0]])
        sol = solver._solve_fm(0.0, q_gamma)
        energy = np.linalg.eigvalsh(
            solver._fm_matrices(q_gamma, sol.m * solver._K) / solver._K
        )[0]
        assert abs(energy - (A * (2 * S - 1) + 6 * S * lam)) < 1e-9

    def test_callen_tc_above_rpa_and_converges(self):
        """CD classical Tc approaches ~1.49 J S^2 (paper/MC comparison)."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J = 0.05
        S = 1.0
        mag = simple_cubic_fm_magnon(J=J, S=S)
        p = default_params(
            thermal_method="callen",
            thermal_spin_regime="classical",
            thermal_qmeshes=[[12, 12, 12], [16, 16, 16]],
            thermal_mesh_tolerance=20.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.transition.converged
        kbtc = r.transition.temperature_K * 8.617333262e-5
        # classical CD ~ 1.45-1.52 J S^2 (docs/sympy/02: 1.4899 grid-corrected)
        assert 1.40 * J * S * S < kbtc < 1.55 * J * S * S

    def test_callen_quantum_tc_exceeds_rpa_for_half_spin(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J = 0.05
        S = 0.5
        mag = simple_cubic_fm_magnon(J=J, S=S)
        p = default_params(
            thermal_method="callen",
            thermal_qmeshes=[[12, 12, 12], [16, 16, 16]],
            thermal_mesh_tolerance=10.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.transition.method_validity == "limited"
        assert r.transition.validity_reason is not None
        # quantum CD Tc(S=1/2) ~ 1.3 J S(S+1)*2 (02: CD 1.3513 J at S=1/2)
        kbtc = r.transition.temperature_K * 8.617333262e-5
        assert kbtc > 1.1 * J  # well above RPA 0.989 J

    def test_hp_reports_breakdown_not_tc(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon(J=0.05, S=0.5)
        p = default_params(
            thermal_method="hp",
            thermal_qmeshes=[[12, 12, 12], [16, 16, 16]],
            thermal_mesh_tolerance=10.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.transition.kind == "temperature_hp_breakdown"
        assert r.transition.breakdown_magnetization > 0.02
        # HP breaks down at finite m ~ 0.14 for S=1/2 (docs/sympy/02)
        assert r.transition.breakdown_magnetization < 0.5
        kbt = r.transition.temperature_K * 8.617333262e-5
        assert kbt > 0.5 * 0.05  # finite, order J

    def test_hp_matches_verified_single_site_curve(self):
        """The matrix HP closure reproduces docs/sympy/02 on its 24^3 grid."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver, gamma_centered_mesh

        J = 0.05
        mesh = [24, 24, 24]
        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=0.5),
            default_params(thermal_method="hp", thermal_qmeshes=[mesh]),
        )
        solver._set_mesh_shape(mesh)
        solution = solver._solve_fm(
            0.6 * J / 8.617333262e-5, gamma_centered_mesh(mesh, 3)
        )
        assert solution.converged
        assert abs(solution.m - 0.4136) < 3e-4

    def test_hp_bands_break_down_status(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = simple_cubic_fm_magnon(J=0.05, S=0.5)
        p = default_params(thermal_method="hp", thermal_qmeshes=[[8, 8, 8]])
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate(
            temperatures_K=[5.0, 700.0], band_kpoints=np.array([[0.5, 0.0, 0.0]])
        )
        assert r.bands[0].status == "ordered"
        assert r.bands[1].status == "hp_breakdown"

    def test_method_curves_order(self):
        """At moderate T: CD magnetization exceeds RPA (docs/sympy/02)."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J = 0.05
        mag = simple_cubic_fm_magnon(J=J, S=0.5)
        qmeshes = [[10, 10, 10], [12, 12, 12]]
        T = 0.6 * J / 8.617333262e-5  # kT = 0.6 J
        results = {}
        for method in ("rpa", "callen"):
            p = default_params(
                thermal_method=method,
                thermal_qmeshes=qmeshes,
                thermal_mesh_tolerance=50.0,
            )
            solver = ThermalMagnonSolver(mag, p)
            r = solver.calculate(
                temperatures_K=[T], band_kpoints=np.array([[0.5, 0, 0]])
            )
            results[method] = r.bands[0].order_parameters[0]
        assert results["callen"] > results["rpa"]


# ----------------------------------------------------------------------------
# Story 021: FM Weiss mean-field (MFA) thermal baseline
# ----------------------------------------------------------------------------


def _mfa_tc_analytic(J0, S):
    """Paper eq. eq:T_mf: k_B Tc^MFA = J0 S(S+1)/3."""
    return J0 * S * (S + 1.0) / 3.0 / KB


class TestFmMfaBaseline:
    """Weiss single-site MFA: Brillouin self-consistency, analytic Tc.

    MFA is the uncorrelated paper-comparison baseline (arXiv:2405.00477
    Sec. II.A): it has no magnon-band interpretation and its transition is
    the exact linearized single-site result, so these tests pin the
    Brillouin closure, the analytic Tc in both spin regimes, the
    Mermin-Wagner validity warning, and the bandless result schema.
    """

    def test_brillouin_function_matches_exact_level_sum(self):
        from TB2J.magnon.thermal_solver import brillouin_function

        for S in (0.5, 1.0, 1.5, 2.5):
            levels = np.arange(2 * S + 1) - S  # eigenvalues m_s of S^z
            for x in (1e-3, 0.1, 1.0, 3.0, 10.0):
                # <Sz> = sum m_s exp(x m_s / S) / Z with <Sz> = S B_S(x)
                w = np.exp(x / S * (levels - S))
                exact = float((levels * w).sum() / w.sum()) / S
                assert np.isclose(float(brillouin_function(S, x)), exact, rtol=1e-11)
            # small-x linearization B_S(x) -> (S+1) x / (3S)
            x = 1e-9
            assert np.isclose(
                float(brillouin_function(S, x)), (S + 1.0) * x / (3.0 * S), rtol=1e-9
            )

    def test_mfa_self_consistency_solves_single_site_partition(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J, S = 0.05, 1.0
        J0 = 6.0 * J
        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=S), default_params(thermal_method="mfa")
        )
        Tc = _mfa_tc_analytic(J0, S)
        assert solver._solve_mfa(0.0) == S
        m = solver._solve_mfa(0.8 * Tc)
        assert 0.0 < m < S
        # residual of the paper's eq. mfa-mag level sum (independent form
        # of the Brillouin function used by the solver)
        levels = np.arange(2 * S + 1) - S
        w = np.exp(J0 * m * (levels - S) / (KB * 0.8 * Tc))
        assert abs(m - float((levels * w).sum() / w.sum())) < 1e-9
        # no ordered solution survives above the linearized transition
        assert solver._solve_mfa(1.001 * Tc) == 0.0
        assert solver._solve_mfa(0.5 * Tc) > m

    def test_mfa_near_tc_and_low_temperature_roots(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J, S = 0.05, 1.0
        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=S), default_params(thermal_method="mfa")
        )
        Tc = _mfa_tc_analytic(6.0 * J, S)
        # regression (story-021 review): the ordered root within 1e-8 of
        # Tc is only ~1e-4*S tall, below any fixed probe floor, and the
        # root at very low T sits above 0.95*S, above the old ladder's
        # top bracket; the monotone iteration from m = S must find both.
        m_near = solver._solve_mfa((1.0 - 1e-8) * Tc)
        assert 0.0 < m_near < S
        m_low = solver._solve_mfa(1e-3 * Tc)
        assert 0.95 * S < m_low <= S
        # m(T) stays monotone between the extremes
        assert solver._solve_mfa(0.5 * Tc) > m_near
        assert solver._solve_mfa(0.5 * Tc) < m_low

    def test_mfa_quantum_tc_is_analytic(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J = 0.05
        for S in (0.5, 1.5):
            solver = ThermalMagnonSolver(
                simple_cubic_fm_magnon(J=J, S=S), default_params(thermal_method="mfa")
            )
            r = solver.calculate()
            assert r.status == "ok"
            assert r.transition.kind == "curie_temperature"
            assert r.transition.converged
            assert r.transition.method_validity == "nominal"
            assert r.transition.temperature_K == pytest.approx(
                _mfa_tc_analytic(6.0 * J, S), rel=1e-12
            )
        # the uncorrelated baseline overestimates: MFA sits above the RPA
        # Watson value S(S+1) * 1.318926 * J (story 015 test)
        r = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=0.5), default_params(thermal_method="mfa")
        ).calculate()
        kbtc = r.transition.temperature_K * KB
        assert 1.2 * J < kbtc < 1.6 * J  # exact MFA: 1.5 J; RPA: 0.989 J

    def test_mfa_classical_regime_scaling_and_langevin_limit(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J, S = 0.05, 1.0
        J0 = 6.0 * J
        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=S),
            default_params(thermal_method="mfa", thermal_spin_regime="classical"),
        )
        r = solver.calculate()
        # classical prescription (K-mapping with S_eff = K S): k_B Tc -> J0 S^2/3
        expected = J0 * S * S / 3.0 / KB
        assert abs(r.transition.temperature_K - expected) / expected < 0.01
        # m(T) follows the classical Langevin closure m = S L(beta S J0 m)
        T = 0.7 * expected
        m = solver._solve_mfa(T)
        assert 0.0 < m <= S
        x = S * J0 * m / (KB * T)
        langevin = 1.0 / np.tanh(x) - 1.0 / x
        assert abs(m - S * langevin) < 0.01 * S

    def test_mfa_low_dimensional_finite_transition_mermin_wagner_warning(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        p2d = default_params(
            thermal_method="mfa",
            thermal_dimensionality=2,
            thermal_qmeshes=[[8, 8, 1], [12, 12, 1]],
        )
        r = ThermalMagnonSolver(sc_square_fm_magnon(), p2d).calculate()
        assert r.status == "ok"  # finite transition, not zero_transition
        assert r.transition.temperature_K > 0.0
        assert r.transition.method_validity == "limited"
        assert "Mermin" in r.transition.validity_reason
        assert "Wagner" in r.transition.validity_reason

        # 1D isotropic: same policy
        p1d = default_params(
            thermal_method="mfa", thermal_dimensionality=1, thermal_qmeshes=[[8, 1, 1]]
        )
        r1d = ThermalMagnonSolver(sc_chain_fm_magnon(), p1d).calculate()
        assert r1d.transition.temperature_K > 0.0
        assert r1d.transition.method_validity == "limited"

        # RPA behaviour on the same model is unchanged: zero transition
        rpa2d = ThermalMagnonSolver(
            sc_square_fm_magnon(),
            default_params(
                thermal_dimensionality=2, thermal_qmeshes=[[16, 16, 1], [24, 24, 1]]
            ),
        ).calculate()
        assert rpa2d.status == "zero_transition"
        assert rpa2d.transition.temperature_K == 0.0
        assert rpa2d.transition.method_validity == "nominal"

        # 3D stays nominal
        assert (
            ThermalMagnonSolver(
                simple_cubic_fm_magnon(), default_params(thermal_method="mfa")
            )
            .calculate()
            .transition.method_validity
            == "nominal"
        )
        # gapped 2D (finite exact Tc) stays nominal as well
        mag = sc_square_fm_magnon(J=0.05, S=1.0)
        mag.JR[0, 0, 0] = np.diag([0.0, 0.0, 0.5e-3])
        rg = ThermalMagnonSolver(mag, p2d).calculate()
        assert rg.transition.method_validity == "nominal"

    def test_mfa_calculate_emits_bandless_order_parameter_blocks(self, tmp_path):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        J, S = 0.05, 1.0
        J0 = 6.0 * J
        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(J=J, S=S), default_params(thermal_method="mfa")
        )
        Tc = _mfa_tc_analytic(J0, S)
        r = solver.calculate(temperatures_K=[0.0, 0.5 * Tc, 1.5 * Tc])
        # AC1: requested temperatures are answered with bandless blocks
        # carrying the Brillouin order parameter m(T), not dropped.
        assert [b.temperature_K for b in r.bands] == [0.0, 0.5 * Tc, 1.5 * Tc]
        for block in r.bands:
            assert block.kpoints.shape == (0, 3)  # bandless: no k-points
            assert block.energies_eV.shape == (0, 1)  # no magnon energies
            assert block.magnetization.shape == (1,)  # per-site <Sz> = m
        m0, m_half, m_hot = (b.order_parameters[0] for b in r.bands)
        assert m0 == S
        assert 0.0 < m_half < S
        # m(0.5 Tc) satisfies the independent single-site partition sum
        levels = np.arange(2 * S + 1) - S
        w = np.exp(J0 * m_half * (levels - S) / (KB * 0.5 * Tc))
        assert abs(m_half - float((levels * w).sum() / w.sum())) < 1e-9
        # above the transition the ordered solution is gone
        assert m_hot == 0.0
        assert r.bands[0].status == "ordered"
        assert r.bands[2].status == "zero_transition"
        # both serializations round-trip the bandless blocks
        fjson = tmp_path / "mfa_blocks.json"
        r.save_json(str(fjson))
        loaded = ThermalMagnonResult.load_json(str(fjson))
        assert len(loaded.bands) == 3
        assert loaded.bands[1].order_parameters[0] == pytest.approx(m_half)
        assert loaded.bands[1].kpoints.shape == (0, 3)
        netcdf4 = pytest.importorskip("netCDF4")
        assert netcdf4 is not None
        fnc = tmp_path / "mfa_blocks.nc"
        r.save_netcdf(str(fnc))
        loaded_nc = ThermalMagnonResult.load_netcdf(str(fnc))
        assert len(loaded_nc.bands) == 3
        assert loaded_nc.bands[1].order_parameters[0] == pytest.approx(m_half)
        assert loaded_nc.bands[1].energies_eV.shape == (0, 1)

    def test_mfa_result_is_bandless_with_baseline_label(self, tmp_path):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        solver = ThermalMagnonSolver(
            simple_cubic_fm_magnon(), default_params(thermal_method="mfa")
        )
        r = solver.calculate(
            temperatures_K=[0.0, 300.0],
            band_kpoints=np.array([[0.0, 0, 0], [0.5, 0, 0]]),
        )
        assert r.method == "mfa"
        # bandless: requested temperatures yield per-temperature blocks
        # carrying only m(T); the band_kpoints request is deliberately
        # ignored because MFA has no thermal magnon-spectrum output.
        assert len(r.bands) == 2
        for block in r.bands:
            assert block.kpoints.shape == (0, 3)
            assert block.energies_eV.shape == (0, 1)
        meta = r.to_dict()["metadata"]
        assert meta["method_class"] == "thermodynamic_baseline"
        assert (
            sample_result().to_dict()["metadata"]["method_class"] == "magnon_spectrum"
        )
        assert len(r.mesh_history) == 1
        assert r.mesh_history[0].status == "converged"
        assert r.transition.detail is not None
        fname = tmp_path / "mfa.json"
        r.save_json(str(fname))
        loaded = ThermalMagnonResult.load_json(str(fname))
        assert loaded.method == "mfa"
        assert len(loaded.bands) == 2
        assert all(b.kpoints.shape == (0, 3) for b in loaded.bands)
        assert loaded.to_dict()["metadata"]["method_class"] == "thermodynamic_baseline"
        # no temperatures requested -> no band blocks at all
        r_plain = ThermalMagnonSolver(
            simple_cubic_fm_magnon(),
            default_params(thermal_method="mfa", thermal_temperatures=[]),
        ).calculate()
        assert r_plain.bands == []

    def test_mfa_afm_rejected(self):
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        with pytest.raises(ThermalModelValidationError, match="mfa"):
            ThermalMagnonSolver(
                bipartite_afm_magnon(1), afm_params(1, thermal_method="mfa")
            )

    def test_mfa_cli_and_toml_registration(self, tmp_path):
        import argparse

        parser = argparse.ArgumentParser()
        add_thermal_args(parser)
        args = parser.parse_args(["--thermal-method", "mfa"])
        p = thermal_parameters_from_args(args)
        assert p.thermal_method == "mfa"
        fname = tmp_path / "mfa.toml"
        p.to_toml(str(fname))
        assert ThermalMagnonParameters.from_toml(str(fname)).thermal_method == "mfa"


# ----------------------------------------------------------------------------
# Unified magnon CLI thermal dispatch (story-021 review fix, ADR-0006)
# ----------------------------------------------------------------------------


class TestThermalCliDispatch:
    """The registered TB2J_magnon.py CLI runs thermal calculations."""

    TB2J_RESULTS = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "data"
        / "tests"
        / "3_CrI3_wannier_SOC"
        / "refs"
        / "TB2J_results"
    )

    def test_thermal_dispatch_serializes_versioned_result(self, monkeypatch, tmp_path):
        import sys

        from TB2J.magnon import magnon_cli

        args = magnon_cli.create_parser().parse_args(
            ["--thermal-method", "mfa", "--thermal-temperatures", "0,50"]
        )
        assert args.thermal_method == "mfa"
        assert thermal_parameters_from_args(args).thermal_temperatures == [0.0, 50.0]

        out = tmp_path / "thermal.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon.py",
                "--path",
                str(self.TB2J_RESULTS),
                "--no-DMI",
                "--no-Jani",
                "--thermal-method",
                "mfa",
                "--thermal-temperatures",
                "0,50",
                "--thermal-output",
                str(out),
            ],
        )
        magnon_cli.main()
        data = json.loads(out.read_text())
        assert data["schema_name"] == "tb2j.magnon.thermal"
        assert data["metadata"]["method"] == "mfa"
        assert data["bands"][0]["order_parameters"] == pytest.approx(
            [data["metadata"]["spins"][0]]
        )
        assert [b["temperature_K"] for b in data["bands"]] == [0.0, 50.0]

    def test_thermal_only_invocation_without_bands_or_dos(self, monkeypatch, tmp_path):
        import sys

        from TB2J.magnon import magnon_cli

        out = tmp_path / "thermal.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon.py",
                "--path",
                str(self.TB2J_RESULTS),
                "--no-DMI",
                "--no-Jani",
                "--thermal-method",
                "mfa",
                "--thermal-output",
                str(out),
            ],
        )
        magnon_cli.main()  # must not error about missing --bands/--dos
        assert (
            json.loads(out.read_text())["metadata"]["method_class"]
            == "thermodynamic_baseline"
        )

    def test_thermal_cli_spectral_temperatures_emit_band_blocks(
        self, monkeypatch, tmp_path
    ):
        """Explicit temperatures with a spectral method produce bands (ADR 0006)."""
        import sys

        from TB2J.magnon import magnon_cli

        out = tmp_path / "thermal_rpa.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon.py",
                "--path",
                str(self.TB2J_RESULTS),
                "--no-DMI",
                "--no-Jani",
                "--thermal-method",
                "rpa",
                "--thermal-temperatures",
                "0,300",
                "--thermal-qmeshes",
                "4x4x4",
                "--thermal-mesh-tolerance",
                "1000",
                "--thermal-output",
                str(out),
            ],
        )
        magnon_cli.main()
        data = json.loads(out.read_text())
        assert data["metadata"]["method"] == "rpa"
        assert [b["temperature_K"] for b in data["bands"]] == [0.0, 300.0]
        for block in data["bands"]:
            assert block["kpoints"]
            assert block["energies_eV"]

    def test_no_thermal_args_preserves_band_only_behavior(self, monkeypatch):
        import sys

        from TB2J.magnon import magnon_cli

        captured = []
        monkeypatch.setattr(magnon_cli, "plot_magnon_bands_from_TB2J", captured.append)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "TB2J_magnon.py",
                "--bands",
                "--path",
                str(self.TB2J_RESULTS),
            ],
        )
        magnon_cli.main()
        assert len(captured) == 1


# ----------------------------------------------------------------------------
# AFM Nambu RPA Neel temperature (docs/sympy/04): K0/eps_q^2 critical kernel
# ----------------------------------------------------------------------------


def bipartite_afm_magnon(dim, K=0.04, S=1.5):
    """Two-sublattice isotropic AFM in TB2J storage (J_iso = -K per bond).

    dim=1: NN chain (z=2), eps_q = 2K|sin(pi q_x)|;
    dim=2: checkerboard square lattice (z=4),
           eps_q = 4K sqrt(1 - cos^2(pi q_x) cos^2(pi q_y));
    dim=3: CsCl/bcc nearest-neighbour lattice (z=8),
           eps_q = 8K sqrt(1 - prod_i cos^2(pi q_i));
    in every case K_0 = z K (docs/sympy/04 Section 4).  Each undirected
    A(cell 0) - B(cell rb) bond is stored as JR[rb,0,1] and JR[-rb,1,0],
    mirroring ``two_site_fm_magnon``.
    """
    if dim == 1:
        neighbour_cells = [(0, 0, 0), (-1, 0, 0)]
    elif dim == 2:
        neighbour_cells = [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)]
    elif dim == 3:
        neighbour_cells = [(x, y, z) for x in (0, -1) for y in (0, -1) for z in (0, -1)]
    else:
        raise ValueError(f"unsupported dimensionality {dim}")
    rset = sorted(
        {tuple(rb) for rb in neighbour_cells}
        | {tuple(-np.array(rb)) for rb in neighbour_cells}
    )
    Rlist = np.array(rset, dtype=float)
    rmap = {r: i for i, r in enumerate(rset)}
    JR = np.zeros((len(Rlist), 2, 2, 3, 3))
    half = 0.5 * S * S
    for rb in neighbour_cells:
        rm = tuple(-np.array(rb))
        JR[rmap[tuple(rb)], 0, 1] = half * np.eye(3) * (-K)
        JR[rmap[rm], 1, 0] = half * np.eye(3) * (-K)
    magmom = np.array([[0.0, 0.0, 2.0 * S], [0.0, 0.0, -2.0 * S]])
    return _finalize(
        Magnon(
            nspin=2,
            magmom=magmom,
            Rlist=Rlist,
            JR=JR,
            cell=_identity_cell(),
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
        )
    )


def afm_params(dim, meshes=None, **kwargs):
    mesh = meshes if meshes is not None else [[8] * dim + [1] * (3 - dim)]
    return default_params(
        thermal_order_mode="bipartite_afm",
        thermal_dimensionality=dim,
        thermal_qmeshes=mesh,
        **kwargs,
    )


class TestAfmRpaNeelTemperature:
    def test_nambu_spectrum_and_K0_match_analytic(self):
        """Grounding: positive Nambu modes are eps_q and Gamma normal-block
        diagonal is K_0 = z K for the chain, checkerboard, and CsCl AFMs."""
        from TB2J.magnon.thermal_solver import (
            ThermalMagnonSolver,
            gamma_centered_mesh,
        )

        K, S = 0.04, 1.5

        def analytic_eps(q, dim):
            c = np.ones(len(q))
            for ax in range(dim):
                c = c * np.cos(np.pi * q[:, ax])
            return 2.0**dim * K * np.sqrt(np.clip(1.0 - c**2, 0.0, None))

        for dim in (1, 2, 3):
            solver = ThermalMagnonSolver(
                bipartite_afm_magnon(dim, K=K, S=S), afm_params(dim)
            )
            mesh = [8] * dim + [1] * (3 - dim)
            q = gamma_centered_mesh(mesh, dim)
            mu = solver._afm_positive_modes(q, 1.0)
            assert np.abs(mu - analytic_eps(q, dim)[:, None]).max() < 1e-10
            H0 = solver.model.M_bdg_q(np.zeros((1, 3)))[0]
            K0 = float(np.mean(np.real(np.diag(H0)[: solver.model.nspin])))
            assert abs(K0 - 2.0**dim * K) < 1e-12

    def test_afm_tn_kernel_is_K0_over_epsilon_squared(self):
        """The AFM critical kernel is mean'_q[K_0/eps_q^2]
        (docs/sympy/04 eq. TN), with exact Goldstone modes excluded as the
        finite-mesh regulator -- not the scalar 1/eps_q weight."""
        from TB2J.magnon.thermal_solver import (
            ThermalMagnonSolver,
            gamma_centered_mesh,
        )

        K, S = 0.04, 1.5
        mesh = [8, 8, 8]
        solver = ThermalMagnonSolver(
            bipartite_afm_magnon(3, K=K, S=S), afm_params(3, meshes=[mesh])
        )
        tn = solver._tc_closed_form(gamma_centered_mesh(mesh, 3))
        # independent analytic CsCl kernel: K_0 = 8K, eps = 8K sqrt(1-c^2)
        q = gamma_centered_mesh(mesh, 3)
        c = np.prod(np.cos(np.pi * q), axis=1)
        eps = 8.0 * K * np.sqrt(np.clip(1.0 - c**2, 0.0, None))
        keep = eps > 1e-10  # exact Goldstone points excluded
        mean_kernel = float((8.0 * K / eps[keep] ** 2).mean())
        expected = S * (S + 1.0) / 3.0 / mean_kernel / KB
        assert np.isclose(tn, expected, rtol=1e-8)

    def test_afm_tn_2d_kernel_diverges_mermin_wagner(self):
        """The weighted kernel diverges logarithmically with mesh in 2D, so
        the closed-form T_N -> 0 (Mermin-Wagner).  A scalar 1/eps kernel
        would converge to a finite, unphysical 2D T_N."""
        from TB2J.magnon.thermal_solver import (
            ThermalMagnonSolver,
            gamma_centered_mesh,
        )

        solver = ThermalMagnonSolver(bipartite_afm_magnon(2), afm_params(2))
        tn = []
        for n in (16, 32, 64):
            mesh = [n, n, 1]
            tn.append(solver._tc_closed_form(gamma_centered_mesh(mesh, 2)))
        assert all(t > 0.0 for t in tn)
        assert tn[0] > tn[1] > tn[2]  # mesh refinement lowers T_N
        assert tn[2] < 0.75 * tn[0]  # 1/ln(N) decay toward zero

    def test_afm_tn_3d_finite_and_converged(self):
        """3D CsCl AFM: finite, mesh-converged Neel temperature from the
        weighted kernel, below the mean-field value."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        K, S = 0.04, 1.5
        mag = bipartite_afm_magnon(3, K=K, S=S)
        p = afm_params(
            3,
            meshes=[[24, 24, 24], [32, 32, 32]],
            thermal_mesh_tolerance=30.0,
        )
        solver = ThermalMagnonSolver(mag, p)
        r = solver.calculate()
        assert r.status == "ok"
        assert r.transition.kind == "neel_temperature"
        assert r.transition.converged
        t_n = r.transition.temperature_K
        assert 0.0 < t_n < S * (S + 1.0) * 8.0 * K / 3.0 / KB
        # matches the analytic weighted kernel on the same (final) mesh
        q = (
            np.stack(np.meshgrid(*[np.arange(32) / 32.0] * 3, indexing="ij"))
            .reshape(3, -1)
            .T
        )
        c = np.prod(np.cos(np.pi * q), axis=1)
        eps = 8.0 * K * np.sqrt(np.clip(1.0 - c**2, 0.0, None))
        keep = eps > 1e-10
        expected = S * (S + 1.0) / 3.0 / float((8.0 * K / eps[keep] ** 2).mean()) / KB
        assert np.isclose(t_n, expected, rtol=1e-6)

    def test_afm_1d_zero_transition(self):
        """Isotropic 1D AFM is gapless: the solver reports T_N = 0."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        solver = ThermalMagnonSolver(
            bipartite_afm_magnon(1),
            afm_params(1, meshes=[[16, 1, 1], [32, 1, 1]]),
        )
        r = solver.calculate()
        assert r.status == "zero_transition"
        assert r.transition.temperature_K == 0.0
        assert r.transition.kind == "neel_temperature"


def frustrated_afm_chain_magnon(K=0.04, Kf=0.03, S=1.5):
    """Chain with NN AFM bonds plus competing A-B exchange of the
    opposite sign at R = +-2 cells.  All bonds still connect opposite
    sublattices (the normal BdG block stays scalar), but omega_q^2 < 0
    away from Gamma: the antiparallel reference is dynamically
    unstable and must be rejected, never read as a Goldstone zero.
    """
    bonds = {(0, 0, 0): -K, (-1, 0, 0): -K, (2, 0, 0): Kf, (-3, 0, 0): Kf}
    rset = sorted(set(bonds) | {tuple(-np.array(r)) for r in bonds})
    Rlist = np.array(rset, dtype=float)
    rmap = {r: i for i, r in enumerate(rset)}
    JR = np.zeros((len(Rlist), 2, 2, 3, 3))
    half = 0.5 * S * S
    for rb, J in bonds.items():
        rm = tuple(-np.array(rb))
        JR[rmap[rb], 0, 1] = half * np.eye(3) * J
        JR[rmap[rm], 1, 0] = half * np.eye(3) * J
    magmom = np.array([[0.0, 0.0, 2.0 * S], [0.0, 0.0, -2.0 * S]])
    return _finalize(
        Magnon(
            nspin=2,
            magmom=magmom,
            Rlist=Rlist,
            JR=JR,
            cell=_identity_cell(),
            _Q=np.zeros(3),
            _uz=np.array([[0.0, 0.0, 1.0]]),
            _n=np.array([0.0, 0.0, 1.0]),
        )
    )


class TestAfmNambuClosureCorrections:
    """Review corrections to the AFM Nambu RPA branch (docs/sympy/04)."""

    def test_afm_closure_is_nambu_contraction_with_zero_point(self):
        """The AFM fluctuation is the local normal contraction
        1/2[A(2nB+1)/omega - 1] (docs/sympy/04 eq. 6) with the
        Bogoliubov weight A/omega = K_0/eps and the zero-point -1/2
        (v_q^2 at T = 0, eq. 7) -- not the bare Bose occupation of the
        FM closure."""
        from TB2J.magnon.thermal_solver import (
            ThermalMagnonSolver,
            bose_factors,
            callen_magnetization,
            gamma_centered_mesh,
        )

        K, S = 0.04, 1.5
        mesh = [8, 8, 8]
        solver = ThermalMagnonSolver(
            bipartite_afm_magnon(3, K=K, S=S), afm_params(3, meshes=[mesh])
        )
        q = gamma_centered_mesh(mesh, 3)
        c = np.prod(np.cos(np.pi * q), axis=1)
        eps = 8.0 * K * np.sqrt(np.clip(1.0 - c**2, 0.0, None))
        keep = eps > 1e-10  # exact Goldstone points excluded
        K0 = 8.0 * K

        sol = solver._solve_afm_rpa(0.0, q)
        n0 = float((0.5 * (K0 / eps[keep] - 1.0)).mean())
        assert sol.m < S - 0.5 * n0  # zero-point depletion present
        assert np.isclose(sol.m, callen_magnetization(S, n0), rtol=1e-8)
        assert abs(sol.m - (S - n0)) < 0.01 * n0  # eq. 7, Callen slope -1

        T = 0.6 * solver._tc_closed_form(q)
        sol = solver._solve_afm_rpa(T, q)
        assert 0.0 < sol.m < S
        w = sol.m * eps
        nB = bose_factors(w[keep], T)
        phi = float((0.5 * (K0 * sol.m / w[keep] * (2.0 * nB + 1.0) - 1.0)).mean())
        assert np.isclose(callen_magnetization(S, phi), sol.m, rtol=1e-7)

    def test_afm_classical_mapping_mirrors_fm(self):
        """Classical AFM solves the quantum closure at S_eff = K S and
        T_eff = K^2 T and reports m/K with omega = m_phys * eps_q, like
        the FM classical mapping -- never the raw effective quantities."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        K, S = 0.04, 1.5
        p = afm_params(
            3,
            meshes=[[12, 12, 12]],
            thermal_spin_regime="classical",
        )
        solver = ThermalMagnonSolver(bipartite_afm_magnon(3, K=K, S=S), p)
        path = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]
        r = solver.calculate(temperatures_K=[800.0], band_kpoints=path)
        assert r.status == "ok"
        assert r.transition.temperature_K > 0.0
        block = r.bands[0]
        m = float(block.order_parameters[0])
        assert 0.0 < m <= S + 1e-12
        q = np.asarray(path)
        c = np.prod(np.cos(np.pi * q), axis=1)
        eps = 8.0 * K * np.sqrt(np.clip(1.0 - c**2, 0.0, None))
        assert np.allclose(block.energies_eV[:, 0], m * eps, rtol=1e-7)
        assert np.allclose(block.energies_eV[:, 1], block.energies_eV[:, 0])
        # staggered global moments: +m on sublattice A, -m on B
        assert np.allclose(block.magnetization, m * np.array([1.0, -1.0]))

    def test_metric_fallback_rejects_complex_unstable_blocks(self):
        """The metric fallback must separate an exactly marginal
        Goldstone block (real zero eigenvalues) from a dynamically
        unstable block (complex eigenvalues, omega^2 < 0): the latter
        comes back as negative pseudo-modes so instability is rejected
        instead of masquerading as a Goldstone zero."""
        from TB2J.magnon.thermal_model import (
            _metric_positive_modes,
            _paraunitary_eigenvalues,
        )
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        def block(b):
            return np.array(
                [[5.0, 0, 0, b], [0, 5.0, b, 0], [0, b, 5.0, 0], [b, 0, 0, 5.0]]
            )

        gold = block(5.0)  # A = |B|: singular, marginal Goldstone
        stable = block(3.0)  # omega = sqrt(25 - 9) = 4
        unstable = block(6.0)  # omega^2 = 25 - 36 < 0: complex pair
        modes = _metric_positive_modes(np.stack([gold, stable, unstable]))
        assert np.max(np.abs(modes[0])) < 1e-10
        assert np.allclose(modes[1], 4.0)
        assert np.all(modes[2] < 0.0)
        assert np.allclose(modes[2], -np.sqrt(36.0 - 25.0))
        assert np.allclose(_paraunitary_eigenvalues(stable), 4.0)

        solver = ThermalMagnonSolver(
            frustrated_afm_chain_magnon(), afm_params(1, meshes=[[16, 1, 1]])
        )
        assert solver.unstable
        r = solver.calculate()
        assert r.status == "unstable_reference"
        assert r.transition.temperature_K == 0.0
        assert r.transition.kind == "neel_temperature"

    def test_afm_rejects_same_sublattice_exchange(self):
        """Same-sublattice transverse exchange makes the BdG normal
        block q-dependent (A_q != m K_0 scalar); such AFMs are outside
        the derived Nambu reduction and must be rejected with an
        actionable message, not silently mis-weighted."""
        from TB2J.magnon.thermal_solver import ThermalMagnonSolver

        mag = bipartite_afm_magnon(1)
        rmap = {tuple(r): i for i, r in enumerate(np.asarray(mag.Rlist, float))}
        mag.JR = mag.JR.copy()
        for r in ((1, 0, 0), (-1, 0, 0)):
            mag.JR[rmap[r], 0, 0] = 0.5 * 1.5 * 1.5 * np.eye(3) * (-0.01)
        with pytest.raises(ThermalModelValidationError, match="same-sublattice"):
            build_thermal_spin_model(mag, afm_params(1))

        # pure A-B multi-shell exchange stays inside the derived form
        ThermalMagnonSolver(
            frustrated_afm_chain_magnon(Kf=0.0), afm_params(1, meshes=[[8, 1, 1]])
        )

    def test_afm_rejects_multisite_covariances(self):
        """A scalar normal block does not make a general multisite
        anomalous block site-equivalent. The current result schema has
        one staggered magnitude and therefore accepts exactly one site
        per sublattice until site-resolved Nambu covariances exist."""
        S = 1.5
        mag = _finalize(
            Magnon(
                nspin=3,
                magmom=np.array(
                    [[0.0, 0.0, 2.0 * S], [0.0, 0.0, 2.0 * S], [0.0, 0.0, -2.0 * S]]
                ),
                Rlist=np.zeros((1, 3)),
                JR=np.zeros((1, 3, 3, 3, 3)),
                cell=_identity_cell(),
                _Q=np.zeros(3),
                _uz=np.array([[0.0, 0.0, 1.0]]),
                _n=np.array([0.0, 0.0, 1.0]),
            )
        )
        with pytest.raises(
            ThermalModelValidationError, match="one magnetic site per sublattice"
        ):
            build_thermal_spin_model(mag, afm_params(1))
