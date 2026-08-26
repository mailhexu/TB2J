"""Tests for the thermal-magnon foundation (story 014).

Covers ThermalMagnonParameters validation/TOML round trip, the
``tb2j.magnon.thermal`` result schema (JSON and NetCDF round trips), and
ThermalSpinModel validation/rejection rules.
"""

import json

import numpy as np
import pytest

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
