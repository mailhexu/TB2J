from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.units import Ha

from TB2J.interfaces import gpaw_projector


def test_gpaw_projection_mapper_prefers_current_wannier_location(monkeypatch):
    mapper = object()
    calls = []

    def import_current(module_name):
        calls.append(module_name)
        return SimpleNamespace(get_projections_in_bz=mapper)

    monkeypatch.setattr(gpaw_projector, "import_module", import_current)

    assert gpaw_projector._projection_mapper() is mapper
    assert calls == ["gpaw.wannier.wannier90"]


def test_gpaw_projection_mapper_falls_back_to_legacy_location(monkeypatch):
    mapper = object()
    calls = []

    def import_compatible(module_name):
        calls.append(module_name)
        if module_name == "gpaw.wannier.wannier90":
            raise ModuleNotFoundError(name=module_name)
        return SimpleNamespace(get_projections_in_bz=mapper)

    monkeypatch.setattr(gpaw_projector, "import_module", import_compatible)

    assert gpaw_projector._projection_mapper() is mapper
    assert calls == ["gpaw.wannier.wannier90", "gpaw.wannier90"]


def test_collect_gpaw_hubbard_metadata_converts_units_and_preserves_sites():
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            setups=[
                SimpleNamespace(hubbard_u=None),
                SimpleNamespace(
                    hubbard_u=SimpleNamespace(U=[2.0 / Ha], l=[2], scale=[True])
                ),
            ]
        )
    )

    assert gpaw_projector._collect_hubbard_metadata(calc) == [
        {"atom_index": 1, "l": [2], "U_eV": [2.0], "scale": [True]}
    ]


def test_gpaw_hubbard_metadata_rejects_malformed_setup():
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            setups=[
                SimpleNamespace(
                    hubbard_u=SimpleNamespace(U=[1.0], l=[2, 3], scale=[True])
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="inconsistent Hubbard metadata"):
        gpaw_projector._collect_hubbard_metadata(calc)


def test_gpaw_total_delta_composes_xc_and_hubbard_once(monkeypatch):
    class Hubbard:
        def __init__(self):
            self.calls = 0

        def calculate(self, setup, density):
            self.calls += 1
            assert density == "unpacked-density"
            return 0.0, np.array([[[3.0]], [[1.0]]])

    hubbard = Hubbard()
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            setups=[SimpleNamespace(hubbard_u=hubbard)],
            nspins=2,
        ),
        density=SimpleNamespace(D_asp={0: "packed-density"}),
    )
    monkeypatch.setattr(
        gpaw_projector,
        "_unpack_gpaw_density",
        lambda density: "unpacked-density",
    )

    total = gpaw_projector._build_gpaw_total_delta(
        calc,
        np.array([[[0.5]]]),
        np.array([1]),
        [{"atom_index": 0, "l": [2], "U_eV": [5.0], "scale": [True]}],
    )

    np.testing.assert_allclose(total, [[[0.5 + 2.0 * Ha]]])
    assert hubbard.calls == 1


def test_gpaw_total_delta_requires_density_and_hubbard_evaluator():
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            setups=[SimpleNamespace(hubbard_u=None)],
            nspins=2,
        ),
        density=SimpleNamespace(D_asp={}),
    )

    with pytest.raises(ValueError, match="converged PAW density"):
        gpaw_projector._build_gpaw_total_delta(
            calc,
            np.zeros((1, 1, 1)),
            np.array([1]),
            [{"atom_index": 0, "l": [2], "U_eV": [5.0], "scale": [True]}],
        )


def test_gpaw_total_delta_requires_hubbard_evaluator(monkeypatch):
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            setups=[SimpleNamespace(hubbard_u=None)],
            nspins=2,
        ),
        density=SimpleNamespace(D_asp={0: "packed-density"}),
    )
    monkeypatch.setattr(
        gpaw_projector,
        "_unpack_gpaw_density",
        lambda density: "unpacked-density",
        raising=False,
    )

    with pytest.raises(ValueError, match="Hubbard evaluator"):
        gpaw_projector._build_gpaw_total_delta(
            calc,
            np.zeros((1, 1, 1)),
            np.array([1]),
            [{"atom_index": 0, "l": [2], "U_eV": [5.0], "scale": [True]}],
        )


def test_gpaw_u_export_adds_total_operator_and_provenance(monkeypatch):
    calc = SimpleNamespace(
        wfs=SimpleNamespace(
            nspins=2,
            mode="pw",
            kd=SimpleNamespace(nbzkpts=1, nibzkpts=1),
            setups=[
                SimpleNamespace(
                    hubbard_u=SimpleNamespace(
                        U=[5.0 / Ha],
                        l=[2],
                        scale=[True],
                        calculate=lambda *_: None,
                    )
                )
            ],
        ),
        get_magnetic_moment=lambda: 1.0,
        get_magnetic_moments=lambda: np.array([1.0]),
        density=SimpleNamespace(D_asp={0: "packed-density"}),
    )
    atoms = SimpleNamespace(
        cell=SimpleNamespace(array=np.eye(3)),
        positions=np.zeros((1, 3)),
        numbers=np.array([26]),
    )
    monkeypatch.setattr(
        gpaw_projector,
        "_collect_kpoint_data",
        lambda _calc: (
            np.zeros((1, 3)),
            np.ones(1),
            np.zeros((2, 1, 1)),
            np.ones((2, 1, 1)),
            np.ones((2, 1, 1, 1)),
        ),
    )
    monkeypatch.setattr(
        gpaw_projector, "_collect_fermi_levels", lambda _calc: np.array([0.0])
    )
    monkeypatch.setattr(
        gpaw_projector,
        "_setup_projector_metadata",
        lambda _setups: {
            "projector_l": np.array([0]),
            "projector_m": np.array([0]),
            "projector_radial": np.array([0]),
            "projector_atom": np.array([0]),
            "projector_site": np.array([0]),
            "site_nproj": np.array([1]),
            "site_projector_indices": np.array([[0]]),
            "paw_N0_p": [[]],
            "overlap_metric": np.eye(1),
        },
    )
    monkeypatch.setattr(
        gpaw_projector, "_collect_hij", lambda *_: np.array([[[[3.0]]], [[[1.0]]]])
    )
    monkeypatch.setattr(
        gpaw_projector, "_collect_delta_xc_paw_xc", lambda *_: np.array([[[0.5]]])
    )
    monkeypatch.setattr(
        gpaw_projector,
        "_build_gpaw_total_delta",
        lambda *_: np.array([[[2.0]]]),
    )

    data = gpaw_projector.gpaw_calc_to_projector_green_data(calc, atoms=atoms)

    np.testing.assert_allclose(data.operator_components["delta_total"], [[[2.0]]])
    assert data.operator_component_metadata["delta_total"]["hubbard_included"] == "true"
    assert (
        data.operator_component_metadata["delta_total"]["hubbard_evaluation_count"]
        == "1"
    )
    assert data.metadata["gpaw_hubbard"][0]["U_eV"] == [5.0]
    snapshot = gpaw_projector.gpaw_calc_to_paw_snapshot(calc, atoms=atoms)
    assert snapshot.operators.policy == "compose"
    assert snapshot.operators.selected_names == ("xc", "hubbard")
    np.testing.assert_allclose(snapshot.operators.components[1].values * Ha, [[[1.5]]])


def test_gpaw_collector_rejects_unsupported_spin_mode(monkeypatch):
    calc = SimpleNamespace(wfs=SimpleNamespace(nspins=1))
    monkeypatch.setattr(
        gpaw_projector,
        "_collect_kpoint_data",
        lambda _calc: (
            np.zeros((1, 3)),
            np.ones(1),
            np.zeros((1, 1, 1)),
            np.ones((1, 1, 1)),
            np.ones((1, 1, 1, 1)),
        ),
    )

    with pytest.raises(ValueError, match="collinear two-spin"):
        gpaw_projector.gpaw_calc_to_projector_green_data(calc, atoms=SimpleNamespace())


class FakeKPoint:
    def __init__(self, spin, q, eps_n, f_n, weight, p_ani):
        self.s = spin
        self.q = q
        self.eps_n = np.asarray(eps_n, dtype=float)
        self.f_n = np.asarray(f_n, dtype=float)
        self.weight = float(weight)
        self.P_ani = p_ani


def _fake_calc(kd, kpt_qs):
    wfs = SimpleNamespace(
        kd=kd,
        nspins=1,
        kpt_qs=kpt_qs,
    )
    return SimpleNamespace(wfs=wfs, get_bz_k_points=lambda: kd.bzk_kc)


def test_collect_kpoint_data_unfolds_symmetry_and_normalizes_occupations(monkeypatch):
    kd = SimpleNamespace(
        bzk_kc=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        nbzkpts=2,
        nibzkpts=1,
        bz2ibz_k=np.array([0, 0]),
    )
    source = FakeKPoint(
        spin=0,
        q=0,
        eps_n=[1.0, 2.0],
        f_n=[0.75, 0.25],
        weight=1.0,
        p_ani={0: np.array([[1.0], [2.0]])},
    )
    calc = _fake_calc(kd, [[source]])

    def fake_maps(calc):
        return object()

    def fake_projectors(wfs, bz_index, spin, ibz2bz):
        phase = 1.0 if bz_index == 0 else -1.0j
        return {0: phase * np.array([[1.0], [2.0]])}

    monkeypatch.setattr(gpaw_projector, "_make_ibz2bz_maps", fake_maps)
    monkeypatch.setattr(gpaw_projector, "_map_projections_in_bz", fake_projectors)

    kpoints, weights, eigenvalues, occupations, coefficients = (
        gpaw_projector._collect_kpoint_data(calc)
    )

    np.testing.assert_allclose(kpoints, kd.bzk_kc)
    np.testing.assert_allclose(weights, [0.5, 0.5])
    np.testing.assert_allclose(eigenvalues, [[[Ha, 2.0 * Ha], [Ha, 2.0 * Ha]]])
    np.testing.assert_allclose(occupations, [[[0.375, 0.125], [0.375, 0.125]]])
    np.testing.assert_allclose(coefficients[0, 0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(coefficients[0, 1, :, 0], [-1.0j, -2.0j])


def test_collect_kpoint_data_keeps_no_symmetry_weighted_occupations():
    kd = SimpleNamespace(
        bzk_kc=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        nbzkpts=2,
        nibzkpts=2,
        bz2ibz_k=np.array([0, 1]),
    )
    kpt_qs = [
        [
            FakeKPoint(
                spin=0,
                q=0,
                eps_n=[1.0],
                f_n=[0.4],
                weight=0.5,
                p_ani={0: np.array([[1.0]])},
            )
        ],
        [
            FakeKPoint(
                spin=0,
                q=1,
                eps_n=[2.0],
                f_n=[0.1],
                weight=0.5,
                p_ani={0: np.array([[2.0]])},
            )
        ],
    ]
    calc = _fake_calc(kd, kpt_qs)

    _, weights, eigenvalues, occupations, coefficients = (
        gpaw_projector._collect_kpoint_data(calc)
    )

    np.testing.assert_allclose(weights, [0.5, 0.5])
    np.testing.assert_allclose(eigenvalues, [[[Ha], [2.0 * Ha]]])
    np.testing.assert_allclose(occupations, [[[0.4], [0.1]]])
    np.testing.assert_allclose(coefficients[0, :, 0, 0], [1.0, 2.0])


@pytest.mark.ecosystem
def test_gpaw_symmetry_reduced_feo_dftu_exchange_is_cubic(tmp_path):
    """Opt-in GPAW 26.7 symmetry-on FeO PAW+U regression."""
    gpw_path = os.environ.get("TB2J_GPAW_FEO_GPW")
    if gpw_path is None:
        pytest.skip("set TB2J_GPAW_FEO_GPW to the symmetry-on FeO .gpw file")
    if not Path(gpw_path).is_file():
        pytest.skip(f"GPAW FeO fixture not found: {gpw_path}")

    pytest.importorskip("gpaw")
    from gpaw import GPAW

    calc = GPAW(gpw_path)
    data = gpaw_projector.gpaw_calc_to_projector_green_data(calc)
    assert data.metadata["gpaw_nbzkpts"] == 343
    assert data.metadata["gpaw_nibzkpts"] == 20
    assert (
        data.operator_component_metadata["delta_total"]["hubbard_evaluation_count"]
        == "1"
    )

    exchange_out, _ = gpaw_projector.gen_exchange_gpaw(
        calc,
        index_magnetic_atoms=[0],
        output_path=tmp_path,
        Rcut=8.0,
        nz=30,
    )
    rows = re.findall(
        r"^\s*Fe1\s+Fe1\s+\([^)]*\)\s+([-+.0-9]+)\s+" r"\([^)]*\)\s+([-+.0-9]+)",
        Path(exchange_out).read_text(),
        re.M,
    )
    first_shell = [
        float(value) for value, distance in rows if abs(float(distance) - 3.063) < 0.01
    ]
    assert len(first_shell) == 12
    assert max(first_shell) - min(first_shell) < 1.0e-3
