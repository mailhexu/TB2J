from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ase.units import Ha

from TB2J.interfaces import gpaw_projector


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
