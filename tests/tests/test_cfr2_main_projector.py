"""Absolute spectral references for main's direct CFR2 projector consumers."""

import numpy as np
from ase.units import kB
from scipy.special import expit

from TB2J.interfaces.abinit_savetb2j import _population_contour
from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    projector_charge_moments_from_green,
)


def test_collinear_projector_exchange_matches_complete_band_reference():
    up = np.array([[-0.3, 0.12 + 0.04j], [0.12 - 0.04j, -0.25]])
    down = np.array([[0.3, 0.12 + 0.04j], [0.12 - 0.04j, 0.25]])
    e_up, v_up = np.linalg.eigh(up)
    e_down, v_down = np.linalg.eigh(down)
    data = ProjectorGreenData(
        kpoints=np.zeros((1, 3)),
        weights=np.ones(1),
        eigenvalues=np.stack((e_up, e_down))[:, None, :],
        coefficients=np.stack((v_up.T, v_down.T))[:, None, :, :],
        efermi=0.0,
        projector_site=np.array([0, 1]),
        projector_atom=np.array([0, 1]),
        site_nproj=np.ones(2, dtype=int),
        site_projector_indices=np.array([[0], [1]]),
    )
    delta = np.real(np.diag(up - down))
    width = kB * 600
    actual = compute_projector_exchange_jdict(
        data,
        Rpts=[(0, 0, 0)],
        nz=80,
        smearing_eV=width,
        local_operators={i: np.array([[delta[i]]]) for i in range(2)},
    )

    f_up = expit(-e_up / width)
    f_down = expit(-e_down / width)
    fermi_difference = (f_up[:, None] - f_down[None, :]) / (
        e_up[:, None] - e_down[None, :]
    )
    for i in range(2):
        for j in range(2):
            # These are the complete band residues of G_up[i,j] G_down[j,i].
            residues = np.outer(
                v_up[i].conj() * v_up[j],
                v_down[j].conj() * v_down[i],
            )
            # CFR2 integrates the divided difference with pi normalization;
            # 1/(4*pi) in the exchange trace therefore leaves the physical 1/4.
            expected = (
                -delta[i] * delta[j] * np.sum(residues * fermi_difference).real / 4.0
            )
            np.testing.assert_allclose(
                actual[((0, 0, 0), i, j)], expected, atol=1.0e-10, rtol=0
            )


def test_abinit_green_population_matches_metric_resolved_band_sum(monkeypatch):
    monkeypatch.setenv("TB2J_GREEN_MODE", "contravariant")
    coefficients = np.array(
        [
            [[[1.0, 0.2 + 0.3j], [0.1j, 0.8]]],
            [[[0.8, 0.1 - 0.2j], [0.2j, 1.1]]],
        ]
    )
    eigenvalues = np.array([[[-0.2, 0.3]], [[-0.1, 0.4]]])
    overlap = np.array([[1.4, 0.2 + 0.1j], [0.2 - 0.1j, 1.1]])
    efermi = 0.07
    width = 0.05
    data = ProjectorGreenData(
        kpoints=np.zeros((1, 3)),
        weights=np.ones(1),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=efermi,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        site_nproj=np.array([2]),
        site_projector_indices=np.array([[0, 1]]),
        overlap_k=overlap[None, :, :],
    )

    actual = projector_charge_moments_from_green(
        ProjectorGreen(data),
        _population_contour(data, nz=80, smearing_eV=width),
    )

    inverse_overlap = np.linalg.inv(overlap)
    expected_by_spin = np.empty(2)
    for spin in range(2):
        band_population = []
        for coefficient_row in coefficients[spin, 0]:
            dual_row = coefficient_row.conj()
            band_population.append(np.vdot(dual_row, inverse_overlap @ dual_row).real)
        occupations = expit(-(eigenvalues[spin, 0] - efermi) / width)
        expected_by_spin[spin] = np.dot(occupations, band_population)

    np.testing.assert_allclose(
        actual["density_by_spin"][:, 0], expected_by_spin, atol=1.0e-10, rtol=0
    )
    np.testing.assert_allclose(
        actual["charges"], [np.sum(expected_by_spin)], atol=1.0e-10, rtol=0
    )
    np.testing.assert_allclose(
        actual["spinat"],
        [[0.0, 0.0, expected_by_spin[0] - expected_by_spin[1]]],
        atol=1.0e-10,
        rtol=0,
    )
