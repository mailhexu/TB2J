"""Fixed-N Fermi-Dirac response: independent finite-rotation free-energy oracle."""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import expit

from TB2J.interfaces.siesta_spiral import BUNDLE_VERSION, SpiralBundle
from TB2J.interfaces.siesta_spiral_metallic import evaluate_metallic_response_set
from TB2J.interfaces.siesta_spiral_response import (
    Ensemble,
    FrozenResponseProblem,
    ResponseSet,
    SpiralResponseError,
)
from TB2J.interfaces.siesta_spiral_validation import validate_finite_rotations
from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex


def _problem():
    bundle = SpiralBundle(
        version=BUNDLE_VERSION,
        norb=2,
        ncell=1,
        q=(0.0,),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse={
            (0, 0, 0): (2.0, 0.5, 0.3, 0.1),
            (1, 1, 0): (1.5, -0.5, -0.2, 0.05),
            (0, 1, 0): (0.05, -0.02, 0.15, -0.08),
            (1, 0, 0): (0.05, -0.02, 0.15, -0.08),
        },
        overlap={
            (0, 0, 0): 1.0,
            (1, 1, 0): 1.0,
            (0, 1, 0): 0.05,
            (1, 0, 0): 0.05,
        },
    )
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    return FrozenResponseProblem(
        bundle,
        vertex,
        axes_by_site={"A": (("y", (0, 1, 0)),), "B": (("y", (0, 1, 0)),)},
    )


def _mu(eigenvalues, width, n_electrons):
    return brentq(
        lambda trial: expit((trial - eigenvalues) / width).sum() - n_electrons,
        eigenvalues.min() - 50 * width,
        eigenvalues.max() + 50 * width,
        xtol=1e-14,
    )


def _free_energy(problem, a, b, width=0.4, n_electrons=2.0):
    h, s = problem.rotated_pencil(0.0, {"A": {"y": a}, "B": {"y": b}})
    eigenvalues = eigh(h, s, eigvals_only=True)
    mu = _mu(eigenvalues, width, n_electrons)
    return -width * np.logaddexp(0, (mu - eigenvalues) / width).sum() + mu * n_electrons


def test_fixed_n_metallic_response_matches_independent_free_energy_fd():
    problem = _problem()
    eig = eigh(*problem.pencil(0.0), eigvals_only=True)
    width = 0.4
    mu = _mu(eig, width, 2.0)
    ensemble = Ensemble(
        kind="metallic_fermi_dirac_fixed_n",
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=mu,
        smearing_width_ev=width,
        gap_ev=float(eig[2] - eig[1]),
    )
    raw = evaluate_metallic_response_set(
        problem,
        ensemble=ensemble,
        coords=(("A", "y"), ("B", "y")),
        kpoints=(0.0,),
        k_weights=(1.0,),
        finite_steps=(4e-3, 2e-3),
    )
    h = 2e-3

    def energy(a, b):
        return _free_energy(problem, a, b, width=width)

    g_a = (energy(h, 0) - energy(-h, 0)) / (2 * h)
    k_aa = (energy(h, 0) - 2 * energy(0, 0) + energy(-h, 0)) / h**2
    k_ab = (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h)) / (4 * h**2)
    assert raw.gradient[("A", "y")] == pytest.approx(g_a, abs=1e-5)
    assert raw.curvature[(("A", "y"), ("A", "y"))] == pytest.approx(k_aa, abs=1e-4)
    assert raw.curvature[(("A", "y"), ("B", "y"))] == pytest.approx(k_ab, abs=1e-4)
    assert raw.ensemble["chemical_potential_ev"] == pytest.approx(mu, abs=1e-9)
    assert raw.ensemble["electron_count_at_mu"] == pytest.approx(2.0, abs=1e-10)
    assert 0.0 < raw.ensemble["occupations_per_k"][0][1] < 1.0
    assert raw.integration["method"] == "fixed_n_fermi_dirac_eigensystem"
    assert raw.contour_vs_band_max_abs is None  # not a fake contour identity
    assert ResponseSet.from_dict(raw.to_dict()).to_dict() == raw.to_dict()
    with pytest.raises(SpiralResponseError, match="gapped contour identity"):
        validate_finite_rotations(
            problem,
            raw,
            axes_by_site=raw.axes,
            reference_directions_by_site={"A": (0, 0, 1), "B": (0, 0, 1)},
            ensemble=ensemble,
            kpoints=(0.0,),
            k_weights=(1.0,),
        )


def test_metallic_ensemble_refuses_unmatched_producer_fermi_level():
    problem = _problem()
    eig = eigh(*problem.pencil(0.0), eigvals_only=True)
    good_mu = _mu(eig, 0.4, 2.0)
    ensemble = Ensemble(
        kind="metallic_fermi_dirac_fixed_n",
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=good_mu + 0.5,
        smearing_width_ev=0.4,
        gap_ev=float(eig[2] - eig[1]),
    )
    with pytest.raises(
        SpiralResponseError, match="electron count|Fermi|chemical potential"
    ):
        evaluate_metallic_response_set(
            problem,
            ensemble=ensemble,
            coords=(("A", "y"), ("B", "y")),
            kpoints=(0.0,),
            k_weights=(1.0,),
        )
