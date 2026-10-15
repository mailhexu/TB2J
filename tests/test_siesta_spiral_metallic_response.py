"""Fixed-N Fermi-Dirac response: independent finite-rotation free-energy oracle."""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import expit

from TB2J.interfaces.siesta_spiral import BUNDLE_VERSION, SpiralBundle
from TB2J.interfaces.siesta_spiral_mapping import (
    compare_mapping_outputs,
    map_response_set,
)
from TB2J.interfaces.siesta_spiral_metallic import (
    evaluate_metallic_response_set,
    recenter_metallic_reference,
)
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


def test_real_recentered_frozen_operator_comparison_changes_mu_not_identity():
    scalar = _problem().bundle
    origin = (0, 0, 0)
    bundle = SpiralBundle(
        version=scalar.version,
        norb=scalar.norb,
        ncell=scalar.ncell,
        q=(0.0, 0.0, 0.25),
        spin_packing=scalar.spin_packing,
        phase_convention=scalar.phase_convention,
        hermitization=scalar.hermitization,
        fermi_shift=scalar.fermi_shift,
        pitch_sign=scalar.pitch_sign,
        sparse={(mu, nu, origin): v for (mu, nu, _r), v in scalar.sparse.items()},
        overlap={(mu, nu, origin): v for (mu, nu, _r), v in scalar.overlap.items()},
        cell=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        orbital_positions=((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site={"A": [0], "B": [1]})
    axes = {
        site: (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0))) for site in ("A", "B")
    }
    problem = FrozenResponseProblem(bundle, vertex, axes_by_site=axes)
    k = (0.0, 0.0, 0.0)
    eig = eigh(*problem.pencil(k), eigvals_only=True)
    width = 0.4
    ensemble = Ensemble(
        kind="metallic_fermi_dirac_fixed_n",
        n_occupied=2,
        electron_count=2.0,
        fermi_energy_ev=_mu(eig, width, 2.0),
        smearing_width_ev=width,
        gap_ev=float(eig[2] - eig[1]),
    )
    coords = (("A", "x"), ("A", "y"), ("B", "x"), ("B", "y"))
    root = evaluate_metallic_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=(k,),
        k_weights=(1.0,),
        finite_steps=(4e-3, 2e-3),
    )
    view, normals = recenter_metallic_reference(
        problem,
        angles_by_site={"B": {"y": 0.5}},
        reference_directions_by_site={"A": (0.0, 0.0, 1.0), "B": (0.0, 0.0, 1.0)},
    )
    h_center, s_center = view.pencil(k)
    h_independent, s_independent = problem.rotated_pencil(k, {"B": {"y": 0.5}})
    np.testing.assert_allclose(h_center, h_independent, atol=1e-12)
    with pytest.raises(SpiralResponseError, match="unknown sites"):
        view.rotated_pencil(k, {"typo": {"y": 0.1}})
    np.testing.assert_allclose(s_center, s_independent, atol=1e-12)
    step = 1e-5
    h_plus, _ = view.rotated_pencil(k, {"B": {"y": step}})
    h_minus, _ = view.rotated_pencil(k, {"B": {"y": -step}})
    np.testing.assert_allclose(
        view.vertex_pencil(k, ("B", "y")), (h_plus - h_minus) / (2 * step), atol=1e-7
    )
    shifted = evaluate_metallic_response_set(
        view,
        ensemble=ensemble,
        coords=coords,
        kpoints=(k,),
        k_weights=(1.0,),
        finite_steps=(4e-3, 2e-3),
    )
    assert shifted.ensemble["electron_count_at_mu"] == pytest.approx(2.0, abs=1e-10)
    assert (
        abs(
            shifted.ensemble["chemical_potential_ev"]
            - root.ensemble["chemical_potential_ev"]
        )
        > 1e-4
    )
    assert (
        shifted.provenance["source_bundle_sha256"]
        == root.provenance["source_bundle_sha256"]
    )
    assert shifted.provenance["reference_center_rotation_by_site"]["B"]["y"] == 0.5
    assert all(root.ladder["converged"].values())
    assert all(shifted.ladder["converged"].values())
    mapped_root = map_response_set(
        root, reference_directions_by_site={"A": (0, 0, 1), "B": (0, 0, 1)}
    )
    mapped_shifted = map_response_set(shifted, reference_directions_by_site=normals)
    comparison = compare_mapping_outputs(mapped_root, mapped_shifted)
    assert comparison["status"] == "matched_frozen_reference_diagnostic"
    assert abs(comparison["per_pair"]["A|B"]["delta_j_pair_once_eV"]) > 1e-8
    assert comparison["mapping"]["left"]["response"] == root.to_dict()
    assert comparison["mapping"]["right"]["response"] == shifted.to_dict()
    unchecked = evaluate_metallic_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=(k,),
        k_weights=(1.0,),
        finite_steps=(),
    )
    unchecked_mapping = map_response_set(
        unchecked,
        reference_directions_by_site={"A": (0, 0, 1), "B": (0, 0, 1)},
    )
    refused = compare_mapping_outputs(unchecked_mapping, mapped_shifted)
    assert refused["status"] == "uncertified_reference"
    assert refused["per_pair"] == {}


def test_metallic_ensemble_refuses_occupied_index_at_band_count():
    problem = _problem()
    with pytest.raises(SpiralResponseError, match="n_occupied|band count"):
        evaluate_metallic_response_set(
            problem,
            ensemble=Ensemble("metallic_fermi_dirac_fixed_n", 4, 2.0, 0.0, 0.4, 0.0),
            coords=(("A", "y"),),
            kpoints=(0.0,),
            k_weights=(1.0,),
            finite_steps=(),
        )
