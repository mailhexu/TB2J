"""Finite-temperature fixed-N frozen response for metallic spiral pencils.

The grand-potential spectral kernel includes the diagonal Fermi derivative;
relaxing one global chemical potential adds the fixed-N Schur correction.
The executable authority is docs/sympy/siesta_spiral_metallic_response.py.
The gapped T=0 contour path remains in siesta_spiral_response.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import expit

from .siesta_spiral_response import (
    CURVATURE_BUDGET_EV_PER_RAD2,
    ENSEMBLE_METALLIC_FD,
    RESPONSE_CONVENTION,
    RESPONSE_VERSION,
    TORQUE_BUDGET_EV_PER_RAD,
    Ensemble,
    ResponseSet,
    SpiralResponseError,
    _all_ordered_pairs,
    _coord_text,
    _normalized_weights,
    _ordered,
)
from .siesta_spiral_vertex import VERTEX_CONVENTION

_METALLIC_COUNT_TOL = 1e-4
_FERMI_DERIVATIVE_MIN = 1e-12


def _occupations(eigenvalues, mu, width):
    return expit((mu - eigenvalues) / width)


def _count(eigenvalues_by_k, weights, mu, width):
    return sum(
        weight * float(_occupations(eig, mu, width).sum())
        for eig, weight in zip(eigenvalues_by_k, weights)
    )


def _solve_mu(eigenvalues_by_k, weights, target, width):
    total_states = sum(
        weight * len(eig) for eig, weight in zip(eigenvalues_by_k, weights)
    )
    if not 0.0 < target < total_states:
        raise SpiralResponseError(
            f"fixed-N electron count {target} must lie inside (0, {total_states})"
        )
    low = min(float(eig[0]) for eig in eigenvalues_by_k) - 50 * width
    high = max(float(eig[-1]) for eig in eigenvalues_by_k) + 50 * width
    return brentq(
        lambda mu: _count(eigenvalues_by_k, weights, mu, width) - target,
        low,
        high,
        xtol=1e-13,
    )


def _kernel(eigenvalues, occupations, derivative, mu, width):
    difference = eigenvalues[:, None] - eigenvalues[None, :]
    numerator = occupations[:, None] - occupations[None, :]
    out = np.empty_like(difference)
    separated = np.abs(difference) > max(1e-10, 1e-8 * width)
    np.divide(numerator, difference, out=out, where=separated)
    midpoint = (eigenvalues[:, None] + eigenvalues[None, :]) / 2
    midpoint_f = _occupations(midpoint, mu, width)
    out[~separated] = (-midpoint_f * (1 - midpoint_f) / width)[~separated]
    np.fill_diagonal(out, derivative)
    return out


def _free_energy(problem, pts, weights, ensemble, rotations):
    spectra = [
        eigh(*problem.rotated_pencil(k, rotations), eigvals_only=True) for k in pts
    ]
    mu = _solve_mu(
        spectra, weights, ensemble.electron_count, ensemble.smearing_width_ev
    )
    width = ensemble.smearing_width_ev
    omega = -width * sum(
        weight * float(np.logaddexp(0, (mu - eig) / width).sum())
        for eig, weight in zip(spectra, weights)
    )
    return omega + mu * ensemble.electron_count


def _finite_rotation_check(problem, coords, pts, weights, ensemble, analytic, step):
    if not np.isfinite(step) or step <= 0:
        raise SpiralResponseError("finite-rotation step must be positive and finite")

    def energy(angles):
        rotations = {}
        for (site, label), angle in angles.items():
            rotations.setdefault(site, {})[label] = angle
        return _free_energy(problem, pts, weights, ensemble, rotations)

    origin = energy({})
    gradient_error = curvature_error = 0.0
    for a in coords:
        plus = energy({a: step})
        minus = energy({a: -step})
        grad = (plus - minus) / (2 * step)
        diag = (plus - 2 * origin + minus) / step**2
        gradient_error = max(gradient_error, abs(grad - analytic["gradient"][a]))
        curvature_error = max(
            curvature_error, abs(diag - analytic["curvature"][(a, a)])
        )
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            mixed = (
                energy({a: step, b: step})
                - energy({a: step, b: -step})
                - energy({a: -step, b: step})
                + energy({a: -step, b: -step})
            ) / (4 * step**2)
            curvature_error = max(
                curvature_error, abs(mixed - analytic["curvature"][(a, b)])
            )
    return {
        "step": float(step),
        "torque_max_abs_mismatch": float(gradient_error),
        "curvature_max_abs_mismatch": float(curvature_error),
        "scope": "independent fixed-N Fermi-Dirac free-energy finite rotations; mu re-solved at every angle",
    }


def evaluate_metallic_response_set(
    problem,
    *,
    ensemble: Ensemble,
    coords,
    kpoints,
    k_weights,
    finite_steps=(4e-3, 2e-3),
):
    """Evaluate a declared metallic frozen response, refusing unmatched μ/N.

    The output uses the same raw ResponseSet contract as the gapped engine,
    but contour_vs_band_max_abs is None (no nonexistent contour identity).
    """
    ensemble.validate()
    if ensemble.kind != ENSEMBLE_METALLIC_FD:
        raise SpiralResponseError(
            "metallic evaluator requires a metallic fixed-N ensemble"
        )
    coords = problem._validate_coords(coords)
    pts, weights = _normalized_weights(kpoints, k_weights)
    width = ensemble.smearing_width_ev
    eigensystems = [eigh(*problem.pencil(k)) for k in pts]
    spectra = [e for e, _ in eigensystems]
    declared_count = _count(spectra, weights, ensemble.fermi_energy_ev, width)
    if abs(declared_count - ensemble.electron_count) > _METALLIC_COUNT_TOL:
        raise SpiralResponseError(
            f"declared Fermi energy gives electron count {declared_count:.12g}, "
            f"not {ensemble.electron_count:.12g} (budget {_METALLIC_COUNT_TOL:.1e})"
        )
    mu = _solve_mu(spectra, weights, ensemble.electron_count, width)
    occupations = [_occupations(e, mu, width) for e in spectra]
    derivatives = [-f * (1 - f) / width for f in occupations]
    d = sum(w * float(fp.sum()) for w, fp in zip(weights, derivatives))
    if not np.isfinite(d) or d >= -_FERMI_DERIVATIVE_MIN:
        raise SpiralResponseError(
            "fixed-N Fermi derivative is unresolved; increase the k mesh or width"
        )
    unique_pairs = [(a, b) for i, a in enumerate(coords) for b in coords[i:]]
    gradient = {coord: 0.0 for coord in coords}
    onsite = {pair: 0.0 for pair in unique_pairs}
    response = {pair: 0.0 for pair in unique_pairs}
    a_terms = {coord: 0.0 for coord in coords}
    for k, weight, (eig, c), f, fp in zip(
        pts, weights, eigensystems, occupations, derivatives
    ):
        vertex = {
            coord: c.conj().T @ problem.vertex_pencil(k, coord) @ c for coord in coords
        }
        diagonal = {coord: np.real(np.diag(v)) for coord, v in vertex.items()}
        kernel = _kernel(eig, f, fp, mu, width)
        for coord in coords:
            gradient[coord] += weight * float(f @ diagonal[coord])
            a_terms[coord] += weight * float(fp @ diagonal[coord])
        for pair in unique_pairs:
            a, b = pair
            if a[0] == b[0]:
                wx = c.conj().T @ problem.second_vertex_pencil(k, a, b) @ c
                onsite[pair] += weight * float(f @ np.real(np.diag(wx)))
            response[pair] += weight * float(
                np.sum((kernel * vertex[a] * vertex[b].T).real)
            )
    correction = {
        pair: -a_terms[pair[0]] * a_terms[pair[1]] / d for pair in unique_pairs
    }
    electronic_response = {
        pair: response[pair] + correction[pair] for pair in unique_pairs
    }
    curvature = {
        pair: onsite[pair] + electronic_response[pair] for pair in unique_pairs
    }
    analytic = {
        "gradient": gradient,
        "curvature": {
            p: curvature[_ordered(p, coords)] for p in _all_ordered_pairs(coords)
        },
        "onsite": {p: onsite[_ordered(p, coords)] for p in _all_ordered_pairs(coords)},
        "response": {
            p: electronic_response[_ordered(p, coords)]
            for p in _all_ordered_pairs(coords)
        },
        "correction": {
            p: correction[_ordered(p, coords)] for p in _all_ordered_pairs(coords)
        },
    }
    finite = [
        _finite_rotation_check(
            problem, coords, pts, weights, ensemble, analytic, float(step)
        )
        for step in finite_steps
    ]
    cross = (
        finite[-1]
        if finite
        else {
            "step": None,
            "torque_max_abs_mismatch": None,
            "curvature_max_abs_mismatch": None,
            "scope": "finite-rotation free-energy cross-check not requested",
        }
    )
    cert = dict(problem._vertex.certificate)
    scalar_digest = hashlib.sha256()
    for key in sorted(problem._bundle.sparse):
        h1, h2, _, _ = problem._bundle.sparse[key]
        scalar_digest.update(repr((key, (h1 + h2) / 2.0)).encode())
    phased_scalar_digest = hashlib.sha256()
    phased_amplitude_digest = hashlib.sha256()
    for k in pts:
        _h, _s, scalar, blocks = problem._phased_blocks(k)
        phased_scalar_digest.update(scalar.tobytes())
        for key in sorted(blocks):
            phased_amplitude_digest.update(blocks[key].tobytes())
    torque = {coord: -value for coord, value in gradient.items()}
    actual_gap = min(float(e[ensemble.n_occupied]) for e in spectra) - max(
        float(e[ensemble.n_occupied - 1]) for e in spectra
    )
    return ResponseSet(
        version=RESPONSE_VERSION,
        convention=RESPONSE_CONVENTION,
        operator_mode=problem.operator_mode,
        site_order=problem.site_order,
        axes={site: dict(problem._axes[site]) for site in problem.site_order},
        coords=coords,
        gradient=gradient,
        torque=torque,
        curvature=analytic["curvature"],
        curvature_split={
            pair: {
                "onsite": analytic["onsite"][pair],
                "response": analytic["response"][pair],
                "fixed_n_correction": analytic["correction"][pair],
            }
            for pair in analytic["curvature"]
        },
        contour_vs_band_max_abs=None,
        finite_rotation_cross_check=cross,
        frozen_declarations={
            "declaration": "ADR-005: density, scalar potential, magnetic amplitudes, basis and S frozen",
            "scalar_potential": "per-k spin-block trace/2 of phased H(k), frozen under local rotations",
            "stored_scalar_sha256": scalar_digest.hexdigest(),
            "scalar_sha256": phased_scalar_digest.hexdigest(),
            "magnetic_amplitudes": {
                "vertex_sha256": cert["vertex_sha256"],
                "source_bundle_sha256": cert["source_bundle_sha256"],
                "certificate": cert,
                "block_count": len(problem._vertex.blocks),
                "phased_blocks_sha256": phased_amplitude_digest.hexdigest(),
                "phase_placement": "operator mode and imported Bloch phases pinned by vertex certificate",
            },
            "overlap": "assembled spin-dependent S(k) frozen",
        },
        ensemble={
            **ensemble.to_record(),
            "chemical_potential_ev": float(mu),
            "electron_count_at_mu": _count(spectra, weights, mu, width),
            "electron_count_at_declared_fermi_ev": declared_count,
            "count_budget_electrons": _METALLIC_COUNT_TOL,
            "actual_gap_ev": actual_gap,
            "occupations_per_k": [f.tolist() for f in occupations],
            "fixed_n_fermi_derivative_ev_inverse": float(d),
        },
        provenance={
            "source_bundle_sha256": cert["source_bundle_sha256"],
            "vertex_sha256": cert["vertex_sha256"],
            "bundle_version": problem._bundle.version,
            "vertex_convention": VERTEX_CONVENTION,
            "operator_mode": problem.operator_mode,
            "q": list(problem._bundle.q),
            "derivation_authority": "docs/sympy/siesta_spiral_metallic_response.py",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metric": "S-orthonormal eigenbasis; fixed-N Fermi-Dirac free-energy Hessian",
        },
        integration={
            "method": "fixed_n_fermi_dirac_eigensystem",
            "kpoints": [np.asarray(k).tolist() for k in pts],
            "k_weights": list(weights),
            "contour": None,
        },
        torque_state={
            "declaration": "FR-008: nonzero reference torques retained; no stationarity identity applied",
            "nonzero_reference_torque": any(abs(v) > 1e-10 for v in torque.values()),
            "reference_torques": {_coord_text(c): v for c, v in torque.items()},
        },
        ladder={
            "budgets": {
                "torque_ev_per_rad": TORQUE_BUDGET_EV_PER_RAD,
                "curvature_ev_per_rad2": CURVATURE_BUDGET_EV_PER_RAD2,
            },
            "finite_rotation": finite,
            "contour_nodes": [],
            "contour_parameters": [],
            "k_mesh": [],
            "converged": {
                "torque": len(finite) > 1
                and finite[-1]["torque_max_abs_mismatch"] <= TORQUE_BUDGET_EV_PER_RAD,
                "curvature": len(finite) > 1
                and finite[-1]["curvature_max_abs_mismatch"]
                <= CURVATURE_BUDGET_EV_PER_RAD2,
            },
        },
    )
