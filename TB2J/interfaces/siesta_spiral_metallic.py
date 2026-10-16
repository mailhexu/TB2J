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
from scipy.linalg import eigh, expm
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.special import expit

from .siesta_spiral_response import (
    CURVATURE_BUDGET_EV_PER_RAD2,
    ENSEMBLE_GAP_TOL_EV,
    ENSEMBLE_METALLIC_FD,
    RESPONSE_CONVENTION,
    RESPONSE_VERSION,
    TORQUE_BUDGET_EV_PER_RAD,
    Ensemble,
    ResponseSet,
    SpiralResponseError,
    _all_ordered_pairs,
    _comm,
    _coord_text,
    _half_generator,
    _normalized_weights,
    _ordered,
    _pair_text,
    _require_hermitian,
    second_vertex_blocks,
)
from .siesta_spiral_vertex import OPERATOR_MODE_ONSITE_DELTA, VERTEX_CONVENTION

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
    separated = np.abs(difference) > 1e-8 * width
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
    gradient_errors, curvature_errors = {}, {}
    for a in coords:
        plus = energy({a: step})
        minus = energy({a: -step})
        grad = (plus - minus) / (2 * step)
        diag = (plus - 2 * origin + minus) / step**2
        gradient_errors[_coord_text(a)] = float(grad - analytic["gradient"][a])
        curvature_errors[_pair_text((a, a))] = float(
            diag - analytic["curvature"][(a, a)]
        )
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            mixed = (
                energy({a: step, b: step})
                - energy({a: step, b: -step})
                - energy({a: -step, b: step})
                + energy({a: -step, b: -step})
            ) / (4 * step**2)
            curvature_errors[_pair_text((a, b))] = float(
                mixed - analytic["curvature"][(a, b)]
            )
    return {
        "step": float(step),
        "torque_max_abs_mismatch": max(map(abs, gradient_errors.values())),
        "curvature_max_abs_mismatch": max(map(abs, curvature_errors.values())),
        "torque_component_errors": gradient_errors,
        "curvature_component_errors": curvature_errors,
        "scope": "independent fixed-N Fermi-Dirac free-energy finite rotations; mu re-solved at every angle",
    }


def _stable_ladder(finite, field, budget):
    if len(finite) < 2 or finite[-1][field + "_max_abs_mismatch"] > budget:
        return False
    return finite[-1][field + "_max_delta_vs_previous"] <= budget


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
    if not 0 < ensemble.n_occupied < 2 * problem._bundle.norb:
        raise SpiralResponseError(
            "n_occupied must be below the spinor band count in a metallic reference"
        )
    width = ensemble.smearing_width_ev
    eigensystems = [eigh(*problem.pencil(k)) for k in pts]
    spectra = [e for e, _ in eigensystems]
    root_problem = getattr(problem, "_base_problem", problem)
    root_spectra = (
        spectra
        if root_problem is problem
        else [eigh(*root_problem.pencil(k), eigvals_only=True) for k in pts]
    )
    root_gap = min(float(e[ensemble.n_occupied]) for e in root_spectra) - max(
        float(e[ensemble.n_occupied - 1]) for e in root_spectra
    )
    if abs(root_gap - ensemble.gap_ev) > ENSEMBLE_GAP_TOL_EV:
        raise SpiralResponseError(
            f"declared metallic reference gap {ensemble.gap_ev:.12g} eV "
            f"differs from root spectrum {root_gap:.12g} eV "
            f"(budget {ENSEMBLE_GAP_TOL_EV:.1e} eV)"
        )
    declared_count = _count(root_spectra, weights, ensemble.fermi_energy_ev, width)
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
    # MathDoc: tb2j/siesta-spiral-metallic-response#schur_sign
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
    for index, rung in enumerate(finite):
        for field in ("torque", "curvature"):
            errors = rung[field + "_component_errors"]
            previous = finite[index - 1][field + "_component_errors"] if index else None
            rung[field + "_max_delta_vs_previous"] = (
                max(abs(value - previous[key]) for key, value in errors.items())
                if previous is not None
                else None
            )
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
            "root_electron_count_at_producer_fermi_ev": declared_count,
            "electron_count_at_declared_fermi_ev": _count(
                spectra, weights, ensemble.fermi_energy_ev, width
            ),
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
            **{key: cert[key] for key in ("vertex_kind", "hubbard") if key in cert},
            "derivation_authority": "docs/sympy/siesta_spiral_metallic_response.py",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "metric": "S-orthonormal eigenbasis; fixed-N Fermi-Dirac free-energy Hessian",
            "reference_center_rotation_by_site": getattr(
                problem, "_reference_center_rotation", {}
            ),
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
                "torque": _stable_ladder(finite, "torque", TORQUE_BUDGET_EV_PER_RAD),
                "curvature": _stable_ladder(
                    finite, "curvature", CURVATURE_BUDGET_EV_PER_RAD2
                ),
            },
        },
    )


class _RecenteredProblem:
    """Same frozen onsite-Delta bundle, evaluated in a shifted local chart.

    It never rewrites the producer bundle. The root H/S and vertex
    certificate remain immutable; only the selected R=0 site magnetic
    blocks are conjugated by the declared center rotation. New local
    derivatives act on those already-rotated blocks, not on the old chart.
    """

    def __init__(self, base, center_unitaries, axes, center_rotation):
        self._base_problem = base
        self._bundle = base._bundle
        self._vertex = base._vertex
        self._axes = axes
        self._unitaries = center_unitaries
        self._reference_center_rotation = center_rotation
        self._center_cache = {}
        self._phased_cache = {}

    @property
    def site_order(self):
        return self._base_problem.site_order

    @property
    def operator_mode(self):
        return self._base_problem.operator_mode

    def _validate_coords(self, coords):
        return self._base_problem._validate_coords(coords)

    def _centered_blocks(self, k):
        cache_key = tuple(np.asarray(k, dtype=float).ravel())
        if cache_key not in self._center_cache:
            frozen, original = self._base_problem._onsite_decomposition(k)
            centered = {}
            for site in self.site_order:
                unitary = self._unitaries[site]
                for key in sorted(self._base_problem._key_partition[site]):
                    block = self._base_problem._stored_block(original[key])
                    centered[key] = (
                        *original[key][:2],
                        unitary @ block @ unitary.conj().T,
                    )
            self._center_cache[cache_key] = frozen, centered
        return self._center_cache[cache_key]

    def _incremental_unitary(self, site, thetas):
        generator = np.zeros((2, 2), dtype=complex)
        for label, angle in thetas.get(site, {}).items():
            if label not in self._axes[site] or not np.isfinite(angle):
                raise SpiralResponseError(
                    f"unknown or nonfinite rotation {site!r}/{label!r}"
                )
            generator += float(angle) * _half_generator(self._axes[site][label])
        return expm(-1j * generator) if np.any(generator) else np.eye(2, dtype=complex)

    def rotated_pencil(self, k, thetas):
        if set(thetas) - set(self.site_order):
            raise SpiralResponseError("rotation contains unknown sites")
        frozen, centered = self._centered_blocks(k)
        h = frozen.copy()
        for site in self.site_order:
            unitary = self._incremental_unitary(site, thetas)
            for key in sorted(self._base_problem._key_partition[site]):
                d11, d22, block = centered[key]
                rotated = unitary @ block @ unitary.conj().T
                self._base_problem._place_key(
                    h,
                    key,
                    d11,
                    d22,
                    rotated[0, 0],
                    rotated[0, 1],
                    rotated[1, 0],
                    rotated[1, 1],
                )
        _h0, s = self._base_problem.pencil(k)
        return _require_hermitian(h, "recentered H(k)"), s

    def pencil(self, k):
        return self.rotated_pencil(k, {})

    def vertex_pencil(self, k, coord):
        site, label = coord
        self._base_problem.axis_of(coord)
        _frozen, centered = self._centered_blocks(k)
        t = _half_generator(self._axes[site][label])
        out = np.zeros((2 * self._bundle.norb,) * 2, dtype=complex)
        for key in sorted(self._base_problem._key_partition[site]):
            d11, d22, block = centered[key]
            vertex = -1j * _comm(t, block)
            self._base_problem._place_key(
                out,
                key,
                d11,
                d22,
                vertex[0, 0],
                vertex[0, 1],
                vertex[1, 0],
                vertex[1, 1],
            )
        return _require_hermitian(out, "recentered V(k)")

    def second_vertex_pencil(self, k, coord_a, coord_b):
        a, b = self._base_problem._ordered_pair(coord_a, coord_b)
        out = np.zeros((2 * self._bundle.norb,) * 2, dtype=complex)
        if a[0] == b[0]:
            _frozen, centered = self._centered_blocks(k)
            for key in sorted(self._base_problem._key_partition[a[0]]):
                d11, d22, block = centered[key]
                seagull = second_vertex_blocks(
                    {key: block},
                    (key,),
                    self._axes[a[0]][a[1]],
                    self._axes[b[0]][b[1]],
                )[key]
                self._base_problem._place_key(
                    out,
                    key,
                    d11,
                    d22,
                    seagull[0, 0],
                    seagull[0, 1],
                    seagull[1, 0],
                    seagull[1, 1],
                )
        return _require_hermitian(out, "recentered W(k)")

    def _phased_blocks(self, k):
        cache_key = tuple(np.asarray(k, dtype=float).ravel())
        if cache_key not in self._phased_cache:
            h, s = self.pencil(k)
            n = self._bundle.norb
            scalar = np.zeros_like(h)
            blocks = {}
            for mu in range(n):
                for nu in range(n):
                    indices = np.ix_([mu, mu + n], [nu, nu + n])
                    block = h[indices]
                    trace = np.trace(block) / 2 * np.eye(2)
                    scalar[indices] = trace
                    blocks[mu, nu] = block - trace
            self._phased_cache[cache_key] = h, s, scalar, blocks
        return self._phased_cache[cache_key]


def recenter_metallic_reference(
    problem, *, angles_by_site, reference_directions_by_site
):
    """Shift an onsite-Delta frozen operator without changing its SCF root.

    Return a recentered problem and certified rotated site normals. The
    caller evaluates a new fixed-N response with the ORIGINAL producer
    ensemble declaration; its chemical potential is re-solved for the
    rotated reference. Weighted-total-H is deliberately not inferred.
    """
    if problem.operator_mode != OPERATOR_MODE_ONSITE_DELTA:
        raise SpiralResponseError("recentered metallic reference requires onsite_delta")
    if set(reference_directions_by_site) != set(problem.site_order):
        raise SpiralResponseError("recenter requires every certified root site normal")
    if not angles_by_site or any(
        site not in problem.site_order for site in angles_by_site
    ):
        raise SpiralResponseError("recenter needs a declared rotation of a known site")
    axes, unitary_by_site, normals, rotation_record = {}, {}, {}, {}
    nonzero = False
    pauli = np.array(
        [
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    )
    for site in problem.site_order:
        chart = problem._axes[site]
        if len(chart) != 2:
            raise SpiralResponseError("recenter needs two transverse axes per site")
        normal = np.asarray(reference_directions_by_site[site], dtype=float)
        vectors = [np.asarray(v, dtype=float) for v in chart.values()]
        if (
            normal.shape != (3,)
            or not np.isfinite(normal).all()
            or abs(np.linalg.norm(normal) - 1) > 1e-6
            or any(abs(np.dot(v, normal)) > 1e-6 for v in vectors)
            or np.linalg.norm(np.cross(*vectors) - normal) > 1e-6
        ):
            raise SpiralResponseError(
                f"uncertified transverse root chart at site {site!r}"
            )
        rotations = angles_by_site.get(site, {})
        if any(
            label not in chart or not np.isfinite(value)
            for label, value in rotations.items()
        ):
            raise SpiralResponseError(
                f"unknown or nonfinite recenter rotation at {site!r}"
            )
        rotvec = sum(
            (
                float(value) * np.asarray(chart[label], dtype=float)
                for label, value in rotations.items()
            ),
            np.zeros(3),
        )
        nonzero |= bool(np.linalg.norm(rotvec) > 0)
        rot = Rotation.from_rotvec(rotvec)
        unitary = expm(
            -1j
            * sum(
                (
                    float(value) * _half_generator(chart[label])
                    for label, value in rotations.items()
                ),
                np.zeros((2, 2), complex),
            )
        )
        rotated_normal = rot.apply(normal)
        if not np.allclose(
            unitary @ np.tensordot(normal, pauli, axes=1) @ unitary.conj().T,
            np.tensordot(rotated_normal, pauli, axes=1),
            atol=1e-11,
            rtol=0,
        ):
            raise SpiralResponseError("SU(2) recenter and spatial normal disagree")
        axes[site] = {label: tuple(rot.apply(v)) for label, v in chart.items()}
        normals[site] = tuple(rotated_normal)
        unitary_by_site[site] = unitary
        rotation_record[site] = {
            label: float(value) for label, value in rotations.items()
        }
    if not nonzero:
        raise SpiralResponseError("recenter rotation must move at least one site")
    return _RecenteredProblem(problem, unitary_by_site, axes, rotation_record), normals
