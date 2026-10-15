"""Story-012 mapping layer tests: response -> Heisenberg parameters (FR-009).

The normative authority is docs/sympy/spiral_heisenberg_mapping.py; the
synthetic tests below differentiate the SAME hand-built energies the
authority differentiates (exact SymPy, no finite differences), so the
mapping identities, ranks, null spaces, masquerade shifts and flagged
chi2 aggregates are checked against the derivation's exact numbers:

- exact Heisenberg: chi2 == 0, J recovered, J_pair_once = 2 J_stored;
- pure biquadratic: chi2_iso = 2752/3115 flagged at the planar
  (noncollinear) reference, exactly J-representable with J_hat = -2B at
  the collinear reference (masquerade);
- pure three-site: chi2_sector = 59/93, flagged through the onsite-block
  violation;
- zero pitch: J_stored = -K_ab/(2 alignment transverse_dot) reproduces
  the story-011 anchor number on the O2 fixture (311.528909 meV).

MathDoc: tb2j/spiral-heisenberg-mapping (paired authority + derivation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from scipy.linalg import eigh

from TB2J.interfaces.siesta_spiral_anchor import (
    DEFAULT_KMESH,
    _fractional_to_cartesian_inv_bohr,
    _oddize_kmesh,
    _site_moment_signs,
    run_frozen_response,
)
from TB2J.interfaces.siesta_spiral_io import KB_EV, load_fixture
from TB2J.interfaces.siesta_spiral_mapping import (
    CHI2_COMPAT_MAX,
    CHI2_FLAG_MIN,
    DEFAULT_CHI2_FLOOR_EV_PER_RAD2,
    RESIDUAL_COMPONENTS,
    MappingOutput,
    _compatibility_label,
    build_synthetic_response_set,
    map_response_set,
)
from TB2J.interfaces.siesta_spiral_response import (
    ENSEMBLE_GAPPED_T0,
    RESPONSE_IDENTITY_TOL_EV,
    EnergyContour,
    Ensemble,
    FrozenResponseProblem,
    SpiralResponseError,
    evaluate_response_set,
)
from TB2J.interfaces.siesta_spiral_vertex import extract_magnetic_vertex
from TB2J.kpoints import monkhorst_pack

J_SYN = sp.Rational(1, 10)
B_SYN = sp.Rational(1, 100)
K3_SYN = sp.Rational(3, 100)

X, Y, Z = (sp.Matrix(v) for v in np.eye(3).tolist())

#: Derivation anchors (site a carries the Cartesian frame; site b's
#: reference is tilted along t1_a inside the x-z plane, 3-4-5 exact).
COLLINEAR_FRAMES = {"A": (X, Y, Z), "B": (X, Y, Z)}
PLANAR_NB = sp.Matrix([sp.Rational(4, 5), 0, sp.Rational(3, 5)])
PLANAR_FRAMES = {"A": (X, Y, Z), "B": (Y, sp.simplify(PLANAR_NB.cross(Y)), PLANAR_NB)}
COLLINEAR_REFERENCES = {"A": (0.0, 0.0, 1.0), "B": (0.0, 0.0, 1.0)}
PLANAR_REFERENCES = {
    "A": (0.0, 0.0, 1.0),
    "B": (float(sp.Rational(4, 5)), 0.0, float(sp.Rational(3, 5))),
}
LABELS = ("1", "2")


def _rodrigues(axis, angle):
    axis = sp.Matrix(axis)
    n = axis.norm()
    skew = sp.Matrix(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return sp.eye(3) + sp.sin(angle) * skew / n + (1 - sp.cos(angle)) * (skew / n) ** 2


def _chart(frame, angles):
    """Two-axis composed exponential chart e = R2 R1 n (authority [A])."""
    t1, t2, n = frame
    return sp.simplify(_rodrigues(t2, angles[1]) * _rodrigues(t1, angles[0]) * n)


def _heisenberg(charts, coupling=J_SYN):
    return -coupling * (charts["A"].T * charts["B"])[0, 0]


def _biquadratic(charts, strength=B_SYN):
    return strength * (charts["A"].T * charts["B"])[0, 0] ** 2


def _three_site(charts, strength=K3_SYN):
    """Pure three-site term with a third site frozen at its reference."""
    e_c = sp.Matrix([sp.Rational(3, 5), sp.Rational(4, 5), 0])
    return (
        strength * ((charts["A"].T * charts["B"])[0, 0]) * ((charts["A"].T * e_c)[0, 0])
    )


def _synthetic_response(energy_builder, *, frames=COLLINEAR_FRAMES, q=(0.0, 0.0, 0.0)):
    """Differentiate a synthetic pair energy exactly and wrap a ResponseSet."""
    thetas = {
        site: sp.symbols(f"theta_{site}_1 theta_{site}_2", real=True) for site in frames
    }
    charts = {site: _chart(frame, thetas[site]) for site, frame in frames.items()}
    energy = sp.simplify(energy_builder(charts))
    zero = {theta: 0 for angles in thetas.values() for theta in angles}
    coords = [(site, label) for site in frames for label in LABELS]
    gradient, curvature = {}, {}
    for index, coord_a in enumerate(coords):
        theta_a = thetas[coord_a[0]][LABELS.index(coord_a[1])]
        gradient[tuple(coord_a)] = float(sp.diff(energy, theta_a).subs(zero).evalf())
        for coord_b in coords[index:]:
            theta_b = thetas[coord_b[0]][LABELS.index(coord_b[1])]
            value = float(sp.diff(energy, theta_a, theta_b).subs(zero).evalf())
            curvature[(tuple(coord_a), tuple(coord_b))] = value
            curvature[(tuple(coord_b), tuple(coord_a))] = value
    axes_by_site = {
        site: {
            label: [float(component) for component in frame[axis_index]]
            for axis_index, label in enumerate(LABELS)
        }
        for site, frame in frames.items()
    }
    return build_synthetic_response_set(
        site_order=tuple(frames),
        axes_by_site=axes_by_site,
        gradient=gradient,
        curvature=curvature,
        q=q,
    )


def _response_digest(response):
    return response.to_dict()["content_sha256"]


def _assert_input_preserved(response, digest_before):
    """The mapping never mutates the response it consumes."""
    assert _response_digest(response) == digest_before


# ---------------------------------------------------------------------------
# exact Heisenberg synthetics
# ---------------------------------------------------------------------------
def test_exact_heisenberg_collinear_zero_residual_and_stored_doubling():
    response = _synthetic_response(_heisenberg)
    digest = _response_digest(response)
    mapped = map_response_set(
        response, reference_directions_by_site=COLLINEAR_REFERENCES
    )
    _assert_input_preserved(response, digest)

    assert mapped.compatibility["label"] == "heisenberg_compatible"
    assert mapped.residuals["chi2_floor_eV_per_rad2"] == (
        DEFAULT_CHI2_FLOOR_EV_PER_RAD2
    )
    assert set(mapped.residuals["per_pair"]["A|B"]["components_eV_per_rad2"]) == set(
        RESIDUAL_COMPONENTS
    )
    assert mapped.residuals["chi2_iso"] == pytest.approx(0.0, abs=1e-15)
    assert mapped.residuals["chi2_sector"] == pytest.approx(0.0, abs=1e-15)
    assert all(
        abs(value) < 1e-12
        for pair in mapped.residuals["per_pair"].values()
        for value in pair["components_eV_per_rad2"].values()
    )

    pair = mapped.parameters["A|B"]
    assert pair["alignment"] == 1
    assert pair["alignment_source"] == "reference_directions_by_site"
    # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
    assert pair["j_pair_once_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["j_hat_scalar_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["scalar_extraction"]["method"] == "zero_pitch_identity"
    for axis in LABELS:
        entry = pair["scalar_extraction"]["per_axis"][axis]
        assert entry["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(0.0, abs=1e-12)
    # The unidentifiable D components are listed, never guessed.
    assert set(pair["dmi_determined_chart_eV"]) == {"D_z"}
    assert pair["dmi_determined_chart_eV"]["D_z"] == pytest.approx(0.0, abs=1e-12)

    ident = mapped.identifiability["per_pair"]["A|B"]
    assert mapped.identifiability["reference_class"] == "collinear_single_reference"
    assert ident["rank_mixed_block"] == 4
    assert ident["rank_onsite_closure"] == 5
    assert ident["rank_with_torque"] == 7
    assert set(ident["undetermined"]) == {"D_x", "D_y", "Gamma_xz", "Gamma_yz"}
    assert set(ident["determined"]) == {
        "J",
        "D_z",
        "Gamma_xx",
        "Gamma_xy",
        "Gamma_yy",
    }
    assert set(ident["nullspace_named"]) == {"D_x", "D_y", "Gamma_xz", "Gamma_yz"}
    assert ident["nullspace_generic"] == []
    assert set(ident["torque_nullspace_named"]) == {
        "trade_D_x_Gamma_yz",
        "trade_D_y_Gamma_xz",
    }
    assert ident["undetermined_trades"] == {
        "D_x": "trade_D_x_Gamma_yz",
        "D_y": "trade_D_y_Gamma_xz",
        "Gamma_yz": "trade_D_x_Gamma_yz",
        "Gamma_xz": "trade_D_y_Gamma_xz",
    }


def test_exact_heisenberg_planar_reference_completes_identifiability():
    response = _synthetic_response(
        _heisenberg, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0)
    )
    digest = _response_digest(response)
    mapped = map_response_set(response, reference_directions_by_site=PLANAR_REFERENCES)
    _assert_input_preserved(response, digest)

    assert mapped.identifiability["reference_class"] == "planar_single_q"
    assert mapped.compatibility["label"] == "heisenberg_compatible"
    assert mapped.residuals["chi2_iso"] == pytest.approx(0.0, abs=1e-15)
    assert mapped.residuals["chi2_sector"] == pytest.approx(0.0, abs=1e-15)

    pair = mapped.parameters["A|B"]
    assert pair["j_pair_once_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["scalar_extraction"]["method"] == "sector_fit"
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(0.0, abs=1e-12)
    assert pair["gamma_transverse_eV"]["22"] == pytest.approx(0.0, abs=1e-12)

    ident = mapped.identifiability["per_pair"]["A|B"]
    assert ident["rank_onsite_closure"] == 5
    assert ident["rank_with_torque"] == 8
    # Torque-completed planar reference: everything but the blind trade
    # members is determined, J included.
    assert "J" in ident["determined"]
    assert set(ident["undetermined"]) == {"D_x", "Gamma_yz"}
    assert ident["undetermined_trades"] == {
        "D_x": "trade_D_x_Gamma_yz",
        "Gamma_yz": "trade_D_x_Gamma_yz",
    }
    # The planar blind direction rotates with the tilt: exactly the
    # derivation's single tradeoff at the t1-tilted reference.
    assert set(ident["torque_nullspace_named"]) == {"trade_D_x_Gamma_yz"}


# ---------------------------------------------------------------------------
# biquadratic and three-site synthetics (flagged / masquerading)
# ---------------------------------------------------------------------------
def test_biquadratic_flagged_noncollinear_and_masquerades_collinear():
    planar_response = _synthetic_response(
        _biquadratic, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0)
    )
    planar_digest = _response_digest(planar_response)
    planar = map_response_set(
        planar_response, reference_directions_by_site=PLANAR_REFERENCES
    )
    _assert_input_preserved(planar_response, planar_digest)
    assert planar.compatibility["label"] == "incompatible"
    # MathDoc: tb2j/spiral-heisenberg-mapping#synthetic
    assert planar.residuals["chi2_iso"] == pytest.approx(2752.0 / 3115.0, rel=1e-9)
    assert planar.residuals["chi2_sector"] == pytest.approx(128.0 / 267.0, rel=1e-9)
    assert planar.residuals["chi2_iso"] >= CHI2_FLAG_MIN

    collinear = map_response_set(
        _synthetic_response(_biquadratic),
        reference_directions_by_site=COLLINEAR_REFERENCES,
    )
    # Invisible at the collinear anchor: exactly J-representable, and the
    # pure biquadratic masquerades as J_hat = -2B (n_a.n_b) = -2B.
    assert collinear.compatibility["label"] == "heisenberg_compatible"
    assert collinear.residuals["chi2_iso"] == pytest.approx(0.0, abs=1e-15)
    assert collinear.residuals["chi2_sector"] == pytest.approx(0.0, abs=1e-15)
    pair = collinear.parameters["A|B"]
    assert pair["j_hat_scalar_eV"] == pytest.approx(float(-2 * B_SYN), rel=1e-12)
    assert pair["j_pair_once_eV"] == pytest.approx(float(-2 * B_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(-B_SYN), rel=1e-12)


def test_three_site_flagged_via_onsite_closure_violation():
    response = _synthetic_response(
        _three_site, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0)
    )
    digest = _response_digest(response)
    mapped = map_response_set(response, reference_directions_by_site=PLANAR_REFERENCES)
    _assert_input_preserved(response, digest)
    assert mapped.compatibility["label"] == "incompatible"
    assert mapped.residuals["chi2_sector"] == pytest.approx(59.0 / 93.0, rel=1e-9)
    assert mapped.residuals["chi2_iso"] >= CHI2_FLAG_MIN
    pair_residuals = mapped.residuals["per_pair"]["A|B"]
    # The three-site term is flagged through the onsite-block violation.
    assert pair_residuals["onsite_max_abs_residual_eV_per_rad2"] > 1e-12
    closure = mapped.parameters["A|B"]["onsite_closure"]
    assert abs(closure["a"]["delta_eV_per_rad2"]) > 1e-12
    assert abs(closure["b"]["delta_eV_per_rad2"]) > 1e-12


# ---------------------------------------------------------------------------
# thresholds, digest gates and refusals
# ---------------------------------------------------------------------------
def test_threshold_constants_and_label_boundaries():
    assert CHI2_COMPAT_MAX == 1.0e-4
    assert CHI2_FLAG_MIN == 2.5e-3 == 1.0 / 400.0
    assert DEFAULT_CHI2_FLOOR_EV_PER_RAD2 == 1.0e-6
    assert _compatibility_label(0.0) == "heisenberg_compatible"
    assert _compatibility_label(CHI2_COMPAT_MAX) == "heisenberg_compatible"
    assert _compatibility_label(0.5 * (CHI2_COMPAT_MAX + CHI2_FLAG_MIN)) == "marginal"
    assert _compatibility_label(CHI2_FLAG_MIN) == "incompatible"


def test_mapping_output_round_trip_and_digest_gates():
    mapped = map_response_set(
        _synthetic_response(_heisenberg),
        reference_directions_by_site=COLLINEAR_REFERENCES,
    )
    record = mapped.to_dict()
    rebuilt = MappingOutput.from_dict(record)
    assert rebuilt == mapped
    assert rebuilt.raw_response.to_dict() == mapped.raw_response.to_dict()

    tampered_version = dict(record)
    tampered_version["version"] = "9.9"
    with pytest.raises(SpiralResponseError, match="version"):
        MappingOutput.from_dict(tampered_version)
    tampered = dict(record)
    tampered["compatibility"] = {**record["compatibility"], "chi2_iso": 0.5}
    with pytest.raises(SpiralResponseError, match="digest"):
        MappingOutput.from_dict(tampered)


def test_alignment_is_refused_not_guessed_without_references():
    response = _synthetic_response(_heisenberg)
    digest = _response_digest(response)
    mapped = map_response_set(response)
    _assert_input_preserved(response, digest)
    pair = mapped.parameters["A|B"]
    assert pair["alignment"] is None
    assert pair["alignment_source"] == "undetermined"
    assert pair["j_pair_once_eV"] is None
    assert pair["j_stored_eV"] is None
    assert pair["gamma_transverse_eV"] is None
    # Sign-independent verdicts are still delivered in full.
    assert mapped.residuals["chi2_iso"] == pytest.approx(0.0, abs=1e-15)
    assert mapped.compatibility["label"] == "heisenberg_compatible"
    assert "torque cannot substitute" in mapped.identifiability["alignment"]["note"]


def test_reference_class_refusals():
    with pytest.raises(SpiralResponseError, match="reference"):
        map_response_set(
            _synthetic_response(_heisenberg, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0))
        )
    with pytest.raises(SpiralResponseError, match="floor"):
        map_response_set(
            _synthetic_response(_heisenberg),
            reference_directions_by_site=COLLINEAR_REFERENCES,
            chi2_floor_eV_per_rad2=-1.0,
        )
    # q = 0 with non-collinear certified references is refused (the
    # references are transverse to their own frames but not parallel).
    with pytest.raises(SpiralResponseError, match="collinear"):
        map_response_set(
            _synthetic_response(_heisenberg, frames=PLANAR_FRAMES),
            reference_directions_by_site=PLANAR_REFERENCES,
        )
    # Nonplanar finite-q references need the multi-reference treatment.
    three_site_frames = {
        "A": (X, Y, Z),
        "B": (Y, sp.simplify(PLANAR_NB.cross(Y)), PLANAR_NB),
        "C": (X, -Z, Y),
    }
    thetas = {
        site: sp.symbols(f"theta_{site}_1 theta_{site}_2", real=True)
        for site in three_site_frames
    }
    charts = {
        site: _chart(frame, thetas[site]) for site, frame in three_site_frames.items()
    }
    zero = {theta: 0 for angles in thetas.values() for theta in angles}
    labels = LABELS
    coords = [(site, label) for site in three_site_frames for label in labels]
    energy = _heisenberg(charts)
    gradient = {
        tuple(c): float(
            sp.diff(energy, thetas[c[0]][labels.index(c[1])]).subs(zero).evalf()
        )
        for c in coords
    }
    curvature = {}
    for i, ca in enumerate(coords):
        for cb in coords[i:]:
            value = float(
                sp.diff(
                    energy,
                    thetas[ca[0]][labels.index(ca[1])],
                    thetas[cb[0]][labels.index(cb[1])],
                )
                .subs(zero)
                .evalf()
            )
            curvature[(tuple(ca), tuple(cb))] = value
            curvature[(tuple(cb), tuple(ca))] = value
    axes = {
        site: {
            label: [float(v) for v in frame[axis_index]]
            for axis_index, label in enumerate(labels)
        }
        for site, frame in three_site_frames.items()
    }
    response = build_synthetic_response_set(
        site_order=tuple(three_site_frames),
        axes_by_site=axes,
        gradient=gradient,
        curvature=curvature,
        q=(0.25, 0.0, 0.0),
    )
    with pytest.raises(SpiralResponseError, match="multi-reference"):
        map_response_set(
            response,
            reference_directions_by_site={
                "A": (0.0, 0.0, 1.0),
                "B": (float(sp.Rational(4, 5)), 0.0, float(sp.Rational(3, 5))),
                "C": (0.0, 1.0, 0.0),
            },
        )


# ---------------------------------------------------------------------------
# zero-pitch reduction on the real O2 fixture (story-011 anchor identity)
# ---------------------------------------------------------------------------
FIXTURE_DIR = Path(__file__).parent / "siesta_spiral_fixtures" / "siesta_collinear_o2"
FIXTURE_NPZ = FIXTURE_DIR / "o2_fm_q0.npz"
SITES = ("O1", "O2")
AXES = ("y", "z")
#: The story-011 anchor number for the repaired response on the 1x1x5 mesh.
ANCHOR_J_STORED_MEV = 311.528909


def _o2_problem(bundle, orbitals_by_site):
    vertex = extract_magnetic_vertex(bundle, orbitals_by_site=orbitals_by_site)
    axes_by_site = {
        site: tuple(
            (
                label,
                {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[
                    label
                ],
            )
            for label in AXES
        )
        for site in SITES
    }
    problem = FrozenResponseProblem(bundle, vertex, axes_by_site=axes_by_site)
    return problem, axes_by_site


@pytest.fixture(scope="module")
def o2_response():
    bundle, reference = load_fixture(FIXTURE_NPZ)
    norb = bundle.norb
    orbitals_by_site = {
        "O1": list(range(norb // 2)),
        "O2": list(range(norb // 2, norb)),
    }
    problem, axes_by_site = _o2_problem(bundle, orbitals_by_site)
    kmesh = _oddize_kmesh(list(DEFAULT_KMESH))
    kpoints = _fractional_to_cartesian_inv_bohr(
        monkhorst_pack(kmesh, gamma_center=False), bundle.cell
    )
    weights = np.full(len(kpoints), 1.0 / len(kpoints))
    n_occupied = int(round(float(reference["electron_count"])))
    homos, lumos, lowest = [], [], []
    for k in kpoints:
        h, s = problem.pencil(k)
        eigenvalues = eigh(h, s, eigvals_only=True)
        homos.append(eigenvalues[n_occupied - 1])
        lumos.append(eigenvalues[n_occupied])
        lowest.append(eigenvalues[0])
    homo_max, lumo_min = max(homos), min(lumos)
    # The reference gate compares against the indirect gap of the mesh
    # (min lumo minus max homo), never the per-k direct gap.
    gap_ev = float(lumo_min - homo_max)
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=n_occupied,
        electron_count=float(reference["electron_count"]),
        fermi_energy_ev=float(reference["ef"]),
        smearing_width_ev=KB_EV * float(reference["temperature"]),
        gap_ev=float(gap_ev),
    )
    contour = EnergyContour.declared(
        z0=float(0.5 * (homo_max + lumo_min)),
        occupied_min=float(min(lowest)),
        gap_ev=float(gap_ev),
        margin_ev=0.8,
        n_nodes=1024,
    )
    coords = tuple((site, axis) for site in SITES for axis in AXES)
    response = evaluate_response_set(
        problem,
        ensemble=ensemble,
        coords=coords,
        kpoints=[tuple(k) for k in kpoints],
        k_weights=[float(w) for w in weights],
        contour=contour,
        finite_steps=(4e-3, 2e-3),
    )
    assert response.contour_vs_band_max_abs <= RESPONSE_IDENTITY_TOL_EV
    return bundle, reference, orbitals_by_site, axes_by_site, kpoints, weights, response


def test_o2_zero_pitch_reduction_matches_story011_anchor(o2_response):
    bundle, reference, orbitals_by_site, axes_by_site, kpoints, weights, response = (
        o2_response
    )
    digest = _response_digest(response)
    mapped = map_response_set(
        response,
        reference_directions_by_site={"O1": (1.0, 0.0, 0.0), "O2": (1.0, 0.0, 0.0)},
    )
    _assert_input_preserved(response, digest)

    pair = mapped.parameters["O1|O2"]
    assert pair["alignment"] == 1
    assert pair["scalar_extraction"]["method"] == "zero_pitch_identity"
    assert 1e3 * pair["j_stored_eV"] == pytest.approx(ANCHOR_J_STORED_MEV, abs=1e-6)
    assert 1e3 * pair["j_pair_once_eV"] == pytest.approx(
        1e3 * 2.0 * pair["j_stored_eV"], rel=1e-12
    )

    # Cross-seam: the story-011 anchor projection on the same mesh.
    moment_signs = _site_moment_signs(bundle, orbitals_by_site)
    assert moment_signs["O1"] * moment_signs["O2"] == 1.0
    _problem, _ensemble, _obs, j_by_axis = run_frozen_response(
        bundle,
        reference,
        orbitals_by_site,
        axes_by_site,
        kpoints,
        [float(w) for w in weights],
        AXES,
        SITES,
        moment_signs,
    )
    for axis in AXES:
        anchor = j_by_axis[axis]["O1|O2"]["j_stored_eV"]
        per_axis = pair["scalar_extraction"]["per_axis"][axis]
        assert per_axis["j_stored_eV"] == pytest.approx(anchor, abs=1e-12)

    assert mapped.provenance["operator_mode"] == "onsite_delta"
    assert mapped.provenance["q"] == [0.0, 0.0, 0.0]
    assert mapped.provenance["vertex_sha256"] == response.provenance["vertex_sha256"]
    assert (
        mapped.provenance["source_bundle_sha256"]
        == response.provenance["source_bundle_sha256"]
    )
    assert mapped.provenance["ensemble_kind"] == ENSEMBLE_GAPPED_T0
    assert mapped.provenance["response_sha256"] == digest
    assert mapped.identifiability["reference_class"] == "collinear_single_reference"


def test_o2_round_trip_preserves_raw_response(o2_response):
    _bundle, _reference, _orbitals, _axes, _kpoints, _weights, response = o2_response
    mapped = map_response_set(
        response,
        reference_directions_by_site={"O1": (1.0, 0.0, 0.0), "O2": (1.0, 0.0, 0.0)},
    )
    record = mapped.to_dict()
    rebuilt = MappingOutput.from_dict(record)
    assert rebuilt.provenance == mapped.provenance
    assert rebuilt.parameters == mapped.parameters
    assert rebuilt.raw_response.to_dict() == response.to_dict()
    assert rebuilt.raw_response == response


def test_o2_without_references_reports_alignment_undetermined(o2_response):
    _bundle, _reference, _orbitals, _axes, _kpoints, _weights, response = o2_response
    mapped = map_response_set(response)
    pair = mapped.parameters["O1|O2"]
    assert pair["alignment"] is None
    assert pair["j_stored_eV"] is None
    assert mapped.compatibility["label"] in {
        "heisenberg_compatible",
        "marginal",
        "incompatible",
    }
    assert mapped.residuals["chi2_sector"] >= 0.0
