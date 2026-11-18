"""Story-012 mapping layer tests: response -> Heisenberg parameters (FR-009).

The normative authority is the CORRECTED
docs/sympy/spiral_heisenberg_mapping.py; the synthetic tests below
differentiate the SAME hand-built energies the authority differentiates
(exact SymPy, no finite differences), so the mapping identities, ranks,
null spaces, masquerade shifts and flagged chi2 aggregates are checked
against the derivation's exact numbers:

- exact Heisenberg: chi2 == 0 (Hessian and measured torque), J recovered,
  J_pair_once = 2 J_stored;
- collinear anisotropic exact pair (J = 0.1, Gamma_xx = 0.01): the
  exported J comes from the determined joint fit (J = 0.1, never the
  common-axis scalar 0.095), which survives only as the j_hat_scalar
  diagnostic;
- AFM charts (n_b = -n_a, same site-local axis): the zero-pitch identity
  divides by the LOCAL tilt dot only (u_a.u_b = -1), so the exact AFM
  Heisenberg pair stores j_stored = +J/2;
- pure biquadratic: flagged at the planar (noncollinear) reference,
  exactly J-representable with J_hat = -2B at the collinear reference
  (masquerade); the K-only projection diagnostics reproduce the
  derivation's chi2_iso = 2752/3115 and chi2_sector = 128/267;
- pure three-site term (frozen third direction): flagged through the
  onsite-block violation, K-only chi2_sector = 59/93 -- distinct from the
  exact THREE-SITE Heisenberg chain, which the joint multisite fit maps
  with zero residual;
- zero pitch: the scalar common-axis identity (j_hat_scalar_eV and its
  per-axis entries) reproduces the story-011 anchor number on the O2
  fixture (311.528909 meV stored, < 1e-6 meV); the exported j_stored is
  the determined fitted J, which on this fixture resolves the onsite
  block's single-site structure into an isotropic Gamma component
  (J_pair_once = (K_ss - 2 K_ab)/3).

MathDoc: tb2j/spiral-heisenberg-mapping (paired authority + derivation).
"""

from __future__ import annotations

from itertools import combinations
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
    CHART_PARAMETER_NAMES,
    CHI2_COMPAT_MAX,
    CHI2_FLAG_MIN,
    DEFAULT_CHI2_FLOOR_EV_PER_RAD2,
    DEFAULT_TORQUE_CHI2_FLOOR_EV_PER_RAD,
    MIXED_RESIDUAL_COMPONENTS,
    ONSITE_RESIDUAL_COMPONENTS,
    THRESHOLD_STATUS,
    MappingOutput,
    _Chart,
    _compatibility_label,
    _determined_parameters,
    _joint_least_squares,
    _nullspace,
    build_synthetic_response_set,
    compare_mapping_outputs,
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
GAMMA_XX_SYN = sp.Rational(1, 100)

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
#: AFM charts: n_b = -n_a with the SAME site-local transverse axes, so
#: the local tilts are u_b = -u_a and the local tilt dot is -1 exactly.
AFM_FRAMES = {"A": (X, Y, Z), "B": (X, Y, -Z)}
AFM_REFERENCES = {"A": (0.0, 0.0, 1.0), "B": (0.0, 0.0, -1.0)}
#: Exact three-site Heisenberg chain: three coplanar references in the
#: x-z plane (the multi-reference refusal only triggers off-plane).
CHAIN_NB_C = sp.Matrix([sp.Rational(3, 5), 0, sp.Rational(4, 5)])
CHAIN_FRAMES = {
    "A": (X, Y, Z),
    "B": (Y, sp.simplify(PLANAR_NB.cross(Y)), PLANAR_NB),
    "C": (Y, sp.simplify(CHAIN_NB_C.cross(Y)), CHAIN_NB_C),
}
CHAIN_REFERENCES = {
    "A": (0.0, 0.0, 1.0),
    "B": (float(sp.Rational(4, 5)), 0.0, float(sp.Rational(3, 5))),
    "C": (float(sp.Rational(3, 5)), 0.0, float(sp.Rational(4, 5))),
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


def _heisenberg_chain(charts, coupling=J_SYN):
    """Exact Heisenberg energy over ALL pairs of the declared sites."""
    return -coupling * sum(
        (charts[a].T * charts[b])[0, 0] for a, b in combinations(sorted(charts), 2)
    )


def _anisotropic_pair(charts):
    """Exact pair with in-plane anisotropy: -J e_a.e_b + e_a.Gamma.e_b.

    Gamma = Gamma_xx (t1 t1 - n n) in site a's Cartesian chart, i.e. the
    traceless tensor diag(1/100, 0, -1/100) for the (X, Y, Z) frame.
    """
    gamma = GAMMA_XX_SYN * (
        sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        - sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    )
    return (
        -J_SYN * (charts["A"].T * charts["B"])[0, 0]
        + (charts["A"].T * gamma * charts["B"])[0, 0]
    )


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


def _assert_zero(mapping):
    """Exact-Heisenberg verdict: both chi2 scopes vanish, J recovered."""
    assert mapping.compatibility["label"] == "heisenberg_compatible"
    assert mapping.compatibility["gate"] == "hessian_and_measured_torque"
    assert mapping.compatibility["threshold_status"] == THRESHOLD_STATUS
    assert mapping.residuals["chi2_hessian"] < 1e-18
    assert mapping.residuals["chi2_torque"] < 1e-18


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
    _assert_zero(mapped)

    assert mapped.residuals["chi2_floor_eV_per_rad2"] == (
        DEFAULT_CHI2_FLOOR_EV_PER_RAD2
    )
    assert mapped.residuals["torque_chi2_floor_eV_per_rad"] == (
        DEFAULT_TORQUE_CHI2_FLOOR_EV_PER_RAD
    )
    pair_residuals = mapped.residuals["per_pair"]["A|B"]
    assert set(pair_residuals["mixed_components_eV_per_rad2"]) == set(
        MIXED_RESIDUAL_COMPONENTS
    )
    assert all(
        abs(value) < 1e-12
        for value in pair_residuals["mixed_components_eV_per_rad2"].values()
    )
    for site in ("A", "B"):
        site_residuals = mapped.residuals["per_site"][site]
        assert set(site_residuals["onsite_components_eV_per_rad2"]) == set(
            ONSITE_RESIDUAL_COMPONENTS
        )
        assert abs(site_residuals["onsite_closure"]["delta_eV_per_rad2"]) < 1e-12
        assert all(
            abs(value) < 1e-12
            for value in site_residuals["torque_components_eV_per_rad"].values()
        )

    pair = mapped.parameters["A|B"]
    assert pair["alignment"] == 1
    assert pair["alignment_source"] == "reference_directions_by_site"
    # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
    assert pair["j_pair_once_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    # Isotropic limit: the scalar common-axis diagnostic coincides.
    assert pair["j_hat_scalar_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["scalar_extraction"]["method"] == "zero_pitch_identity"
    for axis in LABELS:
        entry = pair["scalar_extraction"]["per_axis"][axis]
        assert entry["transverse_dot"] == pytest.approx(1.0, abs=1e-12)
        assert entry["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(0.0, abs=1e-12)
    # Corrected torque rows: D is determined with the torque data.
    assert set(pair["dmi_determined_chart_eV"]) == {"D_x", "D_y", "D_z"}
    assert all(abs(value) < 1e-12 for value in pair["dmi_determined_chart_eV"].values())

    ident = mapped.identifiability["per_pair"]["A|B"]
    assert mapped.identifiability["reference_class"] == "collinear_single_reference"
    assert ident["rank_mixed_block"] == 4
    assert ident["rank_onsite_closure"] == 5
    assert ident["rank_with_torque"] == 9
    assert set(ident["mixed_nullspace_named"]) == {
        "D_x",
        "D_y",
        "Gamma_xz",
        "Gamma_yz",
        "longitudinal_A_zz",
    }
    assert set(ident["onsite_closure_nullspace_named"]) == {
        "D_x",
        "D_y",
        "Gamma_xz",
        "Gamma_yz",
    }
    assert ident["torque_nullspace_named"] == []
    assert set(ident["determined"]) == set(
        (
            "J",
            "D_x",
            "D_y",
            "D_z",
            "Gamma_xx",
            "Gamma_xy",
            "Gamma_xz",
            "Gamma_yz",
            "Gamma_yy",
        )
    )
    assert ident["undetermined"] == []
    # The former D <-> Gamma trades are retracted by the corrected rows:
    # no trade direction is named at any identifiability level.
    assert "undetermined_trades" not in ident
    for key in (
        "mixed_nullspace_named",
        "onsite_closure_nullspace_named",
        "torque_nullspace_named",
    ):
        assert all("trade" not in name.lower() for name in ident[key])


def test_exact_heisenberg_planar_reference_fully_determined():
    response = _synthetic_response(
        _heisenberg, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0)
    )
    digest = _response_digest(response)
    mapped = map_response_set(response, reference_directions_by_site=PLANAR_REFERENCES)
    _assert_input_preserved(response, digest)
    _assert_zero(mapped)

    assert mapped.identifiability["reference_class"] == "planar_single_q"
    pair = mapped.parameters["A|B"]
    assert pair["j_pair_once_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    # The common-axis scalar identity is a zero-pitch instrument only.
    assert pair["scalar_extraction"]["method"] == "finite_pitch"
    assert pair["j_hat_scalar_eV"] is None
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(0.0, abs=1e-12)
    assert pair["gamma_transverse_eV"]["22"] == pytest.approx(0.0, abs=1e-12)

    ident = mapped.identifiability["per_pair"]["A|B"]
    # Corrected authority: planar Hessian-only rank 5, full rank 9.
    assert ident["rank_onsite_closure"] == 5
    assert ident["rank_with_torque"] == 9
    assert ident["torque_nullspace_named"] == []
    assert ident["undetermined"] == []
    assert set(ident["determined"]) == {
        "J",
        "D_x",
        "D_y",
        "D_z",
        "Gamma_xx",
        "Gamma_xy",
        "Gamma_xz",
        "Gamma_yz",
        "Gamma_yy",
    }
    diagnostic = mapped.residuals["diagnostics"]["per_pair"]["A|B"]
    assert diagnostic["scope"] == "pair_12_entry_k_only_layer"
    assert diagnostic["chi2_sector"] == pytest.approx(0.0, abs=1e-18)
    assert diagnostic["chi2_iso"] == pytest.approx(0.0, abs=1e-18)


# ---------------------------------------------------------------------------
# fitted-J export vs the scalar diagnostic (STD-005)
# ---------------------------------------------------------------------------
def test_collinear_anisotropic_exact_pair_exports_fitted_j():
    response = _synthetic_response(_anisotropic_pair)
    mapped = map_response_set(
        response, reference_directions_by_site=COLLINEAR_REFERENCES
    )
    _assert_zero(mapped)

    pair = mapped.parameters["A|B"]
    # The exported J comes from the determined fitted J (theta_fit):
    # J = 0.1 exactly -- NOT the 0.095 of the common-axis scalar mean.
    assert pair["j_pair_once_eV"] == pytest.approx(0.1, rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(0.05, rel=1e-12)
    # The scalar extraction survives as the labeled diagnostic, carrying
    # the in-plane anisotropy contamination J - u.Gamma.u per axis.
    assert pair["j_hat_scalar_eV"] == pytest.approx(0.0475, rel=1e-12)
    per_axis = pair["scalar_extraction"]["per_axis"]
    assert per_axis["1"]["j_stored_eV"] == pytest.approx(0.05, rel=1e-12)
    assert per_axis["2"]["j_stored_eV"] == pytest.approx(0.045, rel=1e-12)
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(0.01, rel=1e-12)
    assert pair["gamma_transverse_eV"]["22"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# AFM convention (STD-004): site-local signed tilts, local tilt dot only
# ---------------------------------------------------------------------------
def test_afm_reference_stores_positive_half_j():
    response = _synthetic_response(_heisenberg, frames=AFM_FRAMES)
    digest = _response_digest(response)
    mapped = map_response_set(response, reference_directions_by_site=AFM_REFERENCES)
    _assert_input_preserved(response, digest)
    _assert_zero(mapped)

    pair = mapped.parameters["A|B"]
    assert pair["alignment"] == -1
    # MathDoc: tb2j/spiral-heisenberg-mapping#zero_pitch_reduction
    # u_a.u_b = -1 exactly; j_stored = -K_ab / (2 u_a.u_b) = +J/2 > 0.
    assert pair["j_pair_once_eV"] == pytest.approx(float(J_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    assert pair["j_hat_scalar_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)
    for axis in LABELS:
        entry = pair["scalar_extraction"]["per_axis"][axis]
        assert entry["transverse_dot"] == pytest.approx(-1.0, abs=1e-12)
        assert entry["j_stored_eV"] == pytest.approx(float(J_SYN) / 2.0, rel=1e-12)


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
    assert planar.residuals["chi2_hessian"] >= CHI2_FLAG_MIN
    assert planar.residuals["chi2_torque"] >= 0.0
    # MathDoc: tb2j/spiral-heisenberg-mapping#synthetic
    # K-only projection diagnostics reproduce the derivation's exact
    # fractions (distinctly labeled; the verdict uses the joint fit).
    diagnostic = planar.residuals["diagnostics"]["per_pair"]["A|B"]
    assert diagnostic["chi2_iso"] == pytest.approx(2752.0 / 3115.0, rel=1e-9)
    assert diagnostic["chi2_sector"] == pytest.approx(128.0 / 267.0, rel=1e-9)

    collinear = map_response_set(
        _synthetic_response(_biquadratic),
        reference_directions_by_site=COLLINEAR_REFERENCES,
    )
    # Invisible at the collinear anchor: exactly J-representable, and the
    # pure biquadratic masquerades as J_hat = -2B (n_a.n_b) = -2B -- the
    # masquerade now appears in the exported fitted J itself.
    assert collinear.compatibility["label"] == "heisenberg_compatible"
    assert collinear.residuals["chi2_hessian"] < 1e-24
    assert collinear.residuals["chi2_torque"] < 1e-24
    pair = collinear.parameters["A|B"]
    assert pair["j_pair_once_eV"] == pytest.approx(float(-2 * B_SYN), rel=1e-12)
    assert pair["j_stored_eV"] == pytest.approx(float(-B_SYN), rel=1e-12)
    # The scalar diagnostic is carried in stored units: -B per axis.
    assert pair["j_hat_scalar_eV"] == pytest.approx(float(-B_SYN), rel=1e-12)


def test_exact_chain_zero_residual_and_flagged_three_site_distinct():
    # (a) The exact THREE-SITE Heisenberg chain maps with zero residual
    # under the joint multisite fit (onsite/torque rows are sums over the
    # incident pairs, never reused per bond).  The three-site site data
    # is shared among its bonds: the global fit keeps a nontrivial null,
    # so the signed per-pair exports stay gated (never guessed) while the
    # fit itself and the per-site closures are exact.
    chain = _synthetic_response(
        _heisenberg_chain, frames=CHAIN_FRAMES, q=(0.25, 0.25, 0.0)
    )
    chain_digest = _response_digest(chain)
    mapped_chain = map_response_set(
        chain, reference_directions_by_site=CHAIN_REFERENCES
    )
    _assert_input_preserved(chain, chain_digest)
    _assert_zero(mapped_chain)
    for site in ("A", "B", "C"):
        site_residuals = mapped_chain.residuals["per_site"][site]
        assert abs(site_residuals["onsite_closure"]["delta_eV_per_rad2"]) < 1e-12
        assert site_residuals["onsite_max_abs_residual_eV_per_rad2"] < 1e-12
        assert all(
            abs(value) < 1e-12
            for value in site_residuals["torque_components_eV_per_rad"].values()
        )
    ident_chain = mapped_chain.identifiability
    for pair_key in ("A|B", "A|C", "B|C"):
        pair_ident = ident_chain["per_pair"][pair_key]
        # Pair-local corrected authority ranks hold at every bond.
        assert pair_ident["rank_mixed_block"] == 4
        assert pair_ident["rank_with_torque"] == 9
        # The shared site data leaves global trade directions: the signed
        # pair parameters are reported undetermined, never guessed.
        assert ident_chain["alignment"]["by_pair"][pair_key] == 1
        assert mapped_chain.parameters[pair_key]["j_pair_once_eV"] is None
        assert "J" in pair_ident["undetermined"]

    # Per-pair identifiability must come from THIS pair's 9-column slice
    # of the global null basis (round-2 STD-R2-001): a shared first-pair
    # projection must never be copied onto later pairs.
    charts_chain = {
        site: _Chart(site, chain.axes[site], CHAIN_REFERENCES[site])
        for site in chain.site_order
    }
    pairs_chain = list(combinations(chain.site_order, 2))
    fit_chain = _joint_least_squares(chain, charts_chain, pairs_chain)
    global_null = _nullspace(fit_chain["design"])
    for pair in pairs_chain:
        pair_key = f"{pair[0]}|{pair[1]}"
        column = fit_chain["pair_col"][pair]
        expected_determined, expected_undetermined = _determined_parameters(
            global_null[:, column : column + 9]
        )
        pair_ident = ident_chain["per_pair"][pair_key]
        assert pair_ident["determined"] == expected_determined
        assert pair_ident["undetermined"] == expected_undetermined
        assert set(pair_ident["determined"]) | set(pair_ident["undetermined"]) == set(
            CHART_PARAMETER_NAMES
        )

    # (b) The derivation's flagged three-site term is a genuine
    # non-Heisenberg two-site response (third direction frozen): flagged
    # through the onsite-block violation.
    response = _synthetic_response(
        _three_site, frames=PLANAR_FRAMES, q=(0.25, 0.25, 0.0)
    )
    digest = _response_digest(response)
    mapped = map_response_set(response, reference_directions_by_site=PLANAR_REFERENCES)
    _assert_input_preserved(response, digest)
    assert mapped.compatibility["label"] == "incompatible"
    assert mapped.residuals["chi2_hessian"] >= CHI2_FLAG_MIN
    for site in ("A", "B"):
        assert (
            mapped.residuals["per_site"][site]["onsite_max_abs_residual_eV_per_rad2"]
            > 1e-12
        )
        assert (
            abs(
                mapped.residuals["per_site"][site]["onsite_closure"][
                    "delta_eV_per_rad2"
                ]
            )
            > 1e-12
        )
    diagnostic = mapped.residuals["diagnostics"]["per_pair"]["A|B"]
    assert diagnostic["chi2_sector"] == pytest.approx(59.0 / 93.0, rel=1e-9)
    assert diagnostic["chi2_iso"] == pytest.approx(1927.0 / 2170.0, rel=1e-9)


# ---------------------------------------------------------------------------
# thresholds, torque gating, digest gates and refusals
# ---------------------------------------------------------------------------
def test_threshold_constants_and_label_boundaries():
    assert CHI2_COMPAT_MAX == 1.0e-4
    assert CHI2_FLAG_MIN == 2.5e-3 == 1.0 / 400.0
    assert DEFAULT_CHI2_FLOOR_EV_PER_RAD2 == 1.0e-6
    assert DEFAULT_TORQUE_CHI2_FLOOR_EV_PER_RAD == 1.0e-6
    assert THRESHOLD_STATUS == "approved_story_012_2026-10-15"
    assert _compatibility_label(0.0, 0.0) == "heisenberg_compatible"
    assert (
        _compatibility_label(CHI2_COMPAT_MAX, CHI2_COMPAT_MAX)
        == "heisenberg_compatible"
    )
    assert (
        _compatibility_label(0.5 * (CHI2_COMPAT_MAX + CHI2_FLAG_MIN), 0.0) == "marginal"
    )
    assert _compatibility_label(0.0, 0.5 * (CHI2_COMPAT_MAX + CHI2_FLAG_MIN)) == (
        "marginal"
    )
    # The verdict gates on BOTH scopes: either at the flag level flags.
    assert _compatibility_label(CHI2_FLAG_MIN, 0.0) == "incompatible"
    assert _compatibility_label(0.0, CHI2_FLAG_MIN) == "incompatible"
    assert _compatibility_label(CHI2_FLAG_MIN, CHI2_COMPAT_MAX) == "incompatible"


def test_residual_scopes_and_verdict_gate():
    # A site-asymmetric onsite tamper breaks the K representability: the
    # Hessian scope flags, the emitted fit moves, and the torque residual
    # g_raw - g_map(theta_fit) is nonzero and reported alongside it
    # (SPEC-007: both scopes enter the verdict dictionary).
    response = _synthetic_response(_heisenberg)
    curvature = dict(response.curvature)
    key = (("A", "1"), ("A", "1"))
    curvature[key] = curvature[key] + 0.05
    curvature[(key[1], key[0])] = curvature[(key[1], key[0])] + 0.05
    tampered = build_synthetic_response_set(
        site_order=response.site_order,
        axes_by_site=response.axes,
        gradient=response.gradient,
        curvature=curvature,
        q=(0.0, 0.0, 0.0),
    )
    mapped = map_response_set(
        tampered, reference_directions_by_site=COLLINEAR_REFERENCES
    )
    assert mapped.residuals["chi2_hessian"] == pytest.approx(1.0 / 12.0, rel=1e-9)
    # Curvature-only tamper at an equilibrium reference: the torque scope
    # stays exactly satisfied, and the two-scope gate still flags via the
    # Hessian scope alone.
    assert mapped.residuals["chi2_torque"] == pytest.approx(0.0, abs=1e-12)
    assert mapped.compatibility["label"] == "incompatible"
    assert mapped.compatibility["gate"] == "hessian_and_measured_torque"
    assert "chi2_hessian" in mapped.compatibility["scope"]
    assert "chi2_torque" in mapped.compatibility["scope"]
    assert (
        mapped.compatibility["aggregate"]
        == "global sums over all pair and site entries"
    )
    # Structural note: at any single reference the torque-only columns
    # (D and the torque-only Gamma entries) never enter the Hessian rows,
    # so a single reference's measured gradients are always exactly
    # representable and chi2_torque stays ~0; the two-scope gate is
    # exercised at the unit level in test_threshold_constants_and_label_
    # boundaries and becomes binding for stacked multi-reference data.


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
    tampered["compatibility"] = {
        **record["compatibility"],
        "chi2_hessian": 0.5,
    }
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
    assert pair["j_hat_scalar_eV"] is None
    assert pair["gamma_transverse_eV"] is None
    # The sign-independent per-axis data is still reported, with the
    # signed j_stored left undetermined per axis.
    for axis in LABELS:
        entry = pair["scalar_extraction"]["per_axis"][axis]
        assert entry["K_ab_eV_per_rad2"] == pytest.approx(-float(J_SYN), rel=1e-12)
        assert entry["transverse_dot"] == pytest.approx(1.0, abs=1e-12)
        assert entry["j_stored_eV"] is None
    diagnostic = mapped.residuals["diagnostics"]["per_pair"]["A|B"]
    assert diagnostic["j_hat_isotropic_eV"] is None
    # Sign-independent verdicts are still delivered in full.
    assert mapped.residuals["chi2_hessian"] < 1e-24
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
    with pytest.raises(SpiralResponseError, match="floor"):
        map_response_set(
            _synthetic_response(_heisenberg),
            reference_directions_by_site=COLLINEAR_REFERENCES,
            torque_chi2_floor_eV_per_rad=0.0,
        )
    # q = 0 with non-collinear certified references is refused (the
    # references are transverse to their own frames but not parallel).
    with pytest.raises(SpiralResponseError, match="collinear"):
        map_response_set(
            _synthetic_response(_heisenberg, frames=PLANAR_FRAMES),
            reference_directions_by_site=PLANAR_REFERENCES,
        )
    # Certified nonplanar finite-q normals still have pair-local rank 9:
    # the physical producer texture need not be exactly planar.
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
    mapped = map_response_set(
        response,
        reference_directions_by_site={
            "A": (0.0, 0.0, 1.0),
            "B": (float(sp.Rational(4, 5)), 0.0, float(sp.Rational(3, 5))),
            "C": (0.0, 1.0, 0.0),
        },
    )
    assert mapped.identifiability["reference_class"] == "nonplanar_single_q"
    assert mapped.compatibility["label"] == "heisenberg_compatible"
    assert mapped.raw_response.to_dict() == response.to_dict()


def test_synthetic_provenance_partition_fields_present():
    mapped = map_response_set(
        _synthetic_response(_heisenberg),
        reference_directions_by_site=COLLINEAR_REFERENCES,
    )
    # Synthetic records carry no vertex certificate beyond the operator
    # mode: the partition/vertex-version fields are present and honest.
    assert mapped.provenance["partition_declaration"] is None
    assert mapped.provenance["vertex_version"] is None
    assert mapped.provenance["operator_mode"] == "onsite_delta"
    assert mapped.provenance["mapping_version"] == "2.1"


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
    # Zero-pitch identity (story-011 anchor): the scalar common-axis
    # extraction and its per-axis entries reproduce the anchor number.
    assert 1e3 * pair["j_hat_scalar_eV"] == pytest.approx(ANCHOR_J_STORED_MEV, abs=1e-6)
    # The exported j_stored comes from the DETERMINED fitted J
    # (theta_fit).  Physical finding on this fixture: the measured onsite
    # transverse curvature (+0.4546 eV/rad^2) differs from the
    # pure-Heisenberg closure (+0.6231 eV/rad^2 = K_ss + 2|K_ab| would
    # need K_ss = -K_ab), i.e. the fixture carries genuine single-site
    # transverse structure.  Under the corrected rank-9 collinear map it
    # maps to an isotropic Gamma_11 = Gamma_22 ~ -0.0561 eV with D = 0,
    # while the story-011 anchor identity stays on the scalar diagnostic.
    mixed_ab = response.curvature[(("O1", "y"), ("O2", "y"))]
    onsite_ss = response.curvature[(("O1", "y"), ("O1", "y"))]
    j_pair_exact = (onsite_ss - 2.0 * mixed_ab) / 3.0
    assert 1e3 * pair["j_stored_eV"] == pytest.approx(
        1e3 * j_pair_exact / 2.0, abs=1e-6
    )
    assert 1e3 * pair["j_pair_once_eV"] == pytest.approx(
        1e3 * 2.0 * pair["j_stored_eV"], rel=1e-12
    )
    gamma_exact = (onsite_ss + mixed_ab) / 3.0
    assert pair["gamma_transverse_eV"]["11"] == pytest.approx(gamma_exact, rel=1e-9)
    assert pair["gamma_transverse_eV"]["22"] == pytest.approx(gamma_exact, rel=1e-9)
    assert all(abs(value) < 1e-12 for value in pair["dmi_determined_chart_eV"].values())
    assert "anchor" in mapped.conventions["zero_pitch_anchor"]
    assert "anchor" in mapped.identifiability["zero_pitch_anchor"]

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
    # SPEC-008: the raw vertex certificate's declarations surface here.
    certificate = response.frozen_declarations["magnetic_amplitudes"]["certificate"]
    assert mapped.provenance["vertex_version"] == certificate["vertex_version"]
    assert (
        mapped.provenance["partition_declaration"]
        == certificate["partition_declaration"]
    )
    assert "onsite delta" in mapped.provenance["partition_declaration"]
    assert mapped.identifiability["reference_class"] == "collinear_single_reference"
    assert mapped.compatibility["threshold_status"] == THRESHOLD_STATUS


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
    assert mapped.residuals["chi2_hessian"] >= 0.0
    assert mapped.residuals["chi2_torque"] >= 0.0


def test_two_reference_fitted_j_departure_and_cross_scf_guard():
    # Exact same frozen synthetic operator, same q/ensemble/site identity:
    # the SymPy authority predicts the FULL joint-fit J shift 148 B / 125
    # for E = -J x - B x² (not the K-only isotropic diagnostic shift).
    def energy(charts):
        return _heisenberg(charts) - _biquadratic(charts)

    q = (0.25, 0.25, 0.0)
    collinear = map_response_set(
        _synthetic_response(energy, frames=COLLINEAR_FRAMES, q=q),
        reference_directions_by_site=COLLINEAR_REFERENCES,
    )
    planar = map_response_set(
        _synthetic_response(energy, frames=PLANAR_FRAMES, q=q),
        reference_directions_by_site=PLANAR_REFERENCES,
    )
    before = (collinear.to_dict(), planar.to_dict())
    comparison = compare_mapping_outputs(collinear, planar)
    assert comparison["status"] == "matched_frozen_reference_diagnostic"
    assert comparison["per_pair"]["A|B"]["delta_j_pair_once_eV"] == pytest.approx(
        float(sp.Rational(148, 125) * B_SYN), abs=1e-11
    )
    assert comparison["per_pair"]["A|B"]["delta_j_stored_eV"] == pytest.approx(
        float(sp.Rational(74, 125) * B_SYN), abs=1e-11
    )
    assert comparison["compatibility"]["left"]["label"] == "heisenberg_compatible"
    assert comparison["compatibility"]["right"]["label"] == "incompatible"
    assert comparison["provenance"]["right"]["reference_directions_by_site"] == {
        site: list(direction) for site, direction in PLANAR_REFERENCES.items()
    }
    assert comparison["converged_exchange_difference"] is False
    assert (collinear.to_dict(), planar.to_dict()) == before
    assert comparison["mapping"]["left"]["response"] == collinear.raw_response.to_dict()
    assert comparison["mapping"]["right"]["response"] == planar.raw_response.to_dict()

    # Same q but a distinct SCF/bundle identity: the comparison may show
    # both records, never emit a numeric exchange difference.
    other_raw = build_synthetic_response_set(
        site_order=planar.raw_response.site_order,
        axes_by_site=planar.raw_response.axes,
        gradient=planar.raw_response.gradient,
        curvature=planar.raw_response.curvature,
        q=q,
        source_bundle_sha256="independent-scf",
    )
    other = map_response_set(other_raw, reference_directions_by_site=PLANAR_REFERENCES)
    unmatched = compare_mapping_outputs(collinear, other)
    assert unmatched["status"] == "unmatched_cross_scf"
    assert unmatched["per_pair"] == {}
    assert unmatched["converged_exchange_difference"] is False
    assert (
        unmatched["provenance"]["left"]["source_bundle_sha256"]
        != unmatched["provenance"]["right"]["source_bundle_sha256"]
    )
