"""Story-012 mapping layer: transverse frozen response -> Heisenberg parameters.

Takes a versioned ``ResponseSet`` (TB2J/interfaces/siesta_spiral_response.py,
version 2.1) and identifies, per site pair, the Heisenberg model parameters
implied by the transverse Hessian K -- or reports, as numbers, exactly why
the data is not Heisenberg-compatible.

The normative authority is the SymPy derivation
``docs/sympy/spiral_heisenberg_mapping.py`` (FR-009 / ADR-006); every
identity used here is pinned there with exact assertions:

# MathDoc: tb2j/spiral-heisenberg-mapping#pair_mapping
    K_ab[a,b]  = -J (u_a.u_b) + D.(u_a x u_b) + u_a.Gamma.u_b
    K_aa[a,b]  = +delta[a,b] (n_a.n_b) J - delta[a,b] D.(n_a x n_b)
                 - delta[a,b] (n_a.Gamma.n_b)
    g_a.alpha  = -J (u.n_b) + D.(u x n_b) + u.Gamma.n_b   (torque = -g)
    g_b.beta   = -J (n_a.u) + D.(n_a x u_b) + u_b.Gamma.n_a
with tilt directions u_alpha = t_alpha x n of the declared transverse
chart, Gamma symmetric traceless, J > 0 ferromagnetic, unit moment
directions and no |m|^2 division.  D enters the mixed cross products and
the onsite DIAGONAL blocks through -D.(n_a x n_b) (zero at collinear
references); the b-site gradient row is D.(n_a x u_b) -- the derivative
of E_DMI = +D.(e_a x e_b) -- NOT D.(u_b x n_a).

# MathDoc: tb2j/spiral-heisenberg-mapping#zero_pitch_reduction
    J_stored = -K_ab / (2 transverse_dot),  J_pair_once = 2 J_stored
with transverse_dot the SITE-LOCAL tilt dot (u_a.u_b): the AFM alignment
sign is already encoded in the charts (n_b = -n_a with the same
site-local axis t gives u_b = -u_a), so j_stored = +J/2 for the exact
AFM Heisenberg pair -- no separate alignment factor divides the identity.

# MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
(the story-011 anchor identity: TB2J stores both ordered directions at
J_stored = J_pair_once / 2; exchange_Jdict keeps (R,i,j) and (-R,j,i)).

# MathDoc: tb2j/spiral-heisenberg-mapping#identifiability
Reference classes and exact ranks (corrected rows): a q = 0 collinear
single reference determines rank 4 from the mixed block (null space
{D_x, D_y, Gamma_xz, Gamma_yz, longitudinal A_zz <-> J trade}), rank 5
with the onsite closure (null space {D_x, D_y, Gamma_xz, Gamma_yz}) and
rank 9 with the torque rows: the corrected b-site gradient row makes the
sums AND differences of the two sites' DMI torque columns measurable, so
D is determined together with Gamma and no D <-> Gamma trade survives.
A planar single-q reference has Hessian rank 5 and full rank 9 with the
torque rows; stacked references stay at rank 9.

# MathDoc: tb2j/spiral-heisenberg-mapping#residual_definitions
Component residuals r = raw - mapped(theta_fit) against the EMITTED
joint least-squares fit, aggregated over the gauge-free K entries
(mixed + onsite) into chi2_hessian and over the measured-gradient
entries into chi2_torque, each with its own explicit denominator floor:
chi2 = r2 / max(d2, floor^2).  The K-only projection residuals are kept
as a distinctly labeled diagnostic.

The fit itself is ONE global least squares over ALL pair parameters
(mixed K rows per pair; onsite and torque rows of a site are sums over
its incident pairs): the measured onsite block and gradient of a site
are single numbers shared by its bonds and are never reused per bond.

The layer never mutates its input, never guesses an unidentifiable
parameter, and never derives a sign it cannot certify: without declared
reference directions the pair alignment sign(n_a.n_b) is undetermined
(torque cannot fix it -- it vanishes at a stationary collinear reference
and otherwise carries the unknown sign of J), so the signed pair
parameters are reported as undetermined instead of guessed.

Prefer boring, bounded input surfaces: one frozen dataclass in, one
frozen dataclass out, every refusal a ``SpiralResponseError``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .siesta_spiral_response import (
    ENSEMBLE_GAPPED_T0,
    RESPONSE_CONVENTION,
    RESPONSE_VERSION,
    ResponseSet,
    SpiralResponseError,
)
from .siesta_spiral_vertex import DEFAULT_OPERATOR_MODE, OPERATOR_MODES

MAPPING_VERSION = "2.1"

# MathDoc: tb2j/spiral-heisenberg-mapping#synthetic
# Flag level chi > 1/20 = CHI_MAX of the derivation's compatibility
# statistic, squared: chi2 >= 1/400.  Both flagged derivation synthetics
# (biquadratic chi2_iso = 2752/3115, three-site chi2_sector = 59/93) sit
# far above it, while the exact-Heisenberg synthetic sits at exactly 0.
#
# Thresholds APPROVED by the user for story-012 (2026-10-15):
# CHI2_COMPAT_MAX = 1e-4, CHI2_FLAG_MIN = 2.5e-3 (1/400), marginal
# strictly between.  MappingOutput.compatibility carries
# 'threshold_status': 'approved_story_012_2026-10-15'.
CHI2_FLAG_MIN = 2.5e-3
CHI2_COMPAT_MAX = 1.0e-4
THRESHOLD_STATUS = "approved_story_012_2026-10-15"

#: Derivation K_FLOOR = 1e-6 (eV/rad^2) denominator floor of the Hessian
#: chi2; the torque aggregate uses the same floor magnitude in its own
#: gradient units (eV/rad).  Both floors were approved together with
#: the thresholds above.
DEFAULT_CHI2_FLOOR_EV_PER_RAD2 = 1.0e-6
DEFAULT_TORQUE_CHI2_FLOOR_EV_PER_RAD = 1.0e-6

HEISENBERG_COMPATIBLE = "heisenberg_compatible"
INCOMPATIBLE = "incompatible"
MARGINAL = "marginal"

REFERENCE_CLASS_COLLINEAR = "collinear_single_reference"
REFERENCE_CLASS_PLANAR = "planar_single_q"
REFERENCE_CLASS_NONPLANAR = "nonplanar_single_q"

#: Chart parameter vector of one pair, in the first site's chart: J, the
#: D vector projected on (t1, t2, n_a), and the traceless symmetric Gamma
#: (in-plane pair Gamma_11/Gamma_12/Gamma_22 plus the
#: longitudinal-transverse components Gamma_1n, Gamma_2n).
CHART_PARAMETER_NAMES = (
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

#: Chart-named null directions of the derivation (corrected authority).
#: The former D_x <-> Gamma_yz and D_y <-> Gamma_xz trades are RETRACTED:
#: the corrected b-site torque row D.(n_a x u_b) determines D together
#: with Gamma once the torque data is used.
LONGITUDINAL_DIRECTION = {"J": 1.0, "Gamma_xx": 1.0, "Gamma_yy": 1.0}

#: Gauge-free 12-entry data layout of one pair
#: (MathDoc: tb2j/spiral-heisenberg-mapping#residual_definitions).
RESIDUAL_COMPONENTS = (
    "mixed_11",
    "mixed_12",
    "mixed_21",
    "mixed_22",
    "onsite_a_11",
    "onsite_a_12",
    "onsite_a_21",
    "onsite_a_22",
    "onsite_b_11",
    "onsite_b_12",
    "onsite_b_21",
    "onsite_b_22",
)
MIXED_RESIDUAL_COMPONENTS = (
    "mixed_11",
    "mixed_12",
    "mixed_21",
    "mixed_22",
)
ONSITE_RESIDUAL_COMPONENTS = ("onsite_11", "onsite_12", "onsite_21", "onsite_22")

_FRAME_UNIT_TOL = 1e-6
AXIS_TRANSVERSE_TOL = 1e-6
CHART_ORTHOGONALITY_TOL = 1e-6
COLLINEARITY_TOL = 1e-6
PLANARITY_TOL = 1e-6
_NULL_PARTICIPATION_TOL = 1e-8
_TRANSVERSE_DOT_MIN = 1e-8
_DIRECTION_MATCH_TOL = 1e-8


def _payload_digest(payload):
    """Canonical-json sha256, the same recipe as the response records."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _compatibility_label(chi2_hessian, chi2_torque):
    """Threshold rule over BOTH scopes of the story-012 verdict.

    Compatible requires the Hessian AND the measured-torque aggregate
    inside the compatibility bound; either scope at or beyond the flag
    level marks the response incompatible; anything between is marginal.
    """
    if chi2_hessian <= CHI2_COMPAT_MAX and chi2_torque <= CHI2_COMPAT_MAX:
        return HEISENBERG_COMPATIBLE
    if chi2_hessian >= CHI2_FLAG_MIN or chi2_torque >= CHI2_FLAG_MIN:
        return INCOMPATIBLE
    return MARGINAL


def _unit_vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise SpiralResponseError(
            f"{name} must be a 3-vector (got shape {vector.shape})"
        )
    if not np.all(np.isfinite(vector)):
        raise SpiralResponseError(f"{name} is not finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise SpiralResponseError(f"{name} is the zero vector")
    if abs(norm - 1.0) > _FRAME_UNIT_TOL:
        raise SpiralResponseError(
            f"{name} must be a declared unit direction (norm {norm!r})"
        )
    return vector / norm


class _Chart:
    """One site's transverse chart: axes (t1, t2), reference n, tilts u."""

    __slots__ = ("site", "labels", "t1", "t2", "n", "tilts")

    def __init__(self, site, axes, reference):
        labels = tuple(axes.keys())
        if len(labels) != 2:
            raise SpiralResponseError(
                f"site {site!r} declares {len(labels)} rotation axes; the "
                "mapping chart needs exactly two transverse axes"
            )
        t1 = _unit_vector(axes[labels[0]], f"axis {site!r}/{labels[0]!r}")
        t2 = _unit_vector(axes[labels[1]], f"axis {site!r}/{labels[1]!r}")
        if abs(float(t1 @ t2)) > CHART_ORTHOGONALITY_TOL:
            raise SpiralResponseError(
                f"site {site!r} declares non-orthogonal chart axes "
                f"{labels!r}; the exponential chart requires an orthonormal "
                "transverse pair"
            )
        if reference is None:
            # The axes determine n only up to a global sign; the residual
            # sign is exactly the pair alignment and is never guessed.
            n = np.cross(t1, t2)
            n = n / float(np.linalg.norm(n))
        else:
            n = _unit_vector(reference, f"reference direction of site {site!r}")
        for label, axis in ((labels[0], t1), (labels[1], t2)):
            if abs(float(axis @ n)) > AXIS_TRANSVERSE_TOL:
                raise SpiralResponseError(
                    f"site {site!r} axis {label!r} is not transverse to the "
                    "reference direction; u = t x n would degenerate and "
                    "the projection is refused"
                )
        self.site = site
        self.labels = labels
        self.t1, self.t2, self.n = t1, t2, n
        self.tilts = (np.cross(t1, n), np.cross(t2, n))


def _chart_directions(chart):
    """Chart direction vectors keyed by the derivation's D names."""
    return {"D_x": chart.t1, "D_y": chart.t2, "D_z": chart.n}


def _chart_gamma_basis(chart):
    """Traceless symmetric chart basis tensors as Cartesian matrices."""

    def sym(a, b):
        return np.outer(a, b) + np.outer(b, a)

    t1, t2, n = chart.t1, chart.t2, chart.n
    return {
        "Gamma_xx": np.outer(t1, t1) - np.outer(n, n),
        "Gamma_xy": sym(t1, t2),
        "Gamma_xz": sym(t1, n),
        "Gamma_yz": sym(t2, n),
        "Gamma_yy": np.outer(t2, t2) - np.outer(n, n),
    }


def _parameter_row(dot_j, d_chart, gamma_chart):
    """Row over CHART_PARAMETER_NAMES from Cartesian channel projections."""
    row = [dot_j]
    row += [float(d_chart[name]) for name in ("D_x", "D_y", "D_z")]
    row += [
        float(gamma_chart[name])
        for name in (
            "Gamma_xx",
            "Gamma_xy",
            "Gamma_xz",
            "Gamma_yz",
            "Gamma_yy",
        )
    ]
    return row


# MathDoc: tb2j/spiral-heisenberg-mapping#pair_mapping
def _pair_rows(chart_a, chart_b):
    """Row blocks of one pair's model, in the pair chart (site a's chart).

    Returns (mixed 4x9, onsite block 4x9, site-a gradient 2x9, site-b
    gradient 2x9).  The onsite rows are diagonal in the chart (the
    composed chart's vanishing mixed second vertex): +J (n_a.n_b) on the
    diagonal entries, -D.(n_a x n_b) and -(n_a.Gamma.n_b) alongside, and
    zeros off the diagonal.  The site-b gradient DMI row is
    D.(n_a x u_b) -- the direct derivative of E_DMI = +D.(e_a x e_b).
    """
    gamma_basis = _chart_gamma_basis(chart_a)
    directions = _chart_directions(chart_a)
    n_a, n_b = chart_a.n, chart_b.n

    def channel_row(dot_j, cross, gamma_proj):
        d_proj = {name: float(cross @ vec) for name, vec in directions.items()}
        g_proj = {
            name: float(np.sum(gamma_proj * tensor))
            for name, tensor in gamma_basis.items()
        }
        return _parameter_row(dot_j, d_proj, g_proj)

    mixed = [
        channel_row(-(u_a @ u_b), np.cross(u_a, u_b), np.outer(u_a, u_b))
        for u_a in chart_a.tilts
        for u_b in chart_b.tilts
    ]
    onsite_entry = channel_row(
        float(n_a @ n_b), -np.cross(n_a, n_b), -np.outer(n_a, n_b)
    )
    zero_row = [0.0] * len(CHART_PARAMETER_NAMES)
    onsite_block = [onsite_entry, zero_row, zero_row, onsite_entry]
    gradient_a = [
        channel_row(-(u @ n_b), np.cross(u, n_b), np.outer(u, n_b))
        for u in chart_a.tilts
    ]
    gradient_b = [
        channel_row(-(u @ n_a), np.cross(n_a, u), np.outer(u, n_a))
        for u in chart_b.tilts
    ]
    return (
        np.asarray(mixed, dtype=float),
        np.asarray(onsite_block, dtype=float),
        np.asarray(gradient_a, dtype=float),
        np.asarray(gradient_b, dtype=float),
    )


def _pair_isotropic_layer(chart_a, chart_b):
    """12-entry isotropic Heisenberg-only layer of one pair (diagnostic)."""
    mixed = [-(u_a @ u_b) for u_a in chart_a.tilts for u_b in chart_b.tilts]
    nab = float(chart_a.n @ chart_b.n)
    onsite = []
    for _chart in (chart_a, chart_b):
        onsite += [nab, 0.0, 0.0, nab]
    return np.asarray(mixed + onsite, dtype=float)


def _nullspace(matrix):
    """Orthonormal basis (rows) of the null space of ``matrix``."""
    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1]))
    _u, s, vt = np.linalg.svd(matrix)
    rank = int(np.linalg.matrix_rank(matrix))
    return vt[rank:]


def _named_chart_directions():
    """Chart-named candidate directions of the derivation's null spaces."""
    candidates = {}
    for index, name in enumerate(CHART_PARAMETER_NAMES):
        basis = np.zeros(len(CHART_PARAMETER_NAMES))
        basis[index] = 1.0
        candidates[name] = basis
    candidates["longitudinal_A_zz"] = np.array(
        [LONGITUDINAL_DIRECTION.get(name, 0.0) for name in CHART_PARAMETER_NAMES]
    )
    return candidates


def _match_nullspace(null_basis):
    """Name the chart directions carried by a null space.

    Returns (named, generic): ``named`` lists derivation names whose
    direction lies in the null space (each matched direction suppresses
    redundant re-expressions of the same ray); ``generic`` lists the
    remaining, unnamed null directions with their chart components.
    """
    null_basis = np.asarray(null_basis, dtype=float)
    if null_basis.size == 0:
        return [], []
    projection = null_basis.T @ null_basis
    named, span = [], np.zeros((0, null_basis.shape[1]))
    for name, vector in _named_chart_directions().items():
        unit = vector / float(np.linalg.norm(vector))
        if float(np.sqrt(unit @ projection @ unit)) <= 1.0 - _DIRECTION_MATCH_TOL:
            continue
        residual = unit - span.T @ (span @ unit) if len(span) else unit
        if float(np.linalg.norm(residual)) <= _DIRECTION_MATCH_TOL:
            continue  # already expressed by matched directions
        named.append(name)
        span = np.vstack([span, residual / float(np.linalg.norm(residual))])
    generic = []
    for vector in null_basis:
        residual = vector - span.T @ (span @ vector)
        norm = float(np.linalg.norm(residual))
        if norm <= _DIRECTION_MATCH_TOL:
            continue
        components = {
            name: float(coef)
            for name, coef in zip(CHART_PARAMETER_NAMES, residual / norm)
            if abs(float(coef)) > _NULL_PARTICIPATION_TOL
        }
        generic.append({"name": None, "components": components})
        span = np.vstack([span, residual / norm])
    return named, generic


def _determined_parameters(null_basis):
    """Chart parameters with no null-space participation are determined."""
    null_basis = np.asarray(null_basis, dtype=float)
    if null_basis.size == 0:
        return list(CHART_PARAMETER_NAMES), []
    determined, undetermined = [], []
    for index, name in enumerate(CHART_PARAMETER_NAMES):
        participation = float(np.linalg.norm(null_basis[:, index]))
        if participation <= _NULL_PARTICIPATION_TOL:
            determined.append(name)
        else:
            undetermined.append(name)
    return determined, undetermined


def _curvature_entry(response, coord_a, coord_b):
    key = (coord_a, coord_b)
    if key not in response.curvature:
        raise SpiralResponseError(
            f"response curvature is missing the declared block {key!r}; "
            "the mapping refuses an incomplete transverse Hessian"
        )
    return float(response.curvature[key])


def _gradient_entry(response, coord):
    if coord not in response.gradient:
        raise SpiralResponseError(
            f"response gradient is missing the declared coordinate {coord!r}"
        )
    return float(response.gradient[coord])


def _mixed_data(response, chart_a, chart_b):
    """Measured mixed K block of one pair, in the derivation layout."""
    a, b = chart_a.site, chart_b.site
    return np.asarray(
        [
            _curvature_entry(response, (a, la), (b, lb))
            for la in chart_a.labels
            for lb in chart_b.labels
        ],
        dtype=float,
    )


def _onsite_data(response, chart):
    """Measured onsite K block of one site, in the (11,12,21,22) layout."""
    site = chart.site
    return np.asarray(
        [
            _curvature_entry(response, (site, l1), (site, l2))
            for l1 in chart.labels
            for l2 in chart.labels
        ],
        dtype=float,
    )


def _gradient_data(response, chart):
    """Measured gradient of one site, one entry per declared axis label."""
    return np.asarray(
        [_gradient_entry(response, (chart.site, label)) for label in chart.labels],
        dtype=float,
    )


def _reference_class(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise SpiralResponseError(
            f"response provenance carries no usable spiral pitch q (got {q!r})"
        )
    if np.allclose(q, 0.0):
        return REFERENCE_CLASS_COLLINEAR
    return REFERENCE_CLASS_PLANAR


def _reference_planes(references):
    """Whether certified reference normals lie in one plane (rank <= 2)."""
    stack = np.column_stack([np.asarray(v, dtype=float) for v in references.values()])
    singular_values = np.linalg.svd(stack, compute_uv=False)
    return len(singular_values) < 3 or (
        float(singular_values[2]) <= PLANARITY_TOL * max(float(singular_values[0]), 1.0)
    )


def _alignment_by_pair(site_order, charts, references_given):
    """alignment = sign(n_a.n_b) per pair, or None when not certifiable."""
    alignment, source = {}, None
    if references_given:
        source = "reference_directions_by_site"
        for a, b in combinations(site_order, 2):
            dot = float(charts[a].n @ charts[b].n)
            alignment[(a, b)] = (
                0.0
                if abs(dot) <= _DIRECTION_MATCH_TOL
                else (1.0 if dot > 0.0 else -1.0)
            )
    else:
        source = "undetermined"
        for pair in combinations(site_order, 2):
            alignment[pair] = None
    return alignment, source


def _scalar_extraction(response, chart_a, chart_b, alignment):
    """Zero-pitch stored-J identity per common chart axis (diagnostic).

    # MathDoc: tb2j/spiral-heisenberg-mapping#zero_pitch_reduction
    # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
    J_stored = -K_ab / (2 transverse_dot) with transverse_dot the
    SITE-LOCAL tilt dot (u_a.u_b) of the same axis LABEL: the AFM
    alignment is already encoded in the charts (u_b = -u_a), so no
    separate alignment factor divides the identity and the exact AFM
    Heisenberg pair yields j_stored = +J/2.  The second-site tilt is
    looked up by axis LABEL, never by position.
    """
    a, b = chart_a.site, chart_b.site
    per_axis = {}
    for label in chart_a.labels:
        if label not in chart_b.labels:
            continue
        u_a = chart_a.tilts[chart_a.labels.index(label)]
        u_b = chart_b.tilts[chart_b.labels.index(label)]
        transverse_dot = float(u_a @ u_b)
        if abs(transverse_dot) < _TRANSVERSE_DOT_MIN:
            per_axis[label] = {
                "j_stored_eV": None,
                "reason": "degenerate transverse projection on this axis",
            }
            continue
        k_ab = _curvature_entry(response, (a, label), (b, label))
        per_axis[label] = {
            "K_ab_eV_per_rad2": k_ab,
            "transverse_dot": transverse_dot,
            "j_stored_eV": (
                None if alignment is None else -k_ab / (2.0 * transverse_dot)
            ),
        }
    stored = [
        entry["j_stored_eV"]
        for entry in per_axis.values()
        if entry["j_stored_eV"] is not None
    ]
    if not stored:
        return None, per_axis
    return float(np.mean(stored)), per_axis


def _positive_floor(value, default, name):
    """Shared floor validation for the two chi2 scopes."""
    if value is None:
        return float(default)
    floor = float(value)
    if not np.isfinite(floor) or floor <= 0.0:
        raise SpiralResponseError(f"{name} must be finite and positive")
    return floor


def _joint_least_squares(response, charts, pairs):
    """ONE global least squares over ALL pair parameters.

    The mixed K rows enter per pair (each pair's 12-entry gauge-free K
    layout reduces to its 4 mixed rows here); the onsite rows and the
    torque rows of a SITE are SUMS over its incident pairs.  The measured
    onsite block and gradient of a site are single numbers shared by its
    bonds and are never reused per bond.  Returns the emitted ``theta_fit``
    together with the row blocks needed for the residual report.
    """
    pair_col = {pair: 9 * index for index, pair in enumerate(pairs)}
    n_col = 9 * len(pairs)
    pair_rows = {pair: _pair_rows(charts[pair[0]], charts[pair[1]]) for pair in pairs}
    incident = {site: [] for site in response.site_order}
    for pair in pairs:
        incident[pair[0]].append(pair)
        incident[pair[1]].append(pair)

    mixed_rows = np.zeros((4 * len(pairs), n_col))
    mixed_data = []
    for index, pair in enumerate(pairs):
        a, b = pair
        column = pair_col[pair]
        mixed_rows[4 * index : 4 * index + 4, column : column + 9] = pair_rows[pair][0]
        mixed_data.extend(_mixed_data(response, charts[a], charts[b]))

    onsite_rows = np.zeros((4 * len(response.site_order), n_col))
    onsite_data = []
    for index, site in enumerate(response.site_order):
        for pair in incident[site]:
            column = pair_col[pair]
            onsite_rows[4 * index : 4 * index + 4, column : column + 9] += pair_rows[
                pair
            ][1]
        onsite_data.extend(_onsite_data(response, charts[site]))

    torque_rows = np.zeros((2 * len(response.site_order), n_col))
    torque_data = []
    for index, site in enumerate(response.site_order):
        for pair in incident[site]:
            column = pair_col[pair]
            block = pair_rows[pair][2] if site == pair[0] else pair_rows[pair][3]
            torque_rows[2 * index : 2 * index + 2, column : column + 9] += block
        torque_data.extend(_gradient_data(response, charts[site]))

    design = np.vstack([mixed_rows, onsite_rows, torque_rows])
    target = np.concatenate([mixed_data, onsite_data, torque_data])
    theta_fit, *_ = np.linalg.lstsq(design, target, rcond=None)
    return {
        "pair_rows": pair_rows,
        "pair_col": pair_col,
        "mixed_rows": mixed_rows,
        "mixed_data": np.asarray(mixed_data, dtype=float),
        "onsite_rows": onsite_rows,
        "onsite_data": np.asarray(onsite_data, dtype=float),
        "torque_rows": torque_rows,
        "torque_data": np.asarray(torque_data, dtype=float),
        "design": design,
        "theta_fit": theta_fit,
    }


@dataclass(frozen=True)
class MappingOutput:
    """Frozen story-012 mapping verdict over one ResponseSet.

    ``raw_response`` is a detached deep copy of the mapped input; the
    mapping never mutates the response it consumes.  ``to_dict`` produces
    a JSON-safe, digest-stamped record; ``from_dict`` refuses mismatched
    mapping versions and any tampered payload.
    """

    conventions: dict
    parameters: dict
    identifiability: dict
    residuals: dict
    compatibility: dict
    provenance: dict
    raw_response: ResponseSet

    def to_dict(self):
        payload = {
            "version": MAPPING_VERSION,
            "conventions": self.conventions,
            "parameters": self.parameters,
            "identifiability": self.identifiability,
            "residuals": self.residuals,
            "compatibility": self.compatibility,
            "provenance": self.provenance,
            "response": self.raw_response.to_dict(),
        }
        payload = json.loads(json.dumps(payload, allow_nan=False))
        payload["content_sha256"] = _payload_digest(
            {k: v for k, v in payload.items() if k != "content_sha256"}
        )
        return payload

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != MAPPING_VERSION:
            raise SpiralResponseError(
                f"unsupported mapping version {data.get('version')!r} "
                f"(expected {MAPPING_VERSION!r})"
            )
        payload = {k: v for k, v in data.items() if k != "content_sha256"}
        if data.get("content_sha256") != _payload_digest(payload):
            raise SpiralResponseError("mapping payload digest mismatch")
        return cls(
            conventions=copy.deepcopy(payload["conventions"]),
            parameters=copy.deepcopy(payload["parameters"]),
            identifiability=copy.deepcopy(payload["identifiability"]),
            residuals=copy.deepcopy(payload["residuals"]),
            compatibility=copy.deepcopy(payload["compatibility"]),
            provenance=copy.deepcopy(payload["provenance"]),
            raw_response=ResponseSet.from_dict(payload["response"]),
        )


def _conventions_record():
    # MathDoc: tb2j/spiral-heisenberg-mapping (normative SymPy authority,
    # corrected results). MathDoc:
    # tb2j/collinear-anchor-projection#stored_ordered_pair_j
    return {
        "mathdoc": (
            "tb2j/spiral-heisenberg-mapping (pair mapping, identifiability, "
            "residuals); tb2j/collinear-anchor-projection "
            "(zero-pitch stored ordered-pair identity)"
        ),
        "pair_once_energy": (
            "E = -sum_{a<b} [J (e_a.e_b) - D.(e_a x e_b) - e_a.Gamma.e_b] "
            "with each site pair counted once; Gamma symmetric traceless"
        ),
        "stored_ordered_pair": (
            "TB2J stores both ordered directions at J_stored = J_pair_once/2"
        ),
        "ferromagnetic_sign": "J > 0 ferromagnetic",
        "moment_scaling": (
            "unit moment directions e; no |m|^2 division anywhere in the map"
        ),
        "dmi_sign": "E_dmi = +D.(e_a x e_b)",
        "torque": "tau = -gradient (FR-008), nonzero reference torques retained",
        "zero_pitch_identity": (
            "J_stored = -K_ab / (2 transverse_dot) with transverse_dot the "
            "SITE-LOCAL tilt dot (u_a.u_b) of the same axis label; the AFM "
            "alignment is encoded in the charts (u_b = -u_a), so the exact "
            "AFM Heisenberg pair stores j_stored = +J/2"
        ),
        "zero_pitch_anchor": (
            "at zero pitch the story-011 anchor identity corresponds to "
            "j_hat_scalar_eV, the common-axis scalar extraction -- "
            "Gamma-contaminated as J_hat = J - u.Gamma.u per the authority "
            "[C]; j_pair_once_eV and j_stored_eV are the distinct "
            "Szilva-hierarchy parameters from the determined joint fit, "
            "which uses the full hierarchy (Heisenberg + DMI + Gamma)"
        ),
        "fit": (
            "one global least squares over ALL pair parameters: mixed K rows "
            "per pair, onsite and torque rows of a site summed over its "
            "incident pairs; residuals are taken against this emitted fit"
        ),
        "chi_statistic": (
            "chi2 = r2 / max(d2, floor^2) per scope; chi2_hessian over the "
            "gauge-free K entries (mixed + onsite), chi2_torque over the "
            "measured-gradient entries; scale-invariant while the floor "
            "is inactive"
        ),
    }


def build_synthetic_response_set(
    *,
    site_order,
    axes_by_site,
    gradient,
    curvature,
    curvature_split=None,
    q=(0.0, 0.0, 0.0),
    operator_mode=DEFAULT_OPERATOR_MODE,
    source_bundle_sha256="synthetic-bundle",
    vertex_sha256="synthetic-vertex",
    ensemble=None,
):
    """Assemble a minimal, digest-consistent ResponseSet for mapping tests.

    This is a test builder for the mapping layer, not a producer: it wraps
    hand-built gradient/curvature dicts (e.g. exact derivatives of a
    synthetic energy) into the versioned ResponseSet surface so the mapping
    is exercised through its real input contract.
    """
    if operator_mode not in OPERATOR_MODES:
        raise SpiralResponseError(f"unknown operator mode {operator_mode!r}")
    site_order = tuple(site_order)
    axes = {
        site: {
            label: tuple(float(v) for v in axes_by_site[site][label])
            for label in axes_by_site[site]
        }
        for site in site_order
    }
    coords = tuple((site, label) for site in site_order for label in axes[site])
    if set(gradient) != set(coords):
        raise SpiralResponseError(
            "synthetic gradient must carry exactly the declared coordinates"
        )
    gradient = {coord: float(gradient[coord]) for coord in coords}
    torque = {coord: -value for coord, value in gradient.items()}
    curvature = {
        (tuple(a), tuple(b)): float(value) for (a, b), value in curvature.items()
    }
    if curvature_split is None:
        curvature_split = {
            pair: {"onsite": 0.0, "response": value}
            for pair, value in curvature.items()
        }
    curvature_split = {
        (tuple(a), tuple(b)): dict(split) for (a, b), split in curvature_split.items()
    }
    return ResponseSet(
        version=RESPONSE_VERSION,
        convention=RESPONSE_CONVENTION,
        operator_mode=operator_mode,
        site_order=site_order,
        axes=axes,
        coords=coords,
        gradient=gradient,
        torque=torque,
        curvature=curvature,
        curvature_split=curvature_split,
        contour_vs_band_max_abs=0.0,
        finite_rotation_cross_check={},
        frozen_declarations={
            "declaration": "synthetic mapping-test record (not a producer)",
            "magnetic_amplitudes": {
                "vertex_sha256": vertex_sha256,
                "source_bundle_sha256": source_bundle_sha256,
                "certificate": {"operator_mode": operator_mode},
            },
        },
        ensemble=ensemble
        or {"kind": ENSEMBLE_GAPPED_T0, "gap_ev": None, "synthetic": True},
        provenance={
            "source_bundle_sha256": source_bundle_sha256,
            "vertex_sha256": vertex_sha256,
            "operator_mode": operator_mode,
            "q": [float(v) for v in q],
        },
        integration={"synthetic": True},
        torque_state={
            "declaration": "synthetic record; FR-008 torques retained",
            "reference_torques": {repr(c): v for c, v in torque.items()},
        },
        ladder={"converged": {}},
    )


def map_response_set(
    response,
    *,
    reference_directions_by_site=None,
    chi2_floor_eV_per_rad2=None,
    torque_chi2_floor_eV_per_rad=None,
):
    """Map one ResponseSet onto per-pair Heisenberg parameters.

    ``reference_directions_by_site`` carries the certified reference moment
    direction of each site (the ResponseSet stores only the transverse
    axes).  When it is absent the chart orientation is used and the pair
    alignment sign(n_a.n_b) is reported as undetermined: torque cannot fix
    it (it vanishes at a stationary collinear reference and otherwise
    carries the unknown sign of J), so every signed pair parameter is
    reported as ``None`` instead of guessed.

    ``chi2_floor_eV_per_rad2`` overrides the derivation's K_FLOOR = 1e-6
    denominator floor of the Hessian statistic;
    ``torque_chi2_floor_eV_per_rad`` the matching gradient-unit floor.
    """
    if not isinstance(response, ResponseSet):
        raise SpiralResponseError(
            "map_response_set consumes a versioned ResponseSet "
            f"(got {type(response).__name__})"
        )
    floor = _positive_floor(
        chi2_floor_eV_per_rad2, DEFAULT_CHI2_FLOOR_EV_PER_RAD2, "chi2_floor_eV_per_rad2"
    )
    torque_floor = _positive_floor(
        torque_chi2_floor_eV_per_rad,
        DEFAULT_TORQUE_CHI2_FLOOR_EV_PER_RAD,
        "torque_chi2_floor_eV_per_rad",
    )
    response_digest = response.to_dict()["content_sha256"]
    raw_response = copy.deepcopy(response)

    provenance = response.provenance
    if "q" not in provenance:
        raise SpiralResponseError(
            "response provenance carries no spiral pitch q; the mapping "
            "refuses to guess the reference class"
        )
    q = [float(v) for v in provenance["q"]]
    reference_class = _reference_class(q)

    references = reference_directions_by_site
    if references is not None:
        missing = [s for s in response.site_order if s not in references]
        extra = [s for s in references if s not in response.site_order]
        if missing or extra:
            raise SpiralResponseError(
                f"reference_directions_by_site must declare exactly the "
                f"response sites {list(response.site_order)} "
                f"(missing {missing}, unknown {extra})"
            )
    if reference_class == REFERENCE_CLASS_PLANAR:
        if references is None:
            raise SpiralResponseError(
                "finite-q treatment needs certified reference directions of "
                "every site; the texture and pair signs are not guessed"
            )
        if not _reference_planes(references):
            # The pair K/torque rows are valid for arbitrary unit normals;
            # the story-013 SymPy authority certifies rank 9 for a generic
            # noncoplanar three-site geometry. The actual joint nullspace
            # is still computed from the declared charts, not assumed.
            reference_class = REFERENCE_CLASS_NONPLANAR

    charts = {}
    for site in response.site_order:
        if site not in response.axes:
            raise SpiralResponseError(
                f"response axes lack the site {site!r}; the mapping refuses "
                "an incomplete transverse chart"
            )
        charts[site] = _Chart(
            site,
            response.axes[site],
            None if references is None else references[site],
        )
    if reference_class == REFERENCE_CLASS_COLLINEAR and references is not None:
        for a, b in combinations(response.site_order, 2):
            if abs(float(np.linalg.norm(np.cross(charts[a].n, charts[b].n)))) > (
                COLLINEARITY_TOL
            ):
                raise SpiralResponseError(
                    f"q = 0 requires collinear reference directions; sites "
                    f"{a!r} and {b!r} are not parallel and the zero-pitch "
                    "reduction is refused"
                )
    alignment_by_pair, alignment_source = _alignment_by_pair(
        response.site_order, charts, references is not None
    )

    pairs = list(combinations(response.site_order, 2))
    fit = _joint_least_squares(response, charts, pairs)
    theta_fit = fit["theta_fit"]
    global_null_basis = _nullspace(fit["design"])

    mixed_residual = fit["mixed_data"] - fit["mixed_rows"] @ theta_fit
    onsite_residual = fit["onsite_data"] - fit["onsite_rows"] @ theta_fit
    torque_residual = fit["torque_data"] - fit["torque_rows"] @ theta_fit
    hessian_data2 = float(fit["mixed_data"] @ fit["mixed_data"]) + float(
        fit["onsite_data"] @ fit["onsite_data"]
    )
    hessian_residual2 = float(mixed_residual @ mixed_residual) + float(
        onsite_residual @ onsite_residual
    )
    chi2_hessian = hessian_residual2 / max(hessian_data2, floor**2)
    torque_data2 = float(fit["torque_data"] @ fit["torque_data"])
    torque_residual2 = float(torque_residual @ torque_residual)
    chi2_torque = torque_residual2 / max(torque_data2, torque_floor**2)

    parameters, residuals_per_pair, identifiability_per_pair = {}, {}, {}
    diagnostics_per_pair = {}
    residuals_per_site = {}
    for index, pair in enumerate(pairs):
        # Identifiability of THIS pair: slice the global null basis at the
        # pair's own 9-column block (a later pair can have a different
        # null-space projection than the first).
        pair_column = fit["pair_col"][pair]
        pair_null_basis = (
            global_null_basis[:, pair_column : pair_column + 9]
            if global_null_basis.size
            else global_null_basis
        )
        determined, undetermined = _determined_parameters(pair_null_basis)
        determined_set = set(determined)
        a, b = pair
        pair_key = f"{a}|{b}"
        chart_a, chart_b = charts[a], charts[b]
        alignment = alignment_by_pair[pair]
        pair_mixed, pair_onsite, pair_grad_a, pair_grad_b = fit["pair_rows"][pair]
        # The pair-local K layout: mixed 2x2 + BOTH onsite 2x2 blocks (the
        # same onsite row serves either site of the pair).
        sector = np.vstack([pair_mixed, pair_onsite, pair_onsite])
        stacked = np.vstack([sector, pair_grad_a, pair_grad_b])

        # Identifiability is a property of the anchor geometry (the
        # derivation's corrected matrix ranks), independent of the data.
        mixed_rank = int(np.linalg.matrix_rank(pair_mixed))
        closure_rank = int(np.linalg.matrix_rank(sector))
        torque_rank = int(np.linalg.matrix_rank(stacked))
        mixed_null = _nullspace(pair_mixed)
        closure_null = _nullspace(sector)
        torque_null = _nullspace(stacked)
        mixed_named, mixed_generic = _match_nullspace(mixed_null)
        closure_named, closure_generic = _match_nullspace(closure_null)
        torque_named, _torque_generic = _match_nullspace(torque_null)

        # Per-pair mixed residual slice of the emitted joint fit.
        mixed_slice = slice(4 * index, 4 * index + 4)
        pair_mixed_residual = mixed_residual[mixed_slice]
        pair_mixed_data = fit["mixed_data"][mixed_slice]
        pair_denominator = max(float(pair_mixed_data @ pair_mixed_data), floor**2)
        chi2_mixed_pair = float(pair_mixed_residual @ pair_mixed_residual) / (
            pair_denominator
        )

        # K-only projection diagnostic (distinctly labeled; derivation [E]
        # 12-entry layer for a two-site response, mixed-only otherwise).
        if len(response.site_order) == 2:
            data12 = np.concatenate([pair_mixed_data, fit["onsite_data"]])
            theta_k, *_ = np.linalg.lstsq(sector, data12, rcond=None)
            k_residual = data12 - sector @ theta_k
            iso = _pair_isotropic_layer(chart_a, chart_b)
            iso_norm2 = float(iso @ iso)
            j_hat_iso = (
                float(iso @ data12) / iso_norm2
                if iso_norm2 > _TRANSVERSE_DOT_MIN
                else None
            )
            k_components = {
                name: float(value)
                for name, value in zip(RESIDUAL_COMPONENTS, k_residual)
            }
            denominator12 = max(float(data12 @ data12), floor**2)
            k_diagnostic = {
                "scope": "pair_12_entry_k_only_layer",
                "chi2_sector": float(k_residual @ k_residual) / denominator12,
                "chi2_iso": (
                    None
                    if j_hat_iso is None
                    else float((data12 - iso * j_hat_iso) @ (data12 - iso * j_hat_iso))
                    / denominator12
                ),
                "j_hat_isotropic_eV": None if alignment is None else j_hat_iso,
                "components_eV_per_rad2": k_components,
            }
        else:
            k_diagnostic = {
                "scope": "mixed_block_k_only_layer",
                "chi2_sector": chi2_mixed_pair,
                "chi2_iso": None,
                "j_hat_isotropic_eV": None,
                "components_eV_per_rad2": {
                    name: float(value)
                    for name, value in zip(
                        MIXED_RESIDUAL_COMPONENTS, pair_mixed_residual
                    )
                },
            }
        diagnostics_per_pair[pair_key] = k_diagnostic

        # Signed parameters: only with a certified alignment sign, and
        # only the parameters the joint fit determines (never guessed).
        # The sign-independent per-axis zero-pitch data is reported at a
        # collinear reference regardless, with the signed j_stored left
        # undetermined when the alignment is uncertified.
        if reference_class == REFERENCE_CLASS_COLLINEAR:
            j_hat_scalar, per_axis = _scalar_extraction(
                response, chart_a, chart_b, alignment
            )
            scalar_method = (
                "zero_pitch_identity"
                if alignment is not None
                else "undetermined_alignment"
            )
        else:
            j_hat_scalar, per_axis = None, {}
            scalar_method = (
                "finite_pitch" if alignment is not None else ("undetermined_alignment")
            )
        if alignment is None:
            j_pair_once = j_stored = None
            gamma_transverse = None
            dmi_determined = None
        else:
            column = fit["pair_col"][pair]
            pair_theta = theta_fit[column : column + 9]
            j_pair_once = float(pair_theta[0]) if "J" in determined_set else None
            j_stored = None if j_pair_once is None else j_pair_once / 2.0
            gamma_transverse = {
                key: (
                    float(pair_theta[CHART_PARAMETER_NAMES.index(name)])
                    if name in determined_set
                    else None
                )
                for key, name in (
                    ("11", "Gamma_xx"),
                    ("12", "Gamma_xy"),
                    ("22", "Gamma_yy"),
                )
            }
            dmi_determined = {
                name: float(pair_theta[CHART_PARAMETER_NAMES.index(name)])
                for name in ("D_x", "D_y", "D_z")
                if name in determined_set
            }

        parameters[pair_key] = {
            "sites": [a, b],
            "reference_class": reference_class,
            "chart_axis_labels": list(chart_a.labels),
            "alignment": alignment,
            "alignment_source": alignment_source,
            "j_pair_once_eV": j_pair_once,
            "j_stored_eV": j_stored,
            "j_hat_scalar_eV": j_hat_scalar,
            "gamma_transverse_eV": gamma_transverse,
            "dmi_determined_chart_eV": dmi_determined,
            "scalar_extraction": {
                "method": scalar_method,
                "per_axis": per_axis,
            },
        }
        residuals_per_pair[pair_key] = {
            "mixed_components_eV_per_rad2": {
                name: float(value)
                for name, value in zip(MIXED_RESIDUAL_COMPONENTS, pair_mixed_residual)
            },
            "chi2_mixed_pair": chi2_mixed_pair,
        }
        identifiability_per_pair[pair_key] = {
            "rank_mixed_block": mixed_rank,
            "rank_onsite_closure": closure_rank,
            "rank_with_torque": torque_rank,
            "mixed_nullspace_named": mixed_named,
            "mixed_nullspace_generic": mixed_generic,
            "onsite_closure_nullspace_named": closure_named,
            "onsite_closure_nullspace_generic": closure_generic,
            "torque_nullspace_named": torque_named,
            "determined": determined,
            "undetermined": undetermined,
            "note": (
                "corrected authority: the b-site torque row D.(n_a x u_b) "
                "determines D together with Gamma; the former "
                "D <-> Gamma trade directions are retracted"
            ),
        }

    for index, site in enumerate(response.site_order):
        onsite_residual_slice = slice(4 * index, 4 * index + 4)
        torque_residual_slice = slice(2 * index, 2 * index + 2)
        site_onsite_residual = onsite_residual[onsite_residual_slice]
        site_torque_residual = torque_residual[torque_residual_slice]
        site_onsite_data = fit["onsite_data"][4 * index : 4 * index + 4]
        site_onsite_model = fit["onsite_rows"][4 * index : 4 * index + 4] @ theta_fit
        residuals_per_site[site] = {
            "chart_axis_labels": list(charts[site].labels),
            "onsite_components_eV_per_rad2": {
                name: float(value)
                for name, value in zip(ONSITE_RESIDUAL_COMPONENTS, site_onsite_residual)
            },
            "onsite_max_abs_residual_eV_per_rad2": float(
                np.max(np.abs(site_onsite_residual))
            ),
            "onsite_closure": {
                "measured_K_ss_11_eV_per_rad2": float(site_onsite_data[0]),
                "mapped_K_ss_11_eV_per_rad2": float(site_onsite_model[0]),
                "delta_eV_per_rad2": float(site_onsite_residual[0]),
            },
            "torque_components_eV_per_rad": {
                label: float(value)
                for label, value in zip(charts[site].labels, site_torque_residual)
            },
        }

    compatibility = {
        "label": _compatibility_label(chi2_hessian, chi2_torque),
        "gate": "hessian_and_measured_torque",
        "chi2_hessian": chi2_hessian,
        "chi2_torque": chi2_torque,
        "scope": (
            "chi2_hessian: mixed + onsite K residuals against the emitted "
            "joint fit; chi2_torque: measured-gradient residuals against "
            "the same fit; the verdict gates on BOTH scopes"
        ),
        "aggregate": "global sums over all pair and site entries",
        "thresholds": {
            "chi2_compat_max": CHI2_COMPAT_MAX,
            "chi2_flag_min": CHI2_FLAG_MIN,
        },
        "threshold_status": THRESHOLD_STATUS,
    }

    ensemble = response.ensemble or {}
    identifiability = {
        "reference_class": reference_class,
        "q": q,
        "zero_pitch_anchor": (
            "at zero pitch the story-011 anchor identity corresponds to "
            "j_hat_scalar_eV (common-axis scalar, Gamma-contaminated per "
            "the authority [C]); j_pair_once_eV/j_stored_eV are the "
            "Szilva-hierarchy parameters of the determined joint fit"
        ),
        "alignment": {
            "source": alignment_source,
            "by_pair": {f"{a}|{b}": alignment_by_pair[(a, b)] for a, b in pairs},
            "note": (
                "alignment = sign(n_a.n_b) from the certified reference "
                "directions; torque cannot substitute for them (it vanishes "
                "at a stationary collinear reference and otherwise carries "
                "the unknown sign of J), so an absent certification leaves "
                "the signed parameters undetermined"
            ),
        },
        "per_pair": identifiability_per_pair,
    }
    residuals = {
        "fit": "joint_multisite_least_squares",
        "chi2_floor_eV_per_rad2": floor,
        "torque_chi2_floor_eV_per_rad": torque_floor,
        "chi2_hessian": chi2_hessian,
        "chi2_torque": chi2_torque,
        "per_pair": residuals_per_pair,
        "per_site": residuals_per_site,
        "diagnostics": {
            "note": (
                "K-only projection residuals of the derivation [E] layer; "
                "distinctly labeled diagnostic -- the compatibility verdict "
                "uses the emitted joint fit above"
            ),
            "per_pair": diagnostics_per_pair,
        },
    }

    certificate = (
        response.frozen_declarations.get("magnetic_amplitudes", {}).get("certificate")
        if isinstance(response.frozen_declarations, dict)
        else None
    ) or {}
    provenance_out = {
        "operator_mode": response.operator_mode,
        "vertex_version": certificate.get("vertex_version"),
        "vertex_sha256": provenance.get("vertex_sha256"),
        "source_bundle_sha256": provenance.get("source_bundle_sha256"),
        "partition_declaration": certificate.get("partition_declaration"),
        "q": q,
        "reference_directions_by_site": (
            None
            if references is None
            else {site: charts[site].n.tolist() for site in response.site_order}
        ),
        "ensemble_kind": ensemble.get("kind"),
        "ensemble_gap_ev": ensemble.get("gap_ev"),
        "response_sha256": response_digest,
        "mapping_version": MAPPING_VERSION,
    }
    return MappingOutput(
        conventions=_conventions_record(),
        parameters=parameters,
        identifiability=identifiability,
        residuals=residuals,
        compatibility=compatibility,
        provenance=provenance_out,
        raw_response=raw_response,
    )


def compare_mapping_outputs(left: MappingOutput, right: MappingOutput):
    """Compare two certified references without a new fit or an E(q) fit.

    Only one immutable frozen bundle/SCF, pitch, ensemble and integration
    permit a numeric per-pair J departure. An unmatched pair is reported
    with both complete raw responses but NEVER a converged exchange delta.
    Chart-local D and Gamma cannot be subtracted across reference frames.
    # MathDoc: tb2j/spiral-reference-dependence#full_fit
    """
    if not isinstance(left, MappingOutput) or not isinstance(right, MappingOutput):
        raise SpiralResponseError("comparison requires two MappingOutput records")
    left_record, right_record = left.to_dict(), right.to_dict()
    left_ref = left.provenance.get("reference_directions_by_site")
    right_ref = right.provenance.get("reference_directions_by_site")
    if left_ref is None or right_ref is None:
        raise SpiralResponseError(
            "comparison requires certified reference directions on both mappings"
        )
    frozen_keys = ("source_bundle_sha256", "vertex_sha256", "operator_mode", "q")
    same_frozen = (
        all(
            left.provenance.get(k) and left.provenance.get(k) == right.provenance.get(k)
            for k in frozen_keys
        )
        and left.raw_response.site_order == right.raw_response.site_order
        and left.raw_response.ensemble == right.raw_response.ensemble
        and left.raw_response.integration == right.raw_response.integration
        and left.raw_response.frozen_declarations["magnetic_amplitudes"]["certificate"]
        == right.raw_response.frozen_declarations["magnetic_amplitudes"]["certificate"]
    )
    provenance = {
        side: {
            **mapping.provenance,
            "ensemble": copy.deepcopy(mapping.raw_response.ensemble),
            "integration": copy.deepcopy(mapping.raw_response.integration),
            "site_order": list(mapping.raw_response.site_order),
        }
        for side, mapping in (("left", left), ("right", right))
    }
    per_pair = {}
    if same_frozen:
        for key in left.parameters:
            if key not in right.parameters:
                continue
            j_left = left.parameters[key]["j_pair_once_eV"]
            j_right = right.parameters[key]["j_pair_once_eV"]
            if j_left is None or j_right is None:
                per_pair[key] = {
                    "delta_j_pair_once_eV": None,
                    "delta_j_stored_eV": None,
                    "reason": "pair J undetermined at one or both references",
                }
                continue
            delta = float(j_left - j_right)
            per_pair[key] = {
                "delta_j_pair_once_eV": delta,
                "delta_j_stored_eV": delta / 2.0,
                "interpretation": "reference-direction dependence of the fitted J; not an exact-Heisenberg claim",
            }
    payload = {
        "version": "1.0",
        "status": (
            "matched_frozen_reference_diagnostic"
            if same_frozen
            else "unmatched_cross_scf"
        ),
        "reason": (
            "same certified frozen operator, pitch, ensemble and integration"
            if same_frozen
            else "unmatched bundle/SCF, pitch, operator, ensemble or integration; "
            "story-004 cross-SCF free-energy spread +0.115644 eV/cell forbids "
            "a converged exchange difference"
        ),
        "converged_exchange_difference": False,
        "comparison_scope": (
            "fitted pair-once scalar J in common site/pair identity only; "
            "chart-local D/Gamma are not compared; neither reference is "
            "relabeled exact Heisenberg"
        ),
        "per_pair": per_pair,
        "compatibility": {
            "left": copy.deepcopy(left.compatibility),
            "right": copy.deepcopy(right.compatibility),
        },
        "provenance": provenance,
        "mapping": {"left": left_record, "right": right_record},
    }
    payload = json.loads(json.dumps(payload, allow_nan=False))
    payload["content_sha256"] = _payload_digest(payload)
    return payload
