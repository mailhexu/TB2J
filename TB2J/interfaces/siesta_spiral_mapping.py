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
    K_aa[a,b]  = +delta[a,b] (n_a.n_b) J - delta[a,b] (n_a.Gamma.n_b)
    g_a.alpha  = -J (u.n_b) + D.(u x n_b) + u.Gamma.n_b   (torque = -g)
with tilt directions u_alpha = t_alpha x n of the declared transverse
chart, Gamma symmetric traceless, J > 0 ferromagnetic, unit moment
directions and no |m|^2 division.

# MathDoc: tb2j/spiral-heisenberg-mapping#zero_pitch_reduction
    J_stored = -K_ab / (2 alignment transverse_dot),  J_pair_once = 2 J_stored

# MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
(the story-011 anchor identity: TB2J stores both ordered directions at
J_stored = J_pair_once / 2; exchange_Jdict keeps (R,i,j) and (-R,j,i)).

# MathDoc: tb2j/spiral-heisenberg-mapping#identifiability
Reference classes and exact ranks: a q = 0 collinear single reference
determines rank 4 from the mixed block, rank 5 with the onsite closure
(null space {D_x, D_y, Gamma_xz, Gamma_yz}) and rank 7 with torque rows
(remaining null space the D_x <-> Gamma_yz and D_y <-> Gamma_xz trades);
a planar single-q reference reaches rank 5 / rank 8 with a single blind
direction; only a multi-reference set completes the rank to 9 (refused
here, never approximated).

# MathDoc: tb2j/spiral-heisenberg-mapping#residual_definitions
Residuals r_i = K_raw - K_map on the gauge-free 12-entry layout of one
pair (mixed 2x2, then both onsite 2x2 blocks), the dimensionless
statistic chi = sqrt(r2 / max(d2, K_floor^2)), and the synthetic flags.

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

MAPPING_VERSION = "1.0"

# MathDoc: tb2j/spiral-heisenberg-mapping#synthetic
# Flag level chi > 1/20 = CHI_MAX of the derivation's compatibility
# statistic, squared: chi2 >= 1/400.  Both flagged derivation synthetics
# (biquadratic chi2_iso = 2752/3115, three-site chi2_sector = 59/93) sit
# far above it, while the exact-Heisenberg synthetic sits at exactly 0.
CHI2_FLAG_MIN = 2.5e-3
# User approval of these two story-012 thresholds is recorded in story-012:
# chi2 <= 1e-4 (chi <= 1e-2) counts as Heisenberg-compatible, chi2 >=
# CHI2_FLAG_MIN as incompatible, and strictly between as marginal.
CHI2_COMPAT_MAX = 1.0e-4

#: Derivation K_FLOOR = 1e-6 (eV/rad^2) denominator floor of chi.
DEFAULT_CHI2_FLOOR_EV_PER_RAD2 = 1.0e-6

HEISENBERG_COMPATIBLE = "heisenberg_compatible"
INCOMPATIBLE = "incompatible"
MARGINAL = "marginal"

REFERENCE_CLASS_COLLINEAR = "collinear_single_reference"
REFERENCE_CLASS_PLANAR = "planar_single_q"

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

#: Chart-named null directions of the derivation.
LONGITUDINAL_DIRECTION = {"J": 1.0, "Gamma_xx": 1.0, "Gamma_yy": 1.0}
NAMED_TRADES = {
    "trade_D_x_Gamma_yz": {"D_x": -1.0, "Gamma_yz": 1.0},
    "trade_D_y_Gamma_xz": {"D_y": 1.0, "Gamma_xz": 1.0},
}

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

_MIXED_ROWS = 4
_ONSITE_BLOCK_ROWS = 4

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


def _compatibility_label(chi2):
    """Threshold rule of the story-012 compatibility verdict."""
    if chi2 <= CHI2_COMPAT_MAX:
        return HEISENBERG_COMPATIBLE
    if chi2 >= CHI2_FLAG_MIN:
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
def _pair_model(chart_a, chart_b):
    """(12x9 sector rows, 12x1 isotropic rows, 4x9 gradient rows) of a pair.

    Parameters are carried in site a's chart; for the derivation anchors
    (site a framed on the Cartesian axes) the chart projections are the
    Cartesian components.  D and J never enter the onsite blocks; the
    onsite rows are diagonal in the chart (the composed chart's seagull).
    """
    gamma_basis = _chart_gamma_basis(chart_a)
    directions = _chart_directions(chart_a)

    def channel_rows(dot_j, cross, gamma_proj):
        d_proj = {name: float(cross @ vec) for name, vec in directions.items()}
        g_proj = {
            name: float(np.sum(gamma_proj * tensor))
            for name, tensor in gamma_basis.items()
        }
        return _parameter_row(dot_j, d_proj, g_proj)

    rows = []
    for u_a in chart_a.tilts:
        for u_b in chart_b.tilts:
            rows.append(
                channel_rows(
                    -(u_a @ u_b), np.cross(u_a, u_b), u_a[:, None] * u_b[None, :]
                )
            )
    n_a, n_b = chart_a.n, chart_b.n
    n_dot = float(n_a @ n_b)
    # Onsite anisotropy carries -delta (n_a.Gamma.n_b); D never enters.
    onsite_row = channel_rows(n_dot, np.zeros(3), -(n_a[:, None] * n_b[None, :]))
    zero_row = [0.0] * len(CHART_PARAMETER_NAMES)
    for _chart in (chart_a, chart_b):
        rows += [onsite_row, zero_row, zero_row, onsite_row]

    gradient_rows = []
    for tilts, other in ((chart_a.tilts, n_b), (chart_b.tilts, n_a)):
        for u in tilts:
            gradient_rows.append(
                channel_rows(
                    -(u @ other), np.cross(u, other), u[:, None] * other[None, :]
                )
            )

    # Heisenberg-only isotropic layer: mixed -u_a.u_b and onsite n_a.n_b
    # on the diagonal chart entries only.
    isotropic = []
    for index, row in enumerate(rows):
        off_diagonal = index >= _MIXED_ROWS and (
            index - _MIXED_ROWS
        ) % _ONSITE_BLOCK_ROWS not in (0, 3)
        isotropic.append(0.0 if off_diagonal else row[0])
    return (
        np.asarray(rows, dtype=float),
        np.asarray(isotropic, dtype=float),
        np.asarray(gradient_rows, dtype=float),
    )


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
    for name, trade in NAMED_TRADES.items():
        candidates[name] = np.array(
            [trade.get(parameter, 0.0) for parameter in CHART_PARAMETER_NAMES]
        )
    return candidates


def _match_nullspace(null_basis):
    """Name the chart directions carried by a null space.

    Returns (named, generic): ``named`` lists derivation names whose
    direction lies in the null space and adds independent content (a
    matched trade suppresses its redundant re-expression through its
    members); ``generic`` lists the remaining, unnamed null directions.
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


def _undetermined_trades(torque_null_basis):
    """Map each traded chart parameter to its matched torque-level trade."""
    named, _generic = _match_nullspace(torque_null_basis)
    trades = {}
    for name in named:
        if name not in NAMED_TRADES:
            continue
        for parameter, coef in NAMED_TRADES[name].items():
            if abs(coef) > _NULL_PARTICIPATION_TOL:
                trades[parameter] = name
    return trades


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


def _pair_data(response, chart_a, chart_b):
    """Gauge-free 12-entry data vector of one pair (derivation layout)."""
    a, b = chart_a.site, chart_b.site
    mixed = [
        _curvature_entry(response, (a, la), (b, lb))
        for la in chart_a.labels
        for lb in chart_b.labels
    ]
    data = list(mixed)
    for chart in (chart_a, chart_b):
        data += [
            _curvature_entry(response, (chart.site, l1), (chart.site, l2))
            for l1 in chart.labels
            for l2 in chart.labels
        ]
    return np.asarray(data, dtype=float)


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
    """Planarity of the declared reference directions (rank <= 2)."""
    stack = np.column_stack([np.asarray(v, dtype=float) for v in references.values()])
    singular_values = np.linalg.svd(stack, compute_uv=False)
    if len(singular_values) < 3:
        return  # at most two directions are always coplanar
    if float(singular_values[2]) > PLANARITY_TOL * max(float(singular_values[0]), 1.0):
        raise SpiralResponseError(
            "declared reference directions are not coplanar; a nonplanar "
            "single-q texture needs the multi-reference identification "
            "(rank 9 completion), which this mapping refuses to approximate"
        )


def _alignment_by_pair(site_order, charts, references_given):
    """alignment = sign(n_a.n_b) per pair, or None when not certifiable."""
    alignment, source = {}, None
    if references_given:
        source = "reference_directions_by_site"
        for a, b in combinations(site_order, 2):
            alignment[(a, b)] = 1.0 if float(charts[a].n @ charts[b].n) >= 0.0 else -1.0
    else:
        source = "undetermined"
        for pair in combinations(site_order, 2):
            alignment[pair] = None
    return alignment, source


def _scalar_extraction(response, chart_a, chart_b, alignment):
    """Zero-pitch stored-J identity per common chart axis.

    # MathDoc: tb2j/spiral-heisenberg-mapping#zero_pitch_reduction
    # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
    J_stored = -K_ab / (2 alignment transverse_dot), J_pair_once = 2 J_stored.
    """
    a, b = chart_a.site, chart_b.site
    per_axis = {}
    for index, label in enumerate(chart_a.labels):
        if label not in chart_b.labels:
            continue
        transverse_dot = float(chart_a.tilts[index] @ chart_b.tilts[index])
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
                None
                if alignment is None
                else -k_ab / (2.0 * alignment * transverse_dot)
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


def _onsite_closure(rows, data, theta, labels_a, labels_b):
    """Per-site onsite closure: measured K_aa[11] vs the mapped row."""
    closure = {}
    for site, labels, block in (
        ("a", labels_a, "onsite_a_11"),
        ("b", labels_b, "onsite_b_11"),
    ):
        row_index = RESIDUAL_COMPONENTS.index(block)
        measured = float(data[row_index])
        mapped = float(rows[row_index] @ theta)
        closure[site] = {
            "chart_axis_labels": list(labels),
            "measured_K_aa_11_eV_per_rad2": measured,
            "mapped_K_aa_11_eV_per_rad2": mapped,
            "delta_eV_per_rad2": measured - mapped,
        }
    return closure


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
    # MathDoc: tb2j/spiral-heisenberg-mapping (normative SymPy authority).
    # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
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
        "chi_statistic": (
            "chi = sqrt(r2 / max(d2, K_floor^2)); scale-invariant while the "
            "floor is inactive"
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
    denominator floor of the compatibility statistic.
    """
    if not isinstance(response, ResponseSet):
        raise SpiralResponseError(
            "map_response_set consumes a versioned ResponseSet "
            f"(got {type(response).__name__})"
        )
    if chi2_floor_eV_per_rad2 is None:
        floor = DEFAULT_CHI2_FLOOR_EV_PER_RAD2
    else:
        floor = float(chi2_floor_eV_per_rad2)
        if not np.isfinite(floor) or floor <= 0.0:
            raise SpiralResponseError(
                "chi2_floor_eV_per_rad2 must be finite and positive"
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
                "finite-q planar treatment needs the certified reference "
                "directions of every site; without them the spiral plane "
                "and the pair alignments are unknowable and are not guessed"
            )
        _reference_planes(references)

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

    parameters, residuals_per_pair, identifiability_per_pair = {}, {}, {}
    chi2_iso_pairs, chi2_sector_pairs = [], []
    for a, b in combinations(response.site_order, 2):
        pair_key = f"{a}|{b}"
        chart_a, chart_b = charts[a], charts[b]
        alignment = alignment_by_pair[(a, b)]
        sector_rows, isotropic_rows, gradient_rows = _pair_model(chart_a, chart_b)
        data = _pair_data(response, chart_a, chart_b)

        # Identifiability is a property of the anchor geometry (the
        # derivation's matrix ranks), independent of the measured data.
        mixed_rank = int(np.linalg.matrix_rank(sector_rows[:_MIXED_ROWS]))
        closure_rank = int(np.linalg.matrix_rank(sector_rows))
        stacked = np.vstack([sector_rows, gradient_rows])
        torque_rank = int(np.linalg.matrix_rank(stacked))
        null_basis = _nullspace(sector_rows)
        torque_null_basis = _nullspace(stacked)
        null_named, null_generic = _match_nullspace(null_basis)
        torque_named, _torque_generic = _match_nullspace(torque_null_basis)

        # Chi statistics are pinned to the K-only least-squares layer of
        # the derivation ([E]); at a planar reference the pair parameters
        # are NOT all determined by K alone (hessian rank 5 of 9), so the
        # parameter fit adds the certified gradient rows (rank 8), exactly
        # like the derivation's identifiability completion.
        theta, *_ = np.linalg.lstsq(sector_rows, data, rcond=None)
        gradient_data = np.asarray(
            [
                _gradient_entry(response, (chart_a.site, label))
                for label in chart_a.labels
            ]
            + [
                _gradient_entry(response, (chart_b.site, label))
                for label in chart_b.labels
            ],
            dtype=float,
        )
        if reference_class == REFERENCE_CLASS_PLANAR:
            augmented = np.vstack([sector_rows, gradient_rows])
            theta_fit, *_ = np.linalg.lstsq(
                augmented, np.concatenate([data, gradient_data]), rcond=None
            )
            fit_basis, fit_kind = augmented, "hessian_plus_torque"
        else:
            theta_fit, fit_basis, fit_kind = theta, sector_rows, "hessian_only"
        # The determined/undetermined report describes the parameter fit
        # actually used: K-only at a collinear reference (the derivation's
        # D2 null space), torque-completed at a planar reference (D4).
        determined, undetermined = _determined_parameters(_nullspace(fit_basis))
        residual = data - sector_rows @ theta
        residual2 = float(residual @ residual)
        data2 = float(data @ data)
        denominator = max(data2, floor**2)
        chi2_sector_pair = residual2 / denominator
        j_hat = float(isotropic_rows @ data) / float(isotropic_rows @ isotropic_rows)
        isotropic_residual = data - isotropic_rows * j_hat
        chi2_iso_pair = float(isotropic_residual @ isotropic_residual) / denominator
        onsite_max = max(abs(float(v)) for v in residual[_MIXED_ROWS:])

        # Signed parameters: only with a certified alignment sign.
        if alignment is None:
            j_pair_once = j_stored = None
            j_hat_reported = None
            gamma_transverse = None
            dmi_determined = None
            per_axis, scalar_method = {}, "undetermined_alignment"
        elif reference_class == REFERENCE_CLASS_COLLINEAR:
            stored, per_axis = _scalar_extraction(response, chart_a, chart_b, alignment)
            j_stored = stored
            j_pair_once = None if stored is None else 2.0 * stored
            j_hat_reported = j_hat
            gamma_transverse = {
                "11": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_xx")]),
                "12": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_xy")]),
                "22": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_yy")]),
            }
            scalar_method = "zero_pitch_identity"
        else:
            j_pair_once = float(theta_fit[0])
            j_stored = j_pair_once / 2.0
            j_hat_reported = j_hat
            gamma_transverse = {
                "11": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_xx")]),
                "12": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_xy")]),
                "22": float(theta_fit[CHART_PARAMETER_NAMES.index("Gamma_yy")]),
            }
            per_axis = {}
            scalar_method = "sector_fit"
        if alignment is None:
            dmi_determined = None
        else:
            dmi_determined = {
                name: float(theta_fit[index])
                for index, name in enumerate(CHART_PARAMETER_NAMES)
                if name in {"D_x", "D_y", "D_z"} and name in determined
            }

        parameters[pair_key] = {
            "sites": [a, b],
            "reference_class": reference_class,
            "chart_axis_labels": list(chart_a.labels),
            "alignment": alignment,
            "alignment_source": alignment_source,
            "j_pair_once_eV": j_pair_once,
            "j_stored_eV": j_stored,
            "j_hat_scalar_eV": j_hat_reported,
            "gamma_transverse_eV": gamma_transverse,
            "dmi_determined_chart_eV": dmi_determined,
            "scalar_extraction": {
                "method": scalar_method,
                "parameter_fit": fit_kind,
                "per_axis": per_axis,
            },
            "onsite_closure": _onsite_closure(
                sector_rows, data, theta_fit, chart_a.labels, chart_b.labels
            ),
        }
        residuals_per_pair[pair_key] = {
            "components_eV_per_rad2": {
                name: float(value) for name, value in zip(RESIDUAL_COMPONENTS, residual)
            },
            "chi_pair": float(np.sqrt(chi2_sector_pair)),
            "chi2_sector_pair": chi2_sector_pair,
            "chi2_iso_pair": chi2_iso_pair,
            "onsite_max_abs_residual_eV_per_rad2": float(onsite_max),
        }
        chi2_iso_pairs.append(chi2_iso_pair)
        chi2_sector_pairs.append(chi2_sector_pair)
        identifiability_per_pair[pair_key] = {
            "rank_mixed_block": mixed_rank,
            "rank_onsite_closure": closure_rank,
            "rank_with_torque": torque_rank,
            "nullspace_named": null_named,
            "nullspace_generic": null_generic,
            "torque_nullspace_named": torque_named,
            "determined": determined,
            "undetermined": undetermined,
            "undetermined_trades": _undetermined_trades(torque_null_basis),
        }

    chi2_iso = max(chi2_iso_pairs)
    chi2_sector = max(chi2_sector_pairs)
    compatibility = {
        "label": _compatibility_label(chi2_iso),
        "chi2_iso": chi2_iso,
        "chi2_sector": chi2_sector,
        "aggregate": "max over site pairs",
        "thresholds": {
            "chi2_compat_max": CHI2_COMPAT_MAX,
            "chi2_flag_min": CHI2_FLAG_MIN,
        },
    }

    ensemble = response.ensemble or {}
    identifiability = {
        "reference_class": reference_class,
        "q": q,
        "alignment": {
            "source": alignment_source,
            "by_pair": {
                f"{a}|{b}": alignment_by_pair[(a, b)]
                for a, b in combinations(response.site_order, 2)
            },
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
        "chi2_floor_eV_per_rad2": floor,
        "per_pair": residuals_per_pair,
        "chi2_iso": chi2_iso,
        "chi2_sector": chi2_sector,
    }
    provenance_out = {
        "operator_mode": response.operator_mode,
        "vertex_sha256": provenance.get("vertex_sha256"),
        "source_bundle_sha256": provenance.get("source_bundle_sha256"),
        "q": q,
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
