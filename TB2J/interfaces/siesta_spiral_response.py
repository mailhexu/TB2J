"""Frozen-MFT response core on the rotating-frame generalized pencil (story 009).

Implements the primitive-cell frozen response about a validated SIESTA
spiral reference, consuming the story-002 importer pencil and the
story-005 production vertex.  Conventions are pinned by
``docs/sympy/siesta_spiral_frozen_response.py`` (story 008, sections
[C]-[H]) and reused, never re-transcribed:

* the one source of truth is bundle -> importer -> phased-block Pauli
  decomposition. Magnetic amplitudes are recomputed per k from the
  assembled rotating-frame blocks after the q/2 sector phases. The vertex
  certificate binds their source and incident partition; no rotated or
  differentiated matrix is re-packed into real-space four-component data.
  Each incident magnetic block transforms as ``B -> U B U^dag`` with its
  spin-scalar trace part and the complete assembled S(k) frozen;
* the joint perturbation composes the passes in the declared site order
  (A first, then B); a shared block receives the ordered composition, and
  the mixed second vertex is ``-i[T_b, -i[T_a, W]]`` on shared blocks only;
* two transverse axes on ONE site use the declared exponential chart, so
  the same-site second vertex is the symmetric seagull
  ``-1/2 ([T_a,[T_b,W]] + [T_b,[T_a,W]])``;
* first-order torques are ``tau = -dE/dtheta`` (nonzero reference torques
  are carried verbatim, FR-008); second-order curvature
  ``K = onsite_seagull + response_kernel`` retains the onsite seagull
  (FR-007), with the response kernel
  ``sum_{n!=m} (f_n - f_m)/(e_n - e_m) Re[V_a[n,m] V_b[m,n]]`` on the
  generalized eigenvectors;
* the contour path evaluates the same quantities as S-weighted resolvent
  traces, ``Tr[G(z)V]``, ``Tr[G(z)W]`` and ``Tr[G(z)V_a G(z)V_b]`` with
  ``G(z) = (zS - H)^-1``, residue-summed over the occupied poles by a
  declared gapped contour; it never diagonalizes the pencil;
* finite rotations of the same frozen problem provide the independent
  cross-check and the convergence ladder (budgets approved in story 009).

The module fails closed on undeclared partitions, unknown sites or axes,
unsupported ensembles, unenclosable contours, broken bundle/vertex
binding, non-Hermitian pencils and any closed gap (ADR-005, FR-016
defers metallic ensembles).
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve

from .siesta_spiral import SpiralBundle, SpiralImporter
from .siesta_spiral_io import EIGENVALUE_TOL_EV, HERMITICITY_TOL
from .siesta_spiral_vertex import (
    VERTEX_CONVENTION,
    MagneticVertex,
    _source_bundle_digest,
)

RESPONSE_VERSION = "1.0"
RESPONSE_CONVENTION = (
    "frozen_total_h_ordered_two_sided: two-sided same-U incident rotations, "
    "ordered joint passes in declared site order, S and scalar part frozen, "
    "exponential chart per site"
)
ENSEMBLE_GAPPED_T0 = "gapped_fixed_occupations"
TORQUE_BUDGET_EV_PER_RAD = 1e-3
CURVATURE_BUDGET_EV_PER_RAD2 = 1e-3
RESPONSE_IDENTITY_TOL_EV = 1e-8
# Recorded producer band energies each carry the story-003 absolute budget.
# A gap is a difference of two such energies.
ENSEMBLE_GAP_TOL_EV = 2.0 * EIGENVALUE_TOL_EV
_GAP_GATE_EV = 1e-8
_NONZERO_TORQUE_TOL = 1e-12
_MIN_CONTOUR_NODES = 32

_PAULI = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.diag([1.0, -1.0]).astype(complex),
)


class SpiralResponseError(RuntimeError):
    """Raised when the response core refuses an uncertifiable input."""


# MathDoc: tb2j/siesta-spiral-frozen-response (package docs/sympy authority).
def _unit_axis(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise SpiralResponseError(
            f"rotation axis must be a finite 3-vector, got {axis!r}"
        )
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise SpiralResponseError(
            "rotation axis must be a nonzero 3-vector; refusing a zero axis"
        )
    return tuple(axis / norm)


def _half_generator(axis):
    """The Hermitian generator T = (axis . sigma) / 2."""
    n = _unit_axis(axis)
    return 0.5 * (n[0] * _PAULI[0] + n[1] * _PAULI[1] + n[2] * _PAULI[2])


def _su2(axis, angle):
    """U = cos(theta/2) I - i sin(theta/2) (axis.sigma), story-008 rotation."""
    angle = float(angle)
    if not np.isfinite(angle):
        raise SpiralResponseError("rotation angles must be finite")
    t = _half_generator(axis)
    return np.cos(angle / 2.0) * np.eye(2, dtype=complex) - 1j * np.sin(angle / 2.0) * (
        2.0 * t
    )


def _comm(t, block):
    return t @ block - block @ t


# ---------------------------------------------------------------------------
# block algebra on generic partition keys (derivation-engine level)
# ---------------------------------------------------------------------------
def rotate_blocks(blocks, keys, axis, angle):
    """Two-sided same-U conjugation on the selected blocks (one pass)."""
    u = _su2(axis, angle)
    return {
        key: u @ np.asarray(blocks[key], dtype=complex) @ u.conj().T for key in keys
    }


def first_vertex_blocks(blocks, keys, axis):
    """V = dW/dtheta at 0 = -i[T, W] on every selected block."""
    t = _half_generator(axis)
    return {key: -1j * _comm(t, np.asarray(blocks[key], dtype=complex)) for key in keys}


def second_vertex_blocks(blocks, keys_a, keys_b, axis_a, axis_b, *, same_site=None):
    """d2W/dtheta_a dtheta_b of the declared composed action.

    Same site (identical incidence sets): symmetric seagull of the
    exponential chart.  Different sites: the ordered A-then-B mixed
    vertex on the shared blocks only (story-008 [C]).
    """
    keys_a = frozenset(keys_a)
    keys_b = frozenset(keys_b)
    ta = _half_generator(axis_a)
    tb = _half_generator(axis_b)
    out = {}
    if same_site if same_site is not None else keys_a == keys_b:
        for key in keys_a:
            w = np.asarray(blocks[key], dtype=complex)
            out[key] = -0.5 * (_comm(ta, _comm(tb, w)) + _comm(tb, _comm(ta, w)))
        return out
    for key in keys_a & keys_b:
        w = np.asarray(blocks[key], dtype=complex)
        out[key] = -1j * _comm(tb, -1j * _comm(ta, w))
    return out


# ---------------------------------------------------------------------------
# single-k observables on a generalized pencil
# ---------------------------------------------------------------------------
def _require_hermitian(matrix, name, tol=HERMITICITY_TOL):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise SpiralResponseError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise SpiralResponseError(f"{name} contains nonfinite entries")
    defect = float(np.max(np.abs(matrix - matrix.conj().T)))
    if defect > tol:
        raise SpiralResponseError(
            f"{name} is not Hermitian (defect {defect:.3e} > {tol:.1e}); "
            "refusing to evaluate the response on an inconsistent pencil"
        )
    return matrix


def _checked_occupations(n_occupied, dim):
    if not isinstance(n_occupied, (int, np.integer)) or isinstance(n_occupied, bool):
        raise SpiralResponseError("n_occupied must be an integer")
    if not 0 < n_occupied < dim:
        raise SpiralResponseError(
            f"n_occupied {n_occupied} outside 1..{dim - 1} for a {dim}-band pencil"
        )
    f = np.zeros(dim, dtype=float)
    f[:n_occupied] = 1.0
    return f


def _require_open_gap(eigenvalues, n_occupied):
    homo = float(eigenvalues[n_occupied - 1])
    lumo = float(eigenvalues[n_occupied])
    if not lumo - homo > _GAP_GATE_EV:
        raise SpiralResponseError(
            "declared gapped fixed-occupation ensemble is invalid: the gap "
            f"closed (homo {homo:.6f}, lumo {lumo:.6f}); gapped references "
            "are the only supported ensemble (FR-016)"
        )
    return homo, lumo


def occupied_band_energy(h, s, n_occupied):
    """Occupied band sum of the pencil, with the gapped-ensemble gate."""
    h = _require_hermitian(h, "H")
    s = _require_hermitian(s, "S")
    _checked_occupations(n_occupied, h.shape[0])
    eigenvalues = eigh(h, s, eigvals_only=True)
    _require_open_gap(eigenvalues, n_occupied)
    return float(eigenvalues[:n_occupied].sum())


def pencil_gradient(h, s, v, n_occupied):
    """g = dE/dtheta = sum_n f_n Re[(C^dag V C)]_nn on the eigenbasis."""
    _require_hermitian(v, "V")
    _checked_occupations(n_occupied, h.shape[0])
    eigenvalues, c = eigh(_require_hermitian(h, "H"), _require_hermitian(s, "S"))
    _require_open_gap(eigenvalues, n_occupied)
    f = _checked_occupations(n_occupied, h.shape[0])
    vx = c.conj().T @ v @ c
    return float(np.sum(f * np.real(np.diag(vx))))


def pencil_curvature(h, s, w, v_a, v_b, n_occupied):
    """(K, onsite, response) with the retained onsite seagull term."""
    _require_hermitian(w, "W")
    _require_hermitian(v_a, "V_a")
    _require_hermitian(v_b, "V_b")
    dim = h.shape[0]
    _checked_occupations(n_occupied, dim)
    eigenvalues, c = eigh(_require_hermitian(h, "H"), _require_hermitian(s, "S"))
    _require_open_gap(eigenvalues, n_occupied)
    f = _checked_occupations(n_occupied, dim)
    wx = c.conj().T @ w @ c
    onsite = float(np.sum(f * np.real(np.diag(wx))))
    vax = c.conj().T @ v_a @ c
    vbx = c.conj().T @ v_b @ c
    kernel = _divided_difference(eigenvalues, f)
    response = float(np.sum((kernel * vax * vbx.T).real))
    return onsite + response, onsite, response


def contour_gradient(h, s, v, contour):
    """g as the occupied-pole residue of Tr[G(z) V], never diagonalizing."""
    h, s = _require_hermitian(h, "H"), _require_hermitian(s, "S")
    _require_hermitian(v, "V")
    total = 0.0 + 0.0j
    for z, weight in zip(contour.nodes, contour.weights):
        g = np.linalg.solve(z * s - h, v)
        total += weight * np.trace(g)
    return float(total.real)


def contour_curvature(h, s, w, v_a, v_b, contour):
    """(K, onsite, response) from S-weighted resolvent traces at z nodes."""
    h, s = _require_hermitian(h, "H"), _require_hermitian(s, "S")
    _require_hermitian(w, "W")
    _require_hermitian(v_a, "V_a")
    _require_hermitian(v_b, "V_b")
    onsite = 0.0 + 0.0j
    response = 0.0 + 0.0j
    for z, weight in zip(contour.nodes, contour.weights):
        lu, piv = lu_factor(z * s - h)
        x_a = lu_solve((lu, piv), v_a)
        x_b = lu_solve((lu, piv), v_b)
        y = lu_solve((lu, piv), w)
        onsite += weight * np.trace(y)
        response += weight * np.trace(x_a @ x_b)
    return (float((onsite + response).real), float(onsite.real), float(response.real))


# ---------------------------------------------------------------------------
# declared ensemble and contour
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Ensemble:
    """Declared electronic ensemble (ADR-005); gapped T=0 only (FR-016)."""

    kind: str
    n_occupied: int
    electron_count: float
    fermi_energy_ev: float
    smearing_width_ev: float
    gap_ev: float
    declaration: str = (
        "step occupations of the n_occupied lowest spinor bands per k; "
        "fixed-N and fixed-occupations coincide for the gapped T=0 reference"
    )

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.kind != ENSEMBLE_GAPPED_T0:
            raise SpiralResponseError(
                f"unsupported ensemble kind {self.kind!r}; metallic ensembles "
                "require explicitly matched conventions (FR-016) and are not "
                "implemented"
            )
        if (
            not isinstance(self.n_occupied, (int, np.integer))
            or isinstance(self.n_occupied, bool)
            or self.n_occupied < 1
        ):
            raise SpiralResponseError(
                f"n_occupied must be a positive integer, got {self.n_occupied!r}"
            )
        for name in (
            "electron_count",
            "fermi_energy_ev",
            "smearing_width_ev",
            "gap_ev",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise SpiralResponseError(f"ensemble {name} must be finite")
        if not self.gap_ev > 0.0:
            raise SpiralResponseError(
                "ensemble gap_ev must be positive for the gapped reference"
            )
        if self.smearing_width_ev < 0.0:
            raise SpiralResponseError("smearing_width_ev must be nonnegative")
        if self.electron_count != self.n_occupied:
            raise SpiralResponseError(
                "electron_count must equal n_occupied: spinor bands have unit multiplicity"
            )

    def to_record(self):
        return {
            "kind": self.kind,
            "n_occupied": int(self.n_occupied),
            "electron_count": float(self.electron_count),
            "fermi_energy_ev": float(self.fermi_energy_ev),
            "smearing_width_ev": float(self.smearing_width_ev),
            "occupation_smearing_width_ev": 0.0,
            "producer_width_declaration": (
                "smearing_width_ev records the reference width; this run uses fixed step occupations"
            ),
            "gap_ev": float(self.gap_ev),
            "declaration": self.declaration,
        }


@dataclass(frozen=True)
class EnergyContour:
    """Declared gapped occupied-pole contour (ellipse, trapezoid rule)."""

    z0: float
    occupied_min: float
    gap_ev: float
    margin_ev: float
    n_nodes: int
    nodes: np.ndarray = field(init=False, repr=False, compare=False)
    weights: np.ndarray = field(init=False, repr=False, compare=False)

    @classmethod
    def declared(cls, *, z0, occupied_min, gap_ev, margin_ev, n_nodes):
        return cls(z0, occupied_min, gap_ev, margin_ev, n_nodes)

    def __post_init__(self):
        z0, occupied_min = float(self.z0), float(self.occupied_min)
        gap_ev, margin_ev = float(self.gap_ev), float(self.margin_ev)
        n_nodes = self.n_nodes
        if not all(np.isfinite(v) for v in (z0, occupied_min, gap_ev, margin_ev)):
            raise SpiralResponseError("contour parameters must be finite")
        if not occupied_min < z0:
            raise SpiralResponseError(
                "contour requires occupied_min < z0 with z0 inside the gap"
            )
        if not 0.0 < margin_ev < gap_ev / 2.0:
            raise SpiralResponseError(
                f"contour margin_ev {margin_ev} must satisfy 0 < margin < "
                f"gap/2 = {gap_ev / 2.0}; the contour must not enclose or "
                "approach empty states"
            )
        if not isinstance(n_nodes, (int, np.integer)) or n_nodes < _MIN_CONTOUR_NODES:
            raise SpiralResponseError(
                f"n_nodes must be an integer >= {_MIN_CONTOUR_NODES}, got {n_nodes!r}"
            )
        center = 0.5 * (z0 + occupied_min)
        semi_major = 0.5 * (z0 - occupied_min) + margin_ev
        t = 2.0 * np.pi * np.arange(int(n_nodes)) / int(n_nodes)
        nodes = center + semi_major * np.cos(t) + 1j * margin_ev * np.sin(t)
        dz = -semi_major * np.sin(t) + 1j * margin_ev * np.cos(t)
        weights = dz / (1j * int(n_nodes))
        nodes.setflags(write=False)
        weights.setflags(write=False)
        for name, value in (
            ("z0", z0),
            ("occupied_min", occupied_min),
            ("gap_ev", gap_ev),
            ("margin_ev", margin_ev),
            ("n_nodes", int(n_nodes)),
            ("nodes", nodes),
            ("weights", weights),
        ):
            object.__setattr__(self, name, value)

    def at_nodes(self, n_nodes):
        return EnergyContour.declared(
            z0=self.z0,
            occupied_min=self.occupied_min,
            gap_ev=self.gap_ev,
            margin_ev=self.margin_ev,
            n_nodes=n_nodes,
        )

    def declaration_record(self):
        return {
            "form": "enclosing_ellipse_trapezoid",
            "z0_ev": self.z0,
            "occupied_min_ev": self.occupied_min,
            "margin_ev": self.margin_ev,
            "n_nodes": int(self.n_nodes),
            "declared_gap_ev": self.gap_ev,
        }


# ---------------------------------------------------------------------------
# production problem: bundle + vertex -> pencils and observables
# ---------------------------------------------------------------------------
def _normalized_weights(kpoints, k_weights):
    kpoints = tuple(kpoints)
    k_weights = tuple(float(w) for w in k_weights)
    if len(kpoints) != len(k_weights) or len(kpoints) == 0:
        raise SpiralResponseError(
            "kpoints and k_weights must be nonempty with matching lengths"
        )
    if not all(np.isfinite(w) and w > 0.0 for w in k_weights):
        raise SpiralResponseError(
            "k_weights must be finite and positive; refusing to weight the "
            "response with degenerate k samples"
        )
    total = sum(k_weights)
    return kpoints, [w / total for w in k_weights]


def _coord_text(coord):
    return json.dumps(list(coord), separators=(",", ":"))


def _pair_text(pair):
    return json.dumps([list(c) for c in pair], separators=(",", ":"))


def _payload_digest(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FrozenResponseProblem:
    """Rotating-frame pencils plus the declared rotation coordinates."""

    def __init__(self, bundle: SpiralBundle, vertex: MagneticVertex, *, axes_by_site):
        bundle.validate()
        if not vertex.partition:
            raise SpiralResponseError(
                "spatial partition undeclared in the magnetic-vertex "
                "certificate; rotation gates must refuse to guess (story-006)"
            )
        digest = _source_bundle_digest(bundle)
        if vertex.certificate.get("source_bundle_sha256") != digest:
            raise SpiralResponseError(
                "magnetic vertex is not bound to this bundle "
                "(source_bundle_sha256 mismatch); refusing to rotate blocks "
                "with a foreign vertex"
            )
        MagneticVertex.from_dict(vertex.to_dict())
        self._bundle = deepcopy(bundle)
        self._vertex = deepcopy(vertex)
        self._source_digest = digest
        self._importer = SpiralImporter()
        self._pencils = {}
        self._partition = {
            site: frozenset((mu, nu) for mu, nu, _r in keys)
            for site, keys in vertex.partition.items()
        }
        self._axes = {}
        for site in self.site_order:
            if not vertex.partition[site]:
                raise SpiralResponseError(
                    f"site {site!r} declares no incident blocks; it cannot "
                    "carry a rotation"
                )
            if site not in axes_by_site:
                raise SpiralResponseError(
                    f"axes_by_site lacks the declared site {site!r}"
                )
            axes = {}
            for label, vector in axes_by_site[site]:
                if not isinstance(label, str) or not label:
                    raise SpiralResponseError("axis labels must be nonempty strings")
                if label in axes:
                    raise SpiralResponseError(
                        f"site {site!r} declares axis label {label!r} twice"
                    )
                axes[label] = _unit_axis(vector)
            if not axes:
                raise SpiralResponseError(f"site {site!r} declares no rotation axes")
            self._axes[site] = axes

    @property
    def site_order(self):
        return tuple(self._vertex.partition)

    @property
    def sites(self):
        return tuple(self._vertex.partition)

    @property
    def bundle(self):
        return deepcopy(self._bundle)

    @property
    def vertex(self):
        return deepcopy(self._vertex)

    def axis_of(self, coord):
        site, label = coord
        try:
            return self._axes[site][label]
        except KeyError:
            if site not in self._axes:
                raise SpiralResponseError(
                    f"unknown site {site!r}; declared sites: {list(self._axes)}"
                ) from None
            raise SpiralResponseError(
                f"unknown axis label {label!r} for site {site!r}; declared: "
                f"{list(self._axes[site])}"
            ) from None

    def _phased_blocks(self, k):
        cache_key = tuple(np.asarray(k, dtype=float).ravel())
        if cache_key not in self._pencils:
            h, s = self._importer.assemble_pencil(self._bundle, k)
            _require_hermitian(h, "H(k)")
            _require_hermitian(s, "S(k)")
            n = self._bundle.norb
            scalar = np.zeros_like(h)
            blocks = {}
            for mu, nu in set().union(*self._partition.values()):
                indices = np.ix_([mu, mu + n], [nu, nu + n])
                block = h[indices]
                spin_scalar = np.trace(block) / 2.0 * np.eye(2)
                scalar[indices] = spin_scalar
                blocks[mu, nu] = block - spin_scalar
            for matrix in (h, s, scalar):
                matrix.setflags(write=False)
            self._pencils[cache_key] = h, s, scalar, blocks
        return self._pencils[cache_key]

    def _place(self, pieces):
        n = self._bundle.norb
        out = np.zeros((2 * n, 2 * n), complex)
        for (mu, nu), block in pieces.items():
            out[np.ix_([mu, mu + n], [nu, nu + n])] = block
        return out

    def pencil(self, k):
        h, s, _scalar, _blocks = self._phased_blocks(k)
        return h, s

    def vertex_pencil(self, k, coord):
        """Assembled first-order vertex V_coord(k)."""
        axis = self.axis_of(coord)
        keys = self._partition[coord[0]]
        _h, _s, _scalar, blocks = self._phased_blocks(k)
        pieces = first_vertex_blocks(blocks, keys, axis)
        return _require_hermitian(self._place(pieces), "V(k)")

    def second_vertex_pencil(self, k, coord_a, coord_b):
        """Assembled second-order vertex W_ab(k) in site order (Clairaut)."""
        a, b = self._ordered_pair(coord_a, coord_b)
        axis_a = self.axis_of(a)
        axis_b = self.axis_of(b)
        keys_a = self._partition[a[0]]
        keys_b = self._partition[b[0]]
        _h, _s, _scalar, blocks = self._phased_blocks(k)
        pieces = second_vertex_blocks(
            blocks, keys_a, keys_b, axis_a, axis_b, same_site=a[0] == b[0]
        )
        return _require_hermitian(self._place(pieces), "W(k)")

    def _ordered_pair(self, coord_a, coord_b):
        order = {site: i for i, site in enumerate(self.site_order)}
        for coord in (coord_a, coord_b):
            self.axis_of(coord)
        ia, ib = order[coord_a[0]], order[coord_b[0]]
        if ia <= ib:
            return coord_a, coord_b
        return coord_b, coord_a

    def rotated_pencil(self, k, thetas):
        """Pencil under finite rotations: ordered two-sided passes."""
        site_thetas = {}
        for site, entries in dict(thetas).items():
            if site not in self._axes:
                raise SpiralResponseError(
                    f"unknown site {site!r}; declared sites: {list(self._axes)}"
                )
            entries = dict(entries)
            generators = []
            for label, angle in entries.items():
                if label not in self._axes[site]:
                    raise SpiralResponseError(
                        f"unknown axis label {label!r} for site {site!r}"
                    )
                if not np.isfinite(angle):
                    raise SpiralResponseError("rotation angles must be finite")
                n = self._axes[site][label]
                generators.append(
                    float(angle) * sum(n[a] * _PAULI[a] for a in range(3))
                )
            if generators:
                site_thetas[site] = sum(generators)
        _h, s, scalar, blocks = self._phased_blocks(k)
        rotated = dict(blocks)
        for site in self.site_order:
            generator = site_thetas.get(site)
            if generator is None:
                continue
            u = _pass_unitary(generator)
            for key in self._partition[site]:
                rotated[key] = u @ rotated[key] @ u.conj().T
        h = scalar + self._place(rotated)
        return _require_hermitian(h, "rotated H(k)"), s

    def occupied_energy(self, kpoints, k_weights, thetas, ensemble):
        pts, weights = _normalized_weights(kpoints, k_weights)
        total = 0.0
        for k, w in zip(pts, weights):
            h, s = self.rotated_pencil(k, thetas)
            energy = occupied_band_energy(h, s, ensemble.n_occupied)
            total += w * energy
        return total

    def band_observables(self, kpoints, k_weights, ensemble, coords):
        """Band-path gradient/curvature with the onsite/response split."""
        coords = self._validate_coords(coords)
        pts, weights = _normalized_weights(kpoints, k_weights)
        dim = 2 * self._bundle.norb
        _checked_occupations(ensemble.n_occupied, dim)
        gradient = {c: 0.0 for c in coords}
        pairs = []
        for i, a in enumerate(coords):
            for b in coords[i:]:
                pairs.append((a, b))
        onsite = {pair: 0.0 for pair in pairs}
        response = {pair: 0.0 for pair in pairs}
        curvature = {pair: 0.0 for pair in pairs}
        for k, w in zip(pts, weights):
            h, s = self.pencil(k)
            eigenvalues, c = eigh(h, s)
            _require_open_gap(eigenvalues, ensemble.n_occupied)
            f = _checked_occupations(ensemble.n_occupied, dim)
            v_mats = {}
            for coord in coords:
                v = self.vertex_pencil(k, coord)
                _require_hermitian(v, "V(k)")
                v_mats[coord] = c.conj().T @ v @ c
                gradient[coord] += w * float(
                    np.sum(f * np.real(np.diag(v_mats[coord])))
                )
            for pair in pairs:
                a, b = self._ordered_pair(*pair)
                w_mat = self.second_vertex_pencil(k, a, b)
                wx = c.conj().T @ w_mat @ c
                value_onsite = w * float(np.sum(f * np.real(np.diag(wx))))
                kernel = _divided_difference(eigenvalues, f)
                value_response = w * float(
                    np.sum((kernel * v_mats[a] * v_mats[b].T).real)
                )
                onsite[pair] += value_onsite
                response[pair] += value_response
                curvature[pair] += value_onsite + value_response
        return {
            "gradient": dict(gradient),
            "curvature": {
                p: curvature[_ordered(p, coords)] for p in _all_ordered_pairs(coords)
            },
            "onsite": {
                p: onsite[_ordered(p, coords)] for p in _all_ordered_pairs(coords)
            },
            "response": {
                p: response[_ordered(p, coords)] for p in _all_ordered_pairs(coords)
            },
        }

    def contour_observables(self, kpoints, k_weights, coords, contour):
        """Contour-path observables via resolvent traces (no eigensolve)."""
        coords = self._validate_coords(coords)
        pts, weights = _normalized_weights(kpoints, k_weights)
        gradient = {c: 0.0 for c in coords}
        pairs = []
        for i, a in enumerate(coords):
            for b in coords[i:]:
                pairs.append((a, b))
        onsite = {pair: 0.0 for pair in pairs}
        response = {pair: 0.0 for pair in pairs}
        for k, w in zip(pts, weights):
            h, s = self.pencil(k)
            v_mats = {coord: self.vertex_pencil(k, coord) for coord in coords}
            w_mats = {}
            for pair in pairs:
                a, b = self._ordered_pair(*pair)
                w_mats[pair] = self.second_vertex_pencil(k, a, b)
            for z, weight in zip(contour.nodes, contour.weights):
                lu, piv = lu_factor(z * s - h)
                x_mats = {coord: lu_solve((lu, piv), v_mats[coord]) for coord in coords}
                for coord in coords:
                    gradient[coord] += w * weight * np.trace(x_mats[coord])
                for pair in pairs:
                    y = lu_solve((lu, piv), w_mats[pair])
                    onsite[pair] += w * weight * np.trace(y)
                    response[pair] += (
                        w * weight * np.trace(x_mats[pair[0]] @ x_mats[pair[1]])
                    )
        curvature = {pair: onsite[pair] + response[pair] for pair in pairs}
        return {
            "gradient": {c: float(gradient[c].real) for c in coords},
            "curvature": {
                p: float(curvature[_ordered(p, coords)].real)
                for p in _all_ordered_pairs(coords)
            },
            "onsite": {
                p: float(onsite[_ordered(p, coords)].real)
                for p in _all_ordered_pairs(coords)
            },
            "response": {
                p: float(response[_ordered(p, coords)].real)
                for p in _all_ordered_pairs(coords)
            },
        }

    def _validate_coords(self, coords):
        coords = tuple(tuple(c) for c in coords)
        if not coords:
            raise SpiralResponseError("at least one rotation coordinate is required")
        if len(set(coords)) != len(coords):
            raise SpiralResponseError("rotation coordinates must be unique")
        for coord in coords:
            self.axis_of(coord)
        return coords


def _ordered(pair, coords):
    a, b = pair
    return (a, b) if coords.index(a) <= coords.index(b) else (b, a)


def _all_ordered_pairs(coords):
    out = []
    for a in coords:
        for b in coords:
            out.append((a, b))
    return out


def _pass_unitary(generator):
    from scipy.linalg import expm

    return expm(-1j * generator / 2.0)


def _divided_difference(eigenvalues, occupations):
    numerator = occupations[:, None] - occupations[None, :]
    denominator = eigenvalues[:, None] - eigenvalues[None, :]
    # Equal-occupation degeneracies contribute exactly zero in the gapped
    # ensemble; never divide 0/0 (including the diagonal).
    kernel = np.zeros_like(denominator)
    np.divide(numerator, denominator, out=kernel, where=numerator != 0.0)
    return kernel


@dataclass(frozen=True)
class ResponseSet:
    """Versioned raw response with declarations, signed torques and evidence.

    ``axes[site][label]`` are the declared frame directions. Torque and
    gradient components use ``(site, label)`` keys; curvature uses coordinate
    pairs. All nested metadata are JSON-safe. No stationarity or spin-model
    identification is applied.
    """

    version: str
    convention: str
    site_order: tuple
    axes: dict
    coords: tuple
    gradient: dict
    torque: dict
    curvature: dict
    curvature_split: dict
    contour_vs_band_max_abs: float
    finite_rotation_cross_check: dict
    frozen_declarations: dict
    ensemble: dict
    provenance: dict
    integration: dict
    torque_state: dict
    ladder: dict

    def to_dict(self):
        payload = {
            "version": self.version,
            "convention": self.convention,
            "site_order": list(self.site_order),
            "axes": {
                site: {label: list(axis) for label, axis in axes.items()}
                for site, axes in self.axes.items()
            },
            "coords": [list(c) for c in self.coords],
            "gradient": {_coord_text(c): v for c, v in self.gradient.items()},
            "torque": {_coord_text(c): v for c, v in self.torque.items()},
            "curvature": {_pair_text(p): v for p, v in self.curvature.items()},
            "curvature_split": {
                _pair_text(p): dict(v) for p, v in self.curvature_split.items()
            },
            "contour_vs_band_max_abs": self.contour_vs_band_max_abs,
            "finite_rotation_cross_check": dict(self.finite_rotation_cross_check),
        }
        for field_name in (
            "frozen_declarations",
            "ensemble",
            "provenance",
            "integration",
            "torque_state",
            "ladder",
        ):
            payload[field_name] = getattr(self, field_name)
        # Return detached containers so mutation of serialized data does not
        # silently mutate the source record.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        payload["content_sha256"] = _payload_digest(payload)
        return payload

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != RESPONSE_VERSION:
            raise SpiralResponseError("unsupported response version")
        if data.get("convention") != RESPONSE_CONVENTION:
            raise SpiralResponseError("unsupported response convention")
        payload = {k: v for k, v in data.items() if k != "content_sha256"}
        if data.get("content_sha256") != _payload_digest(payload):
            raise SpiralResponseError("response payload digest mismatch")

        def coord(text):
            return tuple(json.loads(text))

        def pair(text):
            return tuple(tuple(c) for c in json.loads(text))

        return cls(
            version=data["version"],
            convention=data["convention"],
            site_order=tuple(data["site_order"]),
            axes={
                site: {label: tuple(axis) for label, axis in axes.items()}
                for site, axes in data["axes"].items()
            },
            coords=tuple(tuple(c) for c in data["coords"]),
            gradient={coord(c): v for c, v in data["gradient"].items()},
            torque={coord(c): v for c, v in data["torque"].items()},
            curvature={pair(p): v for p, v in data["curvature"].items()},
            curvature_split={
                pair(p): dict(v) for p, v in data["curvature_split"].items()
            },
            contour_vs_band_max_abs=data["contour_vs_band_max_abs"],
            finite_rotation_cross_check=dict(data["finite_rotation_cross_check"]),
            **{
                name: data[name]
                for name in (
                    "frozen_declarations",
                    "ensemble",
                    "provenance",
                    "integration",
                    "torque_state",
                    "ladder",
                )
            },
        )


def _finite_rotation_observables(problem, ensemble, coords, pts, weights, step):
    if not np.isfinite(step) or step <= 0.0:
        raise SpiralResponseError("finite-rotation step must be finite and positive")

    def energy(parameters):
        theta = {}
        for coord, value in parameters.items():
            theta.setdefault(coord[0], {})[coord[1]] = value
        return problem.occupied_energy(pts, weights, theta, ensemble)

    e0 = energy({})
    gradient, curvature = {}, {}
    for coord in coords:
        ep, em = energy({coord: step}), energy({coord: -step})
        gradient[coord] = (ep - em) / (2.0 * step)
        curvature[coord, coord] = (ep - 2.0 * e0 + em) / step**2
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            value = sum(
                sa * sb * energy({a: sa * step, b: sb * step})
                for sa in (-1, 1)
                for sb in (-1, 1)
            ) / (4.0 * step**2)
            curvature[a, b] = curvature[b, a] = value
    return {"gradient": gradient, "curvature": curvature}


def _rung(observables, previous, **parameters):
    torque = {_coord_text(c): -g for c, g in observables["gradient"].items()}
    curvature = {_pair_text(p): v for p, v in observables["curvature"].items()}
    return {
        **parameters,
        "torque": torque,
        "curvature": curvature,
        "torque_max_delta_vs_previous": (
            max(abs(v - previous["torque"][k]) for k, v in torque.items())
            if previous
            else None
        ),
        "curvature_max_delta_vs_previous": (
            max(abs(v - previous["curvature"][k]) for k, v in curvature.items())
            if previous
            else None
        ),
    }


def _ladder_converged(rungs, field, budget):
    return (
        len(rungs) >= 2
        and rungs[-1].get("identity_passed", True)
        and rungs[-1][field + "_max_delta_vs_previous"] <= budget
    )


def _reference_gate(problem, ensemble, pts, contour):
    """Certify occupations, Fermi level, indirect gap and pole enclosure."""
    ensemble.validate()
    homos, lumos = [], []
    for k in pts:
        h, s = problem.pencil(k)
        eigenvalues = eigh(h, s, eigvals_only=True)
        homo, lumo = _require_open_gap(eigenvalues, ensemble.n_occupied)
        homos.append(homo)
        lumos.append(lumo)
        if not homo < ensemble.fermi_energy_ev < lumo:
            raise SpiralResponseError(
                "declared Fermi energy is not inside the reference gap"
            )
        left = contour.occupied_min - contour.margin_ev
        right = contour.z0 + contour.margin_ev
        if not (left < eigenvalues[0] and homo < right < lumo):
            raise SpiralResponseError(
                "contour does not enclose exactly the occupied spectrum"
            )
    actual_gap = min(lumos) - max(homos)
    if (
        actual_gap <= _GAP_GATE_EV
        or abs(ensemble.gap_ev - actual_gap) > ENSEMBLE_GAP_TOL_EV
    ):
        raise SpiralResponseError(
            f"declared ensemble gap {ensemble.gap_ev:.12g} eV does not match "
            f"sampled spectral gap {actual_gap:.12g} eV "
            f"(budget {ENSEMBLE_GAP_TOL_EV:.1e} eV)"
        )
    return actual_gap


def _observable_mismatch(analytic, other):
    return {
        field: max(
            abs(analytic[field][key] - other[field][key]) for key in analytic[field]
        )
        for field in ("gradient", "curvature")
    }


def _integration_rung(obs, band, previous, **parameters):
    rung = _rung(obs, previous, **parameters)
    mismatch = _observable_mismatch(band, obs)
    rung["identity_max_abs_ev"] = max(mismatch.values())
    rung["identity_passed"] = rung["identity_max_abs_ev"] <= RESPONSE_IDENTITY_TOL_EV
    rung["within_approved_budgets"] = (
        previous is not None
        and rung["torque_max_delta_vs_previous"] <= TORQUE_BUDGET_EV_PER_RAD
        and rung["curvature_max_delta_vs_previous"] <= CURVATURE_BUDGET_EV_PER_RAD2
    )
    return rung


def evaluate_response_set(
    problem,
    *,
    ensemble,
    coords,
    kpoints,
    k_weights,
    contour,
    finite_steps=(4e-3, 2e-3),
    contour_node_rungs=(),
    contour_rungs=(),
    k_mesh_rungs=(),
):
    """Evaluate response with finite-step, node, contour-height and k ladders.

    ``contour_rungs`` contains complete EnergyContour declarations, allowing
    energy-window, imaginary-height (resolvent broadening) and quadrature
    variation. This is not a metallic-ensemble Lorentzian broadening.
    ``k_mesh_rungs`` contains mappings with ``label``, ``kpoints``,
    ``k_weights`` and the explicitly matched ``ensemble`` for each mesh.
    All rung values and adjacent deltas are recorded. Undersampled early
    integration rungs are labeled failed identities; the production contour
    must pass the 1e-8 eV identity gate. Finite-rotation cross-checks record
    actual discrepancies with analytic values, separately from rung deltas.
    """
    coords = problem._validate_coords(coords)
    pts, weights = _normalized_weights(kpoints, k_weights)
    actual_gap = _reference_gate(problem, ensemble, pts, contour)
    band = problem.band_observables(pts, weights, ensemble, coords)
    circ = problem.contour_observables(pts, weights, coords, contour)
    identity = max(_observable_mismatch(band, circ).values())
    if not np.isfinite(identity) or identity > RESPONSE_IDENTITY_TOL_EV:
        raise SpiralResponseError(
            f"contour/eigenbasis identity mismatch {identity:.3e} eV exceeds "
            f"{RESPONSE_IDENTITY_TOL_EV:.1e} eV; refine the production contour"
        )
    finite_rungs, node_rungs, parameter_rungs, mesh_rungs = [], [], [], []
    cross_check = {
        "step": None,
        "torque_max_abs_mismatch": None,
        "curvature_max_abs_mismatch": None,
        "scope": "maximum over coordinates/blocks on the last finite-step rung",
    }
    for step in finite_steps:
        obs = _finite_rotation_observables(
            problem, ensemble, coords, pts, weights, float(step)
        )
        mismatch = _observable_mismatch(band, obs)
        cross_check = {
            "step": float(step),
            "torque_max_abs_mismatch": mismatch["gradient"],
            "curvature_max_abs_mismatch": mismatch["curvature"],
            "scope": "maximum over coordinates/blocks on the last finite-step rung",
        }
        finite_rungs.append(
            _rung(
                obs,
                finite_rungs[-1] if finite_rungs else None,
                step=float(step),
                analytic_vs_finite_rotation=dict(cross_check),
            )
        )
    for nodes in contour_node_rungs:
        varied = contour.at_nodes(nodes)
        obs = problem.contour_observables(pts, weights, coords, varied)
        node_rungs.append(
            _integration_rung(
                obs,
                band,
                node_rungs[-1] if node_rungs else None,
                nodes=int(nodes),
                contour=varied.declaration_record(),
            )
        )
    for varied in contour_rungs:
        _reference_gate(problem, ensemble, pts, varied)
        obs = problem.contour_observables(pts, weights, coords, varied)
        parameter_rungs.append(
            _integration_rung(
                obs,
                band,
                parameter_rungs[-1] if parameter_rungs else None,
                contour=varied.declaration_record(),
                resolvent_imaginary_height_ev=varied.margin_ev,
            )
        )
    for mesh in k_mesh_rungs:
        if "ensemble" not in mesh:
            raise SpiralResponseError(
                "each k-mesh rung requires its matched ensemble declaration"
            )
        varied_pts, varied_weights = _normalized_weights(
            mesh["kpoints"], mesh["k_weights"]
        )
        rung_ensemble = mesh["ensemble"]
        rung_gap = _reference_gate(problem, rung_ensemble, varied_pts, contour)
        varied_band = problem.band_observables(
            varied_pts, varied_weights, rung_ensemble, coords
        )
        obs = problem.contour_observables(varied_pts, varied_weights, coords, contour)
        mesh_rungs.append(
            _integration_rung(
                obs,
                varied_band,
                mesh_rungs[-1] if mesh_rungs else None,
                label=mesh["label"],
                kpoints=[np.asarray(k).tolist() for k in varied_pts],
                k_weights=list(varied_weights),
                ensemble=rung_ensemble.to_record(),
                actual_gap_ev=rung_gap,
            )
        )
    budgets = {
        "torque_ev_per_rad": TORQUE_BUDGET_EV_PER_RAD,
        "curvature_ev_per_rad2": CURVATURE_BUDGET_EV_PER_RAD2,
    }
    enabled_ladders = [
        r for r in (finite_rungs, node_rungs, parameter_rungs, mesh_rungs) if r
    ]
    converged = {
        field: bool(enabled_ladders)
        and all(_ladder_converged(r, field, budget) for r in enabled_ladders)
        for field, budget in (
            ("torque", TORQUE_BUDGET_EV_PER_RAD),
            ("curvature", CURVATURE_BUDGET_EV_PER_RAD2),
        )
    }
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
    cert = dict(problem._vertex.certificate)
    torque = {c: -g for c, g in band["gradient"].items()}
    axes = {site: dict(problem._axes[site]) for site in problem.site_order}
    return ResponseSet(
        version=RESPONSE_VERSION,
        convention=RESPONSE_CONVENTION,
        site_order=problem.site_order,
        axes=axes,
        coords=coords,
        gradient=band["gradient"],
        torque=torque,
        curvature=band["curvature"],
        curvature_split={
            p: {"onsite": band["onsite"][p], "response": band["response"][p]}
            for p in band["curvature"]
        },
        contour_vs_band_max_abs=identity,
        finite_rotation_cross_check=cross_check,
        frozen_declarations={
            "declaration": "ADR-005: density, scalar potential, magnetic amplitudes, basis and S frozen",
            "scalar_potential": "per-k spin-block trace/2 of the phased H(k), invariant under each same-U conjugation",
            "stored_scalar_sha256": scalar_digest.hexdigest(),
            "scalar_sha256": phased_scalar_digest.hexdigest(),
            "magnetic_amplitudes": {
                "vertex_sha256": cert["vertex_sha256"],
                "source_bundle_sha256": cert["source_bundle_sha256"],
                "certificate": cert,
                "block_count": len(problem._vertex.blocks),
                "phased_blocks_sha256": phased_amplitude_digest.hexdigest(),
                "phase_placement": "bundle -> importer -> per-k Pauli split; no real-space repacking",
            },
            "overlap": "assembled spin-dependent S(k) frozen",
        },
        ensemble={
            **ensemble.to_record(),
            "actual_gap_ev": actual_gap,
            "gap_validation_budget_ev": ENSEMBLE_GAP_TOL_EV,
            "occupations_per_k": [
                [1] * ensemble.n_occupied
                + [0] * (2 * problem._bundle.norb - ensemble.n_occupied)
                for _k in pts
            ],
        },
        provenance={
            "source_bundle_sha256": cert["source_bundle_sha256"],
            "vertex_sha256": cert["vertex_sha256"],
            "bundle_version": problem._bundle.version,
            "vertex_convention": VERTEX_CONVENTION,
            "q": list(problem._bundle.q),
            "derivation_authority": "docs/sympy/siesta_spiral_frozen_response.py",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rotation_order": list(problem.site_order),
            "metric": "G=(zS-H)^-1; Tr(G V)=Tr((G S)(S^-1 V))",
        },
        integration={
            "kpoints": [np.asarray(k).tolist() for k in pts],
            "k_weights": list(weights),
            "contour": contour.declaration_record(),
        },
        torque_state={
            "declaration": "FR-008: nonzero reference torques retained; no stationarity identity applied",
            "nonzero_reference_torque": any(
                abs(v) > _NONZERO_TORQUE_TOL for v in torque.values()
            ),
            "reference_torques": {_coord_text(c): v for c, v in torque.items()},
        },
        ladder={
            "budgets": budgets,
            "finite_rotation": finite_rungs,
            "contour_nodes": node_rungs,
            "contour_parameters": parameter_rungs,
            "k_mesh": mesh_rungs,
            "converged": converged,
        },
    )
