"""Displacement seam integration (story 017, ADR-008).

Provides dH/dR and dS/dR displacement derivatives in the *nonorth-spin-
lattice* producer/consumer conventions, reused verbatim (never
reimplemented): sparse derivative records keyed by the SAME
``(mu, nu, R)`` pattern as the reference :class:`SpiralBundle` H/S
sparse dicts, with the pinned 4-component spiral packing
``H1=H11, H2=H22, H3=Re H12, H4=-Im H12`` in file order.

Two producers:

* :class:`DhsdrNcProvider` — the frozen-potential SIESTA ``dHSdR.nc``
  reader (requires the unmerged ``HamiltonIO.siesta.dhsdr`` module);
* :class:`DisplacedReferenceFDProvider` — central finite differences
  from independent displaced SCF HSX exports, read through the
  certified story-014 loader path (sisl with the ``-Ef`` shift undone).

Units after ingestion: ``dH`` in eV/Ang, ``dS`` in 1/Ang (derivatives
with respect to a Cartesian displacement measured in Angstrom).  The
seam gauge label (``periodic-image-sum`` | ``localized``) is carried
verbatim end to end and cross-checked against the bundle's declared
gauge at assembly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path

import numpy as np

from TB2J.interfaces.siesta_spiral import SpiralBundle
from TB2J.interfaces.siesta_spiral_hubbard import (
    BOHR_TO_ANG,
    _bundle_from_hs_geometry,
    xc_source_payload_sha256,
)

BOHR_TO_ANG_ = BOHR_TO_ANG
RY_TO_EV = 13.605693122994
GEOMETRY_TOL_BOHR = 1e-6
PATTERN_TOL = 1e-8

GAUGE_LABELS = frozenset({"periodic-image-sum", "localized"})
SUPPORTED_BRANCHES = frozenset({"central"})
_AXES = {"x": 0, "y": 1, "z": 2}


def expected_units():
    """The declared post-ingestion unit convention for derivative records."""
    return {"dH": "eV/Ang", "dS": "1/Ang"}


# ---------------------------------------------------------------------------
# Refusal classes: one diagnostic class per mismatch kind
# ---------------------------------------------------------------------------


class DisplacementSeamError(RuntimeError):
    """Base class: displacement data inconsistent with the spiral bundle."""


class GeometryMismatch(DisplacementSeamError):
    """Reference positions/cell differ from the bundle beyond tolerance."""


class GaugeMismatch(DisplacementSeamError):
    """Gauge label missing, unknown, or inconsistent between record and bundle."""


class MissingDisplacedReference(DisplacementSeamError):
    """The +h/-h displaced reference pair is incomplete."""


class ConventionDrift(DisplacementSeamError):
    """Units / Fermi flags / declared pattern are inconsistent with the record."""


class SparsePatternMismatch(DisplacementSeamError):
    """Derivative sparse pattern differs from the bundle's H/S pattern."""


class WrongDisplacementSign(DisplacementSeamError):
    """The +/- pair's signed displacements contradict the declared pattern."""


class DigestMismatch(DisplacementSeamError):
    """Path-backed references' sha256 contradicts ``reference_digests``."""


class UnsupportedBranch(DisplacementSeamError):
    """Only the central finite-difference branch is supported."""


class DhsdrUnavailable(DisplacementSeamError):
    """``HamiltonIO.siesta.dhsdr`` (unmerged upstream) is not importable."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DisplacementPattern:
    """A declared atomic displacement.

    ``atom_index`` is 1-based (SIESTA convention); ``axis`` is a
    Cartesian axis ``"x"|"y"|"z"``; ``step_bohr`` is the FD step in
    Bohr.  Only the ``central`` branch is supported.  ``reference_digests``
    pins the displaced-reference set (one digest per +h/-h file, in
    that order).
    """

    atom_index: int
    axis: str
    step_bohr: float
    branch: str = "central"
    reference_digests: tuple = ()

    def __post_init__(self):
        if self.branch not in SUPPORTED_BRANCHES:
            raise UnsupportedBranch(
                f"displacement branch {self.branch!r} is not supported; only "
                "'central' finite differences are implemented (use +/- step "
                "displaced SCF pairs)"
            )
        if not self.step_bohr > 0.0:
            raise UnsupportedBranch(
                f"step must be a positive magnitude, got {self.step_bohr!r}; "
                "the +/- sides are encoded by the displaced reference pair"
            )
        if self.axis not in _AXES:
            raise UnsupportedBranch(
                f"axis {self.axis!r} not recognized; use 'x', 'y' or 'z'"
            )
        if not (isinstance(self.atom_index, int) and self.atom_index >= 1):
            raise UnsupportedBranch(
                f"atom_index must be a 1-based integer, got {self.atom_index!r}"
            )

    @property
    def axis_index(self):
        """1-based SIESTA Cartesian axis index."""
        return _AXES[self.axis] + 1

    @property
    def direction(self):
        d = [0.0, 0.0, 0.0]
        d[_AXES[self.axis]] = 1.0
        return tuple(d)


@dataclass(frozen=True)
class GeometrySource:
    """Reference geometry as recorded by a provider (Bohr).

    ``displaced_axis_offset_bohr`` is the SIGNED displaced atom's offset
    along the declared axis present in the recorded positions, measured
    against the undisplaced reference geometry (0.0 for direct-derivative
    producers whose geometry IS the reference geometry).  A central
    finite-difference record's geometry is the +h side, so this is +step.
    """

    cell: tuple
    orbital_positions: tuple
    displaced_axis_offset_bohr: float = 0.0
    displaced_orbital_indices: tuple = ()


@dataclass(frozen=True)
class DisplacementDerivatives:
    """Sparse dH/dR and dS/dR in the bundle's conventions.

    ``dH`` maps ``(mu, nu, R)`` to the 4-component packing (file order,
    eV/Ang); ``dS`` maps the same keys to a scalar overlap derivative in
    1/Ang.  Keys may be a superset of any particular bundle's pattern;
    :func:`assemble_onto_bundle` enforces exact coverage.
    """

    pattern: DisplacementPattern
    dH: dict
    dS: dict
    producer_recipe: str
    branch: str
    step_bohr: float
    reference_fermi_ev: float | None
    gauge: str
    units: dict
    reference_digests: tuple
    reference_geometry: GeometrySource


def displacement_payload_sha256(record):
    """Bind every field of a seam or vertex derivative record canonically.

    Dataclasses (including the pattern and reference geometry) are expanded
    without copying their arrays. Sparse mappings retain every key/value;
    units, declarations and provenance are included, not merely the onsite
    magnetic components compared by AC-009. The shared AC-008 canonical
    hashing convention refuses unsupported payload values rather than
    certifying their repr. No vertex-module import is needed here.
    """
    if not is_dataclass(record) or isinstance(record, type):
        raise TypeError("displacement payload must be a derivative dataclass record")

    def payload(value):
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "record_type": type(value).__name__,
                "fields": {
                    item.name: payload(getattr(value, item.name))
                    for item in fields(value)
                },
            }
        if isinstance(value, dict):
            return {key: payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return tuple(payload(item) for item in value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    return xc_source_payload_sha256(payload(record))


@dataclass(frozen=True)
class SpiralBundleWithDisplacements:
    """A bundle with displacement derivatives bound to its conventions.

    The original bundle is never mutated; the bundle-level provenance
    (its metadata, plus the ``displacement`` entry) lives here.
    """

    bundle: SpiralBundle
    derivatives: DisplacementDerivatives
    pattern: DisplacementPattern
    provenance: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class DhsdrNcProvider:
    """Frozen-potential dH/dS derivatives from a SIESTA ``dHSdR.nc`` file.

    Requires the unmerged ``HamiltonIO.siesta.dhsdr`` module; the import
    failure is a distinct actionable refusal (use the central-difference
    :class:`DisplacedReferenceFDProvider` on displaced SCF HSX exports
    instead).
    """

    gauge = "periodic-image-sum"

    def __init__(
        self,
        nc_path,
        pattern,
        reference_geometry=None,
        reference_fermi_ev=None,
        sparse_keys=None,
    ):
        self.nc_path = Path(nc_path)
        self.pattern = pattern
        self.reference_geometry = reference_geometry
        self.reference_fermi_ev = reference_fermi_ev
        # the bundle's H/S pattern; when given, the record is emitted over
        # exactly these keys (dense dHSdR blocks are sampled at them)
        self.sparse_keys = (
            frozenset(tuple(k) for k in sparse_keys)
            if sparse_keys is not None
            else None
        )

    def provide(self) -> DisplacementDerivatives:
        try:
            from HamiltonIO.siesta.dhsdr import read_dHSdR
        except ImportError as exc:
            raise DhsdrUnavailable(
                "HamiltonIO.siesta.dhsdr is not importable: the dHSdR.nc "
                "reader is an unmerged upstream dependency of the "
                "nonorth-spin-lattice seam. Install the patched HamiltonIO, "
                "or use DisplacedReferenceFDProvider (central finite "
                "differences over +/- displaced SCF HSX exports) instead."
            ) from exc
        blocks = list(
            read_dHSdR(
                str(self.nc_path),
                displacements=[(self.pattern.atom_index, self.pattern.axis_index)],
            )
        )
        if len(blocks) != 1:
            raise MissingDisplacedReference(
                f"dHSdR.nc {self.nc_path} has no unique derivative block for "
                f"atom {self.pattern.atom_index} axis {self.pattern.axis} "
                f"(got {len(blocks)} blocks; check FC.dHdR.Tolerance)"
            )
        block = blocks[0]
        dhr = np.asarray(block.dHR, dtype=float)
        dsr = np.asarray(block.dSR, dtype=float)
        if dhr.size == 0:
            raise MissingDisplacedReference(
                f"dHSdR block (atom {self.pattern.atom_index}, axis "
                f"{self.pattern.axis}) is empty — check FC.dHdR.Tolerance"
            )
        if dhr.shape[-1] not in (1, 2):
            raise ConventionDrift(
                f"dHSdR spin dimension {dhr.shape[-1]} is not supported "
                "(collinear/non-polarized only); refusing to guess the "
                "spin packing"
            )
        rlist = [tuple(int(v) for v in R) for R in np.atleast_2d(block.Rlist)]
        rindex = {R: iR for iR, R in enumerate(rlist)}
        norb = dhr.shape[1]
        dH_unit = RY_TO_EV / BOHR_TO_ANG_  # Ry/Bohr -> eV/Ang
        dS_unit = 1.0 / BOHR_TO_ANG_  # 1/Bohr -> 1/Ang

        def _components(iR, mu, nu):
            if dhr.shape[-1] == 1:
                up = dn = dhr[iR, :, :, 0]
            else:
                up, dn = dhr[iR, :, :, 0], dhr[iR, :, :, 1]
            # collinear dHSdR is spin-block-diagonal: H3/H4 are zero
            return (
                dH_unit * float(up[mu, nu].real),
                dH_unit * float(dn[mu, nu].real),
                0.0,
                0.0,
            )

        dH, dS = {}, {}
        if self.sparse_keys is None:
            keys = (
                (mu, nu, R) for R in rlist for mu in range(norb) for nu in range(norb)
            )
        else:
            keys = sorted(self.sparse_keys)
        for mu, nu, R in keys:
            if R not in rindex:
                raise SparsePatternMismatch(
                    f"sparse key {(mu, nu, R)} references cell {R} outside the "
                    f"dHSdR.nc R-list {rlist[:5]}...; the derivative file does "
                    "not span the bundle's pattern"
                )
            iR = rindex[R]
            dH[(mu, nu, R)] = _components(iR, mu, nu)
            dS[(mu, nu, R)] = dS_unit * float(dsr[iR, mu, nu].real)
        recipe = (
            "siesta dHSdR.nc frozen-potential central difference "
            "(read_dHSdR); "
            f"atom_index={self.pattern.atom_index}, "
            f"axis={self.pattern.axis}, direction={self.pattern.direction}"
        )
        return DisplacementDerivatives(
            pattern=self.pattern,
            dH=dH,
            dS=dS,
            producer_recipe=recipe,
            branch=self.pattern.branch,
            step_bohr=self.pattern.step_bohr,
            reference_fermi_ev=self.reference_fermi_ev,
            gauge=self.gauge,
            units=expected_units(),
            reference_digests=self.pattern.reference_digests,
            reference_geometry=self.reference_geometry,
        )


class DisplacedReferenceFDProvider:
    """Central finite differences from +/- displaced SCF HSX exports.

    Both displaced runs are read through the certified story-014 loader
    (sisl with the ``-Ef`` shift undone, file packing preserved), so
    ``dX = (X(+h) - X(-h)) / (2h)`` is taken in exactly the bundle's
    sparse key convention and file packing.  ``h`` is the declared step
    (Bohr) converted to Ang; ``dH`` comes out in eV/Ang and ``dS`` in
    1/Ang.

    Either ``plus_path``/``minus_path`` or pre-built siles
    (``plus_sile``/``minus_sile``) may be supplied; the sile form is the
    injection seam used by the tests.
    """

    gauge = "periodic-image-sum"

    def __init__(
        self,
        pattern,
        plus_path=None,
        minus_path=None,
        plus_sile=None,
        minus_sile=None,
        reference_fermi_ev=None,
        hsx_opener=None,
        reference_bundle=None,
    ):
        self.pattern = pattern
        self._paths = (plus_path, minus_path)
        self._siles = (plus_sile, minus_sile)
        self._reference_fermi_ev = reference_fermi_ev
        self._hsx_opener = hsx_opener or self._default_opener
        self.reference_bundle = reference_bundle

    @staticmethod
    def _default_opener(path):
        try:
            import sisl
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise DisplacementSeamError(
                "sisl is required to read displaced HSX exports; install "
                "sisl>=0.16 or inject pre-built siles"
            ) from exc
        return sisl.get_sile(str(path))

    def provide(self) -> DisplacementDerivatives:
        plus, minus = self._resolve_siles()
        plus_bundle = self._read(plus)
        minus_bundle = self._read(minus)
        if self.reference_bundle is None:
            raise DisplacementSeamError(
                "the undisplaced reference bundle is required: the central "
                "difference must be validated as +h/-h displacements OF THE "
                "REFERENCE GEOMETRY, not merely of the pair's own midpoint "
                "(a pair displaced about a wrong centre would otherwise "
                "pass); pass reference_bundle= or an explicit reference "
                "geometry anchor"
            )
        self._check_signed_displacements(plus_bundle, minus_bundle)
        if set(plus_bundle.sparse) != set(minus_bundle.sparse):
            raise SparsePatternMismatch(
                "the +h and -h displaced HSX exports carry different sparse "
                "patterns; central differences require matched displaced "
                "references (same cutoff, same basis, same supercell)"
            )
        h_step_ang = self.pattern.step_bohr * BOHR_TO_ANG_
        dH = {
            key: tuple(
                (p - m) / (2.0 * h_step_ang)
                for p, m in zip(plus_bundle.sparse[key], minus_bundle.sparse[key])
            )
            for key in plus_bundle.sparse
        }
        dS = {
            key: (plus_bundle.overlap[key] - minus_bundle.overlap[key])
            / (2.0 * h_step_ang)
            for key in plus_bundle.sparse
        }
        # displaced geometry: the displaced atom must be the only one moved
        delta = np.asarray(plus_bundle.orbital_positions) - np.asarray(
            minus_bundle.orbital_positions
        )
        moved = sorted(
            int(i) for i in np.where(np.abs(delta).max(axis=1) > GEOMETRY_TOL_BOHR)[0]
        )
        displaced_orbitals = _atom_orbital_group(plus_bundle, self.pattern.atom_index)
        if set(moved) - set(displaced_orbitals):
            raise GeometryMismatch(
                "the +/- displaced references move orbitals outside the "
                f"displaced atom {self.pattern.atom_index}: "
                f"{sorted(set(moved) - set(displaced_orbitals))}; the "
                "displaced SCF pair must differ only by the declared pattern"
            )
        offset = 0.0
        if displaced_orbitals:
            # SIGNED plus-side offset vs the reference geometry (the gate
            # above already required +h; recorded for assembly checks)
            ref_pos = np.mean(
                [
                    np.asarray(self.reference_bundle.orbital_positions[i], dtype=float)
                    for i in _atom_orbital_group(
                        self.reference_bundle, self.pattern.atom_index
                    )
                ],
                axis=0,
            )
            offset = float(
                np.dot(
                    np.mean(
                        [
                            np.asarray(plus_bundle.orbital_positions[i], dtype=float)
                            for i in displaced_orbitals
                        ],
                        axis=0,
                    )
                    - ref_pos,
                    self.pattern.direction,
                )
            )
        if plus_bundle.cell != minus_bundle.cell:
            raise GeometryMismatch(
                "the +/- displaced references have different cells; refusing "
                "to difference different geometries"
            )
        reference_fermi = self._reference_fermi_ev
        if reference_fermi is None and self._siles[0] is None:
            ef = plus.read_fermi_level()
            reference_fermi = float(ef) if ef is not None else 0.0
        digests = self._bind_reference_digests()
        recipe = (
            "siesta displaced-SCF HSX central finite difference "
            "(sisl read, -Ef shift undone per story-014); "
            f"atom_index={self.pattern.atom_index}, "
            f"axis={self.pattern.axis}, direction={self.pattern.direction}, "
            f"step_bohr={self.pattern.step_bohr}"
        )
        return DisplacementDerivatives(
            pattern=self.pattern,
            dH=dH,
            dS=dS,
            producer_recipe=recipe,
            branch=self.pattern.branch,
            step_bohr=self.pattern.step_bohr,
            reference_fermi_ev=reference_fermi,
            gauge=self.gauge,
            units=expected_units(),
            reference_digests=tuple(digests),
            reference_geometry=GeometrySource(
                cell=plus_bundle.cell,
                orbital_positions=plus_bundle.orbital_positions,
                displaced_axis_offset_bohr=offset,
                displaced_orbital_indices=tuple(displaced_orbitals),
            ),
        )

    def _check_signed_displacements(self, plus_bundle, minus_bundle):
        """Signed geometry gate: each side must move the declared atom by
        the SIGNED declared step along the declared axis, measured against
        the declared reference geometry (both bundles' orbital positions
        are in Bohr, like ``step_bohr``).  The reference is the bundle's
        undisplaced geometry — not the pair's midpoint, which would pass
        a pair displaced around a wrong centre.

        Refuses zero-offset (undisplaced file used as a side),
        sign-swapped, and half-step pairs with distinct actionable
        messages; a bare +/- displacement magnitude is not evidence of a
        central pair about the reference.
        """
        pattern = self.pattern
        axis = np.asarray(pattern.direction, dtype=float)
        step = float(pattern.step_bohr)
        side_pos = {}
        for label, bundle in (("+h", plus_bundle), ("-h", minus_bundle)):
            orbitals = _atom_orbital_group(bundle, pattern.atom_index)
            side_pos[label] = np.mean(
                [
                    np.asarray(bundle.orbital_positions[i], dtype=float)
                    for i in orbitals
                ],
                axis=0,
            )
        reference = np.mean(
            [
                np.asarray(self.reference_bundle.orbital_positions[i], dtype=float)
                for i in _atom_orbital_group(self.reference_bundle, pattern.atom_index)
            ],
            axis=0,
        )
        tol = GEOMETRY_TOL_BOHR
        expected = {"+h": step, "-h": -step}
        for label in ("+h", "-h"):
            observed = float(np.dot(side_pos[label] - reference, axis))
            want = expected[label]
            if abs(observed - want) <= tol:
                continue
            if abs(observed) <= tol:
                raise WrongDisplacementSign(
                    f"the {label} displaced reference shows ZERO offset of "
                    f"atom {pattern.atom_index} along {pattern.axis} "
                    f"({observed:+.6f} Bohr, expected {want:+.6f}); the +h/-h "
                    "pair must be two genuinely displaced SCFs — identical "
                    "or undisplaced files are refused"
                )
            if abs(abs(observed) - step) <= tol:
                raise WrongDisplacementSign(
                    f"the {label} displaced reference moves atom "
                    f"{pattern.atom_index} along {pattern.axis} by "
                    f"{observed:+.6f} Bohr but the declared pattern expects "
                    f"{want:+.6f} Bohr; the pair's displacement signs are "
                    "swapped relative to the declared +h/-h roles"
                )
            raise WrongDisplacementSign(
                f"the {label} displaced reference moves atom "
                f"{pattern.atom_index} along {pattern.axis} by "
                f"{observed:+.6f} Bohr but the declared pattern expects "
                f"{want:+.6f} Bohr; the pair is not a full-step central pair "
                f"(step {step} Bohr) — refusing a half-step or corrupted pair"
            )

    def _bind_reference_digests(self):
        """Digest binding: path-backed references are hashed and compared
        to the declared ``reference_digests``; hash failures and
        sile-backed references with a digest declaration are refused as
        missing digest evidence.
        """
        declared = tuple(self.pattern.reference_digests)
        if self._paths[0] is not None and self._paths[1] is not None:
            computed = []
            for label, p in (("+h", self._paths[0]), ("-h", self._paths[1])):
                try:
                    computed.append(_sha256(p))
                except OSError as exc:
                    raise DigestMismatch(
                        f"cannot hash the {label} displaced reference "
                        f"{p}: {exc}; the declared reference_digests "
                        "require verifiable HSX files"
                    ) from exc
            computed = tuple(computed)
            if declared and declared != computed:
                raise DigestMismatch(
                    "displaced reference digests do not match the declared "
                    f"reference_digests: declared {declared}, computed "
                    f"{computed}; the pinned displaced references have "
                    "changed since the pattern was declared"
                )
            return computed if not declared else declared
        if declared:
            raise DigestMismatch(
                "reference_digests are declared but the displaced references "
                "are pre-built siles with no hashable files; provide the HSX "
                "paths or drop the digest declaration"
            )
        return ()

    def _resolve_siles(self):
        if self._siles[0] is not None and self._siles[1] is not None:
            return self._siles
        missing = [str(p) for p in self._paths if p is None or not Path(p).is_file()]
        if missing:
            raise MissingDisplacedReference(
                "incomplete +/- displaced reference pair for central finite "
                f"differences; missing: {missing}. Each displaced reference "
                "is an independent SCF at +step and -step along the declared "
                "axis."
            )
        return tuple(self._hsx_opener(p) for p in self._paths)

    @staticmethod
    def _read(sile):
        return _bundle_from_hs_geometry(sile, stdout_name="displaced", proj_name="")


def _atom_orbital_group(bundle, atom_index):
    """Orbital indices belonging to 1-based ``atom_index`` in a bundle.

    The certified loader records per-orbital positions equal to their
    atom's position, so orbitals group by shared position in order of
    first occurrence.
    """
    positions = [tuple(np.round(p, 8)) for p in bundle.orbital_positions]
    groups = {}
    order = []
    for i, p in enumerate(positions):
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(i)
    if not 1 <= atom_index <= len(order):
        raise GeometryMismatch(
            f"atom_index {atom_index} outside the reference geometry's "
            f"{len(order)} atoms"
        )
    return groups[order[atom_index - 1]]


# ---------------------------------------------------------------------------
# Assembly onto the bundle
# ---------------------------------------------------------------------------


def assemble_onto_bundle(bundle, derivatives, pattern):
    """Bind displacement derivatives to a bundle's rotating-frame conventions.

    Refuses (actionably, one class per mismatch kind) on geometry,
    gauge, convention, or sparse-pattern inconsistency.  Returns a new
    light :class:`SpiralBundleWithDisplacements`; the input bundle is
    never mutated.
    """
    if pattern != derivatives.pattern:
        raise ConventionDrift(
            f"the declared pattern {pattern} does not match the record's "
            f"pattern {derivatives.pattern}; refusing to mix displacement "
            "declarations"
        )
    if (
        derivatives.branch != pattern.branch
        or derivatives.step_bohr != pattern.step_bohr
    ):
        raise ConventionDrift(
            "derivative record branch/step disagree with the declared "
            f"pattern (record: branch={derivatives.branch}, "
            f"step={derivatives.step_bohr})"
        )
    if bundle.fermi_shift != "none":
        raise ConventionDrift(
            f"bundle declares fermi_shift={bundle.fermi_shift!r}; displacement "
            "records are assembled against the raw producer H (sisl's -Ef "
            "shift undone, fermi_shift='none')"
        )
    if derivatives.units != expected_units():
        raise ConventionDrift(
            f"derivative record units {derivatives.units} drift from the "
            f"declared convention {expected_units()}"
        )
    # gauge: carried verbatim, cross-checked against the bundle's declaration
    if derivatives.gauge not in GAUGE_LABELS:
        raise GaugeMismatch(
            f"derivative record gauge {derivatives.gauge!r} is unknown; "
            f"expected one of {sorted(GAUGE_LABELS)}"
        )
    bundle_gauge = bundle.metadata.get("displacement_gauge")
    if bundle_gauge is None:
        raise GaugeMismatch(
            "the bundle does not declare a displacement gauge (set "
            "metadata['displacement_gauge'] to 'periodic-image-sum' or "
            "'localized'); refusing to guess which observable the "
            "displaced references realize"
        )
    if bundle_gauge not in GAUGE_LABELS:
        raise GaugeMismatch(
            f"bundle displacement gauge {bundle_gauge!r} is unknown; "
            f"expected one of {sorted(GAUGE_LABELS)}"
        )
    if bundle_gauge != derivatives.gauge:
        raise GaugeMismatch(
            f"gauge mismatch: record is '{derivatives.gauge}' but the bundle "
            f"declares '{bundle_gauge}'; the periodic-image-sum and "
            "localized gauges are different observables and must not be mixed"
        )
    # geometry: same cell; only the displaced atom may move, by +/- the
    # declared step along the declared axis (0 for direct derivatives)
    geom = derivatives.reference_geometry
    if bundle.cell is None or bundle.orbital_positions is None:
        raise GeometryMismatch(
            "the bundle lacks vector-path geometry (cell/orbital_positions); "
            "displacement derivatives require it to verify the displaced "
            "reference geometry"
        )
    if geom is None:
        raise GeometryMismatch(
            "the derivative record carries no reference geometry; providers "
            "must record the displaced references' cell and positions"
        )
    _check_cell(bundle.cell, geom.cell)
    _check_positions(bundle, derivatives, geom)
    # sparse pattern: exact coverage, no union
    bundle_keys = set(bundle.sparse)
    record_keys = set(derivatives.dH)
    if record_keys - bundle_keys:
        extra = sorted(record_keys - bundle_keys)[:5]
        raise SparsePatternMismatch(
            "the displaced data carry sparse keys outside the bundle's H/S "
            f"pattern (e.g. {extra}); refusing the union of mismatched "
            "patterns"
        )
    missing = bundle_keys - record_keys
    if missing:
        example = sorted(missing)[:5]
        raise SparsePatternMismatch(
            "the derivative record does not cover the bundle's sparse "
            f"pattern ({len(missing)} keys missing, e.g. {example}); "
            "displacement derivatives must span the same pattern as H/S"
        )
    if set(derivatives.dS) != bundle_keys:
        raise SparsePatternMismatch(
            "dS keys differ from dH keys; the overlap derivative must span "
            "the same pattern"
        )
    restricted = replace(
        derivatives,
        dH={key: derivatives.dH[key] for key in bundle.sparse},
        dS={key: derivatives.dS[key] for key in bundle.sparse},
    )
    provenance = dict(bundle.metadata)
    provenance["displacement"] = {
        "producer_recipe": restricted.producer_recipe,
        "atom_index": pattern.atom_index,
        "axis": pattern.axis,
        "branch": restricted.branch,
        "step_bohr": restricted.step_bohr,
        "gauge": restricted.gauge,
        "units": dict(restricted.units),
        "reference_digests": tuple(restricted.reference_digests),
        "reference_fermi_ev": restricted.reference_fermi_ev,
        "reference_geometry": restricted.reference_geometry,
    }
    return SpiralBundleWithDisplacements(
        bundle=bundle,
        derivatives=restricted,
        pattern=pattern,
        provenance=provenance,
    )


def _check_cell(bundle_cell, record_cell):
    a = np.asarray(bundle_cell, dtype=float)
    b = np.asarray(record_cell, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=GEOMETRY_TOL_BOHR, rtol=0):
        raise GeometryMismatch(
            "reference cell differs from the bundle cell beyond "
            f"tolerance {GEOMETRY_TOL_BOHR} Bohr; the displaced references "
            "must be SCFs of the same reference cell"
        )


def _check_positions(bundle, derivatives, geom):
    pattern = derivatives.pattern
    rec = np.asarray(geom.orbital_positions, dtype=float)
    ref = np.asarray(bundle.orbital_positions, dtype=float)
    if rec.shape != ref.shape:
        raise GeometryMismatch(
            f"reference geometry has {rec.shape} orbital positions but the "
            f"bundle has {ref.shape}; the displaced references must be of "
            "the same basis"
        )
    delta = rec - ref
    tol = GEOMETRY_TOL_BOHR
    displaced = set(geom.displaced_orbital_indices)
    for i in range(len(ref)):
        d = delta[i]
        along = float(np.dot(d, pattern.direction))
        if i in displaced:
            # SIGNED: the record's geometry is the +h side of a central
            # pair, anchored to the reference (bundle) geometry
            if abs(along - pattern.step_bohr) > 1e-4:
                raise GeometryMismatch(
                    f"displaced atom's orbital {i} signed offset along "
                    f"{pattern.axis} is {along:+.6f} Bohr; the recorded "
                    f"geometry must be the +h side (+{pattern.step_bohr} "
                    "Bohr) of the central pair about the reference"
                )
            if np.abs(d - along * np.asarray(pattern.direction)).max() > 1e-4:
                raise GeometryMismatch(
                    f"displaced atom's orbital {i} moves off the declared "
                    f"{pattern.axis} axis by more than 1e-4 Bohr"
                )
        elif abs(along) > tol or np.abs(d).max() > tol:
            raise GeometryMismatch(
                f"orbital {i} outside the displaced pattern moves by "
                f"{np.abs(d).max():.3e} Bohr; only atom "
                f"{pattern.atom_index} may be displaced"
            )
