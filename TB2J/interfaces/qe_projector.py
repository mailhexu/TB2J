"""Reader for the Quantum ESPRESSO projector-Green exporter dumps.

Parses the binary dumps written by ``PW/src/becp_dump.f90`` (QE fork branch
``TB2J``) and normalizes them into
:class:`TB2J.projector_green.ProjectorGreenData` for the projector-space
exchange workflow.

Three dump families use gfortran sequential unformatted records with 4-byte
little-endian markers and Fortran/column-major arrays (predecessor schemas
in ``TB2J/qe_patch/README.md``):

KB/beta dumps (``TB2J_PROJECTORS=kb``, magic ``TB2JQEDUMPV1`` / ``V1.1`` /
``V1.2`` / ``V1.3``)
    v1.1 adds the packed-triangular ``becsum`` record (and, for PAW runs,
    ``rho%bec``) after the per-k coefficient stream; v1.0 ends after the
    per-k records; v1.2 adds the projected-xc vertex records.

Atomic-pseudo-wavefunction dumps (``TB2J_PROJECTORS=atomic``, magic
``TB2JQEATWFC1.0`` / ``TB2JQEATWFC1.1``)
    UPF ``PP_PSWFC`` pseudo-atomic orbitals from QE ``atomic_wfc`` in the
    same k-dependent plane-wave basis as ``evc``; collinear ordering atoms,
    radial UPF orbital ``nb``, QE real spherical-harmonic ordinal
    ``m = 1..2l+1``.  Distinct physical-record schema: header, Fermi, cell,
    species, ions, channel metadata, k mesh, spectral bands, then per k a
    coefficient record ``C(nproj, nbnd) = <atomic_wfc|psi>`` followed by a
    full Gram record ``M(nproj, nproj) = <atomic_wfc|atomic_wfc>``, and two
    trailing BZ-weighted R=0 onsite covariant spin-vertex records
    ``cov_xc`` and ``cov_aug``.
    There is deliberately no ``deeq``/``qq_at``/``becsum`` physical field in
    this format.

Non-collinear spinor KB dumps (``TB2JQEDUMPNC1.0`` / ``NC1.1`` / ``NC1.2``)
    Native ``becp%nc(nkb, 2, nbnd)`` per k in QE's dual beta basis;
    native ``(uu, ud, du, dd)`` total and magnetic ``deeq_nc`` blocks,
    Hermitian projected smooth fields, beta Gram and band occupations.
    NC1.1 adds nine source-resolved magnetic response blocks, including
    rotated PAW one-center sources, to construct explicit frozen-source
    tangents without rotating the fixed ionic/SOC Hamiltonian. The beta
    Gram conditions only the smooth-field projection, never the dual Green
    coefficients or native nonlocal response. NC1.0 SO-bearing dumps are
    rejected because they have no source-resolved tangents.

KB1.3, atomic1.1 and NC1.2 append the strictly validated
``TB2JHUBBARD1.0`` native extension. Raw ``.hubbard`` records distinguish
active, explicitly absent and legacy unknown state. Legacy normalization
requires ``legacy_no_u=True`` (user attestation, not measured absence).
Explicit new-format absence needs no assertion. Active records are retained
but normalization refuses until the joint XC+U source is constructed; the
legacy assertion never bypasses a declared or failed source gate.

Pinned physics contract (research memo ``2026-10-08-qe-export-surface-and-
operators``, SymPy pin ``docs/sympy/qe_separable_beta_trace.py`` and the
atomic-PAO pairing pin):

* KB ``becp`` coefficients ``P_ni = <beta_n|psi_i>`` are already dual, so
  ``overlap_k`` stays ``None`` — the ``qq_at`` and beta-Gram records are
  diagnostics only and are never used as a channel metric; the ``deeq``
  spin vertex ``hij = deeq(up) - deeq(down)`` per atom block matches the
  undressed dual Green matrix (v1.2 adds the primal ``dbeta_xc`` vertex,
  jointly metric-transformed with the beta Gram);
* atomic coefficients ``C = <phi|psi>`` are PRIMAL, the per-k full orbital
  overlap ``M(k)`` is exported and normalized as ``overlap_k`` (shared by
  spins), and the shared projector runtime dresses with the k-dependent
  ``M^-1`` (CLI ``--overlap_mode`` / ``--overlap_rcond`` allowed).  The
  atomic spin vertex is the covariant site-local
  ``Delta = sum_k w_k [<phi(k)|V_xc^up - V_xc^dn|phi(k)>``
  ``+ B(k) (deeq_up - deeq_dn) B(k)^dagger] / sum_k w_k``,
  with ``B(k) = <phi(k)|beta(k)>`` and the sum over sampled
  up-spin k points (``isk == 1``): the BZ-weighted R=0 onsite
  covariant matrix, not a first-k matrix. It is exported as
  ``delta_total`` in eV and ``hij[up] = +delta_total/2``,
  ``hij[dn] = -delta_total/2``.  The preconjugated ``M^-1 Delta M^-1``
  pairing is never applied at the reader (the runtime owns the metric).
* energies (``deeq``, ``dvan``, ``et``, ``ef*``, ``cov_*``) are in Ry and
  converted to eV with ``RYTOEV = 13.605693122994``; coefficients are
  dimensionless.

Public API
----------
``parse_qe_dump(path)``
    Raw parse into a :class:`QEProjectorDump` (all records + metadata, in
    dumped units).
``parse_qe_atomic_dump(path)``
    Raw parse into a :class:`QEAtomicDump`.
``read_qe_dump(path, *, legacy_no_u=False)``
    Normalized KB :class:`~TB2J.projector_green.ProjectorGreenData`
    (eV / Angstrom / fractional k-points, spin-degenerate dense mesh).
``read_qe_atomic_dump(path, *, legacy_no_u=False)``
    Normalized atomic :class:`~TB2J.projector_green.ProjectorGreenData`
    with primal coefficients and full k-dependent ``overlap_k = M(k)``.
``parse_qe_spinor_dump(path)`` / ``read_qe_spinor_dump(path, *, legacy_no_u=False)``
    Strict non-collinear spinor parse (NC1.0 / NC1.1 / NC1.2) and normalized
    :class:`~TB2J.projector_green.ProjectorGreenData` with dual spinor
    coefficients and the magnetic-only ``spinor_operator``. NC1.1 adds the
    nine native source-response blocks and explicit frozen-source tangent
    vertices ``spinor_tangent_vertices`` (eV); a single dump yields one
    transverse leg via ``compute_spinor_projector_exchange``, while
    ``gen_exchange_qe({'x': ..., 'y': ..., 'z': ...})`` is required
    for a standard full-tensor ``exchange.out``.
"""

from __future__ import annotations

import os
import re
import struct
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from TB2J.projector_green import (
    PAULI_MATRICES,
    SPINOR_OPERATOR_DEFINITION,
    SPINOR_TANGENT_DEFINITION,
    ProjectorGreenData,
    magnetic_tangent_vertices,
    spinor_dense_block,
)

RYTOEV = 13.605693122994

#: 16-byte magic strings; Fortran pads ``CHARACTER(LEN=16)`` with blanks.
_MAGIC_VERSIONS = {
    b"TB2JQEDUMPV1.3": "1.3",
    b"TB2JQEDUMPV1.2": "1.2",
    b"TB2JQEDUMPV1.1": "1.1",
    b"TB2JQEDUMPV1": "1.0",
}

#: Atomic-pseudo-wavefunction dump family (``TB2J_PROJECTORS=atomic``).
ATOMIC_MAGIC = "TB2JQEATWFC1.0"
MAGIC_ATOMIC = b"TB2JQEATWFC1.0"
_MAGIC_ATOMIC = MAGIC_ATOMIC
MAGIC_ATOMIC_HUBBARD = b"TB2JQEATWFC1.1"
_ATOMIC_MAGICS = {MAGIC_ATOMIC: "1.0", MAGIC_ATOMIC_HUBBARD: "1.1"}

SPINOR_MAGIC = "TB2JQEDUMPNC1.0"
MAGIC_SPINOR = SPINOR_MAGIC.encode("ascii")
#: NC1.1 adds record ``13+nks``: the nine native source-response blocks
#: ``D_response(nhm, nhm, nat, 4, 3, 3)`` (ih, jh, na, spinblock, c, j).
SPINOR_MAGIC_NC11 = "TB2JQEDUMPNC1.1"
MAGIC_SPINOR_NC11 = SPINOR_MAGIC_NC11.encode("ascii")
MAGIC_SPINOR_NC12 = b"TB2JQEDUMPNC1.2"
_SPINOR_MAGICS = {
    MAGIC_SPINOR: "1.0",
    MAGIC_SPINOR_NC11: "1.1",
    MAGIC_SPINOR_NC12: "1.2",
}

HIJ_DEFINITION = "qe_deeq_spin_difference"
HIJ_UNITS = "eV"
HIJ_SOURCE = "QE becp_dump deeq record (PW/src/becp_dump.f90)"
HIJ_PROJECTION = "QE ultrasoft/PAW beta channel (dual basis)"
COEFFICIENT_SOURCE = "qe_becp"
COEFFICIENT_PROJECTOR = "qe_beta"
CHANNEL_INTERPRETATION = "qe_dual_to_beta"
OPERATOR_BASIS = "qe_dual_beta_channel"

#: Atomic-dump projector source/mode and site-vertex definition (registered
#: in ``TB2J/projector_green.py``).
ATOMIC_COEFFICIENT_SOURCE = "qe_atomic_wfc"
ATOMIC_COEFFICIENT_PROJECTOR = "qe_upf_pswfc"
ATOMIC_CHANNEL_INTERPRETATION = "qe_primal_pseudo_atomic"
ATOMIC_OPERATOR_BASIS = "qe_pseudo_atomic_orbital_site"
ATOMIC_HIJ_DEFINITION = "qe_atomic_pao_projected_spin_vertex"
#: Real-harmonic ordinal note shipped in the atomic metadata and .out header.
ATOMIC_M_ORDINAL_NOTE = (
    "projector_m is the QE real spherical-harmonic ordinal 1..2l+1 used by "
    "atomic_wfc (atomic_wfc_mod.f90 real-harmonic convention), not an "
    "m=-l..l label"
)
ATOMIC_VERTEX_NOTE = (
    "BZ-weighted R=0 onsite covariant site vertex Delta = "
    "sum_{k:isk==1} wk [<phi(k)|Vxc_up - Vxc_dn|phi(k)> + "
    "B(k) (deeq_up - deeq_dn) B(k)^dagger] / sum_{k:isk==1} wk, "
    "B(k) = <phi(k)|beta(k)>; primal, dressed by the shared runtime "
    "with the k-dependent M(k)^-1"
)

#: Tolerance (cartesian, 2pi/alat units) for matching the up/down k meshes.
_KPOINT_MATCH_TOL = 1.0e-6

_ITEMSIZE = {"<f8": 8, "<i4": 4, "<c16": 16, "S3": 3, "S16": 16}


# ---------------------------------------------------------------------------
# Low-level sequential-record reader
# ---------------------------------------------------------------------------
class _RecordReader:
    """Sequential unformatted-Fortran reader (4-byte LE record markers)."""

    def __init__(self, path: Path, fh):
        self.path = path
        self.fh = fh
        self.pos = 0

    def _unpack_marker(self, raw: bytes, what: str) -> int:
        if len(raw) < 4:
            raise ValueError(
                f"QE dump {self.path}: truncated file at {what} (offset "
                f"{self.pos}): missing 4-byte record marker"
            )
        return struct.unpack("<i", raw[:4])[0]

    def read_raw(self, what: str) -> bytes:
        head = self.fh.read(4)
        size = self._unpack_marker(head, what)
        if size < 0 or size > self.remaining_bytes() - 4:
            raise ValueError(
                f"QE dump {self.path}: invalid or truncated {what} record "
                f"length {size}; expected a complete 4-byte-marker record"
            )
        payload = self.fh.read(size)
        if len(payload) < size:
            raise ValueError(
                f"QE dump {self.path}: truncated {what} record (offset "
                f"{self.pos}): expected {size} payload bytes, got {len(payload)}"
            )
        tail = self.fh.read(4)
        if len(tail) < 4:
            raise ValueError(
                f"QE dump {self.path}: truncated {what} record (offset "
                f"{self.pos}): missing closing record marker"
            )
        closing = struct.unpack("<i", tail)[0]
        if closing != size:
            raise ValueError(
                f"QE dump {self.path}: record-length mismatch at {what} "
                f"(offset {self.pos}): opening marker {size} != closing "
                f"marker {closing}"
            )
        self.pos += 8 + size
        return payload

    def at_eof(self) -> bool:
        pos = self.fh.tell()
        ended = self.fh.read(1) == b""
        self.fh.seek(pos)
        return ended

    def remaining_bytes(self) -> int:
        pos = self.fh.tell()
        self.fh.seek(0, 2)
        end = self.fh.tell()
        self.fh.seek(pos)
        return end - pos

    def split_record(self, what: str, spec):
        """Read one record and split it into consecutive typed chunks.

        ``spec`` is a sequence of ``(name, dtype, count)``; counts may be
        ``None`` to take all remaining bytes of the record.
        """
        payload = self.read_raw(what)
        chunks = {}
        offset = 0
        for name, dtype, count in spec:
            itemsize = _ITEMSIZE[dtype]
            if count is None:
                count = (len(payload) - offset) // itemsize
            nbytes = count * itemsize
            if offset + nbytes > len(payload):
                raise ValueError(
                    f"QE dump {self.path}: {what} record too short "
                    f"({len(payload)} bytes) for {name}: need "
                    f"{count} x {dtype} at offset {offset}"
                )
            chunks[name] = np.frombuffer(
                payload, dtype=dtype, count=count, offset=offset
            )
            offset += nbytes
        if offset != len(payload):
            raise ValueError(
                f"QE dump {self.path}: {what} record has {len(payload) - offset} "
                f"trailing bytes beyond the declared layout"
            )
        return chunks

    def array(self, what: str, dtype: str, shape):
        """Read one record as a Fortran-order array of the given shape."""
        payload = self.read_raw(what)
        expected = int(np.prod(shape)) * _ITEMSIZE[dtype]
        if len(payload) != expected:
            raise ValueError(
                f"QE dump {self.path}: {what} record is {len(payload)} bytes, "
                f"expected {expected} for shape {shape} x {dtype}"
            )
        flat = np.frombuffer(payload, dtype=dtype)
        if flat.size != int(np.prod(shape)):
            raise ValueError(
                f"QE dump {self.path}: {what} record has {flat.size} values, "
                f"expected {int(np.prod(shape))} for shape {shape}"
            )
        return np.array(flat.reshape(shape, order="F"))


# MathDoc: tb2j/qe-hubbard-native-contract
class QEHubbardError(ValueError):
    """A refused native Hubbard contract, with a stable semantic category."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f"QE Hubbard {code}: {message}")


@dataclass
class QEHubbardSpecies:
    species: int
    label: str
    filename: str
    path: str
    family: int
    flags: int
    md5: str
    digest_origin: str


@dataclass
class QEHubbardParameters:
    species: int
    ev: np.ndarray
    ry: np.ndarray
    excluded_norm: np.ndarray
    quantum_flags: tuple


@dataclass
class QEHubbardChannel:
    channel: int
    atom: int
    species: int
    kind: int
    principal_n: int
    l: int
    dimension: int
    native_offset: int
    matrix_offset: int
    acted: int
    reserved: int


@dataclass
class QEHubbardState:
    channel: int
    atom: int
    species: int
    dimension: int
    occupation: np.ndarray
    potential: np.ndarray
    scalar: np.ndarray
    magnetic: np.ndarray
    # (channel, first matrix row, width); background aggregates keep cross blocks.
    parts: tuple


@dataclass
class QEHubbardKPoint:
    index: int
    npw: int
    spin: int
    xk: np.ndarray
    weight: float
    global_g_index: np.ndarray
    wg: np.ndarray
    et: np.ndarray
    coefficients: np.ndarray


@dataclass
class QEHubbardAction:
    index: int
    algorithm: int
    seed: int
    metrics: np.ndarray
    projection: np.ndarray
    wfcu: np.ndarray
    psi: np.ndarray
    native: np.ndarray
    exported: np.ndarray


@dataclass
class QEHubbardCheck:
    channel: int
    atom: int
    species: int
    kind: int
    absolute_error: float
    relative_error: float
    native: np.ndarray
    reconstructed: np.ndarray


@dataclass
class QEHubbardFrame:
    atom: int
    species: int
    nh: int
    origin: int
    xc_operator_max_eV: float
    xc_trace_vector_eV: np.ndarray
    xc_trace_norm_eV: float
    direction: np.ndarray
    map_id: int
    occupation_map_vector: np.ndarray
    occupation_map_norm: float
    potential_trace_vector_Ry: np.ndarray
    u_map_residual: float
    gram_rank: int
    discarded_rank: int
    gram_gap_ratio: float
    rcond: float
    clean_gap: float
    frame_unit_residual: float


@dataclass
class QEHubbardRotation:
    atom: int
    species: int
    channel: int
    dimension: int
    frame_origin: int
    reference_direction: np.ndarray
    axes: np.ndarray
    angles: np.ndarray
    metrics: dict
    tangent_directions: np.ndarray
    tangent_axes: np.ndarray
    steps: np.ndarray
    analytic: np.ndarray
    central_difference: np.ndarray
    tangent_absolute_error: np.ndarray
    tangent_relative_error: np.ndarray
    step_convergence: np.ndarray


@dataclass
class QEHubbardData:
    """Complete raw native extension; legacy absence of evidence stays unknown.

    All matrix/action records remain in native units. ``complete`` denotes this
    source contract, not an exchange-family or material-convergence certificate.
    """

    state: str = "unknown"
    complete: bool = False
    envelope: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    species: list[QEHubbardSpecies] = field(default_factory=list)
    parameters: list[QEHubbardParameters] = field(default_factory=list)
    channels: list[QEHubbardChannel] = field(default_factory=list)
    columns: np.ndarray = field(default_factory=lambda: np.empty((0, 8), dtype=int))
    spin_columns: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=int)
    )
    layout: dict[str, int] = field(default_factory=dict)
    unit_conversion: float = 0.0
    unit_bridge_residual: float = 0.0
    states: list[QEHubbardState] = field(default_factory=list)
    kpoints: list[QEHubbardKPoint] = field(default_factory=list)
    actions: list[QEHubbardAction] = field(default_factory=list)
    occupation_checks: list[QEHubbardCheck] = field(default_factory=list)
    potential_checks: list[QEHubbardCheck] = field(default_factory=list)
    frames: list[QEHubbardFrame] = field(default_factory=list)
    rotations: list[QEHubbardRotation] = field(default_factory=list)
    restoration: dict[str, float | int] = field(default_factory=dict)
    validation: dict[str, float | int] = field(default_factory=dict)


def _hub_require(condition, code, message):
    if not condition:
        raise QEHubbardError(code, message)


def _hub_close(actual, expected, message, *, atol=1e-12, rtol=1e-10, code="validation"):
    _hub_require(np.allclose(actual, expected, atol=atol, rtol=rtol), code, message)


def _hub_residual(first, second):
    absolute = float(np.linalg.norm(first - second))
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
    return absolute, absolute / scale if scale > np.finfo(float).tiny else 0.0


def _hub_hermitian(matrix, name):
    _hub_close(matrix, matrix.conj().T, name, code="hermiticity")


class _HubbardRecord:
    """Bounds-checked cursor inside one existing sequential Fortran record."""

    def __init__(self, reader, record_id):
        try:
            self.payload = reader.read_raw(f"Hubbard record {record_id}")
        except ValueError as exc:
            raise QEHubbardError("layout", str(exc)) from exc
        self.offset = 0
        _hub_require(
            self.ints(1) == (record_id,), "layout", f"expected record {record_id}"
        )

    def array(self, dtype, shape):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        _hub_require(
            all(n >= 0 for n in shape), "dimension", "negative array dimension"
        )
        count = 1
        for n in shape:
            count *= n
        size = count * np.dtype(dtype).itemsize
        _hub_require(
            self.offset + size <= len(self.payload),
            "layout",
            f"short record: array {shape} of {dtype} at byte {self.offset}",
        )
        result = np.frombuffer(
            self.payload, dtype=dtype, count=count, offset=self.offset
        )
        self.offset += size
        _hub_require(np.isfinite(result).all(), "nonfinite", "non-finite record values")
        return result.reshape(shape, order="F")

    def ints(self, n):
        return tuple(map(int, self.array("<i4", n)))

    def reals(self, shape):
        return self.array("<f8", shape)

    def text(self, n):
        _hub_require(self.offset + n <= len(self.payload), "layout", "short text field")
        raw = self.payload[self.offset : self.offset + n]
        self.offset += n
        try:
            return raw.decode("ascii").rstrip(" \x00")
        except UnicodeDecodeError as exc:
            raise QEHubbardError("provenance", "non-ASCII native identity") from exc

    def end(self):
        _hub_require(
            self.offset == len(self.payload), "layout", "unexpected record bytes"
        )


_HUBBARD_ENVELOPE_FIELDS = (
    "major",
    "minor",
    "endian",
    "active",
    "native_enabled",
    "complete",
    "functional",
    "dc",
    "projector",
    "collinear",
    "source_mode",
    "nchannel",
    "ncol",
    "nat",
    "ntyp",
    "nspin",
    "nks",
    "nbnd",
    "nkb",
    "source_status",
    "unit_ry",
    "unit_ev",
    "integer_bits",
    "real_bits",
    "complex_bits",
)
_HUBBARD_NATIVE_REVISION = "301742e16546b927cfe35fb2fcf4d9338f3e294a"


def _read_hubbard_header(reader, context, profile):
    c = _HubbardRecord(reader, 1)
    magic = c.text(16)
    env = dict(zip(_HUBBARD_ENVELOPE_FIELDS, c.ints(25), strict=True))
    c.end()
    _hub_require(
        magic == "TB2JHUBBARD1.0"
        and (env["major"], env["minor"], env["endian"]) == (1, 0, 16909060)
        and (env["integer_bits"], env["real_bits"], env["complex_bits"])
        == (32, 64, 128),
        "unsupported",
        "unknown native schema, endian or scalar widths",
    )
    _hub_require((env["unit_ry"], env["unit_ev"]) == (1, 2), "unsupported", "unit IDs")
    _hub_require(
        all(env[k] in (0, 1) for k in ("active", "native_enabled", "complete")),
        "dimension",
        "invalid envelope flags",
    )
    _hub_require(env["complete"] == 1, "validation", "incomplete envelope")
    for key in ("nat", "ntyp", "nspin", "nks", "nbnd", "nkb"):
        expected = context.get("nsp" if key == "ntyp" else key)
        if expected is not None:
            _hub_require(
                env[key] == expected, "dimension", f"predecessor {key} mismatch"
            )
    spinor = profile == "spinor"
    _hub_require(
        env["collinear"] == int(not spinor)
        and env["source_mode"] == (2 if profile == "atomic" else 1)
        and env["nspin"] in ((4,) if spinor else (1, 2)),
        "unsupported",
        "predecessor/source/spin convention mismatch",
    )
    _hub_require(env["nkb"] >= 0, "dimension", "negative native beta count")
    if env["active"]:
        _hub_require(
            env["native_enabled"]
            == env["functional"]
            == env["dc"]
            == env["source_status"]
            == 1
            and env["projector"] in (1, 2),
            "unsupported",
            "active source is not native scalar Dudarev atomic/ortho-atomic",
        )
        _hub_require(
            0 < env["nchannel"] <= env["nat"] * (1 if spinor else 3)
            and 0 < env["ncol"] <= 7 * env["nat"] * (2 if spinor else 3),
            "dimension",
            "invalid acted shell/column counts",
        )
    else:
        _hub_require(
            all(
                env[k] == 0
                for k in (
                    "functional",
                    "dc",
                    "projector",
                    "nchannel",
                    "ncol",
                    "source_status",
                )
            ),
            "source",
            "absent-U envelope contains active source metadata",
        )
    result = QEHubbardData(state="active" if env["active"] else "absent", envelope=env)
    c = _HubbardRecord(reader, 2)
    names = (
        "qe_version",
        "native_revision",
        "source_name",
        "projector_name",
        "construction",
        "metric",
        "input_path",
        "input_md5",
        "input_digest_origin",
        "pseudo_dir",
        "pseudo_dir_cur",
    )
    sizes = (80, 64, 16, 30, 64, 80, 256, 32, 48, 256, 256)
    result.provenance = {
        name: c.text(size) for name, size in zip(names, sizes, strict=True)
    }
    c.end()
    p = result.provenance
    _hub_require(
        p["native_revision"] == _HUBBARD_NATIVE_REVISION
        and p["source_name"] == ("atomic" if profile == "atomic" else "kb")
        and p["projector_name"]
        == {0: "inactive", 1: "atomic", 2: "ortho-atomic"}[env["projector"]]
        and all(
            p[k]
            for k in ("qe_version", "construction", "metric", "input_digest_origin")
        ),
        "provenance",
        "missing, unaudited or contradictory native construction identity",
    )
    _hub_require(
        not p["input_md5"] or re.fullmatch(r"[0-9a-fA-F]{32}", p["input_md5"]),
        "provenance",
        "invalid input digest",
    )
    for species in range(1, env["ntyp"] + 1):
        c = _HubbardRecord(reader, 3)
        (nt,) = c.ints(1)
        label, filename, path = c.text(80), c.text(256), c.text(512)
        family, flags = c.ints(2)
        md5, origin = c.text(32), c.text(56)
        c.end()
        _hub_require(nt == species, "mapping", "unordered species identities")
        _hub_require(
            label == context["atm"][species - 1],
            "provenance",
            "Hubbard species label differs from predecessor species identity",
        )
        _hub_require(
            family in (1, 2, 3) and flags >= 0 and flags & ~3 == 0,
            "unsupported",
            "unknown pseudopotential family or flags",
        )
        _hub_require(
            label
            and filename
            and path
            and origin
            and Path(path).name == Path(filename).name
            and (re.fullmatch(r"[0-9a-fA-F]{32}", md5) or md5 == "NOT SET"),
            "provenance",
            "incomplete loaded pseudopotential identity",
        )
        if "tvanp" in context:
            expected = (
                3 if context["tpawp"][nt - 1] else 2 if context["tvanp"][nt - 1] else 1
            )
            _hub_require(
                family == expected, "provenance", "predecessor UPF family mismatch"
            )
        if "has_so" in context:
            _hub_require(
                bool(flags & 2) == bool(context["has_so"][nt - 1]),
                "provenance",
                "predecessor spin-orbit flag mismatch",
            )
        result.species.append(
            QEHubbardSpecies(nt, label, filename, path, family, flags, md5, origin)
        )
    return result


def _read_hubbard_maps(reader, data, context, spinor):
    env = data.envelope
    if not env["active"]:
        c = _HubbardRecord(reader, 13)
        _hub_require(c.ints(1) == (0,), "source", "contradictory absent-U declaration")
        c.end()
        return
    c = _HubbardRecord(reader, 4)
    _hub_require(
        c.ints(3) == (env["ntyp"], 2, 1), "dimension", "parameter dimensions/units"
    )
    data.unit_conversion, data.unit_bridge_residual = map(float, c.reals(2))
    _hub_require(
        c.ints(3) == (1, 1, 0), "unsupported", "functional/double-counting branch"
    )
    c.end()
    _hub_close(
        data.unit_conversion,
        RYTOEV,
        "unrecognized native Ry/eV bridge",
        rtol=64 * np.finfo(float).eps,
        atol=0,
        code="unsupported",
    )
    _hub_require(data.unit_bridge_residual >= 0, "validation", "negative unit residual")
    for nt in range(1, env["ntyp"] + 1):
        c = _HubbardRecord(reader, 4)
        _hub_require(c.ints(1) == (nt,), "mapping", "parameter species order")
        ev, ry, excluded = c.reals(9), c.reals(9), c.reals(8)
        flags = c.ints(9)
        c.end()
        _hub_require(
            not np.any(excluded) and not np.any(ev[2:]) and not np.any(ry[2:]),
            "unsupported",
            "non-Dudarev active parameter inventory",
        )
        _hub_require(
            all(f in (0, 1) for f in flags[6:])
            and (not flags[8] or flags[7])
            and (flags[6] or ry[0] == 0)
            and (flags[7] or ry[1] == 0),
            "source",
            "parameter/shell activation inventory mismatch",
        )
        if spinor:
            _hub_require(
                ry[1] == 0 and flags[7:] == (0, 0) and (not flags[6] or ry[0] > 0),
                "unsupported",
                "spinor background or nonpositive main U",
            )
        _hub_close(
            ev,
            ry * data.unit_conversion,
            "inconsistent eV/Ry parameters",
            atol=32 * np.finfo(float).eps,
            rtol=32 * np.finfo(float).eps,
        )
        data.parameters.append(QEHubbardParameters(nt, ev, ry, excluded, flags))
    bridge_budget = (
        32
        * np.finfo(float).eps
        * max(1.0, *(float(np.abs(p.ev).max()) for p in data.parameters))
    )
    _hub_require(
        data.unit_bridge_residual <= bridge_budget, "validation", "unit conversion gate"
    )
    _hub_require(
        any(np.any(p.ry[:2]) for p in data.parameters),
        "source",
        "active envelope has no acting U",
    )
    expected_channels = []
    for atom, nt in enumerate(context["ityp"], 1):
        flags = data.parameters[nt - 1].quantum_flags
        if flags[6]:
            expected_channels.append((atom, nt, 1, flags[0], flags[1], 0))
        if flags[7]:
            expected_channels.append((atom, nt, 2, flags[2], flags[3], 0))
            if flags[8]:
                expected_channels.append(
                    (atom, nt, 3, flags[4], flags[5], 2 * flags[3] + 1)
                )
    _hub_require(
        len(expected_channels) == env["nchannel"],
        "source",
        "incomplete shell inventory",
    )
    c = _HubbardRecord(reader, 5)
    _hub_require(
        c.ints(2) == (env["nchannel"], env["ncol"]), "dimension", "map dimensions"
    )
    c.end()
    for ch, expected in enumerate(expected_channels, 1):
        c = _HubbardRecord(reader, 6)
        row = c.ints(11)
        c.end()
        channel = QEHubbardChannel(*row)
        atom, nt, kind, principal, ell, moff = expected
        _hub_require(
            row[0] == ch
            and (
                channel.atom,
                channel.species,
                channel.kind,
                channel.principal_n,
                channel.l,
                channel.matrix_offset,
            )
            == (atom, nt, kind, principal, ell, moff)
            and 0 <= ell <= 3
            and principal > ell
            and channel.dimension == 2 * ell + 1
            and channel.native_offset >= 0
            and (channel.acted, channel.reserved) == (1, 0),
            "mapping",
            "invalid shell ownership, dimensions or offsets",
        )
        data.channels.append(channel)
    _hub_require(
        sum(c.dimension for c in data.channels) * (2 if spinor else 1) == env["ncol"],
        "mapping",
        "acted shell columns do not cover native source",
    )
    columns = []
    for export in range(1, env["ncol"] + 1):
        c = _HubbardRecord(reader, 7)
        row = c.ints(8)
        c.end()
        native, exported, ch, atom, nt, kind, ell, orbital = row
        _hub_require(1 <= ch <= len(data.channels), "mapping", "unknown column shell")
        owner = data.channels[ch - 1]
        _hub_require(
            exported == export
            and native > 0
            and 1 <= orbital <= owner.dimension
            and (atom, nt, kind, ell)
            == (owner.atom, owner.species, owner.kind, owner.l),
            "mapping",
            "non-reversible column ownership/ordinal",
        )
        if not spinor:
            _hub_require(
                native == export == owner.native_offset + orbital,
                "mapping",
                "collinear native-column offset",
            )
        columns.append(row)
    data.columns = np.asarray(columns, dtype=int)
    _hub_require(
        np.all(np.diff(data.columns[:, 0]) > 0),
        "mapping",
        "duplicate or unordered native columns",
    )


def _read_hubbard_spin_layout(reader, data):
    c = _HubbardRecord(reader, 15)
    npol, norbital, ncol, order, blocks, components, native_ncol = c.ints(7)
    c.end()
    _hub_require(
        (npol, ncol, order, blocks, components) == (2, data.envelope["ncol"], 1, 4, 2)
        and 2 * norbital == ncol
        and native_ncol >= ncol
        and (data.state == "active" or native_ncol == 0),
        "dimension",
        "invalid spin-major/orbital-inner layout",
    )
    data.layout = dict(
        npol=npol,
        norbital=norbital,
        ncolumn=ncol,
        column_order=order,
        potential_blocks=blocks,
        pw_spin_components=components,
        native_ncolumn=native_ncol,
    )
    rows = []
    for column in data.columns:
        c = _HubbardRecord(reader, 16)
        native, ch, orbital, spin = c.ints(4)
        c.end()
        owner = data.channels[column[2] - 1]
        _hub_require(
            (native, ch, orbital) == (column[0], column[2], column[7])
            and spin in (1, 2)
            and native <= native_ncol
            and native == owner.native_offset + (spin - 1) * owner.dimension + orbital,
            "mapping",
            "native column/orbital/spin map is inconsistent",
        )
        rows.append((native, ch, orbital, spin))
    data.spin_columns = np.asarray(rows, dtype=int).reshape(-1, 4)
    for channel in data.channels:
        actual = {(r[2], r[3]) for r in rows if r[1] == channel.channel}
        _hub_require(
            actual
            == {
                (orbital, spin)
                for orbital in range(1, channel.dimension + 1)
                for spin in (1, 2)
            },
            "mapping",
            "incomplete spin-major shell",
        )


def _hub_dense(blocks):
    dim = blocks.shape[0]
    return (
        blocks.reshape(dim, dim, 2, 2).transpose(2, 0, 3, 1).reshape(2 * dim, 2 * dim)
    )


def _hub_blocks(dense):
    dim = dense.shape[0] // 2
    return dense.reshape(2, dim, 2, dim).transpose(1, 3, 0, 2).reshape(dim, dim, 4)


def _hub_pauli(blocks):
    uu, ud, du, dd = np.moveaxis(blocks, -1, 0)
    return (uu + dd) / 2, np.stack(
        ((ud + du) / 2, (du - ud) / (2j), (uu - dd) / 2), axis=-1
    )


def _hub_magnetic_blocks(magnetic):
    x, y, z = np.moveaxis(magnetic, -1, 0)
    return np.stack((z, x - 1j * y, x + 1j * y, -z), axis=-1)


def _hub_native_potential(occupation, u):
    dense = _hub_dense(occupation)
    return _hub_blocks(u * (0.5 * np.eye(len(dense)) - dense.T))


def _read_hubbard_states(reader, data, spinor):
    for channel in data.channels:
        if channel.kind == 3:
            continue
        dim = channel.dimension
        parts = [(channel.channel, 0, dim)]
        if spinor:
            c = _HubbardRecord(reader, 17)
            _hub_require(
                c.ints(6)
                == (channel.channel, channel.atom, channel.species, dim, 2, 4),
                "dimension",
                "complex source dimensions/ownership",
            )
        elif channel.kind == 1:
            c = _HubbardRecord(reader, 8)
            _hub_require(
                c.ints(5)
                == (
                    channel.channel,
                    channel.atom,
                    channel.species,
                    dim,
                    data.envelope["nspin"],
                ),
                "dimension",
                "main source dimensions/ownership",
            )
        else:
            second = next(
                (
                    ch
                    for ch in data.channels
                    if ch.atom == channel.atom and ch.kind == 3
                ),
                None,
            )
            d2 = second.dimension if second else 0
            c = _HubbardRecord(reader, 9)
            _hub_require(
                c.ints(8)
                == (
                    channel.channel,
                    second.channel if second else 0,
                    channel.atom,
                    channel.species,
                    dim + d2,
                    dim,
                    d2,
                    data.envelope["nspin"],
                ),
                "dimension",
                "background aggregate dimensions/ownership",
            )
            if second:
                parts.append((second.channel, dim, d2))
            dim += d2
        dtype = "<c16" if spinor else "<f8"
        nblock = 4 if spinor else data.envelope["nspin"]
        occupation = c.array(dtype, (dim, dim, nblock))
        potential = c.array(dtype, (dim, dim, nblock))
        scalar = c.array(dtype, (dim, dim))
        magnetic = c.array(dtype, (dim, dim, 3) if spinor else (dim, dim))
        c.end()
        if spinor:
            _hub_hermitian(_hub_dense(occupation), "complex native occupation")
            _hub_hermitian(_hub_dense(potential), "complex native potential")
            expected_scalar, expected_magnetic = _hub_pauli(potential)
            u = data.parameters[channel.species - 1].ry[0]
            _hub_close(
                potential,
                _hub_native_potential(occupation, u),
                "native U/2 I - U n-transpose source",
            )
        else:
            for name, matrix in (("occupation", occupation), ("potential", potential)):
                for spin in range(nblock):
                    _hub_hermitian(matrix[:, :, spin], f"native {name} spin {spin}")
            expected_scalar = potential.mean(axis=2)
            expected_magnetic = (
                (potential[:, :, 0] - potential[:, :, 1]) / 2
                if nblock == 2
                else np.zeros_like(scalar)
            )
        _hub_close(scalar, expected_scalar, "scalar U coefficient split")
        _hub_close(magnetic, expected_magnetic, "magnetic U coefficient split")
        data.states.append(
            QEHubbardState(
                channel.channel,
                channel.atom,
                channel.species,
                dim,
                occupation,
                potential,
                scalar,
                magnetic,
                tuple(parts),
            )
        )


def _hub_channel_columns(data, channel):
    if data.envelope["collinear"]:
        return np.flatnonzero(data.columns[:, 2] == channel)
    rows = np.flatnonzero(data.spin_columns[:, 1] == channel)
    return rows[np.lexsort((data.spin_columns[rows, 2], data.spin_columns[rows, 3]))]


def _hub_source_matrix(data, spin):
    result = np.zeros((data.envelope["ncol"],) * 2, dtype=complex)
    spinor = not data.envelope["collinear"]
    for state in data.states:
        if spinor:
            rows = _hub_channel_columns(data, state.channel)
            result[np.ix_(rows, rows)] = _hub_dense(state.potential)
        else:
            for channel, first, width in state.parts:
                rows = _hub_channel_columns(data, channel)
                result[np.ix_(rows, rows)] = state.potential[
                    first : first + width, first : first + width, spin - 1
                ]
    return result


def _read_hubbard_kpoint(reader, data, context, index, spinor):
    c = _HubbardRecord(reader, 18 if spinor else 10)
    if spinor:
        ik, npw, npol, ncol, nbnd = c.ints(5)
        _hub_require(npol == 2, "dimension", "PW spin components")
    else:
        ik, npw, ncol, nbnd = c.ints(4)
    _hub_require(
        ik == index
        and npw > 0
        and (ncol, nbnd) == (data.envelope["ncol"], data.envelope["nbnd"]),
        "dimension",
        "per-k native coefficient dimensions/order",
    )
    xk, weight = c.reals(3), float(c.reals(1)[0])
    spin = 1 if spinor else c.ints(1)[0]
    _hub_require(
        1 <= spin <= (1 if spinor else data.envelope["nspin"]),
        "mapping",
        "per-k spin index",
    )
    g_index = c.array("<i4", npw)
    _hub_require(
        np.all(g_index > 0) and len(np.unique(g_index)) == npw,
        "mapping",
        "missing or duplicate global PW indices",
    )
    wg, et = c.reals(nbnd), c.reals(nbnd)
    coeff = c.array("<c16", (ncol, nbnd))
    c.end()
    _hub_close(xk, context["xk"][index - 1], "native/predecessor k vector")
    _hub_close(weight, context["wk"][index - 1], "native/predecessor k weight")
    _hub_close(wg, context["wg"][index - 1], "native/predecessor occupations")
    _hub_close(et, context["et"][index - 1], "native/predecessor eigenvalues")
    if not spinor:
        _hub_require(
            spin == context["isk"][index - 1], "mapping", "predecessor spin map"
        )
    data.kpoints.append(
        QEHubbardKPoint(ik, npw, spin, xk, weight, g_index, wg, et, coeff)
    )


def _read_hubbard_action(reader, data, source_matrix, spinor):
    kp = data.kpoints[-1]
    c = _HubbardRecord(reader, 19 if spinor else 11)
    if spinor:
        ik, npw, npol, trials, ncol, algorithm, seed = c.ints(7)
        _hub_require(npol == 2, "dimension", "action PW spin components")
    else:
        ik, npw, trials, ncol, algorithm, seed = c.ints(6)
    _hub_require(
        (ik, npw, trials, ncol, algorithm, seed)
        == (kp.index, kp.npw, 3, data.envelope["ncol"], 1, 617021),
        "validation",
        "missing or inconsistent deterministic full-PW trials",
    )
    metrics = c.reals((7, 3))
    projection = c.array("<c16", (ncol, 3))
    rows = npw * (2 if spinor else 1)
    wfcu = c.array("<c16", (rows, ncol))
    psi, native, exported = (c.array("<c16", (rows, 3)) for _ in range(3))
    c.end()
    _hub_require(np.all(metrics >= 0), "validation", "negative action metrics")
    # First trial is independently reproducible; beta-orthogonality remains
    # a measured native gate because the entire beta bank is not in the ABI.
    g = kp.global_g_index.astype(np.int64)
    samples = []
    for spin in range(2 if spinor else 1):
        components = []
        for stream in (1, 2):
            state = (
                seed
                + ik * 104729
                + 13007
                + g * 48271
                + spin * 15485863
                + stream * 69621
            ) % 2147483647
            components.append(((state * 48271 + 1) % 2147483647) / 2147483647 - 0.5)
        samples.append(components[0] + 1j * components[1])
    trial = np.concatenate(samples)
    trial /= np.linalg.norm(trial)
    _hub_close(psi[:, 0], trial, "deterministic trial/global-G evidence", rtol=0)
    rebuilt_projection = wfcu.conj().T @ psi
    _hub_close(projection, rebuilt_projection, "independent PW trial contractions")
    rebuilt = wfcu @ (source_matrix @ rebuilt_projection)
    input_norm = np.linalg.norm(psi, axis=0)
    native_norm = np.linalg.norm(native, axis=0)
    exported_norm = np.linalg.norm(exported, axis=0)
    difference = np.linalg.norm(native - exported, axis=0)
    scale = np.maximum(native_norm, exported_norm)
    relative = np.divide(
        difference, scale, out=np.zeros(3), where=scale > np.finfo(float).tiny
    )
    for value, expected, name in (
        (input_norm, np.ones(3), "trial normalization"),
        (metrics[0], input_norm, "input action scales"),
        (
            metrics[2],
            np.linalg.norm(rebuilt_projection, axis=0),
            "projector-span scales",
        ),
        (metrics[3], native_norm, "native action scales"),
        (metrics[4], exported_norm, "exported action scales"),
        (metrics[5], difference, "absolute action residuals"),
        (metrics[6], relative, "relative action residuals"),
    ):
        _hub_close(value, expected, name, rtol=1e-8)
    _hub_close(
        exported, rebuilt, "stored action versus independent WfcU V WfcU-dagger action"
    )
    rebuilt_difference = np.linalg.norm(native - rebuilt, axis=0)
    rebuilt_scale = np.maximum(native_norm, np.linalg.norm(rebuilt, axis=0))
    rebuilt_relative = np.divide(
        rebuilt_difference,
        rebuilt_scale,
        out=np.zeros(3),
        where=rebuilt_scale > np.finfo(float).tiny,
    )
    _hub_require(
        np.all(relative[:2] <= 1e-10)
        and np.all(rebuilt_relative[:2] <= 1e-10)
        and native_norm[2] <= 1e-12
        and difference[2] <= 1e-12
        and rebuilt_difference[2] <= 1e-12
        and metrics[1, 1] <= 1e-10
        and metrics[2, 2] <= 1e-10,
        "validation",
        "full-PW relative/analytic-zero/span gate",
    )
    data.actions.append(
        QEHubbardAction(
            ik,
            algorithm,
            seed,
            metrics,
            projection,
            wfcu,
            psi,
            native,
            exported,
        )
    )


def _read_hubbard_checks(reader, data, spinor):
    # Native new_ns is serialized for every atom before native new_nsb.
    states = sorted(
        data.states, key=lambda s: (data.channels[s.channel - 1].kind, s.atom)
    )
    for state in states:
        dim = state.dimension
        c = _HubbardRecord(reader, 20 if spinor else 12)
        expected = (
            (state.channel, state.atom, state.species, dim, 2, 4)
            if spinor
            else (
                state.channel,
                state.atom,
                state.species,
                1 if data.channels[state.channel - 1].kind == 1 else 0,
                dim,
                data.envelope["nspin"],
            )
        )
        _hub_require(
            c.ints(6) == expected, "dimension", "occupation evidence ownership"
        )
        absolute, relative = map(float, c.reals(2))
        shape = (dim, dim, 4 if spinor else data.envelope["nspin"])
        dtype = "<c16" if spinor else "<f8"
        native, rebuilt = c.array(dtype, shape), c.array(dtype, shape)
        c.end()
        if spinor:
            _hub_hermitian(_hub_dense(native), "fresh native complex occupation")
            _hub_hermitian(_hub_dense(rebuilt), "fresh exported complex occupation")
        else:
            for spin in range(data.envelope["nspin"]):
                _hub_hermitian(native[:, :, spin], "fresh native occupation")
                _hub_hermitian(rebuilt[:, :, spin], "fresh exported occupation")
        measured_abs, measured_rel = _hub_residual(native, rebuilt)
        _hub_require(
            0 <= absolute <= 1e-12
            and 0 <= relative <= 1e-10
            and measured_abs <= 1e-12
            and measured_rel <= 1e-10,
            "validation",
            "native occupation reconstruction gate",
        )
        _hub_close(absolute, measured_abs, "reported occupation absolute error")
        _hub_close(
            relative, measured_rel, "reported occupation relative error", rtol=1e-8
        )
        kind = 1 if spinor else expected[3]
        data.occupation_checks.append(
            QEHubbardCheck(
                state.channel,
                state.atom,
                state.species,
                kind,
                absolute,
                relative,
                native,
                rebuilt,
            )
        )
    # Native symmetry may rotate orbitals/spins and exchange equivalent atoms.
    # Species/kind total traces survive those operations; per-site matrices do not.
    groups = {}
    parts_by_channel = {state.channel: state.parts for state in data.states}
    for check in data.occupation_checks:
        groups.setdefault((check.species, check.kind), []).append(check)
    for checks in groups.values():
        rows = np.concatenate(
            [
                _hub_channel_columns(data, channel)
                for check in checks
                for channel, _, _ in parts_by_channel[check.channel]
            ]
        )
        overlap_trace = sum(
            float(np.dot(kp.wg, np.sum(np.abs(kp.coefficients[rows]) ** 2, axis=0)))
            for kp in data.kpoints
        )
        if not spinor and data.envelope["nspin"] == 1:
            overlap_trace *= 0.5
        native_trace = sum(
            float(
                np.trace(check.native[:, :, 0] + check.native[:, :, 3]).real
                if spinor
                else np.trace(check.native, axis1=0, axis2=1).sum()
            )
            for check in checks
        )
        # Mean occupation uses the native per-entry tolerances without an
        # arbitrary atom-count-dependent tolerance. wg already contains k weight.
        count = sum(check.native.shape[0] for check in checks) * (
            2 if spinor else data.envelope["nspin"]
        )
        _hub_close(
            overlap_trace / count,
            native_trace / count,
            "native overlaps disagree with symmetry-invariant fresh occupation",
        )
    if not spinor:
        return
    for state in data.states:
        dim = state.dimension
        c = _HubbardRecord(reader, 23)
        _hub_require(
            c.ints(4) == (state.channel, state.atom, state.species, dim),
            "dimension",
            "potential evidence ownership",
        )
        absolute, relative = map(float, c.reals(2))
        native = c.array("<c16", (dim, dim, 4))
        rebuilt = c.array("<c16", (dim, dim, 4))
        c.end()
        expected = _hub_native_potential(
            state.occupation, data.parameters[state.species - 1].ry[0]
        )
        _hub_close(native, state.potential, "frozen native potential evidence")
        _hub_close(
            rebuilt, expected, "independent frozen occupation/potential relation"
        )
        measured_abs, measured_rel = _hub_residual(native, rebuilt)
        _hub_close(absolute, measured_abs, "reported potential absolute error")
        _hub_close(
            relative, measured_rel, "reported potential relative error", rtol=1e-8
        )
        _hub_require(
            absolute >= 0
            and relative >= 0
            and relative <= 1e-10
            and measured_rel <= 1e-10
            and (np.max(np.abs(expected), initial=0) > 1e-12 or measured_abs <= 1e-12),
            "validation",
            "independent native potential gate",
        )
        data.potential_checks.append(
            QEHubbardCheck(
                state.channel,
                state.atom,
                state.species,
                1,
                absolute,
                relative,
                native,
                rebuilt,
            )
        )


def _read_hubbard_rotations(reader, data):
    for state in data.states:
        c = _HubbardRecord(reader, 22)
        atom, nt, ch, dim, npol, blocks, nrot, ntangent, nstep, origin = c.ints(10)
        _hub_require(
            (atom, nt, ch, dim, npol, blocks, nrot, ntangent, nstep)
            == (
                state.atom,
                state.species,
                state.channel,
                state.dimension,
                2,
                4,
                3,
                2,
                3,
            )
            and origin in (0, 1, 2),
            "dimension",
            "rotation/tangent dimensions or ownership",
        )
        direction, axes, angles = c.reals(3), c.reals((3, 3)), c.reals(3)
        metrics = {
            name: c.reals(3)
            for name in (
                "su2_pauli",
                "native_su2",
                "action_relative",
                "action_absolute",
                "charge_absolute",
                "scalar_absolute",
            )
        }
        tangents, tangent_axes, steps = c.reals((3, 2)), c.reals((3, 2)), c.reals(3)
        analytic = c.array("<c16", (dim, dim, 4, 2))
        finite = c.array("<c16", (dim, dim, 4, 2, 3))
        absolute, relative, convergence = (
            c.reals((2, 3)),
            c.reals((2, 3)),
            c.reals((2, 2)),
        )
        c.end()
        data.rotations.append(
            QEHubbardRotation(
                atom,
                nt,
                ch,
                dim,
                origin,
                direction,
                axes,
                angles,
                metrics,
                tangents,
                tangent_axes,
                steps,
                analytic,
                finite,
                absolute,
                relative,
                convergence,
            )
        )


def _read_hubbard_frames(reader, data, context):
    conversion = data.unit_conversion or RYTOEV
    for atom, nt in enumerate(context["ityp"], 1):
        c = _HubbardRecord(reader, 21)
        site, species, count, origin = c.ints(4)
        operator_max = float(c.reals(1)[0])
        trace, trace_norm, direction = c.reals(3), float(c.reals(1)[0]), c.reals(3)
        (map_id,) = c.ints(1)
        occ_vector, occ_norm = c.reals(3), float(c.reals(1)[0])
        potential_vector, map_residual = c.reals(3), float(c.reals(1)[0])
        rank, discarded = c.ints(2)
        gap, rcond, clean_gap, unit_residual = map(float, c.reals(4))
        c.end()
        _hub_require(
            (site, species, count) == (atom, nt, context["nh"][nt - 1])
            and origin in (0, 1, 2)
            and map_id == 1,
            "mapping",
            "source-frame site/origin/map",
        )
        _hub_require(
            (rcond, clean_gap) == (1e-6, 100.0),
            "validation",
            "source-frame conditioning policy",
        )
        n = count
        if n:
            gram = context["gram"][:n, :n, atom - 1]
            _hub_hermitian(gram, "source-frame Gram")
            spectrum = _qe_gram_spectrum_report(gram)
            _hub_require(spectrum["retained"] > 0, "validation", "empty retained Gram")
            try:
                inverse = _qe_gram_inverse(gram)
            except ValueError as exc:
                raise QEHubbardError("validation", str(exc)) from exc
            eigenvalues = np.asarray(spectrum["eigenvalues"])
            dropped = eigenvalues[eigenvalues <= spectrum["cutoff"]]
            kept = eigenvalues[eigenvalues > spectrum["cutoff"]]
            expected_gap = (
                float(kept.min() / dropped.max())
                if dropped.size and dropped.max() > 0
                else 0.0
            )
            expected_rank, expected_discarded = (
                spectrum["retained"],
                spectrum["discarded"],
            )
            smooth = np.stack(
                [
                    inverse @ context["field"][:n, :n, atom - 1, axis] @ inverse
                    for axis in range(3)
                ],
                axis=-1,
            )
            _, magnetic = _hub_pauli(context["magnetic"][:n, :n, atom - 1])
            xc = 2 * conversion * (smooth + magnetic)
            component_max = max(
                float(np.abs(2 * conversion * smooth).max()),
                float(np.abs(2 * conversion * magnetic).max()),
            )
        else:
            expected_rank = expected_discarded = 0
            expected_gap = component_max = 0.0
            xc = np.empty((0, 0, 3), dtype=complex)
        _hub_require(
            (rank, discarded) == (expected_rank, expected_discarded),
            "validation",
            "source-frame Gram ranks",
        )
        _hub_close(gap, expected_gap, "source-frame eigengap", rtol=1e-8, atol=1e-10)
        expected_max = float(np.abs(xc).max(initial=0))
        expected_trace = np.trace(xc, axis1=0, axis2=1).real
        expected_norm = float(np.linalg.norm(expected_trace))
        _hub_close(operator_max, expected_max, "original-XC source maximum")
        _hub_close(trace, expected_trace, "original-XC source trace")
        _hub_close(trace_norm, expected_norm, "original-XC source trace norm")
        expected_origin = 0
        expected_direction = np.zeros(3)
        expected_occ = np.zeros(3)
        expected_potential = np.zeros(3)
        expected_occ_norm = expected_map = expected_unit = 0.0
        if expected_norm > 1e-10:
            expected_origin = 1
            expected_direction = expected_trace / expected_norm
            expected_unit = abs(np.linalg.norm(expected_direction) - 1)
        else:
            zero_budget = (
                64 * np.finfo(float).eps * max(component_max, np.finfo(float).tiny)
            )
            _hub_require(
                expected_max <= zero_budget,
                "source",
                "nonzero original XC has no Pauli-trace direction",
            )
            state = next((s for s in data.states if s.atom == atom), None)
            if state is not None:
                rho_trace = np.trace(state.occupation, axis1=0, axis2=1)
                expected_occ = np.array(
                    [
                        (rho_trace[1] + rho_trace[2]).real,
                        2 * rho_trace[1].imag,
                        (rho_trace[0] - rho_trace[3]).real,
                    ]
                )
                expected_occ_norm = float(np.linalg.norm(expected_occ))
                expected_potential = np.trace(state.magnetic, axis1=0, axis2=1).real
                u = data.parameters[nt - 1].ry[0]
                native_potential = _hub_native_potential(state.occupation, u)
                scale = max(
                    float(np.abs(state.potential).max()),
                    float(np.abs(native_potential).max()),
                    np.finfo(float).tiny,
                )
                expected_map = (
                    float(np.abs(state.potential - native_potential).max()) / scale
                )
                q_scale = max(
                    np.linalg.norm(expected_potential),
                    0.5 * abs(u) * expected_occ_norm,
                    1e-30,
                )
                q_residual = (
                    float(np.abs(expected_potential + 0.5 * u * expected_occ).max())
                    / q_scale
                )
                _hub_require(
                    expected_map <= 1e-10 and q_residual <= 1e-8,
                    "validation",
                    "native occupation-to-U frame map",
                )
                magnetic_max = float(np.abs(state.magnetic).max())
                _hub_require(
                    2 * conversion * magnetic_max <= 1e-10 or expected_occ_norm > 1e-12,
                    "source",
                    "magnetic U has no mapped reference direction",
                )
                if expected_occ_norm > 1e-12 and 2 * conversion * magnetic_max > 1e-10:
                    expected_origin = 2
                    expected_direction = expected_occ / expected_occ_norm
                    expected_unit = max(
                        q_residual,
                        _hub_residual(state.potential, native_potential)[1],
                        abs(np.linalg.norm(expected_direction) - 1),
                    )
        _hub_require(
            origin == expected_origin, "source", "reference-frame origin is not XC/U"
        )
        for actual, expected, name in (
            (direction, expected_direction, "reference direction"),
            (occ_vector, expected_occ, "occupation-axis map"),
            (occ_norm, expected_occ_norm, "occupation-axis norm"),
            (potential_vector, expected_potential, "U trace-vector map"),
            (map_residual, expected_map, "U coefficient map residual"),
            (unit_residual, expected_unit, "frame-unit residual"),
        ):
            _hub_close(actual, expected, name, rtol=1e-8)
        _hub_require(
            unit_residual >= 0 and (not origin or unit_residual <= 1e-12),
            "validation",
            "source-frame unit/map gate",
        )
        data.frames.append(
            QEHubbardFrame(
                atom,
                nt,
                count,
                origin,
                operator_max,
                trace,
                trace_norm,
                direction,
                map_id,
                occ_vector,
                occ_norm,
                potential_vector,
                map_residual,
                rank,
                discarded,
                gap,
                rcond,
                clean_gap,
                unit_residual,
            )
        )


def _hub_rotation(axis, angle, dim):
    spin = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * np.einsum(
        "a,aij->ij", axis, PAULI_MATRICES
    )
    return np.kron(spin, np.eye(dim))


def _hub_rotated_native(state, u, axis, angle):
    q = _hub_rotation(axis, angle, state.dimension)
    occupation = q.conj() @ _hub_dense(state.occupation) @ q.T
    return _hub_native_potential(_hub_blocks(occupation), u)


def _validate_hubbard_rotations(data):
    base_source = _hub_source_matrix(data, 1)
    action = data.actions[0]
    projection = action.wfcu.conj().T @ action.psi[:, 0]
    for state, rotation in zip(data.states, data.rotations, strict=True):
        frame = data.frames[state.atom - 1]
        _hub_require(
            rotation.frame_origin == frame.origin, "source", "tangent frame origin"
        )
        _hub_close(
            rotation.reference_direction, frame.direction, "rotation reference frame"
        )
        axes = np.array(
            [[1, 0, 1 / np.sqrt(14)], [0, 1, 2 / np.sqrt(14)], [0, 0, 3 / np.sqrt(14)]]
        )
        _hub_close(rotation.axes, axes, "quarter-turn and tilted probe axes")
        _hub_close(
            rotation.angles,
            [np.pi / 2, np.pi / 2, 2 * np.pi / 5],
            "quarter-turn and tilted probe angles",
        )
        _hub_close(
            rotation.steps, [1e-3, 5e-4, 2.5e-4], "tangent refinement steps", rtol=0
        )
        _hub_require(
            all(np.all(v >= 0) for v in rotation.metrics.values())
            and np.all(rotation.tangent_absolute_error >= 0)
            and np.all(rotation.tangent_relative_error >= 0)
            and np.all(rotation.step_convergence >= 0),
            "validation",
            "negative rotation or derivative residual",
        )
        tangents = np.zeros((3, 2))
        tangent_axes = np.zeros((3, 2))
        if frame.origin:
            reference = (
                np.array([1.0, 0.0, 0.0])
                if abs(frame.direction[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            tangents[:, 0] = (
                reference - np.dot(reference, frame.direction) * frame.direction
            )
            tangents[:, 0] /= np.linalg.norm(tangents[:, 0])
            tangents[:, 1] = np.cross(frame.direction, tangents[:, 0])
            tangent_axes = np.cross(frame.direction, tangents.T).T
        _hub_close(
            rotation.tangent_directions, tangents, "transverse tangent directions"
        )
        _hub_close(rotation.tangent_axes, tangent_axes, "positive rotation axes")
        u = data.parameters[state.species - 1].ry[0]
        rows = _hub_channel_columns(data, state.channel)
        source = _hub_dense(state.potential)
        for ir in range(3):
            axis, angle = axes[:, ir], rotation.angles[ir]
            q = _hub_rotation(axis, angle, state.dimension)
            su2 = _hub_blocks(q @ source @ q.conj().T)
            native = _hub_rotated_native(state, u, axis, angle)
            # Rodrigues rotation of the Pauli vector is independent of SU(2).
            magnetic = (
                state.magnetic * np.cos(angle)
                + np.cross(axis, state.magnetic) * np.sin(angle)
                + np.einsum("a,ija->ij", axis, state.magnetic)[:, :, None]
                * axis
                * (1 - np.cos(angle))
            )
            pauli = _hub_magnetic_blocks(magnetic)
            pauli[:, :, 0] += state.scalar
            pauli[:, :, 3] += state.scalar
            _, su2_error = _hub_residual(su2, pauli)
            _, native_error = _hub_residual(native, su2)
            _hub_require(
                su2_error <= 1e-10 and native_error <= 1e-10,
                "validation",
                "finite SU(2)/Pauli/native coefficient rotation",
            )
            _hub_close(
                rotation.metrics["su2_pauli"][ir],
                su2_error,
                "measured SU(2)/Pauli residual",
            )
            _hub_close(
                rotation.metrics["native_su2"][ir],
                native_error,
                "measured native/SU(2) residual",
            )
            native_source, exported_source = base_source.copy(), base_source.copy()
            native_source[np.ix_(rows, rows)] = _hub_dense(native)
            exported_source[np.ix_(rows, rows)] = _hub_dense(pauli)
            native_action = action.wfcu @ (native_source @ projection)
            exported_action = action.wfcu @ (exported_source @ projection)
            absolute, relative = _hub_residual(native_action, exported_action)
            scale = max(np.linalg.norm(native_action), np.linalg.norm(exported_action))
            _hub_require(
                relative <= 1e-10 if scale > 1e-12 else absolute <= 1e-12,
                "validation",
                "finite-rotation full-PW action",
            )
            for name, measured in (
                ("action_absolute", absolute),
                ("action_relative", relative),
            ):
                _hub_close(rotation.metrics[name][ir], measured, f"measured {name}")
            rotated_n = _hub_blocks(q.conj() @ _hub_dense(state.occupation) @ q.T)
            charge_abs, charge_rel = _hub_residual(
                rotated_n[:, :, 0] + rotated_n[:, :, 3],
                state.occupation[:, :, 0] + state.occupation[:, :, 3],
            )
            scalar_abs, scalar_rel = _hub_residual(_hub_pauli(native)[0], state.scalar)
            _hub_require(
                charge_rel <= 1e-10 and scalar_rel <= 1e-10,
                "validation",
                "rotation changed frozen charge or scalar potential",
            )
            _hub_close(
                rotation.metrics["charge_absolute"][ir],
                charge_abs,
                "charge invariant residual",
            )
            _hub_close(
                rotation.metrics["scalar_absolute"][ir],
                scalar_abs,
                "scalar invariant residual",
            )
        if not frame.origin:
            for value in (
                rotation.analytic,
                rotation.central_difference,
                rotation.tangent_absolute_error,
                rotation.tangent_relative_error,
                rotation.step_convergence,
            ):
                _hub_require(
                    not np.any(value),
                    "source",
                    "undefined frame has manufactured tangents",
                )
            continue
        for tangent in range(2):
            axis = tangent_axes[:, tangent]
            analytic = _hub_magnetic_blocks(np.cross(axis, state.magnetic))
            _hub_close(
                rotation.analytic[:, :, :, tangent],
                analytic,
                "analytic dVmag/dtheta, not twice Vmag",
            )
            for step, h in enumerate(rotation.steps):
                plus = _hub_pauli(_hub_rotated_native(state, u, axis, h))[1]
                minus = _hub_pauli(_hub_rotated_native(state, u, axis, -h))[1]
                finite = _hub_magnetic_blocks((plus - minus) / (2 * h))
                _hub_close(
                    rotation.central_difference[:, :, :, tangent, step],
                    finite,
                    "independent native central difference",
                    rtol=1e-9,
                )
                absolute, relative = _hub_residual(analytic, finite)
                _hub_close(
                    rotation.tangent_absolute_error[tangent, step],
                    absolute,
                    "measured derivative absolute error",
                    rtol=1e-8,
                )
                _hub_close(
                    rotation.tangent_relative_error[tangent, step],
                    relative,
                    "measured derivative relative error",
                    rtol=1e-8,
                )
                _hub_require(
                    absolute <= 1e-12
                    if np.abs(analytic).max(initial=0) <= 1e-12
                    else relative <= 1e-6,
                    "validation",
                    "central-difference derivative gate",
                )
            for step in range(2):
                first = rotation.central_difference[:, :, :, tangent, step]
                second = rotation.central_difference[:, :, :, tangent, step + 1]
                absolute, relative = _hub_residual(first, second)
                _hub_close(
                    rotation.step_convergence[tangent, step],
                    relative,
                    "measured tangent refinement",
                    rtol=1e-8,
                )
                _hub_require(
                    absolute <= 1e-12
                    if max(np.abs(first).max(), np.abs(second).max()) <= 1e-12
                    else relative <= 1e-6,
                    "validation",
                    "tangent step-refinement gate",
                )


def _read_hubbard_restoration(reader, data):
    c = _HubbardRecord(reader, 24)
    names = (
        "evc_abs",
        "wfcu_abs",
        "vkb_abs",
        "becp_r_abs",
        "becp_k_abs",
        "becp_nc_abs",
        "deeq_abs",
        "deeq_nc_abs",
        "rho_ns_nc_abs",
        "v_ns_nc_abs",
    )
    data.restoration = dict(zip(names, map(float, c.reals(10)), strict=True))
    flags = (
        "dims_equal",
        "current_k_equal",
        "npw_equal",
        "current_spin_equal",
        "lda_u_equal",
    )
    data.restoration.update(zip(flags, c.ints(5), strict=True))
    data.restoration["native_action_abs"] = float(c.reals(1)[0])
    c.end()
    _hub_require(
        all(0 <= data.restoration[k] <= 1e-12 for k in (*names, "native_action_abs"))
        and all(data.restoration[k] == 1 for k in flags),
        "validation",
        "native state/action was not completely restored",
    )


def _read_hubbard_validation(reader, data, context, profile):
    c = _HubbardRecord(reader, 14)
    active, restored, action_restored, trials, algorithm, seed = c.ints(6)
    measured = c.reals(6)
    (source_method,) = c.ints(1)
    source_difference, source_scale, restore_error, unit_bridge = map(float, c.reals(4))
    c.end()
    _hub_require(
        active == data.envelope["active"] and restored == 1,
        "validation",
        "missing final source-state restoration",
    )
    _hub_require(
        np.all(measured >= 0)
        and min(source_difference, source_scale, restore_error, unit_bridge) >= 0,
        "validation",
        "negative final validation residual/scale",
    )
    if active:
        _hub_require(
            (action_restored, trials, algorithm, seed) == (1, 3, 1, 617021),
            "validation",
            "missing restored native action probe",
        )
        _hub_require(
            np.all(measured <= [1e-10, 1e-12, 1e-10, 1e-12, 1e-10, 1e-12])
            and restore_error <= 1e-12,
            "validation",
            "aggregate action/occupation/restoration gate",
        )
        if profile in ("kb", "spinor"):
            carrier = context["deeq"] if profile == "kb" else context["total"]
            retained_scale = max(
                (
                    float(np.abs(carrier[:count, :count, atom, :]).max())
                    for atom, count in enumerate(context["nh"][context["ityp"] - 1])
                    if count
                ),
                default=0.0,
            )
            if profile == "kb":
                _hub_close(
                    source_scale,
                    retained_scale,
                    "reported deeq scale differs from populated carrier",
                    atol=1e-12,
                    rtol=1e-12,
                    code="source",
                )
                source_scale = retained_scale
                data.provenance["carrier_scale_evidence"] = (
                    "populated_deeq_reconstructed"
                )
            else:
                # QE also measures a real deeq bank that NC1.2 does not export.
                # The retained complex bank provides a lower bound, not equality.
                _hub_require(
                    source_scale + 1e-12 * max(1, retained_scale) >= retained_scale,
                    "source",
                    "reported combined carrier scale is below retained deeq_nc",
                )
                data.provenance["carrier_scale_evidence"] = (
                    "native_combined_scale_deeq_nc_lower_bound_only"
                )
        else:
            data.provenance["carrier_scale_evidence"] = "native_measurement_only"
        _hub_require(
            source_method == (1 if data.envelope["nkb"] else 2)
            and source_difference <= 1e-12 * max(1, source_scale)
            and (source_method != 2 or source_difference == source_scale == 0),
            "source",
            "native deeq carrier not independently shown to be U-free",
        )
        _hub_close(unit_bridge, data.unit_bridge_residual, "final unit bridge")
        for index, kind in ((2, 1), (4, 0)):
            checks = [g for g in data.occupation_checks if g.kind == kind]
            expected = [
                max((g.relative_error for g in checks), default=0),
                max((g.absolute_error for g in checks), default=0),
            ]
            _hub_close(
                measured[index : index + 2], expected, "aggregate occupation evidence"
            )
        expected_action = [
            max(float(a.metrics[6, :2].max()) for a in data.actions),
            max(float(a.metrics[3, 2]) for a in data.actions),
        ]
        _hub_close(measured[:2], expected_action, "aggregate full-PW action evidence")
    else:
        _hub_require(
            (action_restored, trials, algorithm, seed, source_method) == (0, 0, 0, 0, 0)
            and not np.any(measured)
            and source_difference == source_scale == restore_error == unit_bridge == 0,
            "source",
            "absent-U stream contains active validation payload",
        )
    data.validation = dict(
        active=active,
        restored=restored,
        action_restored=action_restored,
        ntrial=trials,
        algorithm=algorithm,
        seed=seed,
        action_relative=float(measured[0]),
        analytic_zero_absolute_Ry=float(measured[1]),
        main_occupation_relative=float(measured[2]),
        main_occupation_absolute=float(measured[3]),
        background_occupation_relative=float(measured[4]),
        background_occupation_absolute=float(measured[5]),
        source_method=source_method,
        source_difference_Ry=source_difference,
        source_scale_Ry=source_scale,
        restored_action_absolute_Ry=restore_error,
        unit_bridge_residual=unit_bridge,
    )
    _hub_require(reader.at_eof(), "layout", "trailing bytes after final validation")
    data.complete = True


def _parse_qe_hubbard(reader, context, profile):
    data = _read_hubbard_header(reader, context, profile)
    spinor = profile == "spinor"
    _read_hubbard_maps(reader, data, context, spinor)
    if spinor:
        _read_hubbard_spin_layout(reader, data)
    if data.state == "active":
        _read_hubbard_states(reader, data, spinor)
        sources = {
            spin: _hub_source_matrix(data, spin)
            for spin in range(1, 2 if spinor else data.envelope["nspin"] + 1)
        }
        for index in range(1, data.envelope["nks"] + 1):
            _read_hubbard_kpoint(reader, data, context, index, spinor)
            _read_hubbard_action(reader, data, sources[data.kpoints[-1].spin], spinor)
        _read_hubbard_checks(reader, data, spinor)
        if spinor:
            _read_hubbard_rotations(reader, data)
    if spinor:
        _read_hubbard_frames(reader, data, context)
        if data.state == "active":
            _validate_hubbard_rotations(data)
            _read_hubbard_restoration(reader, data)
    _read_hubbard_validation(reader, data, context, profile)
    return data


def _qe_hubbard_admission(dump, legacy_no_u):
    if dump.hubbard.state == "unknown":
        _hub_require(
            legacy_no_u is True,
            "legacy_attestation",
            "legacy input has unknown Hubbard state; explicit legacy_no_u=True required",
        )
        return {
            "hubbard_state": "user_attested_no_u",
            "legacy_no_u": True,
            "hubbard_provenance": {
                "origin": "user_attestation",
                "native_state": "unknown",
            },
        }
    _hub_require(
        dump.hubbard.complete, "validation", "incomplete native Hubbard source"
    )
    _hub_require(
        dump.hubbard.state != "active",
        "normalization_pending",
        "active native U requires joint XC+U source normalization",
    )
    return {
        "hubbard_state": "absent",
        "legacy_no_u": False,
        "hubbard_provenance": {
            "origin": "native_extension",
            "native_state": "absent",
            "native_revision": dump.hubbard.provenance["native_revision"],
        },
    }


# ---------------------------------------------------------------------------
# Raw dump container
# ---------------------------------------------------------------------------
@dataclass
class QEProjectorDump:
    """Raw ``becp_dump`` records and metadata, in dumped (QE) units.

    Units: ``deeq``/``dvan``/``qq_at``/``et``/``ef*`` in Ry; ``alat`` in
    Bohr; ``omega`` in Bohr^3; ``tau`` cartesian in units of ``alat``;
    ``xk`` cartesian in 2*pi/alat.  ``at``/``bg`` follow the QE convention
    ``at(i, j)`` = i-th cartesian component of the j-th (reciprocal) lattice
    vector.  ``ityp`` is 1-based; ``ofsbeta`` is the 0-based channel offset
    of each atom in the concatenated beta list (rebuilt from ``ityp`` +
    ``nh``).
    """

    path: Path
    version: str
    magic: str
    nspin: int
    nks: int
    nkstot: int
    nbnd: int
    nat: int
    nsp: int
    nhm: int
    lmaxkb: int
    nkb: int
    nelec: float
    ef: float
    ef_up: float
    ef_dw: float
    degauss: float
    ngauss: int
    ltetra: bool
    lgauss: bool
    at: np.ndarray
    bg: np.ndarray
    alat: float
    omega: float
    ibrav: int
    atm: list
    ityp: np.ndarray
    tau: np.ndarray
    nh: np.ndarray
    tvanp: np.ndarray
    tpawp: np.ndarray
    deeq: np.ndarray
    dvan: np.ndarray
    qq_at: np.ndarray
    gram: np.ndarray
    xk: np.ndarray
    wk: np.ndarray
    isk: np.ndarray
    et: np.ndarray
    wg: np.ndarray
    coefficients: list
    becsum: np.ndarray | None
    rho_bec: np.ndarray | None
    dbeta_xc: np.ndarray | None
    ddd_paw: np.ndarray | None
    ofsbeta: np.ndarray
    hubbard: QEHubbardData = field(default_factory=QEHubbardData)

    @property
    def nh_per_atom(self) -> np.ndarray:
        """Number of beta channels owned by each atom (QE ion order)."""
        return self.nh[self.ityp - 1]


# ---------------------------------------------------------------------------
# Atomic raw dump container
# ---------------------------------------------------------------------------
@dataclass
class QEAtomicDump:
    """Raw atomic-pseudo-wavefunction dump records, in dumped (QE) units.

    Units: ``cov_xc``/``cov_aug``/``et``/``ef*`` in Ry; ``alat`` in Bohr;
    ``omega`` in Bohr^3; ``tau`` cartesian in units of ``alat``; ``xk``
    cartesian in 2*pi/alat.  ``at``/``bg`` follow the QE convention
    ``at(i, j)`` = i-th cartesian component of the j-th (reciprocal) lattice
    vector.  ``ityp`` is 1-based; ``ofswfc`` is the 0-based channel offset of
    each atom in the concatenated atomic-wfc channel list (rebuilt from
    ``ityp`` + ``nsite_species``).  ``coefficients[ik] = C(nproj, nbnd) =
    <atomic_wfc|psi>`` and ``grams[ik] = M(nproj, nproj) =
    <atomic_wfc|atomic_wfc>`` are per-k pairs in dump order (coefficient
    record ``8+2*ik``, Gram record ``9+2*ik``). The trailing vertex records
    ``cov_xc(max_nsite, max_nsite, nat)`` and
    ``cov_aug(max_nsite, max_nsite, nat)`` are BZ-weighted R=0 onsite
    covariant matrices: normalized weighted averages of
    ``<atomic(k)|Vxc_up - Vxc_dn|atomic(k)>`` and
    ``B(k) (deeq_up - deeq_dn) B(k)^dagger``, both in Ry (NC aug naturally
    zero). There is no ``deeq``/``qq_at``/``becsum`` physical record in this
    format.
    """

    path: Path
    magic: str
    nspin: int
    nks: int
    nkstot: int
    nbnd: int
    nat: int
    nsp: int
    nproj: int
    max_nsite: int
    nelec: float
    ef: float
    ef_up: float
    ef_dw: float
    degauss: float
    ngauss: int
    ltetra: bool
    lgauss: bool
    at: np.ndarray
    bg: np.ndarray
    alat: float
    omega: float
    ibrav: int
    atm: list
    ityp: np.ndarray
    tau: np.ndarray
    nsite_species: np.ndarray
    projector_l: np.ndarray
    m_ordinal: np.ndarray
    radial_upf_index: np.ndarray
    xk: np.ndarray
    wk: np.ndarray
    isk: np.ndarray
    et: np.ndarray
    wg: np.ndarray
    coefficients: list
    grams: list
    cov_xc: np.ndarray
    cov_aug: np.ndarray
    ofswfc: np.ndarray
    hubbard: QEHubbardData = field(default_factory=QEHubbardData)

    @property
    def nsite_per_atom(self) -> np.ndarray:
        """Number of atomic-wfc channels owned by each atom (QE ion order)."""
        return self.nsite_species[self.ityp - 1]


def _decode_species_labels(payload: bytes, nsp: int, path: Path) -> list:
    """Decode the first ``nsp`` species labels from the species record.

    Handles both encodings seen in the wild: the spec layout ``S3 x nsp``
    and the QE >= 8 fixed-capacity ``CHARACTER(LEN=6)`` array (blank-padded
    6-byte labels, trailing entries blank).  Preference is widest-first so
    ``b'O     H     ' + padding`` decodes as ``['O', 'H']``, not ``['O']``.
    """
    widths = sorted({w for w in (6, 3) if w * nsp <= len(payload)}, reverse=True)
    for width in widths:
        labels = [payload[i * width : (i + 1) * width] for i in range(nsp)]
        if all(label.isascii() for label in labels) and all(
            label.rstrip(b" \x00") for label in labels
        ):
            return [label.decode("ascii").strip(" \x00") for label in labels]
    raise ValueError(
        f"QE dump {path}: cannot decode {nsp} species labels from the "
        f"{len(payload)}-byte species record"
    )


def _read_fermi_record(reader: _RecordReader):
    """Records 1: Fermi energies and smearing flags (shared by both dumps)."""
    chunks = reader.split_record(
        "fermi/smearing",
        [
            ("energies", "<f8", 5),
            ("smear", "<i4", 3),
        ],
    )
    nelec, ef, ef_up, ef_dw, degauss = (float(v) for v in chunks["energies"])
    ngauss, ltetra, lgauss = (int(v) for v in chunks["smear"])
    return nelec, ef, ef_up, ef_dw, degauss, ngauss, ltetra, lgauss


def _read_cell_record(reader: _RecordReader):
    """Record 2: cell vectors, reciprocal vectors, alat, omega, ibrav."""
    chunks = reader.split_record(
        "cell",
        [("lat", "<f8", 18), ("cell_scalars", "<f8", 2), ("ibrav", "<i4", 1)],
    )
    at = np.array(chunks["lat"][:9].reshape((3, 3), order="F"))
    bg = np.array(chunks["lat"][9:].reshape((3, 3), order="F"))
    alat, omega = (float(v) for v in chunks["cell_scalars"])
    ibrav = int(chunks["ibrav"][0])
    return at, bg, alat, omega, ibrav


def _read_ions_record(reader: _RecordReader, nat: int, nsp: int, path: Path):
    """Record 4: ityp(nat) and tau(3, nat) (shared by both dumps)."""
    chunks = reader.split_record(
        "ions", [("ityp", "<i4", nat), ("tau", "<f8", 3 * nat)]
    )
    ityp = np.array(chunks["ityp"], dtype=int)
    tau = np.array(chunks["tau"].reshape((3, nat), order="F").T)
    if ityp.min() < 1 or ityp.max() > nsp:
        raise ValueError(
            f"QE dump {path}: ityp values {sorted(set(ityp.tolist()))} out "
            f"of range [1, {nsp}]"
        )
    return ityp, tau


def _read_kmesh_record(reader: _RecordReader, nks: int, path: Path):
    """Record 6: xk(3, nks), wk(nks), isk(nks) (shared by both dumps)."""
    chunks = reader.split_record(
        "k-point data",
        [("xk", "<f8", 3 * nks), ("wk", "<f8", nks), ("isk", "<i4", nks)],
    )
    xk = np.array(chunks["xk"].reshape((3, nks), order="F").T)
    wk = np.array(chunks["wk"], dtype=float)
    isk = np.array(chunks["isk"], dtype=int)
    if not np.isin(isk, (1, 2)).all():
        raise ValueError(
            f"QE dump {path}: isk values must be 1 (up) or 2 (down); got "
            f"{sorted(set(isk.tolist()))}"
        )
    return xk, wk, isk


def _read_spectral_record(reader: _RecordReader, nbnd: int, nks: int):
    """Record 7: et(nbnd, nks) and wg(nbnd, nks) (shared by both dumps)."""
    chunks = reader.split_record(
        "bands", [("et", "<f8", nbnd * nks), ("wg", "<f8", nbnd * nks)]
    )
    et = np.array(chunks["et"].reshape((nbnd, nks), order="F").T)
    wg = np.array(chunks["wg"].reshape((nbnd, nks), order="F").T)
    return et, wg


# ---------------------------------------------------------------------------
# parse_qe_dump (KB/beta dump family)
# ---------------------------------------------------------------------------
def parse_qe_dump(path) -> QEProjectorDump:
    """Parse a QE ``becp_dump`` file into raw :class:`QEProjectorDump` data.

    Raises ``ValueError`` on unknown magic/version (including the atomic
    ``TB2JQEATWFC1.0`` family, which has its own parser), non-LSDA
    (``nspin != 2``), pool-parallel dumps (``nkstot != nks``), and record
    length/shape mismatches. NC needs a v1.2 projected-XC exchange vertex.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        reader = _RecordReader(path, fh)

        # -- record 0: magic + dimensions ------------------------------------
        # Manual header handling so an atomic-family file (8 header dims)
        # routed here gets the explicit atomic-parser error instead of a
        # layout error.
        payload = reader.read_raw("header")
        magic_raw = bytes(np.frombuffer(payload[:16], dtype="S16")[0])
        if magic_raw.rstrip(b" \x00") in _ATOMIC_MAGICS:
            raise ValueError(
                f"QE dump {path}: atomic-wavefunction dump (magic "
                f"'{ATOMIC_MAGIC}'); use parse_qe_atomic_dump/"
                f"read_qe_atomic_dump or qe2J.py --input <atomic_dump> "
                f"(the CLI auto-detects the file magic)"
            )
        if len(payload) != 16 + 9 * 4:
            raise ValueError(
                f"QE dump {path}: header record is {len(payload)} bytes, "
                f"expected {16 + 9 * 4} (S16 magic + 9 i4 dims)"
            )
        version = _MAGIC_VERSIONS.get(magic_raw.rstrip())
        if version is None:
            raise ValueError(
                f"QE dump {path}: unknown magic {magic_raw!r}; expected "
                f"TB2JQEDUMPV1.2 (v1.2), TB2JQEDUMPV1.1 (v1.1), "
                f"or TB2JQEDUMPV1 (v1.0)"
            )
        nspin, nks, nkstot, nbnd, nat, nsp, nhm, lmaxkb, nkb = (
            int(v) for v in np.frombuffer(payload[16:], dtype="<i4")
        )
        if nspin != 2 and not (version == "1.3" and nspin == 1):
            raise ValueError(
                f"QE dump {path}: nspin={nspin}; TB2J requires a collinear "
                f"LSDA (nspin=2) calculation"
            )
        if nkstot != nks:
            raise ValueError(
                f"QE dump {path}: nkstot={nkstot} != nks={nks}; pool-parallel "
                f"(-nk/-npools) dumps are not supported; rerun with npool=1"
            )
        if min(nks, nbnd, nat, nsp, nhm, nkb) < 1:
            raise ValueError(
                f"QE dump {path}: invalid dimensions nks={nks}, nbnd={nbnd}, "
                f"nat={nat}, nsp={nsp}, nhm={nhm}, nkb={nkb}"
            )

        # -- record 1: Fermi energies and smearing ---------------------------
        nelec, ef, ef_up, ef_dw, degauss, ngauss, ltetra, lgauss = _read_fermi_record(
            reader
        )

        # -- record 2: cell ---------------------------------------------------
        at, bg, alat, omega, ibrav = _read_cell_record(reader)

        # -- record 3: species labels ----------------------------------------
        # The dump spec lists ``S3 x nsp``; QE >= 8 (the fork's base) stores
        # ``atm`` as CHARACTER(LEN=6) with fixed capacity, so the record can
        # carry blank-padded 6-byte labels for more than nsp entries.  Decode
        # either encoding: take the first nsp labels at the widest width that
        # fills all of them with non-blank ascii.
        atm = _decode_species_labels(reader.read_raw("species labels"), nsp, path)

        # -- record 4: ions ---------------------------------------------------
        ityp, tau = _read_ions_record(reader, nat, nsp, path)

        # -- record 5: projector metadata --------------------------------------
        chunks = reader.split_record(
            "projector metadata",
            [("nh", "<i4", nsp), ("tvanp", "<i4", nsp), ("tpawp", "<i4", nsp)],
        )
        nh = np.array(chunks["nh"], dtype=int)
        tvanp = np.array(chunks["tvanp"], dtype=int) != 0
        tpawp = np.array(chunks["tpawp"], dtype=int) != 0
        if version == "1.0" or version == "1.1":
            # v1.2 carries the beta-projected dbeta_xc vertex which lifts the
            # historical NC+KB blocker; older dumps only have the deeq-only
            # vertex, which vanishes for NC (deeq = dvan, spin-independent).
            if not (tvanp.any() or tpawp.any()):
                raise ValueError(
                    f"QE dump {path}: no ultrasoft (US) or PAW species "
                    "present and no v1.2 dbeta_xc vertex record; NC+KB "
                    "rejected for this dump version: deeq-only separable "
                    "beta spin vertex vanishes (deeq=dvan, Δ≡0); provide a "
                    "v1.2 dump or see research "
                    "2026-10-08-qe-export-surface-and-operators"
                )
        if nh.min() < 0:
            raise ValueError(f"QE dump {path}: negative nh entries {nh.tolist()}")
        if nh.max() > nhm:
            raise ValueError(
                f"QE dump {path}: nh max {int(nh.max())} exceeds nhm={nhm}"
            )
        nkb_expected = int(nh[ityp - 1].sum())
        if nkb_expected != nkb:
            raise ValueError(
                f"QE dump {path}: nkb={nkb} inconsistent with the per-atom "
                f"channel count sum(nh[ityp])={nkb_expected}"
            )

        # -- records 6-9: operators and diagnostics ---------------------------
        deeq = reader.array("deeq", "<f8", (nhm, nhm, nat, nspin))
        dvan = reader.array("dvan", "<f8", (nhm, nhm, nsp))
        qq_at = reader.array("qq_at", "<f8", (nhm, nhm, nat))
        gram = reader.array("beta Gram diagnostic", "<c16", (nhm, nhm, nat))

        # -- record 10: k-point list ------------------------------------------
        xk, wk, isk = _read_kmesh_record(reader, nks, path)

        # -- record 11: bands --------------------------------------------------
        et, wg = _read_spectral_record(reader, nbnd, nks)

        # -- records 12..12+nks-1: per-k becp ----------------------------------
        coefficients = [
            reader.array(f"becp k-point {ik + 1}", "<c16", (nkb, nbnd))
            for ik in range(nks)
        ]

        # -- v1.1 occupations; v1.2 adds the projected-xc vertex records -----
        becsum = None
        rho_bec = None
        dbeta_xc = None
        ddd_paw = None
        npack = nhm * (nhm + 1) // 2
        if version in ("1.1", "1.2", "1.3"):
            becsum = reader.array("becsum", "<f8", (npack, nat, nspin))
            # rho%bec is written only for PAW runs (okpaw at dump time).
            if any(bool(f) for f in tpawp):
                rho_bec = reader.array("rho_bec", "<f8", (npack, nat, nspin))
        if version in ("1.2", "1.3"):
            # <beta_i | V_xc(up) - V_xc(dn) | beta_j> per atom (Ry); plus,
            # for PAW runs, the spin-resolved one-center D^1 coefficients.
            dbeta_xc = reader.array("dbeta_xc", "<f8", (nhm, nhm, nat))
            if any(bool(f) for f in tpawp):
                ddd_paw = reader.array("ddd_paw", "<f8", (npack, nat, nspin))
        hubbard = QEHubbardData()
        if version == "1.3":
            hubbard = _parse_qe_hubbard(
                reader,
                dict(
                    nspin=nspin,
                    nks=nks,
                    nbnd=nbnd,
                    nat=nat,
                    nsp=nsp,
                    nkb=nkb,
                    ityp=ityp,
                    atm=atm,
                    nh=nh,
                    deeq=deeq,
                    tvanp=tvanp,
                    tpawp=tpawp,
                    xk=xk,
                    wk=wk,
                    isk=isk,
                    et=et,
                    wg=wg,
                ),
                "kb",
            )
        if not reader.at_eof():
            raise ValueError(
                f"QE dump {path}: unexpected trailing data after the last "
                f"expected record ({reader.remaining_bytes()} bytes)"
            )

    ofsbeta = np.zeros(nat, dtype=int)
    if nat > 1:
        ofsbeta[1:] = np.cumsum(nh[ityp - 1])[:-1]

    return QEProjectorDump(
        path=path,
        version=version,
        magic=magic_raw.rstrip().decode("ascii"),
        nspin=nspin,
        nks=nks,
        nkstot=nkstot,
        nbnd=nbnd,
        nat=nat,
        nsp=nsp,
        nhm=nhm,
        lmaxkb=lmaxkb,
        nkb=nkb,
        nelec=nelec,
        ef=ef,
        ef_up=ef_up,
        ef_dw=ef_dw,
        degauss=degauss,
        ngauss=ngauss,
        ltetra=bool(ltetra),
        lgauss=bool(lgauss),
        at=at,
        bg=bg,
        alat=alat,
        omega=omega,
        ibrav=ibrav,
        atm=atm,
        ityp=ityp,
        tau=tau,
        nh=nh,
        tvanp=tvanp,
        tpawp=tpawp,
        deeq=deeq,
        dvan=dvan,
        qq_at=qq_at,
        gram=gram,
        xk=xk,
        wk=wk,
        isk=isk,
        et=et,
        wg=wg,
        coefficients=coefficients,
        becsum=becsum,
        rho_bec=rho_bec,
        dbeta_xc=dbeta_xc,
        ddd_paw=ddd_paw,
        ofsbeta=ofsbeta,
        hubbard=hubbard,
    )


# ---------------------------------------------------------------------------
# parse_qe_atomic_dump (atomic pseudo-wavefunction dump family)
# ---------------------------------------------------------------------------
def parse_qe_atomic_dump(path) -> QEAtomicDump:
    """Parse a QE atomic-wavefunction dump (``TB2JQEATWFC1.0``).

    Distinct physical-record schema selected at QE runtime with
    ``TB2J_PROJECTORS=atomic``: no ``deeq``/``qq_at``/``becsum`` records
    exist in this format; the per-k stream carries coefficient/Gram pairs
    and two trailing BZ-weighted R=0 onsite covariant vertex records
    carry the local spin vertex.

    Raises ``ValueError`` on unknown magic (including the KB dump family,
    which has its own parser), non-LSDA (``nspin != 2``) dumps,
    pool-parallel dumps (``nkstot != nks``), and any record-length/shape
    mismatch.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        reader = _RecordReader(path, fh)

        # -- record 0: magic + dimensions ------------------------------------
        # Manual header handling so a KB-family file (9 header dims) routed
        # here gets the friendly magic error instead of a layout error.
        payload = reader.read_raw("header")
        magic_raw = bytes(np.frombuffer(payload[:16], dtype="S16")[0])
        if magic_raw.rstrip(b" \x00") not in _ATOMIC_MAGICS:
            raise ValueError(
                f"QE atomic dump {path}: unknown magic {magic_raw!r}; expected "
                f"'{ATOMIC_MAGIC}' (select at QE runtime with "
                f"TB2J_PROJECTORS=atomic); KB becp dumps (TB2JQEDUMPV1*) are "
                f"read with parse_qe_dump/read_qe_dump"
            )
        if len(payload) != 16 + 8 * 4:
            raise ValueError(
                f"QE atomic dump {path}: header record is {len(payload)} "
                f"bytes, expected {16 + 8 * 4} (S16 magic + 8 i4 dims)"
            )
        nspin, nks, nkstot, nbnd, nat, nsp, nproj, max_nsite = (
            int(v) for v in np.frombuffer(payload[16:], dtype="<i4")
        )
        if nspin != 2:
            raise ValueError(
                f"QE atomic dump {path}: nspin={nspin}; TB2J requires a "
                f"collinear LSDA (nspin=2) calculation"
            )
        if nkstot != nks:
            raise ValueError(
                f"QE atomic dump {path}: nkstot={nkstot} != nks={nks}; "
                f"pool-parallel (-nk/-npools) dumps are not supported; rerun "
                f"with npool=1"
            )
        if min(nks, nbnd, nat, nsp, nproj, max_nsite) < 1:
            raise ValueError(
                f"QE atomic dump {path}: invalid dimensions nks={nks}, "
                f"nbnd={nbnd}, nat={nat}, nsp={nsp}, nproj={nproj}, "
                f"max_nsite={max_nsite}"
            )

        # -- record 1: Fermi energies and smearing ---------------------------
        nelec, ef, ef_up, ef_dw, degauss, ngauss, ltetra, lgauss = _read_fermi_record(
            reader
        )

        # -- record 2: cell ---------------------------------------------------
        at, bg, alat, omega, ibrav = _read_cell_record(reader)

        # -- record 3: species labels ----------------------------------------
        # Same QE >= 8 fixed-capacity CHARACTER(LEN=6) record as the KB dumps.
        atm = _decode_species_labels(reader.read_raw("species labels"), nsp, path)

        # -- record 4: ions ---------------------------------------------------
        ityp, tau = _read_ions_record(reader, nat, nsp, path)

        # -- record 5: channel metadata ---------------------------------------
        chunks = reader.split_record(
            "channel metadata",
            [
                ("nsite_species", "<i4", nsp),
                ("projector_l", "<i4", nproj),
                ("m_ordinal", "<i4", nproj),
                ("radial_upf_index", "<i4", nproj),
            ],
        )
        nsite_species = np.array(chunks["nsite_species"], dtype=int)
        projector_l = np.array(chunks["projector_l"], dtype=int)
        m_ordinal = np.array(chunks["m_ordinal"], dtype=int)
        radial_upf_index = np.array(chunks["radial_upf_index"], dtype=int)
        if nsite_species.min() < 1:
            raise ValueError(
                f"QE atomic dump {path}: nonpositive nsite_species entries "
                f"{nsite_species.tolist()}; every UPF species needs PP_PSWFC "
                "pseudo-atomic wavefunctions"
            )
        if nsite_species.max() > max_nsite:
            raise ValueError(
                f"QE atomic dump {path}: nsite_species max "
                f"{int(nsite_species.max())} exceeds max_nsite={max_nsite}"
            )
        if projector_l.min() < 0 or m_ordinal.min() < 1 or radial_upf_index.min() < 0:
            raise ValueError(
                f"QE atomic dump {path}: invalid channel metadata; require "
                f"projector_l >= 0, m_ordinal >= 1, radial_upf_index >= 0 "
                f"(got l min {int(projector_l.min())}, m min "
                f"{int(m_ordinal.min())}, radial min "
                f"{int(radial_upf_index.min())})"
            )
        for l, m in zip(projector_l.tolist(), m_ordinal.tolist()):
            if m > 2 * l + 1:
                raise ValueError(
                    f"QE atomic dump {path}: channel (l={l}, m_ordinal={m}) "
                    f"exceeds the 2l+1 real-harmonic ordinals"
                )
        nsite_expected = int(nsite_species[ityp - 1].sum())
        if nsite_expected != nproj:
            raise ValueError(
                f"QE atomic dump {path}: nproj={nproj} inconsistent with the "
                f"per-atom channel count sum(nsite_species[ityp])="
                f"{nsite_expected}"
            )

        # -- record 6: k-point list -------------------------------------------
        xk, wk, isk = _read_kmesh_record(reader, nks, path)

        # -- record 7: bands ---------------------------------------------------
        et, wg = _read_spectral_record(reader, nbnd, nks)

        # -- records 8+2*ik / 9+2*ik: per-k coefficient + Gram pairs ----------
        # The stream interleaves one coefficient record with its matching
        # Gram record per k-point (contract records 8+2*ik and 9+2*ik).
        coefficients = []
        grams = []
        for ik in range(nks):
            coefficients.append(
                reader.array(f"atomic wfc k-point {ik + 1}", "<c16", (nproj, nbnd))
            )
            grams.append(
                reader.array(
                    f"atomic wfc Gram k-point {ik + 1}", "<c16", (nproj, nproj)
                )
            )

        # -- records 8+2*nks / 9+2*nks: BZ-weighted R=0 onsite vertex ---------
        cov_xc = reader.array("cov_xc", "<c16", (max_nsite, max_nsite, nat))
        cov_aug = reader.array("cov_aug", "<c16", (max_nsite, max_nsite, nat))
        hubbard = QEHubbardData()
        if magic_raw.rstrip(b" \x00") == MAGIC_ATOMIC_HUBBARD:
            hubbard = _parse_qe_hubbard(
                reader,
                dict(
                    nspin=nspin,
                    nks=nks,
                    nbnd=nbnd,
                    nat=nat,
                    nsp=nsp,
                    ityp=ityp,
                    atm=atm,
                    xk=xk,
                    wk=wk,
                    isk=isk,
                    et=et,
                    wg=wg,
                ),
                "atomic",
            )
        if not reader.at_eof():
            raise ValueError(
                f"QE atomic dump {path}: unexpected trailing data after the "
                f"last expected record ({reader.remaining_bytes()} bytes)"
            )

    ofswfc = np.zeros(nat, dtype=int)
    if nat > 1:
        ofswfc[1:] = np.cumsum(nsite_species[ityp - 1])[:-1]

    return QEAtomicDump(
        path=path,
        magic=magic_raw.rstrip().decode("ascii"),
        nspin=nspin,
        nks=nks,
        nkstot=nkstot,
        nbnd=nbnd,
        nat=nat,
        nsp=nsp,
        nproj=nproj,
        max_nsite=max_nsite,
        nelec=nelec,
        ef=ef,
        ef_up=ef_up,
        ef_dw=ef_dw,
        degauss=degauss,
        ngauss=ngauss,
        ltetra=bool(ltetra),
        lgauss=bool(lgauss),
        at=at,
        bg=bg,
        alat=alat,
        omega=omega,
        ibrav=ibrav,
        atm=atm,
        ityp=ityp,
        tau=tau,
        nsite_species=nsite_species,
        projector_l=projector_l,
        m_ordinal=m_ordinal,
        radial_upf_index=radial_upf_index,
        xk=xk,
        wk=wk,
        isk=isk,
        et=et,
        wg=wg,
        coefficients=coefficients,
        grams=grams,
        cov_xc=cov_xc,
        cov_aug=cov_aug,
        ofswfc=ofswfc,
        hubbard=hubbard,
    )


# ---------------------------------------------------------------------------
# read_qe_dump: ProjectorGreenData normalization (KB/beta family)
# ---------------------------------------------------------------------------
def _match_spin_kmesh(xk_up: np.ndarray, xk_dn: np.ndarray, path: Path) -> np.ndarray:
    """Map each up k-point to the matching down k-point (cartesian 2pi/alat).

    Returns ``perm`` with ``xk_dn[perm[j]] == xk_up[j]`` within tolerance.
    """
    n = xk_up.shape[0]
    perm = np.empty(n, dtype=int)
    for start in range(0, n, 1024):
        block = xk_up[start : start + 1024]
        dist2 = ((block[:, None, :] - xk_dn[None, :, :]) ** 2).sum(axis=-1)
        j = dist2.argmin(axis=1)
        best = dist2[np.arange(block.shape[0]), j]
        if np.any(best > _KPOINT_MATCH_TOL**2):
            raise ValueError(
                f"QE dump {path}: up- and down-spin k meshes differ beyond "
                f"{_KPOINT_MATCH_TOL} (2pi/alat units); a spin-degenerate "
                f"dense mesh is required for the projector Green data"
            )
        perm[start : start + 1024] = j
    if len(np.unique(perm)) != n:
        raise ValueError(
            f"QE dump {path}: up/down k-point matching is not one-to-one; "
            f"a spin-degenerate dense mesh is required"
        )
    return perm


def _band_occupations(wg: np.ndarray, wk: np.ndarray) -> np.ndarray:
    """Convert ``wg`` (which includes the k-weight) to per-band occupations."""
    denom = np.asarray(wk, dtype=float)[:, None]
    out = np.zeros_like(wg)
    np.divide(wg, denom, out=out, where=denom > 0)
    return out


def _atomic_numbers_for_labels(labels) -> np.ndarray:
    """Map QE species labels (e.g. ``'Fe1'``) to atomic numbers."""
    from TB2J.interfaces.abinit_paw import _atomic_numbers_from_symbols

    symbols = []
    for label in labels:
        match = re.match(r"[A-Za-z]{1,2}", label)
        if match is None:
            raise ValueError(
                f"cannot parse an element symbol from QE species label {label!r}"
            )
        symbol = match.group(0)
        symbols.append(symbol[:1].upper() + symbol[1:].lower())
    try:
        return _atomic_numbers_from_symbols(symbols)
    except KeyError as error:
        raise ValueError(
            f"unknown element symbol derived from QE species labels "
            f"{list(labels)}: {error}"
        ) from error


def _fold_spin_mesh(dump, path: Path):
    """Shared isk folding: dense mesh indices, permutation and weights.

    Returns ``(up, dn, perm, nk, kpoints, weights)`` with fractional
    k-points for the up half and symmetrized weights.
    """
    up = np.flatnonzero(dump.isk == 1)
    dn = np.flatnonzero(dump.isk == 2)
    if up.size == 0 or dn.size == 0:
        raise ValueError(
            f"QE dump {path}: both spin channels must be present in isk; "
            f"found {up.size} up and {dn.size} down k-points"
        )
    if up.size != dn.size:
        raise ValueError(
            f"QE dump {path}: spin-degenerate k mesh required; found "
            f"{up.size} up and {dn.size} down k-points"
        )
    perm = _match_spin_kmesh(dump.xk[up], dump.xk[dn], path)
    nk = up.size

    # fractional k: f = at^T xk (QE bg = 2pi inv(at)^T in alat units)
    kpoints = dump.xk[up] @ dump.at
    weights = dump.wk[up] + dump.wk[dn[perm]]
    if np.any(weights < 0):
        raise ValueError(f"QE dump {path}: negative k-point weights")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"QE dump {path}: k-point weights sum to {total}")
    weights = weights / total
    return up, dn, perm, nk, kpoints, weights


def _fermi_pair(dump) -> tuple:
    """Shared Fermi-pair resolution: (efermi, efermi_spin, efermi_source)."""
    if dump.ef_up != 0.0 and dump.ef_dw != 0.0:
        efermi_spin = np.array([dump.ef_up, dump.ef_dw]) * RYTOEV
        efermi_source = "qe ef_up/ef_dw (two Fermi energies)"
    else:
        efermi_spin = None
        efermi_source = "qe ef (single Fermi energy)"
    return dump.ef * RYTOEV, efermi_spin, efermi_source


def read_qe_dump(path, *, legacy_no_u=False) -> ProjectorGreenData:
    """Read a QE ``becp_dump`` file into exchange-ready projector Green data.

    Normalizes dumped units (Ry -> eV, alat/Bohr -> Angstrom, cartesian k in
    2pi/alat -> fractional), folds the ``isk``-split k list into the dense
    spin-degenerate mesh expected by :class:`ProjectorGreenData`, and splits
    the concatenated beta channels per atom (QE ``ofsbeta`` order).
    """
    from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM

    dump = parse_qe_dump(path)
    hubbard_metadata = _qe_hubbard_admission(dump, legacy_no_u)
    _hub_require(dump.nspin == 2, "unsupported", "collinear exchange requires nspin=2")

    # -- dense spin-degenerate k mesh from isk --------------------------------
    up, dn, perm, nk, kpoints, weights = _fold_spin_mesh(dump, dump.path)

    eigenvalues = np.empty((2, nk, dump.nbnd), dtype=float)
    eigenvalues[0] = dump.et[up] * RYTOEV
    eigenvalues[1] = dump.et[dn[perm]] * RYTOEV
    occupations = np.empty_like(eigenvalues)
    occupations[0] = _band_occupations(dump.wg[up], dump.wk[up])
    occupations[1] = _band_occupations(dump.wg[dn[perm]], dump.wk[dn[perm]])

    coefficients = np.empty((2, nk, dump.nbnd, dump.nkb), dtype=complex)
    for j, ik in enumerate(up):
        coefficients[0, j] = dump.coefficients[ik].T
    for j, ik in enumerate(dn[perm]):
        coefficients[1, j] = dump.coefficients[ik].T

    efermi, efermi_spin, efermi_source = _fermi_pair(dump)

    # -- structure -------------------------------------------------------------
    cell = dump.at.T * dump.alat * BOHR_TO_ANGSTROM
    positions = dump.tau * dump.alat * BOHR_TO_ANGSTROM
    atomic_numbers = _atomic_numbers_for_labels(dump.atm)[dump.ityp - 1]

    # -- projector channels: atom na owns ofsbeta(na)..+nh(ityp[na]) -----------
    site_nproj = dump.nh_per_atom.astype(int)
    natom = dump.nat
    nmax = int(site_nproj.max())
    site_projector_indices = -np.ones((natom, nmax), dtype=int)
    projector_site = np.repeat(np.arange(natom), site_nproj)
    for site, count in enumerate(site_nproj):
        start = int(dump.ofsbeta[site])
        site_projector_indices[site, :count] = np.arange(start, start + count)

    # -- separable operator: deeq per spin, eV ---------------------------------
    hij = (
        np.stack(
            [
                dump.deeq[:, :, :, 0].transpose(2, 0, 1),
                dump.deeq[:, :, :, 1].transpose(2, 0, 1),
            ]
        ).astype(complex)
        * RYTOEV
    )
    delta_total = hij[0] - hij[1]
    if dump.dbeta_xc is not None:
        # v1.2: the LKAG site vertex combines the beta-projected spin xc
        # potential with the augmentation-channel spin vertex inside deeq.
        # dbeta_xc = <beta_i|Delta V_xc|beta_j> is a PRIMAL matrix element,
        # so pairing it with the undressed G_beta requires the joint metric
        # transformation M^-1 dbeta M^-1 (the deeq term is already a
        # beta-representation operator and enters unchanged). G3 on bccFe
        # falsified the deeq-only vertex (J1 ~2.6x low); the projected
        # Delta V_xc restores the missing local splitting.
        for site in range(dump.nat):
            count = int(dump.nh_per_atom[site])
            if count == 0:
                continue
            gram_block = np.asarray(dump.gram[:count, :count, site], dtype=complex)
            gram_inv = np.linalg.inv(gram_block)
            dbeta_block = dump.dbeta_xc[:count, :count, site] * RYTOEV
            delta_total[site, :count, :count] += gram_inv @ dbeta_block @ gram_inv
        hij_definition = "qe_dbeta_xc_plus_deeq_spin_difference"
        delta_definition = (
            "QE beta-projected xc spin vertex <beta|V_xc^up - V_xc^dn|beta> "
            "conjugated by the beta Gram (M^-1 dbeta M^-1, joint metric "
            "transformation) plus the deeq_up - deeq_dn augmentation-channel "
            "spin vertex (which already contains the PAW ddd_paw splitting)"
        )
        delta_completeness = "complete"
    else:
        hij_definition = "qe_deeq_spin_difference"
        delta_definition = (
            "QE deeq spin difference (deeq_up - deeq_dn), the covariant "
            "separable beta spin vertex of V_NL = beta D beta^dagger. "
            "FALSIFIED as an exchange vertex by the bccFe G3 gate "
            "(J1 ~2.6x low): deeq carries only the augmentation-channel "
            "spin term. Provide a v1.2 dump (dbeta_xc record) for exchange."
        )
        delta_completeness = "partial_augmentation_channel_only_falsified"
    operator_component_metadata = {
        "delta_total": {
            "units": "eV",
            "input_unit": "Ry",
            "definition": delta_definition,
            "source": HIJ_SOURCE,
            "operator_basis": OPERATOR_BASIS,
            "completeness": delta_completeness,
            "exchange_ready": ("true" if dump.dbeta_xc is not None else "false"),
        }
    }

    metadata = {
        **hubbard_metadata,
        "source": ("Quantum ESPRESSO TB2J_DUMP projector dump (PW/src/becp_dump.f90)"),
        "source_code": "qe",
        "dump_version": dump.version,
        "qe_magic": dump.magic,
        "qe_npool_ok": dump.nkstot == dump.nks,
        "becsum_present": dump.becsum is not None,
        "rho_bec_present": dump.rho_bec is not None,
        "tvanp": [bool(v) for v in dump.tvanp],
        "tpawp": [bool(v) for v in dump.tpawp],
        "projector_basis_type": "ultrasoft+PAW beta",
        "coefficient_convention": "dual_projector_no_inverse",
        "alat_bohr": dump.alat,
        "omega_bohr3": dump.omega,
        "ibrav": dump.ibrav,
        "nelec": dump.nelec,
        "degauss_ry": dump.degauss,
        "ngauss": dump.ngauss,
        "ltetra": dump.ltetra,
        "lgauss": dump.lgauss,
        "nkstot": dump.nkstot,
        "nks": dump.nks,
        "nbnd": dump.nbnd,
        "nat": dump.nat,
        "nsp": dump.nsp,
        "nhm": dump.nhm,
        "lmaxkb": dump.lmaxkb,
        "nkb": dump.nkb,
        "atm": list(dump.atm),
        "ityp": dump.ityp.tolist(),
        "nh": dump.nh.tolist(),
        "ef_ry": dump.ef,
        "efermi_source": efermi_source,
        "qq_at_note": (
            "dump records 8 (qq_at) and 9 (beta Gram) are diagnostics only; "
            "they are never used as a channel metric"
        ),
        "becsum_note": (
            "v1.1 becsum: scf dumps hold converged sum_band occupations; "
            "nscf dumps hold hinit1 scf-mesh restart occupations, not "
            "dense-mesh weights"
            + (
                "; NC-only runs: becsum is zero-filled (no augmentation "
                "charges) and is not an occupation-parity reference"
                if not (dump.tvanp.any() or dump.tpawp.any())
                else ""
            )
        ),
    }

    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
        efermi=efermi,
        efermi_spin=efermi_spin,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        overlap_k=None,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        hij=hij,
        hij_definition=hij_definition,
        hij_units=HIJ_UNITS,
        hij_source=HIJ_SOURCE,
        hij_projection=HIJ_PROJECTION,
        operator_components={"delta_total": delta_total},
        operator_component_metadata=operator_component_metadata,
        coefficient_source=COEFFICIENT_SOURCE,
        coefficient_projector=COEFFICIENT_PROJECTOR,
        channel_interpretation=CHANNEL_INTERPRETATION,
        operator_basis=OPERATOR_BASIS,
        population_metric=(
            "QE becsum packed projector occupations (scf dumps: converged "
            "sum_band; nscf dumps: hinit1 scf-mesh restart occupations)"
        ),
        metadata=metadata,
    )
    data.validate(exchange_ready=True)
    return data


# ---------------------------------------------------------------------------
# read_qe_atomic_dump: ProjectorGreenData normalization (atomic family)
# ---------------------------------------------------------------------------
def read_qe_atomic_dump(path, *, legacy_no_u=False) -> ProjectorGreenData:
    """Read a QE atomic-wavefunction dump into projector Green data.

    The UPF pseudo-atomic coefficients ``C = <phi|psi>`` from QE
    ``atomic_wfc`` are PRIMAL in the k-dependent plane-wave basis, so the
    full per-k orbital overlap ``M(k) = <phi|phi>`` is exported as
    ``overlap_k`` (shared by both spins); the shared runtime dresses with
    the k-dependent ``M^-1``. The BZ-weighted R=0 onsite covariant site
    vertex ``Delta = sum_{k:isk==1} wk [<phi(k)|Vxc_up - Vxc_dn|phi(k)> +
    B(k) (deeq_up - deeq_dn) B(k)^dagger] / sum_{k:isk==1} wk`` (Ry -> eV)
    is exported as the site-local ``delta_total``
    with ``hij[up] = +Delta/2`` and ``hij[dn] = -Delta/2``; it is primal
    and is never preconjugated by ``M^-1`` at the reader.
    """
    from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM

    dump = parse_qe_atomic_dump(path)
    hubbard_metadata = _qe_hubbard_admission(dump, legacy_no_u)

    # -- dense spin-degenerate k mesh from isk --------------------------------
    up, dn, perm, nk, kpoints, weights = _fold_spin_mesh(dump, dump.path)

    eigenvalues = np.empty((2, nk, dump.nbnd), dtype=float)
    eigenvalues[0] = dump.et[up] * RYTOEV
    eigenvalues[1] = dump.et[dn[perm]] * RYTOEV
    occupations = np.empty_like(eigenvalues)
    occupations[0] = _band_occupations(dump.wg[up], dump.wk[up])
    occupations[1] = _band_occupations(dump.wg[dn[perm]], dump.wk[dn[perm]])

    # primal atomic coefficients, spin-folded by isk; full k-dependent Gram M(k)
    coefficients = np.empty((2, nk, dump.nbnd, dump.nproj), dtype=complex)
    overlap_k = np.empty((nk, dump.nproj, dump.nproj), dtype=complex)
    for j, ik in enumerate(up):
        coefficients[0, j] = dump.coefficients[ik].T
        overlap_k[j] = dump.grams[ik]
    for j, ik in enumerate(dn[perm]):
        coefficients[1, j] = dump.coefficients[ik].T

    efermi, efermi_spin, efermi_source = _fermi_pair(dump)

    # -- structure -------------------------------------------------------------
    cell = dump.at.T * dump.alat * BOHR_TO_ANGSTROM
    positions = dump.tau * dump.alat * BOHR_TO_ANGSTROM
    atomic_numbers = _atomic_numbers_for_labels(dump.atm)[dump.ityp - 1]

    # -- projector channels: atom na owns ofswfc(na)..+nsite_species(ityp[na]) -
    site_nproj = dump.nsite_per_atom.astype(int)
    natom = dump.nat
    nmax = int(site_nproj.max())
    site_projector_indices = -np.ones((natom, nmax), dtype=int)
    projector_site = np.repeat(np.arange(natom), site_nproj)
    for site, count in enumerate(site_nproj):
        start = int(dump.ofswfc[site])
        site_projector_indices[site, :count] = np.arange(start, start + count)

    # -- covariant site-local spin vertex (Ry -> eV), hij = ±Delta/2 ----------
    delta_total = (dump.cov_xc + dump.cov_aug).transpose(2, 0, 1) * RYTOEV
    hij = np.stack([delta_total / 2.0, -delta_total / 2.0])
    delta_definition = (
        "QE atomic-wavefunction BZ-weighted R=0 onsite covariant site "
        "vertex Delta = sum_{k:isk==1} wk [<phi(k)|V_xc^up - V_xc^dn|phi(k)> + "
        "B(k) (deeq_up - deeq_dn) B(k)^dagger] / sum_{k:isk==1} wk, "
        "B(k) = <phi(k)|beta(k)> (cov_xc + cov_aug records, Ry). Primal atomic "
        "representation; the shared runtime applies the k-dependent "
        "M(k)^-1 dressing (no reader-side preconjugation)."
    )
    operator_component_metadata = {
        "delta_total": {
            "units": "eV",
            "input_unit": "Ry",
            "definition": delta_definition,
            "source": (
                "QE atomic_wfc BZ-weighted R=0 onsite covariant vertex "
                "records cov_xc + cov_aug "
                "(PW/src/becp_dump.f90, TB2J_PROJECTORS=atomic)"
            ),
            "operator_basis": ATOMIC_OPERATOR_BASIS,
            "completeness": "complete",
            "exchange_ready": "true",
            "m_ordinal_note": ATOMIC_M_ORDINAL_NOTE,
            "overlap_dressing_note": ATOMIC_VERTEX_NOTE,
        }
    }

    metadata = {
        **hubbard_metadata,
        "source": (
            "Quantum ESPRESSO TB2J_DUMP atomic-wavefunction projector dump "
            "(PW/src/becp_dump.f90, TB2J_PROJECTORS=atomic)"
        ),
        "source_code": "qe",
        "dump_version": _ATOMIC_MAGICS[dump.magic.encode("ascii")],
        "qe_magic": dump.magic,
        "qe_npool_ok": dump.nkstot == dump.nks,
        "projector_source_mode": "qe_atomic_wfc",
        "basis_validation": (
            "unvalidated on bccFe: matched-UPF atomic J1 differs substantially "
            "from KB for NC, US and PAW; use for controlled basis comparisons"
        ),
        "projector_basis_type": "UPF PP_PSWFC pseudo-atomic orbitals (atomic_wfc)",
        "coefficient_convention": "primal_pseudo_atomic_with_overlap",
        "alat_bohr": dump.alat,
        "omega_bohr3": dump.omega,
        "ibrav": dump.ibrav,
        "nelec": dump.nelec,
        "degauss_ry": dump.degauss,
        "ngauss": dump.ngauss,
        "ltetra": dump.ltetra,
        "lgauss": dump.lgauss,
        "nkstot": dump.nkstot,
        "nks": dump.nks,
        "nbnd": dump.nbnd,
        "nat": dump.nat,
        "nsp": dump.nsp,
        "nproj": dump.nproj,
        "max_nsite": dump.max_nsite,
        "atm": list(dump.atm),
        "ityp": dump.ityp.tolist(),
        "nsite_species": dump.nsite_species.tolist(),
        "ef_ry": dump.ef,
        "efermi_source": efermi_source,
        "m_ordinal_note": ATOMIC_M_ORDINAL_NOTE,
        "vertex_note": ATOMIC_VERTEX_NOTE,
        "channel_order": (
            "collinear atomic_wfc order: atom, radial UPF orbital "
            "(radial_upf_index = UPF nwfc index nb, 1-based as exported by "
            "QE), QE real-harmonic ordinal m=1..2l+1"
        ),
        "overlap_note": (
            "overlap_k = M(k) = <atomic_wfc|atomic_wfc> per k (Gram record "
            "9+2*ik), full orbital basis, shared by both spins; runtime "
            "dresses with M(k)^-1 (CLI --overlap_mode/--overlap_rcond)"
        ),
    }

    data = ProjectorGreenData(
        kpoints=kpoints,
        weights=weights,
        eigenvalues=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
        efermi=efermi,
        efermi_spin=efermi_spin,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=cell,
        positions=positions,
        atomic_numbers=atomic_numbers,
        overlap_k=overlap_k,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        projector_l=dump.projector_l,
        projector_m=dump.m_ordinal,
        projector_radial=dump.radial_upf_index,
        hij=hij,
        hij_definition=ATOMIC_HIJ_DEFINITION,
        hij_units=HIJ_UNITS,
        hij_source=(
            "QE atomic_wfc cov_xc + cov_aug vertex records "
            "(PW/src/becp_dump.f90, TB2J_PROJECTORS=atomic)"
        ),
        hij_projection=("UPF PP_PSWFC pseudo-atomic orbital channel (primal basis)"),
        operator_components={"delta_total": delta_total},
        operator_component_metadata=operator_component_metadata,
        coefficient_source=ATOMIC_COEFFICIENT_SOURCE,
        coefficient_projector=ATOMIC_COEFFICIENT_PROJECTOR,
        channel_interpretation=ATOMIC_CHANNEL_INTERPRETATION,
        operator_basis=ATOMIC_OPERATOR_BASIS,
        metadata=metadata,
    )
    data.validate(exchange_ready=True)
    return data


@dataclass
class QESpinorDump:
    """Native KB spinor records; energies/operators remain in Ry."""

    path: Path
    magic: str
    version: str
    nspin: int
    nks: int
    nkstot: int
    nbnd: int
    nat: int
    nsp: int
    nhm: int
    lmaxkb: int
    nkb: int
    lspinorb: bool
    domag: bool
    nelec: float
    ef: float
    degauss: float
    ngauss: int
    ltetra: int
    lgauss: int
    at: np.ndarray
    bg: np.ndarray
    alat: float
    omega: float
    ibrav: int
    atm: list
    ityp: np.ndarray
    tau: np.ndarray
    nh: np.ndarray
    tvanp: np.ndarray
    tpawp: np.ndarray
    has_so: np.ndarray
    deeq_nc_total: np.ndarray
    deeq_nc_magnetic: np.ndarray
    gram: np.ndarray
    xk: np.ndarray
    wk: np.ndarray
    et: np.ndarray
    wg: np.ndarray
    coefficients: np.ndarray
    projected_field: np.ndarray
    becsum: np.ndarray
    d_response: np.ndarray | None = None
    hubbard: QEHubbardData = field(default_factory=QEHubbardData)

    @property
    def nh_per_atom(self):
        return self.nh[self.ityp - 1]

    @property
    def ofsbeta(self):
        return np.r_[0, np.cumsum(self.nh_per_atom[:-1])]


def _require_qe_hermitian(matrix, what, path):
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-9, atol=1e-10):
        raise ValueError(f"QE spinor dump {path}: {what} is not complex Hermitian")


def _qe_native_spinor(block):
    """Native (uu,ud,du,dd) to channel/channel/output-spin/input-spin."""
    return block.reshape(block.shape[:2] + (2, 2))


def _validate_qe_spinor_mesh(kpoints, weights, path):
    """Reject reduced/nonuniform meshes; retain QE's k ordering."""
    if np.any(weights <= 0) or not np.allclose(
        weights, 1.0 / len(weights), rtol=1e-7, atol=1e-10
    ):
        raise ValueError(
            f"QE spinor dump {path}: full-BZ spinor weights must be positive, "
            "equal and sum to one; symmetry-reduced meshes are unsupported"
        )
    fractional = np.mod(kpoints, 1.0)
    fractional[np.isclose(fractional, 1.0, atol=_KPOINT_MATCH_TOL, rtol=0)] = 0.0
    labels = np.empty_like(fractional, dtype=int)
    shape = []
    for axis in range(3):
        values = np.sort(fractional[:, axis])
        unique = values[np.r_[True, np.diff(values) > _KPOINT_MATCH_TOL]]
        gaps = np.diff(np.r_[unique, unique[0] + 1.0])
        if not np.allclose(gaps, 1.0 / len(unique), atol=_KPOINT_MATCH_TOL, rtol=0):
            raise ValueError(
                f"QE spinor dump {path}: requires a complete uniform full-BZ grid"
            )
        labels[:, axis] = np.argmin(
            np.abs(fractional[:, axis, None] - unique[None, :]), axis=1
        )
        shape.append(len(unique))
    if np.prod(shape) != len(weights) or len(np.unique(labels, axis=0)) != len(weights):
        raise ValueError(
            f"QE spinor dump {path}: duplicate or incomplete full-BZ k mesh"
        )


def parse_qe_spinor_dump(path) -> QESpinorDump:
    """Strictly parse the KB-only NC spinor ABI (NC1.0, NC1.1 or NC1.2).

    NC1.0 ends after the reserved-zero ``becsum`` record; NC1.1 appends
    record ``13+nks``: the nine native source-response blocks
    ``D_response(nhm, nhm, nat, 4, 3, 3)`` in Fortran order
    ``(ih, jh, na, spinblock, c, j)`` (``c`` = target Pauli slot,
    ``j`` = original source axis), used to build the explicit
    frozen-source tangent vertices. All versions require finite values,
    complex-Hermitian site blocks and exact EOF; NC1.1 and NC1.2 additionally
    enforces the diagonal identity gate
    ``sum_j D_response(:,:,:,:,j,j) == deeq_nc_magnetic`` at 1e-8
    relative tolerance per site.
    NC1.2 then requires the complete typed native Hubbard extension.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        reader = _RecordReader(path, fh)
        header = reader.read_raw("spinor header")
        version = _SPINOR_MAGICS.get(bytes(header[:16]).rstrip(b" "))
        if version is None:
            raise ValueError(
                f"QE spinor dump {path}: unknown magic/version {header[:16]!r}; "
                f"expected {SPINOR_MAGIC} or {SPINOR_MAGIC_NC11} "
                "(KB only; atomic+noncolin unsupported)"
            )
        if len(header) != 60:
            raise ValueError(f"QE spinor dump {path}: header must be S16 + 11 i4 dims")
        names = (
            "nspin",
            "nks",
            "nkstot",
            "nbnd",
            "nat",
            "nsp",
            "nhm",
            "lmaxkb",
            "nkb",
            "lspinorb",
            "domag",
        )
        dims = dict(zip(names, map(int, np.frombuffer(header[16:], dtype="<i4"))))
        if dims["nspin"] != 4:
            raise ValueError(f"QE spinor dump {path}: requires nspin=4")
        if dims["nkstot"] != dims["nks"]:
            raise ValueError(
                f"QE spinor dump {path}: pool-parallel dumps require npool=1"
            )
        if (
            min(dims[n] for n in ("nks", "nbnd", "nat", "nsp", "nhm", "nkb")) < 1
            or dims["lmaxkb"] < 0
            or dims["nsp"] > dims["nat"]
        ):
            raise ValueError(f"QE spinor dump {path}: invalid dimensions/site counts")
        if dims["lspinorb"] not in (0, 1) or dims["domag"] != 1:
            raise ValueError(
                f"QE spinor dump {path}: requires domag=true and valid SOC flag"
            )
        nks, nbnd, nat, nsp, nhm, nkb = (
            dims[n] for n in ("nks", "nbnd", "nat", "nsp", "nhm", "nkb")
        )
        fermi = reader.split_record(
            "spinor fermi/smearing", [("energies", "<f8", 3), ("flags", "<i4", 3)]
        )
        nelec, ef, degauss = map(float, fermi["energies"])
        ngauss, ltetra, lgauss = map(int, fermi["flags"])
        if ltetra not in (0, 1) or lgauss not in (0, 1):
            raise ValueError(f"QE spinor dump {path}: invalid smearing logical flags")
        at, bg, alat, omega, ibrav = _read_cell_record(reader)
        labels = reader.read_raw("species labels")
        # QE Modules/ions_base.f90:47 declares CHARACTER(LEN=6) atm(ntypx);
        # the producer writes the whole array, so unused entries beyond nsp
        # carry undefined memory and are not validated.
        if len(labels) < 6 * nsp or len(labels) % 6 or not labels[: 6 * nsp].isascii():
            raise ValueError(
                f"QE spinor dump {path}: invalid fixed-capacity S6 species record"
            )
        atm = [labels[i * 6 : (i + 1) * 6].decode("ascii").strip() for i in range(nsp)]
        if not all(atm):
            raise ValueError(f"QE spinor dump {path}: blank species label")
        ityp, tau = _read_ions_record(reader, nat, nsp, path)
        if len(np.unique(ityp)) != nsp:
            raise ValueError(f"QE spinor dump {path}: species/site counts disagree")
        meta = reader.split_record(
            "spinor projector metadata",
            [(name, "<i4", nsp) for name in ("nh", "tvanp", "tpawp", "has_so")],
        )
        nh = np.array(meta["nh"], dtype=int)
        minimum_nh = 0 if version == "1.2" else 1
        if (
            np.any(nh < minimum_nh)
            or int(nh.max()) != nhm
            or int(nh[ityp - 1].sum()) != nkb
        ):
            raise ValueError(f"QE spinor dump {path}: nh/nhm/nkb site counts disagree")
        for name in ("tvanp", "tpawp", "has_so"):
            if np.any((meta[name] != 0) & (meta[name] != 1)):
                raise ValueError(f"QE spinor dump {path}: invalid {name} flags")
        tvanp, tpawp, has_so = (
            np.array(meta[n], dtype=bool) for n in ("tvanp", "tpawp", "has_so")
        )
        if dims["lspinorb"] and not has_so.any():
            raise ValueError(
                f"QE spinor dump {path}: SOC-on requires an SO-bearing UPF"
            )
        if version == "1.0" and np.any(~(tvanp | tpawp)):
            raise ValueError(
                f"QE spinor dump {path}: NC1.0 cannot certify a norm-conserving "
                "smooth-only source without NC1.1's nine measured responses"
            )
        total = reader.array("deeq_nc_total", "<c16", (nhm, nhm, nat, 4))
        magnetic = reader.array("deeq_nc_magnetic", "<c16", (nhm, nhm, nat, 4))
        gram = reader.array("beta Gram", "<c16", (nhm, nhm, nat))
        mesh = reader.split_record(
            "spinor k data", [("xk", "<f8", 3 * nks), ("wk", "<f8", nks)]
        )
        xk = np.array(mesh["xk"].reshape((3, nks), order="F").T)
        wk = np.array(mesh["wk"])
        et, wg = _read_spectral_record(reader, nbnd, nks)
        coefficients = np.empty((nks, nkb, 2, nbnd), dtype=complex)
        for ik in range(nks):
            coefficients[ik] = reader.array(
                f"becp_nc k={ik + 1}", "<c16", (nkb, 2, nbnd)
            )
        field = reader.array("projected Pauli field", "<c16", (nhm, nhm, nat, 3))
        becsum = reader.array("spinor becsum", "<f8", (nhm * (nhm + 1) // 2, nat, 4))
        d_response = None
        if version in ("1.1", "1.2"):
            d_response = reader.array(
                "source response D", "<c16", (nhm, nhm, nat, 4, 3, 3)
            )
        hubbard = QEHubbardData()
        if version == "1.2":
            hubbard = _parse_qe_hubbard(
                reader,
                dict(
                    **dims,
                    ityp=ityp,
                    atm=atm,
                    nh=nh,
                    tvanp=tvanp,
                    tpawp=tpawp,
                    has_so=has_so,
                    gram=gram,
                    field=field,
                    magnetic=magnetic,
                    total=total,
                    xk=xk,
                    wk=wk,
                    et=et,
                    wg=wg,
                ),
                "spinor",
            )
        if not reader.at_eof():
            raise ValueError(
                f"QE spinor dump {path}: trailing records beyond NC{version} schema"
            )

    for name, value in (
        ("fermi", [nelec, ef, degauss]),
        ("cell", at),
        ("reciprocal cell", bg),
        ("cell scalars", [alat, omega]),
        ("ions", tau),
        ("kpoints", xk),
        ("weights", wk),
        ("bands", et),
        ("occupations", wg),
        ("becp_nc", coefficients),
        ("deeq_nc_total", total),
        ("deeq_nc_magnetic", magnetic),
        ("Gram", gram),
        ("projected Pauli field", field),
        ("becsum", becsum),
    ) + ((("source response D", d_response),) if d_response is not None else ()):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"QE spinor dump {path}: nonfinite {name}")
    if (
        alat <= 0
        or omega <= 0
        or nelec < 0
        or degauss < 0
        or not np.allclose(at.T @ bg, np.eye(3), atol=1e-7, rtol=1e-7)
        or not np.isclose(abs(np.linalg.det(at)) * alat**3, omega, rtol=1e-6)
    ):
        raise ValueError(f"QE spinor dump {path}: invalid cell or physical scalars")
    _validate_qe_spinor_mesh(xk @ at, wk, path)
    magnetic_source_present = False
    for site, count in enumerate(nh[ityp - 1]):
        for name, array in (("deeq_nc_total", total), ("deeq_nc_magnetic", magnetic)):
            block = _qe_native_spinor(array[:, :, site])
            _require_qe_hermitian(
                block.transpose(0, 2, 1, 3).reshape(2 * nhm, 2 * nhm), name, path
            )
        for axis in range(3):
            _require_qe_hermitian(
                field[:, :, site, axis], "projected Pauli field", path
            )
        _require_qe_hermitian(gram[:, :, site], "beta Gram", path)
        if (
            count
            and version != "1.2"
            and np.linalg.eigvalsh(gram[:count, :count, site]).min() <= 0
        ):
            raise ValueError(
                f"QE spinor dump {path}: beta Gram must be positive definite"
            )
        # ADR-004 per-species family validation: pure NC species
        # (tvanp=tpawp=0) assemble zero augmentation deeq, so the dump
        # must carry a *measured* zero magnetic nonlocal source and zero
        # response slabs; a nonzero NC slab is malformed, not magnetic.
        species = ityp[site] - 1
        if not (tvanp[species] or tpawp[species]):
            if np.any(magnetic[:count, :count, site] != 0):
                raise ValueError(
                    f"QE spinor dump {path}: norm-conserving site {site} "
                    "(tvanp=tpawp=0) has a nonzero deeq_nc_magnetic nonlocal "
                    "source; NC species assemble exactly zero augmentation "
                    "deeq and must measure a zero magnetic nonlocal source"
                )
            if d_response is not None and np.any(d_response[:count, :count, site] != 0):
                raise ValueError(
                    f"QE spinor dump {path}: norm-conserving site {site} "
                    "(tvanp=tpawp=0) has a nonzero D_response slab; all nine "
                    "native response blocks must measure zero for NC species"
                )
        magnetic_source_present |= bool(
            np.any(field[:count, :count, site])
            or np.any(magnetic[:count, :count, site])
        )
        if d_response is not None:
            for c in range(3):
                for j in range(3):
                    block = _qe_native_spinor(d_response[:, :, site, :, c, j])
                    _require_qe_hermitian(
                        block.transpose(0, 2, 1, 3).reshape(2 * nhm, 2 * nhm),
                        f"source response (c={c}, j={j})",
                        path,
                    )
            diagonal = sum(d_response[:count, :count, site, :, j, j] for j in range(3))
            reference = magnetic[:count, :count, site]
            if not np.allclose(
                diagonal,
                reference,
                rtol=1e-8,
                atol=1e-8 * float(np.abs(reference).max(initial=0.0)),
            ):
                raise ValueError(
                    f"QE spinor dump {path}: site {site} source response fails "
                    "the diagonal identity gate sum_j D_response(:,:,:,:,j,j) "
                    "== deeq_nc_magnetic (relative tolerance 1e-8)"
                )
    if not magnetic_source_present and hubbard.state == "unknown":
        raise ValueError(f"QE spinor dump {path}: zero magnetic source")
    return QESpinorDump(
        path=path,
        magic=bytes(header[:16]).rstrip(b" ").decode("ascii"),
        version=version,
        **dims,
        nelec=nelec,
        ef=ef,
        degauss=degauss,
        ngauss=ngauss,
        ltetra=ltetra,
        lgauss=lgauss,
        at=at,
        bg=bg,
        alat=alat,
        omega=omega,
        ibrav=ibrav,
        atm=atm,
        ityp=ityp,
        tau=tau,
        nh=nh,
        tvanp=tvanp,
        tpawp=tpawp,
        has_so=has_so,
        deeq_nc_total=total,
        deeq_nc_magnetic=magnetic,
        gram=gram,
        xk=xk,
        wk=wk,
        et=et,
        wg=wg,
        coefficients=coefficients,
        projected_field=field,
        becsum=becsum,
        d_response=d_response,
        hubbard=hubbard,
    )


_LEVI_CIVITA = np.zeros((3, 3, 3))
_LEVI_CIVITA[0, 1, 2] = _LEVI_CIVITA[1, 2, 0] = _LEVI_CIVITA[2, 0, 1] = 1.0
_LEVI_CIVITA[0, 2, 1] = _LEVI_CIVITA[2, 1, 0] = _LEVI_CIVITA[1, 0, 2] = -1.0


def _qe_gram_inverse(gram_block, rcond=1.0e-6, path=None, site=None):
    """SVD-conditioned dual transform for the smooth beta field.

    Fully relativistic beta manifolds can be numerically rank-deficient
    at a single k (measured Fe FR cond ~1.1e10 with min eigenvalue
    ~5e-9 while the plain inverse explodes the smooth dressing to
    ~1e8 Ry): j-split channels share l space. Truncating eigenvalues
    below ``rcond`` times the largest keeps well-conditioned Grams
    (bccFe scalar cond 58) bit-identical to the plain inverse.

    This is a numerical-conditioning convention only, with NO physical
    guarantee. ``rcond`` bounds the *discarded* modes, not the retained
    ones: measured FR NC (PseudoDojo Fe) retains eigenvalues as small
    as 5.4e-6 at rcond=1e-6, and ``M^+ A M^+`` dressing amplifies by
    1/lambda^2 on those, so the dressed vertex is unbounded in
    principle. The eigengap gate below is a necessary, not sufficient,
    diagnostic; it certifies neither completeness of the retained
    subspace nor exchange readiness. FR-NC exchange additionally
    requires a window-resolved discarded-mode spectral-weight and
    exchange-sensitivity campaign before any certification.
    """
    spectrum = np.linalg.eigvalsh(gram_block)
    cutoff = rcond * float(spectrum.max(initial=0.0))
    retained = spectrum[spectrum > cutoff]
    discarded = spectrum[spectrum <= cutoff]
    if discarded.size:
        largest_discarded = float(discarded.max())
        smallest_retained = float(retained.min()) if retained.size else 0.0
        gap = (
            smallest_retained / largest_discarded if largest_discarded > 0.0 else np.inf
        )
        if not gap >= 100.0:
            where = f"QE spinor dump {path}: " if path is not None else ""
            which = f"site {site} " if site is not None else ""
            raise ValueError(
                f"{where}beta Gram {which}truncated modes lack a clean eigengap: "
                f"smallest retained / largest discarded = {gap:.6e} < 100 at the "
                f"rcond={rcond:g} cutoff {cutoff:.6e} (retained {retained.size}, "
                "discarded {0}). A conditioned pseudoinverse reconstructs only "
                "the retained subspace; refusing an ambiguous truncation.".format(
                    discarded.size
                )
            )
    return np.linalg.pinv(gram_block, rcond=rcond, hermitian=True)


def _qe_gram_spectrum_report(gram_block, rcond=1.0e-6):
    """ADR-003 diagnostic: retained/discarded dimensions, cutoff, eigenvalues."""
    spectrum = np.linalg.eigvalsh(gram_block)
    cutoff = rcond * float(spectrum.max(initial=0.0))
    retained = spectrum > cutoff
    return {
        "retained": int(retained.sum()),
        "discarded": int((~retained).sum()),
        "cutoff": cutoff,
        "eigenvalues": [float(value) for value in spectrum],
    }


def _spinor_tangent_vertices_from_response(dump, operator, path):
    """Build explicit frozen-source tangent vertices (eV) from NC1.1 blocks.

    ``V_a = RYTOEV * sum_{c,j} K^{(a)}_{cj} [(M^+ A_j M^+) (x) sigma_c
    + D_{c<-j}]`` with ``K^{(a)}_{cj} = sum_d eps_{cdj} (n x t_a)_d``; only
    the smooth field ``A_j`` is Gram-dressed, the native dual response
    ``D_{c<-j}`` is used raw. One Ry->eV conversion (``V = dH_mag/dtheta =
    1/2 dDelta/dtheta``). Also validates per-(site, axis) Hermiticity and
    the longitudinal identity ``sum_a n_a V^a = 0``.
    """
    nsite, nmax = operator.shape[0], operator.shape[1]
    vertices = np.zeros((nsite, 3, nmax, nmax, 2, 2), dtype=complex)
    for site, count in enumerate(dump.nh_per_atom):
        response = dump.d_response[:count, :count, site]
        try:
            n = magnetic_tangent_vertices(operator[site])["n"]
        except ValueError as exc:
            # A nonmagnetic site can have zero tangents only if every
            # source-response block vanishes. A zero diagonal sum does
            # not rule out cancelling nonzero off-diagonal responses.
            if "no magnetic splitting" not in str(exc):
                raise
            if np.any(operator[site] != 0):
                raise ValueError(
                    f"QE spinor dump {path}: site {site} has a nonzero "
                    "magnetic operator but its Pauli trace has no "
                    "well-defined moment direction"
                ) from exc
            if np.any(response != 0):
                raise ValueError(
                    f"QE spinor dump {path}: site {site} has zero magnetic "
                    "splitting but nonzero source response"
                ) from exc
            continue
        inverse = _qe_gram_inverse(
            dump.gram[:count, :count, site], path=path, site=site
        )
        smooth_fields = tuple(
            inverse @ dump.projected_field[:count, :count, site, j] @ inverse
            for j in range(3)
        )
        for axis in range(3):
            generator = np.cross(n, np.eye(3)[axis])
            weights = np.einsum("cdj,d->cj", _LEVI_CIVITA, generator)
            block = np.zeros((count, count, 2, 2), dtype=complex)
            for c in range(3):
                for j in range(3):
                    if weights[c, j] == 0.0:
                        continue
                    smooth = smooth_fields[j]
                    block += weights[c, j] * (
                        np.einsum("ij,st->ijst", smooth, PAULI_MATRICES[c])
                        + _qe_native_spinor(response[:, :, :, c, j])
                    )
            vertices[site, axis, :count, :count] = RYTOEV * block
            dense = spinor_dense_block(vertices[site, axis])
            scale = np.abs(dense).max(initial=0.0)
            if not np.allclose(dense, dense.conj().T, rtol=1e-8, atol=1e-8 * scale):
                raise ValueError(
                    f"QE spinor dump {path}: site {site} tangent vertex axis "
                    f"{axis} is not complex Hermitian"
                )
        longitudinal = np.einsum("a,aijst->ijst", n, vertices[site, :, :count, :count])
        scale = np.abs(vertices[site, :, :count, :count]).max(initial=0.0)
        if not np.allclose(longitudinal, 0.0, atol=1e-10 * max(scale, 1.0)):
            raise ValueError(
                f"QE spinor dump {path}: site {site} tangent vertices have a "
                "nonzero longitudinal contraction sum_a n_a V^a"
            )
    return vertices


def read_qe_spinor_dump(path, *, legacy_no_u=False) -> ProjectorGreenData:
    """Map NC spinor coefficients to a spinor operator and tangent vertices.

    NC1.0 stays scalar-relativistic only (SO-bearing rejected fail-closed);
    NC1.1 additionally builds the explicit frozen-source tangent vertices
    ``spinor_tangent_vertices`` (eV) from the nine native response blocks.
    """
    from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM

    dump = parse_qe_spinor_dump(path)
    hubbard_metadata = _qe_hubbard_admission(dump, legacy_no_u)
    if dump.hubbard.state == "absent":
        _hub_require(
            any(frame.origin for frame in dump.hubbard.frames),
            "source",
            "no defined native magnetic reference direction for exchange",
        )
    if dump.version == "1.0" and (dump.lspinorb or np.any(dump.has_so)):
        raise ValueError(
            "QE NC1.0 SO-bearing projector dump lacks native source-resolved "
            "magnetic tangents; fully relativistic exchange is unvalidated"
        )
    counts = dump.nh_per_atom
    gram_spectrum = [
        _qe_gram_spectrum_report(dump.gram[:count, :count, site])
        for site, count in enumerate(counts)
    ]
    nc_sites = [
        site
        for site in range(dump.nat)
        if not (dump.tvanp[dump.ityp[site] - 1] or dump.tpawp[dump.ityp[site] - 1])
    ]
    # QE average_pp clears has_so for an SOC-off FR NC UPF but retains
    # j-split beta radials. A truncated NC Gram needs a physical weight
    # certificate even when neither SOC flag identifies that UPF.
    uncertified_nc = any(
        dump.has_so[dump.ityp[site] - 1] or gram_spectrum[site]["discarded"]
        for site in nc_sites
    )
    nmax = int(counts.max())
    indices = np.full((dump.nat, nmax), -1, dtype=int)
    operator = np.zeros((dump.nat, nmax, nmax, 2, 2), dtype=complex)
    for site, count in enumerate(counts):
        start = int(dump.ofsbeta[site])
        indices[site, :count] = np.arange(start, start + count)
        inverse = _qe_gram_inverse(
            dump.gram[:count, :count, site], path=path, site=site
        )
        projected = np.array(
            [
                inverse @ dump.projected_field[:count, :count, site, axis] @ inverse
                for axis in range(3)
            ]
        )
        magnetic = _qe_native_spinor(dump.deeq_nc_magnetic[:count, :count, site])
        operator[site, :count, :count] = (
            2
            * RYTOEV
            * (np.einsum("aij,ast->ijst", projected, PAULI_MATRICES) + magnetic)
        )
    if not np.any(operator != 0):
        raise ValueError(
            f"QE spinor dump {path}: zero conditioned magnetic source "
            "after dressing the projected Pauli field"
        )
    vertices = None
    tangent_definition = None
    if dump.version in ("1.1", "1.2"):
        vertices = _spinor_tangent_vertices_from_response(dump, operator, path)
        tangent_definition = SPINOR_TANGENT_DEFINITION
        if not dump.lspinorb:
            # Scalar-relativistic explicit-versus-commutator identity.
            for site, count in enumerate(dump.nh_per_atom):
                if dump.has_so[dump.ityp[site] - 1]:
                    continue
                try:
                    commutator = magnetic_tangent_vertices(operator[site])["vertices"]
                except ValueError as exc:
                    if "no magnetic splitting" not in str(exc):
                        raise
                    if np.any(operator[site] != 0) or np.any(vertices[site] != 0):
                        raise ValueError(
                            f"QE spinor dump {path}: site {site} has a "
                            "nonzero magnetic operator or explicit tangent "
                            "without a defined moment direction"
                        ) from exc
                    continue
                for axis in range(3):
                    explicit = spinor_dense_block(vertices[site, axis])
                    reference = commutator[axis]
                    scale = np.abs(reference).max(initial=0.0)
                    residual = float(np.abs(explicit - reference).max(initial=0.0))
                    if residual > 1e-6 * max(scale, 1e-30):
                        raise ValueError(
                            f"QE spinor dump {path}: scalar-relativistic site "
                            f"{site} explicit tangent axis {axis} deviates from "
                            "the magnetic_tangent_vertices commutator by "
                            f"TB2J_TAN_PARITY_RESIDUAL={residual:.6e} "
                            "(relative tolerance 1e-6)"
                        )
    sites = np.repeat(np.arange(dump.nat), counts)
    data = ProjectorGreenData(
        kpoints=dump.xk @ dump.at,
        weights=dump.wk,
        eigenvalues=dump.et[None] * RYTOEV,
        occupations=_band_occupations(dump.wg, dump.wk)[None],
        coefficients=dump.coefficients.transpose(0, 3, 2, 1)[None],
        efermi=dump.ef * RYTOEV,
        projector_site=sites,
        projector_atom=sites.copy(),
        cell=dump.at.T * dump.alat * BOHR_TO_ANGSTROM,
        positions=dump.tau * dump.alat * BOHR_TO_ANGSTROM,
        atomic_numbers=_atomic_numbers_for_labels(dump.atm)[dump.ityp - 1],
        overlap_k=None,
        site_nproj=counts,
        site_projector_indices=indices,
        nspinor=2,
        spinor_operator=operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        spinor_tangent_vertices=vertices,
        spinor_tangent_definition=tangent_definition,
        coefficient_source="qe_becp_nc",
        coefficient_projector=COEFFICIENT_PROJECTOR,
        channel_interpretation=CHANNEL_INTERPRETATION,
        operator_basis=OPERATOR_BASIS,
        population_metric="QE spinor becsum reserved zero record; not an occupation reference",
        metadata={
            **hubbard_metadata,
            "source": f"QE PW/src/becp_dump.f90 NC{dump.version}",
            "source_code": "qe",
            "qe_magic": dump.magic,
            "dump_version": dump.version,
            "coefficient_convention": "dual_projector_no_inverse",
            "native_spinor_order": ["uu", "ud", "du", "dd"],
            "lspinorb": bool(dump.lspinorb),
            "domag": bool(dump.domag),
            # First-k numerical spectrum only. It cannot certify
            # Brillouin-zone completeness or the physical lost weight.
            "gram_spectrum": gram_spectrum,
            **(
                {
                    "nc_exchange_status": (
                        "uncertified: a norm-conserving species is SO-bearing "
                        "or has truncated beta-Gram modes. A numerical "
                        "eigengap does not bound discarded normalized-state "
                        "spectral weight; per-k window weights and exchange "
                        "threshold sensitivity are required. This also "
                        "covers SOC-off FR UPFs whose has_so flag is cleared."
                    ),
                }
                if uncertified_nc
                else {}
            ),
            "has_so": dump.has_so.tolist(),
            "tvanp": dump.tvanp.tolist(),
            "tpawp": dump.tpawp.tolist(),
            "atm": dump.atm,
            "ityp": dump.ityp.tolist(),
            "nh": dump.nh.tolist(),
            "nelec": dump.nelec,
            "spinor_operator_source": (
                "2 * [sum_i (M^+ <beta|B_i|beta> M^+) sigma_i "
                "+ deeq_nc_magnetic]; Hermitian Gram pseudoinverse "
                "rcond=1e-6 for smooth field only; Ry -> eV"
            ),
            **(
                {
                    "spinor_tangent_source": (
                        "V_a = Ry->eV * sum_{c,j} eps_{cdj} (n x t_a)_d "
                        "[(M^+ <beta|B_j|beta> M^+) (x) sigma_c "
                        "+ D_response(:,:,:,:,c,j)]; smooth field only "
                        "uses Hermitian Gram pseudoinverse rcond=1e-6; "
                        "native dual response is raw; "
                        "n from the existing Pauli-trace convention"
                    ),
                    "d_response_note": (
                        "record 13+nks: native same-gauge zero-source-subtracted "
                        "response D(nhm,nhm,nat,4,3,3) with c = target Pauli "
                        "slot and j = original source axis"
                    ),
                }
                if vertices is not None
                else {}
            ),
            "magnetic_nonlocal_source": (
                "same-density/gauge D(smooth B,PAW magnetic Hxc) minus "
                "D(smooth B=0,PAW magnetic Hxc=0); fixed ionic/SOC excluded"
            ),
            "deeq_nc_total_note": (
                "native total deeq_nc (uu,ud,du,dd) stays diagnostic only in the "
                "raw QESpinorDump container; never a tangent vertex"
            ),
            "becsum_note": "NC four-component record zeroed by producer; NSCF accumulator uninitialized, not an occupation-parity reference",
            "family_validation": (
                "NC1.0 scalar-relativistic US/PAW only; native-action and "
                "collinear G3 parities are required. SO-bearing fcoef "
                "projectors need a new native source-tangent ABI"
                if dump.version == "1.0"
                else (
                    "NC1.1: native nine-block source responses passed the "
                    "diagonal identity gate, per-(site,axis) Hermiticity, the "
                    "longitudinal identity and the scalar-relativistic "
                    "explicit-versus-commutator parity on has_so=F species"
                )
            ),
            "units": {
                "eigenvalues": "eV",
                "efermi": "eV",
                "spinor_operator": "eV",
                **({"spinor_tangent_vertices": "eV"} if vertices is not None else {}),
            },
        },
    )
    if uncertified_nc:
        warnings.warn(
            "QE NC spinor dump is read for research only: SOC-bearing or "
            "truncated NC beta modes have uncertified exchange "
            "(see metadata 'nc_exchange_status')",
            stacklevel=2,
        )
    data.validate(exchange_ready=True)
    return data


# ---------------------------------------------------------------------------
# Exchange driver (mirrors gen_exchange_abinit_projector)
# ---------------------------------------------------------------------------
def gen_exchange_qe(
    filename,
    output_path="TB2J_results_qe",
    Rcut=10.0,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    overlap_mode=None,
    overlap_rcond=None,
    leg_consistency_atol_eV=None,
    *,
    legacy_no_u=False,
):
    """Generate projector exchange output from a QE projector dump file.

    The dump family is detected from the file magic.  KB/beta dumps
    (``TB2JQEDUMPV1*``) carry already-dual ``becp`` coefficients whose only
    exchange component is the covariant separable-operator spin splitting
    ``deeq(up) - deeq(down)`` (v1.2: plus the metric-transformed projected
    xc vertex) in the beta-dual channel; ``overlap_mode`` /
    ``overlap_rcond`` are rejected there because the coefficients are dual
    and no inverse exists to control.  Atomic dumps
    (``TB2JQEATWFC1.0``) carry primal atomic-wavefunction coefficients with
    the full per-k overlap ``M(k)``; there the controls select how the
    shared runtime dresses with ``M(k)^-1`` (default ``inverse``).

    A mapping ``{x: dump_x, y: dump_y, z: dump_z}`` selects spinor
    full-tensor output through the existing rank-nine writer; NC1.1 sets
    carry the native source-response tangent, NC1.0 sets remain the
    scalar-relativistic-only legacy family and the two may not be mixed.
    ``leg_consistency_atol_eV`` (spinor three-leg only) bounds the
    repeated-row gate of the merger; it must be finite, positive and
    at most 1e-7 eV, and the measured worst repeated-row spread is
    reported in the output description. A single NC spinor
    dump is available only via ``read_qe_spinor_dump`` and the raw leg API.
    """
    from TB2J.interfaces.gpaw_projector import (
        _magnetic_sites,
        _R_grid_for_cutoff,
        component_local_operators,
        write_projector_exchange_out,
    )

    if isinstance(filename, Mapping):
        if set(filename) != {"x", "y", "z"}:
            raise ValueError(
                "full QE spinor exchange requires all x/y/z reference dumps"
            )
        if overlap_mode is not None or overlap_rcond is not None:
            raise ValueError(
                "overlap controls are invalid for KB spinor coefficients, already dual"
            )
        if leg_consistency_atol_eV is None:
            leg_consistency_atol_eV = 5.0e-8
        leg_consistency_atol_eV = float(leg_consistency_atol_eV)
        if (
            not np.isfinite(leg_consistency_atol_eV)
            or leg_consistency_atol_eV <= 0.0
            or leg_consistency_atol_eV > 1.0e-7
        ):
            raise ValueError(
                "leg_consistency_atol_eV must be finite, positive and at most "
                f"1e-7 eV; got {leg_consistency_atol_eV!r}"
            )
        from TB2J.interfaces.gpaw_spinor_projector import (
            write_spinor_projector_exchange_out,
        )

        references = {
            axis: read_qe_spinor_dump(filename[axis], legacy_no_u=legacy_no_u)
            for axis in ("x", "y", "z")
        }
        baseline = references["z"].metadata
        for axis, reference in references.items():
            if any(
                reference.metadata[key] != baseline[key]
                for key in (
                    "qe_magic",
                    "lspinorb",
                    "has_so",
                    "tvanp",
                    "tpawp",
                    "atm",
                    "ityp",
                    "nh",
                )
            ):
                raise ValueError(
                    f"QE spinor reference {axis!r} differs in UPF/projector family"
                )
        augmented = [bool(t or p) for t, p in zip(baseline["tvanp"], baseline["tpawp"])]
        if any(augmented) and not all(augmented):
            raise ValueError(
                "mixed NC and augmented QE spinor production exchange is "
                "not certified: the real Fe NC + O US smoke has a magnetic "
                "ligand, and a converged zero-source augmented ligand "
                "reference has not passed the source/reader gate. The "
                "per-site validated reader remains available for research; "
                "do not publish a mixed-family tensor."
            )
        if any(
            "nc_exchange_status" in reference.metadata
            for reference in references.values()
        ):
            raise ValueError(
                "norm-conserving QE spinor exchange is not certified for "
                "production output: an SO-bearing NC species or a truncated "
                "NC beta Gram requires per-k window-resolved discarded-mode "
                "spectral weights and measured exchange sensitivity across "
                "conditioning thresholds. A clean first-k eigengap alone "
                "does not bound normalized-state error. The PseudoDojo FR "
                "NC Fe pilot retains ~5.4e-6 modes at rcond=1e-6 and "
                "amplifies max|M^+ A M^+| to ~4.5e3 Ry. The same FR UPF "
                "with SOC disabled clears has_so without averaging betas. "
                "Use read_qe_spinor_dump and compute_spinor_projector_exchange "
                "for research legs only."
            )
        selected_sites = _magnetic_sites(
            references["z"],
            magnetic_elements=magnetic_elements,
            index_magnetic_atoms=index_magnetic_atoms,
        )
        automatic_sites = magnetic_elements is None and index_magnetic_atoms is None
        active_sites = []
        for site in selected_sites:
            if site < 0 or site >= len(references["z"].site_nproj):
                raise ValueError(f"QE spinor magnetic site index {site} is invalid")
            present = [
                bool(np.any(references[axis].spinor_operator[site] != 0))
                for axis in ("x", "y", "z")
            ]
            if not any(present):
                if automatic_sites:
                    continue
                raise ValueError(
                    f"QE spinor site {site} was explicitly selected but "
                    "has zero magnetic splitting in every reference"
                )
            if not all(present):
                raise ValueError(
                    f"QE spinor site {site} lacks magnetic splitting "
                    "in one or more x/y/z reference directions"
                )
            active_sites.append(site)
        if not active_sites:
            raise ValueError("QE spinor references contain no magnetic site")
        return write_spinor_projector_exchange_out(
            references,
            path=output_path,
            Rpts=Rpts,
            Rcut=Rcut,
            nz=nz,
            smearing_eV=smearing_eV,
            magnetic_elements=magnetic_elements,
            index_magnetic_atoms=active_sites,
            leg_consistency_atol_eV=leg_consistency_atol_eV,
            description=(
                f"QE {baseline['qe_magic']} KB spinor full rank-nine exchange "
                "from independent x/y/z "
                "references; NC1.1 uses native frozen magnetic-source response "
                "tangents and NC1.0 uses scalar-relativistic commutator tangents. "
                "Native total deeq_nc (including fixed SOC) is diagnostic only; "
                "coefficients are dual and overlap_k=None."
            ),
        )

    if leg_consistency_atol_eV is not None:
        raise ValueError(
            "leg_consistency_atol_eV applies only to the spinor three-leg "
            "merge (--input_x --input_y --input_z)"
        )

    magic = _sniff_qe_dump_magic(filename)
    if magic.rstrip(b" ") in _SPINOR_MAGICS:
        raise ValueError(
            "a single QE spinor dump measures only a transverse leg; standard "
            "exchange.out requires independent x/y/z dumps (--input_x/y/z). "
            "Use read_qe_spinor_dump and compute_spinor_projector_exchange for a raw leg"
        )
    if magic.rstrip(b" \x00") in _ATOMIC_MAGICS:
        data = read_qe_atomic_dump(filename, legacy_no_u=legacy_no_u)
    elif magic.rstrip(b" ") in _MAGIC_VERSIONS:
        if overlap_mode is not None or overlap_rcond is not None:
            raise ValueError(
                "overlap_mode/overlap_rcond apply only to atomic-wavefunction "
                "dumps (magic 'TB2JQEATWFC1.0', primal coefficients with "
                "k-dependent M(k) overlap); this file is a KB/beta becp dump "
                f"(magic {magic.rstrip().decode('ascii', 'replace')!r}) whose "
                "coefficients are already dual, so there is no inverse to "
                "control"
            )
        data = read_qe_dump(filename, legacy_no_u=legacy_no_u)
    else:
        raise ValueError(f"QE dump {filename}: unknown magic/version {magic!r}")

    sites = None
    if index_magnetic_atoms is not None:
        sites = [int(site) for site in index_magnetic_atoms]
    if sites is None:
        sites = list(range(len(data.site_nproj)))
    if Rpts is None:
        Rpts = _R_grid_for_cutoff(data, sites, Rcut)
    local_operators = component_local_operators(
        data, "delta_total", sites, "QE becp_dump"
    )
    if data.overlap_k is None:
        description = (
            "Projector Green workflow using QE KB beta projections "
            f"({data.coefficient_source}; coefficients already dual, "
            "overlap_k=None) and the projected smooth-XC plus separable "
            "deeq spin vertex in basis "
            f"{data.operator_basis}. QE dump {data.metadata.get('qe_magic', 'unknown')}; "
            "collinear only; npool=1. "
            f"{data.population_metric}."
        )
    else:
        # Mirror the ProjectorGreen fallback chain so the description states
        # the controls actually in effect.
        mode = (
            overlap_mode
            or os.environ.get("TB2J_GREEN_MODE")
            or data.metadata.get("overlap_mode")
            or "inverse"
        )
        mode = {"contravariant": "inverse"}.get(mode, mode)
        rcond = overlap_rcond
        if rcond is None:
            rcond = os.environ.get(
                "TB2J_GREEN_RCOND", data.metadata.get("overlap_rcond", 1.0e-10)
            )
        data.metadata["effective_overlap_mode"] = mode
        data.metadata["effective_overlap_rcond"] = float(rcond)
        description = (
            "Projector Green workflow using QE atomic-wavefunction (UPF "
            f"PP_PSWFC, atomic_wfc) projections ({data.coefficient_source}; "
            "primal coefficients with full k-dependent overlap_k = M(k), "
            f"overlap_mode={data.metadata.get('effective_overlap_mode')}, "
            f"overlap_rcond={data.metadata.get('effective_overlap_rcond')}) "
            "and the BZ-weighted R=0 onsite covariant site spin vertex "
            "Delta = cov_xc + cov_aug with hij(up/dn)=±Delta/2 in basis "
            f"{data.operator_basis}. QE dump {data.metadata.get('qe_magic', 'unknown')}; "
            "collinear only; npool=1. Atomic basis unvalidated quantitatively "
            "on bccFe (matched NC/US/PAW UPFs disagree with KB)."
        )
    return write_projector_exchange_out(
        data,
        path=output_path,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
        description=description,
        population_mode="none",
        Rcut=Rcut,
        local_operators=local_operators,
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )


def _sniff_qe_dump_magic(filename) -> bytes:
    """Read the 16-byte magic of a QE projector dump without full parsing."""
    with open(filename, "rb") as fh:
        payload = _RecordReader(Path(filename), fh).read_raw("header")
        if len(payload) < 16:
            raise ValueError(
                f"QE dump {filename}: header record too short ({len(payload)} bytes)"
            )
    return bytes(payload[:16])
