"""Reader for the Quantum ESPRESSO projector-Green exporter dumps.

Parses the binary dumps written by ``PW/src/becp_dump.f90`` (QE fork branch
``TB2J``) and normalizes them into
:class:`TB2J.projector_green.ProjectorGreenData` for the projector-space
exchange workflow.

Two dump families are supported (record schemas in
``TB2J/qe_patch/README.md``). Both use gfortran sequential unformatted
records with 4-byte little-endian markers and Fortran/column-major
arrays. A third, non-collinear spinor family is read by
:func:`parse_qe_spinor_dump` / :func:`read_qe_spinor_dump`:

KB/beta dumps (``TB2J_PROJECTORS=kb``, magic ``TB2JQEDUMPV1`` / ``V1.1`` /
``V1.2``)
    v1.1 adds the packed-triangular ``becsum`` record (and, for PAW runs,
    ``rho%bec``) after the per-k coefficient stream; v1.0 ends after the
    per-k records; v1.2 adds the projected-xc vertex records.

Atomic-pseudo-wavefunction dumps (``TB2J_PROJECTORS=atomic``, magic
``TB2JQEATWFC1.0``)
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

Non-collinear spinor KB dumps (magic ``TB2JQEDUMPNC1.0`` / ``NC1.1``)
    Native ``becp%nc(nkb, 2, nbnd)`` per k in QE's dual beta basis;
    native ``(uu, ud, du, dd)`` total and magnetic ``deeq_nc`` blocks,
    Hermitian projected smooth fields, beta Gram and band occupations.
    NC1.1 adds nine source-resolved magnetic response blocks, including
    rotated PAW one-center sources, to construct explicit frozen-source
    tangents without rotating the fixed ionic/SOC Hamiltonian. The beta
    Gram conditions only the smooth-field projection, never the dual Green
    coefficients or native nonlocal response. NC1.0 SO-bearing dumps are
    rejected because they have no source-resolved tangents.

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
``read_qe_dump(path)``
    Normalized KB :class:`~TB2J.projector_green.ProjectorGreenData`
    (eV / Angstrom / fractional k-points, spin-degenerate dense mesh).
``read_qe_atomic_dump(path)``
    Normalized atomic :class:`~TB2J.projector_green.ProjectorGreenData`
    with primal coefficients and full k-dependent ``overlap_k = M(k)``.
``parse_qe_spinor_dump(path)`` / ``read_qe_spinor_dump(path)``
    Strict non-collinear spinor parse (magics ``TB2JQEDUMPNC1.0`` /
    ``TB2JQEDUMPNC1.1``) and normalized
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
from dataclasses import dataclass
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
    b"TB2JQEDUMPV1.2": "1.2",
    b"TB2JQEDUMPV1.1": "1.1",
    b"TB2JQEDUMPV1": "1.0",
}

#: Atomic-pseudo-wavefunction dump family (``TB2J_PROJECTORS=atomic``).
ATOMIC_MAGIC = "TB2JQEATWFC1.0"
MAGIC_ATOMIC = b"TB2JQEATWFC1.0"
_MAGIC_ATOMIC = MAGIC_ATOMIC

SPINOR_MAGIC = "TB2JQEDUMPNC1.0"
MAGIC_SPINOR = SPINOR_MAGIC.encode("ascii")
#: NC1.1 adds record ``13+nks``: the nine native source-response blocks
#: ``D_response(nhm, nhm, nat, 4, 3, 3)`` (ih, jh, na, spinblock, c, j).
SPINOR_MAGIC_NC11 = "TB2JQEDUMPNC1.1"
MAGIC_SPINOR_NC11 = SPINOR_MAGIC_NC11.encode("ascii")
_SPINOR_MAGICS = {MAGIC_SPINOR: "1.0", MAGIC_SPINOR_NC11: "1.1"}

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
        if magic_raw.rstrip(b" \x00") == _MAGIC_ATOMIC:
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
        if nspin != 2:
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
        if version in ("1.1", "1.2"):
            becsum = reader.array("becsum", "<f8", (npack, nat, nspin))
            # rho%bec is written only for PAW runs (okpaw at dump time).
            if any(bool(f) for f in tpawp):
                rho_bec = reader.array("rho_bec", "<f8", (npack, nat, nspin))
        if version == "1.2":
            # <beta_i | V_xc(up) - V_xc(dn) | beta_j> per atom (Ry); plus,
            # for PAW runs, the spin-resolved one-center D^1 coefficients.
            dbeta_xc = reader.array("dbeta_xc", "<f8", (nhm, nhm, nat))
            if any(bool(f) for f in tpawp):
                ddd_paw = reader.array("ddd_paw", "<f8", (npack, nat, nspin))
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
        if magic_raw.rstrip(b" \x00") != _MAGIC_ATOMIC:
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


def read_qe_dump(path) -> ProjectorGreenData:
    """Read a QE ``becp_dump`` file into exchange-ready projector Green data.

    Normalizes dumped units (Ry -> eV, alat/Bohr -> Angstrom, cartesian k in
    2pi/alat -> fractional), folds the ``isk``-split k list into the dense
    spin-degenerate mesh expected by :class:`ProjectorGreenData`, and splits
    the concatenated beta channels per atom (QE ``ofsbeta`` order).
    """
    from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM

    dump = parse_qe_dump(path)

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
def read_qe_atomic_dump(path) -> ProjectorGreenData:
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
        "source": (
            "Quantum ESPRESSO TB2J_DUMP atomic-wavefunction projector dump "
            "(PW/src/becp_dump.f90, TB2J_PROJECTORS=atomic)"
        ),
        "source_code": "qe",
        "dump_version": "1.0",
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
    """Strictly parse the KB-only NC spinor ABI (NC1.0 or NC1.1).

    NC1.0 ends after the reserved-zero ``becsum`` record; NC1.1 appends
    record ``13+nks``: the nine native source-response blocks
    ``D_response(nhm, nhm, nat, 4, 3, 3)`` in Fortran order
    ``(ih, jh, na, spinblock, c, j)`` (``c`` = target Pauli slot,
    ``j`` = original source axis), used to build the explicit
    frozen-source tangent vertices. Both versions require finite values,
    complex-Hermitian site blocks and exact EOF; NC1.1 additionally
    enforces the diagonal identity gate
    ``sum_j D_response(:,:,:,:,j,j) == deeq_nc_magnetic`` at 1e-8
    relative tolerance per site.
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
        if np.any(nh <= 0) or int(nh.max()) != nhm or int(nh[ityp - 1].sum()) != nkb:
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
        if version == "1.1":
            d_response = reader.array(
                "source response D", "<c16", (nhm, nhm, nat, 4, 3, 3)
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
        if np.linalg.eigvalsh(gram[:count, :count, site]).min() <= 0:
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
    if not magnetic_source_present:
        raise ValueError(f"QE spinor dump {path}: zero magnetic source")
    return QESpinorDump(
        path=path,
        magic=SPINOR_MAGIC if version == "1.0" else SPINOR_MAGIC_NC11,
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


def read_qe_spinor_dump(path) -> ProjectorGreenData:
    """Map NC spinor coefficients to a spinor operator and tangent vertices.

    NC1.0 stays scalar-relativistic only (SO-bearing rejected fail-closed);
    NC1.1 additionally builds the explicit frozen-source tangent vertices
    ``spinor_tangent_vertices`` (eV) from the nine native response blocks.
    """
    from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM

    dump = parse_qe_spinor_dump(path)
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
    if dump.version == "1.1":
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
            axis: read_qe_spinor_dump(filename[axis]) for axis in ("x", "y", "z")
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
    if magic.rstrip(b" ") in (MAGIC_SPINOR, MAGIC_SPINOR_NC11):
        raise ValueError(
            "a single QE spinor dump measures only a transverse leg; standard "
            "exchange.out requires independent x/y/z dumps (--input_x/y/z). "
            "Use read_qe_spinor_dump and compute_spinor_projector_exchange for a raw leg"
        )
    if magic.rstrip(b" \x00") == _MAGIC_ATOMIC:
        data = read_qe_atomic_dump(filename)
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
        data = read_qe_dump(filename)
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
