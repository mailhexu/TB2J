"""Reader for the Quantum ESPRESSO ``becp_dump`` projector-Green exporter.

Parses the binary dump written by ``PW/src/becp_dump.f90`` (QE fork branch
``TB2J``, enabled with ``TB2J_DUMP=<file>``) and normalizes it into
:class:`TB2J.projector_green.ProjectorGreenData` for the projector-space
exchange workflow.

Format (authoritative: ``DUMP_FORMAT_V1_1.md``): gfortran sequential
unformatted Fortran with 4-byte little-endian record markers; all arrays
Fortran/column-major.  v1.1 adds the packed-triangular ``becsum`` record
(and, for PAW runs, ``rho%bec``) after the per-k coefficient stream;
v1.0 ends after the per-k records.

Pinned physics contract (research memo ``2026-10-08-qe-export-surface-and-
operators`` and SymPy pin ``docs/sympy/qe_separable_beta_trace.py``):

* the ``becp`` coefficients ``P_ni = <beta_n|psi_i>`` are already dual, so
  ``overlap_k`` stays ``None`` — the ``qq_at`` and beta-Gram records are
  diagnostics only and are never used as a channel metric;
* ``deeq`` is the matching covariant separable-operator coefficient
  (``V_NL = beta D beta^dagger``) and the spin vertex is
  ``hij = deeq(up) - deeq(down)`` per atom block;
* energies (``deeq``, ``dvan``, ``et``, ``ef``) are in Ry and converted to
  eV with ``RYTOEV = 13.605693122994``; coefficients are dimensionless.

Public API
----------
``parse_qe_dump(path)``
    Raw parse into a :class:`QEProjectorDump` (all records + metadata, in
    dumped units).
``read_qe_dump(path)``
    Normalized :class:`~TB2J.projector_green.ProjectorGreenData` (eV /
    Angstrom / fractional k-points, spin-degenerate dense mesh).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from TB2J.projector_green import ProjectorGreenData

RYTOEV = 13.605693122994

#: 16-byte magic strings; Fortran pads ``CHARACTER(LEN=16)`` with blanks.
_MAGIC_VERSIONS = {
    b"TB2JQEDUMPV1.2": "1.2",
    b"TB2JQEDUMPV1.1": "1.1",
    b"TB2JQEDUMPV1": "1.0",
}

HIJ_DEFINITION = "qe_deeq_spin_difference"
HIJ_UNITS = "eV"
HIJ_SOURCE = "QE becp_dump deeq record (PW/src/becp_dump.f90)"
HIJ_PROJECTION = "QE ultrasoft/PAW beta channel (dual basis)"
COEFFICIENT_SOURCE = "qe_becp"
COEFFICIENT_PROJECTOR = "qe_beta"
CHANNEL_INTERPRETATION = "qe_dual_to_beta"
OPERATOR_BASIS = "qe_dual_beta_channel"

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
# parse_qe_dump
# ---------------------------------------------------------------------------
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


def parse_qe_dump(path) -> QEProjectorDump:
    """Parse a QE ``becp_dump`` file into raw :class:`QEProjectorDump` data.

    Raises ``ValueError`` on unknown magic/version, non-LSDA (``nspin != 2``)
    dumps, pool-parallel dumps (``nkstot != nks``), dumps without any US or
    PAW species, and any record-length/shape mismatch.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        reader = _RecordReader(path, fh)

        # -- record 0: magic + dimensions ------------------------------------
        chunks = reader.split_record(
            "header", [("magic", "S16", 1), ("dims", "<i4", 9)]
        )
        magic_raw = bytes(chunks["magic"][0])
        version = _MAGIC_VERSIONS.get(magic_raw.rstrip())
        if version is None:
            raise ValueError(
                f"QE dump {path}: unknown magic {magic_raw!r}; expected "
                f"'TB2JQEDUMPV1.1 ' (v1.1) or 'TB2JQEDUMPV1   ' (v1.0)"
            )
        nspin, nks, nkstot, nbnd, nat, nsp, nhm, lmaxkb, nkb = (
            int(v) for v in chunks["dims"]
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
        chunks = reader.split_record(
            "fermi/smearing",
            [
                ("energies", "<f8", 5),
                ("smear", "<i4", 3),
            ],
        )
        nelec, ef, ef_up, ef_dw, degauss = (float(v) for v in chunks["energies"])
        ngauss, ltetra, lgauss = (int(v) for v in chunks["smear"])

        # -- record 2: cell ---------------------------------------------------
        chunks = reader.split_record(
            "cell",
            [("lat", "<f8", 18), ("cell_scalars", "<f8", 2), ("ibrav", "<i4", 1)],
        )
        at = np.array(chunks["lat"][:9].reshape((3, 3), order="F"))
        bg = np.array(chunks["lat"][9:].reshape((3, 3), order="F"))
        alat, omega = (float(v) for v in chunks["cell_scalars"])
        ibrav = int(chunks["ibrav"][0])

        # -- record 3: species labels ----------------------------------------
        # The dump spec lists ``S3 x nsp``; QE >= 8 (the fork's base) stores
        # ``atm`` as CHARACTER(LEN=6) with fixed capacity, so the record can
        # carry blank-padded 6-byte labels for more than nsp entries.  Decode
        # either encoding: take the first nsp labels at the widest width that
        # fills all of them with non-blank ascii.
        atm = _decode_species_labels(reader.read_raw("species labels"), nsp, path)

        # -- record 4: ions ---------------------------------------------------
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

        # -- record 11: bands --------------------------------------------------
        chunks = reader.split_record(
            "bands", [("et", "<f8", nbnd * nks), ("wg", "<f8", nbnd * nks)]
        )
        et = np.array(chunks["et"].reshape((nbnd, nks), order="F").T)
        wg = np.array(chunks["wg"].reshape((nbnd, nks), order="F").T)

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
# read_qe_dump: ProjectorGreenData normalization
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

    if dump.ef_up != 0.0 and dump.ef_dw != 0.0:
        efermi_spin = np.array([dump.ef_up, dump.ef_dw]) * RYTOEV
        efermi_source = "qe ef_up/ef_dw (two Fermi energies)"
    else:
        efermi_spin = None
        efermi_source = "qe ef (single Fermi energy)"
    efermi = dump.ef * RYTOEV

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
):
    """Generate projector exchange output from a QE ``becp_dump`` file.

    The QE operator channel has exactly one exchange component: the
    ``delta_total`` block registered by :func:`read_qe_dump`, i.e. the
    covariant separable-operator spin splitting
    ``deeq(up) - deeq(down)`` in the beta-dual channel (already matching
    the undressed ``becp`` Green matrix).  There is deliberately no
    ``overlap_mode``/``overlap_rcond`` knob: the coefficients are dual.
    """
    from TB2J.interfaces.gpaw_projector import (
        _R_grid_for_cutoff,
        component_local_operators,
        write_projector_exchange_out,
    )

    data = read_qe_dump(filename)
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
    description = (
        "Projector Green workflow using QE becp_dump ultrasoft/PAW beta "
        f"projections ({data.coefficient_source}; coefficients already dual, "
        "overlap_k=None) and the separable-operator spin vertex "
        "deeq(up)-deeq(down) in basis "
        f"{data.operator_basis}. QE dump {data.metadata.get('qe_magic', 'unknown')}; "
        "collinear only; npool=1. "
        f"{data.population_metric}."
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
    )
