"""Reader for the VASP ``tb2j_cso.bin`` one-center SOC dump (versions 1-3).

The dump is produced by the VASP_TB2J_patch Fortran module ``tb2j_cso.F``
(``TB2J_CSO_WRITE_DUMP``), which captures, per ion, the one-center spin-orbit
matrix ``CSO`` and occupancy matrix ``COCC`` (both complex, 4 spinor blocks in
the (uu, ud, du, dd) representation after ``OCC_FLIP4``) together with the
spherical augmentation-sphere potential ``POTAE(:,1,1)`` exactly as consumed by
VASP's ``SPINORB_STRENGTH``.  SAXIS Euler metadata is recorded so the spin
frame of the operator is unambiguous.

Conventions
-----------
- Little-endian raw stream (Fortran ``ACCESS='STREAM'``, no record markers).
- ``cso``/``cocc`` are returned as arrays of shape ``(nions, 4, lmdim, lmdim)``,
  ``potae`` as ``(nmax_max, nions)``; only ``potae[:nmax_ion[i], i]`` is valid
  for ion ``i``.
- ``E_soc`` per ion reproduces VASP's ``CALC_SPINORB_MATRIX_ELEMENTS``
  (relativistic.F): the sum runs over same-``l`` channel pairs only, with the
  inner (row) window following the second channel.  Because ``CSO`` has
  vanishing channel-off-diagonal blocks, this equals ``Re Tr(CSO COCC^dagger)``
  on the ``lmmax`` block.
- ``potae`` carries the raw VASP storage convention (factor ``2*sqrt(pi)``);
  multiply by ``1/(2*sqrt(pi))`` before building xi(r) as in
  ``SPINORB_STRENGTH``.  ``nmax_ion[i]`` is the true per-ion radial-grid
  length (``PP%R%NMAX``); the file zero-pads to ``nmax_max``.
- The collinear strength-0 potential is the spin AVERAGE
  ``(POTAE(:,1,1)+POTAE(:,1,2))/2`` for ISPIN=2 runs, matching the total
  potential the stock noncollinear path consumes; channel 1 as-is for
  ISPIN=1 and LSORBIT runs.  The file stores the full ``nmax_max``-padded
  grid for every ion (zeros beyond ``nmax_ion[i]``); truncate to the POTCAR
  dataset grid via ``nmax_ion`` before use.
- The stream is enforced little-endian at write time
  (``CONVERT='LITTLE_ENDIAN'``); the magic check makes accidental
  host-endian reads fail closed.

Version 2 (band/k provenance)
-----------------------------
Version 2 keeps every version-1 field at an unchanged offset and appends:

- after the SAXIS/alpha/beta header prefix: ``int32 ispin, nkpts, nbands,
  nb_tot``, ``f64 efermi``, ``int32 native_magic, native_version`` (writer
  identity of the companion ``tb2j_native.bin`` snapshot);
- after ``nmax_ion``: ``f64 vkpt(3, nkpts)``, ``f64 wtkpt(nkpts)``.

These tie the dumped ``COCC`` to the same run's CPROJ export
(``COCC = sum_nk w_k f_nk C* C`` per ``fast_aug.F``).  For version-1 files
``CsoDump.provenance`` is ``None``; for version-2 files it is a dict with
the keys above (``vkpt`` shaped ``(nkpts, 3)``, ``wtkpt`` shaped
``(nkpts,)``).

Version 3 (reference-potential provenance)
------------------------------------------
Version 3 keeps every v1/v2 field at an unchanged offset and appends,
after the ``potae`` array:

- ``f64 felect, invmc2, autoa``: the constants entering
  ``APOT(r) = V_H[n_core] + V_nuc - potae_xcr(r) + potae(r)/(2*sqrt(pi))``
  and ``xi(r) = invmc2 * dAPOT/dr`` in ``SPINORB_STRENGTH``
  (``invmc2 = 7.45596E-6 A^2`` exactly as ``relativistic.F:50``);
- ``f64 potae_xcr(nmax_max, nions)``: the per-type
  ``PP%POTAE_XCUPDATED`` radial reference potential broadcast per ion,
  zero-padded to ``nmax_max`` with valid length ``nmax_ion[i]`` (the
  type's ``PP%R%NMAX``, same radial grid as ``potae``).

``CsoDump.potae_xcr`` carries the array and ``CsoDump.constants`` the
dict ``{"felect", "invmc2", "autoa"}`` (both ``None`` for v1/v2 files).
v3 files are only written when the companion native export committed,
so the recorded native identity is defined.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 20260927
VERSION = 3
VERSIONS_SUPPORTED = (1, 2, 3)

_I4 = np.dtype("<i4")
_F8 = np.dtype("<f8")
_C16 = np.dtype("<c16")


@dataclass
class CsoType:
    """Per-species metadata needed to interpret the per-ion blocks."""

    lmmax: int
    lmax: int
    lps: list
    zcore: float
    zvalf_orig: float
    label: str


@dataclass
class CsoDump:
    nions: int
    ntyp: int
    lmdim_max: int
    nmax_max: int
    ncdij: int
    lsorbit: int
    saxis: np.ndarray
    alpha: float
    beta: float
    types: list = field(default_factory=list)
    ityp: np.ndarray = None
    nmax_ion: np.ndarray = None
    cso: np.ndarray = None  # (nions, 4, lmdim, lmdim) complex
    cocc: np.ndarray = None  # (nions, 4, lmdim, lmdim) complex
    potae: np.ndarray = None  # (nmax_max, nions) real
    provenance: dict = None  # band/k provenance dict, or None for v1 dumps
    potae_xcr: np.ndarray = None  # (nmax_max, nions) reference potential, v3
    constants: dict = None  # {"felect","invmc2","autoa"}, or None for v1/v2

    def label(self, ion: int) -> str:
        """Element label for a 0-based ion index."""
        return self.types[self.ityp[ion] - 1].label

    @property
    def labels(self):
        return [t.label for t in self.types]

    def type_of(self, ion: int) -> CsoType:
        """Per-species metadata for a 0-based ion index."""
        return self.types[self.ityp[ion] - 1]

    def lps(self, ityp: int):
        """Channel angular momenta for a 1-based species index (``ityp``)."""
        return list(self.types[ityp - 1].lps)

    def zcore(self, ityp: int) -> float:
        """Core charge for a 1-based species index (``ityp``)."""
        return self.types[ityp - 1].zcore

    def zvalf_orig(self, ityp: int) -> float:
        """Valence charge for a 1-based species index (``ityp``)."""
        return self.types[ityp - 1].zvalf_orig


class _Reader:
    def __init__(self, buf: bytes):
        self._buf = buf
        self._off = 0

    def get(self, dtype, count):
        n = dtype.itemsize * count
        if self._off + n > len(self._buf):
            raise ValueError(
                "tb2j_cso.bin truncated (needed %d bytes at %d)" % (n, self._off)
            )
        out = np.frombuffer(self._buf, dtype=dtype, count=count, offset=self._off)
        self._off += n
        return out

    def rest(self):
        return len(self._buf) - self._off


def read_cso_dump(path) -> CsoDump:
    """Read a ``tb2j_cso.bin`` dump written by ``TB2J_CSO_WRITE_DUMP``."""
    path = Path(path)
    r = _Reader(path.read_bytes())

    magic, version = r.get(_I4, 2)
    if int(magic) != MAGIC:
        raise ValueError(
            "bad magic %d (expected %d); not a tb2j_cso.bin dump" % (magic, MAGIC)
        )
    if int(version) not in VERSIONS_SUPPORTED:
        raise ValueError(
            "unsupported dump version %d (supported: %s)"
            % (version, ", ".join(str(v) for v in VERSIONS_SUPPORTED))
        )

    nions, ntyp, lmdim_max, nmax_max, ncdij, lsorbit = (int(x) for x in r.get(_I4, 6))
    saxis = r.get(_F8, 3).astype(np.float64)
    alpha, beta = (float(x) for x in r.get(_F8, 2))

    provenance = None
    if int(version) >= 2:
        ispin, nkpts, nbands, nb_tot = (int(x) for x in r.get(_I4, 4))
        (efermi,) = (float(x) for x in r.get(_F8, 1))
        native_magic, native_version = (int(x) for x in r.get(_I4, 2))
        if ispin not in (1, 2) or nkpts < 1 or nbands < 1 or nb_tot < nbands:
            raise ValueError(
                "inconsistent provenance header: ispin=%d nkpts=%d "
                "nbands=%d nb_tot=%d" % (ispin, nkpts, nbands, nb_tot)
            )
        provenance = dict(
            ispin=ispin,
            nkpts=nkpts,
            nbands=nbands,
            nb_tot=nb_tot,
            efermi=efermi,
            native_magic=native_magic,
            native_version=native_version,
            vkpt=None,
            wtkpt=None,
        )

    types = []
    for _ in range(ntyp):
        lmmax, lmax = (int(x) for x in r.get(_I4, 2))
        lps = [int(x) for x in r.get(_I4, lmax)]
        zcore, zvalf_orig = (float(x) for x in r.get(_F8, 2))
        label = r.get(np.dtype("S2"), 1)[0].decode("ascii")
        types.append(CsoType(lmmax, lmax, lps, zcore, zvalf_orig, label))

    ityp = r.get(_I4, nions).astype(np.int64)
    nmax_ion = r.get(_I4, nions).astype(np.int64)

    if int(version) >= 2:
        provenance["vkpt"] = (
            r.get(_F8, 3 * provenance["nkpts"])
            .reshape(provenance["nkpts"], 3)
            .astype(np.float64)
        )
        provenance["wtkpt"] = r.get(_F8, provenance["nkpts"]).astype(np.float64)
        if abs(float(provenance["wtkpt"].sum()) - 1.0) > 1.0e-8:
            raise ValueError(
                "provenance k weights sum to %.12f (expected 1 within "
                "1e-8)" % float(provenance["wtkpt"].sum())
            )

    # fail closed on internally inconsistent headers (STD-02)
    if nions <= 0 or ntyp <= 0:
        raise ValueError("degenerate dump: nions=%d ntyp=%d" % (nions, ntyp))
    if lmdim_max <= 0 or nmax_max <= 0:
        raise ValueError(
            "degenerate dims: lmdim_max=%d nmax_max=%d" % (lmdim_max, nmax_max)
        )
    if ncdij not in (1, 2, 4) or lsorbit not in (0, 1):
        raise ValueError(
            "inconsistent branch tags: ncdij=%d lsorbit=%d" % (ncdij, lsorbit)
        )
    for k, t in enumerate(types, start=1):
        if t.lmmax <= 0 or t.lmax < 1 or any(lp < 0 for lp in t.lps):
            raise ValueError(
                "type %d (%s): bad lmmax/lmax/lps %s"
                % (k, t.label, (t.lmmax, t.lmax, t.lps))
            )
        if t.lmmax != sum(2 * lp + 1 for lp in t.lps):
            raise ValueError(
                "type %d (%s): lmmax=%d inconsistent with lps %s (expected %d)"
                % (k, t.label, t.lmmax, t.lps, sum(2 * lp + 1 for lp in t.lps))
            )
        if t.lmmax > lmdim_max:
            raise ValueError(
                "type %d (%s): lmmax=%d exceeds lmdim_max=%d"
                % (k, t.label, t.lmmax, lmdim_max)
            )
    if nmax_ion.min() < 1 or nmax_ion.max() > nmax_max:
        raise ValueError(
            "nmax_ion out of range [%d, %d]" % (nmax_ion.min(), nmax_ion.max())
        )
    # Fortran (l1, l2, spinor, ion) with the row index l1 fastest; undo the
    # reversal so that cso[i, s, row, col] == Fortran CSO_STORE(row, col, s, i).
    cso = (
        r.get(_C16, nions * 4 * lmdim_max * lmdim_max)
        .reshape(nions, 4, lmdim_max, lmdim_max)
        .transpose(0, 1, 3, 2)
    )
    cocc = (
        r.get(_C16, nions * 4 * lmdim_max * lmdim_max)
        .reshape(nions, 4, lmdim_max, lmdim_max)
        .transpose(0, 1, 3, 2)
    )
    potae = r.get(_F8, nmax_max * nions).reshape(nions, nmax_max).T.copy()

    constants = None
    potae_xcr = None
    if int(version) >= 3:
        felect, invmc2, autoa = (float(x) for x in r.get(_F8, 3))
        constants = dict(felect=felect, invmc2=invmc2, autoa=autoa)
        potae_xcr = r.get(_F8, nmax_max * nions).reshape(nions, nmax_max).T.copy()

    if r.rest() != 0:
        raise ValueError("tb2j_cso.bin has %d trailing bytes; format drift?" % r.rest())

    if nions <= 0 or ntyp <= 0:
        raise ValueError("degenerate dump: nions=%d ntyp=%d" % (nions, ntyp))
    if ityp.min() < 1 or ityp.max() > ntyp:
        raise ValueError("ityp out of range")

    return CsoDump(
        nions=nions,
        ntyp=ntyp,
        lmdim_max=lmdim_max,
        nmax_max=nmax_max,
        ncdij=ncdij,
        lsorbit=lsorbit,
        saxis=saxis,
        alpha=alpha,
        beta=beta,
        types=types,
        ityp=ityp,
        nmax_ion=nmax_ion,
        cso=cso,
        cocc=cocc,
        potae=potae,
        provenance=provenance,
        potae_xcr=potae_xcr,
        constants=constants,
    )


def esoc_per_ion(dump: CsoDump, ion: int = None):
    """Per-ion ``E_soc = Re sum(CSO * conj(COCC))`` over same-l channel blocks.

    Mirrors ``CALC_SPINORB_MATRIX_ELEMENTS`` (VASP ``relativistic.F``) with the
    identical LM/LMP window accumulation, so the result is directly comparable
    with the ``Spin-Orbit-Coupling matrix elements`` block in OUTCAR.
    """
    if ion is not None:
        return _esoc_one(dump, ion)
    return np.array([_esoc_one(dump, i) for i in range(dump.nions)])


def _esoc_one(dump: CsoDump, ion: int) -> float:
    t = dump.type_of(ion)
    lps = t.lps
    cso = dump.cso[ion]
    cocc = dump.cocc[ion]
    total = 0.0 + 0.0j
    lm = 0  # lm offset of the outer (column) channel, Fortran LM
    for l1 in lps:
        lmp = 0  # lm offset of the inner (row) channel, Fortran LMP
        for l2 in lps:
            if l1 == l2:
                rows = slice(lmp, lmp + 2 * l2 + 1)
                cols = slice(lm, lm + 2 * l1 + 1)
                blk_s = cso[:, rows, cols]
                blk_o = cocc[:, rows, cols]
                total += np.sum(blk_s * blk_o.conj())
            lmp += 2 * l2 + 1
        lm += 2 * l1 + 1
    return float(total.real)


_ESOC_RE = re.compile(r"Ion:\s+(\d+)\s+E_soc:\s+(-?\d+\.?\d*(?:[Ee][+-]?\d+)?)")


def parse_outcar_esoc(path):
    """Parse per-ion ``E_soc`` from OUTCAR (last SOC block wins)."""
    text = Path(path).read_text(errors="replace")
    marker = "Spin-Orbit-Coupling matrix elements"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError("no '%s' block in %s (not an LSORBIT run?)" % (marker, path))
    pairs = _ESOC_RE.findall(text[idx + len(marker) :])
    if not pairs:
        raise ValueError("SOC block found but no 'Ion: N E_soc:' lines in %s" % path)
    ions = [int(i) for i, _ in pairs]
    if ions != list(range(1, len(pairs) + 1)):
        raise ValueError("non-contiguous ion indices %s in %s" % (ions, path))
    return [float(v) for _, v in pairs]
