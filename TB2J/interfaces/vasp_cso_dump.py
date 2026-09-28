"""Source-neutral reader for the VASP ``tb2j_cso.bin`` dump (format v1).

Vendored copy of ``VASP_TB2J_patch/python/vasp_cso_dump.py`` (story 010,
format v1, magic 20260927) so that TB2J has no runtime dependency on the
patch repository.  Format changes require synchronized updates in both
places; the v1 layout is frozen upstream.

Layout (little-endian raw stream, written by ``TB2J_CSO_WRITE_DUMP``)::

    i4 magic=20260927, i4 version=1
    i4 nions, ntyp, lmdim_max, nmax_max, ncdij, lsorbit
    f64 saxis(3), f64 alpha (PHI argument), f64 beta (THETA argument)
    per type: i4 lmmax, i4 lmax, i4 lps(lmax), f64 zcore, f64 zvalf_orig,
              char(2) label
    i4 ityp(nions), i4 nmax_ion(nions)
    c16 cso  (l1, l2, slot, ion)   row=l1 fastest -> [ion, slot, row, col]
    c16 cocc (l1, l2, slot, ion)   same order
    f64 potae (nmax_max, nions)    raw 2*sqrt(pi) spherical AE potential

``cso``/``cocc`` slots are the (uu, ud, du, dd) spinor blocks with
row=CH2 (bra l' channel) and col=CH1 (ket l channel).  The collinear
strength-0 capture stores the raw collinear augmentation occupations in
slots uu/dd (``CRHODE`` components 1/2) and the ``SPINORB_STRENGTH``
operator evaluated at the SAXIS Euler angles in ``cso``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 20260927
VERSION = 1

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
    if int(version) != VERSION:
        raise ValueError(
            "unsupported dump version %d (expected %d)" % (version, VERSION)
        )

    nions, ntyp, lmdim_max, nmax_max, ncdij, lsorbit = (int(x) for x in r.get(_I4, 6))
    saxis = r.get(_F8, 3).astype(np.float64)
    alpha, beta = (float(x) for x in r.get(_F8, 2))

    types = []
    for _ in range(ntyp):
        lmmax, lmax = (int(x) for x in r.get(_I4, 2))
        lps = [int(x) for x in r.get(_I4, lmax)]
        zcore, zvalf_orig = (float(x) for x in r.get(_F8, 2))
        label = r.get(np.dtype("S2"), 1)[0].decode("ascii")
        types.append(CsoType(lmmax, lmax, lps, zcore, zvalf_orig, label))

    ityp = r.get(_I4, nions).astype(np.int64)
    nmax_ion = r.get(_I4, nions).astype(np.int64)

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
