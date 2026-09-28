"""Story 011 tests: VASP KS-basis split-SOC adapter (tangent contract).

Covers:
- TEST-001: assembled W^K_SO Hermiticity + independent CALC_PAW_OVERLAP
  expected value + SAXIS frame handling: each run is one psi-gauge
  magnetic reference (states are SAXIS-frame eigenstates, vertices
  ``Delta sigma_z``), so the tangent kernel's raw ``J_leg`` must be
  identical across input SAXIS at lam=0 and — the fixture re-expresses
  one physical CSO — match at lam=1 too, with the longitudinal row/
  column exactly masked and ``mask_residual ~ 0``.
- SOC-off anchor: at lam=0 the leg's measured transverse block reduces
  to the isotropic collinear exchange (``J_leg[u,u] = J_leg[v,v] =
  Jiso_collinear`` through the O map).
- COCC reconstruction from the collinear CPROJ against the patch dump
  (CRHODE(LP,L) = conj(CPROJ(LP)) CPROJ(L) fast_aug order; the real-dump
  residual documents the patch's occupation-update skew).
- Dump format v1/v2/v3: v2 band/k provenance (dump-vs-native identity
  gates against the real Ni/FeO story-010 dumps) and v3
  reference-potential provenance (constants + ``potae_xcr``).
- All-atom ligand SOC influence on the measured transverse block (real
  centrosymmetric FeO dump and a non-centrosymmetric synthetic cell).
- Driver/CLI end-to-end over THREE independent x/y/z SAXIS references:
  lattice-frame rotation via ``rotate_transverse_leg`` and rank-nine
  raw-tensor merge via ``merge_transverse_legs`` (never io_merge),
  persisted provenance, and the real retained FeO campaign
  (collinear_x/y/z v2 dump+native triples).
"""

import json
import os
import sys

import numpy as np
import pytest

from TB2J.interfaces.vasp_cso_dump import MAGIC, read_cso_dump
from TB2J.interfaces.vasp_native import read_vasp_native

FE_LPS = [2, 0]  # Fe: d radial, s radial -> 5 + 1 = 6 projectors
O_LPS = [1, 0]  # O: p radial, s radial -> 3 + 1 = 4 projectors
NKPT = 4
NBAND = 4
HARTREE_TO_EV = 27.211386245988

FEO_DIR = os.environ.get(
    "TB2J_FEO_SPLITSOC_DIR",
    "/home/hexu/projects/TB2J_dev/.tmp/story-010-feo-oracle/collinear",
)

SLOT = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

_LPS_BY_SYMBOL = {"Fe": FE_LPS, "O": O_LPS}
_Z_BY_SYMBOL = {"Fe": 26.0, "O": 8.0}


def _ion_layout(symbols):
    """Per-ion channel layout for the given species sequence."""
    return [
        {
            "symbol": s,
            "lps": _LPS_BY_SYMBOL[s],
            "lmmax": sum(2 * l + 1 for l in _LPS_BY_SYMBOL[s]),
        }
        for s in symbols
    ]


def _projector_layout(symbols=("Fe", "O")):
    """Channel layout keyed by ion index (VASP LPS per-channel l)."""
    return {i: dict(v) for i, v in enumerate(_ion_layout(symbols))}


def _ordered_types(symbols):
    """Unique species in first-appearance order (VASP type table)."""
    return list(dict.fromkeys(symbols))


# ---------------------------------------------------------------------------
# synthetic collinear v5 native export + tb2j_cso.bin writers
# ---------------------------------------------------------------------------
def _write_test_native(
    path, cproj, cdij, eigenvalues, occupations, symbols=("Fe", "O"), positions=None
):
    """Write a v5 collinear native export with the given spectral arrays."""
    layout = _ion_layout(symbols)
    types = _ordered_types(symbols)
    nband, nkpt, nspin = eigenvalues.shape
    nprod = cproj.shape[0]
    nions, ntyp = len(layout), len(types)
    lmdim_max = max(v["lmmax"] for v in layout)
    lmax_max = 2
    offsets = [0]
    for v in layout[:-1]:
        offsets.append(offsets[-1] + v["lmmax"])
    if positions is None:
        positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]][:nions]
    positions = np.asarray(positions, dtype=float)
    kpts_3d = np.array([[i * 0.5, j * 0.5, 0.0] for i in range(2) for j in range(2)])
    wts = np.ones(nkpt) / nkpt

    with open(path, "wb") as f:

        def wi(value):
            f.write(np.array(value, dtype="<i4").tobytes())

        def wr(value):
            f.write(np.asarray(value, dtype="<f8").tobytes(order="F"))

        def wc(value):
            f.write(np.asarray(value).astype("<c16").tobytes(order="F"))

        wi(20260812)
        wi(5)
        wi(nspin)
        wi(2)  # ncdij
        wi(nkpt)
        wi(nband)
        wi(nprod)
        wi(nions)
        wi(ntyp)
        wi(lmdim_max)
        wi(lmax_max)
        wr(np.eye(3) * 4.0)
        wr(positions.T)
        wi(offsets)
        wi([v["lmmax"] for v in layout])
        wi([types.index(v["symbol"]) + 1 for v in layout])
        wi([max(v["lmmax"] for v in layout if v["symbol"] == t) for t in types])
        lps = np.zeros((lmax_max, ntyp), dtype=int)
        for ityp, t in enumerate(types):
            lps[: len(_LPS_BY_SYMBOL[t]), ityp] = _LPS_BY_SYMBOL[t]
        f.write(np.asarray(lps, dtype="<i4").tobytes(order="F"))
        qtot = np.zeros((lmax_max, lmax_max, ntyp))
        for ityp in range(ntyp):
            for a in range(lmax_max):
                qtot[a, a, ityp] = 1.0
        wr(qtot)
        wr([_Z_BY_SYMBOL[t] for t in types])
        for t in types:
            f.write(t.ljust(2).encode("ascii"))
        wr(kpts_3d.T)
        wr(wts)
        wr(-1.5)
        wr(eigenvalues)  # (nband, nkpt, nspin) Fortran
        wr(occupations)
        wc(cproj)  # (nprod, nband, nkpt, nspin) Fortran
        wc(cdij)  # (lmdim_max, lmdim_max, nions, ncdij) Fortran
    return path


def _random_hermitian(n, rng, scale=1.0):
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a = 0.5 * (a + a.conj().T)
    return a * scale / max(1.0, float(np.abs(a).max()))


def _write_cso_dump(
    path,
    cso,
    cocc,
    potae=None,
    saxis=(0.0, 0.0, 1.0),
    alpha=0.0,
    beta=0.0,
    lsorbit=0,
    symbols=("Fe", "O"),
    version=1,
    constants=None,
    potae_xcr=None,
    wtkpt=None,
):
    """Serialize a tb2j_cso.bin stream (test-side format writer).

    ``version=2`` adds the provenance block (ispin/nkpts/nbands/nb_tot/
    efermi/native MAGIC+version) and the per-k vkpt/wtkpt arrays, exactly
    as the story-010 v2 Fortran writer does.  ``version=3`` additionally
    appends the reference-potential constants (felect, invmc2, autoa)
    after the potae array and the per-ion ``potae_xcr`` grid, zero-padded
    to ``nmax_max``.
    """
    layout = _ion_layout(symbols)
    types = _ordered_types(symbols)
    lmdim_max = max(v["lmmax"] for v in layout)
    nmax_max = lmdim_max
    with open(path, "wb") as f:

        def wi(value):
            f.write(np.array(value, dtype="<i4").tobytes())

        def wr(value):
            f.write(np.asarray(value, dtype="<f8").tobytes(order="F"))

        def wstream(value):
            f.write(np.asarray(value).astype("<c16").tobytes())

        wi(MAGIC)
        wi(version)
        wi(len(layout))  # nions
        wi(len(types))  # ntyp
        wi(lmdim_max)
        wi(nmax_max)
        wi(2)  # ncdij
        wi(lsorbit)
        wr(np.asarray(saxis, dtype=float))
        wr([alpha, beta])
        if version >= 2:
            wi([2, NKPT, NBAND, NBAND])  # ispin, nkpts, nbands, nb_tot
            wr(-1.5)  # efermi
            wi([20260812, 5])  # companion native MAGIC/version
        for t in types:
            lay_lps = _LPS_BY_SYMBOL[t]
            lmmax = sum(2 * l + 1 for l in lay_lps)
            wi(lmmax)
            wi(len(lay_lps))
            wi(lay_lps)
            wr([10.0, 8.0])
            f.write(t.ljust(2).encode("ascii"))
        wi([types.index(v["symbol"]) + 1 for v in layout])  # ityp
        wi([v["lmmax"] for v in layout])  # nmax_ion
        if version >= 2:
            kpts_3d = np.array(
                [[i * 0.5, j * 0.5, 0.0] for i in range(2) for j in range(2)]
            )
            wr(kpts_3d.T)
            wr(np.ones(NKPT) / NKPT if wtkpt is None else np.asarray(wtkpt))
        wstream(cso)
        wstream(cocc)
        if potae is None:
            potae = np.ones((nmax_max, len(layout)))
        f.write(np.ascontiguousarray(np.asarray(potae).T).tobytes())
        if version >= 3:
            if constants is None:
                constants = {"felect": 0.5, "invmc2": 7.45596e-6, "autoa": 0.529177}
            wr([constants["felect"], constants["invmc2"], constants["autoa"]])
            if potae_xcr is None:
                potae_xcr = 2.0 * np.ones((nmax_max, len(layout)))
            f.write(np.ascontiguousarray(np.asarray(potae_xcr).T).tobytes())
    return path


def _make_synthetic_system(
    tmp_path,
    saxis=(0.0, 0.0, 1.0),
    seed=7,
    symbols=("Fe", "O"),
    positions=None,
    cso_scale=1.0,
):
    """Build a synthetic collinear export + matching CSO dump.

    Returns (native_path, dump_path, snapshot, dump).  The CSO operator is
    generated in the z spin frame and, for SAXIS != z, re-expressed with
    VASP's own EULER/ROTMAT spinor-frame map (independent port of
    SETUP_LS's ROTMAT), so the band data are identical in both cases while
    the dump-frame operator representation rotates.
    """
    from TB2J.interfaces.vasp_split_soc import vasp_spinor_frame

    rng = np.random.default_rng(seed)
    layout = _ion_layout(symbols)
    nions = len(layout)
    lmdim = 6
    nprod = sum(v["lmmax"] for v in layout)
    nspin, nkpt, nband = 2, NKPT, NBAND

    base = np.sort(rng.uniform(0.0, 1.0, size=(nkpt, nband)), axis=1)
    eigenvalues = np.empty((nband, nkpt, nspin), order="F")
    for s in (0, 1):
        eigenvalues[:, :, s] = (base + s * 0.8).T  # exchange splitting
    occupations = np.zeros((nband, nkpt, nspin), order="F")
    occupations[:3, :, :] = 1.0

    cproj = (
        rng.normal(size=(nprod, nband, nkpt, nspin))
        + 1j * rng.normal(size=(nprod, nband, nkpt, nspin))
    ) * 0.3
    cproj = np.asfortranarray(cproj)

    cdij = np.zeros((lmdim, lmdim, nions, 2), dtype=complex, order="F")
    for ion, lay in enumerate(layout):
        npj = lay["lmmax"]
        cdij[:npj, :npj, ion, 0] = _random_hermitian(npj, rng)
        cdij[:npj, :npj, ion, 1] = _random_hermitian(npj, rng, scale=0.5)

    native = _write_test_native(
        tmp_path / "tb2j_native.bin",
        cproj,
        cdij,
        eigenvalues,
        occupations,
        symbols=symbols,
        positions=positions,
    )
    snapshot = read_vasp_native(native)

    # CSO in the z spin frame: L.S-like operator (same-l blocks with
    # l >= 1, du == ud^dagger, dd == -uu), then re-expressed in the SAXIS
    # frame through the VASP ROTMAT if SAXIS != z.
    cso_z = np.zeros((nions, 4, lmdim, lmdim), dtype=complex)
    for ion, lay in enumerate(layout):
        offs = []
        o = 0
        for l in lay["lps"]:
            offs.append((l, o))
            o += 2 * l + 1
        for l1, o1 in offs:
            for l2, o2 in offs:
                n1, n2 = 2 * l1 + 1, 2 * l2 + 1
                if l1 != l2 or l1 == 0:
                    continue
                uu = _random_hermitian(n1, rng) * cso_scale
                ud = (
                    rng.normal(size=(n1, n2)) + 1j * rng.normal(size=(n1, n2))
                ) * cso_scale
                cso_z[ion, SLOT[(0, 0)], o1 : o1 + n1, o2 : o2 + n2] = uu
                cso_z[ion, SLOT[(1, 1)], o1 : o1 + n1, o2 : o2 + n2] = -uu
                cso_z[ion, SLOT[(0, 1)], o1 : o1 + n1, o2 : o2 + n2] = ud
                cso_z[ion, SLOT[(1, 0)], o1 : o1 + n1, o2 : o2 + n2] = ud.conj().T

    saxis = np.asarray(saxis, dtype=float)
    nrm = np.linalg.norm(saxis)
    if nrm < 1e-10:
        alpha = beta = 0.0
        cso = cso_z
        saxis = np.array([0.0, 0.0, 1.0])
    else:
        saxis = saxis / nrm
        beta = float(np.arctan2(np.hypot(saxis[0], saxis[1]), saxis[2]))
        alpha = float(np.arctan2(saxis[1], saxis[0]))
        u = vasp_spinor_frame(alpha, beta)
        # spinor rotation mixes the (uu, ud, du, dd) slot index only:
        # the frame representation of one physical operator,
        # CSO'[c, d] = sum_{a, b} conj(U[a, c]) U[b, d] CSO[a, b]
        # (U columns are the frame basis spinors in z components), so the
        # state-space W obeys W_n = (kron(U, I))^dag W_z (kron(U, I)) and
        # every SAXIS run is an exact SU(2) image of the z reference.
        full = cso_z.reshape(nions, 2, 2, lmdim, lmdim)
        rot = np.einsum("ac,bd,iabpq->icdpq", u.conj(), u, full)
        cso = rot.reshape(nions, 4, lmdim, lmdim)

    # COCC from the CPROJ/occupations with the pinned weight convention:
    # CRHODE(LP, L) = sum_k w_k f_nk conj(CPROJ(LP)) CPROJ(L) — the VASP
    # fast_aug.F outer-product order — accumulated with a plain loop
    # (separate code path from the adapter).
    local = snapshot.coefficients  # (nspin, nkpt, nband, nprod) phase-removed
    cocc = np.zeros((nions, 4, lmdim, lmdim), dtype=complex)
    offsets = [0]
    for lay in layout[:-1]:
        offsets.append(offsets[-1] + lay["lmmax"])
    for ion, lay in enumerate(layout):
        sl = slice(offsets[ion], offsets[ion] + lay["lmmax"])
        for s, slot in ((0, 0), (1, 3)):
            blk = np.zeros((lmdim, lmdim), dtype=complex)
            for k in range(nkpt):
                for m in range(nband):
                    amp = snapshot.weights[k] * snapshot.occupations[s, k, m]
                    vec = local[s, k, m, sl]
                    size = len(vec)
                    # written transposed: read_cso_dump undoes the Fortran
                    # (l1,l2) reversal, so the stored block is conj(CPROJ)
                    # x CPROJ order as reconstruct_cocc computes it
                    blk[:size, :size] += amp * np.outer(vec, vec.conj())
            cocc[ion, slot] = blk

    dump_path = _write_cso_dump(
        tmp_path / "tb2j_cso.bin",
        cso=cso,
        cocc=cocc,
        saxis=saxis,
        alpha=alpha,
        beta=beta,
        symbols=symbols,
    )
    dump = read_cso_dump(dump_path)
    return native, dump_path, snapshot, dump


def _reference_w_soc(cproj, dump):
    """Independent element-by-element CALC_PAW_OVERLAP contraction.

    ``cproj``: (nspin, nkpt, nband, nprod) local (phase-removed) VASP
    coefficients.  States are stacked spin-major: nu = spin*nband + band.
    All loops are explicit (no einsum) so the reference shares no code
    path with the adapter.
    """
    nions = dump.nions
    offsets = {}
    off = 0
    for ion in range(nions):
        offsets[ion] = off
        off += dump.type_of(ion).lmmax
    nspin, nkpt, nband = cproj.shape[:3]
    nstate = nspin * nband
    w = np.zeros((nkpt, nstate, nstate), dtype=complex)
    for k in range(nkpt):
        for ion in range(nions):
            npj = dump.type_of(ion).lmmax
            for p in range(npj):
                for q in range(npj):
                    for s_nu in (0, 1):
                        for m_nu in range(nband):
                            for s_nw in (0, 1):
                                for m_nw in range(nband):
                                    # state (s, m) is a SAXIS-frame eigenstate:
                                    # only its own spinor component is nonzero
                                    v = dump.cso[ion, SLOT[(s_nu, s_nw)], p, q]
                                    nu = s_nu * nband + m_nu
                                    nw = s_nw * nband + m_nw
                                    bp = cproj[s_nu, k, m_nu, offsets[ion] + p]
                                    bq = cproj[s_nw, k, m_nw, offsets[ion] + q]
                                    w[k, nu, nw] += np.conj(bp) * v * bq
    return w


RPTS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])


# ---------------------------------------------------------------------------
# TEST-001: Hermiticity, independent expected value, SAXIS covariance
# ---------------------------------------------------------------------------
def test_w_soc_hermitian_and_matches_independent_contraction(tmp_path):
    """W^K per k equals the element-wise B^dag CSO B reference; Hermitian."""
    from TB2J.interfaces.vasp_split_soc import build_w_soc

    _, _, snapshot, dump = _make_synthetic_system(tmp_path)

    w = build_w_soc(snapshot, dump)
    ref = _reference_w_soc(snapshot.coefficients, dump)

    assert w.shape == ref.shape
    assert np.allclose(w, ref, atol=1e-12)
    assert np.abs(w - w.conj().transpose(0, 2, 1)).max() < 1e-12


def test_leg_pipeline_saxis_covariant(tmp_path):
    """Frame handling of the SAXIS dump representation (psi-gauge legs).

    Each run's collinear states are SAXIS-frame eigenstates (vertices
    ``Delta sigma_z`` — a z magnetic reference for every run), and its
    dump CSO is the state-space SOC operator re-expressed in that frame.
    Two exact consequences are pinned:

    1. lam=0: the strength-0 problem is W-independent and spin-rotation
       invariant, so the RAW ``J_leg`` is identical across input SAXIS.
    2. lam=1: a SAXIS run is the z reference with the SOC operator
       conjugated into its frame (``W_n = U_n^dag W_z U_n``, the exact
       VASP EULER/ROTMAT representation map), so its raw ``J_leg`` must
       equal the z-reference leg computed with that ``W_soc`` override.

    Every leg's longitudinal row/column is exactly masked with
    ``mask_residual`` at the numerical noise level, and the detected leg
    frame is the z reference with the positive Cartesian axis.
    """
    from TB2J.interfaces.vasp_split_soc import compute_split_soc_exchange_leg

    def build(tag):
        sub = tmp_path / tag
        sub.mkdir(exist_ok=True)
        return _make_synthetic_system(
            sub, saxis=(0.0, 0.0, 1.0) if tag == "z" else (1.0, 2.0, 1.5)
        )

    def raw_leg(snapshot, dump, lam, w_override=None):
        res = compute_split_soc_exchange_leg(
            snapshot,
            dump,
            lam=lam,
            sites=(0,),
            rpts=RPTS,
            nz=24,
            w_override=w_override,
        )
        entry = res["exchange"][((1, 0, 0), 0, 0)]
        return res, entry

    for lam in (0.0, 1.0):
        native_z, _, snapshot_z, dump_z = build("z")
        del native_z
        res_z, entry_z = raw_leg(snapshot_z, dump_z, lam)
        native_n, _, snapshot_n, dump_n = build("n")
        del native_n
        res_n, entry_n = raw_leg(snapshot_n, dump_n, lam)
        # both legs are z magnetic references on their own triad
        for entry in (entry_z, entry_n):
            frame = entry["frame"]
            assert int(frame["n"]) == 2
            assert np.allclose(frame["axis"], [0.0, 0.0, 1.0], atol=1e-12)
            j_leg = np.asarray(entry["J_leg"], dtype=float)
            assert j_leg[2, :].max() == 0.0 and j_leg[:, 2].max() == 0.0
            # campaign pin (FR PAW real-data worst 3.3e-10)
            assert float(entry["mask_residual"]) < 1e-8
        # lam=0: W-independent strength-0 covariance across input SAXIS
        if lam == 0.0:
            diff = np.abs(
                np.asarray(entry_z["J_leg"], dtype=float)
                - np.asarray(entry_n["J_leg"], dtype=float)
            ).max()
            assert diff < 1e-10, diff
        else:
            # lam=1: the n-run is the z reference with W re-expressed
            # (W_n = U_n^dag W_z U_n); its raw leg must reproduce the
            # z-frame leg driven with that W_soc override
            _, entry_z_with_wn = raw_leg(
                snapshot_z, dump_z, lam, w_override=res_n["w_soc"]
            )
            diff = np.abs(
                np.asarray(entry_n["J_leg"], dtype=float)
                - np.asarray(entry_z_with_wn["J_leg"], dtype=float)
            ).max()
            assert diff < 1e-10, diff

    # the representation map itself, at the dump slot level: the n-run's
    # CSO is the z-reference operator re-expressed in the SAXIS frame,
    # CSO_n[c, d] = sum_{a,b} conj(U[a, c]) U[b, d] CSO_z[a, b]
    # (pinned channels: the collinear CPROJ entries are frame components,
    # so no state-space conjugation is applied)
    from TB2J.interfaces.vasp_split_soc import vasp_spinor_frame

    native_z, _, snapshot_z, dump_z = build("z")
    del native_z
    native_n, _, snapshot_n, dump_n = build("n")
    del native_n
    u = vasp_spinor_frame(float(dump_n.alpha), float(dump_n.beta))
    for ion in range(dump_z.nions):
        lmm = dump_z.type_of(ion).lmmax
        a = dump_z.cso[ion, :, :lmm, :lmm].reshape(2, 2, lmm, lmm)
        expect = np.einsum("ac,bd,abpq->cdpq", u.conj(), u, a).reshape(4, lmm, lmm)
        got = dump_n.cso[ion, :, :lmm, :lmm]
        assert np.abs(got - expect).max() < 1e-12, ion

    # each run's SO(3) map takes z to its own SAXIS direction
    saxis_n = np.asarray(dump_n.saxis, dtype=float)
    saxis_n /= np.linalg.norm(saxis_n)
    res_z, _ = raw_leg(snapshot_z, dump_z, 0.0)
    res_n, _ = raw_leg(snapshot_n, dump_n, 0.0)
    assert np.allclose(res_n["o_map"] @ [0.0, 0.0, 1.0], saxis_n, atol=1e-12)
    assert np.allclose(res_z["o_map"], np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# TEST-002: SOC-off anchor against the existing collinear v5/v6 kernel
# ---------------------------------------------------------------------------
def test_soc_off_anchor_matches_collinear_exchange(tmp_path):
    """At lam=0 the measured transverse block is the isotropic collinear J.

    The strength-0 problem is spin-rotation invariant, so the tangent
    kernel's transverse 2x2 of the z reference must be ``J * I_2`` with
    ``J`` the existing collinear v5/v6 scalar kernel's Jiso, and the
    lattice-frame rotation must leave it invariant.
    """
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.interfaces.vasp_split_soc import (
        compute_split_soc_exchange_leg,
        rotate_transverse_leg,
    )
    from TB2J.paw_projector import build_projector_green_data

    _, _, snapshot, dump = _make_synthetic_system(tmp_path)

    res = compute_split_soc_exchange_leg(
        snapshot,
        dump,
        lam=0.0,
        sites=(0, 1),
        rpts=RPTS,
        nz=24,
    )
    rotated = rotate_transverse_leg(res["exchange"], res["o_map"], 2)

    data = build_projector_green_data(snapshot)
    jdict = compute_projector_exchange_jdict(
        data, Rpts=RPTS, nz=24, smearing_eV=0.05, sites=(0, 1)
    )
    for (r, i, j), jiso_collinear in jdict.items():
        entry = rotated[(tuple(r), i, j)]
        j_leg = np.asarray(entry["J_leg"], dtype=float)
        assert np.isclose(j_leg[0, 0], jiso_collinear, atol=1e-10)
        assert np.isclose(j_leg[1, 1], jiso_collinear, atol=1e-10)
        # transverse off-diagonals vanish at the contour-noise level
        # (observed ~2.2e-10 eV; campaign pin < 1e-8)
        assert np.abs(j_leg[0, 1]) < 1e-8 and np.abs(j_leg[1, 0]) < 1e-8


# ---------------------------------------------------------------------------
# COCC reconstruction
# ---------------------------------------------------------------------------
def test_cocc_reconstruction_matches_dump(tmp_path):
    from TB2J.interfaces.vasp_split_soc import reconstruct_cocc

    _, _, snapshot, dump = _make_synthetic_system(tmp_path)
    recon = reconstruct_cocc(snapshot, dump)
    assert np.abs(recon[:, 1, :, :]).max() == 0.0
    assert np.abs(recon[:, 2, :, :]).max() == 0.0
    assert np.abs(recon[:, 0] - dump.cocc[:, 0]).max() < 1e-10
    assert np.abs(recon[:, 3] - dump.cocc[:, 3]).max() < 1e-10


# ---------------------------------------------------------------------------
# Real FeO dump: CSO structure, COCC reconstruction, ligand SOC, driver
# ---------------------------------------------------------------------------
def _feo_paths():
    native = os.path.join(FEO_DIR, "tb2j_native.bin")
    dump = os.path.join(FEO_DIR, "tb2j_cso.bin")
    if not (os.path.exists(native) and os.path.exists(dump)):
        return None
    return native, dump


def test_real_feo_cso_structure_and_cocc_reconstruction():
    """COCC reconstruction against the retained (pre-R1, v1) FeO dump.

    NOTE: this archived dump predates the story-010 R1 POTAE semantics —
    it is a valid v1 fixture for the reconstruction convention but must
    NOT be pinned as a v2 regression reference (Main, S10 v2 handoff).
    """
    paths = _feo_paths()
    if paths is None:
        pytest.skip("real FeO story-010 dump not available")
    from TB2J.interfaces.vasp_split_soc import reconstruct_cocc

    snapshot = read_vasp_native(paths[0])
    dump = read_cso_dump(paths[1])
    assert dump.ncdij == 2 and dump.lsorbit == 0

    # Hermitian operator structure: uu/dd Hermitian, du == ud^dagger
    for ion in range(dump.nions):
        cso = dump.cso[ion]
        n = dump.type_of(ion).lmmax
        for s in (0, 3):
            assert np.abs(cso[s, :n, :n] - cso[s, :n, :n].conj().T).max() < 1e-10
        assert np.abs(cso[2, :n, :n] - cso[1, :n, :n].conj().T).max() < 1e-10

    recon = reconstruct_cocc(snapshot, dump)
    for ion in range(dump.nions):
        n = dump.type_of(ion).lmmax
        for slot in (0, 3):
            scale = max(1.0, float(np.abs(dump.cocc[ion, slot, :n, :n]).max()))
            residual = np.abs(
                recon[ion, slot, :n, :n] - dump.cocc[ion, slot, :n, :n]
            ).max()
            # observed skew <= 1.5e-5 x scale (dd slot): the story-010 patch
            # dumps COCC one occupation update away from the exported band
            # occupations; a convention or weight error would sit at O(1)
            assert residual < 2e-5 * scale


def test_real_feo_ligand_soc_influence(tmp_path):
    paths = _feo_paths()
    if paths is None:
        pytest.skip("real FeO story-010 dump not available")
    from TB2J.interfaces.vasp_split_soc import (
        build_w_soc,
        compute_split_soc_exchange_leg,
    )

    snapshot = read_vasp_native(paths[0])
    dump = read_cso_dump(paths[1])

    kwargs = dict(
        sites=(0,),
        rpts=RPTS,
        nz=24,
    )
    res_full = compute_split_soc_exchange_leg(snapshot, dump, lam=1.0, **kwargs)
    w_nolig = build_w_soc(snapshot, dump, skip_atoms=(1,))
    w_full = build_w_soc(snapshot, dump)
    # direct operator-level gate: the O one-center CSO enters the all-atom
    # W^K_SO at the 10% level of its scale (0.0101 vs 0.098 eV observed)
    assert np.abs(w_nolig - w_full).max() > 1e-3 * np.abs(w_full).max()

    res_nolig = compute_split_soc_exchange_leg(
        snapshot, dump, lam=1.0, w_override=w_nolig, **kwargs
    )
    key = ((1, 0, 0), 0, 0)
    full, nolig = res_full["exchange"][key], res_nolig["exchange"][key]
    # the measured transverse scalar channel (uu of the z-leg triad)
    # carries the ligand SOC influence
    full_uu = float(np.asarray(full["J_leg"], dtype=float)[0, 0])
    nolig_uu = float(np.asarray(nolig["J_leg"], dtype=float)[0, 0])
    # Ligand SOC measurably shifts the symmetry-allowed scalar channel
    # (observed shift 2.0e-7 eV on J ~ 7.4e-3 eV; noise floor ~1e-15).
    assert abs(full_uu - nolig_uu) > 1e-9
    # magnetic vertices untouched: the strength-0 onsite splitting is the
    # same to the ligand's second-order correction
    assert np.isclose(full_uu, nolig_uu, rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# v2 dump dispatch + dump/native identity
# ---------------------------------------------------------------------------
def test_cso_v2_synthetic_round_trip(tmp_path):
    """v2 dumps carry the provenance block and k arrays; v1 fields intact."""
    from TB2J.interfaces.vasp_cso_dump import read_cso_dump

    sub_v1 = tmp_path / "v1"
    sub_v1.mkdir()
    _, _, _, dump_v1 = _make_synthetic_system(sub_v1)
    sub = tmp_path / "v2"
    sub.mkdir()
    v2_path = sub / "tb2j_cso_v2.bin"
    # the writer takes RAW (l1, l2, slot, ion) arrays as stored on disk;
    # the reader output is the transposed view, so transpose back
    _write_cso_dump(
        v2_path,
        cso=dump_v1.cso.transpose(0, 1, 3, 2),
        cocc=dump_v1.cocc.transpose(0, 1, 3, 2),
        saxis=tuple(float(x) for x in dump_v1.saxis),
        alpha=dump_v1.alpha,
        beta=dump_v1.beta,
        symbols=("Fe", "O"),
        version=2,
    )
    dump_v2 = read_cso_dump(v2_path)
    assert dump_v2.provenance is not None
    assert dump_v2.provenance["ispin"] == 2
    assert dump_v2.provenance["nkpts"] == NKPT
    assert dump_v2.provenance["nbands"] == NBAND
    assert dump_v2.provenance["nb_tot"] == NBAND
    assert dump_v2.provenance["efermi"] == -1.5
    assert dump_v2.provenance["native_magic"] == 20260812
    assert dump_v2.provenance["native_version"] == 5
    assert dump_v2.provenance["vkpt"].shape == (NKPT, 3)
    assert dump_v2.provenance["wtkpt"].shape == (NKPT,)
    assert np.isclose(dump_v2.provenance["wtkpt"].sum(), 1.0)
    assert np.abs(dump_v2.cso - dump_v1.cso).max() == 0.0
    assert np.abs(dump_v2.cocc - dump_v1.cocc).max() == 0.0
    assert dump_v1.provenance is None


NISOC_V2_DUMP = "/home/hexu/projects/TB2J_dev/.tmp/story-010-ni/nisoc/tb2j_cso.bin"
NISOC_V2_OUTCAR = "/home/hexu/projects/TB2J_dev/.tmp/story-010-ni/nisoc/OUTCAR"
FEO_V2_Z_DUMP = (
    "/home/hexu/projects/TB2J_dev/.tmp/story-010-feo-oracle/collinear_z/tb2j_cso.bin"
)
FEO_V2_Z_NATIVE = (
    "/home/hexu/projects/TB2J_dev/.tmp/story-010-feo-oracle/collinear_z/tb2j_native.bin"
)


def test_real_v2_ni_soc_dump_esoc_oracle():
    """Behavior gate: the real S10 Ni v2 dump reproduces the OUTCAR E_soc.

    Certified by S10: dump -0.08188974 eV vs OUTCAR -0.0818897 eV.
    """
    from TB2J.interfaces.vasp_cso_dump import (
        esoc_per_ion,
        parse_outcar_esoc,
        read_cso_dump,
    )

    if not os.path.exists(NISOC_V2_DUMP):
        pytest.skip("real Ni v2 dump not available")
    dump = read_cso_dump(NISOC_V2_DUMP)
    assert dump.provenance is not None
    assert dump.provenance["ispin"] == 1
    assert dump.provenance["nkpts"] == 36
    assert dump.provenance["native_version"] == 7
    assert dump.label(0).strip() == "Ni"
    assert dump.type_of(0).lmmax == 18
    assert dump.type_of(0).lps == [2, 2, 0, 0, 1, 1]
    assert np.isclose(dump.provenance["wtkpt"].sum(), 1.0, atol=1e-12)
    esoc_dump = float(esoc_per_ion(dump)[0])
    assert abs(esoc_dump - (-0.08188974)) < 1.0e-6
    # and against the run's own OUTCAR block, parsed independently
    esoc_outcar = parse_outcar_esoc(NISOC_V2_OUTCAR)[0]
    assert abs(esoc_dump - esoc_outcar) < 1.0e-6


def test_real_v2_feo_dump_native_identity():
    """v2 FeO dump/native pair passes the consumer identity validation."""
    from TB2J.interfaces.vasp_cso_dump import read_cso_dump
    from TB2J.interfaces.vasp_split_soc import _check_consistency

    if not (os.path.exists(FEO_V2_Z_DUMP) and os.path.exists(FEO_V2_Z_NATIVE)):
        pytest.skip("real FeO v2 dump/native pair not available")
    snapshot = read_vasp_native(FEO_V2_Z_NATIVE)
    dump = read_cso_dump(FEO_V2_Z_DUMP)
    # raises on any identity mismatch (ispin/nbands/nkpt-IBZ/efermi)
    _check_consistency(snapshot, dump)
    assert dump.provenance["nkpts"] == 20
    assert snapshot.provenance["nkpt_ibz"] == 20
    assert dump.provenance["nbands"] == snapshot.eigenvalues.shape[-1] == 32
    assert dump.provenance["efermi"] == snapshot.efermi


def _three_synthetic_runs(tmp_path, merge_consistency_atol=1.0e-8, **driver_kwargs):
    """Build x/y/z SAXIS synthetic runs and run the full driver.

    The synthetic CSO is scaled to 0.05 eV (real FeO one-center SOC is
    ~0.1 eV on a ~1 eV band window): at physical scales the three-leg
    transverse constraints are mutually consistent at the 1e-9 eV level,
    while a full-strength random CSO on a 4-band window mixes bands at
    O(1) and saturates the SOC-frame systematic the merge tolerates.
    """
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

    artifacts = {}
    for tag in ("x", "y", "z"):
        sub = tmp_path / f"run_{tag}"
        sub.mkdir()
        native, dump_path, _, _ = _make_synthetic_system(
            sub,
            saxis=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]["xyz".index(tag)]
            ),
            cso_scale=0.05,
        )
        artifacts[tag] = {"native_input": str(native), "cso_dump": str(dump_path)}
    out = gen_exchange_vasp_split_soc(
        artifacts,
        output_path=str(tmp_path / "results"),
        rpts=RPTS,
        nz=24,
        merge_consistency_atol=merge_consistency_atol,
        **driver_kwargs,
    )
    return out, artifacts


def test_synthetic_driver_end_to_end(tmp_path):
    """Three x/y/z SAXIS references merge through the rank-nine solve.

    The three runs are exact SU(2) images of one physical problem (the
    fixture re-expresses one CSO), so the transverse constraints are
    mutually consistent and the strict merge tolerance must PASS: the
    diagonals measured by two references agree to ~1e-10 eV.  The merged
    decomposition is written as TB2J results with rank-9 diagnostics.
    """
    from TB2J.io_merge import read_pickle

    out, _ = _three_synthetic_runs(tmp_path)
    merged = read_pickle(str(out))
    assert merged.exchange_Jdict
    provenance = json.loads((out / "split_soc_provenance.json").read_text())
    assert provenance["merge_mode"] == "raw_rank_nine"
    diagnostics = provenance["merge_diagnostics"]
    assert diagnostics["min_rank"] == 9
    assert diagnostics["max_repeat_deviation"] < 1e-8
    if diagnostics["max_reciprocity_residual"] is not None:
        assert diagnostics["max_reciprocity_residual"] < 1e-10
    for tag in ("x", "y", "z"):
        leg_prov = json.loads(
            (out / f"leg_{tag}" / "split_soc_provenance.json").read_text()
        )
        assert leg_prov["leg"] == tag
        assert (out / f"leg_{tag}" / "split_soc_leg.npz").exists()


def test_real_feo_driver_end_to_end(tmp_path):
    """The retained FeO campaign (collinear_x/y/z, v2) through the driver.

    Real native fixture gate: the three independent SAXIS runs merge to a
    rank-nine tensor whose transverse constraints are mutually consistent
    (observed repeated-diagonal spread 2.9e-5 eV on the 63 s full driver),
    the DMI/Jani channels of the inversion-symmetric (1,0,0) translation
    pair stay at the numerical/SOC-systematic zero, and the scalar channel
    reproduces the SOC-off collinear exchange of the same run (observed
    SOC shift 2.0e-6 eV on Jiso = 7.386 meV).
    """
    from TB2J.io_merge import read_pickle

    campaign = os.environ.get(
        "TB2J_FEO_SPLITSOC_CAMPAIGN",
        "/home/hexu/projects/TB2J_dev/.tmp/story-010-feo-oracle",
    )
    artifacts = {}
    for tag in ("x", "y", "z"):
        run = os.path.join(campaign, f"collinear_{tag}")
        native, dump = (
            os.path.join(run, "tb2j_native.bin"),
            os.path.join(run, "tb2j_cso.bin"),
        )
        if not (os.path.exists(native) and os.path.exists(dump)):
            pytest.skip(f"real FeO collinear_{tag} artifacts not available")
        artifacts[tag] = {"native_input": native, "cso_dump": dump}

    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

    out = gen_exchange_vasp_split_soc(
        artifacts,
        output_path=str(tmp_path / "TB2J_results_split_soc"),
        rpts=RPTS,
        nz=24,
        magnetic_elements=("Fe",),
        merge_consistency_atol=1.0e-4,
    )
    merged = read_pickle(str(out))
    assert merged.index_spin[0] >= 0 and merged.index_spin[1] == -1
    assert len(merged.exchange_Jdict) >= 1
    # symmetry gates: the pure-translation Fe-Fe pairs are
    # inversion-symmetric, so DMI vanishes and the cubic local environment
    # forbids single-ion exchange anisotropy up to the SOC-frame systematic
    for key, dmi in merged.dmi_ddict.items():
        assert np.abs(dmi).max() < 1e-6, (key, dmi)
    for key, jani in merged.Jani_dict.items():
        assert np.abs(jani).max() < 1e-4, (key, jani)

    # SOC-off anchor: merged Jiso vs the collinear scalar kernel on the
    # same z-run export (band window identical; SOC correction small)
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.interfaces.vasp_native import read_vasp_native
    from TB2J.paw_projector import build_projector_green_data

    snapshot = read_vasp_native(artifacts["z"]["native_input"])
    data = build_projector_green_data(snapshot)
    jdict = compute_projector_exchange_jdict(
        data, Rpts=RPTS, nz=24, smearing_eV=0.05, sites=(0,)
    )
    for key, jiso_merged in merged.exchange_Jdict.items():
        j0 = jdict.get((key[0], key[1], key[2]))
        if j0 is None:
            continue
        assert abs(jiso_merged - j0) < 1e-5, (key, jiso_merged, j0)


def test_driver_rejects_mistagged_saxis_reference(tmp_path):
    """A leg tag must match the run's SAXIS axis (lattice map guard)."""
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

    artifacts = {}
    for tag in ("x", "y", "z"):
        sub = tmp_path / f"run_{tag}"
        sub.mkdir()
        # the z-tag run is deliberately an x-SAXIS run
        native, dump_path, _, _ = _make_synthetic_system(
            sub,
            saxis=(1.0, 0.0, 0.0) if tag == "z" else (0.0, 0.0, 1.0),
        )
        artifacts[tag] = {"native_input": str(native), "cso_dump": str(dump_path)}
    with pytest.raises(ValueError, match="SAXIS.*not parallel"):
        gen_exchange_vasp_split_soc(
            artifacts,
            output_path=str(tmp_path / "results"),
            rpts=RPTS,
            nz=24,
            band_window_study=False,
        )


def test_rank_nine_merge_needs_three_references(tmp_path):
    """Two references cannot determine the raw tensor (design rank 8)."""
    import TB2J.split_soc_kernel as kernel
    from TB2J.interfaces.vasp_split_soc import (
        compute_split_soc_exchange_leg,
        leg_rotation_map,
    )

    def leg(tag):
        sub = tmp_path / f"run_{tag}"
        sub.mkdir()
        native, dump_path, snapshot, dump = _make_synthetic_system(
            sub, saxis=np.eye(3)["xyz".index(tag)]
        )
        del native, dump_path
        res = compute_split_soc_exchange_leg(
            snapshot, dump, lam=0.0, sites=(0,), rpts=RPTS, nz=24
        )
        rotated = kernel.rotate_transverse_leg(
            res["exchange"], leg_rotation_map(dump), "xyz".index(tag)
        )
        return {"exchange": rotated}

    legs = {tag: leg(tag) for tag in ("x", "y")}
    with pytest.raises(ValueError, match="rank|cannot determine"):
        kernel.merge_transverse_legs(legs)


def test_ligand_soc_shifts_allowed_channels(tmp_path):
    """Ligand CSO shifts the symmetry-allowed transverse scalar channel.

    Three-ion cell: Fe at the origin, Fe at (1/2,1/2,1/2), O off-center
    at (0.2,0.3,0.4) — inversion about the Fe-Fe bond midpoint maps the
    O to a vacancy, so the Fe-Fe pairs are not inversion-symmetric.  The
    ligand one-center CSO is part of the all-atom W_SO; dropping it
    (skip_atoms) must shift the allowed transverse scalar channel.
    """
    from TB2J.interfaces.vasp_split_soc import compute_split_soc_exchange_leg

    _, _, snapshot, dump = _make_synthetic_system(
        tmp_path,
        seed=11,
        symbols=("Fe", "Fe", "O"),
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4]],
    )
    kwargs = dict(
        sites=(0, 1),
        rpts=RPTS,
        nz=24,
    )
    res_full = compute_split_soc_exchange_leg(snapshot, dump, lam=1.0, **kwargs)
    res_nolig = compute_split_soc_exchange_leg(
        snapshot, dump, lam=1.0, skip_atoms=(2,), **kwargs
    )
    for key in (((0, 0, 0), 0, 1), ((1, 0, 0), 0, 1)):
        full = np.asarray(res_full["exchange"][key]["J_leg"], dtype=float)
        nolig = np.asarray(res_nolig["exchange"][key]["J_leg"], dtype=float)
        j_scale = max(1.0e-3, abs(full[0, 0]), abs(nolig[0, 0]))
        assert abs(full[0, 0] - nolig[0, 0]) > 1e-2 * j_scale


# ---------------------------------------------------------------------------
# CLI + persisted provenance
# ---------------------------------------------------------------------------
def test_driver_persists_provenance_with_o_maps(tmp_path):
    out, _ = _three_synthetic_runs(tmp_path)
    provenance = json.loads((out / "split_soc_provenance.json").read_text())
    assert provenance["schema"] == "tb2j.vasp_split_soc_provenance/2.0"
    assert set(provenance["o_maps"]) == {"x", "y", "z"}
    for tag, record in provenance["o_maps"].items():
        o_map = np.asarray(record, dtype=float)
        # pinned frame invariants: O in SO(3), O e_z = the run's SAXIS axis
        assert np.allclose(o_map @ o_map.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(o_map), 1.0, atol=1e-12)
        assert np.allclose(
            o_map @ [0.0, 0.0, 1.0],
            [1.0 if i == "xyz".index(tag) else 0.0 for i in range(3)],
            atol=1e-12,
        )
    assert provenance["leg_paths"] == [
        str(out / f"leg_{tag}") for tag in ("x", "y", "z")
    ]
    for tag in ("x", "y", "z"):
        leg_prov = json.loads(
            (out / f"leg_{tag}" / "split_soc_provenance.json").read_text()
        )
        assert leg_prov["mode"] == "second_variation"
        assert leg_prov["backend"] == "vasp"
        assert leg_prov["leg"] == tag
        npz = np.load(out / f"leg_{tag}" / "split_soc_leg.npz")
        assert npz["J_leg"].shape[1:] == (3, 3)
        assert npz["pair_keys"].shape[0] == npz["J_leg"].shape[0]


def test_cli_end_to_end(tmp_path, monkeypatch):
    from TB2J.io_merge import read_pickle
    from TB2J.scripts.vasp_split_soc2J import run_vasp_split_soc2J

    out = tmp_path / "cli_results"
    argv = [
        "vasp_split_soc2J.py",
        "--output_path",
        str(out),
        "--nz",
        "24",
        # the synthetic random projectors are not localized: restrict the
        # cutoff so the pair set stays in the near-field physically-relevant
        # R shell (real systems decay and can use the 10 A default)
        "--Rcut",
        "5.0",
        "--elements",
        "Fe",
    ]
    for tag in ("x", "y", "z"):
        sub = tmp_path / f"run_{tag}"
        sub.mkdir()
        _make_synthetic_system(
            sub,
            saxis=np.eye(3)["xyz".index(tag)],
            cso_scale=0.05,
        )
        argv += ["--leg", f"{tag}={sub}"]
    monkeypatch.setattr(sys, "argv", argv)
    run_vasp_split_soc2J()

    merged = read_pickle(str(out))
    # --elements Fe selects only the iron site; O stays in the all-atom W_SO
    assert merged.index_spin[0] >= 0 and merged.index_spin[1] == -1
    assert len(merged.exchange_Jdict) >= 1
    provenance = json.loads((out / "split_soc_provenance.json").read_text())
    assert provenance["magnetic_sites"] == [0]
    assert provenance["site_species"][1] == "O"
    # CLI output must equal the direct driver call on the same inputs
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

    direct = gen_exchange_vasp_split_soc(
        {
            tag: {
                "native_input": str(tmp_path / f"run_{tag}" / "tb2j_native.bin"),
                "cso_dump": str(tmp_path / f"run_{tag}" / "tb2j_cso.bin"),
            }
            for tag in ("x", "y", "z")
        },
        output_path=str(tmp_path / "direct_results"),
        rcut=5.0,
        nz=24,
        magnetic_elements=("Fe",),
        band_window_study=False,
    )
    direct_merged = read_pickle(str(direct))
    for key, value in direct_merged.exchange_Jdict.items():
        assert np.isclose(merged.exchange_Jdict[key], value, rtol=1e-12)


# ---------------------------------------------------------------------------
# v3 dump dispatch: reference-potential provenance
# ---------------------------------------------------------------------------
def test_cso_v3_synthetic_round_trip(tmp_path):
    """v3 dumps carry the constants block and potae_xcr; v2 fields intact."""
    from TB2J.interfaces.vasp_cso_dump import read_cso_dump

    sub_v1 = tmp_path / "v1"
    sub_v1.mkdir()
    _, _, _, dump_v1 = _make_synthetic_system(sub_v1)
    sub = tmp_path / "v3"
    sub.mkdir()
    v3_path = sub / "tb2j_cso_v3.bin"
    nmax_max = int(max(dump_v1.type_of(i).lmmax for i in range(dump_v1.nions)))
    potae_xcr = (
        2.0
        * np.ones((nmax_max, dump_v1.nions))
        * (np.arange(dump_v1.nions)[None, :] + 1.0)
    )
    # the writer takes RAW (l1, l2, slot, ion) arrays as stored on disk;
    # the reader output is the transposed view, so transpose back
    _write_cso_dump(
        v3_path,
        cso=dump_v1.cso.transpose(0, 1, 3, 2),
        cocc=dump_v1.cocc.transpose(0, 1, 3, 2),
        saxis=tuple(float(x) for x in dump_v1.saxis),
        alpha=dump_v1.alpha,
        beta=dump_v1.beta,
        symbols=("Fe", "O"),
        version=3,
        constants={"felect": 0.25, "invmc2": 7.45596e-6, "autoa": 0.529177},
        potae_xcr=potae_xcr,
    )
    dump_v3 = read_cso_dump(v3_path)
    assert dump_v3.provenance is not None
    assert dump_v3.provenance["native_version"] == 5
    assert dump_v3.constants == {
        "felect": 0.25,
        "invmc2": 7.45596e-6,
        "autoa": 0.529177,
    }
    assert dump_v3.potae_xcr is not None
    assert dump_v3.potae_xcr.shape == potae_xcr.shape
    assert np.abs(dump_v3.potae_xcr - potae_xcr).max() == 0.0
    assert np.abs(dump_v3.cso - dump_v1.cso).max() == 0.0
    assert np.abs(dump_v3.cocc - dump_v1.cocc).max() == 0.0
    # v1 carries neither block
    assert dump_v1.potae_xcr is None and dump_v1.constants is None


def test_cso_v3_rejects_non_unit_k_weights(tmp_path):
    """The reader fails closed on provenance weights that do not sum to 1."""
    import pytest

    from TB2J.interfaces.vasp_cso_dump import read_cso_dump

    sub = tmp_path / "v3bad"
    sub.mkdir()
    _, _, _, dump_v1 = _make_synthetic_system(tmp_path)
    v3_path = sub / "tb2j_cso_bad.bin"
    bad_weights = np.full(NKPT, 1.0 / NKPT)
    bad_weights[0] += 0.1
    _write_cso_dump(
        v3_path,
        cso=dump_v1.cso.transpose(0, 1, 3, 2),
        cocc=dump_v1.cocc.transpose(0, 1, 3, 2),
        saxis=tuple(float(x) for x in dump_v1.saxis),
        alpha=dump_v1.alpha,
        beta=dump_v1.beta,
        symbols=("Fe", "O"),
        version=3,
        wtkpt=bad_weights,
    )
    with pytest.raises(ValueError, match="weights sum"):
        read_cso_dump(v3_path)
