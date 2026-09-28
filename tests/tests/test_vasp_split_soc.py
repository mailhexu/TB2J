"""Story 011 tests: VASP KS-basis split-SOC adapter, rotate/merge, CLI.

Covers:
- TEST-001: assembled W^K_SO Hermiticity + independent CALC_PAW_OVERLAP
  expected value + frame handling of the SAXIS dump representation
  (strength-0 lam=0 covariance across input SAXIS; per-input three-leg
  O-map consistency), with band spinor components and vertices
  conjugated by M = U_leg^dag U_saxis and W_SO kept as the
  frame-independent state-space matrix.
- Leg cross-consistency: the x/y/z legs are frame re-expressions of one
  physical reference, so their lattice-frame tensors coincide at lam=0
  and lam=1 (observed ~1e-17).
- COCC reconstruction from the collinear CPROJ against the patch dump
  (CRHODE(LP,L) = conj(CPROJ(LP)) CPROJ(L) fast_aug order; the real-dump
  residual documents the patch's occupation-update skew).
- All-atom ligand SOC influence: on the real centrosymmetric FeO dump the
  ligand influence is gated on the W_SO operator and the Jiso channel; on
  a non-centrosymmetric synthetic cell (off-center ligand) the allowed
  Jiso channel shifts.  (The Jani/DMI channels of the shared A-channel
  mapping are invalid — cross-story projector_green finding, replacement
  under way per Main — and no Jani/DMI gate is booked.)
- CLI/driver end-to-end with three-leg io_merge and persisted provenance
  (per-leg O maps, O e_z = leg axis).
"""

import json
import os
import sys

import numpy as np
import pytest

from TB2J.interfaces.vasp_cso_dump import MAGIC, VERSION, read_cso_dump
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
):
    """Serialize a tb2j_cso.bin v1 stream (test-side format writer)."""
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
        wi(VERSION)
        wi(len(layout))  # nions
        wi(len(types))  # ntyp
        wi(lmdim_max)
        wi(nmax_max)
        wi(2)  # ncdij
        wi(lsorbit)
        wr(np.asarray(saxis, dtype=float))
        wr([alpha, beta])
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
        wstream(cso)
        wstream(cocc)
        if potae is None:
            potae = np.ones((nmax_max, len(layout)))
        f.write(np.ascontiguousarray(np.asarray(potae).T).tobytes())
    return path


def _make_synthetic_system(
    tmp_path, saxis=(0.0, 0.0, 1.0), seed=7, symbols=("Fe", "O"), positions=None
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
                uu = _random_hermitian(n1, rng)
                ud = rng.normal(size=(n1, n2)) + 1j * rng.normal(size=(n1, n2))
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
        # CSO'[a', b'] = sum_{a,b} conj(U[a', a]) U[b', b] CSO[a, b]
        full = cso_z.reshape(nions, 2, 2, lmdim, lmdim)
        rot = np.einsum("ca,db,iabpq->icdpq", u.conj(), u, full)
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
    """Frame handling of the SAXIS dump representation.

    At lam=0 the strength-0 problem is spin-rotation invariant, so the
    per-leg lattice-frame tensors must be independent of the input SAXIS
    (observed agreement ~7e-10 against a 1e-2 exchange scale; an
    un-conjugated vertex in a transverse leg breaks this).  At lam=1 a
    SAXIS change rotates the magnetization *relative to the
    lattice-locked CSO* — that is the magnetocrystalline anisotropy
    degree of freedom, not a symmetry — so no cross-SAXIS equality is
    asserted there; instead each input's three legs must be internally
    consistent (the O map T_lattice = O T_leg O^T with O e_z = leg axis).
    """
    from TB2J.interfaces.vasp_split_soc import (
        compute_split_soc_exchange_leg,
        so3_from_su2,
        vasp_spinor_frame,
    )

    entries = {}
    for tag, saxis in (("z", (0.0, 0.0, 1.0)), ("n", (1.0, 2.0, 1.5))):
        sub = tmp_path / tag
        sub.mkdir()
        _, _, snapshot, dump = _make_synthetic_system(sub, saxis=saxis)
        per_leg = {}
        for leg_dir in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            res = compute_split_soc_exchange_leg(
                snapshot,
                dump,
                leg_direction=np.asarray(leg_dir),
                lam=1.0,
                sites=(0,),
                rpts=RPTS,
                nz=24,
            )
            key = ((1, 0, 0), 0, 0)
            assert key in res["exchange"]
            per_leg[leg_dir] = res["exchange"][key]
        entries[tag] = (so3_from_su2(vasp_spinor_frame(dump.alpha, dump.beta)), per_leg)

    # lam=0: strength-0 spin-rotation invariance across input SAXIS frames
    anchor = {}
    for tag, saxis in (("z", (0.0, 0.0, 1.0)), ("n", (1.0, 2.0, 1.5))):
        sub = tmp_path / f"{tag}-anchor"
        sub.mkdir()
        _, _, snapshot, dump = _make_synthetic_system(sub, saxis=saxis)
        res = compute_split_soc_exchange_leg(
            snapshot,
            dump,
            leg_direction=np.array([0.0, 0.0, 1.0]),
            lam=0.0,
            sites=(0,),
            rpts=RPTS,
            nz=24,
        )
        anchor[tag] = res["exchange"][((1, 0, 0), 0, 0)]["tensor"]
    assert np.allclose(anchor["z"], anchor["n"], atol=1e-8)

    # lam=1: per-input leg consistency of the covariant channels (frame
    # map, not cross-SAXIS symmetry).  Off-diagonal tensor channels are
    # quarantined (cross-story projector_green finding).
    for tag in ("z", "n"):
        per_leg = entries[tag][1]
        ref = per_leg[(0.0, 0.0, 1.0)]
        for leg_dir, entry in per_leg.items():
            assert np.isclose(entry["Jiso"], ref["Jiso"], atol=1e-10)
            assert np.allclose(
                np.diag(entry["tensor"]), np.diag(ref["tensor"]), atol=1e-10
            )
    # the O map of the z leg is the identity (O e_z = z)
    o_z = so3_from_su2(vasp_spinor_frame(0.0, 0.0))
    assert np.allclose(o_z, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# TEST-002: SOC-off anchor against the existing collinear v5/v6 kernel
# ---------------------------------------------------------------------------
def test_soc_off_anchor_matches_collinear_exchange(tmp_path):
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.interfaces.vasp_split_soc import compute_split_soc_exchange_leg
    from TB2J.paw_projector import build_projector_green_data

    _, _, snapshot, dump = _make_synthetic_system(tmp_path)

    res = compute_split_soc_exchange_leg(
        snapshot,
        dump,
        leg_direction=np.array([0.0, 0.0, 1.0]),
        lam=0.0,
        sites=(0, 1),
        rpts=RPTS,
        nz=24,
    )

    data = build_projector_green_data(snapshot)
    jdict = compute_projector_exchange_jdict(
        data, Rpts=RPTS, nz=24, smearing_eV=0.05, sites=(0, 1)
    )
    for (r, i, j), jiso_collinear in jdict.items():
        assert np.isclose(res["exchange"][(tuple(r), i, j)]["Jiso"], jiso_collinear)


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
        leg_direction=np.array([0.0, 0.0, 1.0]),
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
    # the translation self-pair of rocksalt FeO is inversion-symmetric, so
    # its DMI is forbidden; and with same-axis collinear splitting vertices
    # the shared channel mapping pins DMI to zero for any W (cross-story
    # projector_green finding, under Main's core diagnosis) -- no DMI gate
    # is booked here either way.
    # Ligand SOC measurably shifts the symmetry-allowed scalar channel
    # (observed shift 2.0e-7 eV on Jiso ~ 7.4e-3 eV; noise floor ~1e-15).
    assert abs(full["Jiso"] - nolig["Jiso"]) > 1e-9
    # magnetic vertices untouched: the strength-0 onsite splitting is the
    # same to the ligand's second-order correction
    assert np.isclose(full["Jiso"], nolig["Jiso"], rtol=1e-4, atol=1e-7)


def test_real_feo_driver_end_to_end(tmp_path):
    paths = _feo_paths()
    if paths is None:
        pytest.skip("real FeO story-010 dump not available")
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc
    from TB2J.io_merge import read_pickle

    out = gen_exchange_vasp_split_soc(
        native_input=paths[0],
        cso_dump=paths[1],
        output_path=str(tmp_path / "TB2J_results_split_soc"),
        rpts=RPTS,
        nz=24,
        magnetic_elements=("Fe",),
    )
    merged = read_pickle(str(out))
    assert merged.index_spin[0] >= 0 and merged.index_spin[1] == -1
    assert len(merged.exchange_Jdict) >= 1


def test_synthetic_driver_end_to_end(tmp_path):
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc
    from TB2J.io_merge import read_pickle

    native, dump_path, _, _ = _make_synthetic_system(tmp_path)
    out = gen_exchange_vasp_split_soc(
        native_input=str(native),
        cso_dump=str(dump_path),
        output_path=str(tmp_path / "results"),
        rpts=RPTS,
        nz=24,
    )
    merged = read_pickle(str(out))
    assert merged.exchange_Jdict


# ---------------------------------------------------------------------------
# Leg cross-consistency (regression for the review round-1 frame blocker)
# ---------------------------------------------------------------------------
def test_three_legs_lattice_frame_consistency(tmp_path):
    """The x/y/z legs are frame re-expressions of one physical reference.

    Band spinor components and vertices are conjugated by M = U_leg^dag
    U_saxis while W_SO stays the frame-independent state-space matrix, so
    the lattice-frame Jiso and the tensor diagonal must coincide across
    legs at lam=0 and lam=1 (observed ~1e-17; a state-space W conjugation
    or an un-conjugated vertex breaks this at the 1e-2 level).  The
    off-diagonal tensor channels are NOT gated: the shared A-channel
    mapping produces frame-dependent garbage there (cross-story
    projector_green finding, replacement under way per Main).
    """
    from TB2J.interfaces.vasp_split_soc import compute_split_soc_exchange_leg

    _, _, snapshot, dump = _make_synthetic_system(tmp_path)
    for lam in (0.0, 1.0):
        per_leg = {}
        for leg in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            res = compute_split_soc_exchange_leg(
                snapshot,
                dump,
                leg_direction=np.asarray(leg, dtype=float),
                lam=lam,
                sites=(0, 1),
                rpts=RPTS,
                nz=24,
            )
            per_leg[leg] = res["exchange"]
        for key in ((0, 0, 0), 0, 0), ((1, 0, 0), 0, 1), ((1, 0, 0), 1, 0):
            ref = per_leg[(0.0, 0.0, 1.0)][key]
            for ex in per_leg.values():
                entry = ex[key]
                assert np.isclose(entry["Jiso"], ref["Jiso"], atol=1e-10)
                assert np.allclose(
                    np.diag(entry["tensor"]),
                    np.diag(ref["tensor"]),
                    atol=1e-10,
                )


def test_ligand_soc_shifts_allowed_channels(tmp_path):
    """Ligand CSO shifts the symmetry-allowed Jiso of an allowed pair.

    Three-ion cell: Fe at the origin, Fe at (1/2,1/2,1/2), O off-center
    at (0.2,0.3,0.4) — inversion about the Fe-Fe bond midpoint maps the
    O to a vacancy, so the Fe-Fe pairs are not inversion-symmetric.  The
    ligand one-center CSO is part of the all-atom W_SO; dropping it
    (skip_atoms) must shift the allowed Jiso channel.
    """
    from TB2J.interfaces.vasp_split_soc import compute_split_soc_exchange_leg

    _, _, snapshot, dump = _make_synthetic_system(
        tmp_path,
        seed=11,
        symbols=("Fe", "Fe", "O"),
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.3, 0.4]],
    )
    kwargs = dict(
        leg_direction=np.array([0.0, 0.0, 1.0]),
        sites=(0, 1),
        rpts=RPTS,
        nz=24,
    )
    res_full = compute_split_soc_exchange_leg(snapshot, dump, lam=1.0, **kwargs)
    res_nolig = compute_split_soc_exchange_leg(
        snapshot, dump, lam=1.0, skip_atoms=(2,), **kwargs
    )
    for key in (((0, 0, 0), 0, 1), ((1, 0, 0), 0, 1)):
        full = res_full["exchange"][key]
        nolig = res_nolig["exchange"][key]
        j_scale = max(1.0e-3, abs(full["Jiso"]), abs(nolig["Jiso"]))
        assert abs(full["Jiso"] - nolig["Jiso"]) > 1e-2 * j_scale
    # No Jani/DMI gate: the shared A-channel tensor mapping is invalid
    # (cross-story projector_green finding); its channels are quarantined
    # until the common core replacement lands.


# ---------------------------------------------------------------------------
# CLI + persisted provenance
# ---------------------------------------------------------------------------
def test_driver_persists_provenance_with_o_maps(tmp_path):
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc

    native, dump_path, _, _ = _make_synthetic_system(tmp_path)
    out = gen_exchange_vasp_split_soc(
        native_input=str(native),
        cso_dump=str(dump_path),
        output_path=str(tmp_path / "results"),
        rpts=RPTS,
        nz=24,
    )
    provenance = json.loads((out / "split_soc_provenance.json").read_text())
    assert provenance["schema"] == "tb2j.vasp_split_soc_provenance/1.0"
    assert set(provenance["legs"]) == {"x", "y", "z"}
    axis_of = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    for tag, record in provenance["legs"].items():
        o_map = np.asarray(record["o_map"], dtype=float)
        # pinned frame invariants: O in SO(3), O e_z = leg axis
        assert np.allclose(o_map @ o_map.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(o_map), 1.0, atol=1e-12)
        assert np.allclose(o_map @ [0.0, 0.0, 1.0], axis_of[tag], atol=1e-12)
        assert record["kernel_metadata"]["mode"] == "second_variation"
    assert provenance["leg_paths"] == [
        str(out / "leg_x"),
        str(out / "leg_y"),
        str(out / "leg_z"),
    ]


def test_cli_end_to_end(tmp_path, monkeypatch):
    from TB2J.interfaces.vasp_split_soc import gen_exchange_vasp_split_soc
    from TB2J.io_merge import read_pickle
    from TB2J.scripts.vasp_split_soc2J import run_vasp_split_soc2J

    native, dump_path, _, _ = _make_synthetic_system(tmp_path)
    out = tmp_path / "cli_results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vasp_split_soc2J.py",
            "--native-input",
            str(native),
            "--cso-dump",
            str(dump_path),
            "--output_path",
            str(out),
            "--nz",
            "24",
            "--Rcut",
            "5.0",
            "--elements",
            "Fe",
        ],
    )
    run_vasp_split_soc2J()

    merged = read_pickle(str(out))
    # --elements Fe selects only the iron site; O stays in the all-atom W_SO
    assert merged.index_spin[0] >= 0 and merged.index_spin[1] == -1
    assert len(merged.exchange_Jdict) >= 1
    for tag in ("x", "y", "z"):
        assert (out / f"leg_{tag}" / "TB2J.pickle").exists()
    provenance = json.loads((out / "split_soc_provenance.json").read_text())
    assert provenance["magnetic_sites"] == [0]
    assert provenance["site_species"][1] == "O"
    # CLI output must equal the direct driver call on the same inputs
    direct = gen_exchange_vasp_split_soc(
        native_input=str(native),
        cso_dump=str(dump_path),
        output_path=str(tmp_path / "direct_results"),
        rcut=5.0,
        nz=24,
        magnetic_elements=("Fe",),
    )
    direct_merged = read_pickle(str(direct))
    for key, value in direct_merged.exchange_Jdict.items():
        assert np.isclose(merged.exchange_Jdict[key], value, rtol=1e-12)
