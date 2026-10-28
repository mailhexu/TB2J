"""Tests for the Quantum ESPRESSO ``becp_dump`` projector-Green reader.

Covers the record layout of ``DUMP_FORMAT_V1_1.md`` (gfortran sequential
unformatted Fortran, 4-byte little-endian record markers, Fortran
column-major arrays) via a synthetic dump writer, so no QE build or real
dump file is required.  Pinned math: becp coefficients are already dual,
``hij = deeq(up) - deeq(down)`` is the covariant separable operator,
``overlap_k = None`` (never ``qq_at``/Gram as a channel metric).
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from TB2J.interfaces.abinit_paw import BOHR_TO_ANGSTROM
from TB2J.interfaces.qe_projector import (
    MAGIC_ATOMIC,
    MAGIC_SPINOR,
    MAGIC_SPINOR_NC11,
    RYTOEV,
    QEProjectorDump,
    gen_exchange_qe,
    parse_qe_atomic_dump,
    parse_qe_dump,
    parse_qe_spinor_dump,
    read_qe_atomic_dump,
    read_qe_dump,
    read_qe_spinor_dump,
)
from TB2J.projector_green import ProjectorGreen, ProjectorGreenData

# Real v1.1 golden dumps (Story-1 integration); optional, skipped when absent.
GOLDEN_DIR = Path("/home/hexu/projects/TB2J_dev/.tmp/qe-exporter")

MAGIC_V10 = b"TB2JQEDUMPV1    "
MAGIC_V11 = b"TB2JQEDUMPV1.1  "
MAGIC_V12 = b"TB2JQEDUMPV1.2  "


# ---------------------------------------------------------------------------
# Synthetic dump writer (record layout of DUMP_FORMAT_V1_1.md)
# ---------------------------------------------------------------------------
def _frecord(fh, *parts):
    """Write one Fortran sequential record: 4-byte marker, payload, marker."""
    payload = b"".join(np.asarray(p, order="F").tobytes(order="F") for p in parts)
    fh.write(struct.pack("<i", len(payload)))
    fh.write(payload)
    fh.write(struct.pack("<i", len(payload)))


def write_qe_dump(
    path,
    *,
    magic=MAGIC_V11,
    nspin=2,
    nks=4,
    nkstot=4,
    nbnd=3,
    nat=3,
    nsp=2,
    nhm=3,
    lmaxkb=2,
    atm=(b"Fe", b"O "),
    ityp=(1, 1, 2),
    nh=(3, 2),
    tvanp=(1, 1),
    tpawp=(0, 0),
    isk=None,
    ef_up=-1.2,
    ef_dw=-1.3,
    include_becsum=True,
    include_rho_bec=False,
    include_dbeta_xc=False,
    include_ddd_paw=False,
    seed=20261009,
) -> SimpleNamespace:
    """Write a synthetic QE projector dump; return ground-truth arrays."""
    rng = np.random.default_rng(seed)
    nkb = int(sum(nh[a - 1] for a in ityp))
    alat = 4.0
    at = np.array(
        [[1.0, 0.3, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]]
    )  # columns = lattice vectors, alat units; at^T bg = I
    bg = np.linalg.inv(at).T
    tau = np.array(
        [[0.0, 0.5, 0.25], [0.1, 0.5, 0.25], [0.2, 0.0, 0.5]]
    )  # (3, nat), cartesian, alat units
    if isk is None:
        isk = np.array([1, 2, 1, 2][:nks], dtype="<i4")
    else:
        isk = np.asarray(isk, dtype="<i4")
    xk = np.zeros((3, nks))
    xk[0, 2:] = 0.5  # k-points 3 and 4 at (0.5, 0, 0) 2pi/alat; 1, 2 at Gamma
    wk = np.array([1.0, 1.0, 2.0, 2.0][:nks])
    deeq = rng.normal(size=(nhm, nhm, nat, nspin))
    deeq = deeq + deeq.transpose(1, 0, 2, 3)  # Hermitian atom blocks
    dvan = rng.normal(size=(nhm, nhm, nsp))
    qq_at = rng.normal(size=(nhm, nhm, nat))
    gram = rng.normal(size=(nhm, nhm, nat)) + 1j * rng.normal(size=(nhm, nhm, nat))
    et = rng.normal(size=(nbnd, nks))
    wg = np.abs(rng.normal(size=(nbnd, nks))) * wk
    npack = nhm * (nhm + 1) // 2
    becsum = rng.normal(size=(npack, nat, nspin))
    rho_bec = rng.normal(size=(npack, nat, nspin))
    dbeta_xc = rng.normal(size=(nhm, nhm, nat))
    dbeta_xc = dbeta_xc + dbeta_xc.transpose(1, 0, 2)
    ddd_paw = rng.normal(size=(npack, nat, nspin))
    becps = [
        rng.normal(size=(nkb, nbnd)) + 1j * rng.normal(size=(nkb, nbnd))
        for _ in range(nks)
    ]

    with open(path, "wb") as fh:
        _frecord(
            fh,
            np.array([magic], dtype="S16"),
            np.array(
                [nspin, nks, nkstot, nbnd, nat, nsp, nhm, lmaxkb, nkb], dtype="<i4"
            ),
        )
        _frecord(
            fh,
            np.array([12.0, -1.25, ef_up, ef_dw, 0.01]),
            np.array([1, 0, 1], dtype="<i4"),
        )
        _frecord(
            fh,
            at,
            bg,
            np.array([alat, alat**3 * abs(np.linalg.det(at))]),
            np.array([1], dtype="<i4"),
        )
        _frecord(fh, np.array(atm, dtype="S3"))
        _frecord(fh, np.array(ityp, dtype="<i4"), tau)
        _frecord(
            fh,
            np.array(nh, dtype="<i4"),
            np.array(tvanp, dtype="<i4"),
            np.array(tpawp, dtype="<i4"),
        )
        _frecord(fh, deeq)
        _frecord(fh, dvan)
        _frecord(fh, qq_at)
        _frecord(fh, gram)
        _frecord(fh, xk, wk, isk)
        _frecord(fh, et, wg)
        for becp in becps:
            _frecord(fh, becp)
        if include_becsum:
            _frecord(fh, becsum)
        if include_rho_bec:
            _frecord(fh, rho_bec)
        if include_dbeta_xc:
            _frecord(fh, dbeta_xc)
        if include_ddd_paw:
            _frecord(fh, ddd_paw)

    return SimpleNamespace(
        path=Path(path),
        version={
            MAGIC_V10: "1.0",
            MAGIC_V11: "1.1",
            MAGIC_V12: "1.2",
        }[magic],
        nspin=nspin,
        nks=nks,
        nkstot=nkstot,
        nbnd=nbnd,
        nat=nat,
        nsp=nsp,
        nhm=nhm,
        lmaxkb=lmaxkb,
        nkb=nkb,
        atm=[a.decode("ascii").strip() for a in atm],
        ityp=np.array(ityp, dtype=int),
        tau=tau.T.copy(),
        nh=np.array(nh, dtype=int),
        tvanp=np.array(tvanp, dtype=bool),
        tpawp=np.array(tpawp, dtype=bool),
        at=at,
        bg=bg,
        alat=alat,
        omega=alat**3 * abs(np.linalg.det(at)),
        nelec=12.0,
        ef=-1.25,
        ef_up=ef_up,
        ef_dw=ef_dw,
        deeq=deeq,
        dvan=dvan,
        qq_at=qq_at,
        gram=gram,
        xk=xk.T.copy(),
        wk=wk,
        isk=isk.astype(int),
        et=et.T.copy(),
        wg=wg.T.copy(),
        becps=becps,
        becsum=becsum if include_becsum else None,
        rho_bec=rho_bec if include_rho_bec else None,
        dbeta_xc=dbeta_xc if include_dbeta_xc else None,
        ddd_paw=ddd_paw if include_ddd_paw else None,
    )


# ---------------------------------------------------------------------------
# parse_qe_dump: raw records, magic/version gates
# ---------------------------------------------------------------------------
def test_parse_v10_and_v11(tmp_path):
    p10 = tmp_path / "v10.bin"
    write_qe_dump(p10, magic=MAGIC_V10, include_becsum=False)
    dump = parse_qe_dump(p10)
    assert isinstance(dump, QEProjectorDump)
    assert dump.version == "1.0"
    assert dump.becsum is None
    assert dump.rho_bec is None

    p11 = tmp_path / "v11.bin"
    gt = write_qe_dump(p11)
    dump = parse_qe_dump(p11)
    assert dump.version == "1.1"
    assert dump.becsum.shape == (6, 3, 2)
    assert dump.rho_bec is None
    np.testing.assert_allclose(dump.becsum, gt.becsum)

    # raw record fidelity (units as dumped: Ry / Bohr / alat)
    np.testing.assert_allclose(dump.deeq, gt.deeq)
    np.testing.assert_allclose(dump.dvan, gt.dvan)
    np.testing.assert_allclose(dump.qq_at, gt.qq_at)
    np.testing.assert_allclose(dump.gram, gt.gram)
    np.testing.assert_allclose(dump.xk, gt.xk)
    np.testing.assert_allclose(dump.wk, gt.wk)
    np.testing.assert_array_equal(dump.isk, gt.isk)
    np.testing.assert_allclose(dump.et, gt.et)
    np.testing.assert_allclose(dump.wg, gt.wg)
    np.testing.assert_allclose(dump.tau, gt.tau)
    np.testing.assert_allclose(dump.at, gt.at)
    np.testing.assert_allclose(dump.bg, gt.bg)
    assert dump.alat == 4.0
    assert dump.atm == ["Fe", "O"]
    assert dump.ityp.tolist() == [1, 1, 2]
    assert dump.nh.tolist() == [3, 2]
    assert dump.nkb == 8
    assert dump.ofsbeta.tolist() == [0, 3, 6]
    assert dump.tvanp.tolist() == [True, True]
    assert dump.tpawp.tolist() == [False, False]
    assert dump.nspin == 2
    assert dump.nkstot == dump.nks == 4
    assert dump.nelec == 12.0
    assert dump.ef == -1.25
    for ik in range(4):
        np.testing.assert_allclose(dump.coefficients[ik], gt.becps[ik])

    # v1.1 PAW dump carries the optional rho%bec record
    paw = tmp_path / "paw.bin"
    gtp = write_qe_dump(paw, tvanp=(0, 0), tpawp=(1, 1), include_rho_bec=True)
    dpaw = parse_qe_dump(paw)
    assert dpaw.rho_bec is not None
    np.testing.assert_allclose(dpaw.rho_bec, gtp.rho_bec)


def test_unknown_magic_raises(tmp_path):
    p = tmp_path / "bad.bin"
    write_qe_dump(p)
    raw = bytearray(p.read_bytes())
    raw[4:20] = b"TB2JQEDUMPV9.9  "
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="magic"):
        parse_qe_dump(p)


def test_nspin_1_raises_collinear_message(tmp_path):
    p = tmp_path / "nmag.bin"
    write_qe_dump(p, nspin=1)
    with pytest.raises(ValueError, match="collinear"):
        parse_qe_dump(p)


def test_pool_parallel_raises(tmp_path):
    p = tmp_path / "pool.bin"
    write_qe_dump(p, nkstot=8)
    with pytest.raises(ValueError, match="pool"):
        parse_qe_dump(p)


def test_species_labels_len6_encoding(tmp_path):
    """QE >= 8 writes CHARACTER(LEN=6) atm entries blank-padded to capacity."""
    p = tmp_path / "len6.bin"
    write_qe_dump(p)
    raw = bytearray(p.read_bytes())
    # record 3 spans [292, 306): replace the S3 pair with v8-style 6-byte
    # labels blank-padded to a fixed capacity of 2 entries
    record = b"Fe    " + b"O     "
    raw[292:306] = (
        struct.pack("<i", len(record)) + record + struct.pack("<i", len(record))
    )
    p.write_bytes(bytes(raw))
    dump = parse_qe_dump(p)
    assert dump.atm == ["Fe", "O"]


def test_record_length_mismatch_raises(tmp_path):
    p = tmp_path / "corrupt.bin"
    write_qe_dump(p)
    raw = bytearray(p.read_bytes())
    # record 1 (fermi/smearing) closing marker: 4 + 52 payload bytes in
    raw[56:60] = struct.pack("<i", 999)
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="record"):
        parse_qe_dump(p)

    p2 = tmp_path / "truncated.bin"
    write_qe_dump(p2)
    p2.write_bytes(p2.read_bytes()[:450])  # cut inside the deeq record
    with pytest.raises(ValueError, match="record"):
        parse_qe_dump(p2)


# ---------------------------------------------------------------------------
# read_qe_dump: ProjectorGreenData normalization
# ---------------------------------------------------------------------------
def test_delta_extraction_is_deeq_spin_difference(tmp_path):
    p = tmp_path / "delta.bin"
    gt = write_qe_dump(p)
    data = read_qe_dump(p, legacy_no_u=True)
    expected = (gt.deeq[:, :, :, 0] - gt.deeq[:, :, :, 1]).transpose(2, 0, 1)
    expected = expected * RYTOEV
    assert RYTOEV == 13.605693122994
    assert data.hij.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(
        data.operator_components["delta_total"], expected, rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(data.hij[0] - data.hij[1], expected)
    np.testing.assert_allclose(
        data.hij[0], gt.deeq[:, :, :, 0].transpose(2, 0, 1) * RYTOEV
    )
    np.testing.assert_allclose(
        data.hij[1], gt.deeq[:, :, :, 1].transpose(2, 0, 1) * RYTOEV
    )
    assert data.hij_units == "eV"


def test_overlap_none_and_pinned_names(tmp_path):
    p = tmp_path / "pins.bin"
    write_qe_dump(p)
    data = read_qe_dump(p, legacy_no_u=True)
    assert data.overlap_k is None
    assert data.overlap_metric is None
    assert data.population_metric_matrix is None
    assert data.hij_definition == "qe_deeq_spin_difference"
    assert data.coefficient_source == "qe_becp"
    assert data.coefficient_projector == "qe_beta"
    assert data.channel_interpretation == "qe_dual_to_beta"
    assert data.operator_basis == "qe_dual_beta_channel"
    assert data.metadata["coefficient_convention"] == "dual_projector_no_inverse"
    assert data.metadata["qe_npool_ok"] is True
    assert data.metadata["becsum_present"] is True
    assert data.metadata["rho_bec_present"] is False
    assert data.metadata["tvanp"] == [True, True]
    assert data.metadata["tpawp"] == [False, False]
    assert "TB2JQEDUMP" in data.metadata["qe_magic"]
    green = ProjectorGreen(data)
    assert green.coefficients_are_dual


def test_per_spin_coefficient_mapping_via_isk(tmp_path):
    p = tmp_path / "spins.bin"
    gt = write_qe_dump(p)
    data = read_qe_dump(p, legacy_no_u=True)
    assert data.nspin == 2
    assert data.nkpt == 2
    assert data.nband == 3
    # isk = [1, 2, 1, 2]: up k-points at flat indices 0, 2; down at 1, 3.
    # Fractional coordinates must reconstruct the dumped cartesian xk through
    # the dumped bg (definition of fractional_reciprocal, independent of the
    # reader's internal at^T xk conversion).
    assert data.kpoints.shape == (2, 3)
    np.testing.assert_allclose(gt.at.T @ gt.bg, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(data.kpoints[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(data.kpoints @ gt.bg.T, gt.xk[[0, 2]], atol=1e-14)
    np.testing.assert_allclose(data.eigenvalues[0], gt.et[[0, 2]] * RYTOEV)
    np.testing.assert_allclose(data.eigenvalues[1], gt.et[[1, 3]] * RYTOEV)
    np.testing.assert_allclose(data.coefficients[0, 0], gt.becps[0].T)
    np.testing.assert_allclose(data.coefficients[0, 1], gt.becps[2].T)
    np.testing.assert_allclose(data.coefficients[1, 0], gt.becps[1].T)
    np.testing.assert_allclose(data.coefficients[1, 1], gt.becps[3].T)
    w = gt.wk[[0, 2]] + gt.wk[[1, 3]]
    np.testing.assert_allclose(data.weights, w / w.sum())
    np.testing.assert_allclose(
        data.occupations[0], gt.wg[[0, 2]] / gt.wk[[0, 2]][:, None]
    )
    np.testing.assert_allclose(
        data.occupations[1], gt.wg[[1, 3]] / gt.wk[[1, 3]][:, None]
    )
    np.testing.assert_allclose(data.efermi_spin, np.array([-1.2, -1.3]) * RYTOEV)


def test_single_fermi_energy_run(tmp_path):
    p = tmp_path / "onefermi.bin"
    write_qe_dump(p, ef_up=0.0, ef_dw=0.0)
    data = read_qe_dump(p, legacy_no_u=True)
    assert data.efermi_spin is None
    np.testing.assert_allclose(data.efermi, -1.25 * RYTOEV)


def test_per_atom_channel_splitting(tmp_path):
    p = tmp_path / "channels.bin"
    gt = write_qe_dump(p)
    data = read_qe_dump(p, legacy_no_u=True)
    assert data.projector_site.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert data.projector_atom.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]
    assert data.site_nproj.tolist() == [3, 3, 2]
    assert data.site_projector_indices.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, -1]]
    assert data.nproj == gt.nkb == 8
    np.testing.assert_allclose(data.cell, gt.at.T * gt.alat * BOHR_TO_ANGSTROM)
    np.testing.assert_allclose(data.positions, gt.tau * gt.alat * BOHR_TO_ANGSTROM)
    assert data.atomic_numbers.tolist() == [26, 26, 8]
    assert data.cell.shape == (3, 3)
    assert data.positions.shape == (3, 3)


def test_interleaved_kb_species_preserve_site_resolved_band_action(tmp_path):
    """QE beta rows are Fe(atom0), Fe(atom2), O(atom1), not ion order."""
    path = tmp_path / "interleaved-kb.bin"
    native = write_qe_dump(path, ityp=(1, 2, 1))
    data = read_qe_dump(path, legacy_no_u=True)
    green = ProjectorGreen(data)
    native_rows = ([0, 1, 2], [6, 7], [3, 4, 5])
    for site, rows in enumerate(native_rows):
        count = len(rows)
        native_operator = (
            native.deeq[:count, :count, site, 0] - native.deeq[:count, :count, site, 1]
        ) * RYTOEV
        native_coefficients = native.becps[0][rows]
        expected = native_coefficients.conj().T @ native_operator @ native_coefficients
        for selected in (
            green.get_site_projectors(site),
            np.flatnonzero(data.projector_site == site),
            np.flatnonzero(data.projector_atom == site),
        ):
            coefficients = data.coefficients[0, 0][:, selected]
            actual = (
                coefficients.conj() @ green.get_local_operator(site) @ coefficients.T
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


def test_single_spin_channel_raises(tmp_path):
    p = tmp_path / "onechannel.bin"
    write_qe_dump(p, isk=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="spin"):
        read_qe_dump(p, legacy_no_u=True)


# ---------------------------------------------------------------------------
# Analytic pin: rank-1 separable operator trace via the runtime Green
# ---------------------------------------------------------------------------
def test_rank1_separable_trace_with_becp_green():
    """Tr[(beta Du beta^dag) Gup (beta Dd beta^dag) Gdn] == Tr[Du Gb_up Dd Gb_dn].

    The channel-space Green functions come from ProjectorGreen built on becp
    coefficients with overlap_k=None (dual convention); the reference is the
    dense plane-wave-space contraction with an explicit beta.
    """
    rng = np.random.default_rng(11)
    npw, nch, nbnd = 6, 2, 3
    beta = rng.normal(size=(npw, nch)) + 1j * rng.normal(size=(npw, nch))
    u = rng.normal(size=nch) + 1j * rng.normal(size=nch)
    v = rng.normal(size=nch) + 1j * rng.normal(size=nch)
    d_up = np.outer(u, u.conj())  # rank-1 Hermitian
    d_dn = np.outer(v, v.conj())
    energy, efermi = 0.7, 0.3

    evals = rng.normal(scale=2.0, size=(2, 1, nbnd))
    coefficients = np.empty((2, 1, nbnd, nch), dtype=complex)
    psi = []
    for s in range(2):
        psi_s = rng.normal(size=(npw, nbnd)) + 1j * rng.normal(size=(npw, nbnd))
        psi.append(psi_s)
        becp = beta.conj().T @ psi_s  # (nch, nbnd) = <beta|psi>
        coefficients[s, 0] = becp.T

    data = ProjectorGreenData(
        kpoints=np.zeros((1, 3)),
        weights=np.ones(1),
        eigenvalues=evals,
        coefficients=coefficients,
        efermi=efermi,
        projector_site=np.zeros(nch, dtype=int),
        projector_atom=np.zeros(nch, dtype=int),
    )
    green = ProjectorGreen(data)
    assert green.coefficients_are_dual
    g_up_ch = green.get_Gk(0, energy, ispin=0)
    g_dn_ch = green.get_Gk(0, energy, ispin=1)
    dual_trace = np.trace(d_up @ g_up_ch @ d_dn @ g_dn_ch)

    v_up_pw = beta @ d_up @ beta.conj().T
    v_dn_pw = beta @ d_dn @ beta.conj().T
    g_pw = []
    for s in range(2):
        w = 1.0 / (energy + efermi - evals[s, 0])
        g_pw.append((psi[s] * w) @ psi[s].conj().T)
    reference = np.trace(v_up_pw @ g_pw[0] @ v_dn_pw @ g_pw[1])
    np.testing.assert_allclose(dual_trace, reference, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# v1.2: projected xc vertex records and the conjugated delta_total
# ---------------------------------------------------------------------------
def test_parse_v12_records(tmp_path):
    p = tmp_path / "qe_v12_synth.bin"
    gt = write_qe_dump(
        p,
        magic=MAGIC_V12,
        include_becsum=True,
        include_rho_bec=False,
        include_dbeta_xc=True,
        include_ddd_paw=False,
    )
    dump = parse_qe_dump(p)
    assert dump.version == "1.2"
    np.testing.assert_allclose(dump.dbeta_xc, gt.dbeta_xc)
    assert dump.ddd_paw is None  # US-only synthetic: no PAW record


def test_nc_v12_accepted_vertex_is_conjugated_dbeta_only(tmp_path):
    p = tmp_path / "qe_nc_v12_synth.bin"
    gt = write_qe_dump(
        p,
        magic=MAGIC_V12,
        nspin=2,
        nat=3,
        nsp=1,
        nhm=3,
        atm=(b"Fe",),
        ityp=(1, 1, 1),
        nh=(3,),
        tvanp=(0,),
        tpawp=(0,),
        include_becsum=True,
        include_rho_bec=False,
        include_dbeta_xc=True,
        include_ddd_paw=False,
    )
    dump = parse_qe_dump(p)
    assert dump.version == "1.2"
    assert not dump.tvanp.any() and not dump.tpawp.any()
    data = read_qe_dump(p, legacy_no_u=True)
    data.validate(exchange_ready=True)
    assert data.hij_definition == "qe_dbeta_xc_plus_deeq_spin_difference"
    gram = np.asarray(gt.gram[:3, :3, 0])
    expected = (
        np.linalg.inv(gram) @ (gt.dbeta_xc[:3, :3, 0] * RYTOEV) @ np.linalg.inv(gram)
        + (gt.deeq[:3, :3, 0, 0] - gt.deeq[:3, :3, 0, 1]) * RYTOEV
    )
    np.testing.assert_allclose(
        np.asarray(data.get_operator_component("delta_total", site=0)),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_v12_delta_total_is_conjugated_vertex(tmp_path):
    p = tmp_path / "qe_v12b_synth.bin"
    gt = write_qe_dump(p, magic=MAGIC_V12, include_dbeta_xc=True)
    data = read_qe_dump(p, legacy_no_u=True)
    data.validate(exchange_ready=True)
    assert data.hij_definition == "qe_dbeta_xc_plus_deeq_spin_difference"
    delta = data.get_operator_component("delta_total", site=0)
    nh0 = int(gt.nh[gt.ityp[0] - 1])
    gram = np.asarray(gt.gram[:nh0, :nh0, 0])
    # reader conjugates the primal dbeta_xc by the Gram; deeq term unchanged
    expected = (
        np.linalg.inv(gram)
        @ (gt.dbeta_xc[:nh0, :nh0, 0] * RYTOEV)
        @ np.linalg.inv(gram)
        + (gt.deeq[:nh0, :nh0, 0, 0] - gt.deeq[:nh0, :nh0, 0, 1]) * RYTOEV
    )
    np.testing.assert_allclose(np.asarray(delta), expected, rtol=1e-10, atol=1e-12)


def test_v11_delta_total_remains_deeq_difference_but_not_exchange_ready(tmp_path):
    p = tmp_path / "qe_v11_synth.bin"
    gt = write_qe_dump(p, magic=MAGIC_V11)
    data = read_qe_dump(p, legacy_no_u=True)
    meta = data.operator_component_metadata["delta_total"]
    assert meta["exchange_ready"] == "false"
    from TB2J.interfaces.gpaw_projector import component_local_operators

    with pytest.raises(ValueError):
        component_local_operators(data, "delta_total", [0], "test")
    delta = data.get_operator_component("delta_total", site=0)
    nh0 = int(gt.nh[gt.ityp[0] - 1])
    expected = (gt.deeq[:nh0, :nh0, 0, 0] - gt.deeq[:nh0, :nh0, 0, 1]) * RYTOEV
    np.testing.assert_allclose(np.asarray(delta), expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Real golden dumps (optional; skipped when the files are absent)
# ---------------------------------------------------------------------------
def _check_golden_v11(name, *, expect_rho_bec):
    path = GOLDEN_DIR / name
    dump = parse_qe_dump(path)
    assert dump.version == "1.1"
    assert dump.nspin == 2
    assert dump.nkstot == dump.nks
    assert len(dump.coefficients) == dump.nks
    npack = dump.nhm * (dump.nhm + 1) // 2
    assert dump.becsum.shape == (npack, dump.nat, dump.nspin)
    if expect_rho_bec:
        assert dump.rho_bec.shape == (npack, dump.nat, dump.nspin)
    else:
        assert dump.rho_bec is None
    assert any(dump.tvanp) or any(dump.tpawp)

    data = read_qe_dump(path, legacy_no_u=True)
    data.validate(exchange_ready=True)
    # v1.1 dumps are parseable but their deeq-only vertex is falsified as
    # an exchange vertex (metadata-marked); only v1.2 is exchange-ready.
    if dump.version == "1.1":
        assert (
            data.operator_component_metadata["delta_total"]["exchange_ready"] == "false"
        )
    assert data.nproj == dump.nkb
    assert data.nkpt == dump.nks // 2  # dense spin-degenerate fold
    assert np.isfinite(data.kpoints).all()
    assert np.isfinite(data.coefficients).all()
    assert np.isfinite(data.hij).all()
    assert np.isclose(data.weights.sum(), 1.0)


@pytest.mark.skipif(
    not (GOLDEN_DIR / "o_us_scf_v11.bin").exists(),
    reason="golden dump not present",
)
def test_real_golden_v11_us_scf():
    _check_golden_v11("o_us_scf_v11.bin", expect_rho_bec=False)


@pytest.mark.skipif(
    not (GOLDEN_DIR / "fe_us_scf_v11.bin").exists(),
    reason="golden dump not present",
)
def test_real_golden_fe_us_scf():
    _check_golden_v11("fe_us_scf_v11.bin", expect_rho_bec=False)


@pytest.mark.skipif(
    not (GOLDEN_DIR / "fe_paw_scf_v11.bin").exists(),
    reason="golden dump not present",
)
def test_real_golden_fe_paw_scf():
    _check_golden_v11("fe_paw_scf_v11.bin", expect_rho_bec=True)


# ---------------------------------------------------------------------------
# Atomic-wavefunction dumps (TB2JQEATWFC1.0, TB2J/qe_patch/README.md)
# ---------------------------------------------------------------------------
def write_qe_atomic_dump(
    path,
    *,
    nks=4,
    nbnd=3,
    nat=2,
    nsp=2,
    nsite_species=(3, 1),
    nspin=2,
    nkstot=None,
    magic=MAGIC_ATOMIC,
    seed=20261009,
    corrupt_last_record=False,
    per_k_gram=True,
    block_diagonal_gram=False,
    coeffs=None,
    grams=None,
    cov_xc=None,
    cov_aug=None,
) -> SimpleNamespace:
    """Write a synthetic atomic dump; return ground-truth arrays.

    Layout: header, fermi, cell, species, ions, channel metadata, k mesh,
    spectral bands, per-k coefficient/Gram pairs, then the two BZ-weighted
    R=0 vertex records cov_xc / cov_aug (max_nsite, max_nsite, nat).
    """
    rng = np.random.default_rng(seed)
    nsite_arr = np.asarray(nsite_species, dtype="<i4")
    ityp = np.array(([1, 2] + [1] * max(nat - 2, 0))[:nat], dtype="<i4")
    nproj = int(nsite_arr[ityp - 1].sum())
    max_nsite = int(nsite_arr.max())
    if nkstot is None:
        nkstot = nks
    alat = 4.0
    at = np.array([[1.0, 0.3, 0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]])
    bg = np.linalg.inv(at).T
    tau = np.array([[0.0, 0.5, 0.25], [0.5, 0.25, 0.75], [0.1, 0.0, 0.3]])[:, :nat]
    isk = np.array([1, 2] * (nks // 2) + [1, 2][: nks % 2], dtype="<i4")
    # Regular 3-division mesh along axis 0 in fractional coordinates
    # (f = xk @ at): an irregular k list cannot define the alias-free
    # R window of _R_grid_for_cutoff, and even 2 divisions would leave
    # only R=0. Odd 3 divisions keep the symmetric window {-1,0,1}.
    kfrac = np.zeros((nks, 3))
    kfrac[:, 0] = np.mod((np.arange(nks) // 2) / 3.0, 1.0)
    xk = np.asfortranarray((kfrac @ np.linalg.inv(at)).T)
    wk = np.array([1.0, 1.0, 2.0, 2.0][:nks])
    et = rng.normal(size=(nbnd, nks))
    wg = np.abs(rng.normal(size=(nbnd, nks))) * wk

    # Species 1 has one p shell; species 2 has one s shell.
    l_by_species = ([1, 1, 1], [0])
    projector_l = np.array(
        [l for typ in ityp for l in l_by_species[typ - 1]], dtype="<i4"
    )
    m_ordinal = np.array(
        [m for typ in ityp for m in range(1, nsite_arr[typ - 1] + 1)], dtype="<i4"
    )
    radial_upf_index = np.array(
        [1 for typ in ityp for _ in range(nsite_arr[typ - 1])], dtype="<i4"
    )

    if coeffs is None:
        coeffs = [
            rng.normal(size=(nproj, nbnd)) + 1j * rng.normal(size=(nproj, nbnd))
            for _ in range(nks)
        ]
    else:
        coeffs = [np.array(c, dtype=complex) for c in coeffs]

    def _spd_matrix(n):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return a @ a.conj().T + n * np.eye(n)

    site_counts = nsite_arr[ityp - 1]
    offsets = np.concatenate([[0], np.cumsum(site_counts)])
    if grams is None:
        if block_diagonal_gram:
            # site-block-diagonal SPD Gram (still nonorthogonal within sites)
            matrix = np.zeros((nproj, nproj), dtype=complex)
            for site in range(nat):
                lo, hi = int(offsets[site]), int(offsets[site + 1])
                matrix[lo:hi, lo:hi] = _spd_matrix(hi - lo)
            grams = [matrix.copy() for _ in range(nks)]
        elif per_k_gram:
            # Gram matrices differ per k: random unitary similarity of a
            # fixed SPD reference, so each M(k) is Hermitian positive
            # definite and clearly k-dependent.
            reference = _spd_matrix(nproj)
            grams = []
            for ik in range(nks):
                q = np.linalg.qr(
                    rng.normal(size=(nproj, nproj))
                    + 1j * rng.normal(size=(nproj, nproj))
                )[0]
                grams.append(q @ reference @ q.conj().T)
        else:
            single = _spd_matrix(nproj)
            grams = [single.copy() for _ in range(nks)]
    else:
        grams = [np.array(g, dtype=complex) for g in grams]

    def _default_vertex():
        vertex = rng.normal(size=(max_nsite, max_nsite, nat)) + 1j * rng.normal(
            size=(max_nsite, max_nsite, nat)
        )
        for site in range(nat):
            n = int(site_counts[site])
            vertex[:n, :n, site] += vertex[:n, :n, site].conj().T
        return vertex

    if cov_xc is None:
        cov_xc = _default_vertex()
    else:
        cov_xc = np.array(cov_xc, dtype=complex)
    if cov_aug is None:
        cov_aug = _default_vertex()
    else:
        cov_aug = np.array(cov_aug, dtype=complex)

    with open(path, "wb") as fh:
        # QE Fortran CHARACTER(LEN=16) pads the magic with blanks
        _frecord(
            fh,
            magic.ljust(16, b" "),
            np.array(
                [nspin, nks, nkstot, nbnd, nat, nsp, nproj, max_nsite], dtype="<i4"
            ),
        )
        _frecord(
            fh,
            np.array([12.0, -1.25, -1.2, -1.3, 0.01]),
            np.array([1, 0, 1], dtype="<i4"),
        )
        _frecord(
            fh,
            at,
            bg,
            np.array([alat, alat**3 * abs(np.linalg.det(at))]),
            np.array([1], dtype="<i4"),
        )
        _frecord(fh, np.array([b"Fe", b"O "], dtype="S3"))
        _frecord(fh, ityp, tau)
        _frecord(
            fh,
            nsite_arr,
            projector_l,
            m_ordinal,
            radial_upf_index,
        )
        _frecord(fh, xk, wk, isk)
        _frecord(fh, et, wg)
        for ik in range(nks):
            _frecord(fh, coeffs[ik])
            _frecord(fh, grams[ik])
        if corrupt_last_record:
            payload = np.zeros(10, dtype="<c16")
            fh.write(struct.pack("<i", payload.nbytes))
            fh.write(payload.tobytes(order="F"))
            fh.write(struct.pack("<i", payload.nbytes + 8))  # mismatched marker
            return SimpleNamespace(_aborted=True)
        _frecord(fh, cov_xc)
        _frecord(fh, cov_aug)

    return SimpleNamespace(
        path=Path(path),
        magic=magic.rstrip().decode("ascii"),
        nspin=nspin,
        nks=nks,
        nkstot=nkstot,
        nbnd=nbnd,
        nat=nat,
        nsp=nsp,
        nproj=nproj,
        max_nsite=max_nsite,
        atm=["Fe", "O"],
        ityp=ityp.astype(int),
        tau=tau.T.copy(),
        nsite_species=nsite_arr.astype(int),
        projector_l=projector_l.astype(int),
        m_ordinal=m_ordinal.astype(int),
        radial_upf_index=radial_upf_index.astype(int),
        xk=xk.T.copy(),
        wk=wk,
        isk=isk.astype(int),
        et=et.T.copy(),
        wg=wg.T.copy(),
        coefficients=coeffs,
        grams=grams,
        cov_xc=cov_xc,
        cov_aug=cov_aug,
        alat=alat,
        at=at,
        ef=-1.25,
        ef_up=-1.2,
        ef_dw=-1.3,
    )


def test_atomic_parse_byte_format_and_site_metadata(tmp_path):
    """Header magic, record order/sizes and channel metadata round-trip."""
    gt = write_qe_atomic_dump(tmp_path / "atomic.bin")
    dump = parse_qe_atomic_dump(gt.path)
    assert dump.magic == "TB2JQEATWFC1.0"
    assert (dump.nspin, dump.nks, dump.nkstot) == (2, gt.nks, gt.nks)
    assert (dump.nbnd, dump.nat, dump.nsp) == (gt.nbnd, gt.nat, gt.nsp)
    assert (dump.nproj, dump.max_nsite) == (gt.nproj, gt.max_nsite)
    assert dump.nsite_species.tolist() == [3, 1]
    assert dump.projector_l.tolist() == [1, 1, 1, 0]
    assert dump.m_ordinal.tolist() == [1, 2, 3, 1]
    assert dump.radial_upf_index.tolist() == [1, 1, 1, 1]
    assert dump.ofswfc.tolist() == [0, 3]
    assert dump.nsite_per_atom.tolist() == [3, 1]
    for ik, (coeff, gram) in enumerate(zip(dump.coefficients, dump.grams)):
        np.testing.assert_allclose(coeff, gt.coefficients[ik])
        np.testing.assert_allclose(gram, gt.grams[ik])
        assert np.allclose(gram, gram.conj().T)
    np.testing.assert_allclose(dump.cov_xc, gt.cov_xc)
    np.testing.assert_allclose(dump.cov_aug, gt.cov_aug)
    # header dims payload is exactly 8 i4 words (distinct KB schema has 9)
    with open(gt.path, "rb") as fh:
        assert struct.unpack("<i", fh.read(4))[0] == 16 + 8 * 4
    # KB parser rejects the atomic family explicitly
    with pytest.raises(ValueError, match="TB2JQEATWFC1.0"):
        parse_qe_dump(gt.path)


def test_atomic_parse_rejects_kb_family_magic(tmp_path):
    """The atomic parser accepts only the atomic family magic."""
    write_qe_dump(tmp_path / "kb.bin")
    with pytest.raises(ValueError, match="TB2JQEDUMPV1.1"):
        parse_qe_atomic_dump(tmp_path / "kb.bin")


def test_atomic_malformed_gram_record_raises(tmp_path):
    gt = write_qe_atomic_dump(tmp_path / "corrupt.bin", corrupt_last_record=True)
    assert getattr(gt, "_aborted", False)
    with pytest.raises(ValueError, match="record-length mismatch"):
        parse_qe_atomic_dump(tmp_path / "corrupt.bin")


def test_atomic_trailing_data_raises(tmp_path):
    path = tmp_path / "extra.bin"
    write_qe_atomic_dump(path)
    with open(path, "ab") as fh:
        fh.write(struct.pack("<i", 0) * 2)
    with pytest.raises(ValueError, match="trailing data"):
        parse_qe_atomic_dump(path)


def test_atomic_per_k_varying_gram_exported_as_overlap_k(tmp_path):
    """M(k) is exported per k, full basis, and not reduced to one matrix."""
    gt = write_qe_atomic_dump(tmp_path / "perk.bin")
    data = read_qe_atomic_dump(gt.path, legacy_no_u=True)
    nk = gt.nks // 2
    assert data.overlap_k is not None
    assert data.overlap_k.shape == (nk, gt.nproj, gt.nproj)
    for j, ik in enumerate((0, 2)):
        np.testing.assert_allclose(data.overlap_k[j], gt.grams[ik])
    # distinct k-points really carry distinct metrics (up mesh = ks 0, 2)
    assert not np.allclose(data.overlap_k[0], data.overlap_k[1])
    # constant-metric variant folds one shared M(k) everywhere
    gt2 = write_qe_atomic_dump(tmp_path / "same.bin", per_k_gram=False)
    data2 = read_qe_atomic_dump(gt2.path, legacy_no_u=True)
    np.testing.assert_allclose(data2.overlap_k[0], gt2.grams[0])
    np.testing.assert_allclose(data2.overlap_k[1], gt2.grams[0])


def test_atomic_site_channel_map_and_spin_folding(tmp_path):
    """Channels split by nsite_species[ityp]; spins fold by isk."""
    gt = write_qe_atomic_dump(tmp_path / "sites.bin")
    data = read_qe_atomic_dump(gt.path, legacy_no_u=True)
    assert data.site_nproj.tolist() == [3, 1]
    assert data.projector_site.tolist() == [0, 0, 0, 1]
    assert data.site_projector_indices.tolist() == [[0, 1, 2], [3, -1, -1]]
    assert data.coefficients.shape == (2, gt.nks // 2, gt.nbnd, gt.nproj)
    np.testing.assert_allclose(data.coefficients[0, 0], gt.coefficients[0].T)
    np.testing.assert_allclose(data.coefficients[1, 0], gt.coefficients[1].T)
    np.testing.assert_allclose(data.coefficients[0, 1], gt.coefficients[2].T)
    np.testing.assert_allclose(data.coefficients[1, 1], gt.coefficients[3].T)
    assert data.projector_l.tolist() == gt.projector_l.tolist()
    assert data.projector_m.tolist() == gt.m_ordinal.tolist()
    assert data.projector_radial.tolist() == gt.radial_upf_index.tolist()
    assert "harmonic ordinal" in data.metadata["m_ordinal_note"]


def test_atomic_repeated_species_keeps_each_atom_channel_block(tmp_path):
    """Two Fe atoms and one O count three site blocks, not two species."""
    gt = write_qe_atomic_dump(tmp_path / "repeated.bin", nat=3)
    dump = parse_qe_atomic_dump(gt.path)
    data = read_qe_atomic_dump(gt.path, legacy_no_u=True)
    assert gt.ityp.tolist() == [1, 2, 1]
    assert dump.nproj == 7
    assert dump.ofswfc.tolist() == [0, 3, 4]
    assert data.site_nproj.tolist() == [3, 1, 3]
    assert data.site_projector_indices.tolist() == [[0, 1, 2], [3, -1, -1], [4, 5, 6]]
    np.testing.assert_allclose(data.overlap_k[0], gt.grams[0])


def test_atomic_covariant_vertex_and_hij_signs(tmp_path):
    """delta_total = (cov_xc + cov_aug) Ry->eV; hij(up/dn) = +/-Delta/2."""
    gt = write_qe_atomic_dump(tmp_path / "vertex.bin")
    data = read_qe_atomic_dump(gt.path, legacy_no_u=True)
    expected = (gt.cov_xc + gt.cov_aug).transpose(2, 0, 1) * RYTOEV
    np.testing.assert_allclose(
        np.asarray(data.get_operator_component("delta_total")),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(data.hij[0], expected / 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(data.hij[1], -expected / 2.0, rtol=1e-12, atol=1e-12)
    assert data.hij_definition == "qe_atomic_pao_projected_spin_vertex"
    assert data.hij_units == "eV"
    assert data.coefficient_source == "qe_atomic_wfc"
    assert data.coefficient_projector == "qe_upf_pswfc"
    assert data.channel_interpretation == "qe_primal_pseudo_atomic"
    assert data.operator_basis == "qe_pseudo_atomic_orbital_site"
    meta = data.operator_component_metadata["delta_total"]
    assert meta["input_unit"] == "Ry"
    assert meta["exchange_ready"] == "true"


def test_atomic_overall_and_per_spin_fermi(tmp_path):
    gt = write_qe_atomic_dump(tmp_path / "fermi.bin")
    data = read_qe_atomic_dump(gt.path, legacy_no_u=True)
    np.testing.assert_allclose(data.efermi, -1.25 * RYTOEV)
    np.testing.assert_allclose(data.efermi_spin, np.array([-1.2, -1.3]) * RYTOEV)


def test_atomic_nspin_and_pool_gates(tmp_path):
    write_qe_atomic_dump(tmp_path / "nmag.bin", nspin=1)
    with pytest.raises(ValueError, match="nspin=1"):
        parse_qe_atomic_dump(tmp_path / "nmag.bin")
    write_qe_atomic_dump(tmp_path / "pool.bin", nkstot=8)
    with pytest.raises(ValueError, match="pool-parallel"):
        parse_qe_atomic_dump(tmp_path / "pool.bin")


def test_atomic_zero_channel_species_rejected(tmp_path):
    """A mixed-UPF dump must not silently omit a site's projector basis."""
    path = tmp_path / "missing_pswfc.bin"
    write_qe_atomic_dump(path)
    with open(path, "r+b") as fh:
        for _ in range(5):
            size = struct.unpack("<i", fh.read(4))[0]
            fh.seek(size + 4, 1)
        size = struct.unpack("<i", fh.read(4))[0]
        assert size > 4
        fh.write(struct.pack("<i", 0))
    with pytest.raises(ValueError, match="every UPF species needs PP_PSWFC"):
        parse_qe_atomic_dump(path)


def test_atomic_inconsistent_nproj_raises(tmp_path):
    """Header nproj inconsistent with sum(nsite_species) is rejected."""
    path = tmp_path / "badnp.bin"
    write_qe_atomic_dump(path)
    with open(path, "r+b") as fh:
        # header dims payload: 8 i4 words after the 16-byte magic; nproj is
        # the 7th word
        fh.seek(4 + 16 + 6 * 4)
        fh.write(struct.pack("<i", 5))  # true nproj is 4
    with pytest.raises(ValueError, match="inconsistent"):
        parse_qe_atomic_dump(path)


def test_kb_flags_rejected_for_atomic_input_and_versa(tmp_path):
    """overlap controls are atomic-only; KB inputs reject them explicitly."""
    atomic = write_qe_atomic_dump(tmp_path / "atomic2.bin")
    kb = write_qe_dump(tmp_path / "kb2.bin")
    with pytest.raises(ValueError, match="only to atomic-wavefunction dumps"):
        gen_exchange_qe(
            kb.path,
            overlap_mode="inverse",
            output_path=str(tmp_path / "out_kb"),
            legacy_no_u=True,
        )
    with pytest.raises(ValueError, match="only to atomic-wavefunction dumps"):
        gen_exchange_qe(
            kb.path,
            overlap_rcond=1e-8,
            output_path=str(tmp_path / "out_kb2"),
            legacy_no_u=True,
        )
    # accepted on the atomic family (validated end-to-end below)
    gen_exchange_qe(
        atomic.path,
        output_path=str(tmp_path / "out_atomic"),
        overlap_mode="inverse",
        overlap_rcond=1e-10,
        legacy_no_u=True,
    )
    assert (tmp_path / "out_atomic" / "exchange.out").exists()


def test_atomic_nonorthogonal_overlap_dressing_invariant(tmp_path):
    """Dressed exchange is invariant under per-site orthogonalization.

    The synthetic Gram is site-block-diagonal and nonorthogonal.  Replacing
    M(k) by I while transforming C -> U C and the covariant site vertex
    Delta -> U Delta U^dagger (U = M^-1/2 per site block) leaves the
    M^-1-dressed channel Green function — and therefore the TB2J J(0,1)
    computed by the real pipeline — unchanged.  A reader that ignored
    M(k), or preconjugated Delta by M^-1, would break the equality.
    """
    gt = write_qe_atomic_dump(
        tmp_path / "inv.bin", nks=2, nbnd=2, block_diagonal_gram=True, seed=88
    )
    assert not np.allclose(gt.grams[0], np.eye(gt.nproj))
    counts = gt.nsite_species[gt.ityp - 1]
    offsets = np.concatenate([[0], np.cumsum(counts)])

    us = []
    for site in range(gt.nat):
        lo, hi = int(offsets[site]), int(offsets[site + 1])
        evals, vecs = np.linalg.eigh(gt.grams[0][lo:hi, lo:hi])
        us.append((vecs / np.sqrt(evals)) @ vecs.conj().T)  # M_s^-1/2

    coeffs_t = []
    for coeff in gt.coefficients:
        transformed = coeff.copy()
        for site in range(gt.nat):
            lo, hi = int(offsets[site]), int(offsets[site + 1])
            transformed[lo:hi] = us[site] @ transformed[lo:hi]
        coeffs_t.append(transformed)
    grams_t = [np.eye(gt.nproj, dtype=complex) for _ in gt.coefficients]
    vertices_t = []
    for vertex in (gt.cov_xc, gt.cov_aug):
        out = np.zeros_like(vertex)
        for site in range(gt.nat):
            n = int(counts[site])
            u = np.zeros((gt.max_nsite, gt.max_nsite), dtype=complex)
            u[:n, :n] = us[site]
            out[:, :, site] = u @ vertex[:, :, site] @ u.conj().T
        vertices_t.append(out)

    trans = write_qe_atomic_dump(
        tmp_path / "inv_trans.bin",
        nks=2,
        nbnd=2,
        coeffs=coeffs_t,
        grams=grams_t,
        cov_xc=vertices_t[0],
        cov_aug=vertices_t[1],
        seed=88,
    )
    kw = dict(Rcut=5.0, nz=6, smearing_eV=0.05, overlap_mode="inverse")
    _, jdict = gen_exchange_qe(
        gt.path, output_path=str(tmp_path / "out0"), **kw, legacy_no_u=True
    )
    _, jdict_t = gen_exchange_qe(
        trans.path, output_path=str(tmp_path / "out1"), **kw, legacy_no_u=True
    )
    j01 = jdict[((0, 0, 0), 0, 1)]
    assert j01 != 0.0
    np.testing.assert_allclose(jdict_t[((0, 0, 0), 0, 1)], j01, rtol=1e-6)


def _write_spinor_records(path, records):
    with open(path, "wb") as fh:
        for record in records:
            _frecord(fh, *record)


def write_qe_spinor_dump(path, axis="z", soc=False, total_shift=0.0, version="1.0"):
    """Physical synthetic collinear spectra rotated into x/y/z references.

    Complex orbital phases and a nontrivial beta Gram expose conjugation,
    Pauli-y signs and primal/dual mistakes. Total fixed SOC is diagnostic.
    ``version="1.1"`` appends record 13+nks, the nine native source-response
    blocks: covariant ``D_{c<-p} = magnetic_block (x) sigma_c`` for the
    scalar family, plus Hermitian off-diagonal-response contamination
    (``c != p``) in the ``soc`` variant so the explicit tangent differs
    from the commutator.
    """
    from TB2J.projector_green import PAULI_MATRICES

    rng = np.random.default_rng(741)
    nk, nbnd, nhm, nat, nkb = 2, 8, 2, 2, 4
    sigma = PAULI_MATRICES["xyz".index(axis)]
    _, spin_vectors = np.linalg.eigh(sigma)
    spin_vectors = spin_vectors[:, ::-1]  # up (+1) first
    orbital = np.empty((nk, 4, 4), dtype=complex)
    coefficients = np.zeros((nk, nkb, 2, nbnd), dtype=complex)
    for ik in range(nk):
        orbital[ik], _ = np.linalg.qr(
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        )
        for spin in range(2):
            coefficients[ik, :, :, 4 * spin : 4 * (spin + 1)] = (
                orbital[ik, :, None, :] * spin_vectors[None, :, spin, None]
            )
    gram_block = np.array([[1.4, 0.15j], [-0.15j, 1.1]])
    field_block = np.array([[0.03, 0.009j], [-0.009j, 0.04]])
    magnetic_block = np.array([[0.01, 0.003j], [-0.003j, 0.012]])
    gram = np.repeat(gram_block[:, :, None], nat, axis=2)
    field = np.zeros((nhm, nhm, nat, 3), dtype=complex)
    field[:, :, :, "xyz".index(axis)] = field_block[:, :, None]
    magnetic = np.repeat(
        np.einsum("ij,st->ijst", magnetic_block, sigma).reshape(2, 2, 1, 4), nat, axis=2
    )
    fixed_spin = np.eye(2) * (0.2 + total_shift)
    if soc:
        fixed_spin = fixed_spin + 0.06 * PAULI_MATRICES[1]
    fixed = np.einsum("ij,st->ijst", np.eye(2), fixed_spin).reshape(2, 2, 1, 4)
    total = magnetic + fixed
    xk = np.array([[0, 0.5], [0, 0], [0, 0]], dtype=float)
    wk = np.full(nk, 0.5)
    et = np.array(
        [
            [-0.12, -0.04, 0.03, 0.1, -0.08, -0.01, 0.06, 0.15],
            [-0.11, -0.03, 0.04, 0.12, -0.07, 0.00, 0.07, 0.17],
        ]
    ).T
    wg = (et < 0).astype(float) * wk
    tau = np.array([[0, 0.5], [0, 0.5], [0, 0.5]])
    d_response = None
    if version == "1.1":
        p = "xyz".index(axis)
        d_response = np.zeros((nhm, nhm, nat, 4, 3, 3), dtype=complex)
        for c in range(3):
            resp = np.einsum("ij,st->ijst", magnetic_block, PAULI_MATRICES[c])
            if soc and c != p:
                off_block = np.array([[0.004, -0.002j], [0.002j, 0.007]])
                resp = resp + np.einsum(
                    "ij,st->ijst", off_block, PAULI_MATRICES[(c + 1) % 3]
                )
            d_response[:, :, :, :, c, p] = np.repeat(
                resp.reshape(nhm, nhm, 1, 4), nat, axis=2
            )
    records = [
        (
            np.array([MAGIC_SPINOR.ljust(16, b" ")], dtype="S16"),
            np.array([4, nk, nk, nbnd, nat, 1, nhm, 1, nkb, int(soc), 1], dtype="<i4"),
        ),
        (np.array([3.5, 0, 0.01]), np.array([-99, 0, 1], dtype="<i4")),
        (np.eye(3), np.eye(3), np.array([4.0, 64.0]), np.array([1], dtype="<i4")),
        (np.array([b"Fe    "] + [b"      "] * 9, dtype="S6"),),
        (np.ones(nat, dtype="<i4"), tau),
        tuple(np.array([v], dtype="<i4") for v in (nhm, 1, 0, int(soc))),
        (total,),
        (magnetic,),
        (gram,),
        (xk, wk),
        (et, wg),
        *[(coefficients[ik],) for ik in range(nk)],
        (field,),
        (np.zeros((3, nat, 4)),),
    ]
    if version == "1.1":
        records.append((d_response,))
        records[0] = (
            np.array([MAGIC_SPINOR_NC11.ljust(16, b" ")], dtype="S16"),
            records[0][1],
        )
    _write_spinor_records(path, records)
    return SimpleNamespace(
        path=path,
        records=records,
        coefficients=coefficients,
        gram=gram,
        field=field,
        magnetic=magnetic,
        total=total,
        orbital=orbital,
        et=et,
        wg=wg,
        xk=xk,
        wk=wk,
        tau=tau,
        d_response=d_response,
        magnetic_block=magnetic_block,
        field_block=field_block,
    )


def write_qe_interleaved_spinor_dump(path, *, zero_beta_oxygen=False):
    """Fe/O/Fe beta stream; zero-channel O is admitted only by modern NC1.2."""
    native = write_qe_spinor_dump(path, axis="z", version="1.1")
    records = list(native.records)
    dimensions = records[0][1].copy()
    dimensions[[4, 5, 8]] = [3, 2, 4 if zero_beta_oxygen else 5]
    records[0] = (records[0][0], dimensions)
    records[3] = (np.array([b"Fe    ", b"O     "], dtype="S6"),)
    native.tau = np.column_stack((native.tau, [0.25, 0.0, 0.5]))
    records[4] = (np.array([1, 2, 1], dtype="<i4"), native.tau)
    records[5] = tuple(
        np.array(values, dtype="<i4")
        for values in ([2, 0 if zero_beta_oxygen else 1], [1, 1], [0, 0], [0, 0])
    )
    for name in ("gram", "field", "magnetic", "total", "d_response"):
        values = getattr(native, name)
        values = np.concatenate((values, values[:, :, :1]), axis=2)
        values[1, :, 1] = 0
        values[:, 1, 1] = 0
        if zero_beta_oxygen:
            values[:, :, 1] = 0
        setattr(native, name, values)
    if not zero_beta_oxygen:
        native.coefficients = np.concatenate(
            (native.coefficients, (0.7 + 0.2j) * native.coefficients[:, :1]), axis=1
        )
    records[6:9] = [(native.total,), (native.magnetic,), (native.gram,)]
    records[11:13] = [(coefficients,) for coefficients in native.coefficients]
    records[13:] = [
        (native.field,),
        (np.zeros((3, 3, 4)),),
        (native.d_response,),
    ]
    _write_spinor_records(path, records)
    native.records = records
    return native


def test_interleaved_spinor_species_preserve_site_resolved_band_action(tmp_path):
    from TB2J.projector_green import PAULI_MATRICES

    native = write_qe_interleaved_spinor_dump(tmp_path / "interleaved-spinor.bin")
    data = read_qe_spinor_dump(native.path, legacy_no_u=True)
    green = ProjectorGreen(data)
    for site, rows in enumerate(([0, 1], [4], [2, 3])):
        count = len(rows)
        inverse = np.linalg.inv(native.gram[:count, :count, site])
        smooth = inverse @ native.field[:count, :count, site, 2] @ inverse
        native_operator = (
            2
            * RYTOEV
            * (
                np.einsum("pq,st->pqst", smooth, PAULI_MATRICES[2])
                + native.magnetic[:count, :count, site].reshape(count, count, 2, 2)
            )
        )
        coefficients = native.coefficients[0, rows]
        expected = np.einsum(
            "psb,pqst,qtc->bc", coefficients.conj(), native_operator, coefficients
        )
        for selected in (
            green.get_site_projectors(site),
            np.flatnonzero(data.projector_site == site),
            np.flatnonzero(data.projector_atom == site),
        ):
            coefficients = data.coefficients[0, 0][:, :, selected].transpose(2, 1, 0)
            actual = np.einsum(
                "psb,pqst,qtc->bc",
                coefficients.conj(),
                green.get_local_operator_spinor(site),
                coefficients,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


def test_spinor_complex_vertex_native_packing_and_dual_green(tmp_path):
    from TB2J.projector_green import PAULI_MATRICES

    gt = write_qe_spinor_dump(tmp_path / "y.bin", axis="y")
    raw = parse_qe_spinor_dump(gt.path)
    data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    inverse = np.linalg.inv(gt.gram[:, :, 0])
    smooth = inverse @ gt.field[:, :, 0, 1] @ inverse
    expected = (
        2
        * RYTOEV
        * (
            np.einsum("ij,st->ijst", smooth, PAULI_MATRICES[1])
            + gt.magnetic[:, :, 0].reshape(2, 2, 2, 2)
        )
    )
    np.testing.assert_allclose(data.spinor_operator[0], expected)
    flattened = expected.transpose(0, 2, 1, 3).reshape(4, 4)
    np.testing.assert_allclose(flattened, flattened.conj().T)
    assert np.max(np.abs(expected.imag)) > 0.1
    np.testing.assert_allclose(raw.projected_field, gt.field)
    np.testing.assert_allclose(data.eigenvalues[0], gt.et.T * RYTOEV)
    np.testing.assert_allclose(data.occupations[0], gt.wg.T / gt.wk[:, None])
    assert data.overlap_k is None
    assert data.efermi_spin is None
    data.validate(exchange_ready=True)
    green = ProjectorGreen(data)
    assert green.coefficients_are_dual
    # A spectral Green contraction must preserve the complex coefficients
    # without any extra Gram inverse. This is a consumer-level phase oracle.
    energy = 1.3 + 0.7j
    direct = np.einsum(
        "psb,qtb,b->pqst",
        gt.coefficients[1],
        gt.coefficients[1].conj(),
        1.0 / (energy - gt.et[:, 1] * RYTOEV),
    )
    np.testing.assert_allclose(green.get_Gk_spinor(1, energy), direct)


def test_spinor_nc10_refuses_fully_relativistic_exchange(tmp_path):
    soc = write_qe_spinor_dump(tmp_path / "soc.bin", soc=True)
    raw = parse_qe_spinor_dump(soc.path)
    assert raw.lspinorb and raw.has_so[0]
    with pytest.raises(
        ValueError, match="lacks native source-resolved magnetic tangents"
    ):
        read_qe_spinor_dump(soc.path, legacy_no_u=True)
    references = {
        axis: write_qe_spinor_dump(tmp_path / f"{axis}.bin", axis=axis, soc=True).path
        for axis in "xyz"
    }
    with pytest.raises(
        ValueError, match="lacks native source-resolved magnetic tangents"
    ):
        gen_exchange_qe(
            references, output_path=tmp_path / "no_exchange", legacy_no_u=True
        )
    assert not (tmp_path / "no_exchange" / "exchange.out").exists()


def _write_matching_collinear_dump(path, gt):
    """Scalar-relativistic NC1.0 source written as validated KB v1.2.

    Same spinor spectrum (orbital unitary per k, interleaved up/dn bands),
    identity Gram, z field 0.015*I Ry -> reader vertex dbeta = 2*field.
    """
    dbeta = np.repeat(0.03 * np.eye(2)[:, :, None], 2, axis=2)
    with open(path, "wb") as fh:
        _frecord(
            fh,
            np.array([MAGIC_V12], dtype="S16"),
            np.array([2, 4, 4, 4, 2, 1, 2, 1, 4], dtype="<i4"),
        )
        _frecord(
            fh, np.array([3.5, 0.0, 0.0, 0.0, 0.01]), np.array([-99, 0, 1], dtype="<i4")
        )
        _frecord(
            fh, np.eye(3), np.eye(3), np.array([4.0, 64.0]), np.array([1], dtype="<i4")
        )
        _frecord(fh, np.array([b"Fe"], dtype="S3"))
        _frecord(fh, np.ones(2, dtype="<i4"), gt.tau)
        _frecord(fh, *(np.array([v], dtype="<i4") for v in (2, 1, 0)))
        _frecord(fh, np.zeros((2, 2, 2, 2)))
        _frecord(fh, np.zeros((2, 2, 1)))
        _frecord(fh, np.zeros((2, 2, 2)))
        _frecord(fh, np.repeat(np.eye(2, dtype=complex)[:, :, None], 2, axis=2))
        _frecord(
            fh,
            np.tile(gt.xk, (1, 2)),
            np.tile(gt.wk, 2),
            np.array([1, 1, 2, 2], dtype="<i4"),
        )
        _frecord(
            fh,
            np.concatenate([gt.et[:4], gt.et[4:]], axis=1),
            np.concatenate([gt.wg[:4], gt.wg[4:]], axis=1),
        )
        for _spin in range(2):
            for ik in range(2):
                _frecord(fh, gt.orbital[ik])
        _frecord(fh, np.zeros((3, 2, 2)))
        _frecord(fh, dbeta)


def test_spinor_scalar_relativistic_reduces_to_collinear_reader_and_exchange(tmp_path):
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange

    gt = write_qe_spinor_dump(tmp_path / "z.bin")
    # Identity Gram, zero magnetic deeq, pure z field 0.015*I Ry: the exact
    # scalar-relativistic operator kron(Delta sigma_z, I_orb), Delta=0.03 Ry.
    gt.records[8] = (np.repeat(np.eye(2, dtype=complex)[:, :, None], 2, axis=2),)
    gt.records[13][0][...] = 0
    gt.records[13][0][:, :, :, 2] = 0.015 * np.eye(2)[:, :, None]
    gt.records[7][0][...] = 0
    _write_spinor_records(gt.path, gt.records)
    collinear_path = tmp_path / "collinear.bin"
    _write_matching_collinear_dump(collinear_path, gt)
    collinear, spinor = (
        read_qe_dump(collinear_path, legacy_no_u=True),
        read_qe_spinor_dump(gt.path, legacy_no_u=True),
    )
    delta = collinear.operator_components["delta_total"]
    np.testing.assert_allclose(spinor.spinor_operator[..., 0, 0], delta)
    np.testing.assert_allclose(spinor.spinor_operator[..., 1, 1], -delta)
    kwargs = dict(Rpts=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]], nz=8, smearing_eV=0.1)
    jcol = compute_projector_exchange_jdict(
        collinear, operator_component="delta_total", **kwargs
    )
    leg = compute_spinor_projector_exchange(spinor, **kwargs)
    for key, value in jcol.items():
        np.testing.assert_allclose(
            leg[key]["J_leg"][0, 0], value, atol=1e-11, rtol=1e-8
        )
        np.testing.assert_allclose(
            leg[key]["J_leg"][1, 1], value, atol=1e-11, rtol=1e-8
        )
    assert abs(jcol[((0, 0, 0), 0, 1)]) > 1e-8


@pytest.mark.parametrize(
    "record,part,index,value,message",
    [
        (0, 0, 0, b"TB2JQEDUMPNC9.0", "magic/version"),
        (0, 1, 0, 2, "nspin=4"),
        (0, 1, 2, 3, "pool-parallel"),
        (0, 1, 4, 0, "dimensions/site"),
        (0, 1, 8, 5, "site counts"),
        (0, 1, 9, 1, "SO-bearing"),
        (0, 1, 10, 0, "domag"),
        (5, 0, 0, 0, "site counts"),
        (5, 3, 0, 2, "has_so"),
        (6, 0, (0, 0, 0, 0), 1j, "deeq_nc_total.*Hermitian"),
        (7, 0, (0, 0, 0, 0), 1j, "deeq_nc_magnetic.*Hermitian"),
        (8, 0, (0, 0, 0), -2, "positive definite"),
        (9, 0, (0, 1), 0, "duplicate or incomplete"),
        (9, 0, (0, 1), 0.3, "uniform full-BZ"),
        (9, 1, 0, 0.7, "weights"),
        (10, 0, (0, 0), np.nan, "nonfinite bands"),
        (13, 0, (0, 1, 0, 2), 1j, "Pauli field.*Hermitian"),
    ],
)
def test_spinor_rejects_invalid_physical_records(
    tmp_path, record, part, index, value, message
):
    gt = write_qe_spinor_dump(tmp_path / "bad.bin")
    gt.records[record][part][index] = value
    _write_spinor_records(gt.path, gt.records)
    with pytest.raises(ValueError, match=message):
        read_qe_spinor_dump(gt.path, legacy_no_u=True)


@pytest.mark.parametrize(
    "damage",
    ["missing", "dummy", "trailing", "marker", "negative", "truncate", "real_field"],
)
def test_spinor_rejects_corrupt_schema(tmp_path, damage):
    gt = write_qe_spinor_dump(tmp_path / "bad_schema.bin")
    if damage == "missing":
        del gt.records[7]
    elif damage == "dummy":
        gt.records[7] = (np.zeros(1, dtype="<i4"),)
    elif damage == "trailing":
        gt.records.append((np.zeros(1),))
    elif damage == "real_field":
        gt.records[13] = (gt.records[13][0].real,)
    _write_spinor_records(gt.path, gt.records)
    if damage in ("marker", "negative", "truncate"):
        payload = bytearray(gt.path.read_bytes())
        if damage == "marker":
            payload[64:68] = struct.pack("<i", 61)
        elif damage == "negative":
            payload[:4] = struct.pack("<i", -60)
        else:
            payload = payload[:-10]
        gt.path.write_bytes(payload)
    with pytest.raises(ValueError):
        parse_qe_spinor_dump(gt.path)


def test_spinor_zero_source_and_cross_family_refusals(tmp_path):
    gt = write_qe_spinor_dump(tmp_path / "zero.bin")
    gt.records[7][0][...] = 0
    gt.records[13][0][...] = 0
    _write_spinor_records(gt.path, gt.records)
    with pytest.raises(ValueError, match="zero magnetic source"):
        parse_qe_spinor_dump(gt.path)
    collinear = write_qe_dump(tmp_path / "kb.bin")
    atomic = write_qe_atomic_dump(tmp_path / "atomic.bin")
    for path in (collinear.path, atomic.path):
        with pytest.raises(ValueError, match="KB only"):
            parse_qe_spinor_dump(path)
    good = write_qe_spinor_dump(tmp_path / "good.bin")
    with pytest.raises(ValueError, match="transverse leg"):
        gen_exchange_qe(good.path, legacy_no_u=True)
    with pytest.raises(ValueError, match="x/y/z"):
        gen_exchange_qe({"z": good.path}, legacy_no_u=True)
    for control in ({"overlap_mode": "inverse"}, {"overlap_rcond": 1e-8}):
        with pytest.raises(ValueError, match="already dual"):
            gen_exchange_qe(
                dict.fromkeys(("x", "y", "z"), good.path), **control, legacy_no_u=True
            )


def _spinor_references(tmp_path):
    return {
        axis: write_qe_spinor_dump(tmp_path / f"{axis}.bin", axis=axis).path
        for axis in "xyz"
    }


def test_spinor_three_leg_output_matches_raw_transverse_exchange(tmp_path):
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange

    references = _spinor_references(tmp_path)
    kwargs = dict(Rpts=[[0, 0, 0]], nz=8, smearing_eV=0.1)
    output, jdict = gen_exchange_qe(
        references, output_path=tmp_path / "out", **kwargs, legacy_no_u=True
    )
    leg = compute_spinor_projector_exchange(
        read_qe_spinor_dump(references["z"], legacy_no_u=True), **kwargs
    )
    np.testing.assert_allclose(
        jdict[((0, 0, 0), 0, 1)],
        leg[((0, 0, 0), 0, 1)]["J_leg"][0, 0],
        atol=1e-10,
        rtol=1e-8,
    )
    assert abs(jdict[((0, 0, 0), 0, 1)]) > 1e-8
    from TB2J.io_exchange.io_exchange import SpinIO

    stored = SpinIO.load_pickle(path=output.parent)
    assert np.isclose(
        stored.get_Jiso(0, 1, (0, 0, 0)), jdict[((0, 0, 0), 0, 1)], atol=1e-10
    )


@pytest.mark.parametrize(
    "mismatch", ["axis", "weights", "mesh", "geometry", "field", "family"]
)
def test_spinor_three_leg_merge_refuses_inconsistent_references(tmp_path, mismatch):
    references = _spinor_references(tmp_path)
    if mismatch == "axis":
        write_qe_spinor_dump(references["x"], axis="z")
        message = "not axis-aligned"
    else:
        gt = write_qe_spinor_dump(references["x"], axis="x")
        if mismatch == "weights":
            gt.records[9][1][...] = [0.4, 0.6]
            message = "weights"
        elif mismatch == "mesh":
            gt.records[9][0][0, 1] += 0.3
            message = "uniform full-BZ"
        elif mismatch == "geometry":
            gt.records[4][1][0, 1] += 0.1
            message = "geometry, k mesh, or band window"
        elif mismatch == "field":
            gt.records[13][0][...] *= 1.6
            gt.records[7][0][...] *= 1.6
            message = "leg diagonals disagree"
        else:
            gt.records[5][2][0] = 1
            message = "UPF/projector family"
        _write_spinor_records(gt.path, gt.records)
    with pytest.raises(ValueError, match=message):
        gen_exchange_qe(
            references,
            output_path=tmp_path / "refused",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "refused" / "exchange.out").exists()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--input_x", "x"],
        ["--input", "c", "--input_x", "x"],
        [
            "--input_x",
            "x",
            "--input_y",
            "y",
            "--input_z",
            "z",
            "--overlap_mode",
            "inverse",
        ],
    ],
)
def test_qe_spinor_cli_requires_exclusive_complete_inputs(monkeypatch, argv):
    from TB2J.scripts.qe2J import run_qe2J

    monkeypatch.setattr("sys.argv", ["qe2J.py", *argv])
    with pytest.raises(SystemExit) as failure:
        run_qe2J()
    assert failure.value.code == 2


def test_qe_spinor_cli_writes_three_leg_exchange_without_mocking(tmp_path, monkeypatch):
    from test_qe_hubbard import _spinor_no_u_dump

    from TB2J.scripts.qe2J import run_qe2J

    references = {
        axis: _spinor_no_u_dump(tmp_path / f"{axis}.bin", axis=axis) for axis in "xyz"
    }
    output = tmp_path / "cli"
    argv = ["qe2J.py", "--output_path", str(output), "--nz", "8"]
    for axis, path in references.items():
        argv.extend([f"--input_{axis}", str(path)])
    monkeypatch.setattr("sys.argv", argv)
    run_qe2J()
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
    from TB2J.io_exchange.io_exchange import SpinIO

    stored = SpinIO.load_pickle(path=output)
    leg = compute_spinor_projector_exchange(
        read_qe_spinor_dump(references["z"]),
        Rpts=[[0, 0, 0]],
        nz=8,
    )
    assert np.isclose(
        stored.get_Jiso(0, 1, (0, 0, 0)),
        leg[((0, 0, 0), 0, 1)]["J_leg"][0, 0],
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# NC1.1 native source-response consumer (story 002)
# ---------------------------------------------------------------------------
def test_spinor_nc11_parse_records_and_diagonal_gate(tmp_path):
    gt = write_qe_spinor_dump(tmp_path / "nc11.bin", version="1.1")
    raw = parse_qe_spinor_dump(gt.path)
    assert raw.version == "1.1" and raw.magic == "TB2JQEDUMPNC1.1"
    assert raw.d_response is not None
    np.testing.assert_allclose(raw.d_response, gt.d_response)
    data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    assert data.spinor_tangent_vertices is not None
    assert data.metadata["qe_magic"] == "TB2JQEDUMPNC1.1"
    # diagonal identity gate: corrupt one D_{z<-z} slab
    bad = write_qe_spinor_dump(tmp_path / "bad.bin", version="1.1")
    bad.records[-1][0][:, :, 0, :, 2, 2] += 1e-5
    _write_spinor_records(bad.path, bad.records)
    with pytest.raises(ValueError, match="diagonal identity gate"):
        parse_qe_spinor_dump(bad.path)


def test_spinor_nc11_strict_eof_and_mixed_version_refusals(tmp_path):
    # NC1.0 file with an appended response record is refused
    legacy = write_qe_spinor_dump(tmp_path / "legacy.bin")
    with open(legacy.path, "ab") as fh:
        _frecord(fh, np.zeros((2, 2, 2, 4, 3, 3), dtype=complex))
    with pytest.raises(ValueError, match="trailing records beyond NC1.0"):
        parse_qe_spinor_dump(legacy.path)
    # NC1.1 file missing its response record is refused
    short = write_qe_spinor_dump(tmp_path / "short.bin", version="1.1")
    _write_spinor_records(short.path, short.records[:-1])
    with pytest.raises(ValueError):
        parse_qe_spinor_dump(short.path)
    # a mixed NC1.0/NC1.1 reference set is refused
    references = {
        axis: write_qe_spinor_dump(
            tmp_path / f"{axis}.bin",
            axis=axis,
            version="1.1" if axis == "z" else "1.0",
        ).path
        for axis in "xyz"
    }
    with pytest.raises(ValueError, match="UPF/projector family"):
        gen_exchange_qe(
            references,
            output_path=tmp_path / "mixed",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )


def test_spinor_nc11_scalar_tangent_factor_and_orientation_oracles(tmp_path):
    from TB2J.projector_green import PAULI_MATRICES, magnetic_tangent_vertices

    for axis in ("z", "x"):
        gt = write_qe_spinor_dump(tmp_path / f"{axis}.bin", axis=axis, version="1.1")
        data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
        p = "xyz".index(axis)
        vertices = data.spinor_tangent_vertices
        inverse = np.linalg.inv(gt.gram[:, :, 0])
        q = inverse @ gt.field[:, :, 0, p] @ inverse + gt.magnetic_block
        # longitudinal vertex vanishes; transverse vertex = RYTOEV * Q (x) sigma
        np.testing.assert_allclose(vertices[0, p], 0.0, atol=1e-14)
        for a in range(3):
            if a == p:
                continue
            expected = RYTOEV * np.einsum("ij,st->ijst", q, PAULI_MATRICES[a])
            np.testing.assert_allclose(vertices[0, a], expected, atol=1e-12)
        # exact parity with the existing commutator tangent
        commutator = magnetic_tangent_vertices(data.spinor_operator[0])["vertices"]
        for a in range(3):
            dense = np.stack([vertices[0, a].transpose(2, 0, 3, 1).reshape(4, 4)])
            np.testing.assert_allclose(dense[0], commutator[a], atol=1e-9)


def test_spinor_nc11_rank_deficient_beta_gram_keeps_vertex_bounded(tmp_path):
    from TB2J.projector_green import PAULI_MATRICES

    gt = write_qe_spinor_dump(tmp_path / "weak_beta.bin", version="1.1")
    bright = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    dark = np.array([1.0, -1.0j]) / np.sqrt(2.0)
    gram = 2.0 * np.outer(bright, bright.conj()) + 2e-9 * np.outer(dark, dark.conj())
    # A tiny finite field in the nearly null channel would become enormous
    # under an unconditioned inverse; only the resolved channel is physical.
    field = 0.04 * np.outer(bright, bright.conj()) + 2e-7 * np.outer(dark, dark.conj())
    gt.gram[...] = gram[:, :, None]
    gt.field[:, :, :, 2] = field[:, :, None]
    _write_spinor_records(gt.path, gt.records)
    data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    resolved = 0.01 * np.outer(bright, bright.conj()) + gt.magnetic_block
    expected = 2 * RYTOEV * np.einsum("ij,st->ijst", resolved, PAULI_MATRICES[2])
    np.testing.assert_allclose(data.spinor_operator[0], expected, atol=1e-9)
    np.testing.assert_allclose(
        data.spinor_tangent_vertices[0, 0],
        RYTOEV * np.einsum("ij,st->ijst", resolved, PAULI_MATRICES[0]),
        atol=1e-9,
    )


def test_spinor_nc11_soc_explicit_vertices_drive_exchange(tmp_path):
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
    from TB2J.projector_green import (
        magnetic_tangent_vertices,
        spinor_dense_block,
        spinor_tangent_trace,
    )

    gt = write_qe_spinor_dump(tmp_path / "soc.bin", version="1.1", soc=True)
    data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    vertices = data.spinor_tangent_vertices
    explicit = np.stack([spinor_dense_block(vertices[0, axis]) for axis in range(3)])
    commutator = magnetic_tangent_vertices(data.spinor_operator[0])["vertices"]
    # the off-diagonal response contamination must make the explicit vertex
    # genuinely different from the commutator
    assert np.abs(explicit - commutator).max() > 1e-3 * np.abs(explicit).max()
    kwargs = dict(Rpts=[[0, 0, 0], [1, 0, 0], [-1, 0, 0]], nz=8, smearing_eV=0.1)
    exchange = compute_spinor_projector_exchange(data, **kwargs)
    # oracle: the computed exchange must equal the contour integral of the
    # explicit-vertex tangent trace and differ from the commutator one
    from TB2J.interfaces.gpaw_projector import _R_grid
    from TB2J.mycfr import CFR
    from TB2J.projector_green import ProjectorGreen

    green = ProjectorGreen(data)
    rpts = _R_grid(nmax=1)
    explicit_stack = {
        site: np.stack([spinor_dense_block(vertices[site, a]) for a in range(3)])
        for site in range(data.site_nproj.shape[0])
    }
    commutator_stack = {
        site: magnetic_tangent_vertices(data.spinor_operator[site])["vertices"]
        for site in range(data.site_nproj.shape[0])
    }
    contour = CFR(nz=8, T=0.1 / 8.617333262e-5)
    key = ((0, 0, 0), 0, 0)
    matrices = []
    for energy in contour.path:
        trace = spinor_tangent_trace(
            green,
            rpts,
            energy=energy,
            vertices=explicit_stack,
            sites=list(range(data.site_nproj.shape[0])),
        )
        matrices.append(trace["K_ijR"][key])
    integrated = np.array(
        [
            [
                contour.integrate_values(np.asarray([m[a, b] for m in matrices]))
                for b in range(3)
            ]
            for a in range(3)
        ]
    )
    np.testing.assert_allclose(exchange[key]["K_ijR"], integrated, atol=1e-10)
    matrices_comm = []
    for energy in contour.path:
        trace = spinor_tangent_trace(
            green,
            rpts,
            energy=energy,
            vertices=commutator_stack,
            sites=list(range(data.site_nproj.shape[0])),
        )
        matrices_comm.append(trace["K_ijR"][key])
    integrated_comm = np.array(
        [
            [
                contour.integrate_values(np.asarray([m[a, b] for m in matrices_comm]))
                for b in range(3)
            ]
            for a in range(3)
        ]
    )
    assert (
        np.abs(np.imag(integrated) - np.imag(integrated_comm)).max()
        > 1e-6 * np.abs(np.imag(integrated)).max()
    )
    # hermiticity gate: corrupt one response block
    bad = write_qe_spinor_dump(tmp_path / "bad.bin", version="1.1", soc=True)
    bad.records[-1][0][0, 1, 0, 0, 0, 2] += 0.01
    _write_spinor_records(bad.path, bad.records)
    with pytest.raises(ValueError, match="not complex Hermitian"):
        parse_qe_spinor_dump(bad.path)


def test_spinor_nc11_tangent_netcdf_roundtrip_and_required_field(tmp_path):
    gt = write_qe_spinor_dump(tmp_path / "z.bin", version="1.1")
    data = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    nc_path = tmp_path / "roundtrip.nc"
    data.save_netcdf(nc_path)
    loaded = ProjectorGreenData.load_netcdf(nc_path)
    np.testing.assert_allclose(
        loaded.spinor_tangent_vertices, data.spinor_tangent_vertices
    )
    assert loaded.spinor_tangent_definition == "qe_nc11_frozen_source_tangent"
    loaded.validate(exchange_ready=True)
    # a QE NC1.1-labeled dataset without its tangent fails closed
    stripped = read_qe_spinor_dump(gt.path, legacy_no_u=True)
    stripped.spinor_tangent_vertices = None
    with pytest.raises(ValueError, match="NC1.1"):
        stripped.validate(exchange_ready=True)


def test_spinor_nc11_three_leg_exchange_and_tolerance_gate(tmp_path):
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
    from TB2J.io_exchange.io_exchange import SpinIO

    references = {
        axis: write_qe_spinor_dump(
            tmp_path / f"{axis}.bin", axis=axis, version="1.1"
        ).path
        for axis in "xyz"
    }
    kwargs = dict(Rpts=[[0, 0, 0]], nz=8, smearing_eV=0.1)
    output, jdict = gen_exchange_qe(
        references,
        output_path=tmp_path / "out",
        leg_consistency_atol_eV=5e-8,
        **kwargs,
        legacy_no_u=True,
    )
    stored = SpinIO.load_pickle(path=tmp_path / "out")
    assert "leg_consistency_atol_eV=5.000e-08" in stored.description
    assert "TB2J_TAN_LEG_SPREAD" in stored.description
    leg = compute_spinor_projector_exchange(
        read_qe_spinor_dump(references["z"], legacy_no_u=True), **kwargs
    )
    np.testing.assert_allclose(
        jdict[((0, 0, 0), 0, 1)],
        leg[((0, 0, 0), 0, 1)]["J_leg"][0, 0],
        atol=1e-10,
        rtol=1e-8,
    )
    for bad in (0.0, -1e-8, 2e-7, float("nan")):
        with pytest.raises(ValueError, match="leg_consistency_atol_eV"):
            gen_exchange_qe(
                references,
                output_path=tmp_path / "refused",
                leg_consistency_atol_eV=bad,
                **kwargs,
                legacy_no_u=True,
            )
    # single NC1.1 dump still refuses full-tensor output
    with pytest.raises(ValueError, match="transverse leg"):
        gen_exchange_qe(
            references["z"], output_path=tmp_path / "single", legacy_no_u=True
        )
    # legacy NC1.0 three-leg keeps working (SO rejection test already covers
    # SOC NC1.0 refusal)


def test_qe_spinor_cli_accepts_leg_consistency_flag(tmp_path, monkeypatch):
    from test_qe_hubbard import _spinor_no_u_dump

    from TB2J.scripts.qe2J import run_qe2J

    references = {
        axis: _spinor_no_u_dump(tmp_path / f"{axis}.bin", axis=axis) for axis in "xyz"
    }
    output = tmp_path / "cli"
    argv = [
        "qe2J.py",
        "--output_path",
        str(output),
        "--nz",
        "8",
        "--leg_consistency_atol_eV",
        "1e-8",
    ]
    for axis, path in references.items():
        argv.extend([f"--input_{axis}", str(path)])
    monkeypatch.setattr("sys.argv", argv)
    run_qe2J()
    from TB2J.io_exchange.io_exchange import SpinIO

    stored = SpinIO.load_pickle(path=output)
    assert "leg_consistency_atol_eV=1.000e-08" in stored.description


# ---------------------------------------------------------------------------
# Story 002: per-species NC strict validation and Gram-conditioning gates
# ---------------------------------------------------------------------------
def _write_spinor_dump_custom_gram(path, gram, version="1.0"):
    """Three-channel US spinor dump for Gram-spectrum gate fixtures.

    Nonzero z projected field carries the magnetic source; deeq magnetic
    is zero. ``gram`` is a (3, 3) Hermitian positive-definite block.
    """
    from TB2J.interfaces.qe_projector import MAGIC_SPINOR, MAGIC_SPINOR_NC11

    nk, nbnd, nhm, nat, nkb = 2, 2, 3, 1, 3
    rng = np.random.default_rng(23)
    coefficients = rng.normal(size=(nk, nkb, 2, nbnd)) + 1j * rng.normal(
        size=(nk, nkb, 2, nbnd)
    )
    et = np.array([[-0.2, 0.3], [-0.1, 0.4]]).T
    wg = (et < 0).astype(float) * 0.5
    field = np.zeros((nhm, nhm, nat, 3), dtype=complex)
    field[:, :, :, 2] = 0.02 * np.eye(nhm)[:, :, None]
    magic = MAGIC_SPINOR if version == "1.0" else MAGIC_SPINOR_NC11
    records = [
        (
            np.array([magic.ljust(16, b" ")], dtype="S16"),
            np.array([4, nk, nk, nbnd, nat, 1, nhm, 1, nkb, 0, 1], dtype="<i4"),
        ),
        (np.array([3.5, 0, 0.01]), np.array([-99, 0, 1], dtype="<i4")),
        (np.eye(3), np.eye(3), np.array([4.0, 64.0]), np.array([1], dtype="<i4")),
        (np.array([b"Fe    "] + [b"      "] * 9, dtype="S6"),),
        (np.ones(nat, dtype="<i4"), np.zeros((3, nat))),
        tuple(np.array([v], dtype="<i4") for v in (nhm, 1, 0, 0)),
        (np.zeros((nhm, nhm, nat, 4), dtype=complex),),
        (np.zeros((nhm, nhm, nat, 4), dtype=complex),),
        (gram[:, :, None].astype(complex),),
        (np.array([[0, 0.5], [0, 0], [0, 0]]), np.full(nk, 0.5)),
        (et, wg),
        *[(coefficients[ik],) for ik in range(nk)],
        (field,),
        (np.zeros((nhm * (nhm + 1) // 2, nat, 4)),),
    ]
    if version == "1.1":
        records.append((np.zeros((nhm, nhm, nat, 4, 3, 3), dtype=complex),))
    _write_spinor_records(path, records)
    return path


def _gram_with_eigenvalues(eigenvalues, seed=5):
    """Hermitian positive-definite block with the requested spectrum."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(len(eigenvalues), len(eigenvalues))))
    gram = q @ np.diag(eigenvalues) @ q.T
    return (gram + gram.T) / 2


def test_spinor_nc_species_measured_zero_source_oracles(tmp_path):
    """Per-species NC gates: measured-zero required, nonzero slab rejected.

    A pure NC species (tvanp=tpawp=0) is admitted only with an exactly
    measured-zero nonlocal source while its nonzero projected smooth
    field carries the magnetic presence; nonzero NC slabs are malformed.
    """
    # accepted: NC species, zero deeq magnetic, zero response slabs,
    # nonzero smooth z field
    good = write_qe_spinor_dump(tmp_path / "nc.bin", version="1.1")
    good.records[5][1][0] = 0  # tvanp = 0
    good.records[7][0][...] = 0
    good.records[-1][0][...] = 0
    _write_spinor_records(good.path, good.records)
    data = read_qe_spinor_dump(good.path, legacy_no_u=True)
    assert data.metadata["tvanp"] == [False]
    # smooth-only explicit tangent equals the commutator tangent exactly
    from TB2J.projector_green import magnetic_tangent_vertices, spinor_dense_block

    commutator = magnetic_tangent_vertices(data.spinor_operator[0])["vertices"]
    for axis in range(3):
        np.testing.assert_allclose(
            spinor_dense_block(data.spinor_tangent_vertices[0, axis]),
            commutator[axis],
            atol=1e-9,
        )

    # rejected: NC species with a nonzero native nonlocal source
    for record, part, message in (
        (7, 0, "nonzero deeq_nc_magnetic nonlocal"),
        (None, None, "nonzero D_response slab"),
    ):
        bad = write_qe_spinor_dump(tmp_path / f"bad{record}.bin", version="1.1")
        bad.records[5][1][0] = 0
        bad.records[7][0][...] = 0
        bad.records[-1][0][...] = 0
        if record is None:
            bad.records[-1][0][0, 0, 0, 0, 0, 1] = 1e-9
        else:
            bad.records[record][part][0, 0, 0, 0] = 1e-6
        _write_spinor_records(bad.path, bad.records)
        with pytest.raises(ValueError, match=message):
            parse_qe_spinor_dump(bad.path)

    # has_so=F explicit-versus-commutator parity is asserted: a Hermitian
    # perturbation of an off-diagonal response slab passes the diagonal
    # identity gate but must break the scalar-relativistic parity
    parity = write_qe_spinor_dump(tmp_path / "parity.bin", version="1.1")
    # slab (c=z, j=x): nonzero K weight for the x leg; real diagonal
    # (uu) element keeps the block Hermitian and skips the diagonal gate
    parity.records[-1][0][0, 0, 0, 0, 2, 0] += 1e-3
    _write_spinor_records(parity.path, parity.records)
    with pytest.raises(ValueError, match="deviates from"):
        read_qe_spinor_dump(parity.path, legacy_no_u=True)


def test_spinor_nc_requires_measured_response_and_dressed_source(tmp_path):
    """NC1.0 lacks the oracle; an all-discarded smooth field is not magnetic."""
    legacy = write_qe_spinor_dump(tmp_path / "nc10.bin", version="1.0")
    legacy.records[5][1][0] = 0
    legacy.records[7][0][...] = 0
    _write_spinor_records(legacy.path, legacy.records)
    with pytest.raises(ValueError, match="NC1.0 cannot certify"):
        parse_qe_spinor_dump(legacy.path)

    discard = write_qe_spinor_dump(tmp_path / "discard.bin", version="1.1")
    discard.records[5][1][0] = 0
    discard.records[7][0][...] = 0
    discard.records[-1][0][...] = 0
    discard.records[8][0][...] = np.diag([1.0, 1.0e-9])[:, :, None]
    discard.records[13][0][...] = 0
    discard.records[13][0][1, 1, 0, 2] = 1.0e-9
    _write_spinor_records(discard.path, discard.records)
    with pytest.raises(ValueError, match="zero conditioned magnetic source"):
        read_qe_spinor_dump(discard.path, legacy_no_u=True)


def test_spinor_nc_mixed_nonmagnetic_site_accepted(tmp_path):
    """A nonmagnetic NC site with zero source is admitted (ADR-001)."""
    mix = write_qe_spinor_dump(tmp_path / "mix.bin", version="1.1")
    mix.records[5][1][0] = 0  # tvanp = 0
    mix.records[7][0][...] = 0
    mix.records[-1][0][...] = 0
    mix.records[13][0][:, :, 1, :] = 0  # site 1 fully nonmagnetic
    _write_spinor_records(mix.path, mix.records)
    data = read_qe_spinor_dump(mix.path, legacy_no_u=True)
    assert np.abs(data.spinor_operator[1]).max() == 0.0
    np.testing.assert_allclose(data.spinor_tangent_vertices[1], 0.0, atol=0.0)
    assert np.abs(data.spinor_operator[0]).max() > 0.0


def test_spinor_nonmagnetic_site_with_response_or_traceless_field_refused(tmp_path):
    """A zero moment must not silently erase a nonzero native tangent."""
    bad_response = write_qe_spinor_dump(tmp_path / "response.bin", version="1.1")
    bad_response.records[7][0][:, :, 1, :] = 0
    bad_response.records[13][0][:, :, 1, :] = 0
    bad_response.records[-1][0][:, :, 1, :, :, :] = 0
    bad_response.records[-1][0][0, 0, 1, 0, 2, 0] = 1e-3
    _write_spinor_records(bad_response.path, bad_response.records)
    with pytest.raises(
        ValueError, match="zero magnetic splitting but nonzero source response"
    ):
        read_qe_spinor_dump(bad_response.path, legacy_no_u=True)

    traceless = write_qe_spinor_dump(tmp_path / "traceless.bin", version="1.1")
    gram = traceless.records[8][0][:, :, 1]
    traceless.records[13][0][:, :, 1, :] = 0
    traceless.records[13][0][:, :, 1, 2] = gram @ np.diag([0.02, -0.02]) @ gram
    traceless.records[7][0][:, :, 1, :] = 0
    traceless.records[-1][0][:, :, 1, :, :, :] = 0
    _write_spinor_records(traceless.path, traceless.records)
    with pytest.raises(
        ValueError, match="nonzero magnetic operator.*no well-defined moment"
    ):
        read_qe_spinor_dump(traceless.path, legacy_no_u=True)


def test_spinor_default_three_leg_excludes_zero_source_site(tmp_path):
    """Default selection keeps magnetic pairs and omits the true zero ligand."""
    from TB2J.io_exchange.io_exchange import SpinIO

    refs = {}
    for axis in "xyz":
        dump = write_qe_spinor_dump(tmp_path / f"{axis}.bin", axis=axis, version="1.1")
        dump.records[7][0][:, :, 1, :] = 0
        dump.records[13][0][:, :, 1, :] = 0
        dump.records[-1][0][:, :, 1, :, :, :] = 0
        _write_spinor_records(dump.path, dump.records)
        refs[axis] = dump.path
    output, jdict = gen_exchange_qe(
        refs,
        output_path=tmp_path / "default",
        Rpts=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        nz=8,
        smearing_eV=0.1,
        legacy_no_u=True,
    )
    stored = SpinIO.load_pickle(path=output.parent)
    assert list(stored.index_spin) == [0, -1]
    assert ((1, 0, 0), 0, 0) in jdict
    assert all(i == j == 0 for _, i, j in jdict)
    with pytest.raises(ValueError, match="explicitly selected"):
        gen_exchange_qe(
            refs,
            output_path=tmp_path / "invalid_selection",
            Rpts=[[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
            index_magnetic_atoms=[1],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )


def test_spinor_mixed_scalar_nc_soc_us_readable_but_production_refused(tmp_path):
    """A real zero-ligand smoke is required before mixed production exchange."""
    import warnings

    refs = {}
    for axis in "xyz":
        dump = write_qe_spinor_dump(
            tmp_path / f"mixed_{axis}.bin", axis=axis, version="1.1", soc=True
        )
        dump.records[0][1][5] = 2  # two species, one NC and one US
        dump.records[3][0][1] = b"O     "
        dump.records[4][0][1] = 2
        dump.records[5] = tuple(
            np.asarray(values, dtype="<i4")
            for values in ([2, 2], [0, 1], [0, 0], [0, 1])
        )
        dump.records[7][0][:, :, 0, :] = 0
        dump.records[-1][0][:, :, 0, :, :, :] = 0
        _write_spinor_records(dump.path, dump.records)
        refs[axis] = dump.path
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mixed = read_qe_spinor_dump(refs["z"], legacy_no_u=True)
    assert mixed.metadata["tvanp"] == [False, True]
    assert "nc_exchange_status" not in mixed.metadata
    # Parser retains per-site source checks; production must not infer
    # nonmagnetic-ligand validation from a magnetic augmented-site fixture.
    with pytest.raises(ValueError, match="mixed NC.*augmented"):
        gen_exchange_qe(
            refs,
            output_path=tmp_path / "mixed_uncertified",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "mixed_uncertified").exists()


def test_spinor_nc_with_zero_source_us_ligand_still_refuses_production(tmp_path):
    refs = {}
    for axis in "xyz":
        dump = write_qe_spinor_dump(
            tmp_path / f"nc_us_{axis}.bin", axis=axis, version="1.1"
        )
        dump.records[0][1][5] = 2
        dump.records[3][0][1] = b"O     "
        dump.records[4][0][1] = 2
        dump.records[5] = tuple(
            np.asarray(values, dtype="<i4")
            for values in ([2, 2], [0, 1], [0, 0], [0, 0])
        )
        dump.records[7][0][:, :, 0, :] = 0
        dump.records[-1][0][:, :, 0, :, :, :] = 0
        dump.records[7][0][:, :, 1, :] = 0
        dump.records[13][0][:, :, 1, :] = 0
        dump.records[-1][0][:, :, 1, :, :, :] = 0
        _write_spinor_records(dump.path, dump.records)
        refs[axis] = dump.path
    data = read_qe_spinor_dump(refs["z"], legacy_no_u=True)
    assert data.metadata["tvanp"] == [False, True]
    np.testing.assert_array_equal(data.spinor_operator[1], 0)
    np.testing.assert_array_equal(data.spinor_tangent_vertices[1], 0)
    with pytest.raises(ValueError, match="mixed NC.*augmented"):
        gen_exchange_qe(
            refs,
            output_path=tmp_path / "zero_ligand_uncertified",
            Rpts=[[-1, 0, 0], [1, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "zero_ligand_uncertified").exists()


def test_spinor_gram_no_eigengap_refused(tmp_path):
    """Ambiguous rcond truncation fails closed; a clean gap is accepted.

    The gate is a numerical-conditioning diagnostic only: eigenvalues
    (1, 1.5e-6, 9e-7) truncate at rcond=1e-6 with
    smallest_retained/largest_discarded ~ 1.7 < 100 and are refused;
    (1, 2, 1e-9) has a clean gap and reads normally.
    """
    refused = _write_spinor_dump_custom_gram(
        tmp_path / "nogap.bin", _gram_with_eigenvalues([1.0, 1.5e-6, 9.0e-7])
    )
    with pytest.raises(ValueError, match="lack a clean eigengap"):
        read_qe_spinor_dump(refused, legacy_no_u=True)
    clean = _write_spinor_dump_custom_gram(
        tmp_path / "gap.bin", _gram_with_eigenvalues([1.0, 2.0, 1.0e-9])
    )
    data = read_qe_spinor_dump(clean, legacy_no_u=True)
    report = data.metadata["gram_spectrum"][0]
    assert (report["retained"], report["discarded"]) == (2, 1)
    assert report["cutoff"] == pytest.approx(2.0e-6)
    assert np.all(np.isfinite(data.spinor_operator))


def test_spinor_fr_nc_production_exchange_fail_closed(tmp_path):
    """FR NC reads for research (with warning) but production refuses.

    A Gram eigengap certifies nothing: measured PseudoDojo FR NC Fe
    carries O(1) normalized occupied-projector weight in discarded
    modes, so SOC-bearing *NC-species* three-leg output fails closed
    until a window spectral-weight and threshold-sensitivity
    certificate exists. The refusal is scoped to tvanp=tpawp=0 species:
    SO-bearing US/PAW references must still reach the merge.
    """

    def _nc_references(name):
        references = {}
        for axis in "xyz":
            gt = write_qe_spinor_dump(
                tmp_path / f"{name}{axis}.bin", axis=axis, version="1.1", soc=True
            )
            # pure NC species: measured-zero nonlocal source, smooth-only
            gt.records[5][1][0] = 0
            gt.records[7][0][...] = 0
            gt.records[-1][0][...] = 0
            _write_spinor_records(gt.path, gt.records)
            references[axis] = gt.path
        return references

    soc = write_qe_spinor_dump(tmp_path / "soc_nc.bin", version="1.1", soc=True)
    soc.records[5][1][0] = 0
    soc.records[7][0][...] = 0
    soc.records[-1][0][...] = 0
    _write_spinor_records(soc.path, soc.records)
    with pytest.warns(UserWarning, match="uncertified"):
        data = read_qe_spinor_dump(soc.path, legacy_no_u=True)
    assert "uncertified" in data.metadata["nc_exchange_status"]
    with pytest.raises(ValueError, match="norm-conserving.*not.*certified"):
        gen_exchange_qe(
            _nc_references("nc"),
            output_path=tmp_path / "refused",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "refused" / "exchange.out").exists()
    # average_pp can clear has_so when SOC is disabled for an FR NC
    # pseudopotential while leaving split radial betas intact. A clean
    # numerical gap must not make that scalar-flagged truncation publishable.
    off_refs = {}
    for axis in "xyz":
        gt = write_qe_spinor_dump(tmp_path / f"off{axis}.bin", axis=axis, version="1.1")
        gt.records[5][1][0] = 0
        gt.records[7][0][...] = 0
        gt.records[-1][0][...] = 0
        gt.records[8][0][:, :, 0] = np.diag([1.0, 1.0e-9])
        _write_spinor_records(gt.path, gt.records)
        off_refs[axis] = gt.path
    with pytest.raises(ValueError, match="norm-conserving.*not certified"):
        gen_exchange_qe(
            off_refs,
            output_path=tmp_path / "off_refused",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "off_refused" / "exchange.out").exists()
    # SO-bearing US/PAW references are NOT hit by the NC refusal: they get
    # past the family gate into the real merge, which then fails on its own
    # known physics gate (the synthetic SOC legs are not exchange-consistent,
    # max spread ~1.5e-4 eV). Asserting that exact gate proves the routing.
    uspaw = {
        axis: write_qe_spinor_dump(
            tmp_path / f"uspaw{axis}.bin", axis=axis, version="1.1", soc=True
        ).path
        for axis in "xyz"
    }
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        read_qe_spinor_dump(uspaw["z"], legacy_no_u=True)
    with pytest.raises(ValueError, match="leg diagonals disagree"):
        gen_exchange_qe(
            uspaw,
            output_path=tmp_path / "uspaw_out",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert not (tmp_path / "uspaw_out" / "exchange.out").exists()
    # scalar NC1.1 production output still works
    scalar = {
        axis: write_qe_spinor_dump(
            tmp_path / f"scalar{axis}.bin", axis=axis, version="1.1"
        ).path
        for axis in "xyz"
    }
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        output, _ = gen_exchange_qe(
            scalar,
            output_path=tmp_path / "ok",
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            legacy_no_u=True,
        )
    assert (tmp_path / "ok" / "exchange.out").exists()
