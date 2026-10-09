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
    RYTOEV,
    QEProjectorDump,
    parse_qe_dump,
    read_qe_dump,
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


def test_nc_kb_only_raises_blocker_text(tmp_path):
    p = tmp_path / "nckb.bin"
    write_qe_dump(p, tvanp=(0, 0), tpawp=(0, 0))
    with pytest.raises(ValueError) as excinfo:
        parse_qe_dump(p)
    message = str(excinfo.value)
    assert "separable beta spin vertex vanishes (deeq=dvan, Δ≡0)" in message
    assert "NC+KB" in message


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
    data = read_qe_dump(p)
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
    data = read_qe_dump(p)
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
    data = read_qe_dump(p)
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
    data = read_qe_dump(p)
    assert data.efermi_spin is None
    np.testing.assert_allclose(data.efermi, -1.25 * RYTOEV)


def test_per_atom_channel_splitting(tmp_path):
    p = tmp_path / "channels.bin"
    gt = write_qe_dump(p)
    data = read_qe_dump(p)
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


def test_single_spin_channel_raises(tmp_path):
    p = tmp_path / "onechannel.bin"
    write_qe_dump(p, isk=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="spin"):
        read_qe_dump(p)


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
def test_parse_v12_records():
    p = "/tmp/qe_v12_synth.bin"
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


def test_v12_delta_total_is_conjugated_vertex():
    p = "/tmp/qe_v12b_synth.bin"
    gt = write_qe_dump(p, magic=MAGIC_V12, include_dbeta_xc=True)
    data = read_qe_dump(p)
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


def test_v11_delta_total_remains_deeq_difference_but_not_exchange_ready():
    p = "/tmp/qe_v11_synth.bin"
    gt = write_qe_dump(p, magic=MAGIC_V11)
    data = read_qe_dump(p)
    meta = data.operator_component_metadata["delta_total"]
    assert meta["exchange_ready"] == "false"
    assert "FALSIFIED" in meta["definition"]
    from TB2J.interfaces.gpaw_projector import component_local_operators

    try:
        component_local_operators(data, "delta_total", [0], "test")
    except ValueError as exc:
        assert "not exchange-ready" in str(exc)
    else:
        raise AssertionError("v1.1 deeq-only vertex must not be exchange-ready")
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

    data = read_qe_dump(path)
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
