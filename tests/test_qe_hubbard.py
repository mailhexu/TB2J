"""Semantic tests for the strict native QE Hubbard extension reader.

Fixtures use the published Fortran-record contract and deterministic local
arrays; they do not consume any native export or workstation-specific path.
"""

# MathDoc: tb2j/qe-hubbard-native-contract

from __future__ import annotations

import struct

import numpy as np
import pytest

from test_qe_projector import (
    RYTOEV,
    MAGIC_V12,
    _frecord,
    _write_spinor_records,
    write_qe_atomic_dump,
    write_qe_dump,
    write_qe_spinor_dump,
)


HUBBARD_MAGIC = b"TB2JHUBBARD1.0"
NATIVE_REVISION = "301742e16546b927cfe35fb2fcf4d9338f3e294a"


def _fixed(text: str, width: int) -> np.ndarray:
    return np.asarray([text.encode("ascii")], dtype=f"S{width}")


def _append_record(path, *parts) -> None:
    with path.open("ab") as stream:
        _frecord(stream, *parts)


def _append_common_no_u_extension(
    path,
    *,
    family: str,
    nat: int,
    ntyp: int,
    nspin: int,
    nks: int,
    nbnd: int,
    nkb: int,
    source_mode: int,
    collinear: int = 1,
    species_names=("Fe", "O"),
    species_family=2,
    finish=True,
    metric="S-applied native source retained",
) -> None:
    """Append the exact common no-U profile and EOF validation record."""
    _append_record(
        path,
        np.asarray([1], dtype="<i4"),
        _fixed(HUBBARD_MAGIC.decode("ascii"), 16),
        np.asarray(
            [
                1,
                0,
                16909060,
                0,
                0,
                1,
                0,
                0,
                0,
                collinear,
                source_mode,
                0,
                0,
                nat,
                ntyp,
                nspin,
                nks,
                nbnd,
                nkb,
                0,
                1,
                2,
                32,
                64,
                128,
            ],
            dtype="<i4",
        ),
    )
    _append_record(
        path,
        np.asarray([2], dtype="<i4"),
        _fixed("QE 8.0dev synthetic", 80),
        _fixed(NATIVE_REVISION, 64),
        _fixed(family, 16),
        _fixed("inactive", 30),
        _fixed("not applicable for inactive Hubbard", 64),
        _fixed(metric, 80),
        _fixed("/synthetic/input.in", 256),
        _fixed("", 32),
        _fixed("not available in fixture", 48),
        _fixed("/synthetic/pseudo", 256),
        _fixed("/synthetic/pseudo", 256),
    )
    for species_id, species in enumerate(species_names, start=1):
        _append_record(
            path,
            np.asarray([3, species_id], dtype="<i4"),
            _fixed(species, 80),
            _fixed(f"{species}.upf", 256),
            _fixed(f"/synthetic/pseudo/{species}.upf", 512),
            np.asarray([species_family, 0], dtype="<i4"),
            _fixed(("a" if species_id == 1 else "b") * 32, 32),
            _fixed("synthetic fixture identity", 56),
        )
    _append_record(path, np.asarray([13, 0], dtype="<i4"))
    if finish:
        _append_no_u_validation(path)


def _append_no_u_validation(path) -> None:
    _append_record(
        path,
        np.asarray([14, 0, 1, 0, 0, 0, 0], dtype="<i4"),
        np.zeros(6, dtype="<f8"),
        np.asarray([0], dtype="<i4"),
        np.zeros(4, dtype="<f8"),
    )


def _kb_no_u_dump(path):
    write_qe_dump(path, magic=MAGIC_V12, include_dbeta_xc=True)
    _replace_magic(path, b"TB2JQEDUMPV1.3")
    _append_common_no_u_extension(
        path,
        family="kb",
        nat=3,
        ntyp=2,
        nspin=2,
        nks=4,
        nbnd=3,
        nkb=8,
        source_mode=1,
    )
    return path


def test_kb13_explicit_no_u_is_retained_as_absent_without_legacy_attestation(tmp_path):
    from TB2J.interfaces.qe_projector import parse_qe_dump

    path = _kb_no_u_dump(tmp_path / "kb-no-u.bin")
    raw = parse_qe_dump(path)

    assert raw.hubbard.state == "absent"
    assert raw.hubbard.complete is True
    assert raw.hubbard.envelope["active"] == 0
    assert raw.hubbard.parameters == []
    assert raw.hubbard.channels == []
    assert raw.hubbard.states == []
    assert raw.hubbard.actions == []
    assert raw.hubbard.provenance["native_revision"] == NATIVE_REVISION


def _atomic_no_u_dump(path):
    write_qe_atomic_dump(path, magic=b"TB2JQEATWFC1.1")
    _append_common_no_u_extension(
        path,
        family="atomic",
        nat=2,
        ntyp=2,
        nspin=2,
        nks=4,
        nbnd=3,
        nkb=0,
        source_mode=2,
        species_family=1,
    )
    return path


def _spinor_frame_record(site, gram, field, magnetic):
    inverse = np.linalg.inv(gram[:, :, site])
    smooth = np.stack(
        [inverse @ field[:, :, site, axis] @ inverse for axis in range(3)],
        axis=-1,
    )
    uu, ud, du, dd = np.moveaxis(magnetic[:, :, site, :], -1, 0)
    magnetic_pauli = np.stack(
        ((ud + du) / 2, (du - ud) / (2j), (uu - dd) / 2),
        axis=-1,
    )
    source = 2 * RYTOEV * (smooth + magnetic_pauli)
    trace_vector = np.trace(source, axis1=0, axis2=1).real
    trace_norm = float(np.linalg.norm(trace_vector))
    direction = trace_vector / trace_norm
    rank = gram.shape[0]
    unit_residual = abs(float(np.linalg.norm(direction)) - 1.0)
    return (
        np.asarray([21, site + 1, 1, rank, 1], dtype="<i4"),
        np.asarray([np.abs(source).max()], dtype="<f8"),
        trace_vector.astype("<f8"),
        np.asarray([trace_norm], dtype="<f8"),
        direction.astype("<f8"),
        np.asarray([1], dtype="<i4"),
        np.zeros(3, dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.asarray([0.0], dtype="<f8"),
        np.asarray([rank, 0], dtype="<i4"),
        np.asarray([0.0, 1.0e-6, 100.0, unit_residual], dtype="<f8"),
    )


def _spinor_zero_frame_record(site, rank=2):
    return (
        np.asarray([21, site + 1, 1, rank, 0], dtype="<i4"),
        np.zeros(1, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(1, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.asarray([1], dtype="<i4"),
        np.zeros(3, dtype="<f8"),
        np.zeros(1, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(1, dtype="<f8"),
        np.asarray([rank, 0], dtype="<i4"),
        np.asarray([0.0, 1.0e-6, 100.0, 0.0], dtype="<f8"),
    )


def _spinor_no_u_dump(path, *, axis="z", zero_source=False):
    native = write_qe_spinor_dump(path, axis=axis, version="1.1")
    records = list(native.records)
    if zero_source:
        records[7][0][...] = 0
        records[13][0][...] = 0
        records[-1][0][...] = 0
    records[0] = (
        np.asarray([b"TB2JQEDUMPNC1.2  "], dtype="S16"),
        records[0][1],
    )
    _write_spinor_records(path, records)
    _append_common_no_u_extension(
        path,
        family="kb",
        nat=2,
        ntyp=1,
        nspin=4,
        nks=2,
        nbnd=8,
        nkb=4,
        source_mode=1,
        collinear=0,
        species_names=("Fe",),
        finish=False,
        metric="full-spinor ortho-atomic source metric",
    )
    _append_record(path, np.asarray([15, 2, 0, 0, 1, 4, 2, 0], dtype="<i4"))
    for site in range(2):
        frame = (
            _spinor_zero_frame_record(site)
            if zero_source
            else _spinor_frame_record(site, native.gram, native.field, native.magnetic)
        )
        _append_record(path, *frame)
    _append_no_u_validation(path)
    return path


@pytest.mark.parametrize(
    "fixture,parser",
    [
        (_atomic_no_u_dump, "parse_qe_atomic_dump"),
        (_spinor_no_u_dump, "parse_qe_spinor_dump"),
    ],
)
def test_new_no_u_profiles_retain_explicit_absence_without_attestation(
    tmp_path, fixture, parser
):
    from TB2J.interfaces import qe_projector

    raw = getattr(qe_projector, parser)(fixture(tmp_path / f"{parser}.bin"))

    assert raw.hubbard.state == "absent"
    assert raw.hubbard.complete is True
    assert raw.hubbard.envelope["active"] == 0
    assert raw.hubbard.parameters == []
    assert raw.hubbard.channels == []
    assert raw.hubbard.states == []
    assert raw.hubbard.actions == []
    if parser == "parse_qe_spinor_dump":
        assert [frame.origin for frame in raw.hubbard.frames] == [1, 1]
    else:
        assert raw.hubbard.frames == []


def _record_payloads(path):
    data = path.read_bytes()
    payloads = []
    offset = 0
    while offset < len(data):
        (size,) = struct.unpack_from("<i", data, offset)
        start = offset + 4
        stop = start + size
        payload = data[start:stop]
        (end_size,) = struct.unpack_from("<i", data, stop)
        assert end_size == size
        payloads.append(payload)
        offset = stop + 4
    return payloads


def _write_payloads(path, payloads):
    with path.open("wb") as stream:
        for payload in payloads:
            stream.write(struct.pack("<i", len(payload)))
            stream.write(payload)
            stream.write(struct.pack("<i", len(payload)))


def _replace_magic(path, magic):
    payloads = _record_payloads(path)
    payloads[0] = magic.ljust(16, b" ") + payloads[0][16:]
    _write_payloads(path, payloads)


def _deterministic_collinear_trial(ik, global_g, seed=617021):
    modulus = 2147483647
    g = np.asarray(global_g, dtype=np.int64)
    components = []
    for stream in (1, 2):
        state = (seed + ik * 104729 + 13007 + g * 48271 + stream * 69621) % modulus
        state = (state * 48271 + 1) % modulus
        components.append(state / modulus - 0.5)
    trial = components[0] + 1j * components[1]
    return trial / np.linalg.norm(trial)


def _active_collinear_dump(path, *, family="kb"):
    if family == "kb":
        gt = write_qe_dump(path, magic=MAGIC_V12, include_dbeta_xc=True)
        _replace_magic(path, b"TB2JQEDUMPV1.3")
        nat, nkb, source_mode, atom, species_family = gt.nat, gt.nkb, 1, 3, 2
    else:
        gt = write_qe_atomic_dump(path, magic=b"TB2JQEATWFC1.1")
        nat, nkb, source_mode, atom, species_family = gt.nat, 0, 2, 2, 1
    ntyp, nspin, nks, nbnd = 2, 2, gt.nks, gt.nbnd
    nchannel = ncol = 1
    u_ev = 2.0
    u_ry = u_ev / RYTOEV
    _append_record(
        path,
        np.asarray([1], dtype="<i4"),
        _fixed(HUBBARD_MAGIC.decode("ascii"), 16),
        np.asarray(
            [
                1,
                0,
                16909060,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                source_mode,
                nchannel,
                ncol,
                nat,
                ntyp,
                nspin,
                nks,
                nbnd,
                nkb,
                1,
                1,
                2,
                32,
                64,
                128,
            ],
            dtype="<i4",
        ),
    )
    _append_record(
        path,
        np.asarray([2], dtype="<i4"),
        _fixed("QE 8.0dev synthetic", 80),
        _fixed(NATIVE_REVISION, 64),
        _fixed(family, 16),
        _fixed("atomic", 30),
        _fixed("native radial construction", 64),
        _fixed("S-applied native projectors", 80),
        _fixed("/synthetic/input.in", 256),
        _fixed("", 32),
        _fixed("not available in fixture", 48),
        _fixed("/synthetic/pseudo", 256),
        _fixed("/synthetic/pseudo", 256),
    )
    for species_id, species in enumerate(("Fe", "O"), start=1):
        _append_record(
            path,
            np.asarray([3, species_id], dtype="<i4"),
            _fixed(species, 80),
            _fixed(f"{species}.upf", 256),
            _fixed(f"/synthetic/pseudo/{species}.upf", 512),
            np.asarray([species_family, 0], dtype="<i4"),
            _fixed(("a" if species_id == 1 else "b") * 32, 32),
            _fixed("synthetic fixture identity", 56),
        )
    _append_record(
        path,
        np.asarray([4, ntyp, 2, 1], dtype="<i4"),
        np.asarray([RYTOEV, 0.0], dtype="<f8"),
        np.asarray([1, 1, 0], dtype="<i4"),
    )
    for species_id in range(1, ntyp + 1):
        parameters_ev = np.zeros(9, dtype="<f8")
        if species_id == 2:
            parameters_ev[0] = u_ev
        parameters_ry = parameters_ev / RYTOEV
        quantum_flags = (
            [3, 2, 0, 0, 0, 0, 0, 0, 0]
            if species_id == 1
            else [2, 0, 0, 0, 0, 0, 1, 0, 0]
        )
        _append_record(
            path,
            np.asarray([4, species_id], dtype="<i4"),
            parameters_ev,
            parameters_ry,
            np.zeros(8, dtype="<f8"),
            np.asarray(quantum_flags, dtype="<i4"),
        )
    _append_record(path, np.asarray([5, nchannel, ncol], dtype="<i4"))
    _append_record(
        path,
        np.asarray([6, 1, atom, 2, 1, 2, 0, 1, 0, 0, 1, 0], dtype="<i4"),
    )
    _append_record(path, np.asarray([7, 1, 1, 1, atom, 2, 1, 0, 1], dtype="<i4"))
    _append_record(
        path,
        np.asarray([8, 1, atom, 2, 1, nspin], dtype="<i4"),
        np.zeros((1, 1, nspin), dtype="<f8"),
        np.full((1, 1, nspin), u_ry / 2, dtype="<f8"),
        np.asarray([[u_ry / 2]], dtype="<f8"),
        np.zeros((1, 1), dtype="<f8"),
    )

    npw = 4
    global_g = np.arange(101, 101 + npw, dtype="<i4")
    wfcu = np.zeros((npw, ncol), dtype="<c16")
    wfcu[0, 0] = 1.0
    potential = np.asarray([[u_ry / 2]], dtype="<c16")
    for ik in range(nks):
        psi = np.zeros((npw, 3), dtype="<c16")
        psi[:, 0] = _deterministic_collinear_trial(ik + 1, global_g)
        psi[0:2, 1] = 1.0 / np.sqrt(2.0)
        psi[1, 2] = 1.0
        trial_projection = wfcu.conj().T @ psi
        actions = wfcu @ (potential @ trial_projection)
        action_norms = np.linalg.norm(actions, axis=0)
        metrics = np.vstack(
            (
                np.ones(3),
                np.zeros(3),
                np.linalg.norm(trial_projection, axis=0),
                action_norms,
                action_norms,
                np.zeros(3),
                np.zeros(3),
            )
        ).astype("<f8")
        _append_record(
            path,
            np.asarray([10, ik + 1, npw, ncol, nbnd], dtype="<i4"),
            np.asarray(gt.xk[ik], dtype="<f8"),
            np.asarray([gt.wk[ik]], dtype="<f8"),
            np.asarray([gt.isk[ik]], dtype="<i4"),
            np.arange(101, 101 + npw, dtype="<i4"),
            np.asarray(gt.wg[ik], dtype="<f8"),
            np.asarray(gt.et[ik], dtype="<f8"),
            np.zeros((ncol, nbnd), dtype="<c16"),
        )
        _append_record(
            path,
            np.asarray([11, ik + 1, npw, 3, ncol, 1, 617021], dtype="<i4"),
            metrics,
            trial_projection,
            wfcu,
            psi,
            actions,
            actions,
        )
    _append_record(
        path,
        np.asarray([12, 1, atom, 2, 1, 1, nspin], dtype="<i4"),
        np.asarray([0.0, 0.0], dtype="<f8"),
        np.zeros((1, 1, nspin), dtype="<f8"),
        np.zeros((1, 1, nspin), dtype="<f8"),
    )
    method = 1 if nkb else 2
    source_scale = (
        max(
            float(np.abs(gt.deeq[:count, :count, atom, :]).max())
            for atom, count in enumerate(gt.nh[gt.ityp - 1])
            if count
        )
        if family == "kb"
        else 0.0
    )
    _append_record(
        path,
        np.asarray([14, 1, 1, 1, 3, 1, 617021], dtype="<i4"),
        np.zeros(6, dtype="<f8"),
        np.asarray([method], dtype="<i4"),
        np.asarray([0.0, source_scale, 0.0, 0.0], dtype="<f8"),
    )
    return path


def _assert_error_code(excinfo, code):
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    "family,parser,reader",
    [
        ("kb", "parse_qe_dump", "read_qe_dump"),
        ("atomic", "parse_qe_atomic_dump", "read_qe_atomic_dump"),
    ],
)
def test_collinear_active_stream_retains_inventory_and_defers_normalization(
    tmp_path, family, parser, reader
):
    from TB2J.interfaces import qe_projector

    path = _active_collinear_dump(tmp_path / f"{family}-active.bin", family=family)
    raw = getattr(qe_projector, parser)(path)
    hubbard = raw.hubbard

    assert hubbard.state == "active"
    assert hubbard.complete is True
    assert hubbard.envelope["active"] == 1
    assert hubbard.envelope["nkb"] == (8 if family == "kb" else 0)
    assert hubbard.provenance["native_revision"] == NATIVE_REVISION
    assert len(hubbard.parameters) == 2
    active_parameter = next(item for item in hubbard.parameters if item.species == 2)
    np.testing.assert_allclose(active_parameter.ev[0], 2.0)
    np.testing.assert_allclose(active_parameter.ry[0], 2.0 / RYTOEV)
    assert len(hubbard.channels) == 1
    channel = hubbard.channels[0]
    assert (channel.atom, channel.species, channel.kind, channel.l) == (
        3 if family == "kb" else 2,
        2,
        1,
        0,
    )
    assert hubbard.columns.shape == (1, 8)
    assert len(hubbard.states) == 1
    assert hubbard.states[0].occupation.shape == (1, 1, 2)
    assert hubbard.states[0].potential.shape == (1, 1, 2)
    assert not np.any(hubbard.states[0].occupation)
    np.testing.assert_allclose(
        hubbard.states[0].potential[0, 0], np.full(2, 1.0 / RYTOEV)
    )
    np.testing.assert_allclose(hubbard.actions[0].native, hubbard.actions[0].exported)
    assert np.linalg.norm(hubbard.actions[0].exported) > 0
    assert len(hubbard.kpoints) == 4
    assert len(hubbard.actions) == 4
    assert hubbard.validation
    assert hubbard.validation["source_method"] == (1 if family == "kb" else 2)

    with pytest.raises(ValueError) as excinfo:
        getattr(qe_projector, reader)(path)
    _assert_error_code(excinfo, "normalization_pending")
    with pytest.raises(ValueError) as excinfo:
        getattr(qe_projector, reader)(path, legacy_no_u=True)
    _assert_error_code(excinfo, "normalization_pending")


@pytest.mark.parametrize(
    "fixture,reader",
    [
        (_kb_no_u_dump, "read_qe_dump"),
        (_atomic_no_u_dump, "read_qe_atomic_dump"),
        (_spinor_no_u_dump, "read_qe_spinor_dump"),
    ],
)
def test_explicit_new_no_u_normalizes_without_legacy_assertion(
    tmp_path, fixture, reader
):
    from TB2J.interfaces import qe_projector

    path = fixture(tmp_path / f"{reader}-no-u.bin")
    for legacy_no_u in (False, True):
        data = getattr(qe_projector, reader)(path, legacy_no_u=legacy_no_u)
        assert data.metadata["hubbard_state"] == "absent"
        assert data.metadata["legacy_no_u"] is False
        assert data.metadata["hubbard_provenance"]["origin"] == "native_extension"
        if reader == "read_qe_atomic_dump":
            assert data.metadata["dump_version"] == "1.1"


def _legacy_dump(path, family):
    if family == "kb":
        write_qe_dump(path, magic=MAGIC_V12, include_dbeta_xc=True)
    elif family == "atomic":
        write_qe_atomic_dump(path)
    else:
        write_qe_spinor_dump(path, version="1.1")
    return path


@pytest.mark.parametrize(
    "family,parser,reader",
    [
        ("kb", "parse_qe_dump", "read_qe_dump"),
        ("atomic", "parse_qe_atomic_dump", "read_qe_atomic_dump"),
        ("spinor", "parse_qe_spinor_dump", "read_qe_spinor_dump"),
    ],
)
def test_legacy_stream_remains_unknown_until_user_attests_no_u(
    tmp_path, family, parser, reader
):
    from TB2J.interfaces import qe_projector

    path = _legacy_dump(tmp_path / f"{family}-legacy.bin", family)
    raw = getattr(qe_projector, parser)(path)
    assert raw.hubbard.state == "unknown"
    assert raw.hubbard.complete is False

    with pytest.raises(ValueError) as excinfo:
        getattr(qe_projector, reader)(path)
    _assert_error_code(excinfo, "legacy_attestation")

    data = getattr(qe_projector, reader)(path, legacy_no_u=True)
    assert data.metadata["hubbard_state"] == "user_attested_no_u"
    assert data.metadata["legacy_no_u"] is True
    assert data.metadata["hubbard_provenance"] == {
        "origin": "user_attestation",
        "native_state": "unknown",
    }


def _deterministic_spinor_trial(ik, trial_id, global_g, seed=617021):
    modulus = 2147483647
    vector = np.empty(2 * len(global_g), dtype="<c16")
    for spin in (1, 2):
        for ig, g_index in enumerate(global_g):
            row = (spin - 1) * len(global_g) + ig
            values = []
            for stream in (1, 2):
                state = (
                    seed
                    + ik * 104729
                    + trial_id * 13007
                    + int(g_index) * 48271
                    + (spin - 1) * 15485863
                    + stream * 69621
                ) % modulus
                state = (state * 48271 + 1) % modulus
                values.append(state / modulus - 0.5)
            vector[row] = values[0] + 1j * values[1]
    return vector / np.linalg.norm(vector)


def _spinor_rotation_record(atom, channel, dim, direction):
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(direction[0]) >= 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangents = np.zeros((3, 2), dtype="<f8")
    tangents[:, 0] = reference - np.dot(reference, direction) * direction
    tangents[:, 0] /= np.linalg.norm(tangents[:, 0])
    tangents[:, 1] = np.cross(direction, tangents[:, 0])
    tangent_axes = np.cross(direction[None, :], tangents.T).T
    axes = np.asarray(
        [
            [1.0, 0.0, 1.0 / np.sqrt(14.0)],
            [0.0, 1.0, 2.0 / np.sqrt(14.0)],
            [0.0, 0.0, 3.0 / np.sqrt(14.0)],
        ],
        dtype="<f8",
    )
    return (
        np.asarray([22, atom, 1, channel, dim, 2, 4, 3, 2, 3, 1], dtype="<i4"),
        np.asarray(direction, dtype="<f8"),
        axes,
        np.asarray([np.pi / 2, np.pi / 2, 2 * np.pi / 5], dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        np.zeros(3, dtype="<f8"),
        tangents,
        tangent_axes,
        np.asarray([1.0e-3, 5.0e-4, 2.5e-4], dtype="<f8"),
        np.zeros((dim, dim, 4, 2), dtype="<c16"),
        np.zeros((dim, dim, 4, 2, 3), dtype="<c16"),
        np.zeros((2, 3), dtype="<f8"),
        np.zeros((2, 3), dtype="<f8"),
        np.zeros((2, 2), dtype="<f8"),
    )


def _spinor_active_dump(path):
    native = write_qe_spinor_dump(path, axis="z", version="1.1")
    records = list(native.records)
    records[0] = (
        np.asarray([b"TB2JQEDUMPNC1.2".ljust(16, b" ")], dtype="S16"),
        records[0][1],
    )
    _write_spinor_records(path, records)

    nat, ntyp, nspin = 2, 1, 4
    nks, nbnd, nkb = native.xk.shape[1], native.et.shape[0], 4
    dimension, nchannel, norbital = 5, 2, 10
    ncol, native_ncolumn = 2 * norbital, 2 * norbital
    u_ev, u_ry = 2.0, 2.0 / RYTOEV
    _append_record(
        path,
        np.asarray([1], dtype="<i4"),
        _fixed(HUBBARD_MAGIC.decode("ascii"), 16),
        np.asarray(
            [
                1,
                0,
                16909060,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                nchannel,
                ncol,
                nat,
                ntyp,
                nspin,
                nks,
                nbnd,
                nkb,
                1,
                1,
                2,
                32,
                64,
                128,
            ],
            dtype="<i4",
        ),
    )
    _append_record(
        path,
        np.asarray([2], dtype="<i4"),
        _fixed("QE 8.0dev synthetic", 80),
        _fixed(NATIVE_REVISION, 64),
        _fixed("kb", 16),
        _fixed("atomic", 30),
        _fixed("native radial averaging copy_U_wfc", 64),
        _fixed("native radial averaging; spin-major column slots", 80),
        _fixed("/synthetic/input.in", 256),
        _fixed("", 32),
        _fixed("not available in fixture", 48),
        _fixed("/synthetic/pseudo", 256),
        _fixed("/synthetic/pseudo", 256),
    )
    _append_record(
        path,
        np.asarray([3, 1], dtype="<i4"),
        _fixed("Fe", 80),
        _fixed("Fe.upf", 256),
        _fixed("/synthetic/pseudo/Fe.upf", 512),
        np.asarray([2, 0], dtype="<i4"),
        _fixed("c" * 32, 32),
        _fixed("synthetic fixture identity", 56),
    )
    _append_record(
        path,
        np.asarray([4, ntyp, 2, 1], dtype="<i4"),
        np.asarray([RYTOEV, 0.0], dtype="<f8"),
        np.asarray([1, 1, 0], dtype="<i4"),
    )
    _append_record(
        path,
        np.asarray([4, 1], dtype="<i4"),
        np.asarray([u_ev, 0, 0, 0, 0, 0, 0, 0, 0], dtype="<f8"),
        np.asarray([u_ry, 0, 0, 0, 0, 0, 0, 0, 0], dtype="<f8"),
        np.zeros(8, dtype="<f8"),
        np.asarray([3, 2, 0, 0, 0, 0, 1, 0, 0], dtype="<i4"),
    )
    _append_record(path, np.asarray([5, nchannel, ncol], dtype="<i4"))
    for channel, atom in enumerate((1, 2), start=1):
        native_offset = (channel - 1) * 2 * dimension
        matrix_offset = 0
        _append_record(
            path,
            np.asarray(
                [
                    6,
                    channel,
                    atom,
                    1,
                    1,
                    3,
                    2,
                    dimension,
                    native_offset,
                    matrix_offset,
                    1,
                    0,
                ],
                dtype="<i4",
            ),
        )
    for channel, atom in enumerate((1, 2), start=1):
        native_offset = (channel - 1) * 2 * dimension
        for spin_slot in (1, 2):
            for orbital in range(1, dimension + 1):
                native_column = native_offset + (spin_slot - 1) * dimension + orbital
                export_column = native_column
                _append_record(
                    path,
                    np.asarray(
                        [
                            7,
                            native_column,
                            export_column,
                            channel,
                            atom,
                            1,
                            1,
                            2,
                            orbital,
                        ],
                        dtype="<i4",
                    ),
                )
    _append_record(
        path,
        np.asarray([15, 2, norbital, ncol, 1, 4, 2, native_ncolumn], dtype="<i4"),
    )
    for channel, atom in enumerate((1, 2), start=1):
        native_offset = (channel - 1) * 2 * dimension
        for spin_slot in (1, 2):
            for orbital in range(1, dimension + 1):
                native_column = native_offset + (spin_slot - 1) * dimension + orbital
                _append_record(
                    path,
                    np.asarray(
                        [16, native_column, channel, orbital, spin_slot], dtype="<i4"
                    ),
                )
    identity = np.eye(dimension, dtype="<c16")
    zero_occ = np.zeros((dimension, dimension, 4), dtype="<c16")
    potential = np.zeros_like(zero_occ)
    potential[:, :, 0] = u_ry / 2 * identity
    potential[:, :, 3] = u_ry / 2 * identity
    scalar = u_ry / 2 * identity
    magnetic_u = np.zeros((dimension, dimension, 3), dtype="<c16")
    for channel, atom in enumerate((1, 2), start=1):
        _append_record(
            path,
            np.asarray([17, channel, atom, 1, dimension, 2, 4], dtype="<i4"),
            zero_occ,
            potential,
            scalar,
            magnetic_u,
        )

    npw = 12
    wfcu = np.zeros((2 * npw, ncol), dtype="<c16")
    wfcu[:ncol] = np.eye(ncol, dtype="<c16")
    dense_potential = (u_ry / 2) * np.eye(ncol, dtype="<c16")
    zero_trial_action = 0.0
    for ik in range(nks):
        global_g = np.arange(1, npw + 1, dtype="<i4")
        _append_record(
            path,
            np.asarray([18, ik + 1, npw, 2, ncol, nbnd], dtype="<i4"),
            np.asarray(native.xk[:, ik], dtype="<f8"),
            np.asarray([native.wk[ik]], dtype="<f8"),
            global_g,
            np.asarray(native.wg[:, ik], dtype="<f8"),
            np.asarray(native.et[:, ik], dtype="<f8"),
            np.zeros((ncol, nbnd), dtype="<c16"),
        )
        psi = np.column_stack(
            [
                _deterministic_spinor_trial(ik + 1, trial, global_g)
                for trial in (1, 2, 3)
            ]
        )
        basis = wfcu
        null_trial = psi[:, 2] - basis @ (basis.conj().T @ psi[:, 2])
        psi[:, 2] = null_trial / np.linalg.norm(null_trial)
        projection = wfcu.conj().T @ psi
        actions = wfcu @ (dense_potential @ projection)
        action_norms = np.linalg.norm(actions, axis=0)
        zero_trial_action = max(zero_trial_action, float(action_norms[2]))
        metrics = np.vstack(
            (
                np.ones(3),
                np.zeros(3),
                np.linalg.norm(projection, axis=0),
                action_norms,
                action_norms,
                np.zeros(3),
                np.zeros(3),
            )
        ).astype("<f8")
        _append_record(
            path,
            np.asarray([19, ik + 1, npw, 2, 3, ncol, 1, 617021], dtype="<i4"),
            metrics,
            projection,
            wfcu,
            psi,
            actions,
            actions,
        )
    for channel, atom in enumerate((1, 2), start=1):
        _append_record(
            path,
            np.asarray([20, channel, atom, 1, dimension, 2, 4], dtype="<i4"),
            np.asarray([0.0, 0.0], dtype="<f8"),
            zero_occ,
            zero_occ,
        )
    for channel, atom in enumerate((1, 2), start=1):
        _append_record(
            path,
            np.asarray([23, channel, atom, 1, dimension], dtype="<i4"),
            np.asarray([0.0, 0.0], dtype="<f8"),
            potential,
            potential,
        )
    frames = [
        _spinor_frame_record(site, native.gram, native.field, native.magnetic)
        for site in range(nat)
    ]
    for channel, atom in enumerate((1, 2), start=1):
        _append_record(
            path,
            *_spinor_rotation_record(atom, channel, dimension, frames[atom - 1][4]),
        )
    for frame in frames:
        _append_record(path, *frame)
    _append_record(
        path,
        np.asarray([24], dtype="<i4"),
        np.zeros(10, dtype="<f8"),
        np.ones(5, dtype="<i4"),
        np.zeros(1, dtype="<f8"),
    )
    source_scale = float(np.abs(native.total).max())
    _append_record(
        path,
        np.asarray([14, 1, 1, 1, 3, 1, 617021], dtype="<i4"),
        np.asarray([0.0, zero_trial_action, 0.0, 0.0, 0.0, 0.0], dtype="<f8"),
        np.asarray([1], dtype="<i4"),
        np.asarray([0.0, source_scale, 0.0, 0.0], dtype="<f8"),
    )
    return path


def test_nc12_active_stream_retains_spinor_maps_frames_and_tangents(tmp_path):
    from TB2J.interfaces.qe_projector import parse_qe_spinor_dump, read_qe_spinor_dump

    path = _spinor_active_dump(tmp_path / "nc12-active.bin")
    raw = parse_qe_spinor_dump(path)
    hubbard = raw.hubbard

    assert hubbard.state == "active"
    assert hubbard.complete is True
    assert hubbard.envelope["collinear"] == 0
    assert hubbard.envelope["active"] == 1
    assert hubbard.provenance["native_revision"] == NATIVE_REVISION
    assert len(hubbard.parameters) == 1
    assert hubbard.parameters[0].quantum_flags[6] == 1
    assert len(hubbard.channels) == 2
    assert hubbard.columns.shape == (20, 8)
    assert hubbard.spin_columns.shape == (20, 4)
    assert len(hubbard.states) == 2
    assert [channel.native_offset for channel in hubbard.channels] == [0, 10]
    assert [channel.matrix_offset for channel in hubbard.channels] == [0, 0]
    assert not np.any(hubbard.states[0].occupation)
    np.testing.assert_allclose(
        hubbard.states[0].potential[:, :, 0],
        np.eye(5, dtype=complex) / RYTOEV,
    )
    np.testing.assert_allclose(hubbard.actions[0].native, hubbard.actions[0].exported)
    np.testing.assert_allclose(hubbard.rotations[0].steps, [1e-3, 5e-4, 2.5e-4])
    assert hubbard.rotations[0].tangent_directions.shape == (3, 2)
    assert hubbard.states[0].occupation.shape == (5, 5, 4)
    assert hubbard.states[0].potential.shape == (5, 5, 4)
    assert len(hubbard.kpoints) == len(hubbard.actions) == 2
    assert hubbard.actions[0].wfcu.shape == (24, 20)
    assert hubbard.actions[0].psi.shape == (24, 3)
    assert hubbard.actions[0].native.shape == (24, 3)
    assert hubbard.actions[0].exported.shape == (24, 3)
    assert len(hubbard.frames) == len(hubbard.rotations) == 2
    assert hubbard.frames[0].origin == 1
    assert hubbard.frames[0].direction.shape == (3,)
    assert hubbard.restoration
    assert hubbard.validation

    with pytest.raises(ValueError) as excinfo:
        read_qe_spinor_dump(path)
    _assert_error_code(excinfo, "normalization_pending")
    with pytest.raises(ValueError) as excinfo:
        read_qe_spinor_dump(path, legacy_no_u=True)
    _assert_error_code(excinfo, "normalization_pending")


def _extension_record_index(payloads, record_id, occurrence=0):
    in_extension = False
    for index, payload in enumerate(payloads):
        if payload[4:20].rstrip(b" \x00") == HUBBARD_MAGIC:
            in_extension = True
        if not in_extension or len(payload) < 4:
            continue
        if struct.unpack_from("<i", payload)[0] == record_id:
            if occurrence == 0:
                return index
            occurrence -= 1
    raise AssertionError(f"missing Hubbard record {record_id}")


def _mutate_record(path, record_id, mutate, occurrence=0):
    payloads = _record_payloads(path)
    index = _extension_record_index(payloads, record_id, occurrence)
    payloads[index] = mutate(bytearray(payloads[index]))
    _write_payloads(path, payloads)


def _set_i4(payload, offset, value):
    struct.pack_into("<i", payload, offset, value)
    return payload


def _set_f8(payload, offset, value):
    struct.pack_into("<d", payload, offset, value)
    return payload


def test_kb_source_scale_cannot_expand_carrier_tolerance(tmp_path):
    from TB2J.interfaces.qe_projector import parse_qe_dump

    path = _active_collinear_dump(tmp_path / "inflated-carrier-scale.bin")

    def inflate(payload):
        _set_f8(payload, 80, 1.0)
        return _set_f8(payload, 88, 1.0e12)

    _mutate_record(path, 14, inflate)
    with pytest.raises(ValueError) as excinfo:
        parse_qe_dump(path)
    _assert_error_code(excinfo, "source")


@pytest.mark.parametrize(
    "record_id,occurrence,offset,value,code",
    [
        (5, 0, 8, 2, "dimension"),
        (6, 0, 32, -1, "mapping"),
        (7, 0, 16, 2, "mapping"),
        (10, 0, 20, np.nan, "nonfinite"),
        (14, 0, 76, 2, "source"),
        (14, 0, 80, 1.0, "source"),
        (11, 0, 76, 1.0e-3, "validation"),
        (14, 0, 28, 1.0e-3, "validation"),
    ],
)
def test_collinear_mutations_fail_semantic_gates_with_stable_codes(
    tmp_path, record_id, occurrence, offset, value, code
):
    from TB2J.interfaces.qe_projector import parse_qe_dump

    path = _active_collinear_dump(tmp_path / f"mutated-{record_id}-{offset}.bin")
    mutate = _set_f8 if isinstance(value, float) else _set_i4
    _mutate_record(
        path, record_id, lambda payload: mutate(payload, offset, value), occurrence
    )

    with pytest.raises(ValueError) as excinfo:
        parse_qe_dump(path)
    _assert_error_code(excinfo, code)


def test_active_unsupported_parameters_and_source_mode_cannot_be_attested_away(
    tmp_path,
):
    from TB2J.interfaces.qe_projector import read_qe_dump

    parameter_path = _active_collinear_dump(tmp_path / "unsupported-parameter.bin")

    def set_nonzero_j(payload):
        _set_f8(payload, 56, 0.25)
        _set_f8(payload, 128, 0.25 / RYTOEV)
        return payload

    _mutate_record(parameter_path, 4, set_nonzero_j, occurrence=2)
    for legacy_no_u in (False, True):
        with pytest.raises(ValueError) as excinfo:
            read_qe_dump(parameter_path, legacy_no_u=legacy_no_u)
        _assert_error_code(excinfo, "unsupported")

    excluded_path = _active_collinear_dump(tmp_path / "unsupported-excluded-norm.bin")
    _mutate_record(
        excluded_path,
        4,
        lambda payload: _set_f8(payload, 152, 1.0e-3),
        occurrence=2,
    )
    with pytest.raises(ValueError) as excinfo:
        read_qe_dump(excluded_path, legacy_no_u=True)
    _assert_error_code(excinfo, "unsupported")

    source_mode_path = _active_collinear_dump(tmp_path / "unsupported-source-mode.bin")
    _mutate_record(source_mode_path, 1, lambda payload: _set_i4(payload, 60, 2))
    with pytest.raises(ValueError) as excinfo:
        read_qe_dump(source_mode_path, legacy_no_u=True)
    _assert_error_code(excinfo, "unsupported")


def test_nc12_active_nonhermitian_potential_is_refused(tmp_path):
    from TB2J.interfaces.qe_projector import parse_qe_spinor_dump

    path = _spinor_active_dump(tmp_path / "nonhermitian-potential.bin")

    def make_nonhermitian(payload):
        dimension = 5
        occupation_bytes = dimension * dimension * 4 * 16
        potential = np.frombuffer(
            payload,
            dtype="<c16",
            count=dimension * dimension * 4,
            offset=7 * 4 + occupation_bytes,
        )
        potential[dimension] += 1.0j
        return payload

    _mutate_record(path, 17, make_nonhermitian)
    with pytest.raises(ValueError) as excinfo:
        parse_qe_spinor_dump(path)
    _assert_error_code(excinfo, "hermiticity")


@pytest.mark.parametrize(
    "failure", ["truncated", "extra-eof", "payload-length", "marker-length"]
)
def test_absent_extension_rejects_incomplete_or_nonexact_record_streams(
    tmp_path, failure
):
    from TB2J.interfaces.qe_projector import parse_qe_dump

    path = _kb_no_u_dump(tmp_path / f"{failure}.bin")
    payloads = _record_payloads(path)
    if failure == "truncated":
        _write_payloads(path, payloads[:-1])
    elif failure == "extra-eof":
        _append_record(path, np.asarray([99], dtype="<i4"))
    elif failure == "payload-length":
        index = _extension_record_index(payloads, 13)
        payloads[index] += struct.pack("<i", 0)
        _write_payloads(path, payloads)
    else:
        index = _extension_record_index(payloads, 14)
        data = bytearray(path.read_bytes())
        offset = sum(8 + len(payload) for payload in payloads[:index])
        trailer = offset + 4 + len(payloads[index])
        struct.pack_into("<i", data, trailer, len(payloads[index]) + 1)
        path.write_bytes(data)

    with pytest.raises(ValueError) as excinfo:
        parse_qe_dump(path)
    _assert_error_code(excinfo, "layout")


def test_nc12_zero_source_is_parseable_but_normalization_needs_a_real_frame(tmp_path):
    from TB2J.interfaces.qe_projector import parse_qe_spinor_dump, read_qe_spinor_dump

    path = _spinor_no_u_dump(tmp_path / "nc12-absent-zero-source.bin", zero_source=True)
    raw = parse_qe_spinor_dump(path)
    assert raw.hubbard.state == "absent"
    assert raw.hubbard.complete is True
    assert [frame.origin for frame in raw.hubbard.frames] == [0, 0]
    with pytest.raises(ValueError) as excinfo:
        read_qe_spinor_dump(path)
    _assert_error_code(excinfo, "source")
    with pytest.raises(ValueError) as excinfo:
        read_qe_spinor_dump(path, legacy_no_u=True)
    _assert_error_code(excinfo, "source")


@pytest.mark.parametrize("profile", ["kb", "atomic", "spinor"])
def test_fresh_occupation_rejects_zeroed_native_overlaps(tmp_path, profile):
    from TB2J.interfaces import qe_projector as qe

    path = tmp_path / f"{profile}-fresh-occupation.bin"
    if profile == "spinor":
        _spinor_active_dump(path)
        parse, coefficient_record, check_record = qe.parse_qe_spinor_dump, 18, 20
    else:
        _active_collinear_dump(path, family=profile)
        parse = qe.parse_qe_dump if profile == "kb" else qe.parse_qe_atomic_dump
        coefficient_record, check_record = 10, 12
    raw = parse(path)
    kp = raw.hubbard.kpoints[0]
    band = int(np.argmax(kp.wg))
    weight = float(kp.wg[band])
    assert weight > 0
    ncol, nbnd = kp.coefficients.shape

    def occupied_coefficient(payload):
        offset = len(payload) - ncol * nbnd * 16 + band * ncol * 16
        struct.pack_into("<dd", payload, offset, 1.0, 0.0)
        return payload

    def fresh_occupation(payload):
        check = raw.hubbard.occupation_checks[0]
        fresh = np.zeros_like(check.native)
        fresh[0, 0, 0 if profile == "spinor" else kp.spin - 1] = weight
        block = fresh.tobytes(order="F")
        payload[44:] = block + block
        return payload

    _mutate_record(path, coefficient_record, occupied_coefficient)
    _mutate_record(path, check_record, fresh_occupation)
    # Fresh NSCF evidence is intentionally different from the frozen SCF state.
    fresh = parse(path).hubbard
    assert fresh.occupation_checks[0].native[0, 0, 0] == weight
    assert not np.any(fresh.states[0].occupation)

    def zero_coefficients(payload):
        payload[-ncol * nbnd * 16 :] = bytes(ncol * nbnd * 16)
        return payload

    _mutate_record(path, coefficient_record, zero_coefficients)
    with pytest.raises(ValueError) as excinfo:
        parse(path)
    _assert_error_code(excinfo, "validation")


@pytest.mark.parametrize("profile", ["kb", "atomic", "spinor"])
def test_extension_species_label_must_match_predecessor(tmp_path, profile):
    from TB2J.interfaces import qe_projector as qe

    path = tmp_path / f"{profile}-wrong-species.bin"
    if profile == "spinor":
        _spinor_active_dump(path)
        parse = qe.parse_qe_spinor_dump
    else:
        _active_collinear_dump(path, family=profile)
        parse = qe.parse_qe_dump if profile == "kb" else qe.parse_qe_atomic_dump

    def wrong_label(payload):
        payload[8:88] = b"Co".ljust(80, b" ")
        return payload

    _mutate_record(path, 3, wrong_label)
    with pytest.raises(ValueError) as excinfo:
        parse(path)
    _assert_error_code(excinfo, "provenance")
