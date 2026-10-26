"""Public QE Hubbard source-selection, x/y/z consistency, and CLI behavior."""

from __future__ import annotations

import hashlib
import io
import struct
import sys

import numpy as np
import pytest
from test_qe_hubbard import (
    HUBBARD_MAGIC,
    RYTOEV,
    _active_collinear_dump,
    _legacy_dump,
    _record_payloads,
    _set_f8,
    _set_i4,
    _spinor_active_dump,
    _spinor_frame_record,
    _spinor_rotation_record,
    _write_payloads,
)
from test_qe_projector import _frecord, _write_spinor_records, write_qe_spinor_dump


def _run_qe_cli(monkeypatch, *arguments):
    from TB2J.scripts.qe2J import run_qe2J

    monkeypatch.setattr(sys, "argv", ["tb2j-qe", *(str(arg) for arg in arguments)])
    try:
        run_qe2J()
    except SystemExit as exc:
        return exc.code
    return 0


def _load_spinio(path):
    from TB2J.io_exchange.io_exchange import SpinIO

    return SpinIO.load_pickle(path=path)


def _array_bytes(array):
    return np.asarray(array, order="F").tobytes(order="F")


def _encode_record(parts):
    stream = io.BytesIO()
    _frecord(stream, *parts)
    raw = stream.getvalue()
    (size,) = struct.unpack_from("<i", raw)
    return raw[4 : 4 + size]


def _regular_kb_mesh(path):
    """Represent the fixture's two fractional k points in its skew cell."""
    payloads = _record_payloads(path)
    nks = int(np.frombuffer(payloads[0], dtype="<i4", count=1, offset=20)[0])
    at = np.frombuffer(payloads[2], dtype="<f8", count=9).reshape((3, 3), order="F")
    xk = np.frombuffer(payloads[10], dtype="<f8", count=3 * nks).reshape(
        (3, nks), order="F"
    )
    fractional = np.zeros((nks, 3))
    fractional[:, 0] = xk[0]
    cartesian = fractional @ np.linalg.inv(at)
    predecessor = bytearray(payloads[10])
    predecessor[: 24 * nks] = _array_bytes(cartesian.T)
    payloads[10] = bytes(predecessor)
    extension = next(
        (
            i
            for i, p in enumerate(payloads)
            if p[4:20].rstrip(b" \x00") == HUBBARD_MAGIC
        ),
        len(payloads),
    )
    for index in range(extension + 1, len(payloads)):
        if struct.unpack_from("<i", payloads[index])[0] == 10:
            payload = bytearray(payloads[index])
            ik = struct.unpack_from("<i", payload, 4)[0] - 1
            payload[20:44] = np.asarray(cartesian[ik], dtype="<f8").tobytes()
            payloads[index] = bytes(payload)
    _write_payloads(path, payloads)


def _find_extension(payloads):
    return next(
        index
        for index, payload in enumerate(payloads)
        if payload[4:20].rstrip(b" \x00") == HUBBARD_MAGIC
    )


# MathDoc: tb2j/qe-hubbard-native-contract
def _charge_rotation_evidence(payload, occupation, u_ry):
    """Measure even tiny floating-point derivatives of a charge-only state."""
    header = np.frombuffer(payload, dtype="<i4", count=11)
    atom, channel, dim = (int(header[i]) for i in (1, 3, 4))
    direction = np.frombuffer(payload, dtype="<f8", count=3, offset=44)
    parts = list(_spinor_rotation_record(atom, channel, dim, direction))
    pauli = np.asarray([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    density = occupation * np.eye(2 * dim)

    def magnetic_blocks(axis, angle):
        spin = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * np.einsum(
            "a,aij->ij", axis, pauli
        )
        q = np.kron(spin, np.eye(dim))
        rotated = q.conj() @ density @ q.T
        potential = u_ry * (0.5 * np.eye(2 * dim) - rotated.T)
        uu, ud = potential[:dim, :dim], potential[:dim, dim:]
        du, dd = potential[dim:, :dim], potential[dim:, dim:]
        x, y, z = (ud + du) / 2, (du - ud) / (2j), (uu - dd) / 2
        return np.stack((z, x - 1j * y, x + 1j * y, -z), axis=-1)

    for plane in range(2):
        axis = parts[11][:, plane]
        for step, h in enumerate(parts[12]):
            finite = (magnetic_blocks(axis, h) - magnetic_blocks(axis, -h)) / (2 * h)
            parts[14][:, :, :, plane, step] = finite
            absolute = float(np.linalg.norm(finite))
            parts[15][plane, step] = absolute
            parts[16][plane, step] = 1.0 if absolute > np.finfo(float).tiny else 0.0
        for step in range(2):
            first, second = (
                parts[14][:, :, :, plane, step],
                parts[14][:, :, :, plane, step + 1],
            )
            scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
            parts[17][plane, step] = (
                float(np.linalg.norm(first - second)) / scale
                if scale > np.finfo(float).tiny
                else 0.0
            )
    return _encode_record(parts)


def _patch_active_spinor_state(path, *, occupation, u_ev=2.0):
    """Keep frozen spinor charge/potential/action records mutually coherent."""
    payloads = _record_payloads(path)
    extension = _find_extension(payloads)
    extension_records = {}
    for index, payload in enumerate(payloads[extension + 1 :], start=extension + 1):
        record_id = struct.unpack_from("<i", payload)[0]
        extension_records.setdefault(record_id, []).append(index)

    u_ry = float(u_ev) / RYTOEV
    identity = np.eye(5, dtype="<c16")
    frozen = np.zeros((5, 5, 4), dtype="<c16")
    frozen[:, :, 0] = float(occupation) * identity
    frozen[:, :, 3] = float(occupation) * identity
    potential = np.zeros_like(frozen)
    potential[:, :, 0] = u_ry * (0.5 - float(occupation)) * identity
    potential[:, :, 3] = u_ry * (0.5 - float(occupation)) * identity
    scalar = u_ry * (0.5 - float(occupation)) * identity
    magnetic = np.zeros((5, 5, 3), dtype="<c16")

    if u_ev != 2.0:
        payload = bytearray(payloads[extension_records[4][1]])
        _set_f8(payload, 8, u_ev)
        _set_f8(payload, 80, u_ry)
        payloads[extension_records[4][1]] = bytes(payload)

    occupation_block = _array_bytes(frozen)
    potential_block = _array_bytes(potential)
    scalar_block = _array_bytes(scalar)
    magnetic_block = _array_bytes(magnetic)
    for index in extension_records[17]:
        payload = bytearray(payloads[index])
        offset = 7 * 4
        payload[offset : offset + len(occupation_block)] = occupation_block
        offset += len(occupation_block)
        payload[offset : offset + len(potential_block)] = potential_block
        offset += len(potential_block)
        payload[offset : offset + len(scalar_block)] = scalar_block
        offset += len(scalar_block)
        payload[offset : offset + len(magnetic_block)] = magnetic_block
        payloads[index] = bytes(payload)

    for index in extension_records[23]:
        payload = bytearray(payloads[index])
        offset = 5 * 4 + 2 * 8
        payload[offset : offset + len(potential_block)] = potential_block
        payload[offset + len(potential_block) : offset + 2 * len(potential_block)] = (
            potential_block
        )
        payloads[index] = bytes(payload)

    for index in extension_records[19]:
        payload = bytearray(payloads[index])
        header = np.frombuffer(payload, dtype="<i4", count=8)
        npw, npol, ntrial, ncol = (int(header[i]) for i in (2, 3, 4, 5))
        metrics = (
            np.frombuffer(payload, dtype="<f8", count=7 * ntrial, offset=32)
            .copy()
            .reshape((7, ntrial), order="F")
        )
        offset = 32 + metrics.nbytes
        projection = np.frombuffer(
            payload, dtype="<c16", count=ncol * ntrial, offset=offset
        ).reshape((ncol, ntrial), order="F")
        offset += projection.nbytes
        wfcu = np.frombuffer(
            payload, dtype="<c16", count=npw * npol * ncol, offset=offset
        ).reshape((npw * npol, ncol), order="F")
        action = u_ry * (0.5 - float(occupation)) * (wfcu @ projection)
        metrics[3:5] = np.linalg.norm(action, axis=0)
        payload[32 : 32 + metrics.nbytes] = _array_bytes(metrics)
        action_bytes = _array_bytes(action)
        for offset in (
            len(payload) - 2 * len(action_bytes),
            len(payload) - len(action_bytes),
        ):
            payload[offset : offset + len(action_bytes)] = action_bytes
        payloads[index] = bytes(payload)

    for index in extension_records[22]:
        payloads[index] = _charge_rotation_evidence(payloads[index], occupation, u_ry)

    _write_payloads(path, payloads)


def _patch_shell_map(path):
    payloads = _record_payloads(path)
    extension = _find_extension(payloads)
    records = {}
    for index, payload in enumerate(payloads[extension + 1 :], start=extension + 1):
        records.setdefault(struct.unpack_from("<i", payload)[0], []).append(index)
    # Same d shell and column extent, with a different principal-shell identity.
    parameter = bytearray(payloads[records[4][1]])
    _set_i4(parameter, 216, 4)
    payloads[records[4][1]] = bytes(parameter)
    for index in records[6]:
        channel = bytearray(payloads[index])
        _set_i4(channel, 20, 4)
        payloads[index] = bytes(channel)
    _write_payloads(path, payloads)


def _patch_upf_identity(path, *, upf_path, upf_bytes):
    digest = hashlib.md5(upf_bytes).hexdigest()
    payloads = _record_payloads(path)
    extension = _find_extension(payloads)
    index = next(
        index
        for index, payload in enumerate(payloads[extension + 1 :], start=extension + 1)
        if struct.unpack_from("<i", payload)[0] == 3
    )
    payload = bytearray(payloads[index])
    encoded_path = str(upf_path).encode("ascii")
    assert len(encoded_path) <= 512
    payload[344:856] = encoded_path.ljust(512, b" ")
    payload[864:896] = digest.encode("ascii")
    payload[896:952] = b"fixture_md5".ljust(56, b" ")
    payloads[index] = bytes(payload)
    _write_payloads(path, payloads)
    return digest


def _active_spinor_reference(path, *, axis, occupation, u_ev=2.0, upf_path, upf_bytes):
    """Reuse the strict active NC1.2 fixture, rotating only its x/y/z leg data."""
    _spinor_active_dump(path)
    active_payloads = _record_payloads(path)
    extension = _find_extension(active_payloads)

    base_path = path.with_name(f"{path.stem}-base.bin")
    native = write_qe_spinor_dump(base_path, axis=axis, version="1.1")
    records = list(native.records)
    records[0] = (
        np.asarray([b"TB2JQEDUMPNC1.2".ljust(16, b" ")], dtype="S16"),
        records[0][1],
    )
    _write_spinor_records(base_path, records)
    base_payloads = _record_payloads(base_path)
    assert len(base_payloads) == extension
    merged = base_payloads + active_payloads[extension:]

    frames = [
        _spinor_frame_record(site, native.gram, native.field, native.magnetic)
        for site in range(2)
    ]
    frame_index = rotation_index = 0
    for index in range(extension, len(merged)):
        record_id = struct.unpack_from("<i", merged[index])[0]
        if record_id == 21:
            merged[index] = _encode_record(frames[frame_index])
            frame_index += 1
        elif record_id == 22:
            header = np.frombuffer(merged[index], dtype="<i4", count=11)
            atom, channel, dimension = int(header[1]), int(header[3]), int(header[4])
            merged[index] = _encode_record(
                _spinor_rotation_record(atom, channel, dimension, frames[atom - 1][4])
            )
            rotation_index += 1
    assert frame_index == rotation_index == 2
    _write_payloads(path, merged)
    _patch_active_spinor_state(path, occupation=occupation, u_ev=u_ev)
    _patch_upf_identity(path, upf_path=upf_path, upf_bytes=upf_bytes)
    return path


def _spinor_references(tmp_path, *, occupations=(0.0, 0.1, 0.2), upf_contents=None):
    if upf_contents is None:
        upf_contents = {axis: b"<UPF>same bytes</UPF>\n" for axis in "xyz"}
    references = {}
    upf_paths = {}
    for axis, occupation in zip("xyz", occupations, strict=True):
        upf_path = tmp_path / f"upf-{axis}" / "Fe.upf"
        upf_path.parent.mkdir(parents=True)
        upf_path.write_bytes(upf_contents[axis])
        upf_paths[axis] = upf_path
        references[axis] = _active_spinor_reference(
            tmp_path / f"{axis}-active.bin",
            axis=axis,
            occupation=occupation,
            upf_path=upf_path,
            upf_bytes=upf_contents[axis],
        )
    return references, upf_paths


def test_cli_selects_hubbard_operator_components_numerically_and_legacy_flag_is_non_destructive(
    tmp_path, monkeypatch
):
    from test_qe_hubbard_joint import _native_coefficients_for_targets

    source = _active_collinear_dump(
        tmp_path / "active-collinear.bin",
        state_occupation=(0.2, 0.6),
        native_coefficients=_native_coefficients_for_targets,
    )
    _regular_kb_mesh(source)
    outputs = {}
    common = (
        "--input",
        source,
        "--index_magnetic_atoms",
        "3",
        "--nz",
        "8",
        "--smearing",
        "0.1",
    )
    for component in ("delta_total", "delta_xc", "delta_u"):
        output = tmp_path / component
        assert (
            _run_qe_cli(
                monkeypatch,
                *common,
                "--output_path",
                output,
                "--operator_component",
                component,
            )
            == 0
        )
        spinio = _load_spinio(output)
        values = spinio.exchange_Jdict
        assert values
        assert all(np.isfinite(value) for value in values.values())
        assert max(abs(value) for value in values.values()) > 1e-12
        expected_role = {
            "delta_total": "complete_xc_u",
            "delta_xc": "diagnostic_xc_only",
            "delta_u": "diagnostic_u_only",
        }[component]
        assert component in spinio.description
        assert expected_role in spinio.description
        outputs[component] = values

    keys = set(outputs["delta_total"])
    assert set(outputs["delta_xc"]) == set(outputs["delta_u"]) == keys
    for first, second in (("delta_total", "delta_xc"), ("delta_xc", "delta_u")):
        assert (
            max(abs(outputs[first][key] - outputs[second][key]) for key in keys) > 1e-10
        )

    legacy_output = tmp_path / "active-with-legacy-attestation"
    assert (
        _run_qe_cli(
            monkeypatch,
            *common,
            "--output_path",
            legacy_output,
            "--operator_component",
            "delta_total",
            "--legacy_no_u",
        )
        == 0
    )
    legacy_spinio = _load_spinio(legacy_output)
    assert set(legacy_spinio.exchange_Jdict) == keys
    np.testing.assert_allclose(
        [legacy_spinio.exchange_Jdict[key] for key in sorted(keys)],
        [outputs["delta_total"][key] for key in sorted(keys)],
        rtol=1e-10,
        atol=1e-12,
    )


def test_cli_legacy_attestation_refusal_and_success_use_standard_spinio(
    tmp_path, monkeypatch
):
    source = _legacy_dump(tmp_path / "historical-kb.bin", "kb")
    _regular_kb_mesh(source)
    refused = tmp_path / "unattested"
    with pytest.raises(ValueError, match="legacy"):
        _run_qe_cli(monkeypatch, "--input", source, "--output_path", refused)
    assert not refused.exists()

    output = tmp_path / "attested"
    assert (
        _run_qe_cli(
            monkeypatch,
            "--input",
            source,
            "--output_path",
            output,
            "--legacy_no_u",
            "--nz",
            "8",
            "--smearing",
            "0.1",
        )
        == 0
    )
    spinio = _load_spinio(output)
    assert spinio.exchange_Jdict
    assert all(np.isfinite(value) for value in spinio.exchange_Jdict.values())
    assert max(abs(value) for value in spinio.exchange_Jdict.values()) > 1e-12
    assert "user_attested_no_u" in spinio.description


def test_cli_refuses_unavailable_hubbard_diagnostic_before_standard_output(
    tmp_path, monkeypatch
):
    from test_qe_hubbard import _kb_no_u_dump

    source = _kb_no_u_dump(tmp_path / "native-no-u.bin")
    output = tmp_path / "unavailable-delta-u"
    with pytest.raises(ValueError):
        _run_qe_cli(
            monkeypatch,
            "--input",
            source,
            "--output_path",
            output,
            "--operator_component",
            "delta_u",
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "mismatch",
    ["u_parameter", "shell_map", "upf_digest", "frame_map", "component_convention"],
)
def test_active_spinor_reference_identity_mismatches_refuse_before_output(
    tmp_path, monkeypatch, mismatch
):
    if mismatch == "upf_digest":
        contents = {
            "x": b"<UPF>physical species A</UPF>\n",
            "y": b"<UPF>physical species A</UPF>\n",
            "z": b"<UPF>physical species B</UPF>\n",
        }
    else:
        contents = {axis: b"<UPF>same bytes</UPF>\n" for axis in "xyz"}
    references, _ = _spinor_references(tmp_path, upf_contents=contents)
    if mismatch == "u_parameter":
        _patch_active_spinor_state(references["y"], occupation=0.1, u_ev=2.25)
    elif mismatch == "shell_map":
        _patch_shell_map(references["y"])

    from TB2J.interfaces import qe_projector

    if mismatch in ("frame_map", "component_convention"):
        original_reader = qe_projector.read_qe_spinor_dump

        def injected_reader(path, *, legacy_no_u=False, operator_component=None):
            component = operator_component
            if mismatch == "component_convention" and str(path) == str(references["y"]):
                component = "delta_xc"
            data = original_reader(
                path,
                legacy_no_u=legacy_no_u,
                operator_component=component,
            )
            if mismatch == "frame_map" and str(path) == str(references["y"]):
                data.spinor_operator[0] = 0.0
                data.spinor_tangent_vertices[0] = 0.0
                for operator in data.operator_components.values():
                    operator[0] = 0.0
                frame = data.metadata["qe_source_frame"]
                frame["site_map"] = [1]
                frame["directions"] = [frame["directions"][1]]
                frame["direction_provenance"] = [frame["direction_provenance"][1]]
                data.validate(exchange_ready=True)
            if mismatch == "component_convention" and str(path) == str(references["y"]):
                assert data.metadata["operator_component"] == "delta_xc"
            return data

        monkeypatch.setattr(qe_projector, "read_qe_spinor_dump", injected_reader)

    for reference in references.values():
        qe_projector.read_qe_spinor_dump(reference).validate(exchange_ready=True)

    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        qe_projector.gen_exchange_qe(
            references,
            output_path=output,
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
            operator_component="delta_total",
        )
    assert not output.exists()


def test_unresolved_upf_digests_remain_inspectable_but_refuse_production(tmp_path):
    from TB2J.interfaces.qe_projector import (
        gen_exchange_qe,
        parse_qe_spinor_dump,
        read_qe_spinor_dump,
    )

    references, _ = _spinor_references(
        tmp_path,
        upf_contents={
            "x": b"<UPF>physical species A</UPF>\n",
            "y": b"<UPF>physical species A</UPF>\n",
            "z": b"<UPF>physical species B</UPF>\n",
        },
    )
    for reference in references.values():
        payloads = _record_payloads(reference)
        extension = _find_extension(payloads)
        for index in range(extension + 1, len(payloads)):
            if struct.unpack_from("<i", payloads[index])[0] == 3:
                payload = bytearray(payloads[index])
                payload[864:896] = b"NOT SET".ljust(32, b" ")
                payloads[index] = bytes(payload)
        _write_payloads(reference, payloads)
        assert parse_qe_spinor_dump(reference).hubbard.species[0].md5 == "NOT SET"
        read_qe_spinor_dump(reference).validate(exchange_ready=True)

    output = tmp_path / "unresolved-upf"
    with pytest.raises(ValueError):
        gen_exchange_qe(
            references,
            output_path=output,
            Rpts=[[0, 0, 0]],
            nz=8,
            smearing_eV=0.1,
        )
    assert not output.exists()


def test_active_spinor_references_allow_axis_frames_frozen_occupations_and_identical_upfs(
    tmp_path, monkeypatch
):
    from TB2J.interfaces.qe_projector import parse_qe_spinor_dump, read_qe_spinor_dump

    references, upf_paths = _spinor_references(tmp_path)
    upf_bytes = [upf_paths[axis].read_bytes() for axis in "xyz"]
    assert len(set(upf_paths.values())) == 3
    assert upf_bytes[0] == upf_bytes[1] == upf_bytes[2]

    raw = [parse_qe_spinor_dump(references[axis]) for axis in "xyz"]
    assert not np.array_equal(
        raw[0].hubbard.states[0].occupation, raw[1].hubbard.states[0].occupation
    )
    assert not np.array_equal(
        raw[1].hubbard.states[0].occupation, raw[2].hubbard.states[0].occupation
    )
    directions = [
        np.asarray(
            read_qe_spinor_dump(references[axis]).metadata["qe_source_frame"][
                "directions"
            ]
        )
        for axis in "xyz"
    ]
    assert not np.allclose(directions[0], directions[1])
    assert not np.allclose(directions[1], directions[2])

    outputs = {}
    for component in ("delta_total", "delta_u"):
        output = tmp_path / f"spinor-{component}"
        arguments = (
            "--input_x",
            references["x"],
            "--input_y",
            references["y"],
            "--input_z",
            references["z"],
            "--output_path",
            output,
            "--nz",
            "8",
            "--smearing",
            "0.1",
            "--operator_component",
            component,
        )
        assert _run_qe_cli(monkeypatch, *arguments) == 0
        outputs[component] = _load_spinio(output)

    total, diagnostic = outputs["delta_total"], outputs["delta_u"]
    assert total.exchange_Jdict
    assert max(abs(value) for value in total.exchange_Jdict.values()) > 1e-12
    assert diagnostic.exchange_Jdict
    assert all(abs(value) <= 1e-12 for value in diagnostic.exchange_Jdict.values())
    assert total.description != diagnostic.description
    assert "delta_u" in diagnostic.description
    assert "diagnostic_u_only" in diagnostic.description
    np.testing.assert_allclose(diagnostic.spinat, total.spinat, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fallback_sites", [(0, 1), (1,)])
@pytest.mark.parametrize("component", [None, "delta_xc", "delta_u"])
def test_spinor_output_moments_follow_each_frame_origin(
    tmp_path, monkeypatch, fallback_sites, component
):
    from TB2J.interfaces import qe_projector

    references, _ = _spinor_references(tmp_path)
    original_reader = qe_projector.read_qe_spinor_dump
    original_z = original_reader(references["z"])
    expected_moments = -np.asarray(original_z.metadata["qe_source_frame"]["directions"])

    def normalized_occupation_frame(
        path, *, legacy_no_u=False, operator_component=None
    ):
        # A normalized source-frame fixture, not native-export certification:
        # move each fallback site's XC source and coefficients to its U columns.
        # The same physical source then has the opposite declared frame axis.
        data = original_reader(path, legacy_no_u=legacy_no_u)
        xc = data.operator_components["delta_xc"]
        source_u = data.operator_components["delta_u"]
        xc_tangent = data.spinor_tangent_vertices.copy()
        u_tangent = np.zeros_like(xc_tangent)
        frame = data.metadata["qe_source_frame"]
        for site in fallback_sites:
            indices = data.site_projector_indices[site, : data.site_nproj[site]]
            old = np.flatnonzero(
                np.isin(indices, data.metadata["qe_channel_groups"]["xc"])
            )
            native = np.flatnonzero(
                np.isin(indices, data.metadata["qe_channel_groups"]["u"])
            )[: len(old)]
            source_u[site][np.ix_(native, native)] = xc[site][np.ix_(old, old)]
            u_tangent[site][:, native[:, None], native] = -xc_tangent[site][
                :, old[:, None], old
            ]
            xc[site] = 0.0
            xc_tangent[site] = 0.0
            original_columns, native_columns = indices[old], indices[native]
            data.coefficients[..., native_columns] = data.coefficients[
                ..., original_columns
            ]
            data.coefficients[..., original_columns] = 0.0
            frame_index = frame["site_map"].index(site)
            frame["directions"][frame_index] = (
                -np.asarray(frame["directions"][frame_index])
            ).tolist()
            frame["direction_provenance"][frame_index] = (
                "mapped_native_hubbard_occupation"
            )
        data.operator_components["delta_total"] = xc + source_u
        selected = operator_component or "delta_total"
        data.spinor_operator = data.operator_components[selected]
        data.spinor_tangent_vertices = {
            "delta_total": xc_tangent + u_tangent,
            "delta_xc": xc_tangent,
            "delta_u": u_tangent,
        }[selected]
        data.metadata["operator_component"] = selected
        data.metadata["source_completeness"] = {
            "delta_total": "complete_xc_u",
            "delta_xc": "diagnostic_xc_only",
            "delta_u": "diagnostic_u_only",
        }[selected]
        data.validate(exchange_ready=True)
        return data

    monkeypatch.setattr(
        qe_projector, "read_qe_spinor_dump", normalized_occupation_frame
    )
    output = tmp_path / "occupation-frame-output"
    qe_projector.gen_exchange_qe(
        references,
        output_path=output,
        Rpts=[[0, 0, 0]],
        nz=8,
        smearing_eV=0.1,
        operator_component=component,
    )
    spinio = _load_spinio(output)
    np.testing.assert_allclose(spinio.spinat, expected_moments, rtol=0, atol=1e-12)


def test_spinor_zero_diagnostic_source_and_tangent_keep_the_explicit_runtime_frame(
    tmp_path,
):
    from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
    from TB2J.interfaces.qe_projector import read_qe_spinor_dump

    source = _spinor_active_dump(tmp_path / "active-zero-u-magnetism.bin")
    total = read_qe_spinor_dump(source)
    selected = read_qe_spinor_dump(source, operator_component="delta_u")

    assert np.linalg.norm(total.spinor_operator) > 1e-12
    assert np.linalg.norm(total.spinor_tangent_vertices) > 1e-12
    assert not np.any(selected.spinor_operator)
    assert not np.any(selected.spinor_tangent_vertices)
    frame = selected.qe_source_directions()
    assert set(frame) == {0, 1}
    np.testing.assert_allclose(frame[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(frame[1], [0.0, 0.0, 1.0])

    kwargs = dict(Rpts=[[0, 0, 0]], nz=8, smearing_eV=0.1)
    total_leg = compute_spinor_projector_exchange(total, **kwargs)
    selected_leg = compute_spinor_projector_exchange(selected, **kwargs)
    assert any(np.linalg.norm(entry["J_leg"]) > 1e-12 for entry in total_leg.values())
    assert selected_leg
    for entry in selected_leg.values():
        np.testing.assert_allclose(entry["J_leg"], 0.0, atol=1e-12, rtol=0)
        for site in frame:
            np.testing.assert_allclose(
                entry["frame"]["site_n"][site], frame[site], atol=1e-12, rtol=0
            )


# MathDoc: tb2j/qe-hubbard-native-contract
@pytest.mark.parametrize("noise", [5.0e-13, 2.0e-12])
def test_native_derivative_roundoff_uses_retained_metrics_without_widening_zero_gate(
    tmp_path, noise
):
    from TB2J.interfaces.qe_projector import QEHubbardError, parse_qe_spinor_dump

    source = _spinor_active_dump(tmp_path / "native-roundoff.bin")
    payloads = _record_payloads(source)
    extension = _find_extension(payloads)
    for index in range(extension + 1, len(payloads)):
        payload = payloads[index]
        header = np.frombuffer(payload, dtype="<i4", count=1)
        if header[0] != 22:
            continue
        header = np.frombuffer(payload, dtype="<i4", count=11)
        atom, channel, dim = (int(header[i]) for i in (1, 3, 4))
        direction = np.frombuffer(payload, dtype="<f8", count=3, offset=44)
        parts = list(_spinor_rotation_record(atom, channel, dim, direction))
        # Native and independent central differences can round differently.
        # Their arrays agree within the existing absolute comparison, but the
        # relative residual of an analytically zero derivative is ill-scaled.
        for step in range(3):
            parts[14][0, 0, 0, 0, step] = noise / 2**step
            parts[15][0, step] = noise / 2**step
            parts[16][0, step] = 1.0
        parts[17][0, :] = 0.5
        payloads[index] = _encode_record(parts)
    _write_payloads(source, payloads)

    if noise > 1.0e-12:
        with pytest.raises(QEHubbardError):
            parse_qe_spinor_dump(source)
    else:
        parsed = parse_qe_spinor_dump(source)
        for rotation in parsed.hubbard.rotations:
            assert np.max(rotation.tangent_absolute_error) <= 1.0e-12
            assert np.max(rotation.tangent_relative_error) == 1.0


# MathDoc: tb2j/qe-hubbard-native-contract
def test_scalar_spinor_parity_uses_declared_frame_not_a_second_inferred_direction(
    tmp_path,
):
    from TB2J.interfaces.qe_projector import parse_qe_spinor_dump, read_qe_spinor_dump

    source = _spinor_active_dump(tmp_path / "declared-frame-roundoff.bin")
    payloads = _record_payloads(source)
    extension = _find_extension(payloads)
    direction = np.asarray([5.0e-13, 0.0, 1.0], dtype="<f8")
    for index in range(extension + 1, len(payloads)):
        payload = payloads[index]
        record_id = struct.unpack_from("<i", payload)[0]
        if record_id == 21:
            frame = bytearray(payload)
            frame[60:84] = direction.tobytes()
            payloads[index] = bytes(frame)
        elif record_id == 22:
            header = np.frombuffer(payload, dtype="<i4", count=11)
            atom, channel, dim = (int(header[i]) for i in (1, 3, 4))
            payloads[index] = _encode_record(
                _spinor_rotation_record(atom, channel, dim, direction)
            )
    _write_payloads(source, payloads)

    # This frame is within the original native frame reconstruction tolerance.
    parse_qe_spinor_dump(source)
    data = read_qe_spinor_dump(source)
    assert np.max(np.abs(data.spinor_tangent_vertices)) > 1.0e-6
    for site, declared in data.qe_source_directions().items():
        longitudinal = np.einsum(
            "a,aijst->ijst", declared, data.spinor_tangent_vertices[site]
        )
        np.testing.assert_allclose(longitudinal, 0.0, atol=1.0e-12, rtol=0)
