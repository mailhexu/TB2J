"""Behavioral checks for joint QE XC/native-Hubbard consumer normalization.

All native extensions are written by strict deterministic test fixtures; no QE
installation or workstation export is required.
"""

from __future__ import annotations

import numpy as np
from test_qe_hubbard import (
    _active_collinear_dump,
    _kb_no_u_dump,
    _spinor_native_slot_dump,
)
from test_qe_projector import RYTOEV

from TB2J.interfaces.qe_projector import (
    parse_qe_atomic_dump,
    parse_qe_dump,
    parse_qe_spinor_dump,
    read_qe_atomic_dump,
    read_qe_dump,
    read_qe_spinor_dump,
)
from TB2J.projector_green import (
    QE_SOURCE_FRAME_DEFINITION,
    QE_SPINOR_TANGENT_DEFINITION,
    ProjectorGreen,
)

# MathDoc: tb2j/qe-hubbard-joint-frame#row_filter_transpose
# MathDoc: tb2j/qe-hubbard-native-contract
# MathDoc: tb2j/qe-hubbard-joint-frame#joint_trace


def _native_coefficients_for_targets(gt, targets=(0.12, 0.28)):
    """Construct strict native C overlaps whose weighted traces are targets."""
    coefficients = np.zeros((gt.nks, 1, gt.nbnd), dtype=complex)
    for spin, target in enumerate(targets, start=1):
        indices = np.flatnonzero(gt.isk == spin)
        total_weight = float(np.sum(gt.wg[indices, 0]))
        amplitude = np.sqrt(float(target) / total_weight)
        for ik in indices:
            coefficients[ik, 0, 0] = amplitude * np.exp(0.37j * (ik + 1))
    return coefficients


def _complex_atomic_grams():
    rng = np.random.default_rng(95521)
    first = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    second = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gamma = first @ first.conj().T + 3.0 * np.eye(4)
    boundary = second @ second.conj().T + 4.0 * np.eye(4)
    # Spin-paired records describe the same k; the two distinct k Grams are
    # deliberately full, complex and nonidentical.
    return [gamma, gamma.copy(), boundary, boundary.copy()]


def _direct_separable_trace(c_up, c_down, weights_up, weights_down, operator):
    """Band-pair oracle preserving the operator order in Tr[D G_up D G_down]."""
    forward = c_up.conj() @ operator @ c_down.T
    reverse = c_down.conj() @ operator @ c_up.T
    return np.sum(weights_up[:, None] * weights_down[None, :] * forward * reverse.T)


def test_kb_joint_reader_preserves_native_becp_green_and_frozen_source(tmp_path):
    path = _active_collinear_dump(
        tmp_path / "kb-joint.bin",
        state_occupation=(0.2, 0.6),
        native_coefficients=_native_coefficients_for_targets,
    )
    raw = parse_qe_dump(path)
    baseline = read_qe_dump(_kb_no_u_dump(tmp_path / "kb-no-u.bin"))
    data = read_qe_dump(path)

    state = raw.hubbard.states[0]
    fresh = raw.hubbard.occupation_checks[0].native
    np.testing.assert_allclose(state.occupation[0, 0], [0.2, 0.6])
    np.testing.assert_allclose(
        state.potential[0, 0],
        (2.0 / RYTOEV) * (0.5 - np.array([0.2, 0.6])),
    )
    np.testing.assert_allclose(fresh[0, 0], [0.12, 0.28])
    assert not np.allclose(state.occupation, fresh)
    for spin in (1, 2):
        measured = sum(
            np.dot(kpoint.wg, np.abs(kpoint.coefficients[0]) ** 2)
            for kpoint in raw.hubbard.kpoints
            if kpoint.spin == spin
        )
        np.testing.assert_allclose(measured, fresh[0, 0, spin - 1])

    original_nproj = baseline.nproj
    assert data.coefficients.shape[-1] == original_nproj + raw.hubbard.envelope["ncol"]
    np.testing.assert_allclose(
        data.coefficients[..., :original_nproj], baseline.coefficients
    )
    assert data.overlap_k is None

    mesh_by_spin = ((0, 2), (1, 3))
    for spin, full_mesh in enumerate(mesh_by_spin):
        for ik, raw_ik in enumerate(full_mesh):
            np.testing.assert_allclose(
                data.coefficients[spin, ik, :, original_nproj:],
                raw.hubbard.kpoints[raw_ik].coefficients.T,
            )

    green, old_green = ProjectorGreen(data), ProjectorGreen(baseline)
    energy = 0.73 + 0.29j
    for spin in range(2):
        for ik in range(data.nkpt):
            joint = green.get_Gk(ik, energy, ispin=spin)
            old = old_green.get_Gk(ik, energy, ispin=spin)
            np.testing.assert_allclose(joint[:original_nproj, :original_nproj], old)

    site = 2
    old_xc = baseline.get_operator_component("delta_total", site=site)
    joint_xc = data.get_operator_component("delta_xc", site=site)
    np.testing.assert_allclose(joint_xc[: old_xc.shape[0], : old_xc.shape[1]], old_xc)


def test_atomic_joint_reader_filters_only_original_xc_once_per_k(tmp_path):
    grams = _complex_atomic_grams()
    path = _active_collinear_dump(
        tmp_path / "atomic-joint.bin",
        family="atomic",
        state_occupation=(0.2, 0.6),
        native_coefficients=_native_coefficients_for_targets,
        atomic_grams=grams,
    )
    raw = parse_qe_atomic_dump(path)
    rcond = 3.0e-9
    data = read_qe_atomic_dump(path, overlap_mode="inverse", overlap_rcond=rcond)

    original_nproj = raw.nproj
    assert np.max(np.abs(raw.grams[0].imag)) > 1e-3
    assert not np.allclose(raw.grams[0], raw.grams[2])
    assert data.overlap_k is None
    np.testing.assert_allclose(data.original_overlap_k, np.asarray(raw.grams)[[0, 2]])
    assert data.metadata["effective_overlap_mode"] == "inverse"
    assert data.metadata["effective_overlap_rcond"] == rcond

    mesh_by_spin = ((0, 2), (1, 3))
    for spin, full_mesh in enumerate(mesh_by_spin):
        for ik, raw_ik in enumerate(full_mesh):
            gram = raw.grams[raw_ik]
            filt = np.linalg.inv(gram)
            original = raw.coefficients[raw_ik].T
            expected_xc = original @ filt.T
            np.testing.assert_allclose(
                data.coefficients[spin, ik, :, :original_nproj], expected_xc
            )
            assert np.linalg.norm(expected_xc - original @ filt) > 1e-6
            assert np.linalg.norm(expected_xc - expected_xc @ filt.T) > 1e-6
            np.testing.assert_allclose(
                data.coefficients[spin, ik, :, original_nproj:],
                raw.hubbard.kpoints[raw_ik].coefficients.T,
            )

    second_k_unfiltered = raw.coefficients[2].T
    first_gram_shortcut = second_k_unfiltered @ np.linalg.inv(raw.grams[0]).T
    assert (
        np.linalg.norm(
            data.coefficients[0, 1, :, :original_nproj] - first_gram_shortcut
        )
        > 1e-6
    )
    green = ProjectorGreen(data)
    energy = 0.61 + 0.31j
    coefficients = data.coefficients[0, 1]
    denominator = energy + data.efermi_spin[0] - data.eigenvalues[0, 1]
    expected_green = sum(
        np.outer(row, row.conj()) / pole
        for row, pole in zip(coefficients, denominator, strict=True)
    )
    np.testing.assert_allclose(green.get_Gk(1, energy, ispin=0), expected_green)


def test_spinor_native_slots_keep_spin_major_orbital_inner_packing(tmp_path):
    path = _spinor_native_slot_dump(tmp_path / "spinor-native-slots.bin")
    raw = parse_qe_spinor_dump(path)
    data = read_qe_spinor_dump(path)

    original_nproj = raw.nkb
    norbital = raw.hubbard.layout["norbital"]
    assert raw.hubbard.envelope["ncol"] == 2 * norbital
    assert data.nspinor == 2
    assert data.coefficients.shape == (
        1,
        raw.nks,
        raw.nbnd,
        2,
        original_nproj + norbital,
    )
    assert len(data.metadata["qe_channel_groups"]["u"]) == norbital

    frame = data.metadata["qe_source_frame"]
    assert frame["version"] == 1
    assert frame["definition"] == QE_SOURCE_FRAME_DEFINITION
    assert data.spinor_tangent_definition == QE_SPINOR_TANGENT_DEFINITION
    directions = np.asarray(frame["directions"])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    site_map = frame["site_map"]
    assert len(site_map) == len(set(site_map)) == len(directions)
    assert frame["direction_provenance"] == ["original_xc_pauli_trace"] * len(
        directions
    )
    assert data.spinor_tangent_vertices is not None
    for site, old_count in enumerate(raw.nh_per_atom):
        np.testing.assert_allclose(
            data.spinor_tangent_vertices[site, :, old_count : data.site_nproj[site]],
            0.0,
        )

    channel_starts = {}
    start = 0
    for channel in raw.hubbard.channels:
        channel_starts[channel.channel] = start
        start += channel.dimension
    for ik, kpoint in enumerate(raw.hubbard.kpoints):
        slot_values = []
        for native, channel, orbital, spin_slot in raw.hubbard.spin_columns:
            global_orbital = original_nproj + channel_starts[channel] + orbital - 1
            value = kpoint.coefficients[native - 1, 0]
            slot_values.append(value)
            np.testing.assert_allclose(
                data.coefficients[0, ik, :, spin_slot - 1, global_orbital],
                kpoint.coefficients[native - 1],
            )
        assert np.all(np.abs(slot_values) > 0)
        assert np.all(np.abs(np.imag(slot_values)) > 0)
        assert len(set(slot_values)) == len(slot_values)
    assert np.any(raw.hubbard.occupation_checks[0].native)
    assert not np.any(raw.hubbard.states[0].occupation)

    green = ProjectorGreen(data)
    energy = 0.83 + 0.47j
    for ik in range(data.nkpt):
        coeff = data.coefficients[0, ik]
        poles = energy + data.efermi - data.eigenvalues[0, ik]
        expected = sum(
            np.einsum("sp,tq->pqst", band, band.conj()) / pole
            for band, pole in zip(coeff, poles, strict=True)
        )
        actual = green.get_Gk_spinor(ik, energy)
        np.testing.assert_allclose(actual, expected)
        assert np.linalg.norm(actual[:, :, 0, 1]) > 1e-12


def test_redundant_joint_frame_trace_retains_xc_u_cross_contractions(tmp_path):
    def duplicate_one_xc_channel(gt):
        return np.asarray([0.125 * becp[6:7] for becp in gt.becps])

    path = _active_collinear_dump(
        tmp_path / "kb-rank-deficient-joint.bin",
        state_occupation=(0.2, 0.6),
        native_coefficients=duplicate_one_xc_channel,
    )
    data = read_qe_dump(path)
    groups = data.metadata["qe_channel_groups"]
    site = 2
    global_indices = data.site_projector_indices[site, : data.site_nproj[site]]
    xc_global = set(groups["xc"])
    u_global = set(groups["u"])
    local_xc = [i for i, global_i in enumerate(global_indices) if global_i in xc_global]
    local_u = [i for i, global_i in enumerate(global_indices) if global_i in u_global]
    assert len(local_xc) == 2 and len(local_u) == 1

    local_coefficients = data.coefficients[0, 0][:, global_indices]
    np.testing.assert_allclose(
        local_coefficients[:, local_u[0]],
        0.125 * local_coefficients[:, local_xc[0]],
    )
    assert np.linalg.matrix_rank(local_coefficients.T, tol=1e-11) < len(global_indices)

    delta_xc = data.get_operator_component("delta_xc", site=site)
    delta_u = data.get_operator_component("delta_u", site=site)
    delta_total = data.get_operator_component("delta_total", site=site)
    np.testing.assert_allclose(delta_total, delta_xc + delta_u)
    assert not np.any(delta_xc[np.ix_(local_u, local_xc)])
    assert not np.any(delta_u[np.ix_(local_xc, local_u)])
    assert np.linalg.norm(delta_xc) > 0
    assert np.linalg.norm(delta_u) > 0

    green = ProjectorGreen(data)
    energy = 0.77 + 0.26j
    mixed = isolated_xc = isolated_u = direct_mixed = direct_xc = direct_u = 0j
    for ik, weight in enumerate(data.weights):
        g_up = green.get_Gk(ik, energy, ispin=0)
        g_down = green.get_Gk(ik, energy, ispin=1)
        g_up = g_up[np.ix_(global_indices, global_indices)]
        g_down = g_down[np.ix_(global_indices, global_indices)]
        if ik == 0:
            assert np.linalg.norm(g_up[np.ix_(local_xc, local_u)]) > 0
        mixed += weight * np.trace(delta_total @ g_up @ delta_total @ g_down)
        isolated_xc += weight * np.trace(delta_xc @ g_up @ delta_xc @ g_down)
        isolated_u += weight * np.trace(delta_u @ g_up @ delta_u @ g_down)

        c_up = data.coefficients[0, ik][:, global_indices]
        c_down = data.coefficients[1, ik][:, global_indices]
        poles_up = 1.0 / (energy + data.efermi_spin[0] - data.eigenvalues[0, ik])
        poles_down = 1.0 / (energy + data.efermi_spin[1] - data.eigenvalues[1, ik])
        direct_mixed += weight * _direct_separable_trace(
            c_up, c_down, poles_up, poles_down, delta_total
        )
        direct_xc += weight * _direct_separable_trace(
            c_up, c_down, poles_up, poles_down, delta_xc
        )
        direct_u += weight * _direct_separable_trace(
            c_up, c_down, poles_up, poles_down, delta_u
        )

    np.testing.assert_allclose(mixed, direct_mixed, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(isolated_xc, direct_xc, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(isolated_u, direct_u, rtol=2e-11, atol=2e-11)
    assert abs(mixed - (isolated_xc + isolated_u)) > 1e-8
    assert abs(direct_mixed - (direct_xc + direct_u)) > 1e-8
