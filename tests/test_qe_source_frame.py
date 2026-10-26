from __future__ import annotations

import numpy as np
import pytest

from TB2J.interfaces.gpaw_spinor_projector import compute_spinor_projector_exchange
from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    SPINOR_TANGENT_DEFINITION,
    ProjectorGreen,
    ProjectorGreenData,
)

QE_SPINOR_TANGENT_DEFINITION = "qe_native_xc_u_frozen_source_tangent_v1"
QE_SOURCE_FRAME_DEFINITION = "qe_original_xc_or_mapped_native_u_v1"
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULI = (SIGMA_X, SIGMA_Y, SIGMA_Z)


def _source_frame(
    site_map=(0,),
    directions=((0.0, 0.0, 1.0),),
    direction_provenance=("original_xc_pauli_trace",),
    **overrides,
):
    frame = {
        "version": 1,
        "definition": QE_SOURCE_FRAME_DEFINITION,
        "site_map": list(site_map),
        "directions": [list(direction) for direction in directions],
        "direction_provenance": list(direction_provenance),
    }
    frame.update(overrides)
    return frame


# MathDoc: tb2j/qe-hubbard-native-contract
def _spinor_data(
    *,
    source="zero",
    tangent_sites=(0,),
    include_tangents=True,
    original_overlap_k=None,
):
    rng = np.random.default_rng(428)
    nkpt, nband, nproj = 2, 4, 3
    kpoints = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    eigenvalues = np.array([[[-1.7, -0.4, 0.6, 1.9], [-1.3, -0.1, 0.8, 2.1]]])
    coefficients = rng.normal(size=(1, nkpt, nband, 2, nproj)) + 1j * rng.normal(
        size=(1, nkpt, nband, 2, nproj)
    )
    site_nproj = np.array([2, 1])
    site_projector_indices = np.array([[0, 1], [2, -1]])
    projector_site = np.array([0, 0, 1])
    projector_atom = projector_site.copy()
    spinor_operator = np.zeros((2, 2, 2, 2, 2), dtype=complex)
    if source == "canceling":
        spinor_operator[0, 0, 0] = SIGMA_Z
        spinor_operator[0, 1, 1] = -SIGMA_Z
    elif source == "positive":
        spinor_operator[0, 0, 0] = SIGMA_Z
    elif source == "unmapped":
        spinor_operator[1, 0, 0] = SIGMA_Z
    elif source != "zero":
        raise ValueError(f"unsupported synthetic source mode: {source}")

    tangent_vertices = None
    tangent_definition = None
    if include_tangents:
        tangent_vertices = np.zeros((2, 3, 2, 2, 2, 2), dtype=complex)
        for site in tangent_sites:
            for axis, sigma in enumerate(PAULI):
                tangent_vertices[site, axis, 0, 0] = sigma
                if site_nproj[site] > 1:
                    tangent_vertices[site, axis, 1, 1] = sigma
        tangent_definition = SPINOR_TANGENT_DEFINITION

    return ProjectorGreenData(
        kpoints=kpoints,
        weights=np.full(nkpt, 1.0 / nkpt),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.15,
        projector_site=projector_site,
        projector_atom=projector_atom,
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        nspinor=2,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        spinor_tangent_vertices=tangent_vertices,
        spinor_tangent_definition=tangent_definition,
        **(
            {"original_overlap_k": original_overlap_k}
            if original_overlap_k is not None
            else {}
        ),
    )


def _enable_qe_frame(data, frame=None):
    data.spinor_tangent_definition = QE_SPINOR_TANGENT_DEFINITION
    if frame is not None:
        data.metadata["qe_source_frame"] = frame
    data.validate()
    return data


def _green_data(*, spinor=False, overlap_k=None, original_overlap_k=None):
    kpoints = np.array([[0.0, 0.0, 0.0]])
    if spinor:
        coefficients = np.array(
            [[[[[1.0, 0.2j], [0.3, 0.7]], [[0.4j, 0.8], [0.9, -0.1j]]]]],
            dtype=complex,
        )
        eigenvalues = np.array([[[-0.7, 1.2]]])
        nspinor = 2
    else:
        coefficients = np.array(
            [
                [[[1.0, 0.2j], [0.3, 0.7]]],
                [[[0.4j, 0.8], [0.9, -0.1j]]],
            ],
            dtype=complex,
        )
        eigenvalues = np.array([[[-0.7, 1.2]], [[-0.4, 1.5]]])
        nspinor = 1
    return ProjectorGreenData(
        kpoints=kpoints,
        weights=np.ones(1),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.05,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        site_nproj=np.array([2]),
        site_projector_indices=np.array([[0, 1]]),
        overlap_k=overlap_k,
        **(
            {"original_overlap_k": original_overlap_k}
            if original_overlap_k is not None
            else {}
        ),
        nspinor=nspinor,
    )


def test_filtered_coefficients_reuse_k_filter_for_row_and_leading_spin_axes():
    overlap = np.array([[[2.0, 0.3j], [-0.3j, 1.4]]], dtype=complex)
    data = _green_data(overlap_k=overlap)
    green = ProjectorGreen(data, overlap_mode="inverse")
    filt = np.linalg.inv(overlap[0])

    rows = data.coefficients[1, 0]
    actual = green.get_filtered_coefficients(0, ispin=1)
    np.testing.assert_allclose(actual, rows @ filt.T, rtol=1e-13, atol=1e-13)

    raw = _green_data()
    raw_green = ProjectorGreen(raw)
    np.testing.assert_array_equal(
        raw_green.get_filtered_coefficients(0), raw.coefficients[0, 0]
    )

    spinor = _green_data(spinor=True, overlap_k=overlap)
    spinor_green = ProjectorGreen(spinor, overlap_mode="inverse")
    expected_spinor = np.einsum("...p,pq->...q", spinor.coefficients[0, 0], filt.T)
    np.testing.assert_allclose(
        spinor_green.get_filtered_coefficients(0),
        expected_spinor,
        rtol=1e-13,
        atol=1e-13,
    )


def test_original_overlap_is_provenance_only_and_roundtrips_with_frame(tmp_path):
    pytest.importorskip("netCDF4")
    gram = np.array(
        [
            [
                [1.8, 0.2 + 0.1j, 0.05j],
                [0.2 - 0.1j, 1.3, 0.12],
                [-0.05j, 0.12, 1.1],
            ],
            [
                [1.6, -0.1j, 0.04],
                [0.1j, 1.5, 0.08j],
                [0.04, -0.08j, 1.25],
            ],
        ],
        dtype=complex,
    )
    # Appended native-U columns do not enlarge the original atomic Gram.
    gram = gram[:, :2, :2]
    frame = _source_frame(direction_provenance=("mapped_native_hubbard_occupation",))
    data = _enable_qe_frame(
        _spinor_data(original_overlap_k=gram),
        frame,
    )
    path = tmp_path / "qe_source_frame.nc"
    data.save_netcdf(path)
    restored = ProjectorGreenData.load_netcdf(path)

    np.testing.assert_array_equal(restored.original_overlap_k, gram)
    assert restored.metadata["qe_source_frame"] == frame
    directions = restored.qe_source_directions()
    assert set(directions) == {0}
    np.testing.assert_array_equal(directions[0], [0.0, 0.0, 1.0])

    # This normalized data has no runtime overlap. The retained original Gram
    # must not dress coefficients, Green functions, or get_Sk.
    green = ProjectorGreen(restored)
    np.testing.assert_array_equal(green.get_Sk(0), np.eye(restored.nproj))
    np.testing.assert_array_equal(
        green.get_filtered_coefficients(0), restored.coefficients[0, 0]
    )
    energy = 0.37
    expected = np.einsum(
        "nsp,ntq,n->pqst",
        restored.coefficients[0, 0],
        restored.coefficients[0, 0].conj(),
        1.0 / (energy + restored.efermi - restored.eigenvalues[0, 0]),
    )
    np.testing.assert_allclose(green.get_Gk_spinor(0, energy), expected)


def test_original_overlap_k_requires_full_finite_k_gram():
    valid = np.repeat(np.eye(2, dtype=complex)[None, :, :], 1, axis=0)
    with pytest.raises(ValueError):
        _green_data(original_overlap_k=np.zeros((2, 2)))
    invalid = valid.copy()
    invalid[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        _green_data(original_overlap_k=invalid)


@pytest.mark.parametrize(
    "bad_frame",
    [
        None,
        {"version": 2, "definition": QE_SOURCE_FRAME_DEFINITION},
        _source_frame(definition="other_frame"),
        _source_frame(
            site_map=(0, 0),
            directions=((0, 0, 1), (0, 0, 1)),
            direction_provenance=("original_xc_pauli_trace",) * 2,
        ),
        _source_frame(site_map=(), directions=(), direction_provenance=()),
        _source_frame(site_map=(-1,)),
        _source_frame(site_map=(2,)),
        _source_frame(directions=((0.0, 0.0),)),
        _source_frame(directions=((0.0, 0.0, 2.0),)),
        _source_frame(directions=((0.0, np.nan, 1.0),)),
        _source_frame(direction_provenance=("inferred_from_total",)),
        _source_frame(direction_provenance=()),
    ],
)
def test_new_tangent_contract_rejects_missing_or_malformed_source_frames(bad_frame):
    data = _spinor_data()
    if bad_frame is not None:
        data.metadata["qe_source_frame"] = bad_frame
    data.spinor_tangent_definition = QE_SPINOR_TANGENT_DEFINITION
    with pytest.raises(ValueError):
        data.validate()


def test_new_frame_requires_explicit_tangents_even_for_zero_operator():
    data = _spinor_data(include_tangents=False)
    data.metadata["qe_source_frame"] = _source_frame(
        direction_provenance=("mapped_native_hubbard_occupation",)
    )
    data.spinor_tangent_definition = QE_SPINOR_TANGENT_DEFINITION
    with pytest.raises(ValueError):
        data.validate()


def test_canceling_source_traces_use_nonzero_explicit_tangents():
    results = {}
    for source in ("zero", "canceling"):
        provenance = (
            "mapped_native_hubbard_occupation"
            if source == "zero"
            else "original_xc_pauli_trace"
        )
        data = _enable_qe_frame(
            _spinor_data(source=source),
            _source_frame(direction_provenance=(provenance,)),
        )
        if source == "canceling":
            # The two signed spin sources cancel in their local Pauli trace,
            # while their declared tangent vertices remain nonzero.
            assert np.any(data.spinor_operator[0, 0, 0] != 0.0)
            assert np.any(data.spinor_operator[0, 1, 1] != 0.0)
            np.testing.assert_allclose(
                np.trace(data.spinor_operator[0, :, :, 0, 0])
                - np.trace(data.spinor_operator[0, :, :, 1, 1]),
                0.0,
            )
        assert np.any(data.spinor_tangent_vertices[0] != 0.0)
        results[source] = compute_spinor_projector_exchange(
            data, Rpts=[[0, 0, 0]], nz=4, smearing_eV=0.1
        )
    assert set(results["zero"]) == {((0, 0, 0), 0, 0)}
    entry = results["zero"][((0, 0, 0), 0, 0)]
    np.testing.assert_array_equal(entry["frame"]["site_n"][0], [0.0, 0.0, 1.0])
    assert entry["J_leg"].shape == (3, 3)
    assert np.all(np.isfinite(entry["J_leg"]))
    np.testing.assert_allclose(
        results["canceling"][((0, 0, 0), 0, 0)]["J_leg"], entry["J_leg"]
    )


def test_explicit_frame_controls_default_sites_and_rejects_unmapped_selection():
    frame = _source_frame(
        site_map=(1,),
        directions=((1.0, 0.0, 0.0),),
        direction_provenance=("mapped_native_hubbard_occupation",),
    )
    data = _enable_qe_frame(
        _spinor_data(tangent_sites=(1,)),
        frame,
    )
    result = compute_spinor_projector_exchange(
        data, Rpts=[[0, 0, 0]], nz=4, smearing_eV=0.1
    )
    assert set(result) == {((0, 0, 0), 1, 1)}
    np.testing.assert_array_equal(
        result[((0, 0, 0), 1, 1)]["frame"]["site_n"][1], [1.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError):
        compute_spinor_projector_exchange(
            data, Rpts=[[0, 0, 0]], nz=4, smearing_eV=0.1, sites=[0]
        )


@pytest.mark.parametrize("unmapped", ["source", "tangent"])
def test_new_frame_refuses_nonzero_source_or_tangent_at_unmapped_site(unmapped):
    data = _spinor_data(
        source="unmapped" if unmapped == "source" else "zero",
        tangent_sites=(1,) if unmapped == "tangent" else (0,),
    )
    data.metadata["qe_source_frame"] = _source_frame()
    data.spinor_tangent_definition = QE_SPINOR_TANGENT_DEFINITION
    with pytest.raises(ValueError):
        data.validate()


def test_old_spinor_contract_keeps_operator_trace_direction_inference():
    data = _spinor_data(source="positive", include_tangents=False)
    assert data.qe_source_directions() is None
    result = compute_spinor_projector_exchange(
        data, Rpts=[[0, 0, 0]], nz=4, smearing_eV=0.1, sites=[0]
    )
    np.testing.assert_array_equal(
        result[((0, 0, 0), 0, 0)]["frame"]["site_n"][0], [0.0, 0.0, 1.0]
    )
