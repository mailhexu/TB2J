"""Contract tests for the validated PAW projector snapshot seam."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from TB2J.paw_projector import (
    HARTREE_TO_EV,
    PawOperatorComponent,
    PawOperatorComponents,
    PawProjectorChannel,
    PawProjectorSnapshot,
    PawSiteLayout,
    build_projector_green_data,
    validate_full_bz_mesh,
    validate_paw_projector_snapshot,
)
from TB2J.projector_green import ProjectorGreen, projector_exchange_trace


def _layout() -> tuple[PawSiteLayout, ...]:
    channels = (
        PawProjectorChannel(l=2, m=-2, radial=0, label="dxy"),
        PawProjectorChannel(l=2, m=-1, radial=0, label="dyz"),
    )
    return (
        PawSiteLayout(
            source_site=0,
            species="Fe",
            atomic_number=26,
            projector_slice=slice(0, 2),
            channels=channels,
            setup_hash="fe-paw-hash",
        ),
        PawSiteLayout(
            source_site=1,
            species="O",
            atomic_number=8,
            projector_slice=slice(2, 4),
            channels=channels,
            setup_hash="o-paw-hash",
        ),
    )


def _component(name: str, value: float, *, included_in_total: bool = False):
    blocks = np.zeros((2, 2, 2), dtype=complex)
    blocks[:, 0, 0] = value
    blocks[:, 1, 1] = 2.0 * value
    return PawOperatorComponent(
        name=name,
        values=blocks,
        units="Hartree",
        basis_id="native_paw_projector_hamiltonian",
        definition=f"synthetic {name} spin difference",
        source="synthetic",
        included_in_total=included_in_total,
    )


def _snapshot(
    *,
    operators: PawOperatorComponents | None = None,
    provenance: dict | None = None,
    layout: tuple[PawSiteLayout, ...] | None = None,
) -> PawProjectorSnapshot:
    if operators is None:
        operators = PawOperatorComponents(
            components=(_component("xc", 0.1), _component("hubbard", 0.2)),
            policy="compose",
            selected_names=("xc", "hubbard"),
        )
    if provenance is None:
        provenance = {
            "source_code": "synthetic",
            "source_version": "1.0",
            "functional": "PBE+U",
            "setup_hashes": ["fe-paw-hash", "o-paw-hash"],
            "u_eV": 5.0,
            "j_eV": 0.0,
            "correlated_shells": ["Fe:3d"],
        }
    coefficients = np.zeros((2, 1, 2, 4), dtype=complex)
    coefficients[:, 0, 0, 0] = 1.0
    coefficients[:, 0, 1, 3] = 1.0
    return PawProjectorSnapshot(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        weights=np.array([1.0]),
        eigenvalues=np.array([[[-1.0, 1.0]], [[-0.8, 1.2]]]),
        occupations=np.array([[[1.0, 0.0]], [[1.0, 0.0]]]),
        coefficients=coefficients,
        efermi=0.0,
        cell=np.eye(3) * 3.0,
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
        atomic_numbers=np.array([26, 8]),
        site_layout=_layout() if layout is None else layout,
        operators=operators,
        kpoint_mode="full_bz",
        selected_source_sites=(0,),
        provenance=provenance,
    )


def test_valid_snapshot_builds_dual_projector_green_data():
    snapshot = _snapshot()

    report = validate_paw_projector_snapshot(snapshot)
    data = build_projector_green_data(snapshot)

    assert report.selected_component_names == ("xc", "hubbard")
    assert data.overlap_k is None
    assert data.metadata["paw_operator_policy"] == "compose"
    assert data.metadata["selected_operator_components"] == ["xc", "hubbard"]
    np.testing.assert_allclose(
        data.get_operator_component("delta_total", site=0),
        np.diag([0.3, 0.6]) * HARTREE_TO_EV,
    )
    assert data.operator_component_metadata["delta_total"]["units"] == "eV"


def test_snapshot_owns_immutable_input_arrays_and_provenance():
    snapshot = _snapshot()

    with pytest.raises(ValueError, match="read-only"):
        snapshot.coefficients[0, 0, 0, 0] = 0.0
    with pytest.raises(TypeError):
        snapshot.provenance["functional"] = "LDA"


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (
            lambda: _snapshot(
                layout=(
                    replace(_layout()[0], projector_slice=slice(0, 1)),
                    _layout()[1],
                )
            ),
            "channel count",
        ),
        (
            lambda: _snapshot(
                operators=PawOperatorComponents(
                    components=(replace(_component("xc", 0.1), units="eV"),),
                    policy="compose",
                    selected_names=("xc",),
                )
            ),
            "Hartree",
        ),
        (
            lambda: _snapshot(
                operators=PawOperatorComponents(
                    components=(
                        replace(
                            _component("xc", 0.1),
                            basis_id="unrecognized_projector_basis",
                        ),
                    ),
                    policy="compose",
                    selected_names=("xc",),
                )
            ),
            "native_paw_projector_hamiltonian",
        ),
        (
            lambda: _snapshot(
                operators=PawOperatorComponents(
                    components=(
                        replace(
                            _component("xc", 0.1),
                            values=np.array(
                                [[[0.1, 0.2], [0.0, 0.2]], [[0.1, 0.2], [0.0, 0.2]]],
                                dtype=complex,
                            ),
                        ),
                    ),
                    policy="compose",
                    selected_names=("xc",),
                )
            ),
            "Hermitian",
        ),
        (
            lambda: _snapshot(
                provenance={
                    key: value
                    for key, value in _snapshot().provenance.items()
                    if key != "functional"
                }
            ),
            "functional",
        ),
    ],
)
def test_invalid_snapshot_fails_before_green_construction(snapshot, message):
    with pytest.raises(ValueError, match=message):
        build_projector_green_data(snapshot())


def test_operator_policies_prevent_double_counting_and_record_total_selection():
    invalid = _snapshot(
        operators=PawOperatorComponents(
            components=(_component("xc", 0.1), _component("total", 0.3)),
            policy="compose",
            selected_names=("xc", "total"),
        )
    )
    with pytest.raises(ValueError, match="cannot select total"):
        validate_paw_projector_snapshot(invalid)

    authoritative = _snapshot(
        operators=PawOperatorComponents(
            components=(
                _component("xc", 0.1, included_in_total=True),
                _component("total", 0.3),
            ),
            policy="authoritative_total",
            selected_names=("total",),
        )
    )
    data = build_projector_green_data(authoritative)

    assert data.metadata["paw_operator_policy"] == "authoritative_total"
    assert data.metadata["selected_operator_components"] == ["total"]
    np.testing.assert_allclose(
        data.get_operator_component("delta_total", site=1),
        np.diag([0.3, 0.6]) * HARTREE_TO_EV,
    )


def test_complete_full_bz_mesh_passes():
    validate_full_bz_mesh(
        np.array([[x, y, 0.0] for x in (0.0, 0.5) for y in (0.0, 0.5)]),
        np.full(4, 0.25),
    )


@pytest.mark.parametrize(
    ("kpoints", "weights", "message"),
    [
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([0.5, 0.5]),
            "duplicate",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
            np.full(3, 1 / 3),
            "incomplete",
        ),
        (np.array([[0.0, 0.0, 0.0]]), np.array([0.8]), "normalized"),
    ],
)
def test_invalid_full_bz_mesh_fails(kpoints, weights, message):
    with pytest.raises(ValueError, match=message):
        validate_full_bz_mesh(kpoints, weights)


def test_unproven_ibz_expansion_fails_before_green_construction():
    snapshot = replace(_snapshot(), kpoint_mode="expanded_from_ibz")
    with pytest.raises(ValueError, match="IBZ expansion"):
        build_projector_green_data(snapshot)


def _covariant_component(name: str, values: np.ndarray) -> PawOperatorComponent:
    return PawOperatorComponent(
        name=name,
        values=values,
        units="Hartree",
        basis_id="native_paw_projector_hamiltonian",
        definition=f"synthetic explicit {name} contribution",
        source="synthetic PAW producer",
        included_in_total=False,
    )


def _covariant_snapshot(*, source_code: str = "synthetic") -> PawProjectorSnapshot:
    """Return a two-site fixture with nonzero intersite exchange traces."""
    xc = np.array(
        [
            [[0.30, 0.07 - 0.04j], [0.07 + 0.04j, -0.18]],
            [[-0.22, 0.03 + 0.06j], [0.03 - 0.06j, 0.15]],
        ],
        dtype=complex,
    )
    hubbard = np.array(
        [
            [[0.12, -0.02 + 0.09j], [-0.02 - 0.09j, 0.20]],
            [[0.05, 0.08 - 0.01j], [0.08 + 0.01j, -0.11]],
        ],
        dtype=complex,
    )
    coefficients = np.array(
        [
            [
                [
                    [0.83 + 0.11j, -0.21 + 0.37j, 0.46 - 0.13j, 0.15 + 0.29j],
                    [-0.34 + 0.25j, 0.72 - 0.18j, -0.19 + 0.31j, 0.57 + 0.08j],
                ]
            ],
            [
                [
                    [0.63 - 0.28j, 0.17 + 0.44j, 0.39 + 0.22j, -0.26 + 0.35j],
                    [0.28 + 0.36j, -0.66 + 0.14j, 0.51 - 0.17j, 0.42 + 0.23j],
                ]
            ],
        ],
        dtype=complex,
    )
    snapshot = _snapshot(
        operators=PawOperatorComponents(
            components=(
                _covariant_component("xc", xc),
                _covariant_component("hubbard", hubbard),
            ),
            policy="compose",
            selected_names=("xc", "hubbard"),
        ),
        provenance={
            "source_code": source_code,
            "source_version": "26.7.0" if source_code == "GPAW" else "1.0",
            "functional": "PBE+U",
            "setup_hashes": ["fe-paw-hash", "o-paw-hash"],
            "u_eV": 5.0,
            "j_eV": 0.0,
            "correlated_shells": ["Fe:3d"],
        },
    )
    return replace(
        snapshot,
        coefficients=coefficients,
        selected_source_sites=(0, 1),
    )


def _transform_snapshot(
    snapshot: PawProjectorSnapshot,
    site_changes: tuple[np.ndarray, ...],
    *,
    transform_operators: bool,
) -> PawProjectorSnapshot:
    """Apply a local dual-projector change of basis to a synthetic snapshot.

    For a projector-coordinate change ``X`` on each site, dual coefficients use
    ``C' = C X`` and local PAW operators use
    ``Delta' = X⁻* Delta X⁻ᵀ``. Consequently the dual Green blocks transform
    as ``G' = Xᵀ G X*``, leaving the closed exchange trace invariant.
    """
    coefficients = np.array(snapshot.coefficients, copy=True)
    for layout, change in zip(snapshot.site_layout, site_changes):
        nproj = layout.projector_slice.stop - layout.projector_slice.start
        local_change = np.asarray(change, dtype=complex)
        assert local_change.shape == (nproj, nproj)
        assert np.linalg.det(local_change) != 0.0
        coefficients[..., layout.projector_slice] = (
            coefficients[..., layout.projector_slice] @ local_change
        )

    components = []
    for component in snapshot.operators.components:
        values = np.array(component.values, copy=True)
        if transform_operators:
            for site, (layout, change) in enumerate(
                zip(snapshot.site_layout, site_changes)
            ):
                nproj = layout.projector_slice.stop - layout.projector_slice.start
                inverse_conjugate = np.linalg.inv(
                    np.asarray(change, dtype=complex).conj()
                )
                values[site, :nproj, :nproj] = (
                    inverse_conjugate
                    @ values[site, :nproj, :nproj]
                    @ inverse_conjugate.conj().T
                )
        components.append(replace(component, values=values))
    return replace(
        snapshot,
        coefficients=coefficients,
        operators=replace(snapshot.operators, components=tuple(components)),
    )


def _exchange_trace(snapshot: PawProjectorSnapshot) -> dict[tuple, complex]:
    data = build_projector_green_data(snapshot)
    green = ProjectorGreen(data)
    operators = {
        site: data.get_operator_component("delta_total", site=site)
        for site in range(len(snapshot.site_layout))
    }
    return projector_exchange_trace(
        green,
        np.array([[0, 0, 0]]),
        energy=0.13j,
        local_operators=operators,
        sites=[0, 1],
    )["trace"]


def _deterministic_unitary_changes() -> tuple[np.ndarray, ...]:
    generator = np.random.default_rng(936)
    changes = []
    for _ in range(2):
        raw = generator.normal(size=(2, 2)) + 1j * generator.normal(size=(2, 2))
        unitary, _ = np.linalg.qr(raw)
        changes.append(unitary)
    return tuple(changes)


@pytest.mark.parametrize(
    "site_changes",
    [
        _deterministic_unitary_changes(),
        (
            np.array([[1.3 + 0.2j, 0.1 - 0.3j], [0.0, 0.8 - 0.1j]]),
            np.array([[0.9 - 0.2j, -0.2 + 0.1j], [0.3 + 0.1j, 1.2 + 0.2j]]),
        ),
    ],
    ids=("deterministic_unitary", "deterministic_invertible"),
)
def test_local_dual_projector_basis_change_preserves_exchange_trace(site_changes):
    reference = _exchange_trace(_covariant_snapshot())

    transformed = _exchange_trace(
        _transform_snapshot(
            _covariant_snapshot(),
            site_changes,
            transform_operators=True,
        )
    )

    assert reference.keys() == transformed.keys()
    np.testing.assert_allclose(
        tuple(transformed.values()),
        tuple(reference.values()),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_unpaired_dual_projector_coefficient_change_changes_exchange_trace():
    reference = _exchange_trace(_covariant_snapshot())
    inconsistent = _exchange_trace(
        _transform_snapshot(
            _covariant_snapshot(),
            _deterministic_unitary_changes(),
            transform_operators=False,
        )
    )

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            tuple(inconsistent.values()),
            tuple(reference.values()),
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_gpaw_explicit_dftu_components_are_covariant_under_local_basis_change():
    snapshot = _covariant_snapshot(source_code="GPAW")
    transformed_snapshot = _transform_snapshot(
        snapshot,
        _deterministic_unitary_changes(),
        transform_operators=True,
    )

    transformed_data = build_projector_green_data(transformed_snapshot)
    assert transformed_data.metadata["selected_operator_components"] == [
        "xc",
        "hubbard",
    ]
    assert transformed_data.metadata["source_code"] == "GPAW"
    np.testing.assert_allclose(
        tuple(_exchange_trace(transformed_snapshot).values()),
        tuple(_exchange_trace(snapshot).values()),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
