"""NetCDF persistence contracts for validated PAW projector snapshots."""

from __future__ import annotations

import numpy as np
import pytest

from TB2J.paw_projector import (
    PawOperatorComponent,
    PawOperatorComponents,
    PawProjectorChannel,
    PawProjectorSnapshot,
    PawSiteLayout,
    build_projector_green_data,
)
from TB2J.paw_snapshot_cache import (
    paw_snapshot_cache_identity,
    read_paw_snapshot_netcdf,
    write_paw_snapshot_netcdf,
)
from TB2J.projector_green import ProjectorGreen, projector_exchange_trace


def _snapshot() -> PawProjectorSnapshot:
    channels = (
        PawProjectorChannel(l=2, m=-2, radial=0, label="dxy"),
        PawProjectorChannel(l=2, m=-1, radial=0, label="dyz"),
    )
    layout = (
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
    blocks = np.zeros((2, 2, 2), dtype=complex)
    blocks[:, 0, 0] = 0.1
    blocks[:, 1, 1] = 0.2
    operator = PawOperatorComponent(
        name="xc",
        values=blocks,
        units="Hartree",
        basis_id="native_paw_projector_hamiltonian",
        definition="synthetic XC spin difference",
        source="synthetic",
    )
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
        site_layout=layout,
        operators=PawOperatorComponents(
            components=(operator,), policy="compose", selected_names=("xc",)
        ),
        kpoint_mode="full_bz",
        selected_source_sites=(0,),
        provenance={
            "source_code": "synthetic",
            "source_version": "1.0",
            "functional": "PBE+U",
            "setup_hashes": ["fe-paw-hash", "o-paw-hash"],
            "u_eV": 5.0,
            "j_eV": 0.0,
            "correlated_shells": ["Fe:3d"],
            "input": {"wfk_sha256": "0123456789abcdef"},
        },
    )


def test_paw_snapshot_cache_round_trip_preserves_values_and_provenance(tmp_path):
    pytest.importorskip("netCDF4")
    snapshot = _snapshot()
    filename = tmp_path / "paw-snapshot.nc"

    identity = write_paw_snapshot_netcdf(filename, snapshot)
    cached = read_paw_snapshot_netcdf(filename, expected_identity=identity)

    for name in ("kpoints", "weights", "eigenvalues", "occupations", "coefficients"):
        np.testing.assert_allclose(getattr(cached, name), getattr(snapshot, name))
    assert cached.site_layout == snapshot.site_layout
    assert cached.operators.policy == snapshot.operators.policy
    assert cached.operators.selected_names == snapshot.operators.selected_names
    assert cached.provenance == snapshot.provenance
    assert paw_snapshot_cache_identity(cached) == identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spectral_input_checksum", "different-spectral-input"),
        ("setup_hashes", ["different-setup", "o-paw-hash"]),
        ("selected_source_sites", [1]),
        ("operator_basis", "different-basis"),
        ("operator_policy", "authoritative_total"),
    ],
)
def test_paw_snapshot_cache_rejects_identity_mismatch(tmp_path, field, value):
    pytest.importorskip("netCDF4")
    filename = tmp_path / "paw-snapshot.nc"
    identity = write_paw_snapshot_netcdf(filename, _snapshot())
    expected = dict(identity)
    expected[field] = value

    with pytest.raises(ValueError, match=f"cache identity mismatch.*{field}"):
        read_paw_snapshot_netcdf(filename, expected_identity=expected)


def test_paw_snapshot_cache_rejects_schema_and_payload_checksum_tampering(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    filename = tmp_path / "paw-snapshot.nc"
    identity = write_paw_snapshot_netcdf(filename, _snapshot())

    with netcdf4.Dataset(filename, "a") as nc:
        nc.schema_version = "999.0"
    with pytest.raises(ValueError, match="unsupported PAW snapshot cache schema"):
        read_paw_snapshot_netcdf(filename, expected_identity=identity)

    identity = write_paw_snapshot_netcdf(filename, _snapshot())
    with netcdf4.Dataset(filename, "a") as nc:
        nc.groups["projectors"].variables["coefficients"][0, 0, 0, 0, 0] = 9.0
    with pytest.raises(ValueError, match="payload digest mismatch"):
        read_paw_snapshot_netcdf(filename, expected_identity=identity)


def test_cache_reloaded_projected_state_matches_fresh_exchange_trace(tmp_path):
    pytest.importorskip("netCDF4")
    snapshot = _snapshot()
    filename = tmp_path / "paw-snapshot.nc"
    identity = write_paw_snapshot_netcdf(filename, snapshot)
    cached = read_paw_snapshot_netcdf(filename, expected_identity=identity)

    fresh_trace = projector_exchange_trace(
        ProjectorGreen(build_projector_green_data(snapshot)),
        np.array([[0, 0, 0]]),
        energy=0.25j,
        sites=(0,),
    )
    cached_trace = projector_exchange_trace(
        ProjectorGreen(build_projector_green_data(cached)),
        np.array([[0, 0, 0]]),
        energy=0.25j,
        sites=(0,),
    )

    np.testing.assert_allclose(
        cached_trace["trace"][((0, 0, 0), 0, 0)],
        fresh_trace["trace"][((0, 0, 0), 0, 0)],
    )
