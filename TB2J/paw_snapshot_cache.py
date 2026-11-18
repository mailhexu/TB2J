"""Versioned NetCDF persistence for validated PAW projector snapshots.

The cache stores the source-neutral snapshot rather than derived Green data.  A
cache hit is accepted only after the persisted payload and its identity are
recomputed and validated against the current collector identity.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np

from TB2J.paw_projector import (
    PawOperatorComponent,
    PawOperatorComponents,
    PawProjectorChannel,
    PawProjectorSnapshot,
    PawSiteLayout,
    validate_paw_projector_snapshot,
)

PAW_SNAPSHOT_CACHE_SCHEMA_NAME = "tb2j.paw_projector_snapshot"
PAW_SNAPSHOT_CACHE_SCHEMA_VERSION = "1.0"
_CACHE_IDENTITY_FIELDS = (
    "schema_name",
    "schema_version",
    "spectral_input_checksum",
    "source_code",
    "source_version",
    "site_layout_checksum",
    "setup_hashes",
    "kpoint_mode",
    "kpoint_mesh_checksum",
    "selected_source_sites",
    "operator_basis",
    "operator_policy",
    "selected_operator_components",
    "operator_checksum",
)


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest_arrays(*items: tuple[str, np.ndarray, str]) -> str:
    """Hash labelled arrays after normalizing shape, dtype, and byte order."""
    digest = sha256()
    for name, value, dtype in items:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype(dtype)))
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def _layout_payload(snapshot: PawProjectorSnapshot) -> list[dict]:
    return [
        {
            "source_site": layout.source_site,
            "species": layout.species,
            "atomic_number": layout.atomic_number,
            "projector_slice": [
                layout.projector_slice.start,
                layout.projector_slice.stop,
                layout.projector_slice.step,
            ],
            "setup_hash": layout.setup_hash,
            "channels": [
                {
                    "l": channel.l,
                    "m": channel.m,
                    "radial": channel.radial,
                    "label": channel.label,
                }
                for channel in layout.channels
            ],
        }
        for layout in snapshot.site_layout
    ]


def _operators_payload(snapshot: PawProjectorSnapshot) -> dict:
    return {
        "policy": snapshot.operators.policy,
        "selected_names": list(snapshot.operators.selected_names),
        "components": [
            {
                "name": component.name,
                "units": component.units,
                "basis_id": component.basis_id,
                "definition": component.definition,
                "source": component.source,
                "included_in_total": component.included_in_total,
            }
            for component in snapshot.operators.components
        ],
    }


def _spectral_input_checksum(snapshot: PawProjectorSnapshot) -> str:
    occupations = (
        np.asarray(snapshot.occupations, dtype="<f8")
        if snapshot.occupations is not None
        else np.empty((0,), dtype="<f8")
    )
    return _digest_arrays(
        ("kpoints", snapshot.kpoints, "<f8"),
        ("weights", snapshot.weights, "<f8"),
        ("eigenvalues", snapshot.eigenvalues, "<f8"),
        ("occupations", occupations, "<f8"),
        ("coefficients", snapshot.coefficients, "<c16"),
        ("efermi", np.asarray([snapshot.efermi]), "<f8"),
    )


def _kpoint_mesh_checksum(snapshot: PawProjectorSnapshot) -> str:
    return _digest_arrays(
        ("kpoints", snapshot.kpoints, "<f8"),
        ("weights", snapshot.weights, "<f8"),
    )


def _operator_checksum(snapshot: PawProjectorSnapshot) -> str:
    """Hash selected PAW operator content and declared component semantics."""
    digest = sha256(_canonical_json(_operators_payload(snapshot)).encode("utf-8"))
    digest.update(
        _digest_arrays(
            *(
                (f"operator:{component.name}", component.values, "<c16")
                for component in snapshot.operators.components
            )
        ).encode("ascii")
    )
    return digest.hexdigest()


def paw_snapshot_cache_identity(snapshot: PawProjectorSnapshot) -> dict[str, object]:
    """Return stable identity material required to reuse a snapshot cache.

    Exchange integration settings intentionally do not participate: they act
    after the projected state has been validated and loaded.
    """
    report = validate_paw_projector_snapshot(snapshot)
    layout = _layout_payload(snapshot)
    return {
        "schema_name": PAW_SNAPSHOT_CACHE_SCHEMA_NAME,
        "schema_version": PAW_SNAPSHOT_CACHE_SCHEMA_VERSION,
        "spectral_input_checksum": _spectral_input_checksum(snapshot),
        "source_code": str(snapshot.provenance["source_code"]),
        "source_version": str(snapshot.provenance["source_version"]),
        "site_layout_checksum": sha256(
            _canonical_json(layout).encode("utf-8")
        ).hexdigest(),
        "setup_hashes": [site.setup_hash for site in snapshot.site_layout],
        "kpoint_mode": snapshot.kpoint_mode,
        "kpoint_mesh_checksum": _kpoint_mesh_checksum(snapshot),
        "selected_source_sites": list(snapshot.selected_source_sites),
        "operator_basis": report.operator_basis,
        "operator_policy": snapshot.operators.policy,
        "selected_operator_components": list(report.selected_component_names),
        "operator_checksum": _operator_checksum(snapshot),
    }


def _payload_digest(snapshot: PawProjectorSnapshot) -> str:
    """Hash every persisted scientific field, including provenance and operators."""
    metadata = {
        "layout": _layout_payload(snapshot),
        "operators": _operators_payload(snapshot),
        "provenance": _jsonable(snapshot.provenance),
        "kpoint_mode": snapshot.kpoint_mode,
        "selected_source_sites": list(snapshot.selected_source_sites),
        "efermi": float(snapshot.efermi),
    }
    digest = sha256(_canonical_json(metadata).encode("utf-8"))
    arrays = (
        ("kpoints", snapshot.kpoints, "<f8"),
        ("weights", snapshot.weights, "<f8"),
        ("eigenvalues", snapshot.eigenvalues, "<f8"),
        (
            "occupations",
            snapshot.occupations
            if snapshot.occupations is not None
            else np.empty((0,), dtype=float),
            "<f8",
        ),
        ("coefficients", snapshot.coefficients, "<c16"),
        ("cell", snapshot.cell, "<f8"),
        ("positions", snapshot.positions, "<f8"),
        ("atomic_numbers", snapshot.atomic_numbers, "<i8"),
    )
    for component in snapshot.operators.components:
        arrays += ((f"operator:{component.name}", component.values, "<c16"),)
    digest.update(_digest_arrays(*arrays).encode("ascii"))
    return digest.hexdigest()


def _require_netcdf4():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError("netCDF4 is required for PAW snapshot cache I/O") from exc
    return Dataset


def _require_identity(identity: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise TypeError(
            "expected_identity must be a PAW snapshot cache identity mapping"
        )
    missing = [field for field in _CACHE_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(
            "expected cache identity is missing required field(s): "
            + ", ".join(missing)
        )
    return {field: _jsonable(identity[field]) for field in _CACHE_IDENTITY_FIELDS}


def _assert_identity_matches(
    actual: Mapping[str, object], expected: Mapping[str, object], *, subject: str
) -> None:
    expected = _require_identity(expected)
    for field in _CACHE_IDENTITY_FIELDS:
        if _jsonable(actual[field]) != expected[field]:
            raise ValueError(f"{subject} identity mismatch for {field}")


def _load_json_attribute(nc, name: str):
    raw = getattr(nc, name, None)
    if not isinstance(raw, str):
        raise ValueError(f"PAW snapshot cache is missing {name}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"PAW snapshot cache has invalid {name}") from exc


def write_paw_snapshot_netcdf(
    filename: str | Path, snapshot: PawProjectorSnapshot
) -> dict[str, object]:
    """Validate and serialize a PAW snapshot as a versioned NetCDF cache."""
    validate_paw_projector_snapshot(snapshot)
    Dataset = _require_netcdf4()
    identity = paw_snapshot_cache_identity(snapshot)

    with Dataset(Path(filename), "w") as nc:
        nspin, nkpt, nband, nproj = snapshot.coefficients.shape
        nc.createDimension("nspin", nspin)
        nc.createDimension("nkpt", nkpt)
        nc.createDimension("nband", nband)
        nc.createDimension("nproj", nproj)
        nc.createDimension("nsite", len(snapshot.site_layout))
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.schema_name = PAW_SNAPSHOT_CACHE_SCHEMA_NAME
        nc.schema_version = PAW_SNAPSHOT_CACHE_SCHEMA_VERSION
        nc.cache_identity_json = _canonical_json(identity)
        nc.payload_digest = _payload_digest(snapshot)
        nc.site_layout_json = _canonical_json(_layout_payload(snapshot))
        nc.operators_json = _canonical_json(_operators_payload(snapshot))
        nc.provenance_json = _canonical_json(snapshot.provenance)
        nc.kpoint_mode = snapshot.kpoint_mode
        nc.selected_source_sites_json = _canonical_json(
            list(snapshot.selected_source_sites)
        )

        structure = nc.createGroup("structure")
        structure.createVariable("cell", "f8", ("three", "three"))[:] = np.array(
            snapshot.cell, copy=True
        )
        structure.createVariable("positions", "f8", ("nsite", "three"))[:] = np.array(
            snapshot.positions, copy=True
        )
        structure.createVariable("atomic_numbers", "i8", ("nsite",))[:] = np.array(
            snapshot.atomic_numbers, copy=True
        )

        kpoints = nc.createGroup("kpoints")
        kpoints.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = np.array(
            snapshot.kpoints, copy=True
        )
        kpoints.createVariable("weights", "f8", ("nkpt",))[:] = np.array(
            snapshot.weights, copy=True
        )

        bands = nc.createGroup("bands")
        bands.efermi = snapshot.efermi
        bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = (
            np.array(snapshot.eigenvalues, copy=True)
        )
        if snapshot.occupations is not None:
            bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = (
                np.array(snapshot.occupations, copy=True)
            )

        projectors = nc.createGroup("projectors")
        projectors.createVariable(
            "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
        )[:] = np.stack(
            (snapshot.coefficients.real, snapshot.coefficients.imag), axis=-1
        ).copy()

        operator_group = nc.createGroup("operators")
        nmax = max(
            site.projector_slice.stop - site.projector_slice.start
            for site in snapshot.site_layout
        )
        operator_group.createDimension("nproj_site_max", nmax)
        for component in snapshot.operators.components:
            operator_group.createVariable(
                component.name,
                "f8",
                ("nsite", "nproj_site_max", "nproj_site_max", "complex"),
            )[:] = np.stack(
                (component.values.real, component.values.imag), axis=-1
            ).copy()

    return identity


def _read_layout(payload: object) -> tuple[PawSiteLayout, ...]:
    if not isinstance(payload, list):
        raise ValueError("PAW snapshot cache site_layout_json must be a list")
    try:
        return tuple(
            PawSiteLayout(
                source_site=int(site["source_site"]),
                species=str(site["species"]),
                atomic_number=int(site["atomic_number"]),
                projector_slice=slice(*site["projector_slice"]),
                channels=tuple(
                    PawProjectorChannel(
                        l=int(channel["l"]),
                        m=int(channel["m"]),
                        radial=int(channel["radial"]),
                        label=str(channel["label"]),
                    )
                    for channel in site["channels"]
                ),
                setup_hash=str(site["setup_hash"]),
            )
            for site in payload
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PAW snapshot cache has invalid site layout") from exc


def _read_operators(payload: object, group) -> PawOperatorComponents:
    if not isinstance(payload, dict) or group is None:
        raise ValueError("PAW snapshot cache has invalid operator components")
    try:
        components = tuple(
            PawOperatorComponent(
                name=component["name"],
                values=np.asarray(group.variables[component["name"]][:, :, :, 0])
                + 1j * np.asarray(group.variables[component["name"]][:, :, :, 1]),
                units=str(component["units"]),
                basis_id=str(component["basis_id"]),
                definition=str(component["definition"]),
                source=str(component["source"]),
                included_in_total=bool(component["included_in_total"]),
            )
            for component in payload["components"]
        )
        return PawOperatorComponents(
            components=components,
            policy=payload["policy"],
            selected_names=tuple(payload["selected_names"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PAW snapshot cache has invalid operator components") from exc


def read_paw_snapshot_netcdf(
    filename: str | Path, *, expected_identity: Mapping[str, object]
) -> PawProjectorSnapshot:
    """Read, verify, and revalidate a PAW snapshot cache before Green construction."""
    Dataset = _require_netcdf4()
    with Dataset(Path(filename)) as nc:
        if (
            getattr(nc, "schema_name", None) != PAW_SNAPSHOT_CACHE_SCHEMA_NAME
            or getattr(nc, "schema_version", None) != PAW_SNAPSHOT_CACHE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported PAW snapshot cache schema")
        stored_identity = _load_json_attribute(nc, "cache_identity_json")
        stored_digest = getattr(nc, "payload_digest", None)
        if not isinstance(stored_digest, str):
            raise ValueError("PAW snapshot cache is missing payload_digest")
        layout = _read_layout(_load_json_attribute(nc, "site_layout_json"))
        operators = _read_operators(
            _load_json_attribute(nc, "operators_json"), nc.groups.get("operators")
        )
        bands = nc.groups.get("bands")
        if bands is None:
            raise ValueError("PAW snapshot cache is missing bands group")
        occupations = (
            bands.variables["occupations"][:]
            if "occupations" in bands.variables
            else None
        )
        projectors = nc.groups.get("projectors")
        if projectors is None:
            raise ValueError("PAW snapshot cache is missing projectors group")
        coefficient_data = projectors.variables["coefficients"][:]
        snapshot = PawProjectorSnapshot(
            kpoints=nc.groups["kpoints"].variables["kpoints"][:],
            weights=nc.groups["kpoints"].variables["weights"][:],
            eigenvalues=bands.variables["eigenvalues"][:],
            occupations=occupations,
            coefficients=coefficient_data[..., 0] + 1j * coefficient_data[..., 1],
            efermi=float(getattr(bands, "efermi")),
            cell=nc.groups["structure"].variables["cell"][:],
            positions=nc.groups["structure"].variables["positions"][:],
            atomic_numbers=nc.groups["structure"].variables["atomic_numbers"][:],
            site_layout=layout,
            operators=operators,
            kpoint_mode=str(getattr(nc, "kpoint_mode", "")),
            selected_source_sites=tuple(
                _load_json_attribute(nc, "selected_source_sites_json")
            ),
            provenance=_load_json_attribute(nc, "provenance_json"),
        )

    validate_paw_projector_snapshot(snapshot)
    if _payload_digest(snapshot) != stored_digest:
        raise ValueError("PAW snapshot cache payload digest mismatch")
    identity = paw_snapshot_cache_identity(snapshot)
    _assert_identity_matches(identity, stored_identity, subject="PAW snapshot cache")
    _assert_identity_matches(identity, expected_identity, subject="cache")
    return snapshot
