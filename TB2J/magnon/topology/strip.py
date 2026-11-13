"""Physical open-strip magnon spectra and spin-amplitude edge localization.

The strip is assembled from the canonical real-space tensors of a
:class:`TopologyModel`. Only bonds whose target lies beyond the finite width
are cut; edge and fixed-normal translations remain Bloch-periodic. Stationarity
is evaluated on those retained physical bonds before any bosonic solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from TB2J.magnon.magnon_math import tb2j_to_physical

from TB2J.magnon.topology.assembly import (
    classical_transverse_gradient,
    local_frame_bosonic_hamiltonian,
)
from TB2J.magnon.topology.bosonic import ModeSolverError, solve_bosonic_modes
from TB2J.magnon.topology.embedding import LatticeEmbedding, identity_embedding
from TB2J.magnon.topology.preparation import TopologyModel

__all__ = [
    "StripRequest",
    "StripSpectrumResult",
    "calculate_strip",
    "strip_site_map",
]

SCHEMA_VERSION = "tb2j-topology.strip/1"


@dataclass(frozen=True)
class StripRequest:
    """Request a finite-width strip, periodic along edge and fixed normal.

    ``width`` counts embedding-basis width translations. ``termination`` gives
    the fractional left/right cut offsets ``(left, right)``; the physical cut
    is ``left <= x_width < width + right``. ``reference_magmoms``, when given,
    has one Cartesian moment per entry in :attr:`StripSpectrumResult.site_map`
    (sorted by width coordinate, then source spin index). Moment lengths are
    fixed by the parent model; only an explicit direction correction is
    accepted. ``kpoints`` samples the endpoint-exclusive fractional edge BZ.
    """

    width: int
    kpoints: int = 81
    embedding: Optional[LatticeEmbedding] = None
    termination: Tuple[float, float] = (0.0, 0.0)
    edge_layers: int = 1
    reference_magmoms: Optional[np.ndarray] = None
    stationarity_rtol: float = 1e-6
    stationarity_atol: float = 1e-12
    degeneracy_tol: float = 1e-8
    interior_tol: float = 1e-6

    def __post_init__(self):
        for name in ("width", "kpoints", "edge_layers"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{name} must be an integer.")
        width, kpoints, edge_layers = (
            int(self.width),
            int(self.kpoints),
            int(self.edge_layers),
        )
        if width < 1:
            raise ValueError("width must be at least one embedding width cell.")
        if kpoints < 2:
            raise ValueError("kpoints must be at least 2 for a periodic edge grid.")
        if edge_layers < 1:
            raise ValueError("edge_layers must be a positive integer.")
        if len(self.termination) != 2:
            raise ValueError("termination must contain (left_offset, right_offset).")
        left, right = (float(self.termination[0]), float(self.termination[1]))
        if (
            not np.isfinite([left, right]).all()
            or not (0.0 <= left < 1.0)
            or not (0.0 <= right < 1.0)
        ):
            raise ValueError("termination offsets must be finite fractions in [0, 1).")
        physical_width = width + right - left
        if 2 * edge_layers > physical_width:
            raise ValueError(
                "left and right edge regions overlap: reduce edge_layers "
                "or increase width."
            )
        if self.embedding is not None and not isinstance(
            self.embedding, LatticeEmbedding
        ):
            raise TypeError("embedding must be a LatticeEmbedding.")
        for name in (
            "stationarity_rtol",
            "stationarity_atol",
            "degeneracy_tol",
            "interior_tol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.reference_magmoms is not None:
            moments = np.array(self.reference_magmoms, dtype=float, copy=True)
            if (
                moments.ndim != 2
                or moments.shape[1] != 3
                or not np.isfinite(moments).all()
            ):
                raise ValueError("reference_magmoms must be a finite (nsite, 3) array.")
            moments.setflags(write=False)
            object.__setattr__(self, "reference_magmoms", moments)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "edge_layers", edge_layers)
        object.__setattr__(self, "termination", (left, right))


@dataclass
class StripSpectrumResult:
    """Open-strip energies, physical edge localization and validity evidence."""

    request: StripRequest
    model: TopologyModel
    embedding: LatticeEmbedding
    edge_kpoints: np.ndarray
    energies: Optional[np.ndarray]
    left_edge_fraction: Optional[np.ndarray]
    right_edge_fraction: Optional[np.ndarray]
    interior_fraction: Optional[np.ndarray]
    physical_amplitude_norm: Optional[np.ndarray]
    degenerate_subspaces: List[Dict[str, Any]]
    site_map: List[Dict[str, Any]]
    site_width_coordinates: np.ndarray
    reference_magmoms: np.ndarray
    local_field: np.ndarray
    edge_regions: Dict[str, Any]
    valid: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        """Return versioned strip JSON, including enough parent data for matching."""
        return {
            "kind": "strip",
            "schema_version": SCHEMA_VERSION,
            "model_digest": self.model.digest,
            "parent_model_digest": self.model.digest,
            "canonicalization_version": self.model.canonicalization_version,
            "provenance": self.model.provenance(),
            "model_snapshot": {
                "cell_angstrom": self.model.cell.tolist(),
                "positions_angstrom": self.model.positions.tolist(),
                "positions_fractional": self.model.positions_fractional.tolist(),
                "index_spin": self.model.index_spin.tolist(),
                "spin_atoms": self.model.spin_atoms.tolist(),
                "reference_magmoms_muB": self.model.magmoms.tolist(),
                "Rlist": self.model.Rlist.tolist(),
                "JR_eV": self.model.JR.tolist(),
            },
            "request": {
                "width_cells": self.request.width,
                "edge_kpoints": self.request.kpoints,
                "termination_offsets_width_cells": list(self.request.termination),
                "edge_layers": self.request.edge_layers,
                "stationarity_rtol": self.request.stationarity_rtol,
                "stationarity_atol_eV_muB": self.request.stationarity_atol,
                "degeneracy_tolerance_eV": self.request.degeneracy_tol,
                "interior_reference_tolerance_muB": self.request.interior_tol,
            },
            "embedding": {
                "basis_input_cell": self.embedding.basis.tolist(),
                "reciprocal_dual": self.embedding.reciprocal_dual.tolist(),
                "normal_k_fractional": float(self.embedding.normal_k),
                "orientation_sign": self.embedding.orientation_sign,
            },
            "geometry": {
                "width_cells": self.request.width,
                "cut_bounds_width_fractional": [
                    float(self.request.termination[0]),
                    float(self.request.width + self.request.termination[1]),
                ],
                "edge_translation_input_cell": self.embedding.basis[0].tolist(),
                "width_translation_input_cell": self.embedding.basis[1].tolist(),
                "normal_translation_input_cell": self.embedding.basis[2].tolist(),
                "site_count": len(self.site_map),
                "site_map": self.site_map,
                "edge_regions": self.edge_regions,
                "retained_directed_bond_records": int(
                    self.diagnostics.get("retained_directed_bond_records", 0)
                ),
                "cut_directed_bond_records": int(
                    self.diagnostics.get("cut_directed_bond_records", 0)
                ),
                "retained_translations_input_cell": self.diagnostics.get(
                    "retained_translations_input_cell", []
                ),
            },
            "reference": {
                "source": self.diagnostics.get("reference_source"),
                "site_magmoms_muB": self.reference_magmoms.tolist(),
                "site_width_coordinates": _json_array(self.site_width_coordinates),
                "local_field_eV_muB": _json_array(self.local_field),
                "stationarity": self.diagnostics.get("stationarity"),
                "interior_reference_mismatch": self.diagnostics.get(
                    "interior_reference_mismatch"
                ),
            },
            "edge_k_fractional": _json_array(self.edge_kpoints),
            "energies_eV": _json_array(self.energies),
            "left_edge_fraction": _json_array(self.left_edge_fraction),
            "right_edge_fraction": _json_array(self.right_edge_fraction),
            "interior_fraction": _json_array(self.interior_fraction),
            "physical_amplitude_norm": _json_array(self.physical_amplitude_norm),
            "degenerate_subspaces": self.degenerate_subspaces,
            "array_axes": {
                "edge_k_fractional": ["edge_k_index"],
                "energies_eV": ["edge_k_index", "strip_mode_index"],
                "left_edge_fraction": ["edge_k_index", "strip_mode_index"],
                "right_edge_fraction": ["edge_k_index", "strip_mode_index"],
                "interior_fraction": ["edge_k_index", "strip_mode_index"],
                "physical_amplitude_norm": [
                    "edge_k_index",
                    "strip_mode_index",
                ],
                "reference.site_magmoms_muB": [
                    "strip_site_index",
                    "cartesian_component",
                ],
                "reference.local_field_eV_muB": [
                    "strip_site_index",
                    "cartesian_component",
                ],
            },
            "valid": bool(self.valid),
            "diagnostics": self.diagnostics,
            "conventions": {
                "fourier_phase": "J(k) = sum_R J(R) exp(-2 pi i k.R)",
                "energy_unit": "eV",
                "edge_k_unit": (
                    "fractional reciprocal cycles of the embedding edge vector"
                ),
                "localization_observable": (
                    "sum_i (|mu_i|/2) (|u_i|^2 + |v_i|^2), the sum of squared "
                    "physical transverse spin-amplitude matrix elements"
                ),
                "localization_normalization": (
                    "each resolved mode is divided by its total physical "
                    "spin-amplitude norm"
                ),
                "degeneracy_policy": (
                    "unresolved groups report basis-invariant projected "
                    "edge-fraction spectra; per-mode fractions are null"
                ),
                "certificate": "docs/sympy/magnon_topology_strip_localization.py",
            },
        }


def _json_array(values):
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim == 0:
        value = float(array)
        return value if np.isfinite(value) else None
    return [_json_array(row) for row in array]


def _validate_model(model: TopologyModel):
    """Validate the prepared snapshot, including direct synthetic models."""
    cell = np.asarray(model.cell, dtype=float)
    positions = np.asarray(model.positions, dtype=float)
    fractional = np.asarray(model.positions_fractional, dtype=float)
    moments = np.asarray(model.magmoms, dtype=float)
    rlist = np.asarray(model.Rlist, dtype=float)
    jr = np.asarray(model.JR, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("TopologyModel.cell must be a finite 3x3 matrix.")
    if abs(float(np.linalg.det(cell))) <= 1e-12:
        raise ValueError("TopologyModel.cell is singular.")
    if moments.ndim != 2 or moments.shape[1] != 3:
        raise ValueError("TopologyModel.magmoms must have shape (nspin, 3).")
    nspin = moments.shape[0]
    if np.any(np.linalg.norm(moments, axis=1) <= 0):
        raise ValueError("TopologyModel.magmoms must be nonzero for every spin.")
    if positions.shape != (nspin, 3) or fractional.shape != (nspin, 3):
        raise ValueError(
            "TopologyModel positions and fractional positions must have "
            "shape (nspin, 3)."
        )
    if rlist.ndim != 2 or rlist.shape[1] != 3:
        raise ValueError("TopologyModel.Rlist must have shape (nR, 3).")
    if not np.array_equal(rlist, np.rint(rlist)):
        raise ValueError("TopologyModel.Rlist translations must be integers.")
    if jr.shape != (len(rlist), nspin, nspin, 3, 3):
        raise ValueError("TopologyModel.JR must have shape (nR, nspin, nspin, 3, 3).")
    if not all(
        np.isfinite(array).all()
        for array in (positions, fractional, moments, rlist, jr)
    ):
        raise ValueError("TopologyModel contains non-finite geometry or interactions.")
    if not np.allclose(fractional @ cell, positions, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "TopologyModel fractional positions do not map to its Cartesian positions."
        )
    index_spin = np.asarray(model.index_spin, dtype=float)
    spin_atoms = np.asarray(model.spin_atoms, dtype=float)
    if (
        index_spin.ndim != 1
        or spin_atoms.shape != (nspin,)
        or not np.isfinite(index_spin).all()
        or not np.isfinite(spin_atoms).all()
        or not np.array_equal(index_spin, np.rint(index_spin))
        or not np.array_equal(spin_atoms, np.rint(spin_atoms))
    ):
        raise ValueError("TopologyModel atom/spin mappings are malformed.")
    index_spin = index_spin.astype(int)
    spin_atoms = spin_atoms.astype(int)
    if (
        np.any(index_spin < -1)
        or np.any(spin_atoms < 0)
        or np.any(spin_atoms >= len(index_spin))
        or not np.array_equal(index_spin[spin_atoms], np.arange(nspin))
        or not np.array_equal(np.sort(index_spin[index_spin >= 0]), np.arange(nspin))
    ):
        raise ValueError("TopologyModel atom/spin mappings are inconsistent.")
    if len({tuple(row) for row in rlist.astype(int)}) != len(rlist):
        raise ValueError("TopologyModel.Rlist contains duplicate translations.")
    if not np.any(np.all(rlist == 0, axis=1)):
        raise ValueError("TopologyModel.Rlist must include the origin translation.")
    r_lookup = {tuple(row.astype(int)): index for index, row in enumerate(rlist)}
    for index, translation in enumerate(rlist.astype(int)):
        reverse = r_lookup.get(tuple(-translation))
        if reverse is None or not np.allclose(
            jr[index],
            jr[reverse].transpose(1, 0, 3, 2),
            rtol=1e-6,
            atol=1e-12,
        ):
            raise ValueError(
                "TopologyModel.JR violates pair reversal at translation "
                f"{translation.tolist()}."
            )
    if not isinstance(model.digest, str) or not model.digest:
        raise ValueError("TopologyModel must carry a nonempty content digest.")


def _embedding_positions(model: TopologyModel, embedding: LatticeEmbedding):
    inv_basis = np.linalg.inv(np.asarray(embedding.basis, dtype=float))
    positions = np.asarray(model.positions_fractional, dtype=float) @ inv_basis
    nearest = np.rint(positions)
    positions = np.where(np.abs(positions - nearest) <= 1e-12, nearest, positions)
    offsets = np.floor(positions).astype(np.int64)
    fractional = positions - offsets
    return offsets, fractional


def _site_geometry(
    model: TopologyModel,
    width: int,
    embedding: LatticeEmbedding,
    termination: Tuple[float, float],
    *,
    periodic_width: bool = False,
):
    offsets, fractional = _embedding_positions(model, embedding)
    left, right = termination
    upper = width + right
    records = []
    layers_by_spin: List[List[int]] = [[] for _ in range(model.nspin)]
    for spin in range(model.nspin):
        f_width = float(fractional[spin, 1])
        if periodic_width:
            layers = range(width)
        else:
            first = ceil(left - f_width - 1e-12)
            stop = ceil(upper - f_width - 1e-12)
            layers = range(first, stop)
        for layer in layers:
            coordinate = layer + f_width
            if not periodic_width and not (left <= coordinate < upper):
                continue
            layers_by_spin[spin].append(int(layer))
            site_embedding = fractional[spin].copy()
            site_embedding[1] += layer
            site_input = site_embedding @ embedding.basis
            cartesian = site_input @ model.cell
            source_translation = (
                np.eye(3, dtype=np.int64)[1] * layer - offsets[spin]
            ) @ embedding.basis
            records.append(
                {
                    "source_spin": int(spin),
                    "width_cell": int(layer),
                    "width_coordinate": float(coordinate),
                    "input_fractional": site_input.tolist(),
                    "cartesian_angstrom": cartesian.tolist(),
                    "source_cell_translation_input": source_translation.astype(
                        int
                    ).tolist(),
                }
            )
    records.sort(key=lambda site: (site["width_coordinate"], site["source_spin"]))
    lookup = {}
    for index, site in enumerate(records):
        site["strip_site"] = index
        lookup[(site["width_cell"], site["source_spin"])] = index
    if not records:
        raise ValueError("the requested physical cut contains no magnetic sites.")
    return offsets, fractional, records, lookup, layers_by_spin


def strip_site_map(
    model: TopologyModel,
    width: int,
    embedding: Optional[LatticeEmbedding] = None,
    termination: Tuple[float, float] = (0.0, 0.0),
) -> List[Dict[str, Any]]:
    """Enumerate physical strip sites in the ordering used by corrected references."""
    if not isinstance(model, TopologyModel):
        raise TypeError("strip_site_map requires a topology-ready TopologyModel.")
    _validate_model(model)
    if isinstance(width, (bool, np.bool_)) or int(width) != width or int(width) < 1:
        raise ValueError("width must be a positive integer.")
    embedding = embedding or identity_embedding()
    if len(termination) != 2 or not np.isfinite(termination).all():
        raise ValueError("termination must contain two finite cut offsets.")
    left, right = (float(termination[0]), float(termination[1]))
    if not (0 <= left < 1 and 0 <= right < 1):
        raise ValueError("termination offsets must be in [0, 1).")
    _, _, records, _, _ = _site_geometry(model, int(width), embedding, (left, right))
    return records


def _assemble_strip_bonds(
    model: TopologyModel,
    width: int,
    embedding: LatticeEmbedding,
    termination: Tuple[float, float],
    *,
    periodic_width: bool = False,
):
    """Build finite-site tensors; periodic_width supports the restoration regression."""
    offsets, _fractional, sites, site_lookup, layers_by_spin = _site_geometry(
        model, width, embedding, termination, periodic_width=periodic_width
    )
    basis = np.asarray(embedding.basis, dtype=np.int64)
    inverse = np.linalg.inv(basis.astype(float))
    r_embed_float = np.asarray(model.Rlist, dtype=float) @ inverse
    r_embed = np.rint(r_embed_float).astype(np.int64)
    if not np.allclose(r_embed_float, r_embed, rtol=0.0, atol=1e-10):
        raise ValueError(
            "the embedding does not map every parent interaction translation "
            "to an integer strip translation."
        )

    bond_map: Dict[Tuple[Tuple[int, int, int], int, int], np.ndarray] = {}
    retained = 0
    cut = 0
    for ir, displacement in enumerate(r_embed):
        for source_spin in range(model.nspin):
            for target_spin in range(model.nspin):
                tensor = model.JR[ir, source_spin, target_spin]
                if not np.any(tensor):
                    continue
                delta = displacement + offsets[target_spin] - offsets[source_spin]
                for source_layer in layers_by_spin[source_spin]:
                    target_cell = source_layer + int(delta[1])
                    width_wrap = 0
                    if periodic_width:
                        target_layer = target_cell % width
                        width_wrap = (target_cell - target_layer) // width
                    else:
                        target_layer = target_cell
                    target_site = site_lookup.get((target_layer, target_spin))
                    if target_site is None:
                        cut += 1
                        continue
                    source_site = site_lookup[(source_layer, source_spin)]
                    translation = (
                        int(delta[0]) * basis[0]
                        + int(delta[2]) * basis[2]
                        + int(width_wrap * width) * basis[1]
                    )
                    key = (tuple(int(x) for x in translation), source_site, target_site)
                    if key not in bond_map:
                        bond_map[key] = np.zeros((3, 3), dtype=float)
                    bond_map[key] += tensor
                    retained += 1
    translation_keys = sorted({key[0] for key in bond_map} | {(0, 0, 0)})
    r_index = {translation: index for index, translation in enumerate(translation_keys)}
    nsite = len(sites)
    jr_strip = np.zeros((len(translation_keys), nsite, nsite, 3, 3), dtype=float)
    for (translation, i, j), tensor in bond_map.items():
        jr_strip[r_index[translation], i, j] += tensor
    rlist_strip = np.asarray(translation_keys, dtype=np.int64)
    return jr_strip, rlist_strip, sites, retained, cut


def _local_fields(JR: np.ndarray, Rlist: np.ndarray, moments: np.ndarray):
    spin = np.linalg.norm(moments, axis=1) / 2.0
    exchange = JR / (spin[None, :, None, None, None] * spin[None, None, :, None, None])
    fields = np.zeros_like(moments, dtype=float)
    for ir in range(len(Rlist)):
        fields += np.einsum("ijxy,jy->ix", exchange[ir], moments)
    return fields


def _interior_mismatch(
    model: TopologyModel,
    sites: Sequence[Dict[str, Any]],
    reference_magmoms: np.ndarray,
    interior_mask: np.ndarray,
    tolerance: float,
):
    indices = np.flatnonzero(interior_mask)
    if not len(indices):
        return {
            "checked_sites": 0,
            "max_vector_mismatch_muB": None,
            "rms_vector_mismatch_muB": None,
            "tolerance_muB": float(tolerance),
            "matches_bulk_reference": False,
            "reason": "no sites lie outside the declared left/right edge regions",
        }
    source = np.asarray([sites[i]["source_spin"] for i in indices], dtype=int)
    delta = reference_magmoms[indices] - model.magmoms[source]
    mismatch = np.linalg.norm(delta, axis=1)
    maximum = float(np.max(mismatch))
    return {
        "checked_sites": int(len(indices)),
        "max_vector_mismatch_muB": maximum,
        "rms_vector_mismatch_muB": float(np.sqrt(np.mean(mismatch**2))),
        "tolerance_muB": float(tolerance),
        "matches_bulk_reference": bool(maximum <= tolerance),
    }


def _projected_fraction_spectrum(
    vectors: np.ndarray,
    spin_lengths: np.ndarray,
    region: np.ndarray,
):
    """Generalized projected-observable spectrum, invariant under basis mixing."""
    physical_weight = np.concatenate([spin_lengths, spin_lengths])
    region_weight = np.concatenate([spin_lengths * region, spin_lengths * region])
    total = vectors.conj().T @ (physical_weight[:, None] * vectors)
    observable = vectors.conj().T @ (region_weight[:, None] * vectors)
    total = (total + total.conj().T) / 2.0
    observable = (observable + observable.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(total)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if eigenvalues[0] <= np.finfo(float).eps * scale:
        return None
    inverse_root = (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conj().T
    normalized = inverse_root @ observable @ inverse_root
    fractions = np.linalg.eigvalsh((normalized + normalized.conj().T) / 2.0)
    if fractions[0] < -1e-8 or fractions[-1] > 1.0 + 1e-8:
        return None
    return np.clip(fractions, 0.0, 1.0)


def _mode_localization(
    modes,
    energies: np.ndarray,
    spin_lengths: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    interior_mask: np.ndarray,
    degeneracy_tol: float,
    k_index: int,
):
    nmode = len(energies)
    left = np.full(nmode, np.nan)
    right = np.full(nmode, np.nan)
    interior = np.full(nmode, np.nan)
    amplitude_norm = np.full(nmode, np.nan)
    groups = []
    if modes.vectors is None or modes.positive_indices is None:
        return left, right, interior, amplitude_norm, groups

    band_to_column = {
        int(band): column for column, band in enumerate(modes.positive_indices)
    }
    positive_bands = sorted(band_to_column)
    scale = max(float(np.max(np.abs(energies))), 1.0)
    threshold = degeneracy_tol * scale
    degenerate_sets: List[List[int]] = []
    current: List[int] = []
    for band in positive_bands:
        if current and energies[band] - energies[current[-1]] > threshold:
            degenerate_sets.append(current)
            current = []
        current.append(band)
    if current:
        degenerate_sets.append(current)

    for bands in degenerate_sets:
        if len(bands) > 1:
            columns = [band_to_column[band] for band in bands]
            subspace = modes.vectors[:, columns]
            left_spectrum = _projected_fraction_spectrum(
                subspace, spin_lengths, left_mask.astype(float)
            )
            right_spectrum = _projected_fraction_spectrum(
                subspace, spin_lengths, right_mask.astype(float)
            )
            interior_spectrum = _projected_fraction_spectrum(
                subspace, spin_lengths, interior_mask.astype(float)
            )
            resolved = all(
                values is not None
                for values in (left_spectrum, right_spectrum, interior_spectrum)
            )
            groups.append(
                {
                    "k_index": int(k_index),
                    "mode_indices": [int(band) for band in bands],
                    "energies_eV": [float(energies[band]) for band in bands],
                    "resolution": (
                        "basis_invariant_projected_observables"
                        if resolved
                        else "ambiguous_physical_amplitude_projection"
                    ),
                    "left_fraction_spectrum": (
                        left_spectrum.tolist() if left_spectrum is not None else None
                    ),
                    "right_fraction_spectrum": (
                        right_spectrum.tolist() if right_spectrum is not None else None
                    ),
                    "interior_fraction_spectrum": (
                        interior_spectrum.tolist()
                        if interior_spectrum is not None
                        else None
                    ),
                }
            )
            continue

        band = bands[0]
        vector = modes.vectors[:, band_to_column[band]]
        u = vector[: len(spin_lengths)]
        v = vector[len(spin_lengths) :]
        site_amplitude = spin_lengths * (np.abs(u) ** 2 + np.abs(v) ** 2)
        total_amplitude = float(np.sum(site_amplitude))
        if not np.isfinite(total_amplitude) or total_amplitude <= 0.0:
            continue
        amplitude_norm[band] = total_amplitude
        left[band] = float(np.sum(site_amplitude[left_mask]) / total_amplitude)
        right[band] = float(np.sum(site_amplitude[right_mask]) / total_amplitude)
        interior[band] = float(np.sum(site_amplitude[interior_mask]) / total_amplitude)
    return left, right, interior, amplitude_norm, groups


def calculate_strip(model: TopologyModel, request: StripRequest) -> StripSpectrumResult:
    """Calculate an open strip from retained real-space interactions.

    Boundary sites receive only bonds that remain inside the physical cut;
    local fields, constrained stationarity and the bosonic Hessian are all
    recomputed from that cut model. A nonstationary reference is returned as
    invalid with no mode solve; no edge relaxation is performed automatically.
    """
    if not isinstance(model, TopologyModel):
        raise TypeError(
            "calculate_strip requires a topology-ready TopologyModel from "
            "prepare_topology_model; a bare Magnon is not assumed topology-ready."
        )
    _validate_model(model)
    if not isinstance(request, StripRequest):
        raise TypeError("request must be a StripRequest.")
    embedding = request.embedding or identity_embedding()
    JR, Rlist, sites, retained_count, cut_count = _assemble_strip_bonds(
        model, request.width, embedding, request.termination
    )
    nsite = len(sites)
    source_spins = np.asarray([site["source_spin"] for site in sites], dtype=int)
    reference = (
        model.magmoms[source_spins].copy()
        if request.reference_magmoms is None
        else np.asarray(request.reference_magmoms, dtype=float).copy()
    )
    if reference.shape != (nsite, 3):
        raise ValueError(
            "reference_magmoms must have shape "
            f"({nsite}, 3) in strip_site_map order; got {reference.shape}."
        )
    if not np.isfinite(reference).all() or np.any(
        np.linalg.norm(reference, axis=1) <= 0
    ):
        raise ValueError("reference_magmoms must be finite nonzero vectors.")
    base_lengths = np.linalg.norm(model.magmoms[source_spins], axis=1)
    reference_lengths = np.linalg.norm(reference, axis=1)
    if not np.allclose(reference_lengths, base_lengths, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "reference_magmoms may correct local directions but must preserve "
            "the parent moment length at each mapped site."
        )

    coordinates = np.asarray([site["width_coordinate"] for site in sites], dtype=float)
    left_cut, right_offset = request.termination
    right_cut = request.width + right_offset
    left_mask = coordinates < left_cut + request.edge_layers
    right_mask = coordinates >= right_cut - request.edge_layers
    if not np.any(left_mask) or not np.any(right_mask):
        raise ValueError(
            "the requested edge region contains no magnetic sites; increase "
            "edge_layers or choose a different physical termination."
        )
    if np.any(left_mask & right_mask):
        raise ValueError("the requested left and right edge regions overlap.")
    interior_mask = ~(left_mask | right_mask)
    site_map = [dict(site) for site in sites]
    for site, is_left, is_right, is_interior in zip(
        site_map, left_mask, right_mask, interior_mask
    ):
        site["region"] = (
            "left_edge" if is_left else "right_edge" if is_right else "interior"
        )
    local_field = _local_fields(JR, Rlist, reference)
    stationarity = classical_transverse_gradient(
        JR,
        Rlist,
        reference,
        rtol=request.stationarity_rtol,
        atol=request.stationarity_atol,
    )
    mismatch = _interior_mismatch(
        model, sites, reference, interior_mask, request.interior_tol
    )
    edge_regions = {
        "coordinate": "fractional coordinate along embedding width translation",
        "left": {
            "lower": float(left_cut),
            "upper": float(left_cut + request.edge_layers),
            "site_indices": np.flatnonzero(left_mask).astype(int).tolist(),
        },
        "right": {
            "lower": float(right_cut - request.edge_layers),
            "upper": float(right_cut),
            "site_indices": np.flatnonzero(right_mask).astype(int).tolist(),
        },
        "interior_site_indices": np.flatnonzero(interior_mask).astype(int).tolist(),
    }
    edge_k = np.arange(request.kpoints, dtype=float) / request.kpoints
    diagnostics: Dict[str, Any] = {
        "stationarity": stationarity.as_dict(),
        "transverse_field_eV_muB": stationarity.transverse_field.tolist(),
        "local_field_eV_muB": local_field.tolist(),
        "reference_source": (
            "explicit_strip_reference"
            if request.reference_magmoms is not None
            else "parent_model_reference"
        ),
        "interior_reference_mismatch": mismatch,
        "thresholds": {
            "stationarity_rtol": float(request.stationarity_rtol),
            "stationarity_atol_eV_muB": float(request.stationarity_atol),
            "interior_reference_tolerance_muB": float(request.interior_tol),
            "degeneracy_tolerance_eV": float(request.degeneracy_tol),
            "solver_zero_tolerance_eV": 1e-8,
            "hermiticity_tolerance": 1e-10,
        },
        "retained_directed_bond_records": int(retained_count),
        "cut_directed_bond_records": int(cut_count),
        "retained_translations_input_cell": Rlist.tolist(),
        "solver": {
            "algorithm": "dense certified Colpa bosonic mode solver",
            "classification_counts": {},
            "per_k": [],
        },
        "dense_solver_cost": {
            "bdg_dimension": int(2 * nsite),
            "one_complex128_matrix_bytes": int(
                (2 * nsite) ** 2 * np.dtype(complex).itemsize
            ),
            "time_scaling": "O((2*n_strip_sites)^3) per edge momentum",
            "memory_scaling": (
                "O((2*n_strip_sites)^2) per edge momentum, excluding "
                "eigensolver work arrays"
            ),
        },
        "failures": [],
        "automatic_edge_relaxation": False,
    }
    result = StripSpectrumResult(
        request=request,
        model=model,
        embedding=embedding,
        edge_kpoints=edge_k,
        energies=None,
        left_edge_fraction=None,
        right_edge_fraction=None,
        interior_fraction=None,
        physical_amplitude_norm=None,
        degenerate_subspaces=[],
        site_map=site_map,
        site_width_coordinates=coordinates,
        reference_magmoms=reference,
        local_field=local_field,
        edge_regions=edge_regions,
        valid=False,
        diagnostics=diagnostics,
    )
    if not stationarity.stationary:
        diagnostics["failures"].append(
            "nonstationary open-strip reference: maximum transverse classical "
            f"gradient {stationarity.max_transverse:.3e} exceeds "
            f"{stationarity.tolerance:.3e}; supply an explicitly corrected "
            "reference_magmoms array (no automatic relaxation is performed)."
        )
        return result

    energies = np.full((request.kpoints, nsite), np.nan, dtype=float)
    left_fraction = np.full_like(energies, np.nan)
    right_fraction = np.full_like(energies, np.nan)
    interior_fraction = np.full_like(energies, np.nan)
    amplitude_norm = np.full_like(energies, np.nan)
    classifications: Dict[str, int] = {}
    all_stable = True
    physical_JR = tb2j_to_physical(JR, reference)
    spin_lengths = np.linalg.norm(reference, axis=1) / 2.0
    kpoints_input = embedding.to_input_k(
        np.column_stack(
            [
                edge_k,
                np.zeros(request.kpoints),
                np.full(request.kpoints, embedding.normal_k),
            ]
        )
    )
    for ik, k_input in enumerate(kpoints_input):
        try:
            hamiltonian = local_frame_bosonic_hamiltonian(
                physical_JR, Rlist, reference, k_input[None, :]
            )[0]
            modes = solve_bosonic_modes(hamiltonian)
        except (ModeSolverError, ValueError, np.linalg.LinAlgError) as error:
            all_stable = False
            classifications["solver_error"] = classifications.get("solver_error", 0) + 1
            diagnostics["solver"]["per_k"].append(
                {
                    "k_index": int(ik),
                    "classification": "solver_error",
                    "error": str(error),
                }
            )
            diagnostics["failures"].append(
                f"mode solve failed at edge k index {ik}: {error}"
            )
            continue

        classification = modes.classification
        classifications[classification] = classifications.get(classification, 0) + 1
        record = {
            "k_index": int(ik),
            "classification": classification,
            "hessian_min_eigenvalue_eV": float(modes.h_min_eigenvalue),
            "minimum_energy_eV": float(modes.min_energy),
            "residuals": dict(modes.residuals),
        }
        diagnostics["solver"]["per_k"].append(record)
        if classification not in ("positive_definite", "zero_mode"):
            all_stable = False
            diagnostics["failures"].append(
                f"{classification} bosonic spectrum at edge k index {ik}."
            )
            continue
        if modes.energies.shape != (nsite,):
            all_stable = False
            diagnostics["failures"].append(
                f"solver returned {len(modes.energies)} physical energies for "
                f"{nsite} strip sites at edge k index {ik}."
            )
            continue
        energies[ik] = modes.energies
        if classification == "zero_mode":
            record["zero_mode_indices"] = [
                int(i) for i in np.flatnonzero(modes.energies <= 1e-8)
            ]
        if modes.vectors is None:
            continue
        loc = _mode_localization(
            modes,
            modes.energies,
            spin_lengths,
            left_mask,
            right_mask,
            interior_mask,
            request.degeneracy_tol,
            ik,
        )
        (
            left_fraction[ik],
            right_fraction[ik],
            interior_fraction[ik],
            amplitude_norm[ik],
        ) = loc[:4]
        diagnostics["solver"]["per_k"][-1]["un_normalized_zero_mode_indices"] = [
            int(i)
            for i in range(nsite)
            if not np.isfinite(left_fraction[ik, i]) and modes.energies[i] <= 1e-8
        ]
        result.degenerate_subspaces.extend(loc[4])

    diagnostics["solver"]["classification_counts"] = classifications
    result.energies = energies
    result.left_edge_fraction = left_fraction
    result.right_edge_fraction = right_fraction
    result.interior_fraction = interior_fraction
    result.physical_amplitude_norm = amplitude_norm
    result.valid = bool(all_stable and not diagnostics["failures"])
    diagnostics["mode_resolution"] = (
        "basis-invariant subspace projections used at unresolved degeneracies; "
        "zero modes have no normalized physical edge amplitude"
    )
    diagnostics["edge_hybridization_policy"] = (
        "single-width edge fractions are observables, not a hybridization or "
        "width-convergence claim; compare multiple widths and subspace spectra"
    )
    return result
