"""Signed bulk–edge correspondence for completed topology results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from TB2J.magnon.topology.bulk import (
    SCHEMA_VERSION as BULK_SCHEMA_VERSION,
)
from TB2J.magnon.topology.bulk import (
    BulkTopologyResult,
)

if TYPE_CHECKING:
    from TB2J.magnon.topology.strip import StripSpectrumResult

__all__ = ["BulkEdgeComparison", "compare_bulk_edge"]

SCHEMA_VERSION = "tb2j-topology.comparison/1"


@dataclass(frozen=True)
class BulkEdgeComparison:
    """Inspectable result of comparing one bulk gap with one strip edge."""

    source_results: Dict[str, Any]
    gap: int
    gap_energy_eV: Optional[float]
    bulk_gap_eV: Optional[float]
    edge: str
    chern_below_gap: Optional[int]
    signed_crossings: Optional[int]
    resolved_signed_subtotal: int
    expected_signed_crossings: Optional[int]
    crossings: List[Dict[str, Any]]
    ambiguous_crossings: int
    unmatched_crossings: int
    valid: bool
    matches: bool
    diagnostics: Dict[str, Any]

    def to_json_dict(self) -> Dict[str, Any]:
        """Return the versioned, JSON-safe comparison record."""
        bulk = self.source_results["bulk"]
        strip = self.source_results["strip"]
        same_model = bulk["model_digest"] == strip["model_digest"]
        same_version = (
            bulk["canonicalization_version"] == strip["canonicalization_version"]
        )
        return {
            "kind": "comparison",
            "schema_version": SCHEMA_VERSION,
            "model_digest": bulk["model_digest"] if same_model else None,
            "canonicalization_version": (
                bulk["canonicalization_version"] if same_version else None
            ),
            "provenance": {
                "bulk": bulk["provenance"],
                "strip": strip["provenance"],
            },
            "embeddings": {
                "bulk": bulk["embedding"],
                "strip": strip["embedding"],
            },
            "source_results": self.source_results,
            "gap": {
                "bands_below": int(self.gap),
                "energy_eV": self.gap_energy_eV,
                "minimum_indirect_gap_eV": self.bulk_gap_eV,
            },
            "edge": self.edge,
            "bulk_chern_below_gap": self.chern_below_gap,
            "spectral_flow": {
                "signed_crossings": self.signed_crossings,
                "resolved_signed_subtotal": int(self.resolved_signed_subtotal),
                "expected_signed_crossings": self.expected_signed_crossings,
                "crossings": self.crossings,
                "ambiguous_crossings": int(self.ambiguous_crossings),
                "unmatched_crossings": int(self.unmatched_crossings),
            },
            "valid": bool(self.valid),
            "matches": bool(self.matches),
            "diagnostics": self.diagnostics,
            "conventions": {
                "energy_unit": "eV",
                "bulk_orientation": "FHS plaquette order (k_edge,k_width): (i,j)->(i+1,j)->(i+1,j+1)->(i,j+1)",
                "crossing_sign": "+1 for increasing energy as fractional k_edge increases; -1 for decreasing energy",
                "edge_sides": {
                    "left": "minimum declared width coordinate; expected flow is -sum(Chern below gap)",
                    "right": "maximum declared width coordinate; expected flow is +sum(Chern below gap)",
                },
                "sign_certificate": "docs/sympy/magnon_topology_bulk_edge.py",
            },
        }


def _model_mapping_matches(a, b) -> bool:
    return (
        np.array_equal(a.cell, b.cell)
        and np.array_equal(a.positions, b.positions)
        and np.array_equal(a.positions_fractional, b.positions_fractional)
        and np.array_equal(a.index_spin, b.index_spin)
        and np.array_equal(a.spin_atoms, b.spin_atoms)
        and np.array_equal(a.magmoms, b.magmoms)
        and np.array_equal(a.Rlist, b.Rlist)
        and a.magnetic_cell_transform == b.magnetic_cell_transform
    )


def _interior_mismatch(strip) -> tuple[Optional[float], Optional[float], bool]:
    """Read the strip's measured interior-reference mismatch and tolerance."""
    record = strip.diagnostics.get("interior_reference_mismatch")
    if not isinstance(record, dict):
        return None, None, False
    try:
        mismatch = float(record["max_vector_mismatch_muB"])
        tolerance = float(record["tolerance_muB"])
    except (KeyError, TypeError, ValueError):
        return None, None, False
    matches = bool(record.get("matches_bulk_reference", False))
    if not np.isfinite(mismatch) or not np.isfinite(tolerance) or tolerance < 0:
        return None, None, False
    return mismatch, tolerance, matches and mismatch <= tolerance


def _strip_assembly_evidence(strip):
    """Check and copy the strip's explicit open-boundary assembly record."""
    diagnostics = strip.diagnostics
    try:
        retained = int(diagnostics["retained_directed_bond_records"])
        cut = int(diagnostics["cut_directed_bond_records"])
        translations = np.asarray(
            diagnostics["retained_translations_input_cell"], dtype=float
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        retained < 0
        or cut < 0
        or translations.ndim != 2
        or translations.shape[1] != 3
        or not np.isfinite(translations).all()
        or not np.allclose(translations, np.rint(translations), rtol=0.0, atol=0.0)
    ):
        return None
    return {
        "retained_directed_bond_records": retained,
        "cut_directed_bond_records": cut,
        "retained_translations_input_cell": (
            np.rint(translations).astype(int).tolist()
        ),
    }


def _reference_digest(strip) -> str:
    reference = np.ascontiguousarray(strip.reference_magmoms, dtype=np.float64)
    return hashlib.sha256(reference.tobytes()).hexdigest()


def _lower_group_chern(
    bulk: BulkTopologyResult, gap: int
) -> tuple[Optional[int], List[str]]:
    if not bulk.valid:
        return None, ["bulk result is invalid; no Chern sum can be compared."]
    failures: List[str] = []
    lower = set(range(gap))
    groups = bulk.groups
    if groups is None:
        groups = [
            {
                "bands": bulk.band_indices,
                "chern": bulk.chern,
                "valid": bulk.valid,
            }
        ]
    covered: List[int] = []
    chern_sum = 0
    for group in groups:
        bands = [int(b) for b in group.get("bands", [])]
        if not any(b in lower for b in bands):
            continue
        if any(b not in lower for b in bands):
            failures.append(
                f"bulk band group {bands} straddles the selected gap and cannot be split."
            )
            continue
        if not group.get("valid", False) or group.get("chern") is None:
            failures.append(
                f"bulk band group {bands} below the gap has no valid Chern value."
            )
            continue
        covered.extend(bands)
        chern_sum += int(group["chern"])
    if sorted(covered) != list(range(gap)):
        failures.append(
            f"valid bulk groups below the gap cover {sorted(covered)}, not every band 0..{gap - 1}."
        )
    return (chern_sum if not failures else None), failures


def _degenerate_at(
    energies: np.ndarray, k_index: int, mode: int, tolerance: float
) -> bool:
    if energies.shape[1] <= 1:
        return False
    distances = np.abs(energies[k_index] - energies[k_index, mode])
    distances[mode] = np.inf
    return bool(np.any(distances <= tolerance))


def _scan_edge_crossings(
    strip,
    edge: str,
    gap_energy: float,
    energy_tolerance: float,
    degeneracy_tolerance: float,
) -> tuple[List[Dict[str, Any]], int, int, List[str]]:
    """Count transverse, edge-resolved crossings on an endpoint-exclusive k loop."""
    failures: List[str] = []
    k = np.asarray(strip.edge_kpoints, dtype=float).reshape(-1)
    energies = np.asarray(strip.energies, dtype=float)
    if k.size < 3 or energies.ndim != 2 or energies.shape[0] != k.size:
        return [], 0, 0, ["strip edge mesh or energy array has an invalid shape."]
    if not np.isfinite(k).all() or not np.isfinite(energies).all():
        return [], 0, 0, ["strip edge mesh or energies contain non-finite values."]

    k = np.mod(k, 1.0)
    order = np.argsort(k)
    k = k[order]
    energies = energies[order]
    if np.any(np.diff(k) <= 1e-12):
        return [], 0, 0, ["strip edge mesh repeats a reciprocal coordinate."]
    steps = np.diff(np.r_[k, k[0] + 1.0])
    if not np.allclose(steps, steps[0], rtol=1e-8, atol=1e-10):
        return (
            [],
            0,
            0,
            ["strip edge mesh does not uniformly resolve one reciprocal period."],
        )
    if np.any(np.diff(energies, axis=1) < -degeneracy_tolerance):
        return [], 0, 0, ["strip energies are not ordered by ascending mode index."]

    fractions = (strip.left_edge_fraction, strip.right_edge_fraction)
    if all(value is not None for value in fractions):
        fractions = tuple(np.asarray(value, dtype=float)[order] for value in fractions)
        if any(value.shape != energies.shape for value in fractions):
            return (
                [],
                0,
                0,
                ["strip edge fractions do not match the (k, mode) energy axes."],
            )
    else:
        fractions = None

    old_to_new = np.empty(k.size, dtype=int)
    old_to_new[order] = np.arange(k.size)
    degenerate_groups = {}
    for group in strip.degenerate_subspaces:
        old_index = int(group["k_index"])
        if 0 <= old_index < k.size:
            new_index = int(old_to_new[old_index])
            for mode in group["mode_indices"]:
                degenerate_groups[(new_index, int(mode))] = group

    edge_column = 0 if edge == "left" else 1
    opposite_column = 1 - edge_column
    deg_tol = max(float(degeneracy_tolerance), energy_tolerance)
    crossings: List[Dict[str, Any]] = []
    ambiguous = 0
    unmatched = 0
    subtotal = 0
    processed_degenerate_groups = set()
    near = energies - gap_energy

    def resolve_degenerate_group(
        mode: int, exact_k: int, previous: int, following: int
    ) -> Optional[Dict[str, Any]]:
        group = degenerate_groups.get((exact_k, mode))
        if (
            group is None
            or group.get("resolution") != "basis_invariant_projected_observables"
        ):
            return None
        selected_spectrum = np.asarray(group[f"{edge}_fraction_spectrum"], dtype=float)
        opposite_name = "right" if edge == "left" else "left"
        opposite_spectrum = np.asarray(
            group[f"{opposite_name}_fraction_spectrum"], dtype=float
        )
        if (
            selected_spectrum.ndim != 1
            or opposite_spectrum.shape != selected_spectrum.shape
            or not np.isfinite(selected_spectrum).all()
            or not np.isfinite(opposite_spectrum).all()
            or np.any(np.abs(selected_spectrum - 0.5) <= 1e-8)
            or np.any(np.abs(opposite_spectrum - 0.5) <= 1e-8)
        ):
            return None
        modes = [int(b) for b in group["mode_indices"]]
        selected_rank = int(np.count_nonzero(selected_spectrum > 0.5))
        opposite_rank = int(np.count_nonzero(opposite_spectrum > 0.5))
        record = {
            "k_edge": float(k[exact_k]),
            "mode_indices": modes,
            "edge": edge,
            "edge_fraction_spectrum": selected_spectrum.tolist(),
            "opposite_edge_fraction_spectrum": opposite_spectrum.tolist(),
        }
        if selected_rank == 0:
            if opposite_rank == len(modes):
                record.update(
                    status="opposite_edge",
                    crossing_sign=0,
                    resolution=group["resolution"],
                )
                return record
            if opposite_rank == 0:
                record.update(
                    status="unmatched",
                    crossing_sign=None,
                    reason="degenerate in-gap subspace is not localized to either edge.",
                )
                return record
            return None

        if fractions is None:
            return None
        selected_before = fractions[edge_column][previous, modes]
        selected_after = fractions[edge_column][following, modes]
        opposite_before = fractions[opposite_column][previous, modes]
        opposite_after = fractions[opposite_column][following, modes]
        if not np.isfinite(
            np.concatenate(
                [selected_before, selected_after, opposite_before, opposite_after]
            )
        ).all():
            return None
        if np.any(np.abs(selected_before - 0.5) <= 1e-8) or np.any(
            np.abs(selected_after - 0.5) <= 1e-8
        ):
            return None
        before_edge_modes = selected_before > 0.5
        after_edge_modes = selected_after > 0.5
        if (
            np.count_nonzero(before_edge_modes) != selected_rank
            or np.count_nonzero(after_edge_modes) != selected_rank
        ):
            return None
        below_before = int(
            np.count_nonzero(energies[previous, modes][before_edge_modes] < gap_energy)
        )
        below_after = int(
            np.count_nonzero(energies[following, modes][after_edge_modes] < gap_energy)
        )
        flow = below_before - below_after
        record.update(
            status="resolved_degenerate_subspace",
            crossing_sign=int(flow),
            resolution=group["resolution"],
        )
        return record

    def classify(
        mode: int,
        location: float,
        left_k: int,
        right_k: int,
        fraction: float,
        direction: int,
        exact_k: Optional[int] = None,
    ):
        nonlocal ambiguous, unmatched, subtotal
        if exact_k is not None and _degenerate_at(energies, exact_k, mode, deg_tol):
            group = degenerate_groups.get((exact_k, mode))
            if group is not None:
                group_key = (exact_k, tuple(group["mode_indices"]))
                if group_key in processed_degenerate_groups:
                    return
                processed_degenerate_groups.add(group_key)
            record = resolve_degenerate_group(mode, exact_k, left_k, right_k)
            if record is None:
                ambiguous += 1
                crossings.append(
                    {
                        "k_edge": float(location % 1.0),
                        "mode_index": int(mode),
                        "crossing_sign": None,
                        "edge": edge,
                        "status": "ambiguous_degenerate_subspace",
                        "reason": "basis-invariant edge projection does not resolve the crossing assignment.",
                    }
                )
                return
            crossings.append(record)
            if record["status"] == "resolved_degenerate_subspace":
                subtotal += int(record["crossing_sign"])
            elif record["status"] == "unmatched":
                unmatched += 1
            return

        unresolved = _degenerate_at(energies, left_k, mode, deg_tol) or _degenerate_at(
            energies, right_k, mode, deg_tol
        )
        record: Dict[str, Any] = {
            "k_edge": float(location % 1.0),
            "mode_index": int(mode),
            "crossing_sign": int(direction),
            "edge": edge,
            "status": "resolved",
        }
        if unresolved:
            ambiguous += 1
            record.update(
                status="ambiguous_degenerate_subspace",
                reason="crossing touches a numerically unresolved strip-mode degeneracy.",
            )
        elif fractions is None:
            ambiguous += 1
            record.update(
                status="ambiguous_missing_edge_weights",
                reason="the strip result does not resolve physical edge weights at this crossing.",
            )
        else:
            if exact_k is not None:
                selected = float(fractions[edge_column][exact_k, mode])
                opposite = float(fractions[opposite_column][exact_k, mode])
            else:
                selected = float(
                    (1.0 - fraction) * fractions[edge_column][left_k, mode]
                    + fraction * fractions[edge_column][right_k, mode]
                )
                opposite = float(
                    (1.0 - fraction) * fractions[opposite_column][left_k, mode]
                    + fraction * fractions[opposite_column][right_k, mode]
                )
            record["edge_weight"] = selected
            record["opposite_edge_weight"] = opposite
            if not np.isfinite([selected, opposite]).all():
                ambiguous += 1
                record.update(
                    status="ambiguous_edge_weight",
                    reason="edge localization is unresolved at this crossing.",
                )
            elif selected > 0.5 and opposite < 0.5:
                subtotal += int(direction)
            elif opposite > 0.5 and selected < 0.5:
                record["status"] = "opposite_edge"
            elif (
                abs(selected - 0.5) <= 1e-8
                or abs(opposite - 0.5) <= 1e-8
                or (selected > 0.5 and opposite > 0.5)
            ):
                ambiguous += 1
                record.update(
                    status="ambiguous_edge_assignment",
                    reason="the physical edge fractions do not assign this crossing to one edge.",
                )
            else:
                unmatched += 1
                record.update(
                    status="unmatched",
                    reason="the in-gap crossing is not localized to either declared edge.",
                )
        crossings.append(record)

    for mode in range(energies.shape[1]):
        delta = near[:, mode]
        zeros = np.flatnonzero(np.abs(delta) <= energy_tolerance)
        zero_set = set(int(i) for i in zeros)
        handled: set[int] = set()
        for i in zeros:
            i = int(i)
            if i in handled:
                continue
            previous = (i - 1) % k.size
            following = (i + 1) % k.size
            if previous in zero_set or following in zero_set:
                component = {i}
                pending = [i]
                while pending:
                    index = pending.pop()
                    for neighbor in ((index - 1) % k.size, (index + 1) % k.size):
                        if neighbor in zero_set and neighbor not in component:
                            component.add(neighbor)
                            pending.append(neighbor)
                ambiguous += 1
                crossings.append(
                    {
                        "k_edge": float(k[i]),
                        "k_indices": sorted(component),
                        "mode_index": int(mode),
                        "crossing_sign": None,
                        "edge": edge,
                        "status": "ambiguous_gap_touch",
                        "reason": (
                            "the mode remains within the gap-energy tolerance "
                            "across adjacent samples."
                        ),
                    }
                )
                handled.update(component)
                continue
            if delta[previous] * delta[following] < 0:
                direction = 1 if delta[following] > delta[previous] else -1
                classify(mode, k[i], previous, following, 0.0, direction, exact_k=i)
            elif _degenerate_at(energies, i, mode, deg_tol):
                classify(mode, k[i], previous, following, 0.0, 0, exact_k=i)
            else:
                ambiguous += 1
                crossings.append(
                    {
                        "k_edge": float(k[i]),
                        "mode_index": int(mode),
                        "crossing_sign": None,
                        "edge": edge,
                        "status": "ambiguous_tangency",
                        "reason": "the mode touches the selected gap energy without a resolved transverse crossing.",
                    }
                )
            handled.add(i)

        for i in range(k.size):
            j = (i + 1) % k.size
            if i in zero_set or j in zero_set:
                continue
            if delta[i] * delta[j] < 0:
                fraction = float(-delta[i] / (delta[j] - delta[i]))
                location = float(k[i] + fraction * steps[i])
                direction = 1 if delta[j] > delta[i] else -1
                classify(mode, location, i, j, fraction, direction)

    if unmatched:
        failures.append(
            f"{unmatched} in-gap crossing(s) could not be assigned to the selected edge."
        )
    return crossings, subtotal, ambiguous, failures


def compare_bulk_edge(
    bulk: BulkTopologyResult,
    strip: "StripSpectrumResult",
    gap: int,
    edge: str,
) -> BulkEdgeComparison:
    """Compare signed edge spectral flow with a resolved bulk Chern sum.

    ``gap`` is the number of ascending positive-frequency bands below the
    selected gap. ``edge`` is ``left`` (minimum width coordinate) or ``right``
    (maximum width coordinate). The comparison uses the bulk indirect gap and
    requires valid source results, a matching parent model/embedding/reference,
    complete lower-band Chern coverage, and resolved edge assignments.
    """
    from TB2J.magnon.topology.strip import SCHEMA_VERSION as STRIP_SCHEMA_VERSION
    from TB2J.magnon.topology.strip import StripSpectrumResult

    if not isinstance(bulk, BulkTopologyResult):
        raise TypeError("compare_bulk_edge requires a completed BulkTopologyResult.")
    if not isinstance(strip, StripSpectrumResult):
        raise TypeError("compare_bulk_edge requires a completed StripSpectrumResult.")
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or int(gap) < 1:
        raise ValueError(
            "gap must be a positive integer band count below the selected gap."
        )
    gap = int(gap)
    if edge not in ("left", "right"):
        raise ValueError(
            "edge must be 'left' (minimum width) or 'right' (maximum width)."
        )

    checks: Dict[str, Any] = {}
    failures: List[str] = []

    def check(name: str, passed: bool, reason: str):
        checks[name] = bool(passed)
        if not passed:
            failures.append(reason)

    check(
        "bulk_result_valid",
        bool(bulk.valid),
        "bulk result is invalid; no bulk invariant can be compared.",
    )
    check(
        "strip_result_valid",
        bool(strip.valid),
        "strip result is invalid; no edge spectrum can be compared.",
    )
    check(
        "same_parent_digest",
        bulk.model.digest == strip.model.digest,
        "bulk and strip parent interaction digests differ.",
    )
    check(
        "same_canonicalization_version",
        bulk.model.canonicalization_version == strip.model.canonicalization_version,
        "bulk and strip canonicalization versions differ.",
    )
    check(
        "same_cell_and_image_maps",
        _model_mapping_matches(bulk.model, strip.model),
        "bulk and strip magnetic cells, positions or atom/spin image maps differ.",
    )
    same_embedding = (
        np.array_equal(bulk.embedding.basis, strip.embedding.basis)
        and np.array_equal(
            bulk.embedding.reciprocal_dual, strip.embedding.reciprocal_dual
        )
        and bulk.embedding.orientation_sign == strip.embedding.orientation_sign
    )
    check(
        "same_embedding_and_orientation",
        same_embedding,
        "bulk and strip embeddings or orientations differ.",
    )
    normal_matches = bool(
        np.isclose(
            bulk.embedding.normal_k, strip.embedding.normal_k, rtol=0.0, atol=1e-12
        )
    )
    check(
        "same_fixed_normal_momentum",
        normal_matches,
        "bulk and strip fixed normal momenta differ.",
    )

    mismatch, interior_tol, interior_matches = _interior_mismatch(strip)
    checks["strip_interior_matches_bulk_reference"] = bool(interior_matches)
    if not interior_matches:
        failures.append(
            "strip interior-reference mismatch is missing, unresolved, or exceeds its recorded tolerance."
        )
    assembly_evidence = _strip_assembly_evidence(strip)
    check(
        "strip_open_boundary_assembly_evidence_present",
        assembly_evidence is not None,
        "strip retained/cut bond and translation evidence is missing or malformed.",
    )

    provenance_ok = all(
        isinstance(model.provenance(), dict)
        and "selection" in model.provenance()
        and "exclusions" in model.provenance()
        for model in (bulk.model, strip.model)
    )
    check(
        "source_exclusion_provenance_present",
        provenance_ok,
        "source selection/exclusion provenance is missing.",
    )

    raw_gap_tol = bulk.request.gap_tol
    if isinstance(raw_gap_tol, (bool, np.bool_)):
        gap_tol = None
    else:
        try:
            gap_tol = float(raw_gap_tol)
        except (TypeError, ValueError):
            gap_tol = None
    gap_tol_valid = gap_tol is not None and np.isfinite(gap_tol) and gap_tol >= 0
    check(
        "bulk_gap_tolerance_valid",
        gap_tol_valid,
        "bulk gap tolerance must be finite and nonnegative.",
    )
    energies = None if bulk.energies is None else np.asarray(bulk.energies, dtype=float)
    gap_energy: Optional[float] = None
    bulk_gap: Optional[float] = None
    gap_resolved = False
    if energies is not None and energies.ndim == 3 and energies.shape[2] > gap:
        if np.isfinite(energies).all():
            below_max = float(np.max(energies[:, :, gap - 1]))
            above_min = float(np.min(energies[:, :, gap]))
            bulk_gap = above_min - below_max
            if gap_tol_valid:
                gap_resolved = bulk_gap > gap_tol
                if gap_resolved:
                    gap_energy = (below_max + above_min) / 2.0
    check(
        "resolved_bulk_indirect_gap",
        gap_resolved,
        "selected bulk gap is absent, non-finite, or unresolved on the bulk mesh.",
    )

    chern, group_failures = _lower_group_chern(bulk, gap)
    for failure in group_failures:
        failures.append(failure)
    checks["complete_valid_lower_band_groups"] = chern is not None

    expected = None
    if chern is not None:
        expected = -chern if edge == "left" else chern

    model_matches = (
        checks["same_parent_digest"]
        and checks["same_canonicalization_version"]
        and checks["same_cell_and_image_maps"]
    )
    geometry_matches = (
        checks["same_embedding_and_orientation"]
        and checks["same_fixed_normal_momentum"]
    )
    interior_ok = checks["strip_interior_matches_bulk_reference"]
    assembly_ok = checks["strip_open_boundary_assembly_evidence_present"]
    can_scan = bool(
        bulk.valid
        and strip.valid
        and gap_resolved
        and chern is not None
        and model_matches
        and geometry_matches
        and interior_ok
        and assembly_ok
    )
    crossings: List[Dict[str, Any]] = []
    subtotal = 0
    ambiguous = 0
    unmatched = 0
    scan_failures: List[str] = []
    degeneracy_tol = float(getattr(strip.request, "degeneracy_tol", 1e-8))
    energy_tol = (
        max(
            float(bulk.request.gap_tol),
            degeneracy_tol,
            100 * np.finfo(float).eps * max(1.0, abs(gap_energy)),
        )
        if gap_resolved and gap_energy is not None
        else None
    )
    if can_scan and gap_energy is not None and energy_tol is not None:
        crossings, subtotal, ambiguous, scan_failures = _scan_edge_crossings(
            strip,
            edge,
            gap_energy,
            energy_tol,
            degeneracy_tol,
        )
        unmatched = sum(record.get("status") == "unmatched" for record in crossings)
        failures.extend(scan_failures)

    flow_resolved = can_scan and ambiguous == 0 and unmatched == 0 and not scan_failures
    signed = subtotal if flow_resolved else None
    matches = signed is not None and expected is not None and signed == expected
    checks["edge_crossings_resolved"] = bool(flow_resolved)
    checks["signed_flow_matches_bulk"] = bool(matches)
    if flow_resolved and expected is not None and signed != expected:
        failures.append(
            f"signed edge flow {signed} does not match expected {expected} for the {edge} edge."
        )
    elif can_scan and not flow_resolved and not scan_failures:
        failures.append("edge crossing assignments are ambiguous or unmatched.")

    bulk_embedding = {
        "basis": bulk.embedding.basis.tolist(),
        "reciprocal_dual": bulk.embedding.reciprocal_dual.tolist(),
        "normal_k": float(bulk.embedding.normal_k),
        "orientation_sign": bulk.embedding.orientation_sign,
    }
    strip_embedding = {
        "basis": strip.embedding.basis.tolist(),
        "reciprocal_dual": strip.embedding.reciprocal_dual.tolist(),
        "normal_k": float(strip.embedding.normal_k),
        "orientation_sign": strip.embedding.orientation_sign,
    }
    source_results = {
        "bulk": {
            "kind": "bulk",
            "schema_version": BULK_SCHEMA_VERSION,
            "model_digest": bulk.model.digest,
            "canonicalization_version": bulk.model.canonicalization_version,
            "provenance": bulk.model.provenance(),
            "valid": bool(bulk.valid),
            "mesh": [int(v) for v in bulk.request.mesh],
            "band_indices": [int(b) for b in bulk.band_indices],
            "request_groups": (
                None
                if bulk.request.groups is None
                else [[int(b) for b in group] for group in bulk.request.groups]
            ),
            "gap_tolerance_eV": gap_tol if gap_tol_valid else None,
            "embedding": bulk_embedding,
            "band_groups": [
                {
                    "bands": [int(b) for b in group.get("bands", [])],
                    "chern": (
                        None if group.get("chern") is None else int(group["chern"])
                    ),
                    "valid": bool(group.get("valid", False)),
                }
                for group in (bulk.groups or [])
            ],
        },
        "strip": {
            "kind": "strip",
            "schema_version": STRIP_SCHEMA_VERSION,
            "model_digest": strip.model.digest,
            "canonicalization_version": strip.model.canonicalization_version,
            "provenance": strip.model.provenance(),
            "valid": bool(strip.valid),
            "width": int(strip.request.width),
            "kpoints": int(strip.request.kpoints),
            "termination": list(strip.request.termination),
            "edge_layers": int(strip.request.edge_layers),
            "edge_regions": strip.edge_regions,
            "reference_magmoms_sha256": _reference_digest(strip),
            "assembly": assembly_evidence,
            "embedding": strip_embedding,
            "reference_source": strip.diagnostics.get("reference_source"),
            "interior_reference_mismatch": strip.diagnostics.get(
                "interior_reference_mismatch"
            ),
            "interior_reference_tolerance": interior_tol,
        },
    }
    diagnostics = {
        "consistency_checks": checks,
        "failures": failures,
        "strip_interior_reference": {
            "max_mismatch": mismatch,
            "tolerance": interior_tol,
            "note": (
                "interior reference agreement is separate from edge relaxation "
                "and hybridization."
            ),
        },
        "crossing_scan": {
            "edge_kpoints": int(strip.request.kpoints),
            "edge_weight_threshold": 0.5,
            "energy_tolerance_eV": energy_tol,
            "counterpropagating_pairs_allowed": True,
            "note": (
                "the criterion is net signed spectral flow, not the unsigned "
                "number of edge branches."
            ),
        },
    }
    valid = bool(not failures and matches)
    return BulkEdgeComparison(
        source_results=source_results,
        gap=gap,
        gap_energy_eV=gap_energy,
        bulk_gap_eV=bulk_gap,
        edge=edge,
        chern_below_gap=chern,
        signed_crossings=signed,
        resolved_signed_subtotal=int(subtotal),
        expected_signed_crossings=expected,
        crossings=crossings,
        ambiguous_crossings=int(ambiguous),
        unmatched_crossings=int(unmatched),
        valid=valid,
        matches=bool(matches),
        diagnostics=diagnostics,
    )
