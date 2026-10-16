"""Mode qualification diagnostics for canonical magnon and phonon modes.

Story-008: one qualification report assembled from the Story-006 canonical
site-boson modes (per-point stability/shift/Goldstone residuals) and the
Story-007 phonon modes (acoustic, reciprocity and translation diagnostics),
plus cross-checks: gauge-invariant degenerate-subspace determinism under
repeated diagonalization, and atom/mass/bond-map identity between the
exchange model and the phonon structure.

Numerical kernels only: no filesystem or SCF access here. The phonon side is
duck-typed against the unfold-epc ``PhononModes`` dataclass. Energies are in
eV, frequencies in THz, masses in u, positions in Angstrom.
"""

from __future__ import annotations

import numpy as np

SCHEMA = "tb2j.magnon_phonon.qualification"
SCHEMA_VERSION = "1.0"

UNITS = {
    "energies": "eV",
    "frequencies": "THz",
    "masses": "u",
    "positions": "Ang",
    "force_constants": "eV/Ang^2",
    "force_constant_residuals": "eV/Ang^2",
}


def qualify_magnon_modes(modes) -> dict:
    """Assemble per-point stability/shift/Goldstone diagnostics.

    ``modes`` is a Story-006 ``SiteBosonModes`` result. The report is
    JSON-ready; all energies carry eV units and the producer's recorded
    conventions are echoed as provenance.
    """
    diagnostics = modes.diagnostics
    per_point = []
    warnings = []
    for ik, kpoint in enumerate(np.asarray(modes.kpoints, float)):
        min_eig = float(diagnostics.min_eig[ik])
        stability_tol = float(diagnostics.stability_tol[ik])
        symmetry_residual = float(diagnostics.symmetry_residual[ik])
        n_zero = int(np.count_nonzero(modes.is_zero_mode[ik]))
        record = {
            "kpoint": kpoint.tolist(),
            "min_eig_ev": min_eig,
            "stability_tol_ev": stability_tol,
            "stability_margin_ev": min_eig + stability_tol,
            "shift_ev": float(diagnostics.shift[ik]),
            "cholesky_tier": int(diagnostics.cholesky_tier[ik]),
            "symmetry_residual_ev": symmetry_residual,
            "n_zero_modes": n_zero,
            "zero_mode_indices": np.flatnonzero(modes.is_zero_mode[ik]).tolist(),
        }
        per_point.append(record)
        if n_zero:
            if not np.isfinite(symmetry_residual):
                warnings.append(
                    f"kpoint {kpoint.tolist()}: {n_zero} certified zero mode(s) "
                    "without a finite Goldstone symmetry residual"
                )
            elif symmetry_residual > stability_tol:
                warnings.append(
                    f"kpoint {kpoint.tolist()}: Goldstone residual "
                    f"{symmetry_residual:.3g} eV exceeds stability tolerance "
                    f"{stability_tol:.3g} eV"
                )
    return {
        "species": "magnon",
        "per_point": per_point,
        "warnings": warnings,
        "provenance": {
            "producer": "TB2J.magnon.magnon3.Magnon.get_site_boson_modes",
            "reciprocal_convention": modes.reciprocal_convention,
            "amplitude_convention": modes.amplitude_convention,
            "component_order": list(modes.component_order),
            "spin_lengths_u": np.asarray(modes.spin_lengths, float).tolist(),
        },
        "units": {"energies": "eV", "symmetry_residuals": "eV"},
    }


def qualify_phonon_modes(modes, *, acoustic_tol: float | None = None) -> dict:
    """Assemble acoustic/reciprocity/translation diagnostics.

    ``modes`` is duck-typed against the Story-007 ``PhononModes`` dataclass:
    it must expose ``q``, ``eigenvalues_eV_per_Ang2_per_amu``,
    ``frequencies_thz``, ``diagnostics`` (with the raw reciprocity and
    translation residual keys) and ``instabilities``. Acoustic residuals are
    evaluated only at symmetry zeros of q (Gamma); elsewhere the acoustic
    check is recorded as not applicable.
    """
    q = np.asarray(modes.q, float)
    eigenvalues = np.asarray(modes.eigenvalues_eV_per_Ang2_per_amu, float)
    frequencies_thz = np.asarray(modes.frequencies_thz, float)
    diagnostics = dict(modes.diagnostics)
    scale = max(1.0, float(np.max(np.abs(eigenvalues)))) if eigenvalues.size else 1.0
    if acoustic_tol is None:
        acoustic_tol = np.finfo(float).eps * eigenvalues.size * scale
    at_gamma = eigenvalues.size and np.max(np.abs(q - np.rint(q))) <= 1e-8
    if at_gamma:
        acoustic_mask = eigenvalues <= acoustic_tol
        acoustic_count = int(np.count_nonzero(acoustic_mask))
        acoustic_residual = (
            float(np.max(np.abs(frequencies_thz[acoustic_mask])))
            if acoustic_count
            else None
        )
    else:
        acoustic_mask = np.zeros(eigenvalues.shape, dtype=bool)
        acoustic_count = None
        acoustic_residual = None

    warnings = []
    if at_gamma and acoustic_count == 0:
        warnings.append(
            f"Gamma: no acoustic branch within tolerance {acoustic_tol:.3g} "
            "eV/Ang^2/amu; expected at least the translational null modes"
        )
    instabilities = list(modes.instabilities)
    if instabilities:
        warnings.append(
            f"{len(instabilities)} phonon instability(ies) recorded: unstable "
            "branch(es) with negative eigenvalues at q=" + q.tolist().__str__()
        )
    symmetrization = diagnostics.get("symmetrization")
    corrected_reciprocity = None
    corrected_translation = None
    if symmetrization:
        corrected_reciprocity = symmetrization.get(
            "corrected_reciprocity_residual_eV_per_Ang2"
        )
        corrected_translation = symmetrization.get(
            "corrected_translation_residual_eV_per_Ang2"
        )
    return {
        "species": "phonon",
        "per_point": [
            {
                "q": q.tolist(),
                "at_gamma": bool(at_gamma),
                "acoustic_count": acoustic_count,
                "acoustic_residual_thz": acoustic_residual,
                "acoustic_tol_eV_per_Ang2_per_amu": float(acoustic_tol),
            }
        ],
        "raw_reciprocity_residual_eV_per_Ang2": diagnostics.get(
            "raw_reciprocity_residual_eV_per_Ang2"
        ),
        "translation_residual_eV_per_Ang2": diagnostics.get(
            "translation_residual_eV_per_Ang2"
        ),
        "corrected_reciprocity_residual_eV_per_Ang2": corrected_reciprocity,
        "corrected_translation_residual_eV_per_Ang2": corrected_translation,
        "symmetrization": symmetrization,
        "instabilities": instabilities,
        "warnings": warnings,
        "provenance": {
            "producer": "unfold_epc.phonon campaign_phonon_modes / force_constants_phonon_modes",
            "frequency_conversion_factor_THz": diagnostics.get(
                "raw_frequency_conversion_factor_THz"
            ),
            "atom_map": modes.atom_map,
        },
        "units": {
            "frequencies": "THz",
            "eigenvalues": "eV/Ang^2/amu",
            "force_constant_residuals": "eV/Ang^2",
        },
    }


def _degenerate_clusters(energies: np.ndarray, tol: float) -> list[list[int]]:
    energies = np.asarray(energies, float)
    order = np.argsort(energies)
    clusters = []
    current = [int(order[0])]
    for previous, index in zip(order[:-1], order[1:]):
        if energies[index] - energies[previous] <= tol:
            current.append(int(index))
        else:
            clusters.append(current)
            current = [int(index)]
    clusters.append(current)
    return clusters


def degenerate_subspace_determinism(
    energies_a,
    vectors_a,
    energies_b,
    vectors_b,
    *,
    energy_tol: float = 1e-9,
    projector_tol: float = 1e-8,
) -> dict:
    """Gauge-invariant comparison of two diagonalizations of one operator.

    Vectors are columns (shape (dim, nmode)); comparisons within degenerate
    clusters use the orthogonal projector of the cluster, which is invariant
    under any unitary gauge rotation of the cluster basis (asserted in
    docs/sympy/mode_qualification_projectors.py). Clusters are matched by
    mean energy. ``energy_tol`` is absolute in eV.
    """
    energies_a = np.asarray(energies_a, float)
    energies_b = np.asarray(energies_b, float)
    vectors_a = np.asarray(vectors_a, complex)
    vectors_b = np.asarray(vectors_b, complex)
    if energies_a.shape != energies_b.shape or vectors_a.shape != vectors_b.shape:
        raise ValueError("determinism comparison requires matching mode shapes")
    clusters_a = _degenerate_clusters(energies_a, energy_tol)
    clusters_b = _degenerate_clusters(energies_b, energy_tol)
    if len(clusters_a) != len(clusters_b):
        return {
            "ok": False,
            "reason": "degenerate cluster structure differs between runs",
            "clusters": [],
            "max_projector_deviation": float("inf"),
            "projector_tol": projector_tol,
        }
    records = []
    max_deviation = 0.0
    ok = True
    for cluster_a, cluster_b in zip(clusters_a, clusters_b):
        mean_a = float(np.mean(energies_a[cluster_a]))
        mean_b = float(np.mean(energies_b[cluster_b]))
        matched = abs(mean_a - mean_b) <= energy_tol and len(cluster_a) == len(
            cluster_b
        )
        if not matched:
            ok = False
            records.append(
                {
                    "energy_ev": mean_a,
                    "energy_other_ev": mean_b,
                    "multiplicity": len(cluster_a),
                    "multiplicity_other": len(cluster_b),
                    "matched": False,
                    "projector_deviation": None,
                }
            )
            continue
        block_a = vectors_a[:, cluster_a]
        block_b = vectors_b[:, cluster_b]
        deviation = float(
            np.max(np.abs(block_a @ block_a.conj().T - block_b @ block_b.conj().T))
        )
        max_deviation = max(max_deviation, deviation)
        cluster_ok = deviation <= projector_tol
        ok = ok and cluster_ok
        records.append(
            {
                "energy_ev": mean_a,
                "energy_other_ev": mean_b,
                "multiplicity": len(cluster_a),
                "matched": True,
                "projector_deviation": deviation,
            }
        )
    return {
        "ok": bool(ok),
        "clusters": records,
        "max_projector_deviation": max_deviation,
        "projector_tol": projector_tol,
        "energy_tol_ev": energy_tol,
    }


def magnon_determinism(producer, kpoint, *, energy_tol=1e-9, projector_tol=1e-8):
    """Repeated-diagonalization determinism for canonical magnon modes.

    ``producer(kpoint)`` must return a ``SiteBosonModes``; the positive-norm
    columns of each run's paraunitary are compared through their degenerate
    subspace projectors.
    """
    kpoint = np.asarray(kpoint, float).reshape(1, 3)
    first = producer(kpoint)
    second = producer(kpoint)
    n = first.energies.shape[1]
    result = degenerate_subspace_determinism(
        first.energies[0],
        first.paraunitary[0].T,
        second.energies[0],
        second.paraunitary[0].T,
        energy_tol=energy_tol,
        projector_tol=projector_tol,
    )
    result["kind"] = "magnon"
    result["kpoint"] = kpoint[0].tolist()
    result["n_modes"] = int(n)
    return result


def phonon_determinism(modes, *, repeats=2, energy_tol=1e-9, projector_tol=1e-8):
    """Repeated-diagonalization determinism for phonon modes at their q.

    The dynamical matrix is reconstructed from the recorded eigensystem
    (D = V diag(eigenvalues) V^dagger, phonopy unit-L2 convention), then
    re-diagonalized ``repeats`` times; degenerate subspaces are compared
    through projectors.
    """
    eigenvalues = np.asarray(modes.eigenvalues_eV_per_Ang2_per_amu, float)
    vectors = np.asarray(modes.eigenvectors, complex)
    dynamical = (vectors * eigenvalues[None, :]) @ vectors.conj().T
    dynamical = 0.5 * (dynamical + dynamical.conj().T)
    max_deviation = 0.0
    ok = True
    reference_vals, reference_vecs = np.linalg.eigh(dynamical)
    for _ in range(max(1, repeats) - 1):
        vals, vecs = np.linalg.eigh(dynamical)
        result = degenerate_subspace_determinism(
            reference_vals,
            reference_vecs,
            vals,
            vecs,
            energy_tol=energy_tol,
            projector_tol=projector_tol,
        )
        ok = ok and result["ok"]
        max_deviation = max(max_deviation, result["max_projector_deviation"])
    return {
        "kind": "phonon",
        "q": np.asarray(modes.q, float).tolist(),
        "repeats": int(max(1, repeats)),
        "ok": bool(ok),
        "max_projector_deviation": max_deviation,
        "projector_tol": projector_tol,
        "energy_tol_eV_per_Ang2_per_amu": energy_tol,
    }


def verify_structure_identity(spin_model, phonon_modes, *, position_tol=1e-8) -> dict:
    """Verify one-structure atom/mass/bond-map identity.

    The exchange model must expose ``nspin``, ``magmom`` and ``Rlist``/``JR``
    and, for the full identity check, ``symbols`` and ``positions``. The
    phonon side is a Story-007 ``PhononModes`` with ``masses_amu`` and
    ``atom_map`` (including ``symbols_primitive_order``).
    """
    checks = {}
    warnings = []
    atom_map = dict(getattr(phonon_modes, "atom_map", {}) or {})
    masses = np.asarray(getattr(phonon_modes, "masses_amu", []), float)
    phonon_symbols = list(atom_map.get("symbols_primitive_order", []))

    nspin = int(getattr(spin_model, "nspin"))
    checks["natom"] = bool(masses.size == nspin)
    if not checks["natom"]:
        warnings.append(
            f"atom count mismatch: exchange model has {nspin} spins, "
            f"phonon masses have {masses.size} entries"
        )

    model_symbols = getattr(spin_model, "symbols", None)
    if model_symbols is None:
        model_symbols = getattr(spin_model, "atom_symbols", None)
    if model_symbols is None or phonon_symbols is None:
        checks["symbols"] = False
        warnings.append("symbols unavailable on one side; identity incomplete")
    else:
        checks["symbols"] = list(model_symbols) == phonon_symbols
        if not checks["symbols"]:
            warnings.append(
                f"symbol mismatch: exchange {list(model_symbols)} vs "
                f"phonon {phonon_symbols}"
            )

    model_positions = getattr(spin_model, "positions", None)
    if model_positions is None:
        model_positions = getattr(spin_model, "atom_positions", None)
    if model_positions is None:
        checks["positions"] = False
        warnings.append(
            "positions unavailable on the exchange model; identity incomplete"
        )
    else:
        model_positions = np.asarray(model_positions, float)
        if model_positions.shape[0] != len(phonon_symbols):
            checks["positions"] = False
            warnings.append("position count does not match phonon atom count")
        else:
            # Identity up to a global lattice translation of the whole map.
            shift = model_positions[0] - _phonon_reference_position(phonon_modes, 0)
            checks["positions"] = bool(
                _positions_match(phonon_modes, model_positions + shift, position_tol)
            )
            if not checks["positions"]:
                warnings.append(
                    "positions differ beyond "
                    f"{position_tol:g} Ang (up to a global translation)"
                )

    # Masses: one value per symbol, consistent across occurrences; compared
    # against the exchange model when it carries masses.
    per_symbol = {}
    masses_consistent = masses.size == len(phonon_symbols)
    for symbol, mass in zip(phonon_symbols, masses):
        if symbol in per_symbol and not np.isclose(per_symbol[symbol], mass):
            masses_consistent = False
        per_symbol.setdefault(symbol, float(mass))
    model_masses = getattr(spin_model, "masses", None)
    if model_masses is not None:
        model_masses = np.asarray(model_masses, float)
        if model_masses.size == masses.size:
            masses_consistent = masses_consistent and bool(
                np.allclose(model_masses, masses)
            )
        else:
            masses_consistent = False
    checks["masses"] = masses_consistent
    if not checks["masses"]:
        warnings.append("mass map is inconsistent between the two models")

    # Directed bond map: indices in range, values finite, count recorded.
    Rlist = np.asarray(getattr(spin_model, "Rlist"), int)
    JR = np.asarray(getattr(spin_model, "JR"), float)
    n_bonds = int(Rlist.shape[0]) if Rlist.ndim == 2 else 0
    indices_valid = bool(
        Rlist.ndim == 2
        and JR.ndim == 5
        and JR.shape[0] == n_bonds
        and JR.shape[1] == JR.shape[2] == nspin
        and np.all(np.isfinite(JR))
    )
    bond_map = {
        "n_directed_bonds": n_bonds,
        "indices_valid": indices_valid,
        "supercell_matrix": np.asarray(atom_map.get("supercell_matrix"), int).tolist()
        if atom_map.get("supercell_matrix") is not None
        else None,
    }
    if not indices_valid:
        warnings.append("exchange bond map has out-of-range or non-finite entries")
    return {
        "ok": bool(all(checks.values()) and indices_valid),
        "checks": checks,
        "bond_map": bond_map,
        "warnings": warnings,
        "provenance": {"atom_map": atom_map, "masses_u": masses.tolist()},
    }


def _phonon_reference_position(phonon_modes, index):
    structure = getattr(phonon_modes, "structure", None)
    if structure is None:
        return np.zeros(3)
    return np.asarray(structure["positions"], float)[index]


def _positions_match(phonon_modes, positions, tol):
    structure = getattr(phonon_modes, "structure", None)
    if structure is None:
        # Without phonon positions the check is a count/presence witness only.
        return True
    return bool(
        np.allclose(positions, np.asarray(structure["positions"], float), atol=tol)
    )


def qualification_report(
    spin_model,
    kpoints,
    phonon_modes=None,
    *,
    repeats=3,
    energy_tol=1e-9,
    projector_tol=1e-8,
) -> dict:
    """Assemble one qualification report over both mode bases plus cross-checks."""
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "units": dict(UNITS),
        "provenance": {
            "module": "TB2J.magnon_phonon.qualification",
            "story": "008",
        },
    }
    modes = spin_model.get_site_boson_modes(kpoints)
    report["magnon"] = qualify_magnon_modes(modes)
    report["cross_checks"] = {
        "determinism": magnon_determinism(
            spin_model.get_site_boson_modes,
            kpoints[0],
            energy_tol=energy_tol,
            projector_tol=projector_tol,
        )
    }
    if phonon_modes is not None:
        report["phonon"] = qualify_phonon_modes(phonon_modes)
        report["cross_checks"]["phonon_determinism"] = phonon_determinism(
            phonon_modes, repeats=repeats
        )
        report["cross_checks"]["structure_identity"] = verify_structure_identity(
            spin_model, phonon_modes
        )
    report["warnings"] = (
        report["magnon"]["warnings"]
        + (report.get("phonon", {}).get("warnings", []))
        + [
            warning
            for value in report["cross_checks"].values()
            for warning in value.get("warnings", [])
        ]
    )
    return report
