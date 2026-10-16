"""Consumer-facing tests for Story-008 mode qualification diagnostics."""

import json

import numpy as np
import pytest

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon_phonon.qualification import (
    degenerate_subspace_determinism,
    phonon_determinism,
    qualification_report,
    qualify_magnon_modes,
    qualify_phonon_modes,
    verify_structure_identity,
)


def _magnon(nspin, spins, bonds):
    """Build isotropic exchange models; bonds are (R, i, j, J)."""
    Rlist = np.asarray([bond[0] for bond in bonds], dtype=int)
    JR = np.zeros((len(bonds), nspin, nspin, 3, 3), dtype=float)
    for ir, (_r, i, j, value) in enumerate(bonds):
        JR[ir, i, j] = np.eye(3) * value * spins[i] * spins[j]
    mags = np.zeros((nspin, 3))
    mags[:, 2] = 2.0 * np.asarray(spins)
    mags[1::2, 2] *= -1.0
    magnon = Magnon(
        nspin=nspin,
        magmom=mags,
        Rlist=Rlist,
        JR=JR,
        cell=np.eye(3),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


def _fm_chain():
    return _magnon(
        1,
        [1.0],
        [((0, 0, 0), 0, 0, 1.0), ((1, 0, 0), 0, 0, 1.0), ((-1, 0, 0), 0, 0, 1.0)],
    )


def _two_sublattice_afm():
    return _magnon(
        2,
        [1.0, 1.0],
        [
            ((0, 0, 0), 0, 1, -1.0),
            ((0, 0, 0), 1, 0, -1.0),
            ((1, 0, 0), 0, 1, -0.6),
            ((-1, 0, 0), 1, 0, -0.6),
        ],
    )


class DuckPhononModes:
    """Duck-typed fixture mirroring the Story-007 PhononModes dataclass."""

    def __init__(
        self,
        q,
        eigenvalues,
        eigenvectors,
        force_constants,
        diagnostics,
        masses_amu,
        atom_map,
        corrected_force_constants=None,
    ):
        self.q = np.asarray(q, float)
        self.eigenvalues_eV_per_Ang2_per_amu = np.asarray(eigenvalues, float)
        self.frequencies_thz = (
            np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * 15.633302
        )
        self.frequencies_meV = self.frequencies_thz * 4.135667696
        self.eigenvectors = np.asarray(eigenvectors, complex)
        self.cartesian_patterns = np.asarray(eigenvectors, complex)
        self.force_constants = np.asarray(force_constants, float)
        self.corrected_force_constants = (
            None
            if corrected_force_constants is None
            else np.asarray(corrected_force_constants, float)
        )
        self.masses_amu = np.asarray(masses_amu, float)
        self.atom_map = atom_map
        self.normalization = {"frequency_factor": 15.633302}
        self.diagnostics = diagnostics

    @property
    def instabilities(self):
        return list(self.diagnostics["instabilities"])

    def to_dict(self):
        return {"q_reduced": self.q.tolist()}


def _duck_phonon_gamma(force_constant=0.0, mass=1.0, symbols=("Cr",)):
    """One atom per cell, 1D harmonic chain: exact acoustic zero at Gamma.

    Real-space force constants Phi(+/-a) = -k give D(q) = 2k(1-cos 2pi q),
    i.e. D(Gamma) = 0 (acoustic) and the Gamma force-constant tensor is 0.
    A nonzero onsite term k0 shifts the Gamma branch gapped.
    """
    k0 = force_constant
    # Full 1x1 "supercell" tensor; translation residual 0 exactly.
    fc = np.zeros((1, 1, 3, 3))
    fc[0, 0, 0, 0] = k0
    diag = {
        "raw_reciprocity_residual_eV_per_Ang2": 0.0,
        "translation_residual_eV_per_Ang2": 0.0,
        "instabilities": [],
        "symmetrization": None,
        "raw_frequency_conversion_factor_THz": 15.633302,
    }
    dyn_gamma = np.array([[k0]])
    evals, evecs = np.linalg.eigh(dyn_gamma)
    return DuckPhononModes(
        q=np.zeros(3),
        eigenvalues=evals,
        eigenvectors=evecs,
        force_constants=fc,
        diagnostics=diag,
        masses_amu=[mass],
        atom_map={
            "symbols_primitive_order": list(symbols),
            "primitive_atom": [0],
            "atom_cells": [[0, 0, 0]],
        },
    )


# ---------------------------------------------------------------- magnon side


def test_magnon_report_has_per_point_stability_and_goldstone_residuals():
    magnon = _fm_chain()
    modes = magnon.get_site_boson_modes([[0.0, 0.0, 0.0], [0.17, 0.0, 0.0]])
    report = qualify_magnon_modes(modes)
    assert len(report["per_point"]) == 2
    for point, record in zip(modes.kpoints, report["per_point"]):
        assert record["kpoint"] == pytest.approx(point.tolist())
        assert record["min_eig_ev"] >= -record["stability_tol_ev"]
        assert record["stability_margin_ev"] == pytest.approx(
            record["min_eig_ev"] + record["stability_tol_ev"]
        )
        assert record["cholesky_tier"] in (0, 1, 2)
        assert record["shift_ev"] >= 0.0
        assert isinstance(record["n_zero_modes"], int)
    # Gamma FM chain: one certified Goldstone zero with a residual.
    gamma = report["per_point"][0]
    assert gamma["n_zero_modes"] == 1
    assert 0.0 <= gamma["symmetry_residual_ev"] <= gamma["stability_tol_ev"]
    assert report["provenance"]["reciprocal_convention"] == modes.reciprocal_convention
    assert report["units"]["energies"] == "eV"


def test_magnon_report_flags_afm_double_goldstone_and_no_warning_when_stable():
    modes = _two_sublattice_afm().get_site_boson_modes([[0.0, 0.0, 0.0]])
    report = qualify_magnon_modes(modes)
    assert report["per_point"][0]["n_zero_modes"] == 2
    assert report["warnings"] == []


# ---------------------------------------------------------------- phonon side


def test_phonon_report_has_acoustic_and_reciprocity_translation_diagnostics():
    modes = _duck_phonon_gamma()
    report = qualify_phonon_modes(modes)
    assert report["per_point"][0]["acoustic_count"] == 1
    assert report["per_point"][0]["acoustic_residual_thz"] == pytest.approx(
        0.0, abs=1e-10
    )
    assert report["raw_reciprocity_residual_eV_per_Ang2"] == 0.0
    assert report["translation_residual_eV_per_Ang2"] == 0.0
    assert report["instabilities"] == []
    assert report["units"]["frequencies"] == "THz"


def test_phonon_report_records_instabilities():
    diag = dict(_duck_phonon_gamma().diagnostics)
    diag["instabilities"] = [
        {"index": 0, "eigenvalue_eV_per_Ang2_per_amu": -0.5, "frequency_THz": 0.0}
    ]
    fc = np.zeros((1, 1, 3, 3))
    fc[0, 0, 0, 0] = 1.0
    evals, evecs = np.linalg.eigh(fc[0, 0, :, :].reshape(3, 3) * 0 - 0.5 * np.eye(3))
    modes = DuckPhononModes(
        q=np.zeros(3),
        eigenvalues=evals,
        eigenvectors=evecs,
        force_constants=fc,
        diagnostics=diag,
        masses_amu=[1.0],
        atom_map={"symbols_primitive_order": ["Cr"]},
    )
    report = qualify_phonon_modes(modes)
    assert report["instabilities"] == diag["instabilities"]
    assert any("instability" in w for w in report["warnings"])


def test_phonon_report_exposes_corrected_residuals_when_symmetrized():
    modes = _duck_phonon_gamma()
    modes.diagnostics = dict(modes.diagnostics)
    modes.diagnostics["symmetrization"] = {
        "method": "phonopy traditional translational+permutation symmetrization (level 1)",
        "corrected_reciprocity_residual_eV_per_Ang2": 0.0,
        "corrected_translation_residual_eV_per_Ang2": 0.0,
        "correction_norm_max_eV_per_Ang2": 1e-4,
    }
    report = qualify_phonon_modes(modes)
    assert report["corrected_reciprocity_residual_eV_per_Ang2"] == 0.0
    assert report["corrected_translation_residual_eV_per_Ang2"] == 0.0


# ------------------------------------------------------------------ cross: gauge


def test_degenerate_subspace_comparison_is_gauge_invariant():
    rng = np.random.default_rng(7)
    a = np.linalg.qr(rng.normal(size=(6, 6)))[0].astype(complex)
    energies = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    # Mix each degenerate pair by a different unitary: gauge change.
    block1 = np.array(
        [[np.exp(0.3j), np.exp(0.7j)], [-np.exp(0.7j).conj(), np.exp(0.3j).conj()]]
    )
    block1 /= np.sqrt(2)
    b = a.copy()
    b[:, 0:2] = a[:, 0:2] @ block1
    b[:, 2:4] = a[:, 2:4] @ block1.conj()
    result = degenerate_subspace_determinism(energies, a, energies, b)
    assert result["ok"]
    assert result["max_projector_deviation"] < 1e-12
    assert len(result["clusters"]) == 4
    # A non-degenerate-mixing perturbation must be flagged.
    c = b.copy()
    c[:, 4] = a[:, 4] + 0.1 * a[:, 5]
    c[:, 4] /= np.linalg.norm(c[:, 4])
    result = degenerate_subspace_determinism(energies, a, energies, c)
    assert not result["ok"]


def test_magnon_degenerate_subspace_determinism_under_repeated_diagonalization():
    magnon = _two_sublattice_afm()
    first = magnon.get_site_boson_modes([[0.13, 0.0, 0.0]])
    second = magnon.get_site_boson_modes([[0.13, 0.0, 0.0]])
    result = degenerate_subspace_determinism(
        first.energies[0],
        first.paraunitary[0].T,
        second.energies[0],
        second.paraunitary[0].T,
    )
    assert result["ok"]
    assert result["max_projector_deviation"] < 1e-10


def test_phonon_determinism_under_repeated_diagonalization():
    modes = _duck_phonon_gamma(force_constant=3.0)
    result = phonon_determinism(modes, repeats=3)
    assert result["ok"]
    assert result["repeats"] == 3
    assert result["max_projector_deviation"] < 1e-12


# ------------------------------------------------------------------ map identity


def test_structure_identity_verifies_symbols_positions_masses_bond_map():
    magnon = _two_sublattice_afm()
    magnon.positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    magnon.symbols = ["Cr", "Cr"]
    modes = _duck_phonon_gamma(symbols=("Cr", "Cr"))
    modes.masses_amu = np.array([51.9971, 51.9971])
    report = verify_structure_identity(magnon, modes)
    assert report["ok"]
    assert report["checks"]["natom"] and report["checks"]["symbols"]
    assert report["checks"]["positions"]
    assert report["checks"]["masses"]
    assert report["bond_map"]["n_directed_bonds"] == 4
    assert report["bond_map"]["indices_valid"]


def test_structure_identity_detects_symbol_and_mass_mismatch():
    magnon = _two_sublattice_afm()
    magnon.positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    magnon.symbols = ["Cr", "I"]
    modes = _duck_phonon_gamma(symbols=("Cr", "Cr"))
    modes.masses_amu = np.array([51.9971, 51.9971])
    report = verify_structure_identity(magnon, modes)
    assert not report["ok"]
    assert not report["checks"]["symbols"]


# ------------------------------------------------------------------ full report


def test_qualification_report_json_roundtrip_with_units_and_provenance():
    magnon = _two_sublattice_afm()
    magnon.positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    magnon.symbols = ["Cr", "Cr"]
    kpoints = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]
    phonon = _duck_phonon_gamma(symbols=("Cr", "Cr"))
    phonon.masses_amu = np.array([51.9971, 51.9971])
    report = qualification_report(
        spin_model=magnon,
        kpoints=kpoints,
        phonon_modes=phonon,
    )
    assert report["schema"] == "tb2j.magnon_phonon.qualification"
    assert report["units"]["energies"] == "eV"
    assert report["units"]["frequencies"] == "THz"
    assert report["units"]["masses"] == "u"
    assert "magnon" in report and "phonon" in report
    assert report["cross_checks"]["determinism"]["ok"]
    assert report["cross_checks"]["structure_identity"]["ok"]
    # JSON-serializable (JSON result convention).
    text = json.dumps(report)
    assert "symmetry_residual_ev" in text
