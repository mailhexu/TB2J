"""Same-SCF q=0 collinear anchor: numeric-only ExchangeCL2 comparison.

Story 011 (AC-006, user decision 2026-10-13): run the repaired frozen-MFT
response and the established nonorthogonal LKAG engine ``ExchangeCL2`` from
the **same saved noncollinear SCF** (the q=0 O2 fixture bundle) and report
their image-summed site-pair J values and signed differences as numbers.

This is a numbers-only result: ``agreement_status`` is always
``"not_evaluated"``; no agreement threshold exists and no AC-006/NFR-005
compatibility is claimed. Per the ADR-004 revision (2026-10-14) the
default rotating operator is the **onsite delta** (only R=0 same-site
blocks of the total-H spin splitting rotate, each at w=1), so the new
side matches the legacy LKAG projector by construction; the recorded
diagnostic variant ``weighted_total_h`` (every incident block at
w=1/count of its endpoint owners) remains selectable via ``operator``.
The certificate labels the ``operator_mode`` that produced every number,
identifies the distinct integration methods (exact occupied-band Hessian
vs LKAG contour), and any declaration that does not match the actual
settings, instead of labeling those settings matched.

The HamiltonIO ``SislParser`` NC reader cannot ingest the fixture's
4-component packing (``get_model`` expects SOC's eight components), so the
collinear spin channels for the legacy engine are split directly from the
same fixture bundle's stored components ``h = (h1, h2, h3, h4)`` (packing
``H4=-ImH12``): for an x-collinear reference,

    H_up = (h1 + h2) / 2 + h3,     H_dn = (h1 + h2) / 2 - h3,

with ``max|h1 - h2|`` and ``max|h4|`` required small. The bundle stores
absolute (Fermi-undone) energies while ``ExchangeCL2`` is run with
``efermi=0``, so every channel R-block is shifted by ``-Ef * S(R)``.

MathDoc: tb2j/collinear-anchor-projection (adjacent projection authority).

Prefer boring, bounded input surfaces: explicit site labels, explicit axes,
explicit ensemble, no guessed conventions.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.interfaces.siesta_spiral import SpiralBundle, SpiralImporter
from TB2J.interfaces.siesta_spiral_io import KB_EV, load_fixture
from TB2J.interfaces.siesta_spiral_response import (
    ENSEMBLE_GAPPED_T0,
    Ensemble,
    FrozenResponseProblem,
)
from TB2J.interfaces.siesta_spiral_vertex import (
    DEFAULT_OPERATOR_MODE,
    OPERATOR_MODE_ONSITE_DELTA,
    OPERATOR_MODES,
    extract_magnetic_vertex,
)
from TB2J.kpoints import monkhorst_pack
from TB2J.utils import symbol_number

ANCHOR_VERSION = "1.1"
COMPARISON_KIND = "numeric_only_q0_same_scf"
AGREEMENT_STATUS = "not_evaluated"

#: x-collinearity of the stored components must hold at this level; the
#: recorded fixture shows max|h1-h2| = 9.4e-14 eV and max|h4| = 1.5e-15 eV.
COLLINEAR_TOL_EV = 1e-8
#: Union of the two channel spectra must reproduce the spinor pencil.
SPECTRAL_EQUIVALENCE_BUDGET_EV = 1e-10
#: Cell mismatch between the fdf-parsed geometry (Ang) and the bundle (Bohr).
CELL_MATCH_TOL_ANG = 1e-4
#: The collinear reference is stationary under transverse single-site tilts.
GRADIENT_STATIONARITY_BUDGET_EV_PER_RAD = 1e-8
#: Y/Z transverse axes are symmetry equivalent on the FM dimer.
AXIS_SYMMETRY_BUDGET_EV = 1e-8
#: A rotation axis must be transverse to the magnetic axis to this level;
#: a near-parallel axis drives u = t x n to zero and inflates the projection.
AXIS_TRANSVERSE_TOL = 1e-6

MAGNETIC_AXIS = (1.0, 0.0, 0.0)
DEFAULT_AXES = ("y", "z")
DEFAULT_KMESH = (1, 1, 4)
DEFAULT_NZ = 200
DEFAULT_RCUT_ANG = 8.0

_AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

_CONVENTIONS = {
    "physical_model": (
        "E = -sum_{a<b} J_pair_ab (e_a . e_b); positive J is ferromagnetic"
    ),
    "projection": (
        "K_ab = d2E/dtheta_a dtheta_b on transverse axes t_a; "
        "J_pair_ab = -K_ab / (sign(m_a . m_b) (u_a . u_b)) with u = t x n"
    ),
    "tb2j_stored": (
        "both engines are reported as TB2J ordered-pair stored values: "
        "J_stored = J_pair / 2; exchange_Jdict stores (R,i,j) and (-R,j,i); "
        "magnon/J(q) sums run over all R"
    ),
    "legacy_pair_number": (
        "legacy pair value is the R-image sum over ONE directed (a, b); "
        "each stored direction carries J_pair_bond/2, and the reverse "
        "direction is retained as a same-bond reciprocity audit, never "
        "summed in"
    ),
    "moment_normalization": (
        "no division by moment^2 on either side; the legacy Delta carries "
        "the full onsite H_up - H_dn split"
    ),
    "operator_modes": (
        "the vertex certificate labels operator_mode (ADR-004 2026-10-14): "
        "onsite_delta (default) rotates only R=0 same-site blocks at w=1; "
        "weighted_total_h is the recorded diagnostic variant with "
        "w=1/count endpoint-owner weights on every incident block"
    ),
    "mathdoc": "tb2j/collinear-anchor-projection",
}


class AnchorError(RuntimeError):
    """Raised when the anchor adapter refuses an uncertifiable input."""


class AnchorSettingMismatchError(AnchorError):
    """Raised on refusal when a declared setting mismatches the actual one."""


@dataclass(frozen=True)
class CollinearChannels:
    """x-collinear spin channels split from one noncollinear bundle."""

    rlist: tuple
    h_up: np.ndarray
    h_dn: np.ndarray
    overlap: np.ndarray
    norb: int
    ef_ev: float
    max_diag_split_ev: float
    max_mixing_imag_ev: float
    hermiticity_defect_ev: float

    @property
    def channel_split(self) -> dict:
        return {
            "max_diag_split_ev": self.max_diag_split_ev,
            "max_mixing_imag_ev": self.max_mixing_imag_ev,
            "hermiticity_defect_ev": self.hermiticity_defect_ev,
            "ef_shift_ev": self.ef_ev,
            "formula": "H_up/dn = (h1+h2)/2 +/- h3 from packing H4=-ImH12",
        }


@dataclass(frozen=True)
class AnchorDeclaration:
    """User-declared settings, each checked against the actual ones.

    ``None`` fields are undeclared and only recorded as such. Declarations
    never change the numeric comparison; they only add truthful per-setting
    matched/unmatched reporting (and optional refusal).
    """

    kmesh: tuple | None = None
    j_convention: str | None = None
    ensemble: str | None = None
    projector: str | None = None
    integration: str | None = None


@dataclass
class AnchorComparison:
    """Everything the numbers-only certificate and tests need, in memory."""

    certificate: dict
    channels: CollinearChannels
    legacy_exchange: ExchangeCL2
    problem: FrozenResponseProblem
    ensemble: Ensemble
    kpoints_fractional: np.ndarray
    kpoints_cartesian_inv_bohr: np.ndarray
    kweights: np.ndarray
    response_by_axis: dict
    j_stored_by_axis: dict
    legacy_pair_sum_ev: float
    legacy_pair: dict
    settings: dict

    def to_dict(self) -> dict:
        """JSON-safe deep copy of the certificate."""
        return copy.deepcopy(self.certificate)


# ---------------------------------------------------------------------------
# collinear channel split from the same NC bundle
# ---------------------------------------------------------------------------
def _sorted_rkeys(bundle: SpiralBundle) -> list:
    return sorted({R for _mu, _nu, R in bundle.sparse})


def build_collinear_spin_channels(
    bundle: SpiralBundle,
    *,
    tol_ev: float = COLLINEAR_TOL_EV,
) -> CollinearChannels:
    """Split the stored NC components into x-collinear spin channels.

    The channel Hamiltonians keep the bundle's absolute energy zero before
    the ``-Ef * S`` shift applied here, so ``ExchangeCL2`` can run with its
    ``efermi=0`` on identically shifted data.
    """
    bundle.validate()
    if bundle.fermi_shift != "none":
        raise AnchorError(
            f"bundle fermi_shift {bundle.fermi_shift!r} unsupported; the "
            "channel split requires the recorded absolute (unshifted) "
            "Hamiltonian"
        )
    if float(np.max(np.abs(bundle.q))) != 0.0:
        raise AnchorError(
            "anchor requires a zero-pitch (q=0) bundle; got q="
            f"{tuple(float(v) for v in bundle.q)}"
        )
    if not 0.0 < float(tol_ev) <= 0.05:
        raise AnchorError(
            f"collinear tolerance must be tiny and positive, got {tol_ev!r}"
        )

    ef_ev = float(bundle.metadata["ef_ev"])
    rlist = _sorted_rkeys(bundle)
    rindex = {R: i for i, R in enumerate(rlist)}
    norb = bundle.norb
    shape = (len(rlist), norb, norb)
    h_up = np.zeros(shape)
    h_dn = np.zeros(shape)
    overlap = np.zeros(shape)

    max_diag_split = 0.0
    max_mixing_imag = 0.0
    hermiticity_defect = 0.0
    for (mu, nu, R), (h1, h2, h3, h4) in bundle.sparse.items():
        idx = rindex[R]
        h_up[idx, mu, nu] += 0.5 * (h1 + h2) + h3
        h_dn[idx, mu, nu] += 0.5 * (h1 + h2) - h3
        overlap[idx, mu, nu] += bundle.overlap[(mu, nu, R)]
        max_diag_split = max(max_diag_split, abs(float(h1) - float(h2)))
        max_mixing_imag = max(max_mixing_imag, abs(float(h4)))

    # A real-symmetric channel requires the partner block to carry the same
    # value; the scalar channels have no pencil-level hermitization, so any
    # defect is refused instead of silently symmetrized.
    for (mu, nu, R), comps in bundle.sparse.items():
        partner = bundle.sparse[(nu, mu, bundle._neg(R))]
        mean_up_self = 0.5 * (comps[0] + comps[1])
        mean_up_partner = 0.5 * (partner[0] + partner[1])
        defect = max(
            abs((mean_up_self + comps[2]) - (mean_up_partner + partner[2])),
            abs((mean_up_self - comps[2]) - (mean_up_partner - partner[2])),
        )
        hermiticity_defect = max(hermiticity_defect, float(defect))

    if max_diag_split > tol_ev or max_mixing_imag > tol_ev:
        raise AnchorError(
            "stored components are not x-collinear within "
            f"{tol_ev!r} eV (max|h1-h2| = {max_diag_split:.3e} eV, "
            f"max|h4| = {max_mixing_imag:.3e} eV); refusing to split into "
            "spin channels"
        )
    if hermiticity_defect > tol_ev:
        raise AnchorError(
            "channel blocks are not real-symmetric (defect "
            f"{hermiticity_defect:.3e} eV)"
        )

    for idx in range(len(rlist)):
        h_up[idx] -= ef_ev * overlap[idx]
        h_dn[idx] -= ef_ev * overlap[idx]

    return CollinearChannels(
        rlist=tuple(rlist),
        h_up=h_up,
        h_dn=h_dn,
        overlap=overlap,
        norb=int(norb),
        ef_ev=ef_ev,
        max_diag_split_ev=max_diag_split,
        max_mixing_imag_ev=max_mixing_imag,
        hermiticity_defect_ev=hermiticity_defect,
    )


def _site_moment_signs(bundle: SpiralBundle, orbitals_by_site: dict) -> dict:
    """Signed x-moment proxy per site from the onsite R=0 mixing blocks."""
    signs = {}
    for site, orbitals in orbitals_by_site.items():
        total = 0.0
        for (mu, nu, R), (_h1, _h2, h3, _h4) in bundle.sparse.items():
            if mu == nu and R == (0, 0, 0) and mu in set(orbitals):
                total += float(h3)
        if abs(total) < 1e-12:
            raise AnchorError(
                f"site {site!r} has no resolvable x moment; the anchor "
                "refuses to guess sign(m_a . m_b)"
            )
        signs[site] = 1.0 if total > 0.0 else -1.0
    return signs


# ---------------------------------------------------------------------------
# spectral equivalence of the channel split
# ---------------------------------------------------------------------------
def channel_pencil(channel_values: dict, bundle: SpiralBundle, k):
    """Scalar-channel pencil at ``k`` in the importer phase convention.

    ``channel_values`` maps (mu, nu, R) to a scalar block value; the phasing
    mirrors ``SpiralImporter.assemble_pencil`` exactly (full xij displacement,
    ``exp(-i k.xij)``), so the channel spectra are comparable to the spinor
    pencil of the same bundle.
    """
    cell = np.asarray(bundle.cell, dtype=float)
    pos = np.asarray(bundle.orbital_positions, dtype=float)
    kvec = np.asarray(k, dtype=float)
    norb = bundle.norb
    h = np.zeros((norb, norb), complex)
    s = np.zeros((norb, norb), complex)
    for (mu, nu, R), value in channel_values.items():
        xij = np.asarray(R, dtype=float) @ cell + pos[nu] - pos[mu]
        phase = np.exp(-1j * float(kvec @ xij))
        h[mu, nu] += value * phase
        s[mu, nu] += bundle.overlap[(mu, nu, R)] * phase
    return h, s


def spectral_equivalence_check(bundle: SpiralBundle, kpoints) -> dict:
    """Union of channel spectra vs the spinor pencil on the given kpoints."""
    importer = SpiralImporter()
    up_values = {key: 0.5 * (c[0] + c[1]) + c[2] for key, c in bundle.sparse.items()}
    dn_values = {key: 0.5 * (c[0] + c[1]) - c[2] for key, c in bundle.sparse.items()}
    max_dev = 0.0
    for k in kpoints:
        h_up, s_up = channel_pencil(up_values, bundle, k)
        h_dn, s_dn = channel_pencil(dn_values, bundle, k)
        union = np.concatenate(
            (
                eigh(h_up, s_up, eigvals_only=True),
                eigh(h_dn, s_dn, eigvals_only=True),
            )
        )
        h_spinor, s_spinor = importer.assemble_pencil(bundle, k)
        spinor = eigh(h_spinor, s_spinor, eigvals_only=True)
        union.sort()
        spinor.sort()
        max_dev = max(max_dev, float(np.max(np.abs(union - spinor))))
    return {
        "n_kpoints": int(len(kpoints)),
        "max_dev_ev": max_dev,
        "budget_ev": SPECTRAL_EQUIVALENCE_BUDGET_EV,
        "pass": bool(max_dev <= SPECTRAL_EQUIVALENCE_BUDGET_EV),
        "statement": (
            "union of the up/down channel generalized eigenvalues equals the "
            "NC spinor pencil spectrum (sorted) on every checked kpoint"
        ),
    }


# ---------------------------------------------------------------------------
# legacy engine input assembly
# ---------------------------------------------------------------------------
def _plain_orbital_labels(spinor_labels) -> list:
    """Strip the duplicated spin suffix from every second label of a pair."""
    labels = list(spinor_labels)
    if len(labels) % 2 != 0:
        raise AnchorError(
            "spinor basis carries an odd label count; expected up/down "
            "duplicated pairs"
        )
    plain = []
    for k in range(0, len(labels), 2):
        parts_a = labels[k].rsplit("|", 1)
        parts_b = labels[k + 1].rsplit("|", 1)
        if len(parts_a) != 2 or len(parts_b) != 2:
            raise AnchorError(
                f"labels {labels[k]!r}/{labels[k + 1]!r} lack a '|spin' "
                "suffix; refusing to strip a non-spin token"
            )
        head_a, suffix_a = parts_a
        head_b, suffix_b = parts_b
        for label, suffix in ((labels[k], suffix_a), (labels[k + 1], suffix_b)):
            if suffix not in ("up", "down"):
                raise AnchorError(
                    f"label {label!r} does not end in an up/down spin suffix; "
                    "refusing to strip a non-spin token"
                )
        if head_a != head_b or suffix_a == suffix_b:
            raise AnchorError(
                f"labels {labels[k]!r}/{labels[k + 1]!r} are not one "
                "orbital's up/down spin-suffix duplicate pair"
            )
        plain.append(head_a)
    if len(set(plain)) != len(plain):
        raise AnchorError("stripped orbital labels are not unique")
    return plain


def _load_parser(fdf_path):
    try:
        from HamiltonIO.siesta import SislParser
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AnchorError(
            "the legacy engine requires HamiltonIO with the sisl backend"
        ) from exc
    parser = SislParser(
        fdf_fname=str(fdf_path), ispin=None, read_H_soc=False, orth=False
    )
    parser.read_spin(None)
    if getattr(parser, "spintype", None) != "spinor":
        raise AnchorError(
            "anchor adapter consumes the noncollinear 4-component HSX "
            f"(spintype 'spinor'); got {getattr(parser, 'spintype', None)!r}"
        )
    parser.geom, parser.atoms = parser.read_geometry(fdf=parser.fdf)
    parser.read_basis()
    parser.Rlist = parser.read_Rlist(parser.geom)
    return parser


def _check_orbital_site_assignment(
    bundle: SpiralBundle,
    parser_positions_ang,
    orbitals_by_site: dict,
    atom_index_of_site: dict,
) -> dict:
    """Order-sensitive guard: each bundle orbital sits on its declared atom.

    Spinor spectra are invariant under a basis permutation, so the channel
    matrices' spatial ordering is certified by orbital positions, not by the
    eigenvalue oracle.
    """
    import ase.units

    positions_bohr = np.asarray(parser_positions_ang, dtype=float) / ase.units.Bohr
    orbital_positions = np.asarray(bundle.orbital_positions, dtype=float)
    max_diff = 0.0
    for site, orbitals in orbitals_by_site.items():
        atom_index = atom_index_of_site.get(site)
        if atom_index is None or atom_index >= len(positions_bohr):
            raise AnchorError(f"site {site!r} does not map to a parsed atom row")
        for orb in orbitals:
            max_diff = max(
                max_diff,
                float(
                    np.max(np.abs(orbital_positions[orb] - positions_bohr[atom_index]))
                ),
            )
    if max_diff > 1e-6:
        raise AnchorError(
            "bundle orbital positions do not match the parsed atom positions "
            f"(max diff {max_diff:.3e} Bohr); the channel ordering cannot be "
            "certified, refusing a permuted basis"
        )
    return {"orbital_position_max_diff_bohr": max_diff}


def _check_same_scf_geometry(parser, bundle: SpiralBundle, reference: dict) -> dict:
    import ase.units

    cell_ang = np.asarray(parser.geom.cell, dtype=float)
    cell_from_bundle = np.asarray(bundle.cell, dtype=float) * ase.units.Bohr
    max_diff = float(np.max(np.abs(cell_ang - cell_from_bundle)))
    if max_diff > CELL_MATCH_TOL_ANG:
        raise AnchorError(
            "fdf geometry cell differs from the fixture bundle cell by "
            f"{max_diff:.3e} Ang; refusing a cross-SCF mixture"
        )
    sc_off = {tuple(int(v) for v in R) for R in np.asarray(parser.Rlist)}
    missing = set(_sorted_rkeys(bundle)) - sc_off
    if missing:
        raise AnchorError(
            f"fixture shells absent from the HSX supercell list: {sorted(missing)}"
        )
    nel = parser.read_nel(parser.fdf)
    electron_count = float(reference["electron_count"])
    if nel is not None and abs(float(nel) - electron_count) > 1e-4:
        raise AnchorError(
            f"fdf electron count {nel!r} differs from the fixture's "
            f"{electron_count!r}; refusing a cross-SCF mixture"
        )
    return {
        "cell_max_diff_ang": max_diff,
        "cell_tolerance_ang": CELL_MATCH_TOL_ANG,
        "electron_count": electron_count,
        "parsed_electron_count": None if nel is None else float(nel),
        "all_bundle_shells_in_hsx_supercells": True,
    }


def _channel_model(channels: CollinearChannels, parser, labels, nel, spin_channel):
    from HamiltonIO.siesta.sisl_wrapper import SiestaHamiltonian

    hr = channels.h_up if spin_channel == 0 else channels.h_dn
    return SiestaHamiltonian(
        HR=hr,
        SR=channels.overlap,
        Rlist=np.asarray(channels.rlist, dtype=int),
        nbasis=channels.norb,
        atoms=parser.atoms,
        orbs=list(labels),
        nspin=1,
        nel=nel,
        orth=False,
        sisl_hamiltonian=None,
        spin_channel=spin_channel,
        orb_dict=None,
    )


def run_legacy_exchange(
    channels: CollinearChannels,
    parser,
    labels,
    nel,
    *,
    kmesh=DEFAULT_KMESH,
    nz=DEFAULT_NZ,
    rcut_ang=DEFAULT_RCUT_ANG,
    magnetic_elements=("O",),
):
    """Build both channel models and run ``ExchangeCL2.calculate_all``.

    No band-cutoff knob is exposed: ExchangeCL2 builds its TBGreen objects
    with their own ``initial_emin`` default and derives the adjusted band
    window internally (``ExchangeCL2._adjust_emin``); the actual values are
    recorded from the objects in the certificate's ``band_window``
    diagnostics. ``emax`` and ``ne`` are likewise not exposed.
    """
    basis = dict(zip(labels, range(channels.norb)))
    model_up = _channel_model(channels, parser, labels, nel, 0)
    model_dn = _channel_model(channels, parser, labels, nel, 1)
    exchange = ExchangeCL2(
        tbmodels=(model_up, model_dn),
        atoms=parser.atoms,
        basis=basis,
        efermi=0.0,
        magnetic_elements=list(magnetic_elements),
        kmesh=list(kmesh),
        nz=int(nz),
        Rcut=float(rcut_ang),
        nproc=1,
        use_cache=False,
        orth=False,
        description=(
            "story-011 same-SCF q=0 anchor: x-collinear channels split from "
            "the NC HSX bundle, Fermi-zero shifted"
        ),
    )
    exchange.calculate_all()
    return exchange


def image_summed_pair_J(exchange, name_a: str, name_b: str) -> dict:
    """One-directed R-image sum of ``exchange_Jdict`` for the ordered pair.

    TB2J stores both ordered directions of every R image, each carrying the
    stored value ``J_pair_bond / 2``
    (docs/sympy/collinear_anchor_projection). The pair value comparable to
    the projected response ``-K_ab / 2`` is therefore the sum over ONE
    direction ``(name_a, name_b)`` only; the reverse direction is the same
    physical bond seen from the other endpoint and is retained separately
    for a reciprocity audit, never summed in.
    """
    names = list(symbol_number(exchange.atoms).keys())
    forward, reverse = [], []
    for key, value in exchange.exchange_Jdict.items():
        R = tuple(int(v) for v in np.asarray(key[0]).ravel())
        site_i = names[exchange.ind_mag_atoms[key[1]]]
        site_j = names[exchange.ind_mag_atoms[key[2]]]
        if site_i == site_j or {site_i, site_j} != {name_a, name_b}:
            continue
        entry = {"R": list(R), "i": site_i, "j": site_j, "J_eV": float(value)}
        if (site_i, site_j) == (name_a, name_b):
            forward.append(entry)
        else:
            reverse.append(entry)
    forward_sum = sum(e["J_eV"] for e in forward)
    reverse_sum = sum(e["J_eV"] for e in reverse)
    reverse_by_key = {(tuple(e["R"]), e["i"], e["j"]): e["J_eV"] for e in reverse}
    complete = bool(forward) and len(forward) == len(reverse)
    reciprocity = 0.0
    for entry in forward:
        R = tuple(entry["R"])
        partner = reverse_by_key.get((tuple(-v for v in R), entry["j"], entry["i"]))
        if partner is None:
            complete = False
            reciprocity = float("inf")
            break
        reciprocity = max(reciprocity, abs(partner - entry["J_eV"]))
    return {
        "forward_sum_ev": forward_sum,
        "reverse_sum_ev": reverse_sum,
        "reciprocity_complete": complete,
        "reciprocity_max_abs_diff_ev": reciprocity,
        "forward_entries": forward,
        "reverse_entries": reverse,
    }


# ---------------------------------------------------------------------------
# frozen-response side and projection
# ---------------------------------------------------------------------------
def _fractional_to_cartesian_inv_bohr(k_fractional, cell_bohr) -> np.ndarray:
    """Convert the legacy fractional mesh to the bundle's Cartesian convention.

    The legacy engine phases ``H(R)`` with ``exp(+2*pi*i k_frac . R)`` on
    integer cell vectors while the bundle importer phases with
    ``exp(-i k_cart . xij)``; the sign flip below maps the R-dependent part
    of one onto the other.
    """
    matrix = 2.0 * np.pi * np.linalg.inv(np.asarray(cell_bohr, dtype=float))
    return -np.asarray(k_fractional, dtype=float) @ matrix.T


def _mesh_gap_ev(problem: FrozenResponseProblem, kpoints, n_occupied) -> float:
    gap = np.inf
    for k in kpoints:
        h, s = problem.pencil(k)
        eigenvalues = eigh(h, s, eigvals_only=True)
        gap = min(gap, float(eigenvalues[n_occupied] - eigenvalues[n_occupied - 1]))
    return gap


def _axis_vector(axes_by_site, site, axis) -> np.ndarray:
    for label, vector in axes_by_site[site]:
        if label == axis:
            return np.asarray(vector, dtype=float)
    raise AnchorError(f"site {site!r} declares no axis {axis!r}")


def run_frozen_response(
    bundle: SpiralBundle,
    reference: dict,
    orbitals_by_site: dict,
    axes_by_site: dict,
    kpoints_cart,
    kweights,
    axes,
    sites,
    moment_signs: dict,
    magnetic_axis=MAGNETIC_AXIS,
    operator: str = DEFAULT_OPERATOR_MODE,
):
    """Run ``band_observables`` per transverse axis and project stored J.

    ``operator`` selects the rotating operator passed to the vertex
    extraction (``onsite_delta`` default per ADR-004 2026-10-14,
    ``weighted_total_h`` diagnostic variant).
    """
    n_occupied = int(round(float(reference["electron_count"])))
    vertex = extract_magnetic_vertex(
        bundle, orbitals_by_site=orbitals_by_site, operator=operator
    )
    problem = FrozenResponseProblem(bundle, vertex, axes_by_site=axes_by_site)
    kpoints = [np.asarray(k, dtype=float) for k in kpoints_cart]
    weights = [float(w) for w in kweights]
    gap_ev = _mesh_gap_ev(problem, kpoints, n_occupied)
    if not gap_ev > 0.0:
        raise AnchorError(
            f"comparison mesh is not gapped (min gap {gap_ev:.3e} eV); the "
            "gapped fixed-occupation ensemble refuses the input"
        )
    ensemble = Ensemble(
        kind=ENSEMBLE_GAPPED_T0,
        n_occupied=n_occupied,
        electron_count=float(reference["electron_count"]),
        fermi_energy_ev=float(reference["ef"]),
        smearing_width_ev=float(reference["temperature"]) * KB_EV,
        gap_ev=gap_ev,
    )
    n_hat = np.asarray(magnetic_axis, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    response_by_axis = {}
    j_stored_by_axis = {}
    for axis in axes:
        coords = tuple((site, axis) for site in sites)
        obs = problem.band_observables(kpoints, weights, ensemble, coords)
        response_by_axis[axis] = obs
        t_vec = _axis_vector(axes_by_site, sites[0], axis)
        u_vec = np.cross(t_vec, n_hat)
        u_dot = float(np.dot(u_vec, u_vec))
        j_axis = {}
        for a in sites:
            for b in sites:
                if a == b:
                    continue
                key = ((a, axis), (b, axis))
                k_ab = float(obs["curvature"][key])
                # MathDoc: tb2j/collinear-anchor-projection#stored_ordered_pair_j
                denominator = 2.0 * moment_signs[a] * moment_signs[b] * u_dot
                j_stored = -k_ab / denominator
                j_axis[f"{a}|{b}"] = {
                    "K_ab_eV_per_rad2": k_ab,
                    "onsite_eV_per_rad2": float(obs["onsite"][key]),
                    "response_eV_per_rad2": float(obs["response"][key]),
                    "axis_vector": [float(v) for v in t_vec],
                    "u_vector": [float(v) for v in u_vec],
                    "u_dot": u_dot,
                    "j_pair_once_eV": 2.0 * j_stored,
                    "j_stored_eV": j_stored,
                    "j_stored_meV": 1e3 * j_stored,
                }
        j_stored_by_axis[axis] = j_axis
    return problem, ensemble, response_by_axis, j_stored_by_axis


# ---------------------------------------------------------------------------
# certificate assembly
# ---------------------------------------------------------------------------
def _coord_text(coord) -> str:
    return json.dumps([coord[0], coord[1]])


def _pair_text(pair) -> str:
    return json.dumps([[pair[0][0], pair[0][1]], [pair[1][0], pair[1][1]]])


def _raw_response_record(obs: dict) -> dict:
    return {
        "gradient": {_coord_text(c): float(v) for c, v in obs["gradient"].items()},
        "curvature": {_pair_text(p): float(v) for p, v in obs["curvature"].items()},
        "onsite": {_pair_text(p): float(v) for p, v in obs["onsite"].items()},
        "response": {_pair_text(p): float(v) for p, v in obs["response"].items()},
    }


def _heisenberg_closure(response_by_axis: dict, sites) -> dict:
    """Diagonal closure diagnostic: K_aa + sum_{b!=a} K_ab (model-only)."""
    out = {}
    for axis, obs in response_by_axis.items():
        per_axis = {}
        for site in sites:
            total = obs["curvature"][((site, axis), (site, axis))]
            for other in sites:
                if other != site:
                    total += obs["curvature"][((site, axis), (other, axis))]
            per_axis[site] = float(total)
        out[axis] = per_axis
    return {
        "values_eV_per_rad2": out,
        "statement": (
            "a pure pairwise Heisenberg model obeys K_aa + sum_{b!=a} K_ab = 0; "
            "a frozen band energy need not satisfy this model closure, so it "
            "is recorded as a diagnostic without a gate"
        ),
    }


def _settings_record(
    *,
    operator_mode,
    requested_kmesh,
    actual_kmesh,
    ensemble_record,
    legacy_smearing_ev,
    legacy_contour_nodes,
    legacy_initial_emin_ev,
    legacy_adjusted_emin_ev,
    legacy_engine_emin_ev,
) -> dict:
    if operator_mode == OPERATOR_MODE_ONSITE_DELTA:
        operator = {
            "operator_mode": operator_mode,
            "new": (
                "onsite delta (ADR-004 2026-10-14): only the R=0 same-site "
                "blocks of the total-H spin splitting rotate, each at full "
                "weight w = 1; intersite spin-dependent blocks are not "
                "rotated"
            ),
            "legacy": (
                "onsite R=0 orbital blocks of Delta = H_up - H_dn only; "
                "intersite spin-dependent blocks are not rotated"
            ),
            "matched": True,
            "note": (
                "matched magnetic operators (ADR-004 default): both engines "
                "rotate only onsite R=0 same-site blocks"
            ),
        }
    else:
        operator = {
            "operator_mode": operator_mode,
            "new": (
                "weighted incident total-H magnetic operator: every stored "
                "block incident to a site partition, each contribution "
                "conjugated by the full site unitary with w = 1/count of "
                "distinct endpoint owners"
            ),
            "legacy": (
                "onsite R=0 orbital blocks of Delta = H_up - H_dn only; "
                "intersite spin-dependent blocks are not rotated"
            ),
            "matched": False,
            "note": (
                "non-identical magnetic operators (projector definitions "
                "differ); recorded as unmatched, never absorbed into the "
                "numbers"
            ),
        }
    integration = {
        "new": (
            "exact occupied-band Hessian on the k mesh (gapped T=0 step " "occupations)"
        ),
        "legacy": (
            "LKAG contour integration: CFR imaginary-axis continued-fraction "
            f"Fermi pole sum with nz={legacy_contour_nodes} nodes"
        ),
        "matched": False,
        "note": "distinct integration methods; recorded as unmatched",
    }
    band_window = {
        "tbgreen_initial_emin_ev": float(legacy_initial_emin_ev),
        "tbgreen_adjusted_emin_ev": float(legacy_adjusted_emin_ev),
        "engine_adjusted_emin_ev": float(legacy_engine_emin_ev),
        "mechanism": (
            "band-truncation diagnostics read from the run objects: "
            "ExchangeCL2 builds TBGreen with its own initial_emin default "
            "and derives the adjusted window via _adjust_emin "
            "(find_energy_ingap); the anchor exposes no cutoff control, and "
            "none of these values is a CFR contour endpoint"
        ),
        "matched": None,
    }
    k_mesh = {
        "declared": [int(v) for v in requested_kmesh],
        "actual": [int(v) for v in actual_kmesh],
        "matched": [int(v) for v in requested_kmesh] == [int(v) for v in actual_kmesh],
        "note": (
            "ExchangeCL2 oddizes the requested mesh (n//2*2+1); both engines "
            "consume exactly this oddized mesh, the new side samples "
            "ex.Gup.kpts converted to Cartesian inverse Bohr"
        ),
    }
    ensemble = {
        "new": ensemble_record,
        "legacy": {
            "efermi_ev": 0.0,
            "occupation_smearing_ev": float(legacy_smearing_ev),
            "contour_nodes": int(legacy_contour_nodes),
        },
        "matched": False,
        "matches_new_only": None,
        "note": (
            "gapped fixed-occupation T=0 step occupations vs fermi-smoothed "
            "contour occupations; distinct ensembles, recorded unmatched"
        ),
    }
    j_convention = {
        "actual": (
            "tb2j_ordered_pair_stored on both engines; the pair-once physical "
            "value is twice the stored value; no moment^2 division"
        ),
        "matched": None,
    }
    k_phase_gauge = {
        "new": "exp(-i k . xij) with the full intra-cell orbital displacement",
        "legacy": (
            "exp(+2 pi i k_frac . R) on integer cell vectors only "
            "(LCAOHamiltonian convention 2, no intra-cell displacement)"
        ),
        "matched": False,
        "note": "distinct H(k) phase gauges; recorded as an actual setting",
    }
    return {
        "operator": operator,
        "integration": integration,
        "band_window": band_window,
        "k_mesh": k_mesh,
        "ensemble": ensemble,
        "j_convention": j_convention,
        "k_phase_gauge": k_phase_gauge,
    }


_DECLARATION_FIELDS = {
    "kmesh": "k_mesh",
    "j_convention": "j_convention",
    "ensemble": "ensemble",
    "projector": "operator",
    "integration": "integration",
}


def _declaration_matches(name: str, declared, settings: dict) -> bool:
    if name == "k_mesh":
        return [int(v) for v in declared] == settings["k_mesh"]["actual"]
    if name == "j_convention":
        return declared == "tb2j_ordered_pair_stored"
    if name == "ensemble":
        # The declared ensemble can at most describe the new side; the legacy
        # side runs a fermi-smoothed contour ensemble, so it can never match.
        settings["ensemble"]["matches_new_only"] = (
            declared == settings["ensemble"]["new"]["kind"]
        )
        return False
    if name in ("integration",):
        # Intrinsically different definitions; no declaration can match them.
        return False
    if name == "operator":
        # ADR-004: the declaration matches only when it names the actually
        # recorded operator_mode of the run.
        return str(declared) == settings["operator"].get("operator_mode")
    raise AnchorError(f"unknown declaration field {name!r}")


def apply_declaration(
    settings: dict,
    declaration: AnchorDeclaration,
    *,
    refuse_on_mismatch: bool = False,
) -> list:
    """Fold user declarations into the settings record, truthfully.

    Returns the sorted names of declared settings whose declared value does
    not match the actual one (or whose actual values are intrinsically
    unmatched). With ``refuse_on_mismatch`` the mismatch is refused instead
    of only reported.

    The operator entry is special: its ``matched`` flag records the
    cross-engine comparison (onsite_delta vs the LKAG onsite delta is
    matched; weighted_total_h vs the LKAG onsite delta is genuinely
    unmatched) and is never overwritten by a declaration. A projector
    declaration only records ``declaration_matched`` -- whether it names
    the operator_mode actually selected -- and the entry is reported
    unmatched when either the engines are genuinely unmatched or the
    declaration names another mode, so ``refuse_on_mismatch`` still
    refuses a genuinely unmatched engine pair.
    """
    unmatched = []
    for attr, name in _DECLARATION_FIELDS.items():
        value = getattr(declaration, attr)
        if value is None:
            continue
        entry = settings[name]
        entry["declared"] = [int(v) for v in value] if name == "k_mesh" else str(value)
        if name == "operator":
            entry["declaration_matched"] = bool(
                _declaration_matches(name, value, settings)
            )
            if not entry["matched"] or not entry["declaration_matched"]:
                unmatched.append(name)
            continue
        entry["matched"] = bool(_declaration_matches(name, value, settings))
        if not entry["matched"]:
            unmatched.append(name)
    unmatched = sorted(set(unmatched))
    if refuse_on_mismatch and unmatched:
        raise AnchorSettingMismatchError(
            "declared settings do not match the actual ones: "
            f"{unmatched}; the anchor refuses to present the comparison "
            "under unmatched declarations"
        )
    return unmatched


def _payload_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _oddize_kmesh(kmesh) -> list:
    """Mirror ExchangeCL._prepare_kmesh: n//2*2 + 1 per dimension."""
    return [int(n) // 2 * 2 + 1 for n in kmesh]


def _require_pair_record(record: dict, sites) -> None:
    """Refuse a missing or non-reciprocal one-directed legacy pair record.

    A missing forward direction (e.g. Rcut below the bond, wrong magnetic
    elements) or an incomplete reciprocity audit must stop the comparison;
    reporting J = 0 for an absent pair would be a fabricated number.
    """
    if not record["forward_entries"]:
        raise AnchorError(
            f"no directed ({sites[0]}, {sites[1]}) entries in exchange_Jdict; "
            "the one-directed pair value is unavailable (check Rcut and "
            "magnetic_elements) and must not be reported as zero"
        )
    if not record["reciprocity_complete"]:
        raise AnchorError(
            "reflected (-R, j, i) entries are missing for the site pair; "
            "the one-directed sum cannot be audited for reciprocity and is "
            "refused rather than reported"
        )


def _validate_transverse_axes(axes, magnetic_axis) -> None:
    """Reject rotation axes that are not transverse to the magnetic axis.

    The stored-J projection divides by ``u.u`` with ``u = t x n``; a
    parallel (or near-parallel) axis makes ``u`` vanish or blow up the
    projection, so the refusal happens before any engine runs.
    """
    n_hat = np.asarray(magnetic_axis, dtype=float)
    norm = np.linalg.norm(n_hat)
    if norm == 0.0:
        raise AnchorError("magnetic_axis must be a nonzero vector")
    n_hat = n_hat / norm
    for axis in axes:
        u_vec = np.cross(np.asarray(_AXIS_VECTORS[axis], dtype=float), n_hat)
        if float(np.linalg.norm(u_vec)) < 1.0 - AXIS_TRANSVERSE_TOL:
            raise AnchorError(
                f"rotation axis {axis!r} is not transverse to the magnetic "
                f"axis {tuple(float(v) for v in n_hat)}; the u = t x n "
                "projection denominator would vanish, refusing before any "
                "engine runs"
            )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_anchor_comparison(
    fixture_npz,
    fdf_path,
    *,
    sites=("O1", "O2"),
    axes=DEFAULT_AXES,
    magnetic_axis=MAGNETIC_AXIS,
    kmesh=DEFAULT_KMESH,
    nz=DEFAULT_NZ,
    rcut_ang=DEFAULT_RCUT_ANG,
    magnetic_elements=("O",),
    operator: str = DEFAULT_OPERATOR_MODE,
    declaration: AnchorDeclaration | None = None,
    refuse_on_mismatch: bool = False,
    collinear_tol_ev: float = COLLINEAR_TOL_EV,
) -> AnchorComparison:
    """Run both engines on the same NC SCF and assemble the certificate.

    ``operator`` selects the rotating operator of the new side
    (``onsite_delta`` default per ADR-004 2026-10-14, ``weighted_total_h``
    diagnostic variant); it is recorded as ``operator_mode`` in the
    certificate and the settings record.

    Returns an :class:`AnchorComparison`; the certificate is numbers-only
    (``agreement_status = "not_evaluated"``) with unmatched settings
    reported, never absorbed. Refusals before any engine run: an unknown
    operator mode, non-transverse axes, non-x-collinear components, and
    failed same-SCF spectral equivalence on the recorded kpoints and the
    gated comparison mesh.
    """
    if operator not in OPERATOR_MODES:
        raise AnchorError(
            f"unknown magnetic operator {operator!r}; expected one of "
            f"{sorted(OPERATOR_MODES)}"
        )
    if len(sites) != 2 or len(set(sites)) != 2:
        raise AnchorError("the anchor compares exactly two distinct sites")
    axes = tuple(axes)
    for axis in axes:
        if axis not in _AXIS_VECTORS:
            raise AnchorError(
                f"axis {axis!r} unsupported; known: {sorted(_AXIS_VECTORS)}"
            )
    _validate_transverse_axes(axes, magnetic_axis)

    bundle, reference = load_fixture(Path(fixture_npz))
    channels = build_collinear_spin_channels(bundle, tol_ev=collinear_tol_ev)

    # SPEC-001: the same-SCF spectrum equivalence is a gate, not a note. It
    # covers the recorded producer kpoints and the predicted oddized
    # comparison mesh, and it runs BEFORE the legacy engine so a broken
    # channel split is refused without spending the LKAG run.
    expected_kmesh = _oddize_kmesh(kmesh)
    k_mesh_fractional = monkhorst_pack(expected_kmesh, gamma_center=False)
    k_mesh_cart = _fractional_to_cartesian_inv_bohr(k_mesh_fractional, bundle.cell)
    recorded_check = spectral_equivalence_check(bundle, reference["kp"])
    mesh_check = spectral_equivalence_check(bundle, k_mesh_cart)
    if not (recorded_check["pass"] and mesh_check["pass"]):
        raise AnchorError(
            "same-SCF spectral equivalence failed (recorded kpoints max dev "
            f"{recorded_check['max_dev_ev']:.3e} eV, comparison mesh max dev "
            f"{mesh_check['max_dev_ev']:.3e} eV, budget "
            f"{SPECTRAL_EQUIVALENCE_BUDGET_EV:.1e} eV); the channel split "
            "does not reproduce the NC spinor spectrum and the same-SCF "
            "certificate is refused"
        )

    parser = _load_parser(fdf_path)
    labels = _plain_orbital_labels(parser.orbs)
    if len(labels) != channels.norb:
        raise AnchorError(
            f"parsed basis has {len(labels)} spatial orbitals but the bundle "
            f"stores {channels.norb}"
        )
    same_scf = _check_same_scf_geometry(parser, bundle, reference)

    names = list(symbol_number(parser.atoms).keys())
    for site in sites:
        if site not in names:
            raise AnchorError(f"site {site!r} is not an atom of the structure {names}")
    spinor_counts = {idx: len(parser.orb_dict[idx]) // 2 for idx in range(len(names))}
    orbitals_by_site = {}
    start = 0
    for idx in range(len(names)):
        site = names[idx]
        orbitals_by_site[site] = list(range(start, start + spinor_counts[idx]))
        start += spinor_counts[idx]
    if start != channels.norb:
        raise AnchorError("parsed orbital counts do not partition the bundle basis")
    assignment = _check_orbital_site_assignment(
        bundle,
        parser.geom.xyz,
        orbitals_by_site,
        dict(symbol_number(parser.atoms)),
    )
    same_scf.update(assignment)
    moment_signs = _site_moment_signs(bundle, orbitals_by_site)

    exchange = run_legacy_exchange(
        channels,
        parser,
        labels,
        nel=same_scf["parsed_electron_count"],
        kmesh=kmesh,
        nz=nz,
        rcut_ang=rcut_ang,
        magnetic_elements=magnetic_elements,
    )
    k_fractional = np.asarray(exchange.Gup.kpts, dtype=float)
    kweights = np.asarray(exchange.Gup.kweights, dtype=float)
    actual_kmesh = [int(v) for v in exchange.kmesh]
    if actual_kmesh != expected_kmesh or not np.allclose(
        k_fractional, k_mesh_fractional, atol=1e-12, rtol=0.0
    ):
        raise AnchorError(
            "the engine's actual k mesh deviates from the gated comparison "
            f"mesh (actual {actual_kmesh}, gated {expected_kmesh}); refusing "
            "a certificate whose mesh was not the spectrally checked one"
        )
    k_cart = _fractional_to_cartesian_inv_bohr(k_fractional, bundle.cell)

    axes_by_site = {
        site: tuple((axis, _AXIS_VECTORS[axis]) for axis in axes) for site in sites
    }
    problem, ensemble, response_by_axis, j_stored_by_axis = run_frozen_response(
        bundle,
        reference,
        orbitals_by_site,
        axes_by_site,
        k_cart,
        kweights,
        axes,
        sites,
        moment_signs,
        magnetic_axis=magnetic_axis,
        operator=operator,
    )
    if problem.operator_mode != operator:
        raise AnchorError(
            "the frozen-response problem recorded operator_mode "
            f"{problem.operator_mode!r} instead of the requested "
            f"{operator!r}; refusing a certificate whose rotating operator "
            "was not the declared one"
        )

    legacy_pair = image_summed_pair_J(exchange, *sites)
    _require_pair_record(legacy_pair, sites)
    legacy_sum = float(legacy_pair["forward_sum_ev"])
    per_axis_numbers = {}
    for axis in axes:
        pair_key = f"{sites[0]}|{sites[1]}"
        new_stored = float(j_stored_by_axis[axis][pair_key]["j_stored_eV"])
        legacy = float(legacy_sum)
        per_axis_numbers[axis] = {
            "j_stored_new_eV": new_stored,
            "j_stored_new_meV": 1e3 * new_stored,
            "j_stored_legacy_image_sum_eV": legacy,
            "j_stored_legacy_image_sum_meV": 1e3 * legacy,
            "difference_legacy_minus_new_eV": legacy - new_stored,
            "difference_legacy_minus_new_meV": 1e3 * (legacy - new_stored),
        }
    spread = (
        abs(
            per_axis_numbers[axes[0]]["j_stored_new_eV"]
            - per_axis_numbers[axes[1]]["j_stored_new_eV"]
        )
        if len(axes) == 2
        else None
    )

    settings = _settings_record(
        operator_mode=operator,
        requested_kmesh=kmesh,
        actual_kmesh=actual_kmesh,
        ensemble_record=ensemble.to_record(),
        legacy_smearing_ev=float(exchange.smearing),
        legacy_contour_nodes=int(exchange.nz),
        legacy_initial_emin_ev=float(exchange.Gup.initial_emin),
        legacy_adjusted_emin_ev=float(exchange.Gup.adjusted_emin),
        legacy_engine_emin_ev=float(exchange.emin),
    )
    declared_unmatched = apply_declaration(settings, declaration or AnchorDeclaration())
    if refuse_on_mismatch and declared_unmatched:
        raise AnchorSettingMismatchError(
            "declared settings do not match the actual ones: "
            f"{declared_unmatched}; refusing to present the comparison"
        )
    unmatched_all = sorted(
        name for name, entry in settings.items() if entry.get("matched") is False
    )

    vertex = problem.vertex
    payload = {
        "anchor_version": ANCHOR_VERSION,
        "comparison_kind": COMPARISON_KIND,
        "agreement_status": AGREEMENT_STATUS,
        "operator_mode": operator,
        "agreement_statement": (
            "numbers only (user decision 2026-10-13): no agreement threshold "
            "exists and no AC-006/NFR-005 compatibility is claimed or denied"
        ),
        "conventions": dict(_CONVENTIONS),
        "same_scf": {
            "source_bundle_sha256": vertex.certificate["source_bundle_sha256"],
            "fixture_npz": Path(fixture_npz).name,
            "fdf": Path(fdf_path).name,
            "geometry_checks": same_scf,
            "channel_split": channels.channel_split,
            "site_moment_signs": {k: float(v) for k, v in moment_signs.items()},
            "magnetic_axis": [float(v) for v in np.asarray(magnetic_axis, dtype=float)],
            "note": (
                "both engines consume the H and S of this single noncollinear "
                "SCF; no independent twin SCF enters any number here"
            ),
        },
        "spectral_equivalence": {
            "recorded_producer_kpoints": recorded_check,
            "comparison_mesh": mesh_check,
            "gated": True,
            "gate_statement": (
                "both checks gate the certificate: a failed channel-split "
                "spectrum equivalence refuses the comparison before any "
                "engine output is reported"
            ),
        },
        "settings": settings,
        "unmatched_settings": unmatched_all,
        "declared_mismatched_settings": declared_unmatched,
        "actual_k_mesh": {
            "declared_request": [int(v) for v in kmesh],
            "oddized_actual": actual_kmesh,
            "kpoints_fractional": [[float(v) for v in k] for k in k_fractional],
            "kpoints_cartesian_inv_bohr": [[float(v) for v in k] for k in k_cart],
            "kweights": [float(w) for w in kweights],
        },
        "ensemble_record": ensemble.to_record(),
        "new_response_raw": {
            axis: _raw_response_record(response_by_axis[axis]) for axis in axes
        },
        "heisenberg_closure": _heisenberg_closure(response_by_axis, sites),
        "legacy_response": {
            "n_jdict_keys": int(len(exchange.exchange_Jdict)),
            "pair_image_sum_eV": legacy_sum,
            "pair_image_sum_meV": 1e3 * legacy_sum,
            "reverse_direction_sum_eV": float(legacy_pair["reverse_sum_ev"]),
            "reverse_direction_sum_meV": 1e3 * float(legacy_pair["reverse_sum_ev"]),
            "reciprocity_complete": bool(legacy_pair["reciprocity_complete"]),
            "reciprocity_max_abs_diff_ev": float(
                legacy_pair["reciprocity_max_abs_diff_ev"]
            ),
            "pair_entries": legacy_pair["forward_entries"],
            "reverse_entries": legacy_pair["reverse_entries"],
            "convention_note": (
                "the pair value is the R-image sum over ONE directed "
                "(O1, O2); each stored direction carries J_pair_bond/2, so "
                "the reverse direction is a same-bond audit trail and is "
                "never summed in"
            ),
        },
        "numeric_comparison": {
            "site_pair": list(sites),
            "per_axis": per_axis_numbers,
            "axis_symmetry_spread_new_eV": spread,
            "statement": (
                "raw gradient/curvature and the onsite/response split are "
                "preserved verbatim alongside the mapped exchange; the "
                "difference is a computed number, not an agreement verdict"
            ),
        },
    }
    payload["content_sha256"] = _payload_digest(payload)

    return AnchorComparison(
        certificate=payload,
        channels=channels,
        legacy_exchange=exchange,
        problem=problem,
        ensemble=ensemble,
        kpoints_fractional=k_fractional,
        kpoints_cartesian_inv_bohr=k_cart,
        kweights=kweights,
        response_by_axis=response_by_axis,
        j_stored_by_axis=j_stored_by_axis,
        legacy_pair_sum_ev=legacy_sum,
        legacy_pair=legacy_pair,
        settings=settings,
    )
