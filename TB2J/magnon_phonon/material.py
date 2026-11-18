"""Material (monolayer CrI3) bare and coupled spectral run (Story 013).

This module glues the certified Story-009 vertex seam and the certified
Story-010 Born kernels to the qualified material derivative stores.  It
contains no new physics: every number flows through
:class:`TB2J.magnon_phonon.vertex.MagnonPhononModel` and the kernels in
:mod:`TB2J.magnon_phonon.scattering`.

Declared coverage (Phase 1 research, story-013 Technical Notes):

- exchange derivatives: Story-001 six directed Cr1-Cr2 nearest-neighbour
  bonds; finite-q mode-projected S002 reports from the s013 closing
  campaign (generated ON the qualified relaxed s012-h005-3x reference with
  its mode eigenvectors) at Gamma (3 blocks) and K (4 blocks), all
  elementwise FD gates passed (reports/loader-gates.json); M=(1/2,0,0) is
  declared incommensurate with the 3x supercell and is NOT covered (the
  M coupled channel is absent and the bare-phonon M panel stays
  qualified-dataset-free);
- the CrI3 magnon model is the declared first-neighbour ferromagnet
  (J0 = 1.5292416 meV, Cr3+ S = 3/2) built from the same bond set;
- coupled scattering blocks exist ONLY at hash-joined (q, mode) pairs whose
  table eigenvalue exceeds the declared infrared cutoff; no interpolation
  is performed or claimed anywhere coverage is absent.

The harmonic one-phonon/one-magnon mixing input (SOC mixed-torque or tensor
finite differences) does not exist as a qualified dataset; the hybrid path
is never run here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon_phonon.loaders import load_s002_report
from TB2J.magnon_phonon.scattering import (
    _APPROXIMATION,
    bose_occupation,
    phonon_hot_band_kernel,
    thz_to_ev,
)
from TB2J.magnon_phonon.vertex import (
    MagnonPhononModel,
    ModeProjectedDerivatives,
    PhononModeInput,
)

#: sqrt-unit conversion eV/Ang^2/amu -> THz, validated against the Story-008
#: qualification mapping (raw Gamma acoustic eigenvalue 9.053e-05 -> 0.148747 THz)
FREQUENCY_FACTOR_THZ = 15.633304239856194
#: declared infrared cutoff: covered modes below this table eigenvalue
#: (eV/Ang^2/amu, i.e. below ~0.5 THz) are excluded from the coupled run
#: because their zero-point treatment is infrared-divergent or noise-floor
IR_CUTOFF_EV_PER_ANG2_AMU = 1.0e-3
#: magnetic reference: Cr3+ moment (mu_B) of the SIESTA collinear FM reference
CRI3_MOMENT_MUB = 3.0
#: first-neighbour J0 of the declared magnon model (Story-001 reference
#: constant, unchanged across the closing-campaign source swap)
CRI3_J0_MEV = 1.529241586
#: Story-001 directed Cr1-Cr2 nearest-neighbour bond set (i, j, R)
CRI3_BONDS = (
    (0, 1, (-1, 0, 0)),
    (0, 1, (0, 0, 0)),
    (0, 1, (0, 1, 0)),
    (1, 0, (0, -1, 0)),
    (1, 0, (0, 0, 0)),
    (1, 0, (1, 0, 0)),
)
#: S001 primitive cell (Angstrom rows), hexagonal monolayer with vacuum
CRI3_CELL = np.array(
    [[6.867, 0.0, 0.0], [-3.4335, 5.9469964478, 0.0], [0.0, 0.0, 20.0]]
)
#: run settings declared for the material run
DEFAULT_TEMPERATURE_K = 30.0
DEFAULT_BROADENING_EV = 5.0e-5
DEFAULT_OMEGA_MEV = (-2.0, 30.0, 3201)
#: harmonic mixing statement carried by every artifact
MIXING_UNAVAILABLE_STATEMENT = (
    "harmonic one-phonon/one-magnon mixing input is UNAVAILABLE: no qualified "
    "SOC mixed-torque or tensor finite-difference dataset exists (Story-012 "
    "qualified forces only); the hybrid path is not run on material data and "
    "no substitution is made"
)
THERMAL_PRESCRIPTION = (
    "2D Mermin-Wagner: the isotropic 2D ferromagnet has no finite-temperature "
    "ordered reference; monolayer CrI3 orders below Tc ~ 45 K (Huang et al., "
    "Nature 2017) through out-of-plane single-ion anisotropy. The qualified "
    "stores contain no anisotropy derivative, so the anisotropy gap is NOT "
    "assigned a number and no tiny-gap ordering claim is made. Boundedness is "
    "demonstrated operationally: the declared internal meshes (K, M) carry a "
    "finite magnon gap, so every Bose occupation entering the kernels is "
    "finite; the divergent Gamma occupation is never sampled."
)


@dataclass(frozen=True)
class MaterialSettings:
    """Declared numerical settings of the material run."""

    temperature_K: float = DEFAULT_TEMPERATURE_K
    broadening_eV: float = DEFAULT_BROADENING_EV
    omega_min_meV: float = DEFAULT_OMEGA_MEV[0]
    omega_max_meV: float = DEFAULT_OMEGA_MEV[1]
    omega_points: int = DEFAULT_OMEGA_MEV[2]

    def omega_grid_eV(self):
        return np.linspace(
            self.omega_min_meV * 1e-3, self.omega_max_meV * 1e-3, self.omega_points
        )

    def as_dict(self):
        return {
            "temperature_K": self.temperature_K,
            "broadening_eV": self.broadening_eV,
            "omega_grid_meV": [
                self.omega_min_meV,
                self.omega_max_meV,
                self.omega_points,
            ],
        }


@dataclass(frozen=True)
class CoveredBlock:
    """One hash-joined (q, mode) coupled-scattering block."""

    label: str
    campaign_key: str
    campaign_id: str
    mode: str
    q: np.ndarray
    mode_index: int
    frequency_thz: float
    eigenvalue_eV_per_Ang2_per_amu: float
    polarization: np.ndarray  # (natom, 3) complex, unit Cartesian norm
    masses_amu: np.ndarray
    derivatives: ModeProjectedDerivatives
    evidence_sha256: str
    modes_table_sha256: str
    max_fd_error_meV_per_Ang: float
    j0_eV: float
    #: vault path of the hash-joined mode-evidence file ("" for fixture blocks)
    evidence_path: str = ""

    @property
    def natom(self):
        return len(self.masses_amu)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cri3_magnon_model(j0_eV: float = CRI3_J0_MEV * 1e-3) -> Magnon:
    """Declared first-neighbour FM magnon model of monolayer CrI3.

    ``JR[ir, i, j] = J0 * I3`` in the full-moment ``JR`` convention (as in
    the Story-009 tests), spins S = 3/2 from the 3.0 mu_B collinear
    reference, bonds exactly :data:`CRI3_BONDS`.
    """
    bonds = CRI3_BONDS
    nspin = 2
    moment = CRI3_MOMENT_MUB
    spin = moment / 2.0
    rlist = np.array([bond[2] for bond in bonds], dtype=int)
    jr = np.zeros((len(bonds), nspin, nspin, 3, 3))
    for ir, (i, j, _r) in enumerate(bonds):
        jr[ir, i, j] = np.eye(3) * j0_eV * spin * spin
    magmoms = np.zeros((nspin, 3))
    magmoms[:, 2] = moment
    magnon = Magnon(
        nspin=nspin,
        magmom=magmoms,
        Rlist=rlist,
        JR=jr,
        cell=CRI3_CELL.copy(),
        _Q=np.zeros(3),
        _uz=np.array([[0.0, 0.0, 1.0]]),
        _n=np.array([0.0, 1.0, 0.0]),
    )
    magnon.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    return magnon


#: primitive symbols/site order of the det campaigns (sites 0,1 = Cr)
CRI3_SYMBOLS = ("Cr", "Cr", "I", "I", "I", "I", "I", "I")


def _ase_masses(symbols):
    from ase import Atoms

    return Atoms(symbols=list(symbols)).get_masses()


def load_cri3_blocks(
    s002_store_path,
    evidence_dir,
    mode_tables,
    ir_cutoff=IR_CUTOFF_EV_PER_ANG2_AMU,
):
    """Load every gate-passed, hash-joined covered (q, mode) block.

    ``s002_store_path`` is the closing campaign's matched finite-q rows
    report; every block is hash-verified against its mode-evidence file and
    the sha-pinned closing modes table before it is accepted.

    ``mode_tables`` maps a modes-table sha256 to its JSON path (the tables
    referenced by the evidence ``modes_hash``).  Blocks whose table
    eigenvalue lies below ``ir_cutoff`` are skipped and reported under the
    declared infrared cutoff; nothing else is filtered.
    """
    store_path = Path(s002_store_path)
    store = json.loads(store_path.read_text())
    evidence_dir = Path(evidence_dir)
    masses = _ase_masses(CRI3_SYMBOLS)
    table_cache = {}
    blocks = []
    skipped = []
    for campaign_key, campaign_record in store["campaigns"].items():
        campaign_id = campaign_record["campaign"]["campaign_id"]
        for entry in campaign_record["reports"]:
            evidence_path = evidence_dir / entry["source_evidence"]
            evidence_sha = sha256_file(evidence_path)
            if evidence_sha != entry["source_sha256"]:
                raise ValueError(f"evidence hash mismatch for {evidence_path}")
            evidence = json.loads(evidence_path.read_text())
            derivatives = load_s002_report(
                store_path,
                campaign=campaign_key,
                mode=entry["mode"],
                expected_bonds=CRI3_BONDS,
                mode_evidence_path=evidence_path,
            )
            table_hash = evidence["modes_hash"]
            if table_hash not in mode_tables:
                raise ValueError(f"no modes table supplied for {table_hash}")
            table_path = Path(mode_tables[table_hash])
            if sha256_file(table_path) != table_hash:
                raise ValueError(f"modes table hash mismatch for {table_path}")
            if table_hash not in table_cache:
                table_cache[table_hash] = json.loads(table_path.read_text())
            table = table_cache[table_hash]
            index = int(evidence["mode_eigenvector_index"])
            selected = [
                m for m in table["modes"] if int(m["eigenvector_index"]) == index
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"modes table {table_path} has no unique mode index {index}"
                )
            eigenvalue = float(selected[0]["eigenvalue_eV_per_Ang2_per_amu"])
            label = f"{campaign_key}:{entry['mode']}"
            if eigenvalue < ir_cutoff:
                skipped.append(
                    {
                        "label": label,
                        "eigenvalue_eV_per_Ang2_per_amu": eigenvalue,
                        "reason": "below the declared infrared cutoff",
                    }
                )
                continue
            if eigenvalue <= 0.0:
                raise ValueError(f"negative covered-mode eigenvalue for {label}")
            polarization = np.asarray(
                evidence["polarization_real"], float
            ) + 1j * np.asarray(evidence["polarization_imag"], float)
            blocks.append(
                CoveredBlock(
                    label=label,
                    campaign_key=campaign_key,
                    campaign_id=campaign_id,
                    mode=entry["mode"],
                    q=derivatives.q.copy(),
                    mode_index=index,
                    frequency_thz=FREQUENCY_FACTOR_THZ * np.sqrt(eigenvalue),
                    eigenvalue_eV_per_Ang2_per_amu=eigenvalue,
                    polarization=polarization,
                    masses_amu=masses.copy(),
                    derivatives=derivatives,
                    evidence_sha256=evidence_sha,
                    modes_table_sha256=table_hash,
                    max_fd_error_meV_per_Ang=float(
                        np.max(np.abs(derivatives.O_pattern - derivatives.FD_pattern))
                    )
                    * 1e3,
                    j0_eV=CRI3_J0_MEV * 1e-3,
                    evidence_path=str(evidence_path),
                )
            )
    return blocks, skipped


def phonon_mode_input(block: CoveredBlock) -> PhononModeInput:
    """Single-mode :class:`PhononModeInput` hash-joined to the S002 report.

    The eigenvector column at ``mode_index`` is the mass-weighted pattern of
    the report's unit-Cartesian polarization; all other columns are zero and
    never consumed (the vertex slices the declared ``mode_index`` only).  The
    provenance carries the exact keys the vertex join enforces.
    """
    natom = block.natom
    n_modes = block.mode_index + 1
    eigenvectors = np.zeros((3 * natom, n_modes), dtype=complex)
    eigenvectors[:, block.mode_index] = block.polarization.reshape(
        3 * natom
    ) * np.repeat(np.sqrt(block.masses_amu), 3)
    frequencies = np.full(n_modes, np.nan)
    frequencies[block.mode_index] = block.frequency_thz
    return PhononModeInput(
        q=np.asarray(block.q, dtype=float).copy(),
        frequencies_thz=frequencies,
        masses_amu=np.asarray(block.masses_amu, dtype=float).copy(),
        eigenvectors=eigenvectors,
        provenance={
            "mode_evidence_sha256": block.evidence_sha256,
            "mode": block.mode,
            "campaign_key": block.campaign_key,
            "campaign_id": block.campaign_id,
            "modes_table_sha256": block.modes_table_sha256,
            "story": "013",
        },
    )


def vertex_block(magnon, block: CoveredBlock, k):
    """Materialize the certified vertex at external momentum ``k``."""
    model = MagnonPhononModel(magnon, block.derivatives)
    return model.vertex(
        k,
        np.asarray(block.q, dtype=float),
        phonon_modes=phonon_mode_input(block),
        mode_index=block.mode_index,
    )


def bare_magnon_path(n_points: int = 61):
    """Declared Gamma-K-M-Gamma sampling (fractional coordinates)."""
    gamma = np.zeros(3)
    k_point = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
    m_point = np.array([0.5, 0.0, 0.0])
    segments = (
        (gamma, k_point, "Gamma-K"),
        (k_point, m_point, "K-M"),
        (m_point, gamma, "M-Gamma"),
    )
    points = []
    labels = []
    xlist = []
    x = 0.0
    for start, stop, label in segments:
        seg = np.linspace(0.0, 1.0, n_points)[:, None] * (stop - start)[None, :] + start
        seg[-1] = stop
        seg[0] = start
        if points:
            seg = seg[1:]
            x += 1.0 / (n_points - 1)
        # cumulative reciprocal-space abscissa from the segment parameter
        xlist.append(np.linspace(x, x + 1.0, len(seg)))
        x = xlist[-1][-1]
        points.append(seg)
        labels.append(label)
    return (
        np.concatenate(points),
        np.concatenate(xlist),
        [xlist[i][0] for i in range(len(segments))] + [xlist[-1][-1]],
        labels,
    )


def bare_magnon_dispersion(magnon, n_points: int = 61):
    """Bare magnon energies (eV) along the declared path via the Story-006
    canonical site-boson seam."""
    points, xlist, vertices, labels = bare_magnon_path(n_points)
    modes = magnon.get_site_boson_modes([p.tolist() for p in points])
    return {
        "kpoints": points,
        "x": xlist,
        "vertex_x": np.array(vertices),
        "labels": labels,
        "energies_eV": np.asarray(modes.energies, dtype=float),
    }


def run_coupled_spectra(
    magnon,
    blocks,
    settings: MaterialSettings,
    internal_magnon_points,
    phonon_loop_meshes,
):
    """Coupled one-shot Born spectra at the covered (q, mode) blocks.

    ``internal_magnon_points`` is the declared internal phonon-momentum mesh
    of the magnon loop (the covered points Gamma (optical channels only),
    K and M, weights 1/3).  ``phonon_loop_meshes`` maps each external phonon
    momentum (tuple of 3 floats) to ``(points, weights)`` of its declared
    internal magnon mesh; entries whose ``k + q`` folds to Gamma are EXCLUDED
    (the goldstone Bose occupation diverges there; that channel is the
    anisotropy-regularized contribution which cannot be evaluated without a
    qualified anisotropy gap — the declared infrared limitation of this run,
    see :data:`THERMAL_PRESCRIPTION`).
    """

    def _key(q):
        return tuple(np.round(np.asarray(q, float), 12))

    q_values = sorted({_key(b.q) for b in blocks})
    meshes = {_key(k): v for k, v in phonon_loop_meshes.items()}
    for q in q_values:
        if q not in meshes:
            raise ValueError(f"no declared phonon-loop mesh for q={q}")
    magnon_weights = np.full(
        len(internal_magnon_points), 1.0 / len(internal_magnon_points)
    )
    omega = settings.omega_grid_eV()
    eta = float(settings.broadening_eV)
    temperature = float(settings.temperature_K)
    n_modes = 2

    def _matching(q):
        return [b for b in blocks if np.allclose(b.q, q, atol=1e-8)]

    # ---- magnon channel: external k = magnon-loop mesh points ------------
    # each covered block is one internal (q, nu) channel carrying its q's
    # mesh weight; omega_q has a single branch per channel.
    sigma = np.zeros(
        (len(internal_magnon_points), omega.size, n_modes, n_modes), dtype=complex
    )
    bare = np.zeros((len(internal_magnon_points), n_modes))
    omega_blocks = np.array([[thz_to_ev([b.frequency_thz])[0]] for b in blocks])
    block_weights = np.array(
        [
            magnon_weights[
                next(
                    i
                    for i, p in enumerate(internal_magnon_points)
                    if np.allclose(b.q, p, atol=1e-8)
                )
            ]
            for b in blocks
        ]
    )
    nb_blocks = bose_occupation(omega_blocks, temperature)
    from TB2J.magnon_phonon.scattering import magnon_self_energy

    for i_k, k in enumerate(internal_magnon_points):
        g_blocks = np.zeros((len(blocks), n_modes, n_modes, 1), dtype=complex)
        eps_kq_blocks = np.zeros((len(blocks), n_modes))
        for i_b, block in enumerate(blocks):
            result = vertex_block(magnon, block, k)
            g = np.asarray(result.g, dtype=complex)
            g_blocks[i_b, : g.shape[0], : g.shape[1], 0] = g[0]
            eps_kq_blocks[i_b] = np.asarray(result.magnon_energies[0], dtype=float)
        sigma[i_k] = magnon_self_energy(
            g_blocks, eps_kq_blocks, omega_blocks, nb_blocks, block_weights, omega, eta
        )
    # bare external-k magnon energies do not depend on the block
    for i_k, k in enumerate(internal_magnon_points):
        result = vertex_block(magnon, blocks[0], k)
        bare[i_k] = np.asarray(result.magnon_energies[1], dtype=float)
    green, spectral = _particle_spectral(sigma, bare, omega, eta)
    magnon_out = {
        "kpoints": np.array(internal_magnon_points, dtype=float),
        "bare_energies_eV": bare,
        "sigma": sigma,
        "green": green,
        "spectral": spectral,
    }

    # ---- phonon channel: external q = covered q, per-q internal mesh -----
    phonon_entries = []
    for q in q_values:
        mesh, phonon_weights = meshes[q]
        matching = _matching(q)
        g_k = np.zeros((len(mesh), n_modes, n_modes, len(matching)), dtype=complex)
        eps_k = np.zeros((len(mesh), n_modes))
        eps_kq = np.zeros((len(mesh), n_modes))
        for i_b, block in enumerate(matching):
            for i_k, k in enumerate(mesh):
                result = vertex_block(magnon, block, k)
                g = np.asarray(result.g, dtype=complex)
                g_k[i_k, : g.shape[0], : g.shape[1], i_b] = g[0]
                eps_k[i_k] = np.asarray(result.magnon_energies[1], dtype=float)
                eps_kq[i_k] = np.asarray(result.magnon_energies[0], dtype=float)
        omega_ph = thz_to_ev([b.frequency_thz for b in matching])
        nb_k = bose_occupation(eps_k, temperature)
        nb_kq = bose_occupation(eps_kq, temperature)
        delta_d, sigma_nu = phonon_hot_band_kernel(
            g_k, eps_k, eps_kq, nb_k, nb_kq, omega_ph, phonon_weights, omega, eta
        )
        phonon_entries.append(
            {
                "q": np.asarray(q, dtype=float),
                "modes": [b.mode for b in matching],
                "bare_energies_eV": omega_ph,
                "delta_d": delta_d,
                "sigma": sigma_nu,
            }
        )
    return {
        "magnon": magnon_out,
        "phonon": phonon_entries,
        "omega_grid_eV": omega,
        "internal_magnon_points": np.array(internal_magnon_points, dtype=float),
        "internal_magnon_weights": magnon_weights,
        "phonon_loop_meshes": {
            q: {
                "points": np.asarray(mesh, float).tolist(),
                "weights": np.asarray(w, float).tolist(),
            }
            for q, (mesh, w) in phonon_loop_meshes.items()
        },
    }


def _particle_spectral(sigma, bare, omega, eta):
    """G^R = [(w+i eta) I - E - Sigma^R]^-1 and A = -2 Im G (Story-010)."""
    n_ext, _nw, n, _ = sigma.shape
    green = np.empty_like(sigma)
    eye = np.eye(n, dtype=complex)
    for i in range(n_ext):
        for w in range(omega.size):
            green[i, w] = np.linalg.inv(
                (omega[w] + 1j * eta) * eye - np.diag(bare[i]) - sigma[i, w]
            )
    return green, -2.0 * green.imag


# ---------------------------------------------------------------------------
# self-contained fixture (hermetic tests; generated by the run script)
# ---------------------------------------------------------------------------

FIXTURE_SCHEMA = "s013-material-fixture-v1"


def block_to_fixture(block: CoveredBlock) -> dict:
    """Serialize one covered block + the magnon constants to strict JSON."""
    return {
        "schema": FIXTURE_SCHEMA,
        "magnon": {
            "j0_eV": block.j0_eV,
            "moment_mu_b": CRI3_MOMENT_MUB,
            "bonds": [list(bond) for bond in CRI3_BONDS],
            "cell": CRI3_CELL.tolist(),
        },
        "block": {
            "label": block.label,
            "campaign_key": block.campaign_key,
            "campaign_id": block.campaign_id,
            "mode": block.mode,
            "q": np.asarray(block.q, float).tolist(),
            "mode_index": int(block.mode_index),
            "frequency_thz": float(block.frequency_thz),
            "eigenvalue_eV_per_Ang2_per_amu": float(
                block.eigenvalue_eV_per_Ang2_per_amu
            ),
            "polarization_real": block.polarization.real.tolist(),
            "polarization_imag": block.polarization.imag.tolist(),
            "masses_amu": np.asarray(block.masses_amu, float).tolist(),
            "evidence_sha256": block.evidence_sha256,
            "modes_table_sha256": block.modes_table_sha256,
            "max_fd_error_meV_per_Ang": float(block.max_fd_error_meV_per_Ang),
            "i": np.asarray(block.derivatives.i, int).tolist(),
            "j": np.asarray(block.derivatives.j, int).tolist(),
            "R": np.asarray(block.derivatives.R, int).tolist(),
            "O_real_eV": block.derivatives.O_pattern.real.tolist(),
            "O_imag_eV": block.derivatives.O_pattern.imag.tolist(),
            "FD_real_eV": block.derivatives.FD_pattern.real.tolist(),
            "FD_imag_eV": block.derivatives.FD_pattern.imag.tolist(),
        },
    }


def block_from_fixture(record: dict) -> CoveredBlock:
    """Rebuild a :class:`CoveredBlock` from a fixture record (no I/O)."""
    if record.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"fixture schema mismatch: {record.get('schema')!r}")
    entry = record["block"]
    polarization = np.asarray(entry["polarization_real"], float) + 1j * np.asarray(
        entry["polarization_imag"], float
    )
    derivatives = ModeProjectedDerivatives(
        q=np.asarray(entry["q"], float),
        mode=entry["mode"],
        mode_index=int(entry["mode_index"]),
        polarization=polarization,
        i=np.asarray(entry["i"], int),
        j=np.asarray(entry["j"], int),
        R=np.asarray(entry["R"], int),
        O_pattern=np.asarray(entry["O_real_eV"], float)
        + 1j * np.asarray(entry["O_imag_eV"], float),
        FD_pattern=np.asarray(entry["FD_real_eV"], float)
        + 1j * np.asarray(entry["FD_imag_eV"], float),
        source={"fixture": True},
        mode_evidence_sha256=entry["evidence_sha256"],
    )
    return CoveredBlock(
        label=entry["label"],
        campaign_key=entry["campaign_key"],
        campaign_id=entry["campaign_id"],
        mode=entry["mode"],
        q=np.asarray(entry["q"], float),
        mode_index=int(entry["mode_index"]),
        frequency_thz=float(entry["frequency_thz"]),
        eigenvalue_eV_per_Ang2_per_amu=float(entry["eigenvalue_eV_per_Ang2_per_amu"]),
        polarization=polarization,
        masses_amu=np.asarray(entry["masses_amu"], float),
        derivatives=derivatives,
        evidence_sha256=entry["evidence_sha256"],
        modes_table_sha256=entry["modes_table_sha256"],
        max_fd_error_meV_per_Ang=float(entry["max_fd_error_meV_per_Ang"]),
        j0_eV=float(record["magnon"]["j0_eV"]),
    )


#: provenance-manifest schema (v2: convergence section, per-reference
#: validation statuses, block evidence paths)
MANIFEST_SCHEMA = "s013-material-spectra-manifest-v2"


def provenance_manifest(
    settings: MaterialSettings,
    fixture_path,
    blocks,
    internal_magnon_points,
    phonon_loop_meshes,
    extra: dict | None = None,
    thermal_prescription: str = THERMAL_PRESCRIPTION,
) -> dict:
    """Provenance manifest bound to the artifacts (TEST-002 schema)."""
    required = {
        "schema": MANIFEST_SCHEMA,
        "story": {
            "local_id": "013",
            "project": "nonorth-spin-lattice",
            "subproject": "validated-magnon-phonon-spectra-and-manuscript-evidence",
            "review_base": "f030d57",
        },
        "settings": settings.as_dict(),
        "approximation": _APPROXIMATION,
        "broadening_eV": settings.broadening_eV,
        "temperature_K": settings.temperature_K,
        "internal_mesh": {
            "magnon_loop_points": np.asarray(internal_magnon_points, float).tolist(),
            "magnon_loop_weights": np.full(
                len(internal_magnon_points),
                1.0 / len(internal_magnon_points),
            ).tolist(),
            "phonon_loop_meshes": {
                q: {
                    "points": np.asarray(mesh, float).tolist(),
                    "weights": np.asarray(w, float).tolist(),
                }
                for q, (mesh, w) in (
                    (json.dumps(np.round(np.asarray(k, float), 12).tolist()), v)
                    for k, v in phonon_loop_meshes.items()
                )
            },
        },
        "thermal_prescription": thermal_prescription,
        "mixing_availability": MIXING_UNAVAILABLE_STATEMENT,
        "interpolation_status": (
            "none: bare phonons only at covered commensurate points; coupled "
            "spectra only at hash-joined covered blocks; no interpolated leg"
        ),
        "fixture": {
            "path": str(fixture_path),
            "sha256": sha256_file(fixture_path),
            "schema": FIXTURE_SCHEMA,
        },
        "blocks": [
            {
                "label": b.label,
                "campaign_key": b.campaign_key,
                "campaign_id": b.campaign_id,
                "mode": b.mode,
                "q": np.asarray(b.q, float).tolist(),
                "evidence_sha256": b.evidence_sha256,
                "modes_table_sha256": b.modes_table_sha256,
                "max_fd_error_meV_per_Ang": b.max_fd_error_meV_per_Ang,
                **({"evidence_path": b.evidence_path} if b.evidence_path else {}),
            }
            for b in blocks
        ],
    }
    if extra:
        for key, value in extra.items():
            if value in (None, "", [], {}):
                raise ValueError(f"manifest field {key!r} is empty")
            required[key] = value
    return required


def _manifest_hash_references(manifest: dict):
    """Declared (label, path, expected_sha256) triples referenced by a manifest.

    Covers the matched finite-q rows report, the closing-campaign
    manifest/store and the s012 qualification report (``path`` + ``sha256``
    fields) and every block's mode-evidence file (``evidence_path`` +
    ``evidence_sha256``).  The fixture is validated separately (required).
    """
    references = []
    for key in (
        "s002_store",
        "s013_closing_manifest",
        "s013_closing_store",
        "s012_qualification_report",
    ):
        reference = manifest.get(key)
        if (
            isinstance(reference, dict)
            and "path" in reference
            and "sha256" in reference
        ):
            references.append((key, reference["path"], reference["sha256"]))
    for index, block in enumerate(manifest.get("blocks", [])):
        if "evidence_sha256" in block:
            references.append(
                (
                    f"blocks[{index}]",
                    block.get("evidence_path"),
                    block["evidence_sha256"],
                )
            )
    return references


def validate_manifest(manifest: dict, fixture_path) -> dict:
    """TEST-002 gate: every field non-empty, the fixture hash matches, and
    every referenced file (matched rows report, closing-campaign manifest
    and store, s012 qualification report, block evidence) hashes to its
    recorded value when the file is present.

    Referenced files that are absent are recorded as ``"unresolved"`` in
    ``manifest["reference_validation"]`` (also the returned dict); a present
    file whose sha256 differs from the recorded hash is a hard error.
    """
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema mismatch")

    def _check(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                _check(item, f"{path}.{key}")
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"manifest list {path} is empty")
            for i, item in enumerate(value):
                _check(item, f"{path}[{i}]")
        elif value is None or value == "" or value == {}:
            raise ValueError(f"manifest field {path} is empty")

    _check(manifest, "manifest")
    if manifest["fixture"]["sha256"] != sha256_file(fixture_path):
        raise ValueError("manifest fixture hash does not match the fixture file")
    statuses = {}
    for label, path, expected_sha256 in _manifest_hash_references(manifest):
        if not path or not Path(path).exists():
            statuses[label] = "unresolved"
            continue
        if sha256_file(path) != expected_sha256:
            raise ValueError(
                f"manifest referenced file hash mismatch for {label}: {path}"
            )
        statuses[label] = "resolved"
    manifest["reference_validation"] = statuses
    return statuses
