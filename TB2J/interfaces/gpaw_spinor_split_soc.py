"""GPAW split-SOC second-variational soc-leg adapter (story 003, ADR-4).

Builds per-direction second-variational SOC legs from ONE collinear no-SOC
old-API GPAW calculation (strength-0 protocol ADR-3: the SOC operator is
evaluated on the frozen ``D_asp`` density, no SOC SCF), and converts a leg to
the story-002 KS-band split-SOC kernel input in the **psi gauge**.

Conventions (pinned by story-001 ``docs/sympy/split_soc_gauge.py`` and the
GPAW 26.7.0 source):

- ``gpaw.spinorbit.soc_eigenstates`` diagonalizes ``H + C^dag (sigma.L) C``
  on the paired collinear bands; the rotation is the standard ``C^dag H C``
  (verbatim two-tensordot chain in ``add_soc``), with

  ``C(theta, phi) = exp(-i phi sz/2) exp(-i theta sy/2)``  (theta, phi in
  degrees at the API level), so the leg spinor slots are ``sigma . n``
  eigenstates, ``n = (sin th cos ph, sin th sin ph, cos th)``.

- psi gauge: the collinear exchange vertex stays ``sigma_z``-diagonal
  (``H_psi = h0 (x) 1 + Delta (x) sigma_z + W_v (x) C^dag sigma_v C``); only
  the SOC operator is rotated.  Leg data therefore feed the kernel with the
  UNROTATED ``delta_xc``/``delta_total`` vertices, and output tensors rotate
  at merge time as ``T_lattice = O T_leg O^T`` with
  ``O_wv = Tr[sigma_w C sigma_v C^dag] / 2`` (``O e_z = n``) -- see
  :func:`apply_frame_rotation` and ADR-4 (psi-gauge Option A; per-leg
  ``spinat`` = leg axis).

All-atom ``W_SO``: :func:`collect_soc_leg` rebuilds the strength-0 (paired
collinear) basis from the leg eigenvectors (``P_pre = conj(V) P_soc``) and
contracts the rotated atom blocks into the band-space operator
``W_SO^K(k) = sum_a B_a^dag w_a^leg B_a`` in eV (GPAW does not expose this
operator directly; it is reconstructed from ``P_amj``/``v_mn`` and the frozen
``D_asp`` operator blocks).  At ``scale=0`` the leg data reduce exactly to the
collinear interleaved bands/projections (tested).

Driver seam (story 004, NOT implemented here)::

    leg = collect_soc_leg(calc, theta=90.0, phi=90.0)     # one collinear SCF
    data = soc_leg_to_projector_green_data(leg)
    out = compute_ks_split_soc_exchange(data, leg.w_soc, lam=..., Rpts=...)
    T_lattice = apply_frame_rotation(T_psi, leg.rotation)  # at merge time

Limitations (by contract, not by omission): old-API/legacy calculators or
legacy ``.gpw`` files only; serial (single-process) runs; ``projected=True``
is refused (GPAW's projected operator is a different, spin-diagonal
operator); legs on calculators already carrying ``soc=True`` are refused
(SOC double counting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.units import Ha

from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreenData,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gpaw.old.calculator import GPAW as OldGPAW

SIGMA = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)

#: Canonical three-leg angles in degrees (GPAW theta/phi convention).
LEG_AXES: dict[str, tuple[float, float]] = {
    "x": (90.0, 0.0),
    "y": (90.0, 90.0),
    "z": (0.0, 0.0),
}

_HERM_TOL = 1.0e-8

_DELTA_KINDS = ("delta_xc", "delta_total")


# ---------------------------------------------------------------------------
# gauge primitives (story-001 verbatim forms; angles in RADIANS here)
# ---------------------------------------------------------------------------


def c_gpaw(theta: float, phi: float) -> np.ndarray:
    """GPAW ``spinorbit.py`` C_ss verbatim formula (radians).

    ``C = exp(-i phi sz/2) exp(-i theta sy/2)``; the leg spinor slots are
    ``sigma . n`` eigenstates with ``C sigma_z C^dag = n . sigma``.
    """
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    em, ep = np.exp(-1j * phi / 2), np.exp(1j * phi / 2)
    return np.array(
        [[ct * em, -st * em], [st * ep, ct * ep]],
        dtype=complex,
    )


def axis_of(theta: float, phi: float) -> np.ndarray:
    """Leg axis ``n(theta, phi)`` for unit vector convention ``O e_z = n``."""
    return np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


def o_from_c(c_mat: np.ndarray) -> np.ndarray:
    """``O_wv = Tr[sigma_w C sigma_v C^dag] / 2`` in SO(3), ``O e_z = n``."""
    return np.array(
        [
            [
                0.5 * np.trace(SIGMA[w] @ c_mat @ SIGMA[v] @ c_mat.conj().T).real
                for v in range(3)
            ]
            for w in range(3)
        ]
    )


def pack_sigma_dot_l(dvl_vii: np.ndarray) -> np.ndarray:
    """Verbatim ``add_soc`` packing ``(3, ni, ni) -> (2, 2, ni, ni)``.

    ``H_ssii[0,0] = L_z``, ``H_ssii[0,1] = L_x - i L_y``,
    ``H_ssii[1,0] = L_x + i L_y``, ``H_ssii[1,1] = -L_z`` (i.e. ``sigma.L``).
    """
    dvl_vii = np.asarray(dvl_vii, dtype=complex)
    if dvl_vii.shape[0] != 3:
        raise ValueError(f"dVL_vii must have shape (3, ni, ni); got {dvl_vii.shape}")
    h_ssii = np.zeros((2, 2) + dvl_vii.shape[1:], complex)
    h_ssii[0, 0] = dvl_vii[2]
    h_ssii[0, 1] = dvl_vii[0] - 1j * dvl_vii[1]
    h_ssii[1, 0] = dvl_vii[0] + 1j * dvl_vii[1]
    h_ssii[1, 1] = -dvl_vii[2]
    return h_ssii


def rotate_to_leg_basis(h_ssii: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """Verbatim ``add_soc`` two-tensordot chain: ``H <- C^dag H C``.

    Reproduces ``gpaw/spinorbit.py`` ``add_soc`` exactly::

        H = np.tensordot(C_ss, H_ssii, (0, 1))
        H = np.tensordot(C_ss.T.conj(), H, (1, 1))
    """
    out = np.tensordot(c_mat, h_ssii, (0, 1))
    return np.tensordot(c_mat.T.conj(), out, (1, 1))


def apply_frame_rotation(t_leg: np.ndarray, o_mat: np.ndarray) -> np.ndarray:
    """Rotate a leg-frame Cartesian tensor to the lattice: ``O T O^T``."""
    t_leg = np.asarray(t_leg, dtype=float)
    o_mat = np.asarray(o_mat, dtype=float)
    if t_leg.shape != (3, 3) or o_mat.shape != (3, 3):
        raise ValueError("apply_frame_rotation expects two (3, 3) tensors")
    return np.einsum("wa,ab,ub->wu", o_mat, t_leg, o_mat)


# ---------------------------------------------------------------------------
# leg container
# ---------------------------------------------------------------------------


@dataclass
class GPAWSocLeg:
    """One per-direction second-variational SOC leg (psi gauge).

    Band count is ``2 * nbands`` (the paired collinear window); arrays are
    ordered by BZ k point ``K`` in ``gpaw`` ``kd.bzk_kc`` order.
    """

    theta: float  # degrees, as requested
    phi: float  # degrees, as requested
    scale: float
    axis: np.ndarray  # (3,) leg axis n
    c_matrix: np.ndarray  # (2, 2) GPAW spinor basis matrix
    rotation: np.ndarray  # (3, 3) O with O e_z = n
    kpoints: np.ndarray  # (nk, 3) fractional BZ coordinates
    weights: np.ndarray  # (nk,) uniform 1/nk (full BZ)
    nbands: int  # collinear band count; leg window is 2 * nbands
    eigenvalues_strength0: np.ndarray  # (nk, 2 nb) eV, interleaved up/down
    occupations_strength0: np.ndarray  # (nk, 2 nb)
    eigenvalues_soc: np.ndarray  # (nk, 2 nb) eV, second-variational
    occupations_soc: np.ndarray  # (nk, 2 nb)
    efermi: float  # strength-0 (collinear) fermi level, eV
    efermi_soc: float  # leg fermi level from BZWaveFunctions, eV
    band_energy_soc: float  # exact BZWaveFunctions.calculate_band_energy() result
    w_soc: np.ndarray  # (nk, 2 nb, 2 nb) eV, strength-0 psi band basis
    w_soc_atom: np.ndarray  # (natoms, ni_max, ni_max, 2, 2) eV, leg frame
    p_amj_soc: np.ndarray  # (nk, 2 nb, nproj, 2) psi-gauge leg projections
    v_mn: np.ndarray  # (nk, 2 nb, 2 nb) second-variational eigenvectors
    vertices: np.ndarray  # (nsite, ni_max, ni_max, 2, 2) eV, psi gauge
    vertex_component: str  # which delta the primary vertices carry
    site_nproj: np.ndarray
    site_projector_indices: np.ndarray
    projector_l: np.ndarray | None = None
    projector_m: np.ndarray | None = None
    projector_radial: np.ndarray | None = None
    overlap_metric: np.ndarray | None = None
    cell: np.ndarray | None = None
    positions: np.ndarray | None = None
    atomic_numbers: np.ndarray | None = None
    operator_components: dict[str, np.ndarray] = field(default_factory=dict)
    operator_component_metadata: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def nbands_spinor(self) -> int:
        return 2 * int(self.nbands)


# ---------------------------------------------------------------------------
# refusals (contract gates)
# ---------------------------------------------------------------------------


def _require_legacy_collinear(calc) -> None:
    """Reject new-API, SOC-carrying, or non-two-channel calculators.

    Note: the old-API GPAW in 26.7.0 exposes a ``dft`` compatibility property
    (a shim for ``calc.dft.scf_loop.niter``), so the API kind must be decided
    by the implementation module, not by attribute presence.
    """
    module = type(calc).__module__
    if not module.startswith("gpaw.old."):
        raise TypeError(
            "split-SOC legs require an old-API (legacy) GPAW calculation or "
            "legacy .gpw file; gpaw.spinorbit.soc_eigenstates only supports "
            f"gpaw.old calculators, got {module}.{type(calc).__name__}"
        )
    params = getattr(calc, "parameters", None) or {}
    if params.get("soc", False):
        raise ValueError(
            "the calculation already includes spin-orbit coupling "
            "(soc=True); a second-variational leg on top of an SOC run would "
            "double-count SOC -- collect legs on a collinear no-SOC reference"
        )
    wfs = getattr(calc, "wfs", None)
    if getattr(wfs, "nspins", None) != 2:
        raise ValueError(
            "split-SOC legs require a collinear spin-polarized (two-channel) "
            f"calculation; got nspins={getattr(wfs, 'nspins', None)!r}"
        )
    if getattr(getattr(calc, "density", None), "D_asp", None) is None:
        raise ValueError(
            "split-SOC legs require a converged PAW density (D_asp) to freeze "
            "the strength-0 SOC operator on"
        )


def _require_serial(bzw) -> None:
    comms = (
        bzw.kpt_comm.size,
        bzw.bcomm.size,
        bzw.domain_comm.size,
    )
    if any(size > 1 for size in comms):
        raise NotImplementedError(
            "collect_soc_leg requires a serial (single k-point/band/domain "
            f"process) GPAW state; got communicator sizes {comms}"
        )


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


def _site_layout(calc):
    """Per-site PAW projector layout shared with the collinear exporter."""
    from TB2J.interfaces.gpaw_projector import _setup_projector_metadata

    return _setup_projector_metadata(calc.wfs.setups)


def _collect_vertices(calc, site_nproj, delta_kind):
    """delta_xc (+U delta_total) magnetic vertices in the psi gauge."""
    from TB2J.interfaces.gpaw_projector import (
        _build_gpaw_total_delta,
        _collect_delta_xc_paw_xc,
        _collect_hubbard_metadata,
    )

    delta_xc = _collect_delta_xc_paw_xc(calc, site_nproj)
    hubbard = _collect_hubbard_metadata(calc)
    delta_total = None
    if hubbard:
        delta_total = _build_gpaw_total_delta(calc, delta_xc, site_nproj, hubbard)
    primary = (
        delta_total
        if (delta_kind == "delta_total" and delta_total is not None)
        else delta_xc
    )
    used = (
        "delta_total"
        if (delta_kind == "delta_total" and delta_total is not None)
        else "delta_xc"
    )
    nsite = len(site_nproj)
    nmax = int(np.asarray(site_nproj).max(initial=0))
    vertices = np.zeros((nsite, nmax, nmax, 2, 2), complex)
    for site in range(nsite):
        ni = int(site_nproj[site])
        vertices[site, :ni, :ni] = np.einsum(
            "pq,st->pqst", primary[site, :ni, :ni], SIGMA[2]
        )
    components = {"delta_xc": delta_xc}
    component_meta = {
        "delta_xc": {
            "units": "eV",
            "definition": (
                "explicit V_xc^up - V_xc^down PAW partial-wave matrix "
                "(GPAW xc.calculate_paw_correction spin splitting)"
            ),
            "source": "GPAW hamiltonian.xc.calculate_paw_correction",
            "operator_basis": "paw_partial_wave_channel",
        }
    }
    if delta_total is not None:
        components["delta_total"] = delta_total
        component_meta["delta_total"] = {
            "units": "eV",
            "definition": (
                "explicit PAW XC spin difference plus GPAW Hubbard "
                "derivative spin difference"
            ),
            "source": (
                "GPAW hamiltonian.xc.calculate_paw_correction + "
                "setup.hubbard_u.calculate"
            ),
            "operator_basis": "paw_partial_wave_channel",
            "hubbard_included": "true",
        }
    return vertices, used, components, component_meta, hubbard


def collect_soc_leg(
    calc: "OldGPAW",
    theta: float = 0.0,
    phi: float = 0.0,
    scale: float = 1.0,
    projected: bool = False,
    ignore_xc_potential: bool = False,
    delta_kind: str = "delta_xc",
) -> GPAWSocLeg:
    """Collect one per-direction second-variational SOC leg (ADR-4).

    Parameters
    ----------
    calc : old-API GPAW calculator or legacy ``.gpw`` path
        Converged collinear spin-polarized no-SOC reference (the strength-0
        SCF leg of ADR-3).  New-API calculators, ``soc=True`` runs, and
        non-two-channel states are refused (see module docstring).
    theta, phi : float
        Leg direction in GPAW convention (degrees).
    scale : float
        SOC scaling; ``0.0`` reproduces the collinear data exactly.
    projected : bool
        Must stay ``False``: GPAW's ``projected=True`` switches to the
        spin-diagonal ``projected_soc`` operator, which is a different
        operator class than the ``C^dag (sigma.L) C`` leg required here.
    ignore_xc_potential : bool
        Forwarded to ``gpaw.spinorbit.soc``.
    delta_kind : ``"delta_xc"`` or ``"delta_total"``
        Primary magnetic-vertex component (``delta_total`` adds the +U
        splitting when the setups carry Hubbard evaluators; without +U the
        two coincide and ``delta_xc`` is recorded).

    Returns
    -------
    GPAWSocLeg
        Per-direction leg: strength-0 interleaved eigenvalues/occupations,
        leg (second-variational) eigenvalues/occupations and single fermi
        levels, PAW ``P_amj`` and eigenvectors ``v_mn``, the reconstructed
        all-atom band-space ``W_SO^K(k)`` in eV (``leg.w_soc``), per-atom
        leg-frame operator blocks, psi-gauge magnetic vertices, and the
        frame metadata (axis, C, O) for the story-004 rotate/merge driver.
    """
    if delta_kind not in _DELTA_KINDS:
        raise ValueError(
            f"delta_kind must be one of {_DELTA_KINDS}; got {delta_kind!r}"
        )
    if isinstance(calc, (str, Path)):
        from gpaw import GPAW

        calc = GPAW(calc, legacy_gpaw=True)
    _require_legacy_collinear(calc)
    if projected:
        raise ValueError(
            "projected=True is not supported: GPAW's projected_soc applies a "
            "spin-diagonal projection of the SOC operator instead of the "
            "second-variational C^dag (sigma.L) C leg required by the "
            "split-SOC kernel"
        )

    from gpaw.spinorbit import soc, soc_eigenstates

    theta = float(theta)
    phi = float(phi)
    scale = float(scale)
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
    c_matrix = c_gpaw(theta_rad, phi_rad)
    rotation = o_from_c(c_matrix)
    axis = axis_of(theta_rad, phi_rad)
    if np.abs(rotation @ np.array([0.0, 0.0, 1.0]) - axis).max() > 1e-12:
        raise AssertionError("frame metadata inconsistent: O e_z != n")

    # SOC operator on the FROZEN collinear density (strength-0, all atoms),
    # verbatim source terms of soc_eigenstates:
    dvl_avii = {
        a: soc(
            calc.wfs.setups[a],
            calc.hamiltonian.xc,
            D_sp,
            ignore_xc_potential,
        )
        * scale
        for a, D_sp in calc.density.D_asp.items()
    }

    bzw = soc_eigenstates(
        calc,
        scale=scale,
        theta=theta,
        phi=phi,
        projected=False,
        ignore_xc_potential=ignore_xc_potential,
    )
    _require_serial(bzw)

    kd = calc.wfs.kd
    nk = int(kd.nbzkpts)
    nb = int(calc.get_number_of_bands())
    natoms = len(calc.wfs.setups)

    layout = _site_layout(calc)
    site_nproj = np.asarray(layout["site_nproj"], dtype=int)
    nproj = int(site_nproj.sum())
    nmax = int(site_nproj.max(initial=0))

    # strength-0 interleaved bands (the psi-gauge lambda=0 spectrum)
    eig_pre = np.empty((nk, 2 * nb))
    occ_pre = np.empty((nk, 2 * nb))
    for K in range(nk):
        k = int(kd.bz2ibz_k[K])
        eig_pre[K, 0::2] = calc.get_eigenvalues(kpt=k, spin=0)
        eig_pre[K, 1::2] = calc.get_eigenvalues(kpt=k, spin=1)
        occ_pre[K, 0::2] = np.asarray(calc.get_occupation_numbers(kpt=k, spin=0))
        occ_pre[K, 1::2] = np.asarray(calc.get_occupation_numbers(kpt=k, spin=1))

    eig_soc = np.asarray(bzw.eigenvalues(), dtype=float)
    occ_soc = np.asarray(bzw.occupation_numbers(), dtype=float)
    if eig_soc.shape != (nk, 2 * nb):
        raise ValueError(
            f"unexpected leg spectrum shape {eig_soc.shape}; expected "
            f"{(nk, 2 * nb)}"
        )

    # per-k leg projections (m, i, s) and eigenvectors
    p_amj = np.empty((nk, 2 * nb, nproj, 2), complex)
    v_mn = np.empty((nk, 2 * nb, 2 * nb), complex)
    offsets = np.concatenate(([0], np.cumsum(site_nproj)))
    for K in range(nk):
        wf = bzw[K]
        if wf.v_mn.shape != (2 * nb, 2 * nb):
            raise ValueError(f"k point {K}: leg eigenvector shape {wf.v_mn.shape}")
        v_mn[K] = wf.v_mn
        for a in range(natoms):
            lo, hi = int(offsets[a]), int(offsets[a + 1])
            ni = hi - lo
            block = np.asarray(wf.P_amj[a], dtype=complex)
            if block.shape != (2 * nb, 2 * ni):
                raise ValueError(f"k point {K}, atom {a}: P_amj shape {block.shape}")
            p_amj[K, :, lo:hi, :] = block.reshape(2 * nb, ni, 2)

    # per-atom leg-frame operators (i, j, s, t) in eV
    w_atom = np.zeros((natoms, nmax, nmax, 2, 2), complex)
    for a, dvl in dvl_avii.items():
        ni = int(site_nproj[a])
        h_ssii = rotate_to_leg_basis(pack_sigma_dot_l(dvl), c_matrix) * Ha
        w_atom[a, :ni, :ni] = h_ssii.transpose(2, 3, 0, 1)

    # band-space W_SO^K in the strength-0 psi basis: de-rotate the leg
    # projections (P_pre = conj(V) P_soc) and contract the atom blocks
    w_glob = np.zeros((2 * nproj, 2 * nproj), complex)
    for a in range(natoms):
        ni = int(site_nproj[a])
        lo = int(offsets[a])
        for s in range(2):
            for t in range(2):
                w_glob[
                    s * nproj + lo : s * nproj + lo + ni,
                    t * nproj + lo : t * nproj + lo + ni,
                ] = w_atom[a, :ni, :ni, s, t]
    w_soc = np.empty((nk, 2 * nb, 2 * nb), complex)
    for K in range(nk):
        p_pre = np.einsum("mn,mps->nps", np.conj(v_mn[K]), p_amj[K])
        p_si = p_pre.transpose(0, 2, 1).reshape(2 * nb, 2 * nproj)
        w_soc[K] = p_si.conj() @ w_glob @ p_si.T
    wscale = max(1.0, float(np.abs(w_soc).max(initial=0.0)))
    herm = np.abs(w_soc - np.conj(np.swapaxes(w_soc, 1, 2))).max()
    if herm > _HERM_TOL * wscale:
        raise ValueError(f"assembled W_SO^K is not Hermitian (deviation {herm:.3e} eV)")

    vertices, used_kind, components, component_meta, hubbard = _collect_vertices(
        calc, site_nproj, delta_kind
    )

    atoms = calc.get_atoms()
    metadata = {
        "source": "gpaw.spinorbit.soc_eigenstates (old-API, frozen D_asp)",
        "code": "gpaw",
        "adapter": "gpaw_spinor_split_soc",
        "gauge": "psi (fixed S_z slots; collinear exchange unrotated)",
        "theta_deg": theta,
        "phi_deg": phi,
        "scale": scale,
        "axis": axis.tolist(),
        "rotation": rotation.tolist(),
        "c_matrix": c_matrix.tolist(),
        "output_rotation": "T_lattice = O T_leg O^T with O e_z = n (ADR-4)",
        "per_leg_spinat": "leg axis",
        "strength0_reference": {
            "scf": "collinear no-SOC legacy GPAW",
            "efermi_eV": float(calc.get_fermi_level()),
            "nspins": int(calc.wfs.nspins),
        },
        "soc_operator_source": "gpaw.spinorbit.soc on frozen D_asp (all atoms)",
        "w_soc_source": (
            "reconstructed from leg P_amj/v_mn de-rotation and the rotated "
            "atom blocks (GPAW does not expose W_SO directly)"
        ),
        "vertex_component": used_kind,
        "nbzkpts": nk,
        "nibzkpts": int(kd.nibzkpts),
        "symmetry_unfolded_to_full_bz": bool(nk != int(kd.nibzkpts)),
        "units": {
            "eigenvalues": "eV",
            "efermi": "eV",
            "w_soc": "eV",
            "w_soc_atom": "eV",
            "vertices": "eV",
            "cell": "Angstrom",
            "positions": "Angstrom",
        },
    }
    if hubbard:
        metadata["gpaw_hubbard"] = hubbard

    return GPAWSocLeg(
        theta=theta,
        phi=phi,
        scale=scale,
        axis=axis,
        c_matrix=c_matrix,
        rotation=rotation,
        kpoints=np.asarray(kd.bzk_kc, dtype=float),
        weights=np.full(nk, 1.0 / nk),
        nbands=nb,
        eigenvalues_strength0=eig_pre,
        occupations_strength0=occ_pre,
        eigenvalues_soc=eig_soc,
        occupations_soc=occ_soc,
        efermi=float(calc.get_fermi_level()),
        efermi_soc=float(bzw.fermi_level),
        band_energy_soc=float(bzw.calculate_band_energy()),
        w_soc=w_soc,
        w_soc_atom=w_atom,
        p_amj_soc=p_amj,
        v_mn=v_mn,
        vertices=vertices,
        vertex_component=used_kind,
        site_nproj=site_nproj,
        site_projector_indices=np.asarray(layout["site_projector_indices"], dtype=int),
        projector_l=layout["projector_l"],
        projector_m=layout["projector_m"],
        projector_radial=layout["projector_radial"],
        overlap_metric=layout["overlap_metric"],
        cell=np.asarray(atoms.cell.array, dtype=float),
        positions=np.asarray(atoms.get_positions(), dtype=float),
        atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        operator_components=components,
        operator_component_metadata=component_meta,
        metadata=metadata,
    )


def collect_three_legs(
    calc: "OldGPAW",
    scale: float = 1.0,
    **kwargs,
) -> dict[str, GPAWSocLeg]:
    """Collect the canonical (x, y, z) legs for the story-004 driver seam."""
    return {
        name: collect_soc_leg(calc, theta=theta, phi=phi, scale=scale, **kwargs)
        for name, (theta, phi) in LEG_AXES.items()
    }


# ---------------------------------------------------------------------------
# kernel feed
# ---------------------------------------------------------------------------


def soc_leg_to_projector_green_data(
    leg: GPAWSocLeg,
    metadata: dict | None = None,
) -> ProjectorGreenData:
    """Convert a leg to story-002 kernel input (psi gauge, ``nspinor=2``).

    The data carry the strength-0 (``lambda=0``) interleaved spectrum and the
    de-rotated paired projections ``B_a``; pair them with ``leg.w_soc`` in
    :func:`TB2J.split_soc_kernel.compute_ks_split_soc_exchange`.  At
    ``scale=0`` this is exactly the collinear band data.  Output tensors stay
    in the psi frame; rotate with :func:`apply_frame_rotation` at merge time.
    """
    # P_pre = conj(V) P_soc: strength-0 paired psi basis -> (1, nk, N, 2, nproj)
    coefficients = np.einsum("kmn,kmps->knsp", np.conj(leg.v_mn), leg.p_amj_soc)[None]

    meta = {
        "code": "gpaw",
        "adapter": "gpaw_spinor_split_soc",
        "nspinor": 2,
        "spinor_export": "gpaw_spinor_split_soc",
        "frame": {
            "gauge": "psi",
            "spinaxis": leg.axis.tolist(),
            "theta_deg": leg.theta,
            "phi_deg": leg.phi,
            "rotation": leg.rotation.tolist(),
            "c_matrix": leg.c_matrix.tolist(),
            "output_rotation": "T_lattice = O T_leg O^T",
            "per_leg_spinat": "leg axis",
        },
        "strength0_reference": dict(leg.metadata.get("strength0_reference", {})),
        "soc_operator_source": leg.metadata.get("soc_operator_source"),
        "scale": leg.scale,
        "vertex_component": leg.vertex_component,
        "units": {
            "cell": "Angstrom",
            "positions": "Angstrom",
            "eigenvalues": "eV",
            "efermi": "eV",
            "spinor_operator": "eV",
            "w_soc": "eV",
        },
    }
    if metadata:
        meta.update(metadata)

    return ProjectorGreenData(
        kpoints=leg.kpoints,
        weights=leg.weights,
        eigenvalues=leg.eigenvalues_strength0[None, ...],
        coefficients=coefficients,
        efermi=leg.efermi,
        projector_site=np.repeat(np.arange(len(leg.site_nproj)), leg.site_nproj),
        projector_atom=np.repeat(np.arange(len(leg.site_nproj)), leg.site_nproj),
        cell=leg.cell,
        positions=leg.positions,
        atomic_numbers=leg.atomic_numbers,
        occupations=leg.occupations_strength0[None, ...],
        projector_l=leg.projector_l,
        projector_m=leg.projector_m,
        projector_radial=leg.projector_radial,
        overlap_metric=leg.overlap_metric,
        site_nproj=leg.site_nproj,
        site_projector_indices=leg.site_projector_indices,
        nspinor=2,
        spinor_operator=leg.vertices,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source=(
            "gpaw.soc_eigenstates P_amj de-rotated to the strength-0 psi basis"
        ),
        coefficient_projector="native_paw_projector",
        channel_interpretation="paw_projector_channel",
        operator_basis=(
            "gpaw delta_xc/delta_total pauli, sigma_z-diagonal (psi gauge)"
        ),
        operator_components=dict(leg.operator_components),
        operator_component_metadata=dict(leg.operator_component_metadata),
        metadata=meta,
    )
