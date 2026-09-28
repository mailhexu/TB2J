"""VASP KS-basis split-SOC adapter (ADR-7, story 011).

Consumes two artifacts of one collinear strength-0 VASP run (patched with
the story-010 dump hooks):

- ``tb2j_native.bin`` (v5/v6 collinear native export; complex W%CPROJ,
  bands, occupations, CDIJ) read by :func:`read_vasp_native`;
- ``tb2j_cso.bin`` (per-ion one-center SOC operator ``CSO``, augmentation
  occupations ``COCC``, spherical AE potential) read by the vendored
  :mod:`TB2J.interfaces.vasp_cso_dump` reader.

The KS-band SOC operator follows the ``CALC_PAW_OVERLAP`` contraction
pattern,

.. math:: W^{\\mathcal K}_{SO}(k) = \\sum_a B_a(k)^\\dagger\\, CSO_a\\, B_a(k),

with :math:`B_a` the rectangular projector map of atom ``a`` and the
``(uu, ud, du, dd)`` spinor blocks of ``CSO_a`` in the SAXIS spin frame
VASP uses for the dump.  The second variation and the three-direction
rotate/merge follow the GPAW ADR-4 shape: each leg :math:`d \\in \\{x, y,
z\\} re-expresses the *whole* strength-0 problem in the spin frame with
quantization along :math:`d` through the frame map
:math:`M = U_d^\\dagger U_{SAXIS}` — the band spinor components and the
magnetic vertices (:math:`\\Delta_a M \\sigma_z M^\\dagger` — the
physical SAXIS-axis splitting field in leg-frame components) are
conjugated, while ``W_SO`` is the frame-independent *state-space* matrix
``<psi|W_SO|psi'>`` and enters unchanged, so every leg is the same
physical collinear reference seen from its own frame.  The kernel extracts the tensor in
that leg frame, and the output rotates back with
:math:`T_{lattice} = O\\, T_{leg}\\, O^T`, :math:`O e_z = d` (story-001
corrected O map).  W_SO is all-atom (ligand SOC enters the DMI); the
magnetic rotation vertices are site-local, magnetic-only, and carry the
collinear spin splitting ``D_up - D_down``.

No PROCAR weights and no LOCPROJ/CDIJ-mismatched SOC surrogate are used;
the v7 spinor export remains the fully-relativistic comparison leg only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from TB2J.interfaces.vasp_cso_dump import CsoDump, read_cso_dump
from TB2J.interfaces.vasp_native import HARTREE_TO_EV, read_vasp_native
from TB2J.paw_projector import PawProjectorSnapshot
from TB2J.projector_green import (
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreenData,
    site_magnetization_sign,
)
from TB2J.split_soc_kernel import (
    MODE_SECOND_VARIATION,
    SPLIT_SOC_MODES,
    compute_ks_split_soc_exchange,
)

LEGS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

_PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)

_SOC_OPERATOR_SOURCE = (
    "VASP tb2j_cso.bin v1 CSO (SPINORB_STRENGTH at the SAXIS Euler "
    "angles; story-010 patch dump, collinear branch)"
)
_STRENGTH0_SOURCE = "VASP collinear v5/v6 native export (tb2j_native.bin)"


# ---------------------------------------------------------------------------
# VASP spin-frame helpers (EULER + SETUP_LS ROTMAT, ported verbatim)
# ---------------------------------------------------------------------------
def vasp_spinor_frame(alpha: float, beta: float) -> np.ndarray:
    """VASP SETUP_LS ``ROTMAT`` spinor-frame unitary.

    ``alpha``/``beta`` are the EULER angles of the spin axis (the PHI/THETA
    arguments VASP passes to ``SPINORB_STRENGTH``).  Columns are the frame's
    (+, -) basis spinors in the standard z-frame components, up to the
    per-column phase VASP carries; the dump CSO blocks are matrix elements
    in exactly this frame.
    """
    c = np.cos(0.5 * beta)
    s = np.sin(0.5 * beta)
    ep = np.exp(-0.5j * alpha)
    em = np.exp(0.5j * alpha)
    return np.array([[c * ep, -s * ep], [s * em, c * em]], dtype=complex)


def euler_angles(direction) -> tuple[float, float]:
    """VASP ``EULER``: (alpha, beta) of a spin direction vector."""
    s = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(s))
    if norm < 1.0e-10:
        return 0.0, 0.0
    s = s / norm
    if abs(s[0]) >= 1.0e-10:
        alpha = float(np.arctan2(s[1], s[0]))
    else:
        alpha = 0.0 if abs(s[1]) < 1.0e-10 else 0.5 * np.pi
    beta = float(np.arctan2(np.hypot(s[0], s[1]), s[2]))
    return alpha, beta


def so3_from_su2(u: np.ndarray) -> np.ndarray:
    """SO(3) rotation taking the frame's z axis to its physical direction.

    Defined by ``u sigma_b u^dag = sum_a O[a, b] sigma_a`` (active
    convention), so ``O e_z`` is the physical spin direction of the
    frame's positive basis spinor — the nominal quantization axis — and
    the story-001 output map is ``T_lattice = O T_leg O^T``.
    """
    u = np.asarray(u, dtype=complex)
    r = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            r[a, b] = 0.5 * np.real(
                np.trace(_PAULI[a + 1] @ u @ _PAULI[b + 1] @ u.conj().T)
            )
    return r


def frame_overlap(leg_direction, dump: CsoDump) -> np.ndarray:
    """Overlap ``M = U_leg^dag U_saxis`` of the dump and leg spin frames.

    ``M[t, s] = <chi_t(leg) | chi_s(SAXIS)>`` converts the collinear CPROJ
    spin channels (SAXIS quantization) into leg-frame spinor components.
    """
    leg_alpha, leg_beta = euler_angles(leg_direction)
    u_leg = vasp_spinor_frame(leg_alpha, leg_beta)
    u_saxis = vasp_spinor_frame(dump.alpha, dump.beta)
    return u_leg.conj().T @ u_saxis


# ---------------------------------------------------------------------------
# snapshot / dump consistency
# ---------------------------------------------------------------------------
def _ion_slices(snapshot: PawProjectorSnapshot) -> list[slice]:
    return [site.projector_slice for site in snapshot.site_layout]


def _check_consistency(snapshot: PawProjectorSnapshot, dump: CsoDump) -> None:
    if dump.nions != len(snapshot.site_layout):
        raise ValueError(
            f"dump nions={dump.nions} does not match export "
            f"nions={len(snapshot.site_layout)}"
        )
    if dump.ncdij != 2 or dump.lsorbit:
        raise ValueError(
            "the production W_SO source is the collinear strength-0 dump "
            "(ncdij=2, lsorbit=0); fully-relativistic legs are comparison "
            "only"
        )
    for ion, site in enumerate(snapshot.site_layout):
        npj = site.projector_slice.stop - site.projector_slice.start
        if npj != dump.type_of(ion).lmmax:
            raise ValueError(
                f"ion {ion}: export nproj={npj} != dump lmmax="
                f"{dump.type_of(ion).lmmax}"
            )


# ---------------------------------------------------------------------------
# W^K_SO assembly (CALC_PAW_OVERLAP contraction pattern)
# ---------------------------------------------------------------------------
def build_w_soc(
    snapshot: PawProjectorSnapshot,
    dump: CsoDump,
    skip_atoms: tuple[int, ...] = (),
) -> np.ndarray:
    """Assemble ``W^K_SO(k) = sum_a B_a^dag CSO_a B_a`` for every k.

    The collinear CPROJ channels are the SAXIS-frame spinor components of
    the bands, so the contraction is frame-independent (both the
    projections and ``CSO`` are SAXIS-frame objects); state indices are
    stacked spin-major, ``nu = spin * nband + band``.  ``skip_atoms``
    zeroes selected ions' one-center operator (ligand-SOC studies); W_SO
    itself is all-atom by default.  The result is the frame-independent
    *state-space* matrix ``<psi_nu|W_SO|psi_nu'>`` — legs re-express the
    band spinor components and vertices, never W.

    Returns the complex ``(nkpt, 2*nband, 2*nband)`` matrix in eV.
    """
    _check_consistency(snapshot, dump)
    coeffs = snapshot.coefficients  # (nspin, nkpt, nband, nproj), local
    nspin, nkpt, nband, _ = coeffs.shape
    if nspin != 2:
        raise ValueError(f"collinear export must have nspin=2, got {nspin}")
    nstate = nspin * nband
    w = np.zeros((nkpt, nstate, nstate), dtype=complex)
    for ion, sl in enumerate(_ion_slices(snapshot)):
        if ion in tuple(skip_atoms):
            continue
        npj = sl.stop - sl.start
        cso = np.empty((2 * npj, 2 * npj), dtype=complex)
        cso[:npj, :npj] = dump.cso[ion, 0, :npj, :npj]
        cso[:npj, npj:] = dump.cso[ion, 1, :npj, :npj]
        cso[npj:, :npj] = dump.cso[ion, 2, :npj, :npj]
        cso[npj:, npj:] = dump.cso[ion, 3, :npj, :npj]

        # B[k]: (2*npj, nstate) with rows spinor-major:
        #   row (p, t), column (s * nband + m)
        b = np.zeros((nkpt, 2 * npj, nstate), dtype=complex)
        b[:, :npj, :nband] = coeffs[0, :, :, sl].transpose(0, 2, 1)
        b[:, npj:, nband:] = coeffs[1, :, :, sl].transpose(0, 2, 1)
        w += np.einsum("kpa,pq,kqb->kab", b.conj(), cso, b, optimize="optimal")
    return w


# ---------------------------------------------------------------------------
# COCC reconstruction (weight/phase convention check)
# ---------------------------------------------------------------------------
def reconstruct_cocc(snapshot: PawProjectorSnapshot, dump: CsoDump) -> np.ndarray:
    """Rebuild the collinear augmentation occupations from the CPROJ.

    The dump stores the raw collinear ``CRHODE`` components in the spinor
    slots uu (spin channel 0) and dd (spin channel 1) with ud/du empty,
    accumulated as ``sum_k w_k f_nk conj(CPROJ(LP)) CPROJ(L)`` (VASP
    ``fast_aug.F`` order) over the *local* (translation-phase-removed)
    projector overlaps — the same-ion Bloch phases cancel inside the
    one-center block.  Matching the dump validates the export
    weight/occupation convention end to end; the residual on the real
    FeO dump (<= 1.5e-5 x scale, largest on the dd slot) is the
    story-010 patch's occupation-update skew between the COCC dump point
    and the exported band occupations, not a convention mismatch.

    Returns ``(nions, 4, lmdim_max, lmdim_max)`` in the dump layout.
    """
    _check_consistency(snapshot, dump)
    coeffs = snapshot.coefficients
    occ = snapshot.occupations
    if occ is None:
        raise ValueError("the native export carries no occupations")
    nspin, nkpt, nband, _ = coeffs.shape
    lmdim = dump.lmdim_max
    recon = np.zeros((dump.nions, 4, lmdim, lmdim), dtype=complex)
    for ion, sl in enumerate(_ion_slices(snapshot)):
        npj = sl.stop - sl.start
        for spin, slot in ((0, 0), (1, 3)):
            blk = np.zeros((npj, npj), dtype=complex)
            for k in range(nkpt):
                amp = snapshot.weights[k] * occ[spin, k, :nband]
                vecs = coeffs[spin, k, :, sl] * amp[:, None]
                blk += vecs.conj().T @ coeffs[spin, k, :, sl]
            recon[ion, slot, :npj, :npj] = blk
    return recon


# ---------------------------------------------------------------------------
# collinear snapshot -> spinor ProjectorGreenData (per leg)
# ---------------------------------------------------------------------------
def _site_splitting_operator(
    snapshot: PawProjectorSnapshot,
    frame_overlap_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Leg-frame magnetic vertices ``Delta_a = (D_up - D_down) sigma_z``.

    The collinear spin difference is the physical exchange splitting of
    the strength-0 reference; the vertex is the physical field along the
    SAXIS polarisation axis.  With ``frame_overlap_matrix`` ``M`` (the
    :func:`frame_overlap` map into a leg frame), the vertex is conjugated
    into that frame, ``Delta_a M sigma_z M^dag = Delta_a n^.sigma`` with
    ``n^`` the SAXIS axis in leg-frame coordinates — so a SAXIS-polarised
    collinear state sees its *full* splitting in every leg.  Without
    ``M`` (SAXIS frame) the vertex is diagonal with the full splitting on
    zz (pinned to the collinear reduction of the spinor kernel).  Units
    eV.
    """
    components = {c.name: c for c in snapshot.operators.components}
    if "total" not in components:
        raise ValueError("the native export lacks the total CDIJ component")
    blocks = np.asarray(components["total"].values, dtype=complex) * HARTREE_TO_EV
    if frame_overlap_matrix is None:
        sz = _PAULI[3]
    else:
        m = np.asarray(frame_overlap_matrix, dtype=complex)
        if m.shape != (2, 2):
            raise ValueError("frame overlap must be a 2x2 unitary")
        sz = m @ _PAULI[3] @ m.conj().T
    nions = len(snapshot.site_layout)
    nmax = max(
        site.projector_slice.stop - site.projector_slice.start
        for site in snapshot.site_layout
    )
    operator = np.zeros((nions, nmax, nmax, 2, 2), dtype=complex)
    for ion, site in enumerate(snapshot.site_layout):
        npj = site.projector_slice.stop - site.projector_slice.start
        delta = blocks[ion, :npj, :npj]
        for a in range(2):
            for b in range(2):
                operator[ion, :npj, :npj, a, b] = delta * sz[a, b]
    return operator


def collinear_snapshot_to_spinor_data(
    snapshot: PawProjectorSnapshot,
    dump: CsoDump,
    leg_direction=(0.0, 0.0, 1.0),
    frame_overlap_matrix: np.ndarray | None = None,
) -> ProjectorGreenData:
    """Build normalized no-SOC spinor band data in the leg frame.

    States ``(band m, spin channel s)`` are stacked spin-major with the
    spinor components rotated into the leg frame through
    ``M = U_leg^dag U_saxis``; the magnetic vertices are conjugated by
    the same map so the collinear splitting stays the physical SAXIS-axis
    field ``Delta_a M sigma_z M^dag`` in leg-frame components.
    """
    _check_consistency(snapshot, dump)
    if frame_overlap_matrix is None:
        frame_overlap_matrix = frame_overlap(leg_direction, dump)
    m = np.asarray(frame_overlap_matrix, dtype=complex)
    if m.shape != (2, 2):
        raise ValueError("frame overlap must be a 2x2 unitary")

    coeffs = snapshot.coefficients
    occ = snapshot.occupations
    nspin, nkpt, nband, nproj = coeffs.shape
    nstate = nspin * nband

    # components[k, (s, m), t, p] = coeffs[s, k, m, p] * M[t, s]
    coeff_full = np.zeros((nkpt, nstate, 2, nproj), dtype=complex)
    for s in (0, 1):
        coeff_full[:, s * nband : (s + 1) * nband, :, :] = np.einsum(
            "kmp,t->kmtp", coeffs[s], m[:, s]
        )
    coefficients = coeff_full[None, ...]

    eigenvalues = np.concatenate(
        [snapshot.eigenvalues[0], snapshot.eigenvalues[1]], axis=1
    )[None, ...]
    occupations = None
    if occ is not None:
        occupations = np.concatenate([occ[0], occ[1]], axis=1)[None, ...]

    nions = len(snapshot.site_layout)
    site_nproj = np.array(
        [
            site.projector_slice.stop - site.projector_slice.start
            for site in snapshot.site_layout
        ],
        dtype=int,
    )
    nmax = int(site_nproj.max())
    site_indices = np.full((nions, nmax), -1, dtype=int)
    projector_site = np.repeat(np.arange(nions), site_nproj)
    projector_l = np.empty(int(site_nproj.sum()), dtype=int)
    projector_m = np.empty_like(projector_l)
    projector_radial = np.empty_like(projector_l)
    for site, layout in enumerate(snapshot.site_layout):
        indices = np.arange(layout.projector_slice.start, layout.projector_slice.stop)
        site_indices[site, : len(indices)] = indices
        projector_l[indices] = [channel.l for channel in layout.channels]
        projector_m[indices] = [channel.m for channel in layout.channels]
        projector_radial[indices] = [channel.radial for channel in layout.channels]

    data = ProjectorGreenData(
        kpoints=snapshot.kpoints,
        weights=snapshot.weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=snapshot.efermi,
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=snapshot.cell,
        positions=snapshot.positions,
        atomic_numbers=snapshot.atomic_numbers,
        occupations=occupations,
        projector_l=projector_l,
        projector_m=projector_m,
        projector_radial=projector_radial,
        site_nproj=site_nproj,
        site_projector_indices=site_indices,
        nspinor=2,
        spinor_operator=_site_splitting_operator(snapshot, m),
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source="vasp.CPROJ_collinear_spinor_stacked",
        coefficient_projector="native_paw_projector",
        channel_interpretation="paw_projector_channel",
        operator_basis="vasp CDIJ spin difference D_up - D_down (collinear)",
        metadata={
            "nspinor": 2,
            "code": "vasp",
            "source_version": "6.4.1",
            "native_version": int(snapshot.provenance.get("native_version", 5)),
            "kpoint_storage": snapshot.provenance.get("kpoint_storage", "full_bz"),
            "leg_direction": [float(x) for x in leg_direction],
            "units": {
                "cell": "Angstrom",
                "positions": "Angstrom",
                "eigenvalues": "eV",
                "efermi": "eV",
                "spinor_operator": "eV",
            },
        },
    )
    data.validate(exchange_ready=True)
    return data


# ---------------------------------------------------------------------------
# single-leg exchange through the shared kernel
# ---------------------------------------------------------------------------
def compute_split_soc_exchange_leg(
    snapshot: PawProjectorSnapshot,
    dump: CsoDump,
    leg_direction=(0.0, 0.0, 1.0),
    lam: float = 1.0,
    sites=None,
    rpts=None,
    nz: int = 60,
    smearing_eV: float = 0.05,
    mode: str = MODE_SECOND_VARIATION,
    w_override: np.ndarray | None = None,
    skip_atoms: tuple[int, ...] = (),
    site_signs: dict | None = None,
    metadata: dict | None = None,
):
    """One leg: kernel exchange + story-001 O map into the lattice frame.

    The kernel returns the tensor in the leg frame (psi gauge, quantization
    along ``leg_direction``); the entries are rotated with
    ``T_lattice = O T_leg O^T``, ``O = SO3(U_leg)``, ``O e_z = leg axis``.
    ``Jiso`` is rotationally invariant; DMI and Jani rotate as vectors and
    rank-2 tensors.  The leg frame re-expresses the band spinor components
    and the magnetic vertices by ``M = U_leg^dag U_saxis``; W_SO is the
    frame-independent state-space matrix and enters unchanged.  The
    site-magnetization signs are physical (SAXIS-frame splitting signs),
    so they are computed once from the un-conjugated vertex and passed to
    the kernel explicitly — the leg-frame vertex's z-trace vanishes for
    transverse legs and must not be used as a sign probe.
    """
    if mode not in SPLIT_SOC_MODES:
        raise ValueError(f"unsupported split-SOC mode: {mode!r}")
    leg_direction = np.asarray(leg_direction, dtype=float)
    w = (
        build_w_soc(snapshot, dump, skip_atoms=skip_atoms)
        if w_override is None
        else np.asarray(w_override)
    )
    data = collinear_snapshot_to_spinor_data(snapshot, dump, leg_direction)
    leg_alpha, leg_beta = euler_angles(leg_direction)

    if site_signs is None:
        saxis_operator = _site_splitting_operator(snapshot)
        wanted = range(len(snapshot.site_layout)) if sites is None else sites
        site_signs = {
            int(site): site_magnetization_sign(saxis_operator[site]) for site in wanted
        }
    extra = dict(metadata or {})
    extra.setdefault("soc_operator_source", _SOC_OPERATOR_SOURCE)
    extra.setdefault(
        "strength0_reference",
        {
            "description": _STRENGTH0_SOURCE,
            "native_version": snapshot.provenance.get("native_version"),
        },
    )
    extra.setdefault(
        "frame",
        {
            "saxis": [float(x) for x in dump.saxis],
            "saxis_alpha": float(dump.alpha),
            "saxis_beta": float(dump.beta),
            "leg_direction": [float(x) for x in leg_direction],
            "frame_map": "T_lattice = O T_leg O^T with O = SO3(ROTMAT(alpha, beta)), O e_z = leg axis",
            "leg_frame_conjugation": (
                "band spinor components and magnetic vertices "
                "(Delta M sigma_z M^dag) conjugated by M = U_leg^dag "
                "U_saxis; W_SO is the frame-independent state-space "
                "matrix (unchanged); site signs probed in the SAXIS frame"
            ),
            "pauli_order": "x,y,z",
        },
    )
    extra.setdefault("merge_mode", "three_leg_rotate_merge")
    extra.setdefault("backend", "vasp")
    extra["lambda"] = float(lam)

    res = compute_ks_split_soc_exchange(
        data,
        w,
        lam=lam,
        mode=mode,
        Rpts=rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        site_signs=site_signs,
        metadata=extra,
    )

    o_map = so3_from_su2(vasp_spinor_frame(leg_alpha, leg_beta))
    exchange = {}
    for key, entry in res["exchange"].items():
        tensor = np.asarray(entry["tensor"], dtype=float)
        dmi = np.asarray(entry["dmi"], dtype=float)
        jani = np.asarray(entry["jani"], dtype=float)
        entry = dict(entry)
        entry["tensor"] = o_map @ tensor @ o_map.T
        entry["dmi"] = o_map @ dmi
        entry["jani"] = o_map @ jani @ o_map.T
        entry["tensor_leg_frame"] = tensor
        exchange[key] = entry
    return {
        "exchange": exchange,
        "metadata": res["metadata"],
        "o_map": o_map,
        "leg_direction": leg_direction,
    }


# ---------------------------------------------------------------------------
# three-leg rotate/merge driver
# ---------------------------------------------------------------------------
def _resolve_magnetic_sites(
    snapshot: PawProjectorSnapshot, magnetic_elements, index_magnetic_atoms
):
    nions = len(snapshot.site_layout)
    if index_magnetic_atoms is not None:
        sites = [int(i) - 1 for i in index_magnetic_atoms]
    elif magnetic_elements:
        wanted = {s.strip().capitalize() for s in magnetic_elements}
        sites = [
            i
            for i, site in enumerate(snapshot.site_layout)
            if site.species.capitalize() in wanted
        ]
    else:
        sites = list(range(nions))
    if not sites:
        raise ValueError("no magnetic sites selected")
    if any(s < 0 or s >= nions for s in sites):
        raise ValueError(f"magnetic sites out of range: {sites}")
    return sites


def _leg_tag(direction) -> str:
    axis = int(np.argmax(np.abs(np.asarray(direction))))
    return "xyz"[axis]


def _write_leg_results(
    snapshot: PawProjectorSnapshot,
    leg_result,
    sites,
    signs,
    leg_direction,
    path,
    rpts,
    description,
):
    """Write one leg's TB2J results (noncollinear SpinIO pickle)."""
    from ase import Atoms

    from TB2J.io_exchange.io_exchange import SpinIO

    leg_direction = np.asarray(leg_direction, dtype=float)
    atoms = Atoms(
        numbers=snapshot.atomic_numbers,
        positions=snapshot.positions,
        cell=snapshot.cell,
        pbc=True,
    )
    spinat = np.zeros((len(atoms), 3), dtype=float)
    for site in sites:
        spinat[site] = signs[site] * leg_direction

    index_spin = [-1] * len(atoms)
    site_to_spin = {}
    for ispin, site in enumerate(sites):
        index_spin[site] = ispin
        site_to_spin[site] = ispin

    exchange_jdict = {}
    dmi_ddict = {}
    jani_dict = {}
    distance_dict = {}
    for (r, i, j), entry in leg_result["exchange"].items():
        if i == j and not any(r):
            continue  # onsite pair: excluded (collinear-path cutover)
        key = (tuple(int(x) for x in r), site_to_spin[i], site_to_spin[j])
        vector = np.asarray(r) @ snapshot.cell + atoms.positions[j] - atoms.positions[i]
        distance_dict[key] = (vector, float(np.linalg.norm(vector)))
        exchange_jdict[key] = float(entry["Jiso"])
        dmi_ddict[key] = np.asarray(entry["dmi"], dtype=float)
        jani_dict[key] = np.asarray(entry["jani"], dtype=float)

    output = SpinIO(
        atoms=atoms,
        charges=np.zeros(len(atoms), dtype=float),
        spinat=spinat,
        index_spin=index_spin,
        colinear=False,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_jdict,
        dmi_ddict=dmi_ddict,
        Jani_dict=jani_dict,
        description=description,
    )
    output.write_all(path=path)
    return path


def gen_exchange_vasp_split_soc(
    native_input,
    cso_dump,
    output_path="TB2J_results_vasp_split_soc",
    rpts=None,
    rcut: float = 10.0,
    nz: int = 60,
    smearing_eV: float = 0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    lam: float = 1.0,
    mode: str = MODE_SECOND_VARIATION,
    legs=LEGS,
):
    """Three-direction split-SOC exchange from one collinear VASP run.

    Runs the kernel per leg (x, y, z), applies the corrected O map, writes
    per-leg TB2J results, and merges them through :mod:`TB2J.io_merge`.
    Returns the merged output directory.

    Note: the io_merge stage reconstructs Jani/DMI through the shared
    A-channel mapping, which is invalid (cross-story projector_green
    finding); the merged Jani/DMI are quarantined pending the rank-9
    raw-tensor merge cutover.  The per-leg pickles and
    ``split_soc_provenance.json`` are the authoritative outputs.
    """
    if mode not in SPLIT_SOC_MODES:
        raise ValueError(f"unsupported split-SOC mode: {mode!r}")
    snapshot = read_vasp_native(native_input)
    dump = read_cso_dump(cso_dump)
    _check_consistency(snapshot, dump)

    # input integrity: COCC must reconstruct from the SAME run's CPROJ
    # (catches mismatched native/cso artifact pairs before any physics)
    recon = reconstruct_cocc(snapshot, dump)
    for ion in range(dump.nions):
        n = dump.type_of(ion).lmmax
        for slot in (0, 3):
            scale = max(1.0, float(np.abs(dump.cocc[ion, slot, :n, :n]).max()))
            residual = float(
                np.abs(recon[ion, slot, :n, :n] - dump.cocc[ion, slot, :n, :n]).max()
            )
            if residual >= 1.0e-4 * scale:
                raise ValueError(
                    f"tb2j_cso.bin does not match tb2j_native.bin: ion {ion} "
                    f"COCC slot {slot} residual {residual:.3e} >= "
                    f"{1.0e-4 * scale:.3e} (different runs or corrupted dump)"
                )

    sites = _resolve_magnetic_sites(snapshot, magnetic_elements, index_magnetic_atoms)
    data_probe = collinear_snapshot_to_spinor_data(snapshot, dump)
    if rpts is None:
        from TB2J.interfaces.gpaw_projector import _R_grid_for_cutoff

        rpts = _R_grid_for_cutoff(data_probe, sites, rcut)
    del data_probe

    signs = {}
    operator = _site_splitting_operator(snapshot)
    for site in sites:
        signs[site] = site_magnetization_sign(operator[site])

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    leg_paths = []
    leg_records = {}
    for leg_direction in legs:
        result = compute_split_soc_exchange_leg(
            snapshot,
            dump,
            leg_direction=leg_direction,
            lam=lam,
            sites=sites,
            rpts=rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            mode=mode,
            site_signs=signs,
        )
        tag = _leg_tag(leg_direction)
        leg_dir = output_path / f"leg_{tag}"
        leg_dir.mkdir(exist_ok=True)
        _write_leg_results(
            snapshot,
            result,
            sites,
            signs,
            leg_direction,
            leg_dir,
            rpts,
            description=(
                "VASP KS-basis split-SOC exchange (ADR-7, story 011): "
                f"leg direction {tuple(float(x) for x in leg_direction)}, "
                "second variation of the collinear strength-0 bands with "
                "the patched one-center CSO operator; O map "
                "T_lattice = O T_leg O^T.\n"
            ),
        )
        leg_paths.append(leg_dir)
        leg_records[tag] = {
            "leg_direction": [float(x) for x in result["leg_direction"]],
            "o_map": np.asarray(result["o_map"], dtype=float).tolist(),
            "kernel_metadata": result["metadata"],
        }
        if _leg_tag(legs[0]) == tag and not any(
            not (i == j and not any(r)) for (r, i, j) in result["exchange"]
        ):
            raise ValueError(
                f"no magnetic spin pairs within Rcut={rcut} A: the leg "
                "exchange is empty after excluding the onsite pair.  "
                "Increase Rcut."
            )

    _merge_leg_results(leg_paths, output_path)

    provenance = {
        "schema": "tb2j.vasp_split_soc_provenance/1.0",
        "backend": "vasp",
        "mode": mode,
        "lambda": float(lam),
        "native_input": str(native_input),
        "cso_dump": str(cso_dump),
        "native_version": int(snapshot.provenance.get("native_version", 5)),
        "saxis_frame": {
            "saxis": [float(x) for x in dump.saxis],
            "alpha": float(dump.alpha),
            "beta": float(dump.beta),
        },
        "magnetic_sites": [int(s) for s in sites],
        "site_species": [site.species for site in snapshot.site_layout],
        "magnetic_signs": {str(s): float(signs[s]) for s in sites},
        "rpts": np.asarray(rpts, dtype=int).tolist(),
        "nz": int(nz),
        "smearing_eV": float(smearing_eV),
        "legs": leg_records,
        "leg_paths": [str(p) for p in leg_paths],
        "merge": {
            "inputs": [str(p) for p in leg_paths],
            "write_path": str(output_path),
        },
    }
    with open(output_path / "split_soc_provenance.json", "w") as handle:
        json.dump(provenance, handle, indent=2, default=str)
    return output_path


def _merge_leg_results(leg_paths, output_path):
    """Merge the per-leg results into ``output_path``.

    Seam for the rank-9 raw-tensor merge cutover: currently the legacy
    :func:`TB2J.io_merge.merge` (its Jani/DMI stage is invalid — see the
    gen_exchange_vasp_split_soc quarantine note); it will be replaced by
    ``TB2J.split_soc_kernel.merge_transverse_legs`` once the tangent-core
    contract lands, with the raw per-leg tensors as inputs.
    """
    from TB2J.io_merge import merge

    merge(*[str(p) for p in leg_paths], write_path=str(output_path))
