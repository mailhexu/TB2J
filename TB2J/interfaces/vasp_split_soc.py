"""VASP KS-basis split-SOC adapter (ADR-7, story 011; tangent contract).

Consumes the two artifacts of a collinear strength-0 VASP run patched with
the story-010 dump hooks:

- ``tb2j_native.bin`` (v5/v6 collinear native export; complex W%CPROJ,
  bands, occupations, CDIJ) read by :func:`read_vasp_native`;
- ``tb2j_cso.bin`` (per-ion one-center SOC operator ``CSO``, augmentation
  occupations ``COCC``, spherical AE potential) read by the vendored
  :mod:`TB2J.interfaces.vasp_cso_dump` reader (format v1-v3).

Each run is one *magnetic reference*: its ISPIN=2 collinear states are
SAXIS-frame eigenstates and the dump CSO is the SAXIS-frame one-center
operator, so the leg problem is already in the psi gauge — band spinor
components, magnetic vertices ``Delta_a sigma_z`` and ``W_SO`` all live in
the same (SAXIS) spin frame with the full splitting on the frame's
``zz``.  The shared tangent kernel
(:func:`TB2J.split_soc_kernel.compute_ks_split_soc_exchange`) measures,
per pair, only the transverse 2x2 block of the reference's right-handed
``(u, v, n)`` triad (``J_leg``, masked on the ``n`` row/column).

One reference therefore never determines the full lattice tensor: the
production driver consumes **three independent strength-0 runs** with
``SAXIS = 1 0 0 / 0 1 0 / 0 0 1`` (the retained FeO campaign layout),
rotates each leg's measured block into the lattice frame
(``rotate_transverse_leg`` with ``O = SO3(ROTMAT(alpha, beta))``, the
SO(3) map of the run's SAXIS frame), and solves the raw 3x3 exchange
tensor from the 12 transverse constraints with
:func:`TB2J.split_soc_kernel.merge_transverse_legs` (design-matrix rank 9,
exact least squares; the legacy ``TB2J.io_merge`` scalar/traceless average
is never used — its decomposition biases anisotropy).  The decomposition
into Jiso/DMI/Jani is applied to the solved raw tensor
(:func:`TB2J.Jtensor.decompose_J_tensor`, Levi-Civita DMI convention).

The KS-band SOC operator follows the ``CALC_PAW_OVERLAP`` contraction
pattern,

.. math:: W^{\\mathcal K}_{SO}(k) = \\sum_a B_a(k)^\\dagger\\, CSO_a\\, B_a(k),

with :math:`B_a` the rectangular projector map of atom ``a``.  W_SO is
all-atom (ligand SOC enters the DMI); the magnetic rotation vertices are
site-local, magnetic-only, and carry the collinear spin splitting
``D_up - D_down``.  No PROCAR weights and no LOCPROJ/CDIJ-mismatched SOC
surrogate are used; the v7 spinor export remains the fully-relativistic
comparison leg only.
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
    json_safe_provenance,
    merge_transverse_legs,
    rotate_transverse_leg,
)

LEG_TAGS = ("x", "y", "z")
LEG_DIRECTIONS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

_PAULI = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)

_SOC_OPERATOR_SOURCE = (
    "VASP tb2j_cso.bin CSO (SPINORB_STRENGTH at the SAXIS Euler angles; "
    "story-010 patch dump, collinear branch, format v1-v3)"
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
    frame's positive basis spinor — the run's SAXIS axis — and the
    lattice-frame leg rotation is ``J_lattice = O J_leg O^T``.
    """
    u = np.asarray(u, dtype=complex)
    r = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            r[a, b] = 0.5 * np.real(
                np.trace(_PAULI[a + 1] @ u @ _PAULI[b + 1] @ u.conj().T)
            )
    return r


def leg_rotation_map(dump: CsoDump) -> np.ndarray:
    """SO(3) map of the run's SAXIS frame, ``O e_z = SAXIS`` direction.

    This is the rotation :func:`rotate_transverse_leg` needs to move the
    leg's measured transverse block into the lattice frame.
    """
    return so3_from_su2(vasp_spinor_frame(float(dump.alpha), float(dump.beta)))


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
    if dump.provenance is not None:
        # v2+ provenance: the dump must come from the SAME run as the
        # native export (band/k/efermi identity; k stored in IBZ form)
        prov = dump.provenance
        if prov["ispin"] != snapshot.coefficients.shape[0]:
            raise ValueError(
                f"dump ispin={prov['ispin']} != native export "
                f"nspin={snapshot.coefficients.shape[0]}"
            )
        if prov["nbands"] != snapshot.eigenvalues.shape[-1]:
            raise ValueError(
                f"dump nbands={prov['nbands']} != native export "
                f"nband={snapshot.eigenvalues.shape[-1]}"
            )
        expected_nk = snapshot.provenance.get("nkpt_ibz")
        if expected_nk is None:
            expected_nk = snapshot.kpoints.shape[0]
        if prov["nkpts"] != expected_nk:
            raise ValueError(
                f"dump nkpts={prov['nkpts']} != native export k count "
                f"{expected_nk} (different runs?)"
            )
        if not np.isclose(prov["efermi"], snapshot.efermi, atol=1.0e-8, rtol=0.0):
            raise ValueError(
                f"dump efermi={prov['efermi']} != native export "
                f"efermi={snapshot.efermi}"
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
    itself is all-atom by default.

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
# collinear snapshot -> spinor ProjectorGreenData (psi-gauge leg)
# ---------------------------------------------------------------------------
def _site_splitting_operator(snapshot: PawProjectorSnapshot) -> np.ndarray:
    """Psi-gauge magnetic vertices ``Delta_a sigma_z`` (SAXIS frame).

    The collinear spin difference is the physical exchange splitting of
    the strength-0 reference; the run's states are SAXIS-frame
    eigenstates, so the vertex is diagonal with the full splitting on
    ``zz`` — exactly the z magnetic reference the tangent kernel's leg
    frame expects.  Units eV.
    """
    components = {c.name: c for c in snapshot.operators.components}
    if "total" not in components:
        raise ValueError("the native export lacks the total CDIJ component")
    blocks = np.asarray(components["total"].values, dtype=complex) * HARTREE_TO_EV
    sz = _PAULI[3]
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
) -> ProjectorGreenData:
    """Build the psi-gauge no-SOC spinor band data of one strength-0 run.

    States ``(band m, spin channel s)`` are stacked spin-major; the
    collinear channels ARE the SAXIS-frame spinor components, so no frame
    conjugation is applied: the magnetic vertices stay ``Delta_a sigma_z``
    (full splitting on the frame z) and ``W_SO`` is the state-space matrix
    in the same frame.  This is the z magnetic reference leg the tangent
    kernel measures.
    """
    _check_consistency(snapshot, dump)

    coeffs = snapshot.coefficients
    occ = snapshot.occupations
    nspin, nkpt, nband, nproj = coeffs.shape
    nstate = nspin * nband

    # components[k, (s, m), t, p] with t the spinor slot of channel s
    coeff_full = np.zeros((nkpt, nstate, 2, nproj), dtype=complex)
    for s in (0, 1):
        coeff_full[:, s * nband : (s + 1) * nband, s, :] = coeffs[s]

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
        spinor_operator=_site_splitting_operator(snapshot),
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
# single-leg exchange through the shared tangent kernel
# ---------------------------------------------------------------------------
def compute_split_soc_exchange_leg(
    snapshot: PawProjectorSnapshot,
    dump: CsoDump,
    lam: float = 1.0,
    sites=None,
    rpts=None,
    nz: int = 60,
    smearing_eV: float = 0.05,
    mode: str = MODE_SECOND_VARIATION,
    w_override: np.ndarray | None = None,
    skip_atoms: tuple[int, ...] = (),
    metadata: dict | None = None,
):
    """One strength-0 run's raw leg through the tangent kernel.

    The run is already in the psi gauge (vertices ``Delta sigma_z`` with
    the full splitting on the frame z), so the kernel output is the
    MEASURED transverse 2x2 of the run's ``(u, v, n)`` triad — ``J_leg``
    masked on the ``n`` row/column, ``frame`` the detected leg frame,
    ``mask_residual`` the longitudinal spurion size.  No lattice-frame
    rotation is applied here: the driver maps the measured block with
    :func:`rotate_transverse_leg` and ``leg_rotation_map(dump)``.

    ``site_magnetization_sign`` is NOT passed to the kernel: the tangent
    vertices carry the site collinearity themselves, and the kernel
    derives the ``(u, v, n)`` frame from the measured splitting
    directions.
    """
    if mode not in SPLIT_SOC_MODES:
        raise ValueError(f"unsupported split-SOC mode: {mode!r}")
    w = (
        build_w_soc(snapshot, dump, skip_atoms=skip_atoms)
        if w_override is None
        else np.asarray(w_override)
    )
    data = collinear_snapshot_to_spinor_data(snapshot, dump)

    extra = dict(metadata or {})
    extra.setdefault("soc_operator_source", _SOC_OPERATOR_SOURCE)
    extra.setdefault(
        "strength0_reference",
        {
            "description": _STRENGTH0_SOURCE,
            "native_version": snapshot.provenance.get("native_version"),
            "saxis": [float(x) for x in dump.saxis],
            "saxis_alpha": float(dump.alpha),
            "saxis_beta": float(dump.beta),
        },
    )
    extra.setdefault(
        "frame",
        {
            "psi_gauge": (
                "collinear states are SAXIS-frame eigenstates; vertices "
                "Delta sigma_z with the full splitting on the frame z; "
                "W_SO is the state-space matrix in the same frame"
            ),
            "lattice_map": "J_lattice = O J_leg O^T with O = SO3(ROTMAT(alpha, beta)), O e_z = SAXIS",
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
        metadata=extra,
    )
    return {
        "exchange": res["exchange"],
        "metadata": res["metadata"],
        "o_map": leg_rotation_map(dump),
        "saxis": np.asarray(dump.saxis, dtype=float),
        "data": data,
        "w_soc": w,
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


def _write_leg_results(leg_result, sites, leg_tag, path):
    """Persist one leg's raw measured block (npz) and provenance (json)."""
    rotated = leg_result["rotated"]
    keys = sorted(rotated)
    np.savez_compressed(
        Path(path) / "split_soc_leg.npz",
        pair_keys=np.asarray([[*r, i, j] for r, i, j in keys], dtype=int),
        J_leg=np.asarray([rotated[key]["J_leg"] for key in keys]),
        axis=np.asarray(leg_result["saxis"], dtype=float),
    )
    provenance = json_safe_provenance(
        {
            **leg_result["metadata"],
            "backend": "vasp",
            "leg": leg_tag,
            "merge_mode": "three_leg_rotate_merge",
            "o_map": np.asarray(leg_result["o_map"], dtype=float).tolist(),
            "saxis": [float(x) for x in leg_result["saxis"]],
            "mask_residual_max": max(
                (float(entry.get("mask_residual", 0.0)) for entry in rotated.values()),
                default=0.0,
            ),
        }
    )
    (Path(path) / "split_soc_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    return provenance


def _write_merged_results(
    snapshot, exchange, sites, path, description, saxis_axis, provenance
):
    """Write the decomposed rank-nine exchange as TB2J results."""
    from ase import Atoms

    from TB2J.io_exchange.io_exchange import SpinIO

    atoms = Atoms(
        numbers=snapshot.atomic_numbers,
        positions=snapshot.positions,
        cell=snapshot.cell,
        pbc=True,
    )
    index_spin = [-1] * len(atoms)
    site_to_spin = {}
    for ispin, site in enumerate(sites):
        index_spin[site] = ispin
        site_to_spin[site] = ispin
    exchange_Jdict = {}
    dmi_ddict = {}
    Jani_dict = {}
    distance_dict = {}
    for (r, i, j), entry in exchange.items():
        vector = np.asarray(r) @ snapshot.cell + atoms.positions[j] - atoms.positions[i]
        distance = float(np.linalg.norm(vector))
        if distance < 1e-6:
            continue  # onsite pair excluded (spinor-path convention)
        key = (tuple(int(x) for x in r), site_to_spin[i], site_to_spin[j])
        distance_dict[key] = (vector, distance)
        exchange_Jdict[key] = float(entry["Jiso"])
        dmi_ddict[key] = np.asarray(entry["dmi"], dtype=float)
        Jani_dict[key] = np.asarray(entry["jani"], dtype=float)
    # SpinAT direction: the run's SAXIS axis with the physical moment sign.
    # Delta = D_up - D_down is a potential-like splitting; the majority-spin
    # potential is LOWER on a positive-moment site, so the moment sign is
    # the OPPOSITE of the vertex's z-trace sign (same PAW convention).
    spinat = np.zeros((len(atoms), 3), dtype=float)
    operator = _site_splitting_operator(snapshot)
    for site in sites:
        nproj = int(site_nproj_of(snapshot, site))
        vertex = operator[site, :nproj, :nproj]
        moment_sign = -site_magnetization_sign(vertex)
        spinat[site] = moment_sign * np.asarray(saxis_axis, dtype=float)
    output = SpinIO(
        atoms=atoms,
        charges=np.zeros(len(atoms), dtype=float),
        spinat=spinat,
        index_spin=index_spin,
        colinear=False,
        distance_dict=distance_dict,
        exchange_Jdict=exchange_Jdict,
        dmi_ddict=dmi_ddict,
        Jani_dict=Jani_dict,
        description=description
        + "\nsplit_soc_provenance: "
        + json.dumps(provenance, sort_keys=True, default=str),
    )
    output.split_soc_provenance = provenance
    output.write_all(path=str(path))
    return exchange_Jdict


def site_nproj_of(snapshot: PawProjectorSnapshot, site: int) -> int:
    sl = snapshot.site_layout[site].projector_slice
    return sl.stop - sl.start


def gen_exchange_vasp_split_soc(
    leg_artifacts,
    output_path="TB2J_results_vasp_split_soc",
    rpts=None,
    rcut: float = 10.0,
    nz: int = 60,
    smearing_eV: float = 0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    lam: float = 1.0,
    mode: str = MODE_SECOND_VARIATION,
    merge_consistency_atol: float = 1.0e-8,
    band_window_study: bool = True,
):
    """Three-reference split-SOC exchange from three collinear VASP runs.

    ``leg_artifacts`` maps the leg tags ``x``/``y``/``z`` to that run's
    two artifacts, ``{"native_input": tb2j_native.bin, "cso_dump":
    tb2j_cso.bin}`` — three independent strength-0 references with
    ``SAXIS = 1 0 0 / 0 1 0 / 0 0 1`` (one per tag; the dump's recorded
    SAXIS must be parallel to the tag axis).  Each run measures the
    transverse 2x2 of its own right-handed ``(u, v, n)`` triad through
    the shared tangent kernel; the measured blocks are rotated into the
    lattice frame and merged with the rank-nine raw-tensor solve
    (:func:`TB2J.split_soc_kernel.merge_transverse_legs`), never through
    the legacy ``TB2J.io_merge`` scalar/traceless average.

    Writes, under ``output_path``:

    - ``leg_<tag>/split_soc_leg.npz`` + ``leg_<tag>/split_soc_provenance.json``
      per reference (raw rotated block + full kernel provenance);
    - the merged TB2J results (Jiso/DMI/Jani of the solved raw tensor);
    - ``split_soc_provenance.json`` with the merge diagnostics.

    Returns the output directory.
    """
    if mode not in SPLIT_SOC_MODES:
        raise ValueError(f"unsupported split-SOC mode: {mode!r}")
    if set(leg_artifacts) != set(LEG_TAGS):
        raise ValueError(
            f"leg_artifacts must cover exactly the tags {LEG_TAGS}, got "
            f"{sorted(leg_artifacts)}"
        )

    runs = {}
    for tag in LEG_TAGS:
        spec = leg_artifacts[tag]
        if not isinstance(spec, dict) or not {"native_input", "cso_dump"} <= set(spec):
            raise ValueError(
                f"leg {tag!r} artifacts must be a mapping with "
                "'native_input' and 'cso_dump'"
            )
        snapshot = read_vasp_native(spec["native_input"])
        dump = read_cso_dump(spec["cso_dump"])
        _check_consistency(snapshot, dump)
        # the leg tag is the magnetic reference axis: the run's SAXIS must
        # be parallel to it, or the lattice-frame rotation is meaningless
        saxis = np.asarray(dump.saxis, dtype=float)
        saxis = saxis / np.linalg.norm(saxis)
        axis_vec = np.asarray(LEG_DIRECTIONS[tag], dtype=float)
        if abs(abs(float(np.dot(saxis, axis_vec))) - 1.0) > 1.0e-6:
            raise ValueError(
                f"leg {tag!r}: dump SAXIS {saxis.tolist()} is not parallel "
                f"to the reference axis {axis_vec.tolist()}"
            )
        # input integrity: COCC must reconstruct from the SAME run's CPROJ
        # (catches mismatched native/cso artifact pairs before any physics)
        recon = reconstruct_cocc(snapshot, dump)
        for ion in range(dump.nions):
            n = dump.type_of(ion).lmmax
            for slot in (0, 3):
                scale = max(1.0, float(np.abs(dump.cocc[ion, slot, :n, :n]).max()))
                residual = float(
                    np.abs(
                        recon[ion, slot, :n, :n] - dump.cocc[ion, slot, :n, :n]
                    ).max()
                )
                if residual >= 1.0e-4 * scale:
                    raise ValueError(
                        f"tb2j_cso.bin does not match tb2j_native.bin (leg "
                        f"{tag!r}): ion {ion} COCC slot {slot} residual "
                        f"{residual:.3e} >= {1.0e-4 * scale:.3e} (different "
                        "runs or corrupted dump)"
                    )
        runs[tag] = (snapshot, dump)

    reference_snapshot = runs["z"][0]
    sites = _resolve_magnetic_sites(
        reference_snapshot, magnetic_elements, index_magnetic_atoms
    )
    data_probe = collinear_snapshot_to_spinor_data(reference_snapshot, runs["z"][1])
    if rpts is None:
        from TB2J.interfaces.gpaw_projector import _R_grid_for_cutoff

        rpts = _R_grid_for_cutoff(data_probe, sites, rcut)
    rpts = np.asarray(rpts, dtype=int)
    nband = int(data_probe.eigenvalues.shape[-1])
    if band_window_study and (nband < 4 or nband % 2):
        raise ValueError(
            "band-window study requires at least two paired spinor band "
            f"windows (nband={nband}); disable band_window_study to skip"
        )
    del data_probe

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    leg_metadata = {}
    leg_exchanges = {}
    o_maps = {}
    for tag in LEG_TAGS:
        snapshot, dump = runs[tag]
        result = compute_split_soc_exchange_leg(
            snapshot,
            dump,
            lam=lam,
            sites=sites,
            rpts=rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            mode=mode,
        )
        exchange = result["exchange"]
        if not any(not (i == j and not any(r)) for (r, i, j) in exchange):
            raise ValueError(
                f"leg {tag!r}: no magnetic spin pairs within "
                f"Rcut={rcut} A (empty exchange after excluding the onsite "
                "pair). Increase Rcut."
            )
        if band_window_study:
            pair = (sites[0], sites[1] if len(sites) > 1 else sites[0])
            study = band_window_study_report(
                result,
                pair,
                nband,
                lam=lam,
                mode=mode,
                nz=nz,
                smearing_eV=smearing_eV,
                rpts=rpts,
                sites=sites,
            )
            result["metadata"]["band_window"] = {"convergence_study": study}
        rotated = rotate_transverse_leg(exchange, result["o_map"], LEG_TAGS.index(tag))
        leg_dir = output_path / f"leg_{tag}"
        leg_dir.mkdir(exist_ok=True)
        leg_metadata[tag] = _write_leg_results(
            {**result, "rotated": rotated}, sites, tag, leg_dir
        )
        leg_exchanges[tag] = {"exchange": rotated}
        o_maps[tag] = np.asarray(result["o_map"], dtype=float)

    merged = merge_transverse_legs(
        leg_exchanges, consistency_atol=merge_consistency_atol
    )
    merged_provenance = {
        "schema": "tb2j.vasp_split_soc_provenance/2.0",
        "backend": "vasp",
        "merge_mode": "raw_rank_nine",
        "mode": mode,
        "lambda": float(lam),
        "legs": leg_metadata,
        "rotation": (
            "J_lattice = O J_leg O^T with O = SO3(ROTMAT(alpha, beta)), "
            "O e_z = the run's SAXIS axis"
        ),
        "merge_diagnostics": json_safe_provenance(merged["diagnostics"]),
        "merge_consistency_atol_eV": float(merge_consistency_atol),
        "leg_paths": [str(output_path / f"leg_{tag}") for tag in LEG_TAGS],
        "o_maps": {tag: o_maps[tag].tolist() for tag in LEG_TAGS},
        "magnetic_sites": [int(s) for s in sites],
        "site_species": [site.species for site in reference_snapshot.site_layout],
        "rpts": rpts.tolist(),
        "nz": int(nz),
        "smearing_eV": float(smearing_eV),
        "units": "eV",
    }
    description = (
        "VASP patched-collinear split-SOC exchange (ADR-7, story 011, "
        "tangent contract): rank-nine raw tensor reconstructed from the "
        "x/y/z SAXIS transverse measurements.\n"
    )
    _write_merged_results(
        reference_snapshot,
        merged["exchange"],
        sites,
        output_path,
        description,
        np.asarray(LEG_DIRECTIONS["z"], dtype=float),
        merged_provenance,
    )
    (output_path / "split_soc_provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, default=str) + "\n"
    )
    return output_path


def band_window_study_report(
    leg_result,
    pair,
    nband,
    lam=1.0,
    mode=MODE_SECOND_VARIATION,
    nz=30,
    smearing_eV=0.05,
    rpts=None,
    sites=None,
):
    """ADR-8 band-window enlargement study for the leg's first pair."""
    from TB2J.split_soc_kernel import band_window_convergence_report

    data = leg_result["data"]
    return json_safe_provenance(
        band_window_convergence_report(
            data,
            leg_result["w_soc"],
            [nband - 2, nband],
            pair=pair,
            lam=lam,
            mode=mode,
            nz=nz,
            smearing_eV=smearing_eV,
            Rpts=rpts,
            sites=sites,
        )
    )


def _parse_leg_argument(value: str):
    """Parse ``TAG=RUN_DIR`` or ``TAG=NATIVE_PATH:CSO_PATH``."""
    tag, sep, rest = value.partition("=")
    tag = tag.strip().lower()
    if not sep or tag not in LEG_TAGS:
        raise ValueError(
            f"leg argument {value!r} must be TAG=RUN_DIR with TAG in " f"{LEG_TAGS}"
        )
    native, cso = None, None
    if ":" in rest:
        candidate_native, candidate_cso = rest.split(":", 1)
        from pathlib import Path as _Path

        if _Path(candidate_native).is_file() and _Path(candidate_cso).is_file():
            native, cso = candidate_native, candidate_cso
    if native is None:
        run_dir = Path(rest)
        native = run_dir / "tb2j_native.bin"
        cso = run_dir / "tb2j_cso.bin"
        if not (native.is_file() and cso.is_file()):
            raise ValueError(
                f"leg {tag!r}: {run_dir} must contain tb2j_native.bin and "
                "tb2j_cso.bin (or pass TAG=native_path:cso_path)"
            )
    return {tag: {"native_input": str(native), "cso_dump": str(cso)}}
