"""Reader for VASP native PAW export (tb2j_native.bin v4–v6).

Reads the flat binary written by the VASP source-side exporter patch
(``src/tb2j_export.F``) and constructs a ``PawProjectorSnapshot``.

Version 5 stores complex CPROJ/CDIJ, per-type element labels, and VASP's
AE partial-wave population metric (QTOT).  The metric is used only for
reported charges/moments, never for the exchange Green function.

Version 6 stores compact IBZ spectral arrays plus a self-contained
full-BZ expansion plan authored entirely by VASP.  TB2J validates and
applies the plan before building any snapshot (Story 003).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sisl import Atom

from TB2J.paw_projector import (
    PawOperatorComponent,
    PawOperatorComponents,
    PawProjectorChannel,
    PawProjectorSnapshot,
    PawSiteLayout,
)


@dataclass(frozen=True)
class VaspIbzExpansionPlan:
    """Decoded VASP v6 IBZ-to-full-BZ expansion plan.

    Every full-BZ projector action is authored by VASP through its native
    ``KPOINTS_FULL`` / ``ROTATE_VECTOR`` path.  TB2J only validates and
    applies the dense action; it never reconstructs VASP symmetry algebra.

    Attributes
    ----------
    bz_kpoints : ndarray (nkpt_bz, 3)
        Full-BZ k-points in VASP fractional coordinates.
    parent_ibz : ndarray (nkpt_bz,) intp
        Zero-based source IBZ index for each full-BZ target.
    source_spin : ndarray (nspin, nkpt_bz) intp
        Zero-based parent spin feeding each target spin.
    conjugate : ndarray (nspin, nkpt_bz) bool
        Conjugate parent coefficients before applying the action.
    projector_actions : ndarray (nspin, nkpt_bz, nproj, nproj) complex128
        Dense physical-projector transformation (target = action @ source).
        For v7 spinor plans the action is dense in (spinor x projector):
        (1, nkpt_bz, 2*nproj, 2*nproj).
    symmetry_operation : ndarray (nkpt_bz,) intp
        VASP operation index (audit only).
    spinflip : ndarray (nkpt_bz,) bool
        VASP ``SPINFLIP`` audit flag.
    rssymop : ndarray (nkpt_bz, 2, 2) complex128 or None
        VASP per-BZ-k SU(2) spin rotation (audit; v7 only).
    """

    kpoint_storage_mode: int
    bz_kpoints: np.ndarray
    parent_ibz: np.ndarray
    source_spin: np.ndarray
    conjugate: np.ndarray
    projector_actions: np.ndarray
    symmetry_operation: np.ndarray
    spinflip: np.ndarray
    rssymop: np.ndarray | None = None


def _read_v6_expansion_plan(
    f, nspin: int, nproj_physical: int, nkpt_ibz: int, nrspinors: int = 1
) -> VaspIbzExpansionPlan:
    """Decode the v6 IBZ expansion section from an open binary stream.

    The file position must be immediately after the static type/site
    metadata (type_labels) and before the IBZ spectral arrays.

    Fail-closed: any malformed dimension, out-of-range index, or invalid
    logical flag is rejected before any exchange-capable data is returned.
    """
    kpoint_storage_mode = np.frombuffer(f.read(4), dtype="<i4")[0]
    if kpoint_storage_mode != 1:
        raise ValueError(
            f"unsupported v6 kpoint_storage_mode {kpoint_storage_mode}; "
            f"expected 1 (ibz_with_expansion_plan)"
        )

    nkpt_bz = np.frombuffer(f.read(4), dtype="<i4")[0]
    if nkpt_bz < 1:
        raise ValueError(f"v6 nkpt_bz={nkpt_bz} must be positive")
    if nkpt_bz < nkpt_ibz:
        raise ValueError(
            f"v6 nkpt_bz ({nkpt_bz}) < nkpt_ibz ({nkpt_ibz}); "
            f"cannot cover IBZ parents"
        )

    vkpt_bz = (
        np.frombuffer(f.read(nkpt_bz * 3 * 8), dtype="<f8")
        .reshape(3, nkpt_bz, order="F")
        .T
    )
    parent_ibz = np.frombuffer(f.read(nkpt_bz * 4), dtype="<i4").astype(np.intp)
    if parent_ibz.size != nkpt_bz:
        raise ValueError("truncated v6 parent_ibz array")
    if np.any(parent_ibz < 0) or np.any(parent_ibz >= nkpt_ibz):
        raise ValueError(f"v6 parent_ibz indices out of range [0, {nkpt_ibz})")

    source_spin = (
        np.frombuffer(f.read(nspin * nkpt_bz * 4), dtype="<i4")
        .reshape(nspin, nkpt_bz, order="F")
        .astype(np.intp)
    )
    if np.any(source_spin < 0) or np.any(source_spin >= nspin):
        raise ValueError(f"v6 source_spin indices out of range [0, {nspin})")

    conjugate_raw = np.frombuffer(f.read(nspin * nkpt_bz * 4), dtype="<i4").reshape(
        nspin, nkpt_bz, order="F"
    )
    if np.any((conjugate_raw != 0) & (conjugate_raw != 1)):
        raise ValueError("v6 conjugate flags must be 0 or 1")
    conjugate = conjugate_raw.astype(bool)

    nproj_action = nproj_physical * nrspinors
    action_count = nproj_action * nproj_action * nspin * nkpt_bz
    projector_actions = np.frombuffer(f.read(action_count * 16), dtype="<c16").reshape(
        nproj_action, nproj_action, nspin, nkpt_bz, order="F"
    )
    # Transpose to (nspin, nkpt_bz, nproj_action, nproj_action)
    projector_actions = np.ascontiguousarray(
        np.transpose(projector_actions, (2, 3, 0, 1))
    )
    if not np.all(np.isfinite(projector_actions)):
        raise ValueError("v6 projector_actions contain non-finite entries")

    symmetry_operation = np.frombuffer(f.read(nkpt_bz * 4), dtype="<i4").astype(np.intp)
    if symmetry_operation.size != nkpt_bz:
        raise ValueError("truncated v6 symmetry_operation array")

    spinflip_raw = np.frombuffer(f.read(nkpt_bz * 4), dtype="<i4")
    if spinflip_raw.size != nkpt_bz:
        raise ValueError("truncated v6 spinflip array")
    if np.any((spinflip_raw != 0) & (spinflip_raw != 1)):
        raise ValueError("v6 spinflip flags must be 0 or 1")
    spinflip = spinflip_raw.astype(bool)

    rssymop = None
    if nrspinors == 2:
        rssymop = np.ascontiguousarray(
            np.frombuffer(f.read(4 * nkpt_bz * 16), dtype="<c16")
            .reshape(2, 2, nkpt_bz, order="F")
            .transpose(2, 0, 1)
        )
        if not np.all(np.isfinite(rssymop)):
            raise ValueError("v7 rssymop contains non-finite entries")

    return VaspIbzExpansionPlan(
        kpoint_storage_mode=kpoint_storage_mode,
        bz_kpoints=vkpt_bz,
        parent_ibz=parent_ibz,
        source_spin=source_spin,
        conjugate=conjugate,
        projector_actions=projector_actions,
        symmetry_operation=symmetry_operation,
        spinflip=spinflip,
        rssymop=rssymop,
    )


def _validate_v6_expansion_plan(
    plan: VaspIbzExpansionPlan, nspin: int, nproj: int, nkpt_ibz: int
) -> None:
    """Validate all ADR-005 plan invariants before applying actions."""
    nkpt_bz = len(plan.parent_ibz)

    # BZ mesh completeness: no duplicates, Cartesian-product structure.
    canonical = np.mod(np.round(plan.bz_kpoints, decimals=8), 1.0)
    if len(np.unique(canonical, axis=0)) != nkpt_bz:
        raise ValueError("v6 BZ k-points contain periodic duplicates")
    # Cartesian-product completeness with tolerance grouping.
    tol = 1e-6
    product = 1
    for ax in range(3):
        vals = np.sort(canonical[:, ax])
        groups = 1
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] > tol:
                groups += 1
        product *= groups
    if product != nkpt_bz:
        raise ValueError("v6 BZ mesh is incomplete; not a Cartesian grid")

    # Action unitarity: every action must satisfy A @ A† ≈ I.
    nproj_a = plan.projector_actions.shape[-1]
    identity = np.eye(nproj_a, dtype=complex)
    for sigma in range(nspin):
        for K in range(nkpt_bz):
            A = plan.projector_actions[sigma, K]
            if not np.allclose(A @ A.conj().T, identity, atol=1e-8):
                raise ValueError(
                    f"v6 projector action is not unitary (spin={sigma}, " f"kpt={K})"
                )

    # Parent coverage: every declared IBZ parent must appear at least once.
    if len(np.unique(plan.parent_ibz)) < min(nkpt_ibz, nkpt_bz):
        raise ValueError("v6 plan does not cover all IBZ parents")


def _expand_vasp_native_ibz(
    celtot: np.ndarray,
    fertot: np.ndarray,
    cproj: np.ndarray,
    plan: VaspIbzExpansionPlan,
    nprod: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand VASP v6 IBZ spectral arrays to full-BZ using the decoded plan.

    Returns arrays in TB2J convention:
    - eigenvalues:    (nspin, nkpt_bz, nband)
    - occupations:    (nspin, nkpt_bz, nband)
    - coefficients:   (nspin, nkpt_bz, nband, nproj)
    """
    nspin = plan.source_spin.shape[0]
    nkpt_bz = len(plan.parent_ibz)
    nband = celtot.shape[0]

    _validate_v6_expansion_plan(plan, nspin, nprod, celtot.shape[1])

    eigenvalues = np.empty((nspin, nkpt_bz, nband), dtype=float)
    occupations = np.empty((nspin, nkpt_bz, nband), dtype=float)
    coefficients = np.empty((nspin, nkpt_bz, nband, nprod), dtype=complex)

    for sigma in range(nspin):
        for K in range(nkpt_bz):
            parent = int(plan.parent_ibz[K])
            src_spin = int(plan.source_spin[sigma, K])
            action = plan.projector_actions[sigma, K]

            eigenvalues[sigma, K] = celtot[:, parent, src_spin]
            occupations[sigma, K] = fertot[:, parent, src_spin]

            # Parent projector coefficients: (nproj, nband)
            coeff_parent = cproj[:nprod, :, parent, src_spin]
            if plan.conjugate[sigma, K]:
                coeff_parent = coeff_parent.conj()

            # target = action @ source → (nproj, nband), then → (nband, nproj)
            coefficients[sigma, K] = (action @ coeff_parent).T

    return eigenvalues, occupations, coefficients


HARTREE_TO_EV = 27.211386245988


def read_vasp_native(filename: str | Path) -> PawProjectorSnapshot:
    """Read a VASP native PAW export file and build a PawProjectorSnapshot."""
    filename = Path(filename)
    with open(filename, "rb") as f:
        magic = np.frombuffer(f.read(4), dtype="<i4")[0]
        if magic != 20260812:
            raise ValueError(f"invalid magic number: {magic}")
        raw_ver = np.frombuffer(f.read(4), dtype="<i4")
        if raw_ver.size == 0:
            raise ValueError("truncated VASP export header")
        version = raw_ver[0]
        if version not in {4, 5, 6, 7}:
            raise ValueError(f"unsupported version {version}; need v4, v5, v6, or v7")

        nspin = np.frombuffer(f.read(4), dtype="<i4")[0]
        ncdij = np.frombuffer(f.read(4), dtype="<i4")[0]
        nkpt = np.frombuffer(f.read(4), dtype="<i4")[0]
        nband = np.frombuffer(f.read(4), dtype="<i4")[0]
        nprod_stream = np.frombuffer(f.read(4), dtype="<i4")[0]
        nions = np.frombuffer(f.read(4), dtype="<i4")[0]
        ntyp = np.frombuffer(f.read(4), dtype="<i4")[0]
        lmdim_max = np.frombuffer(f.read(4), dtype="<i4")[0]
        lmax_max = np.frombuffer(f.read(4), dtype="<i4")[0] if version >= 5 else None

        lattice = np.frombuffer(f.read(9 * 8), dtype="<f8").reshape(3, 3, order="F")
        posion = (
            np.frombuffer(f.read(nions * 3 * 8), dtype="<f8")
            .reshape(3, nions, order="F")
            .T
        )
        ion_offset = np.frombuffer(f.read(nions * 4), dtype="<i4")
        ion_nproj = np.frombuffer(f.read(nions * 4), dtype="<i4")
        ion_ityp = np.frombuffer(f.read(nions * 4), dtype="<i4")

        # VASP allocates CPROJ to NPROD = ceil(NPRO / NB_PAR) * NB_PAR.
        # Its tail rows are padding; the per-ion LMMAX metadata is the physical
        # projector layout that must define the exported snapshot.
        nprod = int(ion_nproj.sum())
        if nprod > nprod_stream:
            raise ValueError(
                "VASP CPROJ stream is shorter than its ion projector layout"
            )
        _lmmax_typ = np.frombuffer(f.read(ntyp * 4), dtype="<i4")
        if version >= 5:
            lps_typ = np.frombuffer(f.read(lmax_max * ntyp * 4), dtype="<i4").reshape(
                lmax_max, ntyp, order="F"
            )
            qtot_typ = np.frombuffer(
                f.read(lmax_max * lmax_max * ntyp * 8), dtype="<f8"
            ).reshape(lmax_max, lmax_max, ntyp, order="F")
        else:
            lps_typ = qtot_typ = None
        _zval_typ = np.frombuffer(f.read(ntyp * 8), dtype="<f8")
        type_labels = np.frombuffer(f.read(ntyp * 2), dtype="S2")
        if version >= 7:
            nrspinors = np.frombuffer(f.read(4), dtype="<i4")[0]
        else:
            nrspinors = 1
        v6_plan = None
        if version >= 6:
            if version == 7 and nrspinors != 2:
                raise ValueError(f"v7 file with nrspinors={nrspinors}")
            v6_plan = _read_v6_expansion_plan(
                f, nspin, nprod, nkpt, nrspinors=nrspinors if version == 7 else 1
            )
        vkpt = (
            np.frombuffer(f.read(nkpt * 3 * 8), dtype="<f8")
            .reshape(3, nkpt, order="F")
            .T
        )
        wtkpt = np.frombuffer(f.read(nkpt * 8), dtype="<f8")
        efermi = np.frombuffer(f.read(8), dtype="<f8")[0]

        celtot = np.frombuffer(f.read(nband * nkpt * nspin * 8), dtype="<f8").reshape(
            nband, nkpt, nspin, order="F"
        )
        fertot = np.frombuffer(f.read(nband * nkpt * nspin * 8), dtype="<f8").reshape(
            nband, nkpt, nspin, order="F"
        )

        # CPROJ: complex (NPROD, NBAND, NKPT, NSPIN)
        cproj = np.frombuffer(
            f.read(nprod_stream * nband * nkpt * nspin * 16), dtype="<c16"
        ).reshape(nprod_stream, nband, nkpt, nspin, order="F")

        # CDIJ: complex (LMDIM_MAX, LMDIM_MAX, NIONS, NCDIJ)
        cdij = np.frombuffer(
            f.read(lmdim_max * lmdim_max * nions * ncdij * 16), dtype="<c16"
        ).reshape(lmdim_max, lmdim_max, nions, ncdij, order="F")

    if v6_plan is not None:
        eigenvalues, occupations, coefficients = _expand_vasp_native_ibz(
            celtot, fertot, cproj, v6_plan, nprod
        )
        vkpt = v6_plan.bz_kpoints
        nkpt_bz = len(v6_plan.parent_ibz)
        wtkpt = np.full(nkpt_bz, 1.0 / nkpt_bz)
    else:
        eigenvalues = np.transpose(celtot, (2, 1, 0)).copy()
        occupations = np.transpose(fertot, (2, 1, 0)).copy()
        coefficients = np.transpose(cproj, (3, 2, 1, 0))[..., :nprod].copy()

    # VASP CPROJ carries exp(-i 2π k·τ_i); TB2J applies real-space Bloch
    # factors itself, so convert each projector-site block to local overlaps.
    for ion, (offset, nproj_ion) in enumerate(zip(ion_offset, ion_nproj)):
        phase = np.exp(2j * np.pi * (vkpt @ posion[ion]))
        coefficients[:, :, :, int(offset) : int(offset + nproj_ion)] *= phase[
            None, :, None, None
        ]

    # Build the VASP-native per-site projector layout.
    site_layout = []
    for ion in range(nions):
        nproj_ion = int(ion_nproj[ion])
        offset = int(ion_offset[ion])
        ityp = int(ion_ityp[ion])
        symbol = bytes(type_labels[ityp - 1]).decode("ascii").strip()

        channels = []
        if lps_typ is None:
            idx = 0
            radial = 0
            li = 0
            while idx < nproj_ion:
                n_m = 2 * li + 1
                if idx + n_m <= nproj_ion:
                    channels.extend(
                        PawProjectorChannel(
                            l=li, m=m, radial=radial, label=f"n{radial}l{li}m{m}"
                        )
                        for m in range(-li, li + 1)
                    )
                    idx += n_m
                    li += 1
                else:
                    channels.append(
                        PawProjectorChannel(
                            l=0, m=0, radial=radial, label=f"n{radial}l0m0"
                        )
                    )
                    idx += 1
                if li > 3 and idx < nproj_ion:
                    li = 0
                    radial += 1
        else:
            radial_by_l = {}
            for ln, li in enumerate(lps_typ[:, ityp - 1]):
                if li < 0:
                    continue
                li = int(li)
                radial = radial_by_l.get(li, 0)
                channels.extend(
                    PawProjectorChannel(
                        l=li, m=m, radial=radial, label=f"n{radial}l{li}m{m}"
                    )
                    for m in range(-li, li + 1)
                )
                radial_by_l[li] = radial + 1
            if len(channels) != nproj_ion:
                raise ValueError("VASP LPS metadata does not match ion projector count")

        site_layout.append(
            PawSiteLayout(
                source_site=ion,
                species=symbol,
                atomic_number=Atom(symbol).Z,
                projector_slice=slice(offset, offset + nproj_ion),
                channels=tuple(channels),
                setup_hash=f"vasp_type{ityp}",
            )
        )

    # Build spin-difference operator from CDIJ (complex, should be Hermitian)
    if nspin == 2 and ncdij >= 2:
        delta = cdij[:, :, :, 0] - cdij[:, :, :, 1]
    elif nspin == 2 and ncdij == 1:
        delta = cdij[:, :, :, 0] * 0
    else:
        delta = cdij[:, :, :, 0]

    nmax = max(int(n) for n in ion_nproj)
    blocks = np.zeros((nions, nmax, nmax), dtype=complex)
    for ion in range(nions):
        nproj_ion = int(ion_nproj[ion])
        blocks[ion, :nproj_ion, :nproj_ion] = delta[:nproj_ion, :nproj_ion, ion]

    # VASP CDIJ in complex mode should be Hermitian; symmetrize numerically
    for ion in range(nions):
        nproj_ion = int(ion_nproj[ion])
        blocks[ion, :nproj_ion, :nproj_ion] = 0.5 * (
            blocks[ion, :nproj_ion, :nproj_ion]
            + blocks[ion, :nproj_ion, :nproj_ion].conj().T
        )

    population_metric_matrix = None
    if qtot_typ is not None:
        population_metric_matrix = np.zeros((nprod, nprod), dtype=float)
        for ion, layout in enumerate(site_layout):
            ityp = int(ion_ityp[ion]) - 1
            channel_groups = []
            offset = layout.projector_slice.start
            for ln, li in enumerate(lps_typ[:, ityp]):
                li = int(li)
                if li < 0:
                    continue
                n_m = 2 * li + 1
                channel_groups.append((li, np.arange(offset, offset + n_m, dtype=int)))
                offset += n_m
            for ln1, (l1, indices1) in enumerate(channel_groups):
                for ln2, (l2, indices2) in enumerate(channel_groups):
                    if l1 == l2:
                        population_metric_matrix[np.ix_(indices1, indices2)] = (
                            np.eye(len(indices1)) * qtot_typ[ln1, ln2, ityp]
                        )
    components = PawOperatorComponents(
        components=(
            PawOperatorComponent(
                name="total",
                values=blocks / HARTREE_TO_EV,
                units="Hartree",
                basis_id="native_paw_projector_hamiltonian",
                definition="VASP CDIJ spin difference (D_up - D_down)",
                source="VASP CDIJ",
            ),
        ),
        policy="authoritative_total",
        selected_names=("total",),
    )

    # VASP internal lattice unit is Angstrom; POSION is fractional
    cell = lattice.T.copy()
    positions = (posion @ lattice.T).copy()

    provenance = {
        "source_code": "vasp",
        "source_version": "6.4.1",
        "functional": "unknown",
        "setup_hashes": [f"vasp_type{ion_ityp[i]}" for i in range(nions)],
        "u_eV": 0.0,
        "stream_projector_capacity": int(nprod_stream),
        "j_eV": 0.0,
        "correlated_shells": [],
        "kpoint_storage": "ibz" if v6_plan is not None else "full_bz",
        "expanded_by": "tb2j.vasp_native" if v6_plan is not None else None,
        "nkpt_ibz": int(nkpt) if v6_plan is not None else None,
        "nkpt_bz": (int(len(v6_plan.parent_ibz)) if v6_plan is not None else None),
    }

    return PawProjectorSnapshot(
        kpoints=vkpt,
        weights=wtkpt,
        eigenvalues=eigenvalues,
        occupations=occupations,
        coefficients=coefficients,
        efermi=float(efermi),
        cell=cell,
        positions=positions,
        atomic_numbers=np.array(
            [site.atomic_number for site in site_layout], dtype=int
        ),
        site_layout=tuple(site_layout),
        operators=components,
        kpoint_mode="full_bz",
        selected_source_sites=tuple(range(nions)),
        provenance=provenance,
        population_metric_matrix=population_metric_matrix,
        population_metric=(
            "VASP PAW AE partial-wave overlap QTOT" if qtot_typ is not None else None
        ),
    )


def read_vasp_native_spinor(filename: str | Path):
    """Read a VASP v7 spinor native export into spinor ProjectorGreenData.

    Layout mirrors the Fortran exporter (patch tb2j_export.F, version 7):
    header (with nrspinors=2), static metadata, the v6/v7 expansion plan
    with dense spinor actions SL x conjg(RSSYMOP) and the rssymop audit
    array, IBZ spectral arrays with spinor-packed CPROJ (spinor-major
    halves), the Pauli CDIJ, and the converted 2x2 spin-matrix CDIJ.
    """
    from TB2J.projector_green import (
        SPINOR_OPERATOR_DEFINITION,
        ProjectorGreenData,
    )

    filename = Path(filename)
    with open(filename, "rb") as f:
        magic = np.frombuffer(f.read(4), dtype="<i4")[0]
        if magic != 20260812:
            raise ValueError(f"invalid magic number: {magic}")
        version = np.frombuffer(f.read(4), dtype="<i4")[0]
        if version != 7:
            raise ValueError(
                f"read_vasp_native_spinor requires version 7, got {version}"
            )
        nrspinors = np.frombuffer(f.read(4), dtype="<i4")[0]
        if nrspinors != 2:
            raise ValueError(f"v7 file with nrspinors={nrspinors}")
        nspin = np.frombuffer(f.read(4), dtype="<i4")[0]
        ncdij = np.frombuffer(f.read(4), dtype="<i4")[0]
        nkpt = np.frombuffer(f.read(4), dtype="<i4")[0]
        nband = np.frombuffer(f.read(4), dtype="<i4")[0]
        nprod_stream = np.frombuffer(f.read(4), dtype="<i4")[0]
        nions = np.frombuffer(f.read(4), dtype="<i4")[0]
        ntyp = np.frombuffer(f.read(4), dtype="<i4")[0]
        lmdim_max = np.frombuffer(f.read(4), dtype="<i4")[0]
        lmax_max = np.frombuffer(f.read(4), dtype="<i4")[0]

        lattice = np.frombuffer(f.read(9 * 8), dtype="<f8").reshape(3, 3, order="F")
        posion = (
            np.frombuffer(f.read(nions * 3 * 8), dtype="<f8")
            .reshape(3, nions, order="F")
            .T
        )
        ion_offset = np.frombuffer(f.read(nions * 4), dtype="<i4")
        ion_nproj = np.frombuffer(f.read(nions * 4), dtype="<i4")
        ion_ityp = np.frombuffer(f.read(nions * 4), dtype="<i4")
        nprod = int(ion_nproj.sum())
        if 2 * nprod > nprod_stream:
            raise ValueError("v7 CPROJ stream shorter than 2x scalar projector layout")
        _lmmax_typ = np.frombuffer(f.read(ntyp * 4), dtype="<i4")
        lps_typ = np.frombuffer(f.read(lmax_max * ntyp * 4), dtype="<i4").reshape(
            lmax_max, ntyp, order="F"
        )
        _qtot_typ = np.frombuffer(f.read(lmax_max * lmax_max * ntyp * 8), dtype="<f8")
        _zval_typ = np.frombuffer(f.read(ntyp * 8), dtype="<f8")
        type_labels = np.frombuffer(f.read(ntyp * 2), dtype="S2")

        plan = _read_v6_expansion_plan(f, nspin, nprod, nkpt, nrspinors=2)

        _vkpt_ibz = (
            np.frombuffer(f.read(nkpt * 3 * 8), dtype="<f8")
            .reshape(3, nkpt, order="F")
            .T
        )
        _wtkpt = np.frombuffer(f.read(nkpt * 8), dtype="<f8")
        efermi = np.frombuffer(f.read(8), dtype="<f8")[0]
        celtot = np.frombuffer(f.read(nband * nkpt * nspin * 8), dtype="<f8").reshape(
            nband, nkpt, nspin, order="F"
        )
        fertot = np.frombuffer(f.read(nband * nkpt * nspin * 8), dtype="<f8").reshape(
            nband, nkpt, nspin, order="F"
        )
        cproj = np.frombuffer(
            f.read(nprod_stream * nband * nkpt * nspin * 16), dtype="<c16"
        ).reshape(nprod_stream, nband, nkpt, nspin, order="F")
        _cdij_pauli = np.frombuffer(
            f.read(lmdim_max * lmdim_max * nions * ncdij * 16), dtype="<c16"
        )
        cdij_spinor = np.frombuffer(
            f.read(lmdim_max * lmdim_max * nions * 4 * 16), dtype="<c16"
        ).reshape(lmdim_max, lmdim_max, nions, 4, order="F")

    if nspin != 1:
        raise ValueError(f"v7 spinor file must carry nspin=1, got {nspin}")
    _validate_v6_expansion_plan(plan, nspin, nprod, nkpt)

    nkpt_bz = len(plan.parent_ibz)
    eigenvalues = np.empty((1, nkpt_bz, nband), dtype=float)
    occupations = np.empty((1, nkpt_bz, nband), dtype=float)
    coefficients = np.empty((1, nkpt_bz, nband, 2, nprod), dtype=complex)
    for K in range(nkpt_bz):
        parent = int(plan.parent_ibz[K])
        src_spin = int(plan.source_spin[0, K])
        action = plan.projector_actions[0, K]  # (2*nprod, 2*nprod)
        eigenvalues[0, K] = celtot[:, parent, src_spin]
        occupations[0, K] = fertot[:, parent, src_spin]
        coeff_parent = cproj[: 2 * nprod, :, parent, src_spin]
        if plan.conjugate[0, K]:
            coeff_parent = coeff_parent.conj()
        expanded = action @ coeff_parent  # (2*nprod, nband)
        coefficients[0, K, :, 0, :] = expanded[:nprod].T
        coefficients[0, K, :, 1, :] = expanded[nprod : 2 * nprod].T

    # Bloch phase conversion (identical to the collinear path)
    for ion, (offset, nproj_ion) in enumerate(zip(ion_offset, ion_nproj)):
        phase = np.exp(2j * np.pi * (plan.bz_kpoints @ posion[ion]))
        coefficients[..., int(offset) : int(offset + nproj_ion)] *= phase[
            None, :, None, None, None
        ]

    # Site metadata
    site_nproj = ion_nproj.astype(int)
    nmax = int(site_nproj.max())
    site_projector_indices = np.full((nions, nmax), -1, dtype=int)
    projector_site = np.repeat(np.arange(nions), site_nproj)
    projector_l = []
    projector_m = []
    projector_radial = []
    for ion in range(nions):
        ityp = int(ion_ityp[ion]) - 1
        start = int(ion_offset[ion])
        radial_by_l = {}
        for li in lps_typ[:, ityp]:
            li = int(li)
            if li < 0:
                continue
            radial = radial_by_l.get(li, 0)
            for m in range(-li, li + 1):
                projector_l.append(li)
                projector_m.append(m)
                projector_radial.append(radial)
            radial_by_l[li] = radial + 1
        site_projector_indices[ion, : site_nproj[ion]] = np.arange(
            start, start + site_nproj[ion]
        )

    # 2x2 spin operator from the converted CDIJ blocks
    spinor_operator = np.zeros((nions, nmax, nmax, 2, 2), dtype=complex)
    for ion in range(nions):
        nproj_ion = int(site_nproj[ion])
        block = spinor_operator[ion, :nproj_ion, :nproj_ion]
        block[:, :, 0, 0] = cdij_spinor[:nproj_ion, :nproj_ion, ion, 0]
        block[:, :, 0, 1] = cdij_spinor[:nproj_ion, :nproj_ion, ion, 1]
        block[:, :, 1, 0] = cdij_spinor[:nproj_ion, :nproj_ion, ion, 2]
        block[:, :, 1, 1] = cdij_spinor[:nproj_ion, :nproj_ion, ion, 3]
        herm = 0.5 * (block + block.transpose(1, 0, 3, 2).conj())
        spinor_operator[ion, :nproj_ion, :nproj_ion] = herm

    symbols = [
        bytes(type_labels[int(ion_ityp[i]) - 1]).decode("ascii").strip()
        for i in range(nions)
    ]
    atomic_numbers = np.array([Atom(symbol).Z for symbol in symbols], dtype=int)
    positions = (posion @ lattice.T).copy()

    data = ProjectorGreenData(
        kpoints=plan.bz_kpoints,
        weights=np.full(nkpt_bz, 1.0 / nkpt_bz),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=float(efermi),
        projector_site=projector_site,
        projector_atom=projector_site.copy(),
        cell=lattice.T.copy(),
        positions=positions,
        atomic_numbers=atomic_numbers,
        occupations=occupations,
        projector_l=np.array(projector_l, dtype=int),
        projector_m=np.array(projector_m, dtype=int),
        projector_radial=np.array(projector_radial, dtype=int),
        site_nproj=site_nproj,
        site_projector_indices=site_projector_indices,
        nspinor=2,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source="vasp.CPROJ_spinor_packed",
        coefficient_projector="native_paw_projector",
        channel_interpretation="paw_projector_channel",
        operator_basis="vasp CDIJ Pauli->2x2 (v7)",
        metadata={
            "nspinor": 2,
            "code": "vasp",
            "source_version": "6.4.1",
            "native_version": 7,
            "kpoint_storage": "ibz_with_expansion_plan",
            "expanded_by": "tb2j.vasp_native_spinor",
            "nkpt_ibz": int(nkpt),
            "nkpt_bz": int(nkpt_bz),
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
