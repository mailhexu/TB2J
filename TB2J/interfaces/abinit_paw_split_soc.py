"""ABINIT PAW split-SOC leg builder and three-leg rotate/merge driver.

Story 006 (split-soc-ks spec, ADR-1/ADR-5).  Consumes the schema-1.1
``soc_pauli`` component of the collinear ABINIT ``savetb2j`` export
(story-005, ``pawdijso`` evaluated at the frozen strength-0 density,
unit strength, lattice frame, all atoms) together with the collinear
cprj bands, and feeds the story-002 KS-band split-SOC kernel:

- the normalized no-SOC spinor band data stacks the two collinear
  channels (up, down) into a ``ProjectorGreenData(nspinor=2)`` whose
  coefficient array is ``<p_mu | psi_{n,channel}>`` — the pinned ABINIT
  cprj bra convention, ``W^K_nm = sum_a C^dag_n W_a C_m``;
- ``W_SO^K(k)`` is assembled from the per-site soc_pauli blocks of ALL
  atoms (ligand SOC enters the propagator);
- magnetic rotation vertices are the existing ``delta_total`` (or
  ``delta_xc``) blocks tensored with ``sigma_z`` — magnetic sites only,
  never SOC;
- per leg direction ``n in {x, y, z}`` the SOC operator is re-quantized
  with the ABINIT ``spinaxis`` SU(2) rotation ``U_n`` (``geteuler``),
  the exchange tensors are computed in the leg frame, and rotated back
  to the lattice frame ``T_lattice = O_n T_leg O_n^T`` with
  ``O_n e_z = n``; the three legs are merged with ``TB2J.io_merge``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from TB2J.interfaces.abinit_savetb2j import (
    ABINIT_PAULI_SPIN_TREATMENT,
    load_abinit_savetb2j,
)
from TB2J.split_soc_kernel import band_window_convergence_report, json_safe_provenance

__all__ = [
    "LEG_DIRECTIONS",
    "SOC_PAULI_COMPONENT",
    "build_paw_split_soc_leg",
    "gen_exchange_abinit_paw_split_soc",
    "paw_split_soc_band_w_soc",
    "rotation_matrix_from_su2",
    "su2_leg_rotation",
]

SOC_PAULI_COMPONENT = "soc_pauli"
LEG_DIRECTIONS = ("x", "y", "z")

_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# ABINIT spinaxis SU(2) rotations (m_pawdij.F90 / geteuler conventions)
# ---------------------------------------------------------------------------


def _geteuler(axis):
    """ABINIT ``geteuler``: spinaxis (sx, sy, sz) -> (alpha, beta)."""
    sx, sy, sz = (float(v) for v in axis)
    norm = np.sqrt(sx * sx + sy * sy + sz * sz)
    if norm == 0.0:
        raise ValueError("spinaxis must be a nonzero vector")
    sx, sy, sz = sx / norm, sy / norm, sz / norm
    return float(np.arctan2(sy, sx)), float(np.arctan2(np.hypot(sx, sy), sz))


def _su2_from_euler(alpha, beta):
    """ABINIT ``pawdijso`` spinaxis SU(2) matrix U(alpha, beta)."""
    cb2, sb2 = np.cos(beta / 2.0), np.sin(beta / 2.0)
    em, ep = np.exp(-1j * alpha / 2.0), np.exp(1j * alpha / 2.0)
    return np.array([[cb2 * em, -sb2 * em], [sb2 * ep, cb2 * ep]], dtype=complex)


def su2_leg_rotation(leg):
    """SU(2) rotation quantizing the spin frame along the leg axis."""
    leg = str(leg).lower()
    if leg not in LEG_DIRECTIONS:
        raise ValueError(f"leg must be one of {LEG_DIRECTIONS}, got {leg!r}")
    axis = np.zeros(3)
    axis[LEG_DIRECTIONS.index(leg)] = 1.0
    alpha, beta = _geteuler(axis)
    return _su2_from_euler(alpha, beta)


def rotation_matrix_from_su2(u):
    """Physical rotation O_ab = 1/2 Re Tr[sigma_a U sigma_b U^dag]."""
    u = np.asarray(u, dtype=complex)
    sigma = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        _SIGMA_Y,
        _SIGMA_Z,
    ]
    o = np.empty((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            o[a, b] = np.real(np.trace(sigma[a] @ u @ sigma[b] @ u.conj().T)) / 2.0
    if (
        not np.allclose(o @ o.T, np.eye(3), atol=1e-10)
        or abs(np.linalg.det(o) - 1.0) > 1e-10
    ):
        raise ValueError("SU(2) matrix does not define a proper rotation")
    return o


def _rotate_spin_blocks(block, u):
    """Conjugate the trailing 2x2 spin indices: ``block -> U^dag block U``.

    Matches the ABINIT spinaxis packing ``Drot = U^dag D U`` so the leg
    operators are exactly the exporter's re-quantization convention.
    """
    return np.einsum("ws,pqwv,vt->pqst", u.conj(), np.asarray(block, dtype=complex), u)


def _rotate_entry_tensors(entry, o):
    """Rotate one leg exchange entry to the lattice frame: O T O^T."""
    out = dict(entry)
    dmi = o @ np.asarray(entry["dmi"], dtype=float)
    jani = o @ np.asarray(entry["jani"], dtype=float) @ o.T
    tensor = (
        float(entry["Jiso"]) * np.eye(3)
        + 0.5 * (dmi[:, None] - dmi[None, :])
        + 0.5 * (jani + jani.T)
    )
    out["dmi"] = dmi
    out["jani"] = jani
    out["tensor"] = tensor
    return out


# ---------------------------------------------------------------------------
# leg construction
# ---------------------------------------------------------------------------


def _load_soc_pauli_blocks(data):
    """Return the validated all-atom soc_pauli blocks (eV) from loaded data."""
    if not data.has_operator_component(SOC_PAULI_COMPONENT):
        raise ValueError(
            "no soc_pauli operator component: the ABINIT savetb2j file must "
            "be exported with savetb2j_soc=1 (schema 1.1)"
        )
    metadata = (data.operator_component_metadata or {}).get(SOC_PAULI_COMPONENT, {})
    if metadata.get("spin_treatment") != ABINIT_PAULI_SPIN_TREATMENT:
        raise ValueError(
            f"component {SOC_PAULI_COMPONENT!r} does not carry the "
            f"{ABINIT_PAULI_SPIN_TREATMENT!r} spin treatment"
        )
    if metadata.get("units") != "eV":
        raise ValueError("soc_pauli blocks must already be normalized to eV")
    return data.operator_components[SOC_PAULI_COMPONENT]


def _stacked_spinor_coefficients(coefficients, nband):
    """Interleave equal prefixes of up/down cprj bands in a spinor window.

    State ``2*n+s`` retains ``<p_mu|psi_{n,s}>`` on component ``s``;
    a window prefix of even length includes the same number of each spin.
    """
    coeff = np.asarray(coefficients, dtype=complex)
    nspin, nkpt, _, nproj = coeff.shape
    if nspin != 2:
        raise ValueError("collinear savetb2j data must have nspin=2 channels")
    stacked = np.zeros((1, nkpt, 2 * nband, 2, nproj), dtype=complex)
    for s in range(2):
        stacked[0, :, s::2, s, :] = coeff[s]
    return stacked


def build_paw_split_soc_leg(
    data,
    leg="z",
    sites_magnetic=None,
    vertex_component="delta_total",
):
    """Build the normalized no-SOC spinor band data for one leg.

    The stacked channels carry magnetic rotation vertices
    ``v_a = delta_a (x) sigma_z`` on the magnetic sites only (sourced from
    the existing collinear ``delta_total``/``delta_xc`` component, never
    from the SOC operator); the leg only re-quantizes the frame via the
    vertex convention — the SOC operator is rotated separately through
    :func:`paw_split_soc_band_w_soc`.
    """
    if getattr(data, "nspinor", 1) != 1 or data.coefficients.ndim != 4:
        raise ValueError("build_paw_split_soc_leg expects collinear savetb2j data")
    nband = data.nband
    if sites_magnetic is None:
        sites_magnetic = list(range(len(data.site_nproj)))
    sites_magnetic = [int(s) for s in sites_magnetic]

    from TB2J.projector_green import SPINOR_OPERATOR_DEFINITION

    if vertex_component not in {"delta_total", "delta_xc"}:
        raise ValueError(
            "magnetic vertices must be sourced from the existing collinear "
            f"delta_total or delta_xc component, got {vertex_component!r}"
        )
    if not data.has_operator_component(vertex_component):
        raise ValueError(f"missing magnetic-vertex component: {vertex_component}")

    coefficients = _stacked_spinor_coefficients(data.coefficients, nband)
    eigenvalues = np.empty((1, data.nkpt, 2 * nband), dtype=float)
    for s in range(2):
        eigenvalues[0, :, s::2] = data.eigenvalues[s]
    occupations = None
    if data.occupations is not None:
        occupations = np.empty_like(eigenvalues)
        for s in range(2):
            occupations[0, :, s::2] = data.occupations[s]

    nsite = len(data.site_nproj)
    nmax = data.site_projector_indices.shape[1]
    spinor_operator = np.zeros((nsite, nmax, nmax, 2, 2), dtype=complex)
    for site in sites_magnetic:
        delta = data.get_operator_component(vertex_component, site=site)
        block = np.einsum("pq,st->pqst", delta, _SIGMA_Z)
        nproj_site = data.site_nproj[site]
        spinor_operator[site, :nproj_site, :nproj_site] = block

    metadata = {
        **data.metadata,
        "split_soc_leg": {
            "backend": "abinit_paw",
            "leg": str(leg),
            "coefficient_convention": "cprj bra <p|psi> (ket-side spectral replay)",
            "vertex_component": vertex_component,
            "sites_magnetic": sites_magnetic,
        },
    }
    return type(data)(
        kpoints=data.kpoints,
        weights=data.weights,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=data.efermi,
        projector_site=data.projector_site,
        projector_atom=data.projector_atom,
        cell=data.cell,
        positions=data.positions,
        atomic_numbers=data.atomic_numbers,
        occupations=occupations,
        projector_l=data.projector_l,
        projector_m=data.projector_m,
        projector_radial=data.projector_radial,
        overlap_metric=data.overlap_metric,
        site_nproj=data.site_nproj,
        site_projector_indices=data.site_projector_indices,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        coefficient_source=data.coefficient_source,
        coefficient_projector=data.coefficient_projector,
        channel_interpretation="interleaved collinear channels (up, down)",
        operator_basis=data.operator_basis,
        metadata=metadata,
        nspinor=2,
    )


def paw_split_soc_band_w_soc(data, leg="z"):
    """Assemble the all-atom band-window SOC operator ``W_SO^K(k)`` (eV).

    ``W^K_nm(k) = sum_a C^dag_{n} W_a^{(leg)} C_{m}`` with the per-site
    soc_pauli blocks (already converted to eV by the loader) re-quantized
    to the leg frame with the ABINIT spinaxis SU(2) rotation.  All atoms
    contribute (ligand SOC enters the propagator); returned shape is
    ``(nkpt, 2*nband, 2*nband)``.
    """
    soc_blocks = _load_soc_pauli_blocks(data)
    u = su2_leg_rotation(leg)
    nband = data.nband
    coefficients = _stacked_spinor_coefficients(data.coefficients, nband)
    nkpt = data.nkpt

    w = np.zeros((nkpt, 2 * nband, 2 * nband), dtype=complex)
    nsite = len(data.site_nproj)
    for site in range(nsite):
        nproj_site = int(data.site_nproj[site])
        indices = data.site_projector_indices[site, :nproj_site]
        block = soc_blocks[site, :nproj_site, :nproj_site]
        block_leg = _rotate_spin_blocks(block, u)
        sub = np.take(coefficients[0], indices, axis=-1)
        w += np.einsum(
            "knsp,pqst,kmtq->knm", np.conj(sub), block_leg, sub, optimize="optimal"
        )
    return w


# ---------------------------------------------------------------------------
# three-leg rotate/merge driver
# ---------------------------------------------------------------------------


def _write_leg_results(
    data,
    exchange,
    sites,
    path,
    description,
    spinat_vectors,
    provenance,
):
    """Write one leg's exchange tensors (lattice frame) as TB2J results."""
    from ase import Atoms

    from TB2J.io_exchange.io_exchange import SpinIO

    atoms = Atoms(
        numbers=data.atomic_numbers,
        positions=data.positions,
        cell=data.cell,
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
        vector = np.asarray(r) @ data.cell + atoms.positions[j] - atoms.positions[i]
        distance = float(np.linalg.norm(vector))
        if distance < 1e-6:
            continue  # onsite pair excluded (spinor-path convention)
        key = (tuple(int(x) for x in r), site_to_spin[i], site_to_spin[j])
        distance_dict[key] = (vector, distance)
        exchange_Jdict[key] = float(entry["Jiso"])
        dmi_ddict[key] = np.asarray(entry["dmi"], dtype=float)
        Jani_dict[key] = np.asarray(entry["jani"], dtype=float)
    spinat = np.zeros((len(atoms), 3), dtype=float)
    for site in sites:
        spinat[site] = spinat_vectors[site]
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
        + json.dumps(provenance, sort_keys=True),
    )
    output.split_soc_provenance = provenance
    output.write_all(path=str(path))
    return exchange_Jdict


def gen_exchange_abinit_paw_split_soc(
    filename,
    output_path="TB2J_results_abinit_split_soc",
    legs=LEG_DIRECTIONS,
    Rcut=10.0,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    vertex_component="delta_total",
    lam=1.0,
    mode="second_variation",
    spinat_magnitude=1.0,
):
    """Three-direction split-SOC exchange from one schema-1.1 savetb2j file.

    For each leg ``n``: re-quantize the exported soc_pauli operator with
    the ABINIT spinaxis SU(2) rotation, run the story-002 KS second-
    variation kernel in the leg frame, rotate the output tensors to the
    lattice frame (``T_lattice = O_n T_leg O_n^T``, ``O_n e_z = n``), and
    write one noncollinear TB2J results directory per leg with
    ``spinat`` along the leg axis.  The three legs are then merged with
    ``TB2J.io_merge.merge`` into ``output_path``.

    Returns a dict with ``output_path`` (merged), ``leg_paths``,
    ``metadata`` (merged FR-050 provenance), and per-leg ``metadata``.
    """
    from TB2J.interfaces.gpaw_projector import (
        _magnetic_sites,
        _R_grid_for_cutoff,
    )
    from TB2J.io_merge import merge
    from TB2J.projector_green import site_magnetization_sign
    from TB2J.split_soc_kernel import compute_ks_split_soc_exchange

    legs = tuple(str(leg).lower() for leg in legs)
    if legs != LEG_DIRECTIONS:
        raise ValueError(f"legs must be {LEG_DIRECTIONS} in order, got {legs}")
    if mode != "second_variation":
        raise ValueError(
            "PAW three-leg SpinIO output requires absolute second_variation exchange"
        )
    if not np.isfinite(spinat_magnitude) or spinat_magnitude <= 0:
        raise ValueError("spinat_magnitude must be finite and positive")

    data = load_abinit_savetb2j(filename)
    _load_soc_pauli_blocks(data)
    if (
        magnetic_elements is None
        and index_magnetic_atoms is None
        and data.metadata.get("magnetic_moments") is None
    ):
        raise ValueError(
            "select magnetic sites with index_magnetic_atoms or magnetic_elements; the export includes ligand SOC"
        )
    sites = _magnetic_sites(
        data,
        magnetic_elements=magnetic_elements,
        index_magnetic_atoms=index_magnetic_atoms,
    )
    if not sites:
        raise ValueError(
            "no magnetic sites: pass index_magnetic_atoms or magnetic_elements"
        )
    if len(set(sites)) != len(sites) or any(
        site < 0 or site >= len(data.site_nproj) for site in sites
    ):
        raise ValueError("magnetic sites must be distinct in-range atom indices")
    if Rpts is None:
        Rpts = _R_grid_for_cutoff(data, sites, Rcut)
    Rpts = np.asarray(Rpts, dtype=int)

    output_root = Path(output_path)
    leg_dirs = {}
    leg_metadata = {}
    leg_Jdicts = {}
    digest = hashlib.sha256()
    source = Path(filename).resolve()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    schema_version = data.metadata.get("abinit_schema_version", "unknown")
    for leg in legs:
        u = su2_leg_rotation(leg)
        o = rotation_matrix_from_su2(u)
        leg_axis = o @ np.array([0.0, 0.0, 1.0])
        leg_data = build_paw_split_soc_leg(
            data,
            leg=leg,
            sites_magnetic=sites,
            vertex_component=vertex_component,
        )
        w_leg = paw_split_soc_band_w_soc(data, leg=leg)
        result = compute_ks_split_soc_exchange(
            leg_data,
            w_leg,
            lam=lam,
            mode=mode,
            Rpts=Rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=sites,
            metadata={
                "strength0_reference": {
                    "code": "abinit",
                    "schema": f"abinit.savetb2j.projector/{schema_version}",
                    "checkpoint": str(source),
                    "sha256": digest.hexdigest(),
                    "description": (
                        "collinear PAW savetb2j strength-0 export "
                        "(cprj bands, frozen density)"
                    ),
                },
                "soc_operator_source": (
                    "abinit savetb2j soc_pauli component (pawdijso at the "
                    "frozen density, unit strength, lattice frame)"
                ),
                "frame": {
                    "leg": leg,
                    "spinaxis": "0 0 1 (export) re-quantized with the ABINIT "
                    "spinaxis SU(2) rotation per leg",
                    "leg_axis_lattice": [float(v) for v in leg_axis],
                },
                "merge_mode": "three_leg_rotate_merge",
                "vertex_component": vertex_component,
            },
        )
        if leg_data.nband < 4:
            raise ValueError(
                "a PAW SOC window study requires at least two paired band prefixes"
            )
        study = band_window_convergence_report(
            leg_data,
            w_leg,
            [leg_data.nband - 2, leg_data.nband],
            pair=(sites[0], sites[1] if len(sites) > 1 else sites[0]),
            lam=lam,
            mode=mode,
            Rpts=Rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=sites,
        )
        result["metadata"]["band_window"]["convergence_study"] = json_safe_provenance(
            study
        )
        moments = data.metadata.get("magnetic_moments")
        result["metadata"]["frame"]["spinat_sign_source"] = (
            "exported magnetic_moments"
            if moments is not None
            else "opposite sign of up-minus-down PAW potential trace"
        )
        provenance = json_safe_provenance(
            {
                **result["metadata"],
                "backend": "abinit_paw",
                "leg": leg,
                "merge_mode": "three_leg_rotate_merge",
            }
        )
        spinat_vectors = {}
        for site in sites:
            nproj_site = int(data.site_nproj[site])
            vertex = leg_data.spinor_operator[site, :nproj_site, :nproj_site]
            signed_trace = float(
                np.real(np.trace(vertex[:, :, 0, 0] - vertex[:, :, 1, 1]))
            )
            if abs(signed_trace) <= 1e-10:
                raise ValueError(
                    f"cannot infer magnetic direction for site {site} from zero vertex trace"
                )
            if moments is not None:
                if len(moments) != len(data.site_nproj) or abs(moments[site]) <= 1e-3:
                    raise ValueError(
                        f"no nonzero magnetic moment available for site {site}"
                    )
                moment_sign = float(np.sign(moments[site]))
            else:
                # Majority-spin PAW potential is LOWER on a positive-moment
                # Fe site: Delta = H_up - H_down has the opposite sign to M.
                moment_sign = -site_magnetization_sign(vertex)
            spinat_vectors[site] = spinat_magnitude * moment_sign * leg_axis
        leg_dir = output_root / f"leg_{leg}"
        description = (
            f"ABINIT PAW split-SOC leg '{leg}' (story-006): KS second-variation "
            f"spectra from the schema-{schema_version} soc_pauli operator "
            "(all-atom propagator SOC, magnetic-only delta vertices), tensors "
            "rotated to the lattice frame with T_lattice = O T_leg O^T.\n"
        )
        leg_Jdicts[leg] = _write_leg_results(
            data,
            {
                key: _rotate_entry_tensors(entry, o)
                for key, entry in result["exchange"].items()
            },
            sites,
            leg_dir,
            description,
            spinat_vectors,
            provenance,
        )
        (leg_dir / "split_soc_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )
        leg_dirs[leg] = leg_dir
        leg_metadata[leg] = provenance

    merged_provenance = {
        "schema": leg_metadata[legs[0]]["schema"],
        "backend": "abinit_paw",
        "merge_mode": "three_leg_rotate_merge",
        "legs": leg_metadata,
        "rotation": "T_lattice = O T_leg O^T with O e_z = leg axis",
        "lambda": float(lam),
        "units": "eV",
    }
    merge(
        *[str(leg_dirs[leg]) for leg in legs],
        save=True,
        write_path=str(output_root),
        merged_provenance=merged_provenance,
    )
    (output_root / "split_soc_provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, default=str) + "\n"
    )
    return {
        "output_path": output_root,
        "leg_paths": leg_dirs,
        "metadata": merged_provenance,
        "leg_metadata": leg_metadata,
        "leg_Jdicts": leg_Jdicts,
    }
