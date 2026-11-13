"""Bulk magnon topology: Berry-curvature flux and band Chern numbers.

Implements the public ``calculate_bulk`` contract (architecture section 4):
a bulk request over a topology-ready model returns the endpoint-exclusive
periodic mesh, positive-mode energies in eV, discrete plaquette fluxes and
gauge-safe vertex curvature in the pinned chart/orientation, band Chern
values only for groups whose validity checks pass, and measured diagnostics.

Orientation and sign conventions are pinned by the executed certificate
``docs/sympy/magnon_topology_lswt_ferromagnet.py``: plaquette order
``(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)`` on the ``(edge, width)`` torus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from TB2J.magnon.topology.bosonic import BosonicModes, solve_bosonic_modes
from TB2J.magnon.topology.embedding import LatticeEmbedding, identity_embedding
from TB2J.magnon.topology.preparation import TopologyModel

__all__ = [
    "BulkRequest",
    "BulkTopologyResult",
    "calculate_bulk",
    "refine_bulk",
]

SCHEMA_VERSION = "tb2j-topology.bulk/1"


@dataclass(frozen=True)
class BulkRequest:
    """Bulk topology request over a topology-ready model.

    ``mesh`` is ``(nu, nv)``: the endpoint-exclusive periodic sampling of the
    embedding-basis reciprocal torus. ``bands`` are explicit zero-based
    indices into the ascending physical-mode energies forming ONE requested
    group; ``None`` selects all positive bands. ``groups`` requests several
    independent groups at once (mutually exclusive with ``bands``). Internal
    degeneracies inside a group are allowed; a single-band group is refused
    when that band touches any other band on the mesh.
    """

    mesh: Tuple[int, int]
    bands: Optional[Sequence[int]] = None
    groups: Optional[Sequence[Sequence[int]]] = None
    embedding: Optional[LatticeEmbedding] = None
    gap_tol: float = 1e-8  # eV; resolved isolation of the band group
    link_tol: float = 1e-8  # minimum |det S| on plaquette links
    stationarity_rtol: float = 1e-6

    def __post_init__(self):
        nu, nv = int(self.mesh[0]), int(self.mesh[1])
        if nu < 2 or nv < 2:
            raise ValueError("mesh must be at least 2x2 for plaquette fluxes.")
        if self.groups is not None:
            if self.bands is not None:
                raise ValueError("bands and groups are mutually exclusive.")
            groups = [[int(b) for b in g] for g in self.groups]
            if not groups:
                raise ValueError("groups must contain at least one band group.")
            for g in groups:
                if not g or len(set(g)) != len(g) or any(b < 0 for b in g):
                    raise ValueError(
                        "each group must be nonempty distinct nonnegative integers."
                    )
            flat = [b for g in groups for b in g]
            if len(set(flat)) != len(flat):
                raise ValueError("groups must not share band indices.")
            object.__setattr__(self, "groups", groups)
        elif self.bands is not None:
            bands = [int(b) for b in self.bands]
            if not bands:
                raise ValueError("bands must contain at least one band index.")
            if len(set(bands)) != len(bands) or any(b < 0 for b in bands):
                raise ValueError("bands must be distinct nonnegative integers.")
            object.__setattr__(self, "bands", bands)


@dataclass
class BulkTopologyResult:
    """Bulk topology result: data plus validity diagnostics."""

    request: BulkRequest
    model: TopologyModel
    embedding: LatticeEmbedding
    kpoints: np.ndarray  # (nu, nv, 3) input-basis fractional momenta
    energies: Optional[np.ndarray]  # (nu, nv, n) physical-mode energies in eV
    band_indices: List[int]
    plaquette_flux: Optional[np.ndarray]  # (nu, nv) total flux of the band set
    vertex_curvature: Optional[np.ndarray]  # (nu, nv) gauge-safe vertex sum
    chern: Optional[int]
    valid: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    groups: Optional[List[Dict[str, Any]]] = None  # per-group records

    def to_json_dict(self) -> Dict[str, Any]:
        """Versioned JSON-serializable record (ADR-005)."""

        def _arr(a: Optional[np.ndarray]) -> Optional[Any]:
            return None if a is None else np.asarray(a).tolist()

        def _group(g: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "bands": g["bands"],
                "chern": g["chern"],
                "valid": bool(g["valid"]),
                "plaquette_flux": _arr(g["plaquette_flux"]),
                "vertex_curvature": _arr(g["vertex_curvature"]),
                "min_external_gap_eV": g["min_external_gap"],
                "min_det_overlap": g["min_det_overlap"],
                "failures": g["failures"],
            }

        return {
            "kind": "bulk",
            "schema_version": SCHEMA_VERSION,
            "model_digest": self.model.digest,
            "canonicalization_version": self.model.canonicalization_version,
            "provenance": self.model.provenance(),
            "model_snapshot": {
                "cell_angstrom": self.model.cell.tolist(),
                "positions_angstrom": self.model.positions.tolist(),
                "positions_fractional": self.model.positions_fractional.tolist(),
                "index_spin": self.model.index_spin.tolist(),
                "spin_atoms": self.model.spin_atoms.tolist(),
                "reference_magmoms_muB": self.model.magmoms.tolist(),
                "Rlist": self.model.Rlist.tolist(),
                "JR_eV": self.model.JR.tolist(),
            },
            "embedding": {
                "basis": self.embedding.basis.tolist(),
                "normal_k": float(self.embedding.normal_k),
                "reciprocal_dual": self.embedding.reciprocal_dual.tolist(),
                "orientation_sign": self.embedding.orientation_sign,
            },
            "mesh": [int(self.request.mesh[0]), int(self.request.mesh[1])],
            "band_indices": [int(b) for b in self.band_indices],
            "kpoints_input_fractional": _arr(self.kpoints),
            "energies_eV": _arr(self.energies),
            "plaquette_flux": _arr(self.plaquette_flux),
            "vertex_curvature": _arr(self.vertex_curvature),
            "chern": None if self.chern is None else int(self.chern),
            "valid": bool(self.valid),
            "groups": None if self.groups is None else [_group(g) for g in self.groups],
            "diagnostics": self.diagnostics,
            "conventions": {
                "fourier_phase": "J(k) = sum_R J(R) exp(-2 pi i k.R)",
                "plaquette_order": "(i,j)->(i+1,j)->(i+1,j+1)->(i,j+1)",
                "energy_unit": "eV",
                "flux_chart": "FHS link flux per endpoint-exclusive plaquette (dimensionless, rad units in sum)",
                "group_curvature": "vertex value = average of the four adjacent plaquette fluxes",
                "certificate": "docs/sympy/magnon_topology_lswt_ferromagnet.py",
            },
        }


def calculate_bulk(model: TopologyModel, request: BulkRequest) -> BulkTopologyResult:
    """Run the bulk topology calculation.

    The calculation refuses to emit a validated invariant when the reference
    is nonstationary, the spectrum is unstable or contains a zero-mode
    touching the requested bands, the band group is not isolated on the
    mesh, a plaquette link determinant vanishes, or a mesh point lacks
    validated mode vectors/positive-index maps. In those cases a result
    with ``valid=False``, no Chern value and explicit reasons is returned;
    per-group flux, curvature and Chern fields are ``None`` whenever they
    were not computed.
    """
    if not isinstance(model, TopologyModel):
        raise TypeError(
            "calculate_bulk requires a topology-ready TopologyModel from "
            "TB2J.magnon.topology.prepare_topology_model; a bare Magnon is not "
            "assumed topology-ready. See prepare_topology_model()."
        )
    embedding = request.embedding or identity_embedding()
    nu, nv = int(request.mesh[0]), int(request.mesh[1])

    failures: List[str] = []
    diagnostics: Dict[str, Any] = {}

    # --- reference state: ticket 02 supports the zero-Q collinear reference.
    magnon = model.magnon()
    if not np.all(np.asarray(magnon.Q) == 0):
        failures.append(
            "nonzero ordering vector Q requires the explicit commensurate "
            "magnetic-cell preparation (ticket 05); not supported by this path."
        )

    # --- assembly-owned stationarity on the actual retained bonds
    from TB2J.magnon.topology.assembly import classical_transverse_gradient

    stat = classical_transverse_gradient(
        model.JR,
        model.Rlist,
        model.magmoms,
        Q=magnon.Q,
        n=magnon._n,
        rtol=request.stationarity_rtol,
    )
    diagnostics["stationarity"] = stat.as_dict()
    if not stat.stationary:
        failures.append(
            f"nonstationary reference: max transverse classical gradient "
            f"{stat.max_transverse:.3e} exceeds tolerance {stat.tolerance:.3e}."
        )

    # --- mesh and Hamiltonian assembly
    i_grid = np.arange(nu)
    j_grid = np.arange(nv)
    k_embed = np.stack(
        np.meshgrid(i_grid / nu, j_grid / nv, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    kpoints = embedding.to_input_k(k_embed).reshape(nu, nv, 3)
    modes: List[BosonicModes] = []
    n_mesh = nu * nv
    # Noncollinear references use certificate-verified local-frame LSWT;
    # collinear models retain the legacy Hq assembly unchanged.
    directions = model.magmoms / np.linalg.norm(model.magmoms, axis=1)[:, None]
    collinear = np.allclose(
        np.abs(directions @ directions.T), 1.0, rtol=0.0, atol=1e-10
    )
    noncollinear = not collinear
    if noncollinear:
        from TB2J.magnon.magnon_math import tb2j_to_physical
        from TB2J.magnon.topology.assembly import local_frame_bosonic_hamiltonian

        # the model keeps the raw TB2J tensors; the local-frame assembly
        # needs the physical (spin-normalized) exchange, exactly like the
        # legacy Magnon path.
        H_all = local_frame_bosonic_hamiltonian(
            tb2j_to_physical(model.JR, model.magmoms),
            model.Rlist,
            model.magmoms,
            kpoints.reshape(n_mesh, 3),
        )
    else:
        H_all = magnon.Hq(kpoints.reshape(n_mesh, 3))
    energies = np.empty((n_mesh, 0))
    n_pos = None
    for idx in range(n_mesh):
        m = solve_bosonic_modes(H_all[idx])
        modes.append(m)
        if m.classification == "energetically_unstable":
            failures.append(
                f"energetically unstable spectrum at mesh point {idx}: Hessian "
                f"minimum {m.h_min_eigenvalue:.3e} < 0; no shift is applied."
            )
        elif m.classification == "dynamically_unstable":
            failures.append(
                f"dynamically unstable spectrum at mesh point {idx}: the solver "
                "emitted no validated bosonic mode set; band-group fluxes are "
                "not defined."
            )
    n_pos = len(modes[0].energies)
    lengths = {len(m.energies) for m in modes}
    energies = None
    if len(lengths) == 1:
        energies = np.array([m.energies for m in modes]).reshape(nu, nv, n_pos)
        if not np.isfinite(energies).all():
            energies = None
            failures.append(
                "non-finite physical-mode energies on the mesh; no validated "
                "invariant can be emitted from non-finite data."
            )
    else:
        failures.append("inconsistent physical-mode count across the mesh.")
    missing_mode_points = [
        idx
        for idx, m in enumerate(modes)
        if m.vectors is None
        or m.positive_indices is None
        or len(m.positive_indices) == 0
    ]
    if missing_mode_points:
        diagnostics["missing_mode_points"] = missing_mode_points
        show = missing_mode_points[:8]
        more = "" if len(missing_mode_points) <= 8 else " ..."
        failures.append(
            "no validated mode vectors or positive-index map at "
            f"{len(missing_mode_points)} of {n_mesh} mesh points "
            f"(first {show}{more}, classifications "
            f"{sorted({modes[i].classification for i in missing_mode_points})}); "
            "band-group fluxes are undefined without a normalized projector, "
            "so no invariant is emitted."
        )
    diagnostics["solver"] = {
        "max_generalized_residual": max(
            m.residuals.get("max_generalized_residual", 0.0) for m in modes
        ),
        "max_pairing_residual": max(
            m.residuals.get("pairing_residual", 0.0) for m in modes
        ),
        "min_physical_energy": min(m.min_energy for m in modes),
        "min_hessian_eigenvalue": min(m.h_min_eigenvalue for m in modes),
        "n_mesh_points_with_zero_modes": sum(
            1 for m in modes if m.residuals.get("n_zero_modes", 0) > 0
        ),
        "classifications": {
            c: sum(1 for m in modes if m.classification == c)
            for c in {m.classification for m in modes}
        },
    }
    freq_chars = {
        m.residuals.get("frequency_character")
        for m in modes
        if m.classification == "energetically_unstable"
    }
    if freq_chars - {None}:
        diagnostics["solver"]["unstable_frequency_character"] = sorted(
            freq_chars - {None}
        )
    zero_points = [
        idx for idx, m in enumerate(modes) if m.classification == "zero_mode"
    ]
    if zero_points:
        diagnostics["zero_mode_points"] = zero_points

    # --- group requests
    requested_groups: List[List[int]]
    if request.groups is not None:
        requested_groups = [list(g) for g in request.groups]
    elif request.bands is not None:
        requested_groups = [list(request.bands)]
    elif modes[0].positive_indices is not None and len(modes[0].positive_indices) > 0:
        requested_groups = [[int(i) for i in modes[0].positive_indices]]
    else:
        requested_groups = []
        failures.append(
            "cannot resolve the default all-positive band group: the first "
            f"mesh point has no positive-index map (classification "
            f"{modes[0].classification}); request explicit bands instead."
        )
    for g in requested_groups:
        if g and n_pos > 0 and max(g) >= n_pos:
            raise ValueError(
                f"band index {max(g)} out of range for {n_pos} physical modes."
            )

    def _group_invariant(bands: List[int]) -> Tuple[List[Any], Dict[str, Any]]:
        """FHS invariant of one band group: (failures, record)."""
        gfail: List[str] = []
        # a requested band touching the stable zero level on the mesh is
        # unvalidated: a zero-energy mode is non-normalizable
        if energies is not None and energies.shape[-1] > 0:
            touching = [b for b in bands if float(energies[:, :, b].min()) <= 1e-8]
            if touching:
                gfail.append(
                    f"requested band group touches stable zero modes on the "
                    f"mesh (band indices {sorted(set(touching))}); a validated "
                    "invariant needs a certified finite smooth projector, "
                    "which a zero-mode-touching group does not have."
                )
        # isolation: sorted-energy boundary gap (internal degeneracies are
        # allowed for multi-band groups; a single band has only external
        # boundaries, so an internal touch is refused through min_gap)
        band_set = set(bands)
        min_gap = np.inf
        gap_location = None
        if energies is not None:
            for idx in range(nu * nv):
                w = np.sort(energies.reshape(nu * nv, -1)[idx])
                for b in range(len(w) - 1):
                    if (b in band_set) != ((b + 1) in band_set):
                        gap = float(w[b + 1] - w[b])
                        if gap < min_gap:
                            min_gap = gap
                            gap_location = (idx, b, b + 1)
            if min_gap <= request.gap_tol:
                gfail.append(
                    f"band group is not isolated: minimum external gap "
                    f"{min_gap:.3e} at mesh point {gap_location} within "
                    f"tolerance {request.gap_tol:.3e}."
                )
        flux = np.zeros((nu, nv))
        chern = None
        vertex = None
        min_link = np.inf
        if not gfail and not failures:
            # map requested band indices to the normalized positive-mode
            # columns per mesh point (zero-mode positions may vary); a band
            # that is not a positive mode at any point cannot be projected.
            columns: List[List[int]] = []
            for idx in range(nu * nv):
                m = modes[idx]
                if (
                    m.vectors is None
                    or m.positive_indices is None
                    or len(m.positive_indices) == 0
                ):
                    gfail.append(
                        f"mesh point {idx} has no validated mode vectors or "
                        f"positive-index map (classification "
                        f"{m.classification}); the requested group has no "
                        "computable projector."
                    )
                    break
                band_to_column = {
                    int(b): int(c) for c, b in enumerate(m.positive_indices)
                }
                missing = [b for b in bands if b not in band_to_column]
                if missing:
                    gfail.append(
                        f"requested bands {missing} are not positive modes at "
                        f"mesh point {idx}; the group cannot be projected on "
                        "this mesh."
                    )
                    break
                columns.append([band_to_column[b] for b in bands])
        if not gfail and not failures:
            psi = np.empty(
                (nu * nv, modes[0].vectors.shape[0], len(bands)), dtype=complex
            )
            for idx in range(nu * nv):
                psi[idx] = modes[idx].vectors[:, columns[idx]]

            # FHS links: the determinants of BOTH mesh directions are
            # collected and tested BEFORE any division, because a vanishing
            # determinant would otherwise produce NaN phases and a
            # meaningless Chern number instead of a refusal.
            det_edge = np.empty((nu, nv), dtype=complex)  # (i,j) -> (i+1,j)
            det_width = np.empty((nu, nv), dtype=complex)  # (i,j) -> (i,j+1)
            for i in range(nu):
                for j in range(nv):
                    p = i * nv + j
                    det_edge[i, j] = np.linalg.det(
                        psi[p].conj().T @ psi[((i + 1) % nu) * nv + j]
                    )
                    det_width[i, j] = np.linalg.det(
                        psi[p].conj().T @ psi[i * nv + (j + 1) % nv]
                    )
            mags = np.abs(np.concatenate([det_edge.ravel(), det_width.ravel()]))
            min_link = float(mags.min())
            if not np.isfinite(min_link):
                gfail.append(
                    "non-finite link determinant overlap: the group overlap "
                    "matrices carry NaN or infinite values on this mesh."
                )
            elif min_link <= max(request.link_tol, 0.0):
                gfail.append(
                    f"vanishing link determinant overlap: min |det S| = "
                    f"{min_link:.3e} within tolerance {request.link_tol:.3e}; "
                    "the band group overlaps other states on this mesh."
                )
            else:
                link_edge = det_edge / np.abs(det_edge)
                link_width = det_width / np.abs(det_width)
                prod = np.ones((nu, nv), dtype=complex)
                for i in range(nu):
                    for j in range(nv):
                        # plaquette order (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1);
                        # the reversed links are the complex conjugates of the
                        # forward ones (S_qp = S_pq^dag).
                        prod[i, j] = (
                            link_edge[i, j]
                            * link_width[(i + 1) % nu, j]
                            * np.conj(link_edge[i, (j + 1) % nv])
                            * np.conj(link_width[i, j])
                        )
                flux = np.angle(prod) / (2 * np.pi)
                chern = int(np.rint(flux.sum()))
                # gauge-safe vertex curvature: average of adjacent plaquette fluxes
                vertex = (
                    np.roll(flux, 1, axis=0)
                    + np.roll(flux, 1, axis=1)
                    + np.roll(np.roll(flux, 1, axis=0), 1, axis=1)
                    + flux
                ) / 4.0
        group_valid = not gfail and not failures
        record = {
            "bands": [int(b) for b in bands],
            "chern": int(chern) if group_valid and chern is not None else None,
            "plaquette_flux": flux if group_valid else None,
            "vertex_curvature": vertex if group_valid else None,
            "valid": group_valid,
            "min_external_gap": None if not np.isfinite(min_gap) else float(min_gap),
            "min_det_overlap": None if not np.isfinite(min_link) else float(min_link),
            "failures": gfail,
        }
        return gfail, record

    group_records = []
    group_failures: List[str] = []
    for g in requested_groups:
        gfail, record = _group_invariant(g)
        group_records.append(record)
        for f in gfail:
            group_failures.append(f"groups {record['bands']}: {f}")
    diagnostics["links"] = {
        "link_tol": float(request.link_tol),
        "min_det_overlap_per_group": [r["min_det_overlap"] for r in group_records],
    }
    diagnostics["gap"] = {
        "gap_tol": float(request.gap_tol),
        "min_external_gap_per_group": [r["min_external_gap"] for r in group_records],
    }
    diagnostics["mesh_refinement"] = {
        "mesh": [nu, nv],
        "note": "an integer on one mesh is not a convergence certificate; "
        "refine the mesh and compare when convergence evidence is required.",
    }
    failures.extend(group_failures)
    valid = not failures
    primary = (
        group_records[0]
        if group_records
        else {
            "bands": [],
            "chern": None,
            "plaquette_flux": None,
            "vertex_curvature": None,
        }
    )
    diagnostics["failures"] = failures
    result = BulkTopologyResult(
        request=request,
        model=model,
        embedding=embedding,
        kpoints=kpoints,
        energies=energies,
        band_indices=primary["bands"],
        plaquette_flux=primary["plaquette_flux"] if valid else None,
        vertex_curvature=primary["vertex_curvature"] if valid else None,
        chern=primary["chern"] if valid else None,
        valid=valid,
        diagnostics=diagnostics,
    )
    result.groups = group_records
    return result


def refine_bulk(
    model: TopologyModel,
    request: BulkRequest,
    factor: int = 2,
) -> Dict[str, Any]:
    """Inspectable mesh-refinement report for a bulk request.

    Runs the request on the requested mesh and on the factor-refined mesh and
    reports band Chern values, minimum external gaps and their agreement.
    Agreement is convergence EVIDENCE, not a certificate of continuum
    isolation: a gap closing between mesh points can hide from any discrete
    sampling.
    """
    if factor < 2:
        raise ValueError("refinement factor must be >= 2.")
    fine = BulkRequest(
        mesh=(request.mesh[0] * factor, request.mesh[1] * factor),
        bands=request.bands,
        groups=request.groups,
        embedding=request.embedding,
        gap_tol=request.gap_tol,
        link_tol=request.link_tol,
        stationarity_rtol=request.stationarity_rtol,
    )
    coarse = calculate_bulk(model, request)
    fine_result = calculate_bulk(model, fine)

    def _summary(result: BulkTopologyResult) -> Dict[str, Any]:
        return {
            "mesh": [int(result.request.mesh[0]), int(result.request.mesh[1])],
            "valid": bool(result.valid),
            "chern_per_group": [g["chern"] for g in (result.groups or [])],
            "min_external_gap_per_group": [
                g["min_external_gap"] for g in (result.groups or [])
            ],
        }

    coarse_summary = _summary(coarse)
    fine_summary = _summary(fine_result)
    chern_agrees = (
        coarse_summary["chern_per_group"]
        and coarse_summary["chern_per_group"] == fine_summary["chern_per_group"]
        and all(c is not None for c in fine_summary["chern_per_group"])
    )
    return {
        "factor": int(factor),
        "coarse": coarse_summary,
        "fine": fine_summary,
        "chern_agrees": bool(chern_agrees),
        "note": "integer agreement plus gap growth is convergence evidence, "
        "not proof: a continuum gap closing between mesh points escapes any "
        "discrete sampling.",
    }
