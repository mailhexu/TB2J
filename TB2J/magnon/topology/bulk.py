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
]

SCHEMA_VERSION = "tb2j-topology.bulk/1"


@dataclass(frozen=True)
class BulkRequest:
    """Bulk topology request over a topology-ready model.

    ``mesh`` is ``(nu, nv)``: the endpoint-exclusive periodic sampling of the
    embedding-basis reciprocal torus. ``bands`` are explicit zero-based
    indices into the ascending positive-mode energies; ``None`` selects all
    positive bands (only sensible when they form one isolated group).
    """

    mesh: Tuple[int, int]
    bands: Optional[Sequence[int]] = None
    embedding: Optional[LatticeEmbedding] = None
    gap_tol: float = 1e-8  # eV; resolved isolation of the band group
    link_tol: float = 1e-8  # minimum |det S| on plaquette links
    stationarity_rtol: float = 1e-6

    def __post_init__(self):
        nu, nv = int(self.mesh[0]), int(self.mesh[1])
        if nu < 2 or nv < 2:
            raise ValueError("mesh must be at least 2x2 for plaquette fluxes.")
        if self.bands is not None:
            bands = [int(b) for b in self.bands]
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
    energies: np.ndarray  # (nu, nv, n_pos) positive-mode energies in eV
    band_indices: List[int]
    plaquette_flux: Optional[np.ndarray]  # (nu, nv) total flux of the band set
    vertex_curvature: Optional[np.ndarray]  # (nu, nv) gauge-safe vertex sum
    chern: Optional[int]
    valid: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        """Versioned JSON-serializable record (ADR-005)."""

        def _arr(a: Optional[np.ndarray]) -> Optional[Any]:
            return None if a is None else np.asarray(a).tolist()

        return {
            "kind": "bulk",
            "schema_version": SCHEMA_VERSION,
            "model_digest": self.model.digest,
            "canonicalization_version": self.model.canonicalization_version,
            "provenance": self.model.provenance(),
            "embedding": {
                "basis": self.embedding.basis.tolist(),
                "normal_k": float(self.embedding.normal_k),
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
            "diagnostics": self.diagnostics,
            "conventions": {
                "fourier_phase": "J(k) = sum_R J(R) exp(-2 pi i k.R)",
                "plaquette_order": "(i,j)->(i+1,j)->(i+1,j+1)->(i,j+1)",
                "energy_unit": "eV",
                "flux_chart": "FHS link flux per endpoint-exclusive plaquette (dimensionless, rad units in sum)",
                "certificate": "docs/sympy/magnon_topology_lswt_ferromagnet.py",
            },
        }


def calculate_bulk(model: TopologyModel, request: BulkRequest) -> BulkTopologyResult:
    """Run the bulk topology calculation.

    The calculation refuses to emit a validated invariant when the reference
    is nonstationary, the spectrum is unstable or contains a zero-mode
    touching the requested bands, the band group is not isolated on the
    mesh, or a plaquette link determinant vanishes. In those cases a result
    with ``valid=False``, no Chern value and explicit reasons is returned.
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
    n_pos = len(modes[0].energies)
    lengths = {len(m.energies) for m in modes}
    energies = None
    if len(lengths) == 1:
        energies = np.array([m.energies for m in modes]).reshape(nu, nv, n_pos)
    else:
        failures.append("inconsistent physical-mode count across the mesh.")
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

    # --- band selection
    bands = (
        list(request.bands)
        if request.bands is not None
        else [int(i) for i in modes[0].positive_indices]
    )
    if bands and n_pos > 0 and max(bands) >= n_pos:
        raise ValueError(
            f"band index {max(bands)} out of range for {n_pos} physical modes."
        )
    # a requested band touching the stable zero level on the mesh is
    # unvalidated: a zero-energy mode is non-normalizable
    if energies is not None and energies.shape[-1] > 0:
        touching = []
        for b in bands:
            if float(energies[:, :, b].min()) <= 1e-8:
                touching.append(b)
        if touching:
            failures.append(
                f"requested band group touches stable zero modes on the mesh "
                f"(band indices {sorted(set(touching))}); a validated invariant "
                "needs a certified finite smooth projector, which a "
                "zero-mode-touching group does not have."
            )

    # --- isolation diagnostics: sorted-energy boundary gap of the band set
    band_set = set(bands)
    min_gap = np.inf
    gap_location = None
    if energies is not None:
        for idx in range(nu * nv):
            w = np.sort(energies.reshape(nu * nv, -1)[idx])
            for b in range(len(w) - 1):
                inside_b = b in band_set
                inside_b1 = (b + 1) in band_set
                if inside_b != inside_b1:
                    gap = float(w[b + 1] - w[b])
                    if gap < min_gap:
                        min_gap = gap
                        gap_location = (idx, b, b + 1)
        if min_gap <= request.gap_tol:
            failures.append(
                f"band group is not isolated: minimum external gap {min_gap:.3e} "
                f"at mesh point {gap_location} within tolerance {request.gap_tol:.3e}."
            )
    diagnostics["gap"] = {
        "min_gap": None if not np.isfinite(min_gap) else float(min_gap),
        "gap_location": gap_location,
        "gap_tol": float(request.gap_tol),
    }

    # --- gauge-invariant FHS links over the selected band subspace
    flux = np.zeros((nu, nv))
    chern = None
    vertex = None
    min_link = np.inf
    if not failures:
        # map requested band indices to the normalized positive-mode columns
        # per mesh point (zero-mode positions may vary across the mesh)
        psi = np.empty((nu * nv, modes[0].vectors.shape[0], len(bands)), dtype=complex)
        for idx in range(nu * nv):
            cols = modes[idx].positive_indices
            psi[idx] = modes[idx].vectors[
                :, [int(np.flatnonzero(cols == b)[0]) for b in bands]
            ]

        def link(p, q) -> complex:
            # normalized determinant of the between-points Gram matrix
            d = np.linalg.det(psi[p].conj().T @ psi[q])
            return d / abs(d)

        min_link = min(
            abs(
                np.linalg.det(psi[(i * nv + j)].conj().T @ psi[((i + 1) % nu) * nv + j])
            )
            for i in range(nu)
            for j in range(nv)
        )

        prod = np.ones((nu, nv), dtype=complex)
        for i in range(nu):
            for j in range(nv):
                p0 = i * nv + j
                p1 = ((i + 1) % nu) * nv + j
                p2 = ((i + 1) % nu) * nv + (j + 1) % nv
                p3 = i * nv + (j + 1) % nv
                prod[i, j] = link(p0, p1) * link(p1, p2) * link(p2, p3) * link(p3, p0)
        flux = np.angle(prod) / (2 * np.pi)
        chern = int(np.rint(flux.sum()))
        # gauge-safe vertex curvature: average of adjacent plaquette fluxes
        vertex = (
            np.roll(flux, 1, axis=0)
            + np.roll(flux, 1, axis=1)
            + np.roll(np.roll(flux, 1, axis=0), 1, axis=1)
            + flux
        ) / 4.0
        if min_link <= request.link_tol:
            failures.append(
                f"vanishing link determinant overlap: min |det S| = {min_link:.3e} "
                f"within tolerance {request.link_tol:.3e}; the band group overlaps "
                "other states on this mesh."
            )
        diagnostics["links"] = {
            "min_det_overlap": min_link,
            "link_tol": float(request.link_tol),
        }

    diagnostics["mesh_refinement"] = {
        "mesh": [nu, nv],
        "note": "an integer on one mesh is not a convergence certificate; "
        "refine the mesh and compare when convergence evidence is required.",
    }
    valid = not failures
    diagnostics["failures"] = failures
    return BulkTopologyResult(
        request=request,
        model=model,
        embedding=embedding,
        kpoints=kpoints,
        energies=energies,
        band_indices=bands,
        plaquette_flux=flux if valid else None,
        vertex_curvature=vertex if valid else None,
        chern=chern if valid else None,
        valid=valid,
        diagnostics=diagnostics,
    )
