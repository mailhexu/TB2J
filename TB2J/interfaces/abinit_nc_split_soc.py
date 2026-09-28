"""ABINIT NC split-SOC leg builder and three-leg rotate/merge driver (story-009).

Consumes two stored artifacts (ADR-1, split-soc-ks spec):

- the ``abinit.nc_pao_hs`` **v2** file written by abinao from the strength-0
  collinear (``nsppol=2``, ``nspinor=1``) WFK: PAO projections
  ``C[s, k, n, p] = <phi_p|psi_{k n s}>`` (loaded as ``<phi|psi>`` by the
  existing loader), the k-dependent PAO overlap ``S(k)``, and the
  collinear spin-splitting operators (``delta_total`` &c.) in eV.  The
  file stores Hartree on disk; ``ProjectorGreenData.load_netcdf`` /
  :func:`load_abinit_nc_pao_savetb2j` normalize to eV in memory.
- the ``abinao.nc_soc_ks`` **v1** sidecar (story-008,
  ``abinao.soc_kernel.write_nc_soc_kernel``): the all-atom band-window SOC
  operator ``W^K_SO(k)`` per x/y/z leg in the composite spinor basis
  ``i = 2*n + sigma`` (eV, Hermitian, full-BZ WFK order), with the
  strength-0 WFK / PAO_HS SHA-256 provenance required for pairing.

The consumer maps the composite ``2*n + sigma`` index onto a spinor
``ProjectorGreenData(nspinor=2)`` window of ``2*b`` states whose state
``2*n + s`` carries ``<phi_p|psi_{n s}>`` on spinor component ``s`` and
zero on the other component (the collinear strength-0 limit), dualizes the
nonorthogonal PAO maps ``B(k) = S(k)^-1 C(k)`` (never dropping ``S`` by
fiat; the split-SOC kernel consumes normalized band data with
``overlap_k=None``), and feeds the story-002 KS split-SOC kernel per leg.
The producer supplies three gauge-matched SOC matrices for x/y/z ABINIT
spinaxis rotations. Each rotated reference measures only its transverse
2x2 exchange block; the rank-nine raw tensor is solved from all three
before decomposition into Jiso/DMI/Jani. The strength-0 collinear anchor
and the z-leg transverse projection are checked independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from TB2J.interfaces.abinit_savetb2j import (
    ABINIT_NC_PAO_HS_SCHEMA_NAME,
    ABINIT_NC_PAO_HS_SCHEMA_VERSION,
    load_abinit_nc_pao_savetb2j,
)
from TB2J.projector_green import site_magnetization_sign
from TB2J.split_soc_kernel import json_safe_provenance

__all__ = [
    "NC_SOC_KS_SCHEMA_NAME",
    "NC_SOC_KS_SCHEMA_VERSION",
    "SOC_OFF_LEG_DIRECTIONS",
    "SocKsSidecar",
    "build_nc_split_soc_leg",
    "dualize_pao_coefficients",
    "gen_exchange_abinit_nc_split_soc",
    "load_soc_ks_sidecar",
    "nc_split_soc_soc_off_anchor",
    "sha256_of_file",
]

NC_SOC_KS_SCHEMA_NAME = "abinao.nc_soc_ks"
NC_SOC_KS_SCHEMA_VERSION = "1"
SOC_OFF_LEG_DIRECTIONS = ("x", "y", "z")

_HERMITIAN_TOL = 1.0e-9
_WEIGHT_SUM_TOL = 1.0e-8
_AXIS_ALIGN_TOL = 1.0e-8
_EIGENVALUE_TOL_EV = 1.0e-6
_DEFAULT_TANGENT_TOL_EV = 1.0e-2
_DEFAULT_ANCHOR_RTOL = 1.0e-7
_DEFAULT_ANCHOR_DMI_TOL_EV = 1.0e-9
_HARTREE_TO_EV = 27.211386245988

_LEG_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def sha256_of_file(path) -> str:
    """SHA-256 hexdigest of the raw bytes of ``path`` (FR-050 pairing key)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# sidecar reader (abinao.nc_soc_ks v1, frozen contract)
# ---------------------------------------------------------------------------


@dataclass
class SocKsSidecar:
    """Parsed ``abinao.nc_soc_ks`` v1 file (all energies already eV)."""

    legs: tuple[str, ...]
    kpoints: np.ndarray
    kweights: np.ndarray
    spinaxis: np.ndarray
    w_so: np.ndarray  # (nleg, nkpt, nband_composite, nband_composite) complex, eV
    w_so_site: np.ndarray | None  # (nleg, nkpt, natom, nb, nb) complex, eV
    eigenvalues_ev: np.ndarray | None  # (nleg, nkpt, nband_composite), eV
    band_window: tuple[int, int] | None
    source_wfk: str
    source_wfk_sha256: str
    pao_hs: str | None
    pao_hs_sha256: str | None
    spnorbscl: float
    all_atoms_covered: bool
    fr050: dict = field(default_factory=dict)
    source_path: str | None = None

    @property
    def nband_composite(self) -> int:
        return self.w_so.shape[-1]

    def leg_index(self, leg: str) -> int:
        return self.legs.index(str(leg).lower())


def _require_attr(nc, name, context):
    if not hasattr(nc, name):
        raise ValueError(
            f"{NC_SOC_KS_SCHEMA_NAME} sidecar missing attr {name!r} ({context})"
        )
    return getattr(nc, name)


def _read_split_complex(nc, real_name, imag_name, context):
    for name in (real_name, imag_name):
        if name not in nc.variables:
            raise ValueError(
                f"{NC_SOC_KS_SCHEMA_NAME} sidecar missing variable {name!r} ({context})"
            )
    return np.asarray(nc.variables[real_name][:], dtype=float) + 1j * np.asarray(
        nc.variables[imag_name][:], dtype=float
    )


def _validate_sidecar_self(sidecar: SocKsSidecar) -> None:
    if tuple(sidecar.legs) != SOC_OFF_LEG_DIRECTIONS:
        raise ValueError(
            f"sidecar legs must be {SOC_OFF_LEG_DIRECTIONS} in order, got {sidecar.legs}"
        )
    if sidecar.kpoints.ndim != 2 or sidecar.kpoints.shape[1] != 3:
        raise ValueError("sidecar kpts must have shape (nkpt, 3)")
    if sidecar.kweights.shape != (sidecar.kpoints.shape[0],):
        raise ValueError("sidecar kweights must have shape (nkpt,)")
    if np.any(sidecar.kweights < 0.0):
        raise ValueError("sidecar kweights must be non-negative (full-BZ gauge)")
    if not np.isclose(sidecar.kweights.sum(), 1.0, atol=_WEIGHT_SUM_TOL):
        raise ValueError(
            "sidecar kweights must sum to 1 (full-BZ gauge); got "
            f"{sidecar.kweights.sum()!r} — refusing an ungauged IBZ/BZ mix"
        )
    if sidecar.spinaxis.shape != (len(sidecar.legs), 3):
        raise ValueError("sidecar spinaxis must have shape (nleg, 3)")
    for ileg, leg in enumerate(sidecar.legs):
        axis = sidecar.spinaxis[ileg]
        norm = float(np.linalg.norm(axis))
        if abs(norm - 1.0) > _AXIS_ALIGN_TOL:
            raise ValueError(
                f"sidecar spinaxis for leg {leg!r} is not a unit vector: {axis}"
            )
        if not np.allclose(axis, _LEG_AXES[leg], atol=_AXIS_ALIGN_TOL):
            raise ValueError(
                f"sidecar spinaxis for leg {leg!r} does not align with the "
                f"+{leg} Cartesian axis: {axis}"
            )
    nband = sidecar.nband_composite
    if nband % 2 != 0 or nband <= 0:
        raise ValueError(
            "sidecar SOC matrices must have even composite order 2*n "
            f"(index 2*n+sigma); got {nband}"
        )
    scale = float(np.abs(sidecar.w_so).max(initial=0.0))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("sidecar SOC matrices must be finite and nonzero")
    if not np.allclose(
        sidecar.w_so,
        np.conj(np.swapaxes(sidecar.w_so, 2, 3)),
        rtol=_HERMITIAN_TOL,
        atol=_HERMITIAN_TOL * scale,
    ):
        raise ValueError("sidecar w_so must be Hermitian at every k point")
    if sidecar.w_so_site is not None:
        if sidecar.w_so_site.shape != sidecar.w_so.shape[:2] + (
            sidecar.w_so_site.shape[2],
            nband,
            nband,
        ):
            raise ValueError("sidecar w_so_site has an inconsistent shape")
    if sidecar.eigenvalues_ev is not None and sidecar.eigenvalues_ev.shape != (
        len(sidecar.legs),
        sidecar.kpoints.shape[0],
        nband,
    ):
        raise ValueError("sidecar eigenvalues_ev must have shape (nleg, nkpt, 2*n)")
    if sidecar.band_window is not None:
        lo, hi = sidecar.band_window
        if lo < 0 or hi - lo != nband:
            raise ValueError(
                "sidecar band_window must be a half-open range of width "
                f"{nband} (the stored composite window); got {(lo, hi)}"
            )


def load_soc_ks_sidecar(filename) -> SocKsSidecar:
    """Read and self-validate an ``abinao.nc_soc_ks`` v1 sidecar file."""
    from netCDF4 import Dataset

    path = Path(filename)
    with Dataset(path) as nc:
        schema_name = str(_require_attr(nc, "schema_name", "root"))
        if schema_name != NC_SOC_KS_SCHEMA_NAME:
            raise ValueError(
                f"unsupported SOC sidecar schema_name {schema_name!r} "
                f"(expected {NC_SOC_KS_SCHEMA_NAME!r})"
            )
        schema_version = str(_require_attr(nc, "schema_version", "root"))
        if schema_version != NC_SOC_KS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported SOC sidecar schema_version {schema_version!r} "
                f"(expected {NC_SOC_KS_SCHEMA_VERSION!r})"
            )
        energy_unit = str(_require_attr(nc, "energy_unit", "root"))
        if energy_unit != "eV":
            raise ValueError(
                f"sidecar energies must be stored in eV; got energy_unit={energy_unit!r}"
            )
        hartree_to_ev = float(_require_attr(nc, "hartree_to_ev", "root"))
        if not np.isclose(hartree_to_ev, _HARTREE_TO_EV, rtol=1e-12):
            raise ValueError(f"sidecar hartree_to_ev mismatch: {hartree_to_ev!r}")

        legs = tuple(str(v) for v in np.asarray(nc.variables["leg"][:]).tolist())
        kpoints = np.asarray(nc.variables["kpts"][:], dtype=float)
        kweights = np.asarray(nc.variables["kweights"][:], dtype=float)
        spinaxis = np.asarray(nc.variables["spinaxis"][:], dtype=float)
        w_so = _read_split_complex(nc, "w_so_real", "w_so_imag", "w_so")

        w_so_site = None
        if "w_so_site_real" in nc.variables and "w_so_site_imag" in nc.variables:
            w_so_site = _read_split_complex(
                nc, "w_so_site_real", "w_so_site_imag", "w_so_site"
            )
        eigenvalues_ev = None
        if "eigenvalues_ev" in nc.variables:
            eigenvalues_ev = np.asarray(nc.variables["eigenvalues_ev"][:], dtype=float)

        lo = int(_require_attr(nc, "band_window_lo", "root"))
        hi = int(_require_attr(nc, "band_window_hi", "root"))
        if lo == -1 and hi == -1:
            band_window = None
        else:
            band_window = (lo, hi)
        spnorbscl = float(_require_attr(nc, "spnorbscl", "root"))
        all_atoms_covered = bool(int(_require_attr(nc, "all_atoms_covered", "root")))
        fr050 = json.loads(str(_require_attr(nc, "fr050_metadata", "root")))

        sidecar = SocKsSidecar(
            legs=legs,
            kpoints=kpoints,
            kweights=kweights,
            spinaxis=spinaxis,
            w_so=w_so,
            w_so_site=w_so_site,
            eigenvalues_ev=eigenvalues_ev,
            band_window=band_window,
            source_wfk=str(_require_attr(nc, "source_wfk", "root")),
            source_wfk_sha256=str(_require_attr(nc, "source_wfk_sha256", "root")),
            pao_hs=(str(getattr(nc, "pao_hs")) if hasattr(nc, "pao_hs") else None),
            pao_hs_sha256=(
                str(getattr(nc, "pao_hs_sha256"))
                if hasattr(nc, "pao_hs_sha256")
                else None
            ),
            spnorbscl=spnorbscl,
            all_atoms_covered=all_atoms_covered,
            fr050=fr050,
            source_path=str(path),
        )
    if spnorbscl != 1.0:
        raise ValueError(
            "refusing a non-unit SOC scaling: the sidecar stores the lambda=1 "
            f"kernel (spnorbscl must be 1.0, got {spnorbscl!r}); rescale at "
            "the kernel level with lam, never in the stored operator"
        )
    if not all_atoms_covered:
        raise ValueError(
            "sidecar must cover all atoms (ligand SOC enters the propagator); "
            "all_atoms_covered != 1"
        )
    _validate_sidecar_self(sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# PAO_HS pairing validation
# ---------------------------------------------------------------------------


def validate_sidecar_pairing(sidecar, data, pao_hs_path, wfk_path=None) -> dict:
    """Gate the sidecar against the loaded PAO_HS data (and optional WFK).

    Refuses: hash mismatches (PAO_HS always; strength-0 WFK when a path is
    given), k-point order/gauge mismatches, band-energy mismatches,
    incomplete leg axes, and composite-band-count mismatches.  Returns the
    pairing block merged into the FR-050 provenance.
    """
    if data.metadata.get("kpoint_set") != "full_bz":
        raise ValueError(
            "the split-SOC consumer requires full-BZ PAO_HS coefficients; got "
            f"kpoint_set={data.metadata.get('kpoint_set')!r} — refusing an "
            "ungauged IBZ/BZ mix"
        )
    strength0 = sidecar.fr050.get("strength0_provenance") or {}
    if (int(strength0.get("nspinor", 1)), int(strength0.get("nsppol", 2))) == (2, 1):
        raise ValueError(
            "this consumer pairs a COLLINEAR strength-0 reference (nsppol=2, "
            "nspinor=1, composite 2*n+sigma) with a collinear PAO_HS file; "
            "got a spinor-flavor sidecar (nsppol=1, nspinor=2, contracted "
            "spinor bands).  Its W is not elementwise comparable to a "
            "collinear PAO_HS-side matrix — route it through a spinor-flavor "
            "consumer instead"
        )
    if sidecar.nband_composite != 2 * data.nband:
        raise ValueError(
            "sidecar composite band count "
            f"{sidecar.nband_composite} != 2 * PAO_HS nband {2 * data.nband}"
        )
    pao_path = Path(pao_hs_path)
    pao_sha = sha256_of_file(pao_path)
    pairing = {
        "pao_hs": {"path": str(pao_path), "sha256": pao_sha},
        "wfk": {"name": sidecar.source_wfk, "sha256": sidecar.source_wfk_sha256},
    }
    if sidecar.pao_hs_sha256 is not None and pao_sha != sidecar.pao_hs_sha256:
        raise ValueError(
            "PAO_HS SHA-256 mismatch: sidecar was written from "
            f"{sidecar.pao_hs!r} ({sidecar.pao_hs_sha256}) but got {pao_path} "
            f"({pao_sha})"
        )
    if wfk_path is not None:
        wfk_path = Path(wfk_path)
        wfk_sha = sha256_of_file(wfk_path)
        pairing["wfk"]["checked_path"] = str(wfk_path)
        pairing["wfk"]["checked_sha256"] = wfk_sha
        if wfk_sha != sidecar.source_wfk_sha256:
            raise ValueError(
                "strength-0 WFK SHA-256 mismatch: sidecar was written from "
                f"{sidecar.source_wfk} ({sidecar.source_wfk_sha256}) but got "
                f"{wfk_path} ({wfk_sha})"
            )
    if data.kpoints.shape != sidecar.kpoints.shape or not np.allclose(
        data.kpoints, sidecar.kpoints, atol=1e-8
    ):
        raise ValueError(
            "sidecar k-points do not match the PAO_HS full-BZ k-point order "
            "(same order, same gauge required)"
        )
    if not np.allclose(data.weights, sidecar.kweights, atol=_WEIGHT_SUM_TOL):
        raise ValueError(
            "sidecar k-weights do not match the PAO_HS weights (the two "
            "artifacts must share one BZ gauge)"
        )
    if sidecar.eigenvalues_ev is not None:
        stacked = np.empty((data.nkpt, 2 * data.nband), dtype=float)
        for spin in range(2):
            stacked[:, spin::2] = data.eigenvalues[spin]
        deviation = float(np.abs(sidecar.eigenvalues_ev[0] - stacked).max())
        if deviation > _EIGENVALUE_TOL_EV:
            raise ValueError(
                "sidecar eigenvalues_ev disagree with the PAO_HS band "
                f"energies by {deviation:.3e} eV (tol {_EIGENVALUE_TOL_EV:g}); "
                "the two artifacts must come from one strength-0 WFK"
            )
        pairing["eigenvalues_max_dev_eV"] = deviation
    if sidecar.w_so_site is not None and sidecar.w_so_site.shape[2] != len(
        data.site_nproj
    ):
        raise ValueError(
            "sidecar site-resolved blocks do not cover the PAO_HS atoms: "
            f"{sidecar.w_so_site.shape[2]} vs {len(data.site_nproj)}"
        )
    return pairing


# ---------------------------------------------------------------------------
# nonorthogonal PAO dualization and leg construction
# ---------------------------------------------------------------------------


def dualize_pao_coefficients(data):
    """Return a copy of collinear PAO data with dualized maps ``B = S^-1 C``.

    The split-SOC kernel consumes normalized band data (``overlap_k=None``);
    for the nonorthogonal PAO basis the overlap must be *used*, never
    dropped: the dual projectors ``B(k) = S(k)^{-1} C(k)`` reproduce exactly
    the contravariant Green function of the anchor replay's ``inverse``
    overlap mode.  A condition-number gate mirrors ``ProjectorGreen``.
    """
    import dataclasses

    if data.overlap_k is None:
        raise ValueError(
            "dualize_pao_coefficients requires k-dependent PAO overlap_k; "
            "nothing to dualize for orthogonal data"
        )
    threshold = float(data.metadata.get("overlap_condition_threshold", 1.0e12))
    coefficients = np.asarray(data.coefficients, dtype=complex)
    dual = np.empty_like(coefficients)
    for ik, sk in enumerate(data.overlap_k):
        condition = np.linalg.cond(sk)
        if not np.isfinite(condition) or condition > threshold:
            raise ValueError(
                "PAO overlap_k is singular or ill-conditioned at k-point "
                f"{ik}: condition={condition:.3e}, threshold={threshold:.3e}"
            )
        # B[s, n, q] = sum_p S^-1[q, p] C[s, n, p]  (<~chi_q|psi>)
        dual[:, ik] = np.einsum(
            "qp,sbp->sbq", np.linalg.inv(sk), coefficients[:, ik], optimize=True
        )
    metadata = {
        **data.metadata,
        "pao_dualization": {
            "mode": "inverse",
            "definition": "B(k) = S(k)^-1 C(k) with C = <phi|psi>; "
            "overlap_k dropped only after the dual is taken",
            "overlap_condition_threshold": threshold,
        },
    }
    return dataclasses.replace(
        data, coefficients=dual, overlap_k=None, metadata=metadata
    )


def build_nc_split_soc_reference(
    sidecar,
    data,
    sites_magnetic=None,
    vertex_component="delta_total",
):
    """Build the normalized no-SOC spinor REFERENCE (the z reference).

    Composite state ``2*n + s`` carries the dualized PAO map of collinear
    channel ``s`` on spinor component ``s`` and zero elsewhere; magnetic
    rotation vertices are the collinear spin-splitting component tensored
    with ``sigma_z`` (magnetic sites only, never the SOC operator).
    """
    import dataclasses

    from TB2J.interfaces.abinit_paw_split_soc import (
        _SIGMA_Z,
        _stacked_spinor_coefficients,
    )
    from TB2J.projector_green import SPINOR_OPERATOR_DEFINITION

    if data.nspinor != 1 or data.coefficients.ndim != 4:
        raise ValueError("build_nc_split_soc_reference expects collinear PAO_HS data")
    if sidecar.nband_composite != 2 * data.nband:
        raise ValueError(
            f"sidecar composite bands {sidecar.nband_composite} != 2 * nband "
            f"{2 * data.nband}"
        )
    if vertex_component not in {
        "delta_total",
        "delta_xc_smooth",
        "spectral_spin_split",
    }:
        raise ValueError(
            "magnetic vertices must be sourced from a collinear "
            f"delta_total/delta_xc_smooth/spectral_spin_split component, got {vertex_component!r}"
        )
    if not data.has_operator_component(vertex_component):
        raise ValueError(f"missing magnetic-vertex component: {vertex_component}")

    if sites_magnetic is None:
        sites_magnetic = list(range(len(data.site_nproj)))
    sites_magnetic = [int(s) for s in sites_magnetic]

    coefficients = _stacked_spinor_coefficients(data.coefficients, data.nband)
    eigenvalues = np.empty((1, data.nkpt, 2 * data.nband), dtype=float)
    for spin in range(2):
        eigenvalues[0, :, spin::2] = data.eigenvalues[spin]
    occupations = None
    if data.occupations is not None:
        occupations = np.empty_like(eigenvalues)
        for spin in range(2):
            occupations[0, :, spin::2] = data.occupations[spin]

    nsite = len(data.site_nproj)
    nmax = data.site_projector_indices.shape[1]
    spinor_operator = np.zeros((nsite, nmax, nmax, 2, 2), dtype=complex)
    for site in sites_magnetic:
        delta = data.get_operator_component(vertex_component, site=site)
        nproj_site = int(data.site_nproj[site])
        # site blocks are (nmax, nmax) LOCAL padded blocks (same convention
        # as pack_site_hij / get_site_block_spinor)
        spinor_operator[site, :nproj_site, :nproj_site] = np.einsum(
            "pq,st->pqst", delta, _SIGMA_Z
        )

    metadata = {
        **data.metadata,
        "split_soc_leg": {
            "backend": "abinit_nc",
            "leg": "z",
            "sidecar": sidecar.source_path,
            "coefficient_convention": "abinao PAO <phi|psi> dualized B = S^-1 C "
            "(ket-side spectral replay)",
            "vertex_component": vertex_component,
            "sites_magnetic": sites_magnetic,
        },
    }
    return dataclasses.replace(
        data,
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        occupations=occupations,
        site_nproj=data.site_nproj,
        site_projector_indices=data.site_projector_indices,
        spinor_operator=spinor_operator,
        spinor_operator_definition=SPINOR_OPERATOR_DEFINITION,
        operator_components=data.operator_components,
        operator_component_metadata=data.operator_component_metadata,
        channel_interpretation="interleaved collinear channels (up, down)",
        metadata=metadata,
        nspinor=2,
    )


def _abinit_spinaxis_rotation(axis):
    """ABINIT geteuler z-y spin gauge used by abinao.spin_matrices.

    Band-space W_SO is gauge-dependent: the producer contracts U^dag S U in
    the unrotated collinear WFK basis. The leg must rotate its states by this
    same U; choosing any other SU(2) representative changes W_SO's phases.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    alpha = float(np.arctan2(axis[1], axis[0])) if np.hypot(*axis[:2]) > 1e-8 else 0.0
    beta = float(np.arctan2(np.hypot(*axis[:2]), axis[2]))
    cb, sb = np.cos(beta / 2), np.sin(beta / 2)
    em = np.exp(-0.5j * alpha)
    ep = em.conjugate()
    return np.array([[cb * em, -sb * em], [sb * ep, cb * ep]], dtype=complex)


def build_nc_split_soc_leg(
    sidecar,
    data,
    leg="z",
    sites_magnetic=None,
    vertex_component="delta_total",
    reference=None,
):
    """Build one Cartesian reference leg: an SU(2)-rotated spinor reference.

    The source producer evaluates its matrix in the ABINIT z-y Euler spin
    gauge. Rotate the no-SOC states and magnetic operator by that exact U,
    so the sidecar W_SO for this leg is in the matching composite band basis.
    The physical SOC operator stays fixed in the lattice frame; its band
    matrix generally differs for x/y/z references.
    """
    import dataclasses

    leg = str(leg).lower()
    if leg not in SOC_OFF_LEG_DIRECTIONS:
        raise ValueError(f"leg must be one of {SOC_OFF_LEG_DIRECTIONS}, got {leg!r}")
    if reference is None:
        reference = build_nc_split_soc_reference(
            sidecar,
            data,
            sites_magnetic=sites_magnetic,
            vertex_component=vertex_component,
        )
    if leg == "z":
        return reference

    u = _abinit_spinaxis_rotation(_LEG_AXES[leg])
    coefficients = np.einsum(
        "at,knbtp->knbap",
        u,
        np.asarray(reference.coefficients, dtype=complex),
        optimize=True,
    )
    spinor_operator = np.einsum(
        "as,bt,npqst->npqab",
        u,
        u.conj(),
        np.asarray(reference.spinor_operator, dtype=complex),
        optimize=True,
    )
    metadata = {
        **reference.metadata,
        "split_soc_leg": {**reference.metadata["split_soc_leg"], "leg": leg},
    }
    return dataclasses.replace(
        reference,
        coefficients=coefficients,
        spinor_operator=spinor_operator,
        metadata=metadata,
    )


def w_soc_for_leg(sidecar, leg):
    """Return the leg-gauge SOC operator from the sidecar (eV), (nk, 2b, 2b)."""
    return np.asarray(sidecar.w_so[sidecar.leg_index(leg)], dtype=complex)


# ---------------------------------------------------------------------------
# three-leg rotate/merge driver
# ---------------------------------------------------------------------------


def _default_window_prefixes(nband_composite):
    if nband_composite < 4:
        raise ValueError(
            "an even-prefix SOC window study requires at least two paired "
            f"band prefixes (2*b >= 4); got {nband_composite}"
        )
    return [nband_composite - 2, nband_composite]


def _tangent_projection_check(
    merged_exchange,
    z_leg_exchange,
    cell,
    positions,
    tol_eV,
):
    """FR-032 projection-only dimer gate: merged raw transverse block vs z.

    One collinear z reference determines only the raw 2x2 transverse block
    (J_xx, J_xy, J_yx, J_yy) and D_z of the physical tensor — never the full
    3x3 J (story adjudication).  This gate compares ONLY the raw
    ``tensor[:2, :2]`` of the rank-9 merged tensor against the z one-shot
    leg's ``J_leg`` (already the measured lattice transverse block), with a
    tolerance that respects reference-state differences between the legs.
    """
    report = {
        "gate": (
            "FR-032 dimer projection-only equivalence: merged raw tensor "
            "[:2, :2] (tangent block) vs z one-shot leg"
        ),
        "observable": "(J_xx, J_xy, J_yx, J_yy) — full Jiso only after rank-9 merge",
        "tol_eV": float(tol_eV),
        "pairs_compared": 0,
        "max_transverse_dev_eV": 0.0,
        "worst_pair": None,
    }
    for key, z_entry in z_leg_exchange.items():
        vector = np.asarray(key[0]) @ cell + positions[key[2]] - positions[key[1]]
        if float(np.linalg.norm(vector)) < 1e-6:
            continue
        if key not in merged_exchange:
            raise ValueError(
                "tangent projection gate: merged output is missing pair "
                f"{key}; the three legs and the one-shot run must share k "
                "mesh and R grid"
            )
        merged_entry = merged_exchange[key]
        merged_tensor = np.asarray(merged_entry["tensor"], dtype=float)
        z_block = np.asarray(z_entry["J_leg"], dtype=float)[:2, :2]
        deviation = float(np.abs(merged_tensor[:2, :2] - z_block).max())
        report["pairs_compared"] += 1
        if deviation > report["max_transverse_dev_eV"]:
            report["max_transverse_dev_eV"] = deviation
            report["worst_pair"] = [int(x) for x in key[0]] + [key[1], key[2]]
        if deviation > tol_eV:
            raise ValueError(
                "tangent projection gate FAILED for pair "
                f"{key}: max |dT[:2,:2]|={deviation:.3e} eV exceeds the "
                f"documented tolerance {tol_eV:.3e} eV; diagnose the leg "
                "reference states / merge instead of waiving the gate"
            )
    if report["pairs_compared"] == 0:
        raise ValueError(
            "tangent projection gate has no pairs to compare: the merged "
            "output contains no non-onsite exchange pairs"
        )
    report["passed"] = True
    return report


def nc_split_soc_soc_off_anchor(
    data,
    sidecar,
    dualized,
    sites,
    Rpts,
    nz,
    smearing_eV,
    vertex_component,
    jiso_rtol=_DEFAULT_ANCHOR_RTOL,
    dmi_tol_eV=_DEFAULT_ANCHOR_DMI_TOL_EV,
    reference=None,
):
    """SOC-off collinear anchor gate (existing NC PAO exchange kernel).

    The dualized stacked z reference replayed with ``W_SO = 0`` must
    reproduce the existing collinear ``delta_total`` projector exchange
    kernel exactly on the transverse diagonal (``J_leg[u, u]`` and
    ``J_leg[v, v]``), with a symmetric transverse block.
    """
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    reference_jdict = compute_projector_exchange_jdict(
        data,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        operator_component=vertex_component,
    )
    leg = (
        reference
        if reference is not None
        else build_nc_split_soc_reference(
            sidecar, dualized, sites_magnetic=sites, vertex_component=vertex_component
        )
    )
    zero_w = np.zeros((dualized.nkpt, 2 * dualized.nband, 2 * dualized.nband))
    result = compute_ks_split_soc_exchange(
        leg,
        zero_w,
        lam=1.0,
        mode=MODE_SECOND_VARIATION,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
    )
    report = {
        "gate": "SOC-off collinear anchor (compute_projector_exchange_jdict)",
        "vertex_component": vertex_component,
        "observable": "z reference J_leg[u,u] / J_leg[v,v]",
        "max_rel_Jiso_dev": 0.0,
        "max_transverse_asymmetry_eV": 0.0,
        "pairs_compared": 0,
    }
    for (r, i, j), entry in result["exchange"].items():
        key = (tuple(int(x) for x in r), i, j)
        ref = reference_jdict.get(key)
        if ref is None:
            continue
        frame = entry["frame"]
        u_ax, v_ax = int(frame["u"]), int(frame["v"])
        j_leg = np.asarray(entry["J_leg"], dtype=float)
        deviation = max(
            abs(float(j_leg[u_ax, u_ax]) - float(ref)) / max(1.0, abs(float(ref))),
            abs(float(j_leg[v_ax, v_ax]) - float(ref)) / max(1.0, abs(float(ref))),
        )
        asymmetry = float(abs(j_leg[u_ax, v_ax] - j_leg[v_ax, u_ax]))
        report["pairs_compared"] += 1
        report["max_rel_Jiso_dev"] = max(report["max_rel_Jiso_dev"], deviation)
        report["max_transverse_asymmetry_eV"] = max(
            report["max_transverse_asymmetry_eV"], asymmetry
        )
    if report["pairs_compared"] == 0:
        raise ValueError("SOC-off anchor found no comparable pairs")
    if report["max_rel_Jiso_dev"] > jiso_rtol:
        raise ValueError(
            "SOC-off anchor FAILED: spinor replay transverse diagonal deviates "
            f"from the collinear kernel by {report['max_rel_Jiso_dev']:.3e} "
            f"(rel, tol {jiso_rtol:.1e}); the dualized stacked replay must "
            "reproduce the existing exchange kernel at zero SOC"
        )
    if report["max_transverse_asymmetry_eV"] > dmi_tol_eV:
        raise ValueError(
            "SOC-off anchor FAILED: antisymmetric transverse block "
            f"{report['max_transverse_asymmetry_eV']:.3e} eV at zero SOC (tol "
            f"{dmi_tol_eV:.1e})"
        )
    report["passed"] = True
    return report


def _write_merged_results(data, merged_exchange, sites, path, description, provenance):
    """Write the rank-9 merged tensor set as one TB2J results directory."""
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
    for (r, i, j), entry in merged_exchange.items():
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
        nproj_site = int(data.site_nproj[site])
        vertex = data.spinor_operator[site, :nproj_site, :nproj_site]
        signed_trace = float(np.real(np.trace(vertex[:, :, 0, 0] - vertex[:, :, 1, 1])))
        if abs(signed_trace) <= 1e-10:
            raise ValueError(
                f"cannot infer magnetic direction for site {site} from zero "
                "vertex trace (is the strength-0 state actually magnetic?)"
            )
        # Majority-spin NC PAO potential is LOWER on a positive-moment site:
        # Delta = V_up - V_down has the opposite sign to M (PAW-consumer
        # convention); the reference quantization axis is +z.
        moment_sign = -site_magnetization_sign(vertex)
        spinat[site] = moment_sign * np.array([0.0, 0.0, 1.0])
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


def gen_exchange_abinit_nc_split_soc(
    pao_hs,
    soc_kernel,
    output_path="TB2J_results_abinit_nc_split_soc",
    legs=SOC_OFF_LEG_DIRECTIONS,
    Rcut=10.0,
    Rpts=None,
    nz=30,
    smearing_eV=0.05,
    magnetic_elements=None,
    index_magnetic_atoms=None,
    vertex_component="delta_total",
    lam=1.0,
    mode="second_variation",
    window_prefixes=None,
    verify_tangent_projection=True,
    tangent_tol_eV=_DEFAULT_TANGENT_TOL_EV,
    soc_off_anchor=True,
    anchor_jiso_rtol=_DEFAULT_ANCHOR_RTOL,
    anchor_dmi_tol_eV=_DEFAULT_ANCHOR_DMI_TOL_EV,
    wfk=None,
    frame_tol=1.0e-6,
    merge_consistency_atol=1.0e-6,
):
    """Rank-9 three-reference split-SOC exchange from PAO_HS v2 + nc_soc_ks v1.

    Builds the dualized strength-0 composite spinor reference and rotates
    each leg by the native ABINIT spinaxis matrix (states and magnetic vertex).
    The lattice-fixed SOC operator has distinct band matrices in the three
    rotated references; each is read from the matching sidecar leg. The
    reference-frame gates inspect the full Green function and vertex.
    The kernel returns only the measured transverse ``J_leg`` per leg; the
    full lattice tensor comes exclusively from ``merge_transverse_legs``.
    one TB2J results directory.  Per-leg raw ``J_leg`` blocks are stored as
    ``leg_<x|y|z>/split_soc_leg.npz`` — never as per-leg scalar
    Jiso/DMI/Jani, which a single reference cannot determine.

    Gates (default on): ``soc_off_anchor`` (W=0 z reference vs the existing
    collinear kernel), ``verify_tangent_projection`` (FR-032
    projection-only merged-vs-z transverse block equality).
    """
    from TB2J.interfaces.gpaw_projector import _magnetic_sites, _R_grid_for_cutoff
    from TB2J.split_soc_kernel import (
        band_window_convergence_report,
        compute_ks_split_soc_exchange,
        merge_transverse_legs,
        spinor_frame_rotation_residual,
        split_soc_frame_report,
    )

    legs = tuple(str(leg).lower() for leg in legs)
    if legs != SOC_OFF_LEG_DIRECTIONS:
        raise ValueError(f"legs must be {SOC_OFF_LEG_DIRECTIONS} in order, got {legs}")
    if mode != "second_variation":
        raise ValueError("NC rank-9 merge requires absolute second_variation exchange")

    data = load_abinit_nc_pao_savetb2j(pao_hs)
    sidecar = load_soc_ks_sidecar(soc_kernel)
    pairing = validate_sidecar_pairing(sidecar, data, pao_hs_path=pao_hs, wfk_path=wfk)
    if sidecar.pao_hs_sha256 is None:
        raise ValueError(
            "sidecar carries no pao_hs_sha256: refusing an unverifiable pairing"
        )
    dualized = dualize_pao_coefficients(data)

    if magnetic_elements is None and index_magnetic_atoms is None:
        raise ValueError(
            "select magnetic sites with index_magnetic_atoms or "
            "magnetic_elements; the NC PAO sidecar includes ligand SOC"
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

    reference = build_nc_split_soc_reference(
        sidecar, dualized, sites_magnetic=sites, vertex_component=vertex_component
    )

    if soc_off_anchor:
        anchor_report = nc_split_soc_soc_off_anchor(
            data,
            sidecar,
            dualized,
            sites,
            Rpts,
            nz,
            smearing_eV,
            vertex_component,
            jiso_rtol=anchor_jiso_rtol,
            dmi_tol_eV=anchor_dmi_tol_eV,
            reference=reference,
        )
    else:
        anchor_report = {
            "gate": "SOC-off collinear anchor",
            "passed": None,
            "note": "disabled by request; never ship production values unanchored",
        }

    if window_prefixes is None:
        window_prefixes = _default_window_prefixes(sidecar.nband_composite)
    window_prefixes = sorted(int(nw) for nw in window_prefixes)

    output_root = Path(output_path)
    leg_dirs = {}
    leg_metadata = {}
    leg_exchanges = {}
    pao_sha = pairing["pao_hs"]["sha256"]
    for leg in legs:
        spinaxis = sidecar.spinaxis[sidecar.leg_index(leg)]
        if not np.allclose(spinaxis, _LEG_AXES[leg], atol=_AXIS_ALIGN_TOL):
            raise ValueError(
                f"leg {leg!r}: sidecar spinaxis {spinaxis} does not align with "
                f"the +{leg} Cartesian axis"
            )
        leg_data = build_nc_split_soc_leg(
            sidecar,
            dualized,
            leg=leg,
            sites_magnetic=sites,
            vertex_component=vertex_component,
            reference=reference,
        )
        leg_w = w_soc_for_leg(sidecar, leg)
        result = compute_ks_split_soc_exchange(
            leg_data,
            leg_w,
            lam=lam,
            mode=mode,
            Rpts=Rpts,
            nz=nz,
            smearing_eV=smearing_eV,
            sites=sites,
            metadata={
                "strength0_reference": {
                    "code": "abinit",
                    "schema": (
                        f"{ABINIT_NC_PAO_HS_SCHEMA_NAME}/"
                        f"{ABINIT_NC_PAO_HS_SCHEMA_VERSION}"
                    ),
                    "checkpoint": pairing["pao_hs"]["path"],
                    "sha256": pao_sha,
                    "wfk": pairing["wfk"]["name"],
                    "wfk_sha256": pairing["wfk"]["sha256"],
                    "description": (
                        "collinear NC PAO strength-0 projections (abinao "
                        "<phi|psi>, dualized B = S^-1 C, overlap used not dropped)"
                    ),
                },
                "soc_operator_source": {
                    "sidecar_schema": (
                        f"{NC_SOC_KS_SCHEMA_NAME}/{NC_SOC_KS_SCHEMA_VERSION}"
                    ),
                    "sidecar_path": sidecar.source_path,
                    "operator": (sidecar.fr050.get("operator") or {}).get(
                        "source", "abinao compute_wfk_soc_kernel"
                    ),
                    "units": "eV",
                    "frame": "lattice-fixed operator in ABINIT leg-gauge band states",
                },
                "frame": {
                    "leg": leg,
                    "axis": [float(v) for v in _LEG_AXES[leg]],
                    "construction": "ABINIT z-y Euler spin rotation of states and vertex; sidecar W in matching leg gauge",
                },
                "merge_mode": "rank9_transverse_merge",
                "quantity_scope": (
                    "single magnetic reference axis: only the measured "
                    "transverse J_leg block exists per leg; no per-leg "
                    "Jiso/DMI/Jani is published and the full tensor requires "
                    "all three reference axes (rank-9 merge)"
                ),
                "vertex_component": vertex_component,
                "sidecar_pairing": pairing,
                "soc_off_anchor": anchor_report,
            },
        )
        frame_report = split_soc_frame_report(
            leg_data,
            axis=_LEG_AXES[leg],
            tol=frame_tol,
            reference_data=reference,
            sites=sites,
        )
        if not frame_report["ok"]:
            raise ValueError(
                f"leg {leg!r} frame report FAILED: {frame_report}; a leg must "
                "be a genuine global SU(2) rotation of the reference (vertex "
                "direction and spectrum), not a metadata-only relabelling"
            )
        rotation_residual = spinor_frame_rotation_residual(
            leg_data,
            reference,
            axis=_LEG_AXES[leg],
            energy=float(data.efermi - 3.0),
            rpts=Rpts,
            sites=sites,
            rotation=_abinit_spinaxis_rotation(_LEG_AXES[leg]),
        )
        if rotation_residual["max_g_residual"] > frame_tol:
            raise ValueError(
                f"leg {leg!r} G-covariance proof FAILED: max_g_residual="
                f"{rotation_residual['max_g_residual']:.3e} exceeds tol "
                f"{frame_tol:.1e}; the leg is not an SU(2) rotation of the "
                "reference"
            )
        study = band_window_convergence_report(
            leg_data,
            leg_w,
            window_prefixes,
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
        provenance = json_safe_provenance(
            {
                **result["metadata"],
                "backend": "abinit_nc",
                "leg": leg,
                "frame_report": frame_report,
                "rotation_residual": rotation_residual,
                "merge_mode": "rank9_transverse_merge",
            }
        )
        leg_dir = output_root / f"leg_{leg}"
        leg_dir.mkdir(parents=True, exist_ok=True)
        keys = sorted(result["exchange"])
        np.savez(
            leg_dir / "split_soc_leg.npz",
            keys=np.array([repr(k) for k in keys]),
            j_leg=np.stack(
                [np.asarray(result["exchange"][k]["J_leg"], dtype=float) for k in keys]
            ),
            mask_residual=np.array(
                [float(result["exchange"][k]["mask_residual"]) for k in keys]
            ),
        )
        (leg_dir / "split_soc_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )
        leg_dirs[leg] = leg_dir
        leg_metadata[leg] = provenance
        leg_exchanges[leg] = result["exchange"]

    merged = merge_transverse_legs(
        {leg: {"exchange": leg_exchanges[leg]} for leg in legs},
        consistency_atol=merge_consistency_atol,
    )
    merged_provenance = {
        "schema": leg_metadata[legs[0]]["schema"],
        "backend": "abinit_nc",
        "merge_mode": "rank9_transverse_merge",
        "legs": leg_metadata,
        "construction": "one no-SOC reference; ABINIT z-y rotated states and vertices; matching sidecar SOC matrix per leg",
        "full_tensor": (
            "rank-9 merge over three independent magnetic reference axes; "
            "merged Jiso/DMI/Jani are the final observables and exist ONLY "
            "at this level"
        ),
        "merge_diagnostics": merged["diagnostics"],
        "sidecar_pairing": pairing,
        "soc_off_anchor": anchor_report,
        "lambda": float(lam),
        "units": "eV",
    }
    description = (
        "ABINIT NC split-SOC rank-9 merged exchange (story-009): x/y/z "
        "reference legs from the abinao.nc_soc_ks sidecar W_SO (all-atom "
        "propagator SOC, dualized PAO maps), merged transverse blocks via "
        "merge_transverse_legs.\n"
    )

    tangent_report = None
    if verify_tangent_projection:
        tangent_report = _tangent_projection_check(
            merged["exchange"],
            leg_exchanges["z"],
            data.cell,
            data.positions,
            tangent_tol_eV,
        )
        merged_provenance["tangent_projection_check"] = tangent_report

    _write_merged_results(
        reference,
        merged["exchange"],
        sites,
        output_root,
        description,
        merged_provenance,
    )
    (output_root / "split_soc_provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, default=str) + "\n"
    )
    return {
        "output_path": output_root,
        "leg_paths": leg_dirs,
        "merged_exchange": merged["exchange"],
        "merge_diagnostics": merged["diagnostics"],
        "metadata": merged_provenance,
        "leg_metadata": leg_metadata,
        "soc_off_anchor": anchor_report,
        "tangent_projection_check": tangent_report,
    }
