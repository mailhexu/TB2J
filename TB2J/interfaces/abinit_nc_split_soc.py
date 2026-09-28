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
Per-leg tensors are rotated to the lattice frame with the ABINIT spinaxis
convention ``T_lattice = O T_leg O^T`` (``O e_z`` = leg spinaxis) and the
three legs are merged with ``TB2J.io_merge``.

Two acceptance gates run by default:

- **FR-032 dimer gate**: the merged three-leg output must equal a one-shot
  KS tensor kernel evaluation in the lattice frame (the z leg, whose
  spinaxis is +e_z, so ``O = I``) within a documented tolerance.
- **SOC-off collinear anchor**: with ``W_SO = 0`` the consumer's spinor
  replay must reproduce the existing collinear NC PAO exchange kernel
  (``compute_projector_exchange_jdict``) shell for shell on the same
  strength-0 data.
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
from TB2J.split_soc_kernel import band_window_convergence_report, json_safe_provenance

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


def build_nc_split_soc_leg(
    sidecar,
    data,
    leg="z",
    sites_magnetic=None,
    vertex_component="delta_total",
):
    """Build the normalized no-SOC spinor band data for one NC PAO leg.

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

    leg = str(leg).lower()
    if leg not in SOC_OFF_LEG_DIRECTIONS:
        raise ValueError(f"leg must be one of {SOC_OFF_LEG_DIRECTIONS}, got {leg!r}")
    if data.nspinor != 1 or data.coefficients.ndim != 4:
        raise ValueError("build_nc_split_soc_leg expects collinear PAO_HS data")
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
            "leg": leg,
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


def w_soc_for_leg(sidecar, leg):
    """Return the leg's all-atom band-window SOC operator (eV), (nk, 2b, 2b)."""
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


def _load_merged_results(path):
    from TB2J.io_exchange.io_exchange import SpinIO

    return SpinIO.load_pickle(path=str(path))


def _tangent_projection_check(
    merged_results,
    one_shot_entries,
    sites,
    cell,
    positions,
    tol_eV,
):
    """FR-032 projection-only dimer gate: merged raw transverse block vs z.

    One collinear z reference determines only the raw 2x2 transverse block
    (J_xx, J_xy, J_yx, J_yy) and D_z of the physical tensor — never the full
    3x3 J (SymPy story-001/002 adjudication; confirmed numerically on
    synthetic psi-gauge data where Jiso/DMI_z are leg-consistent to roundoff
    while the transverse block is not).  This gate therefore compares ONLY
    the raw ``tensor[:2, :2]`` of the rank-9 merged tensor against the z
    one-shot leg, with a tolerance that respects reference-state
    differences between the legs.  Single-leg ``Jiso`` is never treated as
    a final observable; the full isotropic/Jani/DMI split is meaningful
    only after the three-leg merge.
    """
    one_shot = {}
    for (r, i, j), entry in one_shot_entries.items():
        vector = np.asarray(r) @ cell + positions[j] - positions[i]
        if float(np.linalg.norm(vector)) < 1e-6:
            continue
        one_shot[(tuple(int(x) for x in r), i, j)] = entry
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

    def _transverse(jiso, dmi, jani):
        from TB2J.Jtensor import combine_J_tensor

        tensor = combine_J_tensor(Jiso=float(jiso), D=dmi, Jani=jani)
        return tensor[:2, :2]

    for key, entry in one_shot.items():
        if key not in merged_results.exchange_Jdict:
            raise ValueError(
                "tangent projection gate: merged output is missing pair "
                f"{key}; the three legs and the one-shot run must share k "
                "mesh and R grid"
            )
        merged_block = _transverse(
            merged_results.exchange_Jdict[key],
            merged_results.dmi_ddict[key],
            merged_results.Jani_dict[key],
        )
        z_block = _transverse(entry["Jiso"], entry["dmi"], entry["jani"])
        deviation = float(np.abs(merged_block - z_block).max())
        report["pairs_compared"] += 1
        if deviation > report["max_transverse_dev_eV"]:
            report["max_transverse_dev_eV"] = deviation
            report["worst_pair"] = list(key)
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
):
    """SOC-off collinear anchor gate (existing NC PAO exchange kernel).

    The dualized stacked leg replayed with ``W_SO = 0`` must reproduce the
    existing collinear ``delta_total`` projector exchange Jiso pair by pair,
    with vanishing DMI.
    """
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.split_soc_kernel import (
        MODE_SECOND_VARIATION,
        compute_ks_split_soc_exchange,
    )

    reference = compute_projector_exchange_jdict(
        data,
        Rpts=Rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        operator_component=vertex_component,
    )
    leg = build_nc_split_soc_leg(
        sidecar,
        dualized,
        leg="z",
        sites_magnetic=sites,
        vertex_component=vertex_component,
    )
    zero_w = np.zeros((data.nkpt, 2 * data.nband, 2 * data.nband))
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
        "max_rel_Jiso_dev": 0.0,
        "max_dmi_norm_eV": 0.0,
        "pairs_compared": 0,
    }
    for (r, i, j), entry in result["exchange"].items():
        key = (tuple(int(x) for x in r), i, j)
        ref = reference.get(key)
        if ref is None:
            continue
        deviation = abs(float(entry["Jiso"]) - float(ref)) / max(1.0, abs(float(ref)))
        report["pairs_compared"] += 1
        report["max_rel_Jiso_dev"] = max(report["max_rel_Jiso_dev"], deviation)
        report["max_dmi_norm_eV"] = max(
            report["max_dmi_norm_eV"], float(np.linalg.norm(entry["dmi"]))
        )
    if report["pairs_compared"] == 0:
        raise ValueError("SOC-off anchor found no comparable pairs")
    if report["max_rel_Jiso_dev"] > jiso_rtol:
        raise ValueError(
            "SOC-off anchor FAILED: spinor replay Jiso deviates from the "
            f"collinear kernel by {report['max_rel_Jiso_dev']:.3e} (rel, tol "
            f"{jiso_rtol:.1e}); the dualized stacked replay must reproduce "
            "the existing exchange kernel at zero SOC"
        )
    if report["max_dmi_norm_eV"] > dmi_tol_eV:
        raise ValueError(
            "SOC-off anchor FAILED: nonzero DMI "
            f"{report['max_dmi_norm_eV']:.3e} eV at zero SOC (tol "
            f"{dmi_tol_eV:.1e})"
        )
    report["passed"] = True
    return report


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
):
    """Three-direction split-SOC exchange from PAO_HS v2 + nc_soc_ks v1.

    For each leg: take the sidecar's eV band-window SOC operator, run the
    story-002 KS second-variation kernel on the dualized composite spinor
    window, attach the even-prefix window convergence study, rotate the
    output tensors to the lattice frame (``T_lattice = O T_leg O^T``, ABINIT
    spinaxis convention), and write one noncollinear TB2J results directory
    per leg.  The three legs are merged with ``TB2J.io_merge.merge``.

    Gates (both on by default): ``verify_tangent_projection`` enforces the
    FR-032 projection-only equivalence — the rank-9 merged tensor's raw
    transverse block vs the z one-shot leg within a documented tolerance
    (never a full-tensor single-leg claim: one collinear reference
    determines only J_xx/J_xy/J_yx/J_yy and D_z); ``soc_off_anchor``
    enforces the SOC-off collinear anchor on the same strength-0 data.
    """
    from TB2J.interfaces.abinit_paw_split_soc import (
        _rotate_entry_tensors,
        _write_leg_results,
        rotation_matrix_from_su2,
        su2_leg_rotation,
    )
    from TB2J.interfaces.gpaw_projector import _magnetic_sites, _R_grid_for_cutoff
    from TB2J.io_merge import merge
    from TB2J.projector_green import site_magnetization_sign
    from TB2J.split_soc_kernel import compute_ks_split_soc_exchange

    legs = tuple(str(leg).lower() for leg in legs)
    if legs != SOC_OFF_LEG_DIRECTIONS:
        raise ValueError(f"legs must be {SOC_OFF_LEG_DIRECTIONS} in order, got {legs}")
    if mode != "second_variation":
        raise ValueError(
            "NC three-leg SpinIO output requires absolute second_variation exchange"
        )

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
    leg_z_entries = None
    pao_sha = pairing["pao_hs"]["sha256"]
    for leg in legs:
        su2 = su2_leg_rotation(leg)
        o_frame = rotation_matrix_from_su2(su2)
        leg_axis = o_frame @ np.array([0.0, 0.0, 1.0])
        spinaxis = sidecar.spinaxis[sidecar.leg_index(leg)]
        if not np.allclose(leg_axis, spinaxis, atol=1e-8):
            raise ValueError(
                f"leg {leg!r}: sidecar spinaxis {spinaxis} disagrees with the "
                f"ABINIT spinaxis rotation axis {leg_axis}"
            )
        leg_data = build_nc_split_soc_leg(
            sidecar,
            dualized,
            leg=leg,
            sites_magnetic=sites,
            vertex_component=vertex_component,
        )
        w_leg = w_soc_for_leg(sidecar, leg)
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
                },
                "frame": {
                    "leg": leg,
                    "spinaxis": [float(v) for v in spinaxis],
                    "leg_axis_lattice": [float(v) for v in leg_axis],
                },
                "merge_mode": "three_leg_rotate_merge",
                "quantity_scope": (
                    "single magnetic reference axis: only the raw transverse "
                    "block (J_xx, J_xy, J_yx, J_yy) and D_z of this leg are "
                    "physical observables; leg Jiso/DMI/Jani scalars are NOT "
                    "final and the full tensor requires all three reference "
                    "axes (rank-9 merge)"
                ),
                "vertex_component": vertex_component,
                "sidecar_pairing": pairing,
                "soc_off_anchor": anchor_report,
            },
        )
        study = band_window_convergence_report(
            leg_data,
            w_leg,
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
                    f"cannot infer magnetic direction for site {site} from zero "
                    "vertex trace (is the strength-0 state actually magnetic?)"
                )
            # Majority-spin NC PAO potential is LOWER on a positive-moment
            # site: Delta = V_up - V_down has the opposite sign to M (same
            # convention as the PAW consumer).
            moment_sign = -site_magnetization_sign(vertex)
            spinat_vectors[site] = moment_sign * leg_axis
        leg_dir = output_root / f"leg_{leg}"
        description = (
            f"ABINIT NC split-SOC leg '{leg}' (story-009): KS second-variation "
            "spectra from the abinao.nc_soc_ks sidecar W_SO (all-atom "
            "propagator SOC, magnetic-only delta vertices, dualized PAO maps), "
            "tensors rotated to the lattice frame with T_lattice = O T_leg O^T.\n"
        )
        rotated = {
            key: _rotate_entry_tensors(entry, o_frame)
            for key, entry in result["exchange"].items()
        }
        _write_leg_results(
            data, rotated, sites, leg_dir, description, spinat_vectors, provenance
        )
        (leg_dir / "split_soc_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str) + "\n"
        )
        leg_dirs[leg] = leg_dir
        leg_metadata[leg] = provenance
        if leg == "z":
            leg_z_entries = rotated

    merged_provenance = {
        "schema": leg_metadata[legs[0]]["schema"],
        "backend": "abinit_nc",
        "merge_mode": "three_leg_rotate_merge",
        "legs": leg_metadata,
        "rotation": "T_lattice = O T_leg O^T with O e_z = leg spinaxis",
        "full_tensor": (
            "rank-9 three-leg merge over three independent magnetic "
            "reference axes; merged Jiso/DMI/Jani are the final observables"
        ),
        "sidecar_pairing": pairing,
        "soc_off_anchor": anchor_report,
        "lambda": float(lam),
        "units": "eV",
    }
    merge(
        *[str(leg_dirs[leg]) for leg in legs],
        save=True,
        write_path=str(output_root),
        merged_provenance=merged_provenance,
    )

    tangent_report = None
    if verify_tangent_projection:
        merged_results = _load_merged_results(output_root)
        tangent_report = _tangent_projection_check(
            merged_results,
            leg_z_entries,
            sites,
            data.cell,
            data.positions,
            tangent_tol_eV,
        )
        merged_provenance["tangent_projection_check"] = tangent_report

    (output_root / "split_soc_provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, default=str) + "\n"
    )
    return {
        "output_path": output_root,
        "leg_paths": leg_dirs,
        "metadata": merged_provenance,
        "leg_metadata": leg_metadata,
        "soc_off_anchor": anchor_report,
        "tangent_projection_check": tangent_report,
    }
