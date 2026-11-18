"""Frozen-band pitch slope for spin spirals (story 006).

Signed fixed-reference pitch response of a folded generalized-Bloch
pencil. For normalized generalized eigenpairs ``H(k) c_n = eps_n S(k)
c_n`` with ``c_n^dag S c_n = 1`` and FROZEN occupations ``f_nk`` (no
re-Fermi refilling, fields and functional untouched), the per-primitive-
cell fractional-reciprocal pitch gradient is

    dE/dq_a = sum_k w_k sum_n f_nk
              c_nk^dag (dH/dq_a - eps_nk dS/dq_a) c_nk

in eV per primitive cell per fractional q component (a = x,y,z).

Normative conventions: ``TB2J/docs/sympy/pitch_slope_generalized.py``
(``U(D) = exp(-i D sigma_y/2)``, folded dressing and its q derivative),
the q-slope research memo, and ADR-R3/R4 of the spiral-first-order-
response architecture: this is the frozen-band spectral slope, never a
relaxed-SCF energy derivative; ``dS/dq`` carries only genuine
representation motion under the pitch (twist phases, rotated overlap).

Two entry points:

* :func:`frozen_pitch_slope` - provider seam (``gen_ham(k)`` /
  ``q_derivative(k)``, e.g. the v2 ``PlanarSpiralProvider``);
  eigenvectors are obtained internally by ``scipy.linalg.eigh``.
* :func:`frozen_pitch_slope_from_eigenpairs` - explicit normalized
  generalized eigenpairs (e.g. reconstructed from a v2 sidecar).

An optional q-path tangent turns the vector into a scalar: a coordinate
derivative along the tangent as given, or an arc-length-normalized
projection; without a tangent only the vector is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "PITCH_SLOPE_UNITS",
    "FrozenPitchSlope",
    "FrozenPitchSlopeError",
    "FrozenPitchShapeError",
    "FrozenPitchEigenpairError",
    "frozen_pitch_slope",
    "frozen_pitch_slope_from_eigenpairs",
]

PITCH_SLOPE_UNITS = "eV / primitive cell / fractional q component"

_PROTOCOL = (
    "frozen-band fixed occupations (no re-Fermi refilling); reference-q "
    "generalized eigenvectors and fields held fixed; "
    "dE/dq_a = sum_k w_k sum_n f_nk c^dag (dH/dq_a - eps_nk dS/dq_a) c_nk"
)

_NORM_TOL_DEFAULT = 1e-8
_HERM_TOL_DEFAULT = 1e-10
_KWEIGHT_SUM_TOL = 1e-6


class FrozenPitchSlopeError(ValueError):
    """Base class for frozen pitch slope input/protocol violations."""


class FrozenPitchShapeError(FrozenPitchSlopeError):
    """Array shapes, k-point metadata or k-weight normalization invalid."""


class FrozenPitchEigenpairError(FrozenPitchSlopeError):
    """Eigenpair/occupation protocol violation (wrong S, Hermiticity, f)."""


@dataclass
class FrozenPitchSlope:
    """Frozen-band pitch response of a spiral reference.

    Attributes:
        vector: ``(3,)`` signed ``dE/dq_a``, in
            :data:`PITCH_SLOPE_UNITS`.
        tangent: ``(3,)`` fractional reciprocal path direction or None.
        slope: ``vector . tangent`` (coordinate derivative) or
            ``vector . tangent/|tangent|`` (arc-length-normalized
            projection); None when no tangent was supplied.
        tangent_normalized: True iff ``slope`` is the normalized
            projection.
        units: unit string for ``vector`` and ``slope``.
        per_k: optional ``(nk, 3)`` per-k-point contributions (before
            k-weighting), for diagnostics.
        metadata: protocol/provenance dict (held-fixed protocol, k mesh
            shape, kweight sum, normalization and Hermiticity audit
            residuals, caller-measured ``fd_residual`` slot, ...).
    """

    vector: np.ndarray
    tangent: np.ndarray | None = None
    slope: float | None = None
    tangent_normalized: bool = False
    units: str = PITCH_SLOPE_UNITS
    per_k: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def slope_kind(self) -> str | None:
        """Identity of the scalar: coordinate or arc-length-normalized."""
        if self.tangent is None:
            return None
        return "arc_length_normalized" if self.tangent_normalized else "coordinate"

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready report (story 007 response schema keys)."""
        return {
            "frozen_band_pitch_slope": [float(x) for x in self.vector],
            "units": self.units,
            "path_tangent": None
            if self.tangent is None
            else [float(x) for x in self.tangent],
            "path_slope": None if self.slope is None else float(self.slope),
            "path_slope_kind": self.slope_kind,
            "metadata": dict(self.metadata),
        }


def _as_finite_complex(name, arr):
    arr = np.asarray(arr, dtype=complex)
    if not (np.isfinite(arr.real).all() and np.isfinite(arr.imag).all()):
        raise FrozenPitchEigenpairError(f"{name} contains non-finite values")
    return arr


def _as_finite_float(name, arr):
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        raise FrozenPitchEigenpairError(f"{name} contains non-finite values")
    return arr


def _check_shapes(eps, vec, dH, dS, occ, kw, overlap):
    nk, nb = eps.shape
    if eps.ndim != 2:
        raise FrozenPitchShapeError(f"eigenvalues must be (nk, nb), got {eps.shape}")
    if vec.shape != (nk, nb, nb):
        raise FrozenPitchShapeError(
            f"eigenvectors must be (nk, nb, nb) columns c[:, n], got {vec.shape} "
            f"for eigenvalues {eps.shape}"
        )
    for name, arr in (("dH", dH), ("dS", dS)):
        if arr is None:
            continue
        if arr.shape != (nk, 3, nb, nb):
            raise FrozenPitchShapeError(
                f"{name} must be (nk, 3, nb, nb), got {arr.shape}; component "
                "order is (x, y, z) fractional reciprocal"
            )
    if occ.shape != (nk, nb):
        raise FrozenPitchShapeError(
            f"occupations must be (nk, nb) = {occ.shape} vs eigenvalues {eps.shape}"
        )
    if kw.shape != (nk,):
        raise FrozenPitchShapeError(f"kweights must be (nk,) = {kw.shape} vs nk={nk}")
    if overlap is not None and overlap.shape != (nk, nb, nb):
        raise FrozenPitchShapeError(
            f"overlap must be (nk, nb, nb), got {overlap.shape}"
        )


def _hermiticity_residual(arr) -> float:
    return float(np.max(np.abs(arr - np.conj(np.swapaxes(arr, -1, -2)))))


def _validate_common(
    eps, vec, dH, dS, occ, kw, overlap, norm_tol, herm_tol, check_normalization
):
    _check_shapes(eps, vec, dH, dS, occ, kw, overlap)
    eps = _as_finite_float("eigenvalues", eps)
    vec = _as_finite_complex("eigenvectors", vec)
    if np.any(occ < 0.0) or np.any(occ > 1.0):
        raise FrozenPitchEigenpairError(
            "fixed occupations must lie in [0, 1]; got range "
            f"[{occ.min():.6g}, {occ.max():.6g}]"
        )
    occ = _as_finite_float("occupations", occ)
    kw = _as_finite_float("kweights", kw)
    wsum = float(kw.sum())
    if abs(wsum - 1.0) > _KWEIGHT_SUM_TOL:
        raise FrozenPitchShapeError(
            "kweights must sum to 1 for eV-per-primitive-cell units; got "
            f"{wsum:.12g} (normalize the mesh weights)"
        )
    herm_res = 0.0
    for name, arr in (("dH", dH), ("dS", dS)):
        if arr is None:
            continue
        res = _hermiticity_residual(arr)
        herm_res = max(herm_res, res)
        if res > herm_tol:
            raise FrozenPitchEigenpairError(
                f"{name} must be Hermitian at every k (max |X - X^dag| = "
                f"{res:.3e} > {herm_tol:.1e})"
            )
    norm_res = None
    if overlap is not None:
        overlap = _as_finite_complex("overlap", overlap)
        c = np.transpose(vec, (0, 2, 1))
        gram = np.einsum("kmi,kij,knj->kmn", np.conj(c), overlap, c)
        norm_res = float(np.max(np.abs(gram - np.eye(eps.shape[1]))))
        if check_normalization and norm_res > norm_tol:
            raise FrozenPitchEigenpairError(
                "eigenvectors are not S-normalized (max |c^dag S c - I| = "
                f"{norm_res:.3e} > {norm_tol:.1e}): wrong overlap S or "
                "eigenvectors not normalized with the supplied S"
            )
    return eps, vec, occ, kw, herm_res, norm_res


def _contract(eps, vec, dH, dS, occ, kw):
    # band-major c: c[k, n, i] = i-th component of band-n eigenvector
    c = np.transpose(vec, (0, 2, 1))
    c_c = np.conj(c)
    pdH = np.einsum("kni,kaij,knj->kna", c_c, dH, c)
    if dS is None:
        val = pdH
    else:
        pdS = np.einsum("kni,kaij,knj->kna", c_c, dS, c)
        val = pdH - eps[:, :, None] * pdS
    per_k = np.einsum("kn,kna->ka", occ, val)
    vector = np.einsum("k,ka->a", kw, per_k)
    imag_round = float(np.max(np.abs(val.imag))) if val.size else 0.0
    return vector.real, per_k.real, imag_round


def _direction(tangent, normalize_tangent):
    if tangent is None:
        if normalize_tangent:
            raise FrozenPitchSlopeError("normalize_tangent requires a path tangent")
        return None, None, False
    t = _as_finite_float("tangent", np.asarray(tangent, dtype=float)).reshape(-1)
    if t.shape != (3,):
        raise FrozenPitchShapeError(
            f"tangent must be a 3-vector in fractional reciprocal units, got {t.shape}"
        )
    norm = float(np.linalg.norm(t))
    if norm <= 0.0:
        raise FrozenPitchSlopeError("path tangent must have positive norm")
    normalized = bool(normalize_tangent)
    if normalized:
        t = t / norm
    return t, None, normalized


def _build_result(
    eps,
    vec,
    dH,
    dS,
    occ,
    kw,
    overlap,
    tangent,
    normalize_tangent,
    fd_residual,
    check_normalization,
    norm_tol,
    herm_tol,
    return_per_k,
    occupations_protocol,
):
    eps, vec, occ, kw, herm_res, norm_res = _validate_common(
        eps, vec, dH, dS, occ, kw, overlap, norm_tol, herm_tol, check_normalization
    )
    t, _, normalized = _direction(tangent, normalize_tangent)
    vector, per_k, imag_round = _contract(eps, vec, dH, dS, occ, kw)
    metadata = {
        "protocol": _PROTOCOL,
        "units": PITCH_SLOPE_UNITS,
        "k_mesh_shape": [int(eps.shape[0]), int(eps.shape[1])],
        "kweight_sum": float(kw.sum()),
        "occupations_protocol": occupations_protocol,
        "normalization_audit": (
            "checked (overlap supplied)"
            if overlap is not None
            else "assumed S-normalized (no overlap supplied)"
        ),
        "normalization_max_residual": norm_res,
        "hermiticity_max_residual": herm_res,
        "max_imaginary_roundoff": imag_round,
        "fd_residual": None if fd_residual is None else float(fd_residual),
    }
    slope = None
    if t is not None:
        slope = float(vector @ t)
    return FrozenPitchSlope(
        vector=vector,
        tangent=t,
        slope=slope,
        tangent_normalized=normalized,
        per_k=per_k if return_per_k else None,
        metadata=metadata,
    )


def frozen_pitch_slope_from_eigenpairs(
    eigenvalues,
    eigenvectors,
    dH,
    dS,
    *,
    occupations,
    kweights,
    overlap=None,
    tangent=None,
    normalize_tangent=False,
    fd_residual=None,
    check_normalization=True,
    norm_tol=_NORM_TOL_DEFAULT,
    hermiticity_tol=_HERM_TOL_DEFAULT,
    return_per_k=False,
) -> FrozenPitchSlope:
    """Frozen-band pitch slope from normalized generalized eigenpairs.

    Args:
        eigenvalues: ``(nk, nb)`` reference-q band energies in eV.
        eigenvectors: ``(nk, nb, nb)`` generalized eigenvectors as
            columns ``c[:, n]`` with ``c^dag S c = I``.
        dH: ``(nk, 3, nb, nb)`` analytic ``dH/dq_a``; component order
            (x, y, z) fractional reciprocal.
        dS: ``(nk, 3, nb, nb)`` analytic ``dS/dq_a`` or None for an
            orthogonal (q-independent) basis.
        occupations: ``(nk, nb)`` fixed occupation numbers in [0, 1],
            frozen against band index; no re-Fermi refilling is done.
        kweights: ``(nk,)`` mesh weights summing to 1 (per-cell units).
        overlap: ``(nk, nb, nb)`` optional ``S(k)`` used to audit the
            eigenvector normalization; audits catch a wrong or
            inconsistent S (``norm_tol``).
        tangent: optional ``(3,)`` fractional reciprocal q-path
            direction; see :class:`FrozenPitchSlope`.
        normalize_tangent: project on ``tangent/|tangent|`` instead of
            the raw tangent.
        fd_residual: optional caller-measured finite-difference
            agreement |analytic - FD| (eV per fractional q unit),
            recorded in metadata; this module never computes FD itself.
        check_normalization: enforce the S-normalization audit.
        norm_tol: maximum ``max|c^dag S c - I|`` when auditing.
        hermiticity_tol: maximum ``max|X - X^dag|`` for dH/dS.
        return_per_k: also return the ``(nk, 3)`` per-k contributions.

    Returns:
        :class:`FrozenPitchSlope`.
    """
    return _build_result(
        np.asarray(eigenvalues, dtype=float),
        np.asarray(eigenvectors, dtype=complex),
        None if dH is None else np.asarray(dH, dtype=complex),
        None if dS is None else np.asarray(dS, dtype=complex),
        np.asarray(occupations, dtype=float),
        np.asarray(kweights, dtype=float),
        None if overlap is None else np.asarray(overlap, dtype=complex),
        tangent,
        normalize_tangent,
        fd_residual,
        check_normalization,
        norm_tol,
        hermiticity_tol,
        return_per_k,
        occupations_protocol=(
            "fixed input occupations f_nk (frozen, no re-Fermi); caller owns "
            "the band-index convention"
        ),
    )


def frozen_pitch_slope(
    provider,
    *,
    kpts,
    occupations,
    kweights,
    tangent=None,
    normalize_tangent=False,
    fd_residual=None,
    check_normalization=True,
    norm_tol=_NORM_TOL_DEFAULT,
    hermiticity_tol=_HERM_TOL_DEFAULT,
    return_per_k=False,
) -> FrozenPitchSlope:
    """Frozen-band pitch slope from a folded-pencil provider.

    The provider seam is duck-typed to the v2 ``PlanarSpiralProvider``
    contract: ``gen_ham(k) -> (H, S)`` and ``q_derivative(k) ->
    (dH, dS)`` with per-k ``dH, dS`` of shape ``(3, n, n)`` in (x, y, z)
    fractional reciprocal order. Eigenpairs are obtained per k with
    ``scipy.linalg.eigh(H, S)`` (ascending eigenvalues, ``c^dag S c =
    I``); the supplied occupations must follow that ascending order and
    stay frozen.

    See :func:`frozen_pitch_slope_from_eigenpairs` for the remaining
    arguments and :class:`FrozenPitchSlope` for the return value.
    """
    kpts = _as_finite_float("kpts", np.asarray(kpts, dtype=float))
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise FrozenPitchShapeError(
            f"kpts must be (nk, 3) fractional reciprocal, got {kpts.shape}"
        )
    nk = kpts.shape[0]
    eps_list, vec_list, overlap_list, dH_list, dS_list = [], [], [], [], []
    for i, k in enumerate(kpts):
        H, S = provider.gen_ham(k)
        dH, dS = provider.q_derivative(k)
        H = _as_finite_complex(f"gen_ham({i}) H", H)
        S = _as_finite_complex(f"gen_ham({i}) S", S)
        if H.ndim != 2 or H.shape != S.shape or H.shape[0] != H.shape[1]:
            raise FrozenPitchShapeError(
                f"gen_ham must return square (n, n) (H, S); got {H.shape} vs {S.shape}"
            )
        dH = _as_finite_complex(f"q_derivative({i}) dH", dH)
        dS = _as_finite_complex(f"q_derivative({i}) dS", dS)
        n = H.shape[0]
        if dH.shape != (3, n, n) or dS.shape != (3, n, n):
            raise FrozenPitchShapeError(
                "q_derivative must return (dH, dS) each (3, n, n); got "
                f"{dH.shape} and {dS.shape} for n={n} (order x,y,z fractional)"
            )
        eps_k, vec_k = eigh(H, S)
        eps_list.append(eps_k)
        vec_list.append(vec_k)
        overlap_list.append(S)
        dH_list.append(dH)
        dS_list.append(dS)
    occ = np.asarray(occupations, dtype=float)
    if occ.shape != (nk, eps_list[0].shape[0]):
        raise FrozenPitchShapeError(
            f"occupations must be (nk, nb) = {(nk, eps_list[0].shape[0])}, got {occ.shape}"
        )
    return _build_result(
        np.stack(eps_list),
        np.stack(vec_list),
        np.stack(dH_list),
        np.stack(dS_list),
        occ,
        np.asarray(kweights, dtype=float),
        np.stack(overlap_list),  # audit the provider S: eigh gives c^dag S c = I
        tangent,
        normalize_tangent,
        fd_residual,
        check_normalization,
        norm_tol,
        hermiticity_tol,
        return_per_k,
        occupations_protocol=(
            "fixed input occupations f_nk (frozen, no re-Fermi) in "
            "scipy.linalg.eigh ascending generalized-eigenvalue order at "
            "the reference q"
        ),
    )
