"""Signed local frozen-band gradients from normalized eigenpairs (story 004).

Evaluates the atom-resolved in-plane (beta) and out-of-plane (delta)
first-rotation gradients of the frozen band energy about a planar spiral
reference, from already-computed normalized generalized eigenpairs:

$$
g_i^a = \\sum_{\\mathbf k} w_{\\mathbf k} \\sum_n f_{n\\mathbf k}\\,
c_{n\\mathbf k}^\\dagger
\\left(\\partial_{\\eta_i^a}H - \\varepsilon_{n\\mathbf k}\\,
\\partial_{\\eta_i^a}S\\right) c_{n\\mathbf k},
\\qquad a \\in \\{\\beta, \\delta\\},
$$

in eV/radian, with the sign convention that ``g_i^a`` is the derivative
of the band energy under a positive rotation of atom ``i`` along channel
``a``; its negative is the frozen-band torque component, which is NOT a
physical self-consistent total-energy torque (FR-005/FR-011): a
constraint potential, Hubbard/DC functional or fixed laboratory field
changes the stationary functional without entering this frozen sum.

Semantics fixed by the approved contract:

* occupations are the FIXED frozen values ``f_nk``; no re-Fermi step is
  ever applied (FR-003);
* the ``-eps dS`` overlap term is included only when the representation
  itself moves under the physical perturbation (rotated overlap/moving
  basis); for a fixed lab-frame or local-frame basis ``dS = 0``. A
  whole-basis unitary rotation is pure gauge and must not be passed as
  a perturbation;
* the occupied spectral density ``rho_spec = sum_k w_k f_nk c c^dag``
  is reconstructed from the very eigenpairs being rotated and compared
  against the stored same-gauge SCF density; a mismatch beyond
  ``density_tol`` raises instead of silently substituting a hard-rotated
  density (FR-003);
* provenance (schema version, gauge, q, electron count, occupation rule,
  constraint record, field role and field rotation policy) is REQUIRED
  and cross-checked; missing provenance raises rather than allowing an
  unlabeled reference (FR-004, ADR-R4).

The evaluator is agnostic to how the pencil was folded: it accepts any
primitive k mesh, so incommensurate q needs no commensurate supercell
(NFR-002).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FrozenBandGradient",
    "FrozenBandEigenpairError",
    "FrozenBandProvenanceError",
    "FrozenBandDensityMismatchError",
    "PROVENANCE_REQUIRED_KEYS",
    "frozen_band_gradient",
]

#: Required provenance keys, mirroring the strict v2 ``metadata_json``
#: (ADR-R4: field role and rotation policy are explicit, no defaults).
PROVENANCE_REQUIRED_KEYS = (
    "schema_version",
    "gauge",
    "q_frac",
    "electron_count",
    "occupation_rule",
    "constraint",
    "field_role",
    "field_rotation_policy",
)

#: Allowed explicit B_local field roles (ADR-R4).
_FIELD_ROLES = ("external", "constraint_proxy", "intrinsic_exchange")

_KWEIGHT_SUM_TOL = 1e-6
_OCCUPANCY_TOL = 1e-8


class FrozenBandEigenpairError(ValueError):
    """Eigenpair/shape/hermiticity contract violation."""


class FrozenBandProvenanceError(ValueError):
    """Missing or inconsistent reference provenance."""


class FrozenBandDensityMismatchError(ValueError):
    """Stored density is not the occupied spectral density of the pencil."""


@dataclass(frozen=True)
class FrozenBandGradient:
    """Signed atom-resolved frozen-band gradients and diagnostics.

    Attributes are eV/rad; ``g_beta``/``g_delta`` are
    ``+dE_band/d(positive local rotation)`` per atom. The ``torque_*``
    fields are their negatives: frozen-band torque COMPONENTS, never a
    self-consistent total-energy torque. ``g_*_orbital`` decompose the
    atom sums over the (spin-expanded) basis orbitals.
    """

    g_beta: np.ndarray
    g_delta: np.ndarray
    torque_beta: np.ndarray
    torque_delta: np.ndarray
    g_beta_orbital: np.ndarray
    g_delta_orbital: np.ndarray
    density_discrepancy: float
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-ready report under the ``frozen_band_gradient`` key."""
        return {
            "frozen_band_gradient": {
                "g_beta": np.asarray(self.g_beta, dtype=float).tolist(),
                "g_delta": np.asarray(self.g_delta, dtype=float).tolist(),
                "torque_beta": np.asarray(self.torque_beta, dtype=float).tolist(),
                "torque_delta": np.asarray(self.torque_delta, dtype=float).tolist(),
                "g_beta_orbital": np.asarray(self.g_beta_orbital, dtype=float).tolist(),
                "g_delta_orbital": np.asarray(
                    self.g_delta_orbital, dtype=float
                ).tolist(),
                "units": "eV/rad",
                "sign": "+dE_band/d(positive local rotation)",
                "torque_kind": (
                    "frozen-band component (negative gradient); NOT a "
                    "self-consistent total-energy torque"
                ),
                "density_discrepancy": float(self.density_discrepancy),
                "provenance": self.provenance,
            }
        }


def _check_hermitian(name, arr, tol):
    scale = max(1.0, float(np.max(np.abs(arr))) if arr.size else 1.0)
    if float(np.max(np.abs(arr - np.conj(np.swapaxes(arr, -1, -2))))) > tol * scale:
        raise FrozenBandEigenpairError(f"{name} must be Hermitian")


def _as_operator_per_k(name, value, nk, natom, n, hermiticity_tol):
    """Normalize an operator argument to a ``(nk, natom, n, n)`` array.

    Accepts ``(natom, n, n)`` (k-independent, the usual local-frame
    case) or ``(nk, natom, n, n)`` (moving basis with k-dependent
    dressing).
    """
    arr = np.asarray(value, dtype=complex)
    if arr.ndim == 3:
        if arr.shape != (natom, n, n):
            raise FrozenBandEigenpairError(
                f"{name} must have shape ({natom}, {n}, {n}) or "
                f"({nk}, {natom}, {n}, {n}), got {arr.shape}"
            )
        arr = np.broadcast_to(arr, (nk,) + arr.shape)
    elif arr.ndim == 4:
        if arr.shape != (nk, natom, n, n):
            raise FrozenBandEigenpairError(
                f"{name} must have shape ({natom}, {n}, {n}) or "
                f"({nk}, {natom}, {n}, {n}), got {arr.shape}"
            )
    else:
        raise FrozenBandEigenpairError(
            f"{name} must be 3-d (natom, n, n) or 4-d (nk, natom, n, n), "
            f"got ndim={arr.ndim}"
        )
    _check_hermitian(name, arr, hermiticity_tol)
    return arr


def _validate_provenance(provenance, nelec_occupations):
    if provenance is None:
        raise FrozenBandProvenanceError(
            "provenance is required for a frozen-band gradient; a v2 "
            "reference must carry schema_version, gauge, q_frac, "
            "electron_count, occupation_rule and constraint"
        )
    if not isinstance(provenance, dict):
        raise FrozenBandProvenanceError(
            f"provenance must be a dict, got {type(provenance).__name__}"
        )
    missing = [k for k in PROVENANCE_REQUIRED_KEYS if k not in provenance]
    if missing:
        raise FrozenBandProvenanceError(
            f"provenance is missing required keys: {missing}"
        )
    for key in ("schema_version", "gauge", "occupation_rule"):
        value = provenance[key]
        if not isinstance(value, str) or not value.strip():
            raise FrozenBandProvenanceError(
                f"provenance {key!r} must be a non-empty string"
            )
    q_frac = np.asarray(provenance["q_frac"], dtype=float)
    if q_frac.shape != (3,) or not np.all(np.isfinite(q_frac)):
        raise FrozenBandProvenanceError(
            "provenance 'q_frac' must be a length-3 finite vector"
        )
    electron_count = provenance["electron_count"]
    if isinstance(electron_count, bool) or not np.isscalar(electron_count):
        raise FrozenBandProvenanceError(
            "provenance 'electron_count' must be a real number"
        )
    electron_count = float(electron_count)
    if not np.isfinite(electron_count):
        raise FrozenBandProvenanceError("provenance 'electron_count' must be finite")
    constraint = provenance["constraint"]
    if isinstance(constraint, dict):
        kind = constraint.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise FrozenBandProvenanceError(
                "provenance 'constraint' dict must carry a non-empty 'kind'"
            )
    elif not isinstance(constraint, str) or not constraint.strip():
        raise FrozenBandProvenanceError(
            "provenance 'constraint' must be a non-empty string or a dict "
            "with an explicit 'kind' (v2 stores {'kind': 'none'} when "
            "unconstrained)"
        )
    if provenance["field_role"] not in _FIELD_ROLES:
        raise FrozenBandProvenanceError(
            "provenance 'field_role' must be one of "
            f"{list(_FIELD_ROLES)}, got {provenance['field_role']!r}"
        )
    policy = provenance["field_rotation_policy"]
    if isinstance(policy, dict):
        if not policy:
            raise FrozenBandProvenanceError(
                "provenance 'field_rotation_policy' dict must not be empty"
            )
    elif not isinstance(policy, str) or not policy.strip():
        raise FrozenBandProvenanceError(
            "provenance 'field_rotation_policy' must be a non-empty string "
            "or a non-empty dict (explicit q/angle rotation policy, no "
            "default)"
        )
    if abs(nelec_occupations - electron_count) > 1e-6:
        raise FrozenBandProvenanceError(
            "provenance electron_count "
            f"({electron_count!r}) is inconsistent with the frozen "
            f"occupations ({nelec_occupations!r})"
        )
    return dict(provenance)


def frozen_band_gradient(
    eigenvalues,
    eigenvectors,
    kweights,
    occupations,
    overlap,
    orbital_to_atom,
    v1_beta,
    v1_delta,
    ds_beta=None,
    ds_delta=None,
    *,
    rho_scf,
    provenance,
    density_tol=1e-8,
    normalization_tol=1e-8,
    hermiticity_tol=1e-10,
):
    """Signed atom-resolved frozen-band gradients (beta/delta) in eV/rad.

    Parameters
    ----------
    eigenvalues:
        ``(nk, nbnd)`` reference eigenvalues in eV.
    eigenvectors:
        ``(nk, n, nbnd)`` generalized eigenvector columns ``c``, S-
        normalized (``c^dag S c = I``); ``n`` is the full spinor basis
        size and ``nbnd <= n`` the stored band window.
    kweights:
        ``(nk,)`` mesh weights summing to one.
    occupations:
        ``(nk, nbnd)`` FIXED frozen occupations in [0, 1]; never
        re-Fermi filled.
    overlap:
        ``(nk, n, n)`` overlap matrices at the reference, or ``None``
        for an orthogonal basis.
    orbital_to_atom:
        Integer atom index per basis orbital: either the v2 spatial map
        ``(norb,)`` with ``n = 2 norb`` (expanded over the interleaved
        spinor basis by repeating each entry for up/down) or the
        already spin-expanded map ``(n,)``.
    v1_beta, v1_delta:
        ``(natom, n, n)`` (or ``(nk, natom, n, n)``) Hermitian first-
        rotation operators ``dH/d eta_i^a`` at the reference. In a
        fixed frame these are on-site field derivatives only (planar
        local basis: ``Bf sx`` / ``Bf sy``); whole-basis unitary
        rotations are gauge and must not be supplied.
    ds_beta, ds_delta:
        Matching overlap derivatives ``dS/d eta_i^a`` with the same
        shapes, or ``None`` when the basis does not move under the
        physical perturbation.
    rho_scf:
        ``(n, n)`` stored SCF density in the SAME gauge/frame as the
        eigenpairs. Required: the spectral density reconstructed from
        these eigenpairs is compared against it and a mismatch beyond
        ``density_tol`` raises :class:`FrozenBandDensityMismatchError`
        instead of silently rotating the projector.
    provenance:
        Dict with :data:`PROVENANCE_REQUIRED_KEYS` (schema_version,
        gauge, q_frac, electron_count, occupation_rule, constraint).
        Missing provenance raises :class:`FrozenBandProvenanceError`;
        ``electron_count`` is cross-checked against the occupations.
    density_tol:
        Frobenius-norm tolerance of the spectral/SCF density comparison.
    normalization_tol, hermiticity_tol:
        Eigenpair S-normalization and operator Hermiticity tolerances.

    Returns
    -------
    FrozenBandGradient
        ``g_beta``, ``g_delta`` (natom,) signed eV/rad atom sums with
        their negatives as frozen-band torque components, per-orbital
        decompositions, the density discrepancy and the provenance
        echo.
    """
    eps = np.asarray(eigenvalues, dtype=float)
    if eps.ndim != 2 or not np.all(np.isfinite(eps)):
        raise FrozenBandEigenpairError(
            f"eigenvalues must be finite (nk, nbnd), got shape {eps.shape}"
        )
    nk, nbnd = eps.shape

    c = np.asarray(eigenvectors, dtype=complex)
    if c.ndim != 3 or c.shape[0] != nk or c.shape[2] != nbnd:
        raise FrozenBandEigenpairError(
            f"eigenvectors must have shape ({nk}, n, {nbnd}), got {c.shape}"
        )
    n = c.shape[1]
    if nbnd > n:
        raise FrozenBandEigenpairError(f"nbnd={nbnd} exceeds basis size n={n}")

    f = np.asarray(occupations, dtype=float)
    if f.shape != (nk, nbnd):
        raise FrozenBandEigenpairError(
            f"occupations must have shape ({nk}, {nbnd}), got {f.shape}"
        )
    if f.min() < -_OCCUPANCY_TOL or f.max() > 1.0 + _OCCUPANCY_TOL:
        raise FrozenBandEigenpairError(
            "occupations must lie in [0, 1], got range " f"[{f.min()!r}, {f.max()!r}]"
        )

    wk = np.asarray(kweights, dtype=float)
    if wk.shape != (nk,):
        raise FrozenBandEigenpairError(
            f"kweights must have shape ({nk},), got {wk.shape}"
        )
    if not np.all(np.isfinite(wk)) or abs(float(wk.sum()) - 1.0) > _KWEIGHT_SUM_TOL:
        raise FrozenBandEigenpairError(
            f"kweights must sum to 1 (got {wk.sum()!r}) and be finite"
        )

    if overlap is None:
        S = np.broadcast_to(np.eye(n, dtype=complex), (nk, n, n))
    else:
        S = np.asarray(overlap, dtype=complex)
        if S.shape != (nk, n, n):
            raise FrozenBandEigenpairError(
                f"overlap must have shape ({nk}, {n}, {n}) or be None, got {S.shape}"
            )
        _check_hermitian("overlap", S, hermiticity_tol)

    # normalized generalized eigenpairs: c^dag S c = I
    eye_b = np.eye(nbnd, dtype=complex)
    for ik in range(nk):
        norm = c[ik].conj().T @ S[ik] @ c[ik]
        residual = float(np.max(np.abs(norm - eye_b)))
        if residual > normalization_tol:
            raise FrozenBandEigenpairError(
                "eigenvectors are not S-normalized: max|c^dag S c - I| = "
                f"{residual:.3e} at k index {ik} exceeds {normalization_tol:.1e}"
            )

    owner = np.asarray(orbital_to_atom)
    if owner.ndim != 1:
        raise FrozenBandEigenpairError("orbital_to_atom must be a 1-d integer array")
    if owner.dtype.kind == "f":
        if not np.all(owner == np.round(owner)):
            raise FrozenBandEigenpairError("orbital_to_atom must contain integers")
        owner = owner.astype(int)
    elif owner.dtype.kind not in "iu":
        raise FrozenBandEigenpairError(
            f"orbital_to_atom must have integer dtype, got {owner.dtype}"
        )
    if owner.shape[0] == n:
        owner = owner.astype(int)
    elif n % 2 == 0 and owner.shape[0] == n // 2:
        # v2 spatial (norb,) map expanded over the interleaved spinor basis
        owner = np.repeat(owner.astype(int), 2)
    else:
        raise FrozenBandEigenpairError(
            f"orbital_to_atom must have length {n} (spin-expanded) or "
            f"{n // 2} (spatial, interleaved spin), got {owner.shape[0]}"
        )
    if owner.min() < 0:
        raise FrozenBandEigenpairError("orbital_to_atom indices must be >= 0")
    natom = int(owner.max()) + 1
    present = np.unique(owner)
    if not np.array_equal(present, np.arange(natom)):
        raise FrozenBandEigenpairError(
            "orbital_to_atom must label atoms contiguously from 0; got "
            f"{present.tolist()} for natom={natom}"
        )

    v1_beta = _as_operator_per_k("v1_beta", v1_beta, nk, natom, n, hermiticity_tol)
    v1_delta = _as_operator_per_k("v1_delta", v1_delta, nk, natom, n, hermiticity_tol)
    ds = {}
    for channel, value in (("beta", ds_beta), ("delta", ds_delta)):
        if value is None:
            ds[channel] = None
        else:
            ds[channel] = _as_operator_per_k(
                f"ds_{channel}", value, nk, natom, n, hermiticity_tol
            )

    rho_scf_arr = np.asarray(rho_scf, dtype=complex)
    if rho_scf_arr.shape != (n, n):
        raise FrozenBandEigenpairError(
            f"rho_scf must have shape ({n}, {n}), got {rho_scf_arr.shape}"
        )

    nelec_occupations = float(np.sum(f * wk[:, None]))
    provenance_echo = _validate_provenance(provenance, nelec_occupations)

    # occupied spectral density of the very pencil being rotated
    rho_spec = np.zeros((n, n), dtype=complex)
    for ik in range(nk):
        rho_spec += wk[ik] * (c[ik] * f[ik][None, :]) @ c[ik].conj().T
    discrepancy = float(np.linalg.norm(rho_spec - rho_scf_arr, ord="fro"))
    if discrepancy > density_tol:
        raise FrozenBandDensityMismatchError(
            "stored rho_scf is not the occupied spectral density of the "
            f"frozen pencil: ||rho_spec - rho_scf||_F = {discrepancy:.3e} "
            f"exceeds density_tol={density_tol:.1e}; refusing to substitute "
            "a hard-rotated density"
        )

    # gradients: g_i^a = sum_k w_k Tr[P_k (dH_i^a - P~_k-weighted dS_i^a)]
    # with P_k = c f c^dag and P~_k = c (f*eps) c^dag — the FULL trace,
    # i.e. the frozen-occupation band-energy derivative (PRD formula).
    # The orbital decomposition carries the per-basis-orbital diagonal of
    # the same products summed over atoms; for block-diagonal local
    # operators (on-site field rotations, the production case) the
    # owner-binned orbital sums reproduce the atom gradients exactly.
    g_orbital = {"beta": np.zeros(n), "delta": np.zeros(n)}
    g_atom = {"beta": np.zeros(natom), "delta": np.zeros(natom)}
    channel_ops = {"beta": v1_beta, "delta": v1_delta}
    for channel in ("beta", "delta"):
        v1 = channel_ops[channel]
        dsk = ds[channel]
        for ia in range(natom):
            vals = np.zeros((nk, n), dtype=complex)
            for ik in range(nk):
                ck = c[ik]
                contrib = ((ck * f[ik][None, :]) @ ck.conj().T) @ v1[ik, ia]
                if dsk is not None:
                    p_eps = (ck * (f[ik] * eps[ik])[None, :]) @ ck.conj().T
                    contrib = contrib - p_eps @ dsk[ik, ia]
                vals[ik] = np.einsum("ii->i", contrib)
            block = np.sum(wk[:, None] * vals.real, axis=0)
            g_atom[channel][ia] = float(block.sum())
            g_orbital[channel] += block

    g_beta = g_atom["beta"]
    g_delta = g_atom["delta"]

    return FrozenBandGradient(
        g_beta=g_beta,
        g_delta=g_delta,
        torque_beta=-g_beta,
        torque_delta=-g_delta,
        g_beta_orbital=g_orbital["beta"],
        g_delta_orbital=g_orbital["delta"],
        density_discrepancy=discrepancy,
        provenance=provenance_echo,
    )
