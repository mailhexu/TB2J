"""Bosonic mode solver with metric normalization and stability diagnostics.

The solver consumes the assembled Hermitian bosonic Hamiltonian ``H`` (the
legacy ``Magnon.Hq`` doubled form ``[[A1-C, B], [B^dag, A2-C]]``) and returns
physical mode energies, metric-normalized mode vectors when a complete
normalized set exists, and the diagnostics required by the validity policy
(architecture section 4). It never supplies a regularized mode set silently:
any necessary diagonal shift is refused, never applied.

Certified Colpa identities (``docs/sympy/magnon_topology_lswt_ferromagnet.py``):
with ``H = L L^dag`` (lower Cholesky), ``M = L^dag g L`` eigendecomposition
``(omega_all, u)`` has the physical energies in the ascending top half
``omega = omega_all[n:]`` (the legacy ``Magnon._magnon_energies`` convention,
without its silent shifts); ``v = L^{-dag} u`` obeys ``H v = omega g v`` with
``v^dag g v = 1/omega``; ``sqrt(omega)`` rescaling yields unit metric norm.
For a numerically singular Hessian (Goldstone or accidental touching) the
branch never solves through the singular factor. For the standard bosonic
doubled form ``H = [[A, B], [B^dag, A]]`` with Hermitian ``B`` and positive
semidefinite ``A - B``, the spectrum is certified by the parity reduction
``lambda = eig((A - B)(A + B))`` with modes ``+-sqrt(lambda)``: the product is
Hermitian-similar and backward stable, so strongly squeezed small modes stay
resolved where the non-normal pencil (condition number ``~||H||/omega``)
loses digits, and exact zeros are ``lambda <= zero_tol^2``. Otherwise the
full-size pencil ``gH v = omega v`` is solved directly, with above-cut modes
classified by their metric spectral weight ``omega*|v^dag g v| = |v^dag H v|``:
a g-degenerate Jordan chain carries it at the rounding level of ``||H||``
(Jordan dust — classified as zeros, never emitted as invented positive
vectors; squeezed genuine modes carry it well above the floor), every discard
is cross-checked against the independent Hermitian Colpa M spectrum
(inconclusive -> refusal). In both cases the mode vectors come from the
pencil at the resolved frequencies, are Colpa-normalized, g-orthogonalized
within degenerate subspaces, Gram-validated and enforced to satisfy the
generalized eigenproblem before emission. Null modes come from the Hessian
null space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "BosonicModes",
    "ModeSolverError",
    "solve_bosonic_modes",
]


class ModeSolverError(ValueError):
    """Raised when the mode solution is physically unusable."""


@dataclass
class BosonicModes:
    """Result of the bosonic mode solve for one momentum."""

    energies: np.ndarray  # physical mode energies, ascending (n,)
    classification: str  # positive_definite | zero_mode | energetically_unstable | dynamically_unstable
    vectors: Optional[np.ndarray] = None  # (2n, n_pos) metric-normalized positive modes
    positive_indices: Optional[np.ndarray] = (
        None  # indices of the positive modes within ``energies``
    )
    null_space: Optional[np.ndarray] = None  # (2n, n_zero) non-normalizable null modes
    h_min_eigenvalue: float = 0.0  # min eigenvalue of the Hessian H itself
    min_energy: float = 0.0  # min physical energy (top-half metric eigenvalue)
    residuals: dict = field(default_factory=dict)


def solve_bosonic_modes(
    H: np.ndarray,
    hermiticity_tol: float = 1e-10,
    zero_tol: float = 1e-8,
) -> BosonicModes:
    """Solve the bosonic eigenproblem ``H v = omega g v`` for Hermitian ``H``.

    Parameters
    ----------
    H : (2n, 2n) complex Hermitian array in the legacy ``Magnon.Hq`` form.
    hermiticity_tol : absolute tolerance on ``max|H - H^dag|`` and on the
        +-pairing residual of the metric spectrum.
    zero_tol : absolute energy tolerance separating a stable zero mode from
        resolved positive energies (eV).

    Returns
    -------
    BosonicModes with ``classification``:
      - ``positive_definite``: stable gapped set; complete metric-normalized
        positive modes attached.
      - ``zero_mode``: a stable zero-frequency direction exists (Goldstone
        or accidental touching); ``null_space`` attached and no normalized
        mode vectors (null modes are not normalizable bosons). Ticket 02
        refuses validated invariants on such meshes.
      - ``energetically_unstable``: the Hessian has a resolved negative
        direction; no diagonal shift is applied and no vectors are provided.
      - ``dynamically_unstable``: the metric spectrum violates the +-pairing
        structure expected of a bosonic BdG set; no vectors are provided.
    """
    H = np.asarray(H, dtype=complex)
    n2 = H.shape[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1] or n2 % 2:
        raise ModeSolverError(
            f"Hamiltonian must be square with even size, got shape {H.shape}."
        )
    asym = float(np.max(np.abs(H - H.conj().T)))
    if asym > hermiticity_tol:
        raise ModeSolverError(
            f"Hamiltonian is not Hermitian within tolerance: max|H-H^dag| = {asym:.3e}."
        )
    n = n2 // 2
    w_all, U_all = np.linalg.eigh(H)
    h_min = float(w_all[0])
    g = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))

    if h_min < -zero_tol:
        # frequency character of the dynamical matrix gH (diagnostics only;
        # no diagonal shift is applied and no vectors are provided)
        g_diag = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))
        freq = np.linalg.eigvals(g_diag @ H)
        scale = max(float(np.max(np.abs(freq))), 1.0)
        freq_character = (
            "real"
            if float(np.max(np.abs(freq.imag))) <= hermiticity_tol * scale
            else "complex"
        )
        return BosonicModes(
            energies=np.array([]),
            classification="energetically_unstable",
            h_min_eigenvalue=h_min,
            residuals={
                "hermiticity_asymmetry": asym,
                "frequency_character": freq_character,
                "max_imag_frequency": float(np.max(np.abs(freq.imag))),
                "note": "Hessian has a resolved negative direction; a diagonal "
                "shift is refused because it would not legitimize topology.",
            },
        )
    null_h = np.abs(w_all) <= zero_tol
    w_clamped = np.where(null_h, 0.0, w_all)
    if null_h.any():
        # Numerically singular Hessian (Goldstone or accidental touching):
        # classification must not depend on a Cholesky pivot coin flip.
        # Solve the full-size pencil gH v = omega v directly — the spectrum
        # is exact (Jordan zeros stay zeros instead of becoming
        # sqrt-of-rounding dust pairs inside an intermediate factorization).
        #
        # Jordan-dust discriminator: a g-degenerate Jordan chain carries
        # vanishing spectral weight omega*|v^dag g v| = |v^dag H v| at the
        # rounding level of ||H||, while a genuine (possibly squeezed)
        # positive mode carries it well above that floor. The absolute
        # metric norm alone is NOT a sound test (squeezed modes can have
        # arbitrarily small Euclidean-normalized metric norm), so every
        # discard is cross-checked against the independent Hermitian Colpa
        # M spectrum below: a count mismatch that M does not explain as
        # sqrt-rounding dust is refused as inconclusive instead of
        # inventing zeros or modes.
        psi = U_all[:, null_h]
        gh = g @ H
        mu_all, vec_all = np.linalg.eig(gh)
        scale_gh = float(np.max(np.abs(mu_all), initial=0.0))
        tol_cplx = hermiticity_tol * max(1.0, scale_gh)
        h_scale = float(np.max(np.abs(H), initial=0.0))
        dust_floor = 100.0 * float(np.finfo(float).eps) * max(1.0, h_scale)
        eye_n2 = np.eye(n2)
        dust_n = 0
        dust_omegas = []  # |omega| of discarded candidates
        # Certified parity reduction for the standard bosonic doubled form
        # H = [[A, B], [B^dag, A]] with B Hermitian and A - B positive
        # semidefinite: the pencil spectrum is exactly {0, +-sqrt(lambda)}
        # with lambda = eig((A - B)(A + B)). The product is Hermitian-similar
        # and backward stable, so a strongly squeezed small mode stays
        # resolved where the non-normal pencil (condition number ~||H||/omega)
        # loses digits, and no heuristic dust discriminator or M witness is
        # needed. Fall back to the pencil route when the form or positivity
        # does not hold.
        A_blk = H[:n, :n]
        B_blk = H[:n, n:]
        lam_ref = None
        if (
            np.array_equal(H, H.conj().T)
            and np.array_equal(H[n:, n:], A_blk)
            and np.array_equal(B_blk, B_blk.conj().T)
        ):
            amb = A_blk - B_blk
            if float(np.linalg.eigvalsh((amb + amb.conj().T) * 0.5)[0]) >= -zero_tol:
                lam = np.linalg.eigvals((A_blk - B_blk) @ (A_blk + B_blk))
                scale_l = max(1.0, float(np.max(np.abs(lam), initial=0.0)))
                if (
                    float(np.max(np.abs(lam.imag), initial=0.0))
                    <= hermiticity_tol * scale_l
                ):
                    lam_ref = np.sort(np.real(lam))
        if lam_ref is not None:
            # exact zeros are lambda <= zero_tol^2; a lambda resolved
            # negative below the rounding floor of the product spectrum is a
            # complex-frequency (dynamically unstable) pair
            neg_floor = 100.0 * float(np.finfo(float).eps) * max(1.0, scale_l)
            lam_pos = lam_ref[lam_ref > zero_tol**2]
            if float(lam_ref[0]) < -neg_floor:
                return BosonicModes(
                    energies=np.concatenate(
                        [np.zeros(max(0, n - lam_pos.size)), np.sqrt(lam_pos)]
                    ),
                    classification="dynamically_unstable",
                    h_min_eigenvalue=h_min,
                    min_energy=0.0,
                    residuals={
                        "hermiticity_asymmetry": asym,
                        "pairing_residual": float(np.sqrt(-lam_ref[0])),
                    },
                )
            genuine = []
            dust_n = 2 * int(np.sum(lam_ref <= zero_tol**2))
            pairing_red = 0.0
            i = 0
            while i < lam_pos.size:
                j0 = i
                while i + 1 < lam_pos.size and abs(
                    lam_pos[i + 1] - lam_pos[j0]
                ) <= 1e-10 * max(1.0, abs(lam_pos[j0])):
                    i += 1
                mult = i - j0 + 1
                om_c = float(np.sqrt(lam_pos[j0]))
                _, _, vt = np.linalg.svd(gh - om_c * eye_n2)
                null_basis = vt[-mult:].conj().T
                for k in range(mult):
                    v_k = null_basis[:, k]
                    denom = float(np.real(v_k.conj() @ (g @ v_k)))
                    if denom <= 0.0:
                        raise ModeSolverError(
                            "Certified singular-Hessian spectrum produced a "
                            "non-positive-metric positive mode (inconclusive); "
                            "refusing to emit an unvalidated mode vector."
                        )
                    genuine.append((om_c, v_k))
                i += 1
        else:
            cands = []  # (omega, LAPACK index) above-cut non-dust candidates
            for j in range(mu_all.size):
                om = mu_all[j]
                if abs(om) <= zero_tol:
                    continue
                met = float(np.real(vec_all[:, j].conj() @ (g @ vec_all[:, j])))
                if abs(om) * abs(met) <= dust_floor:
                    dust_n += 1
                    dust_omegas.append(float(abs(om)))
                    continue
                if abs(om.imag) > tol_cplx:
                    pos_real = np.sort(
                        np.array([float(m.real) for m in mu_all if m.real > zero_tol])
                    )
                    return BosonicModes(
                        energies=np.concatenate(
                            [np.zeros(max(0, n - pos_real.size)), pos_real]
                        ),
                        classification="dynamically_unstable",
                        h_min_eigenvalue=h_min,
                        min_energy=0.0,
                        residuals={
                            "hermiticity_asymmetry": asym,
                            "pairing_residual": float(abs(om.imag)),
                        },
                    )
                cands.append((float(om.real), j))
            # Refine every above-cut candidate: LAPACK pencil eigenvectors carry
            # eps*cond(pencil) direction error, which destroys the
            # g-orthogonality between distinct frequencies for ill-conditioned
            # Hessians. A Rayleigh + singular-subspace pass gives backward-stable
            # eigenpairs (residual ~ eps*||H||), restoring the Colpa structure to
            # rounding level; equal-frequency clusters use the full
            # m-dimensional near-nullspace so degenerate modes stay distinct.
            cands.sort(key=lambda t: t[0])
            genuine = []  # (omega, refined eigenvector) positive genuine modes
            others = []  # (omega, refined eigenvector) negative-metric partners
            refusal_val = 0.0
            refusal = None
            i = 0
            while i < len(cands):
                j0 = i
                while i + 1 < len(cands) and abs(
                    cands[i + 1][0] - cands[j0][0]
                ) <= 1e-10 * max(1.0, abs(cands[j0][0])):
                    i += 1
                mult = i - j0 + 1
                om0 = float(np.mean([cands[k][0] for k in range(j0, i + 1)]))
                _, _, vt = np.linalg.svd(gh - om0 * eye_n2)
                null_basis = vt[-mult:].conj().T  # (n2, mult) near-nullspace
                for k in range(mult):
                    v_k = null_basis[:, k]
                    denom = float(np.real(v_k.conj() @ (g @ v_k)))
                    if abs(denom) <= 1e-12:
                        refusal = "g-neutral frequency"
                        refusal_val = 1.0
                        continue
                    om_k = float(np.real(v_k.conj() @ (H @ v_k)) / denom)
                    if (om_k > 0.0) != (denom > 0.0):
                        refusal = "metric-sign mismatch"
                        refusal_val = float(max(abs(om_k), 1.0))
                    elif om_k > 0.0:
                        genuine.append((om_k, v_k))
                    else:
                        others.append((om_k, v_k))
                i += 1
            if refusal is not None:
                pos_tmp = np.sort(np.array([om for om, _ in genuine]))
                return BosonicModes(
                    energies=np.concatenate([np.zeros(n - pos_tmp.size), pos_tmp]),
                    classification="dynamically_unstable",
                    h_min_eigenvalue=h_min,
                    min_energy=0.0,
                    residuals={
                        "hermiticity_asymmetry": asym,
                        "pairing_residual": refusal_val,
                    },
                )
            genuine.sort(key=lambda t: t[0])
            om_pos = np.array([om for om, _ in genuine])
            om_neg = np.sort(np.array([om for om, _ in others]))
            if len(others) == len(om_pos):
                pairing_red = float(np.max(np.abs(-om_neg - om_pos[::-1]), initial=0.0))
            else:
                pairing_red = float(np.max(np.abs(om_pos), initial=0.0)) + 1.0
            if pairing_red > tol_cplx:
                return BosonicModes(
                    energies=np.concatenate([np.zeros(n - om_pos.size), om_pos]),
                    classification="dynamically_unstable",
                    h_min_eigenvalue=h_min,
                    min_energy=0.0,
                    residuals={
                        "hermiticity_asymmetry": asym,
                        "pairing_residual": pairing_red,
                    },
                )
            # Two-witness rule: the metric discriminator may only discard
            # candidates that the independent Hermitian Colpa M spectrum
            # explains as Jordan dust — M inflates exact zeros into
            # +-sqrt(rounding) pairs, so a discard is accepted only when every
            # discarded candidate lies below the sqrt-rounding plausibility
            # bound. Any other count mismatch is inconclusive and refused
            # instead of inventing zeros or modes.
            L = (U_all * np.sqrt(w_clamped)[None, :]) @ U_all.conj().T
            if np.max(np.abs(L @ L.conj().T - H)) > zero_tol:
                raise ModeSolverError(
                    f"Hessian not decomposable within tolerance (min {h_min:.3e})."
                ) from None
            M = L.conj().T @ g @ L
            m_pos = int(np.sum(np.linalg.eigvalsh(M)[n:] > zero_tol))
            jordan_bound = (
                1e3 * float(np.sqrt(np.finfo(float).eps)) * max(1.0, scale_gh)
            )
            if len(genuine) != m_pos:
                if (
                    len(genuine) < m_pos
                    and dust_n
                    and all(w <= jordan_bound for w in dust_omegas)
                ):
                    pass  # every discard is sqrt-rounding Jordan dust in M too
                else:
                    raise ModeSolverError(
                        "Inconclusive singular-Hessian classification: the "
                        f"metric discriminator resolved {len(genuine)} positive "
                        f"modes while the independent Colpa metric spectrum "
                        f"reports {m_pos}; refusing rather than inventing zeros "
                        "or modes."
                    )  # Colpa vectors, sqrt(omega)-rescaled to unit metric norm (the legacy
        # emission convention: v^dag g v = 1, so v/sqrt(omega) carries metric
        # norm 1/omega), g-orthogonalized within degenerate (equal-frequency)
        # subspaces; every emitted vector is validated against the generalized
        # eigenproblem and the full metric Gram matrix.
        cols = []
        gen_resid = 0.0
        met_err = 0.0
        group = []  # (omega, Colpa vector with v^dag g v = 1/omega)
        group_om = None

        def _flush_group(group, group_om):
            """g-orthogonalize one equal-frequency subspace and enforce."""
            out = []
            basis = []  # unit-metric basis vectors (v^dag g v = 1)
            for om_g, v_colpa in group:
                u = v_colpa
                for b in basis:
                    u = u - (float(np.real(b.conj() @ (g @ u)))) * b
                met_g = float(np.real(u.conj() @ (g @ u)))
                if met_g <= 1e-8 / max(om_g, 1e-300):
                    raise ModeSolverError(
                        "Degenerate positive-mode subspace is metric-"
                        "degenerate; no Colpa-normalizable basis exists. "
                        "Refused with documented diagnostics."
                    )
                u = u / np.sqrt(met_g)  # emitted unit metric norm: v^dag g v = 1
                resid_raw = float(np.linalg.norm(H @ u - om_g * (g @ u))) / max(
                    1e-300,
                    float(np.linalg.norm(H)) * float(np.linalg.norm(u)),
                )
                if resid_raw > 1e-8:
                    raise ModeSolverError(
                        "Singular-Hessian mode vector failed the generalized "
                        f"eigenproblem validation (relative residual "
                        f"{resid_raw:.2e} > 1e-8); no validated mode vector "
                        "is emitted."
                    )
                nonlocal met_err, gen_resid
                met_err = max(
                    met_err,
                    abs(float(np.real(u.conj() @ (g @ u))) - 1.0),
                )
                gen_resid = max(gen_resid, resid_raw)
                basis.append(u)
                out.append(u)
            return out

        for om, v_raw in genuine:
            metric_raw = float(np.real(v_raw.conj() @ (g @ v_raw)))
            v_colpa = v_raw / np.sqrt(metric_raw * om)
            if group_om is None or abs(om - group_om) > 1e-10 * max(1.0, om):
                if group:
                    flushed = _flush_group(group, group_om)
                    cols.extend(flushed)
                group = [(om, v_colpa)]
                group_om = om
            else:
                group.append((om, v_colpa))
        if group:
            flushed = _flush_group(group, group_om)
            cols.extend(flushed)
        p = len(cols)
        energies_red = np.concatenate(
            [np.zeros(n - p), np.sort(np.array([om for om, _ in genuine]))]
        )
        pos_mask_red = np.zeros(n, dtype=bool)
        pos_mask_red[n - p :] = True
        v = np.stack(cols, axis=1) if cols else np.empty((n2, 0))
        if p > 1:
            # full metric Gram validation on the emitted unit-metric set
            # (equal-frequency members were g-orthogonalized above; distinct
            # frequencies are g-orthogonal by the Colpa identity)
            gram = v.conj().T @ (g @ v)
            gram_dev = float(np.max(np.abs(gram - np.eye(p))))
            if gram_dev > 1e-8:
                raise ModeSolverError(
                    "Emitted positive-mode vectors failed the metric Gram "
                    f"orthogonality validation (max |V^dag g V - I| = "
                    f"{gram_dev:.2e} > 1e-8); no validated mode vectors are "
                    "emitted."
                )
        return BosonicModes(
            energies=energies_red,
            classification="zero_mode" if psi.shape[1] else "positive_definite",
            vectors=v if p > 0 else None,
            positive_indices=np.flatnonzero(pos_mask_red),
            null_space=psi if psi.shape[1] else None,
            h_min_eigenvalue=h_min,
            min_energy=float(energies_red[0]),
            residuals={
                "hermiticity_asymmetry": asym,
                "pairing_residual": pairing_red,
                "max_generalized_residual": gen_resid,
                "max_metric_norm_error": met_err,
                "null_space_residual": (float(np.max(np.abs(H @ psi), initial=0.0))),
                "n_zero_modes": n - p,
                "n_jordan_dust_modes": dust_n,
            },
        )
    try:
        L = np.linalg.cholesky(H)  # lower, H = L L^dag (strictly pd here)
    except np.linalg.LinAlgError:
        raise ModeSolverError(
            f"Hessian not decomposable within tolerance (min {h_min:.3e})."
        ) from None
    M = L.conj().T @ g @ L
    omega_all, u = np.linalg.eigh(M)
    omega = omega_all[n:]  # physical top half (legacy [:, n:] convention)
    pairing_residual = float(np.max(np.abs(omega_all[:n] + omega_all[n:][::-1])))
    scale = max(float(np.max(np.abs(omega_all))), 1.0)
    if pairing_residual > hermiticity_tol * scale:
        return BosonicModes(
            energies=omega,
            classification="dynamically_unstable",
            h_min_eigenvalue=h_min,
            min_energy=float(omega[0]),
            residuals={
                "hermiticity_asymmetry": asym,
                "pairing_residual": pairing_residual,
            },
        )

    pos_mask = omega > zero_tol
    zero_mask = ~pos_mask
    null_resid = 0.0
    null_space = None
    if np.any(zero_mask):
        null_space = np.linalg.solve(L.conj().T, u[:, n:][:, zero_mask])
        null_resid = float(np.max(np.abs(H @ null_space))) if null_space.size else 0.0
    if not np.any(pos_mask):
        return BosonicModes(
            energies=omega,
            classification="zero_mode",
            h_min_eigenvalue=h_min,
            min_energy=float(omega[0]),
            residuals={
                "hermiticity_asymmetry": asym,
                "pairing_residual": pairing_residual,
                "null_space_residual": null_resid,
                "n_zero_modes": int(np.sum(zero_mask)),
            },
            null_space=null_space,
        )
    pos_u = u[:, n:][:, pos_mask]
    pos = omega[pos_mask]
    v = np.linalg.solve(L.conj().T, pos_u)  # v = L^{-dag} u for positive modes
    norm = np.real(np.sum(v.conj() * (g @ v), axis=0))
    if np.any(norm <= 0):
        raise ModeSolverError("Nonpositive metric norm in positive subspace.")
    vectors = v * np.sqrt(pos)[None, :]
    resid = H @ vectors - (g @ vectors) * pos[None, :]
    return BosonicModes(
        energies=omega,
        classification="positive_definite",
        vectors=vectors,
        positive_indices=np.flatnonzero(pos_mask),
        null_space=None,
        h_min_eigenvalue=h_min,
        min_energy=float(omega[0]),
        residuals={
            "hermiticity_asymmetry": asym,
            "pairing_residual": pairing_residual,
            "max_generalized_residual": float(np.max(np.abs(resid))),
            "max_metric_norm_error": float(np.max(np.abs(pos * norm - 1.0))),
            "null_space_residual": null_resid,
            "n_zero_modes": int(np.sum(zero_mask)),
        },
    )
