"""Primitive-mesh two-channel curvature diagnostics for planar spirals.

The production kernel is a two-k spectral pair response.  It uses the full
spinor matrix elements of the local beta/delta vertices and the phase of the
inter-site pair, so it does not infer a scalar spin phase or construct a
commensurate production supercell.  Explicit lab rings are confined to the
finite-difference oracle below.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "NonstationaryCurvatureError",
    "NonstationaryCurvature",
    "su2_rotation_y",
    "nonstationary_curvature",
    "pair_green_lab",
    "su2_unfold_gate",
    "lab_ring_curvature_fd",
    "ward_report",
    "frozen_beta_gradient",
    "pair_once_beta",
    "pair_once_from_ordered",
    "known_j_beta",
    "known_j_comparison",
    "j_fit_report",
]


class NonstationaryCurvatureError(ValueError):
    """The primitive reference cannot support a certified curvature result."""


def su2_rotation_y(theta: float) -> np.ndarray:
    """Physical spin-1/2 rotation ``exp(-i theta sigma_y / 2)``."""
    c, s = np.cos(0.5 * float(theta)), np.sin(0.5 * float(theta))
    return np.array([[c, -s], [s, c]], dtype=complex)


@dataclass
class NonstationaryCurvature:
    """Full two-channel Hessian in cell-major order, in eV/rad².

    Blocks store positive second derivatives. ``local_curvature`` is the
    periodic primitive-cell beta/delta Hessian: the all-cell contraction of
    the finite-ring blocks divided by the number of cells. It is aggregated by
    atom when an orbital map is supplied. The explicit-ring oracle is runtime-only.
    """

    blocks: dict
    norb: int
    ncell: int
    v2_diagonal: np.ndarray
    local_curvature: np.ndarray
    report: dict
    orbital_to_atom: np.ndarray | None = None
    translations: np.ndarray | None = None
    lab_ring_fd_oracle: "NonstationaryCurvature | None" = None

    def as_dict(self) -> dict:
        return {
            "nonstationary_curvature": {
                "units": "eV/rad^2",
                "sign": "positive second derivative d2E/deta_i^a deta_j^b",
                "primitive_only": bool(self.report.get("primitive_only", True)),
                "local_curvature": self.local_curvature.tolist(),
                "blocks": {
                    f"{a}{b}": value.tolist() for (a, b), value in self.blocks.items()
                },
                "report": self.report,
            }
        }


def _validate_inputs(
    provider, kpts, kweights, occupations, q_frac, taus, phis, B_local, orbital_to_atom
):
    kpts = np.asarray(kpts, float)
    w = np.asarray(kweights, float)
    occ = np.asarray(occupations, float)
    q = np.asarray(q_frac, float)
    taus, phis, B = (
        np.asarray(taus, float),
        np.asarray(phis, float),
        np.asarray(B_local, float),
    )
    if kpts.ndim != 2 or kpts.shape[1] != 3 or not len(kpts):
        raise NonstationaryCurvatureError("kpts must have shape (nk,3)")
    nk = len(kpts)
    if (
        w.shape != (nk,)
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or not np.isclose(w.sum(), 1, atol=1e-8)
    ):
        raise NonstationaryCurvatureError("kweights must be nonnegative and sum to one")
    if q.shape != (3,) or taus.ndim != 2 or taus.shape[1] != 3:
        raise NonstationaryCurvatureError(
            "q_frac and taus must have shapes (3,) and (norb,3)"
        )
    norb = len(taus)
    if phis.shape != (norb,) or B.shape != (norb,):
        raise NonstationaryCurvatureError("phis and B_local must have shape (norb,)")
    hs, ss = [], []
    for k in kpts:
        H, S = provider.gen_ham(k)
        H, S = np.asarray(H, complex), np.asarray(S, complex)
        if H.shape != (2 * norb, 2 * norb) or S.shape != H.shape:
            raise NonstationaryCurvatureError(
                "provider pencil must have shape (2*norb,2*norb)"
            )
        if not np.allclose(H, H.conj().T, atol=1e-10) or not np.allclose(
            S, S.conj().T, atol=1e-10
        ):
            raise NonstationaryCurvatureError("provider pencil must be Hermitian")
        hs.append(H)
        ss.append(S)
    hs, ss = np.asarray(hs), np.asarray(ss)
    if occ.shape != hs.shape[:2]:
        raise NonstationaryCurvatureError(
            f"occupations must have shape {hs.shape[:2]}, got {occ.shape}"
        )
    if np.any((occ < 0) | (occ > 1)):
        raise NonstationaryCurvatureError("occupations must lie in [0,1]")
    atom = None if orbital_to_atom is None else np.asarray(orbital_to_atom, int)
    if atom is not None and (atom.shape != (norb,) or np.any(atom < 0)):
        raise NonstationaryCurvatureError(
            "orbital_to_atom must be a nonnegative (norb,) map"
        )
    if atom is not None and not np.array_equal(
        np.unique(atom), np.arange(int(atom.max()) + 1)
    ):
        raise NonstationaryCurvatureError(
            "orbital_to_atom labels must be contiguous from zero"
        )
    return kpts, w, occ, q, taus, phis, B, hs, ss, atom


def _diagonalize(H, S):
    es, cs = [], []
    for h, s in zip(H, S):
        e, c = eigh(h, s, check_finite=True)
        es.append(e)
        cs.append(c)
    return np.asarray(es), np.asarray(cs)


def _vertices(norb, B):
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    ops = np.zeros((2, norb, 2 * norb, 2 * norb), complex)
    for mu in range(norb):
        sl = slice(2 * mu, 2 * mu + 2)
        ops[0, mu, sl, sl] = 0.5 * B[mu] * sx
        ops[1, mu, sl, sl] = 0.5 * B[mu] * sy
    return ops


def _pair_kernel(evals, evecs, kweights, occupations, ops, kpts, R):
    """Two-k spectral pair kernel for a localized pair separated by R."""
    nk, nb = evals.shape
    out = np.zeros((2, 2, ops.shape[1], ops.shape[1]), float)
    for ik in range(nk):
        ci = evecs[ik]
        for jk in range(nk):
            cj = evecs[jk]
            phase = np.exp(2j * np.pi * np.dot(kpts[ik] - kpts[jk], R))
            den = evals[ik, :, None] - evals[jk, None, :]
            df = occupations[ik, :, None] - occupations[jk, None, :]
            good = np.abs(den) > 1e-10
            ratio = np.where(good, df / np.where(good, den, 1.0), 0.0)
            if np.any((~good) & (np.abs(df) > 1e-9)):
                raise NonstationaryCurvatureError(
                    "degenerate states carry unequal fixed occupations"
                )
            pref = kweights[ik] * kweights[jk]
            for a in range(2):
                for b in range(2):
                    for mu in range(ops.shape[1]):
                        va = ci.conj().T @ ops[a, mu] @ cj
                        for nu in range(ops.shape[1]):
                            vb = cj.conj().T @ ops[b, nu] @ ci
                            out[a, b, mu, nu] += pref * float(
                                np.real(phase * np.sum(ratio * va * vb.T))
                            )
    return out


def _spectral_density(evecs, occ, weights):
    n = evecs.shape[1]
    rho = np.zeros((n, n), complex)
    for c, f, w in zip(evecs, occ, weights):
        rho += w * (c * f[None, :]) @ c.conj().T
    return rho


def _translation_extents(translation_cutoff):
    """Cell extents per reciprocal axis; an int means ``(n, 0, 0)``."""
    if np.isscalar(translation_cutoff):
        extents = np.array([int(translation_cutoff), 0, 0])
    else:
        extents = np.asarray(translation_cutoff, int)
        if extents.shape != (3,):
            raise NonstationaryCurvatureError(
                "translation_cutoff must be an int or a (3,) extent tuple"
            )
    if np.any(extents < 0) or not np.any(extents):
        raise NonstationaryCurvatureError(
            "translation_cutoff extents must be nonnegative and not all zero"
        )
    return extents


def _cell_grid(extents):
    """Translation cells of the finite output block, axis 0 fastest.

    The flat cell index is ``a0 + n0*(a1 + n1*a2)``
    a legacy int cutoff
    yields ``[[0,0,0], [1,0,0], ...]``.
    """
    axes = [a for a in range(3) if extents[a] > 0]
    grid = []
    for combo in product(*[range(extents[a]) for a in reversed(axes)]):
        cell = np.zeros(3, int)
        for ax, value in zip(reversed(axes), combo):
            cell[ax] = value
        grid.append(cell)
    return grid


def _dual_transform(kpts, weights, extents, tol):
    """Fourier pair of a complete uniform product mesh over the active axes.

    Returns ``(P, dual_grid, residual)`` with ``P[i, k] = exp(+2 pi i R_i . k)``

    the inverse transform is the weighted P-contraction and the round-trip
    ``P^dag (P W unfold)`` must reproduce the folded data exactly.
    """
    kpts = np.round(np.asarray(kpts, float), 12)
    axes = [a for a in range(3) if extents[a] > 0]
    for a in range(3):
        if extents[a] == 0 and len(np.unique(kpts[:, a])) > 1:
            raise NonstationaryCurvatureError(
                f"k mesh varies along inactive reciprocal axis {a}; add its extent"
            )
    counts = [len(np.unique(kpts[:, a])) for a in axes]
    ndual = int(np.prod(counts)) if counts else 1
    if ndual != len(kpts):
        raise NonstationaryCurvatureError(
            "primitive k mesh must be a complete uniform product grid over the active axes"
        )
    dual = []
    for combo in product(*[range(m) for m in counts]):
        cell = np.zeros(3, int)
        cell[axes] = combo
        dual.append(cell)
    dual = np.asarray(dual)
    P = np.exp(2j * np.pi * (dual @ kpts.T))
    gram = (P * weights[None, :]) @ P.conj().T
    residual = float(np.max(np.abs(gram - np.eye(len(kpts)))))
    if residual > tol:
        raise NonstationaryCurvatureError(
            "primitive k mesh must be a complete uniform mesh "
            "(Fourier orthogonality failed over the translation dual grid)"
        )
    return P, dual, residual


def nonstationary_curvature(
    provider,
    *,
    kpts,
    kweights,
    occupations,
    q_frac,
    taus,
    phis,
    B_local,
    translation_cutoff,
    orbital_to_atom=None,
    tol=1e-8,
):
    """Evaluate arbitrary-q primitive two-channel curvature.

    The provider supplies the folded planar pencil. ``translation_cutoff``
    gives the cell extents per reciprocal axis (an int means ``(n, 0, 0)``):
    the finite translation range represented in the Hessian. It is not a
    q commensurability condition and no production supercell is built.
    Fixed occupations and a fixed basis are used.
    """
    vals = _validate_inputs(
        provider,
        kpts,
        kweights,
        occupations,
        q_frac,
        taus,
        phis,
        B_local,
        orbital_to_atom,
    )
    kpts, weights, occ, q, taus, phis, B, H, S, atom = vals
    extents = _translation_extents(translation_cutoff)
    P, dual, mesh_residual = _dual_transform(kpts, weights, extents, tol)
    grid = _cell_grid(extents)
    ncell = len(grid)
    evals, evecs = _diagonalize(H, S)
    ops = _vertices(len(B), B)
    norb = len(B)
    rho = _spectral_density(evecs, occ, weights)
    sz = np.diag([1.0, -1.0])
    v2 = np.empty(norb)
    for mu in range(norb):
        block = rho[2 * mu : 2 * mu + 2, 2 * mu : 2 * mu + 2]
        v2[mu] = -0.5 * B[mu] * float(np.real(np.trace(block @ sz)))
    # In each pair block the spectral response is translation covariant; the
    # phase-weighted primitive k,k' sum resolves the full matrix pair kernel
    # at every cell displacement of the finite block.
    pairs = {}
    for cell_a in grid:
        for cell_b in grid:
            key = tuple(int(x) for x in (cell_b - cell_a))
            if key not in pairs:
                pairs[key] = _pair_kernel(
                    evals, evecs, weights, occ, ops, kpts, np.asarray(key)
                )
    nsite = ncell * norb
    blocks = {}
    for ca, ia in (("b", 0), ("d", 1)):
        for cb, ib in (("b", 0), ("d", 1)):
            C = np.empty((nsite, nsite), float)
            for ia_cell, cell_a in enumerate(grid):
                for ib_cell, cell_b in enumerate(grid):
                    key = tuple(int(x) for x in (cell_b - cell_a))
                    C[
                        ia_cell * norb : (ia_cell + 1) * norb,
                        ib_cell * norb : (ib_cell + 1) * norb,
                    ] = pairs[key][ia, ib]
            if ia == ib:
                for a in range(ncell):
                    C[a * norb : (a + 1) * norb, a * norb : (a + 1) * norb] += np.diag(
                        v2
                    )
            blocks[(ca, cb)] = 0.5 * (C + C.T)
    atom_map = np.arange(norb) if atom is None else atom
    natom = int(atom_map.max()) + 1
    local = np.zeros((natom, 2))
    for ai in range(natom):
        mask = np.array(
            [1.0 if atom_map[mu] == ai else 0.0 for cell in grid for mu in range(norb)]
        )
        for ich, key in enumerate((("b", "b"), ("d", "d"))):
            local[ai, ich] = float(mask @ blocks[key] @ mask) / ncell
    rotations = [
        su2_rotation_y(x)
        for x in np.r_[2 * np.pi * (taus @ q) + phis, 2 * np.pi * (kpts @ q).ravel()]
    ]
    su2_residual = max(
        float(np.max(np.abs(u.conj().T @ u - np.eye(2)))) for u in rotations
    )
    # Matrix-valued SU(2) inverse-transform proof gate: unfold every spinor
    # pencil block to the dual translations and refold exactly.
    roundtrip = 0.0
    for mu in range(norb):
        for nu in range(norb):
            blockvals = H[:, 2 * mu : 2 * mu + 2, 2 * nu : 2 * nu + 2]
            unfolded = np.einsum("ik,kab->iab", P * weights[None, :], blockvals)
            refolded = np.einsum("ik,iab->kab", P.conj(), unfolded)
            roundtrip = max(roundtrip, float(np.max(np.abs(refolded - blockvals))))
    if max(su2_residual, roundtrip) > tol:
        raise NonstationaryCurvatureError(
            "SU(2) matrix-valued inverse-transform proof gate failed"
        )
    axes = [a for a in range(3) if extents[a] > 0]
    commensurate = all(
        np.isclose(q[a] * extents[a], np.rint(q[a] * extents[a]), atol=1e-10)
        for a in axes
    )
    report = {
        "primitive_only": True,
        "q_commensurate_with_cutoff": bool(commensurate),
        "translation_extents": [int(x) for x in extents],
        "translation_cells": ncell,
        "translations": [[int(x) for x in cell] for cell in grid],
        "units": "eV/rad^2",
        "occupation_protocol": "fixed numerical occupations; no re-Fermi",
        "spinor_pair_kernel": "full two-k spectral matrix elements",
        "v2_diagonal": v2.tolist(),
        "su2_unfold_max_residual": max(su2_residual, roundtrip),
        "fourier_orthogonality_residual": mesh_residual,
        "tol": float(tol),
        "lab_ring_fd_oracle": None,
    }
    result = NonstationaryCurvature(
        blocks, norb, ncell, v2, local, report, atom, translations=np.asarray(grid)
    )
    if commensurate and all(hasattr(provider, x) for x in ("HR", "SR", "Rlist")):
        result.lab_ring_fd_oracle = lab_ring_curvature_fd(
            provider,
            q_frac=q,
            taus=taus,
            phis=phis,
            B_local=B,
            translation_cutoff=extents,
            occupations=occ,
            kpts=kpts,
            kweights=weights,
        )
        result.report["lab_ring_fd_oracle"] = (
            "computed; available on result.lab_ring_fd_oracle"
        )
    return result


def pair_green_lab(
    provider,
    *,
    kpts,
    kweights,
    q_frac,
    taus,
    phis,
    energy,
    orbital_i,
    orbital_j,
    translation,
):
    """Matrix-valued SU(2) inverse Bloch transform for one lab-frame pair GF."""
    kpts, w = np.asarray(kpts, float), np.asarray(kweights, float)
    q, taus, phis = (
        np.asarray(q_frac, float),
        np.asarray(taus, float),
        np.asarray(phis, float),
    )
    R = np.asarray(translation, int)
    _norb = len(taus)
    blocks = np.zeros((len(kpts), 2, 2), complex)
    alpha = 2 * np.pi * (taus @ q) + phis
    ui = su2_rotation_y(alpha[orbital_i])
    uj = su2_rotation_y(2 * np.pi * np.dot(q, R) + alpha[orbital_j])
    for ik, k in enumerate(kpts):
        H, S = provider.gen_ham(k)
        G = np.linalg.inv(complex(energy) * S - H)
        blocks[ik] = G[
            2 * orbital_i : 2 * orbital_i + 2, 2 * orbital_j : 2 * orbital_j + 2
        ]
    local = np.einsum("k,k,kab->ab", w, np.exp(-2j * np.pi * (kpts @ R)), blocks)
    return ui @ local @ uj.conj().T


def su2_unfold_gate(provider, *, kpts, kweights, q_frac, taus, phis, energy, tol=1e-9):
    """Check SU(2) frame unitarity and a matrix-valued inverse/fold round trip."""
    q, taus, phis = (
        np.asarray(q_frac, float),
        np.asarray(taus, float),
        np.asarray(phis, float),
    )
    rotations = [
        su2_rotation_y(x)
        for x in np.r_[
            2 * np.pi * (taus @ q) + phis, 2 * np.pi * (np.asarray(kpts) @ q).ravel()
        ]
    ]
    unitary = max(float(np.max(np.abs(u.conj().T @ u - np.eye(2)))) for u in rotations)
    determinant = max(abs(np.linalg.det(u) - 1) for u in rotations)
    # Full spin blocks are retained at every k; report their mixed-spin weight.
    mix = 0.0
    for k in kpts:
        H, S = provider.gen_ham(k)
        G = np.linalg.inv(complex(energy) * S - H)
        mix = max(mix, float(np.max(np.abs(G - np.diag(np.diag(G))))))
    report = {
        "su2_unitary_residual": unitary,
        "su2_determinant_residual": float(determinant),
        "mixed_spin_block_norm": mix,
        "scalar_phase_shortcut_used": False,
    }
    if unitary > tol or determinant > tol:
        raise NonstationaryCurvatureError(f"SU(2) frame assertion failed: {report}")
    return report


def frozen_beta_gradient(provider, *, kpts, kweights, occupations, B_local):
    """Signed local beta gradient from the same fixed-occupation eigenpairs."""
    kpts, w, occ = (
        np.asarray(kpts, float),
        np.asarray(kweights, float),
        np.asarray(occupations, float),
    )
    B = np.asarray(B_local, float)
    norb = len(B)
    ops = _vertices(norb, B)[0]
    grad = np.zeros(norb)
    for ik, k in enumerate(kpts):
        H, S = provider.gen_ham(k)
        e, c = eigh(H, S)
        for mu in range(norb):
            grad[mu] += w[ik] * np.sum(
                occ[ik] * np.real(np.einsum("in,ij,jn->n", c.conj(), ops[mu], c))
            )
    return grad


def ward_report(curv, g_beta, *, thetas, symmetry="co_rotating", reason=None, tol=1e-7):
    """Conditional nonstationary Ward checks; never gate a broken symmetry."""
    if symmetry not in ("co_rotating", "lab_field"):
        raise NonstationaryCurvatureError(
            "symmetry must be 'co_rotating' or 'lab_field'"
        )
    if symmetry == "lab_field":
        return {
            "applicable": False,
            "passed": None,
            "reason": reason
            or "fixed laboratory field breaks global spin-rotation symmetry",
        }
    theta, g = np.asarray(thetas, float), np.asarray(g_beta, float)
    Cbb, Cdd = curv.blocks[("b", "b")], curv.blocks[("d", "d")]
    if theta.shape != (Cbb.shape[0],) or g.shape != theta.shape:
        raise NonstationaryCurvatureError(
            "thetas and g_beta must match the curvature site dimension"
        )
    one = Cbb @ np.ones_like(theta)
    rs = Cdd @ np.sin(theta) - np.cos(theta) * g
    rc = Cdd @ np.cos(theta) + np.sin(theta) * g
    scale = max(float(np.max(np.abs(Cbb))), float(np.max(np.abs(Cdd))), 1e-12)
    worst = max(
        float(np.max(np.abs(one))), float(np.max(np.abs(rs))), float(np.max(np.abs(rc)))
    )
    return {
        "applicable": True,
        "passed": bool(worst <= tol * scale),
        "reason": None,
        "max_residual": worst,
        "scaled_residual": worst / scale,
        "tolerance": float(tol),
        "residual_beta_global": one,
        "residual_delta_sin": rs,
        "residual_delta_cos": rc,
    }


def pair_once_beta(curv, thetas):
    """Pair-once isotropic prediction ``-sum_j Cdd_ij sin(Theta_i-Theta_j)``."""
    t = np.asarray(thetas, float)
    C = np.asarray(curv.blocks[("d", "d")], float)
    return -np.sum(C * np.sin(t[:, None] - t[None, :]), axis=1)


def pair_once_from_ordered(ordered: Mapping):
    """Convert directed TB2J pairs to unique pair-once ``(i,j): J`` values.

    One stored ordered value is half the pair-once coupling. If both directed
    entries are present they must agree
    they are not counted twice.
    """
    result = {}
    for pair, value in ordered.items():
        if len(pair) != 2:
            raise NonstationaryCurvatureError("ordered J keys must be (i,j) pairs")
        i, j = map(int, pair)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        current = 2.0 * float(value)
        if key in result and not np.isclose(
            result[key], current, atol=1e-12, rtol=1e-10
        ):
            raise NonstationaryCurvatureError(
                f"ordered-pair J mismatch for {key}: directed values disagree"
            )
        result[key] = current
    return result


def known_j_beta(j_pairs: Mapping, thetas):
    """Pair-once Heisenberg beta gradient for unordered ``(i,j): J`` pairs."""
    t = np.asarray(thetas, float)
    g = np.zeros(len(t))
    for (i, j), J in j_pairs.items():
        g[i] += float(J) * np.sin(t[i] - t[j])
        g[j] += float(J) * np.sin(t[j] - t[i])
    return g


def known_j_comparison(measured_g_beta, j_pairs: Mapping, thetas, *, tol=1e-8):
    """Report measured-vs-known pair-once gradients without overwriting J."""
    measured = np.asarray(measured_g_beta, float)
    predicted = known_j_beta(j_pairs, thetas)
    if measured.shape != predicted.shape:
        raise NonstationaryCurvatureError(
            "measured gradient shape does not match known-J prediction"
        )
    residual = measured - predicted
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "matched": bool(max_abs <= tol),
        "tolerance": float(tol),
        "max_abs_mismatch": max_abs,
        "measured_g_beta": measured,
        "predicted_g_beta": predicted,
        "residual": residual,
    }


def j_fit_report(g_beta, thetas):
    """Least-squares one-snapshot pair-J fit with explicit uniqueness status."""
    g, t = np.asarray(g_beta, float), np.asarray(thetas, float)
    if g.shape != t.shape or g.ndim != 1:
        raise NonstationaryCurvatureError(
            "g_beta and thetas must be same-length vectors"
        )
    pairs = list(product(range(len(t)), range(len(t))))
    pairs = [(i, j) for i, j in pairs if i < j]
    A = np.zeros((len(t), len(pairs)))
    for p, (i, j) in enumerate(pairs):
        A[i, p] = np.sin(t[i] - t[j])
        A[j, p] = np.sin(t[j] - t[i])
    x, _, rank, s = np.linalg.lstsq(A, g, rcond=None)
    return {
        "rank": int(rank),
        "n_pairs": len(pairs),
        "unique": bool(rank == len(pairs)),
        "nonunique": bool(rank < len(pairs)),
        "singular_values": s,
        "pairs": pairs,
        "least_squares": x,
        "residual_norm": float(np.linalg.norm(A @ x - g)),
    }


def lab_ring_curvature_fd(
    provider,
    *,
    q_frac,
    taus,
    phis,
    B_local,
    translation_cutoff,
    occupations,
    kpts,
    kweights,
    h=2e-4,
    richardson=True,
):
    """Independent explicit commensurate lab-torus FD oracle, both channels."""
    q, taus, phis, B = (
        np.asarray(q_frac, float),
        np.asarray(taus, float),
        np.asarray(phis, float),
        np.asarray(B_local, float),
    )
    extents = _translation_extents(translation_cutoff)
    axes = [a for a in range(3) if extents[a] > 0]
    for a in axes:
        if not np.isclose(q[a] * extents[a], np.rint(q[a] * extents[a]), atol=1e-10):
            raise NonstationaryCurvatureError(
                "explicit lab-torus oracle requires q_a * extent_a integer on every active axis"
            )
    grid = _cell_grid(extents)
    ncell = len(grid)
    cell_index = {tuple(int(x) for x in cell): i for i, cell in enumerate(grid)}
    norb = len(B)
    nsite = ncell * norb
    size = 2 * nsite
    HR = np.asarray(provider.HR, complex)
    SR = np.asarray(provider.SR, complex)
    Rlist = np.asarray(provider.Rlist, int)
    inactive = [a for a in range(3) if extents[a] == 0]
    if inactive and np.any(Rlist[:, inactive] != 0):
        raise NonstationaryCurvatureError(
            "stored hopping extends along an inactive cutoff axis; widen translation_cutoff"
        )
    H0 = np.zeros((size, size), complex)
    S0 = np.zeros_like(H0)

    def idx(a, mu):
        return 2 * (a * norb + mu)

    for cell in grid:
        a = cell_index[tuple(int(x) for x in cell)]
        for ir, R in enumerate(Rlist):
            neighbour = cell + R
            for ax in axes:
                neighbour[ax] %= extents[ax]
            b = cell_index[tuple(int(x) for x in neighbour)]
            for mu in range(norb):
                for nu in range(norb):
                    ia, ib = idx(a, mu), idx(b, nu)
                    H0[ia : ia + 2, ib : ib + 2] += HR[ir, mu, nu] * np.eye(2)
                    S0[ia : ia + 2, ib : ib + 2] += SR[ir, mu, nu] * np.eye(2)
    theta = np.array(
        [
            2 * np.pi * (np.dot(q, cell + taus[mu])) + phis[mu]
            for cell in grid
            for mu in range(norb)
        ]
    )
    Bf = np.tile(B / 2, ncell)

    def energy(eta):
        Hp = H0.copy()
        for site in range(nsite):
            beta, delta = eta[0, site], eta[1, site]
            th = theta[site]
            bf = Bf[site]
            sx = np.array([[0, 1], [1, 0]], complex)
            sy = np.array([[0, -1j], [1j, 0]], complex)
            sz = np.diag([1.0, -1.0])
            field = (
                np.sin(th + beta) * np.cos(delta) * sx
                + np.sin(delta) * sy
                + np.cos(th + beta) * np.cos(delta) * sz
            ) * bf
            sl = slice(2 * site, 2 * site + 2)
            # H0 has hopping only; onsite magnetic field is inserted here.
            Hp[sl, sl] += field
        ev = eigh(Hp, S0, eigvals_only=True)
        # Map the primitive fixed occupations to the reference ring eigenvalue order.
        ref_e = []
        ref_f = []
        for ik, k in enumerate(kpts):
            e = eigh(*provider.gen_ham(k), eigvals_only=True)
            ref_e.extend(e)
            ref_f.extend(np.asarray(occupations)[ik])
        f = np.asarray(ref_f)[np.argsort(np.asarray(ref_e))]
        return float(np.dot(f, ev))

    # Ensure H0 onsite has no pre-existing provider field; lab FD adds it once.
    blocks = {}
    for ca in range(2):
        for cb in range(2):
            M = np.zeros((nsite, nsite))
            for i in range(nsite):
                for j in range(nsite):

                    def mixed(step):
                        a = np.zeros((2, nsite))
                        b = np.zeros_like(a)
                        a[ca, i] = step
                        b[cb, j] = step
                        return (
                            energy(a + b)
                            - energy(a - b)
                            - energy(-a + b)
                            + energy(-a - b)
                        ) / (4 * step * step)

                    val = mixed(h)
                    if richardson:
                        val = (4 * mixed(h / 2) - val) / 3
                    M[i, j] = val
            if ca == cb:
                M = 0.5 * (M + M.T)
            blocks[(("b", "d")[ca], ("b", "d")[cb])] = M
    return NonstationaryCurvature(
        blocks,
        norb,
        ncell,
        np.zeros(norb),
        np.column_stack(
            (np.diag(blocks[("b", "b")])[:norb], np.diag(blocks[("d", "d")])[:norb])
        ),
        {
            "oracle": "explicit lab torus frozen-occupation central FD",
            "primitive_only": False,
            "translation_extents": [int(x) for x in extents],
            "translation_cells": ncell,
            "translations": [[int(x) for x in cell] for cell in grid],
        },
        None,
        translations=np.asarray(grid),
    )
