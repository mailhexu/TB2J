"""Folded-pencil Green functions for spin-spiral frozen states (story 003).

Normative conventions: ``docs/sympy/spiral_green_function_mft.py``
(sections 1-2) and the spiral frozen-bundle contract
(``tbupy_spiral_state`` v1):

* interleaved spin basis ``(orb0 up, orb0 down, orb1 up, ...)``, eV;
* folded pencils ``Hq(k), Sq(k)`` from the contract rebuild rule, with
  the tbupy multi-sublattice assembler as single source of truth;
* resolvent ``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}`` with the S-weighted
  pole sum ``Tr[Sq Gq] = sum_n 1/(E - eps_n)``;
* unfolding to real-space spin blocks by the inverse twisted-Bloch
  transform, Eq. (4) of the derivation script: the basis carries the
  cell half-twist ``exp(+i sigma_s pi q a)`` only, while the sublattice
  phases (``taus``, ``phis``) live in ``Hq(k)`` - the unfold factors
  are therefore tau/phi-independent.

:class:`SpiralState` mirrors ``tbupy.spiral_state.SpiralState``
field-for-field so bundles round-trip between the repos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SpiralState",
    "SpiralGreen",
    "SpiralGreenAtE",
    "assemble_folded_pencil",
]


@dataclass
class SpiralState:
    """TB2J mirror of the ``tbupy_spiral_state`` bundle (schema v1).

    Collinear reference channels ``HR_up``/``HR_dn`` and the
    spin-independent overlap ``SR`` in ``(nR, norb, norb)`` layout over
    the stored signed cells ``Rlist``; spiral wavevector ``q_frac``;
    per-orbital fractional positions ``taus`` and sublattice phase
    offsets ``phis``; twisted-gauge local exchange field magnitudes
    ``B_local``; rotating-frame density matrix ``rho`` and Hubbard
    potential ``V_U`` (both spin-interleaved ``(2 norb, 2 norb)``);
    frozen Fermi level ``efermi``; per-orbital torque residuals
    ``torque_norms``; JSON metadata string (dc type, smearing width,
    guard flags, ...).
    """

    HR_up: np.ndarray
    HR_dn: np.ndarray
    SR: np.ndarray
    Rlist: np.ndarray
    q_frac: np.ndarray
    taus: np.ndarray
    phis: np.ndarray
    B_local: np.ndarray
    rho: np.ndarray | None = None
    V_U: np.ndarray | None = None
    efermi: float = 0.0
    torque_norms: np.ndarray | None = None
    metadata_json: str = "{}"

    @property
    def norb(self) -> int:
        return int(np.asarray(self.HR_up).shape[1])

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)

    @property
    def width(self) -> float:
        """Occupation smearing width (eV) from the bundle metadata."""
        return float(self.metadata.get("width", 0.1))

    def validate(self) -> None:
        """Check bundle shapes against the contract schema."""
        HR = np.asarray(self.HR_up)
        if HR.ndim != 3 or HR.shape[1] != HR.shape[2]:
            raise ValueError(f"HR_up must have shape (nR, norb, norb), got {HR.shape}")
        nR, norb = HR.shape[0], HR.shape[1]
        for name in ("HR_dn", "SR"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != HR.shape:
                raise ValueError(
                    f"{name} must have shape {HR.shape} like HR_up, got {arr.shape}"
                )
        rlist = np.asarray(self.Rlist)
        if rlist.shape != (nR, 3):
            raise ValueError(f"Rlist must have shape (nR, 3), got {rlist.shape}")
        for name, shape in (
            ("q_frac", (3,)),
            ("taus", (norb, 3)),
            ("phis", (norb,)),
            ("B_local", (norb,)),
        ):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        for name in ("rho", "V_U", "torque_norms"):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            want = (norb,) if name == "torque_norms" else (2 * norb, 2 * norb)
            if arr.shape != want:
                raise ValueError(f"{name} must have shape {want}, got {arr.shape}")


def _sigma_z_field(B_local) -> np.ndarray:
    """Interleaved on-site field ``diag(+B/2 on up, -B/2 on down)``."""
    B = np.asarray(B_local, dtype=float)
    out = np.zeros((2 * len(B), 2 * len(B)))
    out[0::2, 0::2] += np.diag(0.5 * B)
    out[1::2, 1::2] -= np.diag(0.5 * B)
    return out


def assemble_folded_pencil(state: SpiralState, k_frac) -> tuple[np.ndarray, np.ndarray]:
    """Contract rebuild rule at one k.

    ``Hq(k) = assemble_multisublattice_hq_sq(HR_up, HR_dn, SR, Rlist,
    config, k) + interleave(B_local x sigma_z) + V_U``, with
    ``config = MultiSublatticeSpiralConfig(q_frac, taus, phis)``; ``V_U``
    is a rotating-frame operator added cell-periodically (R=0 block).
    Returns the interleaved spinor pair ``(Hq, Sq)`` of shape
    ``(2 norb, 2 norb)``.
    """
    from tbupy.generalized_bloch import (
        MultiSublatticeSpiralConfig,
        assemble_multisublattice_hq_sq,
    )

    config = MultiSublatticeSpiralConfig(state.q_frac, state.taus, state.phis)
    Hq, Sq = assemble_multisublattice_hq_sq(
        state.HR_up, state.HR_dn, state.SR, state.Rlist, config, k_frac
    )
    Hq = Hq + _sigma_z_field(state.B_local)
    if state.V_U is not None:
        Hq = Hq + np.asarray(state.V_U, dtype=complex)
    return Hq, Sq


@dataclass
class SpiralGreenAtE:
    """Energy snapshot of :class:`SpiralGreen` (returned by ``at(E)``)."""

    green: "SpiralGreen"
    E: complex
    evals: np.ndarray  # (nk, n) pencil eigenvalues
    evecs: np.ndarray  # (nk, n, n) standardized eigenvectors, C^dag Sq C = I
    f: np.ndarray  # (nk, n) frozen occupations at efermi/width
    Gq: np.ndarray  # (nk, n, n) resolvent (E Sq - Hq)^{-1}

    def pole_sum(self) -> np.ndarray:
        """``sum_n 1/(E - eps_n(k))`` per k point, shape ``(nk,)``."""
        return np.sum(1.0 / (self.E - self.evals), axis=1)

    def pole_sum_trace(self, S=None) -> np.ndarray:
        """Per-k S-weighted pole sum ``Tr[S Gq(k, E)]``, shape ``(nk,)``.

        With ``S = Sq(k)`` this equals :meth:`pole_sum`
        (derivation-script section 1); with ``S=None`` it is the plain
        trace ``Tr Gq(k, E)``.  ``S`` may be a single ``(n, n)`` operator
        or a stacked ``(nk, n, n)`` array.
        """
        if S is None:
            return np.trace(self.Gq, axis1=1, axis2=2)
        S = np.asarray(S, dtype=complex)
        if S.ndim == 2:
            return np.einsum("ij,kji->k", S, self.Gq)
        if S.shape != self.Gq.shape:
            raise ValueError(f"S must have shape {self.Gq.shape}, got {S.shape}")
        return np.einsum("kij,kji->k", S, self.Gq)

    def unfold(self, i, j_cells) -> np.ndarray:
        """Real-space spin blocks ``G_q[(0, i), (R, j)](E)``.

        ``i`` is the orbital index in cell 0; ``j_cells`` is an iterable
        of ``(j, R)`` pairs with orbital index ``j`` and integer lattice
        displacement ``R = b - a``.  Returns complex array
        ``(len(j_cells), 2, 2)`` indexed by spin ``(s, s')``.

        Inverse twisted-Bloch transform, Eq. (4)::

            G_s[(a i s), (b j s')](E) = (1/N) sum_m e^{2 pi i k_m (a-b)}
                conj(tw(s, m)) Gq(k_m)[2i+s, 2j+s'] tw(s', m),
            tw(s, m) = exp(+i sigma_s pi q m),  sigma_up = +1.

        The mesh must be the uniform commensurate mesh
        ``k_m = (m/N, 0, 0)`` of the N-cell supercell.
        """
        green = self.green
        nk = green.kmesh.shape[0]
        m = np.arange(nk)
        if not (
            np.allclose(green.kmesh[:, 0], m / nk)
            and np.allclose(green.kmesh[:, 1:], 0.0)
        ):
            raise ValueError(
                "unfold requires the uniform commensurate mesh k_m = (m/N, 0, 0)"
            )
        q = float(np.asarray(green.state.q_frac, dtype=float)[0])
        sigma = np.array([1.0, -1.0])  # sigma_up = +1, sigma_dn = -1
        tw = np.exp(1j * np.pi * sigma[:, None] * q * m[None, :])  # (2, nk)
        i = int(i)
        out = np.empty((len(j_cells), 2, 2), dtype=complex)
        for row, item in enumerate(j_cells):
            j, R = item
            j = int(j)
            R = np.asarray(R, dtype=float)
            if R.ndim == 0:  # scalar displacement along the ring axis
                R = np.array([R, 0.0, 0.0])
            bloch = np.exp(-2j * np.pi * (green.kmesh @ R)) / nk  # a - b = -R
            sub = self.Gq[:, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            out[row] = np.einsum("m,sm,mst,tm->st", bloch, np.conj(tw), sub, tw)
        return out


class SpiralGreen:
    """Folded-pencil Green function of a frozen spiral bundle.

    Prebuilds ``Hq(k), Sq(k)`` on the given k mesh (contract rebuild
    rule) and the standardized pencil eigenbasis
    ``Hq C = eps Sq C, C^dag Sq C = I`` once (batched over k); per
    energy it exposes the frozen occupations, the resolvent
    ``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}``, S-weighted pole sums, and
    unfolding to real-space spin blocks.
    """

    def __init__(self, state: SpiralState, kmesh, kweights=None):
        state.validate()
        kmesh = np.asarray(kmesh, dtype=float)
        if kmesh.ndim != 2 or kmesh.shape[1] != 3:
            raise ValueError(f"kmesh must have shape (nk, 3), got {kmesh.shape}")
        nk = kmesh.shape[0]
        if kweights is None:
            kweights = np.full(nk, 1.0 / nk)
        kweights = np.asarray(kweights, dtype=float)
        if kweights.shape != (nk,):
            raise ValueError(
                f"kweights must have shape (nk,) = ({nk},), got {kweights.shape}"
            )
        self.state = state
        self.kmesh = kmesh
        self.kweights = kweights
        self.norb = state.norb
        self.n = 2 * self.norb
        Hq = np.empty((nk, self.n, self.n), dtype=complex)
        Sq = np.empty((nk, self.n, self.n), dtype=complex)
        for ik, k in enumerate(kmesh):
            Hq[ik], Sq[ik] = assemble_folded_pencil(state, k)
        self.Hq = Hq
        self.Sq = Sq
        # Standardized pencil eigenbasis, batched over k: S = L L^dag,
        # A = L^{-1} Hq L^{-dag}, C = L^{-dag} U with A U = eps U.
        L = np.linalg.cholesky(Sq)
        Li = np.linalg.inv(L)
        Li_dag = np.conj(np.swapaxes(Li, 1, 2))
        A = Li @ Hq @ Li_dag
        A = 0.5 * (A + np.conj(np.swapaxes(A, 1, 2)))
        evals, U = np.linalg.eigh(A)
        self.evals = evals
        self.evecs = Li_dag @ U

    def frozen_occupations(self, evals=None) -> np.ndarray:
        """Fermi occupations from the frozen efermi and smearing width."""
        if evals is None:
            evals = self.evals
        return 1.0 / (1.0 + np.exp((evals - self.state.efermi) / self.state.width))

    def at(self, E: complex) -> SpiralGreenAtE:
        """Energy snapshot: eigenbasis quantities, occupations, resolvent."""
        E = complex(E)
        f = self.frozen_occupations()
        Gq = np.linalg.inv(E * self.Sq - self.Hq)
        return SpiralGreenAtE(
            green=self, E=E, evals=self.evals, evecs=self.evecs, f=f, Gq=Gq
        )

    def eigenbasis(self, E: complex):
        """Eigenbasis quantities at ``E``: ``(evals, evecs, frozen f_n)``."""
        at = self.at(E)
        return at.evals, at.evecs, at.f

    def resolvent(self, E: complex) -> np.ndarray:
        """``Gq(k, E) = (E Sq(k) - Hq(k))^{-1}``, vectorized over k."""
        return self.at(E).Gq
