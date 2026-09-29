"""Non-orthogonal spin-lattice downfolding: displacement vertex and producers.

Implements the approved nonorth-spin-lattice architecture (TB2J vault
specs/nonorth-spin-lattice/): a localized atomic displacement in a
non-orthogonal LCAO basis perturbs both the Hamiltonian and the overlap
matrix, and the Green-function derivative at complex energy z is

    dG/dtau_a (z) = G(z) [ dH/dtau_a - z dS/dtau_a ] G(z)

(assertion-checked derivation: docs/sympy/nonorthogonal_spin_lattice.py).

The vertex is *energy-dependent*: it must be assembled at each contour
energy, never precomputed at a fixed z.  This module provides the
``DisplacementVertex`` provider consumed by ``TBGreen.get_dGR*`` through
the ``get_vertex_two_spin(Rq, Rk, z)`` contract, plus derivative producers
(``LocalizedDisplacementDerivative``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class DisplacementVertex:
    """Energy-dependent localized displacement vertex V(z) = dH - z dS.

    Holds two-legged real-space derivative matrices keyed by the pair
    (Rq, Rk), with both legs measured relative to the cell Ru of the
    displaced atom, in the primitive-orbital gauge of the reference model.
    ``dH`` may carry the two-spin (spinor) layout; ``dS`` is spin-blind and
    is duplicated into the same layout on access.

    The contract consumed by ``TBGreen.get_dGR*`` is
    ``get_vertex_two_spin(Rq, Rk, z)`` returning the contraction-ready
    matrix ``dH(Rq,Rk) - z*dS(Rq,Rk)`` (the transpose is applied by the
    provider, mirroring ``.T`` in the legacy EPW branch).  A missing key
    yields a zero vertex: the R-support is data-driven.
    """

    def __init__(self, dH, dS, nbasis, Rqlist=None, Rklist=None):
        """
        Parameters
        ----------
        dH, dS : dict[(Rq, Rk) -> (nbasis, nbasis) complex ndarray]
            Displacement derivatives with legs relative to the displaced
            cell. Entries may be spinor-sized (2*nbasis_scalar) already;
            dS is spin-blind and its scalar form is accepted.
        nbasis : int
            Row/column size of the returned vertex matrices (the size the
            Green functions have, i.e. spinor layout if the model is NCL).
        """
        self.dH = {tuple(k): np.asarray(v, dtype=complex) for k, v in dH.items()}
        self.dS = {tuple(k): np.asarray(v, dtype=complex) for k, v in dS.items()}
        self.nbasis = nbasis
        if Rqlist is not None and Rklist is not None:
            self.Rqdict = {tuple(R): None for R in Rqlist}
            self.Rkdict = {tuple(R): None for R in Rklist}
        else:
            rq = sorted({key[0] for key in self.dH} | {key[0] for key in self.dS})
            rk = sorted({key[1] for key in self.dH} | {key[1] for key in self.dS})
            self.Rqdict = {R: None for R in rq}
            self.Rkdict = {R: None for R in rk}
        for name, store in (("dH", self.dH), ("dS", self.dS)):
            for key, val in store.items():
                if val.shape != (nbasis, nbasis):
                    raise ValueError(
                        f"{name}{key} has shape {val.shape}, expected "
                        f"{(nbasis, nbasis)}; wrap scalar-channel data in the "
                        "spinor layout first"
                    )

    def vertex(self, Rq, Rk, z):
        """Physical (untransposed) vertex dH(Rq,Rk) - z*dS(Rq,Rk)."""
        key = (tuple(Rq), tuple(Rk))
        dh = self.dH.get(key)
        ds = self.dS.get(key)
        if dh is None and ds is None:
            return np.zeros((self.nbasis, self.nbasis), dtype=complex)
        out = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        if dh is not None:
            out += dh
        if ds is not None:
            out -= z * ds
        return out

    def get_vertex_two_spin(self, Rq, Rk, z, avg=False):
        """Contraction-ready vertex for ``TBGreen.get_dGR*``.

        The provider branch in ``green._get_dV`` applies no transpose, so
        this returns the physical vertex ``dH - z*dS`` directly (for a
        legacy ``EpmatOneMode`` the raw matrix needs ``.T``; providers do
        not).  ``avg`` is accepted for signature parity and ignored (there
        is no degenerate-q average for a localized displacement).
        """
        return self.vertex(Rq, Rk, z)


class LocalizedDisplacementDerivative:
    """Localized two-legged derivative data dH(Rq,Rk), dS(Rq,Rk).

    Producer-agnostic container (architecture ADR-002). Provenance string
    distinguishes frozen-potential direct derivatives (ABACUS
    ``out_mat_dh``/``out_mat_ds``) from self-consistent supercell finite
    differences.
    """

    def __init__(self, dH, dS, Ru, atoms, provenance="unlabeled"):
        self.dH = {tuple(k): np.asarray(v, dtype=complex) for k, v in dH.items()}
        self.dS = {tuple(k): np.asarray(v, dtype=complex) for k, v in dS.items()}
        self.Ru = tuple(Ru)
        self.atoms = atoms
        self.provenance = str(provenance)

    def to_vertex(self, nbasis):
        return DisplacementVertex(self.dH, self.dS, nbasis)

    @staticmethod
    def _interleave_two_spin(up, dn):
        """Interleaved spinor layout [up_1, dn_1, up_2, dn_2, ...]."""
        n = up.shape[0]
        out = np.zeros((2 * n, 2 * n), dtype=complex)
        out[::2, ::2] = up
        out[1::2, 1::2] = dn
        return out

    @classmethod
    def from_abacus_dh(cls, parser, direction=(1, 0, 0), atoms=None, binary=None):
        """Producer A: ABACUS ``out_mat_dh``/``out_mat_ds`` direct loader.

        The CSR derivative blocks are already the localized two-legged
        vertex data for displacing the selected atom in the home cell, so
        every R-block maps to the key (Rq, Rk) = ((0,0,0), R) with
        Ru = (0,0,0); no unfolding is needed. Provenance is frozen
        potential (single SCF at the undisplaced density).
        """
        nbasis, Rlist, dH_up, dH_dn, dS = parser.read_dH_dS(
            direction=direction, binary=binary
        )
        dH = {
            ((0, 0, 0), tuple(int(x) for x in R)): cls._interleave_two_spin(
                dH_up[iR], dH_dn[iR]
            )
            for iR, R in enumerate(Rlist)
        }
        dS_sc = {
            ((0, 0, 0), tuple(int(x) for x in R)): cls._interleave_two_spin(
                dS[iR], dS[iR]
            )
            for iR, R in enumerate(Rlist)
        }
        return cls(
            dH,
            dS_sc,
            Ru=(0, 0, 0),
            atoms=getattr(parser, "atoms", atoms),
            provenance="abacus out_mat_dh/out_mat_ds (frozen potential)",
        )

    @classmethod
    def from_supercell_runs(
        cls, ref_model, plus_model, minus_model, orb_map, atom_orbs, delta
    ):
        """Producer B: central differences of +-displaced supercell runs.

        Parameters
        ----------
        ref_model, plus_model, minus_model : HamiltonIO LCAOHamiltonian-like
            Supercell models with ``HR``, ``SR``, ``Rlist`` attributes,
            identical basis ordering (asserted).
        orb_map : list of (prim_orb_index, cell tuple) per supercell orbital
            Maps each supercell orbital to its primitive orbital and cell
            (integer supercell-cell coordinates).
        atom_orbs : list of primitive orbital indices on the displaced atom
            The displaced atom sits in supercell cell (0, 0, 0).
        delta : float
            Displacement magnitude in Angstrom (plus/minus runs at +-delta).

        The supercell R-sum of each model gives the full matrix between all
        orbital pairs; the difference restricted to elements touching the
        displaced atom image is the two-legged derivative, keyed
        (Rq, Rk) = (-n1, n2 - n1) for legs (n1, n2). Provenance is the
        self-consistent response of the displaced runs.
        """
        for name, model in (("plus", plus_model), ("minus", minus_model)):
            if list(map(tuple, np.atleast_2d(plus_model.Rlist))) != list(
                map(tuple, np.atleast_2d(model.Rlist))
            ):
                raise ValueError(f"{name} model R-list differs from the plus model")

        def rsum(model, array):
            return sum(np.asarray(array[iR]) for iR in range(len(model.Rlist)))

        def diff(array_plus, array_minus):
            return (rsum(plus_model, array_plus) - rsum(minus_model, array_minus)) / (
                2.0 * delta
            )

        dM_up = diff(plus_model.HR, minus_model.HR)
        # collinear spinor handled by passing already-merged spinor models
        dM_S = diff(plus_model.SR, minus_model.SR)
        n_sc = dM_up.shape[1]
        if n_sc != len(orb_map):
            raise ValueError(
                f"orb_map has {len(orb_map)} entries but the supercell basis has {n_sc}"
            )
        displaced_sc_orbs = [
            i
            for i, (mu, n) in enumerate(orb_map)
            if mu in atom_orbs and tuple(n) == (0, 0, 0)
        ]
        if not displaced_sc_orbs:
            raise ValueError("no supercell orbital maps to the displaced atom image")
        dH = {}
        dS = {}
        for i in range(n_sc):
            mu_i, n_i = orb_map[i]
            for j in range(n_sc):
                mu_j, n_j = orb_map[j]
                if i not in displaced_sc_orbs and j not in displaced_sc_orbs:
                    continue
                key = (
                    tuple(int(-c) for c in n_i),
                    tuple(int(b - a) for a, b in zip(n_i, n_j)),
                )
                if np.all(dM_up[i, j] == 0) and np.all(dM_S[i, j] == 0):
                    continue
                dH.setdefault(key, np.zeros_like(dM_up))
                dS.setdefault(key, np.zeros_like(dM_S))
                dH[key][i, j] += dM_up[i, j]
                dS[key][i, j] += dM_S[i, j]
        return cls(
            dH,
            dS,
            Ru=(0, 0, 0),
            atoms=getattr(ref_model, "atoms", None),
            provenance="supercell +/- finite difference (self-consistent)",
        )


def fold_two_legged(GR, disp_vertex, Rpts, Rjlist, Ru, energy):
    """Reference folding of dG(R) for tests and cross-checks.

    Mirrors the (Rq, Rk) R-map contraction of ``TBGreen.get_dGR`` without
    caching or cutoff: forward Rm = Ru - Rq, Rn = Rm + Rk, Rnj = Rj - Rn;
    reverse Rn' = Ru - Rq, Rm' = Rn' + Rk, Rjn = Rn' - Rj, Rmi = -Rm'.
    z is the absolute contour energy entering V(z) = dH - z dS.
    """
    Rpts = [tuple(R) for R in Rpts]
    Rset = set(Rpts)
    Ru = np.array(Ru)
    dGRij = defaultdict(lambda: 0.0 + 0.0j)
    dGRji = defaultdict(lambda: 0.0 + 0.0j)
    for Rj in Rjlist:
        Rj = tuple(Rj)
        for Rq in disp_vertex.Rqdict:
            for Rk in disp_vertex.Rkdict:
                V = disp_vertex.vertex(Rq, Rk, energy)
                if not np.any(V):
                    continue
                Rm = tuple(Ru - np.array(Rq))
                Rn = tuple(np.array(Rm) + np.array(Rk))
                Rnj = tuple(np.array(Rj) - np.array(Rn))
                if Rm in Rset and Rnj in Rset:
                    dGRij[Rj] += GR[Rm] @ V @ GR[Rnj]
                Rmi = tuple(-np.array(Rn))
                Rjn = tuple(np.array(Rm) - np.array(Rj))
                if Rmi in Rset and Rjn in Rset:
                    dGRji[Rj] += GR[Rjn] @ V @ GR[Rmi]
    return dGRij, dGRji
