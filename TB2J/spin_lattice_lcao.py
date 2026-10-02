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
from pathlib import Path

import numpy as np
from HamiltonIO.model.occupations import myfermi

from TB2J.exchange_pert2 import ExchangePert2
from TB2J.green import TBGreen


class DisplacementVertex:
    """Energy-dependent localized displacement vertex V(z) = dH - z dS.

    Holds two-legged real-space derivative matrices keyed by the pair
    (Rq, Rk), with both legs measured relative to the cell Ru of the
    displaced atom, in the primitive-orbital gauge of the reference model.
    Both stores use the reference model's full basis layout; producers
    duplicate spin-blind overlap data into the spinor layout.

    The contract consumed by ``TBGreen.get_dGR*`` is
    ``get_vertex_two_spin(Rq, Rk, z)`` returning the contraction-ready
    physical matrix ``dH(Rq,Rk) - z*dS(Rq,Rk)`` without a transpose.
    A missing key yields a zero vertex: the R-support is data-driven.
    """

    def __init__(self, dH, dS, nbasis, Rqlist=None, Rklist=None):
        """
        Parameters
        ----------
        dH, dS : dict[(Rq, Rk) -> (nbasis, nbasis) complex ndarray]
            Displacement derivatives with legs relative to the displaced
            cell, already in the full (spinor, for NCL) model layout.
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

    def __init__(
        self,
        dH,
        dS,
        Ru,
        atoms,
        provenance="unlabeled",
        atom_index=None,
        direction=None,
    ):
        self.dH = {tuple(k): np.asarray(v, dtype=complex) for k, v in dH.items()}
        self.dS = {tuple(k): np.asarray(v, dtype=complex) for k, v in dS.items()}
        self.Ru = tuple(Ru)
        self.atoms = atoms
        self.provenance = str(provenance)
        self.atom_index = atom_index
        self.direction = tuple(direction) if direction is not None else None

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
            atom_index=None,
            direction=tuple(direction),
        )

    @classmethod
    def from_siesta_dhsdr(cls, block, atoms=None, reference_fermi=0.0):
        """Producer C: SIESTA ``dHSdR.nc`` loader (upstream dev/rel-5.4).

        ``block`` is a ``HamiltonIO.siesta.dhsdr.DisplacementBlock`` from
        ``read_dHSdR`` (one displaced unit-cell atom, one Cartesian axis).
        Its rows are unit-cell orbitals (cell 0) and columns live in
        supercell cells ``R``. Under the adopted convention (user decision
        2026-09-30, following the upstream developer's description) the
        R-resolved blocks are treated as the localized derivative of the
        displaced home-cell atom: every R-block maps to the vertex key
        (Rq, Rk) = ((0,0,0), R) with Ru = (0,0,0), exactly like the ABACUS
        direct loader. The derivative is a self-consistent central
        difference from the FC run.
        ``reference_fermi`` is the undisplaced raw SIESTA Fermi energy in
        eV. Pass it for sisl models (which store H - E_F S); leave zero
        for an unshifted model. The derivative of this fixed gauge is
        dH - E_F(reference) dS, not a moving Fermi-energy shift.
        """
        from HamiltonIO.siesta.dhsdr import block_to_R_lists

        Rlist, dHR, dSR = block_to_R_lists(block)
        dHR = dHR - reference_fermi * dSR[..., None]
        if dHR.size == 0:
            raise ValueError(
                f"dHSdR block (atom {block.atom_index}, axis {block.axis}) "
                "is empty — check FC.dHdR.Tolerance"
            )
        if dHR.shape[-1] not in (1, 2):
            raise ValueError(
                f"dHSdR spin dimension {dHR.shape[-1]} not supported "
                "(collinear/non-polarized only)"
            )
        if dHR.shape[-1] == 1:
            dH_up = dH_dn = dHR[..., 0]
            dS = dSR
        else:
            dH_up = dHR[..., 0]
            dH_dn = dHR[..., 1]
            dS = dSR
        dH = {
            ((0, 0, 0), tuple(R)): cls._interleave_two_spin(dH_up[iR], dH_dn[iR])
            for iR, R in enumerate(Rlist)
        }
        dS_sc = {
            ((0, 0, 0), tuple(R)): cls._interleave_two_spin(dS[iR], dS[iR])
            for iR, R in enumerate(Rlist)
        }
        direction = tuple(1 if i == block.axis - 1 else 0 for i in range(3))
        return cls(
            dH,
            dS_sc,
            Ru=(0, 0, 0),
            atoms=atoms,
            provenance=(
                "siesta dHSdR.nc FC central difference (self-consistent); "
                f"atom_index={block.atom_index}, direction={direction}; "
                f"raw_reference_fermi_eV={reference_fermi}"
            ),
            atom_index=block.atom_index,
            direction=direction,
        )

    @classmethod
    def from_supercell_runs(
        cls,
        ref_model,
        plus_model,
        minus_model,
        orb_map,
        atom_orbs,
        delta,
        atom_index=None,
        direction=(1, 0, 0),
    ):
        """Producer B: central differences of +-displaced supercell runs.

        Parameters
        ----------
        ref_model, plus_model, minus_model : HamiltonIO LCAOHamiltonian-like
            Supercell models with ``HR``, ``SR``, ``Rlist`` attributes.
            All three must share the same R-list and basis size (checked).
        orb_map : list of (prim_orb_index, cell tuple) per supercell orbital
            Maps each supercell orbital to its primitive orbital and cell.
            Use the symmetric cell range (e.g. -2,-1,0,1 for N=4) so
            boundary bonds fold to the shortest image.
        atom_orbs : list of primitive orbital indices on the displaced atom
            The displaced atom sits in supercell cell (0, 0, 0).
        delta : float
            Displacement magnitude in Angstrom (plus/minus runs at +-delta).
        atom_index, direction : displaced-atom identity recorded in the
            provenance metadata (FR-005).

        Each supercell matrix element ``HR[iR][i, j]`` connects supercell
        orbitals ``i`` (cell n_i) and ``j`` (cell n_j + R_sc): its two legs
        are (n_i, n_j + R_sc) and the entry folds into the primitive
        two-legged key (Rq, Rk) = (Ru - n_i, (n_j + R_sc) - n_i) at the
        primitive orbital indices (mu_i, mu_j). Blocks are stored in the
        PRIMITIVE basis so ``to_vertex(primitive_nbasis)`` accepts them.
        """
        rlists = {
            name: [tuple(int(c) for c in R) for R in np.atleast_2d(model.Rlist)]
            for name, model in (
                ("reference", ref_model),
                ("plus", plus_model),
                ("minus", minus_model),
            )
        }
        if not (rlists["reference"] == rlists["plus"] == rlists["minus"]):
            raise ValueError(
                "reference/plus/minus supercell R-lists differ; the runs "
                "must share the same R convention (FR-007)"
            )
        n_sc = np.asarray(plus_model.HR).shape[1]
        for name, model in (("reference", ref_model), ("minus", minus_model)):
            n = np.asarray(model.HR).shape[1]
            if n != n_sc:
                raise ValueError(
                    f"{name} model basis size {n} differs from the plus model {n_sc}"
                )
        if n_sc != len(orb_map):
            raise ValueError(
                f"orb_map has {len(orb_map)} entries but the supercell basis "
                f"has {n_sc}"
            )

        # supercell multiplicity per axis from the complete cell labeling
        cells = [tuple(int(c) for c in n) for _, n in orb_map]
        Ns = []
        for ax in range(3):
            vals = sorted({c[ax] for c in cells})
            N = len(vals)
            if sorted({v % N for v in vals}) != list(range(N)):
                raise ValueError(
                    "orb_map cells do not form a complete supercell residue "
                    "system per axis (FR-007)"
                )
            Ns.append(N)
        n_prim = 1 + max(mu for mu, _ in orb_map)
        # each primitive orbital must appear exactly once per supercell cell
        # (a duplicated or misordered same-length orb_map must not pass)
        from collections import Counter

        cell_of = {}
        counts = Counter((int(mu), tuple(int(c) for c in n)) for mu, n in orb_map)
        if counts and max(counts.values()) > 1:
            raise ValueError(
                "orb_map labels a (primitive orbital, cell) position more "
                "than once — the map is inconsistent with the supercell "
                "basis (FR-007)"
            )
        distinct_cells = set(cells)
        per_orb = Counter(int(mu) for mu, _ in orb_map)
        if (
            set(per_orb) != set(range(n_prim))
            or len(orb_map) != len(distinct_cells) * n_prim
        ):
            raise ValueError(
                "orb_map is not a complete labeling: every primitive "
                "orbital must appear exactly once in every supercell cell"
            )
        for mu, n in orb_map:
            cell_of.setdefault(mu, set()).add(tuple(int(c) for c in n))
        if any(s != distinct_cells for s in cell_of.values()):
            raise ValueError(
                "orb_map is not a complete labeling: some primitive orbital "
                "does not appear in every supercell cell"
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
        # (key, mu_i, mu_j) -> source (R_sc, i, j): each position of the
        # unfolded vertex must be written exactly once; a second nonzero
        # contribution means two displaced-image derivative supports alias
        # into the same position, i.e. the supercell is too small
        seen = {}
        for iR, R_sc in enumerate(rlists["plus"]):
            dH_R = (np.asarray(plus_model.HR[iR]) - np.asarray(minus_model.HR[iR])) / (
                2.0 * delta
            )
            dS_R = (np.asarray(plus_model.SR[iR]) - np.asarray(minus_model.SR[iR])) / (
                2.0 * delta
            )
            for i in range(n_sc):
                mu_i, n_i = orb_map[i]
                for j in range(n_sc):
                    mu_j, n_j = orb_map[j]
                    if i not in displaced_sc_orbs and j not in displaced_sc_orbs:
                        continue
                    if np.all(dH_R[i, j] == 0) and np.all(dS_R[i, j] == 0):
                        continue
                    # second leg in primitive cells (R_sc is in supercell
                    # lattice units = N primitive cells per axis), reduced
                    # to the shortest image inside the supercell labeling
                    diff = [n_j[ax] + Ns[ax] * R_sc[ax] - n_i[ax] for ax in range(3)]
                    diff = [
                        d - Ns[ax] * int(round(d / Ns[ax])) if Ns[ax] > 1 else d
                        for ax, d in enumerate(diff)
                    ]
                    key = (
                        tuple(int(-c) for c in n_i),
                        tuple(int(d) for d in diff),
                    )
                    pos = (key, int(mu_i), int(mu_j))
                    if pos in seen and seen[pos] != (tuple(R_sc), i, j):
                        raise ValueError(
                            "supercell too small: displaced-image derivative "
                            "supports overlap — use a larger supercell "
                            f"(position {pos} written from {seen[pos]} and "
                            f"{(tuple(R_sc), i, j)})"
                        )
                    seen[pos] = (tuple(R_sc), i, j)
                    dH.setdefault(key, np.zeros((n_prim, n_prim), dtype=complex))
                    dS.setdefault(key, np.zeros((n_prim, n_prim), dtype=complex))
                    dH[key][mu_i, mu_j] += dH_R[i, j]
                    dS[key][mu_i, mu_j] += dS_R[i, j]
        # drop keys where both derivative channels vanish
        for key in list(dH):
            if not np.any(dH[key]) and not np.any(dS[key]):
                del dH[key]
                del dS[key]
        return cls(
            dH,
            dS,
            Ru=(0, 0, 0),
            atoms=getattr(ref_model, "atoms", None),
            provenance=(
                "supercell +/- finite difference (self-consistent); "
                f"atom_index={atom_index}, direction={tuple(direction)}"
            ),
            atom_index=atom_index,
            direction=tuple(direction),
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


class _FullSpectrumGreen(TBGreen):
    def _prepare_eigen(self, solve=True, saveH=False):
        self._collinear_interleaved = getattr(
            self.tbmodel, "collinear_interleaved", False
        )
        self._fourier_phases = {}
        super()._prepare_eigen(solve=solve, saveH=saveH)
        if self._use_cache:
            self.evecs = np.memmap(
                Path(self.cache_path) / "evecs.dat",
                mode="r",
                shape=self.evecs_shape,
                dtype=complex,
            )
            if not self.is_orthogonal:
                self.S = np.memmap(
                    Path(self.cache_path) / "S.dat",
                    mode="r",
                    shape=(self.nkpts, self.nbasis, self.nbasis),
                    dtype=complex,
                )
        if self._collinear_interleaved:
            self._spin_vectors = tuple(
                np.ascontiguousarray(self.evecs[:, s::2, s::2]) for s in (0, 1)
            )
            self._spin_adjoint = tuple(
                v.conj().swapaxes(-1, -2) for v in self._spin_vectors
            )
            self._spin_overlap = (
                None
                if self.is_orthogonal
                else tuple(np.ascontiguousarray(self.S[:, s::2, s::2]) for s in (0, 1))
            )
            # All subsequent contractions use scalar spin blocks. Full-matrix
            # accessors reconstruct one k-point instead of retaining zeros.
            del self.evecs, self.S

    def get_evecs(self, ik):
        if not self._collinear_interleaved:
            return super().get_evecs(ik)
        return LocalizedDisplacementDerivative._interleave_two_spin(
            self._spin_vectors[0][ik], self._spin_vectors[1][ik]
        )

    def get_Sk(self, ik):
        if not self._collinear_interleaved:
            return super().get_Sk(ik)
        if self.is_orthogonal:
            return None
        return LocalizedDisplacementDerivative._interleave_two_spin(
            self._spin_overlap[0][ik], self._spin_overlap[1][ik]
        )

    def get_density_matrix(self):
        if not self._collinear_interleaved:
            return super().get_density_matrix()
        rho = np.zeros((self.nbasis, self.nbasis), complex)
        for ik, weight in enumerate(self.kweights):
            occupation = myfermi(
                self.evals[ik], self.efermi, width=self.fermi_width, nspin=2
            )
            for spin in (0, 1):
                sl = slice(spin, None, 2)
                block = (
                    self._spin_vectors[spin][ik] * occupation[sl]
                ) @ self._spin_adjoint[spin][ik]
                if not self.is_orthogonal:
                    block = block @ self._spin_overlap[spin][ik]
                rho[sl, sl] += weight * block
        return rho

    def get_spin_realspace(self, energy, Rpts, derivatives=None, with_rho=False):
        """Full-spectrum k sums with bounded temporary storage per energy."""
        n = self.nbasis // 2
        phase = self._fourier_phase(Rpts, self.kpts)
        gr = np.zeros((len(Rpts), self.nbasis, self.nbasis), complex)
        dgr = np.zeros_like(gr) if derivatives is not None else None
        rho = np.zeros((self.nbasis, self.nbasis), complex) if with_rho else None
        z = energy + self.efermi
        for spin in (0, 1):
            sl = slice(spin, None, 2)
            scalar_gr = np.zeros((len(Rpts), n, n), complex)
            scalar_dgr = np.zeros_like(scalar_gr) if derivatives is not None else None
            for start in range(0, self.nkpts, 64):
                part = slice(start, min(start + 64, self.nkpts))
                denominator = 1.0 / (z - self.evals[part, sl])
                g = (
                    self._spin_vectors[spin][part] * denominator[:, None, :]
                ) @ self._spin_adjoint[spin][part]
                count = g.shape[0]
                scalar_gr += (phase[:, part] @ g.reshape(count, -1)).reshape(
                    len(Rpts), n, n
                )
                if derivatives is not None:
                    dh, ds = derivatives
                    dg = g @ (dh[spin][part] - z * ds[spin][part]) @ g
                    scalar_dgr += (phase[:, part] @ dg.reshape(count, -1)).reshape(
                        len(Rpts), n, n
                    )
                if with_rho:
                    sg = g if self.is_orthogonal else self._spin_overlap[spin][part] @ g
                    rho[sl, sl] += np.einsum("k,kij->ij", self.kweights[part], sg)
            gr[:, sl, sl] = scalar_gr
            if dgr is not None:
                dgr[:, sl, sl] = scalar_dgr
        return gr, dgr, rho

    def get_spin_Gk_all(self, energy):
        blocks = []
        for spin in (0, 1):
            denominators = 1.0 / (energy + self.efermi - self.evals[:, spin::2])
            blocks.append(
                (self._spin_vectors[spin] * denominators[:, None, :])
                @ self._spin_adjoint[spin]
            )
        return blocks

    def _fourier_phase(self, Rpts, kpts):
        key = tuple(tuple(R) for R in Rpts)
        if key not in self._fourier_phases:
            self._fourier_phases[key] = (
                np.exp(
                    self.k2Rfactor
                    * (np.asarray(Rpts).reshape(-1, 3) @ np.asarray(kpts).T)
                )
                * self.kweights[None, :]
            )
        return self._fourier_phases[key]

    def compute_GR_spin(self, Rpts, kpts, blocks):
        phase = self._fourier_phase(Rpts, kpts)
        result = np.zeros((len(Rpts), self.nbasis, self.nbasis), complex)
        for spin, scalar in enumerate(blocks):
            sl = slice(spin, None, 2)
            result[:, sl, sl] = (phase @ scalar.reshape(self.nkpts, -1)).reshape(
                len(Rpts), self.nbasis // 2, self.nbasis // 2
            )
        return result

    def get_Gk_all(self, energy, ispin=0):
        if not self._collinear_interleaved:
            return super().get_Gk_all(energy, ispin=ispin)
        result = np.zeros((self.nkpts, self.nbasis, self.nbasis), complex)
        for spin, block in enumerate(self.get_spin_Gk_all(energy)):
            result[:, spin::2, spin::2] = block
        return result

    def compute_GR(self, Rpts, kpts, Gks):
        if not self._collinear_interleaved:
            return super().compute_GR(Rpts, kpts, Gks)
        blocks = [np.ascontiguousarray(Gks[:, s::2, s::2]) for s in (0, 1)]
        return self.compute_GR_spin(Rpts, kpts, blocks)

    def _reduce_eigens(self, evals, evecs, emin, emax):
        # G V G differentiates the full pencil inverse, not a moving band cut.
        return evals, evecs


class ExchangePert2Sl(ExchangePert2):
    """Spin-lattice exchange with a localized non-orthogonal displacement.

    The displacement enters through a ``LocalizedDisplacementDerivative``:
    ``self.epc`` becomes the energy-dependent vertex provider
    (``V(z)=dH-z*dS`` at each contour energy) and the A-matrix derivative
    gains the two on-site magnetic-vertex terms dP_i, dP_j (H-like, from
    the vertex on-site entry at z=0, where the dS term vanishes). The
    collinear reduction and channel algebra are certified in
    docs/sympy/nonorthogonal_spin_lattice.py (checks F0-F2).
    """

    _green_class = _FullSpectrumGreen
    response = "localized"

    def set_displacement(
        self, disp, J_only=False, density_method="eigenvector", response="localized"
    ):
        if response not in ("localized", "periodic"):
            raise ValueError("response must be localized or periodic")
        self.response = response
        if disp is None:
            # J-only run: no derivative data; epc is unused with J_only
            self.J_only = True
            self.density_method = density_method
            self.epc = None
            self.Ru = (0, 0, 0)
            self._dPdict = None
            if not hasattr(self, "basis_is_separated"):
                orbs = self.iorb(self.ind_mag_atoms[0])
                self.basis_is_separated = (
                    len(orbs) > 1 and orbs[-1] - orbs[0] != len(orbs) - 1
                )
            self._init_pert_dicts()
            return
        vertex = disp.to_vertex(self.nbasis)
        # a new provider can change the (Rq, Rk) support: invalidate the
        # TBGreen R-map cache built from the previous provider
        self.G._Rmap = None
        self.G._Rmap_rev = None
        self.epc = vertex
        self.Ru = tuple(disp.Ru)
        if getattr(self, "orb_decomposition", False) and not J_only:
            raise NotImplementedError(
                "orbital-decomposed dJdx_orb does not include the dP terms "
                "yet; disable orb_decomposition for spin-lattice runs"
            )
        self.J_only = J_only
        self.density_method = density_method
        self.disp = disp
        if response == "periodic":
            # Sum translated images: the two-legged provider reduces to a
            # diagonal k-space vertex, with the phase carried by Rk.
            Rlist = sorted({Rk for _, Rk in set(disp.dH) | set(disp.dS)})
            phase = np.exp(
                self.G.R2kfactor * (self.G.kpts @ np.asarray(Rlist).reshape(-1, 3).T)
            )
            index = {R: i for i, R in enumerate(Rlist)}
            collinear = getattr(self.G, "_collinear_interleaved", False)
            n = self.nbasis // 2 if collinear else self.nbasis
            spins = (0, 1) if collinear else (None,)
            transformed = []
            for store in (disp.dH, disp.dS):
                blocks = []
                for spin in spins:
                    realspace = np.zeros((len(Rlist), n, n), complex)
                    for (_, Rk), matrix in store.items():
                        realspace[index[Rk]] += (
                            matrix[spin::2, spin::2] if collinear else matrix
                        )
                    blocks.append(
                        (phase @ realspace.reshape(len(Rlist), n * n)).reshape(
                            self.G.nkpts, n, n
                        )
                    )
                transformed.append(tuple(blocks))
            if collinear:
                self._spin_dHk, self._spin_dSk = transformed
                self._periodic_dH0 = np.zeros((self.nbasis, self.nbasis), complex)
                for spin in (0, 1):
                    self._periodic_dH0[spin::2, spin::2] = np.einsum(
                        "k,kij->ij", self.G.kweights, self._spin_dHk[spin]
                    )
            else:
                self._dHk, self._dSk = (blocks[0] for blocks in transformed)
                self._periodic_dH0 = np.einsum("k,kij->ij", self.G.kweights, self._dHk)
        else:
            support = set(self.short_Rlist)
            for Rq, Rk in set(disp.dH) | set(disp.dS):
                m = np.asarray(self.Ru) - np.asarray(Rq)
                n = m + np.asarray(Rk)
                support.update((tuple(m), tuple(-n)))
                for Rj in self.short_Rlist:
                    support.update(
                        (tuple(np.asarray(Rj) - n), tuple(m - np.asarray(Rj)))
                    )
            self._green_Rlist = sorted(support)
        self._dPdict = None
        if not hasattr(self, "basis_is_separated"):
            orbs = self.iorb(self.ind_mag_atoms[0])
            self.basis_is_separated = (
                len(orbs) > 1 and orbs[-1] - orbs[0] != len(orbs) - 1
            )
        self._init_pert_dicts()

    def _init_pert_dicts(self):
        # attribute parity with the EPW path (set_epw initializes these)
        from collections import defaultdict

        self.dA_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.dA2_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))

    def get_dP_iatom(self, iatom, R=(0, 0, 0)):
        """Displacement derivative of the magnetic on-site vertex P_i.

        The localized onsite is at cell R, with provider key (Ru-R, 0).
        Periodic response uses the zero-R Fourier component of the periodic
        derivative for every cell, not the k=0 sum over orbital-cell offsets.
        The overlap term drops at z=0; project on the reference moment.
        Missing onsite keys mean zero response under the supplied localized
        representation; R-resolved unit-cell FC data alone do not certify
        the onsite response of a distant atom in isolated bulk.
        """
        if getattr(self, "_dPdict", None) is None:
            self._dPdict = {}
        key = (iatom, tuple(R))
        if key in self._dPdict:
            return self._dPdict[key]
        Rq = tuple(np.asarray(self.Ru) - np.asarray(R))
        onsite = (
            self._periodic_dH0
            if self.response == "periodic"
            else self.epc.vertex(Rq, (0, 0, 0), 0.0)
        )
        orbs = self.iorb(iatom)
        block = onsite[np.ix_(orbs, orbs)]
        ni = block.shape[0] // 2
        if self.basis_is_separated:
            uu, dd, ud, du = (
                block[:ni, :ni],
                block[ni:, ni:],
                block[:ni, ni:],
                block[ni:, :ni],
            )
        else:
            uu, dd, ud, du = (
                block[::2, ::2],
                block[1::2, 1::2],
                block[::2, 1::2],
                block[1::2, ::2],
            )
        dm1 = (ud + du) / 2.0
        dm2 = (ud - du) * 0.5j
        dm3 = (uu - dd) / 2.0
        # reference local-moment direction from _prepare_Patom's evec
        H = self.get_H_atom(iatom)
        if self.basis_is_separated:
            hu, hd = H[:ni, :ni], H[ni:, ni:]
            hud, hdu = H[:ni, ni:], H[ni:, :ni]
        else:
            hu, hd = H[::2, ::2], H[1::2, 1::2]
            hud, hdu = H[::2, 1::2], H[1::2, ::2]
        m1 = (hud + hdu) / 2.0
        m2 = (hud - hdu) * 0.5j
        m3 = (hu - hd) / 2.0
        evec = np.array((np.trace(m1).real, np.trace(m2).real, np.trace(m3).real))
        norm = np.linalg.norm(evec)
        if norm > 1e-8:
            evec = evec / norm
        else:
            evec = np.array((0.0, 0.0, 1.0))
        dP = dm1 * evec[0] + dm2 * evec[1] + dm3 * evec[2]
        self._dPdict[key] = dP
        return dP

    def get_AijR_rhoR(self, e):
        """Energy-point worker: deterministic loop contraction with dP terms.

        Uses the loop ``get_all_A`` path (no vectorized try/except swap) so
        the dP insertion is always active (architecture ADR-003, W3).
        """
        Rpts = (
            self.short_Rlist
            if self.response == "periodic" or self.J_only
            else self._green_Rlist
        )
        collinear = getattr(self.G, "_collinear_interleaved", False)
        if collinear:
            vertices = (
                (self._spin_dHk, self._spin_dSk)
                if self.response == "periodic" and not self.J_only
                else None
            )
            gr, streamed_dgr, rho0 = self.G.get_spin_realspace(
                e,
                Rpts,
                derivatives=vertices,
                with_rho=self.density_method == "greens_function",
            )
            GR = dict(zip(Rpts, gr))
        else:
            Gk = self.G.get_Gk_all(e)
            GR = dict(zip(Rpts, self.G.compute_GR(Rpts, self.G.kpts, Gk)))
        dGRij, dGRji = {}, {}
        if not self.J_only:
            if self.response == "periodic":
                z = e + self.G.efermi
                if collinear:
                    dGR = streamed_dgr
                else:
                    dGk = Gk @ (self._dHk - z * self._dSk) @ Gk
                    dGR = self.G.compute_GR(self.short_Rlist, self.G.kpts, dGk)
                dGRij = dict(zip(self.short_Rlist, dGR))
                dGRji = {R: dGRij[tuple(-x for x in R)] for R in self.short_Rlist}
            else:
                dGRij, dGRji = self.G.get_dGR(
                    GR,
                    Rpts,
                    self.short_Rlist,
                    self.epc,
                    self.Ru,
                    cutoff=np.inf,
                    energy=e,
                )
        rho = None
        if self.density_method == "greens_function":
            if not collinear:
                rho0 = np.einsum(
                    "k,kij->ij",
                    self.G.kweights,
                    Gk if self.G.is_orthogonal else self.G.S @ Gk,
                )
            rho = self.get_rho_e_spin({(0, 0, 0): rho0})
        A, dA, Aorb, dAorb = self.get_all_A(GR, dGRij, dGRji)
        return A, dA, rho, Aorb, dAorb

    def get_A_ijR(self, G, dG, dGrev, R, iatom, jatom):
        """Parent contraction plus the two on-site dP product-rule terms."""
        tmp, dtmp, torb, dtorb = super().get_A_ijR(G, dG, dGrev, R, iatom, jatom)
        if self.J_only or getattr(self, "disp", None) is None:
            return tmp, dtmp, torb, dtorb
        Gij = self.GR_atom(G[R], iatom, jatom)
        Rm = tuple(-x for x in R)
        Gji = self.GR_atom(G[Rm], jatom, iatom)

        def pauli_blocks(M):
            ni, nj = M.shape[0] // 2, M.shape[1] // 2
            if self.basis_is_separated:
                uu, dd, ud, du = M[:ni, :nj], M[ni:, nj:], M[:ni, nj:], M[ni:, :nj]
            else:
                uu, dd, ud, du = M[::2, ::2], M[1::2, 1::2], M[::2, 1::2], M[1::2, ::2]
            return (uu + dd) / 2.0, (ud + du) / 2.0, (ud - du) * 0.5j, (uu - dd) / 2.0

        Pi = self.get_P_iatom(iatom)
        Pj = self.get_P_iatom(jatom)
        dPi = self.get_dP_iatom(iatom)
        dPj = self.get_dP_iatom(jatom, R)
        for a, b in ([0, 0], [3, 3]):
            Gij_a = pauli_blocks(Gij)[a]
            Gji_b = pauli_blocks(Gji)[b]
            extra = (dPi @ Gij_a @ Pj + Pi @ Gij_a @ dPj) @ Gji_b
            dtmp[a, b] += np.trace(extra) / np.pi
        return tmp, dtmp, torb, dtorb


def gen_exchange_spin_lattice_lcao(
    tbmodel,
    atoms,
    disp,
    basis=None,
    efermi=0.0,
    magnetic_elements=None,
    kmesh=(5, 5, 5),
    emin=-10.0,
    emax=0.0,
    nz=50,
    np=1,
    exclude_orbs=[],
    description="",
    list_iatom=None,
    output_path="TB2J_results",
    Jonly=False,
    integration_method="CFR",
    density_method="eigenvector",
    response="localized",
    Rcut=None,
    smearing=0.05,
):
    """Spin-lattice exchange for a non-orthogonal LCAO model (driver).

    Parameters mirror ``gen_exchange_Oiju_epw3`` where meaningful; the
    displacement comes from a ``LocalizedDisplacementDerivative`` (ABACUS
    ``out_mat_dh``/``out_mat_ds`` direct loader or supercell +/- FD)
    instead of EPW q-space data. ``tbmodel`` is the merged spinor model
    (interleaved or separated spin layout; auto-detected). ``efermi`` must
    be pinned from the undisplaced reference. The output ``SpinIO``
    carries ``J_iso`` and the spin-lattice derivative ``dJdx`` for the one
    displaced mode, with the derivative provenance in the description.
    """
    from TB2J.io_exchange.io_exchange import SpinIO

    if disp is None and not Jonly:
        raise ValueError("disp is required unless Jonly=True")
    meta = f" [spin-lattice response={response}; full_spectrum=True; smearing_eV={smearing}]"
    if disp is not None:
        meta += (
            f" [spin-lattice displacement provenance: {disp.provenance};"
            f" atom_index={disp.atom_index}, Ru={disp.Ru},"
            f" direction={disp.direction}]"
        )
    ex = ExchangePert2Sl(
        tbmodels=tbmodel,
        atoms=atoms,
        basis=basis,
        efermi=efermi,
        magnetic_elements=magnetic_elements,
        kmesh=kmesh,
        emin=emin,
        emax=emax,
        nz=nz,
        nproc=np,
        exclude_orbs=exclude_orbs,
        index_magnetic_atoms=list_iatom,
        description=description + meta,
        integration_method=integration_method,
        Rcut=Rcut,
        smearing=smearing,
    )
    if disp is not None:
        ex.set_displacement(
            disp, J_only=Jonly, density_method=density_method, response=response
        )
    else:
        ex.set_displacement(
            None, J_only=True, density_method=density_method, response=response
        )
    ex.run(output_path)
    return SpinIO.load_pickle(output_path)
