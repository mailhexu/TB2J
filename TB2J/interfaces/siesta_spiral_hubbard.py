"""+U bundle ingestion and Hubbard operator assembly (story 014).

Ingests a converged SIESTA SO-mode DFT+U reference (story-004 producer
probe: 4-atom Fe chain, U=3.0 eV J=0.7 eV, FLL double counting per
Bultmark-09 Eq. 13, projector method 2) and assembles the site-local
spin-dependent Hubbard potential together with the projector occupation
matrix, with every convention named (ADR-007).

Pinned SIESTA source conventions (mirror, never reinterpret):

* occupation matrix (m_occ_proj_dftu.F90:681-741):
  n(m,m') = sum_{mu,nu in supercell} rho_{mu nu} S(m,mu) S(m',nu),
  with the runtime-only projector overlap S = <proj_m | PAO_mu>
  recomputed offline by the certified two-center engine below;
* spinor density layout, file component order D1..D8 with the diag3k
  imaginary-sign fix mirrored exactly (m_occ_proj_dftu.F90:727-741):
  rho_upup   = D1 + i D5
  rho_dndn   = D2 + i D6
  rho_updn   = D3 - i D4   (file stores -Im; sign fix restores physics)
  rho_dnup   = D7 + i D8
* Hubbard potential (m_pot_dftu.F90:1057-1141), direct/fock splitting
    direct = vee(m, m'', m', m''')
    fock   = vee(m, m''', m'', m')
  V_hub[sigma=uu](m,m')  += direct*(n[m2,m3,uu] + n[m2,m3,dd])
                             - fock*n[m2,m3,uu]
  V_hub[sigma=dd](m,m')  += direct*(n[m2,m3,uu] + n[m2,m3,dd])
                             - fock*n[m2,m3,dd]
  V_hub[sigma=ud](m,m')  += -fock*n[m2,m3,ud]
  V_hub[sigma=du](m,m')  += -fock*n[m2,m3,du]
* double-counting potential (m_pot_dftu.F90:974-1035)
  V_dc[uu] = -delta(m,m') (U (n_tot - 1/2) - J (n_uu - 1/2))
  V_dc[dd] = -delta(m,m') (U (n_tot - 1/2) - J (n_dd - 1/2))
  V_dc[ud] = +delta(m,m') J n_ud
  V_dc[du] = +delta(m,m') J n_du
* double-counting energy per site (m_pot_dftu.F90:753-754)
  E_dc += 0.25 (U - J) n_corr(I)
* Slater integrals (dftu_specs.f:1749-1758), d shell:
  F0 = U, F2 = 14 J / (1 + F4/F2), F4 = 0.625 F2.

All internal energies are Ry; provenance fields carry eV alongside
(1 Ry = 13.605693122994 eV).

The stored HSX Hamiltonian already contains the +U potential folded in
(H_stored = H_xc + H_U, final_H_f_stress.F); the HubbardBundle records
that fact so no consumer double-counts.  The story-015 local rotation
contract (ADR-007: the occupation matrix co-rotates with the site spin,
the double-counting term is frozen under the MFT variation) is derived
in docs/sympy/siesta_spiral_hubbard_rotation.py and implemented below as
site_spin_rotation / rotate_site_occupations / rotate_hubbard_site_operator;
the matched frozen-rotation response gates are story-016 scope.

VALIDATION STATUS (story 014, against the env-instrumented occ_dump
run): the projector-overlap tables match SIESTA's runtime Ski dump to
max |dS| = 2.8e-6 over all 10004 nonzero entries; the recomputed
occupations match the ground-truth dump to max |dn_d| = 0.076 on the
two chain-end sites (0.0001 on the middle sites), max element deviation
0.263, and reproduce the m_z / t_updn texture.  OPEN ITEM, reported to
the reviewer: the Edftu oracle (mirroring m_pot_dftu.F90:1221-1239,
E = 1/2 Re sum H^U conj(D) + E_correc_dc) still deviates from the
producer's printed value by O(100) eV even when fed the ground-truth
occupations, so a convention beyond the pinned sources (candidate: the
runtime pairing of Dscf components into the energy loop, or an
unexported piece of H_dftu_so_dc) remains to be identified from a
SIESTA-side energy-truth dump.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903

#: Story-015 approved numeric tolerance (relative) for the +U rotation
#: contract gates: W[n'] vs U W[n] U^dag, spin-trace invariance, traces.
HUBBARD_ROTATION_TOLERANCE = 1e-12

_SUPPORTED_SCHEMES = frozenset({"fll_bultmark09"})
_REQUIRED_SOURCE_REFS = frozenset(
    {
        "u_j_stdout",
        "slater_integrals",
        "dc_potential",
        "hubbard_potential",
        "occupation_loop",
        "projector_dump",
    }
)
_SPINOR_LAYOUT = "file_order_D1..D8_diag3k_fix:uu=D1+iD5,dd=D2+iD6,ud=D3-iD4,du=D7+iD8"
_HUBBARD_CONVENTIONS = {
    "vertex_kind": "hubbard_local",
    "scheme": "fll_bultmark09",
    "liechtenstein_params": "(U, J)",
    "projector_method": 2,
    "spinor_layout": _SPINOR_LAYOUT,
    "units": "Ry",
}


class HubbardImportError(ImportError):
    """Refusal base: any +U ingestion inconsistency (fail-closed)."""


class DftuProvenanceError(HubbardImportError):
    """Incomplete, inconsistent or unknown +U provenance."""


class OccupationMatrixError(HubbardImportError):
    """Missing occupation-matrix payload where one is required."""


class HubbardBundleError(HubbardImportError):
    """Partial or inconsistent +U payload on a SpiralBundle."""


class EngineUncertifiedError(HubbardImportError):
    """Two-center engine used before it certified against a known table."""


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DftuProvenance:
    """Complete, self-describing +U provenance (fail-closed on build).

    All energies carried twice: SIESTA-native Ry and eV; consistency is
    enforced at construction time (u_ev == u_ry * RY_TO_EV, ditto J).
    """

    u_ev: float
    j_ev: float
    u_ry: float
    j_ry: float
    projector_rc_bohr: float
    projector_npts: int
    slater_f0_ry: float
    slater_f2_ry: float
    slater_f4_ry: float
    scheme: str = "fll_bultmark09"
    projector_method: int = 2
    projector_l: int = 2
    f4_over_f2: float = 0.625
    source_refs: dict = field(default_factory=dict)
    # subspace identity (no defaults: __post_init__ refuses when absent)
    correlated_species: str | None = None
    principal_n: int | None = None
    site_indices: tuple | None = None
    projector_identity: str | None = None

    def __post_init__(self):
        for name in (
            "u_ev",
            "j_ev",
            "u_ry",
            "j_ry",
            "projector_rc_bohr",
            "slater_f0_ry",
            "slater_f2_ry",
            "slater_f4_ry",
        ):
            val = getattr(self, name)
            if not np.isfinite(val):
                raise DftuProvenanceError(
                    f"DftuProvenance.{name} must be finite, got {val!r}"
                )
        if self.u_ry <= 0 or self.j_ry <= 0:
            raise DftuProvenanceError(
                "DftuProvenance requires U > 0 and J > 0 in Ry; got "
                f"U={self.u_ry}, J={self.j_ry}"
            )
        for name, ry, ev in (("U", self.u_ry, self.u_ev), ("J", self.j_ry, self.j_ev)):
            if abs(ev - ry * RY_TO_EV) > 1e-9 * max(1.0, abs(ev)):
                raise DftuProvenanceError(
                    f"DftuProvenance {name} eV/Ry mismatch: "
                    f"{ev} eV vs {ry} Ry * {RY_TO_EV}"
                )
        if self.scheme not in _SUPPORTED_SCHEMES:
            raise DftuProvenanceError(
                f"unknown double-counting scheme {self.scheme!r}; supported: "
                f"{sorted(_SUPPORTED_SCHEMES)} (refusing to guess)"
            )
        if self.projector_method != 2:
            raise DftuProvenanceError(
                f"projector method {self.projector_method} unsupported; the "
                "story-014 producer used DFTU.ProjectorGenerationMethod 2"
            )
        if self.projector_l != 2:
            raise DftuProvenanceError(
                f"projector l={self.projector_l} unsupported: Slater "
                "integrals are only pinned for the d shell (l=2), see "
                "dftu_specs.f:1749-1758"
            )
        if self.projector_rc_bohr <= 0 or self.projector_npts <= 0:
            raise DftuProvenanceError(
                "projector radial table must have positive cutoff and "
                f"npts; got rc={self.projector_rc_bohr}, "
                f"npts={self.projector_npts}"
            )
        f4of2 = self.f4_over_f2
        f2_expect = 14.0 * self.j_ry / (1.0 + f4of2)
        f4_expect = f4of2 * f2_expect
        for name, got, expect in (
            ("slater_f2_ry", self.slater_f2_ry, f2_expect),
            ("slater_f4_ry", self.slater_f4_ry, f4_expect),
            ("slater_f0_ry", self.slater_f0_ry, self.u_ry),
        ):
            if abs(got - expect) > 1e-10 * max(1.0, abs(expect)):
                raise DftuProvenanceError(
                    f"{name} inconsistent with dftu_specs.f:1749-1758 "
                    f"pinning: got {got}, expected {expect}"
                )
        refs = self.source_refs or {}
        missing = sorted(_REQUIRED_SOURCE_REFS - set(refs))
        if missing:
            raise DftuProvenanceError(
                "incomplete +U provenance: missing source references "
                f"{missing}; every convention must name its pinned "
                "SIESTA source (file:line)"
            )
        if not isinstance(self.correlated_species, str) or not self.correlated_species:
            raise DftuProvenanceError(
                "incomplete +U provenance: correlated_species must name the "
                "correlated species label (e.g. 'Fe'); refusing to guess "
                "which species carries the Hubbard subspace"
            )
        if (
            not isinstance(self.principal_n, int)
            or isinstance(self.principal_n, bool)
            or self.principal_n <= 0
        ):
            raise DftuProvenanceError(
                "incomplete +U provenance: principal_n must be the positive "
                f"principal quantum number of the correlated shell, got "
                f"{self.principal_n!r}"
            )
        if self.site_indices is None:
            raise DftuProvenanceError(
                "incomplete +U provenance: site_indices must list the "
                "correlated site indices; refusing to guess which sites "
                "carry the Hubbard subspace"
            )
        object.__setattr__(
            self, "site_indices", tuple(int(s) for s in self.site_indices)
        )
        if not self.site_indices or any(s < 0 for s in self.site_indices):
            raise DftuProvenanceError(
                "incomplete +U provenance: site_indices must be a non-empty "
                f"tuple of non-negative site indices, got "
                f"{self.site_indices!r}"
            )
        if not isinstance(self.projector_identity, str) or not self.projector_identity:
            raise DftuProvenanceError(
                "incomplete +U provenance: projector_identity must label the "
                "projector source (file path with rc/npts); refusing an "
                "unidentified projector table"
            )

    def to_dict(self) -> dict:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, payload):
        fields = cls.__dataclass_fields__
        if not isinstance(payload, dict):
            raise DftuProvenanceError(
                "DftuProvenance.from_dict requires a mapping payload"
            )
        # No defaults are filled here: every declaration must be present
        # in the payload (fail-closed; spec AC-2).
        _DECLARATIONS = {
            "scheme": (
                "double-counting scheme",
                "accepted: 'fll_bultmark09'",
            ),
            "projector_method": (
                "projector generation method",
                "accepted: 2 only (the construction supports DFTU."
                "ProjectorGenerationMethod 2; the story-014 producer "
                "used 2)",
            ),
            "correlated_species": (
                "correlated species label",
                "accepted: species symbol string (e.g. 'Fe')",
            ),
            "principal_n": (
                "principal quantum number of the correlated shell",
                "accepted: positive int (e.g. 3 for Fe 3d)",
            ),
            "site_indices": (
                "correlated site indices",
                "accepted: iterable of non-negative ints",
            ),
        }
        for key, (what, accepted) in _DECLARATIONS.items():
            if key not in payload:
                raise DftuProvenanceError(
                    f"incomplete +U provenance payload: missing declaration "
                    f"'{key}' ({what}); {accepted}. from_dict never fills "
                    "defaults -- the producer must declare it explicitly"
                )
        payload = dict(payload)
        if "source_refs" in payload:
            payload["source_refs"] = dict(payload["source_refs"])
        missing = [name for name in fields if name not in payload]
        if missing:
            raise DftuProvenanceError(
                "incomplete +U provenance payload: missing fields "
                f"{missing}; refusing partial hubbard metadata "
                "(U/J values come from the stdout 'DFTUprojs with U/J' "
                "lines and the projector table from the dftu_proj header)"
            )
        return cls(**payload)


def spinor_components_to_rho(d8):
    """File-order D1..D8 block -> complex spinor (2, 2) with diag3k fix.

    Mirrors m_occ_proj_dftu.F90:727-741 exactly, including the sign fix
    on the imaginary part of the (up,down) element:
    Dij(1)=uu=D1+iD5, Dij(2)=dd=D2+iD6, Dij(3)=ud=D3-iD4, Dij(4)=du=D7+iD8.
    ``d8`` has file component order (..., 8); returns (..., 2, 2) with
    spin order ((uu, ud), (du, dd)) matching SIESTA's spin box.
    """
    d8 = np.asarray(d8, dtype=float)
    if d8.shape[-1] != 8:
        raise OccupationMatrixError(
            f"spinor density blocks need 8 file components, got shape " f"{d8.shape}"
        )
    uu = d8[..., 0] + 1j * d8[..., 4]
    dd = d8[..., 1] + 1j * d8[..., 5]
    ud = d8[..., 2] - 1j * d8[..., 3]
    du = d8[..., 6] + 1j * d8[..., 7]
    rho = np.empty(d8.shape[:-1] + (2, 2), dtype=complex)
    rho[..., 0, 0] = uu
    rho[..., 0, 1] = ud
    rho[..., 1, 0] = du
    rho[..., 1, 1] = dd
    return rho


# ----------------------------------------------------------------------
# two-center overlap engine (Gaunt x spherical-Bessel), Bohr units
# ----------------------------------------------------------------------


def _Ylm(l, m, theta, phi):
    from scipy.special import sph_harm

    return sph_harm(m, l, phi, theta)


@lru_cache(maxsize=None)
def gaunt3(l1, m1, l2, m2, l3, m3):
    """int dOmega conj(Y_{l1m1}) Y_{l2m2} Y_{l3m3} (sympy, exact)."""
    from sympy.physics.wigner import gaunt as sym_gaunt

    return (-1) ** m1 * complex(sym_gaunt(l1, l3, l2, -m1, m3, m2).evalf())


@lru_cache(maxsize=None)
def real_transform(l):
    """T[mr, a] = int dOmega Yreal_mr conj(Y_a) (SIESTA rlylm ordering)."""
    n = 4 * (2 * l + 1) + 24
    x, wth = np.polynomial.legendre.leggauss(n)
    theta = np.arccos(x)
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    W = np.outer(wth, 2 * np.pi / n)
    T = np.zeros((2 * l + 1, 2 * l + 1), complex)
    for i, m in enumerate(range(-l, l + 1)):
        if m < 0:
            yreal = np.sqrt(2.0) * (-1) ** m * _Ylm(l, -m, TH, PH).imag
        elif m > 0:
            yreal = np.sqrt(2.0) * (-1) ** m * _Ylm(l, m, TH, PH).real
        else:
            yreal = _Ylm(l, 0, TH, PH).real
        for j, a in enumerate(range(-l, l + 1)):
            T[i, j] = np.sum(W * yreal * np.conj(_Ylm(l, a, TH, PH)))
    return T


class Radial:
    """Radial function R(r) on a dense grid covering [0, rc]."""

    def __init__(self, grid, vals, label="radial"):
        self.grid = np.asarray(grid, float)
        self.vals = np.asarray(vals, float)
        self.label = str(label)

    @property
    def rc(self):
        return float(self.grid[-1])


def _on_site_overlap(r1, r2):
    """int r^2 R1 R2 dr by composite 40-node Gauss-Legendre panels.

    Integrated over the SHARED support [0, min(rc1, rc2)]: each radial
    is zero outside its own cutoff, and integrating past it would hit
    cubic-spline extrapolation artifacts.
    """
    spl1 = CubicSpline(r1.grid, r1.vals)
    spl2 = CubicSpline(r2.grid, r2.vals)
    rc = min(r1.rc, r2.rc)
    x, w = np.polynomial.legendre.leggauss(40)
    total = 0.0
    edges = np.linspace(0.0, rc, 31)
    for a, b in zip(edges[:-1], edges[1:]):
        rr = (a + b) / 2 + (b - a) / 2 * x
        ww = (b - a) / 2 * w
        total += float(np.sum(ww * rr * rr * spl1(rr) * spl2(rr)))
    return total


class TwoCenterEngine:
    """Gaunt x spherical-Bessel two-center overlap engine (Bohr units).

    Two-center expansion (complex harmonics):
        S = 8 sum_{l3 m3} i^{l1-l2-l3} Y*_{l3m3}(dhat)
              G(l1m1, l2m2, l3m3) I_{l3}(d)
        I_{l3}(d) = int dk k^2 Rt1(k) Rt2(k) j_{l3}(k d)
        Rt_l(k)   = int dr r^2 R(r) j_l(k r)

    Certification gate: overlap requests fail closed until
    ``self_certify`` succeeded in this process (or the caller passes
    ``certified=True`` explicitly, accepting responsibility).
    """

    def __init__(self, kmax=130.0, kpanel=0.04, ngl_k=10, ngl_r=600):
        self.kmax = float(kmax)
        self.kpanel = float(kpanel)
        self.ngl_k = int(ngl_k)
        self.ngl_r = int(ngl_r)
        self._kx, self._kw = np.polynomial.legendre.leggauss(self.ngl_k)
        edges = np.arange(0.0, self.kmax + 1e-9, self.kpanel)
        nodes, weights = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            nodes.append((a + b) / 2 + (b - a) / 2 * self._kx)
            weights.append((b - a) / 2 * self._kw)
        self.kn = np.concatenate(nodes)
        self.kw = np.concatenate(weights)
        self._r_cache = {}
        self._Rk_cache = {}
        self.certified = False

    def _r_quad(self, radial):
        key = id(radial)
        if key not in self._r_cache:
            spl = CubicSpline(radial.grid, radial.vals)
            x, w = np.polynomial.legendre.leggauss(self.ngl_r)
            rc = radial.rc
            rr = (x + 1) * rc / 2
            ww = w * rc / 2
            self._r_cache[key] = (rr, ww * rr * rr * spl(rr))
        return self._r_cache[key]

    def _Rk(self, radial, l):
        key = (id(radial), l)
        if key not in self._Rk_cache:
            rr, wr = self._r_quad(radial)
            self._Rk_cache[key] = wr @ spherical_jn(l, self.kn[:, None] * rr[None, :]).T
        return self._Rk_cache[key]

    def I3_batch(self, r1, l1, r2, l2, l3, dnorm):
        """I_{l3}(d) for an array of distances."""
        R1 = self._Rk(r1, l1)
        R2 = self._Rk(r2, l2)
        env = self.kw * self.kn**2 * R1 * R2
        out = np.empty(dnorm.shape)
        chunk = max(1, int(4e6 / len(self.kn)))
        flat = dnorm.ravel()
        for s in range(0, flat.size, chunk):
            dc = flat[s : s + chunk]
            J = spherical_jn(l3, self.kn[:, None] * dc[None, :])
            out.ravel()[s : s + chunk] = env @ J
        return out

    def overlap_block(self, r1, l1, r2, l2, dvecs, certified=None):
        """Real-harmonic overlap matrices, shape (2l1+1, 2l2+1, nd).

        Row/col index order is m = -l..l (SIESTA rlylm convention).
        Refuses to run before certification unless ``certified=True``.
        """
        if certified is not True and not self.certified:
            raise EngineUncertifiedError(
                "two-center overlap engine is not certified: run "
                "self_certify against a known overlap table (or pass "
                "certified=True explicitly) before requesting overlaps"
            )
        dvecs = np.atleast_2d(np.asarray(dvecs, float))
        dn = np.linalg.norm(dvecs, axis=1)
        n1, n2, nd = 2 * l1 + 1, 2 * l2 + 1, len(dn)
        S = np.zeros((n1, n2, nd), complex)
        onsite = dn < 1e-9
        if onsite.any():
            val = _on_site_overlap(r1, r2)
            if l1 == l2:
                idx = np.arange(n1)
                for o in np.flatnonzero(onsite):
                    S[idx, idx, o] = val
        off = np.flatnonzero(~onsite)
        if off.size:
            doff = dn[off]
            th = np.arccos(np.clip(dvecs[off, 2] / doff, -1.0, 1.0))
            ph = np.arctan2(dvecs[off, 1], dvecs[off, 0])
            for l3 in range(0, l1 + l2 + 1):
                if (l1 + l2 + l3) % 2:
                    continue
                I3 = self.I3_batch(r1, l1, r2, l2, l3, doff)
                for m3 in range(-l3, l3 + 1):
                    amp = (
                        8.0
                        * (1j) ** (l1 - l2 - l3)
                        * np.conj(_Ylm(l3, m3, th, ph))
                        * I3
                    )
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            g = gaunt3(l1, m1, l2, m2, l3, m3)
                            if abs(g) < 1e-15:
                                continue
                            S[m1 + l1, m2 + l2, off] += amp * g
        T1 = real_transform(l1)
        T2 = real_transform(l2)
        Sreal = np.einsum("ma,abn->mbn", np.conj(T1), S)
        Sreal = np.einsum("mbn,cb->mcn", Sreal, T2)
        if np.max(np.abs(Sreal.imag)) > 1e-8:
            raise HubbardImportError(
                f"non-real overlap block {r1.label}/{r2.label}: "
                f"max imag {np.max(np.abs(Sreal.imag)):.2e}"
            )
        return Sreal.real

    def self_certify(self, entries, max_abs_error=1e-10):
        """Certify against a known overlap table before production use.

        ``entries`` is a sequence of
        ``(radial1, l1, m1, radial2, l2, m2, dvec, S_ref)``.
        Returns the max abs residual; flips the certified gate on success.
        """
        if not entries:
            raise EngineUncertifiedError(
                "self_certify requires a non-empty overlap table"
            )
        groups = {}
        radials = {}
        for r1, l1, m1, r2, l2, m2, dvec, ref in entries:
            key = (id(r1), l1, id(r2), l2)
            groups.setdefault(key, []).append((m1, m2, dvec, ref))
            radials[(id(r1), l1, 1)] = r1
            radials[(id(r2), l2, 2)] = r2
        worst = 0.0
        worst_key = None
        for (rid1, l1, rid2, l2), items in groups.items():
            r1 = radials[(rid1, l1, 1)]
            r2 = radials[(rid2, l2, 2)]
            dvecs = np.array([it[2] for it in items])
            blocks = self.overlap_block(r1, l1, r2, l2, dvecs, certified=True)
            for k, (m1, m2, _, ref) in enumerate(items):
                got = float(blocks[m1 + l1, m2 + l2, k])
                err = abs(got - float(ref))
                if err > worst:
                    worst = err
                    worst_key = (
                        r1.label,
                        l1,
                        m1,
                        r2.label,
                        l2,
                        m2,
                        dvec,
                        float(ref),
                        got,
                    )
        if worst > max_abs_error:
            raise EngineUncertifiedError(
                f"two-center engine certification failed: max |err| "
                f"{worst:.3e} > {max_abs_error:.1e} at {worst_key}"
            )
        self.certified = True
        return worst


# ----------------------------------------------------------------------
# vee 4-center tensor
# ----------------------------------------------------------------------


def vee_tensor(f0_ry, f2_ry, f4_ry, l=2):
    """Real-harmonic 4-center on-site repulsion vee(m, m'', m', m''').

    SIESTA rlylm ordering (m index -l..l).  Angular factor per
    m_vee_integrals.F90 ee_integrals_cmplx: a_k = (4pi/(2k+1)) sum_q
    <lm|Y_kq|lm'> <lm''|Y*_{kq}|lm'''>; the complex-harmonic tensor is
    transformed to the real-harmonic basis exactly like
    ee_4index_int_real.  Ry units.
    """
    f0_ry = float(f0_ry)
    f2_ry = float(f2_ry)
    f4_ry = float(f4_ry)
    slater = {0: f0_ry, 2: f2_ry, 4: f4_ry}
    n = 2 * l + 1
    T = real_transform(l)
    vc = np.zeros((n, n, n, n), dtype=complex)
    ms = list(range(-l, l + 1))
    for ia, a in enumerate(ms):
        for ic, c in enumerate(ms):
            for k in range(0, 2 * l + 1, 2):
                pref = 4.0 * np.pi / (2 * k + 1)
                for q in range(-k, k + 1):
                    g1 = gaunt3(l, a, k, q, l, c)
                    if abs(g1) < 1e-15:
                        continue
                    for ib, b in enumerate(ms):
                        for idd, dd in enumerate(ms):
                            # <l b|Y*_{kq}|l dd> = (-1)^q int conj(Y_b) Y_{k,-q} Y_dd
                            g2 = ((-1) ** q) * gaunt3(l, b, k, -q, l, dd)
                            if abs(g2) < 1e-15:
                                continue
                            vc[ia, ib, ic, idd] += pref * g1 * g2 * slater[k]
    # transform (complex -> real harmonics): <ij|V|kl>
    v = np.einsum("ia,jb,ck,ld,abkd->ijcl", np.conj(T), np.conj(T), T, T, vc)
    if np.max(np.abs(v.imag)) > 1e-10:
        raise HubbardImportError("vee tensor acquired an imaginary part")
    return v.real.copy()


# ----------------------------------------------------------------------
# occupations
# ----------------------------------------------------------------------


def recompute_occupations(dscf_rows, dscf_cols, dscf_comp8, s_proj_pao, prov=None):
    """Per-site complex 5x5 occupation matrices per spin sigma.

    Mirrors m_occ_proj_dftu.F90:681-741: n(m,m') = sum_{mu,nu}
    rho_{mu nu} S(m,mu) S(m',nu), summed over ALL supercell sparse
    entries (both orderings appear in the stored pair list).

    Parameters
    ----------
    dscf_rows, dscf_cols : int arrays
        Sparse spinor density matrix: unit-cell row orbital, supercell
        column orbital (sisl column layout, 0-based).
    dscf_comp8 : float array (nnz, 8)
        FILE-order spinor density components D1..D8; the diag3k sign
        fix is applied here (never before).
    s_proj_pao : float array (nsites, 5, no_s)
        Dense projector-overlap tables S(m, js) per correlated site.
    prov : DftuProvenance, optional
        Required to be present and valid (refusal otherwise); carried
        for provenance discipline.

    Returns
    -------
    occupations : complex array (nsites, 4, 5, 5)
        Spin order (uu, dd, ud, du) matching SIESTA's Dij(1..4).
    """
    if prov is None:
        raise DftuProvenanceError(
            "recompute_occupations requires valid +U provenance; refusing "
            "to assemble an occupation matrix without it"
        )
    if dscf_rows is None or dscf_cols is None or dscf_comp8 is None:
        raise OccupationMatrixError(
            "occupation matrix absent: the spinor density sparse payload "
            "(rows, cols, 8 components) is required and was None"
        )
    if s_proj_pao is None:
        raise OccupationMatrixError(
            "occupation matrix absent: the projector-overlap table "
            "S(proj, PAO) is required and was None"
        )
    rows = np.asarray(dscf_rows, dtype=int)
    cols = np.asarray(dscf_cols, dtype=int)
    comp = np.asarray(dscf_comp8, dtype=float)
    if comp.shape != (len(rows), 8):
        raise OccupationMatrixError(
            f"density components shape {comp.shape} inconsistent with "
            f"{len(rows)} sparse entries"
        )
    sp = np.asarray(s_proj_pao, dtype=float)
    if sp.ndim != 3 or sp.shape[1] != 5:
        raise OccupationMatrixError(
            f"projector overlap table must be (nsites, 5, no_s), got " f"{sp.shape}"
        )
    # complex spinor density per sparse entry, diag3k fix applied
    uu = comp[:, 0] + 1j * comp[:, 4]
    dd = comp[:, 1] + 1j * comp[:, 5]
    ud = comp[:, 2] - 1j * comp[:, 3]
    du = comp[:, 6] + 1j * comp[:, 7]
    norb = int(rows.max()) + 1
    no_s = sp.shape[2]
    if no_s % norb:
        raise OccupationMatrixError(
            f"supercell size {no_s} not a multiple of {norb} orbitals"
        )
    occ = np.zeros((sp.shape[0], 4, 5, 5), dtype=complex)
    for site in range(sp.shape[0]):
        P_sup = sp[site]
        P_cell = P_sup[:, :norb]  # isc = 0 block (js < norb)
        for sigma, vals in enumerate((uu, dd, ud, du)):
            D = np.zeros((norb, no_s), dtype=complex)
            D[rows, cols] = vals
            occ[site, sigma] = P_cell @ D @ P_sup.T
    return occ


# ----------------------------------------------------------------------
# Hubbard potential + double counting
# ----------------------------------------------------------------------


def assemble_hubbard_operator(occupations, prov):
    """Site-local Hubbard part V_U^{sigma} and FLL dc potential.

    Mirrors m_pot_dftu.F90:1057-1141 (Hubbard) and :974-1035 (dc),
    with direct/fock splitting vee(m,m'',m',m''') / vee(m,m''',m'',m').

    Returns
    -------
    (v_hubbard, v_dc, conventions) : tuple
        ``v_hubbard``, ``v_dc``: complex arrays (nsites, 4, 5, 5),
        spin order (uu, dd, ud, du), Ry units.  ``conventions``: dict
        naming every convention (vertex kind, scheme, projector method,
        spinor layout, units).
    """
    if prov is None:
        raise DftuProvenanceError(
            "assemble_hubbard_operator requires valid +U provenance"
        )
    occ = np.asarray(occupations, dtype=complex)
    if occ.ndim != 4 or occ.shape[1:] != (4, 5, 5):
        raise OccupationMatrixError(
            f"occupations must be (nsites, 4, 5, 5), got {occ.shape}"
        )
    vee = vee_tensor(
        prov.slater_f0_ry, prov.slater_f2_ry, prov.slater_f4_ry, l=prov.projector_l
    )
    u_ry, j_ry = prov.u_ry, prov.j_ry
    nsites = occ.shape[0]
    v_hub = np.zeros((nsites, 4, 5, 5), dtype=complex)
    v_dc = np.zeros((nsites, 4, 5, 5), dtype=complex)
    for site in range(nsites):
        n = occ[site]  # (4, 5, 5) sigma = uu, dd, ud, du
        n_uu, n_dd, n_ud, n_du = n[0], n[1], n[2], n[3]
        n_tot = np.trace(n_uu) + np.trace(n_dd)
        for m in range(5):
            for mp in range(5):
                for m2 in range(5):
                    for m3 in range(5):
                        # Certified against SIESTA's own E split (env
                        # DFTU_DUMP_E on m_pot_dftu.F90, build_gcc13):
                        # W(m,m') = sum vee(m,m'',m',m''') n(m'',m''') —
                        # vee slots 2,3 carry the contracted occupation
                        # pair and the AO element's (m, m').  The
                        # bra-ket-looking alternative vee(m,m',m'',m''')
                        # gives E_HUB +39.4 eV vs SIESTA's +188.73 eV.
                        direct = vee[m, m2, mp, m3]
                        fock = vee[m, m3, m2, mp]
                        v_hub[site, 0, m, mp] += (
                            direct * (n[0][m2, m3] + n[1][m2, m3]) - fock * n[0][m2, m3]
                        )
                        v_hub[site, 1, m, mp] += (
                            direct * (n[0][m2, m3] + n[1][m2, m3]) - fock * n[1][m2, m3]
                        )
                        v_hub[site, 2, m, mp] += -fock * n[2][m2, m3]
                        v_hub[site, 3, m, mp] += -fock * n[3][m2, m3]
        dc_uu = -(u_ry * (n_tot - 0.5) - j_ry * (np.trace(n_uu) - 0.5))
        dc_dd = -(u_ry * (n_tot - 0.5) - j_ry * (np.trace(n_dd) - 0.5))
        dc_ud = j_ry * np.trace(n_ud)
        dc_du = j_ry * np.trace(n_du)
        for m in range(5):
            v_dc[site, 0, m, m] = dc_uu
            v_dc[site, 1, m, m] = dc_dd
            v_dc[site, 2, m, m] = dc_ud
            v_dc[site, 3, m, m] = dc_du
    conventions = dict(_HUBBARD_CONVENTIONS)
    conventions["scheme"] = prov.scheme
    conventions["projector_method"] = prov.projector_method
    return v_hub, v_dc, conventions


def edftu_oracle(occupations, v_hubbard, v_dc, rho_local=None, prov=None):
    """Edftu reconstruction in eV, mirroring m_pot_dftu.F90:1221-1253.

    E = 1/2 Re [ Tr(n_uu W_uu) + Tr(n_dd W_dd) + Tr(n_ud W_du)
                 + Tr(n_du W_ud) ]
        + sum_I 0.25 (U - J) n_corr(I)

    The ud/du channels cross exactly as in the source's H fill
    (m_pot_dftu.F90:1200-1214: H(:,3) stores the du-channel potential,
    paired with the ud density component in the E loop) and
    write_subs.F:327 prints E_dftu_so + E_correc_dc as "siesta: Edftu".
    Certified against the pinned build's own energy split on the fe4 +U
    probe (DFTU_DUMP_E): E_HUB +188.733, E_DC -196.905 eV.

    ``rho_local`` optionally supplies an alternative local occupation
    payload for the trace (same layout as ``occupations``); when given
    it is used instead of ``occupations`` and the two must agree to
    1e-8, otherwise the refusal is raised.
    """
    if prov is None:
        raise DftuProvenanceError("edftu_oracle requires +U provenance")
    occ = np.asarray(occupations, dtype=complex)
    if rho_local is not None:
        rl = np.asarray(rho_local, dtype=complex)
        if rl.shape != occ.shape:
            raise OccupationMatrixError(
                f"rho_local shape {rl.shape} does not match occupations " f"{occ.shape}"
            )
        if np.max(np.abs(rl - occ)) > 1e-8:
            raise OccupationMatrixError(
                "rho_local disagrees with the occupation payload beyond "
                "1e-8; refusing ambiguous Edftu oracle"
            )
    vh = np.asarray(v_hubbard, dtype=complex)
    vd = np.asarray(v_dc, dtype=complex)
    e_ry = 0.0
    for site in range(occ.shape[0]):
        # Source-faithful channel pairing (m_pot_dftu.F90:1200-1214 +
        # :1232-1240): the E loop pairs the du-channel potential with
        # the ud density component and vice versa.
        for sigma, vchan in ((0, 0), (1, 1), (2, 3), (3, 2)):
            e_ry += 0.5 * np.real(
                np.trace(occ[site, sigma] @ (vh[site, vchan] + vd[site, vchan]))
            )
        n_corr = np.trace(occ[site, 0]) + np.trace(occ[site, 1])
        e_ry += 0.25 * (prov.u_ry - prov.j_ry) * n_corr
    return float(np.real(e_ry) * RY_TO_EV)


# ----------------------------------------------------------------------
# story-015: local rotation contract (ADR-007, derivation-first)
# ----------------------------------------------------------------------


def site_spin_rotation(axis, angle):
    """Local spin rotation U_s = exp(-i angle sigma.n_hat / 2) (2x2).

    Right-handed active rotation about the unit axis ``n_hat``; a zero
    angle is exactly the identity.  Refuses a zero or non-finite axis.
    """
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0.0:
        raise HubbardImportError("rotation axis must be a nonzero finite 3-vector")
    n = axis / np.linalg.norm(axis)
    half = 0.5 * float(angle)
    c, s = np.cos(half), np.sin(half)
    kx, ky, kz = n
    return np.array(
        [
            [c - 1j * s * kz, -1j * s * (kx - 1j * ky)],
            [-1j * s * (kx + 1j * ky), c + 1j * s * kz],
        ]
    )


def rotate_site_occupations(occupations, site, *, axis=(0.0, 0.0, 1.0), angle=0.0):
    """Co-rotate ONE site's spinor occupation blocks (contract item (i)).

    ``N' = (U_s (x) I_orb) N (U_s^dag (x) I_orb)`` on the site's
    (4, 5, 5) channels (uu, dd, ud, du).  The full spinor trace
    ``Tr N`` is invariant while the single channel traces generally are
    NOT (story-015 derivation, section [A] — the algebraic reason the
    double-counting freeze is a contract, not a consequence).  Every
    other site is returned bit-exactly unchanged; a zero angle is an
    exact no-op.
    """
    occ = np.asarray(occupations, dtype=complex)
    if occ.ndim != 4 or occ.shape[1:] != (4, 5, 5):
        raise OccupationMatrixError(
            f"occupations must be (nsites, 4, 5, 5), got {occ.shape}"
        )
    site = int(site)
    if not 0 <= site < occ.shape[0]:
        raise HubbardImportError(
            f"rotation requested for unknown site index {site}; the payload "
            f"holds {occ.shape[0]} correlated sites"
        )
    u2 = site_spin_rotation(axis, angle)
    n = np.zeros((2, 2, 5, 5), dtype=complex)
    n[0, 0] = occ[site, 0]
    n[1, 1] = occ[site, 1]
    n[0, 1] = occ[site, 2]
    n[1, 0] = occ[site, 3]
    n_rot = np.einsum("ia,abpq,jb->ijpq", u2, n, u2.conj())
    out = occ.copy()
    out[site, 0] = n_rot[0, 0]
    out[site, 1] = n_rot[1, 1]
    out[site, 2] = n_rot[0, 1]
    out[site, 3] = n_rot[1, 0]
    return out


# MathDoc: tb2j/siesta-spiral-hubbard-rotation
def rotate_hubbard_site_operator(
    occupations, prov, site, *, axis=(0.0, 0.0, 1.0), angle=0.0
):
    """Story-015 contract: finite local rotation of one +U vertex (ADR-007).

    The site's occupation matrix co-rotates, the FLL Hubbard part is
    reassembled from the co-rotated occupations (certified vee slot
    order), and the double-counting potential is FROZEN bit-exactly at
    the converged reference value (MFT: the density is frozen, so
    delta V_dc = 0 exactly while delta V_U != 0; the counterfactual
    rebuild drift is recorded in the report).  The story-006
    rotation-legitimacy discipline is applied to the +U vertex: the
    Hubbard contribution transforms per the derived contract
    (``W[n'] == U W[n] U^dag`` within ``HUBBARD_ROTATION_TOLERANCE``
    relative), the spin-trace (charge-channel) Hubbard part is
    untouched, every other site is bit-exactly unchanged, traces are
    preserved and a zero angle is an exact no-op.

    Returns a dict with ``occupations_rotated`` (nsites, 4, 5, 5),
    ``v_hubbard_rotated`` (nsites, 4, 5, 5), ``v_dc_frozen``
    (nsites, 4, 5, 5, identical to the reference assembly) and a
    ``report`` dict recording every gate value (Ry units).
    """
    if prov is None:
        raise DftuProvenanceError(
            "rotate_hubbard_site_operator requires valid +U provenance"
        )
    occ = np.asarray(occupations, dtype=complex)
    if occ.ndim != 4 or occ.shape[1:] != (4, 5, 5):
        raise OccupationMatrixError(
            f"occupations must be (nsites, 4, 5, 5), got {occ.shape}"
        )
    site = int(site)
    if not 0 <= site < occ.shape[0]:
        raise HubbardImportError(
            f"rotation requested for unknown site index {site}; the payload "
            f"holds {occ.shape[0]} correlated sites"
        )
    v_hub_ref, v_dc_ref, conventions = assemble_hubbard_operator(occ, prov)
    occ_rot = rotate_site_occupations(occ, site, axis=axis, angle=angle)
    v_hub_rot_site, v_dc_rebuilt, _ = assemble_hubbard_operator(
        occ_rot[site : site + 1], prov
    )
    v_hub_rot = v_hub_ref.copy()
    v_hub_rot[site] = v_hub_rot_site[0]

    # covariance gate: reassembled W[n'] vs explicit U W[n] U^dag
    u2 = site_spin_rotation(axis, angle)
    v_ref = np.zeros((2, 2, 5, 5), dtype=complex)
    v_ref[0, 0] = v_hub_ref[site, 0]
    v_ref[1, 1] = v_hub_ref[site, 1]
    v_ref[0, 1] = v_hub_ref[site, 2]
    v_ref[1, 0] = v_hub_ref[site, 3]
    v_conj = np.einsum("ia,abpq,jb->ijpq", u2, v_ref, u2.conj())
    v_conj_channels = np.zeros((4, 5, 5), dtype=complex)
    v_conj_channels[0] = v_conj[0, 0]
    v_conj_channels[1] = v_conj[1, 1]
    v_conj_channels[2] = v_conj[0, 1]
    v_conj_channels[3] = v_conj[1, 0]
    scale = max(float(np.max(np.abs(v_ref))), 1e-30)
    covariance = float(np.max(np.abs(v_hub_rot[site] - v_conj_channels))) / scale

    # charge channel (spin trace) of the Hubbard part: SU(2) invariant
    spin_trace_change = float(
        np.max(
            np.abs(
                0.5 * (v_hub_rot[site, 0] + v_hub_rot[site, 1])
                - 0.5 * (v_hub_ref[site, 0] + v_hub_ref[site, 1])
            )
        )
    )
    tr0 = np.trace(occ[site, 0]) + np.trace(occ[site, 1])
    tr1 = np.trace(occ_rot[site, 0]) + np.trace(occ_rot[site, 1])
    trace_dev = float(abs(tr1 - tr0))
    magnetic_change = float(np.sqrt(np.sum(np.abs(v_conj - v_ref) ** 2)))
    untouched = all(
        np.array_equal(occ_rot[other], occ[other])
        for other in range(occ.shape[0])
        if other != site
    )
    zero_noop = float(angle) == 0.0
    # the contract freezes the dc at the reference: the returned dc IS
    # the reference array; record what a rebuild WOULD have moved
    would_drift = float(np.max(np.abs(v_dc_rebuilt[0] - v_dc_ref[site])))
    passed = (
        covariance <= HUBBARD_ROTATION_TOLERANCE
        and spin_trace_change <= HUBBARD_ROTATION_TOLERANCE
        and trace_dev <= HUBBARD_ROTATION_TOLERANCE
        and untouched
        and (magnetic_change > 1e-8 if not zero_noop else magnetic_change == 0.0)
    )
    report = {
        "passed": bool(passed),
        "rotated_site": site,
        "axis": [float(x) for x in np.asarray(axis, dtype=float)],
        "angle_rad": float(angle),
        "hubbard_covariance_relative": covariance,
        "hubbard_covariance_tolerance": HUBBARD_ROTATION_TOLERANCE,
        "magnetic_change": magnetic_change,
        "spin_trace_change_max": spin_trace_change,
        "dc_change_max": 0.0,  # contract: reference dc returned untouched
        "would_be_dc_drift_ry": would_drift,
        "untouched_sites_unchanged": bool(untouched),
        "max_trace_deviation": trace_dev,
        "zero_angle_noop": bool(zero_noop),
        "convention": {
            "adr": "ADR-007",
            "vertex_kind": conventions.get("vertex_kind", "hubbard_local"),
            "scheme": prov.scheme,
            "dc": "frozen_at_reference_traces",
            "hubbard": (
                "occupation co-rotates; W reassembled from the co-rotated "
                "occupations (certified vee slot order)"
            ),
            "units": conventions.get("units", "Ry"),
            "spinor_layout": conventions.get("spinor_layout", _SPINOR_LAYOUT),
        },
    }
    return {
        "occupations_rotated": occ_rot,
        "v_hubbard_rotated": v_hub_rot,
        "v_dc_frozen": v_dc_ref,
        "report": report,
    }


# ----------------------------------------------------------------------
# bundle artifacts
# ----------------------------------------------------------------------


def _canonical_json(obj) -> str:
    """Deterministic JSON text (sorted keys, tuples as lists)."""

    def _jsonable(o):
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    return json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"))


def _digest(arrays: dict, meta: dict | None = None) -> str:
    """sha256 over sorted array bytes plus canonical-JSON metadata.

    ``meta`` (provenance, conventions, note, scalar energies, fixture
    identity) is serialized with :func:`_canonical_json` and appended to
    the hash so provenance substitution is detected, not just array
    tampering.
    """
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        arr = np.ascontiguousarray(arrays[key])
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    if meta is not None:
        h.update(b"meta:")
        h.update(_canonical_json(meta).encode())
    return h.hexdigest()


@dataclass
class HubbardBundle:
    """+U artifact: provenance + occupations + potentials + Edftu check.

    Records explicitly that the stored HSX Hamiltonian already contains
    the +U potential (H_stored = H_xc + H_U); consumers must not add
    V_U to the stored H again (story-015/016 handle the rotated-frame
    operator algebra on xc and hubbard vertices separately).

    ``stored_h`` (optional) is the stored total H over the certified
    correlated subspace (H_stored = H_xc + H_U), i.e. the intrinsic XC
    carrier: it is digested into the certificate so the XC provenance
    audit derives its digest from certified arrays, never from
    caller-supplied bytes (FR-014).  XC PRESENCE IS A PRODUCER FACT:
    the certified SIESTA +U run's stored total H carries the PBE XC
    contribution in every SCF step, so a valid ``stored_h`` always
    comes from the producer HSX through :func:`ingest_dftu_run`
    (never from a Hubbard-only pencil), and it MUST be accompanied by
    ``stored_h_provenance`` naming that producer anchor.
    """

    prov: DftuProvenance
    occupations: np.ndarray  # (nsites, 4, 5, 5) complex
    v_hubbard: np.ndarray  # (nsites, 4, 5, 5) complex, Ry
    v_dc: np.ndarray  # (nsites, 4, 5, 5) complex, Ry
    edftu_ev: float
    edftu_reference_ev: float
    fixture_digest: str = ""
    conventions: dict = field(default_factory=lambda: dict(_HUBBARD_CONVENTIONS))
    stored_h: np.ndarray | None = None
    stored_h_provenance: dict | None = None
    note: str = (
        "H_stored = H_xc + H_U (final_H_f_stress.F): the stored HSX "
        "Hamiltonian already contains the +U potential; do not add "
        "V_U to it again."
    )

    def __post_init__(self):
        if not isinstance(self.prov, DftuProvenance):
            raise HubbardBundleError(
                "HubbardBundle requires a validated DftuProvenance"
            )
        occ = np.asarray(self.occupations)
        vh = np.asarray(self.v_hubbard)
        vd = np.asarray(self.v_dc)
        if occ.shape[1:] != (4, 5, 5):
            raise HubbardBundleError(
                f"HubbardBundle occupations must be (nsites, 4, 5, 5), "
                f"got {occ.shape}"
            )
        if vh.shape != occ.shape or vd.shape != occ.shape:
            raise HubbardBundleError(
                "HubbardBundle potentials must match the occupation shape "
                f"{occ.shape}; got {vh.shape} and {vd.shape}"
            )
        if self.stored_h is not None:
            sh = np.asarray(self.stored_h, dtype=complex)
            if sh.ndim != 2 or sh.shape[0] != sh.shape[1]:
                raise HubbardBundleError(
                    "HubbardBundle stored_h must be a square (n, n) "
                    f"stored-total-H array, got {sh.shape}"
                )
            if not isinstance(self.stored_h_provenance, dict) or (
                self.stored_h_provenance.get("kind") != "producer_hsx_stored_total_H"
            ):
                raise HubbardBundleError(
                    "HubbardBundle stored_h requires stored_h_provenance "
                    "with kind 'producer_hsx_stored_total_H': XC presence "
                    "is a producer fact, anchored to the certified +U "
                    "run's stored total H extracted from the producer HSX "
                    "by ingest_dftu_run; a Hubbard-only pencil is not an "
                    "XC carrier (FR-014)"
                )

    @staticmethod
    def _complex_payload(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=complex)
        return {"re": arr.real.tolist(), "im": arr.imag.tolist()}

    @staticmethod
    def _digest_with(payload: dict, arrays: dict) -> str:
        meta = {
            "prov": payload["prov"],
            "conventions": payload.get("conventions", {}),
            "note": payload.get("note", ""),
            "fixture_digest": payload.get("fixture_digest", ""),
            "edftu_ev": float(payload["edftu_ev"]),
            "edftu_reference_ev": float(payload["edftu_reference_ev"]),
            "stored_h_provenance": payload.get("stored_h_provenance"),
        }
        return _digest(arrays, meta=meta)

    def to_certificate(self) -> dict:
        """JSON-serializable certificate with sha256 over arrays AND
        provenance/conventions/note/fixture identity (canonical JSON),
        so provenance substitution is detected by the digest."""
        payload = {
            "prov": self.prov.to_dict(),
            "occupations": self._complex_payload(self.occupations),
            "v_hubbard": self._complex_payload(self.v_hubbard),
            "v_dc": self._complex_payload(self.v_dc),
            "edftu_ev": float(self.edftu_ev),
            "edftu_reference_ev": float(self.edftu_reference_ev),
            "fixture_digest": self.fixture_digest,
            "conventions": self.conventions,
            "note": self.note,
        }
        arrays = {
            "occupations": np.asarray(self.occupations, dtype=complex),
            "v_hubbard": np.asarray(self.v_hubbard, dtype=complex),
            "v_dc": np.asarray(self.v_dc, dtype=complex),
        }
        if self.stored_h is not None:
            payload["stored_h"] = self._complex_payload(self.stored_h)
            payload["stored_h_provenance"] = dict(self.stored_h_provenance)
            arrays["stored_h"] = np.asarray(self.stored_h, dtype=complex)
        digest = self._digest_with(payload, arrays)
        return {
            "kind": "hubbard_bundle_certificate",
            "version": "1.1",
            "payload_sha256": digest,
            "payload": payload,
        }

    @classmethod
    def from_certificate(cls, cert):
        if (
            not isinstance(cert, dict)
            or cert.get("kind") != "hubbard_bundle_certificate"
        ):
            raise HubbardBundleError(
                "not a hubbard_bundle_certificate; refusing to import"
            )
        payload = cert.get("payload")
        if not isinstance(payload, dict):
            raise HubbardBundleError("certificate payload missing")

        def _complex(key):
            block = payload[key]
            if not isinstance(block, dict) or "re" not in block or "im" not in block:
                raise HubbardBundleError(
                    f"certificate array {key!r} must be a "
                    '{"re": [...], "im": [...]} block'
                )
            return np.asarray(block["re"], dtype=float) + 1j * np.asarray(
                block["im"], dtype=float
            )

        arrays = {
            "occupations": _complex("occupations"),
            "v_hubbard": _complex("v_hubbard"),
            "v_dc": _complex("v_dc"),
        }
        if "stored_h" in payload:
            arrays["stored_h"] = _complex("stored_h")
        digest = cls._digest_with(payload, arrays)
        if digest != cert.get("payload_sha256"):
            raise HubbardBundleError(
                "certificate payload sha256 mismatch: "
                f"expected {cert.get('payload_sha256')}, computed {digest}; "
                "refusing tampered payload"
            )
        prov = DftuProvenance.from_dict(payload["prov"])
        return cls(
            prov=prov,
            occupations=arrays["occupations"],
            v_hubbard=arrays["v_hubbard"],
            v_dc=arrays["v_dc"],
            edftu_ev=payload["edftu_ev"],
            edftu_reference_ev=payload["edftu_reference_ev"],
            fixture_digest=payload.get("fixture_digest", ""),
            conventions=payload.get("conventions", {}),
            stored_h=arrays.get("stored_h"),
            stored_h_provenance=payload.get("stored_h_provenance"),
            note=payload.get("note", ""),
        )


# ----------------------------------------------------------------------
# ingestion from a genuine SIESTA +U run (spec AC-1)
# ----------------------------------------------------------------------


def parse_dftu_stdout(text: str):
    """U, J (Ry) and the last printed 'siesta: Edftu' (eV) from stdout.

    Refuses when the 'DFTUprojs with U/J' declaration lines or the
    converged Edftu line are absent.
    """
    u_ry = j_ry = edftu = None
    for line in text.splitlines():
        if "DFTUprojs with U" in line:
            u_ry = float(line.split("=")[1])
        elif "DFTUprojs with J" in line:
            j_ry = float(line.split("=")[1])
        elif "siesta: Edftu" in line and "=" in line:
            edftu = float(line.split("=")[1])
    if u_ry is None or j_ry is None:
        raise DftuProvenanceError(
            "stdout does not declare 'DFTUprojs with U/J' lines; the U and "
            "J values must come from the producer run (refusing to guess)"
        )
    if edftu is None:
        raise DftuProvenanceError(
            "stdout does not carry a 'siesta: Edftu' line; the Edftu "
            "oracle cannot be certified without the producer's value"
        )
    return u_ry, j_ry, edftu


def parse_dftu_proj_header(path):
    """(npts, delta, rc) from a dftu_proj SIESTA table header."""
    lines = Path(path).read_text().split("\n")
    npts, delta, rc = map(float, lines[2].split("#")[0].split())
    return int(npts), float(delta), float(rc)


def ski_to_sp_proj(ski_path, geometry, site_indices):
    """Dense (nsites, 5, no_s) projector-overlap table from the runtime
    Ski dump (SIESTA's own S_proj-PAO table), matched to sisl column
    indices by (site, in-cell orbital, displacement rounded to 1e-6)
    exactly as the story-014 fixture verification does.  Entries the
    dump omits stay zero, mirroring the zero-initialized runtime table.

    ``site_indices`` are the declared global 1-based atom numbers, as
    printed in the Ski dump's ``ka`` column.  A dump row is placed at
    its declared site's position in ``site_indices``; rows belonging to
    atoms outside the declared correlated subspace (uncorrelated
    atoms) are dropped, so a non-prefix declaration such as ``(2, 4)``
    selects exactly the dumped rows of global atoms 2 and 4.
    """
    from collections import defaultdict

    site_indices = tuple(int(s) for s in site_indices)
    site_pos = {ka: pos for pos, ka in enumerate(site_indices)}

    g = geometry
    no_l = int(g.lasto[-1]) + 1
    if no_l % g.na:
        raise OccupationMatrixError(
            f"{no_l} orbitals over {g.na} atoms is not a whole number of "
            "orbitals per atom"
        )
    norb_pa = no_l // g.na
    cell = np.asarray(g.cell) / BOHR_TO_ANG
    pos = np.asarray(g.xyz) / BOHR_TO_ANG
    sc_off = np.asarray(g.sc_off, dtype=int)
    sp = np.zeros((len(site_indices), 5, no_l * len(sc_off)))
    by_site = {}
    for line in Path(ski_path).read_text().split("\n"):
        p = line.split()
        if len(p) != 8:
            continue
        ka, mp, iono = int(p[0]), int(p[1]), int(p[2])
        x, y, z, s = (float(v) for v in p[4:8])
        # ka is the global 1-based atom number from the dump; only rows
        # of DECLARED correlated atoms are kept, the rest belong to
        # uncorrelated atoms and are dropped.
        isite = site_pos.get(ka)
        if isite is None:
            continue
        ia = ka - 1  # 0-based position for geometry lookup
        if not 0 <= ia < g.na:
            raise OccupationMatrixError(
                f"Ski dump site index {ka} outside the cell's {g.na} atoms"
            )
        if ia not in by_site:
            lk = defaultdict(list)
            for iscell in range(len(sc_off)):
                nsc = tuple(int(v) for v in sc_off[iscell])
                sc = g.sc_index(nsc)
                for ja in range(g.na):
                    for j in range(norb_pa):
                        dvec = tuple(
                            np.round(
                                np.asarray(nsc, float) @ cell + pos[ja] - pos[ia],
                                6,
                            )
                        )
                        lk[(ja, j, dvec)].append(ja * norb_pa + j + no_l * sc)
            by_site[ia] = lk
        cands = by_site[ia].get(
            (
                ((iono - 1) % no_l) // norb_pa,
                (iono - 1) % norb_pa,
                tuple(np.round((x, y, z), 6)),
            ),
            [],
        )
        for js in cands:
            if sp[isite, mp - 1, js] == 0.0:
                sp[isite, mp - 1, js] = s
    return sp


def _bundle_from_hs_geometry(hsx_sile, stdout_name, proj_name):
    """SpiralBundle (vector path, q=0) from a sisl-read SIESTA HSX.

    The stored H is mapped through the pinned file packing
    ``H1=H11, H2=H22, H3=Re H12, H4=-Im H12``; a significant imaginary
    diagonal (D5/D6, i.e. a non-spiral SO Hamiltonian) is refused
    rather than silently truncated.

    Fermi level: sisl's ``read_hamiltonian`` shifts H by ``-Ef`` on
    read (``hsxSileSiesta._r_hamiltonian_v1``).  The bundle declares
    ``fermi_shift="none"``, so the shift is undone here by adding the
    Fermi level back (read via ``read_fermi_level``); the stored H is
    therefore the unshifted producer H.  When the file carries no
    usable Ef (sisl's sentinel), it reads as 0 and the undo is a
    no-op.
    """
    from TB2J.interfaces.siesta_spiral import BUNDLE_VERSION, SpiralBundle

    g = hsx_sile.read_geometry()
    hamiltonian = hsx_sile.read_hamiltonian()
    # sisl returns None for HSX version 0 files (no Ef stored, and the
    # reader leaves H unshifted) - treat as zero so the undo is a no-op.
    ef_raw = hsx_sile.read_fermi_level()
    Ef = float(ef_raw) if ef_raw is not None else 0.0
    if Ef != 0.0:
        # undo sisl's H.shift(-Ef) so the stored H is the raw producer H
        # (shift modifies the Hamiltonian in place)
        hamiltonian.shift(Ef)
    hcsr = hamiltonian._csr
    scsr = hsx_sile.read_overlap()._csr
    no_l = int(g.lasto[-1]) + 1
    if hcsr.shape[0] != no_l or scsr.shape[0] != no_l:
        raise HubbardBundleError(
            f"HSX sparse shape {hcsr.shape[0]} does not match the "
            f"geometry orbital count {no_l}"
        )
    hcomp = np.array(hcsr._D[:, :8], dtype=float)
    hcomp[:, 3] = -hcomp[:, 3]  # back to raw file order (diag3k fix)
    scale = max(1.0, float(np.abs(hcomp[:, :4]).max()))
    if np.abs(hcomp[:, 4:6]).max() > 1e-8 * scale:
        raise HubbardBundleError(
            "HSX Hamiltonian carries a significant imaginary diagonal "
            "(file components D5/D6); that is a non-spiral SO reference "
            "and cannot be declared under the pinned spiral packing "
            "(refusing silent truncation)"
        )
    if not (
        len(hcsr.ptr) == len(scsr.ptr)
        and bool((hcsr.ptr == scsr.ptr).all())
        and bool((hcsr.col == scsr.col).all())
    ):
        raise HubbardBundleError(
            "HSX Hamiltonian and overlap sparse patterns differ; refusing "
            "to pair blocks positionally"
        )
    norb_pa = no_l // g.na
    cell = np.asarray(g.cell) / BOHR_TO_ANG
    pos = np.asarray(g.xyz) / BOHR_TO_ANG
    sc_off = np.asarray(g.sc_off, dtype=int)
    sparse = {}
    overlap = {}
    for io in range(no_l):
        for ptr in range(scsr.ptr[io], scsr.ptr[io + 1]):
            js = int(scsr.col[ptr])
            iscell, o = divmod(js, no_l)
            nsc = tuple(int(v) for v in sc_off[iscell])
            s_val = float(scsr._D[ptr, 0])
            c = hcomp[ptr]
            comps = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            key = (io, o, nsc)
            if key in sparse:
                if not np.allclose(sparse[key], comps, atol=1e-10):
                    raise HubbardBundleError(
                        f"duplicate HSX key {key} with inconsistent blocks"
                    )
                continue
            sparse[key] = comps
            overlap[key] = s_val
    orb_pos = np.zeros((no_l, 3))
    for ia in range(g.na):
        first = int(g.firsto[ia])
        orb_pos[first : first + norb_pa] = pos[ia]
    bundle = SpiralBundle(
        version=BUNDLE_VERSION,
        norb=no_l,
        ncell=len(sc_off),
        q=(0.0, 0.0, 0.0),
        spin_packing="H4=-ImH12",
        phase_convention="q_half_per_spin_sector",
        hermitization="none_hamiltonian_path",
        fermi_shift="none",
        pitch_sign="qphs_up_exp_minus",
        sparse=sparse,
        overlap=overlap,
        cell=tuple(tuple(float(v) for v in row) for row in cell),
        orbital_positions=tuple(tuple(float(v) for v in p) for p in orb_pos),
        metadata={
            "origin": "ingest_dftu_run",
            "stdout": stdout_name,
            "dftu_proj": proj_name,
        },
    )
    return bundle


def extract_producer_stored_h(hsx_sile, site_indices):
    """Stored total H over the certified correlated subspace (Ry).

    XC presence is a PRODUCER fact (story-014/016): the certified
    SIESTA +U run stores its TOTAL Hamiltonian H_stored = H_xc + H_U in
    the HSX, with the PBE XC contribution present in every SCF step, so
    the correlated-subspace rows/cols of that stored H are the XC
    carrier digested by the FR-014 provenance audit.  This helper reads
    the producer HSX the certified ingestion path already consumes --
    never caller-supplied bytes.

    The pinned file packing is unfolded per sparse entry exactly as in
    ``_bundle_from_hs_geometry``: ``H1=H11, H2=H22, H3=Re H12,
    H4=-Im H12`` (the diag3k sign fix restores the raw file order), and
    sisl's ``read_hamiltonian`` ``-Ef`` shift is undone so the returned
    matrix is the unshifted producer H.  Rows/cols are the correlated
    atoms' in-cell orbital windows; the Gamma-point matrix folds every
    supercell image (unit phase at Gamma).  The two spinor channels are
    packed in the same blocked convention as
    :meth:`SpiralImporter.assemble_pencil`: the up block first, then the
    down block, with ``H12`` in the up-down quadrant.

    Returns a complex ``(2n, 2n)`` array with ``n`` the total number of
    correlated orbitals.
    """
    g = hsx_sile.read_geometry()
    no_l = int(g.lasto[-1]) + 1
    hamiltonian = hsx_sile.read_hamiltonian()
    ef_raw = hsx_sile.read_fermi_level()
    Ef = float(ef_raw) if ef_raw is not None else 0.0
    if Ef != 0.0:
        hamiltonian.shift(Ef)
    hcsr = hamiltonian._csr
    sites = tuple(int(s) for s in site_indices)
    if not sites:
        raise DftuProvenanceError(
            "extract_producer_stored_h requires the declared site_indices"
        )
    didx = []
    norb_site = None
    for ka in sites:
        if not 1 <= ka <= g.na:
            raise DftuProvenanceError(
                f"site_indices entry {ka} is outside the producer cell " f"(1..{g.na})"
            )
        first = int(g.firsto[ka - 1])
        last = int(g.lasto[ka - 1])
        count = last - first + 1
        if norb_site is None:
            norb_site = count
        elif count != norb_site:
            raise DftuProvenanceError(
                "correlated sites carry different orbital counts "
                f"({norb_site} vs {count}); the correlated subspace must "
                "be a set of equivalent atoms"
            )
        didx.extend(range(first, last + 1))
    n = len(didx)
    row_of = {io: mu for mu, io in enumerate(didx)}
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    hD = hcsr._D
    seen = 0
    for io in range(no_l):
        mu = row_of.get(io)
        if mu is None:
            continue
        for ptr in range(hcsr.ptr[io], hcsr.ptr[io + 1]):
            jo = row_of.get(int(hcsr.col[ptr]) % no_l)
            if jo is None:
                continue
            c = hD[ptr]
            h11, h22, h3, h4 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
            h12 = complex(h3, h4)  # raw file order H3=Re H12, H4=Im H12
            h[mu, jo] += h11
            h[mu + n, jo + n] += h22
            h[mu, jo + n] += h12
            h[mu + n, jo] += np.conj(h12)
            seen += 1
    if seen == 0:
        raise HubbardBundleError(
            "the producer HSX carries no sparse entries over the "
            "declared correlated subspace; the XC carrier cannot be "
            "extracted from this export"
        )
    return h


def ingest_dftu_run(hsx, dm, stdout, proj_path, occ_dump=None, **ids):
    """Build (SpiralBundle with hubbard, HubbardBundle) from a genuine
    SIESTA +U export (spec AC-1).

    Parameters
    ----------
    hsx, dm : path
        SIESTA HSX and DM files of the converged +U run (read via sisl).
    stdout : path
        Producer stdout carrying the 'DFTUprojs with U/J' declaration
        lines (Ry) and the converged 'siesta: Edftu' line (last
        occurrence is the reference).
    proj_path : path
        The dftu_proj projector table (header gives npts/rc; data kept
        in the bundle metadata identity).
    occ_dump : path, optional
        Directory with the env-instrumented run's ``*.dftu_ski`` dump
        (SIESTA's own runtime S_proj-PAO table).  Required: recomputing
        the projector overlaps from PAO radial tables would need the
        ion.nc basis files, which a bare export does not carry.
    **ids : required keyword identity declarations
        ``species_label`` (str), ``principal_n`` (int),
        ``site_indices`` (iterable of int) -- REFUSED when absent;
        ``site_indices`` are the correlated atoms' global 1-based atom
        numbers, as printed in the runtime Ski dump's ``ka`` column;
        plus optional ``label`` propagated to the bundle metadata.

    Returns
    -------
    (bundle, hubbard_bundle) : (SpiralBundle, HubbardBundle)
        The spiral bundle carries ``hubbard=provenance``; the Hubbard
        bundle carries occupations, potentials and the Edftu oracle
        value with the producer's printed value as reference.  Its
        ``stored_h`` is the stored total H extracted from the producer
        HSX over the certified correlated subspace (H_stored =
        H_xc + H_U): the XC carrier, anchored via
        ``stored_h_provenance`` to the producer HSX identity and the
        run's Edftu reference.  XC presence is a producer fact delivered
        by this certified ingestion path -- never assembled downstream.
    """
    for req in ("species_label", "principal_n", "site_indices"):
        if ids.get(req) is None:
            raise DftuProvenanceError(
                f"ingest_dftu_run requires the explicit identity kwarg "
                f"'{req}'; the correlated subspace (species, shell n, "
                "site indices) must be declared, never guessed"
            )
    if occ_dump is None:
        raise OccupationMatrixError(
            "ingest_dftu_run requires occ_dump with the instrumented "
            "*.dftu_ski projector-overlap dump; a bare HSX/DM export "
            "does not carry the projector tables"
        )
    occ_dump = Path(occ_dump)
    stdout_path = Path(stdout)
    u_ry, j_ry, edftu_ref = parse_dftu_stdout(stdout_path.read_text())
    npts, _delta, rc = parse_dftu_proj_header(proj_path)
    ski_path = sorted(occ_dump.glob("*.dftu_ski"))
    if not ski_path:
        raise OccupationMatrixError(
            f"no *.dftu_ski dump found under {occ_dump}; the runtime "
            "projector-overlap table is required to recompute the "
            "occupations"
        )
    import sisl

    hsx_sile = sisl.get_sile(str(hsx))
    g = hsx_sile.read_geometry()
    dcsr = sisl.get_sile(str(dm)).read_density_matrix()._csr
    no_l = int(g.lasto[-1]) + 1
    dm_comp = np.array(dcsr._D[:, :8], dtype=float)
    dm_comp[:, 3] = -dm_comp[:, 3]  # back to raw file order (diag3k)
    dm_rows = np.repeat(np.arange(no_l), np.diff(dcsr.ptr))
    dm_cols = dcsr.col.copy()
    sp_proj = ski_to_sp_proj(ski_path[0], g, tuple(int(s) for s in ids["site_indices"]))
    proj_name = Path(proj_path).name
    prov = DftuProvenance(
        u_ev=u_ry * RY_TO_EV,
        j_ev=j_ry * RY_TO_EV,
        u_ry=u_ry,
        j_ry=j_ry,
        projector_rc_bohr=rc,
        projector_npts=npts,
        slater_f0_ry=u_ry,
        slater_f2_ry=14.0 * j_ry / 1.625,
        slater_f4_ry=0.625 * 14.0 * j_ry / 1.625,
        source_refs={
            "u_j_stdout": f"{stdout_path.name}:DFTUprojs lines",
            "slater_integrals": "dftu_specs.f:1749-1758",
            "dc_potential": "m_pot_dftu.F90:974-1035",
            "hubbard_potential": "m_pot_dftu.F90:1057-1141",
            "occupation_loop": "m_occ_proj_dftu.F90:681-741",
            "projector_dump": f"{proj_name} header",
        },
        correlated_species=str(ids["species_label"]),
        principal_n=int(ids["principal_n"]),
        site_indices=tuple(int(s) for s in ids["site_indices"]),
        projector_identity=f"{proj_name}:rc={rc:.6f},npts={npts}",
    )
    occ = recompute_occupations(dm_rows, dm_cols, dm_comp, sp_proj, prov)
    vh, vd, conv = assemble_hubbard_operator(occ, prov)
    e = edftu_oracle(occ, vh, vd, prov=prov)
    bundle = _bundle_from_hs_geometry(hsx_sile, stdout_path.name, proj_name)
    bundle.hubbard = prov
    bundle.validate()
    # XC presence is a PRODUCER fact: the XC carrier is the stored total
    # H read from THIS producer HSX (H_stored = H_xc + H_U; PBE XC in
    # every SCF step), extracted here over the certified correlated
    # subspace -- never a Hubbard-only pencil assembled downstream.
    sites = tuple(int(s) for s in ids["site_indices"])
    stored_h = extract_producer_stored_h(hsx_sile, sites)
    hsx_name = Path(hsx).name
    stored_h_provenance = {
        "kind": "producer_hsx_stored_total_H",
        "anchor": (
            "XC presence is a producer fact: the certified +U run's "
            "stored total H (read from the producer HSX by "
            "ingest_dftu_run) carries the PBE XC contribution in every "
            "SCF step; H_stored = H_xc + H_U (story-014)"
        ),
        "hsx": hsx_name,
        "extraction": (
            "extract_producer_stored_h: certified correlated-subspace "
            "rows/cols, Gamma-folded over supercell images; spinor "
            "channels stacked [up; down], H12 = H3 + i*H4 (raw file "
            "order after the diag3k fix); sisl's -Ef shift undone"
        ),
        "site_indices": list(sites),
        "edftu_reference_ev": float(edftu_ref),
    }
    hubbard_bundle = HubbardBundle(
        prov=prov,
        occupations=occ,
        v_hubbard=vh,
        v_dc=vd,
        edftu_ev=e,
        edftu_reference_ev=edftu_ref,
        conventions=conv,
        fixture_digest="",
        stored_h=stored_h,
        stored_h_provenance=stored_h_provenance,
        note=ids.get(
            "note",
            "Ingested by ingest_dftu_run from a genuine SIESTA +U run; "
            "H_stored = H_xc + H_U (final_H_f_stress.F): do not add V_U "
            "to the stored Hamiltonian again.",
        ),
    )
    return bundle, hubbard_bundle


# ----------------------------------------------------------------------
# story-016: AC-008 gate — matched frozen-rotation checks on the genuine
# +U reference, plus the XC+Hubbard provenance audit (FR-014).
# ----------------------------------------------------------------------

#: AC-008 angle ladder (rad), FIXED AT STORY-016 APPROVAL (2026-10-16).
#: Coarse rungs demonstrate the theta**2 finite-difference truncation
#: regime; the three finest rungs form the approved stable-error window.
AC008_ANGLE_LADDER_RAD = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.025, 0.02)

#: AC-008 budgets, FIXED AT STORY-016 APPROVAL (2026-10-16).  Torque and
#: curvature budgets inherit the story-009/010 approved values
#: (TB2J.interfaces.siesta_spiral_response.TORQUE_BUDGET_EV_PER_RAD /
#: CURVATURE_BUDGET_EV_PER_RAD2); the contract-energy budget pins the
#: story-015 rotation contract at the frozen-problem energy level.
#: Measured worst values on the genuine fe4 +U reference: torque mismatch
#: 1.2e-7 eV/rad at theta=0.5 decaying as theta**2, curvature mismatch
#: 6.5e-4 eV/rad2 at the plateau angles, contract-vs-conjugation energy
#: delta 1.7e-13 eV, hubbard covariance 5.8e-16 relative.
AC008_TORQUE_BUDGET_EV_PER_RAD = 1e-3
AC008_CURVATURE_BUDGET_EV_PER_RAD2 = 1e-3
AC008_CONTRACT_ENERGY_BUDGET_EV = 1e-11

#: Stable-error window (rung indices into AC008_ANGLE_LADDER_RAD) where
#: all analytic-vs-finite mismatches, adjacent deltas and roundoff floors
#: sit within the budgets above; measured on the genuine reference.
AC008_STABLE_WINDOW_RUNGS = (4, 5, 6)

#: Packing fidelity budget (eV), FIXED AT STORY-016 APPROVAL.  The 4-component
#: file packing (uu, dd, ud, -Im ud) discards the residual imaginary part of
#: the spin-diagonal channels and stores the du channel implicitly (du =
#: conj(ud) transposed); the certified operator is reproduced from the packed
#: representation to within this measured bound on the genuine reference
#: (measured worst 2.2978e-13 eV, dominated by the discarded Im uu/dd).
AC008_PACKING_FIDELITY_BUDGET_EV = 2.3e-13

#: Packing validity ceiling (eV) for the documented restriction of
#: :func:`hubbard_site_block_components`: the packing is only defined for
#: xz-plane / real-channel textures (real spin-diagonal channels, du =
#: conj-swap of ud).  Textures violating that class lose O(eV) information
#: and are refused outright; the genuine reference sits ~4 orders of
#: magnitude below this ceiling (2.3e-13 eV, AC008_PACKING_FIDELITY_BUDGET_EV).
HUBBARD_PACKING_RESTRICTION_BUDGET_EV = 1e-9


class ProvenanceAuditError(DftuProvenanceError):
    """XC+Hubbard provenance audit refusal (AC-008 / FR-014), fail-closed.

    Carries a stable machine-readable ``reason`` so gate consumers can
    distinguish the refusal classes; the message is the actionable
    diagnostic naming the producer source to re-export or re-certify.
    """

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = str(reason)


def rotation_contract_mathdoc_sha256():
    """sha256 of the +U rotation-contract derivation authority, or None.

    The authority is ``docs/sympy/siesta_spiral_hubbard_rotation.py`` in
    the same source tree as this module.  An installed distribution
    without the docs tree yields ``None``; the audit then fails closed
    (uncertified rotation contract) instead of trusting a stale claim.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "sympy"
        / "siesta_spiral_hubbard_rotation.py"
    )
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xc_source_payload_sha256(payload):
    """Canonical sha256 over an XC source payload (AC-008 / FR-014).

    Accepts numpy arrays, (nested) mappings, sequences, byte strings and
    scalars; digests them in a deterministic canonical byte form so the
    same payload always yields the same digest.  Anything else is
    refused (fail-closed) rather than digested through ``repr``.
    """
    if isinstance(payload, np.ndarray):
        arr = np.ascontiguousarray(payload)
        return hashlib.sha256(
            arr.dtype.str.encode("utf-8")
            + (str(arr.shape) + "|").encode("utf-8")
            + arr.tobytes()
        ).hexdigest()
    if isinstance(payload, bytes):
        return hashlib.sha256(payload).hexdigest()
    if isinstance(payload, str):
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if isinstance(payload, dict):
        h = hashlib.sha256()
        for key in sorted(payload, key=str):
            h.update(f"{key!r}:".encode("utf-8"))
            h.update(xc_source_payload_sha256(payload[key]).encode("utf-8"))
        return h.hexdigest()
    if isinstance(payload, (list, tuple)):
        h = hashlib.sha256()
        for item in payload:
            h.update(xc_source_payload_sha256(item).encode("utf-8"))
        return h.hexdigest()
    if isinstance(payload, (int, float, complex, bool)) or payload is None:
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    raise HubbardBundleError(
        f"xc_source payload of type {type(payload).__name__} cannot be "
        "digested canonically; pass the array/bytes/mapping that carries "
        "the XC contribution"
    )


def build_xc_hubbard_audit_artifact(
    bundle,
    *,
    vertex_kind,
    producer_u_ry,
    producer_j_ry,
    producer_u_j_source,
    xc_source=None,
    rotation_contract_sha256=None,
):
    """Assemble the serializable XC+Hubbard provenance-audit artifact.

    ``bundle`` is the genuine :class:`HubbardBundle`; its full
    certificate (arrays + provenance + conventions, sha256-digested) is
    embedded, alongside the declared vertex kind, the producer stdout
    U/J reference (the substitution anchor) and the rotation-contract
    MathDoc sha256.

    The XC source digest is INTRINSIC (FR-014): it is derived from the
    certificate's own stored-H payload (``bundle.stored_h``, the stored
    total H over the certified subspace, H_stored = H_xc + H_U), never
    from caller-supplied bytes.  The CARRIER is the producer-extracted
    ``stored_h``: XC presence is a producer fact anchored through
    ``ingest_dftu_run`` (the certified ingestion path reads the +U
    run's HSX, where the PBE XC contribution is present in every SCF
    step).  A certificate whose stored-H payload is missing, not
    producer-anchored (i.e. set outside ``ingest_dftu_run`` -- a
    Hubbard-only pencil is exactly a state with no XC contribution) --
    or all-zero is the stripped-XC output and is refused here with the
    ``empty_xc_source`` reason: an artifact with no XC carrier cannot
    pass the audit.  The audit detects POST-PRODUCER stripping or
    substitution of the XC carrier (FR-014); it does not re-derive the
    physics of the producer SCF.

    ``xc_source`` only carries the label of where the XC contribution
    enters the +U total H: ``{"kind": "stored_total_H"}`` (default) or
    ``{"kind": "vertex"}``.  A ``payload`` key is refused -- the digest
    provenance is the certificate's, not the caller's.

    The recorded digest is bound to the certificate digest
    (``xc_digest_binding``), so post-export substitution of the XC
    digest alone is detected by the audit.  Declaring
    ``vertex_kind="xc"`` on a +U run -- or exporting without an
    XC carrier at all -- is exactly the stripped-XC output the audit
    must refuse (FR-014/FR-005).
    """
    sha = rotation_contract_sha256
    if sha is None:
        sha = rotation_contract_mathdoc_sha256()
    if xc_source is None:
        xc_source = {"kind": "stored_total_H"}
    if not isinstance(xc_source, dict):
        raise HubbardBundleError(
            "build_xc_hubbard_audit_artifact xc_source must be a label "
            "dict {'kind': 'stored_total_H'|'vertex'}; the digest is "
            "derived from the certificate's stored-H payload, not from "
            "caller bytes (FR-014)"
        )
    if "payload" in xc_source:
        raise HubbardBundleError(
            "xc_source carries a caller payload; the XC digest is "
            "intrinsic to the certificate's stored-H payload and must "
            "not be presented as caller bytes (FR-014)"
        )
    xc_kind = str(xc_source.get("kind", "stored_total_H"))
    if xc_kind not in ("stored_total_H", "vertex"):
        raise HubbardBundleError(
            f"xc_source kind {xc_kind!r} is not one of 'stored_total_H' "
            "or 'vertex'; identify where the XC contribution enters the "
            "+U total H (FR-014)"
        )
    certificate = bundle.to_certificate()
    stored_h = certificate["payload"].get("stored_h")
    if stored_h is None:
        raise ProvenanceAuditError(
            "empty_xc_source",
            "the certified +U state carries no stored-H payload "
            "(H_stored = H_xc + H_U); an artifact whose certified state "
            "has no XC carrier cannot pass the provenance audit -- "
            "re-export from a genuine +U run that retains the stored "
            "total H (FR-014)",
        )
    # Producer anchor: the carrier must be the stored_h extracted from
    # the producer HSX by the certified ingestion path (ingest_dftu_run
    # sets stored_h_provenance); a Hubbard-only bundle is not an XC
    # carrier.
    carrier_prov = certificate["payload"].get("stored_h_provenance")
    if (
        not isinstance(carrier_prov, dict)
        or carrier_prov.get("kind") != "producer_hsx_stored_total_H"
    ):
        raise ProvenanceAuditError(
            "empty_xc_source",
            "the certified stored-H payload is not producer-anchored: "
            "XC presence must come from the certified +U run's stored "
            "total H extracted by ingest_dftu_run "
            "(stored_h_provenance kind 'producer_hsx_stored_total_H'); "
            "a Hubbard-only pencil carries no XC contribution -- "
            "re-ingest from the producer run (FR-014)",
        )
    stored_h_arr = np.asarray(stored_h["re"], dtype=float) + 1j * np.asarray(
        stored_h["im"], dtype=float
    )
    if stored_h_arr.size == 0 or not np.any(stored_h_arr):
        raise ProvenanceAuditError(
            "empty_xc_source",
            "the certified +U state carries an all-zero stored-H "
            "payload: no XC contribution is present in the certified "
            "total H (H_stored = H_xc + H_U) -- the XC channel was "
            "stripped; re-export from a genuine +U run (FR-014)",
        )
    xc_digest = xc_source_payload_sha256(stored_h_arr)
    binding = hashlib.sha256(
        (certificate["payload_sha256"] + xc_digest).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "xc_hubbard_provenance_audit",
        "version": "1.3",
        "vertex_kind": str(vertex_kind),
        "xc_source": {"kind": xc_kind, "digest": xc_digest},
        "xc_carrier_provenance": {
            "carrier": "producer HSX via ingest_dftu_run",
            "kind": carrier_prov.get("kind"),
            "hsx": carrier_prov.get("hsx"),
            "edftu_reference_ev": carrier_prov.get("edftu_reference_ev"),
            "note": (
                "XC presence is anchored to the certified producer "
                "output via the certified ingestion path; the audit "
                "detects post-producer stripping/substitution (FR-014), "
                "not the physics of the SCF"
            ),
        },
        "xc_digest_binding": binding,
        "certificate": certificate,
        "producer_u_ry": float(producer_u_ry),
        "producer_j_ry": float(producer_j_ry),
        "producer_u_j_source": str(producer_u_j_source),
        "rotation_contract_mathdoc_sha256": sha,
    }


def audit_xc_hubbard_provenance(artifact):
    """Audit the XC+Hubbard provenance of a +U gate artifact (AC-008).

    Passes only on genuine outputs: the declared vertex kind is
    ``xc+hubbard`` with both operator contributions present, the U/J
    values match the producer stdout reference, the projector identity
    is recorded, and the rotation contract is certified against the
    current derivation-authority MathDoc sha256.  Every refusal raises
    :class:`ProvenanceAuditError` with a distinct ``reason``:

    * ``malformed_audit_artifact`` -- not an audit envelope/certificate;
    * ``tampered_certificate_payload`` -- certificate digest mismatch;
    * ``empty_xc_source`` -- the certified state carries no stored-H
      payload, one that is not producer-anchored (kind
      'producer_hsx_stored_total_H', set by ingest_dftu_run), or an
      all-zero one: no XC carrier exists to audit;
    * ``stripped_xc_contribution`` -- missing/malformed XC source
      identity or digest, a digest not matching the recomputed
      stored-H digest, or a digest breaking its export-time binding;
    * ``stripped_hubbard_contribution`` -- vertex kind xc(-only) or
      vanishing Hubbard operator where +U is declared;
    * ``substituted_u_j`` -- U/J differing from the producer reference;
    * ``stripped_projector_identity`` -- unidentified projector table;
    * ``uncertified_rotation_contract`` -- missing or stale MathDoc sha.

    Returns a verdict dict recording every check plus informational
    Edftu oracle/reference values (story-014 keeps the oracle outside
    the gating set; recorded verbatim, never gated).
    """

    def refuse(reason, message):
        raise ProvenanceAuditError(reason, message)

    if not isinstance(artifact, dict) or artifact.get("kind") != (
        "xc_hubbard_provenance_audit"
    ):
        refuse(
            "malformed_audit_artifact",
            "provenance audit requires an 'xc_hubbard_provenance_audit' "
            "envelope; build it with build_xc_hubbard_audit_artifact from "
            "the genuine HubbardBundle",
        )
    certificate = artifact.get("certificate")
    if (
        not isinstance(certificate, dict)
        or certificate.get("kind") != "hubbard_bundle_certificate"
    ):
        refuse(
            "malformed_audit_artifact",
            "audit artifact carries no 'hubbard_bundle_certificate'; the "
            "+U operator, provenance and conventions must be presented "
            "through the digested certificate",
        )
    try:
        checked = HubbardBundle.from_certificate(certificate)
    except HubbardBundleError as exc:
        # An unanchored stored-H payload (no producer XC carrier) is the
        # stripped-XC defect, not a generic tamper.
        if "producer_hsx_stored_total_H" in str(exc):
            refuse(
                "empty_xc_source",
                "the certified stored-H payload is not producer-anchored: "
                "XC presence must come from the certified +U run's stored "
                "total H extracted by ingest_dftu_run "
                "(stored_h_provenance kind 'producer_hsx_stored_total_H'); "
                f"a Hubbard-only pencil carries no XC contribution "
                f"(FR-014) (constructor: {exc})",
            )
        refuse(
            "tampered_certificate_payload",
            f"certificate payload digest mismatch ({exc}); the +U artifact "
            "was modified after export -- re-export from the producer run",
        )
    except DftuProvenanceError as exc:
        # A re-digested certificate can carry a structurally invalid
        # provenance (fail-closed constructor).  Blanking the projector
        # identity is exactly the stripped-identity producer defect.
        if "projector_identity" in str(exc):
            refuse(
                "stripped_projector_identity",
                "certificate provenance carries no projector_identity; the "
                "projector table (file, rc, npts) must be identified for "
                "the occupation reconstruction to be auditable "
                f"(constructor: {exc})",
            )
        refuse(
            "tampered_certificate_payload",
            f"certificate provenance is incomplete or altered ({exc}); "
            "re-export the +U artifact from the producer run",
        )
    prov = checked.prov

    # (0) stripped XC contribution: the XC source identity + content
    # digest must be present, well-formed and bound to this certificate.
    xc_source = artifact.get("xc_source")
    if (
        not isinstance(xc_source, dict)
        or xc_source.get("kind") not in ("stored_total_H", "vertex")
        or not xc_source.get("digest")
    ):
        refuse(
            "stripped_xc_contribution",
            "audit artifact carries no XC source identity/content digest "
            "(xc_source with kind 'stored_total_H'|'vertex' and a sha256 "
            "digest); a +U export without a digested XC contribution is "
            "indistinguishable from an H0+V^U output with the XC vertex "
            "stripped -- re-export with build_xc_hubbard_audit_artifact "
            "(FR-014)",
        )
    xc_digest = str(xc_source["digest"])
    if len(xc_digest) != 64 or any(c not in "0123456789abcdef" for c in xc_digest):
        refuse(
            "stripped_xc_contribution",
            f"xc_source digest {xc_digest!r} is not a sha256 hex digest; "
            "re-export the +U artifact so the XC payload is digested at "
            "export time (FR-014)",
        )
    # The digest provenance is intrinsic: recompute it from the
    # certificate's own stored-H payload and refuse any drift.
    stored_h = certificate["payload"].get("stored_h")
    if stored_h is None:
        refuse(
            "empty_xc_source",
            "the certified +U state carries no stored-H payload "
            "(H_stored = H_xc + H_U): the certified state has no XC "
            "carrier, so the output is indistinguishable from an "
            "H0+V^U export with the XC contribution stripped -- "
            "re-export from a genuine +U run (FR-014)",
        )
    # The carrier itself must be the producer-extracted stored total H:
    # ingest_dftu_run sets stored_h_provenance on the bundle; a payload
    # without it comes from a Hubbard-only pencil (no XC contribution).
    carrier_prov = certificate["payload"].get("stored_h_provenance")
    if (
        not isinstance(carrier_prov, dict)
        or carrier_prov.get("kind") != "producer_hsx_stored_total_H"
    ):
        refuse(
            "empty_xc_source",
            "the certified stored-H payload is not producer-anchored: "
            "XC presence must come from the certified +U run's stored "
            "total H extracted by ingest_dftu_run "
            "(stored_h_provenance kind 'producer_hsx_stored_total_H'); "
            "a Hubbard-only pencil carries no XC contribution -- "
            "re-ingest from the producer run (FR-014)",
        )
    stored_h_arr = np.asarray(stored_h["re"], dtype=float) + 1j * np.asarray(
        stored_h["im"], dtype=float
    )
    if stored_h_arr.size == 0 or not np.any(stored_h_arr):
        refuse(
            "empty_xc_source",
            "the certified +U state carries an all-zero stored-H "
            "payload: no XC contribution is present in the certified "
            "total H -- the XC channel was stripped; re-export from a "
            "genuine +U run (FR-014)",
        )
    intrinsic_digest = xc_source_payload_sha256(stored_h_arr)
    if xc_digest != intrinsic_digest:
        refuse(
            "stripped_xc_contribution",
            "xc_source digest does not match the digest recomputed from "
            "the certificate's stored-H payload (payload sha "
            f"{certificate['payload_sha256'][:12]}...): the XC "
            "contribution was substituted or re-digested after export -- "
            "re-export from the producer run",
        )
    expected_binding = hashlib.sha256(
        (certificate["payload_sha256"] + xc_digest).encode("utf-8")
    ).hexdigest()
    if artifact.get("xc_digest_binding") != expected_binding:
        refuse(
            "stripped_xc_contribution",
            "xc_source digest does not match the binding recorded at "
            "export against this certificate (payload sha "
            f"{certificate['payload_sha256'][:12]}...); the XC "
            "contribution was substituted or re-digested after export -- "
            "re-export from the producer run",
        )

    # (1) stripped Hubbard contribution: the +U run must keep BOTH the
    # XC channel and the Hubbard operator (vertex kind xc+hubbard).
    vertex_kind = artifact.get("vertex_kind")
    w_scale = (
        float(np.max(np.abs(checked.v_hubbard)))
        if np.asarray(checked.v_hubbard).size
        else 0.0
    )
    if vertex_kind != "xc+hubbard":
        refuse(
            "stripped_hubbard_contribution",
            f"declared vertex kind {vertex_kind!r} on a run with U="
            f"{prov.u_ev:.6g} eV / J={prov.j_ev:.6g} eV: a +U output must "
            "declare vertex kind 'xc+hubbard' with the Hubbard operator "
            "channels retained (FR-005/FR-014); re-export without "
            "stripping the Hubbard contribution",
        )
    if prov.u_ry > 0 and w_scale <= 0.0:
        refuse(
            "stripped_hubbard_contribution",
            "certificate declares U > 0 but carries an all-zero Hubbard "
            "operator; the V^U channels were stripped after export",
        )
    if checked.conventions.get("vertex_kind") != "hubbard_local":
        refuse(
            "stripped_hubbard_contribution",
            f"certificate conventions declare vertex_kind "
            f"{checked.conventions.get('vertex_kind')!r} instead of "
            "'hubbard_local'; the certified +U assembly was altered",
        )

    # (2) substituted U/J: the certificate is internally digest-consistent,
    # so substitution is caught against the independent producer reference.
    source = artifact.get("producer_u_j_source")
    if not source:
        refuse(
            "substituted_u_j",
            "audit artifact carries no producer U/J reference source; the "
            "substitution anchor (stdout 'DFTUprojs with U/J' lines) is "
            "required",
        )
    for name, producer, actual in (
        ("U", artifact.get("producer_u_ry"), prov.u_ry),
        ("J", artifact.get("producer_j_ry"), prov.j_ry),
    ):
        if producer is None:
            refuse(
                "substituted_u_j",
                f"audit artifact carries no producer {name} reference; the "
                f"stdout value from {source} is required to detect "
                "substitution",
            )
        if abs(float(producer) - float(actual)) > 1e-12 * max(1.0, abs(float(actual))):
            refuse(
                "substituted_u_j",
                f"certificate {name}={float(actual):.12g} Ry does not match "
                f"the producer declaration {float(producer):.12g} Ry "
                f"({source}); the U/J values were substituted after export",
            )

    # (3) stripped projector identity.
    if not prov.projector_identity:
        refuse(
            "stripped_projector_identity",
            "certificate provenance carries no projector_identity; the "
            "projector table (file, rc, npts) must be identified for the "
            "occupation reconstruction to be auditable",
        )

    # (4) uncertified rotation contract: the pinned MathDoc sha must be
    # present AND match the current derivation authority.
    sha = artifact.get("rotation_contract_mathdoc_sha256")
    if not sha:
        refuse(
            "uncertified_rotation_contract",
            "audit artifact carries no rotation_contract_mathdoc_sha256; "
            "the story-015 contract must be certified against "
            "docs/sympy/siesta_spiral_hubbard_rotation.py before the +U "
            "output can pass the gate (FR-014)",
        )
    computed = rotation_contract_mathdoc_sha256()
    if computed is None:
        refuse(
            "uncertified_rotation_contract",
            "rotation-contract derivation authority "
            "(docs/sympy/siesta_spiral_hubbard_rotation.py) not found in "
            "this source tree; refusing to certify the contract against "
            "an unverifiable sha (fail-closed)",
        )
    if sha != computed:
        refuse(
            "uncertified_rotation_contract",
            f"rotation-contract MathDoc sha {sha} is stale (current "
            f"authority sha {computed}); re-run the derivation authority "
            "and re-export the +U artifact",
        )

    edftu_delta = float(checked.edftu_ev) - float(checked.edftu_reference_ev)
    return {
        "passed": True,
        "vertex_kind": vertex_kind,
        "checks": {
            "certificate_digest": "verified",
            "xc_source_kind": xc_source["kind"],
            "xc_source_digest": xc_digest,
            "xc_digest_intrinsic": True,
            "xc_carrier_producer_anchored": (
                carrier_prov.get("kind") == "producer_hsx_stored_total_H"
            ),
            "xc_carrier_hsx": carrier_prov.get("hsx"),
            "vertex_kind_xc_plus_hubbard": True,
            "hubbard_operator_scale_ry": w_scale,
            "u_ry_matches_producer": True,
            "j_ry_matches_producer": True,
            "projector_identity": prov.projector_identity,
            "rotation_contract_mathdoc_sha256": computed,
        },
        "informational": {
            "edftu_oracle_ev": float(checked.edftu_ev),
            "edftu_reference_ev": float(checked.edftu_reference_ev),
            "edftu_delta_ev": edftu_delta,
            "edftu_gating": (
                "recorded verbatim, never gated (story-014 open item on "
                "the SIESTA-side energy convention)"
            ),
        },
    }


def hubbard_site_block_components(v_hubbard_site, *, energy_scale=RY_TO_EV):
    """(4, 5, 5) channel array -> {(p, q): (h1, h2, h3, h4)} file packing.

    Maps the site's Hubbard operator onto the pinned 4-component spin
    block packing used by SpiralBundle/HSX files (uu, dd, ud, -Im ud),
    scaled to ``energy_scale`` (default eV).  The spin-diagonal channels
    of the certified +U operator are real to machine precision on the
    declared reference (xz-plane texture keeps the occupation channels
    real); the residual imaginary part is bounded by the gate's packing
    fidelity check (``AC008_PACKING_FIDELITY_BUDGET_EV``, measured
    2.2978e-13 eV on the genuine reference).

    Documented restriction: the packing is only defined for xz-plane /
    real-channel textures.  Textures whose spin-diagonal channels carry
    an imaginary part, or whose du channel is not the conj-swap of ud
    (``du = conj(ud).T``), would lose O(eV) information through this
    packing and are refused outright (fail-closed) rather than silently
    truncated.
    """
    w = np.asarray(v_hubbard_site, dtype=complex) * energy_scale
    if w.shape != (4, 5, 5):
        raise HubbardBundleError(f"v_hubbard_site must be (4, 5, 5), got {w.shape}")
    ceiling = HUBBARD_PACKING_RESTRICTION_BUDGET_EV
    im_diag = max(float(np.max(np.abs(w[0].imag))), float(np.max(np.abs(w[1].imag))))
    du_swap_residual = float(np.max(np.abs(w[3] - np.conj(w[2].T))))
    if im_diag > ceiling or du_swap_residual > ceiling:
        raise HubbardBundleError(
            "Hubbard site operator violates the 4-component packing "
            f"restriction: spin-diagonal |Im| = {im_diag:.3e} eV, "
            f"|du - conj(ud).T| = {du_swap_residual:.3e} eV exceed the "
            f"{ceiling:.1e} eV ceiling; the (uu, dd, ud, -Im ud) packing "
            "is only valid for xz-plane / real-channel textures -- store "
            "the full complex operator instead of packing it"
        )
    return {
        (p, q): (
            float(w[0, p, q].real),
            float(w[1, p, q].real),
            float(w[2, p, q].real),
            float(-w[2, p, q].imag),
        )
        for p in range(5)
        for q in range(5)
    }


def hubbard_site_block_unpack(components, *, energy_scale=RY_TO_EV):
    """Inverse of :func:`hubbard_site_block_components` for the declared
    texture class: rebuild the (4, 5, 5) complex site operator (in
    ``energy_scale`` units, default eV) from the packed components,
    restoring ud = h3 - i*h4 and the implicit du channel du =
    conj(ud).T.  Valid only where the packing restriction holds; the
    residual against the certified operator is the packing-fidelity
    budget (``AC008_PACKING_FIDELITY_BUDGET_EV``).
    """
    w = np.zeros((4, 5, 5), dtype=complex)
    for (p, q), (h1, h2, h3, h4) in components.items():
        w[0, p, q] = h1
        w[1, p, q] = h2
        w[2, p, q] = complex(h3, -h4)
    w[3] = np.conj(w[2]).T
    return w / energy_scale


def rotated_hubbard_site_components(
    occupations, prov, site, *, axis, angle, energy_scale=RY_TO_EV
):
    """Story-015 contract rotation of one site, in bundle file packing.

    Runs :func:`rotate_hubbard_site_operator` (occupation co-rotation,
    W reassembly, dc frozen) and packs the rotated site operator as
    ``{(p, q): (h1, h2, h3, h4)}`` components.  Returns the components
    and the rotation report (covariance, dc drift, gates) verbatim.
    """
    result = rotate_hubbard_site_operator(
        occupations, prov, site, axis=axis, angle=angle
    )
    comps = hubbard_site_block_components(
        result["v_hubbard_rotated"][site], energy_scale=energy_scale
    )
    return comps, result["report"]


def run_ac008_gate(
    bundle,
    *,
    cross_check,
    contract_table,
    ensemble,
    texture_provenance,
    audit_artifact,
):
    """AC-008 gate driver (story-016): assemble the +U verdict report.

    Parameters
    ----------
    bundle : HubbardBundle
        The genuine +U artifact under gate.
    cross_check : dict
        The ``finite_rotation_cross_check`` summary produced by
        ``TB2J.interfaces.siesta_spiral_validation.validate_finite_rotations``
        on the frozen problem built from this bundle's operator
        (analytic local response vs finite rotations, story-010 machinery).
    contract_table : list of dict
        Per-rotation rows binding the story-015 contract to the same
        frozen problem at the energy level; each row carries ``site``,
        ``axis``, ``angle_rad``, ``energy_conjugation_ev``,
        ``energy_contract_ev``, ``abs_delta_ev``,
        ``hubbard_covariance_relative``, ``dc_change_max`` and
        ``would_be_dc_drift_ry``.
    ensemble : TB2J.interfaces.siesta_spiral_response.Ensemble
        The declared ADR-005 ensemble of the frozen problem.
    texture_provenance : dict
        Producer run id, convergence, Edftu and texture description,
        recorded verbatim in the verdict.
    audit_artifact : dict
        Envelope from :func:`build_xc_hubbard_audit_artifact`; the
        provenance audit is wired in here and its verdict recorded.

    Returns the verdict dict (agreement table, tolerances, provenance
    verdicts, declared ensemble string, texture provenance verbatim,
    pass/fail with failure reasons).  JSON-serializable for the
    evidence pack; the audit refusal is recorded, never swallowed.
    """
    from .siesta_spiral_response import (
        CURVATURE_BUDGET_EV_PER_RAD2,
        TORQUE_BUDGET_EV_PER_RAD,
    )

    # Pinned budgets are cross-checked against the story-009/010 approved
    # values they inherit; a drift there invalidates the AC-008 pin.
    if (
        AC008_TORQUE_BUDGET_EV_PER_RAD != TORQUE_BUDGET_EV_PER_RAD
        or AC008_CURVATURE_BUDGET_EV_PER_RAD2 != CURVATURE_BUDGET_EV_PER_RAD2
    ):
        raise HubbardBundleError(
            "AC-008 torque/curvature budgets have drifted from the "
            "story-009/010 approved values they inherit; re-pin the gate "
            "before running it"
        )
    try:
        audit_verdict = audit_xc_hubbard_provenance(audit_artifact)
    except ProvenanceAuditError as exc:
        audit_verdict = {"passed": False, "reason": exc.reason, "diagnostic": str(exc)}

    record = ensemble.to_record()
    declared_ensemble_string = (
        "ADR-005 {kind}: n_occupied={n}, electron_count={ne}, "
        "gap={gap:.12g} eV, Fermi={fermi:.12g} eV; density, scalar "
        "potential, amplitudes, basis and S frozen; only the declared "
        "magnetic operator rotates; finite rotations reuse the same "
        "frozen problem (no cross-SCF comparison enters the gate)"
    ).format(
        kind=record["kind"],
        n=record["n_occupied"],
        ne=record["electron_count"],
        gap=record["gap_ev"],
        fermi=record["fermi_energy_ev"],
    )

    ladder = tuple(cross_check.get("angle_ladder_rad", ()))
    # Pinned-ladder cross-check: the gate may only report the approved
    # AC-008 ladder (or exactly the pinned stable window).  A cross-check
    # computed over a different angle ladder could pass every budget
    # while the verdict still claims the approved rungs -- refuse here,
    # naming the pinned values.
    pinned_window_ladder = tuple(
        AC008_ANGLE_LADDER_RAD[i] for i in AC008_STABLE_WINDOW_RUNGS
    )
    supplied_ladder = tuple(float(a) for a in ladder)
    if supplied_ladder not in (AC008_ANGLE_LADDER_RAD, pinned_window_ladder):
        raise HubbardBundleError(
            "cross_check angle ladder "
            f"{[round(a, 6) for a in supplied_ladder]} does not match the "
            "pinned AC-008 ladder; the gate may only run over the "
            f"approved ladder {[round(a, 6) for a in AC008_ANGLE_LADDER_RAD]} "
            "or exactly its pinned stable window "
            f"{[round(a, 6) for a in pinned_window_ladder]} (rungs "
            f"{AC008_STABLE_WINDOW_RUNGS}); re-run the story-010 "
            "cross-check on the pinned ladder"
        )
    table_angles = sorted({float(row["angle_rad"]) for row in contract_table})
    reasons = []
    if not audit_verdict.get("passed"):
        reasons.append("provenance_audit_" + audit_verdict.get("reason", "refused"))
    if not cross_check.get("passed"):
        reasons.extend(cross_check.get("failure_reasons", ["cross_check_failed"]))
    contract_budget = AC008_CONTRACT_ENERGY_BUDGET_EV
    # fail-closed: recompute the deltas from the reported energies instead
    # of trusting the rows' own abs_delta_ev bookkeeping
    contract_table = [
        dict(
            row,
            abs_delta_ev=abs(
                float(row["energy_contract_ev"]) - float(row["energy_conjugation_ev"])
            ),
        )
        for row in contract_table
    ]
    worst_contract = max(
        (float(row["abs_delta_ev"]) for row in contract_table), default=0.0
    )
    worst_covariance = max(
        (float(row["hubbard_covariance_relative"]) for row in contract_table),
        default=0.0,
    )
    # fail-closed: absent dc evidence is a refusal, not a pass.  A row
    # without 'dc_change_max' proves nothing about double counting being
    # frozen; only an explicit zero passes.
    missing_dc_evidence = any("dc_change_max" not in row for row in contract_table)
    dc_moved = not missing_dc_evidence and any(
        float(row["dc_change_max"]) != 0.0 for row in contract_table
    )
    if worst_contract > contract_budget:
        reasons.append("contract_energy_budget_exceeded")
    if worst_covariance > HUBBARD_ROTATION_TOLERANCE:
        reasons.append("hubbard_covariance_budget_exceeded")
    if missing_dc_evidence:
        reasons.append("missing_dc_freeze_evidence")
    if dc_moved:
        reasons.append("double_counting_not_frozen")
    if not contract_table:
        reasons.append("empty_contract_table")
    elif table_angles != sorted(set(float(a) for a in ladder)):
        reasons.append("contract_table_angles_differ_from_ladder")

    agreement_table = [
        {
            "angle_rad": rung["angle_rad"],
            "gradient_max_abs_mismatch": rung["gradient_max_abs_mismatch"],
            "curvature_max_abs_mismatch": rung["curvature_max_abs_mismatch"],
            "gradient_roundoff_floor": rung["gradient_roundoff_floor"],
            "curvature_roundoff_floor": rung["curvature_roundoff_floor"],
            "minimum_rotated_gap_ev": rung["minimum_rotated_gap_ev"],
        }
        for rung in cross_check.get("rungs", ())
    ]
    verdict = {
        "gate": "AC-008",
        "version": "1.0",
        "passed": not reasons,
        "failure_reasons": reasons,
        "declared_ensemble": record,
        "declared_ensemble_string": declared_ensemble_string,
        "tolerances": {
            "angle_ladder_rad": list(AC008_ANGLE_LADDER_RAD),
            "torque_ev_per_rad": AC008_TORQUE_BUDGET_EV_PER_RAD,
            "curvature_ev_per_rad2": AC008_CURVATURE_BUDGET_EV_PER_RAD2,
            "contract_energy_abs_ev": AC008_CONTRACT_ENERGY_BUDGET_EV,
            "hubbard_covariance_relative": HUBBARD_ROTATION_TOLERANCE,
            "stable_window_rungs": list(AC008_STABLE_WINDOW_RUNGS),
            "pinned_at": "story-016 approval",
        },
        "agreement_table": agreement_table,
        "stable_error_regime": {
            "gradient": cross_check.get("stable_error_regime", {}).get("gradient"),
            "curvature": cross_check.get("stable_error_regime", {}).get("curvature"),
            "common_stable_windows": cross_check.get("common_stable_windows", []),
            "reference_gradient": cross_check.get("reference_gradient"),
            "reference_torque": cross_check.get("reference_torque"),
            "torque_state": cross_check.get("torque_state"),
        },
        "contract_wire_in": {
            "rows": list(contract_table),
            "worst_abs_delta_ev": worst_contract,
            "worst_hubbard_covariance_relative": worst_covariance,
            "double_counting_frozen": not dc_moved and not missing_dc_evidence,
        },
        "provenance_verdicts": {
            "audit": audit_verdict,
            "rotation_contract": {
                "mathdoc": "tb2j/siesta-spiral-hubbard-rotation",
                "tolerance_relative": HUBBARD_ROTATION_TOLERANCE,
                "worst_covariance_relative": worst_covariance,
                "passed": worst_covariance <= HUBBARD_ROTATION_TOLERANCE,
            },
        },
        "texture_provenance": dict(texture_provenance),
    }
    return verdict
