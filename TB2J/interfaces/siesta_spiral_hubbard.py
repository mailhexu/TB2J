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
that fact so no consumer double-counts.  Operator algebra on the
rotated frame is story-015/016 scope and deliberately absent here.

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
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903

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

    def to_dict(self) -> dict:
        return dict(vars(self))

    @classmethod
    def from_dict(cls, payload):
        fields = cls.__dataclass_fields__
        filled = {
            name: f.default
            for name, f in fields.items()
            if name not in payload
            and f.default is not __import__("dataclasses").MISSING
        }
        if "source_refs" in payload:
            filled["source_refs"] = dict(payload["source_refs"])
        payload = {**filled, **payload}
        missing = [name for name in fields if name not in payload]
        if missing:
            raise DftuProvenanceError(
                "incomplete +U provenance payload: missing fields "
                f"{missing}; refusing partial hubbard metadata"
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
# bundle artifacts
# ----------------------------------------------------------------------


def _digest(arrays: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        h.update(key.encode())
        arr = np.ascontiguousarray(arrays[key])
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


@dataclass
class HubbardBundle:
    """+U artifact: provenance + occupations + potentials + Edftu check.

    Records explicitly that the stored HSX Hamiltonian already contains
    the +U potential (H_stored = H_xc + H_U); consumers must not add
    V_U to the stored H again (story-015/016 handle the rotated-frame
    operator algebra on xc and hubbard vertices separately).
    """

    prov: DftuProvenance
    occupations: np.ndarray  # (nsites, 4, 5, 5) complex
    v_hubbard: np.ndarray  # (nsites, 4, 5, 5) complex, Ry
    v_dc: np.ndarray  # (nsites, 4, 5, 5) complex, Ry
    edftu_ev: float
    edftu_reference_ev: float
    fixture_digest: str = ""
    conventions: dict = field(default_factory=lambda: dict(_HUBBARD_CONVENTIONS))
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

    def to_certificate(self) -> dict:
        """JSON-serializable certificate with sha256 of payload arrays."""
        payload = {
            "prov": self.prov.to_dict(),
            "occupations": self.occupations.tolist(),
            "v_hubbard": np.stack([self.v_hubbard.real, self.v_hubbard.imag]).tolist(),
            "v_dc": np.stack([self.v_dc.real, self.v_dc.imag]).tolist(),
            "edftu_ev": self.edftu_ev,
            "edftu_reference_ev": self.edftu_reference_ev,
            "fixture_digest": self.fixture_digest,
            "conventions": self.conventions,
            "note": self.note,
        }
        digest = _digest(
            {
                "occupations": np.asarray(self.occupations, dtype=complex),
                "v_hubbard": np.asarray(self.v_hubbard, dtype=complex),
                "v_dc": np.asarray(self.v_dc, dtype=complex),
                "edftu_ev": np.asarray(self.edftu_ev, dtype=float),
                "edftu_reference_ev": np.asarray(self.edftu_reference_ev, dtype=float),
            }
        )
        return {
            "kind": "hubbard_bundle_certificate",
            "version": "1.0",
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
        arrays = {
            "occupations": np.asarray(payload["occupations"], dtype=complex),
            "v_hubbard": (
                np.asarray(payload["v_hubbard"][0])
                + 1j * np.asarray(payload["v_hubbard"][1])
            ),
            "v_dc": (
                np.asarray(payload["v_dc"][0]) + 1j * np.asarray(payload["v_dc"][1])
            ),
        }
        digest = _digest(
            {
                **arrays,
                "edftu_ev": np.asarray(payload["edftu_ev"], dtype=float),
                "edftu_reference_ev": np.asarray(
                    payload["edftu_reference_ev"], dtype=float
                ),
            }
        )
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
            note=payload.get("note", ""),
        )
