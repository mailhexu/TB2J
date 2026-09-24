"""Spin-spiral MFT exchange calculator through the standard TB2J output (story 005).

:class:`ExchangeSpiral` maps the story-004 two-channel curvature
(:mod:`TB2J.spiral_kernels`) onto exchange tensors and writes the standard
TB2J ``SpinIO`` tree.  It deliberately does **not** reuse the
:class:`~TB2J.exchange.ExchangeNCL` constructor path (its input is a frozen
spiral-state bundle, not tbmodels/Green functions); it reuses the *output*
conventions instead:

* Heisenberg mapping (ADR-S8): ``J^spiral_ab = -C^dd_ab`` for ``a != b``,
  with the diagonal consistency identity
  ``C^dd_aa = sum_b J_ab cos(Theta_a - Theta_b)`` (exact to machine zero in
  the probe class) and the ``C^db = 0`` check;
* tensor conversion through the **existing**
  :meth:`~TB2J.exchange.ExchangeNCL.A_to_Jtensor` formulas: the
  local-frame out-of-plane channel is mapped onto the ``{0,x,y,z}`` Pauli
  basis as ``A^{00}_{ij}(R) = -i C^dd_ij(R)`` (all other components zero),
  so that ``J_iso = Im(A00 - Axx - Ayy - Azz) = -C^dd_ij`` verbatim while
  the DMI/Jani/biquadratic slots evaluate to their symmetry-expected zero
  (the planar single-axis path carries no antisymmetric or
  longitudinal content);
* AFM normalization ``J / sgn(S_i . S_j)`` from the reference local
  moments (the TB2J stored convention of the collinear/q-space paths);
* the non-Heisenberg diagnostic ``||C^bb + J o cos(Theta)||`` reported on
  every run (off-diagonal Heisenberg form; the diagonal is fixed by the
  Goldstone zero mode).

Gates (Goldstone, torque, diagonal consistency, q=0 LKAG anchor) hard-fail
with diagnostics unless ``SpiralParameters.gate_override`` is set.

Normative conventions: ``docs/sympy/spiral_state_mft.py`` and the
spiral frozen-bundle contract.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ase import Atoms

from TB2J.io_exchange import SpinIO
from TB2J.spiral_green import SpiralState
from TB2J.spiral_kernels import (
    SpiralCurvature,
    SpiralGateError,
    _dense_ed,
    _frozen_f,
    contour_kernels_dense,
    eigenbasis_kernels_dense,
    goldstone_gate,
    inplane_response_fd,
    lab_supercell,
    planar_kmesh,
    q0_anchor_check,
    spiral_angles,
    torque_gate,
)

__all__ = [
    "SpiralParameters",
    "SpiralJtensors",
    "ExchangeSpiral",
    "a_tensors_to_jtensors",
    "derive_ncell",
]


@dataclass
class SpiralParameters:
    """Configuration of a spiral-state exchange run (MagnonParameters style).

    The reference protocol is the frozen torque-free bundle (P-c, ADR-S4):
    TBUpy owns the per-q SCF; TB2J evaluates the force-theorem curvature
    about it.  The stored bundle convention puts the spiral rotation axis
    on +y (the local field rotates in the x-z plane).
    """

    #: ring size (number of primitive cells); 0 = derive from q commensurability
    ncell: int = 0
    #: folded mesh (N, 1, 1) for the optional E(q) table; None = planar mesh
    kmesh: Optional[List[int]] = None
    #: curvature kernel path: "contour" (primary) or "eigenbasis" (reference)
    kernel: str = "contour"
    #: number of Matsubara points of the contour kernel
    n_matsubara: int = 3000
    #: smearing width override (eV); None = bundle metadata value
    width: Optional[float] = None
    #: reference protocol (P-c frozen bundle is the required one, ADR-S4)
    reference_protocol: str = "frozen-bundle"
    #: spiral rotation axis of the stored convention (only +/-y implemented)
    spiral_axis: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    # gates
    goldstone_tol: float = 1e-7
    torque_tol: float = 1e-6
    diag_consistency_tol: float = 1e-6
    q0_anchor: bool = True
    q0_anchor_tol: float = (
        1e-5  # FD-limited (inplane_response_fd), story-004 convention
    )
    #: report gate violations instead of raising
    gate_override: bool = False
    # diagnostics
    #: write the E(q) frozen-band-energy table
    eq_table: bool = False
    #: q points (fractional, 3 components) of the E(q) table; None = [q, -q, 0]
    q_set: Optional[List[List[float]]] = None
    # output geometry
    #: pair distance cutoff (Angstrom); None = all pairs on the ring
    Rcut: Optional[float] = None
    #: lattice vectors of the primitive cell (Angstrom), row-major 3x3
    cell: List[List[float]] = field(default_factory=lambda: np.eye(3).tolist())
    #: chemical symbols of the norb sites; None = "X" placeholders
    symbols: Optional[List[str]] = None

    def __post_init__(self):
        if self.kernel not in ("contour", "eigenbasis"):
            raise ValueError(
                f"kernel must be 'contour' or 'eigenbasis', got {self.kernel!r}"
            )
        if self.kernel == "contour" and self.n_matsubara < 1:
            raise ValueError(f"n_matsubara must be >= 1, got {self.n_matsubara}")
        if self.ncell < 0:
            raise ValueError(f"ncell must be >= 0 (0 = derive), got {self.ncell}")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.reference_protocol != "frozen-bundle":
            raise ValueError(
                "reference_protocol must be 'frozen-bundle' (the torque-free "
                f"per-q SCF bundle, ADR-S4), got {self.reference_protocol!r}"
            )
        if self.Rcut is not None and self.Rcut <= 0:
            raise ValueError(f"Rcut must be positive, got {self.Rcut}")
        for name in (
            "goldstone_tol",
            "torque_tol",
            "diag_consistency_tol",
            "q0_anchor_tol",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.kmesh is not None:
            if len(self.kmesh) != 3 or list(self.kmesh[1:]) != [1, 1]:
                raise ValueError(
                    f"kmesh must be (N, 1, 1) on the 1D spiral ring, got {self.kmesh}"
                )
            if int(self.kmesh[0]) < 1:
                raise ValueError(f"kmesh[0] must be >= 1, got {self.kmesh[0]}")
        cell = np.asarray(self.cell, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
            raise ValueError("cell must be a finite 3x3 matrix")
        if np.linalg.matrix_rank(cell) != 3:
            raise ValueError("cell must be non-singular")
        axis = np.asarray(self.spiral_axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("spiral_axis must be a finite 3-vector")
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("spiral_axis must be non-zero")
        y = np.array([0.0, 1.0, 0.0])
        dev = min(np.linalg.norm(axis / norm - y), np.linalg.norm(axis / norm + y))
        if dev > 1e-9:
            raise NotImplementedError(
                "only the stored-convention spiral axis +/-(0, 1, 0) (field in "
                f"the x-z plane) is implemented; got {list(self.spiral_axis)}. "
                "Rotate the bundle instead."
            )
        if self.q_set is not None:
            for qv in self.q_set:
                if len(qv) != 3:
                    raise ValueError(f"q_set entries must have 3 components, got {qv}")

    def resolved_cell(self) -> np.ndarray:
        return np.asarray(self.cell, dtype=float)


@dataclass
class SpiralJtensors:
    """Raw conversion output of the ExchangeNCL ``A -> J`` hierarchy."""

    exchange_Jdict: dict
    dmi_ddict: dict
    Jani_dict: dict
    biquadratic_Jdict: dict
    debug_dict: dict


def derive_ncell(state: SpiralState) -> int:
    """Ring size of the commensurate 1D spiral ``q = p/N`` along e_1."""
    q = float(np.asarray(state.q_frac, dtype=float)[0])
    if abs(q) < 1e-12:
        raise ValueError(
            "q = 0 bundle has no commensurate ring size; set ncell explicitly"
        )
    n = int(round(1.0 / abs(q)))
    if abs(abs(q) * n - 1.0) > 1e-9:
        raise ValueError(
            f"q = {q!r} is not commensurate on a ring (qN = {q * n!r}); "
            "the explicit-supercell kernel path needs a commensurate q"
        )
    return n


def reference_density_matrix(state: SpiralState, ncell: int):
    """Ground-state density matrix of the explicit lab-frame supercell.

    Returns ``(rho, evals)`` with ``rho`` the frozen-occupation density
    matrix of the supercell Hamiltonian built by
    :func:`~TB2J.spiral_kernels.lab_supercell`.
    """
    H, S = lab_supercell(state, ncell)
    evals, evecs = _dense_ed(H, S)
    f = _frozen_f(state, evals=evals)
    rho = (evecs * f[None, :]) @ evecs.conj().T
    return rho, evals


def reference_moments(state: SpiralState, ncell: int):
    """Local spin moments and charges of the frozen spiral reference.

    ``moms[a, mu, :]`` is the spin moment ``tr(rho sigma)`` (lab frame) of
    ring site ``(a, mu)``, ``charges[a, mu]`` its frozen occupation.  The
    moment directions carry the spiral texture used by the AFM
    normalization.
    """
    rho, _ = reference_density_matrix(state, ncell)
    norb = state.norb
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    moms = np.zeros((ncell, norb, 3))
    charges = np.zeros((ncell, norb))
    for a in range(ncell):
        for mu in range(norb):
            s = 2 * (a * norb + mu)
            blk = rho[s : s + 2, s : s + 2]
            charges[a, mu] = float(np.trace(blk).real)
            for ix, sig in enumerate((sx, sy, sz)):
                moms[a, mu, ix] = float(np.trace(blk @ sig).real)
    return moms, charges


def a_tensors_to_jtensors(A_ijR: dict, spinat: np.ndarray) -> SpiralJtensors:
    """Convert the four-component ``A^{uv}_{ij}(R)`` tensors via the existing
    :meth:`~TB2J.exchange.ExchangeNCL.A_to_Jtensor` hierarchy.

    ``A_ijR`` keys are ``(R, iatom, jatom)`` with ``iatom == ispin`` (one
    site per atom/spin); ``spinat`` are the reference local moments.  The
    conversion formulas are the existing ExchangeNCL ones, applied
    verbatim by delegating to the real method on a minimal attribute
    harness (the only attributes it reads are ``A_ijR``, ``spinat`` and
    ``ispin``).  Pinned by ``tests/test_exchange_spiral.py``.
    """
    from TB2J.exchange import ExchangeNCL

    exch = ExchangeNCL.__new__(ExchangeNCL)
    exch.A_ijR = A_ijR
    exch.spinat = np.asarray(spinat, dtype=float)
    exch._spin_dict = {iatom: iatom for iatom in range(len(exch.spinat))}
    exch.A_to_Jtensor()
    return SpiralJtensors(
        exchange_Jdict=exch.exchange_Jdict,
        dmi_ddict=exch.DMI,
        Jani_dict=exch.Jani,
        biquadratic_Jdict=exch.B,
        debug_dict=exch.debug_dict,
    )


def _pair_sign(moms: np.ndarray, ai: int, mi: int, aj: int, mj: int) -> float:
    """``sgn(S_i . S_j)`` of the reference local moments (0 -> +1)."""
    s = float(np.dot(moms[ai, mi], moms[aj, mj]))
    return 1.0 if s == 0.0 else float(np.sign(s))


def nonheisenberg_diagnostic(curv: SpiralCurvature, thetas: np.ndarray) -> dict:
    """``||C^bb + J o cos(Theta)||`` against the extracted ``J = -C^dd``.

    The off-diagonal comparison is the Heisenberg-form deviation; the
    diagonal of the predicted matrix is the Heisenberg value
    ``sum_b J_ab cos(Theta_a - Theta_b)`` (equal to the Goldstone-zero-mode
    row sum when ``C^bb . 1 = 0`` holds exactly), so a strictly bilinear
    energy reports exactly zero on the full matrix.
    """
    Cbb = curv[("b", "b")]
    J = -curv[("d", "d")]
    np.fill_diagonal(J, 0.0)
    th = np.asarray(thetas, dtype=float).ravel()
    cosM = np.cos(th[:, None] - th[None, :])
    off = ~np.eye(len(th), dtype=bool)
    pred = -(J * cosM)
    np.fill_diagonal(pred, (J * cosM).sum(axis=1))
    scale = max(float(np.max(np.abs(Cbb))), 1e-12)
    full = Cbb - pred
    return {
        "max_abs_offdiag": float(np.max(np.abs((Cbb + J * cosM)[off]))),
        "max_abs_full": float(np.max(np.abs(full))),
        "scale": scale,
    }


def _jsonable(obj):
    """Recursively convert numpy containers of a report to plain Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


class ExchangeSpiral:
    """Spiral-state MFT exchange calculator producing standard SpinIO output.

    Construct with :meth:`from_spiral_state`, :meth:`run` it, then fetch
    the result with :meth:`get_spinio` and write it with
    :meth:`write_output` (``SpinIO.write_all`` plus
    ``spiral_diagnostics.json``).
    """

    def __init__(self, state: SpiralState, params: Optional[SpiralParameters] = None):
        state.validate()
        self.state = self._apply_width(state, params or SpiralParameters())
        self.params = params or SpiralParameters()
        self._ncell = None
        self.curvature: Optional[SpiralCurvature] = None
        self.report: dict = {}
        self.moms: Optional[np.ndarray] = None
        self.charges: Optional[np.ndarray] = None
        self.A_ijR: dict = {}
        self.distance_dict: dict = {}
        self._pair_sites: dict = {}
        self._raw_jtensors: Optional[SpiralJtensors] = None
        self.exchange_Jdict: dict = {}
        self.dmi_ddict: dict = {}
        self.Jani_dict: dict = {}
        self.biquadratic_Jdict: dict = {}
        self._spinio: Optional[SpinIO] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @staticmethod
    def _mirror_state(obj) -> SpiralState:
        """TB2J :class:`SpiralState` mirror of a tbupy bundle (or passthrough)."""
        if isinstance(obj, SpiralState):
            return obj
        fields = {f.name for f in dataclasses.fields(SpiralState)}
        try:
            kwargs = {name: getattr(obj, name) for name in fields}
        except AttributeError as exc:
            raise TypeError(
                "spiral_state must be a SpiralState bundle or an object with "
                f"the tbupy_spiral_state fields; got {type(obj).__name__}"
            ) from exc
        return SpiralState(**kwargs)

    @classmethod
    def load_spiral_state(cls, path) -> SpiralState:
        """Load a ``*.spiral.nc`` bundle through the lazy tbupy reader."""
        try:
            from tbupy.spiral_state import load_spiral_state as _tbupy_load
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Reading a spiral-state bundle file requires tbupy "
                "(pip install tbupy); a TB2J SpiralState can be passed directly."
            ) from exc
        return cls._mirror_state(_tbupy_load(os.fspath(path)))

    @classmethod
    def from_spiral_state(cls, spiral_state, kmesh=None, **params) -> "ExchangeSpiral":
        """Build a calculator from a bundle (path, tbupy or TB2J state).

        ``kmesh`` is the optional folded ``(N, 1, 1)`` mesh of the E(q)
        diagnostic table; remaining keyword arguments feed
        :class:`SpiralParameters`.
        """
        if isinstance(spiral_state, (str, os.PathLike)):
            state = cls.load_spiral_state(spiral_state)
        else:
            state = cls._mirror_state(spiral_state)
        if kmesh is not None:
            params.setdefault("kmesh", [int(v) for v in np.asarray(kmesh).ravel()])
        return cls(state, SpiralParameters(**params))

    @staticmethod
    def _apply_width(state: SpiralState, params: SpiralParameters) -> SpiralState:
        if params.width is None:
            return state
        meta = dict(state.metadata)
        meta["width"] = float(params.width)
        return dataclasses.replace(state, metadata_json=json.dumps(meta))

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    @property
    def ncell(self) -> int:
        if self._ncell is None:
            self._ncell = (
                self.params.ncell if self.params.ncell > 0 else derive_ncell(self.state)
            )
        return self._ncell

    def _validate_state_geometry(self):
        q = np.asarray(self.state.q_frac, dtype=float)
        if not np.allclose(q[1:], 0.0):
            raise NotImplementedError(
                "the spiral kernel path is a 1D ring along e_1; got "
                f"q_frac = {q.tolist()} with transverse components"
            )

    def run(self) -> dict:
        """Evaluate the curvature, gates, mapping and diagnostics."""
        p = self.params
        self._validate_state_geometry()
        ncell = self.ncell
        state = self.state

        if p.kernel == "contour":
            curv = contour_kernels_dense(state, ncell, n_matsubara=p.n_matsubara)
        else:
            curv = eigenbasis_kernels_dense(state, ncell)
        self.curvature = curv

        thetas = spiral_angles(state, ncell)  # (ncell, norb)
        self.moms, self.charges = reference_moments(state, ncell)

        report = {
            "q_frac": np.asarray(state.q_frac, dtype=float).tolist(),
            "ncell": int(ncell),
            "kernel": p.kernel,
            "n_matsubara": int(p.n_matsubara),
            "reference_protocol": p.reference_protocol,
            "gate_override": bool(p.gate_override),
            "torque_norms": (
                None
                if state.torque_norms is None
                else np.asarray(state.torque_norms, dtype=float).tolist()
            ),
        }

        # ---- gates (hard fail unless overridden) ----
        reports = {}
        reports["goldstone"] = goldstone_gate(
            curv, tol=p.goldstone_tol, allow_violation=p.gate_override
        )
        reports["torque"] = torque_gate(
            curv, state, ncell, tol=p.torque_tol, allow_violation=p.gate_override
        )
        self._diag_consistency_gate(curv, thetas, reports)
        self._q0_anchor_gate(curv, reports)

        report["gates"] = {
            name: {
                "passed": bool(rep.get("passed")),
                "flagged": bool(rep.get("flagged", False)),
                "max_residual": float(rep.get("max_residual", np.nan)),
                "tol": float(rep.get("tol", np.nan)),
            }
            for name, rep in reports.items()
        }
        report["zero_modes"] = {
            "goldstone_residuals": np.asarray(
                reports["goldstone"]["residuals"], dtype=float
            ).tolist(),
            "torque_residuals_cos": np.asarray(
                reports["torque"]["residuals_cos"], dtype=float
            ).tolist(),
            "torque_residuals_sin": np.asarray(
                reports["torque"]["residuals_sin"], dtype=float
            ).tolist(),
        }

        # ---- mapping: C^dd -> A tensors -> J via the existing hierarchy ----
        self._build_pairs(curv, p)
        raw = a_tensors_to_jtensors(self.A_ijR, self.moms[0])
        self._raw_jtensors = raw
        sgn = {
            key: _pair_sign(self.moms, *self._pair_sites[key])
            for key in self._pair_sites
        }
        self.exchange_Jdict = {
            key: val / sgn[key] for key, val in raw.exchange_Jdict.items()
        }
        self.dmi_ddict = dict(raw.dmi_ddict)
        self.Jani_dict = dict(raw.Jani_dict)
        self.biquadratic_Jdict = {
            key: (jprime / sgn[key], b)
            for key, (jprime, b) in raw.biquadratic_Jdict.items()
        }

        # ---- diagnostics ----
        Cdb = curv[("d", "b")]
        report["cdb_max"] = float(np.max(np.abs(Cdb)))
        report["nonheisenberg_Cbb"] = nonheisenberg_diagnostic(curv, thetas)
        report["curvature_scale_Cdd"] = float(np.max(np.abs(curv[("d", "d")])))
        report["n_pairs"] = len(self.exchange_Jdict)
        report["moments"] = self.moms.tolist()

        if p.eq_table:
            report["eq_table"] = self.compute_eq_table()

        self.report = _jsonable(report)
        return self.report

    # ------------------------------------------------------------------
    # gates and mapping helpers
    # ------------------------------------------------------------------
    def _diag_consistency_gate(self, curv, thetas, reports) -> None:
        """``C^dd_aa = sum_b J_ab cos(Theta_a - Theta_b)`` (ADR-S8)."""
        Cdd = curv[("d", "d")]
        J = -Cdd.copy()
        np.fill_diagonal(J, 0.0)
        th = np.asarray(thetas, dtype=float).ravel()
        cosM = np.cos(th[:, None] - th[None, :])
        resid = np.diag(Cdd) - (J * cosM).sum(axis=1)
        scale = max(float(np.max(np.abs(Cdd))), 1e-12)
        worst = float(np.max(np.abs(resid)))
        passed = worst <= self.params.diag_consistency_tol
        rep = {
            "gate": "diag_consistency",
            "residuals": resid,
            "max_residual": worst,
            "scale": scale,
            "tol": float(self.params.diag_consistency_tol),
            "passed": passed,
            "allow_violation": bool(self.params.gate_override),
        }
        if not passed and not self.params.gate_override:
            raise SpiralGateError(
                "Diagonal consistency violated: C^dd_aa = sum_b J_ab "
                f"cos(Theta_a - Theta_b) requires a pairwise-Heisenberg-form "
                f"response (max residual {worst:.3e} > tol "
                f"{self.params.diag_consistency_tol:g}).",
                rep,
            )
        reports["diag_consistency"] = rep

    def _q0_anchor_gate(self, curv, reports) -> None:
        """Mandatory q=0 LKAG anchor (``C^dd = C^bb = M``, ``C^db = 0``)."""
        if not self.params.q0_anchor:
            return
        q = float(np.asarray(self.state.q_frac, dtype=float)[0])
        if abs(q) > 1e-12:
            return
        M = inplane_response_fd(self.state, self.ncell)
        try:
            rep = q0_anchor_check(curv, M, tol=self.params.q0_anchor_tol)
        except SpiralGateError as exc:
            if not self.params.gate_override:
                raise
            rep = exc.report
            rep["allow_violation"] = True
        rep["max_residual"] = max(
            rep["max_dd_minus_bb"], rep["max_bb_minus_M"], rep["max_db"]
        )
        reports["q0_anchor"] = rep

    def _build_pairs(self, curv: SpiralCurvature, p: SpiralParameters) -> None:
        """Enumerate ring pairs, assemble ``A^{00} = -i C^dd`` and distances."""
        state = self.state
        ncell, norb = self.ncell, state.norb
        cell = p.resolved_cell()
        positions = np.asarray(state.taus, dtype=float) @ cell
        Cdd = curv[("d", "d")]
        A = np.zeros((4, 4), dtype=complex)
        A[0, 0] = -1.0j
        self.A_ijR = {}
        self.distance_dict = {}
        self._pair_sites = {}
        for ai in range(ncell):
            for mi in range(norb):
                si = ai * norb + mi
                for aj in range(ncell):
                    for mj in range(norb):
                        sj = aj * norb + mj
                        if si == sj:
                            continue
                        R = (aj - ai, 0, 0)
                        vec = (
                            positions[mj]
                            + np.asarray(R, dtype=float) @ cell
                            - positions[mi]
                        )
                        dist = float(np.linalg.norm(vec))
                        if p.Rcut is not None and dist >= p.Rcut:
                            continue
                        key = (R, mi, mj)
                        self.A_ijR[key] = A * Cdd[si, sj]
                        self.distance_dict[key] = (vec, dist)
                        self._pair_sites[key] = (ai, mi, aj, mj)

    # ------------------------------------------------------------------
    # diagnostics: optional E(q) table
    # ------------------------------------------------------------------
    def compute_eq_table(self) -> dict:
        """Frozen band energy ``E(q)`` of the folded pencil over ``q_set``.

        Uses the contract rebuild rule through ``SpiralGreen`` (lazy tbupy
        import) on the half-shifted commensurate mesh of each q.
        """
        from TB2J.spiral_green import SpiralGreen
        from TB2J.spiral_kernels import planar_flux_shift

        state = self.state
        q_ref = np.asarray(state.q_frac, dtype=float)
        if self.params.q_set is None:
            q_list = [[0.0, 0.0, 0.0], q_ref.tolist(), (-q_ref).tolist()]
        else:
            q_list = [list(map(float, qv)) for qv in self.params.q_set]
        rows = []
        for qv in q_list:
            state_q = dataclasses.replace(state, q_frac=np.asarray(qv, dtype=float))
            try:
                planar_flux_shift(state_q, self.ncell)
                mesh = planar_kmesh(state_q, self.ncell)
            except ValueError as exc:
                raise ValueError(f"E(q) table point q = {qv}: {exc}") from exc
            green = SpiralGreen(state_q, mesh)
            f = green.frozen_occupations()
            energy = float(np.sum(green.kweights[:, None] * f * green.evals))
            rows.append({"q_frac": qv, "E_frozen_eV": energy})
        return {"points": rows, "width_eV": state.width, "ncell": self.ncell}

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def build_atoms(self) -> Atoms:
        """Primitive-cell :class:`ase.Atoms` of the spiral ring."""
        p = self.params
        taus = np.asarray(self.state.taus, dtype=float)
        symbols = p.symbols or ["X"] * self.state.norb
        if len(symbols) != self.state.norb:
            raise ValueError(
                f"symbols must have norb = {self.state.norb} entries, got {len(symbols)}"
            )
        return Atoms(
            symbols=symbols,
            positions=taus @ p.resolved_cell(),
            cell=p.resolved_cell(),
            pbc=(True, True, True),
        )

    def get_spinio(self) -> SpinIO:
        """Standard ``SpinIO`` of the extracted tensors (run first)."""
        if self._spinio is not None:
            return self._spinio
        if not self.exchange_Jdict:
            raise RuntimeError(
                "ExchangeSpiral.run() must be called before get_spinio()"
            )
        natom = self.state.norb
        q = np.asarray(self.state.q_frac, dtype=float).tolist()
        self._spinio = SpinIO(
            atoms=self.build_atoms(),
            spinat=np.asarray(self.moms[0], dtype=float),
            charges=np.asarray(self.charges[0], dtype=float),
            index_spin=list(range(natom)),
            orbital_names={},
            colinear=False,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            dmi_ddict=self.dmi_ddict,
            Jani_dict=self.Jani_dict,
            biquadratic_Jdict=self.biquadratic_Jdict,
            debug_dict=self._raw_jtensors.debug_dict,
            description=(
                "Spin-spiral MFT exchange (ExchangeSpiral, story 005)\n"
                f"q_frac = {q}, ncell = {self.ncell}, kernel = {self.params.kernel}\n"
            ),
        )
        # make the in-memory object consistent (Rlist/nspin/ind_atoms), as
        # load_pickle does on the round-trip
        self._spinio._build_Rlist()
        return self._spinio

    def write_report(self, path="TB2J_results") -> str:
        """Write ``spiral_diagnostics.json``; returns the file path."""
        if not self.report:
            raise RuntimeError(
                "ExchangeSpiral.run() must be called before write_report()"
            )
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, "spiral_diagnostics.json")
        with open(fname, "w") as myfile:
            json.dump(self.report, myfile, indent=2)
        return fname

    def write_output(self, path="TB2J_results") -> str:
        """Write the standard TB2J tree plus the spiral diagnostic report."""
        spinio = self.get_spinio()
        spinio.write_all(path=path)
        return self.write_report(path=path)
