"""Temperature-dependent magnon solver (RPA, Callen, HP, RPA+CD).

Deep module behind the thermal-magnon architecture: validates the input
(fail-closed), runs the thermal self-consistency for the selected method,
converges the transition over a declared q-mesh sequence, classifies the
physical outcome, and evaluates explicit-temperature bands on a separate
k path. All equations follow the verified derivations in ``docs/sympy``
(01 LSWT, 02 RPA/Callen/HP + Tc, 03 anisotropy/multisite + TB2J bridge).

Energies are in eV, temperatures in K, k-points in fractional reciprocal
coordinates.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from ase.units import kB as KB_EV_PER_K

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.thermal_model import (
    ThermalModelValidationError,
    ThermalSpinModel,
    build_thermal_spin_model,
)
from TB2J.magnon.thermal_parameters import ThermalMagnonParameters
from TB2J.magnon.thermal_result import (
    MeshHistoryEntry,
    ThermalBandBlock,
    ThermalMagnonResult,
    TransitionRecord,
)

_ZERO_MODE_TOL = 1e-10
_STABILITY_TOL = -1e-9
_BISECTION_STEPS = 60
_MSTAR_FRACTION = 1e-3


def callen_magnetization(S: float, phi: float) -> float:
    """Callen formula <Sz>(phi), numerically stable in both limits.

    For phi below a cancellation-safe threshold the exact rho form is used;
    beyond it the verified large-phi asymptote m = S(S+1)/(3 phi) (1 + O(1/phi^2))
    from docs/sympy/02 takes over.
    """
    phi = max(float(phi), 0.0)
    if phi < 1e-12:
        return S
    if phi > 1e5:
        return S * (S + 1.0) / (3.0 * phi)
    rho = np.exp((2 * S + 1) * (np.log(phi) - np.log1p(phi)))
    m = ((S - phi) + (S + 1 + phi) * rho) / (1.0 - rho)
    if m < 0.0:
        return S * (S + 1.0) / (3.0 * phi)
    return m


def classical_magnetization(S: float, phi: float) -> float:
    """Exact S -> infinity limit of the Callen closure.

    With y = phi / S:  m/S = [(1 - y) + (1 + y) exp(-2/y)] / (1 - exp(-2/y)),
    which tends to S at y -> 0 and to S^2 / (3 phi) near the transition
    (the paper's classical prescription S(S+1) -> S^2).
    """
    y = float(phi) / S
    if y < 1e-8:
        return S
    if y > 1e4:
        x = (1.0 / (3.0 * y)) * (1.0 + 1.0 / y)
    else:
        e = np.exp(-2.0 / y)
        x = ((1.0 - y) + (1.0 + y) * e) / (1.0 - e)
    return S * float(np.clip(x, 0.0, 1.0))


def bose_factors(omega_eV: np.ndarray, T_K: float) -> np.ndarray:
    """Bose occupation 1/(exp(w/kT) - 1), clipped for stability."""
    if T_K <= 0:
        return np.zeros_like(omega_eV)
    x = np.clip(omega_eV / (KB_EV_PER_K * T_K), 1e-12, 700.0)
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(x)


def gamma_centered_mesh(mesh: List[int], dimensionality: int) -> np.ndarray:
    """Gamma-centered fractional q mesh, Gamma at index 0.

    Aperiodic directions (dimensionality < 3) contribute a single q = 0
    division regardless of the stored mesh entry.
    """
    divisions = []
    for axis in range(3):
        n = int(mesh[axis])
        if axis >= dimensionality or n <= 1:
            divisions.append(np.zeros(1))
        else:
            divisions.append(np.arange(n) / float(n))
    grid = np.stack(np.meshgrid(*divisions, indexing="ij"), axis=-1).reshape(-1, 3)
    # keep Gamma at index 0 (meshgrid origin)
    return grid


class ThermalSolution:
    """Converged thermal state at one temperature (internal)."""

    def __init__(self, m: float, converged: bool, iterations: int, min_energy: float):
        self.m = m
        self.converged = converged
        self.iterations = iterations
        self.min_energy = min_energy


class ThermalMagnonSolver:
    """Solve temperature-dependent magnons and Curie/Néel temperatures."""

    def __init__(self, magnon: Magnon, params: ThermalMagnonParameters):
        self.params = params
        self.model: ThermalSpinModel = build_thermal_spin_model(magnon, params)
        self._validate_method_policy()

        first_mesh = params.thermal_qmeshes[0]
        qpts = gamma_centered_mesh(first_mesh, self.model.dimensionality)
        self._min_mu = self.model.check_stability(qpts)
        if self._min_mu < _STABILITY_TOL:
            self.unstable = True
        else:
            self.unstable = False
        self._gamma_gap = self._compute_gamma_gap()

    # ------------------------------------------------------------------
    # policy checks
    # ------------------------------------------------------------------

    def _validate_method_policy(self):
        p = self.params
        model = self.model
        if (
            p.thermal_method == "rpa"
            and p.thermal_spin_regime == "quantum"
            and np.max(np.abs(model.A)) > 1e-12
            and np.all(np.isclose(model.S, 0.5))
        ):
            raise ThermalModelValidationError(
                "plain RPA with single-ion anisotropy is rejected for exact "
                "quantum S=1/2: the standard RPA ordering opens the unphysical "
                "gap 2AS although -A(Sz)^2 is then a constant. Use "
                "thermal_method='rpa_callen', 'callen', or 'hp'."
            )

    def _method_validity(self):
        """Method-validity status for known limited regimes."""
        p = self.params
        if p.thermal_spin_regime != "quantum":
            return "nominal", None
        S = self.model.S[0]
        if p.thermal_method in ("callen", "hp", "rpa_callen") and S < 1.0 + 1e-12:
            reason = (
                f"{p.thermal_method} is known to be unreliable for low spin "
                f"(S={S:g}); the Callen decoupling overestimates critical "
                "temperatures for S=1/2 systems (arXiv:2405.00477 Sec. II.D, "
                "Swendsen PRB 11, 1935 (1975))"
            )
            return "limited", reason
        return "nominal", None

    # ------------------------------------------------------------------
    # mesh-level thermal machinery (FM)
    # ------------------------------------------------------------------

    def _compute_gamma_gap(self) -> float:
        gap_q = np.zeros((1, 3))
        if self.model.order_mode == "ferromagnetic":
            vals = np.linalg.eigvalsh(self.model.M_normal_q(gap_q))
            return float(vals.min())
        vals = self._paraunitary_eig(self.model.M_bdg_q(gap_q))
        return float(np.sort(vals.real)[: vals.shape[-1] // 2].min())

    @staticmethod
    def _paraunitary_eig(H: np.ndarray):
        n = H.shape[-1] // 2
        I = np.eye(n)
        K = np.linalg.cholesky(H)
        g = np.block([[1 * I, 0 * I], [0 * I, -1 * I]])
        return np.linalg.eigvalsh(K.swapaxes(-1, -2).conj() @ g @ K)[:, n:]

    def _fm_matrices(self, qpoints, m: float, psi_q=None):
        """FM dynamical matrices H_q for the selected method (eV)."""
        model = self.model
        method = self.params.thermal_method
        S = model.S[0]
        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        Nq = qpoints.shape[0]
        gamma = np.zeros((1, 3))
        Jp = model.Jp_q(qpoints)
        Jp0 = model.Jp_q(gamma)[0]
        lam0 = model.lambda_q(gamma)[0]
        base = np.diag(m * (Jp0 + lam0).sum(axis=1))[None, :, :] - m * Jp

        if method == "rpa":
            Lam = model.Lambda_q(qpoints)
            Lam0 = model.Lambda_q(gamma)[0]
            return np.diag(m * (Jp0 + Lam0).sum(axis=1))[None, :, :] - m * Jp

        if method in ("callen", "rpa_callen"):
            if psi_q is None or psi_q.shape[0] != Nq:
                psi_q = np.zeros((Nq, model.nspin, model.nspin), dtype=complex)
            psi_bar = np.real(np.diag(psi_q.mean(axis=0)))
            pref = m / (2.0 * S * S * Nq)
            if method == "callen":
                kernel = model.Jp_q(qpoints) + model.Lambda_q(qpoints)
                # q - q' differences on the Gamma-centered mesh
                acc = np.zeros_like(base)
                for iq2 in range(Nq):
                    iqd = (np.arange(Nq) - iq2) % Nq
                    acc += kernel[iqd] * psi_q[iq2]
                H = base - pref * acc
                Jpsi = np.einsum("qac,qca->a", Jp, psi_q) / Nq
                H = H + np.diag(m / (2.0 * S * S) * Jpsi)[None, :, :]
            else:  # rpa_callen: SIA-only Callen feedback
                A_mat = np.diag(2.0 * model.A)
                acc = np.zeros_like(base)
                for iq2 in range(Nq):
                    acc += A_mat[None, :, :] * psi_q[iq2]
                H = base - pref * acc
            a_term = np.diag(model.A * (2.0 * m - (m / S**2) * (m + psi_bar)))
            return H + a_term[None, :, :]

        if method == "hp":
            # Appendix H^HP with S -> m in the exchange terms; A(2S-1) exact.
            if psi_q is None or psi_q.shape[0] != Nq:
                nq = np.zeros((Nq, model.nspin, model.nspin), dtype=complex)
            else:
                nq = psi_q  # reused slot: normal correlators n^{ab}_q
            Lam = model.Jp_q(qpoints) + model.lambda_q(qpoints)
            Jp0Lam0 = Jp0 + lam0
            H = base + np.diag(model.A * (2.0 * S - 1.0))[None, :, :]
            for iq2 in range(Nq):
                iqd = (np.arange(Nq) - iq2) % Nq
                n2 = nq[iq2]
                d = np.diag(n2).real
                H = (
                    H
                    + (
                        Jp * 0.5 * (d[:, None] + d[None, :])[None, :, :]
                        - Lam[iqd] * n2.T[None, :, :]
                        + 0.5
                        * np.diag(
                            np.einsum("ac,ca->a", Jp[iq2], n2)
                            + np.einsum("ca,ac->a", Jp[iq2], n2)
                        )[None, :, :]
                        - np.diag(np.einsum("ac,cc->a", Jp0Lam0, n2))[None, :, :]
                        - np.diag(4.0 * model.A * np.diag(n2).real)[None, :, :]
                    )
                    / Nq
                )
            return H

        raise ValueError(f"unknown thermal method {method!r}")

    def _closure_m(self, phi: float, m_prev: float) -> float:
        S = self.model.S[0]
        if self.params.thermal_method == "hp":
            return S - phi  # HP closure (paper eq. HP_mag)
        if self.params.thermal_spin_regime == "quantum":
            return callen_magnetization(S, phi)
        return classical_magnetization(S, phi)

    def _phi_of_m(self, T_K: float, qpoints, m: float, psi_q=None):
        """Bose-sum phi and minimum energy at magnetization m."""
        H = self._fm_matrices(qpoints, m, psi_q)
        Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
        w = np.linalg.eigvalsh(Hh)
        min_energy = float(w.min())
        mask = w > _ZERO_MODE_TOL
        nB = np.zeros_like(w)
        nB[mask] = bose_factors(w[mask], T_K)
        phi = float(nB.sum() / w.size)
        return phi, min_energy

    def _root_m(self, T_K: float, qpoints, psi_q=None):
        """Order-parameter root of m = closure(phi(m)); None if disordered."""
        S = self.model.S[0]
        if T_K <= 0.0:
            return S
        from scipy.optimize import brentq

        def F(m):
            phi, _ = self._phi_of_m(T_K, qpoints, m, psi_q)
            return m - self._closure_m(phi, m)

        probes = [
            S * f
            for f in (
                1.0,
                0.95,
                0.9,
                0.8,
                0.7,
                0.5,
                0.35,
                0.2,
                0.1,
                0.05,
                0.02,
                0.01,
                0.005,
                0.002,
                0.001,
            )
        ]
        f_hi = F(probes[0])
        if f_hi <= 0.0:
            return S
        for idx in range(1, len(probes)):
            if F(probes[idx]) < 0.0:
                return float(
                    brentq(F, probes[idx], probes[idx - 1], xtol=1e-14, rtol=1e-13)
                )
        return None

    def _solve_fm(self, T_K: float, qpoints) -> ThermalSolution:
        """Self-consistent FM state at temperature T_K on the given mesh."""
        p = self.params
        method = p.thermal_method
        psi_q = None
        iterations = 1
        if method in ("callen", "hp", "rpa_callen"):
            for _ in range(p.thermal_max_iterations // 50 + 1):
                m = self._root_m(T_K, qpoints, psi_q)
                if m is None or m <= 0.0:
                    return ThermalSolution(0.0, True, iterations, 0.0)
                H = self._fm_matrices(qpoints, m, psi_q)
                Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                w, U = np.linalg.eigh(Hh)
                min_energy = float(w.min())
                if method == "hp" and min_energy <= _ZERO_MODE_TOL:
                    return ThermalSolution(m, True, iterations, min_energy)
                mask = w > _ZERO_MODE_TOL
                nB = np.zeros_like(w)
                nB[mask] = bose_factors(w[mask], T_K)
                if method == "hp":
                    psi_new = (U * nB[:, None, :]) @ U.conj().transpose(0, 2, 1)
                    scale = 2.0
                else:
                    psi_new = (
                        2.0 * m * (U * nB[:, None, :]) @ U.conj().transpose(0, 2, 1)
                    )
                    scale = 2.0 * m
                if psi_q is not None and np.max(
                    np.abs(psi_new - psi_q)
                ) < p.thermal_tolerance * max(scale, 1.0):
                    psi_q = psi_new
                    iterations += 1
                    break
                psi_q = psi_new if psi_q is None else 0.5 * psi_q + 0.5 * psi_new
                iterations += 1
                if iterations > 200:
                    break
            phi, min_energy = self._phi_of_m(T_K, qpoints, m, psi_q)
            m = self._closure_m(phi, m)
            if method == "hp":
                H = self._fm_matrices(qpoints, m, psi_q)
                Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                min_energy = float(np.linalg.eigvalsh(Hh).min())
            return ThermalSolution(max(m, 0.0), True, iterations, min_energy)
        # RPA: no correlator feedback, single scalar root
        m = self._root_m(T_K, qpoints)
        if m is None or m <= 0.0:
            return ThermalSolution(0.0, True, 1, 0.0)
        phi, min_energy = self._phi_of_m(T_K, qpoints, m)
        return ThermalSolution(m, True, 1, min_energy)

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    def _tc_closed_form(self, qpoints) -> Optional[float]:
        """Linearized RPA transition (closed form), Gamma mode excluded."""
        if self.params.thermal_method != "rpa":
            return None
        if self.model.order_mode != "ferromagnetic":
            return None
        mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
        mask = mu > _ZERO_MODE_TOL
        if not mask.any():
            return 0.0
        g = float((1.0 / mu[mask]).mean()) / self.model.nspin
        if not np.isfinite(g) or g <= 0:
            return 0.0
        S = self.model.S[0]
        if self.params.thermal_spin_regime == "quantum":
            coeff = S * (S + 1.0) / 3.0
        else:
            coeff = S * S / 3.0
        return coeff / g / KB_EV_PER_K

    def _tc_self_consistent(self, qpoints, mstar) -> Optional[float]:
        """Temperature at which the self-consistent order reaches mstar."""
        lo, hi = 0.0, None
        # exponential search for a bracketing upper bound
        T = 10.0 * self._energy_scale(qpoints) / KB_EV_PER_K
        for _ in range(80):
            sol = self._solve_fm(T, qpoints)
            if self.params.thermal_method == "hp":
                if sol.min_energy <= _ZERO_MODE_TOL:
                    hi = T
                    break
            elif sol.m <= mstar:
                hi = T
                break
            lo = T
            T *= 1.6
        if hi is None:
            return None
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            sol = self._solve_fm(mid, qpoints)
            if self.params.thermal_method == "hp":
                satisfied = sol.min_energy <= _ZERO_MODE_TOL
            else:
                satisfied = sol.m <= mstar
            if satisfied:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-6 * max(hi, 1.0):
                break
        return hi

    def _energy_scale(self, qpoints) -> float:
        if self.model.order_mode == "ferromagnetic":
            mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
        else:
            mu = self._paraunitary_eig(self.model.M_bdg_q(qpoints))
        pos = mu[mu > _ZERO_MODE_TOL]
        return float(np.median(pos)) if pos.size else 1.0

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        temperatures_K=None,
        band_kpoints=None,
    ) -> ThermalMagnonResult:
        """Run the thermal calculation and return the versioned result."""
        p = self.params
        model = self.model
        method_label = p.thermal_method
        validity, reason = self._method_validity()

        if self.unstable:
            return ThermalMagnonResult(
                method=method_label,
                spin_regime=p.thermal_spin_regime,
                spin_interpretation=model.spin_interpretation,
                spins=model.S.tolist(),
                order_mode=model.order_mode,
                dimensionality=model.dimensionality,
                status="unstable_reference",
                transition=TransitionRecord(
                    kind=self._transition_kind(),
                    temperature_K=0.0,
                    converged=True,
                    method_validity=validity,
                    validity_reason=reason,
                    detail=(
                        "T=0 harmonic spectrum has negative modes "
                        f"(min = {self._min_mu:.3e} eV per unit magnetization); "
                        "the reference state is not a stable minimum"
                    ),
                ),
                mesh_history=[],
                bands=[],
            )

        zero_transition = model.dimensionality < 3 and self._gamma_gap <= _ZERO_MODE_TOL

        history: List[MeshHistoryEntry] = []
        transition_T = 0.0 if zero_transition else None
        converged_flag = True
        if zero_transition:
            history.append(
                MeshHistoryEntry(
                    qmesh=list(p.thermal_qmeshes[0]),
                    estimate_K=0.0,
                    residual=0.0,
                    iterations=0,
                    min_energy_eV=self._min_mu,
                    status="zero_transition",
                )
            )
        else:
            previous = None
            for mesh in p.thermal_qmeshes:
                qpoints = gamma_centered_mesh(mesh, model.dimensionality)
                if method_label == "rpa":
                    estimate = self._tc_closed_form(qpoints)
                else:
                    S = model.S[0]
                    mstar = _MSTAR_FRACTION * S
                    t1 = self._tc_self_consistent(qpoints, mstar)
                    if t1 is None:
                        estimate = None
                    elif method_label == "hp":
                        estimate = t1
                    else:
                        t2 = self._tc_self_consistent(qpoints, 2.0 * mstar)
                        estimate = 2.0 * t1 - t2 if t2 is not None else t1
                if estimate is None or not np.isfinite(estimate):
                    entry = MeshHistoryEntry(
                        qmesh=list(mesh),
                        estimate_K=float("nan"),
                        residual=float("nan"),
                        iterations=0,
                        min_energy_eV=self._min_mu,
                        status="unconverged",
                    )
                    history.append(entry)
                    converged_flag = False
                    transition_T = history[-2].estimate_K if len(history) > 1 else None
                    continue
                residual = (
                    abs(estimate - previous) if previous is not None else float("nan")
                )
                status = "refined"
                if previous is not None and residual < p.thermal_mesh_tolerance:
                    status = "converged"
                history.append(
                    MeshHistoryEntry(
                        qmesh=list(mesh),
                        estimate_K=estimate,
                        residual=residual,
                        iterations=0,
                        min_energy_eV=self._min_mu,
                        status=status,
                    )
                )
                transition_T = estimate
                if status == "converged":
                    converged_flag = True
                    break
                previous = estimate
            else:
                converged_flag = False

        if transition_T is None:
            overall_status = "unconverged"
        elif zero_transition:
            overall_status = "zero_transition"
        else:
            overall_status = "ok"

        strict_failure = (
            p.thermal_strict
            and not zero_transition
            and not (converged_flag and transition_T is not None)
        )
        if strict_failure:
            raise RuntimeError(
                "thermal_strict: the q-mesh sequence "
                f"{p.thermal_qmeshes} failed the tolerance "
                f"{p.thermal_mesh_tolerance} K for the "
                f"{self._transition_kind()}"
            )

        bands = []
        temperatures = (
            list(temperatures_K)
            if temperatures_K is not None
            else list(p.thermal_temperatures)
        )
        band_kpoints = (
            np.atleast_2d(np.asarray(band_kpoints, dtype=float))
            if band_kpoints is not None
            else None
        )
        if temperatures and band_kpoints is not None and not self.unstable:
            final_mesh = p.thermal_qmeshes[-1]
            qpoints = gamma_centered_mesh(final_mesh, model.dimensionality)
            for T in temperatures:
                if zero_transition and T > _ZERO_MODE_TOL:
                    # ordered solution does not exist; report zero order
                    bands.append(
                        ThermalBandBlock(
                            temperature_K=float(T),
                            kpoints=band_kpoints.copy(),
                            energies_eV=np.zeros((band_kpoints.shape[0], model.nspin)),
                            order_parameters=np.zeros(band_kpoints.shape[0]),
                            status="zero_transition",
                        )
                    )
                    continue
                sol = self._solve_fm(float(T), qpoints)
                H = self._fm_matrices(band_kpoints, sol.m)
                Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                energies = np.linalg.eigvalsh(Hh)
                block_status = "ordered"
                if (
                    self.params.thermal_method == "hp"
                    and sol.min_energy <= _ZERO_MODE_TOL
                ):
                    block_status = "hp_breakdown"
                bands.append(
                    ThermalBandBlock(
                        temperature_K=float(T),
                        kpoints=band_kpoints.copy(),
                        energies_eV=energies,
                        order_parameters=np.full(band_kpoints.shape[0], sol.m),
                        status=block_status,
                        magnetization=np.full(model.nspin, sol.m),
                    )
                )

        transition = TransitionRecord(
            kind=self._transition_kind(),
            temperature_K=float(transition_T)
            if transition_T is not None
            else float("nan"),
            converged=bool(converged_flag and transition_T is not None),
            method_validity=validity,
            validity_reason=reason,
        )
        if method_label == "hp" and transition_T is not None:
            final_mesh = p.thermal_qmeshes[-1]
            qpoints = gamma_centered_mesh(final_mesh, model.dimensionality)
            sol = self._solve_fm(float(transition_T), qpoints)
            transition.breakdown_magnetization = float(sol.m)

        return ThermalMagnonResult(
            method=method_label,
            spin_regime=p.thermal_spin_regime,
            spin_interpretation=model.spin_interpretation,
            spins=model.S.tolist(),
            order_mode=model.order_mode,
            dimensionality=model.dimensionality,
            status=overall_status,
            transition=transition,
            mesh_history=history,
            bands=bands,
        )

    def _transition_kind(self) -> str:
        if self.params.thermal_method == "hp":
            return "temperature_hp_breakdown"
        if self.model.order_mode == "ferromagnetic":
            return "curie_temperature"
        return "neel_temperature"
