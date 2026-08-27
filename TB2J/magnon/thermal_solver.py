"""Temperature-dependent magnon solver (RPA, Callen, HP, RPA+CD).

Deep module behind the thermal-magnon architecture: validates the input
(fail-closed), runs the thermal self-consistency for the selected method,
converges the transition over a declared q-mesh sequence, classifies the
physical outcome, and evaluates explicit-temperature bands on a separate
k path. All equations follow the verified derivations in ``docs/sympy``
(01 LSWT, 02 RPA/Callen/HP + Tc, 03 anisotropy/multisite + TB2J bridge,
04 local-frame AFM Nambu RPA).

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
    _metric_positive_modes,
    _paraunitary_eigenvalues,
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
_HP_BREAKDOWN_TOL = -1e-11
_STABILITY_TOL = -1e-9
_BISECTION_STEPS = 60
_MSTAR_FRACTION = 5e-3


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


def brillouin_function(S: float, x):
    """Weiss single-site Brillouin function B_S(x), stable near x = 0.

    ``B_S(x) = (1 + 1/2S) coth((1 + 1/2S) x) - (1/2S) coth(x/2S)`` is the
    exact thermal average ``<S^z> / S`` of a spin S in a field along z
    (arXiv:2405.00477 Sec. II.A eq. mfa-mag).  The two ``coth`` terms
    cancel catastrophically at small x, so the small-argument branch uses
    the cancellation-free series ``B_S(x) = x(a^2-b^2)/3 - x^3(a^4-b^4)/45
    + O(x^5)`` with ``a = 1 + 1/2S``, ``b = 1/2S``, whose leading term is
    the linearization ``B_S -> (S+1)x/(3S)`` behind
    ``k_B Tc^MFA = J_0 S(S+1)/3``.
    """
    x = np.asarray(x, dtype=float)
    a = 1.0 + 1.0 / (2.0 * S)
    b = 1.0 / (2.0 * S)
    small = np.abs(x) < 1e-4
    x_series = np.where(small, x, 0.0)  # masked -> 0: no overflow in x^3
    x_direct = np.where(small, 1.0, x)  # masked -> 1: no division by tanh(0)
    series = x_series * (a * a - b * b) / 3.0 - x_series**3 * (a**4 - b**4) / 45.0
    direct = a / np.tanh(a * x_direct) - b / np.tanh(b * x_direct)
    return np.where(small, series, direct)


def bose_factors(omega_eV: np.ndarray, T_K: float) -> np.ndarray:
    """Bose occupation 1/(exp(w/kT) - 1), clipped for stability."""
    if T_K <= 0:
        return np.zeros_like(omega_eV)
    x = np.clip(omega_eV / (KB_EV_PER_K * T_K), 1e-12, 700.0)
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(x)


def classical_bose_factors(omega_eV: np.ndarray, T_K: float) -> np.ndarray:
    """Classical-limit occupation kBT/omega - 1/2 (S -> infinity of Bose).

    This is the occupation implied by the paper's classical prescription
    (replace S(S+1) by S^2); it is clipped at zero for modes below 2 kBT
    where the classical expansion is invalid (low-T saturation of classical
    spins).
    """
    if T_K <= 0:
        return np.zeros_like(omega_eV)
    return np.clip(KB_EV_PER_K * T_K / omega_eV - 0.5, 0.0, None)


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

    def __init__(self, m, converged, iterations, min_energy, psi_q=None):
        self.m = m
        self.converged = converged
        self.iterations = iterations
        self.min_energy = min_energy
        self.psi_q = psi_q


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
        self._fcache = {}
        self._current_mesh_shape = None
        # Classical regime: the paper's prescription solves the quantum
        # equations at S_eff = K S, T_eff = K^2 T and rescales outputs
        # (S -> infinity limit); K = 200 gives <0.5% discretization error.
        self._K = 200.0 if params.thermal_spin_regime == "classical" else 1.0

    # ------------------------------------------------------------------
    # policy checks
    # ------------------------------------------------------------------

    def _validate_method_policy(self):
        p = self.params
        model = self.model
        if model.order_mode == "bipartite_afm" and p.thermal_method == "mfa":
            raise ThermalModelValidationError(
                "thermal_method='mfa' is not implemented for "
                "thermal_order_mode='bipartite_afm': the Weiss single-site "
                "baseline is defined here only for the ferromagnetic order "
                "mode. Use thermal_method='rpa' for the supported bipartite "
                "AFM Nambu RPA calculation."
            )
        if model.order_mode == "bipartite_afm" and p.thermal_method != "rpa":
            raise ThermalModelValidationError(
                f"thermal_method={p.thermal_method!r} is not implemented for "
                "thermal_order_mode='bipartite_afm': its Callen/HP "
                "correlator closure is derived only for the ferromagnetic "
                "normal (non-Nambu) spectrum. Use thermal_method='rpa' for "
                "the supported bipartite AFM Nambu RPA calculation."
            )
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
        if p.thermal_method == "mfa":
            if self.model.dimensionality < 3 and self._gamma_gap <= _ZERO_MODE_TOL:
                return "limited", (
                    "MFA returns a finite transition for an isotropic "
                    f"{self.model.dimensionality}D ferromagnet, in direct "
                    "violation of the Mermin-Wagner theorem "
                    "(arXiv:2405.00477 Sec. II.A): the uncorrelated Weiss "
                    "baseline cannot represent the critical long-wavelength "
                    "fluctuations that enforce a zero transition. Treat the "
                    "value as a baseline only; use thermal_method='rpa' for "
                    "the correlated zero-transition result."
                )
            return "nominal", None
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
        # Use the same metric diagonalization as the AFM Nambu spectrum so
        # the Goldstone detection noise floor (1e-13-ish) sits safely below
        # _ZERO_MODE_TOL; the Cholesky route leaves ~1e-9 residual noise at
        # exactly singular Goldstone blocks, which would defeat the
        # zero-transition (Mermin-Wagner) policy in 1D/2D.
        return float(self._afm_positive_modes(gap_q, 1.0).min())

    def _fourier_cache(self, qpoints):
        """Cache Fourier-space quantities for a fixed q-point set."""
        key = (qpoints.shape[0], qpoints.tobytes())
        cache = self._fcache.get(key)
        if cache is None:
            gamma = np.zeros((1, 3))
            cache = {
                "Jp": self.model.Jp_q(qpoints),
                "Lam": self.model.Lambda_q(qpoints),
                "Jp0": self.model.Jp_q(gamma)[0],
                "lam0": self.model.lambda_q(gamma)[0],
                "lam": self.model.lambda_q(qpoints),
            }
            self._fcache[key] = cache
        return cache

    def _circular_conv(self, kernel, correl):
        """acc[i] = sum_j kernel[(i - j) % N] * correl[j] per matrix element.

        FFT circular convolution over the q grid (meshgrid 'ij' order with
        Gamma at index 0 makes index differences modular per axis).
        """
        Nq = kernel.shape[0]
        shape = self._mesh_shape(Nq)
        if shape is None:
            acc = np.zeros_like(kernel, dtype=np.result_type(kernel, correl))
            for iq2 in range(Nq):
                iqd = (np.arange(Nq) - iq2) % Nq
                acc += kernel[iqd] * correl[iq2]
            return acc
        k = kernel.reshape(shape + kernel.shape[1:])
        c = correl.reshape(shape + correl.shape[1:])
        fk = np.fft.fftn(k, axes=(0, 1, 2))
        fc = np.fft.fftn(c, axes=(0, 1, 2))
        return np.fft.ifftn(fk * fc, axes=(0, 1, 2)).reshape(kernel.shape)

    def _mesh_shape(self, Nq):
        shape = self._current_mesh_shape
        if shape is not None and int(np.prod(shape)) == Nq:
            return shape
        return None

    def _fm_matrices(self, qpoints, m: float, psi_q=None, cache=None):
        """FM dynamical matrices H_q for the selected method (eV)."""
        model = self.model
        method = self.params.thermal_method
        S = self._S_eff()
        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        Nq = qpoints.shape[0]
        if cache is None:
            cache = self._fourier_cache(qpoints)
        Jp = cache["Jp"]
        Jp0 = cache["Jp0"]
        lam0 = cache["lam0"]
        base = np.diag(m * (Jp0 + lam0).sum(axis=1))[None, :, :] - m * Jp

        if method == "rpa":
            Lam0 = cache["Lam"][0]
            return np.diag(m * (Jp0 + Lam0).sum(axis=1))[None, :, :] - m * Jp

        if method in ("callen", "rpa_callen"):
            if psi_q is None or psi_q.shape[0] != Nq:
                psi_q = np.zeros((Nq, model.nspin, model.nspin), dtype=complex)
            psi_bar = np.real(np.diag(psi_q.mean(axis=0)))
            pref = m / (2.0 * S * S * Nq)
            if method == "callen":
                kernel = Jp + cache["Lam"]
                acc = self._circular_conv(kernel, psi_q)
                H = base - pref * acc
                Jpsi = np.einsum("qac,qca->a", Jp, psi_q) / Nq
                H = H + np.diag(m / (2.0 * S * S) * Jpsi)[None, :, :]
            else:  # rpa_callen: SIA-only Callen feedback
                A_mat = np.diag(2.0 * model.A)
                acc = self._circular_conv(
                    np.broadcast_to(A_mat, (Nq, model.nspin, model.nspin)), psi_q
                )
                H = base - pref * acc
            a_term = np.diag(model.A * (2.0 * m - (m / S**2) * (m + psi_bar)))
            return H + a_term[None, :, :]

        if method == "hp":
            if psi_q is None or psi_q.shape[0] != Nq:
                nq = np.zeros((Nq, model.nspin, model.nspin), dtype=complex)
            else:
                nq = psi_q  # slot reused: normal correlators n^{ab}_q
            d = np.real(np.diag(nq.mean(axis=0)))
            # ``base`` already contains m = S - d in the HP closure.
            # Do not add the appendix's J_q*d - (J_0+lambda_0)*d terms:
            # those are exactly the S -> m conversion and would subtract
            # the magnon stiffness twice.
            kernel = Jp + cache["lam"]
            conv = self._circular_conv(kernel, nq.transpose(0, 2, 1))
            jpsi = (
                0.5
                * (np.einsum("qac,qca->a", Jp, nq) + np.einsum("qca,qac->a", Jp, nq))
                / Nq
            )
            H = base + np.diag(model.A * (2.0 * S - 1.0))[None, :, :]
            H = H + (-conv / Nq + np.diag(jpsi - 4.0 * model.A * d)[None, :, :])
            return H

        raise ValueError(f"unknown thermal method {method!r}")

    def _S_eff(self) -> float:
        return self.model.S[0] * self._K

    def _closure_m(self, phi: float, m_prev: float) -> float:
        S = self._S_eff()
        if self.params.thermal_method == "hp":
            return S - phi  # HP closure (paper eq. HP_mag)
        return callen_magnetization(S, phi)

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

    def _afm_positive_modes(self, qpoints, m: float) -> np.ndarray:
        """Positive Nambu energies for the bipartite AFM RPA spectrum.

        Metric diagonalization keeps exact Goldstone blocks at the
        ~1e-13 noise floor (below ``_ZERO_MODE_TOL``).  Its reality gate
        returns negative markers for complex unstable blocks, so an
        instability cannot masquerade as a Goldstone zero.
        """
        return _metric_positive_modes(m * self.model.M_bdg_q(qpoints))

    def _afm_phi_of_m(self, T_eff: float, qpoints, m: float):
        """Local AFM Nambu normal contraction ``Phi(m, T)``.

        The RPA/Callen closure uses docs/sympy/04 eq. (6), not the FM's
        bare Bose mean:

        ``Phi = mean'[ (A_q (2 n_B + 1) / omega_q - 1) / 2 ]``.

        Model validation enforces the equivalent two-sublattice form
        ``A_q = m K_0``.  Thus ``A_q / omega_q = K_0 / eps_q`` supplies
        both the Bogoliubov amplification of the finite-T occupation and
        the ``v_q^2`` zero-point depletion at ``T = 0``.  Exact
        Goldstone modes are excluded from this finite-mesh regulator,
        consistently with the weighted transition kernel.
        """
        w = self._afm_positive_modes(qpoints, m)
        mask = w > _ZERO_MODE_TOL
        if not mask.any():
            return 0.0, float(w.min())
        nB = bose_factors(w[mask], T_eff)
        phi = float(
            (0.5 * (m * self._afm_K0() / w[mask] * (2.0 * nB + 1.0) - 1.0)).mean()
        )
        return phi, float(w.min())

    def _solve_afm_rpa(self, T_K, qpoints) -> ThermalSolution:
        """Self-consistent staggered order in the supported AFM RPA branch.

        The Callen order relation is evaluated with the local Nambu
        normal contraction above, which has the same
        ``K_0 / eps_q^2`` critical kernel as ``_tc_closed_form``.  The
        classical branch mirrors the FM prescription: solve at
        ``S_eff = K S`` and ``T_eff = K^2 T``, then report ``m/K`` and
        energies divided by ``K``.
        """
        from scipy.optimize import brentq

        K = self._K
        S = self._S_eff()
        T_eff = T_K * K * K
        if T_eff <= 0.0:
            # At T=0, A_q / omega_q is m-independent, so this direct
            # closure evaluation retains the Nambu v_q^2 depletion.
            phi, minimum = self._afm_phi_of_m(0.0, qpoints, S)
            return ThermalSolution(
                self._closure_m(phi, S) / K,
                True,
                1,
                minimum / K,
            )

        def residual(m):
            phi, _ = self._afm_phi_of_m(T_eff, qpoints, m)
            return m - self._closure_m(phi, m)

        probes = S * np.array(
            [1.0, 0.95, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        )
        if residual(probes[0]) <= 0.0:
            m = S
        else:
            m = None
            for low, high in zip(probes[1:], probes[:-1]):
                if residual(low) < 0.0:
                    m = float(brentq(residual, low, high, xtol=1e-14, rtol=1e-13))
                    break
            if m is None:
                return ThermalSolution(0.0, True, 1, 0.0)
        _, minimum = self._afm_phi_of_m(T_eff, qpoints, m)
        return ThermalSolution(m / K, True, 1, minimum / K)

    def _root_m(self, T_eff: float, qpoints, psi_q=None):
        """Order-parameter root of m = closure(phi(m)); None if disordered."""
        S = self._S_eff()
        if T_eff <= 0.0:
            return S
        from scipy.optimize import brentq

        def F(m):
            phi, _ = self._phi_of_m(T_eff, qpoints, m, psi_q)
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

    def _solve_fm(self, T_K, qpoints, warm=None) -> ThermalSolution:
        """Self-consistent FM state at temperature T_K on the given mesh.

        ``warm`` is an optional (m, psi_q) state from a nearby temperature
        used as continuation seed so ordered CD/HP solutions survive beyond
        the RPA transition.
        """
        p = self.params
        method = p.thermal_method
        K = self._K
        T_eff = T_K * K * K
        psi_q = None
        m_last = 0.95 * self._S_eff()
        if warm is not None:
            if warm[1] is not None:
                psi_q = warm[1] * K
            if warm[0] is not None:
                m_last = warm[0] * K
        iterations = 1
        if method == "hp":
            # Direct port of the verified docs/sympy/02 HP scheme: m is
            # slaved to the current Bose sum (m = S - phi), correlators are
            # iterated with adaptive mixing, and negative iterate energies
            # signal the HP breakdown.
            S_eff = self._S_eff()
            na = self.model.nspin
            psi = (
                psi_q
                if psi_q is not None
                else np.zeros((qpoints.shape[0], na, na), dtype=complex)
            )
            mix, prev_res, bad = 0.4, np.inf, 0
            m = S_eff
            w_min = 0.0
            converged_state = None
            for it in range(2000):
                phi = float(np.real(np.trace(psi, axis1=1, axis2=2).mean()) / na)
                m = S_eff - phi
                H = self._fm_matrices(qpoints, m, psi)
                Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                w, U = np.linalg.eigh(Hh)
                w_min = float(w.min())
                iterations = it + 1
                if w_min < _HP_BREAKDOWN_TOL:
                    return ThermalSolution(m / K, True, iterations, w_min / K, psi / K)
                mask = w > _ZERO_MODE_TOL
                nB = np.zeros_like(w)
                nB[mask] = bose_factors(w[mask], T_eff)
                psi_new = (U * nB[:, None, :]) @ U.conj().transpose(0, 2, 1)
                res = float(np.max(np.abs(psi_new - psi)))
                if res < 1e-9 * (1.0 + float(np.abs(psi_new).max())):
                    converged_state = (m, w_min, psi_new)
                    break
                if res > prev_res:
                    bad += 1
                else:
                    bad = max(0, bad - 1)
                if bad > 3:
                    mix = max(0.002, 0.4 * mix)
                    bad = 0
                prev_res = res
                psi = (1.0 - mix) * psi + mix * psi_new
            if converged_state is None:
                return ThermalSolution(m / K, False, iterations, w_min / K, psi / K)
            m, w_min, psi = converged_state
            return ThermalSolution(m / K, True, iterations, w_min / K, psi / K)

        if method in ("callen", "rpa_callen"):
            # Paper's nested closure: for each Callen magnetization, converge
            # the correlator/Bose occupations first, then update phi.  Solving
            # a scalar root against a stale correlator selects an unphysical
            # high-m branch near the transition and causes critical slowdown.
            S_eff = self._S_eff()
            na = self.model.nspin
            phi = 1e-8
            if psi_q is None:
                psi = np.zeros((qpoints.shape[0], na, na), dtype=complex)
            else:
                psi = psi_q
            m = m_last
            min_energy = 0.0
            iterations = 0
            for outer in range(400):
                m = self._closure_m(phi, m)
                if not np.isfinite(m) or m <= 0.0:
                    return ThermalSolution(0.0, True, iterations, min_energy / K, None)
                mix, previous_residual, bad = p.thermal_mixing, np.inf, 0
                for inner in range(p.thermal_max_iterations):
                    H = self._fm_matrices(qpoints, m, psi)
                    Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                    w, U = np.linalg.eigh(Hh)
                    min_energy = float(w.min())
                    mask = w > _ZERO_MODE_TOL
                    nB = np.zeros_like(w)
                    nB[mask] = bose_factors(w[mask], T_eff)
                    psi_new = (
                        2.0 * m * (U * nB[:, None, :]) @ U.conj().transpose(0, 2, 1)
                    )
                    residual = float(np.max(np.abs(psi_new - psi)))
                    iterations += 1
                    if residual < p.thermal_tolerance * (
                        1.0 + float(np.abs(psi_new).max())
                    ):
                        psi = psi_new
                        break
                    if residual > previous_residual:
                        bad += 1
                    else:
                        bad = max(0, bad - 1)
                    if bad > 3:
                        mix = max(0.002, 0.4 * mix)
                        bad = 0
                    previous_residual = residual
                    psi = (1.0 - mix) * psi + mix * psi_new
                else:
                    return ThermalSolution(
                        m / K, False, iterations, min_energy / K, psi / K
                    )
                phi_new = float(nB.sum() / w.size)
                if abs(phi_new - phi) < p.thermal_tolerance:
                    return ThermalSolution(
                        m / K, True, iterations, min_energy / K, psi / K
                    )
                phi = 0.5 * phi + 0.5 * phi_new
            return ThermalSolution(m / K, False, iterations, min_energy / K, psi / K)
        # RPA: no correlator feedback, single scalar root
        m = self._root_m(T_eff, qpoints)
        if m is None or m <= 0.0:
            return ThermalSolution(0.0, True, 1, 0.0)
        phi, min_energy = self._phi_of_m(T_eff, qpoints, m)
        return ThermalSolution(m / K, True, 1, min_energy / K)

    # ------------------------------------------------------------------
    # Weiss mean-field (MFA) baseline
    # ------------------------------------------------------------------

    def _weiss_field(self) -> float:
        """Single-site Weiss field per unit magnetization (eV).

        The mean-field exchange field of site ``a`` is the Gamma row sum of
        the transverse exchange kernel, ``sum_b (J'_0 + Lambda_0)[a, b]``:
        the paper's ``J_0`` plus ``lambda_0`` and the ``2 A`` mean-field
        single-ion-anisotropy field (``3 k_B Tc = S(S+1)(J_0 + 2A)``,
        arXiv:2405.00477 Secs. II.A/III.C).  Equivalent spins make the
        site average the natural single-site value.
        """
        gamma = np.zeros((1, 3))
        Jp0 = self.model.Jp_q(gamma)[0]
        Lam0 = self.model.Lambda_q(gamma)[0]
        return float(np.real((Jp0 + Lam0).sum(axis=1)).mean())

    def _solve_mfa(self, T_K: float) -> float:
        """Weiss single-site Brillouin self-consistency m(T), 0 <= m <= S.

        Solves ``m = S B_S(beta J_W S m)`` for the Weiss field ``J_W`` in
        the effective (classical-mapping) variables and rescales by ``K``,
        exactly like the magnon closures.  ``B_S`` is strictly concave
        with ``B_S(0) = 0`` and ``B_S(x) < 1``, so below the transition
        the residual ``g(m) = m - S B_S(beta J_W S m)`` is negative just
        above zero (the curve starts above the diagonal) and positive at
        ``m = S`` (the curve ends below it): ``[eps*S, S]`` brackets the
        unique ordered root for every ``T < Tc``, however close, and
        root finding converges superlinearly instead of the critically
        slow ``1/sqrt(n)`` decay of the plain fixed-point iteration.  The
        linearized slope at the origin decides existence exactly
        (``k_B Tc = J_W S(S+1)/3``): at or above it the only solution is
        the disordered ``m = 0`` one, returned directly.
        """
        from scipy.optimize import brentq

        K = self._K
        S = self._S_eff()
        T_eff = T_K * K * K
        if T_eff <= 0.0:
            return S / K
        beta = 1.0 / (KB_EV_PER_K * T_eff)
        J0 = self._weiss_field()
        if beta * J0 * S * (S + 1.0) / 3.0 <= 1.0:
            return 0.0

        def residual(m):
            return m - S * float(brillouin_function(S, beta * J0 * S * m))

        lo = 1e-12 * S
        root = brentq(
            residual,
            lo,
            S,
            xtol=1e-14 * S,
            rtol=8.881784197001252e-16,
            maxiter=self.params.thermal_max_iterations,
        )
        return float(root) / K

    def _tc_mfa(self) -> float:
        """Analytic linearized Weiss transition temperature (K).

        Linearizing the Brillouin closure at ``m -> 0`` gives
        ``k_B Tc = J_W S(S+1)/3`` (arXiv:2405.00477 eq. eq:T_mf).  The
        classical regime follows the shared ``S_eff = K S`` prescription,
        i.e. ``k_B Tc -> J_W S^2/3``; the result is q-independent, so no
        mesh sequence is needed.
        """
        S = self._S_eff()
        coeff = S * (S + 1.0) / 3.0 / (self._K * self._K)
        return coeff * self._weiss_field() / KB_EV_PER_K

    def _calculate_mfa(self, validity, reason, temperatures) -> ThermalMagnonResult:
        """Bandless MFA result: analytic transition plus Weiss m(T) blocks.

        MFA neglects correlations entirely, so it has no
        temperature-dependent magnon spectrum to report: each requested
        temperature is answered with a bandless block carrying only the
        Brillouin order parameter ``m(T)`` (empty k-points and energies,
        per-site magnetization ``m``; the same ``zero_transition`` block
        status the spectrum methods use once the ordered solution is
        gone), and the exported metadata labels the method a
        thermodynamic baseline.
        """
        p = self.params
        model = self.model
        tc = self._tc_mfa()
        transition = TransitionRecord(
            kind=self._transition_kind(),
            temperature_K=tc,
            converged=True,
            method_validity=validity,
            validity_reason=reason,
            detail=(
                "Weiss mean-field baseline: analytic linearized single-site "
                "Brillouin transition, mesh-independent and bandless by "
                "construction (k_B Tc = J_0 S(S+1)/3 in the quantum regime, "
                "the S(S+1) -> S^2 classical prescription otherwise)"
            ),
        )
        bands = []
        for T in temperatures:
            m = self._solve_mfa(float(T))
            bands.append(
                ThermalBandBlock(
                    temperature_K=float(T),
                    kpoints=np.zeros((0, 3)),
                    energies_eV=np.zeros((0, model.nspin)),
                    order_parameters=np.array([m]),
                    status="ordered" if m > 0.0 else "zero_transition",
                    magnetization=np.full(model.nspin, m),
                )
            )
        return ThermalMagnonResult(
            method="mfa",
            spin_regime=p.thermal_spin_regime,
            spin_interpretation=model.spin_interpretation,
            spins=model.S.tolist(),
            order_mode=model.order_mode,
            dimensionality=model.dimensionality,
            status="ok",
            transition=transition,
            mesh_history=[
                MeshHistoryEntry(
                    qmesh=list(p.thermal_qmeshes[0]),
                    estimate_K=tc,
                    residual=0.0,
                    iterations=0,
                    min_energy_eV=self._min_mu,
                    status="converged",
                )
            ],
            bands=bands,
        )

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    def _set_mesh_shape(self, mesh):
        shape = []
        for axis in range(3):
            n = int(mesh[axis])
            shape.append(n if (axis < self.model.dimensionality and n > 1) else 1)
        self._current_mesh_shape = tuple(shape)

    def _afm_K0(self) -> float:
        """On-site Nambu mean field K_0 per unit magnetization (eV).

        With the RPA scaling A_q = m K_0 (docs/sympy/04 Section 4), K_0 is
        the q-independent diagonal of the BdG normal block, i.e. the
        sublattice exchange field at Gamma per unit magnetization.
        """
        n = self.model.nspin
        H0 = self.model.M_bdg_q(np.zeros((1, 3)))[0]
        return float(np.mean(np.real(np.diag(H0)[:n])))

    def _tc_closed_form(self, qpoints) -> Optional[float]:
        """Linearized RPA transition, using the relevant positive spectrum."""
        if self.params.thermal_method != "rpa":
            return None
        if self.model.order_mode == "ferromagnetic":
            mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
            mask = mu > _ZERO_MODE_TOL
            if not mask.any():
                return 0.0
            g = float((1.0 / mu[mask]).mean()) / self.model.nspin
        else:
            # AFM Nambu RPA (docs/sympy/04 eq. TN): the local contraction
            # n_q = 1/2[A_q(2n_B+1)/omega_q - 1] with A_q = m K_0 and
            # omega_q = m epsilon_q has an m-independent ratio A_q/omega_q =
            # K_0/epsilon_q, so as m -> 0 the Bose-divergent part of the
            # site occupation is (k_B T/m) K_0/epsilon_q^2.  Callen's
            # linearization then gives k_B T_N = S(S+1)/(3 mean[K_0/eps^2]).
            # A scalar 1/epsilon weight is wrong: it would converge in 2D
            # and violate Mermin-Wagner.  Exact Goldstone modes
            # (epsilon_q <= _ZERO_MODE_TOL, e.g. Gamma on any mesh) are
            # excluded from the mean: that exclusion is the finite-mesh
            # regulator of the weighted kernel.
            mu = self._afm_positive_modes(qpoints, 1.0)
            mask = mu > _ZERO_MODE_TOL
            if not mask.any():
                return 0.0
            g = float((self._afm_K0() / mu[mask] ** 2).mean())
        if not np.isfinite(g) or g <= 0:
            return 0.0
        S = self._S_eff()
        coeff = S * (S + 1.0) / 3.0 / (self._K * self._K)
        return coeff / g / KB_EV_PER_K

    def _tc_self_consistent(self, qpoints, mstar) -> Optional[float]:
        """Temperature at which the self-consistent order reaches mstar.

        Anchored on the RPA closed form for the initial bracket, then solved
        by bisection on the (monotone) ordered solution m(T).
        """
        from scipy.optimize import brentq

        if self.params.thermal_method in ("callen", "rpa_callen"):
            # Follow docs/sympy/02 T_of_m: hold m fixed while converging the
            # CD occupations, then bisect T on phi(T, m).  Iterating m(T)
            # directly is critically slow and can choose the wrong branch.
            K = self._K
            m_fixed = mstar * K
            S = self._S_eff()
            target_phi = brentq(
                lambda phi: self._closure_m(phi, m_fixed) - m_fixed,
                0.0,
                1e8,
            )
            na = self.model.nspin
            state_psi = None

            def phi_at_fixed_m(T):
                nonlocal state_psi
                psi = state_psi
                if psi is None:
                    psi = np.zeros((qpoints.shape[0], na, na), dtype=complex)
                T_eff = T * K * K
                for _ in range(self.params.thermal_max_iterations):
                    H = self._fm_matrices(qpoints, m_fixed, psi)
                    Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                    w, U = np.linalg.eigh(Hh)
                    mask = w > _ZERO_MODE_TOL
                    nB = np.zeros_like(w)
                    nB[mask] = bose_factors(w[mask], T_eff)
                    psi_new = (
                        2.0
                        * m_fixed
                        * (U * nB[:, None, :])
                        @ U.conj().transpose(0, 2, 1)
                    )
                    if np.max(np.abs(psi_new - psi)) < self.params.thermal_tolerance * (
                        1.0 + float(np.abs(psi_new).max())
                    ):
                        state_psi = psi_new
                        return float(nB.sum() / w.size)
                    psi = 0.5 * psi + 0.5 * psi_new
                return None

            mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
            mask = mu > _ZERO_MODE_TOL
            if not mask.any():
                return 0.0
            g = float((1.0 / mu[mask]).mean()) / self.model.nspin
            guess = S * (S + 1.0) / (3.0 * K * K * g * KB_EV_PER_K)
            lo, hi = 0.1 * guess, 5.0 * guess
            for _ in range(8):
                value = phi_at_fixed_m(lo)
                if value is not None and value < target_phi:
                    break
                lo *= 0.4
            else:
                return None
            for _ in range(8):
                value = phi_at_fixed_m(hi)
                if value is not None and value > target_phi:
                    break
                hi *= 2.5
            else:
                return None
            for _ in range(34):
                mid = 0.5 * (lo + hi)
                value = phi_at_fixed_m(mid)
                if value is None:
                    return None
                if value < target_phi:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        state = {"m": None, "psi": None}  # m=None handled by the seed

        def m_of_T(T):
            sol = self._solve_fm(T, qpoints, warm=(state["m"], state["psi"]))
            if self.params.thermal_method == "hp":
                if not sol.converged:
                    return -1.0
                return sol.min_energy - _HP_BREAKDOWN_TOL
            return sol.m - mstar

        if self.params.thermal_method == "hp":
            # The HP breakdown lies on the RPA energy scale.  Derive that
            # scale directly rather than beginning an exponential search at
            # an arbitrary band-energy estimate.
            mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
            mask = mu > _ZERO_MODE_TOL
            if not mask.any():
                return 0.0
            g = float((1.0 / mu[mask]).mean()) / self.model.nspin
            S = self._S_eff()
            guess = S * (S + 1.0) / (3.0 * self._K * self._K * g * KB_EV_PER_K)
            lo, hi = 0.5 * guess, 1.5 * guess
            while m_of_T(lo) < 0.0 and lo > 1e-6 * guess:
                lo *= 0.5
            if m_of_T(lo) < 0.0:
                return None
            while m_of_T(hi) >= 0.0:
                hi *= 1.5
                if hi > 20.0 * guess:
                    return None
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if m_of_T(mid) < 0.0:
                    hi = mid
                else:
                    lo = mid
            return hi

        guess = self._tc_closed_form(qpoints)
        if guess is None or not np.isfinite(guess) or guess <= 0:
            scale = self._energy_scale(qpoints)
            guess = scale / KB_EV_PER_K
        lo, hi = 0.2 * guess, 5.0 * guess
        f_lo, f_hi = m_of_T(lo), m_of_T(hi)
        widen = 0
        while f_lo <= 0.0 and widen < 6:
            lo *= 0.4
            f_lo = m_of_T(lo)
            widen += 1
        if f_lo <= 0.0:
            return None
        widen = 0
        while f_hi >= 0.0 and widen < 6:
            hi *= 2.5
            f_hi = m_of_T(hi)
            widen += 1
        if f_hi >= 0.0:
            return hi
        return float(brentq(m_of_T, lo, hi, xtol=1e-6 * guess, rtol=1e-6))

    def _energy_scale(self, qpoints) -> float:
        if self.model.order_mode == "ferromagnetic":
            mu = np.linalg.eigvalsh(self.model.M_normal_q(qpoints))
        else:
            mu = _paraunitary_eigenvalues(self.model.M_bdg_q(qpoints))
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

        if method_label == "mfa":
            # Bandless Weiss baseline: analytic, mesh-independent
            # transition; deliberately bypasses the Mermin-Wagner
            # zero-transition gate (flagged as limited validity instead).
            # Requested temperatures still get bandless m(T) blocks.
            temperatures = (
                list(temperatures_K)
                if temperatures_K is not None
                else list(p.thermal_temperatures)
            )
            return self._calculate_mfa(validity, reason, temperatures)
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
                self._set_mesh_shape(mesh)
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
            self._set_mesh_shape(final_mesh)
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
                sol = (
                    self._solve_fm(float(T), qpoints)
                    if model.order_mode == "ferromagnetic"
                    else self._solve_afm_rpa(float(T), qpoints)
                )
                if model.order_mode == "ferromagnetic":
                    H = self._fm_matrices(band_kpoints, sol.m * self._K) / self._K
                    Hh = 0.5 * (H + H.conj().transpose(0, 2, 1))
                    energies = np.linalg.eigvalsh(Hh)
                    magnetization = np.full(model.nspin, sol.m)
                else:
                    # AFM mirrors the FM classical map: ``sol.m`` is
                    # physical, so restore m_eff for the RPA spectrum
                    # and divide its energies by K.
                    energies = (
                        self._afm_positive_modes(band_kpoints, sol.m * self._K)
                        / self._K
                    )
                    magnetization = sol.m * np.sign(model.magmoms @ model.axis)
                block_status = "ordered"
                if self.params.thermal_method == "hp" and (
                    not sol.converged or sol.min_energy < _HP_BREAKDOWN_TOL
                ):
                    block_status = "hp_breakdown"
                bands.append(
                    ThermalBandBlock(
                        temperature_K=float(T),
                        kpoints=band_kpoints.copy(),
                        energies_eV=energies,
                        order_parameters=np.full(band_kpoints.shape[0], sol.m),
                        status=block_status,
                        magnetization=magnetization,
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
            self._set_mesh_shape(final_mesh)
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
