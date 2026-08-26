"""Validated collinear thermal spin model extracted from a Magnon object.

Implements the fail-closed input boundary of the thermal-magnon architecture:
only collinear ferromagnetic or equivalent bipartite antiferromagnetic
references with tensors of the form ``J_iso + lambda z z`` plus on-site
single-ion anisotropy are accepted. Unsupported DMI, transverse/off-diagonal
tensors, non-collinear references, and inequivalent spins are rejected with
actionable diagnostics — never silently projected away.

The extraction conventions follow the verified bridge in
``docs/sympy/03_anisotropy_multisite_conventions.py`` (Section 7):

- the stored per-directed-pair tensors are interpreted exactly as
  ``Magnon.Jq`` does (normalization by ``1/(S_i S_j)`` and Fourier phase
  ``exp(-2 pi i q R)``);
- with the reference storage ``JR = (1/2) S_a S_b [(J_iso) I + lambda zz]``
  per directed pair (and on-site SIA ``k1 = A S^2``) the extracted paper
  quantities are ``Jp_q = 2 sum_R J~xx e^{+2pi q R}`` and
  ``Lambda_q = lambda_q + 2 A delta = 2 sum_R (J~zz - J~xx) e^{+2pi q R}``,
  and the per-unit-magnetization RPA matrix reproduces ``Magnon.Hq``
  at T = 0.
"""

from __future__ import annotations

import numpy as np

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon.thermal_parameters import ThermalMagnonParameters

_COLLINEARITY_TOL = 1e-6
_EQUAL_SPIN_RTOL = 1e-6
_TENSOR_TOL = 1e-8
_STABILITY_TOL = 1e-9


class ThermalModelValidationError(ValueError):
    """Raised when input cannot be represented as a supported thermal model."""


def _is_half_integer(value: float) -> bool:
    twice = 2.0 * value
    return abs(twice - round(twice)) < 1e-9


class ThermalSpinModel:
    """Validated scalar thermal model: J_iso, lambda, SIA, spins, frames.

    Attributes
    ----------
    order_mode : str
        ``ferromagnetic`` or ``bipartite_afm`` (validated against moments).
    dimensionality : int
        Declared physical periodicity (1/2/3).
    S : ndarray (nspin,)
        Spin lengths (equal on all sites by validation).
    A : ndarray (nspin,)
        On-site single-ion anisotropy constants A_i (eV).
    magmoms : ndarray (nspin, 3)
        Collinear unit reference moments (plus/minus the common axis).
    spin_interpretation : str
        ``physical_quantum_spin`` or ``effective_quantum_spin``.
    """

    def __init__(
        self,
        magnon: Magnon,
        params: ThermalMagnonParameters,
    ):
        self._magnon = magnon
        self.nspin = magnon.nspin
        self.order_mode = params.thermal_order_mode
        self.dimensionality = params.thermal_dimensionality
        magmoms = np.asarray(magnon.magmom, dtype=float)
        norms = np.linalg.norm(magmoms, axis=1)
        if np.any(norms < 1e-12):
            raise ThermalModelValidationError(
                "all magnetic sites must carry a non-zero reference moment"
            )
        unit = magmoms / norms[:, None]

        # --- collinearity along one axis ------------------------------------
        axis = unit[0]
        projections = unit @ axis
        if not np.allclose(np.abs(projections), 1.0, atol=_COLLINEARITY_TOL):
            bad = int(np.argmax(np.abs(np.abs(projections) - 1.0)))
            raise ThermalModelValidationError(
                "the thermal solver requires a collinear reference state "
                f"(all moments parallel/antiparallel to one axis); site {bad} "
                f"deviates (cos angle = {projections[bad]:.6f}). Non-collinear, "
                "canted, and spiral finite-temperature thermodynamics are not "
                "supported."
            )
        self.magmoms = unit
        self._upmask = projections > 0
        self.axis = axis

        # --- order-mode consistency -----------------------------------------
        n_up = int(self._upmask.sum())
        n_down = int((~self._upmask).sum())
        if self.order_mode == "ferromagnetic":
            if n_down != 0:
                raise ThermalModelValidationError(
                    "thermal_order_mode='ferromagnetic' but "
                    f"{n_down} of {magnon.nspin} reference moments are "
                    "antiparallel to the first. Declare "
                    "thermal_order_mode='bipartite_afm' or provide a uniform "
                    "ferromagnetic reference."
                )
        else:  # bipartite_afm
            if n_up == 0 or n_down == 0:
                raise ThermalModelValidationError(
                    "thermal_order_mode='bipartite_afm' requires antiparallel "
                    "moments forming two sublattices; the reference is uniform. "
                    "Use thermal_order_mode='ferromagnetic' or supply an AFM "
                    "reference configuration."
                )

        # --- spins ------------------------------------------------------------
        if magnon.Snorm is None:
            raise ThermalModelValidationError(
                "Magnon object lacks Snorm; call set_reference() first"
            )
        if params.thermal_spin is not None:
            if len(params.thermal_spin) != magnon.nspin:
                raise ThermalModelValidationError(
                    f"thermal_spin has {len(params.thermal_spin)} entries but "
                    f"the cell has {magnon.nspin} magnetic sites"
                )
            spins = np.asarray(params.thermal_spin, dtype=float)
        else:
            spins = np.asarray(magnon.Snorm, dtype=float)
        if not np.allclose(spins, spins[0], rtol=_EQUAL_SPIN_RTOL, atol=1e-12):
            raise ThermalModelValidationError(
                "the thermal methods require equivalent spins: site spin "
                f"lengths range over [{spins.min():.6f}, {spins.max():.6f}]. "
                "Provide an explicit uniform thermal_spin override or use a "
                "model with equal |moments|."
            )
        self.S = np.full(magnon.nspin, spins[0], dtype=float)
        if _is_half_integer(spins[0]):
            self.spin_interpretation = "physical_quantum_spin"
        else:
            self.spin_interpretation = "effective_quantum_spin"

        # --- tensor validation and extraction (global collinear frame) ------
        self._validate_tensors()
        self._extract()

        # --- dimensionality vs exchange support ------------------------------
        self._validate_dimensionality()

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------

    def _normalized_JR(self) -> np.ndarray:
        inv = 1.0 / self.S[:, None, None, None] / self.S[None, :, None, None]
        return self._magnon.JR * inv

    def _validate_tensors(self):
        JRn = self._normalized_JR()
        scale = max(np.abs(JRn).max(), 1e-30)
        tol = _TENSOR_TOL * scale
        for iR, R in enumerate(np.asarray(self._magnon.Rlist, dtype=float)):
            on_site = bool(np.allclose(R, 0.0))
            for i in range(self._magnon.nspin):
                for j in range(self._magnon.nspin):
                    tensor = JRn[iR, i, j]
                    if on_site and i == j:
                        # only the zz (SIA) element may be non-zero
                        mask = np.ones((3, 3), dtype=bool)
                        mask[2, 2] = False
                        if np.abs(tensor[mask]).max() > tol:
                            raise ThermalModelValidationError(
                                "on-site tensors may only carry single-ion "
                                f"anisotropy in zz; site {i} has additional "
                                "on-site tensor components"
                            )
                        continue
                    antisym = 0.5 * (tensor - tensor.T)
                    if np.abs(antisym).max() > tol:
                        raise ThermalModelValidationError(
                            "DMI (antisymmetric exchange) is not supported by "
                            f"the thermal methods: pair ({i},{j}) at "
                            f"R={R.tolist()} has antisymmetric part up to "
                            f"{np.abs(antisym).max():.3e} eV. Disable DMI or "
                            "reformulate the model."
                        )
                    offdiag = tensor - np.diag(np.diag(tensor))
                    if np.abs(offdiag).max() > tol:
                        raise ThermalModelValidationError(
                            "off-diagonal (transverse) exchange tensors are not "
                            f"supported: pair ({i},{j}) at R={R.tolist()} has "
                            f"off-diagonal entries up to "
                            f"{np.abs(offdiag).max():.3e} eV."
                        )
                    if abs(tensor[0, 0] - tensor[1, 1]) > tol:
                        raise ThermalModelValidationError(
                            "transverse exchange anisotropy (Jxx != Jyy) is "
                            f"not supported: pair ({i},{j}) at R={R.tolist()} "
                            f"has Jxx={tensor[0,0]:.6f}, Jyy={tensor[1,1]:.6f} "
                            "eV. Only the longitudinal lambda = Jzz - J_perp "
                            "form is supported."
                        )

    def _validate_dimensionality(self):
        if self.dimensionality >= 3:
            return
        Rlist = np.asarray(self._magnon.Rlist, dtype=float)
        aperiodic = Rlist[:, self.dimensionality :]
        if np.abs(aperiodic).max() > 1e-8:
            raise ThermalModelValidationError(
                f"thermal_dimensionality={self.dimensionality} but exchange "
                "tensors extend along aperiodic cell directions (max "
                f"component {np.abs(aperiodic).max():.3f}). Declare "
                "dimensionality 3 or provide a model whose exchange support "
                "matches the declared periodicity."
            )

    def _extract(self):
        JRn = self._normalized_JR()
        Rlist = np.asarray(self._magnon.Rlist, dtype=float)
        self._JRn = JRn
        self._Rlist = Rlist
        # on-site SIA (normalized zz at R=0)
        self.A = np.zeros(self._magnon.nspin)
        for iR, R in enumerate(Rlist):
            if np.allclose(R, 0.0):
                self.A += JRn[iR][
                    np.arange(self._magnon.nspin), np.arange(self._magnon.nspin), 2, 2
                ]

    # ------------------------------------------------------------------
    # paper-quantity accessors
    # ------------------------------------------------------------------

    def _fourier(self, component, qpoints, sign=+1.0):
        """Fourier transform of an (nR, nspin, nspin) component, e^{+2pi i qR}."""
        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        phases = np.exp(sign * 2j * np.pi * (qpoints @ self._Rlist.T))  # (nk, nR)
        return np.tensordot(phases, component, axes=(1, 0))

    def Jp_q(self, qpoints) -> np.ndarray:
        """Paper isotropic exchange J_q (nkpt, nspin, nspin), Hermitian."""
        comp = self._JRn[..., 0, 0]
        return 2.0 * self._fourier(comp, qpoints)

    def Lambda_q(self, qpoints) -> np.ndarray:
        """lambda_q + 2 A delta (nkpt, nspin, nspin), Hermitian."""
        comp = self._JRn[..., 2, 2] - self._JRn[..., 0, 0]
        return 2.0 * self._fourier(comp, qpoints)

    def lambda_q(self, qpoints) -> np.ndarray:
        """Paper exchange anisotropy lambda_q (on-site SIA 2A excluded)."""
        lam = self.Lambda_q(qpoints).copy()
        idx = np.arange(self._magnon.nspin)
        lam[:, idx, idx] -= 2.0 * self.A
        return lam

    # ------------------------------------------------------------------
    # per-unit-magnetization dynamical matrices
    # ------------------------------------------------------------------

    def M_normal_q(self, qpoints) -> np.ndarray:
        """FM RPA matrix per unit magnetization: diag(sum(Jp0+Lam0)) - Jp_q.

        At T = 0 (m = S) this equals the magnon3 positive block ``A1 - C``
        elementwise for supported collinear ferromagnets (docs/sympy/03).
        """
        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        gamma = np.zeros((1, 3))
        Jp0 = self.Jp_q(gamma)[0]
        Lam0 = self.Lambda_q(gamma)[0]
        Jp = self.Jp_q(qpoints)
        diag = np.diag((Jp0 + Lam0).sum(axis=1))
        return diag[None, :, :] - Jp

    def check_stability(self, qpoints) -> float:
        """Minimum T=0 mode energy per unit magnetization over ``qpoints``.

        Returns the minimum eigenvalue of the T=0 dynamical matrix (eV per
        unit magnetization); negative values flag an unstable reference.
        """
        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        if self.order_mode == "ferromagnetic":
            matrices = self.M_normal_q(qpoints)
            values = np.linalg.eigvalsh(matrices)
            return float(values.min())
        # AFM: full bosonic BdG via magnon3 contractions at unit magnetization
        H = self.M_bdg_q(qpoints)
        values = _paraunitary_eigenvalues(H)
        return float(np.sort(values.real)[: H.shape[1] // 2].min())

    def M_bdg_q(self, qpoints) -> np.ndarray:
        """Bosonic BdG matrix per unit magnetization (collinear frames).

        Mirrors ``Magnon.Hq`` with all spin factors set to one, using the
        same rotation-frame construction, so ``H(T=0) = S * M_bdg`` for
        equivalent spins.
        """
        from TB2J.magnon.magnon_math import get_rotation_arrays

        qpoints = np.atleast_2d(np.asarray(qpoints, dtype=float))
        magmoms = self.magmoms.copy()
        U, V = get_rotation_arrays(magmoms, u=self.axis[None, :])

        def _jq(kpts):
            phases = np.exp(-2j * np.pi * (kpts @ self._Rlist.T))
            return np.tensordot(phases, self._JRn, axes=(1, 0))

        J0 = -_jq(np.zeros((1, 3)))[0]
        Jq = -_jq(qpoints)
        Jmq = Jq.swapaxes(-1, -2).swapaxes(1, 2)

        A1 = np.einsum("ix,qijxy,jy->qij", U, Jmq, U.conj())
        A2 = np.einsum("ix,qijxy,jy->qij", U.conj(), Jq.conj(), U)
        B = np.einsum("ix,qijxy,jy->qij", U, Jmq, U)
        C = np.diag(np.einsum("ix,ijxy,jy->i", V, 2 * J0, V))
        return np.block(
            [
                [A1 - C, B],
                [B.swapaxes(-1, -2).conj(), A2 - C],
            ]
        )


def _paraunitary_eigenvalues(H: np.ndarray) -> np.ndarray:
    """Positive-mode eigenvalues of bosonic BdG matrices via Cholesky/metric."""
    n = H.shape[-1] // 2
    I = np.eye(n)
    K = np.linalg.cholesky(H)
    g = np.block([[1 * I, 0 * I], [0 * I, -1 * I]])
    eig_matrix = K.swapaxes(-1, -2).conj() @ g @ K
    return np.linalg.eigvalsh(eig_matrix)[:, n:]


def build_thermal_spin_model(
    magnon: Magnon,
    params: ThermalMagnonParameters,
) -> ThermalSpinModel:
    """Validate and build the thermal spin model (fail-closed)."""
    return ThermalSpinModel(magnon, params)
