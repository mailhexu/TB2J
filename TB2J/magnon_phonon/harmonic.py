"""Harmonic mixed spin/lattice quadratic solver (Story 011, ADR-005).

The full bosonic quadratic mixed problem keeps the counter-rotating
terms and is diagonalized through the certified Colpa/paraunitary
primitive of the Story-006 magnon path (same convention as
``docs/sympy/site_boson_modes.md``: Cholesky factor ``M = K K^+``,
eigenproblem ``K^+ g K``, physical columns ``t = sqrt(|eps|) K^{-+} x``,
positive-norm columns first).  No regularization is applied: a stability
threshold crossing is flagged, and the flagged points carry NaN
energies and a zero shift.

All energies are in eV.  The magnon block uses the Story-006 ``Magnon.Hq``
Nambu convention; the phonon block is a Hermitian mass-weighted dynamical
matrix whose eigenvalues are phonon energies squared (eV^2) in the
displacement basis of the mixed Hessian.  The mixed block ``Gamma(k)`` is
the transverse-spin/displacement mixed Hessian in eV with Fourier
convention ``Gamma(k) = sum_R Gamma(R) exp(+2 pi i k . R)``.

The solver never derives a coupling from a scalar collinear derivative:
inputs that would fabricate mixing are rejected at construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "MixedSpinDisplacementHessian",
    "ControlledTwoBosonModel",
    "HybridModes",
    "HybridDiagnostics",
    "HarmonicMixedModel",
    "hybrid_modes",
]

#: producers whose observables cannot certify a transverse-spin/
#: displacement mixed derivative; their use would fabricate mixing
FORBIDDEN_PRODUCERS = frozenset(
    {
        "scalar_collinear_exchange_striction",
        "scalar_exchange_striction_maximum",
    }
)

FOURIER_CONVENTION = "gamma(k) = sum_R gamma(R) exp(+2 pi i k . R)"
UNITS = {"energy": "eV", "displacement": "mass-weighted sqrt(amu) Angstrom"}


@dataclass(frozen=True)
class MixedSpinDisplacementHessian:
    """Typed input for ``hybrid_modes``: bare sectors plus the mixed block.

    ``spin_coordinate_labels`` names one local complex transverse HP
    coordinate (its real/imaginary parts are the local transverse
    quadratures) per magnetic site.  ``displacement_labels`` names each
    Cartesian atom/axis coordinate.  Both arrays use the declared units
    and Fourier convention; ``provenance`` identifies how the mixed
    channels were observed.
    """

    n_mag_sites: int
    displacement_labels: Sequence[Sequence[int]]
    magnon_bare: Callable[[np.ndarray], np.ndarray]
    phonon_bare: Callable[[np.ndarray], np.ndarray]
    magnon_phonon_coupling: Callable[[np.ndarray], np.ndarray]
    producer: str
    provenance: Mapping
    spin_coordinate_labels: Sequence[Sequence] | None = None
    units: Mapping = field(default_factory=lambda: dict(UNITS))
    fourier_convention: str = FOURIER_CONVENTION

    def __post_init__(self):
        if self.producer in FORBIDDEN_PRODUCERS:
            raise ValueError(
                f"MixedSpinDisplacementHessian rejects producer '{self.producer}': "
                "a scalar collinear derivative does not observe the "
                "longitudinal-transverse mixed channels and would fabricate "
                "harmonic mixing."
            )
        if self.n_mag_sites < 1:
            raise ValueError("n_mag_sites must be positive")
        if not self.displacement_labels:
            raise ValueError("displacement_labels must name at least one coordinate")
        if self.spin_coordinate_labels is None:
            object.__setattr__(
                self,
                "spin_coordinate_labels",
                tuple((i, "local-transverse-complex") for i in range(self.n_mag_sites)),
            )
        if len(self.spin_coordinate_labels) != self.n_mag_sites:
            raise ValueError(
                "spin_coordinate_labels must name one local transverse "
                "coordinate per magnetic site"
            )
        if not self.provenance:
            raise ValueError("mixed-Hessian producer provenance is required")
        if self.units.get("energy") != "eV":
            raise ValueError("mixed-Hessian energy units must be eV")
        if self.fourier_convention != FOURIER_CONVENTION:
            raise ValueError(
                f"unsupported Fourier convention {self.fourier_convention!r}; "
                f"expected {FOURIER_CONVENTION!r}"
            )
        if not callable(self.magnon_bare) or not callable(self.phonon_bare):
            raise TypeError("bare magnon and phonon Hessian providers must be callable")
        if not callable(self.magnon_phonon_coupling):
            raise TypeError("magnon_phonon_coupling must be callable over points")


@dataclass(frozen=True)
class ControlledTwoBosonModel:
    """Controlled one-magnon/one-phonon model (ADR-005 benchmark).

    ``H = alpha b^+ b + beta a^+ a + g (b + b^+)(a + a^+)`` with
    counter-rotating terms; derived energies, stability condition
    ``4 g^2 < alpha beta`` and weights are asserted in
    ``docs/sympy/harmonic_magnon_phonon.py``.
    """

    alpha_eV: float
    beta_eV: float
    g_eV: float

    def to_mixed_hessian(self) -> MixedSpinDisplacementHessian:
        a, b, g = float(self.alpha_eV), float(self.beta_eV), float(self.g_eV)

        def magnon_bare(points):
            n = len(points)
            out = np.zeros((n, 2, 2), dtype=complex)
            out[:, 0, 0] = a
            out[:, 1, 1] = a
            return out

        def phonon_bare(points):
            return np.full((len(points), 1, 1), b**2, dtype=float)

        def coupling(points):
            return np.full((len(points), 1, 1), g, dtype=complex)

        return MixedSpinDisplacementHessian(
            n_mag_sites=1,
            displacement_labels=((0, 0),),
            magnon_bare=magnon_bare,
            phonon_bare=phonon_bare,
            magnon_phonon_coupling=coupling,
            producer="controlled_two_boson",
            provenance={
                "derivation": "docs/sympy/harmonic_magnon_phonon.py",
                "alpha_eV": a,
                "beta_eV": b,
                "g_eV": g,
                "stability_condition": "4 g^2 < alpha beta",
            },
        )


@dataclass(frozen=True)
class HybridDiagnostics:
    """Per-point solver diagnostics; ``shift`` is always zero (flagged,
    never regularized)."""

    min_eig: np.ndarray
    stability_tol: np.ndarray
    stability_flag: np.ndarray
    shift: np.ndarray
    pairing_residual: np.ndarray
    hermiticity_residual: np.ndarray


@dataclass(frozen=True)
class HybridModes:
    """Positive harmonic hybrid modes and observable weights.

    ``energies`` has NaN entries where the stability threshold is crossed;
    ``magnetic_weight``/``displacement_weight`` are the squared-amplitude
    shares of each positive mode on the transverse-spin and displacement
    coordinates (each pair sums to one).  ``paraunitary`` columns are
    ordered by ascending positive energy and satisfy
    ``T^+ g T = g``.
    """

    kpoints: np.ndarray
    energies: np.ndarray
    paraunitary: np.ndarray
    magnetic_weight: np.ndarray
    displacement_weight: np.ndarray
    is_stable: np.ndarray
    diagnostics: HybridDiagnostics
    mixed_hessian: MixedSpinDisplacementHessian


@dataclass(frozen=True)
class HarmonicMixedModel:
    """Small facade exposing the requested ``model.hybrid_modes`` seam."""

    mixed_hessian: MixedSpinDisplacementHessian

    def hybrid_modes(
        self,
        points: np.ndarray,
        mixed_hessian: MixedSpinDisplacementHessian | None = None,
    ) -> HybridModes:
        """Return harmonic hybrid modes using the supplied or stored Hessian."""
        return hybrid_modes(
            points, mixed_hessian if mixed_hessian is not None else self.mixed_hessian
        )


def _assemble_nambu(mixed: MixedSpinDisplacementHessian, kpoint) -> np.ndarray:
    """Assemble the full Hermitian Nambu Hessian at one point."""
    n_mag = mixed.n_mag_sites
    Hb = np.asarray(mixed.magnon_bare(kpoint[None, :])[0], dtype=complex)
    D = np.asarray(mixed.phonon_bare(kpoint[None, :])[0], dtype=complex)
    Gamma = np.asarray(mixed.magnon_phonon_coupling(kpoint[None, :])[0], dtype=complex)
    n_ph = D.shape[0]
    if Hb.shape != (2 * n_mag, 2 * n_mag):
        raise ValueError(
            f"magnon_bare returned {Hb.shape}, expected {(2 * n_mag, 2 * n_mag)}"
        )
    if D.shape != (len(mixed.displacement_labels), len(mixed.displacement_labels)):
        raise ValueError("phonon_bare matrix does not match displacement_labels")
    if not np.allclose(D, D.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("phonon_bare dynamical matrix must be Hermitian")
    if Gamma.shape != (n_mag, n_ph):
        raise ValueError(
            f"magnon_phonon_coupling returned {Gamma.shape}, expected {(n_mag, n_ph)}"
        )
    # phonon sector of the Nambu Hessian is the positive square root of D
    evals, evecs = np.linalg.eigh(D)
    if evals.min() < 0.0:
        raise ValueError(
            f"phonon_bare dynamical matrix has a negative eigenvalue {evals.min():.6g}; "
            "unstable bare phonons are not clipped or regularized"
        )
    omega = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    A_b, B_b = Hb[:n_mag, :n_mag], Hb[:n_mag, n_mag:]
    A = np.block([[A_b, Gamma], [Gamma.conj().T, omega]])
    B = np.block([[B_b, Gamma], [Gamma.T, np.zeros((n_ph, n_ph), dtype=complex)]])
    M = np.block([[A, B], [B.conj().T, A.conj()]])
    return M


def hybrid_modes(points, mixed_hessian: MixedSpinDisplacementHessian) -> HybridModes:
    """Diagonalize the harmonic mixed spin/lattice problem at ``points``.

    Full quadratic problem with counter-rotating terms through the
    certified central Colpa/paraunitary primitive.  Stability threshold
    crossings are flagged (NaN energies, zero shift), never regularized.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    mixed = mixed_hessian
    n_mag = mixed.n_mag_sites
    n_ph = len(mixed.displacement_labels)
    N = n_mag + n_ph

    energies = np.full((len(points), N), np.nan)
    paraunitary = np.full((len(points), 2 * N, 2 * N), np.nan + 0j)
    magnetic_weight = np.full((len(points), N), np.nan)
    displacement_weight = np.full((len(points), N), np.nan)
    is_stable = np.zeros(len(points), dtype=bool)
    min_eig = np.full(len(points), np.nan)
    stability_tol = np.full(len(points), np.nan)
    stability_flag = np.zeros(len(points), dtype=bool)
    shift = np.zeros(len(points))
    pairing_residual = np.full(len(points), np.nan)
    hermiticity_residual = np.full(len(points), np.nan)

    g_metric = np.diag(np.r_[np.ones(N), -np.ones(N)])
    for ik, kpoint in enumerate(points):
        M = _assemble_nambu(mixed, kpoint)
        hermiticity_residual[ik] = np.max(np.abs(M - M.conj().T))
        if hermiticity_residual[ik] > np.finfo(float).eps * 4 * N * max(
            1.0, np.max(np.abs(M))
        ):
            raise ValueError(
                f"Assembled Nambu Hessian at k={kpoint} is not Hermitian "
                f"(residual {hermiticity_residual[ik]:.3g})."
            )
        evals_M = np.linalg.eigvalsh(M)
        min_eig[ik] = evals_M[0]
        scale = max(1.0, np.max(np.abs(evals_M)))
        tol = np.finfo(float).eps * (2 * N) * scale
        stability_tol[ik] = tol
        if min_eig[ik] <= tol:
            # stability threshold crossed: flag, do not regularize
            stability_flag[ik] = True
            continue
        B_block = M[:N, N:]
        pairing_residual[ik] = np.max(np.abs(B_block - B_block.T))
        K = np.linalg.cholesky(M)
        KH = K.conj().T
        eigvals, eigvecs = np.linalg.eigh(KH @ g_metric @ K)
        if np.any(eigvals[:N] >= 0) or np.any(eigvals[N:] <= 0):
            raise ValueError(
                "Cholesky metric spectrum does not have stable positive/negative "
                f"branches at k={kpoint}."
            )
        physical = np.linalg.solve(KH, eigvecs) * np.sqrt(np.abs(eigvals))[None, :]
        paraunitary[ik] = np.concatenate(
            (physical[:, N:], physical[:, :N][:, ::-1]), axis=1
        )
        energies[ik] = eigvals[N:]
        pos = paraunitary[ik, :, :N]  # (2N coordinates, N modes)
        weight_sq = np.abs(pos) ** 2
        mag = weight_sq[:n_mag, :] + weight_sq[N : N + n_mag, :]
        ph = weight_sq[n_mag:N, :] + weight_sq[N + n_mag :, :]
        total = mag + ph
        magnetic_weight[ik] = mag / total
        displacement_weight[ik] = ph / total
        is_stable[ik] = True

    return HybridModes(
        kpoints=points,
        energies=energies,
        paraunitary=paraunitary,
        magnetic_weight=magnetic_weight,
        displacement_weight=displacement_weight,
        is_stable=is_stable,
        diagnostics=HybridDiagnostics(
            min_eig=min_eig,
            stability_tol=stability_tol,
            stability_flag=stability_flag,
            shift=shift,
            pairing_residual=pairing_residual,
            hermiticity_residual=hermiticity_residual,
        ),
        mixed_hessian=mixed,
    )
