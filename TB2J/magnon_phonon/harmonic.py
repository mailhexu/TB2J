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

from TB2J.magnon.eigenstates import colpa_diagonalize_factor

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
UNITS = {
    "energy": "eV",
    "spin_coordinate": "dimensionless local transverse spin components (hbar=1)",
    "displacement_coordinate": "mass_weighted | cartesian_angstrom",
}
DISPLACEMENT_COORDINATES = ("mass_weighted", "cartesian_angstrom")


@dataclass(frozen=True)
class MixedSpinDisplacementHessian:
    """Typed physical input for ``hybrid_modes``.

    The mixed input is the second derivative of the energy with respect to
    (i) the local transverse spin components of each magnetic site,
    expressed through ``s_i = S_i1 + i S_i2`` (dimensionless spin units,
    ``hbar = 1``), and (ii) the displacement coordinates named by
    ``displacement_labels``.  ``displacement_coordinate`` declares that
    basis: ``"mass_weighted"`` (``w = sqrt(m) u``) or
    ``"cartesian_angstrom"`` (then ``masses_amu`` is required and the
    solver converts ``w = sqrt(m) u`` exactly once).  The mixed derivative
    is in eV per (spin unit, displacement unit).

    The bare magnon provider returns the full Story-006 Nambu Hessian
    ``(2 n_mag, 2 n_mag)`` per point in eV, including its own
    opposite-k lower blocks (nonreciprocal systems are preserved, never
    reconstructed from the upper block).  The bare phonon provider returns
    the Hermitian mass-weighted dynamical matrix in the same displacement
    basis with eigenvalues equal to phonon energies squared in eV^2.

    ``docs/sympy/harmonic_magnon_phonon.py`` derives the solver's
    conversion to the bosonic coupling
    ``Gamma = G sqrt(S_i/2) e / sqrt(2 omega)``: the spin-length and
    phonon zero-point factors are applied exactly once, here and nowhere
    else.
    """

    n_mag_sites: int
    displacement_labels: Sequence[Sequence[int]]
    magnon_bare: Callable[[np.ndarray], np.ndarray]
    phonon_bare: Callable[[np.ndarray], np.ndarray]
    mixed_derivative: Callable[[np.ndarray], np.ndarray]
    spin_lengths: Sequence[float]
    producer: str
    provenance: Mapping
    displacement_coordinate: str = "mass_weighted"
    masses_amu: Sequence[float] | None = None
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
        if len(tuple(self.spin_lengths)) != self.n_mag_sites:
            raise ValueError("spin_lengths must give one spin length per magnetic site")
        if self.displacement_coordinate not in DISPLACEMENT_COORDINATES:
            raise ValueError(
                f"displacement_coordinate must be one of {DISPLACEMENT_COORDINATES}"
            )
        if self.displacement_coordinate == "cartesian_angstrom":
            if self.masses_amu is None or len(tuple(self.masses_amu)) != len(
                tuple(self.displacement_labels)
            ):
                raise ValueError(
                    "cartesian_angstrom displacement basis requires masses_amu "
                    "for every displacement coordinate"
                )
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
        if not callable(self.mixed_derivative):
            raise TypeError("mixed_derivative must be callable over points")

    @classmethod
    def scalar_collinear_forbidden_control(
        cls,
        *,
        n_mag_sites: int,
        displacement_labels: Sequence[Sequence[int]],
        magnon_bare: Callable[[np.ndarray], np.ndarray],
        phonon_bare: Callable[[np.ndarray], np.ndarray],
        spin_lengths: Sequence[float],
        provenance: Mapping,
        displacement_coordinate: str = "mass_weighted",
        masses_amu: Sequence[float] | None = None,
        spin_coordinate_labels: Sequence[Sequence] | None = None,
    ) -> MixedSpinDisplacementHessian:
        """Build the symmetry-forbidden scalar-collinear zero-mixing control.

        No scalar exchange derivative is promoted to a mixed Hessian; the
        mixed derivative is identically zero while the independently
        supplied bare sectors are preserved.
        """
        return cls(
            n_mag_sites=n_mag_sites,
            displacement_labels=displacement_labels,
            magnon_bare=magnon_bare,
            phonon_bare=phonon_bare,
            mixed_derivative=lambda points: np.zeros(
                (len(points), n_mag_sites, len(displacement_labels)), dtype=complex
            ),
            spin_lengths=spin_lengths,
            producer="scalar_collinear_forbidden_control",
            provenance={
                **dict(provenance),
                "control": "scalar-collinear input; symmetry-forbidden mixed Hessian",
                "mixed_derivative_inferred": False,
            },
            displacement_coordinate=displacement_coordinate,
            masses_amu=masses_amu,
            spin_coordinate_labels=spin_coordinate_labels,
        )


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
        """Express the model through its physical mixed derivative.

        With spin length ``S = 2`` (so ``sqrt(S/2) = 1``) and the single
        mass-weighted normal coordinate (``e = 1``, ``omega = beta``), the
        derived conversion ``Gamma = G sqrt(S/2)/sqrt(2 omega)`` maps the
        physical Hessian ``G = g sqrt(2 beta)`` back to the bosonic
        coupling ``g`` exactly once.
        """
        a, b, g = float(self.alpha_eV), float(self.beta_eV), float(self.g_eV)

        def magnon_bare(points):
            n = len(points)
            out = np.zeros((n, 2, 2), dtype=complex)
            out[:, 0, 0] = a
            out[:, 1, 1] = a
            return out

        def phonon_bare(points):
            return np.full((len(points), 1, 1), b**2, dtype=float)

        def mixed_derivative(points):
            return np.full((len(points), 1, 1), g * np.sqrt(2.0 * b))

        return MixedSpinDisplacementHessian(
            n_mag_sites=1,
            displacement_labels=((0, 0),),
            magnon_bare=magnon_bare,
            phonon_bare=phonon_bare,
            mixed_derivative=mixed_derivative,
            spin_lengths=(2.0,),
            producer="controlled_two_boson",
            provenance={
                "derivation": "docs/sympy/harmonic_magnon_phonon.py",
                "alpha_eV": a,
                "beta_eV": b,
                "g_eV": g,
                "spin_length": 2.0,
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

    ``energies`` has NaN entries where the stability threshold is crossed.
    ``magnetic_strength`` is the operator-defined one-quantum transverse
    spin strength ``sum_i |<0|S_i1 + i S_i2|mode>|^2`` (spin-length
    squared units); ``displacement_strength`` is the physical displacement
    strength ``sum_beta |<0|w_beta|mode>|^2`` in the declared displacement
    basis (mass-weighted or Angstrom, including the phonon zero-point
    factor ``1/sqrt(2 omega)``).  ``magnetic_weight``/``displacement_weight``
    remain the normalized particle-amplitude species shares (each pair
    sums to one).  ``paraunitary`` columns are ordered by ascending
    positive energy and satisfy ``T^+ g T = g``.
    """

    kpoints: np.ndarray
    energies: np.ndarray
    paraunitary: np.ndarray
    magnetic_weight: np.ndarray
    displacement_weight: np.ndarray
    magnetic_strength: np.ndarray
    displacement_strength: np.ndarray
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


def _bosonic_coupling(mixed, kpoint, phonon_energies, phonon_modes):
    """Convert the physical mixed derivative to the bosonic energy coupling.

    Applies the derived conversion exactly once:
    ``Gamma = G sqrt(S_i/2) e / sqrt(2 omega)`` with the optional
    Cartesian-to-mass-weighted conversion ``G -> G / sqrt(m)`` first.
    """
    n_ph = len(mixed.displacement_labels)
    G = np.asarray(mixed.mixed_derivative(kpoint[None, :])[0], dtype=complex)
    if G.shape != (mixed.n_mag_sites, n_ph):
        raise ValueError(
            f"mixed_derivative returned {G.shape}, expected "
            f"{(mixed.n_mag_sites, n_ph)}"
        )
    if mixed.displacement_coordinate == "cartesian_angstrom":
        G = G / np.sqrt(np.asarray(mixed.masses_amu, dtype=float))[None, :]
    Gamma_raw = G @ phonon_modes  # (n_mag, n_ph modes)
    spin = np.sqrt(np.asarray(mixed.spin_lengths, dtype=float) / 2.0)
    return Gamma_raw * spin[:, None] / np.sqrt(2.0 * phonon_energies)[None, :]


def _assemble_nambu(mixed: MixedSpinDisplacementHessian, kpoint):
    """Assemble the full Hermitian Nambu Hessian at one point.

    The supplied magnon block is embedded exactly as given, including its
    own opposite-k lower blocks (never reconstructed from the upper
    block).  The mixed rectangles use the converted ``Gamma(k)`` for both
    the normal and counter-rotating pairing; both pair k with -k through
    the stated Fourier convention.  Returns the Hessian plus the phonon
    energies and mass-weighted modes needed for the observable strengths.
    """
    n_mag = mixed.n_mag_sites
    Hb = np.asarray(mixed.magnon_bare(kpoint[None, :])[0], dtype=complex)
    D = np.asarray(mixed.phonon_bare(kpoint[None, :])[0], dtype=complex)
    n_ph = D.shape[0]
    if Hb.shape != (2 * n_mag, 2 * n_mag):
        raise ValueError(
            f"magnon_bare returned {Hb.shape}, expected {(2 * n_mag, 2 * n_mag)}"
        )
    if D.shape != (len(mixed.displacement_labels), len(mixed.displacement_labels)):
        raise ValueError("phonon_bare matrix does not match displacement_labels")
    if not np.allclose(D, D.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("phonon_bare dynamical matrix must be Hermitian")
    # phonon sector: eigendecomposition gives energies and mass-weighted modes
    evals, evecs = np.linalg.eigh(D)
    if evals.min() < 0.0:
        raise ValueError(
            f"phonon_bare dynamical matrix has a negative eigenvalue {evals.min():.6g}; "
            "unstable bare phonons are not clipped or regularized"
        )
    omega = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
    phonon_energies = np.sqrt(evals)
    Gamma = _bosonic_coupling(mixed, kpoint, phonon_energies, evecs)

    # grouped Nambu layout [b(n), a(m), b^+(n), a^+(m)]; the magnon blocks
    # are embedded exactly as supplied (Hq), including nonreciprocal ones.
    # Mixed placement mirrors the Colpa bookkeeping of the magnon K block:
    # normal pairs (b,a)=G,(a,b)=G^+; counter-rotating pairs
    # (b,a^+)=G,(a,b^+)=G^T with conjugate partners (a^+,b)=G*,
    # (b^+,a)=G*.
    dim = 2 * n_mag + 2 * n_ph
    M = np.zeros((dim, dim), dtype=complex)
    # A block: normal part
    M[:n_mag, :n_mag] = Hb[:n_mag, :n_mag]
    M[:n_mag, n_mag : n_mag + n_ph] = Gamma
    M[n_mag : n_mag + n_ph, :n_mag] = Gamma.conj().T
    M[n_mag : n_mag + n_ph, n_mag : n_mag + n_ph] = omega
    # B block: anomalous part, symmetric like the magnon K block
    M[:n_mag, n_mag + n_ph : 2 * n_mag + n_ph] = Hb[:n_mag, n_mag:]
    M[:n_mag, 2 * n_mag + n_ph :] = Gamma
    M[n_mag : n_mag + n_ph, n_mag + n_ph : 2 * n_mag + n_ph] = Gamma.T
    # B^+ block
    M[n_mag + n_ph : 2 * n_mag + n_ph, :n_mag] = Hb[n_mag:, :n_mag]
    M[n_mag + n_ph : 2 * n_mag + n_ph, n_mag : n_mag + n_ph] = Gamma.conj()
    M[2 * n_mag + n_ph :, :n_mag] = Gamma.conj()
    # bottom-right block: Nambu conjugate of A with the supplied magnon
    # lower block embedded (nonreciprocal-safe) instead of Hb00*
    M[n_mag + n_ph : 2 * n_mag + n_ph, n_mag + n_ph : n_mag + n_ph + n_mag] = Hb[
        n_mag:, n_mag:
    ]
    M[n_mag + n_ph : 2 * n_mag + n_ph, n_mag + n_ph + n_mag :] = Gamma.conj()
    M[2 * n_mag + n_ph :, n_mag + n_ph : n_mag + n_ph + n_mag] = Gamma.T.conj()
    M[2 * n_mag + n_ph :, n_mag + n_ph + n_mag :] = omega.conj()
    return M, phonon_energies, evecs


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
    magnetic_strength = np.full((len(points), N), np.nan)
    displacement_strength = np.full((len(points), N), np.nan)
    is_stable = np.zeros(len(points), dtype=bool)
    min_eig = np.full(len(points), np.nan)
    stability_tol = np.full(len(points), np.nan)
    stability_flag = np.zeros(len(points), dtype=bool)
    shift = np.zeros(len(points))
    pairing_residual = np.full(len(points), np.nan)
    hermiticity_residual = np.full(len(points), np.nan)

    g_metric = np.diag(np.r_[np.ones(N), -np.ones(N)])
    spin_lengths = np.asarray(mixed.spin_lengths, dtype=float)
    for ik, kpoint in enumerate(points):
        M, phonon_energies, phonon_modes = _assemble_nambu(mixed, kpoint)
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
        energies[ik], paraunitary[ik] = colpa_diagonalize_factor(K, metric=g_metric)
        pos = paraunitary[ik, :, :N]  # (2N coordinates, N modes)
        weight_sq = np.abs(pos) ** 2
        mag = weight_sq[:n_mag, :] + weight_sq[N : N + n_mag, :]
        ph = weight_sq[n_mag:N, :] + weight_sq[N + n_mag :, :]
        total = mag + ph
        magnetic_weight[ik] = mag / total
        displacement_weight[ik] = ph / total
        # Operator-defined physical strengths (derived in
        # docs/sympy/harmonic_magnon_phonon.py):
        # <0|S_i1 + i S_i2|mode> = sqrt(2 S_i) T_{b_i, mode}
        magnetic_strength[ik] = 2.0 * np.sum(
            spin_lengths[:, None] * weight_sq[:n_mag, :], axis=0
        )
        # <0|w_beta|mode> = sum_nu e_{beta nu} T_{a_nu, mode}/sqrt(2 omega_nu)
        displacement_amp = (phonon_modes @ pos[n_mag:N, :]) / np.sqrt(
            2.0 * phonon_energies
        )[:, None]
        if mixed.displacement_coordinate == "cartesian_angstrom":
            displacement_amp = (
                displacement_amp
                / np.sqrt(np.asarray(mixed.masses_amu, dtype=float))[:, None]
            )
        displacement_strength[ik] = np.sum(np.abs(displacement_amp) ** 2, axis=0)
        is_stable[ik] = True

    return HybridModes(
        kpoints=points,
        energies=energies,
        paraunitary=paraunitary,
        magnetic_weight=magnetic_weight,
        displacement_weight=displacement_weight,
        magnetic_strength=magnetic_strength,
        displacement_strength=displacement_strength,
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
