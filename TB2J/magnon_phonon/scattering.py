"""Full-frequency one-shot Born scattering spectra (Story 010, ADR-004).

Retarded second-order self-energies and spectral functions of BOTH species
from the certified Story-009 vertex, with bare internal propagators, one
declared temperature, and blocked (k, q) contractions (only requested
blocks are materialized).  The kernels are the ones asserted in
``docs/sympy/born_scattering_spectra.py``; that script is the mathematical
authority for every coefficient below.

    Sigma^R_{ab}(k, w) = sum_q w_q sum_{m nu}
        g[m, a, nu] g*[m, b, nu] [ (n_B+1)/(w - eps_{k+q,m} - Om_qnu + i eta)
                                 +  n_B  /(w - eps_{k+q,m} + Om_qnu + i eta) ]

    Pi^R_{nu nu'}(q, W) = sum_k w_k sum_{m n}
        g[m, n, nu'] g*[m, n, nu] (n_{k,n} - n_{k+q,m})
                                 / (W + eps_{k,n} - eps_{k+q,m} + i eta)

The static tadpole/magnetostriction contraction of the vertex is reference
bookkeeping, frequency-independent, and is excluded here (asserted in the
derivation script; see ``metadata["static_tadpole_excluded"]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from TB2J.magnon_phonon.vertex import MagnonPhononModel

#: Boltzmann constant in eV/K
KB_EV_K = 8.617333262e-5
#: Planck constant in eV s (THz phonon frequency -> eV)
H_EV_S = 4.135667696e-15

_APPROXIMATION = "one-shot O(g^2) Born (bare internal propagators, not SCBA)"


def bose_occupation(energy_eV, temperature_K):
    """Bose occupation n_B(E) = 1/(exp(E/kT) - 1); exactly zero at T = 0."""
    # MathDoc (docs/sympy/born_scattering_spectra, check_bose_weights):
    # emission (n+1) / absorption n weights from the canonical thermal
    # geometric series, with the asserted T -> 0 emission-only limit.
    energy = np.asarray(energy_eV, dtype=float)
    temperature = float(temperature_K)
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if np.any(energy < 0.0):
        raise ValueError("Bose occupations require non-negative energies")
    if temperature == 0.0:
        return np.zeros_like(energy)
    return 1.0 / np.expm1(energy / (KB_EV_K * temperature))


def thz_to_ev(frequencies_thz):
    """Phonon frequency THz -> energy eV (E = h nu)."""
    # MathDoc (docs/sympy/born_scattering_spectra, "Units" section): E = h nu.
    return np.asarray(frequencies_thz, dtype=float) * 1e12 * H_EV_S


def magnon_self_energy(g, eps_kq, omega_q, n_bose_q, weights, omega, eta):
    """One-loop retarded magnon self-energy (asserted Born kernel).

    Parameters
    ----------
    g : (nq, m, n, nu) complex
        Vertex blocks ``g[m at k+q, n at k, nu]`` per internal momentum.
    eps_kq : (nq, m) float
        Bare magnon energies at ``k+q`` (eV).
    omega_q, n_bose_q : (nq, nu) float
        Internal phonon energies (eV) and their Bose occupations.
    weights : (nq,) float
        Internal-mesh weights (sum to one).
    omega : (nw,) float
        Frequency grid (eV).
    eta : float
        Artificial retarded broadening (eV), reported separately.

    Returns
    -------
    (nw, n, n) complex
        ``Sigma^R`` on the branch space of ``k``.
    """
    g = np.asarray(g, dtype=complex)
    eps_kq = np.asarray(eps_kq, dtype=float)
    omega_q = np.asarray(omega_q, dtype=float)
    n_bose_q = np.asarray(n_bose_q, dtype=float)
    weights = np.asarray(weights, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if g.ndim != 4:
        raise ValueError("g must have shape (nq, m, n, nu)")
    if eps_kq.shape[0] != g.shape[0] or omega_q.shape != g.shape[0:1] + g.shape[3:]:
        raise ValueError("internal-mesh array shapes are inconsistent")
    if weights.shape != (g.shape[0],) or not np.all(weights >= 0.0):
        raise ValueError("weights must match the internal mesh and be >= 0")
    if eta <= 0.0:
        raise ValueError("broadening eta must be positive")

    denom_em = (
        omega[None, None, None, :]
        - eps_kq[:, :, None, None]
        - omega_q[:, None, :, None]
        + 1j * eta
    )  # (nq, m, nu, nw)
    denom_ab = (
        omega[None, None, None, :]
        - eps_kq[:, :, None, None]
        + omega_q[:, None, :, None]
        + 1j * eta
    )
    kernel = (n_bose_q + 1.0)[:, None, :, None] / denom_em + n_bose_q[
        :, None, :, None
    ] / denom_ab  # (nq, m, nu, nw)
    weighted = (
        g[:, :, :, :, None]
        * weights[:, None, None, None, None]
        * kernel[:, :, None, :, :]
    )  # (nq, m, n, nu, nw)
    # MathDoc (docs/sympy/born_scattering_spectra, check_magnon_kernel_t0):
    # Sigma^R_{ab}(w) = sum_{q,m,nu} w_q g[m,a,nu] g*[m,b,nu]
    #   [(n+1)/(w - eps_{k+q,m} - Om_nu + i eta) + n/(w - eps_{k+q,m} + Om_nu + i eta)]
    # the phonon branch index nu is contracted between BOTH g operands
    # (label u below) and the external frequency stays a distinct free
    # output axis (label w); the high-frequency limit and threshold gating
    # are asserted in check_high_frequency_limits and
    # check_zero_coupling_and_thresholds.
    sigma = np.einsum("qmauw,qmbu->wab", weighted, np.conj(g), optimize=True)
    return sigma


def phonon_hot_band_kernel(
    g, eps_k, eps_kq, n_bose_k, n_bose_kq, omega_ph_ev, weights, omega, eta
):
    """Certified one-shot O(g^2) phonon coordinate-channel kernel.

    For the declared first-order Hamiltonian the phonon coordinate Green
    function correction is NOT a static Lindhard self-energy.  The certified
    closed form (derivation script, ``check_phonon_hot_band_prefactor``:
    single-dressing Lehmann intermediate-state sum over the Bose ladders,
    asserted against the truncated-Fock reference at T = 0 and finite T) is,
    per propagating branch ``nu`` and channel ``(k, m, n)`` with
    ``deps = eps_{k+q,m} - eps_{k,n}``,

        delta D_nu(omega) = sum_k w_k sum_{mn} |g[m,n,nu]|^2 *
            [ S_nu / (omega - Omega_nu + i eta)^2
            + S_nu / (omega + Omega_nu + i eta)^2
            + W_nu / (omega - deps + i eta) ],
        S_nu = (n_{k+q,m} - n_{k,n}) deps / (Omega_nu (deps^2 - Omega_nu^2)),
        W_nu = 2 Omega_nu (n_{k,n} - n_{k+q,m}) / (deps^2 - Omega_nu^2)^2.

    One hot-band pole is carried per internal channel; on the full momentum
    meshes used by the seam the reversed channel ``k' = k + q`` supplies the
    ``-deps`` partner pole, reproducing the exactly KMS-antisymmetric
    hot-band pair of the derivation (which writes both terms for a mesh
    containing a single hop direction).

    The ``S`` pieces are the O(g^2) second-order restructuring of the bare
    ``+- Omega_nu`` poles (the ladder-shifted phonon line); the ``W`` pieces
    are the hot-band poles at ``omega = +- deps`` created by the
    magnon-density dressing of the coordinate vertex.  Both weights depend
    only on the magnon Bose occupations: at T = 0 (``n_{k,n} = n_{k+q,m} =
    0``) the correction vanishes exactly.  The full-mesh hot-band pair is
    exactly KMS antisymmetric and the double-pole pair symmetric, so the
    coordinate spectral function stays exactly odd.  The kernel is diagonal
    in the phonon branch index (each correction attaches to a single
    propagating line, one-shot with bare internal propagators); the returned
    Green function correction carries the full ``(nw, nu, nu)`` matrix form.

    Returns
    -------
    (delta_d, sigma) : ((nw, nu) complex, (nw, nu) complex)
        ``delta_d`` is the additive coordinate Green-function correction
        (NOT a self-energy); ``sigma`` is the Dyson-partitioned phonon
        self-energy ``Sigma_nu(w) = D0_nu^-1 deltaD_nu(w) D0_nu^-1`` whose
        on-shell imaginary part defines the calculated linewidth.

    The one-shot expansion is undefined on resonance (``deps^2 = Omega_nu^2``).
    """
    g = np.asarray(g, dtype=complex)
    eps_k = np.asarray(eps_k, dtype=float)
    eps_kq = np.asarray(eps_kq, dtype=float)
    n_bose_k = np.asarray(n_bose_k, dtype=float)
    n_bose_kq = np.asarray(n_bose_kq, dtype=float)
    omega_ph_ev = np.asarray(omega_ph_ev, dtype=float)
    weights = np.asarray(weights, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if g.ndim != 4:
        raise ValueError("g must have shape (nk, m, n, nu)")
    nk = g.shape[0]
    if eps_k.shape != (nk, g.shape[2]) or eps_kq.shape != (nk, g.shape[1]):
        raise ValueError("internal-mesh energy shapes are inconsistent")
    if weights.shape != (nk,) or not np.all(weights >= 0.0):
        raise ValueError("weights must match the internal mesh and be >= 0")
    if eta <= 0.0:
        raise ValueError("broadening eta must be positive")
    if omega_ph_ev.shape != (g.shape[3],):
        raise ValueError("omega_ph_ev must match the phonon branch count")

    om = omega_ph_ev[None, None, None, :]  # (1, 1, 1, nu)
    deps = eps_kq[:, :, None, None] - eps_k[:, None, :, None]  # (nk, m, n, 1)
    if np.any(np.abs(deps**2 - om**2) == 0.0):
        raise ValueError(
            "one-shot Born hot-band kernel is undefined on resonance "
            "(deps^2 == Omega_nu^2); detune the controlled model"
        )
    dn = n_bose_kq[:, :, None, None] - n_bose_k[:, None, :, None]  # (nk, m, n, 1)
    s_ker = dn * deps / (om * (deps**2 - om**2))  # (nk, m, n, nu)
    w_ker = 2.0 * om * (-dn) / (deps**2 - om**2) ** 2  # (nk, m, n, nu)
    g2 = (g * np.conj(g))[:, :, :, :, None]  # (nk, m, n, nu, 1)
    nw = omega.size
    omega_c = omega.reshape(1, 1, 1, 1, nw)
    deps5 = deps[:, :, :, None, :]  # (nk, m, n, 1, nu)
    om5 = omega_ph_ev[None, None, :, None]  # (1, 1, nu, 1)
    # bare +-Omega_nu double poles (branch-diagonal), shape (1,1,nu,nw)
    d_pole = (
        1.0 / (omega_c - om5 + 1j * eta) ** 2 + 1.0 / (omega_c + om5 + 1j * eta) ** 2
    )  # (1, 1, nu, nw)
    # one hot-band pole per channel at omega = deps_k: the reversed
    # channel k' = k + q of a full momentum mesh carries deps_k' = -deps_k
    # and supplies the -deps_k pole, so no second term is added here
    h_pole = 1.0 / (omega_c - deps5 + 1j * eta)  # (nk, m, n, 1, nw)
    weighted = (
        g2
        * weights[:, None, None, None, None]
        * (s_ker[:, :, :, :, None] * d_pole + w_ker[:, :, :, :, None] * h_pole)
    )  # (nk, m, n, nu, nw)
    # MathDoc (docs/sympy/born_scattering_spectra,
    # check_phonon_hot_band_prefactor): the additive coordinate correction
    # delta D_nu(w) = sum_k w_k sum_{mn} |g[m,n,nu]|^2 [S/(w -+ Om)^2 +
    # W/(w -+ deps)] asserted against the independently constructed
    # truncated-Fock Lehmann pole sum (test oracle keeps its own Fock-space
    # interaction and energy shifts).  MathDoc (check_phonon_kernel_bubble):
    # the bubble chi enters ONLY through the hot-band shifted arguments;
    # delta D is an additive Green-function correction, NOT a self-energy.
    delta_d = np.einsum("kmnuw->wu", weighted, optimize=True)
    # Dyson-partitioned phonon self-energy of the additive correction:
    # D = D0 + D0 Sigma D0  =>  Sigma_nu(w) = D0_nu^-1 deltaD_nu(w) D0_nu^-1
    # (branch-diagonal; check_phonon_hot_band: the on-shell linewidth is
    # eta-suppressed because the O(g^2) weight sits in the hot bands).
    d0_inv = 1.0 / ((omega[:, None] + 1j * eta) ** 2 - omega_ph_ev[None, :] ** 2)
    sigma = d0_inv**2 * delta_d
    return delta_d, sigma


@dataclass(frozen=True)
class ScatteringRequest:
    """Typed full-frequency scattering request (ADR-006).

    External points are the reported momenta; internal meshes carry the
    blocked momentum contractions with explicit weights (summing to one).
    ``phonon_modes``/``magnon_phonon_modes`` are :class:`PhononModeInput`
    (or equivalent duck-typed) providers aligned with the external phonon
    momenta and the internal magnon-loop momenta respectively.  ``species``
    selects ``("magnon",)``, ``("phonon",)`` or both (default).
    """

    omega_eV: np.ndarray
    temperature_K: float
    broadening_eV: float
    magnon_kpoints: np.ndarray
    phonon_qpoints: np.ndarray
    phonon_modes: tuple
    magnon_mesh_qpoints: np.ndarray
    magnon_mesh_weights: np.ndarray
    magnon_phonon_modes: tuple
    phonon_mesh_kpoints: np.ndarray
    phonon_mesh_weights: np.ndarray
    species: tuple = ("magnon", "phonon")

    def __post_init__(self):
        omega = np.asarray(self.omega_eV, dtype=float)
        if omega.ndim != 1 or omega.size < 2 or not np.all(np.diff(omega) > 0):
            raise ValueError("omega_eV must be a strictly increasing grid")
        object.__setattr__(self, "omega_eV", omega)
        if float(self.broadening_eV) <= 0.0:
            raise ValueError("broadening_eV must be positive")
        if float(self.temperature_K) < 0.0:
            raise ValueError("temperature_K must be non-negative")
        for name in (
            "magnon_kpoints",
            "phonon_qpoints",
            "magnon_mesh_qpoints",
            "phonon_mesh_kpoints",
        ):
            pts = np.asarray(getattr(self, name), dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"{name} must have shape (n, 3)")
            object.__setattr__(self, name, pts)
        for name, npts in (
            ("magnon_mesh_weights", self.magnon_mesh_qpoints.shape[0]),
            ("phonon_mesh_weights", self.phonon_mesh_kpoints.shape[0]),
        ):
            w = np.asarray(getattr(self, name), dtype=float)
            if (
                w.shape != (npts,)
                or np.any(w < 0.0)
                or not np.isclose(w.sum(), 1.0, atol=1e-12)
            ):
                raise ValueError(
                    f"{name} must be non-negative and sum to one over the mesh"
                )
            object.__setattr__(self, name, w)
        if len(self.phonon_modes) != self.phonon_qpoints.shape[0]:
            raise ValueError("phonon_modes must align with the external phonon qpoints")
        if len(self.magnon_phonon_modes) != self.magnon_mesh_qpoints.shape[0]:
            raise ValueError("magnon_phonon_modes must align with the internal q mesh")
        known = {"magnon", "phonon"}
        species = self.species if self.species is not None else ("magnon", "phonon")
        if not set(species) <= known or not species:
            raise ValueError(f"species must be a nonempty subset of {known}")
        object.__setattr__(self, "species", tuple(species))


@dataclass(frozen=True)
class SpeciesSpectrum:
    """One species' bare references, self-energy, spectral function and
    separately reported calculated linewidths."""

    species: str
    momenta: np.ndarray  # (n_ext, 3)
    bare_energies: np.ndarray  # (n_ext, nmodes) eV
    sigma: np.ndarray  # (n_ext, nw, nmodes, nmodes) complex, eV
    green: np.ndarray  # (n_ext, nw, nmodes, nmodes) complex
    spectral: np.ndarray  # A = -2 Im G^R, same shape
    linewidths: np.ndarray  # (n_ext, nmodes) eV, -Im Sigma at the bare pole
    #: (n_ext, nmodes) eV, nearest-grid sampling offset of the on-shell
    #: linewidth evaluation (grid-error estimate, not a linewidth)
    grid_error: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


@dataclass(frozen=True)
class ScatteringResult:
    """Both species' spectra with temperature, broadening and grid metadata
    reported separately from the calculated linewidths."""

    magnon: SpeciesSpectrum | None
    phonon: SpeciesSpectrum | None
    omega_grid_eV: np.ndarray
    temperature_K: float
    broadening_eV: float
    metadata: dict = field(default_factory=dict)


def _oscillator_bare(bare, omega, eta):
    """Bare coordinate-channel phonon Green function
    ``[(omega + i eta)^2 I - Omega^2]^-1``."""
    n_ext, n = bare.shape
    eye = np.eye(n, dtype=complex)
    out = np.empty((n_ext, omega.size, n, n), dtype=complex)
    for i in range(n_ext):
        for w in range(omega.size):
            out[i, w] = np.linalg.inv(
                (omega[w] + 1j * eta) ** 2 * eye - np.diag(bare[i] ** 2)
            )
    return out


def _grid_sampling_error(bare, omega):
    """Nearest-grid sampling offset ``|bare pole - nearest omega grid
    point|`` (eV) of every on-shell linewidth evaluation.  This is the
    reported grid-error estimate; it is NOT a linewidth.  MathDoc: the
    internal momentum meshes are exactly weighted, so this sampling offset
    is the only grid error of the on-shell evaluation."""
    n_ext, n = bare.shape
    out = np.zeros((n_ext, n))
    for i in range(n_ext):
        for a in range(n):
            i_w = int(np.argmin(np.abs(omega - bare[i, a])))
            out[i, a] = abs(bare[i, a] - omega[i_w])
    return out


def _oscillator_linewidths(sigma, bare, omega):
    """Calculated phonon linewidths ``-Im Sigma_aa(Omega)`` at the nearest
    grid point to the bare pole, from the Dyson self-energy (not from the
    additive correction), reported separately from the artificial
    broadening.  For the hot-band kernel the on-shell linewidth is
    eta-suppressed (the O(g^2) weight sits in the hot bands, asserted in
    the derivation script, check_phonon_hot_band)."""
    n_ext, n = bare.shape
    out = np.zeros((n_ext, n))
    for i in range(n_ext):
        for a in range(n):
            i_w = int(np.argmin(np.abs(omega - bare[i, a])))
            out[i, a] = -sigma[i, i_w, a, a].imag
    return out


def _particle_spectral(sigma, bare, omega, eta):
    """G^R = [(omega + i eta) I - E - Sigma^R]^{-1}, A = -2 Im G."""
    # MathDoc (docs/sympy/born_scattering_spectra,
    # check_observable_normalization_and_moments): particle-channel
    # commutator normalization, free-boson pole pinned, A = -2 Im G^R.
    n_ext, nw, n, _ = sigma.shape
    green = np.empty_like(sigma)
    eye = np.eye(n, dtype=complex)
    for i in range(n_ext):
        for w in range(nw):
            green[i, w] = np.linalg.inv(
                (omega[w] + 1j * eta) * eye - np.diag(bare[i]) - sigma[i, w]
            )
    return green, -2.0 * green.imag


def _linewidths(sigma, bare, omega):
    """``-Im Sigma^R_{aa}(k, eps_{k a})`` at the nearest grid point to the
    bare pole; reported separately from the artificial broadening.  Also
    returns the nearest-grid sampling offset of each on-shell evaluation
    (the grid-error estimate, not a linewidth).  MathDoc
    (docs/sympy/born_scattering_spectra, check_zero_coupling_and_thresholds):
    the absorptive part is threshold-gated, so the sampled on-shell value
    carries the reported sampling offset."""
    n_ext, n = bare.shape
    out = np.zeros((n_ext, n))
    for i in range(n_ext):
        for a in range(n):
            i_w = int(np.argmin(np.abs(omega - bare[i, a])))
            out[i, a] = -sigma[i, i_w, a, a].imag
    return out, _grid_sampling_error(bare, omega)


class _ScatteringMixin:
    """Seam methods added to :class:`MagnonPhononModel` (ADR-006)."""

    def scattering_spectra(self, request: ScatteringRequest) -> ScatteringResult:
        """Full-frequency one-shot Born spectra for the requested species.

        Per ADR-004 the (k, q) contractions are blocked: only the requested
        external points and internal meshes are materialized through
        :meth:`vertex`; no full tensor is allocated.  The full branch matrix
        structure is kept (no diagonal-only gauge).
        """
        omega = request.omega_eV
        eta = float(request.broadening_eV)
        magnon = None
        phonon = None
        if "magnon" in request.species:
            magnon = self._magnon_spectrum(request, omega, eta)
        if "phonon" in request.species:
            phonon = self._phonon_spectrum(request, omega, eta)
        grid_magnon = magnon.grid_error if magnon is not None else None
        grid_phonon = phonon.grid_error if phonon is not None else None
        metadata = {
            "approximation": _APPROXIMATION,
            "broadening_eV": eta,
            "omega_grid_eV": omega.copy(),
            "temperature_K": float(request.temperature_K),
            "magnon_mesh_qpoints": request.magnon_mesh_qpoints.copy(),
            "magnon_mesh_weights": request.magnon_mesh_weights.copy(),
            "phonon_mesh_kpoints": request.phonon_mesh_kpoints.copy(),
            "phonon_mesh_weights": request.phonon_mesh_weights.copy(),
            "static_tadpole_excluded": True,
            "phonon_kernel_status": (
                "certified: closed-form hot-band prefactor (S poles at "
                "+-Omega_nu, W hot-band poles at +-deps, T=0 exact "
                "vanishing, KMS-odd coordinate channel) asserted against "
                "the truncated-Fock Lehmann intermediate-state sum in "
                "docs/sympy/born_scattering_spectra and against the "
                "ring-model reference with truncation convergence"
            ),
            "tadpole_note": (
                "the thermal <a^dag a> contraction of the vertex is the "
                "static magnetostriction force (q = 0, linear in the "
                "phonon amplitude); it is reference bookkeeping and is "
                "not part of Sigma^R_dyn"
            ),
            "normalization": (
                "particle-channel commutator Green functions, A = -2 Im G^R; "
                "free-boson pole pinned in docs/sympy/born_scattering_spectra"
            ),
            # SPEC-002: the phonon self-energy is reported separately from
            # the additive coordinate correction
            "phonon_sigma_convention": (
                "sigma is the Dyson-partitioned self-energy "
                "Sigma_nu(w) = D0_nu^-1 deltaD_nu(w) D0_nu^-1 (branch-"
                "diagonal); green carries the ADDITIVE coordinate "
                "Green-function correction delta D on D0, which is not a "
                "self-energy; linewidths are -Im Sigma_aa(Omega_nu) from "
                "sigma, never from delta D"
            ),
            # SPEC-004: real grid-error estimate, separate from linewidths
            "grid_error": {
                "convention": (
                    "nearest-grid sampling offset |bare pole - nearest "
                    "omega grid point| (eV) of the on-shell linewidth "
                    "evaluation; internal momentum meshes are exactly "
                    "weighted, so this is the only grid error of the "
                    "on-shell evaluation"
                ),
                "magnon_max_offset_eV": (
                    float(np.max(grid_magnon)) if grid_magnon is not None else None
                ),
                "phonon_max_offset_eV": (
                    float(np.max(grid_phonon)) if grid_phonon is not None else None
                ),
            },
        }
        return ScatteringResult(
            magnon=magnon,
            phonon=phonon,
            omega_grid_eV=omega.copy(),
            temperature_K=float(request.temperature_K),
            broadening_eV=eta,
            metadata=metadata,
        )

    def _magnon_spectrum(self, request, omega, eta):
        n_k = request.magnon_kpoints.shape[0]
        n_q = request.magnon_mesh_qpoints.shape[0]
        blocks = np.empty((n_k, n_q), dtype=object)
        eps_kq = np.empty((n_k, n_q), dtype=object)
        bare = None
        om_q = np.empty(n_q, dtype=object)
        for i_q, (q, phonon) in enumerate(
            zip(request.magnon_mesh_qpoints, request.magnon_phonon_modes)
        ):
            for i_k, k in enumerate(request.magnon_kpoints):
                # blocked contraction: one requested (k, q) block at a time
                vertex = self.vertex(k, q, phonon_modes=phonon)
                blocks[i_k, i_q] = np.asarray(vertex.g, dtype=complex)
                eps_kq[i_k, i_q] = np.asarray(vertex.magnon_energies[0], dtype=float)
                if i_q == 0:
                    if bare is None:
                        bare = np.zeros(
                            (n_k, vertex.magnon_energies.shape[1]), dtype=float
                        )
                    bare[i_k] = np.asarray(vertex.magnon_energies[1], dtype=float)
            om_q[i_q] = thz_to_ev(vertex.phonon_frequencies_thz)
        n_modes = bare.shape[1]
        for i_q in range(n_q):
            om_q[i_q] = np.asarray(om_q[i_q], dtype=float)
        om_mesh = np.stack([om_q[i_q] for i_q in range(n_q)])
        nb_mesh = bose_occupation(om_mesh, request.temperature_K)
        # the kernel is per external k: eps_{k+q} differs between externals
        sigma = np.zeros((n_k, omega.size, n_modes, n_modes), dtype=complex)
        for i_k in range(n_k):
            g_k = np.stack([blocks[i_k, i_q] for i_q in range(n_q)])
            eps_k = np.stack([eps_kq[i_k, i_q] for i_q in range(n_q)])
            sigma[i_k] = magnon_self_energy(
                g_k,
                eps_k,
                om_mesh,
                nb_mesh,
                request.magnon_mesh_weights,
                omega,
                eta,
            )
        green, spectral = _particle_spectral(sigma, bare, omega, eta)
        linewidths, grid_error = _linewidths(sigma, bare, omega)
        return SpeciesSpectrum(
            species="magnon",
            momenta=request.magnon_kpoints.copy(),
            bare_energies=bare,
            sigma=sigma,
            green=green,
            spectral=spectral,
            linewidths=linewidths,
            grid_error=grid_error,
        )

    def _phonon_spectrum(self, request, omega, eta):
        n_k = request.phonon_mesh_kpoints.shape[0]
        spectra = []
        for i_q, (q, phonon) in enumerate(
            zip(request.phonon_qpoints, request.phonon_modes)
        ):
            g_list = []
            eps_k_list = []
            eps_kq_list = []
            for i_k, k in enumerate(request.phonon_mesh_kpoints):
                vertex = self.vertex(
                    k,
                    q,
                    phonon_modes=phonon,
                    mode_index=getattr(phonon, "mode_index", None),
                )
                g_list.append(np.asarray(vertex.g, dtype=complex))
                eps_k_list.append(np.asarray(vertex.magnon_energies[1], dtype=float))
                eps_kq_list.append(np.asarray(vertex.magnon_energies[0], dtype=float))
            n_modes_k = max(block.shape[1] for block in g_list)
            n_modes_kq = max(block.shape[0] for block in g_list)
            n_nu = g_list[0].shape[2]
            g = np.zeros((n_k, n_modes_kq, n_modes_k, n_nu), dtype=complex)
            eps_k = np.zeros((n_k, n_modes_k), dtype=float)
            eps_kq = np.zeros((n_k, n_modes_kq), dtype=float)
            for i_k, block in enumerate(g_list):
                g[i_k, : block.shape[0], : block.shape[1], : block.shape[2]] = block
                eps_k[i_k, : len(eps_k_list[i_k])] = eps_k_list[i_k]
                eps_kq[i_k, : len(eps_kq_list[i_k])] = eps_kq_list[i_k]
            nb_k = bose_occupation(eps_k, request.temperature_K)
            nb_kq = bose_occupation(eps_kq, request.temperature_K)
            bare = thz_to_ev(np.asarray(phonon.frequencies_thz, dtype=float))
            delta_d, sigma_nu = phonon_hot_band_kernel(
                g,
                eps_k,
                eps_kq,
                nb_k,
                nb_kq,
                bare,
                request.phonon_mesh_weights,
                omega,
                eta,
            )
            # Dyson self-energy as a branch-diagonal (1, nw, nu, nu) matrix;
            # green keeps the ADDITIVE coordinate correction delta D on the
            # bare D0 (delta D is not a self-energy and is reported as such
            # through metadata["phonon_sigma_convention"]).
            n_nu = bare.size
            pi = np.zeros((1, omega.size, n_nu, n_nu), dtype=complex)
            for i_nu in range(n_nu):
                pi[0, :, i_nu, i_nu] = sigma_nu[:, i_nu]
            d0 = _oscillator_bare(bare[None], omega, eta)
            pi_g = d0.copy()
            for i_nu in range(n_nu):
                pi_g[0, :, i_nu, i_nu] += delta_d[:, i_nu]
            pi_a = -2.0 * pi_g.imag
            linewidths = _oscillator_linewidths(pi, bare[None], omega)
            grid_error = _grid_sampling_error(bare[None], omega)
            spectra.append(
                SpeciesSpectrum(
                    species="phonon",
                    momenta=q[None].copy(),
                    bare_energies=bare[None],
                    sigma=pi,
                    green=pi_g,
                    spectral=pi_a,
                    linewidths=linewidths,
                    grid_error=grid_error,
                )
            )
        return _stack_species(spectra)


def _pad_block(block, n_m, n_nu):
    out = np.zeros((n_m, n_m, n_nu), dtype=complex)
    out[: block.shape[0], : block.shape[1], : block.shape[2]] = block
    return out


def _stack_species(spectra):
    """Concatenate per-momentum spectra along the external axis."""
    if len(spectra) == 1:
        return spectra[0]
    return SpeciesSpectrum(
        species=spectra[0].species,
        momenta=np.concatenate([s.momenta for s in spectra]),
        bare_energies=np.concatenate([s.bare_energies for s in spectra]),
        sigma=np.concatenate([s.sigma for s in spectra]),
        green=np.concatenate([s.green for s in spectra]),
        spectral=np.concatenate([s.spectral for s in spectra]),
        linewidths=np.concatenate([s.linewidths for s in spectra]),
        grid_error=np.concatenate([s.grid_error for s in spectra]),
    )


# ADR-006 seam attachment: the explicit calculation methods live in this
# module (numeric kernels are filesystem-free) and extend the Story-009
# model facade without modifying Story-009 files.
for _name in (
    "scattering_spectra",
    "_magnon_spectrum",
    "_phonon_spectrum",
):
    setattr(MagnonPhononModel, _name, getattr(_ScatteringMixin, _name))
