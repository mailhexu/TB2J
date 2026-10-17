# Born scattering spectra: retarded kernels for magnons and phonons

Asserted derivation companion of `born_scattering_spectra.py` (Story 010,
ADR-004). Every formula below is pinned by an assertion in that script; none
is transplanted from a report. Authoritative source: the script, not this file.

## Declared Hamiltonian

The scattering calculation starts from the certified Story-009 first-order
spin-lattice Hamiltonian and nothing else:

    H = H0 + V,
    H0 = sum_{k mu} eps_{k mu} a^dag_{k mu} a_{k mu}
       + sum_{q nu} Omega_{q nu} b^dag_{q nu} b_{q nu},
    V = 1/sqrt(N_cell) sum_{k q} sum_{m n nu} g[m, n, nu](k, q)
        a^dag_{k+q, m} a_{k, n} (b_{q, nu} + b^dag_{-q, nu}).

`g[m, n, nu](k, q)` is the materialized `VertexResult.g` block (modes at
`k+q`, modes at `k`, phonon modes at `q`), in eV, with the zero-point
amplitude applied exactly once (Story 009). The calculation is one-shot
O(g^2): bare internal propagators, no vertex corrections, no
self-consistency (not SCBA), not on-shell-only.

## Observable normalization (commutator, not fermionic)

All Green functions are particle-channel commutator functions,

    G^R_AB(t) = -i theta(t) <[A(t), B^dag]>,   G^R(omega) = integral,
    A_AB(omega) = -2 Im G^R_AB(omega),

i.e. the bosonic normalization with `A = -2 Im G`, NOT the fermionic
`-Im G / pi`. The free boson pins sign and prefactor:

    G^R_free(omega) = 1/(omega - Omega + i 0),
    A_free(omega) = 2 eta / ((omega - Omega)^2 + eta^2)

(eta the artificial broadening): unit positive pole weight, integral
`2 pi`. The equal-time commutator moments of the bare propagators are
`m0 = int A/2pi = [b, b^dag] = 1` and
`m1 = int omega A/2pi = Omega = -<[[H, b], b^dag]>`; the Hermitian
coordinate channel `x = (b + b^dag)/sqrt(2 Omega)` has the antisymmetric
spectral function with `m0 = 0` and the occupation-independent commutator
moment `m1 = 1`. Both are asserted symbolically with the explicit
Lorentzian integrals.

## Bose weights

From the canonical thermal expectations (asserted from the geometric
series): `<b b^dag> = n_B + 1`, `<b^dag b> = n_B`,
`n_B(E) = 1/(exp(E/k_B T) - 1)`, and `n_B -> 0` exactly as `T -> 0`.
These are the only temperature-dependent prefactors of the kernels.

## Magnon kernel (one loop)

To O(g^2) the only T = 0 intermediate state behind a created magnon at
`(k+q, m)` is that magnon plus an emitted phonon (energy
`eps_{k+q, m} + Omega_{q nu}`, matrix element `g`); the absorption channel
carries the thermal weight `n_B`. With `G0^{-1}(omega) = omega - eps` on
the external leg, the asserted kernel is

    Sigma^R_{alpha beta}(k, omega) = 1/N_cell sum_{q} sum_{m nu}
        g[m, alpha, nu](k, q) g*[m, beta, nu](k, q)
        [ (n_B(Omega_{q nu}) + 1) / (omega - eps_{k+q, m} - Omega_{q nu} + i eta)
        +  n_B(Omega_{q nu})        / (omega - eps_{k+q, m} + Omega_{q nu} + i eta) ].

The full matrix structure in the branch indices `(alpha, beta)` is kept;
no diagonal gauge is taken, so degenerate subspaces transform covariantly
(`Sigma -> Theta Sigma Theta^dag`).

## Phonon correction (coordinate channel, hot bands)

The declared first-order Hamiltonian conserves magnon number (asserted on
a Fock space), so there is no two-magnon phonon decay: the bubble chi is

    chi_nu(nu) = g^2 sum_k (n_{k,n} - n_{k+q,m})
                 / (nu + eps_{k,n} - eps_{k+q,m} + i eta),

which vanishes exactly at T = 0 and has its particle-hole pole at
eps_{k+q} - eps_k.  CERTIFIED against exact truncated-Fock Lehmann
reference (two-mode model): the phonon COORDINATE-channel correction
x = (b + b^dag)/sqrt(2 Omega_nu)

- is O(g^2) with the thermal density (vanishes at T = 0, quadratic in g),
- carries absorptive weight in HOT BANDS at
  omega = +- Omega_nu -+ (eps_{k+q,m} - eps_{k,n})
  (Lang-Firsov/polaron structure), NOT at the bare pole omega = Omega_nu,
- and the coordinate spectral function A = -2 Im D is exactly odd (KMS).

PENDING (Story-010 follow-up, flagged as `phonon_kernel_status` in the
seam metadata): the closed-form prefactor of the hot-band kernel
(candidate forms: Lindhard-Dyson `D^-1 = D0^-1 - 2 Omega chi` and polaron
structure-factor shifts) is NOT yet certified against the exact Lehmann
reference at the asserted tolerance; the seam implements the certified
hot-band kernel shape with its derivation status reported, and the
multi-branch phonon certification test is skipped until the prefactor is
derived.  The magnon kernel above is fully certified.

## High-frequency limits and moments

Both kernels decay as `1/omega` with no `omega^0` term:

    Sigma(omega) -> sum |g|^2 [(n_B + 1) + n_B] / omega = sum |g|^2 (2 n_B + 1)/omega,
    Pi(omega)    -> sum |g|^2 (n_{k,n} - n_{k+q,m}) / omega.

The magnon asymptote equals the thermal two-particle spectral sum
`sum_m p_m g^2 [(m + 1) + m]` (asserted on a truncated Fock space,
including the exact truncation-boundary correction), i.e. the second
commutator moment of the coupling. The absence of an `omega^0` constant
is what separates `Sigma^R_dyn` from static bookkeeping below.

## KMS / detailed balance

For a transition of gap Delta between states of Boltzmann weights
`p` and `p e^{-beta Delta}`, the dynamic-structure-factor weights are
`S(+Delta) = p`, `S(-Delta) = p e^{-beta Delta}` (asserted), so
`S(-omega) = e^{-beta omega} S(omega)`, and the commutator spectral
function is odd. In kernel form: `(n_B + 1) = e^{beta Omega} n_B` for the
emission/absorption pair, and the Lindhard numerator
`(n_{k,n} - n_{k+q,m})` combines the same adjacent-ladder Boltzmann
weights whose ratio is `e^{-beta (eps_{k+q} - eps_k)}`. The reported
one-sided particle-channel self-energies inherit these identities
through the two-sided construction.

## Static tadpole / magnetostriction bookkeeping

Contracting `a^dag_{k+q, m} a_{k, n}` in V with the equilibrium magnon
state gives `(delta_{q,0} n_{k,n}/sqrt(N_cell)) g (b_0 + b^dag_0)`: a term
LINEAR in the q = 0 phonon amplitude, i.e. the static magnetostriction
force `delta E / delta tau_0`. It is real, frequency-independent, and
vanishes in the magnon vacuum (`n_k = 0` at T = 0) and for a stationary
reference. It is therefore NOT part of `Sigma^R_dyn` and is not added to
the returned self-energies; conversely `Sigma^R_dyn` is exactly the
frequency-dependent object above and carries no `omega^0` asymptote
(asserted). Reference-equilibrium static shifts belong to a separately
relaxed reference, not to these spectra.

## Zero-coupling recovery and thresholds

`g -> 0` gives `Sigma = Pi = 0` exactly and the bare spectra. At T = 0
the magnon absorptive part exists only in the emission window
`omega > eps_{k+q} + Omega`; below threshold `Im Sigma` is a pure
broadening artifact scaling linearly with `eta`, and the real part is the
principal value. The phonon bubble is absorptive only across its
particle-hole gap. Artificial broadening and grid error are reported
separately from calculated linewidths by the seam (Story 010 contract).

## Units

Energies in eV throughout the kernels; phonon frequencies enter through
`E = h nu` from THz, magnon energies as exported by the Story-006
canonical modes; temperature in K with `k_B` in eV/K.
