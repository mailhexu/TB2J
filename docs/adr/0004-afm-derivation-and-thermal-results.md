# ADR 0004: AFM theory gate and thermal result format

- **Status:** Accepted
- **Date:** 2026-08-25
- **Decision owners:** TB2J maintainers

## Context

A collinear AFM produces anomalous bosonic terms after the local sublattice rotation.
The scalar FM thermal equations in arXiv:2405.00477 are not an AFM implementation.
Swendsen, *Phys. Rev. B* **11**, 1935 (1975), DOI
[10.1103/PhysRevB.11.1935](https://doi.org/10.1103/PhysRevB.11.1935), provides a
published Callen AFM Green-function precedent, including low-spin instability warnings.

TB2J's registered CLI is `TB2J_magnon.py`; current band/eigenstate persistence is
versioned JSON with optional NetCDF.

## Decision

1. Before AFM code, derive and assert a local-frame Nambu formulation for RPA, CD, HP,
   and RPA+CD: normal and anomalous correlators, sublattice-resolved order, and TN
   condition. Link the derivation to the literature and benchmark it.
2. In AFM RPA+CD, use RPA for isotropic and longitudinal exchange anisotropy, and Callen
   decoupling only for SIA.
3. Extend `TB2J_magnon.py` and its existing flat TOML `MagnonParameters` convention;
   do not promote the unregistered legacy `TB2J_magnon2.py`.
4. Create a versioned `tb2j.magnon.thermal` result schema in JSON, with optional NetCDF,
   for explicit temperatures/bands, Tc/TN/HP-breakdown status, method, spin regime,
   physical dimensionality, order mode, and q-mesh history.
5. Release-blocking validation is model-only: SymPy/exact identities, isotropic 1D/2D
   zero-transition constraints, simple-cubic FM, and published bipartite/type-I AFM
   benchmarks. Material examples are nonblocking.

## Consequences

No AFM thermal solver can be presented as a direct implementation of the scalar
ferromagnetic paper. The schema is a new public contract, so it must be independently
round-tripped in JSON/NetCDF and versioned from its first release.
