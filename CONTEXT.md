# TB2J Thermal-Magnon Domain Model

## Glossary

| Term | Meaning | Boundary |
|---|---|---|
| **Thermal-magnon solver** | A self-consistent solver that evaluates magnon energies, Bose occupations, and ordered moments at a specified temperature. | Does not mean the existing $T=0$ `Magnon` band solver. |
| **Thermal method** | One of RPA (Tyablikov), Callen decoupling (CD), HP mean-field, or RPA+CD; selecting `mfa` instead runs the Weiss single-site thermodynamic baseline, which sits outside these four magnon-spectrum methods. | A method is part of the physical result, not an interchangeable numerical option; MFA answers Brillouin order parameters m(T) and an analytic $T_\mathrm C$ but no temperature-dependent magnon bands. |
| **RPA+CD** | RPA for isotropic and exchange-anisotropy terms, with Callen decoupling only for single-ion anisotropy. | Required because SIA has an operator-ordering ambiguity. |
| **Curie temperature ($T_\mathrm C$)** | Continuous-loss temperature of uniform ferromagnetic order. | HP reports a finite-magnetization instability estimate rather than a continuous $m\to0$ $T_\mathrm C$. |
| **Néel temperature ($T_\mathrm N$)** | Continuous-loss temperature of staggered collinear antiferromagnetic order. | Requires a local-frame, multi-sublattice thermal formulation; it is not a Curie temperature with negative $J$. |
| **Quantum spin regime** | Uses dimensionless $S_i=|\mu_i|/(2\mu_\mathrm B)$ and quantum $S(S+1)$ factors. | Per-site $S_i$ must be recorded in the output. |
| **Classical spin regime** | Uses the $S(S+1)\to S^2$ prescription. | Intended for itinerant/local-moment classical modeling; it is not the quantum default. |
| **Longitudinal exchange anisotropy ($\lambda$)** | The coefficient of $S_i^zS_j^z$ beyond the isotropic exchange in the reference frame. | Does not include DMI or transverse/off-diagonal tensor couplings. |

| **Supported AFM thermal state** | A collinear bipartite reference with two equivalent antiparallel sublattices. | It is the initial $T_\mathrm N$ contract; arbitrary collinear, canted, spiral, and inequivalent AFMs are unsupported. |
| **Unsupported thermal tensor term** | DMI or transverse/off-diagonal bilinear exchange outside the $J_\mathrm{iso}+\lambda S_i^zS_j^z$ form. | Thermal solving must reject it; it must not silently project it away. |
| **Mesh-converged transition** | A $T_\mathrm C$ or $T_\mathrm N$ whose prescribed q-mesh sequence satisfies its reported tolerance. | A fixed-mesh result must be explicitly marked nonconverged rather than presented as a material prediction. |
| **Zero transition** | Absence of finite-temperature long-range order within a stable model, e.g. isotropic 1D/2D RPA. | Distinct from an invalid or unstable reference state. |
| **Unstable reference** | Input whose $T=0$ harmonic modes violate the solver’s stability condition. | It precludes a transition result. |
| **HP breakdown** | The first temperature where HP mean field makes a magnon energy nonpositive at finite magnetization. | A method-specific estimate, not a continuous $m\\to0$ critical point. |

| **Thermal dimensionality** | User-declared physical periodicity: 1D, 2D, or 3D. | It must agree with the exchange support; crystallographic cell dimensionality alone is insufficient. |
| **Order mode** | User-declared `ferromagnetic` or `bipartite_afm` ordered reference. | The solver validates supplied reference moments; it never guesses from exchange signs. |
| **Flagged transition estimate** | Last q-mesh result returned when tolerance fails, with `converged=false` and mesh history. | Strict mode converts this outcome into an error. |

| **AFM thermal Nambu solver** | A local-frame, two-sublattice self-consistency using normal and anomalous bosonic/spin correlators. | It is required for all-method $T_\mathrm N$; the FM scalar convolution is invalid here. |
| **Thermal result schema** | Versioned `tb2j.magnon.thermal` JSON with optional NetCDF representation. | It carries thermal bands, transition statuses, inputs, and q-mesh history; it is not the `eigenstates` schema overloaded. |

| **Method-validity status** | Provenance attached when a selected method has a known limited regime, e.g. low-spin Callen. | It never triggers an implicit method fallback. |
| **Physical versus effective quantum spin** | An integer/half-integer $S$ versus a real spin length continued from a DFT moment. | Both are permitted; only the former supports exact finite-spin identities. |
| **Single-ion anisotropy (SIA, $A$)** | The coefficient in $-A(S_i^z)^2$. | RPA/TB2J-$T=0$ has gap $2AS+S\lambda_0$; HP/CD has $A(2S-1)+S\lambda_0$. |
| **Temperature-dependent band** | The magnon eigenvalues evaluated after thermal self-consistency at one temperature. | It must retain method, spin regime, q-mesh, and convergence metadata. |

## References

- [Sympy derivations](docs/sympy/README.md): authoritative mathematics and checks.
- [Thermal-magnon scope ADR](docs/adr/0001-thermal-magnon-scope.md): accepted product-level decisions.
