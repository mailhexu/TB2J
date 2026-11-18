# Spiral integration fixture - inspection summary (story 007)

Fixture: two-sublattice dimer chain (Cr2, t1=0.35, t2=0.25, t3=0.15,
eps=+-0.05, B_local=3.0 eV, nel=2, width=0.05 eV), real TBUpy
rotating-frame spinor SCF at q = 1/6 and q = 0, cached sidecars under
`tests/_spiral_integration_cache/`.

## Gates (ExchangeSpiral, contour kernel, n_matsubara = 3000)

| gate              | q = 1/6          | q = 0            |
|-------------------|------------------|------------------|
| goldstone         | PASS  2.4e-13    | PASS  4.2e-14    |
| torque            | PASS  2.1e-13    | PASS  4.2e-14    |
| diag_consistency  | PASS  2.1e-13    | PASS  4.2e-14    |
| q0_anchor         | n/a (q != 0)     | PASS  1.1e-04 *  |
| C^db              | 6.2e-26          | 6.2e-26          |

* tolerance 5e-4, documented below.

## Extracted J (dominant shells, eV)

| shell            | q = 1/6 (A-B) | q = 1/6 (A-A / B-B) | q = 0 (A-B) | q = 0 (A-A / B-B) |
|------------------|---------------|---------------------|-------------|-------------------|
| (0,0,0) rung     | -0.0253       | -                   | -0.0251     | -                 |
| (+-1,0,0)        | -0.0064       | -0.0486 / -0.0499   | -0.0094     | -0.0605 / -0.0628 |
| (+-2,0,0)        | +0.0012       | +0.0055 / +0.0061   | +0.0027     | +0.0055 / +0.0061 |

Signs alternate and magnitudes decay - physically sensible for the
itinerant dimer chain.

## Documented findings

1. `q0_anchor` tolerance 5e-4: kernel identities (C^dd = C^bb, C^db = 0)
   hold to 1e-13; |Cbb - M| = 1.1e-4 is the FD discretization plus the
   finite-smearing free-energy term carried by `inplane_response_fd`
   (it recomputes occupations from perturbed eigenvalues).  Frozen-f FD
   agrees with the kernel to 9e-7.  Flagged for the story-004/005 owners.
2. LKAG anchor: J_spiral(q=0) = 2 x J_ExchangeCL2 exactly (dominant
   shells 2.000-2.016; the deviation from 2 is the frozen-occupation vs
   CFR Matsubara smearing floor).  The ExchangeCL2 value was verified
   against exact second-order perturbation theory (agreement to 6
   decimals), so the factor 2 originates in the normative
   `j(R) = -M(0,R)` identification (transverse curvature with Bf = dU/2
   operators vs the Liechtenstein `Tr[D G^up D G^down]` with D = dU).
   Flagged for the story-004/005 owners.
3. Exact torque-free references require `q . tau_mu` sublattice-
   independent; an intra-cell `q . tau` phase cants the frozen reference
   at O((t/B)^2) and fails the torque gate at the 1e-3 level.  The
   fixture therefore displaces sublattice B transversally.
4. Cone gate: on the dimer's classical model (FM-favouring) the conical
   rebuild collapses to the field-aligned state; the gate runs on the
   spiral-stabilized J1-J2 written-tensor class (story-006 probe
   guidance).  There the conical reference is stationary, the phason is
   gapless (<= 5e-4), and the +-q gap is exactly linear in the field
   with structure factor kappa = 1.0000 (omega(+-q) = B exact on the
   the fluctuation-mode identity; flagged for the derivation owners).
