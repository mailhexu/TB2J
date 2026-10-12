# SIESTA rotating-frame spiral pencil — derivation report

Paired executable: `siesta_spiral_pencil.py` (authoritative; run in the mydev
environment). Story 001 of `siesta-fixed-spin-spiral-nonorthogonal-xc-mft`
(epic 001). All checks below are asserted by the script; it exits nonzero on
any failure. Revised after independent review (round 1) corrected two
convention errors: the spiral phase is **q/2** (`diag2kspiral.F90` line 141,
`qwh = qw*0.5`), and `setup_k` applies **no** Hermitization on the Hamiltonian
path.

## [A] Spinor packing and phase placement (symbolic, source-accurate)

Stored components `H1=H↑↑, H2=H↓↓, H3=Re H↑↓, H4=−Im H↑↓` reconstruct a
Hermitian lab block. With `kphs = e^{−ik·x}` and `qphs = e^{−iq·x/2}`:

$$
\uparrow\uparrow: D_{11} = k_{\rm phs}\, q_{\rm phs}, \qquad
\downarrow\downarrow: D_{22} = k_{\rm phs}\, \overline{q_{\rm phs}}, \qquad
H_{12} \to (H_3 - iH_4) D_{11},\quad H_{21} \to (H_3 + iH_4) D_{22}
$$

- **No Hermitization** on this path; the `(D12+conj(D21))/2` averaging in
  the source converts density/energy matrices only.
- Spin-diagonal channels carry pure momenta `k ± q/2` — exactly the pinned
  TBUpy convention `diag(H(k−q/2), H(k+q/2))`.
- **Hermiticity requirement on stored components:** the full `H(k)` is
  Hermitian iff the partner entry `(ν,μ,−R)` stores `H12' = conj(H21)`
  (stored-component form: `H4' = −H4`) at `x' = −x`; then the partner's
  assembled block is the elementwise conjugate (verified symbolically).
  Whether a producer's stored sparse components satisfy this is a
  **fixture-certifiable** convention question (story-003 oracle).

## [B] Ring/folded equivalence (numeric; spin-diagonal blocks, even m)

For random nonorthogonal multi-orbital models and `q = 2πm/N` with **even**
m, each channel is a plain Fourier transform at `k±q/2` and the folded mesh
is invariant under the q/2-shift: the union of generalized eigenvalues of
the SIESTA-assembled pencils equals the supercell ring pencil spectrum to
`1e-14` (phases on **both** H and S included).

**Scope boundary (honest):** odd m places the shifted channels on the
magnetic-BZ (2N-point) mesh, and spin-mixing blocks (`H3, H4 ≠ 0`) have
their stored-component frame fixed by the SCF (`vmatsp`/`rhoofdsp`), not by
`diag2kspiral` alone. Both require a real producer fixture and are deferred
to stories 003/004 (AC-001/AC-002).

## [C] The rotating-frame overlap is spin-dependent

Even with a spin-independent lab overlap, the assembled rotating-frame
overlap differs per spin channel:

$$
S_{\uparrow}(k) = S_{\rm lab}(k+q/2), \qquad
S_{\downarrow}(k) = S_{\rm lab}(k-q/2)
$$

(exact to machine precision). **Consequence for the response core:** all
dual-basis formulas (traces, resolvents, DOS, later Green-function
contractions) must use the assembled spinor `S(k)`, never the scalar lab S.

## [D] Dual-basis identities on the generalized pencil

Oroszlány et al. (PRB 99, 224412, App. B) carry **two** dual forms; the
trace rule pairs with the coefficient form (conflation trap — the research
memo §2 has been clarified accordingly):

$$
\mathrm{Tr}\,A \;=\; \mathrm{Tr}\!\left(S\, S^{-1}A\right)
\;=\; \mathrm{Tr}\!\left(S^{-1} A S\right)
$$

Pencil resolvent factorization (exact, `S⁻¹` on the **right**):

$$
(zS - H)^{-1} \;=\; \left(zI - S^{-1}H\right)^{-1} S^{-1}
$$

Spectral decomposition and DOS (numeric, machine precision):

$$
(zS-H)^{-1} = \sum_n \frac{c_n c_n^\dagger}{z - \varepsilon_n},
\qquad H c_n = \varepsilon_n S c_n, \quad c_n^\dagger S c_n = 1
$$

$$
\rho(\varepsilon) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\!\left[
S\,(zS-H)^{-1} \right] = \sum_n \frac{\delta/\pi}{(\varepsilon-\varepsilon_n)^2+\delta^2}
$$

## Downstream consequences (importer contract, story 002)

1. Spiral phases are **q/2** per spin sector on both H and S blocks.
2. Reconstruct complex blocks as `H12 = H3 − iH4`; no hermitization when
   assembling the eigenproblem.
3. Carry the assembled spinor S(k) into every pencil computation.
4. The fixture oracle (stories 003/004) must certify (i) the stored-component
   pairing `H12' = conj(H21)` for Hermiticity, (ii) the odd-m magnetic-BZ
   folding, and (iii) the textured-frame convention for mixing components.
