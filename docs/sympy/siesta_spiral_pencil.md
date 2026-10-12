# SIESTA rotating-frame spiral pencil — derivation report

Paired executable: `siesta_spiral_pencil.py` (authoritative; run in the mydev
environment). Story 001 of `siesta-fixed-spin-spiral-nonorthogonal-xc-mft`
(epic 001). All checks below are asserted by the script; it exits nonzero on
any failure.

## [A] Spinor packing and Hermitization (symbolic)

Stored components `H1=H↑↑, H2=H↓↓, H3=Re H↑↓, H4=−Im H↑↓` reconstruct a
Hermitian lab block. With `kphs = e^{−ik·x}`, `qphs = e^{−iq·x}` the rotating
frame applies `D11 = kphs·qphs` to the ↑↑ sector, `D22 = kphs·conj(qphs)` to
↓↓, and Hermitizes the spin off-diagonal **within each stored entry**:

$$
\mathrm{D}_{12} \;=\; \tfrac{1}{2}\left[(H_3 - i H_4) D_{11} \;+\;
\overline{(H_3 + i H_4) D_{22}}\right] \;=\; (H_3 - i H_4)\, e^{-i q x}\,
\cos(k x)
$$

Two distinct Hermiticity mechanisms are pinned:

1. **Spin off-diagonal** — each entry contributes a conjugate `C12/C21` pair.
2. **Orbital off-diagonal** — via the partner entry `(ν,μ,−R)` of
   `W(ν,μ,−R) = W(μ,ν,R)†`, whose stored components are
   `H1'=H1, H2'=H2, H3'=H3, H4'=−H4` at `x' = −x`; the partner's assembled
   block is the elementwise conjugate (verified symbolically). Per-entry
   blocks alone are **not** Hermitian — only the summed pair is.

Spin-diagonal channels carry pure momenta:

$$
H_{\uparrow\uparrow}(k) \sim W_{\uparrow\uparrow}(k+q), \qquad
H_{\downarrow\downarrow}(k) \sim W_{\downarrow\downarrow}(k-q)
$$

## [B] Ring/folded equivalence (numeric, spin-diagonal blocks)

For random nonorthogonal multi-orbital models (SPD S, general spin-diagonal
W), the union of generalized eigenvalues of the SIESTA-assembled pencils
over the folded commensurate k mesh equals the supercell ring pencil
spectrum to `1e-13`. The mechanism: each channel is a plain Fourier
transform at `k±q`, and the commensurate mesh is invariant under the
q-shift.

**Scope boundary (important):** for blocks with spin mixing (`H3, H4 ≠ 0`)
the stored-component frame is fixed by the SCF (`vmatsp`/`rhoofdsp`), not by
`diag2kspiral` alone — the `cos(kx)` off-diagonal mixes ±k branches, so the
full textured-frame equivalence cannot be pinned from static source analysis
with synthetic blocks. It is deferred to the producer-fixture stories
(AC-001/AC-002), where the importer oracle arbitrates. What the importer can
already rely on analytically is the entire [A] structure.

## [C] The rotating-frame overlap is spin-dependent

Even with a spin-independent lab overlap, the assembled rotating-frame
overlap differs per spin channel:

$$
S_{\uparrow}(k) = S_{\mathrm{lab}}(k+q), \qquad
S_{\downarrow}(k) = S_{\mathrm{lab}}(k-q)
$$

(exact to machine precision). **Consequence for the response core:** all
dual-basis formulas (traces, resolvents, DOS, later Green-function
contractions) must use the assembled spinor `S(k)`, never the scalar lab S.

## [D] Dual-basis identities on the generalized pencil

Oroszlány et al. (PRB 99, 224412, App. B) carry **two** dual forms; the
trace rule pairs with the coefficient form (a conflation trap — the
transcription in the research memo §2 has been clarified accordingly):

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

1. Reconstruct complex blocks as `H12 = H3 − iH4`; hermitize per entry.
2. Apply `e^{−i(k±q)·x}` per spin sector to **both** H and S blocks.
3. Carry the assembled spinor S(k) into every pencil computation.
4. The fixture oracle (stories 003/004) must arbitrate the textured-frame
   convention for mixing components; the [A] structure is the analytic
   pre-check the importer can run without a fixture.
