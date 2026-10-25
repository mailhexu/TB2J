# EMC raw full-block rotation: derivation and oracle contract

**Canonical executable:** `docs/sympy/emc_raw_block_rotation.py`  
**Authority scope:** Story 015, full-block frozen response and endpoint-image certificate.  
**Execution status:** The parent executed the stabilized script in `mydev`; all assertions passed. The Observed verification section records its `run_checks()` measurements. These synthetic checks do not establish material accuracy.

## Source conventions and authority

The approved Story 015 and its ADRs set the model being derived; this note does not amend them.

- ADR-008 defines the physical response as an equal partition of full magnetic-block conjugations in a fixed lab basis. The scalar block remains fixed. The endpoint set is the set of distinct `(owner atom, actual lattice cell)` pairs, not primitive orbital or site labels. Its equal weights are an approximation; a saved Hamiltonian does not uniquely determine its magnetic derivative.
- ADR-009 requires both structural common-unitary covariance and the actual response map's uniform energy/curvature test. Rotating all owners, including ligand owners, is the structural comparison. A restricted response that freezes a polarized owner may fail the physical gate, and that failure withholds certification. A transverse derivative under common rotation is allowed.
- ADR-010 requires explicit orbital ownership and response; this authority models the ownership keys but does not claim to infer a material map.
- ADR-012 resolves each effective image before applying endpoint phases. It fixes the cell-referenced endpoint phase to `exp(+2 pi i q dot d)` and the HamiltonIO convention-2 electronic/operator phase to `exp(+2 pi i k dot d)`, with projection from `k` to `k+q`.
- The amended research memo's endpoint discussion (F2-F3) distinguishes a fixed-basis physical response from `G^dagger H G` and explains why owner cells and images cannot be dropped.

The inspected implementation sources agree with these pins. `HamiltonIO/HamiltonIO/wannier/wannier_hamiltonian.py:246-272` applies the convention-2 phase with a positive `R2kfactor * dot(k,R)` and divides by `R_degens`; lines 283-305 expand the per-pair WS transform over `R+T`, with denominator `ndegen * N_T`. The script's explicit supercell Fourier check uses that image weight once. `TB2J/docs/sympy/planar_su2_frame.py:41-58` identifies whole-basis dressing as a gauge transformation and distinguishes it from physical field rotation. `TB2J/docs/sympy/siesta_spiral_vertex.py:25-59` is evidence for a declared equal partition in its own spiral representation, not authority to copy its `q/2` sector gauge into this ordinary `k -> k+q` construction.

## Block decomposition and named physical approximation

Take an arbitrary complex spinor block; no Hermiticity is imposed on a directed block:

$$
W = h_0 I + B,\qquad h_0 = \tfrac12\operatorname{Tr}W,\qquad \operatorname{Tr}B=0.
$$

For one effective image let its directed orbital endpoints be `i` in cell `n` and `j` in cell `n+d`, where `d=R+T`. If their owner offsets are `C_i` and `C_j`, the physical endpoint keys are

$$
E_i=(\mathrm{owner}_i,n+C_i),\qquad
E_j=(\mathrm{owner}_j,n+d+C_j).
$$

Deduplicate by the complete owner-and-cell key. Thus two orbitals owned by one atom in the same cell share an endpoint; the same atom in another cell is a different endpoint. Different owner atoms remain different endpoints even if their cells match. The offset and effective image enter before phases and weights are applied.

Let `P` be that set of distinct physical endpoints and let `U_e=exp(-i theta_e sigma_y/2)`. The named **equal-physical-endpoint frozen response** is

$$
W'(\{\theta_e\}) = h_0 I + \sum_{e\in P}\frac{1}{|P|}U_e B U_e^\dagger.
$$

The scalar hopping is unchanged. A one-endpoint block receives one full conjugation; a two-endpoint block receives one half-weighted full conjugation per physical endpoint. The sum restores the undeformed `B` at zero angle, and equal endpoint weights sum to one. This is not a sequence of left/right dressing operations.

For a directed block with two distinct endpoints, put `D_y(B)=-i[ sigma_y, B ]/2`. Its exact first derivative at the reference is

$$
\delta W = \tfrac12\theta_i D_y(B)+\tfrac12\theta_j D_y(B).
$$

For a single shared physical endpoint the coefficient is one, not two halves. The script derives the general complex `W` result symbolically and asserts the one-endpoint, two-endpoint, scalar-fixed, weight-conservation, common-`U`, and partner identities.

A common angle gives

$$
h_0 I+\sum_{e\in P}|P|^{-1}U B U^\dagger=h_0 I+UBU^\dagger=U W U^\dagger,
$$

for every owner and any directed `W`. This structural identity does not choose the equal weights: it only depends on their sum being one.

## Hermitian partner

The directed block may be non-Hermitian. Its separate partner is constructed as

$$
W_{ji}(-d)=W_{ij}(d)^\dagger=h_0^* I+B^\dagger,
$$

with endpoints exchanged and the same endpoint allocation and image weight. For real finite angles, applying the same endpoint rotations to that partner gives the adjoint of the rotated forward block. The executable oracle builds both directions explicitly; it never replaces an individual directed block by its Hermitian part.

## Cell, image, and Fourier phases

For cell-referenced motion, the endpoint amplitudes at the actual owner cells are

$$
a_i(q)e^{+2\pi i q\cdot C_i},\qquad
a_j(q)e^{+2\pi i q\cdot(d+C_j)}.
$$

For distinct endpoints, the complex `+q` first-order block before the electronic Fourier factor is

$$
\frac12\left[e^{+2\pi i q\cdot C_i}+e^{+2\pi i q\cdot(d+C_j)}\right]D_y(B).
$$

HamiltonIO convention 2 then contributes `exp(+2 pi i k dot d)`, and the response is projected from `k` to `k+q`. The finite-supercell projection uses Bloch basis amplitudes `exp(+2 pi i k n)`; its matrix element is `<k+q|delta H|k>`. The `+q` endpoint modulation is therefore selected with the stated signs. This is the ordinary `k -> k+q` operator convention; no SIESTA `q/2` sector gauge is used.

Every effective WS image is evaluated independently at its own `d=R+T`. Each shell/image coefficient `1/(ndegen * N_T)` multiplies that image's complete block response exactly once. Image expansion cannot in general be moved after differentiation because the endpoint phases depend on `d`.

The authority implements the cell-referenced mode above. If a caller supplies a site-referenced magnetic-site amplitude instead, the corresponding magnetic-site `tau` phase is included once in that amplitude. A Wannier center is not substituted for `tau` and contributes no accidental endpoint phase.

## Independent commensurate finite-angle oracle

The oracle builds the real-space supercell from the original directed matrices and their explicit adjoints. For every endpoint it computes the actual owner-cell angle, constructs finite `U(theta) B U(theta)^dagger`, adds the fixed scalar part, and applies the image weight once. It does not call the analytic derivative builder. Independent real patterns are built and centrally differentiated:

$$
\theta_{n,e}^{(c)}=\alpha\cos(2\pi q\cdot c_{n,e}),\qquad
\theta_{n,e}^{(s)}=\alpha\sin(2\pi q\cdot c_{n,e}).
$$

With the declared Bloch projection, the `+q` Fourier coefficient of the cosine derivative is `V_c=A(+q)/2`; that of the sine derivative is `V_s=A(+q)/(2i)=-iA(+q)/2`. Consequently the complex channel is `V_c+iV_s=A(+q)`, with no additional factor of two.

The inequivalent-owner test uses a commensurate eight-cell chain with `q=1/4`, `k=1/8`, effective images `d=1,2` from one `R=1` shell, nonzero owner-cell offsets, and a complex nonsymmetric `W`. Here `q N=2`, `k N=1`, `(k+q)N=3`, and `q` and `-q` are distinct supercell harmonics. The test includes the adjoint partner and checks a direct unrotated supercell projection against the explicit convention-2 Fourier sum.

An independent same-owner intercell test uses `q=1/4`, `d=2`, so `q dot d=1/2`. Its physical endpoints are `(same atom, cell n)` and `(same atom, cell n+2)`, not one primitive-site endpoint. Equal endpoint coefficients give

$$
\tfrac12(1+e^{+2\pi i q\cdot d})D_y(B)
=\tfrac12(1+e^{i\pi})D_y(B)=0.
$$

The independent finite-angle supercell remains the correct physical, Hermitian construction throughout. The cell-collapsed control instead builds a deliberately wrong analytic candidate (collapsing the same-owner endpoints to one primitive-site key) and checks that candidate against the correct finite-angle cancellation oracle; it does not feed a non-Hermitian corrupted Hamiltonian into the positive oracle path. A second executable negative control uses the wrong sine sign (`V_c-iV_s`) in the `k -> k+q` projection and requires a large mismatch. These are assertions, not prose-only warnings.

### Error and tolerance protocol

The central angle step is `2e-6 rad`. If the analytic reference Frobenius norm is at most `1e-12`, the comparison uses maximum absolute matrix error and requires `<=5e-9`; it never divides by a near-zero reference. Otherwise it uses relative Frobenius error and requires `<=5e-8`. The undeformed convention-2 supercell/Fourier comparison uses maximum absolute error `<=2e-12`. Uniform energy curvature is evaluated with a `2e-3 rad` central step and `2e-6` curvature tolerance. Observed results from the stabilized run are recorded below.

## Separate uniform-response gates and gauge control

### Structural all-owner common-unitary gate

For a finite Hermitian spinor Hamiltonian, the structural comparison applies one common spinor unitary to every block, including ligand-owned blocks:

$$
H_{\mathrm{struct}}(\theta)=G(\theta)H G(\theta)^\dagger.
$$

The executable check compares the equal-partition full-block response to this separately constructed similarity, compares the spectrum, and evaluates the occupied-energy curvature. It deliberately checks a nonzero transverse derivative without requiring it to vanish. This gate is mathematical covariance, not a physical-response or Goldstone certificate.

### Actual-map rigid-rotation gate

Separately, the actual declared endpoint response is evaluated on

$$
H=\begin{pmatrix}b\sigma_z&tI\\tI&b\sigma_z\end{pmatrix},\qquad b=2,\;t=1,
$$
with the two orbitals assigned to a magnetic owner and a polarized ligand owner. When both owners follow the rotation, the response is the common-unitary deformation and its uniform occupied-energy curvature is zero. When only the magnetic owner rotates and the ligand is frozen, the exact perturbative curvature for the two occupied states is

$$
E_{\mathrm{occ}}''(0)=-\frac{b t^2}{2(b^2-t^2)}=-\frac13.
$$

The script checks this deliberately restricted map against that nonzero value and asserts that its physical uniform-response gate is false. It reports certification as withheld; passing the structural gate cannot convert this physical failure into a Goldstone pass. The exact formula and fixture are mathematical specifications, not a reported execution result.

### Pure-gauge negative control

A separate `G^dagger H G` construction is checked for spectral invariance under endpoint-local basis rotations. Its matrix is compared with the fixed-basis physical response. In particular, basis dressing rotates a scalar inter-owner hopping through `U_i^dagger (tI) U_j`, while the declared physical response leaves that scalar hopping fixed and rotates only `B`. Spectral invariance of the former is therefore not the physical-response certificate.

## Executable API and run protocol

The canonical script exports:

- `derive_expressions()`: a lexically named mapping of SymPy expressions/matrices, with exact assertions;
- `derive()`: the Kosmic MathDoc list of `{name, srepr, latex}` results, sorted by identifier;
- `run_checks()`: executes all symbolic assertions, the independent finite-angle oracles, uniform gates, and negative controls, then returns the actual observation mapping without printing.

The registry's machine protocol is `--kosmic-json`: the script runs the full assertion/oracle suite first and then emits only `{schema_version: 1, python_version, sympy_version, results}`. Normal mode prints the actual observations. Commands for the parent verification are:

```text
python docs/sympy/emc_raw_block_rotation.py
python docs/sympy/emc_raw_block_rotation.py --kosmic-json
```

### Observed verification

Executed from the TB2J root in `mydev` with Python 3.11 and SymPy 1.14.0:
`python docs/sympy/emc_raw_block_rotation.py`. All assertions passed.

| Check | Observed value |
|-------|----------------|
| Nonzero finite-angle/Fourier relative error | `6.358381558672055e-11` |
| Exact-zero cancellation maximum absolute residual | `2.938735877055719e-33` |
| Unrotated convention-2 Fourier/weight error | `1.734723475976807e-17` |
| Wrong-q-sign relative mismatch | `1.000000000001194` |
| Collapsed-owner candidate gap from correct oracle | `2.8284271247461903` |
| Structural common-U spectrum drift | `1.332267629550188e-15` |
| Structural occupied-energy curvature | `-4.440892098500626e-10` |
| Actual all-owner occupied-energy curvature | `8.881784197001252e-10` |
| Restricted-map curvature (expected `-1/3`) | `-0.33333304738647485` |
| Pure-gauge spectrum drift | `3.330669073875470e-16` |
| Pure-gauge/fixed-basis matrix gap | `0.7931524547529532` |

The restricted-map physical gate returned false and certification remained
withheld, as required. An initial generic fixture accidentally cancelled
the entire response; its second image was changed to retain a nonzero
general channel. The collapsed-endpoint negative control compares a wrong
analytic candidate with the correct Hermitian finite-angle oracle; a
corrupted non-Hermitian producer is not accepted as an oracle.

## Scope and limitations

This derivation identifies a declared equal-endpoint frozen approximation, not a unique physical derivative inferred from raw `H`. It does not calculate self-consistent ligand, itinerant, spin-density, or charge-density motion; it does not establish a material response map, AFM mode conversion, production vertex, or material accuracy. The finite-angle examples are deterministic synthetic algebra oracles, not DFT or experimental validation. Independent scientific review and the Story 015 parent-run evidence remain required.
