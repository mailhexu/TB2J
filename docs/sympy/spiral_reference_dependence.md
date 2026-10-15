# Spiral reference dependence and comparison scope (story 013)

Authority script: `spiral_reference_dependence.py` (run in mydev:
`python docs/sympy/spiral_reference_dependence.py`; nonzero exit on any
failure). Normative mapping rows are those of
`spiral_heisenberg_mapping.py` (story 012); this file states the
derivation, the comparison-scope policy and the exact results.

## 1. Definition of reference dependence [A]

Reference dependence is **not** an `E(q)` fit. It is defined as the
difference of the pair-once isotropic Heisenberg coefficient fitted by
exact least squares on the transverse response of the *same frozen model*
at *two reference orientations* of the *same pair/site identity*:

$$\delta J \;=\; \hat J(\mathrm{ref}_1) - \hat J(\mathrm{ref}_2),$$

with $\hat J(\mathrm{ref}) = (L^T d)/(L^T L)$, $L$ the 12-entry isotropic
layer at that reference (mixed $2\times2$ block plus both onsite
$2\times2$ blocks, story-012 layout) and $d$ the raw differentiated data
`d = d²E/dθ_a dθ_b`, evaluated at $\theta = 0$ on the two-axis composed
exponential chart. No ensemble, SCF, or pair changes between the two
orientations; only the frozen reference direction moves.

Two fitted quantities are kept strictly distinct throughout:

- **Isotropic diagnostic** $\hat J$: one-parameter K-only least-squares
  projection onto the isotropic layer (a pure diagnostic; it is NOT what
  production exports).
- **Full joint fit $J_{\mathrm{fit}}$**: the J component of the
  nine-parameter least-squares solution of the FULL layer (12 Hessian rows
  + 8 gradient rows over $J, D, \Gamma$) — this is the layer whose J
  component production exports as `j_pair_once_eV`. The reference
  dependence is asserted for BOTH.

## 2. Exact bilinear Heisenberg invariance [B]

For the exact bilinear pair $E = -J\,(e_a\!\cdot e_b)$, direct
differentiation at any reference gives the story-012 rows
$K_{ab} = -J\,(u_a u_b)$, $K_{aa} = +J\,(n_a\!\cdot n_b)\,\delta$,
$g = -J\,(u\,n)$, which are exactly represented by the isotropic layer:
residual $r^2 = 0$ and $\hat J = J$ identically. Hence

$$\delta J = 0 \quad\text{(symbolically, and numerically at } J = 1/10).$$

Verified exactly at three reference classes: collinear, planar 3-4-5
($n_b = (4/5,0,3/5)$), and noncollinear rational ($n_b = (2,1,2)/3$).
**The exact bilinear Heisenberg $J$ is reference invariant.**

## 3. Biquadratic departure: exact counterexample [C]

Let $E = -J x - B x^2$ with $x = e_a\!\cdot e_b$. $x^2$ is quartic in the
tilt angles, so its Hessian is not bilinear-layer representable.

- **Collinear reference** ($x|_0 = 1$): the biquadratic Hessian is exactly
  isotropic-representable with zero residual and the fit *absorbs* $B$:

  $$\hat J_{\mathrm{col}} = J + 2B,\qquad r^2 = 0 .$$

  A single reference cannot distinguish the pair: a fitted $J$ from the
  full $\Gamma$/D hierarchy at one collinear reference silently contains
  $B$. (This is the story-012 masquerade $\hat J = -2B$ for pure
  $+B x^2$, shifted by the sign convention here.)

- **Planar 3-4-5 reference**: direct differentiation leaves an exact
  nonzero residual and

  $$\delta J(\mathrm{col},\mathrm{planar}) = \hat J_{\mathrm{col}} - \hat J_{\mathrm{planar}}
   = \tfrac{284}{175}\,B \neq 0,$$

  vanishing identically at $B = 0$ (the whole shift is the $B$ channel).

- **Noncollinear rational reference**:
  $\delta J(\mathrm{col},\mathrm{nonplanar}) = \tfrac{118}{87} B$, distinct
  from the planar shift — the departure is orientation-specific, not a
  collinear artifact.

- **Numeric verdict** at $J = 1/10$, $B = 1/100$:
  $|\delta J| \approx 0.01623$;
  $\chi^2_{\mathrm{planar}} = 516/52045 \approx 9.91\times10^{-3}$,
  $\chi^2_{\mathrm{nonplanar}} = 1275/215644 \approx 5.91\times10^{-3}$ —
  both **incompatible** under the boundaries below. The nonzero $\delta J$
  is asserted symbolically (exact rational multiples of $B$) and
  numerically.

**Full nine-parameter joint fit (K + torque).** Repeating the
differentiation with the full layer (12 Hessian + 8 gradient rows over
$J, D, \Gamma$), whose J component production exports as
`j_pair_once_eV`:

- Collinear: exact fit, zero residual, and **still** $J_{\mathrm{fit}} = J + 2B$
  — the full hierarchy absorbs $B$ at a single reference just like the
  isotropic projection.
- Planar 3-4-5: $J_{\mathrm{fit}} = J + \tfrac{102}{125}B$,
  $\Delta J_{\mathrm{fit}}(\mathrm{col},\mathrm{planar}) = \tfrac{148}{125}B$
  (numeric $|\Delta J_{\mathrm{fit}}| \approx 0.01184$); residual
  $\chi^2 = 8/2271 \approx 3.52\times10^{-3}$ → **incompatible**.
- Noncollinear rational: $J_{\mathrm{fit}} = J + \tfrac{26}{27}B$,
  $\Delta J_{\mathrm{fit}}(\mathrm{col},\mathrm{nonplanar}) = \tfrac{28}{27}B$;
  residual $\chi^2 = 25/10326 \approx 2.42\times10^{-3}$ → **indeterminate**
  (grey band; never compatible). The fit spreads part of $B$ into the
  $D/\Gamma$ channels (visible in the exact fitted parameter vector),
  which is precisely why the single-reference full-fit residual is the
  weaker diagnostic — the two-orientation $\Delta J_{\mathrm{fit}}$
  exposes the departure either way, exactly.
- Sanity: with exact Heisenberg data the full joint fit recovers
  $J_{\mathrm{fit}} = J$ with zero residual at every reference.

## 4. Approved compatibility boundaries [D]

$$\chi^2 = \frac{r^2}{\max(d^2, k_{\mathrm{floor}}^2)}:\quad
\begin{cases}
\chi^2 \le 10^{-4} & \Rightarrow\ \text{compatible},\\
\chi^2 \ge 2.5\times10^{-3} = (1/20)^2 & \Rightarrow\ \text{incompatible},\\
\text{otherwise} & \Rightarrow\ \text{indeterminate (grey band, width } 3/1250).
\end{cases}$$

Exact rational pins: compatible max $= 1/10000$, incompatible min
$= 1/400$ (story-012 $\chi^2_{\max}$). Exact-Heisenberg data ($\chi^2=0$)
lands compatible; the section-3 biquadratic data lands incompatible.

## 5. Comparison scope declaration [E]

**Permitted**

- Scalar pair-once $J$ compared across reference orientations of the same
  frozen model, same pair/site identity, same SCF ensemble, with recorded
  ensemble and reference provenance on both sides.

**Forbidden**

- Chart-local $D/\Gamma$ comparisons without the site-local frame
  transforms: those parameters are projections onto per-site tilt charts
  and are not scalars under a change of reference frame.
- Cross-pitch or cross-SCF comparisons without matched provenance: the
  story-004 $\sim 0.1$ eV/cell cross-SCF spread forbids presenting
  unmatched cross-SCF exchange differences as converged results.
- Labeling an incompatible response (e.g. any detected reference
  dependence or $\chi^2$ above the boundaries) as an exact Heisenberg
  result; it is reported as a non-Heisenberg departure alongside the
  untouched raw response.

## 6. Nonplanar three-site identifiability [F]

Motivation: the Fe4 producer fixture's Mulliken moment normals are not
exactly coplanar, and `map_response_set` currently refuses nonplanar
finite-q references. Exact test with generic rational site normals
$(2,1,2)/3,\ (1,2,2)/3,\ (2,2,1)/3$, triple product $-5/27 \neq 0$
(noncoplanarity asserted), and distinct generic tangent seeds per site:

- For **every** of the three pairs, the existing story-012 pair rows —
  mixed block + onsite closure + corrected torque rows — have **exact
  rank 9 with trivial null space**. Identifiability is asserted
  **pair-locally only**; no claim is made about a joint stacked
  multi-pair map.
- The extracted $J$ is **identical** from (i) a single mixed entry
  against its tilt dot, (ii) the full isotropic least-squares layer,
  (iii) the onsite closure $K_{aa} = J (n_a\!\cdot n_b)\,\delta$, and
  (iv) the torque rows $g_{a,\alpha} = -J (u_a\!\cdot n_b)$ — and
  identical across all three pairs: **reference-class independent**.

No planarity of the moment normals is required for pair
identifiability; the planar-only gate is not justified by the mapping
layer and can be removed for the real producer fixture.

## 7. Provenance

Conventions: docs/sympy/collinear_anchor_projection.py (story 011) and
spiral_heisenberg_mapping.py (story 012): pair-once energy, transverse
two-axis exponential chart, torque = −gradient, $E_{\mathrm{dmi}} =
+D\!\cdot(e_a\times e_b)$. Literature anchor: A. Szilva et al., Rev. Mod.
Phys. **95**, 035004 (2023) (reference-state dependence). SHA-256 of both
files is printed by the script on every run.
