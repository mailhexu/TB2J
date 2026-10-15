# A-tensor → Heisenberg mapping: single-reference identifiability and compatibility flags

Story 012 (Epic 004, FR-009 / ADR-006, SymPy-first derivation gate). Script:
[`spiral_heisenberg_mapping.py`](spiral_heisenberg_mapping.py) — all assertions
green in mydev (`python docs/sympy/spiral_heisenberg_mapping.py`, ~20 s, nonzero
exit on any failure).

Authorities consumed, not imported: story 011
[`collinear_anchor_projection.py`](collinear_anchor_projection.py) (stored
ordered-pair convention, zero-pitch identity) and the frozen-response core
[`siesta_spiral_frozen_response.py`](siesta_spiral_frozen_response.py)
(version-2 weighted site rotations, q = 0 Ward identities). The consumer data
model is the `ResponseSet` (version 2.1) of
`TB2J/interfaces/siesta_spiral_response.py`: per-site declared frame axes,
gradient/torque per `(site, label)`, curvature per coordinate pair with the
onsite + response split, `operator_mode` labeling (ADR-004: production default
`onsite_delta`). Literature anchor for the A-tensor hierarchy: A. Szilva et al.,
*Quantitative theory of magnetic interactions in solids*, Rev. Mod. Phys.
**95**, 035004 (2023) (arXiv:2206.02415).

## Conventions pinned

- **Pair-once classical energy, unit directions, no $|\mathbf m|^2$ division**:

$$
E_{ab}(\hat e_a, \hat e_b) = -J_{ab}\,(\hat e_a\!\cdot\!\hat e_b)
+ \mathbf D_{ab}\!\cdot\!(\hat e_a\times\hat e_b)
+ \hat e_a\,\Gamma_{ab}\,\hat e_b ,\qquad a<b,
$$

  with $\Gamma_{ab}$ symmetric traceless ($\Gamma_{zz}=-\Gamma_{xx}-\Gamma_{yy}$)
  and $J>0$ ferromagnetic. The **DMI sign is the $+\mathbf D\cdot(\hat e_a\times\hat e_b)$
  convention**; flipping it flips every $\mathbf D$ and nothing else.
- **TB2J storage**: `exchange_Jdict` keeps both ordered directions at
  $J^{\rm stored}_{ab}=J^{\rm pair}_{ab}/2$ (story 011, `exchange_spiral.py`).
  All $J$ in this document are pair-once values.
- **Chart**: each site declares a right-handed orthonormal local frame
  $(\mathbf t_{a,1},\mathbf t_{a,2},\mathbf n_a)$; tilt directions are
  $\mathbf u_{a,\alpha}=\mathbf t_{a,\alpha}\times\mathbf n_a$; the two-axis
  rotations compose as $R(\mathbf t_2,\theta_2)R(\mathbf t_1,\theta_1)\,\mathbf n$
  (the declared exponential chart of the ResponseSet).
- **Torque** $\boldsymbol\tau=-\mathbf g$ with $\mathbf g=\partial E/\partial\theta$
  (FR-008, carried verbatim by the ResponseSet).

## Data model (asserted, section A)

From **one** spiral reference the observables are the transverse Hessian
$K_{a\alpha,b\beta}=\partial^2E/\partial\theta_{a\alpha}\partial\theta_{b\beta}\big|_0$
($\alpha,\beta\in\{1,2\}$, $a=b$ allowed; onsite and response split recorded but
consumed as a total) and the gradients. Asserted chart identities:

- $\hat e\cdot\hat e=1$ for arbitrary angles, $\hat e\cdot\partial_\alpha\hat e=0$
  identically, and $\mathbf n_a\cdot\partial_\alpha\hat e_a\big|_0=0$: the chart
  spans exactly the two tangent directions — **longitudinal response is absent
  by construction**, never merely unmeasured.
- The same-site off-diagonal second vertex vanishes in the composed chart and
  the diagonal seagull is $-\mathbf n$ (Clairaut-symmetric): the onsite blocks
  of any pair energy are $\propto\delta_{\alpha\beta}$.

## Pair mapping (asserted, section B)

Exact differentiation of the parametrized pair energy gives, for arbitrary frames:

$$
K_{ab}[\alpha,\beta] = -J\,(\mathbf u_{a,\alpha}\!\cdot\!\mathbf u_{b,\beta})
+ \mathbf D\!\cdot\!(\mathbf u_{a,\alpha}\times\mathbf u_{b,\beta})
+ \mathbf u_{a,\alpha}\,\Gamma\,\mathbf u_{b,\beta},
$$

$$
K_{aa}[\alpha,\beta] = +\delta_{\alpha\beta}\,J\,(\mathbf n_a\!\cdot\!\mathbf n_b)
- \delta_{\alpha\beta}\,(\mathbf n_a\,\Gamma\,\mathbf n_b),
\qquad
\mathbf g_{a,\alpha} = -J\,(\mathbf u_{a,\alpha}\!\cdot\!\mathbf n_b)
+ \mathbf D\!\cdot\!(\mathbf u_{a,\alpha}\times\mathbf n_b)
+ \mathbf u_{a,\alpha}\,\Gamma\,\mathbf n_b .
$$

Structure asserted channel by channel with fully symbolic frames: $J$ and
$\mathbf D$ never enter onsite blocks; $\mathbf D$ enters only through the mixed
cross products $\mathbf u_{a,\alpha}\times\mathbf u_{b,\beta}$ (and the mixed
cross products with $\mathbf n_b$ in the gradient); the anisotropy reaches the
onsite blocks only through the longitudinal combinations
$\mathbf n_a\Gamma\mathbf n_b$.

## Zero-pitch reduction to the story-011 identity (asserted, section C)

At $q=0$ (collinear, common transverse tilt direction $\mathbf t$, $\mathbf u=\mathbf t\times\hat n$):
$K_{ab}=-J$ exactly and the DMI channel is silent ($\mathbf u\times\mathbf u=0$).
With distinct transverse axes ($\mathbf u_a\cdot\mathbf u_b\neq0$),
$J^{\rm pair}_{ab}=-K_{ab}/(\mathbf u_a\!\cdot\!\mathbf u_b)$, and the story-011
stored ordered-pair identity holds verbatim:

$$
J^{\rm stored}_{ab} = -\frac{K_{ab}}{2\,{\rm sign}(\hat e_a\!\cdot\!\hat e_b)\,
(\mathbf u_a\!\cdot\!\mathbf u_b)},
\qquad J^{\rm pair}_{ab} = 2\,J^{\rm stored}_{ab},
$$

with the AFM reference flip checked explicitly ($\hat e_b=-\hat n$,
$K_{ab}=+J(\mathbf u_a\cdot\mathbf u_b)$). **In-plane-anisotropy contamination**:
the common-direction extraction absorbs the transverse anisotropy projection,

$$
\hat J = J - \mathbf u\,\Gamma\,\mathbf u ,
$$

and the onsite blocks close it: $K_{aa}=J+\Gamma_{11}+\Gamma_{22}$ in the chart
basis (traceless), so $(J,\Gamma_{\rm in\,plane})$ are jointly recoverable from
(mixed, onsite) data at a collinear anchor.

## Single-reference identifiability for two sites (asserted, section D)

Parameters $p=(J,\,D_x,D_y,D_z,\,\Gamma_{xx},\Gamma_{xy},\Gamma_{xz},\Gamma_{yz},\Gamma_{yy})$
(traceless closure; $D$ projected on site $a$'s local chart, which the anchors
place on the Cartesian axes). Data: the gauge-free pair block = the $3\times3$
Szilva block $A_{ab}[i,j]$ (Cartesian probe axes) restricted to the tangent
planes, i.e. $C_a\,A_{ab}\,C_b^{\,\rm T}$ with $C[\alpha,i]=(\mathbf t_{a,\alpha})_i$
— 4 of 9 entries, the other 5 being gauge (asserted: the annihilator
$\{M: C_a M C_b^{\rm T}=0\}$ is exactly 5-dimensional) — plus the onsite blocks.

Exact ranks and null spaces (all matrices built from the closed-form mapping,
ranks in exact arithmetic; symbolic $(c,s)$ means arbitrary tilt with
$c^2+s^2=1$):

| data (2 sites) | rank | undetermined combinations (null space) |
|---|---|---|
| scalar common-direction $K$ (story-011 extraction) | 1 (null 8) | all of $\mathbf D$ — **incl. the reference-axis component** — $\Gamma_{xz},\Gamma_{yz}$, and the $J\leftrightarrow$ in-plane-trace/$A_{zz}$ trade |
| full $2\times2$ mixed block, collinear | 4 (null 5) | $D_x,D_y,\Gamma_{xz},\Gamma_{yz}$, and $(\delta J,\delta\Gamma_{xx},\delta\Gamma_{yy})=(1,1,1)$ — the **longitudinal $A_{zz}$ shift** ($\delta\Gamma_{zz}=-2$) trading against $J$ |
| + onsite closure, collinear | 5 (null 4) | $D_x,D_y,\Gamma_{xz},\Gamma_{yz}$ only; $J$ and $\Gamma_{\rm in\,plane}$ determined |
| + torque, collinear | 7 (null 2) | the two tradeoffs $D_x\leftrightarrow\Gamma_{yz}$ and $D_y\leftrightarrow\Gamma_{xz}$ (the collinear torque sees only the sums $\mathbf D\cdot\mathbf t_\alpha+\mathbf t_\alpha\Gamma\mathbf n$) |
| planar single-$q$ reference (symbolic $c,s$), torque + full Hessian | Hessian-only 5; full 8 (null 1) | exactly $\delta D_{t1_a}=-\lambda,\ \delta\Gamma_{t2_a n}=+\lambda$; with the tilt along $t2_a$ the blind pair rotates to $D_{t2_a}\leftrightarrow\Gamma_{t1_a n}$ (both asserted symbolically) |

**Reference-axis DMI note.** The contract's example "DMI along the reference
axis undetermined" is exact for the *scalar* common-direction extraction
(story-011 protocol). The full $2\times2$ mixed block **does** determine the
reference-axis $D$ through its antisymmetric off-diagonal part
($\mathbf u_{a,1}\times\mathbf u_{b,2}\parallel\hat n\neq0$); the in-plane $D$
components stay blind at any collinear anchor.

**Multi-reference completion** (exact rational anchors): collinear + planar
noncollinear still leaves the shared planar blind direction (rank 8); two
mutually generic noncollinear references give rank 9, as does collinear +
two noncollinear. Full separation of $(J,\mathbf D,\Gamma)$ from transverse
data alone is therefore a *multi-reference* statement; a single reference
always leaves at least one $D$–$\Gamma$ tradeoff.

## Residual and compatibility statistic (asserted, section E)

For one pair, collect the gauge-free raw entries
$d=(K_{ab}[\alpha,\beta],K_{aa},K_{bb})$ (12 numbers). A mapping layer is a
linear map $M$ from $p$ to that layout; its representation of $d$ is the exact
least-squares projection $P=M\,M^{+}$. Pinned definitions:

- **per-component residual** $r_i = d_i - (M\hat p)_i$ with
  $\hat p=\arg\min\|d-Mp\|$ (exact pseudo-inverse);
- **aggregate** $R=\sqrt{r^{\rm T}W\,r}$ over the gauge-free coordinates
  (unit weights $W=\mathbb 1$ pinned; $W$ an explicit argument), i.e. the
  weighted distance of $d$ from the layer-representable subspace;
- **compatibility statistic** (dimensionless, explicit denominator floor)

$$
\chi \;=\; \sqrt{\frac{r^{\rm T}W r}{\max\!\big(d^{\rm T}W d,\;K_{\rm floor}^2\big)}},
\qquad K_{\rm floor}=10^{-6}\ {\rm eV/rad^2},\qquad
\text{flag incompatibility iff } \chi>\chi_{\max}=\tfrac{1}{20}.
$$

$\chi$ is exactly scale-invariant while the floor is inactive (asserted
symbolically and numerically on $2d$); $\chi=0$ for $d=0$.

## Synthetic constructions (asserted, section F)

Exact differentiation of the composed-chart energies; layers fitted by exact
pseudo-inverse; $\chi_{\max}^2=1/400$.

| synthetic (pure term, added to nothing) | collinear anchor | noncollinear $3\!-\!4\!-\!5$ reference |
|---|---|---|
| exact Heisenberg $-J\,\hat e_a\!\cdot\!\hat e_b$, $J=\tfrac1{10}$ | $\chi^2_{\rm iso}=\chi^2_{\rm sector}=0$; $\hat J=J$ recovered | same: zero residual, $\hat J=J$ |
| pure biquadratic $B(\hat e_a\!\cdot\!\hat e_b)^2$, $B=\tfrac1{100}$ | **invisible**: $\chi^2_{\rm iso}=\chi^2_{\rm sector}=0$; masquerades as $\hat J=-2B(\mathbf n_a\!\cdot\!\mathbf n_b)=-\tfrac1{50}$ — flag silent | **flagged**: $\chi^2_{\rm iso}=\tfrac{2752}{3115}$, $\chi^2_{\rm sector}=\tfrac{128}{267}$, both $>\tfrac1{400}$ |
| pure three-site $K_3(\hat e_a\!\cdot\!\hat e_b)(\hat e_a\!\cdot\!\hat e_c)$, $K_3=\tfrac3{100}$ | — | **flagged**: $\chi^2_{\rm iso}=\tfrac{1927}{2170}$, $\chi^2_{\rm sector}=\tfrac{59}{93}$; the sector flag fires through the onsite-block violation ($K_{aa}\propto-\Gamma_{nn}\mathbb 1$ fails) |

The biquadratic row is the sharp statement: **a collinear anchor cannot flag
biquadratic exchange at all** (it is exactly $J$-representable there), while a
noncollinear single reference flags it at both the isotropic and the full-sector
layer. This is a derivation-level justification for noncollinear anchors in the
flagging protocol.

## Assumptions and limits (honest)

1. **Pair model at one reference.** The mapping and every identifiability
   statement assume the 9-parameter pair energy above. Beyond-pair physics is
   constrained only through what violates the pair-shaped data (demonstrated
   for three-site terms via the onsite blocks); nothing is claimed about
   terms that stay inside the pair rowspace at the anchor.
2. **Single-reference determinacy.** A single reference never determines the
   full pair tensor from transverse data: one $D$–$\Gamma$ tradeoff always
   survives (rank 8 of 9 with torque and onsite closure; more without torque).
   The planar blind direction is exact and chart-relative
   ($D_{\rm tilt}\leftrightarrow\Gamma_{\rm other\,transverse,\,n}$); a
   second, suitably different reference closes it (rank 9 asserted).
3. **Reference-state dependence (Szilva).** $K$ and $\mathbf g$ are properties
   of the *state*, not of the pair alone; the mapping returns parameters
   defined *at* the reference. Comparing $J$ across different references mixes
   different states; only the zero-pitch identity (story 011) is
   reference-independent within its anchor class.
4. **Chart relativity.** $D_x,D_y,D_z$ and $\Gamma_{ij}$ are local-chart
   projections; null-space directions are chart-relative statements. The
   anchors place site $a$'s frame on the Cartesian axes, so "local-projected"
   coincides with Cartesian there.
5. **Operator-split agnosticism.** The layer consumes the total transverse
   Hessian and gradients; the ResponseSet onsite/response split
   (`curvature_split`, operator modes of ADR-004) is diagnostic and never
   enters the mapping. Nothing here re-weights or re-phases any vertex.
6. **Conventions, not invariants.** The DMI sign, the traceless-$\Gamma$ gauge
   (which is what pins $\mathbf D$ uniquely against $\Gamma\to\Gamma+\lambda\mathbb 1$-type
   shifts), the pair-once doubling, the $1/20$ flag threshold and the $10^{-6}$
   eV/rad² floor are policy choices pinned here; changing any of them changes
   the flagged outcomes and must be re-adjudicated.
