# Collinear-limit projection: transverse curvature → Heisenberg J

Story 011 (AC-006, ADR-003 derivation gate). Script:
[`collinear_anchor_projection.py`](collinear_anchor_projection.py) — all
assertions green in mydev.

## Conventions pinned

- **Physical pair-once convention**:
  $E=-\sum_{a<b}J^{\rm pair}_{ab}\,\hat e_a\cdot\hat e_b$;
  positive $J^{\rm pair}$ = FM. **TB2J storage** differs:
  `exchange_Jdict` stores both ordered directions with
  $J^{\rm stored}_{ab}=J^{\rm pair}_{ab}/2$; see
  `TB2J/exchange_spiral.py:490-497` and its ExchangeCL2 adjudication.
  The values are in eV per unit-vector pair; there is **no division by
  $|\mathbf m|^2$**. Reference-moment sign enters the AFM projection.
- **Frozen-response observable** (story 009): the raw transverse Hessian
  $K_{ab} = \partial^2 E/\partial\theta_a\partial\theta_b$ in eV/rad², where
  $\theta_a$ tilts site $a$'s magnetic operator about a declared transverse
  axis $\mathbf t_a$ of the collinear reference axis $\hat n$.

## Identities (symbolically asserted)

**IDENTITY-1.** With $\hat e_a(\theta_a)=R(\mathbf t_a,\theta_a)\,\hat n$,
$\mathbf t_a\perp\hat n$, and $\mathbf u_a=\mathbf t_a\times\hat n$ a unit
vector:

$$
\left.\frac{\partial^2(\hat e_a\cdot\hat e_b)}{\partial\theta_a\,\partial\theta_b}\right|_0
=\mathbf u_a\cdot\mathbf u_b .
$$

**IDENTITY-2 (projection).** For the pair-once energy:

$$
K_{aa}=\sum_{b\ne a}J^{\rm pair}_{ab},\qquad
K_{ab}=-J^{\rm pair}_{ab}(\mathbf u_a\cdot\mathbf u_b)\quad(a\ne b).
$$

The ordered-pair TB2J value for non-orthogonal transverse axes is:

$$
J^{\rm stored}_{ab}=-\frac{K_{ab}}{2(\mathbf u_a\cdot\mathbf u_b)}.
$$

For a **common tilt direction** the denominator's dot product equals one.
The sum rule $K_{aa}=-\sum_{b\ne a}K_{ab}$ applies to a *pure pairwise
Heisenberg* energy. Frozen band energy is not guaranteed to satisfy it;
report the residual, do not force diagonal curvature into the mapping.

**IDENTITY-3 (AFM).** An antiparallel reference ($\hat e_b=-\hat n$) flips
the sign:

$$
K_{ab}=+J^{\rm pair}_{ab}(\mathbf u_a\cdot\mathbf u_b),\qquad
J^{\rm stored}_{ab}=
-\frac{K_{ab}}{2\,\mathrm{sign}(\hat e_a\cdot\hat e_b)
(\mathbf u_a\cdot\mathbf u_b)}.
$$

**IDENTITY-4 (numeric).** Finite differences of the classical Heisenberg
energy with random pair-once $J$ and random per-site transverse axes
project back to the ordered-pair $J^{\rm stored}=J^{\rm pair}/2$
(agreement < 1e-9).

## Consequences for the anchor comparison (AC-006)

1. The new pipeline maps offsite curvature to the **ordered-pair** value
   $J^{\rm stored}_{ab}=-K_{ab}/[2(\mathbf u_a\cdot\mathbf u_b)]$
   on the FM zero-pitch reference. The FM anchor needs no sign correction.
2. Site-pair curvature sums all periodic images (the pencil contains
   intercell hopping), so compare with the image-summed established-path
   $\sum_R J^{\rm stored}(R,i,j)$ on the same k-mesh. Neither side resolves
   individual $R$ through this site-only rotation; the certificate says so.
3. Preserve $K_{aa}$ and its Heisenberg closure residual as diagnostics, not
   as a normalization adjustment or a required identity for band energy.
