# Planar SU(2) Frame: Hopping, Overlap and Both Rotation Vertices

Companion script: `planar_su2_frame.py` (assertion-checked; run in the
`mydev` environment). Status: **all assertions pass** (2026-09-25).

Story 001 (spiral-first-order-response), NFR-001/NFR-005; pins ADR-R1
conventions before any production code. The script imports neither TB2J
nor TBUpy; every builder is an independent oracle, and production
(`tbupy/planar_spiral.py`, TB2J planar kernels) must match these
identities, not the other way round.

## Pinned conventions

* Interleaved spinor layout `2*(cell*norb + mu) + s`; folded primitive
  blocks indexed `2*mu + s`.
* Site unitary about the spiral normal $y$:
  $U(\theta)=e^{-i\theta\sigma_y/2}=
  \begin{psmallmatrix}\cos\frac\theta2&-\sin\frac\theta2\\
  \sin\frac\theta2&\cos\frac\theta2\end{psmallmatrix}$, so the
  lab-frame planar field at angle $\Theta$ is
  $B_f(\cos\Theta\,\sigma_z+\sin\Theta\,\sigma_x)=U(\Theta)\,\sigma_z\,U(\Theta)^\dagger$
  with $B_f = B_{\mathrm{local}}/2$; the stored up/down splitting is
  $2B_f$ (asserted by the flat local-frame field `Bf*sz`).
* Lab spiral angles
  $\Theta_{a\mu}=2\pi\,\mathbf q\cdot(\mathbf R_a+\boldsymbol\tau_\mu)+\varphi_\mu$,
  sublattice phase $\alpha_\mu=2\pi\,\mathbf q\cdot\boldsymbol\tau_\mu+\varphi_\mu$.
* Gauge transform of a spin-scalar class, with **both rotation
  vertices** — the left vertex $\mu$ and the right vertex $\nu$ enter
  only through
  $\Delta_{\mu\nu}(\mathbf R)=2\pi\,\mathbf q\cdot\mathbf R+\alpha_\nu-\alpha_\mu$:

$$
H_q(\mathbf k)_{\mu\nu}=\sum_{\mathbf R}e^{-2\pi i\,\mathbf k\cdot\mathbf R}\,
 h_{\mu\nu}(\mathbf R)\,U\!\big(\Delta_{\mu\nu}(\mathbf R)\big),
 \qquad
 S_q(\mathbf k)_{\mu\nu}=\sum_{\mathbf R}e^{-2\pi i\,\mathbf k\cdot\mathbf R}\,
 s_{\mu\nu}(\mathbf R)\,U\!\big(\Delta_{\mu\nu}(\mathbf R)\big),
$$

  with the Hermitian pairing $h_{\nu\mu}(-\mathbf R)=h_{\mu\nu}(\mathbf R)^*$
  (overlap alike) and the flat local-frame field $B_f\sigma_z$ on site.
  The vertex decomposition
  $\Delta = 2\pi\mathbf q\cdot\mathbf R + \alpha_\nu - \alpha_\mu$ is
  asserted with $\partial\Delta/\partial\varphi_\mu=-1$,
  $\partial\Delta/\partial\varphi_\nu=+1$,
  $\partial\Delta/\partial q_{\alpha}=2\pi(R_\alpha+\tau_{\nu\alpha}-\tau_{\mu\alpha})$:
  the twist vertex, and each end of the intra-cell vertex, are
  separately pinned.
* Folded mesh on $N$ cells: $k_m=(m+s)/N$ with the half flux shift
  $s=\tfrac12$ iff $\mathrm{round}(q_x N)$ is odd, else $s=0$. A wrong
  shift yields a plausible but wrong spectrum; the rule is verified for
  both parities.

## Checks and observed residuals

| Check | Content | Observed |
|-------|---------|----------|
| [A] SU(2) basics | unitarity, $U(a)U(b)=U(a+b)$, lab-field form | exact |
| [B] dimer gauge transform | $G^\dagger H_{\rm lab}G=H_{\rm loc}$, $G^\dagger S_{\rm lab}G=S_{\rm loc}$, both vertices | exact (symbolic) |
| [C] lab ring vs folded pencil | multi-sublattice, nonorthogonal $S$, spectra + eigenvector map, even and odd $qN$ | spectrum dev $\le 7.0\times10^{-16}$; mapped-vector pencil residual $\le 1.4\times10^{-14}$ |
| [D] mirrored pair | $(e^{+2\pi i\mathbf k\cdot\mathbf R},U(-\Delta))$ spectrum | dev $1.0\times10^{-15}$ |
| [E] pure-gauge invariance | whole-basis site unitaries change no eigenvalue | drift $6.2\times10^{-15}$ |

## The gauge transform, both vertices

For the dimer the script verifies symbolically, with
$G=\mathrm{diag}(U(\vartheta_1),U(\vartheta_2))$:

$$
G^\dagger
\begin{pmatrix} B_1\hat{\mathbf n}_1\!\cdot\!\boldsymbol\sigma & h\,\mathbb 1\\
h\,\mathbb 1 & B_2\hat{\mathbf n}_2\!\cdot\!\boldsymbol\sigma\end{pmatrix}
G
=
\begin{pmatrix} B_1\sigma_z & h\,U(\vartheta_2-\vartheta_1)\\
h\,U(\vartheta_1-\vartheta_2) & B_2\sigma_z\end{pmatrix},
$$

and identically for the overlap with $s_{12}\,\mathbb 1\to
s_{12}\,U(\pm\Delta)$. Each physical bond is touched by exactly two
site rotations — its two vertices — and both appear once, in
$U(\Delta)$ and $U(\Delta)^\dagger$. The multi-sublattice numeric check
exercises the same structure for every class simultaneously.

## Eigenvector gauge map (not just spectra)

Spectrum equality alone cannot distinguish a correct frame from a
flipped one. The script therefore also verifies the map of generalized
eigenvectors: with $c_{\rm fold}(\mathbf k_m)$ solving
$H_q\,c=\varepsilon\,S_q\,c$ and $c^\dagger S_q c=1$,

$$
c_{\rm lab}\big[(a\mu s)\big]
=\sum_{s'}\big[U(\Theta_{a\mu})\big]_{ss'}\;
 e^{-2\pi i k_m a}\; c_{\rm fold}\big[(\mu s')\big],
$$

and every mapped vector satisfies the explicit lab-ring pencil equation
$H_{\rm lab}c=\varepsilon\,S_{\rm lab}c$ to $\le1.4\times10^{-14}$
(absolute, on $\mathcal O(1)$ matrices). This is the verified gauge
transform required by the story's first acceptance criterion,
generalized-eigenvector level, nonorthogonal overlap included.

## Mirrored Fourier/dressing pair

Production code may equivalently use
$H_q(\mathbf k)=\sum_{\mathbf R}e^{+2\pi i\,\mathbf k\cdot\mathbf R}h\,U(-\Delta)$:
it is the $\mathbf k\to-\mathbf k$ image of the pinned pair, and its
mesh spectrum coincides with the lab ring through the time-reversal
evenness of $E(\mathbf q)$. Asserted numerically. Note the eigenvector
map of the mirrored pair is the $-\mathbf k$ map; only spectra (not
vector identities) are transferable between the two pairs.

## What is NOT a physical perturbation

Check [E] pins the invariant that production gradients must respect: a
whole-basis local unitary (every operator conjugated by fixed site
unitaries) is pure gauge — the spectrum, hence any frozen-occupation
band energy, is exactly invariant. Consequently:

* the **local** $\beta_i,\delta_i$ gradient operators act on the
  magnetic on-site field only (plus any *declared* co-rotating local
  operator, e.g. a constraint potential with a recorded rotation
  policy), with $\partial S=0$: in the local frame
  $V_1^{\beta}=B_f\sigma_x$, $V_1^{\delta}=B_f\sigma_y$
  (verified in `local_gradient_nonorthogonal.py`);
* a global $q$ twist is different: it moves the SU(2) dressing phases
  of hopping AND overlap, so $\partial_q H_q$ **and** $\partial_q S_q$
  are generically nonzero (quantified in `pitch_slope_generalized.py`).

Confusing the two — differentiating the dressing along a local
direction — measures a gauge artefact that cancels only if the
$-\varepsilon\,\partial S$ term is retained; the cancellation is
asserted explicitly in `local_gradient_nonorthogonal.py`.

## Implementation consequences

1. Production planar providers must assemble $H_q$ and $S_q$ with the
   same dressing $U(\Delta)$ on both, the same Fourier sign, and the
   same half-shift rule; any deviation is falsifiable by check [C].
2. v2 sidecar `gauge='planar_y'` means exactly this frame;
   `field_role` and `field_rotation_policy` metadata distinguish the
   physical field rotation from the pure-gauge directions of [E].
3. Commensurate lab rings are oracles only; arbitrary
   (incommensurate) $\mathbf q$ needs the primitive folded pencil.
4. The overlap participates fully: nonorthogonal $S$ is dressed,
   positive definite on the folded mesh, and its derivative enters the
   pitch slope with the $-\varepsilon\,\partial S$ structure.
