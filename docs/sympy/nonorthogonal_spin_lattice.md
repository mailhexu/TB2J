# Spin-Lattice Downfolding on a Non-Orthogonal Basis

Companion script: `nonorthogonal_spin_lattice.py` (assertion-checked; run in
the `mydev` environment). All checks below are exact (sympy) or
high-precision oracles (mpmath, dps 40). Status: **see table** (2026-09-29).

This derivation extends the electron–phonon → spin-lattice downfolding of
TB2J (`TB2J/exchange_pert2.py`, `TB2J/Oiju_epw3.py`; manuscript
`epcdownfolding`) from orthogonal Wannier functions to non-orthogonal LCAO
bases (SIESTA/ABACUS via HamiltonIO `LCAOHamiltonian`, which already carries
real-space `SR`). Previously a displacement $\tau_a$ entered only through the
EPW vertex $g_a=\partial V/\partial\tau_a$; in a non-orthogonal basis the
overlap $S$ also moves, and both derivatives below are required.

## 1. Setup and conventions

Covariant resolvent (the convention of TB2J `green.py::get_Gk`, which already
forms $G(z)=(zS-H)^{-1}$ when the model carries an overlap):

$$
G^{\sigma}(z)=\bigl(z\,S-H^{\sigma}\bigr)^{-1},\qquad z\in\mathbb C ,
$$

with the LKAG exchange (orthogonal-basis form, `epcdownfolding` note §2.4)

$$
J_{IJ}^{\mathrm{iso}}
=\frac{1}{4\pi}\int_{-\infty}^{E_F} d\varepsilon\,
\operatorname{Im}\operatorname{Tr}
\Bigl[\Delta_I\,G^{\uparrow}_{IJ}(\varepsilon)\,
\Delta_J\,G^{\downarrow}_{JI}(\varepsilon)\Bigr],
$$

where $\Delta_I$ is the on-site exchange splitting of atom $I$
($H$-like/covariant; $\Pi_i$ Pauli blocks in `exchange_pert2`). Production
never touches the real axis: `TB2J/contour.py` integrates $A(z)$ over a
semicircle from `emin` to $E_F$ dipping into $\operatorname{Im}z<0$, and the
oracle below uses exactly that contour. The Fermi level is held fixed when
differentiating (magnetic-force-theorem convention).

## 2. Central identities

**(R1) Displacement one-form.** For $A(z,\tau)=zS(\tau)-H(\tau)$,
$G=A^{-1}$:

$$
\frac{\partial G^{\sigma}}{\partial\tau_a}
=G^{\sigma}\,V_a^{\sigma}(z)\,G^{\sigma},
\qquad
\boxed{\,V_a^{\sigma}(z)=\frac{\partial H^{\sigma}}{\partial\tau_a}
-z\,\frac{\partial S}{\partial\tau_a}\,}
$$

The EPW vertex $g_a$ is replaced by an **energy-dependent pencil vertex**
$V_a(z)$. This is exact for every $z$ off the spectrum, so it holds at every
contour/CFR quadrature point with no extra approximation. $S$ is
spin-independent; $H^\sigma$ is not, so $V_a^\uparrow\ne V_a^\downarrow$
through $\partial_a\Delta_I$ on-site terms.

**(R2) Consistency with level shifts.** With the generalized eigenproblem
$Hc=\varepsilon Sc$, $c^{\dagger}Sc=1$, the spectral form
$G=\sum_n c_nc_n^{\dagger}/(z-\varepsilon_n)$ gives the double-pole residue

$$
(z-\varepsilon_n)^2\,G\,V_a(z)\,G\Big|_{z\to\varepsilon_n}
=c_nc_n^{\dagger}\;
c_n^{\dagger}\Bigl(\partial_a H-\varepsilon_n\,\partial_a S\Bigr)c_n ,
$$

i.e. the vertex reproduces the pencil identity
$\partial_a\varepsilon_n=c_n^{\dagger}(\partial_aH-\varepsilon_n\partial_aS)c_n$
(pinned independently in `TB2J/docs/sympy/local_gradient_nonorthogonal.py`,
check [B]). On the occupied real axis the $-z\,\partial_aS$ term is what makes
band-bottom level shifts physical rather than gauge.

**(R3) The $J$-contraction is basis-invariant in the covariant form.** For
$S>0$, Loewdin orthogonalization $\tilde H=S^{-1/2}HS^{-1/2}$,
$\tilde\Delta=S^{-1/2}\Delta S^{-1/2}$ gives
$\tilde G=S^{1/2}GS^{1/2}$ and hence

$$
\operatorname{Tr}\bigl[\Delta_I G^{\uparrow}\Delta_J G^{\downarrow}\bigr]
=\operatorname{Tr}\bigl[\tilde\Delta_I\tilde G^{\uparrow}
\tilde\Delta_J\tilde G^{\downarrow}\bigr]
$$

by cyclicity of the trace. The covariant LKAG formula used by TB2J's SIESTA
path is therefore the correct non-orthogonal generalization (site-resolved
covariant blocks are the natural LCAO decomposition, validated end-to-end by
the oracle below).

**(R4) First-order spin-lattice parameters.** Differentiating the LKAG
integrand (four terms; `dA/dx = P dG P G + P G P dG` in
`exchange_pert2.get_A_ijR`, plus the two $\partial\Delta$ terms):

$$
\begin{aligned}
\partial_a\operatorname{Tr}\bigl[\Delta_I G^{\uparrow}_{IJ}\Delta_J G^{\downarrow}_{JI}\bigr]
=\operatorname{Tr}\bigl[&
(\partial_a\Delta_I)\,G^{\uparrow}_{IJ}\Delta_J G^{\downarrow}_{JI}
+\Delta_I\,\bigl(G^{\uparrow}V_a^{\uparrow}G^{\uparrow}\bigr)_{IJ}\Delta_J G^{\downarrow}_{JI}\\
&+\Delta_I\,G^{\uparrow}_{IJ}(\partial_a\Delta_J)\,G^{\downarrow}_{JI}
+\Delta_I\,G^{\uparrow}_{IJ}\Delta_J\,\bigl(G^{\downarrow}V_a^{\downarrow}G^{\downarrow}\bigr)_{JI}
\bigr],
\end{aligned}
$$

$$
O_{IJa}=\frac{1}{4\pi}\int d\varepsilon\,\operatorname{Im}\;
\partial_a\operatorname{Tr}\bigl[\cdots\bigr] .
$$

$\partial_a\Delta_I$ is the on-site spin splitting of $\partial_aH$;
it has **no** $S$-term because $\Delta$ is $H$-like.
`exchange_pert2.get_dP_iatom` computes this derivative, but the current
`get_A_ijR` contraction holds $P_i$ and $P_j$ fixed: production must
insert both $\partial_aP$ terms rather than assume that method is used.
In the NCL A-matrix convention $A^{ab}_{ij}=\operatorname{Tr}
[P_iG^a_{ij}P_jG^b_{ji}]/\pi$, the **same** on-site spin-splitting $P$
enters the 0 and 3 channels. The collinear crossed-spin reduction is
proved symbolically (F0) and checked against the full contour (F1).

**(R5) Second-order kernel** (for $Q_{IJ,ab}$ force constants):
$$
\partial_a\partial_b G
=G\bigl(V_aGV_b+V_bGV_a\bigr)G
+G\,V_{ab}^{(2)}(z)\,G,
\qquad
V_{ab}^{(2)}(z)=\partial_a\partial_b H-z\,\partial_a\partial_b S .
$$

The full $Q$ integrand is the product rule over both propagators, both
$\Delta$s, and their mixed derivatives (check E1 assembles all ten terms).

## 3. Physical notes pinned by the oracle

1. **Dropping $-z\,\partial_aS$ is quantitatively wrong** (check D2): the
   orthogonal-basis formula applied verbatim to a non-orthogonal model misses
   the overlap-relaxation channel at first order in $s_0$.
2. **A pure overlap derivative produces spin-lattice coupling by itself**
   (check D4): with $\partial_aH=0$ but $s(\tau_a)=s_0e^{-\lambda\tau_a}$,
   $O_{IJa}\neq0$. The $dS$ channel is not a correction; it is a leading
   physical vertex in LCAO bases.
3. **Orthogonal limit exact** (D3): $s_0=0\Rightarrow\partial S=0$ and the
   vertex degenerates to $g_a=\partial_aH$, reproducing the published EPW
   downfolding.
4. All identities carry the contour with them: because $V_a(z)$ is evaluated
   at each quadrature $z$ (arc points or CFR Matsubara-like poles), the
   existing `Contour`/`CFR` machinery in TB2J needs **no structural change** —
   only the vertex assembly at each energy point.

## 4. Checks and observed values

| Check | Content | Observed |
|-------|---------|----------|
| A1 | $\partial_\lambda G=G(\partial_\lambda H-z\,\partial_\lambda S)G$, exact 2×2 symbolic pencil | exact (sympy zero) |
| A2 | $\partial^2_\lambda G=2GVGVG$ for a linear pencil, exact | exact |
| A3 | $\partial^2_\lambda G=G(VGV+VGV)G+GV_2G$ on a random quadratic pencil, exact | exact |
| B1 | double-pole residue $\Rightarrow$ pencil identity $\partial_a\varepsilon_n$ | rel-err 7.3e-10 |
| C1 | Loewdin invariance of $\operatorname{Tr}[\Delta G^{\uparrow}\Delta G^{\downarrow}]$ (random 6×6) | err 6.0e-15 |
| D1 | $O_{IJa}$: assembled vertex formula == central FD of $J(\tau)$ (2-site model, arc contour) | rel 5.2e-13 |
| D2 | negative control: $-z\,\partial_aS$ dropped | 70% deviation |
| D3 | orthogonal limit ($s_0=0$) == EPW $\partial_aV$-only formula | rel < 1e-8; $dS$-term vanishes |
| D4 | pure-$\partial_aS$ model: $O\neq0$ and formula == FD | $O=-0.0492$, rel 7.3e-13 |
| E1 | $Q=\partial^2J$: assembled 10-term kernel == FD² of $J$ | rel 1.7e-8 |
| F0 | $G^0_{IJ}G^0_{JI}-G^3_{IJ}G^3_{JI}=\tfrac12(G^\uparrow_{IJ}G^\downarrow_{JI}+G^\downarrow_{IJ}G^\uparrow_{JI})$ | exact (sympy zero) |
| F1 | TB2J $P_i(G^0G^0-G^3G^3)P_j/\pi$ == collinear LKAG and derivative via $dH-zdS$ == FD | $J$ difference 0; rel 5.2e-13 |
| F2 | channel derivative without $-z\,dS$ (negative control) | 70% deviation |

(Numeric values in the last column are re-verified by each run of the
script; the run log in `../memnotes/Projects/non-orthogonal-spin-lattice/log/`
records them.)

## 5. Implementation consequences (TB2J/HamiltonIO touchpoints)

1. `green.py::get_GR_and_dGRdx{,1,_and_dGRdx2}` and the EPW kernels
   (`get_dGR`, `get_dGR_vectorized`, `get_GR_and_dGRdx_from_epw*`): replace
   $G\,dH\,G\to G\,V_a(z)\,G$ with $V_a(z)=dH-z\,dS$ at each energy point;
   thread `dSdx` through the same Fourier pipeline as `dHdx` (R-lists must
   coincide).
2. `exchange_pert2.py`: `set_epw` path gains an overlap-derivative source
   alongside `EpmatOneMode`; `get_dP_iatom` stays $H$-block-only (no $dS$).
3. Vertex sources, non-orthogonal: finite differences of displaced-geometry
   $H(\mathbf R),S(\mathbf R)$ from SIESTA/ABACUS via HamiltonIO
   (`LCAOHamiltonian.SR` already exposes $S(\mathbf R)$), i.e.
   $\partial_{\tau_a}H\approx[H(+h)-H(-h)]/2h$ and likewise for $S$; basis
   orbitals follow their atoms (lab-frame blocks change), which is exactly
   the gauge the pencil identity assumes.
4. Gauge remark: the optional `contravariant` dressings
   ($S^{-1}GS^{-1}$, `TB2J_SIESTA_GREEN_MODE`) transform the vertex
   covariantly; this derivation pins the **covariant/default** mode used by
   the SIESTA $J$ path. Do not mix gauges between $G$ and $V_a$.
5. Per `TB2J/AGENTS.md`, when this math is implemented in TB2J the script
   and this report must be mirrored under `TB2J/docs/sympy/`.
