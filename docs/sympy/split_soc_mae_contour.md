# KS-band SOC: second-order contour factor

`split_soc_mae_contour.py` asserts the result with SymPy and checks the CFR
implementation numerically in the `mydev` environment.

For $H(\lambda)=\begin{pmatrix}-1&\lambda w\\\lambda w&1\end{pmatrix}$,
SymPy verifies the lower eigenvalue $-\sqrt{1+\lambda^2 w^2}$ has a
$\lambda^2$ coefficient $-w^2/2$. For $G_0(z)=(z-H_0)^{-1}$, the corresponding
band-space diagnostic is

$$E^{(2)}=-\frac{1}{2\pi}\operatorname{Im}\int_{\mathcal C}
\operatorname{Tr}[(G_0(z) W)^2]\,dz.$$

The script evaluates the same continued-fraction contour as TB2J for
$w=0.1$ eV, $E=(-1,1)$ eV, smearing $0.05$ eV. At 12/30 poles it
reproduces $-0.005$ eV within $10^{-7}/10^{-9}$ eV. GPAW's exact
second-variational band energy uses each direction's own occupations and
Fermi level; this perturbative trace can differ through higher SOC orders,
window truncation, occupation/Fermi-surface effects, and contour convergence.
The driver reports rather than hides this residual.
