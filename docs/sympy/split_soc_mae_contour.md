# KS-band SOC: second-order contour factor

`split_soc_mae_contour.py` asserts the result with SymPy and checks the CFR
implementation numerically in the `mydev` environment.

For the two-level Hamiltonian `H(lambda)=[[mu-1, lambda*w],
[lambda*w, mu+1]]`, SymPy verifies that the lower eigenvalue
`mu - sqrt(1 + lambda**2*w**2)` has a lambda-squared coefficient
`-w**2/2` independent of the Fermi reference. Use the resolvent
`G0(z) = (z + mu - H0)**-1`; the corresponding band-space diagnostic is

$$E^{(2)}=-\frac{1}{2\pi}\operatorname{Im}\int_{\mathcal C}
\operatorname{Tr}[(G_0(z) W)^2]\,dz.$$

The script evaluates the same continued-fraction contour as TB2J for
`w=0.1` eV, `E=(mu-1,mu+1)` eV with `mu=0` and `mu=6.3` eV, and
smearing 0.05 eV. At 12/30 poles both gauges reproduce -0.005 eV
within 1e-7/1e-9 eV. GPAW's exact
second-variational band energy uses each direction's own occupations and
Fermi level; this perturbative trace can differ through higher SOC orders,
window truncation, occupation/Fermi-surface effects, and contour convergence.
The driver reports rather than hides this residual.
