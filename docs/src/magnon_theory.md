# Magnon Band Structure and Density of States: Theory

```{note}
For usage instructions and command-line interface, see [Magnon Band Structure](magnon_band.rst).
```

This document describes the theoretical background for calculating magnon band structures and density of states (DOS) from exchange parameters computed by TB2J.

## Introduction

Magnons are quantized spin waves in magnetic materials. Understanding magnon dispersions is essential for:

- Predicting thermodynamic properties (magnetization, specific heat)
- Analyzing magnetic stability and phase transitions
- Designing spintronic devices
- Interpreting neutron scattering experiments

TB2J calculates magnon properties from first-principles-derived exchange parameters using linear spin wave theory (LSWT).

## Heisenberg Model

### General Spin Hamiltonian

The magnetic interactions in a crystal are described by a generalized Heisenberg Hamiltonian:

$$
\mathcal{H} = -\sum_{\langle ij\rangle} \mathbf{S}_i \cdot \mathbf{J}_{ij}(\mathbf{R}) \cdot \mathbf{S}_j(\mathbf{R})
$$

where:
- $\langle ij\rangle$ runs over **unordered** magnetic pairs once (each bond is
  counted a single time; the "+$\mathbf{R}$" and the "-$\mathbf{R}$" entries of a
  directed exchange list are two different bonds)
- $i, j$ index magnetic atoms in the unit cell
- $\mathbf{R}$ is the lattice vector connecting unit cells
- $\mathbf{S}_i$ is the spin operator at site $i$, with $|\mathbf{S}_i| = S_i$
  (the physical spin, $S_i = |\mathbf{m}_i|/2$ for a moment $\mathbf{m}_i$ given in
  $\mu_B$)
- $\mathbf{J}_{ij}(\mathbf{R})$ is the $3\times3$ exchange tensor, in meV, as
  coupling the *physical* spins $S_i$

```{note}
The double sum $\sum_{i,j,\mathbf{R}}$ that appears in some references (for
example Toth & Lake, Eq. 1) counts every bond twice. The operative convention
of the reference — and the one used by TB2J — is one coupling per bond.
```

```{note}
$\mathbf{J}_{ij}$ is the exchange of the spins whose moments are given to the
magnon calculation. If the moments are changed with respect to the ones used
in the exchange calculation, the exchange has to be rescaled with the DFT
rule $J \propto 1/m^2$: $J(m') = J(m)\,(m/m')^2$.
```

### Exchange Tensor Components

The exchange tensor can be decomposed into:

$$
\mathbf{J}_{ij} = J_{ij}^{\text{iso}} \mathbf{I} + \mathbf{J}_{ij}^{\text{sym}} + \mathbf{D}_{ij} \times
$$

where:
- $J_{ij}^{\text{iso}}$: Isotropic (Heisenberg) exchange
- $\mathbf{J}_{ij}^{\text{sym}}$: Symmetric anisotropic exchange (traceless)
- $\mathbf{D}_{ij}$: Dzyaloshinskii-Moriya interaction (DMI) vector

In component form:

$$
J_{ij}^{\alpha\beta} = J_{ij}^{\text{iso}}\delta_{\alpha\beta} + J_{ij}^{\text{sym},\alpha\beta} + \epsilon_{\alpha\beta\gamma} D_{ij}^\gamma
$$

### Single-Ion Anisotropy

For on-site terms ($i=j$, $\mathbf{R}=0$), single-ion anisotropy (SIA) contributes:

$$
\mathcal{H}_{\text{SIA}} = -\sum_i \mathbf{S}_i \cdot \mathbf{A}_i \cdot \mathbf{S}_i
$$

where $\mathbf{A}_i$ is the anisotropy tensor.

## Fourier Transform to Reciprocal Space

### Exchange at the Magnon Wave Vector

For a magnon Bloch wave vector $\mathbf{k}$, real-space exchange tensors are
transformed as:

$$
\mathbf{J}_{ij}(\mathbf{k}) =
\sum_{\mathbf{R}} \mathbf{J}_{ij}(\mathbf{R})
e^{-2\pi i\mathbf{k}\cdot\mathbf{R}}
$$

`Magnon.Jq()` implements this negative Fourier-phase convention:

```python
for iR, R in enumerate(Rlist):
    for ikpt, kpt in enumerate(kpoints):
        phase = 2 * np.pi * R @ kpt
        Jq[ikpt] += np.exp(-1j * phase) * JRprime[iR]
```

### Single-Q Incommensurate Reference Structures

The known magnetic ordering/propagation vector $\mathbf{Q}$ defines the
reference (normally ground-state) structure; it is distinct from the sampled
magnon vector $\mathbf{k}$. For each exchange displacement $\mathbf{R}$:

$$
\phi = 2\pi\mathbf{R}\cdot\mathbf{Q},
\qquad
\mathbf{J}'_{mn}(\mathbf{R}) =
\mathbf{J}_{mn}(\mathbf{R})\mathcal{R}_{\mathbf{n}}(\phi)
$$

where $\mathbf{n}$ is the global spiral rotation axis. TB2J then Fourier
transforms $\mathbf{J}'(\mathbf{R})$ at each independent $\mathbf{k}$ to build
the existing $A(\mathbf{k})$, $B(\mathbf{k})$, and $C$ matrices. TB2J accepts a
known $\mathbf{Q}$; it does not determine the ground-state ordering vector.

```{figure} magnon_band.assets/afm_zone_folding.png
:alt: Alternating G-type antiferromagnetic chemical cells and directional folding between chemical Gamma/R and magnetic Gamma.
:width: 100%
:align: center

For $\mathbf{Q}=(\tfrac12,\tfrac12,\tfrac12)$, alternate chemical cells differ
by a $\pi$ rotation. **Folding** maps chemical-cell $\Gamma$ and R, which differ
by $\mathbf{Q}$, to the same magnetic-zone $\Gamma$. Conversely, **unfolding**
that magnetic-$\Gamma$ mode gives chemical-zone representatives at both
$\Gamma$ and R$=\mathbf{Q}$. Thus
$\omega(\Gamma)=\omega(\mathrm{R})$ when the combined translation–spin-rotation
symmetry is present, although their spectral intensities can differ.
```

```{figure} magnon_band.assets/afm_magnon_unfolding.png
:alt: A magnetic-Gamma antiferromagnetic magnon whose uniform transverse x component is at chemical Gamma and staggered transverse y component is at chemical R.
:width: 100%
:align: center

This is one magnetic-$\Gamma$ normal mode, drawn in two phases separated by a
quarter period. With the B sublattice related to A by
$\mathcal{R}_{\hat{x}}(\pi)$, its transverse components satisfy
$\delta S_A^x=\delta S_B^x$ and $\delta S_A^y=-\delta S_B^y$. The first is a
chemical-$\Gamma$ Fourier component and the second is a chemical-R component;
they are quadratures of the same mode and therefore have the same frequency
$\omega$.
```

## Magnon Hamiltonian Construction

### Local Reference Frame

For each magnetic atom, we define a local coordinate system where the $z'$-axis aligns with the spin direction. The rotation from global to local frame is characterized by:

$$
U_i = R_{i,x'} + iR_{i,y'}, \quad V_i = R_{i,z'}
$$

where $\mathbf{R}_i$ is the rotation matrix for site $i$.

### Holstein-Primakoff Transformation

In linear spin wave theory, spin operators are expressed in terms of bosonic creation/annihilation operators:

$$
S_i^+ \approx \sqrt{2S_i} a_i, \quad S_i^- \approx \sqrt{2S_i} a_i^\dagger, \quad S_i^z = S_i - a_i^\dagger a_i
$$

### Dynamical Matrix

The magnon Hamiltonian in the local frame takes the form:

$$
\mathcal{H} = \frac{1}{2}\sum_{\mathbf{q}} \begin{pmatrix} \mathbf{a}_\mathbf{q}^\dagger & \mathbf{a}_{-\mathbf{q}} \end{pmatrix} \mathbf{H}(\mathbf{q}) \begin{pmatrix} \mathbf{a}_\mathbf{q} \\ \mathbf{a}_{-\mathbf{q}}^\dagger \end{pmatrix}
$$

where $\mathbf{a}_\mathbf{q}$ is a vector of bosonic operators for all $N$ magnetic atoms.

The dynamical matrix has the block structure:

$$
\mathbf{H}(\mathbf{q}) = \begin{pmatrix} \mathbf{A}(\mathbf{q}) & \mathbf{B}(\mathbf{q}) \\ \mathbf{B}^*(\mathbf{-q}) & \mathbf{A}^*(\mathbf{-q}) \end{pmatrix}
$$

### Matrix Elements

The matrix elements are computed from the exchange tensors. With the
**unnormalised** transverse frame $\mathbf{U}_i = \mathbf{e}_{1i} + i\,\mathbf{e}_{2i}$
($|\mathbf{U}_i|^2 = 2$), the elements carry a factor $1/2$ *inside* $A$ and
$B$ (this is the same $1/2$ that the quadratic form above carries in front of
the $\mathbf{q}$-sum; moving it between the two places is an identity):

$$
A_{ij}(\mathbf{q}) = \frac{\sqrt{S_i S_j}}{2}\; \mathbf{U}_i^T \mathbf{J}_{ij}(-\mathbf{q}) \overline{\mathbf{U}}_j
$$

$$
B_{ij}(\mathbf{q}) = \frac{\sqrt{S_i S_j}}{2}\; \mathbf{U}_i^T \mathbf{J}_{ij}(-\mathbf{q}) \mathbf{U}_j
$$

$$
C_{ij} = \delta_{ij} \sum_l S_l\, \mathbf{V}_i^T \mathbf{J}_{il}(\mathbf{0}) \mathbf{V}_l
$$

This is the construction of Toth & Lake (2015), Eq. (26) (with their
$\mathbf{u}\to\mathbf{U}$, $\mathbf{v}\to\mathbf{V}$), and it is implemented in
`Magnon.Hq()` as:

```python
# U = e1 + i e2 is not unit-normalised (|U|^2 = 2): the 1/2 below is the
# one printed inside A(q) and B(q) of Toth & Lake (2015), Eq. (26).
A1 = 0.5 * np.einsum("ix,kijxy,jy->kij", U, Jmq, U.conj())
A2 = 0.5 * np.einsum("ix,kijxy,jy->kij", U.conj(), Jq.conj(), U)
B = 0.5 * np.einsum("ix,kijxy,jy->kij", U, Jmq, U)
# the mean field of the reference state is sum_j J_ij M_j, i.e. Jq(0) once
C = np.diag(np.einsum("ix,ijxy,jy,j->i", V, J0, V, self.Snorm))

# the sqrt(S_i S_j) of Eq. (26) is applied here
A1 = np.einsum("kij, i, j-> kij", A1, Ssqrt, Ssqrt)
A2 = np.einsum("kij, i, j-> kij", A2, Ssqrt, Ssqrt)
B = np.einsum("kij, i, j-> kij", B, Ssqrt, Ssqrt)

H = np.block([[A1 - C, B], [B.swapaxes(-1,-2).conj(), A2 - C]])
```

Here `Jq` is the Fourier transform of the real-space exchange tensors (no
$1/(S_iS_j)$ rescale: the spin-length factors enter only through
$\sqrt{S_iS_j}$), `Jmq` is the same tensor at $-\mathbf{q}$ with transposed
sublattice indices, and `J0 = -Jq(0)`. The module's overall minus sign
(`Jq_code = -Jq`) is a sign convention of the code and does not change the
frequencies.

With these conventions the frequencies are the textbook values of linear spin
wave theory, for example

$$
\omega(k) = 2 J S (1 - \cos ka) \quad \text{(ferromagnetic chain)}, \qquad
\omega_{\pm}(k) = 3 J S \left(1 \mp |\gamma(k)|\right) \quad \text{(ferromagnetic honeycomb)},
$$

with $J$ in meV and $S$ the physical spin.

## Band Structure Calculation

### Bogoliubov Transformation

The magnon eigenfrequencies are obtained by diagonalizing the Hamiltonian with the bosonic metric:

$$
\mathbf{g} = \begin{pmatrix} \mathbf{I}_N & 0 \\ 0 & -\mathbf{I}_N \end{pmatrix}
$$

Using Cholesky decomposition $\mathbf{H} = \mathbf{K}^\dagger \mathbf{K}$, we solve:

$$
\mathbf{K}^\dagger \mathbf{g} \mathbf{K} \mathbf{v} = \omega \mathbf{v}
$$

The positive eigenvalues give the magnon energies:

```python
K = np.linalg.cholesky(H)
g = np.block([[1*I, 0*I], [0*I, -1*I]])
energies = np.linalg.eigvalsh(K.T.conj() @ g @ K)[:, n:]
```

```{warning}
The Cholesky route is only accurate to about $10^{-6}$ and it needs a positive
definite $\mathbf{H}$. At a Goldstone point $\mathbf{H}$ is singular, and the
shifted fallback (`H - (min_eig - 1e-6) I`) can return negative frequencies for
ferromagnetic references. For exact frequencies, and for any test that pins a
band value, use the exact Colpa route instead: the eigenvalues of
$\mathbf{g}\,\mathbf{H}$, whose positive half are the magnon energies,

```python
g = np.block([[1*I, 0*I], [0*I, -1*I]])
energies = np.sort(np.linalg.eigvals(g @ H).real)[n:]
```
```

### Goldstone Mode

For ferromagnetic systems, the magnon energy at $\mathbf{q}=0$ should be zero (Goldstone mode). Deviations indicate:

1. Insufficient exchange parameter convergence
2. Incorrect magnetic ground state
3. Missing long-range interactions
4. The Cholesky artefact described above (use the exact $\mathbf{g}\,\mathbf{H}$ route)

### k-Path Selection

Band structures are calculated along high-symmetry paths in the Brillouin zone. TB2J supports:

1. **Automatic path detection**: Based on crystal symmetry (default)
2. **Manual specification**: Using string notation (e.g., "GXMR")

```python
# Automatic
xlist, kptlist, Xs, knames, spk = auto_kpath(cell, None, npoints=300)

# Manual
bandpath = cell.bandpath(path="GXMR", npoints=300)
```

## Spin-Wave Stiffness

### Definition

The spin-wave stiffness is the leading coefficient of the long-wavelength
expansion of the acoustic magnon branch around its Goldstone (or gap) point:

$$
\omega(\mathbf{k}) = \omega_0 + \mathbf{k}^{T} A \mathbf{k}
\qquad\text{(quadratic branch)}
$$

$$
\omega(\mathbf{k}) = \omega_0 + \mathbf{v}\cdot\mathbf{k}
\qquad\text{(linear branch)}
$$

$A$ is a cartesian tensor in meV Å² and $\mathbf{v}$ collects the directional
spin-wave speeds in meV Å. A ferromagnet has a quadratic branch
($\omega_0 = 0$ at $\Gamma$), an antiferromagnet a linear one
($\omega_0 = 0$ at the ordering vector $\mathbf{Q}$). The stiffness is
extracted from the *exact* $\mathbf{g}\,\mathbf{H}$ spectrum — the shifted
Cholesky route is not accurate enough at a Goldstone point.

### Usage

```python
from TB2J.magnon.magnon3 import Magnon

magnon = Magnon.from_TB2J_results(path="TB2J_results")
stiffness = magnon.get_spin_stiffness()          # expansion at Gamma
print(stiffness)

# antiferromagnet: expand at the ordering vector
stiffness = magnon.get_spin_stiffness(k0=[0.5, 0.0, 0.0])
```

From the command line:

```bash
TB2J_magnon.py --stiffness -p TB2J_results
TB2J_magnon.py --stiffness --stiffness-k0 0.5 0 0 --stiffness-step 0.005 -p TB2J_results
```

`--stiffness-step` is the finite-difference step in cartesian 1/Å
(default 0.01); `--stiffness-k0` is the expansion point in reciprocal lattice
coordinates (default $\Gamma$).

### Sum rule and its limits

The textbook route is the real-space sum rule
(Liechtenstein–Katsnelson–Gubanov; Pajda et al., PRB **64**, 174402 (2001)):

$$
A = -\frac{S}{2} \sum_{\text{bonds}} J\,\mathbf{v}\,\mathbf{v}^{T}
$$

where $\mathbf{v}$ are the *true* bond vectors (lattice offset plus intra-cell
displacement). It is available as a cross-check
(`stiffness_sum_rule`, or `get_spin_stiffness(J_iso=..., bond_vectors=...)`)
and reproduces the dispersion route for a single-sublattice ferromagnet. It
does **not** generalise: for a multi-sublattice lattice the exact LSWT band is
not the bond sum (for the honeycomb the sum rule gives
$\mathrm{diag}(|JS|/2, |JS|)$, while the exact stiffness is
$|JS|/4$ in both directions).

The published spin-wave stiffness constants of itinerant magnets
($\mathrm{meV\,Å^2}$ for Fe/Co/Ni) carry additional $\mu_B/M$ normalisation
factors of the itinerant model; TB2J reports the stiffness of the mapped
Heisenberg model, $H = -\sum_{\langle ij\rangle} J\,\mathbf{S}_i\!\cdot\!\mathbf{S}_j$
with $J$ in meV and $|\mathbf{S}| = |\mathbf{m}|/2$.

## Density of States Calculation

### Definition

The magnon DOS is defined as:

$$
g(\omega) = \frac{1}{N_k} \sum_{\mathbf{k},n} \delta(\omega - \omega_{n\mathbf{k}})
$$

where $\omega_{n\mathbf{k}}$ is the energy of magnon band $n$ at wavevector $\mathbf{k}$.

### Gaussian Smearing

In practice, the delta function is approximated by a Gaussian:

$$
g(\omega) = \frac{1}{N_k} \sum_{\mathbf{k},n} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(\omega - \omega_{n\mathbf{k}})^2}{2\sigma^2}\right)
$$

where $\sigma$ is the smearing width (typically 0.001 eV ≈ 0.08 meV).

### k-Mesh Sampling

DOS calculations require dense k-mesh sampling:

```python
kpts = monkhorst_pack([20, 20, 20], gamma_center=True)
```

The gamma-centered mesh ensures better convergence for magnetic systems.

### Implementation

```python
from TB2J.magnon.magnon_dos import MagnonDOSCalculator

calculator = MagnonDOSCalculator(magnon)
calculator.set_kmesh(kmesh=[20, 20, 20], gamma=True)
dos = calculator.get_dos(width=0.001, window=None, npts=401)
```

## Energy Units

TB2J uses consistent energy units:

| Quantity | Internal Unit | Output Unit |
|----------|---------------|-------------|
| Exchange parameters | eV | meV |
| Magnon energies | eV | meV |
| DOS energy axis | eV | meV |
| DOS values | states/eV | states/meV |

The conversion factors are applied during plotting:

```python
energies_meV = energies_eV * 1000
dos_states_per_meV = dos_states_per_eV / 1000
```

## Validation and Quality Checks

### Positive Definiteness

The Hamiltonian should be positive semi-definite at all q-points. If not:

```python
try:
    K = np.linalg.cholesky(H)
except np.linalg.LinAlgError:
    min_eig = np.min(np.linalg.eigvalsh(H))
    warn(f"WARNING: The system may be far from the magnetic ground-state. "
         f"Minimum eigenvalue: {min_eig}")
```

### Stability Indicators

1. **Zero Goldstone mode**: Expected for ferromagnets
2. **No negative energies**: Indicates ground state stability
3. **Smooth band dispersion**: Validates k-path continuity

## Spin Quantization

In linear spin wave theory, the spin quantum number $S = n/2$ ($n = 1, 2, 3, ...$). 
TB2J uses magnetic moment values in units of $\mu_B$ (Bohr magneton). The 
input moment corresponds to $2S\,\mu_B$, so $S=3/2$ requires $3\,\mu_B$.

DFT calculations typically give moments that deviate slightly from these 
values (e.g., $\sim 3.1\,\mu_B$ instead of $3\,\mu_B$ for Cr³⁺). While the DFT 
value is accurate for exchange calculations, using $2S\,\mu_B$ values in 
magnon calculations ensures internal consistency with the bosonic 
Holstein-Primakoff transformation.

You can specify custom moments via `--spin-conf` or TOML `spin_conf` 
(see [Magnon Band Structure](magnon_band.rst) for usage).

## References

1. Toth, S. & Lake, B. Linear spin wave theory for single-Q incommensurate magnetic structures. *J. Phys.: Condens. Matter* **27**, 166002 (2015).

2. Colpa, J. H. P. Diagonalizing the quadratic boson Hamiltonian with a generalized Bogoliubov transformation. *Physica A* **134**, 417-442 (1986).

3. Mankovsky, S. & Ebert, H. Magnetic exchange interactions and magnon dispersion in multilayer systems from first principles. *Phys. Rev. B* **94**, 144424 (2016).

## See Also

- [Magnon Band Structure Usage](magnon_band.rst) - Command-line interface and examples
- [Exchange Parameters](parameters.rst) - Input parameters from TB2J calculations
- [Tutorial](tutorial.rst) - Step-by-step guide for magnon calculations
