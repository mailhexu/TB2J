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
\mathcal{H} = -\sum_{i,j,\mathbf{R}} \mathbf{S}_i \cdot \mathbf{J}_{ij}(\mathbf{R}) \cdot \mathbf{S}_j(\mathbf{R})
$$

where:
- $i, j$ index magnetic atoms in the unit cell
- $\mathbf{R}$ is the lattice vector connecting unit cells
- $\mathbf{S}_i$ is the spin operator at site $i$
- $\mathbf{J}_{ij}(\mathbf{R})$ is the $3\times3$ exchange tensor

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

### Exchange in q-Space

The real-space exchange tensors are transformed to reciprocal space:

$$
\mathbf{J}_{ij}(\mathbf{q}) = \sum_{\mathbf{R}} \mathbf{J}_{ij}(\mathbf{R}) e^{i\mathbf{q}\cdot\mathbf{R}}
$$

This is implemented in `Magnon.Jq()` (see `TB2J/magnon/magnon3.py`):

```python
for iR, R in enumerate(Rlist):
    for iqpt, qpt in enumerate(kpoints):
        phase = 2 * np.pi * R @ qpt
        Jq[iqpt] += np.exp(1j * phase) * JRprime[iR]
```

### Non-collinear Magnetic Structures

For helimagnetic or spin-spiral systems with propagation vector $\mathbf{Q}$, each exchange tensor is rotated before the Fourier transform:

$$
\mathbf{J}'_{mn}(\mathbf{R}) = \mathbf{R}_m(\phi)^T \mathbf{J}_{mn}(\mathbf{R}) \mathbf{R}_n(\phi)
$$

where $\phi = 2\pi \mathbf{R}\cdot\mathbf{Q}$ and $\mathbf{R}(\phi)$ is the rotation matrix.

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

The matrix elements are computed from the exchange tensors:

$$
A_{ij}(\mathbf{q}) = \sqrt{S_i S_j} \left[ \sum_k V_k^T \mathbf{J}_{ik}(\mathbf{0}) V_k S_k \delta_{ij} - U_i^* \mathbf{J}_{ij}(\mathbf{q}) U_j \right]
$$

$$
B_{ij}(\mathbf{q}) = -\sqrt{S_i S_j} \, U_i \mathbf{J}_{ij}(\mathbf{q}) U_j
$$

This is implemented in `Magnon.Hq()`:

```python
A1 = np.einsum("ix,kijxy,jy->kij", U, Jq, U.conj())
B = np.einsum("ix,kijxy,jy->kij", U, Jq, U)
C = np.diag(np.einsum("ix,ijxy,jy,j->i", V, 2*J0, V, self.Snorm))
H = np.block([[A1 - C, B], [B.swapaxes(-1,-2).conj(), A2 - C]])
```

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

### Goldstone Mode

For ferromagnetic systems, the magnon energy at $\mathbf{q}=0$ should be zero (Goldstone mode). Deviations indicate:

1. Insufficient exchange parameter convergence
2. Incorrect magnetic ground state
3. Missing long-range interactions

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
