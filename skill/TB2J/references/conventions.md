# Heisenberg Hamiltonian Conventions

TB2J uses a specific convention for the Heisenberg Hamiltonian. It is crucial to check this when comparing results with other codes or experiments.

## The Hamiltonian

The full Hamiltonian calculated by TB2J includes Isotropic Exchange ($J$), Anisotropic Exchange ($J^{ani}$), Dzyaloshinskii-Moriya Interaction (DMI, $D$), and Single-Ion Anisotropy ($K$).

$$
E = -\sum_i K_i (\vec{S}_i \cdot \vec{e}_i)^2 
    -\sum_{i \neq j} \left[ J_{ij} \vec{S}_i \cdot \vec{S}_j 
    + \vec{S}_i \mathbf{J}^{ani}_{ij} \vec{S}_j 
    + \vec{D}_{ij} \cdot (\vec{S}_i \times \vec{S}_j) \right]
$$

### Key Conventions

1.  **Sign Convention**: A **minus sign** is used in the exchange terms.
    *   **Positive $J_{ij} > 0$**: **Ferromagnetic** interaction (favors parallel alignment).
    *   **Negative $J_{ij} < 0$**: **Antiferromagnetic** interaction (favors anti-parallel alignment).

2.  **Spin Normalization**:
    *   The spin vectors $\vec{S}_i$ are **normalized to 1** ($|\vec{S}_i| = 1$).
    *   The parameters ($J, D, K$) are in units of energy (typically eV or meV as output).

3.  **Summation (Double Counting)**:
    *   The sum runs over all pairs $i \neq j$.
    *   Both terms $J_{ij}$ and $J_{ji}$ are included in the Hamiltonian.
    *   This means the total energy of a bond is effectively $2 J_{ij} \vec{S}_i \cdot \vec{S}_j$ if $J_{ij} = J_{ji}$.

## Comparison with Other Codes

Different codes use different conventions. Common variations include:

*   **Factor of 1/2**: Some codes write $-\frac{1}{2} \sum_{i \neq j} J_{ij} \vec{S}_i \cdot \vec{S}_j$. If converting from TB2J to such a convention, you might need to multiply/divide by 2.
*   **Summation over $i < j$**: Some codes sum over unique pairs only ($-\sum_{i < j} ...$). Since TB2J sums over all $i \neq j$, the energy is counted twice per pair compared to the $i < j$ convention (assuming symmetric $J$).
*   **Spin Magnitude**: Some codes use non-normalized spins (e.g., $|\vec{S}_i| \approx \mu_B$). In TB2J, the magnitude of the magnetic moment is absorbed into the interaction parameter $J$. To convert to a convention using unnormalized spins, you generally divide the TB2J $J$ parameter by $|\vec{M}_i||\vec{M}_j|$.

## Mapping to Output Files

*   **`exchange.out`**: Uses the TB2J convention described above.
*   **`Multibinit` / `Vampire` outputs**: TB2J automatically attempts to convert parameters to the input formats required by these codes, handling factors of 2 or normalization differences where possible. However, always verify the header of the generated files.
