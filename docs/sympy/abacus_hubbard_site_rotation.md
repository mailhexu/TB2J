# ABACUS native Hubbard projector: finite site-rotation checks

Run `python docs/sympy/abacus_hubbard_site_rotation.py` with assertions enabled in the `mydev` environment. The paired Python script is the exact-arithmetic authority. Its checks apply to finite Hilbert spaces and a **prescribed** spin rotation; they are not a self-consistent noncollinear DFT+U derivation or a periodic-crystal exchange derivation.

## Definitions and assumptions

- The columns of `phi` are linearly independent, nonorthogonal export functions. Their Hermitian Gram is nonsingular. The complete two-dimensional counterexample spans its ambient space; in a truncated space, equality with the full physical trace is not guaranteed.
- `x` is the cross-overlap of the export basis with the functions of one native separable correlated projector. Its entries may be complex. `d` is the corresponding finite site-operator coefficient matrix; values must be finite and use a consistent energy unit. Overlaps and spin rotations are dimensionless. Native U coefficients are in Ry and require the established Ry-to-eV conversion at serialization.
- Spin is the final tensor index. The prescribed rotation is right-handed about the y axis with spinor convention `exp(-i theta sigma_y/2)`. Its derivative is evaluated at zero. This is a fixed co-rotation model, not proof that ABACUS has performed or supports the associated self-consistent spin response.
- There are no hidden assumptions that orbitals on distinct atoms are orthogonal or that the native correlated radial function equals the UPF atomic wavefunction. Bloch phases, k-point sampling, Fourier translations and the representation of a locally rotated nonperiodic perturbation require additional proof.

## Assertions

The script verifies the generic two-channel separable projection in exact symbolic arithmetic. It independently expands every finite sum and compares the result with the matrix product. It then builds an exactly normalized, full-rank two-site basis with nonzero imaginary intersite overlap, applies a rank-one first-site operator, and checks:

1. Its covariant projected vertex contains cross-site and second-site elements, although its native coefficient is associated with one atom.
2. The exact infinitesimal rotation of the prescribed spin-dependent site operator retains that full covariant structure.
3. For spin-dependent resolvents of the corresponding two-level Hamiltonian, the exact full-space and full-covariant inverse-metric traces both equal `1/2`, while keeping only the first-site covariant block gives `7/9`.
4. For the native collinear scalar potential `U*(I/2 - n_spin.T)` in `dftu_pw_tools.cpp:65-85,235-249`, the difference is `U*(n_down - n_up).T`; equal occupations give zero and swapping the channels reverses its sign. This does not establish a self-consistent transverse spin response.
5. The full-vertex and factorized traces agree for generic complex entries in two-dimensional ambient and local spaces, and for a separate example with two distinct rectangular complex vertex factors and nontrivial Green matrices. Zeroing one site operator removes its contribution; doubling it doubles that finite trace. These are finite checks, not a proof for arbitrary dimensions or a Fourier-periodic algorithm.
6. An exact two-cell periodic toy with an off-cell hopping demonstrates that a localized site rotation couples distinct k points. Direct ambient, full covariant inverse-metric and explicit double-k block traces agree; retaining only equal-k vertex blocks does not. This checks that finite toy's convention, not the general periodic Fourier normalization or arbitrary-k implementation.

The comparison is a counterexample to naïvely discarding projected cross-site terms; it does not disprove an exactly equivalent factorized formulation. It also does not establish that the existing SIESTA PAW/FLL local-rotation contract can be transplanted to the ABACUS Dudarev implementation.

## Native connection and outstanding proof

ABACUS source `source/source_pw/module_pwdft/op_pw_proj.cpp:204-307` selects correlated angular channels and applies spin-specific native onsite coefficients through projector overlaps; `source/source_pw/module_pwdft/kernels/onsite_op.cpp:57-131` performs the block action. Story001 includes a separately compiled exercise of that CPU kernel. The present finite check has not compared a complete native plane-wave Hamiltonian, converged/mixed U state, SCF spin rotation, Fourier-transformed multisite vertices or physical ABINIT reference. Those remain proof gates before enabling an exchange-ready exporter.
