# Gauge-invariant degenerate-subspace comparison (Story-008)

Authoritative script: `mode_qualification_projectors.py` (run with the project
environment; all assertions must pass).

## Claim

For a cluster of exactly degenerate eigenvalues with orthonormal eigenvector
block $X$, any re-diagonalization of the same operator returns $X' = XW$ with
$W$ unitary within the cluster. The orthogonal projector

$$P = X X^{\dagger}$$

is invariant: $P' = XW (XW)^{\dagger} = X X^{\dagger}$, since $WW^{\dagger} =
I$. Comparing projectors between repeated diagonalizations is therefore a
gauge-invariant determinism witness; comparing individual eigenvectors is
not. Mixing with a vector outside the cluster changes $P$ by $O(\epsilon^2)$
in the perturbation $\epsilon$, so genuine degeneracy-breaking is flagged.

## Production use

`TB2J/magnon_phonon/qualification.py::degenerate_subspace_determinism` clusters
energies with an absolute tolerance, matches clusters across runs by mean
energy and multiplicity, and compares cluster projectors within
`projector_tol`. Magnon side uses the positive-norm columns of the Story-006
paraunitary; phonon side reconstructs the dynamical matrix from the recorded
Story-007 eigensystem and re-diagonalizes it.

## Numerical witness

The script also re-checks the claim numerically with `numpy.linalg.eigh` on a
random Hermitian matrix with an exact degenerate pair, rotating the cluster by
a random unitary and verifying the projector residual is below `1e-12`, and
that $P^2 = P$ to the same tolerance.
