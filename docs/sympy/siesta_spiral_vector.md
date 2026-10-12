# SIESTA spiral vector and orbital-offset conventions

Executable authority: [siesta_spiral_vector.py](siesta_spiral_vector.py),
complementing [siesta_spiral_pencil.py](siesta_spiral_pencil.py).

For a direct cell $A$ whose rows are lattice vectors, the script checks on
an exact skew cell (not only the orthogonal producer fixture) that

$$
B = 2\pi A^{-T},\qquad AB^T = 2\pi I.
$$

With integer cell image $R$ and fractional orbital centers $\tau_i,\tau_j$,

$$
x_{ij} = (R+\tau_j-\tau_i)A,\qquad
q_{\rm cart} = q_{\rm frac}B,
$$

and the exact assertion verifies

$$
q_{\rm cart}\cdot x_{ij}
= 2\pi q_{\rm frac}\cdot(R+\tau_j-\tau_i).
$$

The partner displacement is exactly $-x_{ij}$. At $q=0$, intra-cell Bloch
phases are a unitary basis gauge; a collinear spectrum comparison therefore
cannot establish whether the importer retained orbital positions. For a
spiral the $q/2$ sector phases require the full displacement.

Let $h_{12}=H_3-iH_4$. The source-exact assembly is adjoint-paired when the
stored partner satisfies

$$
h_{12}(j,i,-R) = e^{-iq\cdot x_{ij}}h_{12}(i,j,R).
$$

SymPy asserts the full off-diagonal block adjoint equality with these
partners. The script also checks all 2,607 stored Fe-fixture entries:
maximum mixing-partner defect **8.882e-15 eV**. This is producer evidence,
not an assumption inferred from the phase placement.

Run: `python docs/sympy/siesta_spiral_vector.py` in the `mydev` environment.
