# Finite-q nonorthogonal spin-lattice contraction

Assertion script: `q_phased_exchange_contraction.py`.

Canonical counterpart: [nonorth-spin-lattice q-phased exchange mathdoc](../../../memnotes/Projects/non-orthogonal-spin-lattice/mathdocs/q-phased-exchange-contraction.md). Producer phase authority: [two-index q transform](../../../memnotes/Projects/non-orthogonal-spin-lattice/mathdocs/q-transform-phase.md). These links resolve from the canonical research-workspace checkout `non-orthogonal-spin-lattice/TB2J`, where the shared `memnotes` vault is a sibling of the repository; in other checkouts use the vault path `Projects/non-orthogonal-spin-lattice/mathdocs/`.

Convention: $H(k)=\sum_R H(0,R)e^{2\pi ik\cdot R}$, displacement $e_a e^{-2\pi iq\cdot L}$, bra $k$, ket $k+q$. All bands and a fixed energy contour are required. Finite supercells certify commensurate q only.

In the fixed reference gauge,

$$
V(k,q;z)=\delta H(k,q)-(E_F^{\rm ref}+z)\delta S(k,q),\qquad \delta G=G(k)V G(k+q).
$$

The forward bond from 0 to $R$ uses $e^{-2\pi i(k+q)\cdot R}$; its reverse uses $e^{2\pi ik\cdot R}$. The second endpoint on-site splitting derivative carries $e^{-2\pi iq\cdot R}$. All four product-rule terms of $\operatorname{Tr}(P_iG_{ij}P_jG_{ji})$ are retained. The script proves these phases and a noncommuting 2-by-2 block resolvent identity by independent inverse differentiation.

Contour reduction for a complex mode must retain complex linearity. Integrate both q and its conjugate partner, then use $(C_q-C_{-q}^{*})/(2i)$, equivalent to the cosine response minus $i$ times the sine response. Taking `imag(C_q)` alone is incorrect. Units: $H$ eV, $S$ dimensionless, displacement Ang, response eV/Ang (multiply 1000 for meV/Ang).

Verification: `python docs/sympy/q_phased_exchange_contraction.py` asserts seven identities. The same script is mirrored and hash-checked by the project's `check-math` gate.
