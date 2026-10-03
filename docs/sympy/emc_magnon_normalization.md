# EMC magnon amplitude and metric normalization

Executable certificate: [`emc_magnon_normalization.py`](emc_magnon_normalization.py). Run in `mydev` with `python docs/sympy/emc_magnon_normalization.py`; all assertions passed.

For the +z ferromagnet, define $n_+=n_x+in_y=\sqrt{2/S}\,u b_q$ and $n_-=n_x-in_y=\sqrt{2/S}\,u^*b_q^\dagger$. Substitution into $D(n_x\sigma_x+n_y\sigma_y)$ gives the annihilation coefficient $D\sqrt{2/S}\,u\sigma_-$ and the adjoint creation coefficient $D\sqrt{2/S}\,u^*\sigma_+$. The script differentiates the exact symbolic perturbation with respect to independent boson variables and asserts both matrices. The small-angle expansion of $\sin\theta$ fixes the linear angle conversion.

For a positive-energy TB2J Cholesky pencil, $H=KK^\dagger$, $g=\operatorname{diag}(1,-1)$, and $A=K^\dagger gK$, with $Av=\omega v$ and $v^\dagger v=1$. The physical vector is $\psi=\sqrt\omega K^{-\dagger}v$. The script verifies for a generic invertible lower-triangular $K$ that $A^{-1}=K^{-1}gK^{-\dagger}$, the generalized-pencil residual is proportional to $(A-\omega I)v$, and $\psi^\dagger g\psi=\omega v^\dagger A^{-1}v=1$. At a singular zero mode $K^{-\dagger}$ is unavailable; substituting $\omega=0$ into a regularized expression yields zero metric norm, not a bosonic unit state. No eV-per-boson normalization is assigned there.

The corresponding electron-field/q-phase certificate is in the EMC package at `docs/sympy/emc_vertex.py` and `docs/sympy/emc_vertex.md`.
