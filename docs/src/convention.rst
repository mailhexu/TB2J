=======================================
 Conventions of Heisenberg Model
=======================================

Before you use the TB2J output, please read very carefully this section. There are many conventions of the Heisenberg. We strongly suggest that you clearly specify the convention you use in any published work. Here we describe the convention used in TB2J. 

The Heisenberg Hamiltonian contains four different parts and reads as 

.. math::

   E = &  -\sum_i K_i \vec{S}_i^2 \\
   &-\sum_{i \neq j} \biggl[ J^{iso}_{ij} \vec{S}_i\cdot\vec{S}_j \hspace{0.8cm} \\
   &+ \vec{S}_i \mathbf{J}^{ani}_{ij} \vec{S}_j \\
   &+ \vec{D}_{ij} \cdot \left( \vec{S}_i\times\vec{S}_j\right) \biggl],  

where the first term represents the single-ion anisotropy (SIA), the second the isotropic exchange, and the third term is the symmetric anisotropic exchange, where :math:`\mathbf{J}^{ani}` is a :math:`3\times 3` tensor with  :math:`J^{ani}=J^{ani,T}`. The final term is the DMI, which is antisymmetric. Importantly, the SIA is not accessible from Wannier 90 as it requires separately the spin-orbit coupling part of the Hamiltonian. However, it is readily accessible from constrained DFT calculations. 

We note that there are several conventions for the Heisenberg Hamiltonian, here we take a commonly used one in atomic spin dynamics: we use a minus sign in the exchange terms, i.e. positive exchange :math:`J` values favor ferromagnetic alignment. Every pair :math:`ij` is taken into account twice, :math:`J_{ij}` and :math:`J_{ji}` are both in the Hamiltonian. Similarly, both :math:`uv` and :math:`vu` are in the symmetric anisotropic term. The spin vectors :math:`\vec{S}_i` are normalized to 1, so that the parameters are in units of energy. The other commonly used conventions differ in a prefactor 1/2 or a summation over different :math:`ij` pairs only. The conversion factors to other conventions are given in the following table. For other conventions in which the spins are not normalized, the parameters need to be divided by :math:`|\vec{S}_i\cdot\vec{S}_j|` in addition.
