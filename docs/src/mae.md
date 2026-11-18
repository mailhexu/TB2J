### Computing Magnetocrystalline anisotropy energy (MAE)

:warning: This feature is currently under development and internal test. Do not use it for production yet.
:warning: This feature is only available with the ABACUS and SIESTA code.

`TB2J/MAEGreen.py` evaluates the second-order SOC response with normalized CFR2 weights and the separate physical `-Im/(2*pi)` factor. At finite temperature and fixed chemical potential this is a grand-potential coefficient, not a general fixed-particle-number internal-energy difference. The optional eigenvalue calculation uses its own occupation prescription and is not an identical ensemble by construction.

The ABACUS LCAO basis is non-orthogonal. Generalized eigenvectors and the overlap matrix must remain consistent when forming the Green function and operator traces; an identity-metric replacement is not valid. A full-space inverse-overlap moment and a retained/projected spectral moment are distinct. Pole convergence alone does not establish retained-state or k-mesh convergence.

Use an even integer `nz >= 2` and converge the pole depth at the selected temperature and spectral range. CLI temperatures in K are converted to an internal width in eV with the Boltzmann constant; no implicit spin-degeneracy factor is supplied by the contour. The implementation follows the continued-fraction representation of Ozaki, *Phys. Rev. B* **75**, 035123 (2007).


To compute the magnetocrystalline anisotropy energy (MAE) of a magnetic system with the magnetic force theorem, two steps of DFT calculations are needed.

- The first step is to do  an collinear spin calculation. The density and the Hamiltonian is saved at this step. Note that the current implementation requires the SOC to be turned on in ABACUS, but setting the SOC strength to zero (soc_lambda=0).

- The second step is to do a non-SCF non-collinear spin calculation with SOC turned on. The density is read from the previous step. In practice, one step of SCF calculation is done (as the current implementation does not write the Hamiltonian and the energy). The Hamiltonian should be saved in this step, too. 

Here is one example: 
Step one: collinear spin calculation.  Note that  instead of using  nspin=2, we use nspin=4, and  lspinorb=1 to enable the SOC but set the soc\_lambda  to 0.0 to turn off the SOC.  This is to make the Hamiltonian saved in the spinor form, so it can be easily compared with the next step of a real calculation with SOC. 


``` text
INPUT_PARAMETERS
# SCF calculation with SOC turned on, but soc_lambda=0. 
calculation                             scf
nspin                                   4
symmetry                                0
noncolin                                1
lspinorb                                1
ecutwfc                                 100
scf_thr                                 1e-06
init_chg                                atomic
out_mul                                 1
out_chg                                 1
out_dos                                 0
out_band                                0
out_wfc_lcao                            1
out_mat_hs2                             1
ks_solver                               scalapack_gvx
scf_nmax                                500
out_bandgap                             0
basis_type                              lcao
gamma_only                              0
smearing_method                         gaussian
smearing_sigma                          0.01
mixing_type                             broyden
mixing_beta                             0.5
soc_lambda                              0.0
ntype                                   1
dft_functional                          PBE
```


- Step two: non-SCF non-collinear spin calculation. 

In this step, we need to start from the density saved in the previous step. So we can copy the output directory of the previous step to a the present directory.
To mimic the non-SCF calculation: 
    * The scf\_nmax should be set to 1 to end the scf calculation in one step.
    * The "scf\_thr" should be set to a large value to make the calculation "converge" in one step.
    * The mixing\_beta should be set to a small value to suppress the density mixing.
    * Then we can set the "init\_chg" to "file" to read the density from the previous step. The lspionorb should be set to 1, and the soc_lambda should be set to a 1.0 to enable the SOC. 
The "out\_mat\_hs2" should be set to 1 to save the Hamiltonian.

:warning: Once ABACUS can output the Hamiltonian in the non-SCF calculation, we can use a "calculation=nscf" to do the non-SCF calculation.


```text
INPUT_PARAMETERS
# Non-SCF non-collinear spin calculation with SOC turned on. soc_lambda=1.0
calculation                             scf
nspin                                   4
symmetry                                0
noncolin                                1
lspinorb                                1
ecutwfc                                 100
scf_thr                                 1000000.0 
init_chg                                file
out_mul                                 1
out_chg                                 0
out_mat_hs2                             1
ks_solver                               scalapack_gvx
scf_nmax                                1
basis_type                              lcao
smearing_method                         gaussian
smearing_sigma                          0.01
mixing_type                             broyden
mixing_beta                             1e-06
soc_lambda                              1.0
init_wfc                                file
ntype                                   1
dft_functional                          PBE
```


After the two calculations, `abacus_get_MAE` reads the paired Hamiltonians and overlap matrices. The reader separates the charge and spin parts of their difference: the charge part enters the reference Hamiltonian, and the spin part is the SOC perturbation. Use a compatible frozen-density reference and record the chemical potential and temperature when interpreting the force-theorem response.


The reference spin direction is rotated relative to the SOC perturbation. `MAE.dat` stores raw directional second-order corrections in meV, obtained by multiplying the internal eV values by1000. Form the anisotropy as `E(direction) - E(reference)`; positive means the direction is harder. Disabled matrix/orbital decomposition does not publish new placeholder files; use a fresh output directory because existing files are not deleted.

Here is an example of the usage of the function.

```python
import numpy as np
from TB2J.MAEGreen import abacus_get_MAE

def run():
    # theta, phi: along the xz plane, rotating from z to x. 
    thetas = np.linspace(0, 180, 19) * np.pi / 180
    phis = np.zeros(19)
    abacus_get_MAE(
        path_nosoc= 'soc0/OUT.ABACUS',   
        path_soc= 'soc1/OUT.ABACUS',
        kmesh=[6,6,1],
        gamma=True,
        thetas=thetas, 
        phis=phis,
        nel = 16,
        )

if __name__ == '__main__':
    run()
```

#### Advanced Usage

The `MAEGreen` class also supports various angle presets and options:

```python
from TB2J.MAEGreen import MAEGreen

# Use predefined angle sets
mae = MAEGreen(
    tbmodels=model,
    atoms=model.atoms,
    kmesh=[6,6,1],
    angles="xyz",  # x, y, z axes
    # angles="fib",  # Fibonacci sphere sampling
    # angles="miller",  # Miller index directions
    # angles="scan",  # Full angular scan
)

# Custom angle arrays
mae = MAEGreen(
    tbmodels=model,
    atoms=model.atoms,
    kmesh=[6,6,1],
    angles=[thetas, phis],  # Custom theta, phi arrays
)

# Run with both perturbation and eigenvalue methods
mae.run(output_path="my_mae_results", with_eigen=True)
```


Here the soc0 and soc1 directories are where the two ABACUS calculations are done. We use a 6x6x1 Gamma-centered k-mesh for the integration (which is too small for practical usage!). And explore the energy with the magnetic moments in the x-z plane.

After running the python script above, several output files will be created in the `TB2J_anisotropy` directory:

- **MAE.dat**: Contains theta, phi, and total MAE values (in meV)
- **MAE_matrix.dat**: Atom-atom interaction matrices for each angle
- **MAE_orb.dat**: Orbital-resolved MAE contributions for each atom
- **MAE_eigen.dat**: (optional) Exact eigenvalue results if `with_eigen=True`

The MAE values are computed using second-order perturbation theory and reported in meV relative to the reference magnetization direction.  






