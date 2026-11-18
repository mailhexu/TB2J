### Computing Magnetocrystalline anisotropy energy (MAE)

:warning: This feature is currently under development and internal test. Do not use it for production yet.
:warning: This feature is only available with the ABACUS and SIESTA code.

The MAE calculation is implemented in `TB2J/MAEGreen.py` using the magnetic force theorem with Green's function perturbation theory. The implementation provides both efficient perturbative and exact eigenvalue approaches for computing magnetocrystalline anisotropy energies. 


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


After the two steps of calculations, we can use the `abacus_get_MAE` function from `TB2J.MAEGreen` to compute the MAE. The function reads the Hamiltonian from the two steps of calculations and uses the magnetic force theorem to compute the band energy differences for different magnetization directions. 


The non-SOC Hamiltonian is rotated along different magnetization directions (defined by theta, phi in spherical coordinates), and the energy differences are computed using second-order perturbation theory.

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






