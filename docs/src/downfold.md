### The ligand spin problem: downfolding the Heisenberg Hamiltonian

Usually the magnetic moments are dominantly on the metal sites. But in some structures, the spin magnetic momentums on the ligand are non-negligible. They are however, not due to the spin splitting of the ligand atom, but the hybridization to the orbitals of surrounding  atoms (often transitional metals). Therefore, the ligand magnetic moments are not independent, and will move together with the that of the metal.

In these cases, a "downfolding" method could be used to used to get an effective Heisenberg model with only the transitional metal spins as independent variables from the exchange parameters with both transitional metal spins and ligand spins.  Note that the current method is only valid for ferromagnetic structures. The extenstion to antiferromagnetic and non-collinear magnetic structures are under development. Even for FM state, this result of this method should still be carefully validated. We don't recommend the usage unless you're sure about what you're doing. 



#### Usage:

To do the downfolding, one has to first generate the Heisenberg Hamiltonian with both the transitional metal and the ligands as magnetic elements, e.g. in CrI3, 

```
wann2J.py --elements Cr I ....
```

The result is saved to TB2J\_results directory. Then run the downfolding to get the Cr only effective exchange parameters, e.g.

```bash
TB2J_downfold.py --inpath TB2J_results --outpath TB2J_results_downfold --metals Cr --ligands I
```

will generate the downfolded result to TB2J\_results\_downfold. 





