# The ligand spin problem: downfolding the Heisenberg Hamiltonian

Usually the magnetic moments are dominantly on the metal sites. But in some structures, the spin magnetic moments on the ligand are non-negligible. They are, however, not due to the spin splitting of the ligand atom, but the hybridization to the orbitals of surrounding atoms (often transition metals). Therefore, the ligand magnetic moments are not independent, and will move together with that of the metal.

In these cases, a "downfolding" method could be used to get an effective Heisenberg model with only the transition metal spins as independent variables from the exchange parameters with both transition metal spins and ligand spins.  Note that the current method is only valid for ferromagnetic structures. The extension to antiferromagnetic and non-collinear magnetic structures is under development. Even for the FM state, the result of this method should still be carefully validated. We don't recommend the usage unless you're sure about what you're doing. 



## Usage:

To do the downfolding, one has to first generate the Heisenberg Hamiltonian with both the transition metal and the ligands as magnetic elements, e.g. in CrI3, 

```
wann2J.py --elements Cr I ....
```

The result is saved to the TB2J\_results directory. Then run the downfolding to get the Cr only effective exchange parameters, e.g.

```bash
TB2J_downfold.py --inpath TB2J_results --outpath TB2J_results_downfold --metals Cr --ligands I
```

will generate the downfolded result to TB2J\_results\_downfold. 

#### Mryasov: induced moments with variable magnitude

For a collinear reference with parallel metal moments and nonzero induced
moments, use the isotropic [Mryasov model as formulated by Borisov et al.,
Eqs. (3)–(12)](https://arxiv.org/html/2611.26697v1#S2.E3):

```bash
TB2J_downfold.py --method mryasov --metals Fe --ligands Pt \
    --inpath TB2J_results --outpath TB2J_results_mryasov
```

The input must contain the original Fe and Pt spins, their reference moments
and reciprocal isotropic Fe–Pt bonds. The output is a standard TB2J result
with only the selected metal spins active. Use a new or empty output
directory. The method retains the direct
metal exchange and adds the interaction mediated by each independent
induced moment. It ignores ligand–ligand exchange and all anisotropic terms.
The reduction works on real-space bonds and can generate longer translations
than were present in the input; it does not use `--qmesh`.

`--dressing` specifies the positive scale $\lambda$ of the mediated energy.
The default $\lambda=1$ uses only the input bonds and does not require spin
spirals. Set another value only if supplied or independently calibrated; it
does not scale the direct metal exchange or the induced moment. The method
does not extract a missing electronic response kernel. Use the original
metal–ligand result, not exchange already downfolded to the metals.

For a model Hamiltonian with existing isotropic bonds,
`TB2J.mryasov.mryasov_downfold(exchange, metals, ligands, dressing=1.0)`
accepts the ordered-pair dictionary `{(R, i, j): J_eV}` and sequences of
spin indices. It returns metal-only bonds in the same convention and original
spin indices. It assumes an FM-normalized reference; unlike the command
interface it receives no site moments to check that assumption.





