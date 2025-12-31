---
name: TB2J
description: Guide for using TB2J command-line tools to calculate magnetic interaction parameters (J, DMI, etc.) from DFT outputs (Wannier90, Siesta, Abacus). Use this skill when the user asks about running TB2J commands, their parameters, inputs, or outputs.
---

# TB2J Skill

This skill provides expert guidance on using the TB2J package to calculate magnetic interaction parameters (Heisenberg exchange $J$, DMI $D$, Anisotropy $J_{ani}$) from DFT inputs.

## References

*   **[Command Reference](references/commands.md)**: Detailed flags and arguments for all scripts (`wann2J.py`, `siesta2J.py`, etc.).
*   **[Workflows & DFT Prep](references/workflows.md)**: Step-by-step guides for preparing VASP/Wannier90/Siesta inputs.
*   **[Conventions](references/conventions.md)**: Vital information on Hamiltonian signs, units, and spin normalization.
*   **[Outputs](references/outputs.md)**: Explanations of `exchange.out`, `exchange.xml`, and interfaces to Vampire/Multibinit.

## Common Workflows

### 1. Wannier90 (VASP, QE, Abinit)
*   **Command**: `wann2J.py`
*   **Key Step**: Ensure `projections` in `.win` file cover magnetic orbitals.
*   **See**: [Workflows > Wannier90](references/workflows.md#1-wannier90-workflow-general)

### 2. Siesta
*   **Command**: `siesta2J.py`
*   **Key Step**: Set `SaveHS True` in `.fdf`.
*   **See**: [Workflows > Siesta](references/workflows.md#2-siesta-workflow)

### 3. Abacus
*   **Command**: `abacus2J.py`
*   **Key Step**: Set `out_mat_hs2 1` in `INPUT`.
*   **See**: [Workflows > Abacus](references/workflows.md#3-abacus-workflow)

### 4. Non-Collinear / Anisotropy
*   **Option A**: Full non-collinear DFT + `wann2J.py --spinor`.
*   **Option B**: "Rotate and Merge" using `TB2J_rotate.py` (generate structures) -> DFT -> `TB2J_merge.py` (combine results).
*   **See**: [Workflows > Non-Collinear](references/workflows.md#4-non-collinear--spin-orbit-coupling-soc)

## Troubleshooting / FAQ

*   **Results contradict experiment?**
    *   Check [Conventions](references/conventions.md). Are you comparing $J$ vs $2J$?
    *   Verify the **Fermi Energy** (`--efermi`) matches the DFT calculation exactly.
    *   Is the system metallic? Try increasing K-mesh or adjusting `--nz`.

*   **Wannier functions not localized?**
    *   Check `wannier90.wout` for spreads.
    *   Adjust `dis_win` (outer window) to include all relevant bands.
    *   Check projections.

*   **"Orbitals not found" error?**
    *   Ensure `--elements` matches the atoms in the structure file.
    *   Check if `exclude_orbs` is needed for semi-core states.
