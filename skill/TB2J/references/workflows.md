# DFT Preparation & Workflows

Preparing your DFT calculation correctly is the most critical step for getting accurate results with TB2J.

## 1. Wannier90 Workflow (General)
Used for **VASP**, **Quantum ESPRESSO**, **Abinit**, and others via `wann2J.py`.

### Step 0: DFT Calculation
*   Perform a standard self-consistent field (SCF) calculation.
*   **Spin**: Must be spin-polarized.
*   **Orbitals**: Identify the relevant magnetic orbitals (e.g., d-orbitals for transition metals) from the DOS.

### Step 1: Wannierization (`wannier90.win`)
You must construct Maximally Localized Wannier Functions (MLWFs) covering the magnetic bands.

*   **`projections`**: Explicitly project onto the magnetic orbitals and ligand p-orbitals.
    ```text
    begin projections
    Mn: d
    O : p
    end projections
    ```
*   **Energy Windows**:
    *   `dis_win_min` / `dis_win_max`: Outer window. Include all bands with magnetic character.
    *   `dis_froz_min` / `dis_froz_max`: Inner (frozen) window. Ensure this covers the main magnetic bands and the Fermi energy.
*   **Output Flags**:
    *   `write_hr = true`: Writes the Hamiltonian (`*_hr.dat`).
    *   `write_xyz = true`: Writes Wannier centers (`*_centres.xyz`).
    *   *(Newer versions)* `write_tb = true`: Writes both to `_tb.dat`.

### Step 2: Run TB2J
```bash
wann2J.py --posfile structure.vasp --efermi <E_F> --kmesh 5 5 5 --elements Mn --prefix_up wan_up --prefix_down wan_down
```

---

## 2. Siesta Workflow
Used via `siesta2J.py`.

### Input Settings (`.fdf`)
You must save the Hamiltonian and Overlap matrices.

```fdf
# Save Hamiltonian and Overlap
SaveHS True

# Use NetCDF format (Recommended)
CDF.Save True
```

### Run TB2J
```bash
siesta2J.py --fdf_fname input.fdf --elements Fe --kmesh 7 7 7
```

---

## 3. ABACUS Workflow
Used via `abacus2J.py` (LCAO mode).

### Input Settings (`INPUT`)
You need to output the Hamiltonian and Overlap matrices.

```text
out_mat_hs2  1
suffix       Fe   # Example suffix
```
This generates `data-HR-*` and `data-SR-*` files in the `OUT.Fe` directory.

### Run TB2J
```bash
abacus2J.py --path . --suffix Fe --elements Fe --kmesh 7 7 7
```

---

## 4. Non-Collinear & Spin-Orbit Coupling (SOC)

For DMI and Anisotropic Exchange, you generally need **Spin-Orbit Coupling (SOC)** enabled in DFT.

### Method A: Single Non-Collinear Calculation (Wannier90)
1.  Run DFT with Non-Collinear Spin + SOC.
2.  Generate **Spinor** Wannier functions (TB2J requires spinor format).
3.  Run:
    ```bash
    wann2J.py --spinor --groupby spin --prefix_spinor wan ...
    ```

### Method B: Rotate and Merge (Perturbation)
If full non-collinear Wannierization is difficult, or for verifying anisotropy:

1.  **Generate Rotated Structures**:
    ```bash
    TB2J_rotate.py structure.vasp --ftype vasp --noncollinear
    ```
    This creates folders `atoms_0`, `atoms_1`, etc., with rotated crystal structures (effectively rotating the spin axis relative to the lattice).

2.  **Run DFT + TB2J for each**:
    Run the standard collinear TB2J workflow in each `atoms_x` directory.

3.  **Merge Results**:
    ```bash
    TB2J_merge.py atoms_1 atoms_2 atoms_0
    ```
    This reconstructs the full anisotropic tensor from the scalar relativistic calculations.
