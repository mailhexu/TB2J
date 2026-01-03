# TB2J_edit

TB2J_edit is a tool for modifying TB2J results. It allows you to post-process the exchange parameters calculated by TB2J, such as adding single-ion anisotropy, toggling DMI or anisotropic exchange, and symmetrizing the exchange parameters.

## Command Line Interface

The `TB2J_edit.py` command provides several subcommands:

### Load and Save

Simply load and save the results (useful for format conversion or checking if the file is valid).

```bash
TB2J_edit.py load -i TB2J_results/TB2J.pickle -o modified_results
```

### Set Single-Ion Anisotropy

Set the single-ion anisotropy (SIA) for specific atomic species.

```bash
# Set Sm anisotropy to 5 meV along z-axis
TB2J_edit.py set-anisotropy -i TB2J_results/TB2J.pickle -o modified_results -s Sm 5.0 -d "0 0 1" -m
```

*   `-s SPECIES K1`: Set the anisotropy constant `K1` for `SPECIES`. Can be used multiple times.
*   `-d "x y z"`: Set the direction of the anisotropy axis (default is [0, 0, 1]).
*   `-m`: Interpret `K1` in meV (default is eV).

### Toggle DMI and Anisotropic Exchange

Enable or disable Dzyaloshinskii-Moriya Interaction (DMI) or anisotropic exchange (Jani).

```bash
# Disable DMI
TB2J_edit.py toggle-dmi -i TB2J_results/TB2J.pickle -o modified_results --disable

# Enable Anisotropic Exchange
TB2J_edit.py toggle-jani -i TB2J_results/TB2J.pickle -o modified_results --enable
```

### Symmetrize Exchange

Symmetrize the isotropic exchange parameters based on a reference crystal structure. This is useful if the calculated J values slightly break symmetry due to numerical noise or if you want to enforce a higher symmetry (e.g., from a high-temperature phase).

```bash
TB2J_edit.py symmetrize -i TB2J_results/TB2J.pickle -S structure.cif -o modified_results
```

*   `-S STRUCTURE`: Path to the reference structure file (e.g., CIF, POSCAR).

### Info

Show information about the TB2J results, including composition, number of magnetic atoms, and enabled interactions.

```bash
TB2J_edit.py info -i TB2J_results/TB2J.pickle
```

## Python API

You can also use the `TB2J.io_exchange.edit` module in your Python scripts.

```python
from TB2J.io_exchange.edit import load, set_anisotropy, toggle_DMI, save

# Load results
spinio = load('TB2J_results/TB2J.pickle')

# Modify results
set_anisotropy(spinio, species='Sm', k1=5.0, k1dir=[0, 0, 1]) # k1 in eV
toggle_DMI(spinio, enabled=False)

# Save results
save(spinio, 'modified_results')
```

### Functions

*   `load(path)`: Load TB2J results from a pickle file.
*   `save(spinio, path)`: Save modified results to a directory.
*   `set_anisotropy(spinio, species, k1, k1dir)`: Set single-ion anisotropy.
*   `toggle_DMI(spinio, enabled)`: Enable/disable DMI.
*   `toggle_Jani(spinio, enabled)`: Enable/disable anisotropic exchange.
*   `symmetrize_exchange(spinio, atoms, symprec)`: Symmetrize exchange parameters.
