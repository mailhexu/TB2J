# Magnon Band Structure and DOS Plotting

TB2J provides a unified command-line tool `TB2J_magnon.py` to calculate and plot magnon band structures and density of states (DOS) from exchange parameters.

## Command Line Interface

### Basic Usage

```bash
# Plot magnon band structure
TB2J_magnon.py --bands

# Plot magnon DOS
TB2J_magnon.py --dos

# Plot both band structure and DOS
TB2J_magnon.py --bands --dos
```

### Command Line Options

```
usage: TB2J_magnon.py [-h] [--bands] [--dos]
                      [--config CONFIG | --save-config SAVE_CONFIG] [-p PATH]
                      [--no-Jiso] [--no-Jani] [--no-DMI] [--no-SIA]
                      [-c SPIN_CONF_FILE | --spin-conf M [M ...]]
                      [-s]
                      # Band options
                      [-k KPATH] [--npoints NPOINTS] [--qpoints QPOINTS] [--band-output BAND_OUTPUT]
                      # DOS options
                      [--kmesh nx ny nz] [--no-gamma] [--width WIDTH]
                      [--window emin emax] [--npts NPTS] [--dos-output DOS_OUTPUT]

Calculate and plot magnon band structure and/or DOS from TB2J results

options:
  -h, --help            show this help message and exit
  --bands               Plot magnon band structure
  --dos                 Plot magnon density of states
  --config CONFIG       Path to TOML configuration file
  --save-config SAVE_CONFIG
                        Save default configuration to specified TOML file
  -p PATH, --path PATH  Path to TB2J results directory (default: TB2J_results)
  --no-Jiso             Exclude isotropic exchange interactions
  --no-Jani             Exclude anisotropic exchange interactions
  --no-DMI              Exclude Dzyaloshinskii-Moriya interactions
  --no-SIA              Exclude single-ion anisotropy
  -c SPIN_CONF_FILE, --spin-conf-file SPIN_CONF_FILE
                        Path to file containing magnetic moments (nspin×3 array)
  --spin-conf M [M ...]
                        Spin configuration as flat list: mx1 my1 mz1 mx2 my2 mz2 ...
                        Values in μB. E.g., --spin-conf 0 0 3 0 0 -3
  -s, --show            Show figure on screen

Band structure options:
  -k KPATH, --kpath KPATH
                        k-path specification (default: auto-detected)
  --npoints NPOINTS     Number of k-points along the path (default: 300)
  --qpoints QPOINTS     Custom q-points as name:coord pairs (e.g., "G:0,0,0,X:0.5,0,0")
                        Overrides ASE default special points
  --band-output BAND_OUTPUT
                        Output file name for band structure (default: magnon_bands.png)

DOS options:
  --kmesh nx ny nz      k-point mesh dimensions (default: 20, 20, 20)
  --no-gamma            Exclude Gamma point from k-mesh
  --width WIDTH         Gaussian smearing width in eV (default: 0.001)
  --window emin emax    Energy window in meV (optional)
  --npts NPTS           Number of energy points (default: 401)
  --dos-output DOS_OUTPUT
                        Output filename for DOS plot (default: magnon_dos.png)
```

### Examples

```bash
# Plot band structure with custom k-path
TB2J_magnon.py --bands --kpath GXMR --band-output my_bands.png

# Plot DOS with finer k-mesh
TB2J_magnon.py --dos --kmesh 30 30 30 --width 0.0005

# Plot both, excluding anisotropic and DMI interactions
TB2J_magnon.py --bands --dos --no-Jani --no-DMI

# Use configuration file
TB2J_magnon.py --save-config config.toml
TB2J_magnon.py --config config.toml --bands --dos

# Specify spin configuration via CLI (two antiparallel spins of 3 μB)
TB2J_magnon.py --bands --spin-conf 0 0 3 0 0 -3

# Use custom q-points (see Custom Q-Points section below)
TB2J_magnon.py --bands --kpath GXMR --qpoints "G:0,0,0,X:0.5,0,0,M:0.5,0.5,0,R:0.5,0.5,0.5"
```

### Configuration File

You can save and load parameters from a TOML configuration file:

```bash
# Save default configuration
TB2J_magnon.py --save-config config.toml

# Use configuration file
TB2J_magnon.py --config config.toml --bands --dos
```

Example `config.toml`:

```toml
# TB2J results location
path = "TB2J_results"

# Exchange interactions (all enabled by default)
Jiso = true
Jani = true
DMI = false
SIA = true

# Spin configuration (optional, uses TB2J results if not specified)
# Values in μB (Bohr magneton), for spin S use 2S
# spin_conf = [[0.0, 0.0, 3.0]]           # Single spin: S=3/2 → 3 μB
# spin_conf = [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]  # Two antiparallel spins

# Band structure parameters
kpath = "GXMR"
npoints = 300
filename = "magnon_bands.png"

# Custom q-points (optional, overrides ASE defaults)
# Uncomment to use custom q-point definitions
# [qpoints]
# G = [0.0, 0.0, 0.0]
# X = [0.5, 0.0, 0.0]
# M = [0.5, 0.5, 0.0]
# R = [0.5, 0.5, 0.5]

# DOS parameters
kmesh = [20, 20, 20]
gamma = true
width = 0.001
npts = 401
```

## Python Interface

### Using MagnonParameters

```python
from TB2J.magnon.magnon_parameters import MagnonParameters
from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
from TB2J.magnon.magnon_dos import plot_magnon_dos_from_TB2J

# Create parameters with all interactions enabled by default
params = MagnonParameters(
    path="TB2J_results",
    Jiso=True,
    Jani=True,
    DMI=False,  # Disable DMI
    SIA=True,
)

# Plot band structure
params.filename = "magnon_bands.png"
params.kpath = "GXMR"
plot_magnon_bands_from_TB2J(params)

# Plot DOS
params.filename = "magnon_dos.png"
params.kmesh = [30, 30, 30]
plot_magnon_dos_from_TB2J(params)
```

#### With Custom Q-Points

```python
from TB2J.magnon.magnon_parameters import MagnonParameters
from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J

# Define custom q-points
custom_qpoints = {
    "G": [0.0, 0.0, 0.0],
    "X": [0.5, 0.0, 0.0],
    "M": [0.5, 0.5, 0.0],
    "R": [0.5, 0.5, 0.5],
}

params = MagnonParameters(
    path="TB2J_results",
    kpath="GXMRG",
    qpoints=custom_qpoints,  # Pass custom q-points
    filename="magnon_bands_custom.png",
)

plot_magnon_bands_from_TB2J(params)
```

### Advanced Usage with Magnetic Structure

```python
import numpy as np
from TB2J.magnon.magnon_parameters import MagnonParameters
from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J

# Create quantization axes file for non-collinear magnetism
nspin = 4
uz = np.array([
    [0.0, 0.0, 1.0],   # atom 1: spin up
    [0.0, 0.0, -1.0],  # atom 2: spin down
    [1.0, 0.0, 0.0],   # atom 3: spin along x
    [-1.0, 0.0, 0.0],  # atom 4: spin along -x
])
np.savetxt("spins.txt", uz)

# Plot with custom magnetic configuration
params = MagnonParameters(
    kpath="GXMR",
    uz_file="spins.txt",
)
plot_magnon_bands_from_TB2J(params)
```

## Output Files

### Band Structure

When plotting band structure, the following files are created:
- `magnon_bands.png` (or specified output): The band structure plot
- `magnon_bands.json`: Band structure data (k-points, energies, labels)

### DOS

When plotting DOS, the following files are created:
- `magnon_dos.png` (or specified output): The DOS plot
- `magnon_dos.json`: DOS data (energies, DOS values)

## Spin Configuration

By default, TB2J uses magnetic moments from the DFT calculation stored in 
TB2J results. You can override this for several reasons:

### Use Cases

1. **Quantum Theory Alignment**: In linear spin wave theory, the spin 
   quantum number $S$ should be an integer or half-integer (1/2, 1, 3/2, ...).
   
   The magnetic moment input uses units of $\mu_B$ (Bohr magneton), 
   which corresponds to $2S$ (i.e., $S=3/2$ → $3\,\mu_B$).
   
   For example, Cr³⁺ has $S = 3/2$, so the moment should be $3\,\mu_B$, 
   even though DFT may give $\sim 3.1\,\mu_B$. Using $3\,\mu_B$ ensures 
   the spin wave theory is internally consistent.

2. **Explore Different Magnetic States**: Calculate magnon dispersion for 
   FM state even if DFT used AFM ordering.

3. **Rotate Spin Directions**: DFT calculated with spins along $z$, explore 
   magnons with spins along $x$.

### Methods to Specify

| Method | Example |
|--------|---------|
| Default (TB2J results) | *(no option)* |
| TOML config | `spin_conf = [[0, 0, 3], [0, 0, -3]]` |
| CLI | `--spin-conf 0 0 3 0 0 -3` |
| File | `--spin-conf-file spins.txt` |

### CLI Examples

```bash
# Two antiparallel spins of 3 μB each
TB2J_magnon.py --bands --spin-conf 0 0 3 0 0 -3

# Four spins in different directions
TB2J_magnon.py --bands --spin-conf 3 0 0 -3 0 0 0 0 3 0 0 -3
```

### TOML Example

```toml
# Cr³⁺ with S=3/2: use 3 μB (not DFT value ~3.1 μB)
spin_conf = [[0.0, 0.0, 3.0]]

# Two antiparallel spins
spin_conf = [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]

# Four spins in different directions
spin_conf = [
    [3.0, 0.0, 0.0],   # along +x
    [-3.0, 0.0, 0.0],  # along -x
    [0.0, 0.0, 3.0],   # along +z
    [0.0, 0.0, -3.0],  # along -z
]
```

The moment values are in $\mu_B$. For spin $S$, use $m = 2S\,\mu_B$.

### File Format

The `--spin-conf-file` should contain a nspin×3 array with magnetic moments:

```
0.0  0.0  3.0   # atom 1: 3 μB along +z
0.0  0.0  -3.0  # atom 2: 3 μB along -z
1.0  0.0  0.0   # atom 3: 1 μB along +x
```

Each row represents the magnetic moment vector $(m_x, m_y, m_z)$ in $\mu_B$. 
The number of rows must match the number of magnetic atoms in your system.

## Custom Q-Points

By default, TB2J uses ASE's built-in special k-points for standard crystal structures. 
These are high-symmetry points (Γ, X, M, R, etc.) that ASE automatically identifies 
based on the crystal lattice type and symmetry.

However, you can specify **custom q-points** with your own names and coordinates. 
This is useful when:

1. **Non-standard high-symmetry points**: Your crystal has high-symmetry points 
   that don't match ASE's defaults
2. **Custom k-path**: You want to explore a specific region of the Brillouin zone
3. **Different naming convention**: You prefer different labels (e.g., "G" vs "Γ")

### Three Methods to Specify Custom Q-Points

#### Method 1: Command Line

Use the `--qpoints` option with format `name:x,y,z` pairs separated by commas:

```bash
TB2J_magnon.py --bands \
    --kpath GXMRG \
    --qpoints "G:0,0,0,X:0.5,0,0,M:0.5,0.5,0,R:0.5,0.5,0.5"
```

**Format**: Each q-point is specified as `NAME:x,y,z` where:
- `NAME`: The label for the q-point (e.g., "G", "X", "M")
- `x,y,z`: The three coordinates in fractional reciprocal lattice units

Multiple q-points are separated by commas.

#### Method 2: TOML Configuration

In your TOML configuration file, add a `[qpoints]` section:

```toml
kpath = "GXMRG"

[qpoints]
G = [0.0, 0.0, 0.0]     # Gamma point
X = [0.5, 0.0, 0.0]     # X point
M = [0.5, 0.5, 0.0]     # M point
R = [0.5, 0.5, 0.5]     # R point
```

The `kpath` parameter specifies the path connecting these q-points, and the 
`[qpoints]` section defines their coordinates.

#### Method 3: Python API

When using the Python API, pass a dictionary to the `qpoints` parameter:

```python
from TB2J.magnon.magnon_parameters import MagnonParameters
from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J

# Define custom q-points as a dictionary
custom_qpoints = {
    "G": [0.0, 0.0, 0.0],    # Gamma point
    "X": [0.5, 0.0, 0.0],    # X point
    "M": [0.5, 0.5, 0.0],    # M point
    "R": [0.5, 0.5, 0.5],    # R point
}

params = MagnonParameters(
    path="TB2J_results",
    kpath="GXMRG",           # Path connecting the q-points
    qpoints=custom_qpoints,  # Your custom q-points
    filename="magnon_bands.png",
)

plot_magnon_bands_from_TB2J(params)
```

### How It Works

1. **Default Behavior**: When `qpoints` is not specified, TB2J uses ASE's 
   `bandpath()` function with built-in special points for the crystal structure.

2. **Custom Q-Points**: When you provide custom q-points, they are passed to 
   ASE's `bandpath()` function via the `special_points` parameter, overriding 
   the defaults.

3. **K-Path**: The `kpath` parameter (e.g., "GXMRG") specifies the sequence of 
   q-points to visit. Each letter should correspond to a name in your q-points 
   definition.

### Examples

#### Example 1: Hexagonal Lattice with Custom K-Point

```bash
# For a hexagonal lattice where you want to define a specific K-point
TB2J_magnon.py --bands \
    --kpath GMKG \
    --qpoints "G:0,0,0,M:0.5,0,0,K:0.333,0.333,0"
```

#### Example 2: Complete TOML Example

```toml
# config_custom.toml
path = "TB2J_results"

# Exchange interactions
Jiso = true
Jani = true
DMI = true
SIA = true

# Band structure with custom q-points
kpath = "GXMRG"
npoints = 300
filename = "magnon_bands_custom.png"

[qpoints]
G = [0.0, 0.0, 0.0]
X = [0.5, 0.0, 0.0]
M = [0.5, 0.5, 0.0]
R = [0.5, 0.5, 0.5]
```

Then use it:

```bash
TB2J_magnon.py --config config_custom.toml --bands
```

### Important Notes

1. **Coordinate System**: Q-point coordinates are in **fractional reciprocal 
   lattice units** (not Cartesian). For example, [0.5, 0, 0] means halfway 
   along the first reciprocal lattice vector.

2. **Matching Names**: The names in your `kpath` must match the keys in your 
   `qpoints` dictionary/TOML section. Case-sensitive.

3. **Backward Compatibility**: If you don't specify custom q-points, TB2J 
   works exactly as before, using ASE's default special points.

4. **Partial Override**: You only need to define the q-points referenced in 
   your kpath. Any additional points in ASE's defaults that aren't used won't 
   matter.
