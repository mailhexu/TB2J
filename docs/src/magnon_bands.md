# Magnon Band Structure Plotting

TB2J provides functionality to calculate and plot magnon band structures from the exchange parameters. This can be done either through the Python interface or using the command-line tool.

## Python Interface

### Band Structure Data and Plotting

When you plot the magnon band structure using any of the available methods, the data is automatically saved to a JSON file (with the same name as the output figure but with .json extension). Additionally, an executable Python script `plot_magnon_bands.py` is created that can be used to replot the data.

```bash
# Original plotting command
TB2J_plot_magnon_bands.py --kpath GXMR --output bands.png

# This creates:
# - bands.png: the band structure plot
# - bands.json: the band structure data
# - plot_magnon_bands.py: a script to replot the data

# Replot the data with the provided script
./plot_magnon_bands.py bands.json -o new_plot.png
```

The JSON data file contains:
- k-point coordinates
- Band energies (in meV)
- k-path labels
- Axis labels

### Basic Usage

```python
from TB2J.magnon.magnon3 import MagnonParameters, plot_magnon_bands_from_TB2J

# Using default parameters
params = MagnonParameters()
plot_magnon_bands_from_TB2J(params)

# Or customize parameters
params = MagnonParameters(
    path="TB2J_results",
    kpath="GXMR",
    npoints=300,
    filename="magnon_bands.png",
    Jiso=True,
    Jani=False,
    DMI=False,
)
plot_magnon_bands_from_TB2J(params)
```

### Configuration File Support

```python
# Save parameters to TOML file
params = MagnonParameters(
    kpath="GXMR",
    Q=[0.5, 0.0, 0.0],
    uz_file="spins.txt"
)
params.to_toml("config.toml")

# Load parameters from TOML file
params = MagnonParameters.from_toml("config.toml")
plot_magnon_bands_from_TB2J(params)
```

### Advanced Usage with Magnetic Structure

```python
import numpy as np

# Create quantization axes file for non-collinear magnetism
nspin = 4  # number of magnetic atoms
uz = np.array([
    [0.0, 0.0, 1.0],   # atom 1: spin up
    [0.0, 0.0, -1.0],  # atom 2: spin down
    [1.0, 0.0, 0.0],   # atom 3: spin along x
    [-1.0, 0.0, 0.0],  # atom 4: spin along -x
])
np.savetxt("spins.txt", uz)

# Plot bands with custom magnetic configuration
params = MagnonParameters(
    kpath="GXMR",
    Q=[0.5, 0.0, 0.0],    # Propagation vector
    uz_file="spins.txt",   # Quantization axes for each spin
    n=[0.0, 0.0, 1.0]     # Normal vector for rotation
)
plot_magnon_bands_from_TB2J(params)
```

## Command Line Interface

The `TB2J_plot_magnon_bands.py` script provides easy access to magnon band structure plotting.

### Basic Usage

```bash
# Default settings
TB2J_plot_magnon_bands.py

# Specify k-path and output file
TB2J_plot_magnon_bands.py --kpath GXMR --output bands.png

# Include different interactions
TB2J_plot_magnon_bands.py --Jani --DMI
```

### Configuration File Support

```bash
# Save default configuration
TB2J_plot_magnon_bands.py --save-config config.toml

# Use configuration file
TB2J_plot_magnon_bands.py --config config.toml
```

### Example Configuration File (config.toml)

```toml
# TB2J results location
path = "TB2J_results"

# Band structure parameters
kpath = "GXMR"
npoints = 300
filename = "magnon_bands.png"

# Exchange interactions to include
Jiso = true
Jani = false
DMI = false

# Magnetic structure parameters
Q = [0.5, 0.0, 0.0]      # Propagation vector
uz_file = "spins.txt"     # Quantization axes file
n = [0.0, 0.0, 1.0]      # Normal vector for rotation
```

### Command Line Options

```bash
TB2J_plot_magnon_bands.py --help
```

Will show all available options:

```
  --config CONFIG      Path to TOML configuration file
  --save-config FILE  Save default configuration to specified TOML file
  --path PATH         Path to TB2J results directory (default: TB2J_results)
  --kpath KPATH       k-path specification (default: GXMR)
  --npoints NPOINTS   Number of k-points along the path (default: 300)
  --output OUTPUT     Output file name (default: magnon_bands.png)
  --Jiso             Include isotropic exchange interactions (default: True)
  --no-Jiso          Exclude isotropic exchange interactions
  --Jani             Include anisotropic exchange interactions (default: False)
  --DMI              Include Dzyaloshinskii-Moriya interactions (default: False)
  --Q Qx Qy Qz       Propagation vector [Qx, Qy, Qz] (default: [0, 0, 0])
  --uz-file FILE     Path to file containing quantization axes for each spin
  --n nx ny nz       Normal vector for rotation [nx, ny, nz] (default: [0, 0, 1])
```

### Quantization Axes File Format

The `--uz-file` should point to a space-separated text file containing a natom×3 array, where each row represents the quantization axis direction for that spin. For example:

```
0.0  0.0  1.0   # atom 1: spin up
0.0  0.0  -1.0  # atom 2: spin down
1.0  0.0  0.0   # atom 3: spin along x
0.0  1.0  0.0   # atom 4: spin along y
```

Each row must have exactly 3 components representing the x, y, and z components of the quantization axis for that spin. The number of rows must match the number of magnetic atoms in your system.
