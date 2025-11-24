# Oiju_FD2.py - Unified Interface for Spin-Phonon Coupling

## Overview
`Oiju_FD2.py` provides a unified interface matching `Oiju_epw2.py` but using the finite difference implementation from `Oiju_FD.py`.

## Purpose
- Allows switching between perturbation theory (EPW) and finite difference methods using the same API
- Useful for comparing or validating results between the two approaches
- Maintains compatibility with code written for `Oiju_epw2.py`

## Key Differences from Original Files

### Interface (from Oiju_epw2.py)
Uses parameter names:
- `idisp` - displacement/mode index
- `Ru` - R vector for displacement
- `epw_up_path`, `epw_prefix_up` - EPW file locations
- `Rcut` - cutoff radius (accepted but not used in FD method)

### Implementation (from Oiju_FD.py)
Uses finite difference approach:
- Generates distorted tight-binding model using `epw_to_dHdx()` and `generate_TB_with_distortion()`
- Computes exchange for both original and distorted structures
- Saves results in separate directories: `original/` and `idisp{n}_Ru{x}_{y}_{z}/`

## Parameter Mapping

| Oiju_epw2.py | Oiju_FD.py | Oiju_FD2.py | Notes |
|--------------|------------|-------------|-------|
| `idisp` | `imode` | `idisp` | Displacement/phonon mode index |
| `Ru` | `Rp` | `Ru` | R vector for phonon |
| `epw_up_path` | `epw_path` | `epw_up_path` | Path to EPW files |
| `epw_prefix_up` | `epw_prefix` | `epw_prefix_up` | EPW file prefix |
| `Rcut` | N/A | `Rcut` | Accepted but not used in FD |
| N/A | `supercell_matrix` | `supercell_matrix` | Added as optional parameter |
| N/A | `amplitude` | `amplitude` | Added as optional parameter |

## Additional Parameters
These parameters are specific to the finite difference method:
- `supercell_matrix` (default: identity) - Supercell matrix for calculations
- `amplitude` (default: 0.01) - Displacement amplitude for finite difference

## Usage Example

```python
from TB2J.Oiju_FD2 import gen_exchange_Oiju_epw
import numpy as np

# Same interface as Oiju_epw2.py
gen_exchange_Oiju_epw(
    path="/path/to/wannier90",
    colinear=True,
    posfile='scf.pwi',
    prefix_up="SrMnO3_up",
    prefix_dn="SrMnO3_down",
    epw_up_path='/path/to/epw',
    epw_prefix_up='SrMnO3',
    idisp=0,                    # displacement index
    Ru=(0, 0, 0),              # R vector
    supercell_matrix=np.eye(3) * 3,  # 3x3x3 supercell
    amplitude=0.01,            # displacement amplitude
    efermi=10.67,
    magnetic_elements=['Mn'],
    kmesh=[3, 3, 3],
    emin=-7.33,
    emax=0.0,
    nz=70,
    output_path=f"FD_results"
)
```

## Output Structure
```
output_path/
├── original/                           # Exchange for undistorted structure
│   ├── exchange.txt
│   └── ...
└── idisp{n}_Ru{x}_{y}_{z}/            # Exchange for distorted structure
    ├── exchange.txt
    └── ...
```

## Comparison: Perturbation vs Finite Difference

| Method | File | Approach | Pros | Cons |
|--------|------|----------|------|------|
| Perturbation | `Oiju_epw2.py` | EPW matrix elements | Direct dJ/dx, faster | Requires EPW calculation |
| Finite Difference | `Oiju_FD2.py` | Distorted structures | Conceptually simple | Two full calculations, numerical errors |

## When to Use Which

### Use Oiju_epw2.py (Perturbation Theory) when:
- You have EPW electron-phonon matrix elements available
- You need direct derivatives (dJ/dx)
- Computational efficiency is important
- You want a single calculation per mode

### Use Oiju_FD2.py (Finite Difference) when:
- You want to validate perturbation theory results
- EPW calculation is difficult or unavailable
- You prefer conceptual simplicity over efficiency
- You want to study finite-amplitude effects

## Implementation Notes

### Spin-Resolved dH/dx Merging
The `merge_dHdx_spin()` function combines spin-up and spin-down dH/dx matrices:
```python
def merge_dHdx_spin(dH_up, dH_dn):
    """
    Creates block diagonal structure:
    [[dH_up,  0   ],
     [0,      dH_dn]]
    """
```

Key features:
- Merges all R vectors from both spin channels
- Creates spin-resolved matrix with spin-up in even indices, spin-down in odd indices
- Preserves Wigner-Seitz degeneracy weights (wsdeg) from input channels
- Returns unified `dHdx` object compatible with `generate_TB_with_distortion()`

### Non-collinear Case
Currently only the collinear case is implemented. Non-collinear support will raise `NotImplementedError`.

### EPW Parameters
- Both spin-up and spin-down EPW files are required for proper spin-resolved calculations
- `epw_up_path` and `epw_prefix_up` specify spin-up EPW files
- `epw_down_path` and `epw_prefix_dn` specify spin-down EPW files
- The dH/dx matrices from both channels are merged using `merge_dHdx_spin()`

### Supercell
If `supercell_matrix` is not provided, defaults to identity (no supercell).
