### Cut exchange parameters in a supercell calculation into a unitcell calculation. 

Sometimes the DFT calculation has to be done in a supercell to describe the magnetic-ground state. The exchange parameters can be written in the unitcell to make the analysis simpler.  Another case is when the symmetry of the magnetic sublattice is higher than the full structure, thus the magnetic model can be simpified in the same way. 

TB2J provide a tool to cut the magnetic parameters inside a unitcell. Besides the TB2J_results in the supercell, one has to provide the sc_matrix which is the matrix that transform the unitcell to the suprecell. The new cell will put the origin at one of the atom with the index of origin_atom_id. 

With a python script (eg. "run_cut_cell.py") like below, we can 

``` python
import numpy as np
from TB2J.cut_cell import cut_cell
cut_cell(path='TB2J_results',
    output_path='TB2J_results_cutted',
    sc_matrix=np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]), 
    origin_atom_id=0, 
    thr=1e-5)
```

The parameters in this function are:
 - path: the original TB2J_results path
 - output_path: the output path.
 - sc_matrix: the matrix which maps the primitive cell to supercell.
 - origin: the origin of the primitive cell.
 - thr: the atoms which the reduced position is within -thr to 1.0+thr are considered as inside the primitive atoms

Then we can run:

```bash
python run_cut_cell.py

```

The result will be written to the directory "TB2J_results_cutted".


There are some limitations currently:
- It is a cut of the original supercell. If the unitcell is not inside the suprecell, then there could be atoms missing. 
- Some atoms at the primtive cell boundary is not properly treated. 

This function should be used with care, as this cut is not always valid. The exchange parameters does not necessarily fit into the unitcell. 
