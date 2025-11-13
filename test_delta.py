import sys

import numpy as np

sys.path.insert(0, "/home/hexu/projects/TB2J")

from TB2J.interfaces.siesta_interface import SislParser

# Test with orth=True
parser = SislParser(
    fdf_fname="/home/hexu/projects/TB2J_examples/Siesta/bccFe/DFT/siesta.fdf",
    ispin=None,
    read_H_soc=False,
    orth=True,
)

tbmodel_up, tbmodel_dn = parser.get_model()

print(f"HR0_up shape: {tbmodel_up.H0.shape}")
print(f"HR0_dn shape: {tbmodel_dn.H0.shape}")
print(f"HR0_up[0:5, 0:5]:\n{tbmodel_up.H0[0:5, 0:5]}")
print(f"HR0_dn[0:5, 0:5]:\n{tbmodel_dn.H0[0:5, 0:5]}")

Delta = tbmodel_up.H0 - tbmodel_dn.H0
print(f"\nDelta[0:5, 0:5]:\n{Delta[0:5, 0:5]}")
print(f"\nDelta diagonal: {np.diag(Delta)}")
print(f"Delta max: {np.max(np.abs(Delta))}")
print(f"Delta has NaN: {np.any(np.isnan(Delta))}")
