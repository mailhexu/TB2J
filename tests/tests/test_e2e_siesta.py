"""E2E validation test for the SIESTA exchange path (Epic 011-4).

The legacy ``spin=None`` xfail is stale: current HamiltonIO/sisl returns a proper
collinear spin Hamiltonian for this input, so the workflow runs to completion.
This case exercises the public ``siesta2J`` entry point and checks the canonical
``SpinIO`` with the layered oracle.
"""

from __future__ import annotations

import pytest
from conftest import require_input, resolve_example
from utils.runners import run_tb2j_module
from utils.spinio_checks import check_pair_reversal, check_schema, compare_J

# Cr-Cr exchange in the SIESTA CrI3 collinear result, in meV. The same-direction
# (0,0,0) and in-plane neighbour pairs are the physical nearest-neighbour shell.
_CRI3_SIESTA_NN_J_MEV = {
    ((0, 0, 0), 1, 0): 1.1374,
    ((1, 0, 0), 1, 0): 1.1370,
}


@pytest.mark.tier2
def test_siesta_cri3_collinear(tmp_path):
    """Inventory: SIESTA exchange. Tier T2, default profile.

    Runs siesta2J on the CrI3 SIESTA input and checks the canonical SpinIO:
    schema, nearest-neighbour Cr-Cr J, and pair-reversal symmetry. Replaces the
    legacy ``spin=None`` xfail (now resolved by HamiltonIO/sisl).
    """
    data_dir = require_input(
        "inputs/4_CrI3_SIESTA_collinear/data", "SIESTA exchange", "CrI3"
    )
    args = [
        "--fdf_fname",
        str(data_dir / "siesta.fdf"),
        "--kmesh",
        "5",
        "5",
        "1",
        "--elements",
        "Cr_3d",
        "--nz",
        "50",
    ]
    sio = run_tb2j_module("TB2J.scripts.siesta2J", args, tmp_path)

    check_schema(sio)
    compare_J(sio, _CRI3_SIESTA_NN_J_MEV, tol=1e-3, unit="meV")
    check_pair_reversal(sio)


# bcc Fe nearest-neighbour J (R=(1,1,1)) from the SIESTA bccFe Hamiltonian, meV.
_BCCFE_NN_J_MEV = {
    ((1, 1, 1), 0, 0): 20.5145,
    ((-1, -1, -1), 0, 0): 20.5145,
}


@pytest.mark.tier2
def test_siesta_bccfe_collinear(tmp_path):
    """Inventory: SIESTA exchange (bcc Fe collinear). Tier T2, default profile."""
    data_dir = resolve_example("Siesta/bccFe/DFT", "SIESTA exchange", "bccFe")
    args = [
        "--fdf_fname",
        str(data_dir / "siesta.fdf"),
        "--elements",
        "Fe",
        "--kmesh",
        "5",
        "5",
        "5",
        "--nz",
        "50",
    ]
    sio = run_tb2j_module("TB2J.scripts.siesta2J", args, tmp_path)

    check_schema(sio)
    compare_J(sio, _BCCFE_NN_J_MEV, tol=1e-2, unit="meV")
    check_pair_reversal(sio)
