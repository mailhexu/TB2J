"""E2E validation test for the ABACUS exchange path (Epic 012-2).

Exercises the public ``abacus2J`` entry point on the governed ABACUS Fe
collinear input (stored H/S sparse output; no DFT run required) and checks the
canonical ``SpinIO`` with the layered oracle.

Data provenance: ``~/projects/TB2J_examples/Abacus/Fe_no_SOC`` (~39 MB,
self-contained). Resolved via :func:`conftest.resolve_example` so the test runs
locally against the examples tree and against the curated ``tests/data`` copy
once present; skips cleanly otherwise (ADR-006). A reduced k-mesh keeps the
default profile fast while still validating the public workflow end to end.
"""

from __future__ import annotations

import pytest
from conftest import resolve_example
from utils.runners import run_tb2j_module
from utils.spinio_checks import check_pair_reversal, check_schema, compare_J

# bcc Fe nearest-neighbour J (R=(1,1,1), the (1/2,1/2,1/2) shell) at kmesh 3x3x3,
# in meV. Only the (i=0,j=1) pair and its reversal carry the NN value; the
# (1,0) key at the same +R is a different (smaller) shell.
_FE_NN_J_MEV = {
    ((1, 1, 1), 0, 1): 50.7138,
    ((-1, -1, -1), 1, 0): 50.7138,
}


@pytest.mark.tier2
def test_abacus_fe_collinear(tmp_path):
    """Inventory: ABACUS exchange and split SOC (collinear). Tier T2, default."""
    data_dir = resolve_example("Abacus/Fe_no_SOC/DFT", "ABACUS exchange", "Fe")
    args = [
        "--path",
        str(data_dir),
        "--suffix",
        "Fe",
        "--elements",
        "Fe",
        "--kmesh",
        "3",
        "3",
        "3",
    ]
    sio = run_tb2j_module("TB2J.scripts.abacus2J", args, tmp_path)

    check_schema(sio)
    compare_J(sio, _FE_NN_J_MEV, tol=1e-2, unit="meV")
    check_pair_reversal(sio)
