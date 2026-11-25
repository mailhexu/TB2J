"""siesta2J --sparse_hk gives the same exchange as the default dense H(R) path.

With --sparse_hk, HamiltonIO builds H(k) and S(k) directly from the sparse
SIESTA matrices (SislParser(sparse=True)) and the dense real-space arrays are
not stored. The exchange parameters must not change beyond rounding (H(k) is
summed in a different order), checked against 1e-6 of the largest |J|. Skipped
if the installed HamiltonIO has no sparse option.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import resolve_example
from utils.runners import run_tb2j_module

pytest.importorskip("sisl")
from HamiltonIO.siesta.sisl_wrapper import SislParser  # noqa: E402

pytestmark = pytest.mark.skipif(
    "sparse" not in inspect.signature(SislParser).parameters,
    reason="HamiltonIO without SislParser(sparse=...)",
)


@pytest.mark.tier2
@pytest.mark.parametrize("orth", [False, True])
def test_siesta_bccfe_sparse_hk_matches_dense(tmp_path, orth):
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
    ] + (["--orth"] if orth else [])
    dense = run_tb2j_module("TB2J.scripts.siesta2J", args, tmp_path / "dense")
    sparse = run_tb2j_module(
        "TB2J.scripts.siesta2J", args + ["--sparse_hk"], tmp_path / "sparse"
    )

    assert set(sparse.exchange_Jdict) == set(dense.exchange_Jdict)
    Jmax = max(abs(J) for J in dense.exchange_Jdict.values())
    for key, J in dense.exchange_Jdict.items():
        np.testing.assert_allclose(
            sparse.exchange_Jdict[key], J, rtol=0, atol=1e-6 * Jmax
        )
