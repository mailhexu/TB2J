"""E2E validation tests for the Wannier90 exchange path (Epic 011).

Each case runs the public ``wann2J`` entry point on governed stored input, loads
the canonical ``SpinIO``, and applies the layered oracle (ADR-001/005): schema,
selected toleranced quantities, and independent invariants. This replaces the
legacy full-text ``exchange.out`` comparison in ``test_e2e_tb2j.py``.

Reference provenance:
- SrMnO3 collinear: J_iso values are byte-identical to the pre-ws-weights
  reference (scheme 1, global ndegen); the only legacy diff was added
  presentation (WS-interpolation line + "Combined J tensor" blocks). So the
  stored reference values are scientifically unchanged.
"""

from __future__ import annotations

import pytest
from utils.runners import run_tb2j_module
from utils.spinio_checks import (
    check_dmi_antisymmetry,
    check_exchange_out_section,
    check_jani_hermiticity,
    check_pair_reversal,
    check_schema,
    compare_J,
)

# Nearest-neighbour SrMnO3 Mn-Mn exchange, in meV (stored values are in eV;
# compare_J converts meV -> eV before comparing). Verified against the stored
# reference; unchanged by the ws-weights epic for this scheme-1 collinear case.
_SRMNO3_NN_J_MEV = {
    ((0, 0, 1), 0, 0): -6.7855,
    ((0, 1, 0), 0, 0): -6.7845,
    ((1, 0, 0), 0, 0): -6.7845,
}


@pytest.mark.tier2
def test_wannier_srmno3_collinear(tmp_path):
    """Inventory: Wannier90 collinear exchange. Tier T2, default profile.

    Runs wann2J on the ABINIT-derived SrMnO3 Wannier input and checks the
    canonical SpinIO: schema, nearest-neighbour J values, and pair-reversal
    symmetry, plus the WS-interpolation description in exchange.out.
    """
    from conftest import require_input

    data_dir = require_input(
        "inputs/2_SrMnO3_wannier/data", "Wannier90 collinear", "SrMnO3"
    )
    args = [
        "--path",
        str(data_dir),
        "--posfile",
        "abinit.in",
        "--efermi",
        "6.15",
        "--kmesh",
        "5",
        "5",
        "5",
        "--nz",
        "50",
        "--elements",
        "Mn",
        "--prefix_up",
        "abinito_w90_up",
        "--prefix_down",
        "abinito_w90_down",
    ]
    sio = run_tb2j_module("TB2J.scripts.wann2J", args, tmp_path)

    check_schema(sio)
    # Nearest-neighbour J, tight tolerance (meV input -> eV comparison).
    compare_J(sio, _SRMNO3_NN_J_MEV, tol=5e-5, unit="meV")
    check_pair_reversal(sio)
    # Focused exchange.out contract: the WS-interpolation scheme is recorded.
    check_exchange_out_section(tmp_path / "exchange.out", "Wannier90 WS interpolation")


# Cr-Cr in-plane nearest-neighbour isotropic exchange in the merged CrI3 SOC
# result, in meV. The merge runs on the stored x/y/z direction results with the
# current TB2J tensor-reconstruction logic; the small anisotropy drift vs the
# pre-ws-weights reference is the documented ws-weights correctness change.
# J_iso is stable; the merged tensor invariants (DMI antisymmetry, Jani
# Hermiticity, pair reversal) are the scientific checks.
_CRI3_NN_J_MEV = {
    ((1, 0, 0), 0, 0): 0.6877,
    ((1, 0, 0), 1, 1): 0.6877,
}


@pytest.mark.tier2
def test_wannier_cri3_soc_merge(tmp_path):
    """Inventory: Wannier90 spinor/SOC exchange + merge. Tier T2, default profile.

    Merges the stored CrI3 x/y/z spinor-direction results via the public
    ``TB2J_merge`` entry point and checks the merged canonical SpinIO: schema,
    nearest-neighbour J, pair reversal, DMI antisymmetry, and Jani Hermiticity
    (the tensor-reconstruction invariants that the text comparison obscured).
    """
    from conftest import require_input

    ref_root = require_input(
        "tests/3_CrI3_wannier_SOC/refs", "Wannier90 SOC merge", "CrI3"
    )
    args = [
        "-T",
        "structure",
        str(ref_root / "TB2J_results_x"),
        str(ref_root / "TB2J_results_y"),
        str(ref_root / "TB2J_results_z"),
    ]
    sio = run_tb2j_module("TB2J.scripts.TB2J_merge", args, tmp_path)

    check_schema(sio)
    compare_J(sio, _CRI3_NN_J_MEV, tol=1e-2, unit="meV")
    check_pair_reversal(sio)
    check_dmi_antisymmetry(sio)
    check_jani_hermiticity(sio)
