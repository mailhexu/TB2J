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


def _assert_jdict_close(cpu, gpu, atol_mev=1e-1):
    """Assert two SpinIO results agree on isotropic J within ``atol_mev`` (meV).

    GPU runs in float32, so a loose absolute tolerance (0.1 meV) is used.
    """
    cj, gj = cpu.exchange_Jdict, gpu.exchange_Jdict
    assert set(cj) == set(gj), "CPU and GPU J-dict key sets differ"
    worst = 0.0
    for key in cj:
        d = abs(cj[key] - gj[key]) * 1e3  # Hartree -> meV
        worst = max(worst, d)
    assert worst < atol_mev, f"max |J_cpu - J_gpu| = {worst:.4f} meV > {atol_mev}"


@pytest.mark.gpu
def test_siesta_bccfe_collinear_gpu_matches_cpu(tmp_path):
    """Collinear SIESTA exchange: the GPU path (ExchangeCL2GPU) must match CPU.

    Guards the ``_compute_collinear_A_batch`` einsum regression (it contracted
    over the orbital index instead of an element-wise product, giving ~52 meV
    vs ~17.6 meV). bccFe is governed test data, so this runs in CI when JAX is
    available.
    """
    pytest.importorskip("jax")
    data_dir = resolve_example("Siesta/bccFe/DFT", "SIESTA GPU exchange", "bccFe")
    common = [
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
    cpu = run_tb2j_module("TB2J.scripts.siesta2J", common, tmp_path / "cpu")
    gpu = run_tb2j_module(
        "TB2J.scripts.siesta2J", common + ["--use_gpu"], tmp_path / "gpu"
    )
    _assert_jdict_close(cpu, gpu)


@pytest.mark.gpu
def test_siesta_bccfe_soc_gpu_matches_cpu(tmp_path):
    """Non-collinear (SOC) SIESTA exchange: the GPU path (ExchangeNCLGPU),
    including orbital decomposition, must match CPU.

    Uses the bccFe SOC dataset (spin-orbit, double-zeta d + 4p polarization)
    resolved via ``resolve_example`` from ``tests/data`` or
    ``$TB2J_EXAMPLES_DIR``. With ``Fe_3d`` this exercises both element selection
    (19 spatial orbitals -> 5 d-groups) and zeta contraction (Z1+Z2 -> 1 group)
    on the GPU, plus CPU/GPU equivalence. This is the only e2e coverage for the
    non-collinear GPU path.
    """
    pytest.importorskip("jax")
    data_dir = resolve_example(
        "Siesta/bccFe_SOC/DFT", "SIESTA SOC GPU exchange", "bccFe_SOC"
    )
    common = [
        "--fdf_fname",
        str(data_dir / "siesta.fdf"),
        "--elements",
        "Fe_3d",
        "--kmesh",
        "5",
        "5",
        "5",
        "--nz",
        "50",
        "--rcut",
        "6",
        "--orb_decomposition",
    ]
    cpu = run_tb2j_module("TB2J.scripts.siesta2J", common, tmp_path / "cpu")
    gpu = run_tb2j_module(
        "TB2J.scripts.siesta2J", common + ["--use_gpu"], tmp_path / "gpu"
    )
    _assert_jdict_close(cpu, gpu)
    # Orbital-resolved output must be zeta-contracted and element-selected
    # (Fe_3d -> 5 groups), independent of CPU/GPU agreement. This guards the
    # regression where the GPU path skipped simplify_orbital_contributions.
    co, go = cpu.Jiso_orb, gpu.Jiso_orb
    expected = ("3dxy", "3dyz", "3dz2", "3dxz", "3dx2-y2")
    assert cpu.orbital_names[0] == expected, cpu.orbital_names[0]
    sample = next(iter(co.values()))
    assert sample.shape == (5, 5), f"Jiso_orb not 5x5: {sample.shape}"
    # And CPU must agree with GPU on the orbital matrices.
    assert set(co) == set(go)
    worst = max(abs(co[k] - go[k]).max() for k in co) * 1e3
    assert worst < 1e-1, f"max |Jiso_orb cpu-gpu| = {worst:.4f} meV"
