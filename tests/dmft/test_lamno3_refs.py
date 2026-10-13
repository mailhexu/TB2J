import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.dmft import SigInpParser

ROOT = Path(__file__).resolve().parents[2]
REF_DATA = ROOT.parent / "Refs" / "Refs" / "DMFT" / "LaMnO3_DMFT_data"


def _require_lamno3_refs():
    required = [
        REF_DATA / "sig.inp",
        REF_DATA / "dmft_params.dat",
        REF_DATA / "DMFT_mu.out",
        REF_DATA / "wannier90_hr.dat",
        REF_DATA / "wannier90.win",
        REF_DATA / "POSCAR",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"LaMnO3 DMFT reference data not available: {missing}")


def test_lamno3_siginp_static_sigma_embedding():
    _require_lamno3_refs()

    parser = SigInpParser(REF_DATA / "sig.inp")
    sigma_static, _ = parser.get_static_sigma()

    assert np.allclose(parser.mesh.real, 0.0)
    assert np.all(parser.mesh.imag > 0.0)
    assert sigma_static.shape == (2, 5, 5)
    assert parser.orbital_map["n_correlated_atoms"] == 4
    assert parser.orbital_map["spin_channels"] == [0, 1, 1, 0]

    sigma_up = np.diag(sigma_static[0]).real
    sigma_dn = np.diag(sigma_static[1]).real
    assert not np.allclose(sigma_up, sigma_dn)


@pytest.mark.skipif(
    os.environ.get("TB2J_RUN_REFS_TESTS") != "1",
    reason="set TB2J_RUN_REFS_TESTS=1 to run LaMnO3 reference integration test",
)
def test_lamno3_static_sigma_cpu_gpu_equivalence(tmp_path):
    _require_lamno3_refs()
    pytest.importorskip("jax")

    cpu_out = tmp_path / "cpu"
    gpu_out = tmp_path / "gpu"
    base_cmd = [
        sys.executable,
        str(ROOT / "TB2J" / "scripts" / "dmft2J.py"),
        "--path",
        str(REF_DATA),
        "--posfile",
        "POSCAR",
        "--prefix",
        "wannier90",
        "--dmft_file",
        str(REF_DATA / "sig.inp"),
        "--parser-type",
        "siginp",
        "--dmft-params",
        str(REF_DATA / "dmft_params.dat"),
        "--mu-file",
        str(REF_DATA / "DMFT_mu.out"),
        "--static-sigma",
        "--nspin",
        "1",
        "--magnetic-elements",
        "Mn",
        "--kmesh",
        "1",
        "1",
        "1",
        "--nz",
        "4",
        "--emin",
        "-2",
    ]

    subprocess.run(
        [*base_cmd, "--output_path", str(cpu_out)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            *base_cmd,
            "--use_gpu",
            "--e_batch_size",
            "2",
            "--output_path",
            str(gpu_out),
        ],
        check=True,
        cwd=ROOT,
    )

    with open(cpu_out / "TB2J.pickle", "rb") as handle:
        cpu = pickle.load(handle)
    with open(gpu_out / "TB2J.pickle", "rb") as handle:
        gpu = pickle.load(handle)

    assert set(cpu["exchange_Jdict"]) == set(gpu["exchange_Jdict"])
    for key, cpu_value in cpu["exchange_Jdict"].items():
        assert np.isclose(cpu_value, gpu["exchange_Jdict"][key], atol=1e-10)

    np.testing.assert_allclose(cpu["charges"], gpu["charges"], atol=1e-12)
    np.testing.assert_allclose(cpu["spinat"], gpu["spinat"], atol=1e-12)


@pytest.mark.skipif(
    os.environ.get("TB2J_RUN_REFS_TESTS") != "1",
    reason="set TB2J_RUN_REFS_TESTS=1 to run LaMnO3 reference integration test",
)
def test_lamno3_dynamic_matsubara_cpu_gpu_equivalence(tmp_path):
    _require_lamno3_refs()
    pytest.importorskip("jax")

    cpu_out = tmp_path / "cpu_dynamic"
    gpu_out = tmp_path / "gpu_dynamic"
    base_cmd = [
        sys.executable,
        str(ROOT / "TB2J" / "scripts" / "dmft2J.py"),
        "--path",
        str(REF_DATA),
        "--posfile",
        "POSCAR",
        "--prefix",
        "wannier90",
        "--dmft_file",
        str(REF_DATA / "sig.inp"),
        "--parser-type",
        "siginp",
        "--dmft-params",
        str(REF_DATA / "dmft_params.dat"),
        "--mu-file",
        str(REF_DATA / "DMFT_mu.out"),
        "--nspin",
        "1",
        "--magnetic-elements",
        "Mn",
        "--kmesh",
        "1",
        "1",
        "1",
    ]

    subprocess.run(
        [*base_cmd, "--output_path", str(cpu_out)],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            *base_cmd,
            "--use_gpu",
            "--e_batch_size",
            "64",
            "--output_path",
            str(gpu_out),
        ],
        check=True,
        cwd=ROOT,
    )

    with open(cpu_out / "TB2J.pickle", "rb") as handle:
        cpu = pickle.load(handle)
    with open(gpu_out / "TB2J.pickle", "rb") as handle:
        gpu = pickle.load(handle)

    assert set(cpu["exchange_Jdict"]) == set(gpu["exchange_Jdict"])
    for key, cpu_value in cpu["exchange_Jdict"].items():
        assert np.isclose(cpu_value, gpu["exchange_Jdict"][key], atol=1e-10)

    np.testing.assert_allclose(cpu["charges"], gpu["charges"], atol=1e-12)
    np.testing.assert_allclose(cpu["spinat"], gpu["spinat"], atol=1e-12)
