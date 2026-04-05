"""Compare q-space vs real-space exchange coupling results.

Runs both ExchangeCL2 (real-space) and ExchangeCLQspace (q-space) on the
SrMnO3 Wannier90 test data with identical parameters and verifies that the
resulting exchange_Jdict values match within numerical tolerance.

Run from the repository root:

    pytest tests/tests/test_qspace_vs_realspace.py -v

"""

from __future__ import annotations

import os
import tempfile

import pytest
from ase.io import read
from HamiltonIO.wannier import WannierHam

from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.utils import auto_assign_basis_name

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "tests", "data", "inputs", "2_SrMnO3_wannier", "data")
)


def _load_wannier_data(path, prefix_up, prefix_dn):
    atoms = read(os.path.join(path, "abinit.in"))
    tbmodel_up = WannierHam.read_from_wannier_dir(
        path=path, prefix=prefix_up, atoms=atoms, nls=False
    )
    tbmodel_dn = WannierHam.read_from_wannier_dir(
        path=path, prefix=prefix_dn, atoms=atoms, nls=False
    )
    return atoms, (tbmodel_up, tbmodel_dn)


def _run_exchange(ExchangeClass, atoms, tbmodels, basis, output_path, label):
    common_kwargs = dict(
        efermi=6.15,
        kmesh=[5, 5, 5],
        magnetic_elements=["Mn"],
        basis=basis,
        description=f"Test {label}",
        output_path=output_path,
        nz=10,
        nproc=1,
    )
    exchange = ExchangeClass(tbmodels=tbmodels, atoms=atoms, **common_kwargs)
    exchange.calculate_all()
    return exchange


def _compare_results(ex_rs, ex_qs, rtol=1e-4):
    rs_dict = ex_rs.exchange_Jdict
    qs_dict = ex_qs.exchange_Jdict
    all_keys = set(rs_dict.keys()) | set(qs_dict.keys())

    failures = []
    for key in sorted(all_keys):
        rs_val = rs_dict.get(key, None)
        qs_val = qs_dict.get(key, None)
        if rs_val is None or qs_val is None:
            failures.append(
                f"{key}: missing in {'q-space' if rs_val is None else 'real-space'}"
            )
            continue
        if abs(rs_val) < 1e-15:
            diff = abs(qs_val - rs_val)
        else:
            diff = abs(qs_val - rs_val) / abs(rs_val)
        if diff > rtol:
            failures.append(
                f"{key}: rs={rs_val:.6e} qs={qs_val:.6e} rel_diff={diff:.6e}"
            )

    return failures


@pytest.mark.skipif(
    not os.path.isdir(DATA_DIR),
    reason="SrMnO3 test data not available (submodule not initialized)",
)
def test_qspace_matches_realspace():
    """Q-space and real-space exchange couplings must agree within tolerance."""
    prefix_up = "abinito_w90_up"
    prefix_dn = "abinito_w90_down"

    atoms, tbmodels = _load_wannier_data(DATA_DIR, prefix_up, prefix_dn)

    with tempfile.TemporaryDirectory() as tmpdir:
        basis_file = os.path.join(tmpdir, "assigned_basis.txt")
        basis, _ = auto_assign_basis_name(
            tbmodels[0].xred, atoms, write_basis_file=basis_file
        )

        rs_out = os.path.join(tmpdir, "realspace")
        qs_out = os.path.join(tmpdir, "qspace")

        ex_rs = _run_exchange(ExchangeCL2, atoms, tbmodels, basis, rs_out, "realspace")
        ex_qs = _run_exchange(
            ExchangeCLQspace, atoms, tbmodels, basis, qs_out, "qspace"
        )

    failures = _compare_results(ex_rs, ex_qs)
    assert not failures, (
        f"Q-space vs real-space mismatch in {len(failures)} pairs:\n"
        + "\n".join(failures)
    )
