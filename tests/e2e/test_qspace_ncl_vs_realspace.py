"""Compare q-space vs real-space non-collinear (NCL) exchange coupling results.

Runs both ExchangeNCL (real-space) and ExchangeNCLQspace (q-space) on the
CrI3 SOC Wannier90 test data with identical parameters and verifies that the
resulting exchange_Jdict, Jani, and DMI values match within numerical tolerance.

Run from the repository root:

    pytest tests/e2e/test_qspace_ncl_vs_realspace.py -v

"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase.io import read
from HamiltonIO.wannier import WannierHam

from TB2J.exchange import ExchangeNCL
from TB2J.exchange_qspace import ExchangeNCLQspace
from TB2J.utils import auto_assign_basis_name

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.abspath(
    os.path.join(ROOT_DIR, "data", "inputs", "3_CrI3_wannier_SOC", "data", "z")
)


def _load_spinor_data(path, prefix="wannier90"):
    """Load spinor (non-collinear) Wannier90 data."""
    atoms = read(os.path.join(path, "POSCAR"))
    tbmodel = WannierHam.read_from_wannier_dir(
        path=path, prefix=prefix, atoms=atoms, groupby="orbital", nls=True
    )
    return atoms, tbmodel


def _run_exchange_ncl(ExchangeClass, atoms, tbmodel, basis, output_path, label):
    """Run an NCL exchange calculation with common parameters."""
    common_kwargs = dict(
        efermi=-0.86,
        kmesh=[3, 3, 1],
        magnetic_elements=["Cr"],
        basis=basis,
        description=f"Test NCL {label}",
        output_path=output_path,
        nz=10,
        nproc=1,
    )
    exchange = ExchangeClass(tbmodels=tbmodel, atoms=atoms, **common_kwargs)
    exchange.calculate_all()
    return exchange


def _compare_ncl_results(ex_rs, ex_qs, rtol=1e-4):
    """Compare NCL results: Jiso, Jani, DMI."""
    failures = []

    # --- Compare exchange_Jdict (isotropic J) ---
    rs_J = ex_rs.exchange_Jdict
    qs_J = ex_qs.exchange_Jdict
    all_J_keys = set(rs_J.keys()) | set(qs_J.keys())
    for key in sorted(all_J_keys):
        rs_val = rs_J.get(key, None)
        qs_val = qs_J.get(key, None)
        if rs_val is None or qs_val is None:
            failures.append(
                f"Jiso {key}: missing in {'q-space' if rs_val is None else 'real-space'}"
            )
            continue
        if abs(rs_val) < 1e-15:
            diff = abs(qs_val - rs_val)
        else:
            diff = abs(qs_val - rs_val) / abs(rs_val)
        if diff > rtol:
            failures.append(
                f"Jiso {key}: rs={rs_val:.6e} qs={qs_val:.6e} rel_diff={diff:.6e}"
            )

    # --- Compare Jani (anisotropic exchange, 3x3 tensor) ---
    rs_Jani = ex_rs.Jani
    qs_Jani = ex_qs.Jani
    all_Jani_keys = set(rs_Jani.keys()) | set(qs_Jani.keys())
    for key in sorted(all_Jani_keys):
        rs_val = rs_Jani.get(key, None)
        qs_val = qs_Jani.get(key, None)
        if rs_val is None or qs_val is None:
            failures.append(
                f"Jani {key}: missing in {'q-space' if rs_val is None else 'real-space'}"
            )
            continue
        diff_mat = np.abs(qs_val - rs_val)
        ref_norm = np.max(np.abs(rs_val))
        if ref_norm < 1e-15:
            max_diff = np.max(diff_mat)
        else:
            max_diff = np.max(diff_mat) / ref_norm
        if max_diff > rtol:
            failures.append(
                f"Jani {key}: max_rel_diff={max_diff:.6e}\n"
                f"  rs={rs_val}\n  qs={qs_val}"
            )

    # --- Compare DMI (Dzyaloshinskii-Moriya interaction, 3-vector) ---
    rs_DMI = ex_rs.DMI
    qs_DMI = ex_qs.DMI
    all_DMI_keys = set(rs_DMI.keys()) | set(qs_DMI.keys())
    for key in sorted(all_DMI_keys):
        rs_val = rs_DMI.get(key, None)
        qs_val = qs_DMI.get(key, None)
        if rs_val is None or qs_val is None:
            failures.append(
                f"DMI {key}: missing in {'q-space' if rs_val is None else 'real-space'}"
            )
            continue
        diff_vec = np.abs(qs_val - rs_val)
        ref_norm = np.max(np.abs(rs_val))
        if ref_norm < 1e-15:
            max_diff = np.max(diff_vec)
        else:
            max_diff = np.max(diff_vec) / ref_norm
        if max_diff > rtol:
            failures.append(
                f"DMI {key}: max_rel_diff={max_diff:.6e}\n"
                f"  rs={rs_val}\n  qs={qs_val}"
            )

    return failures


@pytest.mark.skipif(
    not os.path.isdir(DATA_DIR),
    reason="CrI3 SOC test data not available (submodule not initialized)",
)
def test_qspace_ncl_matches_realspace():
    """Q-space and real-space NCL exchange couplings must agree within tolerance."""
    atoms, tbmodel = _load_spinor_data(DATA_DIR)

    with tempfile.TemporaryDirectory() as tmpdir:
        basis_file = os.path.join(tmpdir, "assigned_basis.txt")
        basis, _ = auto_assign_basis_name(
            tbmodel.xred, atoms, write_basis_file=basis_file
        )

        rs_out = os.path.join(tmpdir, "realspace")
        qs_out = os.path.join(tmpdir, "qspace")

        ex_rs = _run_exchange_ncl(
            ExchangeNCL, atoms, tbmodel, basis, rs_out, "realspace"
        )
        ex_qs = _run_exchange_ncl(
            ExchangeNCLQspace, atoms, tbmodel, basis, qs_out, "qspace"
        )

    failures = _compare_ncl_results(ex_rs, ex_qs)
    assert not failures, (
        f"Q-space vs real-space NCL mismatch in {len(failures)} items:\n"
        + "\n".join(failures)
    )
