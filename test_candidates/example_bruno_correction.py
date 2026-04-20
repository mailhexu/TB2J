"""Example: Bruno's correction in real-space and reciprocal-space.

Bruno's correction renormalizes the isotropic exchange J to account for the
spin moment magnitude, improving the mapping from DFT to the Heisenberg model.

Two modes are available for real-space (ExchangeCL2):
  - "fft":  exact correction via FFT R→q, Bruno formula at each q, iFFT q→R
  - "local": fast per-R-vector local approximation

For reciprocal-space (ExchangeCLQspace), the Bruno renormalization is applied
directly at each q-point before transforming back to real space.

This example:
  1. Loads SrMnO3 Wannier90 data from the test data submodule
  2. Runs four calculations: real-space (no Bruno), real-space FFT Bruno,
     real-space local Bruno, and q-space Bruno
  3. Compares the exchange coupling values across all methods

Run from the TB2J repository root:

    python test_candidates/example_bruno_correction.py

Or as a pytest test (requires E2E test data):

    pytest test_candidates/example_bruno_correction.py -v -s
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup: resolve test data directory
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TB2J_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(
    TB2J_ROOT, "tests", "data", "inputs", "2_SrMnO3_wannier", "data"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_wannier_data(path, prefix_up="abinito_w90_up", prefix_dn="abinito_w90_down"):
    """Load collinear Wannier90 data for SrMnO3."""
    from ase.io import read
    from HamiltonIO.wannier import WannierHam

    atoms = read(os.path.join(path, "abinit.in"))
    tbmodel_up = WannierHam.read_from_wannier_dir(
        path=path, prefix=prefix_up, atoms=atoms, nls=False
    )
    tbmodel_dn = WannierHam.read_from_wannier_dir(
        path=path, prefix=prefix_dn, atoms=atoms, nls=False
    )
    return atoms, (tbmodel_up, tbmodel_dn)


def run_exchange(
    ExchangeClass, atoms, tbmodels, basis, output_path, label, bruno_correction=""
):
    """Run exchange calculation with common parameters."""
    common_kwargs = dict(
        efermi=6.15,
        kmesh=[5, 5, 5],
        magnetic_elements=["Mn"],
        basis=basis,
        description=f"Bruno example - {label}",
        output_path=output_path,
        nz=10,
        nproc=1,
        bruno_correction=bruno_correction,
    )
    exchange = ExchangeClass(tbmodels=tbmodels, atoms=atoms, **common_kwargs)
    exchange.calculate_all()
    return exchange


def _fmt_R(R):
    return f"({R[0]},{R[1]},{R[2]})"


def _fmt_val(v):
    """Format a value that may be complex — show real part if imag ≈ 0."""
    import numpy as np

    v = np.real_if_close(v, tol=100)
    return f"{np.real(v):+.4f}"


def print_jdict(label, jdict, bruno_dict=None):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    if not jdict:
        print("  (empty)")
        return

    for key in sorted(jdict.keys()):
        R, i, j = key
        R_str = _fmt_R(R)
        j_iso = jdict[key] * 1000
        line = f"  R={R_str:>12s}  i={i} j={j}  J_iso={j_iso:+10.4f} meV"
        if bruno_dict is not None and key in bruno_dict:
            j_bruno = bruno_dict[key] * 1000
            diff = j_bruno - j_iso
            line += f"  J(Bruno)={_fmt_val(j_bruno):>10s}  Δ={_fmt_val(diff):>10s} meV"
        print(line)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: Test data not found at {DATA_DIR}")
        print("Initialize the test data submodule first:")
        print("  cd TB2J && ./tests/init_test_data.sh")
        sys.exit(1)

    print("Loading SrMnO3 Wannier90 data...")
    atoms, tbmodels = load_wannier_data(DATA_DIR)

    with tempfile.TemporaryDirectory() as tmpdir:
        from TB2J.exchange_qspace import ExchangeCLQspace
        from TB2J.exchangeCL2 import ExchangeCL2
        from TB2J.utils import auto_assign_basis_name

        basis_file = os.path.join(tmpdir, "assigned_basis.txt")
        basis, _ = auto_assign_basis_name(
            tbmodels[0].xred, atoms, write_basis_file=basis_file
        )

        # --- 1. Real-space, no Bruno correction ---
        print("\n[1/4] Real-space exchange (no Bruno)...")
        ex_rs = run_exchange(
            ExchangeCL2,
            atoms,
            tbmodels,
            basis,
            os.path.join(tmpdir, "rs_nobruno"),
            "realspace-no-bruno",
            bruno_correction="",
        )

        # --- 2. Real-space, Bruno correction via FFT ---
        print("[2/4] Real-space exchange (Bruno FFT)...")
        ex_rs_fft = run_exchange(
            ExchangeCL2,
            atoms,
            tbmodels,
            basis,
            os.path.join(tmpdir, "rs_bruno_fft"),
            "realspace-bruno-fft",
            bruno_correction="fft",
        )

        # --- 3. Real-space, Bruno correction via local approximation ---
        print("[3/4] Real-space exchange (Bruno local)...")
        ex_rs_local = run_exchange(
            ExchangeCL2,
            atoms,
            tbmodels,
            basis,
            os.path.join(tmpdir, "rs_bruno_local"),
            "realspace-bruno-local",
            bruno_correction="local",
        )

        # --- 4. Q-space, Bruno correction ---
        print("[4/4] Q-space exchange (Bruno)...")
        ex_qs = run_exchange(
            ExchangeCLQspace,
            atoms,
            tbmodels,
            basis,
            os.path.join(tmpdir, "qs_bruno"),
            "qspace-bruno",
            bruno_correction="fft",
        )

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  BRUNO CORRECTION COMPARISON — SrMnO3")
    print("=" * 60)

    print_jdict("Real-space (no Bruno)", ex_rs.exchange_Jdict)
    print_jdict(
        "Real-space (no Bruno) + Bruno FFT side-by-side",
        ex_rs_fft.exchange_Jdict,
        bruno_dict=ex_rs_fft.exchange_Jdict_bruno,
    )
    print_jdict(
        "Real-space (no Bruno) + Bruno local side-by-side",
        ex_rs_local.exchange_Jdict,
        bruno_dict=ex_rs_local.exchange_Jdict_bruno,
    )
    print_jdict(
        "Q-space (no Bruno) + Bruno side-by-side",
        ex_qs.exchange_Jdict,
        bruno_dict=ex_qs.exchange_Jdict_bruno,
    )

    # --- Cross-method comparison of Bruno values ---
    print(f"\n{'=' * 60}")
    print("  CROSS-METHOD BRUNO COMPARISON")
    print(f"{'=' * 60}")
    fft_bruno = ex_rs_fft.exchange_Jdict_bruno or {}
    local_bruno = ex_rs_local.exchange_Jdict_bruno or {}
    qs_bruno = ex_qs.exchange_Jdict_bruno or {}

    all_keys = sorted(set(fft_bruno) | set(local_bruno) | set(qs_bruno))
    print(f"  {'Key':<36s}  {'FFT':>12s}  {'Local':>12s}  {'Q-space':>12s}")
    print(f"  {'-' * 36}  {'-' * 12}  {'-' * 12}  {'-' * 12}")
    for key in all_keys:
        R_str = _fmt_R(key[0])
        short_key = f"R={R_str} i={key[1]} j={key[2]}"
        fft_v = fft_bruno.get(key, float("nan")) * 1000
        loc_v = local_bruno.get(key, float("nan")) * 1000
        qs_v = qs_bruno.get(key, float("nan")) * 1000
        print(
            f"  {short_key:<36s}  {_fmt_val(fft_v):>12s}  {_fmt_val(loc_v):>12s}  {_fmt_val(qs_v):>12s}  meV"
        )

    print(f"\n{'=' * 60}")
    print("  Done. Bruno correction typically shifts J values by a few meV.")
    print("  FFT and q-space methods should agree closely.")
    print("  Local approximation may differ for longer-range interactions.")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Pytest entry point (optional: run as `pytest this_file.py -v -s`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
