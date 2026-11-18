"""E2E ecosystem validation: ABINIT + abinao projector handoff (Epic 013-1).

T3 ecosystem case (opt-in profile): runs the maintained ABINIT + abinao PAO
projector -> TB2J exchange path on the primitive two-atom FeO input, reusing the
stored ABINIT WFK/VXC (no DFT rerun). Validates the full producer -> exchange
contract and the converged Fe-Fe exchange.

The FeO calculation is unconstrained collinear PBE; qualification of the
converged magnetic state is recorded separately (the exchange itself is the
contract checked here). Depends on the ``abinao`` package; skips if absent or if
the governed input is unavailable.
"""

from __future__ import annotations

import os
import re

import pytest
from conftest import resolve_example

# abinao is an ecosystem-only dependency.
pytest.importorskip("abinao")

# Fe-Fe nearest-neighbour exchange in the ABINIT+abinao FeO result (meV), at the
# ~3.06 A rocksalt NN distance. Verified against the supplied reference.
_FEO_NN_J_MEV = 7.5753
_FEO_NN_DIST_A = 3.065


@pytest.mark.ecosystem
def test_abinit_abinao_feo(tmp_path):
    """Inventory: ABINIT + abinao projector handoff. Tier T3, ecosystem profile."""
    from abinao.exchange import gen_exchange_from_orbitals

    data = resolve_example("ABINIT/FeO", "ABINIT+abinao FeO", "FeO")
    gen_exchange_from_orbitals(
        data / "feo_abinaoo_WFK.nc",
        data / "feo_abinaoo_VXC.nc",
        [str(data / "Fe.upf"), str(data / "O.upf")],
        output_path=str(tmp_path),
        magnetic_elements=["Fe"],
        Rcut=10.0,
        nz=30,
        smearing_eV=0.05,
        operator_component="delta_total",
        population_mode="projector",
    )

    txt = (tmp_path / "exchange.out").read_text()
    # Parse "Fe1 Fe1 (Rx,Ry,Rz) Jval (x,y,z) dist" rows; take the NN shell.
    rows = re.findall(r"Fe1\s+Fe1\s+\([^)]*\)\s+(-?[\d.]+)\s+\([^)]*\)\s+([\d.]+)", txt)
    assert rows, "no Fe-Fe exchange rows found in exchange.out"
    nn = [float(j) for j, d in rows if abs(float(d) - _FEO_NN_DIST_A) < 0.01]
    assert nn, f"no Fe-Fe NN shell at ~{_FEO_NN_DIST_A} A found"
    # All NN pairs share the same J; check the mean against the reference.
    mean_j = sum(nn) / len(nn)
    assert (
        abs(mean_j - _FEO_NN_J_MEV) < 1e-2
    ), f"Fe-Fe NN J {mean_j:.4f} meV vs expected {_FEO_NN_J_MEV} meV"


# Mn-Mn exchange in the SrMnO3 spin-phonon k222q222 undisplaced (idisp=0) result,
# meV. Fermi energy 11.26 eV (maintainer-provided); emin=-7.336, emax=0 relative
# to Fermi. AFM (negative) as expected for SrMnO3. ~13 s on CPU at kmesh 2x2x2.
_SRMNO3_SPINPHON_J_MEV = {
    ((0, 0, 1), 0, 0): -16.9083,
    ((0, 0, -1), 0, 0): -16.9083,
}


@pytest.mark.ecosystem
def test_spinphon_srmno3_k222q222(tmp_path):
    """Inventory: spin-phonon exchange (SrMnO3 k222q222, idisp=0). T3, ecosystem.

    Runs the TB2J spin-phonon path (Wannier TB + EPW electron-phonon) on the
    curated nk=2/nq=2 SrMnO3 data and checks the Mn-Mn exchange. The data lives
    outside the examples tree; set TB2J_SPINPHON_DIR to its root (defaults to the
    workstation path). Skips if absent.
    """
    from utils.spinio_checks import check_pair_reversal, check_schema, compare_J

    sp_dir = os.environ.get(
        "TB2J_SPINPHON_DIR",
        "/home_phythema/hexu/spinphon/2025-10-02_newdata/k222q222",
    )
    if not os.path.isdir(sp_dir):
        pytest.skip(f"spin-phonon data not found: TB2J_SPINPHON_DIR={sp_dir}")

    # Oiju_epw3 has no CLI __main__; call the public function directly.
    from TB2J.io_exchange.io_exchange import SpinIO
    from TB2J.Oiju_epw3 import gen_exchange_Oiju_epw3

    gen_exchange_Oiju_epw3(
        path=sp_dir,
        colinear=True,
        posfile="scf.pwi",
        prefix_up="up/SrMnO3",
        prefix_dn="down/SrMnO3.down",
        epw_up_path=f"{sp_dir}/up",
        epw_down_path=f"{sp_dir}/down",
        epw_prefix_up="epmat",
        epw_prefix_dn="epmat",
        idisp=0,
        Ru=(0, 0, 0),
        Rcut=8,
        efermi=11.26,
        magnetic_elements=["Mn"],
        kmesh=[2, 2, 2],
        emin=-7.336,
        emax=0.0,
        nz=70,
        np=1,
        output_path=str(tmp_path),
        use_gpu=False,
    )
    sio = SpinIO.load_pickle(str(tmp_path))

    check_schema(sio)
    compare_J(sio, _SRMNO3_SPINPHON_J_MEV, tol=1e-2, unit="meV")
    check_pair_reversal(sio)
