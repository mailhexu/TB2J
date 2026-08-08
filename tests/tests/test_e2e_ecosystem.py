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
