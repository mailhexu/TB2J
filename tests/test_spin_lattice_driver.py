"""Public driver: serialized local response vs displaced-supercell exchange."""

import numpy as np
import pytest
import test_spin_lattice_fd as FD
from ase import Atoms

from TB2J.spin_lattice_lcao import gen_exchange_spin_lattice_lcao


def run(model, atoms, basis, path, disp=None, kmesh=(1, 1, 1)):
    return gen_exchange_spin_lattice_lcao(
        tbmodel=model,
        atoms=atoms,
        disp=disp,
        basis=basis,
        efermi=FD.EFERMI,
        magnetic_elements=["Fe"],
        kmesh=kmesh,
        emin=-10,
        emax=0,
        nz=48,
        integration_method="semicircle",
        smearing=0.1,
        Rcut=1.1,
        output_path=str(path),
        Jonly=disp is None,
    )


@pytest.mark.parametrize("displaced_cell", (0, 1))
def test_serialized_local_response_matches_displaced_supercell(
    tmp_path, displaced_cell
):
    n = 17
    prim = FD._SpinorChain(ncell=1)
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.diag([1, 10, 10]), pbc=True)
    disp = FD._analytic_vertex()
    disp.Ru = (displaced_cell, 0, 0)
    result = run(
        prim,
        atoms,
        ["Fe1|orb1", "Fe1|orb1"],
        tmp_path / "analytic",
        disp=disp,
        kmesh=(n, 1, 1),
    )
    sc_atoms = Atoms(
        ["Fe"] * n,
        positions=[[i, 0, 0] for i in range(n)],
        cell=np.diag([n, 10, 10]),
        pbc=True,
    )
    basis = [f"Fe{i+1}|orb1" for i in range(n) for _ in (0, 1)]
    h = 1e-5

    class TranslatedChain(FD._SpinorChain):
        def HSE_k(self, kpt, convention=2):
            H, S, values, vectors = super().HSE_k(kpt, convention)
            perm = np.array(
                [
                    2 * ((i - displaced_cell) % n) + spin
                    for i in range(n)
                    for spin in (0, 1)
                ]
            )
            return H[np.ix_(perm, perm)], S[np.ix_(perm, perm)], values, vectors[perm]

    plus = run(TranslatedChain(tau=h, ncell=n), sc_atoms, basis, tmp_path / "plus")
    minus = run(TranslatedChain(tau=-h, ncell=n), sc_atoms, basis, tmp_path / "minus")
    key = ((0, 0, 0), 0, 1)
    finite_difference = (plus.exchange_Jdict[key] - minus.exchange_Jdict[key]) / (2 * h)
    assert abs(finite_difference) > 1e-6
    np.testing.assert_allclose(
        result.dJdx[((1, 0, 0), 0, 0)], finite_difference, rtol=2e-6, atol=1e-9
    )
    # J-only mode must not infer a displacement from a model's changing H.
    assert all(value == 0.0 for value in plus.dJdx.values())


def test_driver_rejects_missing_displacement(tmp_path):
    model = FD._SpinorChain(ncell=1)
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=np.eye(3), pbc=[True, False, False])
    with pytest.raises(ValueError, match="disp is required"):
        gen_exchange_spin_lattice_lcao(
            tbmodel=model,
            atoms=atoms,
            disp=None,
            basis=["Fe1|orb1", "Fe1|orb1"],
            efermi=FD.EFERMI,
            magnetic_elements=["Fe"],
            kmesh=(17, 1, 1),
            output_path=str(tmp_path),
        )
