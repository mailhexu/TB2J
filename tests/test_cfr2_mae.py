"""Actual MAEGreen response and output against finite-T spectral energies.

MathDoc: tb2j/cfr2-integration-contract (fixed-mu grand potential).
"""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from scipy.special import expit
from test_cfr2_exchange import ORIGIN, execution_context

from TB2J.MAEGreen import MAEGreen


def mae_calculator(kind="cpu", mu=0.0):
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[10, 10, 10])
    metric = 1.4 * np.eye(2)
    h0 = 1.4 * np.diag([mu - 0.2, mu + 0.2])
    soc = 1.4 * np.array([[0, 0.1], [0.1, 0]])
    model = LCAOHamiltonian(
        HR=np.array([h0]),
        SR=np.array([metric]),
        Rlist=np.array([ORIGIN]),
        nbasis=2,
        atoms=atoms,
        nspin=2,
        orth=False,
        split_soc=True,
        HR_nosoc=np.array([h0]),
        HR_soc=np.array([soc]),
        so_strength=0.0,
    )
    cls = MAEGreen
    if kind != "cpu":
        from TB2J.gpu.mae_green_gpu import MAEGreenGPU

        cls = MAEGreenGPU
    return cls(
        tbmodels=model,
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|d|up", "Fe1|d|down"],
        kmesh=[1, 1, 1],
        efermi=mu,
        nz=80,
        nproc=1,
        Rcut=2,
        angles=[[0.0, np.pi / 2], [0.0, 0.0]],
    )


def spectral_response():
    # The two rotated vertices are v*sigma_x and -v*sigma_z. The
    # partial-fraction/equal-energy limits are verified by the MathDoc.
    levels = np.array([-0.2, 0.2])
    f = expit(-levels / (kB * 600))
    off_diagonal = 0.1**2 * (f[0] - f[1]) / (levels[0] - levels[1])
    diagonal = -0.5 * 0.1**2 * np.sum(f * (1 - f)) / (kB * 600)
    return np.array([off_diagonal, diagonal])


@pytest.mark.parametrize("mu", [0.0, 6.3])
@pytest.mark.parametrize(
    "kind",
    [
        "cpu",
        pytest.param("sequential_gpu", marks=pytest.mark.gpu),
        pytest.param("batched_gpu", marks=pytest.mark.gpu),
    ],
)
def test_actual_mae_directional_response_and_mev_output(kind, mu, tmp_path):
    with execution_context(kind):
        calc = mae_calculator(kind, mu)
        options = {}
        if kind != "cpu":
            options = dict(vectorize_energy=kind == "batched_gpu", e_batch_size=7)
        calc.run(output_path=tmp_path, with_eigen=False, **options)
    expected = spectral_response()
    np.testing.assert_allclose(calc.es, expected, atol=1e-10, rtol=0)
    # Actual published numeric column: raw second-order corrections, not
    # silently re-zeroed MAE. Positive E(direction)-E(reference) is harder.
    output = np.loadtxt(tmp_path / "MAE.dat")[:, 2]
    np.testing.assert_allclose(output, expected * 1000, atol=5.1e-9, rtol=0)
    assert abs((output[1] - output[0]) / 1000 - (expected[1] - expected[0])) < 1e-10
    assert output[1] - output[0] > 0
    # Disabled decompositions must not be published as computed zero data.
    assert not (tmp_path / "MAE_matrix.dat").exists()
    assert not (tmp_path / "MAE_orb.dat").exists()
