"""Absolute spectral references through actual localized exchange consumers."""

from contextlib import nullcontext

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from scipy.linalg import eigh
from scipy.special import expit

from TB2J.exchange import ExchangeNCL
from TB2J.exchange_pert2 import ExchangePert2
from TB2J.exchangeCL2 import ExchangeCL2

ORIGIN = (0, 0, 0)
PAIR = (ORIGIN, 0, 1)
UP = np.array([[-0.3, 0.12], [0.12, -0.25]])
DOWN = np.array([[0.3, 0.12], [0.12, 0.25]])
DISPLACEMENT = np.array([[0, 0.07], [0.07, 0]])


def spectral_exchange(up=UP, down=DOWN):
    """Complete spectral sum, independent of contour and Green implementation.

    Partial fractions are verified in cfr2_integration_contract.py. The
    collinear kernel owns 1/(4*pi); its extracted J has no extra spin factor.
    """
    eu, vu = np.linalg.eigh(up)
    ed, vd = np.linalg.eigh(down)
    fu, fd = expit(-eu / (kB * 600)), expit(-ed / (kB * 600))
    residues = np.outer(vu[0] * vu[1].conj(), vd[1] * vd[0].conj())
    divided_difference = (fu[:, None] - fd[None, :]) / (eu[:, None] - ed[None, :])
    delta = np.diag(up - down)
    return (-delta[0] * delta[1] * np.sum(residues * divided_difference) / 4).real


class DimerDisplacement:
    """One-cell hopping derivative supplied through the EP mode input API."""

    Rqdict = {ORIGIN: 0}
    Rkdict = {ORIGIN: 0}

    def get_epmat_RgRk_two_spin(self, Rq, Rk, avg=False):
        assert Rq == Rk == ORIGIN
        return DISPLACEMENT


def calculator(kind, **options):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])

    def model(h, nspin=1):
        return LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([np.eye(len(h))]),
            Rlist=np.array([ORIGIN]),
            nbasis=len(h),
            atoms=atoms,
            nspin=nspin,
            orth=True,
        )

    parameters = dict(
        atoms=atoms,
        magnetic_elements=["Fe"],
        kmesh=[1, 1, 1],
        efermi=0,
        nz=80,
        nproc=1,
        Rcut=2,
        orb_decomposition=True,
    )
    parameters.update(options)
    classes = {"cl": ExchangeCL2, "ncl": ExchangeNCL, "pert": ExchangePert2}
    if kind.endswith("gpu"):
        from TB2J.gpu.exchange_ncl_gpu import ExchangeNCLGPU
        from TB2J.gpu.exchange_pert2_gpu import ExchangePert2GPU
        from TB2J.gpu.exchangeCL_gpu import ExchangeCL2GPU

        classes.update(
            cl_gpu=ExchangeCL2GPU, ncl_gpu=ExchangeNCLGPU, pert_gpu=ExchangePert2GPU
        )
    if kind.startswith("cl"):
        parameters["basis"] = ["Fe1|d", "Fe2|d"]
        tbmodels = (model(UP), model(DOWN))
    else:
        spinor = np.zeros((4, 4))
        spinor[::2, ::2], spinor[1::2, 1::2] = UP, DOWN
        tbmodels = model(spinor, nspin=2)
        parameters["basis"] = ["Fe1|d|up", "Fe1|d|down", "Fe2|d|up", "Fe2|d|down"]
    result = classes[kind](tbmodels=tbmodels, **parameters)
    if kind.startswith("pert"):
        result.set_epw(
            Ru=ORIGIN, epmode_up=DimerDisplacement(), epmode_dn=DimerDisplacement()
        )
    return result


def execution_context(kind):
    if not kind.endswith("gpu"):
        return nullcontext()
    jax = pytest.importorskip("jax")
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("CUDA/ROCm device is unavailable")
    return jax.enable_x64()


@pytest.mark.parametrize(
    "kind",
    [
        "cl",
        "ncl",
        "pert",
        pytest.param("cl_gpu", marks=pytest.mark.gpu),
        pytest.param("ncl_gpu", marks=pytest.mark.gpu),
        pytest.param("pert_gpu", marks=pytest.mark.gpu),
    ],
)
def test_exchange_and_orbitals_against_absolute_spectral_reference(kind):
    with execution_context(kind):
        calc = calculator(kind)
        calc.calculate_all()
    expected = spectral_exchange()
    assert abs(calc.exchange_Jdict[PAIR] - expected) < 1e-10
    np.testing.assert_allclose(calc.Jiso_orb[PAIR], [[expected]], atol=1e-10, rtol=0)
    if kind.startswith("pert"):
        step = 1e-5
        derivative = (
            spectral_exchange(UP + step * DISPLACEMENT, DOWN + step * DISPLACEMENT)
            - spectral_exchange(UP - step * DISPLACEMENT, DOWN - step * DISPLACEMENT)
        ) / (2 * step)
        assert abs(calc.dJdx[PAIR] - derivative) < 1e-9
        np.testing.assert_allclose(
            calc.dJdx_orb[PAIR], [[derivative]], atol=1e-9, rtol=0
        )


@pytest.mark.parametrize("method", ["SeMiCiRcLe", "LeGeNdRe"])
def test_non_cfr_override_executes_geometric_contour(method):
    calc = calculator("cl", integration_method="cfr2", method=method, nz=800)
    calc.calculate_all()
    # This contour is a zero-temperature energy integral, not a Fermi-weighted
    # finite-temperature sum. Its independent reference uses step occupations.
    eu, vu = np.linalg.eigh(UP)
    ed, vd = np.linalg.eigh(DOWN)
    residues = np.outer(vu[0] * vu[1], vd[1] * vd[0])
    kernel = ((eu < 0)[:, None].astype(float) - (ed < 0)[None, :]) / (
        eu[:, None] - ed[None, :]
    )
    delta = np.diag(UP - DOWN)
    expected = -delta[0] * delta[1] * np.sum(residues * kernel) / 4
    assert abs(calc.exchange_Jdict[PAIR] - expected) < 1e-7


def test_explicit_cfr2_precedence_against_spectral_reference():
    calc = calculator("cl", integration_method="semicircle", method="CfR2")
    calc.calculate_all()
    assert abs(calc.exchange_Jdict[PAIR] - spectral_exchange()) < 1e-10


def test_retired_legacy_selector_is_rejected():
    with pytest.raises(ValueError):
        calculator("cl", method="cfr")


@pytest.mark.parametrize("gpu", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_multi_orbital_exchange_and_decomposition_against_spectral_trace(gpu):
    """Off-diagonal orbital products must close the trace, not sum a product."""
    up = np.array(
        [
            [-0.4, 0.06j, 0.12, 0.03],
            [-0.06j, -0.22, -0.04j, 0.09],
            [0.12, 0.04j, -0.3, 0.05],
            [0.03, 0.09, 0.05, -0.15],
        ]
    )
    down = up + np.diag([0.65, 0.5, 0.6, 0.45])
    overlap = np.array(
        [
            [1.3, 0.08j, 0.02, 0],
            [-0.08j, 1.1, 0, 0.03],
            [0.02, 0, 1.2, -0.04j],
            [0, 0.03, 0.04j, 1.4],
        ]
    )
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = tuple(
        LCAOHamiltonian(
            HR=h[None],
            SR=overlap[None],
            Rlist=np.array([ORIGIN]),
            nbasis=4,
            atoms=atoms,
            nspin=1,
            orth=False,
        )
        for h in (up, down)
    )
    eu, vu = eigh(up, overlap)
    ed, vd = eigh(down, overlap)
    diff = (expit(-eu[:, None] / (600 * kB)) - expit(-ed[None, :] / (600 * kB))) / (
        eu[:, None] - ed[None, :]
    )
    delta_i, delta_j = (up - down)[:2, :2], (up - down)[2:, 2:]
    expected = np.zeros((2, 2))
    for n in range(4):
        for m in range(4):
            x = delta_i @ np.outer(vu[:2, n], vu[2:, n].conj())
            y = delta_j @ np.outer(vd[2:, m], vd[:2, m].conj())
            expected -= (x * y.T * diff[n, m]).real / 4
    kind = "cl_gpu" if gpu else "cl"
    with execution_context(kind):
        cls = ExchangeCL2
        if gpu:
            from TB2J.gpu.exchangeCL_gpu import ExchangeCL2GPU

            cls = ExchangeCL2GPU
        calc = cls(
            tbmodels=models,
            atoms=atoms,
            basis=["Fe1|d1", "Fe1|d2", "Fe2|d1", "Fe2|d2"],
            magnetic_elements=["Fe"],
            kmesh=[1, 1, 1],
            efermi=0,
            nz=80,
            nproc=1,
            Rcut=2,
            orb_decomposition=True,
        )
        calc.calculate_all()
    assert abs(calc.exchange_Jdict[PAIR] - expected.sum()) < 1e-10
    np.testing.assert_allclose(calc.Jiso_orb[PAIR], expected, atol=1e-10, rtol=0)
