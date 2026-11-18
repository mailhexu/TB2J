"""Metric-dressed populations through real localized and DMFT consumers."""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from scipy.linalg import eigh
from scipy.special import expit
from test_cfr2_exchange import DimerDisplacement, execution_context

from TB2J.dmft_model import TBModelDMFT
from TB2J.exchange import ExchangeNCL
from TB2J.exchange_dmft import ExchangeCLDMFT, ExchangeDMFTNCL
from TB2J.exchange_pert2 import ExchangePert2
from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.myTB import MyTB

ORIGIN = (0, 0, 0)
ATOMS = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
METRIC = np.array([[1.0, 0.2 + 0.1j], [0.2 - 0.1j, 1.3]])
BASE = np.array([[0.02, 0.12 + 0.03j], [0.12 - 0.03j, -0.03]])
SIGMA = np.array([np.diag([-0.3, -0.25]), np.diag([0.3, 0.25])])
PAULI = np.array([np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def spectral_density(h, s):
    energies, vectors = eigh(h, s)
    return (vectors * expit(-energies / (600 * kB))) @ vectors.conj().T


def populations(density, metric):
    dressed = metric @ density
    return np.array(
        [
            [
                np.trace(sigma @ dressed[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]).real
                for sigma in PAULI
            ]
            for i in range(2)
        ]
    )


def spinor_matrices():
    h = np.zeros((4, 4), complex)
    h[::2, ::2], h[1::2, 1::2] = BASE + SIGMA[0], BASE + SIGMA[1]
    angle = 0.37
    rotation = np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * PAULI[1]
    unitary = np.kron(np.eye(2), rotation)
    return unitary @ h @ unitary.conj().T, np.kron(METRIC, np.eye(2))


def parameters(spinor=False):
    basis = ["Fe1|d", "Fe2|d"]
    if spinor:
        basis = ["Fe1|d|up", "Fe1|d|down", "Fe2|d|up", "Fe2|d|down"]
    return dict(
        atoms=ATOMS,
        basis=basis,
        magnetic_elements=["Fe"],
        kmesh=[1, 1, 1],
        efermi=0,
        smearing=600 * kB,
        nz=80,
        nproc=1,
        Rcut=2,
        method="cfr2",
    )


@pytest.mark.parametrize(
    "kind,density_method",
    [
        ("pert", "greens_function"),
        ("pert", "eigenvector"),
        pytest.param("pert_gpu", "greens_function", marks=pytest.mark.gpu),
        pytest.param("pert_gpu", "eigenvector", marks=pytest.mark.gpu),
    ],
)
def test_complex_metric_charge_and_all_spin_components(kind, density_method):
    h, s = spinor_matrices()
    model = LCAOHamiltonian(
        HR=np.array([h]),
        SR=np.array([s]),
        Rlist=np.array([ORIGIN]),
        nbasis=4,
        atoms=ATOMS,
        nspin=2,
    )
    cls = ExchangePert2
    if kind.endswith("gpu"):
        from TB2J.gpu.exchange_pert2_gpu import ExchangePert2GPU

        cls = ExchangePert2GPU
    with execution_context(kind):
        kwargs = parameters(spinor=True)
        calc = cls(tbmodels=model, **kwargs)
        calc.set_epw(
            Ru=ORIGIN,
            epmode_up=DimerDisplacement(),
            epmode_dn=DimerDisplacement(),
            density_method=density_method,
        )
        calc.calculate_all()
    expected = populations(spectral_density(h, s), s)
    np.testing.assert_allclose(calc.charges, expected[:, 0], atol=1e-10, rtol=0)
    np.testing.assert_allclose(calc.spinat, expected[:, 1:], atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", ["cl", "ncl"])
def test_debug_contour_charge_contains_complex_metric_coherence(kind):
    if kind == "cl":
        models = tuple(
            LCAOHamiltonian(
                HR=np.array([BASE + sigma]),
                SR=np.array([METRIC]),
                Rlist=np.array([ORIGIN]),
                nbasis=2,
                atoms=ATOMS,
                nspin=1,
            )
            for sigma in SIGMA
        )
        calc = ExchangeCL2(
            tbmodels=models,
            **parameters(),
            debug_options={"compute_charge_moments": True},
        )
        densities = [spectral_density(BASE + sigma, METRIC) for sigma in SIGMA]
        expected = np.array([np.diag(METRIC @ d).real for d in densities])
    else:
        h, s = spinor_matrices()
        model = LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([s]),
            Rlist=np.array([ORIGIN]),
            nbasis=4,
            atoms=ATOMS,
            nspin=2,
        )
        calc = ExchangeNCL(
            tbmodels=model,
            **parameters(spinor=True),
            debug_options={"compute_charge_moments": True},
        )
        expected = populations(spectral_density(h, s), s)
    calc.calculate_all()
    for atom in range(2):
        if kind == "cl":
            for spin, stored in enumerate((calc.G_diagonal_up, calc.G_diagonal_dn)):
                charge = (
                    -calc.contour.integrate_values(np.array(stored[atom])).imag.sum()
                    / np.pi
                )
                assert abs(charge - expected[spin, atom]) < 1e-10
        else:
            charge = (
                -calc.contour.integrate_values(
                    np.array(calc.G_diagonal[atom])
                ).imag.sum()
                / np.pi
            )
            assert abs(charge - expected[atom, 0]) < 1e-10


class MetricDimer(MyTB):
    """Full numerical generalized-Hamiltonian input for the DMFT model API."""

    def __init__(self, h, s, nspin=1):
        super().__init__(nbasis=len(h), data={ORIGIN: h / 2}, nspin=nspin)
        self.HR = np.array([h])
        self.SR = np.array([s])
        self.is_orthogonal = False
        self.set_atoms(ATOMS)

    def HSE_k(self, kpt, convention=2):
        h = self.gen_ham(kpt, convention=convention)
        values, vectors = eigh(h, self.SR[0])
        return h, self.SR[0], values, vectors


class SelfEnergyInput:
    """Constant self-energy data, also usable on a finite Matsubara mesh."""

    def __init__(self, sigma, nfreq=1024):
        self.sigma = sigma
        self.mesh = 1j * (2 * np.arange(nfreq) + 1) * np.pi * (600 * kB)

    def get_chemical_potential(self):
        return 0.0

    def get_static_sigma(self):
        return self.sigma, None

    def read_self_energy(self):
        return np.zeros((2, len(self.mesh), 2, 2), complex), self.mesh


@pytest.mark.parametrize(
    "kind",
    [
        "cl",
        "ncl",
        pytest.param("cl_gpu", marks=pytest.mark.gpu),
        pytest.param("ncl_gpu", marks=pytest.mark.gpu),
    ],
)
def test_static_dmft_metric_populations_at_the_contour_temperature(kind):
    if kind.startswith("cl"):
        static = MetricDimer(BASE, METRIC)
        sigma = SIGMA
        ref = [
            np.diag(spectral_density(BASE + sig, METRIC) @ METRIC).real for sig in SIGMA
        ]
        expected = np.array([ref[0] + ref[1], ref[0] - ref[1]])
        cls = ExchangeCLDMFT
    else:
        h, s = spinor_matrices()
        static = MetricDimer(h, s, nspin=2)
        sigma = np.zeros((4, 4), complex)
        expected = populations(spectral_density(h, s), s)
        cls = ExchangeDMFTNCL
    if kind.endswith("gpu"):
        from TB2J.gpu.exchange_dmft_gpu import ExchangeCLDMFTGPU, ExchangeDMFTNCLGPU

        cls = ExchangeCLDMFTGPU if kind.startswith("cl") else ExchangeDMFTNCLGPU
    model = TBModelDMFT(static, SelfEnergyInput(sigma), use_static_sigma=True)
    with execution_context(kind):
        calc = cls(tbmodels=model, **parameters(spinor=not kind.startswith("cl")))
        calc.calculate_all()
    if kind.startswith("cl"):
        np.testing.assert_allclose(calc.charges, expected[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(calc.spinat[:, 2], expected[1], atol=1e-10, rtol=0)
    else:
        np.testing.assert_allclose(calc.charges, expected[:, 0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(calc.spinat, expected[:, 1:], atol=1e-10, rtol=0)
    # Independent generalized spectral exchange checks that the corrected static
    # density convention does not change the response normalization/sign.
    eu, cu = eigh(BASE + SIGMA[0], METRIC)
    ed, cd = eigh(BASE + SIGMA[1], METRIC)
    fu, fd = expit(-eu / (600 * kB)), expit(-ed / (600 * kB))
    residues = np.outer(cu[0] * cu[1].conj(), cd[1] * cd[0].conj())
    kernel = (fu[:, None] - fd[None, :]) / (eu[:, None] - ed[None, :])
    delta = np.diag(SIGMA[0] - SIGMA[1])
    reference_j = (-delta[0] * delta[1] * np.sum(residues * kernel) / 4).real
    np.testing.assert_allclose(
        calc.exchange_Jdict[(ORIGIN, 0, 1)], reference_j, atol=1e-10, rtol=0
    )


def test_dynamical_matsubara_population_control_stays_on_its_own_contract():
    static = MetricDimer(BASE.real, np.eye(2))
    static.is_orthogonal = True
    model = TBModelDMFT(static, SelfEnergyInput(SIGMA), use_static_sigma=False)
    kwargs = parameters()
    kwargs["method"] = "matsubara"
    calc = ExchangeCLDMFT(tbmodels=model, **kwargs)
    calc.calculate_all()
    ref = [np.diag(spectral_density(BASE.real + sig, np.eye(2))).real for sig in SIGMA]
    np.testing.assert_allclose(calc.charges, ref[0] + ref[1], atol=1e-3, rtol=0)
    np.testing.assert_allclose(calc.spinat[:, 2], ref[0] - ref[1], atol=1e-3, rtol=0)
