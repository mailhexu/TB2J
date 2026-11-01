"""Absolute J/K/X references through the real reciprocal-space consumer.

Ported from owner revision 4355ca6 (CFR2 subproject Story 006). The helper
module import was inlined: this tree has no tests/test_cfr2_exchange.py and
its GPU/JAX contexts are out of scope here. Oracle expressions, kernels and
tolerances are the owner's, unchanged.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.units import kB
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian
from scipy.special import expit

from TB2J.exchange_qspace import ExchangeCLQspace
from TB2J.interfaces.manager import Manager

ORIGIN = (0, 0, 0)
UP = np.array([[-0.3, 0.12], [0.12, -0.25]])
DOWN = np.array([[0.3, 0.12], [0.12, 0.25]])


def qspace_calculator(method="cfr2", nz=80, nk=1):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = tuple(
        LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([np.eye(2)]),
            Rlist=np.array([ORIGIN]),
            nbasis=2,
            atoms=atoms,
            nspin=1,
            orth=True,
        )
        for h in (UP, DOWN)
    )
    return ExchangeCLQspace(
        tbmodels=models,
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|d", "Fe2|d"],
        kmesh=[nk, 1, 1],
        efermi=0,
        nz=nz,
        nproc=1,
        Rcut=2,
        method=method,
    )


def spectral_jkx(temperature=600):
    """Complete spectral pair sum, not samples of the implementation contour.

    MathDoc: tb2j/cfr2-integration-contract verifies the divided difference.
    This intentionally preserves the existing K1+2*K2 kernel, not a physical
    redefinition of K or a validation of Bruno renormalization.
    """
    eu, vu = np.linalg.eigh(UP)
    ed, vd = np.linalg.eigh(DOWN)
    fu = expit(-eu / (kB * temperature)) if temperature else (eu < 0).astype(float)
    fd = expit(-ed / (kB * temperature)) if temperature else (ed < 0).astype(float)
    divided = (fu[:, None] - fd[None, :]) / (eu[:, None] - ed[None, :])
    delta = np.diag(UP - DOWN)
    j, k, x = (np.empty((1, 2, 2)) for _ in range(3))
    for i in range(2):
        for l in range(2):
            residue = np.outer(vu[i] * vu[l].conj(), vd[l] * vd[i].conj())
            reverse = np.outer(vu[l] * vu[i].conj(), vd[i] * vd[l].conj())
            ud = -1j * np.pi * np.sum(residue * divided)
            du = -1j * np.pi * np.sum(reverse * divided)
            j[0, i, l] = ((ud + du) * delta[i] * delta[l] / (8 * np.pi)).imag
            k[0, i, l] = (-(ud + 2 * du) * delta[l] / (2 * np.pi)).imag
            x[0, i, l] = (2 * ud / np.pi).imag
    return j, k, x


@pytest.mark.parametrize("nk", [1, 3])
def test_actual_qspace_cfr2_against_all_spectral_observables(nk):
    calc = qspace_calculator(nk=nk)
    calc.calculate_all()
    for actual, expected in zip((calc.Jq, calc.Kq, calc.Xq), spectral_jkx()):
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected, actual.shape), atol=1e-10, rtol=0
        )


@pytest.mark.parametrize("rule", ["simpson", "trapezoidal"])
def test_geometric_qspace_controls_against_zero_temperature_spectrum(rule):
    calc = qspace_calculator(method="semicircle", nz=4097)
    calc._prepare()
    calc.get_all_A()
    calc.integrate(method=rule)
    # The retained geometric contour is zero-temperature, not Fermi weighted.
    # Smooth gapped two-level products: at4097 nodes O(h^2) error is <2e-6.
    for actual, expected in zip((calc.Jq, calc.Kq, calc.Xq), spectral_jkx(0)):
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)


def test_pair_table_respects_distance_cutoff(tmp_path):
    # R = (±2,0,0) images are 3 A apart, beyond Rcut=2: the finite-grid
    # transform keeps them, but the emitted pair table must filter them,
    # matching the real-space route's distance_dict keys.
    calc = qspace_calculator(method="cfr2", nk=5)
    calc.calculate_all()
    calc.write_output(path=str(tmp_path / "results"))
    assert set(calc.exchange_Jdict) <= set(calc.distance_dict)
    assert ((2, 0, 0), 0, 1) not in calc.exchange_Jdict


def test_cfr2_is_default_for_qspace_driver():
    from TB2J.mycfr import CFR2

    calc = qspace_calculator(method=None)
    assert calc.integration_method.lower() == "cfr2"
    # The default must build the CFR2 contour, not a CFR pole path that the
    # CFR2 branch would then mis-contract.
    assert isinstance(calc.contour, CFR2)


def test_explicit_none_method_builds_cfr2_contour():
    from TB2J.mycfr import CFR2

    calc = qspace_calculator(method=None)
    assert isinstance(calc.contour, CFR2)


def test_conflicting_contour_selectors_refused():
    from TB2J.exchange_qspace import ExchangeCLQspace

    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = tuple(
        LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([np.eye(2)]),
            Rlist=np.array([ORIGIN]),
            nbasis=2,
            atoms=atoms,
            nspin=1,
            orth=True,
        )
        for h in (UP, DOWN)
    )
    with pytest.raises(ValueError, match="conflicting"):
        ExchangeCLQspace(
            tbmodels=models,
            atoms=atoms,
            magnetic_elements=["Fe"],
            basis=["Fe1|d", "Fe2|d"],
            kmesh=[1, 1, 1],
            efermi=0,
            nz=8,
            nproc=1,
            Rcut=2,
            method="cfr",
            integration_method="semicircle",
        )


def test_manager_refuses_unsupported_qspace_combinations(tmp_path):
    atoms, models = _toy_manager_models()
    with pytest.raises(ValueError, match="collinear"):
        Manager(
            atoms,
            models,
            ["Fe1|d", "Fe2|d"],
            colinear=False,
            qspace=True,
            efermi=0,
            nz=8,
            Rcut=2,
            magnetic_elements=["Fe"],
            output_path=str(tmp_path / "r1"),
        )
    with pytest.raises(ValueError, match="GPU"):
        Manager(
            atoms,
            models,
            ["Fe1|d", "Fe2|d"],
            colinear=True,
            qspace=True,
            use_gpu=True,
            efermi=0,
            nz=8,
            Rcut=2,
            magnetic_elements=["Fe"],
            output_path=str(tmp_path / "r2"),
        )


def test_deprecated_cfr_refused_for_qspace_full_driver():
    calc = qspace_calculator(method="cfr")
    with pytest.raises(ValueError, match="deprecated"):
        calc.calculate_all()


def test_raw_route_still_requires_geometric_contour():
    calc = qspace_calculator(method="cfr2")
    with pytest.raises(ValueError, match="geometric"):
        calc.calculate_raw()


def _toy_manager_models():
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = tuple(
        LCAOHamiltonian(
            HR=np.array([h]),
            SR=np.array([np.eye(2)]),
            Rlist=np.array([ORIGIN]),
            nbasis=2,
            atoms=atoms,
            nspin=1,
            orth=True,
        )
        for h in (UP, DOWN)
    )
    return atoms, models


def test_select_exchange_routes_qspace():
    manager = Manager.__new__(Manager)
    assert manager.select_exchange(True, qspace=True, use_gpu=False) is ExchangeCLQspace
    assert (
        manager.select_exchange(True, qspace=False, use_gpu=False).__name__
        == "ExchangeCL2"
    )


def test_manager_qspace_flag_runs_qspace_driver(tmp_path, monkeypatch):
    import TB2J.interfaces.manager as manager_mod

    atoms, models = _toy_manager_models()
    selected = {}
    original = manager_mod.ExchangeCLQspace

    def spy_calculate_all(self):
        selected["class"] = type(self)
        raise _StopAfterSelection()

    class _StopAfterSelection(Exception):
        pass

    monkeypatch.setattr(original, "calculate_all", spy_calculate_all)
    with pytest.raises(_StopAfterSelection):
        Manager(
            atoms,
            models,
            ["Fe1|d", "Fe2|d"],
            colinear=True,
            qspace=True,
            magnetic_elements=["Fe"],
            efermi=0,
            nz=80,
            Rcut=2,
            nproc=1,
            output_path=str(tmp_path / "results"),
        )
    assert selected["class"] is ExchangeCLQspace


def test_production_cli_parses_and_dispatches_qspace(monkeypatch):
    # The production parser and dispatch, exercised without wannier files:
    # gen_exchange is replaced by a recorder so the test observes exactly
    # what run_wann2J passes to the manager.
    import TB2J.scripts.wann2J as w2j

    received = {}
    monkeypatch.setattr(w2j, "gen_exchange", lambda **kwargs: received.update(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "wann2J",
            "--efermi",
            "0.0",
            "--elements",
            "Fe",
            "--qspace",
            "--integration-method",
            "cfr2",
        ],
    )
    w2j.run_wann2J()
    assert received["qspace"] is True
    assert received["integration_method"] == "cfr2"
    assert received["colinear"] is True

    # The underscore spelling is accepted too (house convention alias).
    monkeypatch.setattr(
        "sys.argv",
        [
            "wann2J",
            "--efermi",
            "0.0",
            "--elements",
            "Fe",
            "--qspace",
            "--integration_method",
            "semicircle",
        ],
    )
    w2j.run_wann2J()
    assert received["integration_method"] == "semicircle"


def test_cli_python_parity_same_driver_and_j(tmp_path):
    # Python API: qspace=True selects ExchangeCLQspace and produces J.
    # The CLI test above proves the production parser dispatches the same
    # kwargs; here the physics side of the parity chain is pinned.
    atoms, models = _toy_manager_models()
    m = Manager(
        atoms,
        models,
        ["Fe1|d", "Fe2|d"],
        colinear=True,
        qspace=True,
        magnetic_elements=["Fe"],
        efermi=0,
        nz=80,
        Rcut=2,
        nproc=1,
        kmesh=[1, 1, 1],
        output_path=str(tmp_path / "results"),
    )
    assert m.exchange_class is ExchangeCLQspace
    from TB2J.io_exchange import SpinIO

    spi = SpinIO.load_pickle(path=str(tmp_path / "results"))
    assert ((0, 0, 0), 0, 1) in spi.exchange_Jdict


@pytest.fixture()
def srmno3_fixture():
    from conftest import require_input

    return require_input(
        "inputs/2_SrMnO3_wannier/data",
        capability="qspace_cli_python_parity",
        description="SrMnO3 wannier fixture for CLI/Python J parity",
    )


def test_cli_and_python_routes_produce_identical_j(
    tmp_path, monkeypatch, srmno3_fixture
):
    """Full production parity: run_wann2J vs WannierManager on the pinned
    fixture, same reduced settings, identical J dictionaries."""
    import TB2J.scripts.wann2J as w2j

    common = dict(
        path=str(srmno3_fixture),
        posfile="abinit.in",
        prefix_up="abinito_w90_up",
        prefix_dn="abinito_w90_down",
        efermi=6.15,
        kmesh=[3, 3, 3],
        nz=40,
        magnetic_elements=["Mn"],
        Rcut=6.0,
        nproc=1,
        integration_method="cfr2",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "wann2J",
            "--path",
            str(srmno3_fixture),
            "--posfile",
            "abinit.in",
            "--prefix_up",
            "abinito_w90_up",
            "--prefix_down",
            "abinito_w90_down",
            "--efermi",
            "6.15",
            "--kmesh",
            "3",
            "3",
            "3",
            "--nz",
            "40",
            "--elements",
            "Mn",
            "--rcut",
            "6.0",
            "--qspace",
            "--integration_method",
            "cfr2",
            "--output_path",
            "cli_results",
        ],
    )
    w2j.run_wann2J()

    from TB2J.interfaces.wannier90_interface import WannierManager
    from TB2J.io_exchange import SpinIO

    WannierManager(
        prefix_SOC="unused",
        colinear=True,
        qspace=True,
        output_path="py_results",
        **common,
    )
    cli = SpinIO.load_pickle(path=str(tmp_path / "cli_results"))
    py = SpinIO.load_pickle(path=str(tmp_path / "py_results"))
    assert cli.exchange_Jdict, "CLI run produced no exchange pairs"
    assert set(cli.exchange_Jdict) == set(py.exchange_Jdict)
    for key in sorted(cli.exchange_Jdict):
        assert cli.exchange_Jdict[key] == py.exchange_Jdict[key], key
