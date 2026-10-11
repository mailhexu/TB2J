"""GPAW noncollinear+SOC spinor export tests (story 004)."""

import numpy as np
import pytest

gpaw = pytest.importorskip("gpaw")

from TB2J.interfaces.gpaw_spinor_projector import (  # noqa: E402
    compute_spinor_projector_exchange,
    gpaw_spinor_calc_to_projector_green_data,
)
from TB2J.projector_green import (  # noqa: E402
    SPINOR_OPERATOR_DEFINITION,
    ProjectorGreen,
    spinor_tangent_trace,
)


def _fe_box(soc=True, magmoms=None):
    from ase import Atoms
    from gpaw import GPAW, PW

    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True)
    kwargs = {}
    if soc or magmoms is not None:
        kwargs["magmoms"] = magmoms if magmoms is not None else [[0, 0, 3.0]]
        kwargs["soc"] = soc
    calc = GPAW(
        mode=PW(300),
        xc="LDA",
        kpts=(2, 2, 2),
        symmetry="off",
        txt=None,
        convergence={"energy": 1e-4},
        **kwargs,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc


def test_spinor_export_from_soc_calc(tmp_path):
    calc = _fe_box(soc=True)
    data = gpaw_spinor_calc_to_projector_green_data(calc)
    assert data.nspinor == 2
    assert data.coefficients.shape == (1, 8, data.nband, 2, 18)
    assert data.nproj == 18
    assert data.spinor_operator.shape == (1, 18, 18, 2, 2)
    assert data.spinor_operator_definition == SPINOR_OPERATOR_DEFINITION
    assert data.validate(exchange_ready=True)
    assert np.isfinite(data.coefficients).all()
    assert np.isfinite(data.eigenvalues).all()
    np.testing.assert_allclose(data.weights.sum(), 1.0, atol=1e-12)
    # Hermiticity of the 2x2 operator blocks
    block = data.spinor_operator[0]
    np.testing.assert_allclose(block, block.transpose(1, 0, 3, 2).conj(), atol=1e-12)
    data.save_netcdf(tmp_path / "fe_soc.nc")


def test_spinor_export_rejects_collinear():
    calc = _fe_box(soc=False)
    # plain collinear (spinpol) calc: soc/magmoms kwargs omitted -> collinear
    with pytest.raises(ValueError, match="noncollinear"):
        gpaw_spinor_calc_to_projector_green_data(calc)


def test_spinor_kernel_consumes_export():
    calc = _fe_box(soc=True)
    data = gpaw_spinor_calc_to_projector_green_data(calc)
    green = ProjectorGreen(data)
    Rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
    trace = spinor_tangent_trace(green, Rpts, energy=0.05)
    matrix = trace["K_ijR"][((1, 0, 0), 0, 0)]
    reverse = trace["K_ijR"][((-1, 0, 0), 0, 0)]
    np.testing.assert_allclose(matrix, reverse.T, atol=1e-10)
    exchange = compute_spinor_projector_exchange(
        data, Rpts=Rpts, nz=6, smearing_eV=0.05, sites=[0]
    )
    for entry in exchange.values():
        frame = entry["frame"]
        assert frame["n"] == 2
        assert "Jiso" not in entry and "dmi" not in entry and "jani" not in entry
        np.testing.assert_allclose(entry["J_leg"][2, :], 0.0, atol=1e-12)
        np.testing.assert_allclose(entry["J_leg"][:, 2], 0.0, atol=1e-12)
    with pytest.raises(ValueError, match="three independent x/y/z"):
        from TB2J.interfaces.gpaw_spinor_projector import (
            write_spinor_projector_exchange_out,
        )

        write_spinor_projector_exchange_out(data)


def test_symmetry_forced_off_for_sc_noncollinear():
    """Story 005: GPAW itself refuses symmetry for SC noncollinear PW runs
    (gpaw/new/builder.py assertion), so the native export path needs no
    unfolding; the exporter guards against symmetrized k-weights."""
    import pytest as _pytest
    from ase import Atoms
    from gpaw import GPAW, PW

    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True)
    calc = GPAW(
        mode=PW(200),
        xc="LDA",
        kpts=(2, 2, 2),
        soc=True,
        magmoms=[[0, 0, 3.0]],
        txt=None,
        convergence={"energy": 1e-3},
    )
    atoms.calc = calc
    with _pytest.raises(AssertionError):
        atoms.get_potential_energy()
