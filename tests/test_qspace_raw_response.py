"""Raw q-space response through the public library entry point."""

import numpy as np
import pytest
from ase import Atoms
from HamiltonIO.lcao_hamiltonian import LCAOHamiltonian

from TB2J.exchange_qspace import ExchangeCLQspace


def calculator(method, matrices=None, nz=None):
    atoms = Atoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
    models = []
    matrices = matrices or (
        [[-0.3, 0.12], [0.12, -0.25]],
        [[0.3, 0.12], [0.12, 0.25]],
    )
    for h in matrices:
        models.append(
            LCAOHamiltonian(
                HR=np.array([h]),
                SR=np.array([np.eye(2)]),
                Rlist=np.zeros((1, 3), int),
                nbasis=2,
                atoms=atoms,
                nspin=1,
                orth=True,
            )
        )
    return ExchangeCLQspace(
        tbmodels=tuple(models),
        atoms=atoms,
        magnetic_elements=["Fe"],
        basis=["Fe1|d", "Fe2|d"],
        kmesh=[1, 1, 1],
        efermi=0,
        nz=nz or (4097 if method == "semicircle" else 8),
        nproc=1,
        Rcut=2,
        method=method,
    )


def test_raw_exchange_matches_zero_temperature_spectral_reference():
    raw = calculator("semicircle").calculate_raw()
    # Independent occupied-band spectral sum at zero temperature, in eV.
    np.testing.assert_allclose(raw[0, 0, 1], -0.016201620162016202, atol=2e-10, rtol=0)
    np.testing.assert_allclose(raw[0, 0, 0], 0.1662016201620161, atol=2e-10, rtol=0)


def test_raw_exchange_does_not_require_well_conditioned_bruno_matrix():
    h = [[0.0, 0.5], [0.5, 0.0]]
    calc = calculator("semicircle", matrices=(h, h), nz=64)
    raw = calc.calculate_raw()
    np.testing.assert_allclose(raw, 0.0, atol=1e-14, rtol=0)
    assert np.linalg.cond(calc.Xq[0]) > 1e12


def test_longitudinal_stiffness_matches_spectral_reference():
    # Metallic up channel (mu = 0 cuts through it), empty down channel:
    # the exact band longitudinal stiffness is the divided-difference sum
    # over same-channel eigenpairs (MathDoc: tb2j/bruno-renormalization-dimer).
    up = np.array([[-0.3, 0.5], [0.5, -0.25]])
    dn = np.array([[0.3, 0.12], [0.12, 0.25]])
    calc = calculator("semicircle", matrices=(up, dn))
    calc.calculate_raw()
    calc.bruno_renormalize()

    C = np.zeros((2, 2))
    for h, f in ((up, None), (dn, None)):
        w, v = np.linalg.eigh(h)
        occ = (w < 0).astype(float)
        for i in range(2):
            for j in range(2):
                for n in range(2):
                    for m in range(2):
                        if n == m:
                            continue
                        C[i, j] += (
                            (occ[n] - occ[m])
                            / (w[n] - w[m])
                            * (v[n, i] * v[m, i] * v[m, j] * v[n, j])
                        )
    # CR-006: Cq stays complex Hermitian.  The -i*M extraction with
    # Hermitian symmetrization reproduces the spectral stiffness here and
    # preserves the complex phases a multi-site C(q) carries.
    np.testing.assert_allclose(calc.Cq[0], calc.Cq[0].conj().T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(calc.Cq[0].imag, 0.0, atol=1e-8, rtol=0)
    np.testing.assert_allclose(calc.Cq[0].real, C, atol=1e-10, rtol=0)


def test_bruno_correction_vanishes_for_collinear_reference(tmp_path):
    # At a collinear reference the transverse-longitudinal cross response B
    # vanishes identically (MathDoc: tb2j/bruno-renormalization-dimer, S4),
    # so the renormalized exchange equals the raw one and no longitudinal
    # correction is applied.
    up = np.array([[-0.3, 0.5], [0.5, -0.25]])
    dn = np.array([[0.3, 0.12], [0.12, 0.25]])
    calc = calculator("semicircle", matrices=(up, dn))
    calc.calculate_raw()
    calc.bruno_renormalize()
    np.testing.assert_allclose(calc.Bq[0], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(calc.Jnorm_q[0], calc.Jq[0], atol=1e-12, rtol=0)
    assert calc.longitudinal_correction_applied is False
    # CR-007: the Bruno-renormalized table is emitted as a separate,
    # labeled dict (never overwriting the raw one) and here coincides
    # with it because the correction vanishes.
    calc.q_to_r()
    calc.get_Jdict()
    assert set(calc.exchange_Jdict_bruno) == set(calc.exchange_Jdict)
    np.testing.assert_allclose(
        sorted(calc.exchange_Jdict_bruno.values()),
        sorted(calc.exchange_Jdict.values()),
        atol=1e-12,
        rtol=0,
    )
    # R2-004: the text output carries the labeled Bruno result and the
    # availability/scope note next to the untouched raw table.
    calc.write_output(path=str(tmp_path))
    text = (tmp_path / "exchange.out").read_text()
    assert "Bruno-renormalized exchange" in text
    assert "longitudinal_correction_applied: False" in text
    assert "scope:" in text


def test_raw_exchange_rejects_a_pole_on_the_contour():
    h = [[-0.5, 0.5], [0.5, -0.5]]
    calc = calculator("semicircle", matrices=(h, h), nz=64)
    with pytest.raises(ValueError, match="non-finite"):
        calc.calculate_raw()


@pytest.mark.parametrize("method", ["cfr", "cfr2"])
def test_raw_exchange_rejects_pole_contours_without_compatible_integration(method):
    with pytest.raises(ValueError, match="requires a geometric contour"):
        calculator(method).calculate_raw()
