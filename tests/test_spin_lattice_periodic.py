"""Periodic image-sum response against independent pencil perturbations."""

import numpy as np
import pytest
from ase import Atoms
from HamiltonIO.model.occupations import myfermi
from scipy.linalg import eigh

from TB2J.spin_lattice_lcao import ExchangePert2Sl, LocalizedDisplacementDerivative


class HybridizedChain:
    """A magnetic low orbital hybridized with a virtual orbital above 5 eV."""

    name = "LCAOHamiltonian"
    R2kfactor = 2j * np.pi
    is_orthogonal = False
    norb = 2
    nbasis = 4
    nel = 1.0

    def __init__(self, tau=0, collinear=False):
        self.tau = tau
        self.collinear_interleaved = collinear
        self.atoms = Atoms(
            "Fe", positions=[[0, 0, 0]], cell=np.diag([1, 10, 10]), pbc=True
        )

    def matrices(self, k):
        c = np.cos(2 * np.pi * k[0])
        h = np.zeros((4, 4), complex)
        s = np.eye(4, dtype=complex)
        dh = np.zeros_like(h)
        ds = np.zeros_like(s)
        for spin in (0, 1):
            a, b = spin, 2 + spin
            h[a, a] = -0.7 + 0.6 * (1 - 2 * spin) - 0.8 * c
            h[b, b] = 12 + 0.2 * (1 - 2 * spin)
            h[a, b] = h[b, a] = 1.5
            s[a, b] = s[b, a] = 0.06
            dh[a, a] = 0.2 * (1 - 2 * spin) + 0.3 * c
            dh[a, b] = dh[b, a] = 0.4
            ds[a, b] = ds[b, a] = 0.02
        return h + self.tau * dh, s + self.tau * ds, dh, ds

    def HSE_k(self, k, convention=2):
        h, s, _, _ = self.matrices(k)
        if self.collinear_interleaved:
            ev = np.empty(4)
            v = np.zeros((4, 4), complex)
            for spin in (0, 1):
                sl = slice(spin, None, 2)
                ev[sl], v[sl, sl] = eigh(h[sl, sl], s[sl, sl])
        else:
            ev, v = eigh(h, s)
        return h, s, ev, v


def calculation(tau=0, disp=None, collinear=False, cache_path=None):
    model = HybridizedChain(tau, collinear=collinear)
    ex = ExchangePert2Sl(
        tbmodels=model,
        atoms=model.atoms,
        basis=["Fe1|s|up", "Fe1|s|down", "Fe1|p|up", "Fe1|p|down"],
        magnetic_elements=["Fe"],
        efermi=0.3,
        kmesh=(12, 1, 1),
        Rcut=3.1,
        nz=48,
        integration_method="semicircle",
        smearing=0.1,
        use_cache=cache_path is not None,
        cache_path=str(cache_path) if cache_path is not None else None,
    )
    ex.set_displacement(disp, J_only=disp is None, response="periodic")
    ex.calculate_all()
    return ex


def derivative():
    m = HybridizedChain()
    _, _, d0, s0 = m.matrices([0.25, 0, 0])
    _, _, d1, _ = m.matrices([0, 0, 0])
    hopping = (d1 - d0) / 2
    zero = (0, 0, 0)
    return LocalizedDisplacementDerivative(
        {(zero, zero): d0, (zero, (1, 0, 0)): hopping, (zero, (-1, 0, 0)): hopping},
        {(zero, zero): s0},
        Ru=zero,
        atoms=m.atoms,
        provenance="analytic periodic chain",
    )


@pytest.mark.parametrize("collinear", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_periodic_overlap_and_splitting_response_matches_fd(
    collinear, cached, tmp_path
):
    cache_path = tmp_path if cached else None
    ex = calculation(disp=derivative(), collinear=collinear, cache_path=cache_path)
    step = 1e-5
    plus, minus = (
        calculation(step, collinear=collinear, cache_path=cache_path),
        calculation(-step, collinear=collinear, cache_path=cache_path),
    )
    keys = sorted(ex.dJdx)
    actual = np.array([ex.dJdx[k] for k in keys])
    expected = np.array(
        [(plus.exchange_Jdict[k] - minus.exchange_Jdict[k]) / (2 * step) for k in keys]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-9)
    assert np.max(np.abs(expected)) > 1e-5
    # High-energy virtual states must remain in the pencil inverse.
    k = ex.G.kpts[0]
    h, s, _, _ = HybridizedChain().matrices(k)
    z = -0.4 + 0.7j
    np.testing.assert_allclose(
        ex.G.get_Gk(0, z - ex.G.efermi), np.linalg.inv(z * s - h), atol=2e-13
    )
    for result in (ex, plus, minus):
        result.finalize()


def test_invalid_response_rejected():
    model = HybridizedChain()
    ex = ExchangePert2Sl(
        tbmodels=model,
        atoms=model.atoms,
        basis=["Fe1|s|up", "Fe1|s|down", "Fe1|p|up", "Fe1|p|down"],
        magnetic_elements=["Fe"],
        efermi=0.3,
        kmesh=(3, 1, 1),
        nz=12,
        integration_method="semicircle",
    )
    with pytest.raises(ValueError, match="response"):
        ex.set_displacement(derivative(), response="localish")


@pytest.mark.parametrize("cached", [False, True])
def test_streamed_k_sum_keeps_last_partial_block_and_density(cached, tmp_path):
    model = HybridizedChain(collinear=True)
    ex = ExchangePert2Sl(
        tbmodels=model,
        atoms=model.atoms,
        basis=["Fe1|s|up", "Fe1|s|down", "Fe1|p|up", "Fe1|p|down"],
        magnetic_elements=["Fe"],
        efermi=0.3,
        kmesh=(67, 1, 1),
        Rcut=3.1,
        nz=48,
        integration_method="semicircle",
        smearing=0.1,
        use_cache=cached,
        cache_path=str(tmp_path) if cached else None,
    )
    ex.set_displacement(
        derivative(), response="periodic", density_method="greens_function"
    )
    energy = 0.41 + 0.73j
    z = energy + ex.G.efermi
    Rpts = ex.short_Rlist
    gr = np.zeros((len(Rpts), 4, 4), complex)
    dg = np.zeros_like(gr)
    rho_energy = np.zeros((4, 4), complex)
    rho_occupied = np.zeros_like(rho_energy)
    for k, weight in zip(ex.G.kpts, ex.G.kweights):
        h, s, dh, ds = model.matrices(k)
        g = np.linalg.inv(z * s - h)
        d = g @ (dh - z * ds) @ g
        phase = weight * np.exp(ex.G.k2Rfactor * (np.asarray(Rpts) @ k))
        gr += phase[:, None, None] * g
        dg += phase[:, None, None] * d
        rho_energy += weight * s @ g
        values, vectors = eigh(h, s)
        rho_occupied += weight * (
            (vectors * myfermi(values, ex.G.efermi, width=ex.G.fermi_width, nspin=2))
            @ vectors.conj().T
            @ s
        )
    gr_dict, dg_dict = dict(zip(Rpts, gr)), dict(zip(Rpts, dg))
    reverse = {R: dg_dict[tuple(-x for x in R)] for R in Rpts}
    expected_a, expected_da, _, _ = ex.get_all_A(gr_dict, dg_dict, reverse)
    actual_a, actual_da, actual_rho, _, _ = ex.get_AijR_rhoR(energy)
    for key in expected_a:
        np.testing.assert_allclose(actual_a[key], expected_a[key], atol=2e-12)
        np.testing.assert_allclose(actual_da[key], expected_da[key], atol=2e-12)
    np.testing.assert_allclose(
        actual_rho, ex.get_rho_e_spin({(0, 0, 0): rho_energy}), atol=2e-12
    )
    np.testing.assert_allclose(ex.G.get_density_matrix(), rho_occupied, atol=2e-12)
    ex.finalize()


def test_cached_collinear_preparation_avoids_full_k_spinor_stacks(
    monkeypatch, tmp_path
):
    """Disk caching must bound the *preparation* peak, not just later G(E)."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    original_zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        if isinstance(shape, (tuple, list)) and tuple(shape) == (67, 4, 4):
            raise AssertionError("allocated full k-point spinor stack in RAM")
        return original_zeros(shape, *args, **kwargs)

    model = HybridizedChain(collinear=True)
    with monkeypatch.context() as patch:
        patch.setattr(np, "zeros", bounded_zeros)
        ex = ExchangePert2Sl(
            tbmodels=model,
            atoms=model.atoms,
            basis=["Fe1|s|up", "Fe1|s|down", "Fe1|p|up", "Fe1|p|down"],
            magnetic_elements=["Fe"],
            efermi=0.3,
            kmesh=(67, 1, 1),
            Rcut=3.1,
            nz=48,
            integration_method="semicircle",
            smearing=0.1,
            use_cache=True,
        )
    assert all(isinstance(v, np.memmap) for v in ex.G._spin_vectors)
    assert all(isinstance(s, np.memmap) for s in ex.G._spin_overlap)
    ex.finalize()
