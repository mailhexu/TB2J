"""Ticket 01 regressions: topology-ready input preparation.

Covers the ADR-001 seam requirements: channel-union translations, spin-ordered
geometry/moments, exact-once SIA canonicalization, unsupported-channel
handling, immutable identity digests, and unchanged legacy band/DOS behavior.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402

from TB2J.io_exchange import SpinIO  # noqa: E402
from TB2J.magnon.magnon3 import Magnon  # noqa: E402
from TB2J.magnon.topology import (  # noqa: E402
    TopologyPreparationError,
    prepare_topology_model,
)


def make_exc(
    index_spin=(0, 1, -1),
    spinat=((0.0, 0.0, 3.0), (0.0, 0.0, -3.0), (0.0, 0.02, 0.0)),
    exchange=None,
    dmi=None,
    jani=None,
    biquadratic=None,
    sia_tensor=None,
    k1=None,
    k1dir=None,
    njt_dmi=None,
    njt_exchange=None,
):
    atoms = Atoms(
        "Cr2X",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [0.0, 3.0, 0.0]],
        cell=3.0 * np.eye(3),
        pbc=True,
    )
    if exchange is None:
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 1e-6))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 1e-6))
    exc = SpinIO(
        atoms=atoms,
        spinat=np.array(spinat, dtype=float),
        charges=np.zeros(len(atoms)),
        index_spin=list(index_spin),
        colinear=False,
        exchange_Jdict=exchange,
        dmi_ddict=dmi,
        Jani_dict=jani,
        biquadratic_Jdict=biquadratic,
        sia_tensor=sia_tensor,
        k1=k1,
        k1dir=k1dir,
        NJT_ddict=njt_dmi,
        NJT_Jdict=njt_exchange,
    )
    exc._build_Rlist()
    return exc


def iso_pair(R, i, j, value):
    return {(tuple(int(x) for x in R), i, j): value}


def dmi_pair(R, i, j, vector):
    """DMI pair with the reverse entry following the TB2J convention
    D_ji(-R) = -D_ij(R)."""
    d = np.asarray(vector)
    return {
        (tuple(int(x) for x in R), i, j): d,
        (tuple(-int(x) for x in R), j, i): -d,
    }


def jani_pair(R, i, j, tensor):
    t = np.asarray(tensor)
    return {
        (tuple(int(x) for x in R), i, j): t,
        (tuple(-int(x) for x in R), j, i): t,
    }


def basic_channels():
    exchange = {}
    exchange.update(iso_pair((0, 0, 1), 0, 0, 0.002))
    exchange.update(iso_pair((0, 0, -1), 0, 0, 0.002))
    exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01))
    exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01))
    dmi = {}
    # (0,1,0) shell exists only in the DMI channel.
    dmi.update(dmi_pair((0, 1, 0), 0, 1, (0.0, 0.0, 0.001)))
    jani = {}
    # (1,1,0) shell exists only in the anisotropic-exchange channel.
    jani.update(jani_pair((1, 1, 0), 0, 1, np.diag([0.0002, -0.0001, -0.0001])))
    return exchange, dmi, jani


class TestTranslationUnion:
    def test_dmi_only_shell_is_retained(self):
        exchange, dmi, _ = basic_channels()
        # (0,1,0) exists only in the DMI channel.
        model = prepare_topology_model(make_exc(exchange=exchange, dmi=dmi))
        Rs = {tuple(R) for R in model.Rlist}
        assert (0, 1, 0) in Rs
        assert (0, -1, 0) in Rs
        iR = next(idx for idx, R in enumerate(model.Rlist) if tuple(R) == (0, 1, 0))
        expected = np.array([[0, 0.001, 0], [-0.001, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(model.JR[iR, 0, 1], expected, atol=1e-15)

    def test_jani_only_shell_is_retained(self):
        exchange, _, jani = basic_channels()
        slim = dict(exchange)
        slim.update(iso_pair((0, 0, 0), 0, 0, 0.0))
        model = prepare_topology_model(make_exc(exchange=slim, jani=jani))
        Rs = {tuple(R) for R in model.Rlist}
        assert (1, 1, 0) in Rs and (-1, -1, 0) in Rs

    def test_onsite_only_spin_is_covered_by_mapping(self):
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01))
        sia = {1: np.diag([0.0, 0.0, 0.0005])}
        # Spin 1 has only the on-site SIA term beyond the shared pair bonds.
        exc = make_exc(
            index_spin=(0, 1, -1),
            exchange=exchange,
            sia_tensor=sia,
        )
        model = prepare_topology_model(exc)
        assert model.nspin == 2
        iR0 = next(idx for idx, R in enumerate(model.Rlist) if np.all(R == 0))
        np.testing.assert_allclose(model.JR[iR0, 1, 1], np.diag([0.0, 0.0, 0.0005]))

    def test_isolated_spin_with_no_bonds_is_retained(self):
        """A magnetic site absent from every pair key still enters the model."""
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 0, 0.002))
        exchange.update(iso_pair((-1, 0, 0), 0, 0, 0.002))
        sia = {0: np.diag([0.0, 0.0, 0.0005]), 1: np.diag([0.0, 0.0, 0.0003])}
        exc = make_exc(index_spin=(0, 1, -1), exchange=exchange, sia_tensor=sia)
        model = prepare_topology_model(exc)
        assert model.nspin == 2
        iR0 = next(idx for idx, R in enumerate(model.Rlist) if np.all(R == 0))
        np.testing.assert_allclose(model.JR[iR0, 1, 1], np.diag([0.0, 0.0, 0.0003]))
        # Legacy loader drops the isolated spin from the tensor silently.
        legacy = Magnon.load_from_io(exc)
        assert legacy.JR.shape[1:3] == (1, 1)


class TestSpinOrdering:
    def test_permuted_index_spin_with_unequal_moments(self):
        exc = make_exc(
            index_spin=(1, -1, 0),
            spinat=((0.0, 0.0, 2.0), (0.1, 0.0, 0.0), (0.0, 0.0, -4.0)),
            exchange=dict(basic_channels()[0]),
        )
        model = prepare_topology_model(exc)
        assert model.nspin == 2
        np.testing.assert_allclose(model.spin_atoms, [2, 0])
        np.testing.assert_allclose(model.magmoms[0], [0.0, 0.0, -4.0])
        np.testing.assert_allclose(model.magmoms[1], [0.0, 0.0, 2.0])
        np.testing.assert_allclose(model.positions[0], [0.0, 3.0, 0.0])
        np.testing.assert_allclose(model.positions[1], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(model.index_spin, [1, -1, 0])

    def test_collinear_spinat_becomes_z_moments(self):
        exc = make_exc(spinat=(3.0, -3.0, 0.0))
        model = prepare_topology_model(exc)
        np.testing.assert_allclose(model.magmoms[:, 2], [3.0, -3.0])

    def test_zero_moment_spin_rejected(self):
        with pytest.raises(TopologyPreparationError, match="zero moment"):
            prepare_topology_model(
                make_exc(spinat=((0.0, 0.0, 0.0), (0.0, 0.0, -3.0), (0.0, 0.0, 0.0)))
            )

    def test_duplicate_spin_index_rejected(self):
        with pytest.raises(TopologyPreparationError, match="exactly once"):
            prepare_topology_model(make_exc(index_spin=(1, 1, -1)))

    def test_gapped_spin_index_rejected(self):
        with pytest.raises(TopologyPreparationError, match="duplicates or gaps"):
            prepare_topology_model(make_exc(index_spin=(0, 2, -1)))


class TestSIAcanonicalization:
    def test_constructor_merged_uniaxial_counted_once(self):
        via_k1 = prepare_topology_model(make_exc(k1=[0.0005], k1dir=[(0.0, 0.0, 1.0)]))
        via_tensor = prepare_topology_model(
            make_exc(sia_tensor={0: 0.0005 * np.outer([0, 0, 1], [0, 0, 1])})
        )
        np.testing.assert_allclose(via_k1.JR, via_tensor.JR, atol=1e-15)
        assert via_k1.digest == via_tensor.digest

    def test_post_construction_uniaxial_is_independent(self):
        sia_tensor = {0: 0.0004 * np.outer([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])}
        exc = make_exc(sia_tensor=sia_tensor)
        exc.k1 = np.array([0.0002])
        exc.k1dir = np.array([[0.0, 1.0, 0.0]])
        exc.has_uniaxial_anistropy = True
        model = prepare_topology_model(exc)
        iR0 = next(idx for idx, R in enumerate(model.Rlist) if np.all(R == 0))
        expected = 0.0004 * np.outer([1, 0, 0], [1, 0, 0]) + 0.0002 * np.outer(
            [0, 1, 0], [0, 1, 0]
        )
        np.testing.assert_allclose(model.JR[iR0, 0, 0], expected, atol=1e-15)
        # The input is not mutated by preparation.
        assert exc.k1 is not None

    def test_half_set_uniaxial_is_rejected(self):
        exc = make_exc(sia_tensor={0: np.zeros((3, 3))})
        exc.k1 = np.array([0.0002])
        exc.has_uniaxial_anistropy = True
        with pytest.raises(TopologyPreparationError, match="Ambiguous"):
            prepare_topology_model(exc)


class TestUnsupportedChannels:
    def test_biquadratic_rejected(self):
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            biquadratic={((1, 0, 0), 0, 1): 0.001},
        )
        with pytest.raises(TopologyPreparationError, match="biquadratic"):
            prepare_topology_model(exc)

    def test_imaginary_only_unsupported_channel_rejected(self):
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            biquadratic={((1, 0, 0), 0, 1): np.complex128(1e-3j)},
        )
        with pytest.raises(TopologyPreparationError, match="biquadratic"):
            prepare_topology_model(exc)

    def test_biquadratic_exclusion_recorded(self):
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            biquadratic={((1, 0, 0), 0, 1): 0.001},
        )
        model = prepare_topology_model(exc, exclude=("biquadratic",))
        assert model.exclusions == ("biquadratic",)
        assert model.unsupported_nonzero["biquadratic"] == 1

    def test_excluding_clean_channel_is_rejected(self):
        exchange, dmi, jani = basic_channels()
        exc = make_exc(exchange=exchange, dmi=dmi, jani=jani)
        with pytest.raises(TopologyPreparationError, match="no nonzero content"):
            prepare_topology_model(exc, exclude=("biquadratic",))

    def test_njt_channels_rejected(self):
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            njt_dmi={((1, 0, 0), 0, 1): np.array([0.0, 0.0, 0.0009])},
        )
        with pytest.raises(TopologyPreparationError, match="NJT_DMI"):
            prepare_topology_model(exc)


class TestPairReversal:
    def test_missing_reverse_dmi_rejected(self):
        exchange, dmi, _ = basic_channels()
        broken = dict(dmi)
        del broken[((0, -1, 0), 1, 0)]
        with pytest.raises(TopologyPreparationError, match="reverse"):
            prepare_topology_model(make_exc(exchange=exchange, dmi=broken))

    def test_asymmetric_jani_rejected(self):
        exchange, _, _ = basic_channels()
        asymmetric = jani_pair(
            (0, 1, 0),
            0,
            1,
            np.array([[0.0, 0.001, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        with pytest.raises(TopologyPreparationError, match="symmetric"):
            prepare_topology_model(make_exc(exchange=exchange, jani=asymmetric))


class TestComplexChannels:
    """Selected exchange channels must never discard imaginary content."""

    def test_material_imaginary_jiso_rejected(self):
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01 + 1e-4j))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01 + 1e-4j))
        with pytest.raises(TopologyPreparationError, match="Jiso.*imaginary"):
            prepare_topology_model(make_exc(exchange=exchange))

    def test_material_imaginary_dmi_rejected(self):
        exchange, _, _ = basic_channels()
        dmi = dmi_pair((1, 0, 0), 0, 1, np.array([0.0, 0.0, 1e-3 + 1e-5j]))
        with pytest.raises(TopologyPreparationError, match="DMI.*imaginary"):
            prepare_topology_model(make_exc(exchange=exchange, dmi=dmi))

    def test_material_imaginary_jani_rejected(self):
        exchange, _, _ = basic_channels()
        jani = jani_pair(
            (1, 1, 0),
            0,
            1,
            np.diag([2e-4, -1e-4, -1e-4]) + 1e-5j * np.eye(3),
        )
        with pytest.raises(TopologyPreparationError, match="Jani.*imaginary"):
            prepare_topology_model(make_exc(exchange=exchange, jani=jani))

    def test_material_imaginary_sia_rejected(self):
        sia = {0: np.diag([0.0, 0.0, 5e-4 + 1e-5j])}
        with pytest.raises(TopologyPreparationError, match="SIA.*imaginary"):
            prepare_topology_model(make_exc(sia_tensor=sia))

    def test_unselected_complex_channel_not_rejected(self):
        exchange, dmi, _ = basic_channels()
        jani = jani_pair(
            (1, 1, 0),
            0,
            1,
            np.diag([2e-4, -1e-4, -1e-4]) + 1e-5j * np.eye(3),
        )
        model = prepare_topology_model(
            make_exc(exchange=exchange, dmi=dmi, jani=jani),
            include={"Jiso": True, "Jani": False, "DMI": True, "SIA": True},
        )
        assert model.selection["Jani"] is False

    def test_complex_zero_input_matches_real_model(self):
        exchange, dmi, jani = basic_channels()
        sia = {0: 5e-4 * np.outer([0, 0, 1], [0, 0, 1])}
        real = prepare_topology_model(
            make_exc(exchange=exchange, dmi=dmi, jani=jani, sia_tensor=sia)
        )
        complexed = prepare_topology_model(
            make_exc(
                exchange={k: complex(v) for k, v in exchange.items()},
                dmi={k: np.asarray(v, dtype=complex) for k, v in dmi.items()},
                jani={k: np.asarray(v, dtype=complex) for k, v in jani.items()},
                sia_tensor={k: np.asarray(v, dtype=complex) for k, v in sia.items()},
            )
        )
        assert complexed.digest == real.digest
        np.testing.assert_array_equal(complexed.JR, real.JR)

    def test_imaginary_roundoff_within_tolerance_accepted(self):
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01 + 1e-9j))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01 + 1e-9j))
        model = prepare_topology_model(make_exc(exchange=exchange))
        iR = next(idx for idx, R in enumerate(model.Rlist) if tuple(R) == (1, 0, 0))
        np.testing.assert_array_equal(model.JR[iR, 0, 1], 0.01 * np.eye(3))

    def test_imaginary_content_above_roundoff_tolerance_rejected(self):
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01 + 2e-8j))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01 + 2e-8j))
        with pytest.raises(TopologyPreparationError, match="Jiso.*imaginary"):
            prepare_topology_model(make_exc(exchange=exchange))

    def test_roundoff_tolerance_follows_hermiticity_parameters(self):
        exchange = {}
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01 + 1e-9j))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01 + 1e-9j))
        with pytest.raises(TopologyPreparationError, match="Jiso.*imaginary"):
            prepare_topology_model(make_exc(exchange=exchange), hermiticity_rtol=1e-12)


class TestIdentity:
    def test_digest_is_content_identity(self):
        exchange, dmi, jani = basic_channels()
        m1 = prepare_topology_model(make_exc(exchange=exchange, dmi=dmi, jani=jani))
        m2 = prepare_topology_model(make_exc(exchange=exchange, dmi=dmi, jani=jani))
        assert m1.digest == m2.digest

    def test_changed_coupling_changes_digest(self):
        exchange, dmi, jani = basic_channels()
        m1 = prepare_topology_model(make_exc(exchange=exchange, dmi=dmi, jani=jani))
        tweaked = dict(exchange)
        tweaked[((1, 0, 0), 0, 1)] = 0.0100001
        tweaked[((-1, 0, 0), 1, 0)] = 0.0100001
        m2 = prepare_topology_model(
            make_exc(exchange=tweaked, dmi=dmi, jani=jani),
            source_path="/somewhere/else",
        )
        assert m1.digest != m2.digest
        # Source path is provenance, not identity.
        m3 = prepare_topology_model(
            make_exc(exchange=exchange, dmi=dmi, jani=jani), source_path="/x"
        )
        assert m1.digest == m3.digest

    def test_model_is_immutable(self):
        model = prepare_topology_model(make_exc(exchange=basic_channels()[0]))
        with pytest.raises(Exception):
            model.JR = None
        with pytest.raises(Exception):
            model.digest = "forged"

    def test_model_arrays_cannot_be_changed_through_magnon(self):
        model = prepare_topology_model(make_exc(exchange=basic_channels()[0]))
        expected = model.JR.copy()
        digest = model.digest
        with pytest.raises(ValueError):
            model.JR[0, 0, 0, 0, 0] = 1.0
        legacy = model.magnon()
        legacy.JR[0, 0, 0, 0, 0] += 1.0
        np.testing.assert_array_equal(model.JR, expected)
        assert model.digest == digest


class TestLegacyCompatibility:
    @pytest.mark.parametrize("topology", [False, True])
    def test_multisite_tb2j_bands_match_unit_spin_hamiltonian(self, topology):
        spins = ((0.0, 0.0, 3.0), (0.0, 0.0, 4.0), (0.0, 0.0, 0.0))
        exchange = {}
        families = (
            (0, 1, [(0, 0, 0), (0, 1, 0), (-1, 0, 0)], 0.001),
            (0, 0, [(1, 0, 0), (0, 1, 0), (1, 1, 0)], 0.0005),
            (1, 1, [(1, 0, 0), (0, 1, 0), (1, 1, 0)], 0.0005),
            (0, 1, [(1, 1, 0), (-1, 1, 0), (-1, -1, 0)], 0.00025),
        )
        for i, j, translations, coupling in families:
            for R in translations:
                exchange[(R, i, j)] = coupling
                exchange[(tuple(-x for x in R), j, i)] = coupling
        exc = make_exc(spinat=spins, exchange=exchange)
        if topology:
            model = prepare_topology_model(exc).magnon()
        else:
            model = Magnon.load_from_io(exc, Jani=False, DMI=False, SIA=False)
            model.set_reference((0, 0, 0), [[0, 0, 1]], (1, 0, 0))
        h = model.Hq(np.array([[1 / 3, 1 / 3, 0.0]]))[0]
        w = np.linalg.eigvals(np.diag([1, 1, -1, -1]) @ h)
        np.testing.assert_allclose(np.sort(np.abs(w.real))[::2] * 1000,
                                   [8.25, 11.0], atol=1e-8)

    def test_legacy_loader_accepts_biquadratic_source(self):
        """The prior band/DOS path silently ignores unsupported channels;
        topology-only strictness must not change that behavior."""
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            biquadratic={((1, 0, 0), 0, 1): 0.001},
        )
        magnon = Magnon.load_from_io(exc)
        magnon.set_reference((0, 0, 0), [[0.0, 0.0, 1.0]], (1, 0, 0))
        energies = magnon._magnon_energies(np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]))
        assert np.all(np.isfinite(energies))

    def test_legacy_band_cli_accepts_biquadratic_pickle(self, tmp_path):
        """The actual band workflow on a pickled source with an unsupported
        channel keeps working: strict rejection is topology-only."""
        exchange, dmi, jani = basic_channels()
        exc = make_exc(
            exchange=exchange,
            dmi=dmi,
            jani=jani,
            biquadratic={((1, 0, 0), 0, 1): 0.001},
        )
        results = tmp_path / "TB2J_results"
        exc.write_pickle(path=str(results))
        from TB2J.magnon.magnon3 import plot_magnon_bands_from_TB2J
        from TB2J.magnon.magnon_parameters import MagnonParameters

        output_file = tmp_path / "bands.png"
        params = MagnonParameters(
            path=str(results),
            kpath="GX",
            npoints=20,
            filename=str(output_file),
            show=False,
        )
        magnon = plot_magnon_bands_from_TB2J(params)
        assert output_file.exists()
        assert magnon.nspin == 2

    def test_prepared_magnon_matches_legacy_energies(self):
        """Same source, same translations: the seam must be energy-identical.

        All channels share the same shells here, so the legacy isotropic-only
        translation union and the canonical union coincide.
        """
        exchange = {}
        exchange.update(iso_pair((0, 0, 1), 0, 0, 0.002))
        exchange.update(iso_pair((0, 0, -1), 0, 0, 0.002))
        exchange.update(iso_pair((1, 0, 0), 0, 1, 0.01))
        exchange.update(iso_pair((-1, 0, 0), 1, 0, 0.01))
        dmi = dmi_pair((1, 0, 0), 0, 1, (0.0, 0.0, 0.001))
        jani = jani_pair((0, 0, 1), 0, 0, np.diag([0.0002, -0.0001, -0.0001]))
        exc = make_exc(exchange=exchange, dmi=dmi, jani=jani)
        legacy = Magnon.load_from_io(exc)
        legacy.set_reference((0, 0, 0), [[0.0, 0.0, 1.0]], (1, 0, 0))
        model = prepare_topology_model(exc)
        assert np.array(legacy.Rlist).shape == model.Rlist.shape
        modern = model.magnon()
        kpoints = np.array([[0.13, 0.27, 0.41], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        np.testing.assert_allclose(
            modern._magnon_energies(kpoints),
            legacy._magnon_energies(kpoints),
            atol=1e-14,
        )


class TestSelection:
    def test_channel_selection_drops_shells(self):
        exchange, dmi, jani = basic_channels()
        model = prepare_topology_model(
            make_exc(exchange=exchange, dmi=dmi, jani=jani),
            include={"Jiso": True, "Jani": False, "DMI": False, "SIA": True},
        )
        Rs = {tuple(R) for R in model.Rlist}
        assert (0, 1, 0) not in Rs  # DMI-only shell dropped
        assert (-1, -1, 0) not in Rs  # Jani-only shell dropped
        assert (1, 0, 0) in Rs
        assert model.selection == {
            "Jiso": True,
            "Jani": False,
            "DMI": False,
            "SIA": True,
        }
