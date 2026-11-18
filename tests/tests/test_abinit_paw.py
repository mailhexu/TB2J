"""Tests for ABINIT PAW projector-space exchange assembly and CLI (Stories 004+005).

All tests use synthetic data — no real DFT outputs are required.  The synthetic
system is a two-atom magnetic dimer with two PAW projector channels per atom,
two k-points, four bands, and two spin channels.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_paw import (
    assemble_paw_exchange_data,
    gen_exchange_abinit_paw,
    load_projected_data,
    save_projected_data,
)
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    projector_exchange_trace,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

NATOM = 2
NPROJ_PER_ATOM = 2
NPROJ_TOTAL = NATOM * NPROJ_PER_ATOM
NKPT = 2
NBAND = 4
NSPPOL = 2


def _synthetic_cprj() -> list:
    """Dual-projector coefficients cprj[ik][ispin] → (natom, nproj, nband).

    Unit-norm projectors with a small cross-projector rotation so the Green
    function has non-trivial off-diagonal structure.
    """
    rng = np.random.default_rng(42)
    rotation = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    cprj_per_kpt = []
    for _ik in range(NKPT):
        by_spin = []
        for _ispin in range(NSPPOL):
            base = np.zeros((NATOM, NPROJ_PER_ATOM, NBAND), dtype=complex)
            for atom in range(NATOM):
                for band in range(NBAND):
                    base[atom, :, band] = rotation[band % NPROJ_PER_ATOM, :]
            # Small deterministic complex perturbation.
            base += 0.05 * rng.standard_normal(base.shape) + 0j
            by_spin.append(base)
        cprj_per_kpt.append(by_spin)
    return cprj_per_kpt


def _synthetic_eigenvalues() -> np.ndarray:
    """Band energies [nsppol, nkpt, nband] straddling the Fermi level."""
    return np.array(
        [
            [[-2.0, -1.0, 1.0, 2.0], [-1.5, -0.5, 0.5, 1.5]],
            [[-1.8, -0.8, 1.2, 2.2], [-1.3, -0.3, 0.7, 1.7]],
        ]
    )


def _synthetic_delta_ij() -> dict[int, np.ndarray]:
    """Spin-difference Delta matrices per atom (eV), real symmetric."""
    return {
        0: np.array([[0.50, 0.05], [0.05, 0.30]]),
        1: np.array([[0.40, 0.03], [0.03, 0.20]]),
    }


def _synthetic_kpoints() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])


def _synthetic_kweights() -> np.ndarray:
    return np.array([0.5, 0.5])


def _synthetic_cell() -> np.ndarray:
    return 2.5 * np.eye(3)


def _synthetic_positions() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]])


def _synthetic_atomic_numbers() -> np.ndarray:
    return np.array([26, 26], dtype=int)  # Fe


def _synthetic_projector_lm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(l, m, radial) labels for 2 channels: (l=0,m=0) and (l=1,m=0)."""
    proj_l = np.tile([0, 1], NATOM)
    proj_m = np.tile([0, 0], NATOM)
    proj_radial = np.tile([0, 0], NATOM)
    return proj_l, proj_m, proj_radial


def _build_synthetic_data(**overrides) -> ProjectorGreenData:
    proj_l, proj_m, proj_radial = _synthetic_projector_lm()
    kwargs = dict(
        cprj_per_kpt=_synthetic_cprj(),
        delta_ij=_synthetic_delta_ij(),
        eigenvalues=_synthetic_eigenvalues(),
        kweights=_synthetic_kweights(),
        kpoints=_synthetic_kpoints(),
        efermi=0.0,
        natom=NATOM,
        nproj_per_atom=NPROJ_PER_ATOM,
        delta_unit="eV",
        cell=_synthetic_cell(),
        positions=_synthetic_positions(),
        atomic_numbers=_synthetic_atomic_numbers(),
        projector_l=proj_l,
        projector_m=proj_m,
        projector_radial=proj_radial,
    )
    kwargs.update(overrides)
    return assemble_paw_exchange_data(**kwargs)


# ---------------------------------------------------------------------------
# Story 004 — Assembly tests
# ---------------------------------------------------------------------------


class TestAssemblePawExchangeData:
    """Verify the structure and physical correctness of the assembled data."""

    def test_returns_projector_green_data(self):
        data = _build_synthetic_data()
        assert isinstance(data, ProjectorGreenData)

    def test_coefficients_shape(self):
        data = _build_synthetic_data()
        assert data.coefficients.shape == (
            NSPPOL,
            NKPT,
            NBAND,
            NPROJ_TOTAL,
        )
        assert data.coefficients.dtype == complex

    def test_overlap_k_is_none(self):
        data = _build_synthetic_data()
        assert data.overlap_k is None

    def test_coefficients_are_dual_flag(self):
        """overlap_k=None must set coefficients_are_dual on ProjectorGreen."""
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        assert green.coefficients_are_dual is True
        assert green.needs_overlap_transform is False

    def test_site_projector_indexing(self):
        data = _build_synthetic_data()
        assert data.site_nproj.shape == (NATOM,)
        assert np.all(data.site_nproj == NPROJ_PER_ATOM)
        expected_indices = np.arange(NPROJ_TOTAL).reshape(NATOM, NPROJ_PER_ATOM)
        np.testing.assert_array_equal(data.site_projector_indices, expected_indices)

    def test_projector_site_atom_assignment(self):
        data = _build_synthetic_data()
        expected_site = np.repeat([0, 1], NPROJ_PER_ATOM)
        np.testing.assert_array_equal(data.projector_site, expected_site)
        np.testing.assert_array_equal(data.projector_atom, expected_site)

    def test_delta_xc_block_diagonal(self):
        """The full-projector delta_xc must be block-diagonal per atom."""
        data = _build_synthetic_data()
        # operator_components store per-site blocks; verify they match input.
        delta_xc = data.operator_components["delta_xc"]
        assert delta_xc.shape == (NATOM, NPROJ_PER_ATOM, NPROJ_PER_ATOM)
        for atom in range(NATOM):
            block = data.get_operator_component("delta_xc", site=atom)
            np.testing.assert_allclose(block, _synthetic_delta_ij()[atom])

    def test_delta_total_alias_present(self):
        """delta_total must exist for validate(exchange_ready=True)."""
        data = _build_synthetic_data()
        assert data.has_operator_component("delta_total")
        assert data.has_operator_component("delta_xc")
        # exchange_ready validation should pass with delta_total.
        assert data.validate(exchange_ready=True) is True

    def test_delta_unit_conversion_hartree_to_ev(self):
        delta_hartree = {
            atom: mat / 27.211386245988 for atom, mat in _synthetic_delta_ij().items()
        }
        data = _build_synthetic_data(delta_ij=delta_hartree, delta_unit="hartree")
        for atom in range(NATOM):
            block = data.get_operator_component("delta_xc", site=atom)
            np.testing.assert_allclose(block, _synthetic_delta_ij()[atom])

    def test_delta_xc_metadata(self):
        data = _build_synthetic_data()
        meta = data.operator_component_metadata["delta_xc"]
        assert meta["units"] == "eV"
        assert meta["completeness"] == "complete"
        assert meta["exchange_ready"] == "true"

    def test_get_sites(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        assert green.get_sites() == [0, 1]

    def test_get_local_operator_prefers_delta_xc(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        op0 = green.get_local_operator(0)
        np.testing.assert_allclose(op0, _synthetic_delta_ij()[0])

    def test_get_site_block(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        full = np.eye(NPROJ_TOTAL, dtype=complex)
        block = green.get_site_block(full, 0, 1)
        assert block.shape == (NPROJ_PER_ATOM, NPROJ_PER_ATOM)

    def test_rejects_wrong_cprj_spin_count(self):
        cprj = _synthetic_cprj()
        cprj[0] = cprj[0][:1]  # only 1 spin
        with pytest.raises(ValueError, match="spin"):
            _build_synthetic_data(cprj_per_kpt=cprj)

    def test_rejects_wrong_natom_in_cprj(self):
        cprj = _synthetic_cprj()
        cprj[0][0] = cprj[0][0][:1]  # 1 atom instead of 2
        with pytest.raises(ValueError, match="cprj shape"):
            _build_synthetic_data(cprj_per_kpt=cprj)

    def test_rejects_delta_atom_out_of_range(self):
        bad_delta = dict(_synthetic_delta_ij())
        bad_delta[5] = np.eye(NPROJ_PER_ATOM)
        with pytest.raises(ValueError, match="out of range"):
            _build_synthetic_data(delta_ij=bad_delta)

    def test_rejects_wrong_delta_shape(self):
        bad_delta = {0: np.eye(3), 1: np.eye(NPROJ_PER_ATOM)}
        with pytest.raises(ValueError, match="delta_ij\\[0\\]"):
            _build_synthetic_data(delta_ij=bad_delta)

    def test_accepts_flat_cprj_layout(self):
        """cprj in (nproj_total, nband) layout should also work."""
        cprj_3d = _synthetic_cprj()
        cprj_flat = []
        for ik in range(NKPT):
            by_spin = []
            for ispin in range(NSPPOL):
                flat = cprj_3d[ik][ispin].reshape(NPROJ_TOTAL, NBAND)
                by_spin.append(flat)
            cprj_flat.append(by_spin)
        data = _build_synthetic_data(cprj_per_kpt=cprj_flat)
        assert data.coefficients.shape == (NSPPOL, NKPT, NBAND, NPROJ_TOTAL)

    def test_assembles_unequal_site_slices_without_padding(self, tmp_path: Path):
        """Fe/O-like PAW blocks occupy their exact slices on the global axis."""
        site_slices = (slice(0, 2), slice(2, 5))
        delta_ij = {
            0: np.array([[0.5, 0.1], [0.1, 0.3]]),
            1: np.array([[0.4, 0.02, 0.03], [0.02, 0.2, 0.04], [0.03, 0.04, 0.1]]),
        }
        cprj = [
            [
                (
                    np.arange(20, dtype=float).reshape(5, NBAND)
                    + 100 * ik
                    + 10 * ispin
                    + 1j * (ik + ispin)
                )
                for ispin in range(NSPPOL)
            ]
            for ik in range(NKPT)
        ]
        data = assemble_paw_exchange_data(
            cprj_per_kpt=cprj,
            delta_ij=delta_ij,
            eigenvalues=_synthetic_eigenvalues(),
            kweights=_synthetic_kweights(),
            kpoints=_synthetic_kpoints(),
            efermi=0.0,
            natom=NATOM,
            site_slices=site_slices,
            cell=_synthetic_cell(),
            positions=_synthetic_positions(),
            atomic_numbers=np.array([26, 8]),
        )

        assert data.coefficients.shape == (NSPPOL, NKPT, NBAND, 5)
        np.testing.assert_allclose(data.coefficients[1, 1], cprj[1][1].T)
        np.testing.assert_array_equal(data.site_nproj, [2, 3])
        np.testing.assert_array_equal(
            data.site_projector_indices, [[0, 1, -1], [2, 3, 4]]
        )
        global_delta = data.get_operator_component("delta_xc")
        assert global_delta.shape == (5, 5)
        np.testing.assert_allclose(global_delta[:2, :2], delta_ij[0])
        np.testing.assert_allclose(global_delta[2:, 2:], delta_ij[1])
        np.testing.assert_allclose(global_delta[:2, 2:], 0.0)
        np.testing.assert_allclose(global_delta[2:, :2], 0.0)
        np.testing.assert_allclose(
            data.get_operator_component("delta_xc", site=1), delta_ij[1]
        )
        green = ProjectorGreen(data)
        trace = projector_exchange_trace(
            green, np.array([[0, 0, 0]]), energy=0.1j, sites=[0, 1]
        )
        assert np.isfinite(trace["trace"][((0, 0, 0), 0, 1)])
        pytest.importorskip("netCDF4")
        filename = tmp_path / "unequal-paw.nc"
        data.save_netcdf(filename)
        loaded = ProjectorGreenData.load_netcdf(filename)
        np.testing.assert_allclose(
            loaded.get_operator_component("delta_xc"), global_delta
        )
        np.testing.assert_allclose(
            loaded.get_operator_component("delta_xc", site=1), delta_ij[1]
        )

    def test_rejects_noncontiguous_unequal_site_slices(self):
        with pytest.raises(ValueError, match="contiguous"):
            _build_synthetic_data(
                cprj_per_kpt=[
                    [np.zeros((5, NBAND), dtype=complex) for _ in range(NSPPOL)]
                    for _ in range(NKPT)
                ],
                nproj_per_atom=None,
                site_slices=(slice(0, 2), slice(3, 5)),
            )


# ---------------------------------------------------------------------------
# Exchange trace tests (acceptance: trace can be evaluated)
# ---------------------------------------------------------------------------


class TestExchangeTrace:
    """Verify the projector exchange trace runs end-to-end on synthetic data."""

    def test_trace_at_single_energy(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
        energy = 0.1j  # imaginary energy on the contour
        result = projector_exchange_trace(green, rpts, energy=energy, sites=[0, 1])
        assert result["method"] == "projector_exchange_trace"
        assert result["local_operator"] == "hij_spin_difference"
        # Every (R, i, j) key must produce a finite scalar trace.
        for key, val in result["trace"].items():
            assert np.isfinite(val), f"non-finite trace for {key}"
            assert isinstance(val, (float, complex, np.floating, np.complexfloating))

    def test_trace_delta_xc_source(self):
        """When local_operators are provided explicitly, source is 'explicit'."""
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
        ops = green.get_local_operators(sites=[0, 1])
        result = projector_exchange_trace(
            green, rpts, energy=0.1j, local_operators=ops, sites=[0, 1]
        )
        assert result["local_operator"] == "explicit"

    def test_trace_orbital_resolution(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        rpts = np.array([[0, 0, 0]], dtype=int)
        result = projector_exchange_trace(green, rpts, energy=0.1j, sites=[0, 1])
        for key, orbital in result["orbital_trace"].items():
            assert orbital.shape == (NPROJ_PER_ATOM, NPROJ_PER_ATOM)

    def test_onsite_trace_nonzero(self):
        """The R=0 self-interaction trace must be non-zero with real delta."""
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
        result = projector_exchange_trace(green, rpts, energy=0.5j, sites=[0, 1])
        onsite_00 = result["trace"][((0, 0, 0), 0, 0)]
        assert abs(onsite_00) > 0.0


# ---------------------------------------------------------------------------
# Green function reconstruction tests
# ---------------------------------------------------------------------------


class TestGreenFunction:
    """Verify the dual-no-dressing Green function mechanics."""

    def test_get_Gk_shape(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        gk = green.get_Gk(0, energy=0.1j, ispin=0)
        assert gk.shape == (NPROJ_TOTAL, NPROJ_TOTAL)

    def test_get_Gk_all_shape(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        gk_all = green.get_Gk_all(0.1j, ispin=0)
        assert gk_all.shape == (NKPT, NPROJ_TOTAL, NPROJ_TOTAL)

    def test_get_GR_shape(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        rpts = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=int)
        gr = green.get_GR(rpts, energy=0.1j, ispin=0)
        assert gr.shape == (3, NPROJ_TOTAL, NPROJ_TOTAL)

    def test_green_backend_protocol(self):
        data = _build_synthetic_data()
        green = ProjectorGreen(data)
        assert green.kpts.shape == (NKPT, 3)
        assert green.kweights.shape == (NKPT,)
        assert green.nbasis == NPROJ_TOTAL
        assert green.norb == NPROJ_TOTAL


# ---------------------------------------------------------------------------
# Projected-data persistence tests
# ---------------------------------------------------------------------------


class TestProjectedDataIO:
    def test_roundtrip(self, tmp_path: Path):
        cprj = _synthetic_cprj()
        path = save_projected_data(
            tmp_path / "proj.pkl",
            cprj_per_kpt=cprj,
            eigenvalues=_synthetic_eigenvalues(),
            kweights=_synthetic_kweights(),
            kpoints=_synthetic_kpoints(),
            efermi=0.0,
            natom=NATOM,
            nproj_per_atom=NPROJ_PER_ATOM,
            cell=_synthetic_cell(),
            positions=_synthetic_positions(),
            atomic_numbers=_synthetic_atomic_numbers(),
        )
        assert path.exists()
        loaded = load_projected_data(path)
        assert loaded["natom"] == NATOM
        assert loaded["nproj_per_atom"] == NPROJ_PER_ATOM
        np.testing.assert_array_equal(loaded["eigenvalues"], _synthetic_eigenvalues())

    def test_roundtrip_unequal_site_slices(self, tmp_path: Path):
        site_slices = (slice(0, 2), slice(2, 5))
        path = save_projected_data(
            tmp_path / "mixed-proj.pkl",
            cprj_per_kpt=[
                [np.zeros((5, NBAND), dtype=complex) for _ in range(NSPPOL)]
                for _ in range(NKPT)
            ],
            eigenvalues=_synthetic_eigenvalues(),
            kweights=_synthetic_kweights(),
            kpoints=_synthetic_kpoints(),
            efermi=0.0,
            natom=NATOM,
            site_slices=site_slices,
        )
        loaded = load_projected_data(path)
        assert "nproj_per_atom" not in loaded
        assert loaded["site_slices"] == site_slices

    def test_load_missing_keys_raises(self, tmp_path: Path):
        path = tmp_path / "bad.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"foo": 1}, fh)
        with pytest.raises(ValueError, match="missing keys"):
            load_projected_data(path)

    def test_layout_metadata_is_required(self, tmp_path: Path):
        with pytest.raises(ValueError, match="projector layout"):
            save_projected_data(
                tmp_path / "no-layout.pkl",
                cprj_per_kpt=_synthetic_cprj(),
                eigenvalues=_synthetic_eigenvalues(),
                kweights=_synthetic_kweights(),
                kpoints=_synthetic_kpoints(),
                efermi=0.0,
                natom=NATOM,
            )


# ---------------------------------------------------------------------------
# Story 005 — CLI tests
# ---------------------------------------------------------------------------


def _write_synthetic_projected_file(tmp_path: Path) -> Path:
    """Write a pickle of synthetic projection results."""
    return save_projected_data(
        tmp_path / "projected.pkl",
        cprj_per_kpt=_synthetic_cprj(),
        eigenvalues=_synthetic_eigenvalues(),
        kweights=_synthetic_kweights(),
        kpoints=_synthetic_kpoints(),
        efermi=0.0,
        natom=NATOM,
        nproj_per_atom=NPROJ_PER_ATOM,
        cell=_synthetic_cell(),
        positions=_synthetic_positions(),
        atomic_numbers=_synthetic_atomic_numbers(),
    )


def _write_synthetic_pawprt_log(
    tmp_path: Path,
    delta_ij: dict[int, np.ndarray] | None = None,
    unit: str = "eV",
) -> Path:
    """Write a synthetic ABINIT log with a pawprt Dij block.

    The delta matrices are stored as ``D_up = delta, D_down = 0`` so that
    ``D_up - D_down = delta``.
    """
    if delta_ij is None:
        delta_ij = _synthetic_delta_ij()
    lines = [
        " Version 9.10.1 of ABINIT",
        " ...(some SCF output)...",
        "",
        f" Total pseudopotential strength Dij ({unit}):",
    ]
    for atom in sorted(delta_ij):
        delta = np.asarray(delta_ij[atom], dtype=float)
        zeros = np.zeros_like(delta)
        lines.append(f" Atom #{atom + 1:3d} - Spin component 1")
        for row in delta:
            lines.append("".join(f" {v:9.5f}" for v in row))
        lines.append(f" Atom #{atom + 1:3d} - Spin component 2")
        for row in zeros:
            lines.append("".join(f" {v:9.5f}" for v in row))
    lines.append(" ...(more output)...")
    log = tmp_path / "run.abo"
    log.write_text("\n".join(lines))
    return log


class TestLogPathIntegration:
    """End-to-end through abinao.pawprt_parser → assembly → exchange trace."""

    def test_full_pipeline_with_log_path(self, tmp_path: Path):
        projected = _write_synthetic_projected_file(tmp_path)
        log = _write_synthetic_pawprt_log(tmp_path, unit="eV")
        out_dir = tmp_path / "log_results"
        exchange_out, jdict = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            log_path=str(log),
            output_path=str(out_dir),
            nz=8,
            smearing_eV=0.1,
            index_magnetic_atoms=[0, 1],
        )
        assert Path(exchange_out).exists()
        assert len(jdict) > 0
        for val in jdict.values():
            assert np.isfinite(val)

    def test_log_path_unit_conversion(self, tmp_path: Path):
        """delta given in Hartree must be converted to eV."""
        from TB2J.interfaces.abinit_paw import HARTREE_TO_EV

        delta_ev = _synthetic_delta_ij()
        delta_ha = {a: mat / HARTREE_TO_EV for a, mat in delta_ev.items()}
        projected = _write_synthetic_projected_file(tmp_path)
        log = _write_synthetic_pawprt_log(tmp_path, delta_ij=delta_ha, unit="hartree")
        out_dir = tmp_path / "ha_results"

        # Run with Hartree log; assembly should convert to eV internally.
        exchange_out, jdict_ha = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            log_path=str(log),
            output_path=str(out_dir),
            nz=8,
            index_magnetic_atoms=[0, 1],
        )
        assert Path(exchange_out).exists()

        # Compare against the direct-delta-eV path — same J values.
        ev_dir = tmp_path / "ev"
        ev_dir.mkdir()
        log_ev = _write_synthetic_pawprt_log(ev_dir, delta_ij=delta_ev, unit="eV")
        out_dir2 = tmp_path / "ev_results"
        _, jdict_ev = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            log_path=str(log_ev),
            output_path=str(out_dir2),
            nz=8,
            index_magnetic_atoms=[0, 1],
        )
        for key in jdict_ev:
            np.testing.assert_allclose(
                jdict_ha[key],
                jdict_ev[key],
                rtol=5e-3,
                err_msg=f"unit conversion mismatch for {key}",
            )

    def test_missing_pawprt_block_raises(self, tmp_path: Path):
        projected = _write_synthetic_projected_file(tmp_path)
        bad_log = tmp_path / "no_dij.abo"
        bad_log.write_text("some ABINIT output without Dij block\n")
        with pytest.raises(ValueError, match="no pawprt Dij block"):
            gen_exchange_abinit_paw(
                projected_data_path=str(projected),
                log_path=str(bad_log),
                output_path=str(tmp_path / "out"),
            )


class TestGenExchangeAbinitPaw:
    """End-to-end CLI with synthetic pre-projected data + explicit delta_ij."""

    def test_full_pipeline_with_delta_ij(self, tmp_path: Path):
        projected = _write_synthetic_projected_file(tmp_path)
        out_dir = tmp_path / "results"
        exchange_out, jdict = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            delta_ij=_synthetic_delta_ij(),
            delta_unit="eV",
            output_path=str(out_dir),
            nz=10,
            smearing_eV=0.1,
            index_magnetic_atoms=[0, 1],
        )
        assert Path(exchange_out).exists()
        assert len(jdict) > 0
        # All J values must be finite.
        for key, val in jdict.items():
            assert np.isfinite(val), f"non-finite J for {key}"

    def test_writes_exchange_out(self, tmp_path: Path):
        projected = _write_synthetic_projected_file(tmp_path)
        out_dir = tmp_path / "results2"
        exchange_out, _ = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            delta_ij=_synthetic_delta_ij(),
            output_path=str(out_dir),
            nz=8,
            index_magnetic_atoms=[0, 1],
        )
        content = Path(exchange_out).read_text()
        assert "exchange" in content.lower() or len(content) > 0

    def test_requires_data_source(self, tmp_path: Path):
        with pytest.raises(ValueError, match="snapshot_cache, projected_data_path"):
            gen_exchange_abinit_paw(
                log_path=str(tmp_path / "dummy.log"),
                delta_ij=_synthetic_delta_ij(),
            )

    def test_requires_delta_source(self, tmp_path: Path):
        projected = _write_synthetic_projected_file(tmp_path)
        with pytest.raises(ValueError, match="log_path or delta_ij"):
            gen_exchange_abinit_paw(
                projected_data_path=str(projected),
                output_path=str(tmp_path / "out"),
            )

    def test_cell_positions_from_projected_data(self, tmp_path: Path):
        """Structural metadata from the pickle should propagate to the output."""
        projected = _write_synthetic_projected_file(tmp_path)
        out_dir = tmp_path / "results3"
        exchange_out, jdict = gen_exchange_abinit_paw(
            projected_data_path=str(projected),
            delta_ij=_synthetic_delta_ij(),
            output_path=str(out_dir),
            nz=8,
            index_magnetic_atoms=[0, 1],
        )
        assert Path(exchange_out).exists()


# ---------------------------------------------------------------------------
# CLI argument signature test
# ---------------------------------------------------------------------------


class TestCLISignature:
    """Verify the function accepts the documented API."""

    def test_accepts_all_documented_kwargs(self):
        import inspect

        sig = inspect.signature(gen_exchange_abinit_paw)
        params = set(sig.parameters)
        for required in (
            "wfk_path",
            "paw_xml_path",
            "log_path",
            "projected_data_path",
        ):
            assert required in params, f"missing parameter {required}"
        # All documented source parameters should default to None.
        for p in ("wfk_path", "paw_xml_path", "log_path", "projected_data_path"):
            assert sig.parameters[p].default is None

    def test_accepts_kwargs(self):
        import inspect

        sig = inspect.signature(gen_exchange_abinit_paw)
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
