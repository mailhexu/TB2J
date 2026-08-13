from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
from TB2J.projector_green import (
    ProjectorGreen,
    ProjectorGreenData,
    build_site_projector_indices,
    encode_complex,
    pack_site_hij,
    project_potential_to_hij,
    projector_charge_moments_from_green,
    projector_exchange_trace,
    validate_green_backend,
)

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_projector_green_example():
    path = (
        ROOT_DIR / "examples" / "projector_green" / "build_synthetic_projector_green.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_synthetic_projector_green", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_gpaw_bcc_fe_example():
    path = ROOT_DIR / "examples" / "projector_green" / "gpaw_bcc_fe_projector_green.py"
    spec = importlib.util.spec_from_file_location("gpaw_bcc_fe_projector_green", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_bcc_fe_projector_data():
    a = 2.86
    cell = (
        0.5
        * a
        * np.array(
            [
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        )
    )
    eigenvalues = np.array(
        [
            [[0.0, 2.0], [0.5, 2.5]],
            [[0.2, 2.2], [0.7, 2.7]],
        ]
    )
    coefficients = np.zeros((2, 2, 2, 2), dtype=complex)
    coefficients[:, :, 0, 0] = 1.0
    coefficients[:, :, 1, 1] = 1.0
    hij = np.array(
        [
            [[[1.0, 0.1j], [-0.1j, 2.0]]],
            [[[0.4, 0.0], [0.0, 1.5]]],
        ],
        dtype=complex,
    )
    return ProjectorGreenData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        weights=np.array([0.5, 0.5]),
        eigenvalues=eigenvalues,
        coefficients=coefficients,
        efermi=0.0,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        cell=cell,
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([26]),
        projector_l=np.array([2, 2]),
        projector_m=np.array([-2, -1]),
        projector_radial=np.array([0, 0]),
        overlap_metric=np.array([[1.0, 0.2j], [-0.2j, 1.5]]),
        site_nproj=np.array([2]),
        site_projector_indices=np.array([[0, 1]]),
        hij=hij,
        hij_definition="paw_dij_projector_hamiltonian",
        hij_units="eV",
        hij_source="GPAW dH_asp",
        hij_projection="native PAW projector Hamiltonian matrix",
        coefficient_source="gpaw.P_ani",
        coefficient_projector="dual_paw_projector",
        channel_interpretation="paw_partial_wave_channel",
        overlap_metric_definition="GPAW PAW onsite dO_ii correction",
        population_metric="GPAW PAW N0_p packed density contraction",
        operator_basis="native_paw_projector_hamiltonian",
        metadata={
            "source": "synthetic bcc Fe primitive cell",
            "hij_source_name": "PAW d_ij / dH_asp",
            "hij_usage": "spin-dependent part H_ij^up - H_ij^down",
        },
    )


def make_overlap_k_projector_data():
    coefficients = np.array(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0j], [0.5, -1.0]],
            ]
        ],
        dtype=complex,
    )
    return ProjectorGreenData(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        weights=np.array([0.25, 0.75]),
        eigenvalues=np.array([[[0.0, 2.0], [0.5, 3.0]]]),
        coefficients=coefficients,
        efermi=0.1,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        overlap_k=np.array(
            [
                [[2.0, 0.1j], [-0.1j, 1.5]],
                [[1.5, 0.2 - 0.1j], [0.2 + 0.1j, 2.0]],
            ],
            dtype=complex,
        ),
        overlap_metric=np.array([[9.0, 0.0], [0.0, 9.0]], dtype=complex),
        overlap_metric_definition="synthetic k-dependent NC PAO overlap",
    )


def make_nc_pao_exchange_data():
    coefficients = np.zeros((2, 1, 2, 2), dtype=complex)
    coefficients[:, 0, 0, 0] = 1.0
    coefficients[:, 0, 1, 1] = 1.0
    return ProjectorGreenData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        weights=np.array([1.0]),
        eigenvalues=np.array([[[0.0, 2.0]], [[0.2, 2.2]]]),
        occupations=np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        coefficients=coefficients,
        efermi=0.5,
        projector_site=np.array([0, 0]),
        projector_atom=np.array([0, 0]),
        cell=np.eye(3) * 3.0,
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([26]),
        overlap_k=np.array([[[1.2, 0.1j], [-0.1j, 1.5]]], dtype=complex),
        site_nproj=np.array([2]),
        site_projector_indices=np.array([[0, 1]]),
        operator_components={
            "delta_total": np.array([[[1.0, 0.05], [0.05, 0.8]]], dtype=complex),
            "delta_xc": np.array([[[0.9, 0.0], [0.0, 0.7]]], dtype=complex),
        },
        operator_component_metadata={
            "delta_total": {
                "source": "synthetic",
                "units": "eV",
                "operator_basis": "abinit_nc_pao",
                "spin_treatment": "spin_difference",
                "completeness": "complete",
            },
            "delta_xc": {
                "source": "synthetic",
                "units": "eV",
                "operator_basis": "abinit_nc_pao",
                "spin_treatment": "spin_difference",
                "completeness": "complete",
            },
        },
        coefficient_source="abinit.nc_pao",
        coefficient_projector="nc_pao",
        channel_interpretation="norm_conserving_pao",
        overlap_metric_definition="validated synthetic k-dependent NC PAO overlap",
        operator_basis="abinit_nc_pao",
        metadata={"source_code": "abinit"},
    )


def write_abinit_nc_pao_fixture(filename, data, overlap_definition=True):
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(filename, "w") as nc:
        nc.createDimension("nspin", data.nspin)
        nc.createDimension("nkpt", data.nkpt)
        nc.createDimension("nband", data.nband)
        nc.createDimension("nproj", data.nproj)
        nc.createDimension("nsite", len(data.site_nproj))
        nc.createDimension("nproj_site_max", data.site_projector_indices.shape[1])
        nc.createDimension("natom", len(data.atomic_numbers))
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.schema_name = "abinit.savetb2j.nc_pao"
        nc.schema_version = "1.0"
        nc.source_code = "abinit"
        nc.spin_mode = "collinear"
        nc.spin_channel_order = "up,down"
        nc.full_bz = "true"
        nc.coefficient_source = "abinit.nc_pao"
        nc.operator_basis = "abinit_nc_pao"

        structure = nc.createGroup("structure")
        structure.createVariable("cell", "f8", ("three", "three"))[:] = data.cell
        structure.createVariable("positions", "f8", ("natom", "three"))[:] = (
            data.positions
        )
        structure.createVariable("atomic_numbers", "i4", ("natom",))[:] = (
            data.atomic_numbers
        )
        kpoints = nc.createGroup("kpoints")
        kpoints.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = data.kpoints
        kpoints.createVariable("weights", "f8", ("nkpt",))[:] = data.weights
        bands = nc.createGroup("bands")
        bands.efermi = data.efermi
        bands.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = (
            data.eigenvalues
        )
        bands.createVariable("occupations", "f8", ("nspin", "nkpt", "nband"))[:] = (
            data.occupations
        )
        projectors = nc.createGroup("projectors")
        projectors.coefficient_projector = "nc_pao"
        projectors.channel_interpretation = "norm_conserving_pao"
        projectors.createVariable(
            "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
        )[:] = encode_complex(data.coefficients)
        overlap_k = projectors.createVariable(
            "overlap_k", "f8", ("nkpt", "nproj", "nproj", "complex")
        )
        overlap_k[:] = encode_complex(data.overlap_k)
        if overlap_definition:
            overlap_k.definition = data.overlap_metric_definition
        projectors.createVariable("projector_site", "i4", ("nproj",))[:] = (
            data.projector_site
        )
        projectors.createVariable("projector_atom", "i4", ("nproj",))[:] = (
            data.projector_atom
        )
        projectors.createVariable("site_nproj", "i4", ("nsite",))[:] = data.site_nproj
        projectors.createVariable(
            "site_projector_indices", "i4", ("nsite", "nproj_site_max")
        )[:] = data.site_projector_indices
        operators = nc.createGroup("operators")
        components = operators.createGroup("operator_components")
        for name, component in data.operator_components.items():
            variable = components.createVariable(
                name, "f8", ("nsite", "nproj_site_max", "nproj_site_max", "complex")
            )
            variable[:] = encode_complex(component)
            for key, value in data.operator_component_metadata[name].items():
                setattr(variable, key, value)


def write_abinit_nc_pao_hs_v2_fixture(
    filename,
    data,
    overlap_exchange_ready=1,
    full_bz=False,
    abinit_structure_units=False,
):
    netcdf4 = pytest.importorskip("netCDF4")
    hartree = 27.211386245988
    with netcdf4.Dataset(filename, "w") as nc:
        nc.createDimension("nproj", data.nproj)
        nc.createDimension("nkpt_ibz", data.nkpt)
        if full_bz:
            nc.createDimension("nkpt_bz", data.nkpt)
        nc.createDimension("nsppol", data.nspin)
        nc.createDimension("nband", data.nband)
        nc.createDimension("natom", len(data.atomic_numbers))
        nc.createDimension("ntypat", len(data.atomic_numbers))
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.schema_name = "abinit.nc_pao_hs"
        nc.schema_version = 2
        nc.basis_type = "pseudo_atomic_orbital"
        nc.complex_storage = "split_real_imag_variables"
        nc.overlap_exchange_ready = overlap_exchange_ready
        nc.overlap_validation_status = "validated_synthetic_fixture"
        nc.pao_overlap_convention = "synthetic k-dependent NC PAO overlap"
        nc.pao_coefficient_kpoint_set = "direct_full_bz" if full_bz else "ibz_only"
        nc.fermi_energy = data.efermi / hartree

        nc.createVariable("atom_index", "i4", ("nproj",))[:] = data.projector_atom + 1
        nc.createVariable("l_quantum", "i4", ("nproj",))[:] = np.array([0, 0])
        nc.createVariable("m_quantum", "i4", ("nproj",))[:] = np.array([0, 0])
        nc.createVariable("n_quantum", "i4", ("nproj",))[:] = np.array([1, 2])
        atom_positions = nc.createVariable("atom_positions", "f8", ("natom", "three"))
        primitive_vectors = nc.createVariable(
            "primitive_vectors", "f8", ("three", "three")
        )
        if abinit_structure_units:
            from ase.units import Bohr

            atom_positions[:] = np.linalg.solve(data.cell.T, data.positions.T).T
            atom_positions.units = "dimensionless"
            atom_positions.mnemonics = "Reduced atomic positions"
            primitive_vectors[:] = data.cell / Bohr
            primitive_vectors.units = "dimensionless"
            primitive_vectors.mnemonics = "Primitive cell vectors (bohr)"
        else:
            atom_positions[:] = data.positions
            primitive_vectors[:] = data.cell
        nc.createVariable("atom_types", "i4", ("natom",))[:] = np.ones(
            len(data.atomic_numbers), dtype=int
        )
        nc.createVariable("atomic_numbers", "f8", ("ntypat",))[:] = data.atomic_numbers
        nc.createVariable("kpoints_ibz", "f8", ("nkpt_ibz", "three"))[:] = data.kpoints
        nc.createVariable("kweights_ibz", "f8", ("nkpt_ibz",))[:] = data.weights
        nc.createVariable("eigenvalues", "f8", ("nsppol", "nkpt_ibz", "nband"))[:] = (
            data.eigenvalues / hartree
        )
        nc.createVariable("occupations", "f8", ("nsppol", "nkpt_ibz", "nband"))[:] = (
            data.occupations
        )
        # ABINIT PAO_HS stores <psi|phi>, unlike ProjectorGreen's <phi|psi>.
        coeff = data.coefficients.conj()
        nc.createVariable(
            "coefficients_ibz_real", "f8", ("nsppol", "nkpt_ibz", "nband", "nproj")
        )[:] = coeff.real
        nc.createVariable(
            "coefficients_ibz_imag", "f8", ("nsppol", "nkpt_ibz", "nband", "nproj")
        )[:] = coeff.imag
        overlap = data.overlap_k
        nc.createVariable("overlap_ibz_real", "f8", ("nkpt_ibz", "nproj", "nproj"))[
            :
        ] = overlap.real
        nc.createVariable("overlap_ibz_imag", "f8", ("nkpt_ibz", "nproj", "nproj"))[
            :
        ] = overlap.imag
        if full_bz:
            nc.createVariable("kpoints_bz", "f8", ("nkpt_bz", "three"))[:] = (
                data.kpoints
            )
            nc.createVariable("bz_to_ibz", "i4", ("nkpt_bz",))[:] = np.arange(
                1, data.nkpt + 1
            )
            nc.createVariable("bz_to_sym", "i4", ("nkpt_bz",))[:] = np.ones(
                data.nkpt, dtype=int
            )
            nc.createVariable(
                "coefficients_bz_real", "f8", ("nsppol", "nkpt_bz", "nband", "nproj")
            )[:] = coeff.real
            nc.createVariable(
                "coefficients_bz_imag", "f8", ("nsppol", "nkpt_bz", "nband", "nproj")
            )[:] = coeff.imag
            nc.createVariable("overlap_bz_real", "f8", ("nkpt_bz", "nproj", "nproj"))[
                :
            ] = overlap.real
            nc.createVariable("overlap_bz_imag", "f8", ("nkpt_bz", "nproj", "nproj"))[
                :
            ] = overlap.imag
        delta_total = data.get_operator_component("delta_total", site=0) / hartree
        delta_xc = data.get_operator_component("delta_xc", site=0) / hartree
        delta_u = delta_total - delta_xc
        for name, matrix in (
            ("delta_total", delta_total),
            ("delta_xc_smooth", delta_xc),
            ("delta_U", delta_u),
        ):
            nc.createVariable(f"{name}_real", "f8", ("nproj", "nproj"))[:] = matrix.real
            nc.createVariable(f"{name}_imag", "f8", ("nproj", "nproj"))[:] = matrix.imag


def test_projector_data_validates_exchange_ready_and_hij_difference():
    data = make_bcc_fe_projector_data()

    assert data.validate(exchange_ready=True)
    np.testing.assert_allclose(
        data.get_hij_spin_difference(site=0),
        np.array([[0.6, 0.1j], [-0.1j, 0.5]]),
    )


def test_projector_data_requires_explicit_hij_definition():
    data = make_bcc_fe_projector_data()
    data.hij_definition = ""

    with pytest.raises(ValueError, match="hij requires an explicit definition"):
        data.validate(exchange_ready=True)


def test_projector_green_reconstructs_gk_from_spectral_data():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j

    gk = green.get_Gk(ik=0, energy=energy, ispin=0)

    expected = np.diag(
        [
            1.0 / (energy - data.eigenvalues[0, 0, 0]),
            1.0 / (energy - data.eigenvalues[0, 0, 1]),
        ]
    )
    np.testing.assert_allclose(gk, expected)


def test_projector_green_reconstructs_gk_with_spin_resolved_fermi():
    data = make_bcc_fe_projector_data()
    data.efermi_spin = np.array([0.0, 0.4])
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j

    gk_up = green.get_Gk(ik=0, energy=energy, ispin=0)
    gk_dn = green.get_Gk(ik=0, energy=energy, ispin=1)

    expected_up = np.diag(
        [
            1.0 / (energy - data.eigenvalues[0, 0, 0]),
            1.0 / (energy - data.eigenvalues[0, 0, 1]),
        ]
    )
    expected_dn = np.diag(
        [
            1.0 / (energy + data.efermi_spin[1] - data.eigenvalues[1, 0, 0]),
            1.0 / (energy + data.efermi_spin[1] - data.eigenvalues[1, 0, 1]),
        ]
    )

    np.testing.assert_allclose(gk_up, expected_up)
    np.testing.assert_allclose(gk_dn, expected_dn)


def test_projector_green_uses_band_mask_in_spectral_sum():
    data = make_bcc_fe_projector_data()
    data.band_mask = np.array(
        [
            [[True, False], [False, True]],
            [[True, True], [True, False]],
        ]
    )
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j

    gk0 = green.get_Gk(ik=0, energy=energy, ispin=0)
    gk1 = green.get_Gk(ik=1, energy=energy, ispin=0)

    np.testing.assert_allclose(
        gk0,
        np.diag([1.0 / (energy - data.eigenvalues[0, 0, 0]), 0.0]),
    )
    np.testing.assert_allclose(
        gk1,
        np.diag([0.0, 1.0 / (energy - data.eigenvalues[0, 1, 1])]),
    )


def test_projector_green_reconstructs_gk_with_k_dependent_overlap():
    data = make_overlap_k_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.4j
    ik = 1

    gk = green.get_Gk(ik=ik, energy=energy, ispin=0)

    coeff = data.coefficients[0, ik]
    inv_denom = 1.0 / (energy + data.efermi - data.eigenvalues[0, ik])
    covariant = np.einsum("np,nq,n->pq", coeff, coeff.conj(), inv_denom)
    Sinv = np.linalg.inv(data.overlap_k[ik])
    expected = Sinv @ covariant @ Sinv
    np.testing.assert_allclose(green.get_Sk(ik), data.overlap_k[ik])
    np.testing.assert_allclose(gk, expected)


def test_projector_green_marks_paw_coefficients_as_already_dual():
    green = ProjectorGreen(make_bcc_fe_projector_data())

    assert green.coefficients_are_dual
    assert not green.needs_overlap_transform
    assert not hasattr(green, "is_orthogonal")


def test_projector_green_svd_truncates_small_overlap_mode():
    data = make_overlap_k_projector_data()
    data.overlap_k[0] = np.diag([1.0, 1.0e-12])
    green = ProjectorGreen(data, overlap_mode="svd", overlap_rcond=1.0e-8)
    energy = 1.0 + 0.4j
    coeff = data.coefficients[0, 0]
    inv_denom = 1.0 / (energy + data.efermi - data.eigenvalues[0, 0])
    covariant = np.einsum("np,nq,n->pq", coeff, coeff.conj(), inv_denom)
    truncated_inverse = np.diag([1.0, 0.0])

    np.testing.assert_allclose(
        green.get_Gk(0, energy=energy),
        truncated_inverse @ covariant @ truncated_inverse,
    )


def test_projector_green_lowdin_matches_svd_at_same_cutoff():
    data = make_overlap_k_projector_data()
    unitary, _ = np.linalg.qr(np.array([[1.0, 1.0j], [0.5j, 1.0]], dtype=complex))
    data.overlap_k[0] = unitary @ np.diag([2.0, 1.0e-8]) @ unitary.conj().T
    energy = 1.0 + 0.4j

    svd = ProjectorGreen(data, overlap_mode="svd", overlap_rcond=1.0e-6)
    lowdin = ProjectorGreen(data, overlap_mode="lowdin", overlap_rcond=1.0e-6)

    np.testing.assert_allclose(
        lowdin.get_Gk(0, energy=energy),
        svd.get_Gk(0, energy=energy),
        atol=1.0e-12,
    )


def test_projector_green_tikhonov_smoothly_damps_small_overlap_modes():
    data = make_overlap_k_projector_data()
    data.overlap_k[0] = np.diag([1.0, 1.0e-4])
    green = ProjectorGreen(data, overlap_mode="tikhonov", overlap_rcond=1.0e-2)
    regularized = green._inverse_overlap(data.overlap_k[0], 0)

    singular = np.array([1.0, 1.0e-4])
    expected = singular / (singular**2 + 1.0e-4)
    np.testing.assert_allclose(regularized, np.diag(expected))
    assert 0.0 < regularized[1, 1] < 1.0 / singular[1]


def test_projector_green_gk_matches_operator_matrix_element():
    """Phase-sensitive regression for the projector coefficient convention.

    get_Gk must return the true dual-dual Green matrix <p~_p|G(z)|p~_q>, not its
    transpose. With complex non-diagonal projector overlaps the conj-on-1st-index
    form (the pre-fix code) yields G^T and fails; conj-on-2nd (docs +
    sympy identity [F], PAW_green_convention_check.py) passes.
    """
    rng = np.random.default_rng(20260804)
    nproj = nband = 3
    a = rng.standard_normal((nproj, nband)) + 1j * rng.standard_normal((nproj, nband))
    psi, _ = np.linalg.qr(a)  # psi[:, n] = |psi~_n>, orthonormal columns
    pmat = rng.standard_normal((nproj, nproj)) + 1j * rng.standard_normal(
        (nproj, nproj)
    )
    evals = np.array([-0.5, 0.3, 1.7])
    z = 0.9 + 0.4j

    # coefficients C[n, p] = <p~_p | psi~_n> = pmat[:, p]^dagger . psi[:, n]
    coeff = (pmat.conj().T @ psi).T
    # operator resolvent and the true G^{pq} = p_p^dagger G p_q
    gop = psi @ np.diag(1.0 / (z - evals)) @ psi.conj().T
    gtrue = pmat.conj().T @ gop @ pmat

    data = ProjectorGreenData(
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        weights=np.array([1.0]),
        eigenvalues=evals[None, None, :],
        coefficients=coeff[None, None, :, :],
        efermi=0.0,
        projector_site=np.zeros(nproj, dtype=int),
        projector_atom=np.zeros(nproj, dtype=int),
        cell=np.eye(3),
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([26]),
        projector_l=np.zeros(nproj, dtype=int),
        projector_m=np.zeros(nproj, dtype=int),
        projector_radial=np.zeros(nproj, dtype=int),
    )
    green = ProjectorGreen(data)
    gk = green.get_Gk(ik=0, energy=z, ispin=0)
    np.testing.assert_allclose(gk, gtrue, atol=1e-10)
    # the pre-fix transpose form must NOT match
    assert not np.allclose(gk, gtrue.T, atol=1e-6)


def test_operator_component_api_prefers_delta_xc_and_is_selectable():
    """Lock the GPAW operator-component API contract.

    get_local_operator prefers an exported 'delta_xc' component; operator_component_names
    lists stored components; compute_projector_exchange_jdict(operator_component=...) uses
    that component.
    """
    data = make_bcc_fe_projector_data()
    # inject a distinct delta_xc operator component (2x the hij spin difference)
    hij_split = data.get_hij_spin_difference(site=0)
    delta_xc_block = 2.0 * hij_split[None, :, :]
    data.operator_components = {"delta_xc": delta_xc_block}
    data.operator_component_metadata = {"delta_xc": {}}

    assert data.operator_component_names == ["delta_xc"]
    green = ProjectorGreen(data)
    np.testing.assert_allclose(green.get_local_operator(0), hij_split * 2.0)

    # explicit selection runs and produces an exchange dict
    jdict = compute_projector_exchange_jdict(
        data, Rpts=[(0, 0, 0)], nz=6, smearing_eV=0.05, operator_component="delta_xc"
    )
    assert jdict  # non-empty


def test_projector_green_transforms_full_bz_gk_to_gr():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j
    rpts = np.array([[0, 0, 0], [1, 0, 0]])

    gks = green.get_Gk_all(energy, ispin=0)
    gr = green.get_GR(rpts, energy, Gk_all=gks, ispin=0)

    phase = np.exp(-2.0j * np.pi * np.einsum("ri,ki->rk", rpts, data.kpoints))
    expected = np.einsum("kpq,rk,k->rpq", gks, phase, data.weights)
    np.testing.assert_allclose(gr, expected)


def test_projector_green_transforms_overlap_k_corrected_gk_to_gr():
    data = make_overlap_k_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.4j
    rpts = np.array([[0, 0, 0], [1, 0, 0]])

    gr = green.get_GR(rpts, energy=energy, ispin=0)

    corrected = []
    for ik in range(data.nkpt):
        coeff = data.coefficients[0, ik]
        inv_denom = 1.0 / (energy + data.efermi - data.eigenvalues[0, ik])
        covariant = np.einsum("np,nq,n->pq", coeff, coeff.conj(), inv_denom)
        Sinv = np.linalg.inv(data.overlap_k[ik])
        corrected.append(Sinv @ covariant @ Sinv)
    corrected = np.asarray(corrected)
    phase = np.exp(-2.0j * np.pi * np.einsum("ri,ki->rk", rpts, data.kpoints))
    expected = np.einsum("kpq,rk,k->rpq", corrected, phase, data.weights)
    np.testing.assert_allclose(gr, expected)


def test_projector_green_rejects_invalid_gr_shapes():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    gks = green.get_Gk_all(1.0 + 0.5j, ispin=0)

    with pytest.raises(ValueError, match="Rpts must have shape"):
        green.compute_GR(np.array([0, 0, 0]), data.kpoints, gks)

    with pytest.raises(ValueError, match="Gks must have shape"):
        green.compute_GR(np.array([[0, 0, 0]]), data.kpoints, gks[:, :1, :1])


def test_projector_data_rejects_invalid_overlap_k_shape():
    data = make_overlap_k_projector_data()
    data.overlap_k = data.overlap_k[:, :1, :1]

    with pytest.raises(ValueError, match="overlap_k must have shape"):
        data.validate()


def test_projector_green_netcdf_roundtrip(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    data = make_bcc_fe_projector_data()
    filename = tmp_path / "projector_green.nc"

    data.save_netcdf(filename)
    with netcdf4.Dataset(filename) as nc:
        assert "greens_k" not in nc.groups
        assert "greens_R" not in nc.groups
        assert nc.groups["projectors"].variables["coefficients"].dimensions[-1] == (
            "complex"
        )
        projectors = nc.groups["projectors"]
        assert projectors.coefficient_source == "gpaw.P_ani"
        assert projectors.coefficient_projector == "dual_paw_projector"
        assert projectors.channel_interpretation == "paw_partial_wave_channel"
        assert (
            projectors.population_metric == "GPAW PAW N0_p packed density contraction"
        )
        assert (
            projectors.variables["overlap_metric"].definition
            == "GPAW PAW onsite dO_ii correction"
        )
        assert (
            nc.groups["operators"].variables["hij"].operator_basis
            == "native_paw_projector_hamiltonian"
        )

    loaded = ProjectorGreenData.load_netcdf(filename)

    assert loaded.metadata["storage_level"] == "spectral"
    assert loaded.metadata["source"] == "synthetic bcc Fe primitive cell"
    np.testing.assert_allclose(loaded.kpoints, data.kpoints)
    np.testing.assert_allclose(loaded.eigenvalues, data.eigenvalues)
    np.testing.assert_allclose(loaded.coefficients, data.coefficients)
    np.testing.assert_array_equal(loaded.projector_site, data.projector_site)
    np.testing.assert_allclose(loaded.overlap_metric, data.overlap_metric)
    np.testing.assert_allclose(loaded.hij, data.hij)
    assert loaded.hij_definition == data.hij_definition
    assert loaded.hij_units == "eV"
    assert loaded.hij_source == "GPAW dH_asp"
    assert loaded.hij_projection == "native PAW projector Hamiltonian matrix"
    assert loaded.coefficient_source == "gpaw.P_ani"
    assert loaded.coefficient_projector == "dual_paw_projector"
    assert loaded.channel_interpretation == "paw_partial_wave_channel"
    assert loaded.overlap_metric_definition == "GPAW PAW onsite dO_ii correction"
    assert loaded.population_metric == "GPAW PAW N0_p packed density contraction"
    assert loaded.operator_basis == "native_paw_projector_hamiltonian"


def test_projector_green_netcdf_roundtrip_with_spin_resolved_fermi(tmp_path):
    pytest.importorskip("netCDF4")
    data = make_bcc_fe_projector_data()
    data.efermi_spin = np.array([0.0, 0.4])
    filename = tmp_path / "projector_green_spin_fermi.nc"

    data.save_netcdf(filename)
    loaded = ProjectorGreenData.load_netcdf(filename)

    np.testing.assert_allclose(loaded.efermi_spin, data.efermi_spin)
    assert loaded.efermi == np.mean(data.efermi_spin)

    green = ProjectorGreen(loaded)
    energy = 0.9 + 0.3j
    gk_up = green.get_Gk(ik=0, energy=energy, ispin=0)
    gk_dn = green.get_Gk(ik=0, energy=energy, ispin=1)

    expected_up = np.diag(
        [
            1.0 / (energy + data.efermi_spin[0] - data.eigenvalues[0, 0, 0]),
            1.0 / (energy + data.efermi_spin[0] - data.eigenvalues[0, 0, 1]),
        ]
    )
    expected_dn = np.diag(
        [
            1.0 / (energy + data.efermi_spin[1] - data.eigenvalues[1, 0, 0]),
            1.0 / (energy + data.efermi_spin[1] - data.eigenvalues[1, 0, 1]),
        ]
    )

    np.testing.assert_allclose(gk_up, expected_up)
    np.testing.assert_allclose(gk_dn, expected_dn)


def test_projector_green_netcdf_roundtrips_overlap_k(tmp_path):
    pytest.importorskip("netCDF4")
    data = make_overlap_k_projector_data()
    filename = tmp_path / "projector_green_overlap_k.nc"

    data.save_netcdf(filename)
    loaded = ProjectorGreenData.load_netcdf(filename)

    np.testing.assert_allclose(loaded.overlap_metric, data.overlap_metric)
    np.testing.assert_allclose(loaded.overlap_k, data.overlap_k)
    assert loaded.overlap_metric_definition == data.overlap_metric_definition
    np.testing.assert_allclose(ProjectorGreen(loaded).get_Sk(1), data.overlap_k[1])


def test_projector_green_netcdf_roundtrips_operator_components(tmp_path):
    pytest.importorskip("netCDF4")
    data = make_nc_pao_exchange_data()
    filename = tmp_path / "projector_green_operator_components.nc"

    data.save_netcdf(filename)
    loaded = ProjectorGreenData.load_netcdf(filename)

    np.testing.assert_allclose(
        loaded.get_operator_component("delta_total", site=0),
        data.get_operator_component("delta_total", site=0),
    )
    assert loaded.operator_component_metadata["delta_total"]["source"] == "synthetic"


def test_projector_green_loads_minimal_nc_pao_netcdf(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    source = make_overlap_k_projector_data()
    filename = tmp_path / "minimal_nc_pao.nc"

    with netcdf4.Dataset(filename, "w") as nc:
        nc.createDimension("nspin", source.nspin)
        nc.createDimension("nkpt", source.nkpt)
        nc.createDimension("nband", source.nband)
        nc.createDimension("nproj", source.nproj)
        nc.createDimension("three", 3)
        nc.createDimension("complex", 2)
        nc.efermi = source.efermi
        nc.coefficient_source = "synthetic NC PAO fixture"
        nc.createVariable("kpoints", "f8", ("nkpt", "three"))[:] = source.kpoints
        nc.createVariable("weights", "f8", ("nkpt",))[:] = source.weights
        nc.createVariable("eigenvalues", "f8", ("nspin", "nkpt", "nband"))[:] = (
            source.eigenvalues
        )
        nc.createVariable(
            "coefficients", "f8", ("nspin", "nkpt", "nband", "nproj", "complex")
        )[:] = encode_complex(source.coefficients)
        overlap_k = nc.createVariable(
            "overlap_k", "f8", ("nkpt", "nproj", "nproj", "complex")
        )
        overlap_k[:] = encode_complex(source.overlap_k)
        overlap_k.definition = source.overlap_metric_definition
        nc.createVariable("projector_site", "i4", ("nproj",))[:] = source.projector_site
        nc.createVariable("projector_atom", "i4", ("nproj",))[:] = source.projector_atom

    loaded = ProjectorGreenData.load_nc_pao_netcdf(filename)

    assert loaded.coefficient_source == "synthetic NC PAO fixture"
    assert loaded.coefficient_projector == "nc_pao"
    assert loaded.channel_interpretation == "norm_conserving_pao"
    np.testing.assert_allclose(loaded.overlap_k, source.overlap_k)
    np.testing.assert_allclose(
        ProjectorGreen(loaded).get_Gk(1, energy=1.0 + 0.4j),
        ProjectorGreen(source).get_Gk(1, energy=1.0 + 0.4j),
    )


def test_abinit_nc_pao_savetb2j_loader_preserves_metric_and_operator(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    source = make_nc_pao_exchange_data()
    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    write_abinit_nc_pao_fixture(filename, source)

    loaded = load_abinit_nc_pao_savetb2j(filename)

    assert loaded.coefficient_projector == "nc_pao"
    assert loaded.channel_interpretation == "norm_conserving_pao"
    assert loaded.operator_basis == "abinit_nc_pao"
    np.testing.assert_allclose(loaded.overlap_k, source.overlap_k)
    np.testing.assert_allclose(
        loaded.get_operator_component("delta_total", site=0),
        source.get_operator_component("delta_total", site=0),
    )
    assert loaded.operator_component_metadata["delta_xc"]["source"] == "synthetic"


def test_abinit_nc_pao_hs_v2_loader_decodes_split_schema(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    source = make_nc_pao_exchange_data()
    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source)

    loaded = load_abinit_nc_pao_savetb2j(filename)

    assert loaded.operator_basis == "norm_conserving_pao"
    assert loaded.coefficient_projector == "nc_pao"
    assert loaded.metadata["kpoint_set"] == "ibz"
    np.testing.assert_allclose(loaded.eigenvalues, source.eigenvalues)
    np.testing.assert_allclose(loaded.coefficients, source.coefficients)
    np.testing.assert_allclose(loaded.overlap_k, source.overlap_k)
    np.testing.assert_allclose(
        loaded.get_operator_component("delta_total", site=0),
        source.get_operator_component("delta_total", site=0),
    )


def test_abinit_nc_pao_hs_v2_loader_converts_abinit_structure_units(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    source = make_nc_pao_exchange_data()
    filename = tmp_path / "abinit_nc_pao_hs_v2_abinit_units.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, abinit_structure_units=True)

    loaded = load_abinit_nc_pao_savetb2j(filename)

    np.testing.assert_allclose(loaded.cell, source.cell)
    np.testing.assert_allclose(loaded.positions, source.positions, atol=1e-12)


def test_abinit_nc_pao_hs_v2_loader_prefers_full_bz_coefficients(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    source = make_nc_pao_exchange_data()
    filename = tmp_path / "abinit_nc_pao_hs_v2_full_bz.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, full_bz=True)

    loaded = load_abinit_nc_pao_savetb2j(filename)

    assert loaded.metadata["kpoint_set"] == "full_bz"
    np.testing.assert_allclose(loaded.kpoints, source.kpoints)
    np.testing.assert_allclose(loaded.coefficients, source.coefficients)
    np.testing.assert_allclose(loaded.overlap_k, source.overlap_k)


def test_abinit_nc_pao_shell_populations_group_by_site_radial_and_l(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import (
        compute_nc_pao_shell_populations,
        load_abinit_nc_pao_savetb2j,
    )

    source = make_nc_pao_exchange_data()
    source.overlap_k = np.eye(2, dtype=complex)[None, :, :]
    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, full_bz=True)
    data = load_abinit_nc_pao_savetb2j(filename)

    shells = compute_nc_pao_shell_populations(data)

    assert [(s["site"], s["n_quantum"], s["l_quantum"]) for s in shells] == [
        (0, 1, 0),
        (0, 2, 0),
    ]
    np.testing.assert_allclose([s["charge"] for s in shells], [1.0, 1.0])
    np.testing.assert_allclose([s["moment_z"] for s in shells], [1.0, -1.0])


def test_abinit_nc_pao_shell_populations_require_shell_metadata():
    from TB2J.interfaces.abinit_savetb2j import compute_nc_pao_shell_populations

    data = make_nc_pao_exchange_data()

    with pytest.raises(ValueError, match="shell diagnostics require"):
        compute_nc_pao_shell_populations(data)


def test_abinit_nc_pao_shell_threshold_masks_local_operator(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import (
        compute_nc_pao_shell_populations,
        load_abinit_nc_pao_savetb2j,
        mask_local_operators_by_shell_selection,
        select_nc_pao_shells,
    )

    source = make_nc_pao_exchange_data()
    source.overlap_k = np.eye(2, dtype=complex)[None, :, :]
    source.occupations = np.array([[[1.0, 0.0]], [[0.0, 0.005]]])
    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, full_bz=True)
    data = load_abinit_nc_pao_savetb2j(filename)
    local_operators = {0: np.ones((2, 2), dtype=complex)}

    shells = select_nc_pao_shells(
        compute_nc_pao_shell_populations(data), charge_threshold=0.01
    )
    masked = mask_local_operators_by_shell_selection(data, local_operators, shells)

    assert [shell["selected"] for shell in shells] == [True, False]
    np.testing.assert_allclose(masked[0], [[1.0, 0.0], [0.0, 0.0]])


def test_abinit_nc_pao_shell_moment_threshold_masks_nonmagnetic_shell(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import (
        compute_nc_pao_shell_populations,
        load_abinit_nc_pao_savetb2j,
        mask_local_operators_by_shell_selection,
        select_nc_pao_shells,
    )

    source = make_nc_pao_exchange_data()
    source.overlap_k = np.eye(2, dtype=complex)[None, :, :]
    source.occupations = np.array([[[1.0, 1.0]], [[1.0, 0.0]]])
    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, full_bz=True)
    data = load_abinit_nc_pao_savetb2j(filename)
    local_operators = {0: np.ones((2, 2), dtype=complex)}

    shells = select_nc_pao_shells(
        compute_nc_pao_shell_populations(data),
        charge_threshold=0.01,
        moment_threshold=0.01,
    )
    masked = mask_local_operators_by_shell_selection(data, local_operators, shells)

    np.testing.assert_allclose([shell["moment_norm"] for shell in shells], [0.0, 1.0])
    assert [shell["selected"] for shell in shells] == [False, True]
    np.testing.assert_allclose(masked[0], [[0.0, 0.0], [0.0, 1.0]])


def test_abinit_nc_pao_diagnostics_report_summarizes_selection(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import (
        compute_nc_pao_shell_populations,
        gen_exchange_abinit_nc_pao,
        load_abinit_nc_pao_savetb2j,
        select_nc_pao_shells,
    )

    source = make_nc_pao_exchange_data()
    source.overlap_k = np.eye(2, dtype=complex)[None, :, :]
    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    report_path = tmp_path / "report.md"
    write_abinit_nc_pao_hs_v2_fixture(filename, source, full_bz=True)

    data = load_abinit_nc_pao_savetb2j(filename)
    shells = select_nc_pao_shells(
        compute_nc_pao_shell_populations(data), charge_threshold=0.01
    )
    assert all(shell["selected"] for shell in shells)

    gen_exchange_abinit_nc_pao(
        filename,
        output_path=tmp_path / "TB2J_results",
        Rmax=0,
        nz=2,
        report_path=report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "excluded_shell_count: 0" in report
    assert "all shells are selected by the current threshold" in report

    disabled_report_path = tmp_path / "disabled_report.md"
    gen_exchange_abinit_nc_pao(
        filename,
        output_path=tmp_path / "TB2J_results_disabled_filter",
        Rmax=0,
        nz=2,
        shell_charge_threshold=None,
        shell_moment_threshold=None,
        report_path=disabled_report_path,
    )

    disabled_report = disabled_report_path.read_text(encoding="utf-8")
    assert "shell_filtering: disabled" in disabled_report
    assert "shell filtering is disabled" in disabled_report


def test_abinit_nc_pao_report_handles_disabled_filter_without_shell_metadata(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    report_path = tmp_path / "report.md"
    write_abinit_nc_pao_fixture(filename, make_nc_pao_exchange_data())

    gen_exchange_abinit_nc_pao(
        filename,
        output_path=tmp_path / "TB2J_results",
        Rmax=0,
        nz=2,
        shell_charge_threshold=None,
        shell_moment_threshold=None,
        report_path=report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "shell diagnostics unavailable" in report
    assert "shell_filtering: disabled" in report


def test_abinit_nc_pao_band_mask_from_absolute_and_relative_cutoff():
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    data = make_nc_pao_exchange_data()
    data.occupations = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])

    absolute, absolute_meta = build_nc_pao_band_mask(data, emax_eV=2.05)
    relative, relative_meta = build_nc_pao_band_mask(
        data, emax_relative_to_fermi_eV=1.6
    )

    np.testing.assert_array_equal(
        absolute,
        np.array([[[True, True]], [[True, False]]]),
    )
    np.testing.assert_array_equal(
        relative,
        np.array([[[True, True]], [[True, False]]]),
    )
    assert absolute_meta["included_band_count_min"] == 1
    assert absolute_meta["included_band_count_max"] == 2
    assert relative_meta["emax_relative_to_fermi_eV"] == 1.6


def test_abinit_nc_pao_band_mask_rejects_excluded_occupied_band():
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    data = make_nc_pao_exchange_data()

    with pytest.raises(ValueError, match="excludes occupied bands"):
        build_nc_pao_band_mask(data, emax_eV=2.05)


def test_abinit_nc_pao_relative_band_mask_uses_spin_fermi():
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    data = make_nc_pao_exchange_data()
    data.occupations = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    data.efermi_spin = np.array([0.5, 1.0])

    mask, metadata = build_nc_pao_band_mask(data, emax_relative_to_fermi_eV=1.3)

    np.testing.assert_array_equal(
        mask,
        np.array([[[True, False]], [[True, True]]]),
    )
    assert metadata["cutoff_eV"] == [[[1.8]], [[2.3]]]


def test_abinit_nc_pao_band_mask_from_fixed_empty_count():
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    data = make_nc_pao_exchange_data()
    data.eigenvalues = np.array(
        [
            [[-1.0, 0.0, 2.0, 1.0]],
            [[-1.0, 0.2, 3.0, 1.5]],
        ]
    )
    data.occupations = np.array([[[1.0, 1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]])
    data.coefficients = np.zeros((2, 1, 4, 2), dtype=complex)

    mask, metadata = build_nc_pao_band_mask(data, n_empty=1)

    np.testing.assert_array_equal(
        mask,
        np.array([[[True, True, False, True]], [[True, True, False, False]]]),
    )
    assert metadata["n_empty"] == 1
    assert metadata["included_unoccupied_min"] == 1
    assert metadata["included_unoccupied_max"] == 1


def test_abinit_nc_pao_band_mask_rejects_conflicting_window_options():
    from TB2J.interfaces.abinit_savetb2j import build_nc_pao_band_mask

    data = make_nc_pao_exchange_data()

    with pytest.raises(ValueError, match="Specify only one"):
        build_nc_pao_band_mask(data, emax_eV=1.0, n_empty=1)


def test_abinit_nc_pao_hs_v2_full_bz_exchange_smoke(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    filename = tmp_path / "abinit_nc_pao_hs_v2_full_bz.nc"
    write_abinit_nc_pao_hs_v2_fixture(
        filename, make_nc_pao_exchange_data(), full_bz=True
    )

    exchange_out, jdict = gen_exchange_abinit_nc_pao(
        filename, output_path=tmp_path / "TB2J_results", Rmax=0, nz=2
    )

    assert exchange_out.is_file()
    assert jdict


def test_abinit_nc_pao_hs_v2_loader_rejects_unready_overlap(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    filename = tmp_path / "abinit_nc_pao_hs_v2_unready.nc"
    write_abinit_nc_pao_hs_v2_fixture(
        filename, make_nc_pao_exchange_data(), overlap_exchange_ready=0
    )

    with pytest.raises(ValueError, match="overlap metadata is not exchange-ready"):
        load_abinit_nc_pao_savetb2j(filename)


def test_abinit_nc_pao_savetb2j_loader_rejects_missing_overlap_metadata(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    filename = tmp_path / "abinit_nc_pao_bad_overlap.nc"
    write_abinit_nc_pao_fixture(
        filename, make_nc_pao_exchange_data(), overlap_definition=False
    )

    with pytest.raises(ValueError, match="overlap_k requires validation metadata"):
        load_abinit_nc_pao_savetb2j(filename)


def test_abinit_nc_pao_savetb2j_loader_rejects_missing_delta_total(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    data = make_nc_pao_exchange_data()
    data.operator_components.pop("delta_total")
    data.operator_component_metadata.pop("delta_total")
    filename = tmp_path / "abinit_nc_pao_missing_delta.nc"
    write_abinit_nc_pao_fixture(filename, data)

    with pytest.raises(ValueError, match="requires delta_total"):
        load_abinit_nc_pao_savetb2j(filename)


def test_projector_green_rejects_ill_conditioned_overlap_k():
    data = make_overlap_k_projector_data()
    data.overlap_k[0] = np.array([[1.0, 0.0], [0.0, 1.0e-16]])
    data.metadata["overlap_condition_threshold"] = 1.0e8
    green = ProjectorGreen(data)

    with pytest.raises(ValueError, match="ill-conditioned"):
        green.get_Gk(0, energy=1.0 + 0.4j)


def test_abinit_nc_pao_exchange_api_writes_exchange_out(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    output_path = tmp_path / "TB2J_results"
    write_abinit_nc_pao_fixture(filename, make_nc_pao_exchange_data())

    exchange_out, jdict = gen_exchange_abinit_nc_pao(
        filename,
        output_path=output_path,
        Rmax=0,
        nz=2,
        smearing_eV=0.1,
        overlap_mode="svd",
        overlap_rcond=1.0e-8,
    )

    assert exchange_out.is_file()
    assert jdict


def test_abinit_nc_pao_exchange_projector_population_uses_dual_metric(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    write_abinit_nc_pao_fixture(filename, make_nc_pao_exchange_data())

    exchange_out, _ = gen_exchange_abinit_nc_pao(
        filename,
        output_path=tmp_path / "TB2J_results_projector_population",
        Rmax=0,
        nz=2,
        population_mode="projector",
    )

    text = exchange_out.read_text(encoding="utf-8")
    assert "Fe1" in text
    assert "0.0000    0.0000" not in text


def test_abinit_nc_pao_exchange_rejects_ibz_schema_v2(tmp_path):
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    filename = tmp_path / "abinit_nc_pao_hs_v2.nc"
    write_abinit_nc_pao_hs_v2_fixture(filename, make_nc_pao_exchange_data())

    with pytest.raises(ValueError, match="requires full-BZ spectral coefficients"):
        gen_exchange_abinit_nc_pao(filename)


def test_abinit_nc_pao_cli_writes_exchange_out(tmp_path, monkeypatch):
    from TB2J.scripts.abinit_nc_pao2J import run_abinit_nc_pao2J

    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    output_path = tmp_path / "TB2J_cli_results"
    write_abinit_nc_pao_fixture(filename, make_nc_pao_exchange_data())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "abinit_nc_pao2J.py",
            "--input",
            str(filename),
            "--output_path",
            str(output_path),
            "--Rmax",
            "0",
            "--nz",
            "2",
            "--no_shell_filter",
            "--emax",
            "2.5",
        ],
    )

    run_abinit_nc_pao2J()

    assert (output_path / "exchange.out").is_file()


def test_project_potential_to_hij_for_non_paw_projectors():
    projectors = np.array(
        [
            [1.0, 0.0, 1.0j],
            [0.0, 2.0, 1.0],
            [1.0, -1.0j, 0.0],
        ],
        dtype=complex,
    )
    potential = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 2.5],
        ]
    )
    weights = np.array([0.2, 0.3, 0.5])

    hij_global = project_potential_to_hij(projectors, potential, weights=weights)

    expected = np.zeros((2, 3, 3), dtype=complex)
    for ispin in range(2):
        for i in range(3):
            for j in range(3):
                expected[ispin, i, j] = np.sum(
                    projectors[i].conj() * potential[ispin] * projectors[j] * weights
                )
    np.testing.assert_allclose(hij_global, expected)


def test_pack_site_hij_from_projected_non_paw_hij():
    projectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=complex)
    potential = np.array([[1.0, 2.0], [3.0, 4.0]])
    projector_site = np.array([0, 1, 1])
    site_nproj, site_projector_indices = build_site_projector_indices(projector_site)
    hij_global = project_potential_to_hij(projectors, potential)

    hij = pack_site_hij(hij_global, site_projector_indices, site_nproj)

    assert hij.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(site_nproj, np.array([1, 2]))
    np.testing.assert_array_equal(site_projector_indices, np.array([[0, -1], [1, 2]]))
    np.testing.assert_allclose(hij[:, 0, :1, :1], hij_global[:, :1, :1])
    np.testing.assert_allclose(hij[:, 1], hij_global[:, 1:, 1:])


def test_green_backend_protocol_accepts_projector_green():
    green = ProjectorGreen(make_bcc_fe_projector_data())

    assert validate_green_backend(green)


def test_green_backend_protocol_rejects_missing_members():
    with pytest.raises(TypeError, match="missing required protocol members"):
        validate_green_backend(object())


def test_projector_exchange_trace_matches_direct_reference():
    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    energy = 1.0 + 0.5j
    rpts = np.array([[0, 0, 0]])

    result = projector_exchange_trace(green, rpts, energy)

    Delta = data.get_hij_spin_difference(site=0)
    Gup = green.get_GR(rpts, energy=energy, ispin=0)[0]
    Gdn = green.get_GR(rpts, energy=energy, ispin=1)[0]
    orbital = np.einsum("ij,ji->ij", Delta @ Gup, Delta @ Gdn) / (4.0 * np.pi)

    assert result["method"] == "projector_exchange_trace"
    assert result["local_operator"] == "hij_spin_difference"
    np.testing.assert_allclose(result["orbital_trace"][((0, 0, 0), 0, 0)], orbital)
    np.testing.assert_allclose(result["trace"][((0, 0, 0), 0, 0)], np.sum(orbital))


def test_projector_charge_moments_from_green_matches_manual_contour_trace():
    class FakeContour:
        path = np.array([1.0 + 0.2j, 1.5 + 0.3j])
        weights = np.array([0.7 + 0.1j, -0.2 + 0.4j])

        def integrate_values(self, values):
            return np.einsum("e,e...->...", self.weights, values)

    data = make_bcc_fe_projector_data()
    green = ProjectorGreen(data)
    contour = FakeContour()

    density = projector_charge_moments_from_green(green, contour)

    manual = np.zeros(2)
    for ispin in range(2):
        diags = []
        for energy in contour.path:
            GR0 = green.get_GR([(0, 0, 0)], energy=energy, ispin=ispin)[0]
            diags.append(np.diag(GR0))
        manual[ispin] = np.sum(
            -np.imag(contour.integrate_values(np.asarray(diags))) / np.pi
        )

    np.testing.assert_allclose(density["density_by_spin"][:, 0], manual)
    np.testing.assert_allclose(density["charges"], [np.sum(manual)])
    np.testing.assert_allclose(density["spinat"][:, 2], [manual[0] - manual[1]])


def test_projector_charge_moments_from_green_contracts_population_metric():
    class FakeContour:
        path = np.array([1.0 + 0.2j, 1.5 + 0.3j])
        weights = np.array([0.7 + 0.1j, -0.2 + 0.4j])

        def integrate_values(self, values):
            return np.einsum("e,e...->...", self.weights, values)

    data = make_bcc_fe_projector_data()
    data.coefficients[0, 0, 0] = [1.0, 0.5j]
    data.population_metric_matrix = np.array([[2.0, 0.3j], [-0.3j, 1.5]], dtype=complex)
    green = ProjectorGreen(data)
    contour = FakeContour()

    density = projector_charge_moments_from_green(green, contour)

    manual = np.zeros(2)
    for ispin in range(2):
        values = []
        for energy in contour.path:
            GR0 = green.get_GR([(0, 0, 0)], energy=energy, ispin=ispin)[0]
            values.append(np.trace(GR0 @ data.population_metric_matrix))
        manual[ispin] = -np.imag(contour.integrate_values(np.asarray(values))) / np.pi

    assert density["method"] == "projector_green_contour_population_metric"
    np.testing.assert_allclose(density["density_by_spin"][:, 0], manual)
    np.testing.assert_allclose(density["charges"], [np.sum(manual)])
    np.testing.assert_allclose(density["spinat"][:, 2], [manual[0] - manual[1]])


def test_abinit_nc_pao_green_population_matches_occupations(tmp_path, monkeypatch):
    from TB2J.interfaces import abinit_savetb2j

    source = make_nc_pao_exchange_data()
    source.eigenvalues = np.array([[[-80.0, 1.0]], [[-80.0, 1.0]]])
    source.efermi = 0.0
    source.occupations = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    source.coefficients[:, 0, 0] = [1.0, 0.25j]
    source.overlap_k = np.eye(2, dtype=complex)[None, :, :]
    filename = tmp_path / "abinit_nc_pao_savetb2j.nc"
    write_abinit_nc_pao_fixture(filename, source)
    captured = {}

    def capture_output(data, **kwargs):
        captured.update(kwargs)
        return tmp_path / "exchange.out", {}

    monkeypatch.setattr(abinit_savetb2j, "write_projector_exchange_out", capture_output)
    abinit_savetb2j.gen_exchange_abinit_nc_pao(
        filename,
        output_path=tmp_path / "TB2J_results",
        Rmax=0,
        nz=30,
        smearing_eV=0.05,
        population_mode="green",
        shell_charge_threshold=None,
        shell_moment_threshold=None,
    )

    from TB2J.interfaces.abinit_savetb2j import compute_nc_pao_projected_charges_moments

    expected_charges, expected_spinat, _ = compute_nc_pao_projected_charges_moments(
        source
    )
    np.testing.assert_allclose(captured["charges"], expected_charges)
    np.testing.assert_allclose(captured["spinat"], expected_spinat)


def test_projector_exchange_trace_rejects_unsupported_hij_definition():
    data = make_bcc_fe_projector_data()
    data.hij_definition = "projected_density_matrix"
    green = ProjectorGreen(data)

    with pytest.raises(ValueError, match="unsupported hij definition"):
        projector_exchange_trace(green, np.array([[0, 0, 0]]), 1.0 + 0.5j)


def test_projector_exchange_trace_accepts_explicit_local_operator():
    data = make_bcc_fe_projector_data()
    data.hij_definition = "projected_density_matrix"
    green = ProjectorGreen(data)
    Delta = data.get_hij_spin_difference(site=0)

    result = projector_exchange_trace(
        green,
        np.array([[0, 0, 0]]),
        1.0 + 0.5j,
        local_operators={0: Delta},
    )

    assert result["local_operator"] == "explicit"
    assert ((0, 0, 0), 0, 0) in result["trace"]


def test_synthetic_nonpaw_projector_green_example_builds_valid_data():
    example = load_projector_green_example()
    data = example.build_synthetic_projector_green_data()
    green = ProjectorGreen(data)

    assert data.validate(exchange_ready=True)
    assert data.hij_definition == "projected_spin_dependent_potential"
    assert data.hij_source == "synthetic non-PAW projected potential"
    assert data.coefficient_projector == "custom_discrete_grid_projector"
    assert data.operator_basis == "projected_spin_dependent_potential"
    GR = green.get_GR(np.array([[0, 0, 0]]), energy=0.1 + 0.02j, ispin=0)
    assert GR.shape == (1, data.nproj, data.nproj)


def test_gpaw_bcc_fe_projector_green_workflow(tmp_path):
    pytest.importorskip("gpaw")
    pytest.importorskip("netCDF4")
    example = load_gpaw_bcc_fe_example()
    filename = tmp_path / "gpaw_bcc_fe_projector_green.nc"

    data, GR, trace = example.run_gpaw_bcc_fe_projector_green_workflow(filename)
    exchange_out, exchange_Jdict = example.write_projector_exchange_out(
        data, path=tmp_path / "TB2J_results"
    )

    assert filename.exists()
    assert exchange_out.exists()
    assert data.validate(exchange_ready=True)
    assert data.metadata["source_code"] == "gpaw"
    assert data.hij_definition == "paw_dh_asp_projector_hamiltonian"
    assert data.hij_source == "GPAW dH_asp"
    assert data.coefficient_source == "gpaw"
    assert data.coefficient_projector == "dual_paw_projector"
    assert data.channel_interpretation == "paw_partial_wave_channel"
    assert data.operator_basis == "native_paw_projector_hamiltonian"
    assert data.nspin == 2
    assert data.nkpt == np.prod(data.metadata["kmesh"])
    assert data.nproj == data.site_nproj[0]
    assert data.metadata["magnetic_moment_total"] > 2.0
    assert data.metadata["magnetic_moments"][0] > 2.0
    assert np.linalg.norm(data.get_hij_spin_difference(site=0)) > 1.0
    assert GR.shape == (1, data.nproj, data.nproj)
    assert trace["local_operator"] == "hij_spin_difference"
    assert ((0, 0, 0), 0, 0) in trace["trace"]
    assert ((1, 0, 0), 0, 0) in exchange_Jdict
    assert exchange_Jdict[((1, 0, 0), 0, 0)] > 0.0
    assert "Exchange:" in exchange_out.read_text()


# ---------------------------------------------------------------------------
# Story 10: Bundled NiO NC PAO fixture (schema-v2, overlap_exchange_ready=1)
# Generated from NiO NC DFT+U, single k-point, istwfk=1, full-BZ.
# ---------------------------------------------------------------------------

_NIO_NC_PAO_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "inputs"
    / "abinit_savetb2j"
    / "nio_nc_pao_hs.nc"
)


def test_bundled_nio_nc_pao_loader():
    """Story 10 AC2: Load the bundled NiO NC PAO fixture and validate schema."""
    pytest.importorskip("netCDF4")
    if not _NIO_NC_PAO_FIXTURE.is_file():
        pytest.skip("Bundled NiO NC PAO fixture not found")
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    data = load_abinit_nc_pao_savetb2j(_NIO_NC_PAO_FIXTURE)
    assert data.coefficient_projector == "nc_pao"
    assert data.channel_interpretation == "norm_conserving_pao"
    assert data.kpoints.shape[1] == 3
    assert data.eigenvalues.ndim == 3
    assert data.coefficients.ndim == 4
    assert data.nproj > 0
    assert data.overlap_k is not None
    nkpt, nproj, _ = data.overlap_k.shape
    assert nkpt == data.nkpt
    assert nproj == data.nproj


def test_bundled_nio_nc_pao_exchange_smoke(tmp_path):
    """Story 10 AC3: Exchange smoke test on the bundled NiO NC PAO fixture.

    Limitations documented:
    - Single k-point (Gamma-shifted): exchange values are not physically meaningful.
    - paral_kgb=1 is fail-closed; this fixture was generated with paral_kgb=0.
    """
    pytest.importorskip("netCDF4")
    if not _NIO_NC_PAO_FIXTURE.is_file():
        pytest.skip("Bundled NiO NC PAO fixture not found")
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    exchange_out, jdict = gen_exchange_abinit_nc_pao(
        _NIO_NC_PAO_FIXTURE,
        output_path=tmp_path / "TB2J_results",
        Rmax=0,
        nz=2,
        operator_component="spectral_spin_split",
    )
    assert exchange_out.is_file()
    assert jdict
    text = exchange_out.read_text(encoding="utf-8")
    assert "ABINIT" in text


# ---------------------------------------------------------------------------
# Story 10: Real ABINIT NC PAO validation fixture (integration tests)
# These tests are skipped when no real ABINIT NC PAO fixture is available.
# Set the ABINIT_NC_PAO_FIXTURE environment variable to a path to enable them.
# ---------------------------------------------------------------------------

_NC_PAO_FIXTURE = None
_NC_PAO_FIXTURE_REASON = "ABINIT_NC_PAO_FIXTURE not set; skipping real-fixture test"
try:
    import os as _os

    _fixture_path = _os.environ.get("ABINIT_NC_PAO_FIXTURE", "")
    if _fixture_path and Path(_fixture_path).is_file():
        _NC_PAO_FIXTURE = Path(_fixture_path)
except Exception:
    pass

_needs_fixture = pytest.mark.skipif(
    _NC_PAO_FIXTURE is None, reason=_NC_PAO_FIXTURE_REASON
)


@_needs_fixture
def test_real_abinit_nc_pao_fixture_loader():
    """Story 10 TEST-001: Load a real ABINIT NC PAO file and validate schema."""
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import load_abinit_nc_pao_savetb2j

    try:
        data = load_abinit_nc_pao_savetb2j(_NC_PAO_FIXTURE)
    except ValueError as exc:
        if "overlap metadata is not exchange-ready" in str(exc):
            pytest.skip(str(exc))
        raise
    assert data.coefficient_projector == "nc_pao"
    assert data.channel_interpretation == "norm_conserving_pao"
    assert data.kpoints.shape[1] == 3
    assert data.eigenvalues.ndim == 3
    assert data.coefficients.ndim == 4
    assert data.nproj > 0
    # overlap_k may or may not be present; if present shape must be correct
    if data.overlap_k is not None:
        nkpt, nproj, _ = data.overlap_k.shape
        assert nkpt == data.nkpt
        assert nproj == data.nproj


@_needs_fixture
def test_real_abinit_nc_pao_fixture_exchange_smoke(tmp_path):
    """Story 10 TEST-002: Exchange smoke test on real ABINIT NC PAO fixture."""
    pytest.importorskip("netCDF4")
    from TB2J.interfaces.abinit_savetb2j import gen_exchange_abinit_nc_pao

    try:
        exchange_out, jdict = gen_exchange_abinit_nc_pao(
            _NC_PAO_FIXTURE,
            output_path=tmp_path / "TB2J_results",
            Rmax=0,
            nz=2,
            operator_component="spectral_spin_split",
        )
    except ValueError as exc:
        if "overlap metadata is not exchange-ready" in str(
            exc
        ) or "requires full-BZ spectral coefficients" in str(exc):
            pytest.skip(str(exc))
        raise
    assert exchange_out.is_file()
    assert jdict
