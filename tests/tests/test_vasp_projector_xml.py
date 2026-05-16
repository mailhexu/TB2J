from __future__ import annotations

import numpy as np
import pytest

from TB2J.interfaces.vasp_projector_xml import (
    compare_green_to_vasp_outcar_populations,
    gen_exchange_vasp_projector_xml,
    load_vasp_outcar_charges_moments,
    load_vasp_projector_xml,
    vasp_projected_charges_moments,
)
from TB2J.projector_green import ProjectorGreen


def _numbers(values):
    return " ".join(str(value) for value in values)


def _complex_numbers(values):
    out = []
    for value in np.asarray(values, dtype=complex).ravel():
        out.extend([value.real, value.imag])
    return _numbers(out)


def _write_fixture(
    path,
    *,
    bz="full",
    isym=None,
    malformed_coefficients=False,
    nonnumeric_weights=False,
    nonfinite_weights=False,
    fractional_projector_site=False,
    include_efermi=True,
):
    coefficients = np.zeros((2, 2, 2, 2), dtype=complex)
    coefficients[:, :, 0, 0] = 1.0
    coefficients[:, :, 1, 1] = 1.0
    if malformed_coefficients:
        coefficient_text = _numbers([1.0, 0.0, 2.0])
    else:
        coefficient_text = _complex_numbers(coefficients)
    if nonnumeric_weights:
        weights_text = "0.5 0.5 garbage"
    elif nonfinite_weights:
        weights_text = "0.5 nan"
    else:
        weights_text = "0.5 0.5"
    projector_site_text = "0.2 0" if fractional_projector_site else "0 0"
    efermi_attr = ' efermi="0.0"' if include_efermi else ""
    isym_text = "" if isym is None else f'    <item name="isym">{isym}</item>\n'

    text = f"""<?xml version="1.0"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp" source_version="6.4.1"
    spin_mode="collinear" kpoint_convention="fractional_reciprocal"
    phase_convention="exp(-2*pi*i*k.R)">
  <metadata>
    <item name="source">"synthetic VASP XML fixture"</item>
    <item name="symmetry_provenance">"full-BZ fixture"</item>
{isym_text.rstrip()}
  </metadata>
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="2"/>
    <dim name="nband" value="2"/>
    <dim name="nproj" value="2"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="{bz}">
    <array name="kpoints" dims="nkpt three">0 0 0 0.5 0 0</array>
    <array name="weights" dims="nkpt">{weights_text}</array>
  </kpoints>
  <bands{efermi_attr}>
    <array name="eigenvalues" dims="nspin nkpt nband">
      0.0 2.0 0.5 2.5 0.2 2.2 0.7 2.7
    </array>
    <array name="occupations" dims="nspin nkpt nband">
      1 0 1 0 1 0 1 0
    </array>
  </bands>
  <projectors coefficient_source="vasp.LPRJ_COVL"
      coefficient_projector="vasp_locproj"
      channel_interpretation="vasp_projector_channel">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{coefficient_text}</array>
    <array name="projector_site" dims="nproj" dtype="int">{projector_site_text}</array>
    <array name="projector_atom" dims="nproj" dtype="int">0 0</array>
    <array name="projector_l" dims="nproj" dtype="int">2 2</array>
    <array name="projector_m" dims="nproj" dtype="int">-2 -1</array>
    <array name="projector_radial" dims="nproj" dtype="int">0 0</array>
    <array name="site_nproj" dims="nsite" dtype="int">2</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0 1</array>
  </projectors>
</tb2j_projector_green>
"""
    path.write_text(text)


def test_vasp_projector_xml_loads_spectral_fixture(tmp_path):
    filename = tmp_path / "vasp_projector.xml"
    _write_fixture(filename)

    data = load_vasp_projector_xml(filename)

    assert data.metadata["source_code"] == "vasp"
    assert data.metadata["kpoint_set"] == "full_bz"
    assert data.coefficient_source == "vasp.LPRJ_COVL"
    assert data.coefficient_projector == "vasp_locproj"
    assert data.channel_interpretation == "vasp_projector_channel"
    assert data.nspin == 2
    assert data.nkpt == 2
    assert data.nband == 2
    assert data.nproj == 2
    np.testing.assert_allclose(data.weights, [0.5, 0.5])

    green = ProjectorGreen(data)
    gk = green.get_Gk(ik=0, energy=1.0 + 0.5j, ispin=0)
    expected = np.diag([1.0 / (1.0 + 0.5j), 1.0 / (-1.0 + 0.5j)])
    np.testing.assert_allclose(gk, expected)


def test_vasp_projector_xml_loads_generated_spectral_groups(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    filename.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <metadata writer_status="spectral" full_bz_required="true">
    <source name="vasp" writer="tb2j_projector_xml" />
    <item name="symmetry_provenance">"native_full_bz_symmetry_off"</item>
    <item name="isym">0</item>
  </metadata>
  <dimensions>
    <dim name="nspin" value="1"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="2"/>
    <dim name="nproj" value="0"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="0"/>
    <dim name="nproj_site_max" value="0"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
    <array name="symbols" dims="natom" dtype="string">Fe</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">0 1</array>
    <array name="occupations" dims="nspin nkpt nband">1 0</array>
  </bands>
  <projectors coefficient_source="pending" coefficient_projector="pending"
      channel_interpretation="pending">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved"></array>
    <array name="projector_site" dims="nproj" dtype="int"></array>
    <array name="projector_atom" dims="nproj" dtype="int"></array>
  </projectors>
  <operators />
</tb2j_projector_green>
"""
    )

    data = load_vasp_projector_xml(filename)

    assert data.metadata["symmetry_provenance"] == "native_full_bz_symmetry_off"
    assert data.nspin == 1
    assert data.nkpt == 1
    assert data.nband == 2
    assert data.nproj == 0
    np.testing.assert_allclose(data.eigenvalues, [[[0.0, 1.0]]])
    np.testing.assert_allclose(data.occupations, [[[1.0, 0.0]]])


def test_vasp_projector_xml_converts_legacy_direct_positions(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    filename.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <dimensions>
    <dim name="nspin" value="1"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="1"/>
    <dim name="nproj" value="0"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="0"/>
    <dim name="nproj_site_max" value="0"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">2 0 0 0 2 0 0 0 2</array>
    <array name="positions" dims="natom three">0.5 0.5 0.5</array>
    <array name="atomic_numbers" dims="natom" dtype="int">28</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">0</array>
    <array name="occupations" dims="nspin nkpt nband">1</array>
  </bands>
  <projectors coefficient_source="pending" coefficient_projector="pending"
      channel_interpretation="pending">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved"></array>
    <array name="projector_site" dims="nproj" dtype="int"></array>
    <array name="projector_atom" dims="nproj" dtype="int"></array>
  </projectors>
</tb2j_projector_green>
"""
    )

    data = load_vasp_projector_xml(filename)

    np.testing.assert_allclose(data.positions, [[1.0, 1.0, 1.0]])


def test_vasp_projector_xml_loads_generated_operator_groups(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = _complex_numbers(np.ones((2, 1, 1, 1), dtype=complex))
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <metadata writer_status="spectral" full_bz_required="true">
    <source name="vasp" writer="tb2j_projector_xml" />
  </metadata>
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="1"/>
    <dim name="nproj" value="1"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="1"/>
    <dim name="noperator_spin" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">0 0.1</array>
    <array name="occupations" dims="nspin nkpt nband">1 1</array>
  </bands>
  <projectors coefficient_source="vasp.LPRJ_COVL"
      coefficient_projector="vasp_locproj"
      channel_interpretation="vasp_locproj_function"
      overlap_metric_definition="real(CQIJ) in LPRJ function basis">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{coefficients}</array>
    <array name="projector_site" dims="nproj" dtype="int">0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0</array>
    <array name="projector_l" dims="nproj" dtype="int">2</array>
    <array name="projector_m" dims="nproj" dtype="int">0</array>
    <array name="projector_radial" dims="nproj" dtype="int">1</array>
    <array name="site_nproj" dims="nsite" dtype="int">1</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0</array>
    <array name="overlap_metric" dims="nproj nproj">1</array>
  </projectors>
  <operators operator_basis="vasp_cdij_paw_hamiltonian"
      hij_definition="spin-resolved real(CDIJ) in native PAW projector basis"
      hij_units="eV" hij_source="vasp.CDIJ"
      hij_projection="site_local_paw_channel_subblocks">
    <array name="hij" dims="noperator_spin nsite nproj_site_max nproj_site_max">
      2 0.5
    </array>
  </operators>
</tb2j_projector_green>
"""
    )

    data = load_vasp_projector_xml(filename)

    data.validate(exchange_ready=True)
    assert data.coefficient_source == "vasp.LPRJ_COVL"
    assert data.operator_basis == "vasp_cdij_paw_hamiltonian"
    np.testing.assert_allclose(data.overlap_metric, [[1.0]])
    np.testing.assert_allclose(data.get_hij_spin_difference(site=0), [[1.5]])


def test_vasp_projector_xml_uses_qtot_population_metric(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = np.zeros((2, 1, 1, 2), dtype=complex)
    coefficients[0, 0, 0, 0] = 1.0
    coefficients[0, 0, 0, 1] = 2.0
    coefficients[1, 0, 0, 0] = 0.5
    coefficients[1, 0, 0, 1] = 1.0
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="1"/>
    <dim name="nproj" value="2"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">-1 -1</array>
    <array name="occupations" dims="nspin nkpt nband">1 1</array>
  </bands>
  <projectors coefficient_source="vasp.W_CPROJ"
      coefficient_projector="native_paw_projector"
      channel_interpretation="paw_projector_channel"
      population_metric="VASP QTOT LORBIT projector population metric">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{_complex_numbers(coefficients)}</array>
    <array name="projector_site" dims="nproj" dtype="int">0 0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0 0</array>
    <array name="projector_l" dims="nproj" dtype="int">2 2</array>
    <array name="projector_m" dims="nproj" dtype="int">0 0</array>
    <array name="projector_radial" dims="nproj" dtype="int">1 2</array>
    <array name="site_nproj" dims="nsite" dtype="int">2</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0 1</array>
    <array name="population_metric_matrix" dims="nproj nproj">2 0.25 0.25 3</array>
  </projectors>
</tb2j_projector_green>
"""
    )
    (tmp_path / "OUTCAR").write_text(
        """
 total charge
# of ion       s       p       d       tot
    1        0.0 0.0 18.7500 18.7500
tot          0.0 0.0 18.7500 18.7500
 magnetization (x)
# of ion       s       p       d       tot
    1        0.0 0.0 11.2500 11.2500
tot          0.0 0.0 11.2500 11.2500
"""
    )

    data = load_vasp_projector_xml(filename)
    populations = vasp_projected_charges_moments(data)
    comparison = compare_green_to_vasp_outcar_populations(data, tmp_path / "OUTCAR")

    np.testing.assert_allclose(data.population_metric_matrix, [[2, 0.25], [0.25, 3]])
    np.testing.assert_allclose(populations["charges"], [18.75])
    np.testing.assert_allclose(populations["spinat"][:, 2], [11.25])
    assert populations["method"] == "vasp_qtot_projected_population"
    assert comparison["matches"]
    assert comparison["method"] == "vasp_qtot_projected_population"


def test_vasp_projector_xml_uses_delta_total_component_for_exchange(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = _complex_numbers(np.ones((2, 1, 2, 1), dtype=complex))
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="2"/>
    <dim name="nproj" value="1"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="1"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">-1 1 -0.8 1.2</array>
    <array name="occupations" dims="nspin nkpt nband">1 0 1 0</array>
  </bands>
  <projectors coefficient_source="vasp.LPRJ_COVL"
      coefficient_projector="vasp_locproj"
      channel_interpretation="vasp_locproj_function">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{coefficients}</array>
    <array name="projector_site" dims="nproj" dtype="int">0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0</array>
    <array name="site_nproj" dims="nsite" dtype="int">1</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0</array>
  </projectors>
  <operators operator_basis="vasp_locproj_trial_function"
      operator_kind="spin_splitting_components"
      delta_definition="spin-splitting matrix in VASP LOCPROJ trial-function basis"
      delta_units="eV" delta_source="vasp.delta_xc_plus_u"
      delta_projection="site_local_locproj_trial_function_blocks">
    <array name="delta_total" dims="nsite nproj_site_max nproj_site_max">1.5</array>
    <array name="delta_xc" dims="nsite nproj_site_max nproj_site_max">1.0</array>
    <array name="delta_u" dims="nsite nproj_site_max nproj_site_max">0.5</array>
    <array name="delta_u_paw_aug" dims="nsite nproj_site_max nproj_site_max">0.5</array>
  </operators>
</tb2j_projector_green>
"""
    )

    data = load_vasp_projector_xml(filename)
    exchange_out, exchange_Jdict = gen_exchange_vasp_projector_xml(
        filename,
        output_path=tmp_path / "TB2J_results_vasp_xml",
        Rmax=0,
        nz=4,
        population_source="none",
    )

    assert data.operator_basis == "vasp_locproj_trial_function"
    np.testing.assert_allclose(
        data.get_operator_component("delta_total", site=0), [[1.5]]
    )
    np.testing.assert_allclose(data.get_operator_component("delta_xc", site=0), [[1.0]])
    np.testing.assert_allclose(data.get_operator_component("delta_u", site=0), [[0.5]])
    np.testing.assert_allclose(
        data.get_operator_component("delta_u_paw_aug", site=0), [[0.5]]
    )
    assert exchange_out.exists()
    assert ((0, 0, 0), 0, 0) in exchange_Jdict


def test_vasp_projector_xml_exchange_writes_qtot_populations(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = np.zeros((2, 1, 1, 2), dtype=complex)
    coefficients[0, 0, 0, 0] = 1.0
    coefficients[0, 0, 0, 1] = 2.0
    coefficients[1, 0, 0, 0] = 0.5
    coefficients[1, 0, 0, 1] = 1.0
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="1"/>
    <dim name="nproj" value="2"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="2"/>
    <dim name="noperator_spin" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">-1 -1</array>
    <array name="occupations" dims="nspin nkpt nband">1 1</array>
  </bands>
  <projectors coefficient_source="vasp.W_CPROJ"
      coefficient_projector="native_paw_projector"
      channel_interpretation="paw_projector_channel"
      population_metric="VASP QTOT LORBIT projector population metric">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{_complex_numbers(coefficients)}</array>
    <array name="projector_site" dims="nproj" dtype="int">0 0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0 0</array>
    <array name="site_nproj" dims="nsite" dtype="int">2</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0 1</array>
    <array name="population_metric_matrix" dims="nproj nproj">2 0.25 0.25 3</array>
  </projectors>
  <operators operator_basis="vasp_cdij_paw_hamiltonian"
      hij_definition="spin-resolved real(CDIJ) in native PAW projector basis"
      hij_units="eV" hij_source="vasp.CDIJ"
      hij_projection="site_local_paw_channel_subblocks">
    <array name="hij" dims="noperator_spin nsite nproj_site_max nproj_site_max">
      2 0 0 2 0.5 0 0 0.5
    </array>
  </operators>
</tb2j_projector_green>
"""
    )

    exchange_out, _ = gen_exchange_vasp_projector_xml(
        filename,
        output_path=tmp_path / "TB2J_results_vasp_xml",
        Rmax=0,
        nz=4,
        population_source="green",
    )

    text = exchange_out.read_text()
    assert "vasp_qtot_projected_population" in text
    assert "  18.7500   11.2500" in text


def test_vasp_projector_xml_writes_exchange_out(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = _complex_numbers(np.ones((2, 1, 2, 1), dtype=complex))
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <metadata writer_status="projectors_operators" full_bz_required="true">
    <source name="vasp" writer="tb2j_projector_xml" />
  </metadata>
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="2"/>
    <dim name="nproj" value="1"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="1"/>
    <dim name="noperator_spin" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">-1 1 -0.8 1.2</array>
    <array name="occupations" dims="nspin nkpt nband">1 0 1 0</array>
  </bands>
  <projectors coefficient_source="vasp.LPRJ_COVL"
      coefficient_projector="vasp_locproj"
      channel_interpretation="vasp_locproj_function"
      overlap_metric_definition="real(CQIJ) in LPRJ function basis">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{coefficients}</array>
    <array name="projector_site" dims="nproj" dtype="int">0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0</array>
    <array name="projector_l" dims="nproj" dtype="int">2</array>
    <array name="projector_m" dims="nproj" dtype="int">0</array>
    <array name="projector_radial" dims="nproj" dtype="int">1</array>
    <array name="site_nproj" dims="nsite" dtype="int">1</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0</array>
    <array name="overlap_metric" dims="nproj nproj">1</array>
  </projectors>
  <operators operator_basis="vasp_cdij_paw_hamiltonian"
      hij_definition="spin-resolved real(CDIJ) in native PAW projector basis"
      hij_units="eV" hij_source="vasp.CDIJ"
      hij_projection="site_local_paw_channel_subblocks">
    <array name="hij" dims="noperator_spin nsite nproj_site_max nproj_site_max">
      2 0.5
    </array>
  </operators>
</tb2j_projector_green>
"""
    )
    (tmp_path / "OUTCAR").write_text(
        """
 total charge

# of ion       s       p       d       tot
------------------------------------------
    1        0.100   0.200   1.200   1.500
--------------------------------------------------
tot          0.100   0.200   1.200   1.500

 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1        0.010   0.020   0.470   0.500
--------------------------------------------------
tot          0.010   0.020   0.470   0.500
"""
    )

    exchange_out, exchange_Jdict = gen_exchange_vasp_projector_xml(
        filename,
        output_path=tmp_path / "TB2J_results_vasp_xml",
        Rmax=0,
        nz=4,
        population_source="outcar",
        allow_basis_mismatch=True,
    )

    assert exchange_out.exists()
    assert ((0, 0, 0), 0, 0) in exchange_Jdict
    text = exchange_out.read_text()
    assert "VASP XML" in text
    assert "vasp_cdij_paw_hamiltonian" in text
    assert "copied from VASP OUTCAR" in text
    assert "Fe1" in text
    assert "   1.5000    0.5000" in text


def test_vasp_outcar_charges_moments_uses_final_lorbit_blocks(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """
 total charge
# of ion       s       p       d       tot
    1        0.0 0.0 0.0 9.0
tot          0.0 0.0 0.0 9.0
 magnetization (x)
# of ion       s       p       d       tot
    1        0.0 0.0 0.0 9.0
tot          0.0 0.0 0.0 9.0
 total charge
# of ion       s       p       d       tot
    1        0.819   6.246   7.421  14.487
    2        0.819   6.246   7.421  14.487
    3        1.756   3.081   0.000   4.837
    4        1.756   3.081   0.000   4.837
tot          5.150  18.656  14.843  38.649
 magnetization (x)
# of ion       s       p       d       tot
    1        0.023   0.060   1.612   1.694
    2       -0.023  -0.060  -1.612  -1.694
    3        0.000   0.000   0.000   0.000
    4        0.000   0.000   0.000   0.000
tot          0.000   0.000   0.000   0.000
"""
    )

    charges, spinat = load_vasp_outcar_charges_moments(outcar, natom=4)

    np.testing.assert_allclose(charges, [14.487, 14.487, 4.837, 4.837])
    np.testing.assert_allclose(spinat[:, 2], [1.694, -1.694, 0.0, 0.0])


def test_vasp_outcar_charges_moments_reads_f_column_without_tot_line(tmp_path):
    outcar = tmp_path / "OUTCAR"
    outcar.write_text(
        """
 total charge

# of ion       s       p       d       f       tot
--------------------------------------------------
    1        0.615   0.836   5.450   0.076   6.976


 magnetization (x)

# of ion       s       p       d       f       tot
--------------------------------------------------
    1        0.003  -0.013   0.763   0.005   0.758

 total amount of memory used by VASP MPI-rank0    31846. kBytes
"""
    )

    charges, spinat = load_vasp_outcar_charges_moments(outcar, natom=1)

    np.testing.assert_allclose(charges, [6.976])
    np.testing.assert_allclose(spinat[:, 2], [0.758])


def test_vasp_green_population_mismatch_raises_with_outcar(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    _write_fixture(filename)
    (tmp_path / "OUTCAR").write_text(
        """
 total charge
# of ion       s       p       d       tot
    1        0.0 0.0 0.0 99.0
tot          0.0 0.0 0.0 99.0
 magnetization (x)
# of ion       s       p       d       tot
    1        0.0 0.0 0.0 99.0
tot          0.0 0.0 0.0 99.0
"""
    )

    data = load_vasp_projector_xml(filename)
    comparison = compare_green_to_vasp_outcar_populations(
        data, tmp_path / "OUTCAR", nz=4, charge_atol=0.1, moment_atol=0.1
    )

    assert not comparison["matches"]
    assert comparison["charge_diff"][0] < -90.0


def test_vasp_projector_xml_exchange_rejects_isym_by_default(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    _write_fixture(filename, isym=2)

    with pytest.raises(ValueError, match="ISYM>0"):
        gen_exchange_vasp_projector_xml(
            filename,
            output_path=tmp_path / "TB2J_results_vasp_xml",
            Rmax=0,
            nz=4,
            population_source="none",
        )


def test_vasp_projector_xml_exchange_rejects_lprj_cdij_by_default(tmp_path):
    filename = tmp_path / "tb2j_projector.xml"
    coefficients = _complex_numbers(np.ones((2, 1, 2, 1), dtype=complex))
    filename.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<tb2j_projector_green schema_name="tb2j.projector_green.xml"
    schema_version="0.1" source_code="vasp">
  <dimensions>
    <dim name="nspin" value="2"/>
    <dim name="nkpt" value="1"/>
    <dim name="nband" value="2"/>
    <dim name="nproj" value="1"/>
    <dim name="natom" value="1"/>
    <dim name="nsite" value="1"/>
    <dim name="nproj_site_max" value="1"/>
    <dim name="noperator_spin" value="2"/>
  </dimensions>
  <structure>
    <array name="cell" dims="three three">1 0 0 0 1 0 0 0 1</array>
    <array name="positions" dims="natom three">0 0 0</array>
    <array name="atomic_numbers" dims="natom" dtype="int">26</array>
  </structure>
  <kpoints bz="full" convention="fractional_reciprocal">
    <array name="kpoints" dims="nkpt three">0 0 0</array>
    <array name="weights" dims="nkpt">1</array>
  </kpoints>
  <bands efermi="0.0">
    <array name="eigenvalues" dims="nspin nkpt nband">-1 1 -0.8 1.2</array>
    <array name="occupations" dims="nspin nkpt nband">1 0 1 0</array>
  </bands>
  <projectors coefficient_source="vasp.LPRJ_COVL"
      coefficient_projector="vasp_locproj"
      channel_interpretation="vasp_locproj_function">
    <array name="coefficients" dims="nspin nkpt nband nproj"
        dtype="complex128" complex="interleaved">{coefficients}</array>
    <array name="projector_site" dims="nproj" dtype="int">0</array>
    <array name="projector_atom" dims="nproj" dtype="int">0</array>
    <array name="site_nproj" dims="nsite" dtype="int">1</array>
    <array name="site_projector_indices" dims="nsite nproj_site_max" dtype="int">0</array>
  </projectors>
  <operators operator_basis="vasp_cdij_paw_hamiltonian"
      hij_definition="spin-resolved real(CDIJ) in native PAW projector basis"
      hij_units="eV" hij_source="vasp.CDIJ"
      hij_projection="site_local_paw_channel_subblocks">
    <array name="hij" dims="noperator_spin nsite nproj_site_max nproj_site_max">
      2 0.5
    </array>
  </operators>
</tb2j_projector_green>
"""
    )

    with pytest.raises(ValueError, match="LPRJ_COVL coefficients with native CDIJ"):
        gen_exchange_vasp_projector_xml(
            filename,
            output_path=tmp_path / "TB2J_results_vasp_xml",
            Rmax=0,
            nz=4,
            population_source="none",
        )


def test_vasp_projector_xml_rejects_malformed_complex_array(tmp_path):
    filename = tmp_path / "bad_vasp_projector.xml"
    _write_fixture(filename, malformed_coefficients=True)

    with pytest.raises(ValueError, match="interleaved complex data"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_rejects_nonnumeric_array_data(tmp_path):
    filename = tmp_path / "bad_weights_vasp_projector.xml"
    _write_fixture(filename, nonnumeric_weights=True)

    with pytest.raises(ValueError, match="non-numeric data"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_rejects_nonfinite_array_data(tmp_path):
    filename = tmp_path / "bad_nonfinite_vasp_projector.xml"
    _write_fixture(filename, nonfinite_weights=True)

    with pytest.raises(ValueError, match="non-finite values"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_rejects_fractional_integer_array(tmp_path):
    filename = tmp_path / "bad_indices_vasp_projector.xml"
    _write_fixture(filename, fractional_projector_site=True)

    with pytest.raises(ValueError, match="non-integer values"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_requires_fermi_level(tmp_path):
    filename = tmp_path / "missing_efermi_vasp_projector.xml"
    _write_fixture(filename, include_efermi=False)

    with pytest.raises(ValueError, match="requires efermi"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_rejects_ibz_only_data(tmp_path):
    filename = tmp_path / "ibz_vasp_projector.xml"
    _write_fixture(filename, bz="ibz")

    with pytest.raises(ValueError, match="full-BZ k-point data"):
        load_vasp_projector_xml(filename)


def test_vasp_projector_xml_spectral_only_is_not_exchange_ready(tmp_path):
    filename = tmp_path / "vasp_projector.xml"
    _write_fixture(filename)
    data = load_vasp_projector_xml(filename)

    with pytest.raises(ValueError, match="exchange-ready projector data"):
        data.validate(exchange_ready=True)
