"""Public QE source-selection and legacy-admission behavior tests."""

from __future__ import annotations

import numpy as np
import pytest
from ase.units import kB
from test_qe_hubbard import (
    _active_collinear_dump,
    _legacy_dump,
    _mutate_record,
    _set_f8,
)
from test_qe_hubbard_joint import (
    _complex_atomic_grams,
    _native_coefficients_for_targets,
)
from test_qe_projector import RYTOEV

from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
from TB2J.interfaces.qe_projector import (
    gen_exchange_qe,
    read_qe_atomic_dump,
    read_qe_dump,
)
from TB2J.io_exchange import SpinIO
from TB2J.mycfr import CFR
from TB2J.projector_green import ProjectorGreen, projector_exchange_trace

_SOURCE_OCCUPATION = (0.2, 0.6)
_OVERLAP_RCOND = 3.0e-9
_NZ = 6
_SMEARING_EV = 0.05
_RPTS = [[0, 0, 0]]
_COMPONENTS = ("delta_total", "delta_xc", "delta_u")
_COMPLETENESS = {
    "delta_total": "complete_xc_u",
    "delta_xc": "diagnostic_xc_only",
    "delta_u": "diagnostic_u_only",
}


def _active_source(path, family):
    kwargs = {
        "family": family,
        "state_occupation": _SOURCE_OCCUPATION,
        "native_coefficients": _native_coefficients_for_targets,
    }
    if family == "atomic":
        kwargs["atomic_grams"] = _complex_atomic_grams()
    return _active_collinear_dump(path, **kwargs)


def _reader(family):
    return read_qe_dump if family == "kb" else read_qe_atomic_dump


def _reader_kwargs(family):
    if family == "atomic":
        return {"overlap_mode": "inverse", "overlap_rcond": _OVERLAP_RCOND}
    return {}


def _source_site(family):
    # The active fixture places the native Hubbard channel on the last KB site
    # and on the middle atomic-wfc site.
    return 2 if family == "kb" else 1


# MathDoc: tb2j/qe-hubbard-joint-frame#joint_trace
def _direct_single_site_exchange(data, site, component, *, overlap_mode, overlap_rcond):
    """Independent one-site CFR oracle for Tr[Delta G_up Delta G_down]."""
    green = ProjectorGreen(data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond)
    delta = data.get_operator_component(component, site=site)
    contour = CFR(nz=_NZ, T=_SMEARING_EV / kB)
    values = []
    for energy in contour.path:
        green_up = green.get_GR(_RPTS, energy=energy, ispin=0)[0]
        green_down = green.get_GR(_RPTS, energy=energy, ispin=1)[0]
        local_up = green.get_site_block(green_up, site, site)
        local_down = green.get_site_block(green_down, site, site)
        values.append(np.trace(delta @ local_up @ delta @ local_down) / (4.0 * np.pi))
    integrated = contour.integrate_values(np.asarray(values))
    # This is one site at both ends: its pair moment-sign product is +1.
    return float(np.imag(integrated))


@pytest.mark.parametrize("family", ["kb", "atomic"])
def test_active_reader_selects_complete_xc_or_u_without_refiltering_joint_data(
    tmp_path, family
):
    path = _active_source(tmp_path / f"{family}-selected.bin", family)
    read = _reader(family)
    reader_kwargs = _reader_kwargs(family)
    site = _source_site(family)

    # The legacy flag cannot relabel a source that explicitly declares active U.
    default = read(path, legacy_no_u=True, **reader_kwargs)
    assert default.metadata["hubbard_state"] == "active"
    assert np.linalg.norm(default.get_operator_component("delta_u")) > 0.0
    assert (
        np.linalg.norm(
            default.coefficients[
                ..., len(default.metadata["qe_channel_groups"]["xc"]) :
            ]
        )
        > 0.0
    )

    selected = {
        component: read(
            path,
            legacy_no_u=True,
            operator_component=component,
            **reader_kwargs,
        )
        for component in _COMPONENTS
    }
    assert default.metadata["operator_component"] == "delta_total"
    assert default.metadata["source_completeness"] == _COMPLETENESS["delta_total"]
    np.testing.assert_allclose(
        default.get_operator_component("delta_total"),
        default.get_operator_component("delta_xc")
        + default.get_operator_component("delta_u"),
    )

    for component, data in selected.items():
        assert data.metadata["hubbard_state"] == "active"
        assert data.metadata["operator_component"] == component
        assert data.metadata["source_completeness"] == _COMPLETENESS[component]
        assert data.operator_component_metadata[component]["completeness"] == "complete"
        np.testing.assert_allclose(
            data.get_hij_spin_difference(),
            data.get_operator_component(component),
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(data.coefficients, default.coefficients)

    assert (
        np.linalg.norm(selected["delta_xc"].get_operator_component("delta_xc", site))
        > 0.0
    )
    assert (
        np.linalg.norm(selected["delta_u"].get_operator_component("delta_u", site))
        > 0.0
    )

    channel_groups = default.metadata["qe_channel_groups"]
    assert channel_groups["u"]
    site_indices = default.site_projector_indices[site, : default.site_nproj[site]]
    assert set(channel_groups["u"]).issubset(set(site_indices))
    green = ProjectorGreen(default, overlap_mode="plain")
    cross_green = green.get_Gk(0, 0.47 + 0.23j, ispin=0)
    assert (
        np.linalg.norm(cross_green[np.ix_(channel_groups["xc"], channel_groups["u"])])
        > 1e-10
    )

    if family == "atomic":
        assert default.overlap_k is None
        assert default.original_overlap_k is not None
        assert default.metadata["effective_overlap_mode"] == "inverse"
        assert default.metadata["effective_overlap_rcond"] == _OVERLAP_RCOND


@pytest.mark.parametrize("family", ["kb", "atomic"])
@pytest.mark.parametrize("component", [None, "delta_xc", "delta_u"])
def test_selected_reader_drives_default_collinear_trace_and_exchange(
    tmp_path, family, component
):
    path = _active_source(tmp_path / f"{family}-default-consumer.bin", family)
    data = _reader(family)(path, operator_component=component, **_reader_kwargs(family))
    site = _source_site(family)
    overlap_mode = "inverse" if family == "atomic" else "plain"
    overlap_rcond = _OVERLAP_RCOND if family == "atomic" else 1.0e-10
    expected = _direct_single_site_exchange(
        data,
        site,
        component or "delta_total",
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )

    green = ProjectorGreen(data, overlap_mode=overlap_mode, overlap_rcond=overlap_rcond)
    contour = CFR(nz=_NZ, T=_SMEARING_EV / kB)
    values = [
        projector_exchange_trace(green, _RPTS, energy=energy, sites=[site])["trace"][
            ((0, 0, 0), site, site)
        ]
        for energy in contour.path
    ]
    assert float(
        np.imag(contour.integrate_values(np.asarray(values)))
    ) == pytest.approx(expected, rel=1e-10, abs=1e-12)
    generated = compute_projector_exchange_jdict(
        data,
        Rpts=_RPTS,
        nz=_NZ,
        smearing_eV=_SMEARING_EV,
        sites=[site],
        overlap_mode=overlap_mode,
        overlap_rcond=overlap_rcond,
    )
    assert generated[((0, 0, 0), 0, 0)] == pytest.approx(expected, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("family", ["kb", "atomic"])
def test_collinear_generator_writes_numerical_selected_exchange_to_spinio(
    tmp_path, family
):
    path = _active_source(tmp_path / f"{family}-generator.bin", family)
    read = _reader(family)
    reader_kwargs = _reader_kwargs(family)
    site = _source_site(family)
    overlap_mode = "inverse" if family == "atomic" else "plain"
    overlap_rcond = _OVERLAP_RCOND if family == "atomic" else 1.0e-10
    output_values = {}

    for component in (None, *_COMPONENTS):
        selected_component = component or "delta_total"
        label = component or "default"
        output_dir = tmp_path / "generated" / family / label
        output_file, generated = gen_exchange_qe(
            path,
            output_path=output_dir,
            Rpts=_RPTS,
            nz=_NZ,
            smearing_eV=_SMEARING_EV,
            index_magnetic_atoms=[site],
            operator_component=component,
            legacy_no_u=True,
            **_reader_kwargs(family),
        )

        normalized = read(
            path,
            legacy_no_u=True,
            operator_component=selected_component,
            **reader_kwargs,
        )
        expected = _direct_single_site_exchange(
            normalized,
            site,
            selected_component,
            overlap_mode=overlap_mode,
            overlap_rcond=overlap_rcond,
        )
        key = ((0, 0, 0), 0, 0)
        assert output_file == output_dir / "exchange.out"
        assert output_file.is_file()
        assert generated[key] == pytest.approx(expected, rel=1e-10, abs=1e-12)

        spinio = SpinIO.load_pickle(path=output_dir)
        assert spinio.colinear is True
        assert spinio.exchange_Jdict[key] == pytest.approx(
            expected, rel=1e-10, abs=1e-12
        )
        output_values[label] = spinio.exchange_Jdict[key]

    assert output_values["default"] == pytest.approx(
        output_values["delta_total"], rel=1e-12, abs=1e-12
    )
    # The native C/U and XC channels remain in one Green frame: the complete
    # exchange contains their mixed contractions and is not a sum of two runs.
    mixed_contraction = (
        output_values["delta_total"]
        - output_values["delta_xc"]
        - output_values["delta_u"]
    )
    assert abs(mixed_contraction) > 1e-12


@pytest.mark.parametrize("family", ["kb", "atomic"])
def test_legacy_unknown_requires_attestation_and_attested_writer_is_real(
    tmp_path, family
):
    path = _legacy_dump(tmp_path / f"{family}-legacy.bin", family)
    refused_output = tmp_path / f"{family}-unattested"
    with pytest.raises(ValueError):
        gen_exchange_qe(
            path,
            output_path=refused_output,
            Rpts=_RPTS,
            nz=_NZ,
            smearing_eV=_SMEARING_EV,
        )
    assert not refused_output.exists()

    attested_output = tmp_path / f"{family}-attested"
    output_file, generated = gen_exchange_qe(
        path,
        output_path=attested_output,
        Rpts=_RPTS,
        nz=_NZ,
        smearing_eV=_SMEARING_EV,
        legacy_no_u=True,
        **_reader_kwargs(family),
    )
    assert output_file.is_file()
    spinio = SpinIO.load_pickle(path=attested_output)
    assert spinio.exchange_Jdict == generated

    for component in ("delta_u", "not_a_qe_component"):
        unavailable_output = tmp_path / f"{family}-{component}-unavailable"
        with pytest.raises(ValueError):
            gen_exchange_qe(
                path,
                output_path=unavailable_output,
                Rpts=_RPTS,
                nz=_NZ,
                smearing_eV=_SMEARING_EV,
                legacy_no_u=True,
                operator_component=component,
                **_reader_kwargs(family),
            )
        assert not unavailable_output.exists()


@pytest.mark.parametrize("failure", ["unsupported", "incomplete"])
def test_legacy_attestation_cannot_bypass_active_source_gates_or_write_output(
    tmp_path, failure
):
    path = _active_source(tmp_path / f"active-{failure}.bin", "kb")
    if failure == "unsupported":

        def add_unsupported_j(payload):
            _set_f8(payload, 56, 0.25)
            _set_f8(payload, 128, 0.25 / RYTOEV)
            return payload

        _mutate_record(path, 4, add_unsupported_j, occurrence=2)
    else:
        path.write_bytes(path.read_bytes()[:-1])

    output_dir = tmp_path / f"refused-{failure}"
    with pytest.raises(ValueError):
        gen_exchange_qe(
            path,
            output_path=output_dir,
            Rpts=_RPTS,
            nz=_NZ,
            smearing_eV=_SMEARING_EV,
            legacy_no_u=True,
        )
    assert not output_dir.exists()
