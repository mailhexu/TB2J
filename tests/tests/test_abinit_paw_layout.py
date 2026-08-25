"""Contract tests for heterogeneous ABINIT PAW XML layouts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.interfaces.abinit_paw import (
    build_abinit_paw_site_layout,
    build_abinit_paw_snapshot,
    normalize_paw_xml_mapping,
)
from TB2J.paw_projector import build_projector_green_data


class _Pseudo:
    def __init__(self, labels):
        self.channel_labels = tuple(labels)


def _xmls(tmp_path: Path):
    fe = tmp_path / "Fe.xml"
    oxygen = tmp_path / "O.xml"
    fe.write_text("<paw>Fe</paw>")
    oxygen.write_text("<paw>O</paw>")
    return fe, oxygen


def test_two_species_mapping_builds_distinct_physical_layout(tmp_path: Path):
    fe_xml, o_xml = _xmls(tmp_path)
    mapping = normalize_paw_xml_mapping(["Fe", "O", "Fe"], {"Fe": fe_xml, "O": o_xml})
    layout = build_abinit_paw_site_layout(
        ["Fe", "O", "Fe"],
        mapping,
        {"Fe": _Pseudo(((3, 2),)), "O": _Pseudo(((2, 0), (2, 1)))},
    )

    assert [(site.species, site.projector_slice) for site in layout] == [
        ("Fe", slice(0, 5)),
        ("O", slice(5, 9)),
        ("Fe", slice(9, 14)),
    ]
    assert [
        (channel.radial, channel.l, channel.m) for channel in layout[1].channels
    ] == [
        (0, 0, 0),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    assert layout[0].setup_hash == layout[2].setup_hash
    assert layout[0].setup_hash != layout[1].setup_hash


@pytest.mark.parametrize(
    ("species", "mapping", "message"),
    [
        (["Fe", "O"], {"Fe": "Fe.xml"}, "source site 1 species 'O'"),
        (["Fe"], {"Fe": "Fe.xml", "O": "O.xml"}, "unknown species mapping"),
        (["Fe", "O"], "Fe.xml", "single PAW XML"),
    ],
)
def test_mapping_errors_identify_source_remediation(species, mapping, message):
    with pytest.raises(ValueError, match=message):
        normalize_paw_xml_mapping(species, mapping)


def test_layout_dimension_error_identifies_site_species_channels(tmp_path: Path):
    fe_xml, o_xml = _xmls(tmp_path)
    mapping = normalize_paw_xml_mapping(["Fe", "O"], {"Fe": fe_xml, "O": o_xml})

    with pytest.raises(
        ValueError, match="source site 1 species 'O'.*expected 4 channels"
    ):
        build_abinit_paw_site_layout(
            ["Fe", "O"],
            mapping,
            {"Fe": _Pseudo(((3, 2),)), "O": _Pseudo(((2, 0), (2, 1)))},
            site_slices=(slice(0, 5), slice(5, 8)),
        )


def test_magnetic_selection_preserves_full_source_layout_and_provenance(tmp_path: Path):
    fe_xml, o_xml = _xmls(tmp_path)
    mapping = normalize_paw_xml_mapping(["Fe", "O"], {"Fe": fe_xml, "O": o_xml})
    layout = build_abinit_paw_site_layout(
        ["Fe", "O"],
        mapping,
        {"Fe": _Pseudo(((3, 0),)), "O": _Pseudo(((2, 0),))},
    )
    snapshot = build_abinit_paw_snapshot(
        cprj_per_kpt=[[np.ones((2, 1), dtype=complex), np.ones((2, 1), dtype=complex)]],
        delta_ij={0: np.array([[0.1]]), 1: np.array([[0.2]])},
        eigenvalues=np.array([[[-1.0]], [[-0.8]]]),
        kweights=np.array([1.0]),
        kpoints=np.zeros((1, 3)),
        efermi=0.0,
        site_layout=layout,
        cell=np.eye(3),
        positions=np.zeros((2, 3)),
        atomic_numbers=np.array([26, 8]),
        delta_unit="hartree",
        provenance={"functional": "PBE"},
        selected_source_sites=(0,),
    )
    data = build_projector_green_data(snapshot)

    assert data.metadata["selected_source_sites"] == [0]
    assert [item["species"] for item in data.metadata["site_layout"]] == ["Fe", "O"]
    assert data.projector_site.tolist() == [0, 1]
