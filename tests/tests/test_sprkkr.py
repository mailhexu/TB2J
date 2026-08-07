from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from TB2J.io_exchange import SpinIO
from TB2J.magnon.magnon3 import Magnon

ROOT_DIR = Path(__file__).resolve().parents[2]
REF_DIR = ROOT_DIR.parent / "Refs" / "SPRKKR_RuO2"
STRUCTURE_FILE = REF_DIR / "RuO2.str"
EXCHANGE_FILE = REF_DIR / "RuO2_JXC_Jij.dat"
FULL_EXCHANGE_FILE = REF_DIR / "RuO2_JXC_XCPLTEN_Jij.dat"


def test_read_sprkkr_structure_preserves_sites_and_spin_mapping():
    from TB2J.interfaces.sprkkr import read_sprkkr_structure

    structure = read_sprkkr_structure(
        STRUCTURE_FILE,
        magnetic_species=["Ru"],
    )

    assert structure.lattice_parameter_au == pytest.approx(8.490671147504)
    assert len(structure.symbols) == 14
    assert structure.symbols[:6] == ["Ru", "Ru", "O", "O", "O", "O"]
    assert structure.symbols[6:] == ["X"] * 8
    assert structure.site_type_labels[1] == "Ru"
    assert structure.site_type_labels[7] == "Vc_1"
    assert structure.index_spin[:2] == [0, 1]
    assert structure.index_spin[2:] == [-1] * 12
    assert structure.atoms.get_chemical_symbols()[:2] == ["Ru", "Ru"]
    assert np.allclose(structure.atoms.get_pbc(), [True, True, True])
    assert structure.atoms.cell.lengths()[0] == pytest.approx(
        8.490671147504 * 0.529177210903
    )


def test_read_sprkkr_structure_reports_missing_sections(tmp_path):
    from TB2J.interfaces.sprkkr import SprkkrParseError, read_sprkkr_structure

    bad_file = tmp_path / "bad.str"
    bad_file.write_text("lattice parameter A  [a.u.]\n 1.0\n")

    with pytest.raises(SprkkrParseError, match="missing required section"):
        read_sprkkr_structure(bad_file, magnetic_species=["Ru"])


def test_read_sprkkr_exchange_table_parses_ru_only_rows():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange_table

    table = read_sprkkr_exchange_table(EXCHANGE_FILE)

    assert len(table.rows) == 118
    assert table.units["exchange"] == "meV"
    assert table.units["displacement"] == "lattice_parameter"
    assert table.rows[0].site_i == 1
    assert table.rows[0].site_j == 1
    assert table.rows[0].n123 == (0, 0, 1)
    assert table.rows[0].j_xx_mev == pytest.approx(1.55334513)
    assert table.rows[0].j_xy_mev == pytest.approx(4.81466537)
    assert {row.site_i for row in table.rows} == {1, 2}
    assert {row.site_j for row in table.rows} == {1, 2}


def test_read_sprkkr_exchange_table_rejects_bad_row_after_table_starts(tmp_path):
    from TB2J.interfaces.sprkkr import SprkkrParseError, read_sprkkr_exchange_table

    bad_file = tmp_path / "bad_Jij.dat"
    bad_file.write_text(
        "header text\n"
        " 1 1 1 1 0 0 0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0\n"
        " trailing prose after numeric table\n"
    )

    with pytest.raises(SprkkrParseError, match="unexpected text after exchange table"):
        read_sprkkr_exchange_table(bad_file)


def test_read_sprkkr_exchange_table_rejects_missing_tensor_column(tmp_path):
    from TB2J.interfaces.sprkkr import SprkkrParseError, read_sprkkr_exchange_table

    bad_file = tmp_path / "missing_column_Jij.dat"
    bad_file.write_text(" 1 1 1 1 0 0 0 0.0 0.0 0.0 0.0 1.0 1.0 0.0\n")

    with pytest.raises(SprkkrParseError, match="invalid exchange row shape"):
        read_sprkkr_exchange_table(bad_file)


def test_read_sprkkr_exchange_table_filters_full_table_to_magnetic_sites():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange_table

    table = read_sprkkr_exchange_table(FULL_EXCHANGE_FILE)
    filtered = table.filter_by_sites({1, 2})

    assert len(table.rows) == 18251
    assert len(filtered.rows) > len(read_sprkkr_exchange_table(EXCHANGE_FILE).rows)
    assert {row.site_i for row in filtered.rows} == {1, 2}
    assert {row.site_j for row in filtered.rows} == {1, 2}


def test_read_sprkkr_exchange_combines_structure_and_exchange():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
    )

    assert len(data.structure.symbols) == 14
    assert len(data.rows) == 118
    assert data.moments_by_site == {
        1: (0.0, 0.0, 0.5674),
        2: (0.0, 0.0, 0.5674),
    }
    assert data.source_files["structure"] == str(STRUCTURE_FILE)


def test_read_sprkkr_exchange_accepts_signed_site_moments():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=[0.5674, -0.5674],
    )
    spinio = sprkkr_to_spinio(data, tensor_policy="isotropic")

    assert data.moments_by_site == {
        1: (0.0, 0.0, 0.5674),
        2: (0.0, 0.0, -0.5674),
    }
    assert np.allclose(
        spinio.get_magnetic_moments(),
        [[0.0, 0.0, 0.5674], [0.0, 0.0, -0.5674]],
    )


def test_read_sprkkr_exchange_accepts_explicit_moment_vectors():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=[0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
    )
    spinio = sprkkr_to_spinio(data, tensor_policy="isotropic")

    assert data.moments_by_site == {
        1: (0.1, 0.2, 0.3),
        2: (-0.1, -0.2, -0.3),
    }
    assert np.allclose(
        spinio.get_magnetic_moments(),
        [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]],
    )


def test_read_sprkkr_exchange_accepts_dict_moments_and_rejects_bad_lengths():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment={1: 0.1, 2: [0.0, 0.0, -0.2]},
    )

    assert data.moments_by_site == {1: (0.0, 0.0, 0.1), 2: (0.0, 0.0, -0.2)}

    with pytest.raises(ValueError, match="1, N, or 3N"):
        read_sprkkr_exchange(
            STRUCTURE_FILE,
            EXCHANGE_FILE,
            magnetic_species=["Ru"],
            moment=[0.1, 0.2, 0.3],
        )

    with pytest.raises(ValueError, match="Missing moment"):
        read_sprkkr_exchange(
            STRUCTURE_FILE,
            EXCHANGE_FILE,
            magnetic_species=["Ru"],
            moment={1: 0.1},
        )


def test_write_sprkkr_tb2j_results_writes_and_reloads_pickle(tmp_path):
    from TB2J.interfaces.sprkkr import write_sprkkr_tb2j_results

    output_path = tmp_path / "TB2J_results"

    spinio = write_sprkkr_tb2j_results(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        output_path=output_path,
        magnetic_species=["Ru"],
        moment=0.5674,
    )

    pickle_path = output_path / "TB2J.pickle"
    assert pickle_path.exists()
    assert (output_path / "exchange.out").exists()
    assert (output_path / "structure.vasp").exists()
    assert (output_path / "Multibinit" / "exchange.xml").exists()
    assert spinio.nspin == 2
    assert list(spinio.index_spin[:2]) == [0, 1]

    loaded = SpinIO.load_pickle(path=str(output_path))
    assert loaded.nspin == 2
    assert list(loaded.index_spin[:2]) == [0, 1]
    metadata = getattr(loaded, "sprkkr_metadata")
    assert metadata["tensor_policy"] == "transverse-block"
    assert metadata["source_files"]["structure"] == str(STRUCTURE_FILE)
    assert "SPRKKR reference-format files" in loaded.description


def test_sprkkr_to_spinio_builds_magnon_ready_bridge():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
    )
    spinio = sprkkr_to_spinio(data, tensor_policy="transverse-block")

    assert isinstance(spinio, SpinIO)
    assert spinio.nspin == 2
    assert list(spinio.index_spin[:2]) == [0, 1]
    assert list(spinio.index_spin[2:]) == [-1] * 12
    assert spinio.get_magnetic_moments().shape == (2, 3)
    assert spinio.get_magnetic_moments()[0, 2] == pytest.approx(0.5674)

    key = ((0, 0, 1), 0, 0)
    assert spinio.Jani_dict is not None
    jani_dict = spinio.Jani_dict
    assert key in jani_dict
    exchange_dict = spinio.exchange_Jdict
    assert exchange_dict is not None
    assert exchange_dict[key] == pytest.approx(0.0)
    assert jani_dict[key][0, 0] == pytest.approx(0.00155334513)
    assert jani_dict[key][0, 1] == pytest.approx(0.00481466537)

    magnon = Magnon.load_from_io(spinio, Jiso=True, Jani=True, DMI=False, SIA=False)
    assert magnon.nspin == 2
    assert magnon.JR.shape[-2:] == (3, 3)


def test_sprkkr_to_spinio_supports_isotropic_policy():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
    )
    spinio = sprkkr_to_spinio(data, tensor_policy="isotropic")

    key = ((0, 0, 1), 0, 0)
    assert spinio.Jani_dict is None
    exchange_dict = spinio.exchange_Jdict
    assert exchange_dict is not None
    assert exchange_dict[key] == pytest.approx(0.00155334513)


def test_sprkkr_to_spinio_supports_transverse_block_jzz_policy():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
    )
    spinio = sprkkr_to_spinio(data, tensor_policy="transverse-block-jzz")

    key = ((0, 0, 1), 0, 0)
    assert spinio.Jani_dict is not None
    tensor = spinio.Jani_dict[key]
    assert tensor[0, 0] == pytest.approx(0.00155334513)
    assert tensor[1, 1] == pytest.approx(0.00155334513)
    assert tensor[2, 2] == pytest.approx(0.00155334513)
    assert tensor[0, 2] == pytest.approx(0.0)
    assert tensor[2, 0] == pytest.approx(0.0)


def test_sprkkr_to_spinio_rejects_unsupported_tensor_policy():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio

    data = read_sprkkr_exchange(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
    )

    with pytest.raises(ValueError, match="tensor_policy"):
        sprkkr_to_spinio(data, tensor_policy="full-tensor")


def test_read_sprkkr_exchange_requires_moment():
    from TB2J.interfaces.sprkkr import read_sprkkr_exchange

    with pytest.raises(ValueError, match="moment is required"):
        read_sprkkr_exchange(
            STRUCTURE_FILE,
            EXCHANGE_FILE,
            magnetic_species=["Ru"],
        )


def test_magnon_from_sprkkr_returns_configured_magnon():
    from TB2J.interfaces.sprkkr import magnon_from_sprkkr

    magnon = magnon_from_sprkkr(
        STRUCTURE_FILE,
        EXCHANGE_FILE,
        magnetic_species=["Ru"],
        moment=0.5674,
        tensor_policy="isotropic",
    )

    labels, bands, _xlist = magnon.get_magnon_bands(path="GX", npoints=4)

    assert magnon.nspin == 2
    assert bands.shape == (4, 2)
    assert labels
