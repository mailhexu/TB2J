"""Consumer regressions for the hand-first shell exchange format."""

from dataclasses import replace
from datetime import date, time

import numpy as np
import pytest

from TB2J.io_exchange import shells
from TB2J.io_exchange.io_exchange import SpinIO

HAND_FILE = '''exchange = "isotropic"
[structure]
coordinates = "fractional"
lattice = """
4 0 0
0.3 5 0
0.2 0.4 6
"""
atoms = """
1 Fe7 -0.87 0.21 0.17 Y 1
2 O2 0.27 1.11 0.41 N -2
3 Co4 1.39 0.36 0.23 Y 0
"""
[reference_spins]
moments = """
Co4 -1
Fe7 2
O2 0.03
"""
[symmetry]
mode = "spacegroup"
[[shell]]
representative = "Fe7 Co4 1 0 -1"
J = 2
[metadata]
date = 2026-10-25
time = 10:33:00
[metadata.sample]
tags = ["raw", "editable"]
'''


def test_hand_file_raw_frame_annotations_and_generated_summaries(tmp_path):
    path = tmp_path / "hand.shells.toml"
    path.write_text(HAND_FILE)
    document = shells.read_shells(path)
    assert document.site_labels == ("Fe7", "O2", "Co4")
    assert document.index_spin == (1, -2, 0)
    assert document.moments == (2.0, 0.03, -1.0)
    assert document.shells[0].name is None
    assert document.metadata["date"] == date(2026, 10, 25)
    assert document.metadata["time"] == time(10, 33)
    with pytest.raises(TypeError):
        document.metadata["sample"]["tags"][0] = "changed"
    shells.write_shells(document, path, equivalent_comments=False)
    reloaded = shells.read_shells(path)
    shell = reloaded.shells[0]
    vector = (
        np.array([1.39, 0.36, 0.23]) + [1, 0, -1] - [-0.87, 0.21, 0.17]
    ) @ np.array(document.cell_angstrom)
    assert shell.distance == pytest.approx(np.linalg.norm(vector))
    assert shell.n_equivalent_pairs == 1 and shell.name is None
    output = shells.expand(reloaded).spinio
    key = ((1, 0, -1), 1, 0)
    assert output.exchange_Jdict == {key: 0.002}
    np.testing.assert_allclose(output.distance_dict[key][0], vector, atol=1e-14)
    output.write_pickle(path=tmp_path)
    native = SpinIO.load_pickle(path=tmp_path)
    restored = shells.compress(native, mode="isotropic").document
    assert restored.site_labels == reloaded.site_labels
    assert restored.metadata == reloaded.metadata
    np.testing.assert_allclose(restored.positions_frac, reloaded.positions_frac)


@pytest.mark.parametrize(
    "before,after,field",
    [
        ("2 O2 0.27 1.11 0.41 N -2", "2 O2 0.27 1.11 0.41 N -", None),
        ("2 O2 0.27 1.11 0.41 N -2", "2 O2 0.27 1.11 0.41 N 0", "structure.atoms"),
        ("3 Co4", "4 Co4", "structure.atoms"),
        ("3 Co4", "3 Fe7", "structure.atoms"),
        ("-0.87", "nan", "structure.atoms"),
        ("-0.87", "1_000", "structure.atoms"),
        ("Fe7 Co4 1 0 -1", "Fe7 Missing 1 0 -1", "representative"),
        ("Co4 -1", "Co4 -1 0 0", "moments"),
        ("O2 0.03\n", "", "moments"),
        ("Co4 -1", "Fe7 -1", "moments"),
        ("J = 2", "J = 2\ndistance = 1", "distance"),
        ("J = 2", "J = 2\nn_equivalent_pairs = 2", "n_equivalent_pairs"),
        ("J = 2", "J = 2\nsupport_members = [0]", "snapshot"),
        ('exchange = "isotropic"', 'exchange_mode = "isotropic"', "exchange"),
        ("J = 2", "Jiso = 2", "Jiso"),
        ("J = 2", "J = 2\norbit_size = 1", "orbit_size"),
        ("date = 2026-10-25", "date = nan", "metadata"),
    ],
)
def test_hand_authoring_boundaries(tmp_path, before, after, field):
    path = tmp_path / "hand.shells.toml"
    path.write_text(HAND_FILE.replace(before, after))
    if field is None:
        assert shells.read_shells(path).index_spin == (1, -1, 0)
    else:
        with pytest.raises(ValueError, match=field):
            shells.read_shells(path)


def test_tensor_name_metadata_and_snapshot_replay(tmp_path):
    path = tmp_path / "tensor.shells.toml"
    path.write_text(
        HAND_FILE.replace('exchange = "isotropic"', 'exchange = "tensor"').replace(
            "J = 2",
            'name = "nearest interspecies"\nJ = [[2, 0.4, 0.3], [-0.4, 1, 0.2], [0.3, 0.2, 3]]',
        )
    )
    document = shells.read_shells(path)
    alias = {"sample": {"tags": ["frozen"]}}
    document = replace(document, metadata=alias)
    alias["sample"]["tags"][0] = "changed"
    alias["sample"]["new"] = 3
    assert document.metadata["sample"] == {"tags": ("frozen",)}
    shells.write_shells(document, path)
    saved = shells.read_shells(path)
    assert saved.operations == document.operations
    assert saved.shells[0].name == "nearest interspecies"
    np.testing.assert_array_equal(saved.shells[0].tensor, document.shells[0].tensor)
    result = shells.expand(saved)
    assert result.spinio.index_spin == [1, -2, 0]
    base = result.spinio
    base._shell_user_metadata = {"stale": True}
    base._shell_site_labels = ["Fe1", "O1", "Co1"]
    edited = replace(saved, metadata={})
    replay = shells.expand(edited, base=base).spinio
    assert replay._shell_user_metadata == {}
    assert replay._shell_site_labels == ["Fe7", "O2", "Co4"]
    for attribute, value in [
        ("_shell_user_metadata", {"bad": float("inf")}),
        ("_shell_site_labels", ["Fe7", "O2", "Co4", "Fe8"]),
        ("_shell_site_labels", None),
        ("_shell_site_labels", 7),
    ]:
        native = shells.expand(saved).spinio
        setattr(native, attribute, value)
        with pytest.raises(ValueError):
            shells.compress(native)


@pytest.mark.parametrize("mode", ["isotropic", "tensor"])
@pytest.mark.parametrize("symmetry", ["spacegroup", "magnetic"])
def test_snapshot_free_cubic_orbit_pins_full_count_and_replays(
    tmp_path, mode, symmetry
):
    path = tmp_path / "cubic.shells.toml"
    path.write_text(f'''exchange = "{mode}"
[structure]
coordinates = "fractional"
lattice = """
2 0 0
0 2 0
0 0 2
"""
atoms = "1 Fe1 1 0 -1 Y 0"
[reference_spins]
moments = "Fe1 0 0 2"
[symmetry]
mode = "{symmetry}"
[[shell]]
representative = "Fe1 Fe1 0 0 1"
J = "unfilled"
''')
    document = shells.read_shells(path)
    shells.write_shells(document, path, equivalent_comments=False)
    canonical = shells.read_shells(path)
    assert canonical.shells[0].distance == 2.0
    assert canonical.shells[0].n_equivalent_pairs == (
        6 if symmetry == "spacegroup" else 2
    )
    assert canonical.shells[0].name is None
    assert canonical.operations == document.operations
    assert canonical.positions_frac == ((1.0, 0.0, -1.0),)
