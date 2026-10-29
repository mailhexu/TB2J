"""Standard results remain honest when physical fields are unavailable."""

import numpy as np
import pytest
from ase import Atoms

from TB2J.io_exchange import SpinIO


def _single_spin(*, moments=None, charges=None):
    key = ((1, 0, 0), 0, 0)
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)
    return SpinIO(
        atoms=atoms,
        spinat=moments,
        charges=charges,
        index_spin=[0],
        exchange_Jdict={key: 0.002},
        distance_dict={key: (np.array([3.0, 0.0, 0.0]), 3.0)},
    )


def test_write_all_keeps_missing_moments_and_charges_honest(tmp_path):
    model = _single_spin()
    results = tmp_path / "TB2J_results"
    report = model.write_all(path=results)
    loaded = SpinIO.load_pickle(path=results)
    assert loaded.spinat is None and loaded.charges is None
    assert loaded.exchange_Jdict == model.exchange_Jdict
    text = (results / "exchange.out").read_text()
    assert "unavailable" in text.lower()
    assert (results / "structure.vasp").is_file()
    assert (results / "JvsR.pdf").is_file()
    assert {name for name, _ in report.omitted} >= {
        "Multibinit",
        "TomASD",
        "Vampire",
        "ESPInS",
    }
    assert not (results / "TomASD").exists()


def test_write_all_with_moments_does_not_require_charges(tmp_path):
    model = _single_spin(moments=np.array([[0.0, 0.0, 2.0]]))
    report = model.write_all(path=tmp_path / "results")
    assert not report.omitted
    assert {"Multibinit", "TomASD", "Vampire", "ESPInS"} <= set(report.written)
    assert SpinIO.load_pickle(path=tmp_path / "results").charges is None
    assert "unavailable" in (tmp_path / "results" / "exchange.out").read_text().lower()
    np.testing.assert_array_equal(model.spinat, [[0.0, 0.0, 2.0]])


def test_zero_magnetic_moment_omits_solver_outputs(tmp_path):
    model = _single_spin(moments=np.zeros((1, 3)), charges=np.array([8.0]))
    report = model.write_all(path=tmp_path / "results")
    assert {target for target, reason in report.omitted if "zero" in reason} == {
        "Multibinit",
        "TomASD",
        "Vampire",
        "ESPInS",
    }
    assert not (tmp_path / "results" / "Vampire").exists()
    np.testing.assert_array_equal(model.spinat, [[0.0, 0.0, 0.0]])


def test_invalid_context_is_rejected_before_output_creation(tmp_path):
    model = _single_spin(charges=np.array([8.0, 9.0]))
    with pytest.raises(ValueError, match="charges"):
        model.write_all(path=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_pickle_write_error_is_not_suppressed(tmp_path):
    model = _single_spin()
    model.debug_dict = {"unpicklable": lambda: None}
    with pytest.raises((AttributeError, TypeError)):
        model.write_all(path=tmp_path / "results")


def test_sparse_dmi_only_is_reported_without_inventing_isotropic_exchange(tmp_path):
    model = _single_spin()
    key = next(iter(model.exchange_Jdict))
    model.exchange_Jdict = None
    model.dmi_ddict = {key: np.array([0.001, 0.002, 0.003])}
    report = model.write_all(path=tmp_path / "results")
    output = SpinIO.load_pickle(path=tmp_path / "results")
    assert output.exchange_Jdict is None
    np.testing.assert_array_equal(output.dmi_ddict[key], model.dmi_ddict[key])
    text = (tmp_path / "results" / "exchange.out").read_text()
    assert "J_iso: unavailable" in text and "DMI:" in text
    assert "JvsR.pdf" in report.written


def test_missing_distance_is_derived_for_writing_without_mutating_native_model(
    tmp_path,
):
    model = _single_spin()
    model.distance_dict = None
    model.write_all(path=tmp_path / "results")
    assert model.distance_dict is None
    assert SpinIO.load_pickle(path=tmp_path / "results").distance_dict is None
    text = (tmp_path / "results" / "exchange.out").read_text()
    assert "3.000" in text and "J_iso:" in text


def test_list_moments_are_not_mutated_by_writer(tmp_path):
    model = _single_spin(moments=[[0.0, 0.0, 2.0]], charges=[8.0])
    report = model.write_all(path=tmp_path / "results")
    assert "TomASD" in report.written
    assert model.spinat == [[0.0, 0.0, 2.0]]
    assert SpinIO.load_pickle(path=tmp_path / "results").spinat == model.spinat


def test_edit_save_does_not_modify_omitted_multibinit_file(tmp_path):
    from TB2J.io_exchange.edit import save

    model = _single_spin()
    model.sia_tensor = {0: np.eye(3)}
    model.has_sia_tensor = True
    target = tmp_path / "results" / "Multibinit"
    target.mkdir(parents=True)
    existing = target / "mb.in"
    existing.write_text("spin_sia_add = 1\n")
    save(model, path=tmp_path / "results")
    assert existing.read_text() == "spin_sia_add = 1\n"


def test_sparse_without_isotropic_section_omits_unreadable_multibinit(tmp_path):
    model = _single_spin(moments=np.array([[0.0, 0.0, 2.0]]))
    key = next(iter(model.exchange_Jdict))
    model.exchange_Jdict = None
    model.dmi_ddict = {key: np.array([0.001, 0.002, 0.003])}
    report = model.write_all(tmp_path / "results")
    assert "isotropic" in dict(report.omitted)["Multibinit"]
    assert not (tmp_path / "results" / "Multibinit").exists()
    assert SpinIO.load_pickle(tmp_path / "results").exchange_Jdict is None


def test_legacy_anisotropy_omits_unsupported_espins(tmp_path):
    from TB2J.io_exchange.edit import set_anisotropy

    model = _single_spin(moments=np.array([[0.0, 0.0, 2.0]]))
    set_anisotropy(model, "Fe", k1=0.001, k1dir=[0, 0, 1])
    report = model.write_all(tmp_path / "results")
    assert "anisotropy" in dict(report.omitted)["ESPInS"]
    assert not (tmp_path / "results" / "ESPInS").exists()
    assert model.k1 == [0.001]


def test_unwritable_structure_is_rejected_before_pickle_publication(tmp_path):
    from ase.constraints import FixedLine

    model = _single_spin()
    model.atoms.set_constraint(FixedLine(0, [1, 1, 1]))
    output = tmp_path / "results"
    output.mkdir()
    pickle = output / "TB2J.pickle"
    pickle.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="parallel"):
        model.write_all(output)
    assert pickle.read_bytes() == b"existing"


def test_reloaded_model_with_missing_distance_can_plot_exchange(tmp_path):
    model = _single_spin()
    model.distance_dict = None
    model.write_all(tmp_path / "results")
    loaded = SpinIO.load_pickle(tmp_path / "results")
    ax = loaded.plot_JvsR()
    points = ax.collections[0].get_offsets()
    np.testing.assert_allclose(points, [[3.0, 2.0]])
    assert loaded.distance_dict is None
