import json

import numpy as np
import pytest
from tbupy.spiral_state import save_spiral_state

from TB2J.spiral_green import (
    SpiralGreen,
    SpiralState,
    load_spiral_state,
    spectral_density,
)


def _state():
    rho = np.diag([0.4, 0.6]).astype(complex)
    metadata = {
        "schema_version": 2,
        "gauge": "planar_y",
        "q_frac": [0.2, 0, 0],
        "electron_count": 1.0,
        "occupation_rule": "fixed",
        "constraint": {
            "kind": "spinor_moment_component",
            "sites": [
                {
                    "operator": "sigma_y(atom0)",
                    "type": "spinor_moment_component",
                    "target": 0.0,
                    "multiplier": 0.1,
                    "rotation": "lab_fixed",
                    "hold_fixed": True,
                }
            ],
        },
        "field_symmetry": {"local_field_axis": "y"},
        "field_role": "external",
        "field_rotation_policy": "co_rotating",
        "width": 0.05,
        "hubbard": {"hubbard_dict": {}, "hubbard_type": "dudarev", "dc_type": "FLL-ns"},
        "density_tolerance": 1e-10,
    }
    return SpiralState(
        HR_up=np.zeros((1, 1, 1)),
        HR_dn=np.zeros((1, 1, 1)),
        SR=np.ones((1, 1, 1)),
        Rlist=np.zeros((1, 3), int),
        q_frac=np.array([0.2, 0, 0]),
        taus=np.zeros((1, 3)),
        phis=np.zeros(1),
        B_local=np.zeros(1),
        rho=rho,
        V_U=np.zeros((2, 2)),
        metadata_json=json.dumps(metadata),
        schema_version=2,
        kpts=np.zeros((1, 3)),
        kweights=np.ones(1),
        occupations=np.array([[0.4, 0.6]]),
        orbital_to_atom=np.array([0]),
        constraint_potential=np.array([[0, 1j], [-1j, 0]], complex),
        evals=np.array([[0, 1]]),
        evecs=np.eye(2, dtype=complex)[None, :, :],
        field_symmetry=metadata["field_symmetry"],
    )


def test_tbupy_netcdf_roundtrip_into_tb2j_mirror(tmp_path):
    source = _state()
    path = tmp_path / "v2.nc"
    save_spiral_state(path, source)
    loaded = load_spiral_state(path)
    assert loaded.schema_version == 2
    np.testing.assert_allclose(loaded.constraint_potential, source.constraint_potential)
    np.testing.assert_allclose(loaded.occupations, source.occupations)
    np.testing.assert_allclose(spectral_density(loaded), loaded.rho)


def test_mirror_rejects_density_mismatch_and_missing_gauge():
    state = _state()
    state.rho[0, 0] += 0.2
    with pytest.raises(ValueError, match="spectral density mismatch"):
        state.validate()
    state = _state()
    metadata = state.metadata
    del metadata["gauge"]
    state.metadata_json = json.dumps(metadata)
    with pytest.raises(ValueError, match="missing 'gauge'"):
        state.validate()


def test_v2_mirror_rejects_missing_smearing_protocol():
    state = _state()
    meta = state.metadata
    del meta["width"]
    state.metadata_json = json.dumps(meta)
    with pytest.raises(ValueError, match="width"):
        state.validate()


def test_unconstrained_tag_refuses_nonzero_constraint_operator():
    state = _state()
    meta = state.metadata
    meta["constraint"] = {"kind": "none"}
    state.metadata_json = json.dumps(meta)
    with pytest.raises(ValueError, match="constraint_potential"):
        state.validate()


def test_hard_rotated_density_is_never_a_spectral_projector():
    state = _state()
    metadata = state.metadata
    metadata["hard_rotation_applied"] = True
    state.metadata_json = json.dumps(metadata)
    with pytest.raises(ValueError, match="hard-rotated rho"):
        state.validate()


def test_v2_green_uses_persisted_occupations_and_kmesh():
    state = _state()
    state.constraint_potential[:] = 0
    green = SpiralGreen(
        state, state.kpts, assembler=lambda _state, _k: (np.diag([0.0, 1.0]), np.eye(2))
    )
    np.testing.assert_array_equal(green.frozen_occupations(), state.occupations)
    with pytest.raises(ValueError, match="exactly match persisted kpts"):
        SpiralGreen(
            state,
            np.array([[0.1, 0.0, 0.0]]),
            assembler=lambda _state, _k: (np.diag([0.0, 1.0]), np.eye(2)),
        )
