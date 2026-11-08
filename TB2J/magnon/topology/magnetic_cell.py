"""Commensurate real-space expansion for single-Q magnetic references."""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from TB2J.magnon.topology.preparation import (
    TopologyModel,
    TopologyPreparationError,
    _digest_model,
)

__all__ = ["expand_magnetic_cell"]


def _commensurate_component(
    q: float, max_denominator: int, tolerance: float
) -> Fraction:
    fraction = Fraction(float(q)).limit_denominator(max_denominator)
    if abs(float(fraction) - q) > tolerance:
        raise TopologyPreparationError(
            f"Ordering vector component {q:g} is incommensurate within "
            f"denominator limit {max_denominator}; increase max_denominator "
            "only when a larger real-space cell is acceptable."
        )
    return fraction


def expand_magnetic_cell(
    model: TopologyModel,
    Q,
    axis=(0.0, 0.0, 1.0),
    *,
    max_denominator: int = 64,
    max_sites: int = 4096,
    tolerance: float = 1e-10,
) -> TopologyModel:
    """Expand a topology model and twist moments into a Q-commensurate cell.

    ``Q`` is expressed in reciprocal coordinates of the input cell. The
    resulting model has Q=0: the ordering is represented explicitly by its
    real-space moments, so callers must not apply the spiral rotation again.
    """
    if not isinstance(model, TopologyModel):
        raise TypeError("expand_magnetic_cell requires a TopologyModel.")
    q = np.asarray(Q, dtype=float)
    u = np.asarray(axis, dtype=float)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("Q must be a finite three-component vector.")
    if u.shape != (3,) or not np.isfinite(u).all() or np.linalg.norm(u) == 0:
        raise ValueError("axis must be a finite nonzero three-component vector.")
    u /= np.linalg.norm(u)
    if model.magnetic_cell_transform is not None and np.any(q != 0.0):
        raise TopologyPreparationError(
            "This model already contains an explicit magnetic-cell transform; "
            "additional Q rotation would apply the ordering twice."
        )
    fractions = [_commensurate_component(x, max_denominator, tolerance) for x in q]
    q_commensurate = np.array([float(x) for x in fractions])
    periods = np.array([x.denominator for x in fractions], dtype=int)
    images = list(product(*(range(int(m)) for m in periods)))
    nspin = model.nspin * len(images)
    if nspin > max_sites:
        raise TopologyPreparationError(
            f"Commensurate expansion requires {nspin} spins, above max_sites={max_sites}."
        )
    new_cell = periods[:, None] * model.cell
    positions = []
    moments = []
    for image in images:
        for i in range(model.nspin):
            coordinate = model.positions_fractional[i] + np.asarray(image)
            positions.append(coordinate @ model.cell)
            angle = 2.0 * np.pi * float(np.dot(q_commensurate, coordinate))
            moments.append(Rotation.from_rotvec(angle * u).apply(model.magmoms[i]))
    positions = np.asarray(positions)
    moments = np.asarray(moments)
    fractional = positions @ np.linalg.inv(new_cell)

    lookup = {
        (image, i): image_index * model.nspin + i
        for image_index, image in enumerate(images)
        for i in range(model.nspin)
    }
    bond_map = {}
    for r_index, R in enumerate(model.Rlist):
        for image in images:
            target = np.asarray(image) + R
            target_image = tuple((target % periods).tolist())
            new_R = tuple(((target - np.asarray(target_image)) // periods).tolist())
            for i in range(model.nspin):
                ii = lookup[(image, i)]
                for j in range(model.nspin):
                    jj = lookup[(target_image, j)]
                    key = (new_R, ii, jj)
                    bond_map[key] = (
                        bond_map.get(key, np.zeros((3, 3))) + model.JR[r_index, i, j]
                    )
    Rlist = np.asarray(
        sorted({key[0] for key in bond_map} | {(0, 0, 0)}), dtype=np.int64
    )
    r_index = {tuple(R): k for k, R in enumerate(Rlist)}
    JR = np.zeros((len(Rlist), nspin, nspin, 3, 3), dtype=float)
    for (R, i, j), tensor in bond_map.items():
        JR[r_index[R], i, j] = tensor

    # Repeat all atom mappings, including nonmagnetic atoms, per image.
    image_map = [
        {
            "new_spin": image_index * model.nspin + i,
            "source_spin": i,
            "source_atom": int(model.spin_atoms[i]),
            "image": [int(x) for x in image],
        }
        for image_index, image in enumerate(images)
        for i in range(model.nspin)
    ]
    transformation = {
        "source_model_digest": model.digest,
        "supercell_matrix": np.diag(periods).tolist(),
        "ordering_vector_input_fractional": q.tolist(),
        "ordering_vector_commensurate_input_fractional": q_commensurate.tolist(),
        "ordering_vector_rational": [
            f"{x.numerator}/{x.denominator}" for x in fractions
        ],
        "commensurability_error_max": float(np.max(np.abs(q - q_commensurate))),
        "rotation_axis_cartesian": u.tolist(),
        "image_map": image_map,
        "folded_mode_count": int(nspin),
        "additional_ordering_vector": [0.0, 0.0, 0.0],
    }
    index_spin = np.tile(model.index_spin, len(images)).astype(np.int64)
    for image_index in range(len(images)):
        start = image_index * model.natom
        block = index_spin[start : start + model.natom]
        block[block >= 0] += image_index * model.nspin
    spin_atoms = np.concatenate(
        [
            model.spin_atoms + image_index * model.natom
            for image_index in range(len(images))
        ]
    )
    digest = _digest_model(
        new_cell, positions, index_spin, spin_atoms, moments, Rlist, JR
    )
    return TopologyModel(
        cell=new_cell,
        positions=positions,
        positions_fractional=fractional,
        index_spin=index_spin,
        spin_atoms=spin_atoms,
        magmoms=moments,
        Rlist=Rlist,
        JR=JR,
        digest=digest,
        selection=dict(model.selection),
        exclusions=tuple(model.exclusions),
        unsupported_nonzero=dict(model.unsupported_nonzero),
        auxiliary_present=tuple(model.auxiliary_present),
        source_path=model.source_path,
        canonicalization_version=model.canonicalization_version,
        magnetic_cell_transform=transformation,
    )
