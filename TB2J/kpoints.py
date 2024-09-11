from collections import Counter

import numpy as np
import spglib


def monkhorst_pack(size, gamma_center=False):
    """Construct a uniform sampling of k-space of given size.
    Modified from ase.dft.kpoints with gamma_center option added"""
    if np.less_equal(size, 0).any():
        raise ValueError("Illegal size: %s" % list(size))
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    asize = np.array(size)
    shift = 0.5 * ((asize + 1) % 2) / asize
    mkpts = (kpts + 0.5) / size - 0.5
    if gamma_center:
        mkpts += shift
    return mkpts


def get_ir_kpts(atoms, mesh):
    """
    Gamma-centered IR kpoints. mesh : [nk1,nk2,nk3].
    """
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    cell = (lattice, positions, numbers)
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
    return grid[np.unique(mapping)] / np.array(mesh, dtype=float)


def ir_kpts(
    atoms, mp_grid, is_shift=[0, 0, 0], verbose=True, ir=True, is_time_reversal=False
):
    """
    generate kpoints for structure
    Parameters:
    ------------------
    atoms: ase.Atoms
      structure
    mp_grid: [nk1,nk2,nk3]
    is_shift: shift of k points. default is Gamma centered.
    ir: bool
    Irreducible or not.
    """
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    # print(spglib.get_spacegroup(cell, symprec=1e-5))
    mesh = mp_grid
    # Gamma centre mesh
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        mesh, cell, is_shift=is_shift, is_time_reversal=is_time_reversal, symprec=1e-4
    )
    if not ir:
        return (np.array(grid).astype(float) + np.asarray(is_shift) / 2.0) / mesh, [
            1.0 / len(mapping)
        ] * len(mapping)
    # All k-points and mapping to ir-grid points
    # for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
    #    print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh))
    cnt = Counter(mapping)
    ids = list(cnt.keys())
    weight = list(cnt.values())
    weight = np.array(weight) * 1.0 / sum(weight)
    ird_kpts = [
        (grid[id].astype(float) + np.asarray(is_shift) / 2.0) / mesh for id in ids
    ]

    # Irreducible k-points
    # print("Number of ir-kpoints: %d" % len(np.unique(mapping)))
    # print(grid[np.unique(mapping)] / np.array(mesh, dtype=float))

    new_ir_kpts = []
    new_weight = []
    for ik, k in enumerate(ird_kpts):
        # add k and -k to ird_kpts if -k is not in ird_kpts
        if not any([np.allclose(-1.0 * k, kpt) for kpt in new_ir_kpts]):
            new_ir_kpts.append(k)
            new_ir_kpts.append(-1.0 * k)
            new_weight.append(weight[ik] / 2)
            new_weight.append(weight[ik] / 2)
        else:
            new_ir_kpts.append(k)
            new_weight.append(weight[ik])
    # return ird_kpts, weight
    return np.array(new_ir_kpts), np.array(new_weight)


def test_ir_kpts():
    from ase.build import bulk

    atoms = bulk("Si")
    mesh = [14, 14, 14]
    kpts = get_ir_kpts(atoms, mesh)
    print(kpts)
    print(len(kpts))


if __name__ == "__main__":
    test_ir_kpts()
