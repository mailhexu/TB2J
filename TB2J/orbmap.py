import re
from collections import defaultdict

import numpy as np


def split_orb_name(name):
    """
    split name to : n, l, label
    """
    m = re.findall(r"([a-z\d\-\^\*]*)(.*)", name)
    m = m[0]
    return m[0], m[1]


def map_orbs_matrix(orblist, spinor=False, include_only=None, group_by_zeta=False):
    """
    Build the orbital-grouping matrix.

    Parameters
    ----------
    orblist : list[str]
        Orbital labels (e.g. ``"3dz2Z1"``, ``"3dz2Z2"``, ``"4pxZ1P"``).
    spinor : bool
        If True, every orbital is duplicated for spin up/down; take only the
        even-indexed entries.
    include_only : list[str] | None
        If given, keep only orbitals whose ``(n,l)`` prefix matches an entry
        (e.g. ``["3d"]`` or ``["d"]``).
    group_by_zeta : bool
        If False (default), orbitals sharing the same ``(n,l,m)`` are summed
        over all zeta and polarisation shells — this is the historical
        behaviour.
        If True, each ``(n,l,m,zeta)`` combination becomes its own group so
        that per-zeta contributions are retained individually.

    Returns
    -------
    mmat : ndarray, shape (norb, ngroup)
        Integer grouping matrix.  ``mmat.T @ M @ mmat`` aggregates the
        full-orbital matrix *M* into the grouped representation.
    reduced_orbs : tuple[str]
        Group labels in column order.
    """

    if spinor:
        orblist = orblist[::2]

    norb = len(orblist)

    ss = [split_orb_name(orb) for orb in orblist]
    orbdict = dict(zip(ss, range(norb)))

    reduced_orbdict = defaultdict(lambda: [])

    if include_only is None:
        for key, val in orbdict.items():
            group_key = key if group_by_zeta else key[0]
            reduced_orbdict[group_key].append(val)
    else:
        for key, val in orbdict.items():
            if key[0][:2] in include_only or key[0][:1] in include_only:
                # [:2] for 3d, 4d, 5d, etc. and [:1] for s, p, d, etc
                group_key = key if group_by_zeta else key[0]
                reduced_orbdict[group_key].append(val)

    # When grouping by zeta, flatten tuple keys to readable strings
    if group_by_zeta:
        flat_orbdict = defaultdict(list)
        for key, val in reduced_orbdict.items():
            flat_orbdict[key[0] + key[1]].extend(val)
        reduced_orbdict = flat_orbdict

    reduced_orbs = tuple(reduced_orbdict.keys())
    ngroup = len(reduced_orbdict)
    mmat = np.zeros((norb, ngroup), dtype=int)

    for i, (key, val) in enumerate(reduced_orbdict.items()):
        for j in val:
            mmat[j, i] = 1
    return mmat, reduced_orbs


def test_split():
    split_orb_name("3sZ1")
    split_orb_name("3dxyZ1")
    split_orb_name("5dxyZ1")
    split_orb_name("5dx2-y2Z1P")


def test():
    odict = {
        0: [
            "3sZ1",
            "3sZ1",
            "4sZ1",
            "4sZ1",
            "4sZ2",
            "4sZ2",
            "3pyZ1",
            "3pyZ1",
            "3pzZ1",
            "3pzZ1",
            "3pxZ1",
            "3pxZ1",
            "3dxyZ1",
            "3dxyZ1",
            "3dyzZ1",
            "3dyzZ1",
            "3dz2Z1",
            "3dz2Z1",
            "3dxzZ1",
            "3dxzZ1",
            "3dx2-y2Z1",
            "3dx2-y2Z1",
            "3dxyZ2",
            "3dxyZ2",
            "3dyzZ2",
            "3dyzZ2",
            "3dz2Z2",
            "3dz2Z2",
            "3dxzZ2",
            "3dxzZ2",
            "3dx2-y2Z2",
            "3dx2-y2Z2",
            "4pyZ1P",
            "4pyZ1P",
            "4pzZ1P",
            "4pzZ1P",
            "4pxZ1P",
            "4pxZ1P",
        ],
        1: [
            "3sZ1",
            "3sZ1",
            "4sZ1",
            "4sZ1",
            "4sZ2",
            "4sZ2",
            "3pyZ1",
            "3pyZ1",
            "3pzZ1",
            "3pzZ1",
            "3pxZ1",
            "3pxZ1",
            "3dxyZ1",
            "3dxyZ1",
            "3dyzZ1",
            "3dyzZ1",
            "3dz2Z1",
            "3dz2Z1",
            "3dxzZ1",
            "3dxzZ1",
            "3dx2-y2Z1",
            "3dx2-y2Z1",
            "3dxyZ2",
            "3dxyZ2",
            "3dyzZ2",
            "3dyzZ2",
            "3dz2Z2",
            "3dz2Z2",
            "3dxzZ2",
            "3dxzZ2",
            "3dx2-y2Z2",
            "3dx2-y2Z2",
            "4pyZ1P",
            "4pyZ1P",
            "4pzZ1P",
            "4pzZ1P",
            "4pxZ1P",
            "4pxZ1P",
        ],
        2: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
        3: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
        4: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
        5: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
        6: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
        7: [
            "5sZ1",
            "5sZ1",
            "5sZ2",
            "5sZ2",
            "5pyZ1",
            "5pyZ1",
            "5pzZ1",
            "5pzZ1",
            "5pxZ1",
            "5pxZ1",
            "5pyZ2",
            "5pyZ2",
            "5pzZ2",
            "5pzZ2",
            "5pxZ2",
            "5pxZ2",
            "5dxyZ1P",
            "5dxyZ1P",
            "5dyzZ1P",
            "5dyzZ1P",
            "5dz2Z1P",
            "5dz2Z1P",
            "5dxzZ1P",
            "5dxzZ1P",
            "5dx2-y2Z1P",
            "5dx2-y2Z1P",
        ],
    }

    olist = odict[0]
    r1 = map_orbs_matrix(olist, spinor=True)
    print("Default (sum over zeta):", r1[1])
    assert len(r1[1]) == 13, f"Expected 13 reduced orbitals, got {len(r1[1])}"

    r2 = map_orbs_matrix(olist, spinor=True, include_only=["3d"])
    print("3d only (sum over zeta):", r2[1])
    assert len(r2[1]) == 5, f"Expected 5 d orbitals, got {len(r2[1])}"

    r3 = map_orbs_matrix(olist, spinor=True, group_by_zeta=True)
    print("Group by zeta:", r3[1])
    assert len(r3[1]) == 19, f"Expected 19 per-zeta groups, got {len(r3[1])}"
    # Verify that 3dz2Z1 and 3dz2Z2 are separate groups
    assert "3dz2Z1" in r3[1] and "3dz2Z2" in r3[1]

    r4 = map_orbs_matrix(olist, spinor=True, group_by_zeta=True, include_only=["3d"])
    print("3d only (group by zeta):", r4[1])
    assert len(r4[1]) == 10, f"Expected 10 per-zeta d groups, got {len(r4[1])}"

    # Sanity: summing per-zeta columns should reproduce zeta-summed result
    mmat_sum, orbs_sum = r1
    mmat_zeta, orbs_zeta = r3
    # Build a summation matrix from zeta → summed
    rebuild = np.zeros((len(orbs_zeta), len(orbs_sum)), dtype=int)
    for iz, oz in enumerate(orbs_zeta):
        prefix = split_orb_name(oz)[0]
        js = [j for j, os in enumerate(orbs_sum) if os == prefix]
        for j in js:
            rebuild[iz, j] = 1
    mmat_combined = mmat_zeta @ rebuild
    assert np.array_equal(
        mmat_combined, mmat_sum
    ), "Per-zeta groups don't sum to zeta-summed groups"
    print("All tests passed.")


if __name__ == "__main__":
    test()
