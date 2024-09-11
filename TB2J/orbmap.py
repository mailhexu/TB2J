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


def map_orbs_matrix(orblist, spinor=False, include_only=None):
    """
    map the orbitals to a matrix
    Method:
    1. split the orbital name to n, l, label
    2. group the orbitals by n, l
    3. create a matrix with 1 for each orbital in the group
    4. return the matrix and the group names
    """

    if spinor:
        orblist = orblist[::2]

    norb = len(orblist)

    # print("orblist: ", orblist)
    ss = [split_orb_name(orb) for orb in orblist]
    orbdict = dict(zip(ss, range(norb)))

    reduced_orbdict = defaultdict(lambda: [])

    # print(f"Orbital dictionary: {orbdict}")
    # print("include_only: ", include_only)

    if include_only is None:
        for key, val in orbdict.items():
            reduced_orbdict[key[0]].append(val)
    else:
        for key, val in orbdict.items():
            if key[0][:2] in include_only or key[0][:1] in include_only:
                # [:2] for 3d, 4d, 5d, etc. and [:1] for s, p, d, etc
                reduced_orbdict[key[0]].append(val)

    # print(f"reduced_orbdict: {reduced_orbdict}")
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
    print(r1)
    r2 = map_orbs_matrix(olist, spinor=True, include_only=["3d"])
    print(r2)


if __name__ == "__main__":
    test()
