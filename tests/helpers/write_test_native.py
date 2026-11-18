"""Test fixture: generate a minimal tb2j_native.bin for testing the reader."""

from pathlib import Path

import numpy as np


def write_test_native(path):
    """Write a minimal synthetic tb2j_native.bin for testing."""
    path = Path(path)
    nspin = 2
    nkpt = 3
    nband = 4
    nprod = 6  # 2 atoms × 3 channels (1 s + 1 p = 1 + 3 = 4... let's do 3 per atom)
    nions = 2
    ntyp = 1
    lmdim_max = 3

    with open(path, "wb") as f:
        # Header
        f.write(np.array(20260812, dtype="<i4").tobytes())
        f.write(np.array(1, dtype="<i4").tobytes())
        # Dimensions
        for v in [nspin, nkpt, nband, nprod, nions, ntyp, lmdim_max]:
            f.write(np.array(v, dtype="<i4").tobytes())
        # Lattice
        lattice = np.eye(3, dtype="<f8") * 5.0  # 5 Bohr cubic
        f.write(lattice.tobytes())
        # Positions
        posion = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype="<f8").T
        f.write(posion.T.copy().tobytes())
        # Ion offsets and nproj
        f.write(np.array([0, 3], dtype="<i4").tobytes())  # offsets
        f.write(np.array([3, 3], dtype="<i4").tobytes())  # nproj per ion
        f.write(np.array([1, 1], dtype="<i4").tobytes())  # ityp
        f.write(np.array([3], dtype="<i4").tobytes())  # lmmaxc per type
        # K-points
        vkpt = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype="<f8"
        ).T
        f.write(vkpt.T.copy().tobytes())
        f.write(np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype="<f8").tobytes())
        # Fermi
        f.write(np.array(5.0, dtype="<f8").tobytes())
        # Eigenvalues and occupations
        celtot = np.arange(nband * nkpt * nspin, dtype="<f8").reshape(
            nband, nkpt, nspin
        )
        f.write(celtot.tobytes())
        fertot = (
            np.ones(nband * nkpt * nspin, dtype="<f8").reshape(nband, nkpt, nspin) * 0.5
        )
        f.write(fertot.tobytes())
        # CPROJ
        cproj = np.ones(nprod * nband * nkpt * nspin, dtype="<c16").reshape(
            nprod, nband, nkpt, nspin
        )
        f.write(cproj.tobytes())
        # CDIJ
        cdij = np.eye(lmdim_max, dtype="<c16").reshape(lmdim_max, lmdim_max, 1, 1)
        cdij_full = np.tile(cdij, (1, 1, nions, nspin))
        f.write(cdij_full.tobytes())

    return path


if __name__ == "__main__":
    import sys

    path = write_test_native(
        sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_tb2j_native.bin"
    )
    print(f"Wrote test native binary to {path}")
