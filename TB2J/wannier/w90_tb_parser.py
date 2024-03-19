"""
parse the tb files in Wannier90
"""

import os
import re
import numpy as np
from typing import List, Tuple, Dict


def ssrline(file):
    """
    Read a line and split it into a list of strings"""
    line = file.readline().strip()
    return re.split(r"\s+", line)


def parse_vector(file, vector_type, size):
    result = np.zeros((size,), dtype=vector_type)
    for i in range(size):
        result[i] = np.fromstring(file.readline().strip(), sep=" ")
        result[i] = result[i].astype(vector_type)
    return result


def find_Rvectors(Rvectors, R):
    for i, Rvec in enumerate(Rvectors):
        if np.all(Rvec == R):
            return i
    return -1


def parse_tb_file(filename):
    with open(filename, "r") as io:
        header = io.readline().strip()
        a1 = np.fromstring(io.readline().strip(), sep=" ")
        a2 = np.fromstring(io.readline().strip(), sep=" ")
        a3 = np.fromstring(io.readline().strip(), sep=" ")
        # a1, a2, a3 = np.array(list(map(float, [a1, a2, a3])), dtype=np.float64)
        # lattice = np.vstack((a1.reshape(-1, 1), a2.reshape(-1, 1), a3.reshape(-1, 1))).T
        lattice = np.vstack((a1, a2, a3))

        n_wann = int(ssrline(io)[0])
        n_Rvecs = int(ssrline(io)[0])

        Rdegens = []
        line = io.readline().strip()
        while line != "":
            Rdegens += [int(x) for x in line.split()]
            line = io.readline().strip()
        assert len(Rdegens) == n_Rvecs, "Rdegens length does not match n_Rvecs"

        Rvectors = np.zeros((n_Rvecs, 3), dtype=np.int32)
        # H = np.zeros((n_Rvecs,n_wann, n_wann), dtype=np.complex128)
        H = {}
        r_x = np.zeros((n_Rvecs, n_wann, n_wann), dtype=np.complex128)
        r_y = np.zeros((n_Rvecs, n_wann, n_wann), dtype=np.complex128)
        r_z = np.zeros((n_Rvecs, n_wann, n_wann), dtype=np.complex128)
        pos_operator = np.zeros((n_Rvecs, n_wann, n_wann, 3), dtype=np.complex128)

        for iR in range(n_Rvecs):
            # read Rvectors
            Rvectors[iR] = np.fromstring(io.readline().strip(), sep=" ")
            Rvectors[iR] = Rvectors[iR].astype(np.int32)
            R = tuple(Rvectors[iR])
            H[R] = np.zeros((n_wann, n_wann), dtype=np.complex128)
            # read H
            for n in range(n_wann):
                for m in range(n_wann):
                    line = ssrline(io)
                    assert (
                        m == int(line[0]) - 1 and n == int(line[1]) - 1
                    ), "Unexpected indices"
                    reH, imH = float(line[2]), float(line[3])
                    # H[iR][m, n] = reH + 1j * imH
                    H[R][m, n] = (reH + 1j * imH) / 2.0
            io.readline()  # empty line
        # set the onsite term to half
        # np.fill_diagonal(H[(0, 0, 0)], H[(0, 0, 0)].diagonal() / 2.0)
        # print("onsite H from TB: ", H[(0, 0, 0)].diagonal())

        iR0 = find_Rvectors(Rvectors, [0, 0, 0])
        for iR in range(n_Rvecs):
            # read Rvectors
            line = io.readline().strip()
            Rvectors[iR] = np.fromstring(line, sep=" ")
            Rvectors[iR] = Rvectors[iR].astype(np.int32)
            # read r_x, r_y, r_z
            for n in range(n_wann):
                for m in range(n_wann):
                    line = ssrline(io)
                    assert (
                        m == int(line[0]) - 1 and n == int(line[1]) - 1
                    ), f"Unexpected indices in line {line}"
                    pos_operator[iR, m, n, 0] = complex(float(line[2]), float(line[3]))
                    pos_operator[iR, m, n, 1] = complex(float(line[4]), float(line[5]))
                    pos_operator[iR, m, n, 2] = complex(float(line[6]), float(line[7]))
            io.readline()

        # print(
        #    f"Reading tb.dat file: {filename} | Header: {header} | n_wann: {n_wann} | n_Rvecs: {n_Rvecs}"
        # )

        centers = pos_operator[iR0, :, :, :].diagonal(offset=0, axis1=0, axis2=1).T.real

        return {
            "n_wann": n_wann,
            "lattice": lattice,
            "Rvectors": Rvectors,
            "Rdegens": Rdegens,
            "H": H,
            "pos_operator": pos_operator,
            "centers": centers,
            "header": header,
        }


def test_parse_tb_file():
    filename = "cri3_up_tb.dat"
    result = parse_tb_file(filename)
    # rvecs=result["Rvectors"]
    # iR=find_Rvectors(rvecs, [0, 0, 0])
    # for i in range(30):
    #    print(np.real(result["pos_operator"][iR, i, i, :]))
    print(result["centers"])


if __name__ == "__main__":
    test_parse_tb_file()
