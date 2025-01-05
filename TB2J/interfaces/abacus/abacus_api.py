# -*- coding: utf-8 -*-
import re
import struct

import numpy as np
from scipy.sparse import csr_matrix


class XR_matrix:
    def __init__(self, nspin, XR_fileName):
        self.nspin = nspin
        self.XR_fileName = XR_fileName

    def read_file(self):
        with open(self.XR_fileName, "r") as fread:
            while True:
                line = fread.readline().split()
                if line[0] == "Matrix":
                    break

            self.basis_num = int(line[-1])
            line = fread.readline()
            self.R_num = int(line.split()[-1])
            self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
            if self.nspin != 4:
                self.XR = np.zeros(
                    [self.R_num, self.basis_num, self.basis_num], dtype=float
                )
            else:
                self.XR = np.zeros(
                    [self.R_num, self.basis_num, self.basis_num], dtype=complex
                )

            for iR in range(self.R_num):
                line = fread.readline().split()
                self.R_direct_coor[iR, 0] = int(line[0])
                self.R_direct_coor[iR, 1] = int(line[1])
                self.R_direct_coor[iR, 2] = int(line[2])
                data_size = int(line[3])

                if self.nspin != 4:
                    data = np.zeros((data_size,), dtype=float)
                else:
                    data = np.zeros((data_size,), dtype=complex)

                indices = np.zeros((data_size,), dtype=int)
                indptr = np.zeros((self.basis_num + 1,), dtype=int)

                if data_size != 0:
                    if self.nspin != 4:
                        line = fread.readline().split()
                        if len(line) != data_size:
                            print("size = ", len(line), " data_size = ", data_size)
                        for index in range(data_size):
                            data[index] = float(line[index])
                    else:
                        line = re.findall("[(](.*?)[])]", fread.readline())
                        for index in range(data_size):
                            value = line[index].split(",")
                            data[index] = complex(float(value[0]), float(value[1]))

                    line = fread.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line[index])

                    line = fread.readline().split()
                    for index in range(self.basis_num + 1):
                        indptr[index] = int(line[index])

                self.XR[iR] = csr_matrix(
                    (data, indices, indptr), shape=(self.basis_num, self.basis_num)
                ).toarray()

    def read_file_binary(self):
        with open(self.XR_fileName, "rb") as fread:
            self.basis_num = struct.unpack("i", fread.read(4))[0]
            self.R_num = struct.unpack("i", fread.read(4))[0]
            self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
            if self.nspin != 4:
                self.XR = np.zeros(
                    [self.R_num, self.basis_num, self.basis_num], dtype=float
                )
            else:
                self.XR = np.zeros(
                    [self.R_num, self.basis_num, self.basis_num], dtype=complex
                )

            for iR in range(self.R_num):
                self.R_direct_coor[iR, 0] = struct.unpack("i", fread.read(4))[0]
                self.R_direct_coor[iR, 1] = struct.unpack("i", fread.read(4))[0]
                self.R_direct_coor[iR, 2] = struct.unpack("i", fread.read(4))[0]
                data_size = struct.unpack("i", fread.read(4))[0]

                if self.nspin != 4:
                    data = np.zeros((data_size,), dtype=float)
                else:
                    data = np.zeros((data_size,), dtype=complex)

                indices = np.zeros((data_size,), dtype=int)
                indptr = np.zeros((self.basis_num + 1,), dtype=int)

                if data_size != 0:
                    if self.nspin != 4:
                        for index in range(data_size):
                            data[index] = struct.unpack("d", fread.read(8))[0]
                    else:
                        for index in range(data_size):
                            real = struct.unpack("d", fread.read(8))[0]
                            imag = struct.unpack("d", fread.read(8))[0]
                            data[index] = complex(real, imag)

                    for index in range(data_size):
                        indices[index] = struct.unpack("i", fread.read(4))[0]

                    for index in range(self.basis_num + 1):
                        indptr[index] = struct.unpack("i", fread.read(4))[0]

                self.XR[iR] = csr_matrix(
                    (data, indices, indptr), shape=(self.basis_num, self.basis_num)
                ).toarray()


def read_HR_SR(
    nspin=4,
    binary=False,
    HR_fileName="data-HR-sparse_SPIN0.csr",
    SR_fileName="data-SR-sparse_SPIN0.csr",
):
    """
    IN:
        nspin: int, different spins.
        binary: bool, whether the HR and SR matrices are binary files.
        HR_fileName: if nspin=1 or 4, str, HR file name;
                     if nspin=2, list or tuple, size=2, [HR_up_fileName, HR_dn_fileName].
        SR_fileName: str, SR file name.

    OUT:
        if nspin = 1 or 4:
            return basis_num, R_direct_coor, HR, SR
        elif nspin = 2:
            return basis_num, R_direct_coor, HR_up, HR_dn, SR

        basis_num: int, number of atomic orbital basis.
        R_direct_coor: numpy ndarray, shape=[R_num, 3], fractional coordinates (x, y, z) of the R-th primitive cell.
        HR or HR_up or HR_dn: numpy ndarray, shape=[R_num, basis_num, basis_num], HR matrix and unit is eV.
        SR: numpy ndarray, shape=[R_num, basis_num, basis_num], SR matrix.
    """
    Ry_to_eV = 13.605698066

    if nspin == 1 or nspin == 4:
        if not isinstance(HR_fileName, str):
            raise ValueError("The HR_fileName must be a str for nspin=1 or 4.")

        HR = XR_matrix(nspin, HR_fileName)
        SR = XR_matrix(nspin, SR_fileName)

        if binary:
            HR.read_file_binary()
            SR.read_file_binary()
        else:
            HR.read_file()
            SR.read_file()

        return HR.basis_num, HR.R_direct_coor, HR.XR * Ry_to_eV, SR.XR
    else:
        if not isinstance(HR_fileName, (list, tuple)):
            raise ValueError("The HR_fileName must be a list or a tuple for nspin=2.")

        if len(HR_fileName) != 2:
            raise ValueError("The size of the HR_fileName must be 2 for nspin=2.")

        HR_up = XR_matrix(nspin, HR_fileName[0])
        HR_dn = XR_matrix(nspin, HR_fileName[1])
        SR = XR_matrix(nspin, SR_fileName)

        if binary:
            HR_up.read_file_binary()
            HR_dn.read_file_binary()
            SR.read_file_binary()
        else:
            HR_up.read_file()
            HR_dn.read_file()
            SR.read_file()

        return (
            HR_up.basis_num,
            HR_up.R_direct_coor,
            HR_up.XR * Ry_to_eV,
            HR_dn.XR * Ry_to_eV,
            SR.XR,
        )
