#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   test_read_stru.py
@Time    :   2024/02/02 09:52:23
@Author  :   Shen Zhen-Xiong
@Email   :   shenzx@iai.ustc.edu.cn
"""

import os

from stru_api import read_abacus, read_input, write_abacus


def main():
    # stru_fe = read_abacus(os.path.join(os.getcwd(), "input/Fe.STRU"))
    stru_sr2mn2o6 = read_abacus(
        os.path.join(os.getcwd(), "input/Sr2Mn2O6.STRU"), verbose=True
    )
    write_abacus(
        file=os.path.join(os.getcwd(), "STRU"),
        atoms=stru_sr2mn2o6,
        pp=stru_sr2mn2o6.info["pp"],
        basis=stru_sr2mn2o6.info["basis"],
    )
    input_file = read_input(os.path.join(os.getcwd(), "input/INPUT"))
    print(input_file["pseudo_dir"])
    return


if __name__ == "__main__":
    main()
