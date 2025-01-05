from abacus_api import read_HR_SR


def main():
    # nspin = 2 case
    nspin = 2
    binary = False
    file_path = "../abacus_example/case_Sr2Mn2O6/1_no_soc/OUT.Sr2Mn2O6/"
    HR_file = [
        file_path + "data-HR-sparse_SPIN0.csr",
        file_path + "data-HR-sparse_SPIN1.csr",
    ]
    SR_file = file_path + "data-SR-sparse_SPIN0.csr"

    basis_num, R_direct_coor, HR_up, HR_dn, SR = read_HR_SR(
        nspin, binary, HR_file, SR_file
    )

    print("basis_num =", basis_num)
    print("R_direct_coor =", R_direct_coor)
    print("HR_up[1, 0, 15:22] =", HR_up[1, 0, 15:22])
    print("HR_dn[1, 0, 15:22] =", HR_dn[1, 0, 15:22])
    print("SR[1, 0, 15:22] =", SR[1, 0, 15:22])

    # nspin = 4 case
    nspin = 4
    binary = False
    file_path = "../abacus_example/case_Sr2Mn2O6/2_soc/OUT.Sr2Mn2O6/"
    HR_file = file_path + "data-HR-sparse_SPIN0.csr"
    SR_file = file_path + "data-SR-sparse_SPIN0.csr"

    basis_num, R_direct_coor, HR, SR = read_HR_SR(nspin, binary, HR_file, SR_file)

    print("basis_num =", basis_num)
    print("R_direct_coor =", R_direct_coor)
    print("HR[1, 0, 30:37] =", HR[1, 0, 30:37])
    print("SR[1, 0, 30:37] =", SR[1, 0, 30:37])

    return


if __name__ == "__main__":
    main()
