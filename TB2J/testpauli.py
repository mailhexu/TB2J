from TB2J.pauli import *


def test():
    a = vec_to_mat(1.3, 0.0, 0.0, 0.7)
    print(a)

    print(pauli_block_all(a))

    for i in range(1, 4):
        print(commutate(a, i))


test()
