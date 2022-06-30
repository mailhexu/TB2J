import numpy as np
from TB2J.supercell import find_primitive_cell, map_to_primitive
from TB2J.io_exchange.io_exchange import SpinIO, gen_distance_dict


def cut_cell(path, output_path, sc_matrix, origin_atom_id, thr=1e-5):
    """
    Cut the exchange parameters
    :param path: the original TB2J_results path
    :param output_path: the output path.
    :param sc_matrix: the matrix which maps the primitive cell to supercell.
    :param origin: the origin of the primitive cell.
    :param thr: the atoms which the reduced position is within -thr to 1.0+thr are considered as inside the primitive atoms
    :returns:
    """
    sc_matrix = np.asarray(sc_matrix, dtype=int)
    sc_excparams = SpinIO.load_pickle(path=path, fname='TB2J.pickle')
    sc_atoms = sc_excparams.atoms
    uc_atoms, ids = find_primitive_cell(sc_atoms,
                                        sc_matrix,
                                        origin_atom_id=origin_atom_id,
                                        thr=thr,
                                        perfect=False)
    uc_charges = sc_excparams.charges[ids]
    uc_spinat = sc_excparams.spinat[ids]
    indmap, Rmap = map_to_primitive(sc_atoms, uc_atoms)

    # TODO index_spin: {iatom: ispin}

    # list of iatom for each spin index.
    uc_index_spin = [sc_excparams.index_spin[i] for i in ids]

    uc_ind_atoms = {}
    for iatom, ispin in enumerate(uc_index_spin):
        if ispin >= 0:
            uc_ind_atoms[ispin] = iatom

    uc_Jdict = {}
    uc_Rset = set()
    for key, val in sc_excparams.exchange_Jdict.items():
        R, ispin, jspin = key
        iatom = sc_excparams.ind_atoms[ispin]
        jatom = sc_excparams.ind_atoms[jspin]
        uc_ispin = uc_index_spin[indmap[iatom]]
        uc_jspin = uc_index_spin[indmap[jatom]]
        uc_R = R @ sc_matrix + Rmap[jatom] - Rmap[iatom]
        uc_R = tuple(uc_R)
        if iatom in ids:
            #print(f"{iatom=}, {indmap[iatom]=},{uc_ispin=}, {uc_jspin=}, {uc_R=} ")
            uc_Jdict[(uc_R, uc_ispin, uc_jspin)] = val
            uc_Rset.add(uc_R)

    uc_distance_dict = gen_distance_dict(uc_ind_atoms, uc_atoms, list(uc_Rset))

    assert sc_excparams.colinear, "Cut supercell for non-collinear spin is not yet implemented."

    uc_exc = SpinIO(uc_atoms,
                    spinat=uc_spinat,
                    charges=uc_charges,
                    index_spin=uc_index_spin,
                    colinear=sc_excparams.colinear,
                    distance_dict=uc_distance_dict,
                    exchange_Jdict=uc_Jdict,
                    description="Cutted")

    uc_exc.write_all(output_path)


def run_cut_cell(
    path="TB2J_results",
    output_path="./TB2J_cutted",
    sc_matrix=np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]]),
):
    cut_cell(path,
             output_path,
             np.array(sc_matrix),
             origin_atom_id=0,
             thr=1e-19)


if __name__ == "__main__":
    run_cut_cell()
