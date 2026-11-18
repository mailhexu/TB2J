import copy
from collections import defaultdict

import numpy as np
from sympair import SymmetryPairFinder

from TB2J.io_exchange import SpinIO
from TB2J.versioninfo import print_license


class TB2JSymmetrizer:
    def __init__(self, exc, symprec=1e-8, verbose=True, Jonly=False):
        # list of pairs with the index of atoms
        ijRs = exc.ijR_list_index_atom()
        finder = SymmetryPairFinder(atoms=exc.atoms, pairs=ijRs, symprec=symprec)
        self.verbose = verbose
        self.Jonly = Jonly

        if verbose:
            print("=" * 30)
            print_license()
            print("-" * 30)
            print(
                "WARNING: The symmetry detection is based on the crystal symmetry, not the magnetic symmetry. Make sure if this is what you want."
            )
            print("-" * 30)
            if exc.has_dmi:
                print(
                    "WARNING: Currently only the isotropic exchange is symmetrized. Symmetrization of DMI and anisotropic exchange are not yet implemented."
                )

            print(f"Finding crystal symmetry with symprec of {symprec} Angstrom.")
            print("Symmetry found:")
            print(finder.spacegroup)
            print("-" * 30)
        self.pgdict = finder.get_symmetry_pair_list_dict()
        self.exc = exc
        self.new_exc = copy.deepcopy(exc)
        self.Jonly = Jonly

    def print_license(self):
        print_license()

    def symmetrize_J(self):
        """
        Symmetrize the exchange parameters J.
        """
        symJdict = {}
        # Jdict = self.exc.exchange_Jdict
        # ngroup = self.pgdict
        for pairgroup in self.pgdict.pairlists:
            ijRs = pairgroup.get_all_ijR()
            ijRs_spin = [self.exc.ijR_index_atom_to_spin(*ijR) for ijR in ijRs]
            Js = []
            for ijR_spin in ijRs_spin:
                i, j, R = ijR_spin
                J = self.exc.get_J(i, j, R)
                if J is not None:
                    Js.append(J)
            if Js:
                Javg = np.average(Js)
                for i, j, R in ijRs_spin:
                    symJdict[(R, i, j)] = Javg
        self.new_exc.exchange_Jdict = symJdict
        if self.Jonly:
            self.new_exc.has_dmi = False
            self.new_exc.dmi_ddict = None
            self.new_exc.has_bilinear = False
            self.new_exc.Jani_dict = None
            self.has_uniaxial_anisotropy = False
            self.k1 = None
            self.k1dir = None

    def output(self, path="TB2J_symmetrized"):
        if path is None:
            path = "TB2J_symmetrized"
        self.new_exc.write_all(path=path)

    def run(self, path="TB2J_symmetrized"):
        print("** Symmetrizing exchange parameters.")
        self.symmetrize_J()
        print("** Outputing the symmetrized exchange parameters.")
        print(f"** Output path: {path} .")
        self.output(path=path)
        print("** Finished.")


def symmetrize_J(
    exc=None,
    path=None,
    fname="TB2J.pickle",
    symprec=1e-5,
    output_path="TB2J_symmetrized",
    Jonly=False,
):
    """
    symmetrize the exchange parameters
    parameters:
    exc: exchange
    """
    if exc is None:
        if path is None:
            raise ValueError("Please provide the path to the exchange parameters.")
        exc = SpinIO.load_pickle(path=path, fname=fname)
    symmetrizer = TB2JSymmetrizer(exc, symprec=symprec, Jonly=Jonly)
    symmetrizer.run(path=output_path)


def _map_atoms_to_spinio(atoms, spinio, symprec=1e-3):
    """
    Map atoms from input structure to SpinIO structure.

    Uses species and position matching within symprec tolerance.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atomic structure.
    spinio : SpinIO
        The SpinIO object containing the reference structure.
    symprec : float, optional
        Position tolerance in Angstrom. Default is 1e-3.

    Returns
    -------
    dict
        Mapping from SpinIO atom index to input structure atom index.
    """
    mapping = {}
    symbols_in = atoms.get_chemical_symbols()
    pos_in = atoms.get_positions()
    symbols_s = spinio.atoms.get_chemical_symbols()
    pos_s_array = spinio.atoms.get_positions()

    for i_in, (sym, pos) in enumerate(zip(symbols_in, pos_in)):
        for i_s, (sym_s, pos_s) in enumerate(zip(symbols_s, pos_s_array)):
            if sym == sym_s:
                if np.linalg.norm(pos - pos_s) < symprec:
                    mapping[i_s] = i_in
                    break
    return mapping


def symmetrize_exchange(spinio, atoms, symprec=1e-3):
    """
    Symmetrize isotropic exchange based on a provided atomic structure.

    The symmetry is detected from the provided atomic structure using spglib.
    Exchange parameters for symmetry-equivalent atom pairs are averaged.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    atoms : ase.Atoms
        Atomic structure that defines the target symmetry.
        For example, provide a cubic structure to symmetrize to cubic symmetry.
    symprec : float, optional
        Symmetry precision in Angstrom. Default is 1e-3.

    Notes
    -----
    - Only isotropic exchange (exchange_Jdict) is modified.
    - DMI and anisotropic exchange are unchanged.
    - The spinio.atoms structure is NOT modified; only exchange values change.
    - Atoms are mapped between input and SpinIO structures by species and position.

    Examples
    --------
    >>> from ase.io import read
    >>> # Symmetrize to cubic symmetry
    >>> cubic_structure = read('cubic_smfeo3.cif')
    >>> symmetrize_exchange(spinio, atoms=cubic_structure)

    >>> # Symmetrize to original Pnma symmetry (averaging within groups)
    >>> symmetrize_exchange(spinio, atoms=spinio.atoms)
    """
    try:
        from spglib import spglib as spg
    except ImportError:
        raise ImportError(
            "spglib is required for symmetrization. Install it with: pip install spglib"
        )

    # Get symmetry dataset from the provided structure
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    dataset = spg.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
    if dataset is None:
        raise ValueError(
            "spglib could not detect symmetry from the provided structure. "
            "Check that the structure is valid and try adjusting symprec."
        )

    equivalent_atoms = dataset["equivalent_atoms"]

    # Map atoms between input structure and SpinIO structure
    atom_mapping = _map_atoms_to_spinio(atoms, spinio, symprec=symprec)

    # Group equivalent pairs and average J values
    groups = defaultdict(list)

    for (R, i, j), J in spinio.exchange_Jdict.items():
        # Get corresponding atom indices in SpinIO
        iatom = spinio.iatom(i)
        jatom = spinio.iatom(j)

        # Find matching atoms in input structure
        i_in = atom_mapping.get(iatom)
        j_in = atom_mapping.get(jatom)

        if i_in is not None and j_in is not None:
            # Key based on equivalent atom orbits
            key = (equivalent_atoms[i_in], equivalent_atoms[j_in], R)
            groups[key].append(J)

    # Average and reassign
    for (R, i, j), J in spinio.exchange_Jdict.items():
        iatom = spinio.iatom(i)
        jatom = spinio.iatom(j)
        i_in = atom_mapping.get(iatom)
        j_in = atom_mapping.get(jatom)

        if i_in is not None and j_in is not None:
            key = (equivalent_atoms[i_in], equivalent_atoms[j_in], R)
            if key in groups:
                spinio.exchange_Jdict[(R, i, j)] = np.mean(groups[key])


def symmetrize_J_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Symmetrize exchange parameters. Currently, it take the crystal symmetry into account and  not the magnetic moment."
    )
    parser.add_argument(
        "-i",
        "--inpath",
        default=None,
        help="input path to the exchange parameters",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        default="TB2J_results_symmetrized",
        help="output path to the symmetrized exchange parameters",
    )
    parser.add_argument(
        "-s",
        "--symprec",
        type=float,
        default=1e-5,
        help="precision for symmetry detection. default is 1e-5 Angstrom",
    )

    parser.add_argument(
        "--Jonly",
        action="store_true",
        help="symmetrize only the exchange parameters and discard the DMI and anisotropic exchange",
        default=False,
    )

    args = parser.parse_args()
    if args.inpath is None:
        parser.print_help()
        raise ValueError("Please provide the input path to the exchange.")
    symmetrize_J(
        path=args.inpath,
        output_path=args.outpath,
        symprec=args.symprec,
        Jonly=args.Jonly,
    )


if __name__ == "__main__":
    symmetrize_J_cli()
