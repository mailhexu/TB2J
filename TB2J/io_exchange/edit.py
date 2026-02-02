"""
TB2J_edit: A library for modifying TB2J results.

This library provides a simple interface for editing TB2J exchange parameters,
including single ion anisotropy, DMI, anisotropic exchange, and symmetry operations.

Example:
    >>> from TB2J.io_exchange.edit import load, set_anisotropy, toggle_DMI, save
    >>> spinio = load('TB2J_results/TB2J.pickle')
    >>> set_anisotropy(spinio, species='Sm', k1=5.0, k1dir=[0, 0, 1])
    >>> toggle_DMI(spinio, enabled=False)
    >>> save(spinio, 'modified_results')
"""

import os

import numpy as np

from TB2J.io_exchange.io_exchange import SpinIO

__all__ = [
    "load",
    "save",
    "set_anisotropy",
    "set_sia_tensor",
    "remove_sia_tensor",
    "toggle_DMI",
    "toggle_Jani",
    "toggle_exchange",
    "remove_sublattice",
    "symmetrize_exchange",
]


def load(path="TB2J_results/TB2J.pickle"):
    """
    Load TB2J results from a pickle file.

    Parameters
    ----------
    path : str, optional
        Path to the pickle file. Default is 'TB2J_results/TB2J.pickle'.

    Returns
    -------
    spinio : SpinIO
        The loaded SpinIO object containing all TB2J results.

    Raises
    ------
    FileNotFoundError
        If the specified pickle file does not exist.

    Examples
    --------
    >>> spinio = load('my_results/TB2J.pickle')
    >>> print(spinio.exchange_Jdict)
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, "TB2J.pickle")
    if not os.path.exists(path):
        raise FileNotFoundError(f"TB2J pickle file not found: {path}")

    dirname, fname = os.path.split(path)
    return SpinIO.load_pickle(path=dirname, fname=fname)


def save(spinio, path="modified_results"):
    """
    Save modified TB2J results to disk in all supported formats.

    This writes the pickle file, TXT format, Multibinit XML, and other formats.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to save.
    path : str, optional
        Output directory path. Default is 'modified_results'.
        Will be created if it doesn't exist.

    Examples
    --------
    >>> save(spinio, 'my_modified_results')
    """
    spinio.write_all(path=path)

    # Fix mb.in to use SIA from exchange.xml when SIA is present
    if spinio.has_uniaxial_anistropy or (
        hasattr(spinio, "has_sia_tensor") and spinio.has_sia_tensor
    ):
        mb_in_path = os.path.join(path, "Multibinit", "mb.in")
        if os.path.exists(mb_in_path):
            with open(mb_in_path, "r") as f:
                content = f.read()
            # Replace spin_sia_add = 1 with spin_sia_add = 0
            content = content.replace("spin_sia_add = 1", "spin_sia_add = 0")
            with open(mb_in_path, "w") as f:
                f.write(content)


def set_anisotropy(spinio, species, k1=None, k1dir=None):
    """
    Set single ion anisotropy for all atoms of a specified species.

    Modifies the k1 (amplitude) and/or k1dir (direction) for all magnetic
    atoms of the given chemical species.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    species : str
        Chemical species symbol (e.g., 'Sm', 'Fe').
    k1 : float, optional
        Anisotropy amplitude in eV. If None, only k1dir is modified.
    k1dir : array-like, optional
        Anisotropy direction as a 3D vector [x, y, z].
        Will be normalized automatically. If None, only k1 is modified.

    Notes
    -----
    - Only magnetic atoms (those with index_spin >= 0) are modified.
    - If k1/k1dir are None in the SpinIO object, they will be initialized.
    - k1dir is always normalized to a unit vector.

    Examples
    --------
    >>> # Set Sm anisotropy to 5 meV along z-axis
    >>> set_anisotropy(spinio, species='Sm', k1=0.005, k1dir=[0, 0, 1])

    >>> # Set only the direction for Fe
    >>> set_anisotropy(spinio, species='Fe', k1dir=[1, 0, 0])
    """
    # Get symbols and find target atoms
    symbols = spinio.atoms.get_chemical_symbols()

    target_indices = [
        i
        for i, (sym, idx) in enumerate(zip(symbols, spinio.index_spin))
        if sym == species and idx >= 0
    ]

    if not target_indices:
        import warnings

        warnings.warn(
            f"No magnetic atoms found for species '{species}'. "
            f"Either the species doesn't exist or has no magnetic atoms.",
            UserWarning,
        )
        return

    # Initialize k1/k1dir if not present
    n_spins = (
        max(spinio.index_spin) + 1
        if max(spinio.index_spin) >= 0
        else len(spinio.index_spin)
    )
    if spinio.k1 is None:
        spinio.k1 = [0.0] * n_spins
    if spinio.k1dir is None:
        spinio.k1dir = [[0.0, 0.0, 1.0]] * n_spins

    # Set values for matching atoms
    for iatom in target_indices:
        ispin = spinio.index_spin[iatom]
        if k1 is not None:
            spinio.k1[ispin] = k1  # eV
        if k1dir is not None:
            k1dir_array = np.array(k1dir, dtype=float)
            norm = np.linalg.norm(k1dir_array)
            if norm == 0:
                raise ValueError(
                    f"k1dir cannot be a zero vector for species '{species}'"
                )
            spinio.k1dir[ispin] = k1dir_array / norm

    # Set the flag to enable SIA in output
    spinio.has_uniaxial_anistropy = True


def set_sia_tensor(spinio, species, tensor):
    """
    Set full single ion anisotropy tensor for all atoms of a specified species.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    species : str
        Chemical species symbol (e.g., 'Sm', 'Fe').
    tensor : array-like
        3x3 anisotropy tensor in eV.

    Examples
    --------
    >>> # Set Sm anisotropy tensor
    >>> tensor = np.diag([0.001, 0.002, 0.003])
    >>> set_sia_tensor(spinio, species='Sm', tensor=tensor)
    """
    # Get symbols and find target atoms
    symbols = spinio.atoms.get_chemical_symbols()

    target_indices = [
        i
        for i, (sym, idx) in enumerate(zip(symbols, spinio.index_spin))
        if sym == species and idx >= 0
    ]

    if not target_indices:
        import warnings

        warnings.warn(
            f"No magnetic atoms found for species '{species}'. "
            f"Either the species doesn't exist or has no magnetic atoms.",
            UserWarning,
        )
        return

    # Initialize sia_tensor dict if not present
    if not hasattr(spinio, "sia_tensor") or spinio.sia_tensor is None:
        spinio.sia_tensor = {}

    tensor_array = np.array(tensor, dtype=float)
    if tensor_array.shape != (3, 3):
        raise ValueError("SIA tensor must be a 3x3 matrix")

    # Set values for matching atoms
    for iatom in target_indices:
        ispin = spinio.index_spin[iatom]
        spinio.sia_tensor[ispin] = tensor_array.copy()

    # Set the flag to enable SIA tensor in output
    spinio.has_sia_tensor = True


def remove_sia_tensor(spinio, species=None):
    """
    Remove single ion anisotropy tensor.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    species : str, optional
        Chemical species symbol (e.g., 'Sm', 'Fe').
        If None, remove for all atoms.
    """
    if not hasattr(spinio, "sia_tensor") or spinio.sia_tensor is None:
        return

    if species is None:
        spinio.sia_tensor = None
        spinio.has_sia_tensor = False
    else:
        symbols = spinio.atoms.get_chemical_symbols()
        target_indices = [
            i
            for i, (sym, idx) in enumerate(zip(symbols, spinio.index_spin))
            if sym == species and idx >= 0
        ]
        for iatom in target_indices:
            ispin = spinio.index_spin[iatom]
            if ispin in spinio.sia_tensor:
                del spinio.sia_tensor[ispin]
        if not spinio.sia_tensor:
            spinio.has_sia_tensor = False


def toggle_DMI(spinio, enabled=None):
    """
    Enable or disable Dzyaloshinskii-Moriya interactions (DMI).

    When disabling, the DMI values are backed up and can be restored
    by re-enabling.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    enabled : bool, optional
        If True, enable DMI. If False, disable DMI.
        If None, toggle the current state.

    Examples
    --------
    >>> # Disable DMI
    >>> toggle_DMI(spinio, enabled=False)

    >>> # Toggle DMI (disable if enabled, enable if disabled)
    >>> toggle_DMI(spinio)

    >>> # Re-enable DMI
    >>> toggle_DMI(spinio, enabled=True)
    """
    if enabled is None:
        # Toggle current state
        enabled = not spinio.has_dmi

    if enabled and not spinio.has_dmi:
        # Re-enable: restore from backup or initialize empty
        if hasattr(spinio, "_dmi_backup"):
            spinio.dmi_ddict = spinio._dmi_backup
        else:
            spinio.dmi_ddict = {}
    elif not enabled and spinio.has_dmi:
        # Disable: backup and clear
        spinio._dmi_backup = spinio.dmi_ddict
        spinio.dmi_ddict = None


def toggle_Jani(spinio, enabled=None):
    """
    Enable or disable symmetric anisotropic exchange (Jani).

    When disabling, the Jani values are backed up and can be restored
    by re-enabling.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    enabled : bool, optional
        If True, enable Jani. If False, disable Jani.
        If None, toggle the current state.

    Examples
    --------
    >>> # Disable anisotropic exchange
    >>> toggle_Jani(spinio, enabled=False)

    >>> # Toggle anisotropic exchange
    >>> toggle_Jani(spinio)

    >>> # Re-enable anisotropic exchange
    >>> toggle_Jani(spinio, enabled=True)
    """
    if enabled is None:
        # Toggle current state
        enabled = not spinio.has_bilinear

    if enabled and not spinio.has_bilinear:
        # Re-enable: restore from backup or initialize empty
        if hasattr(spinio, "_jani_backup"):
            spinio.Jani_dict = spinio._jani_backup
        else:
            spinio.Jani_dict = {}
    elif not enabled and spinio.has_bilinear:
        # Disable: backup and clear
        spinio._jani_backup = spinio.Jani_dict
        spinio.Jani_dict = None


def toggle_exchange(spinio, enabled=None):
    """
    Enable or disable isotropic exchange parameters.

    When disabling, the exchange values are backed up and can be restored
    by re-enabling.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    enabled : bool, optional
        If True, enable exchange. If False, disable exchange.
        If None, toggle the current state.

    Examples
    --------
    >>> # Disable isotropic exchange
    >>> toggle_exchange(spinio, enabled=False)

    >>> # Toggle isotropic exchange
    >>> toggle_exchange(spinio)

    >>> # Re-enable isotropic exchange
    >>> toggle_exchange(spinio, enabled=True)
    """
    if enabled is None:
        # Toggle current state
        enabled = not spinio.has_exchange

    if enabled and not spinio.has_exchange:
        # Re-enable: restore from backup or initialize empty
        if hasattr(spinio, "_exchange_backup"):
            spinio.exchange_Jdict = spinio._exchange_backup
        else:
            spinio.exchange_Jdict = {}
    elif not enabled and spinio.has_exchange:
        # Disable: backup and clear
        spinio._exchange_backup = spinio.exchange_Jdict
        spinio.exchange_Jdict = None


def remove_sublattice(spinio, sublattice_name):
    """
    Remove all magnetic interactions associated with a specific sublattice.

    This includes:
    - Single-ion anisotropy (SIA) for atoms in the sublattice.
    - Exchange interactions (J) where i or j belongs to the sublattice.
    - Dzyaloshinskii-Moriya interactions (DMI) where i or j belongs to the sublattice.
    - Anisotropic exchange where i or j belongs to the sublattice.

    Parameters
    ----------
    spinio : SpinIO
        The SpinIO object to modify.
    sublattice_name : str
        The name of the sublattice (species symbol) to remove.

    Examples
    --------
    >>> # Remove all interactions involving Sm atoms
    >>> remove_sublattice(spinio, 'Sm')
    """
    symbols = spinio.atoms.get_chemical_symbols()
    sublattice_indices = [
        i
        for i, (sym, idx) in enumerate(zip(symbols, spinio.index_spin))
        if sym == sublattice_name and idx >= 0
    ]

    if not sublattice_indices:
        import warnings

        warnings.warn(f"No magnetic atoms found for sublattice '{sublattice_name}'.")
        return

    sublattice_spin_indices = set(spinio.index_spin[i] for i in sublattice_indices)

    for i in sublattice_indices:
        spinio.index_spin[i] = -1

    # Re-index spins
    # Map old spin indices to new spin indices
    # Remaining spins will be compacted to 0, 1, 2...

    old_to_new_spin_index = {}
    current_spin_index = 0
    # max_old_spin = max(spinio.index_spin)

    for iatom, ispin in enumerate(spinio.index_spin):
        if ispin >= 0:
            if ispin not in old_to_new_spin_index:
                old_to_new_spin_index[ispin] = current_spin_index
                current_spin_index += 1
            spinio.index_spin[iatom] = old_to_new_spin_index[ispin]

    def reindex_interaction(interaction_dict):
        if interaction_dict is None:
            return None
        new_dict = {}
        for key, val in interaction_dict.items():
            R, i, j = key
            if i in old_to_new_spin_index and j in old_to_new_spin_index:
                new_i = old_to_new_spin_index[i]
                new_j = old_to_new_spin_index[j]
                new_dict[(R, new_i, new_j)] = val
        return new_dict

    def filter_interaction(interaction_dict):
        if interaction_dict is None:
            return None
        new_dict = {}
        for key, val in interaction_dict.items():
            R, i, j = key
            if i not in sublattice_spin_indices and j not in sublattice_spin_indices:
                new_dict[key] = val
        return new_dict

    def filter_distance_dict(distance_dict):
        if distance_dict is None:
            return None
        new_dict = {}
        for key, val in distance_dict.items():
            R, i, j = key
            if i not in sublattice_spin_indices and j not in sublattice_spin_indices:
                new_dict[key] = val
        return new_dict

    if spinio.has_exchange:
        spinio.exchange_Jdict = filter_interaction(spinio.exchange_Jdict)
        spinio.exchange_Jdict = reindex_interaction(spinio.exchange_Jdict)
        spinio.distance_dict = filter_distance_dict(spinio.distance_dict)
        spinio.distance_dict = reindex_interaction(spinio.distance_dict)
        if hasattr(spinio, "Jiso_orb") and spinio.Jiso_orb:
            spinio.Jiso_orb = filter_interaction(spinio.Jiso_orb)
            spinio.Jiso_orb = reindex_interaction(spinio.Jiso_orb)

    if spinio.has_dmi:
        spinio.dmi_ddict = filter_interaction(spinio.dmi_ddict)
        spinio.dmi_ddict = reindex_interaction(spinio.dmi_ddict)
        if hasattr(spinio, "DMI_orb") and spinio.DMI_orb:
            spinio.DMI_orb = filter_interaction(spinio.DMI_orb)
            spinio.DMI_orb = reindex_interaction(spinio.DMI_orb)

    if spinio.has_bilinear:
        spinio.Jani_dict = filter_interaction(spinio.Jani_dict)
        spinio.Jani_dict = reindex_interaction(spinio.Jani_dict)
        if hasattr(spinio, "Jani_orb") and spinio.Jani_orb:
            spinio.Jani_orb = filter_interaction(spinio.Jani_orb)
            spinio.Jani_orb = reindex_interaction(spinio.Jani_orb)

    if hasattr(spinio, "dJdx") and spinio.dJdx:
        spinio.dJdx = filter_distance_dict(spinio.dJdx)
        spinio.dJdx = reindex_interaction(spinio.dJdx)
    if hasattr(spinio, "dJdx2") and spinio.dJdx2:
        spinio.dJdx2 = filter_distance_dict(spinio.dJdx2)
        spinio.dJdx2 = reindex_interaction(spinio.dJdx2)
    if hasattr(spinio, "biquadratic_Jdict") and spinio.biquadratic_Jdict:
        spinio.biquadratic_Jdict = filter_interaction(spinio.biquadratic_Jdict)
        spinio.biquadratic_Jdict = reindex_interaction(spinio.biquadratic_Jdict)
    if hasattr(spinio, "NJT_Jdict") and spinio.NJT_Jdict:
        spinio.NJT_Jdict = filter_interaction(spinio.NJT_Jdict)
        spinio.NJT_Jdict = reindex_interaction(spinio.NJT_Jdict)
    if hasattr(spinio, "NJT_ddict") and spinio.NJT_ddict:
        spinio.NJT_ddict = filter_interaction(spinio.NJT_ddict)
        spinio.NJT_ddict = reindex_interaction(spinio.NJT_ddict)

    if spinio.has_uniaxial_anistropy:
        if spinio.k1 is not None:
            new_k1 = [0.0] * current_spin_index
            new_k1dir = [[0.0, 0.0, 1.0]] * current_spin_index

            for old_idx, new_idx in old_to_new_spin_index.items():
                if old_idx < len(spinio.k1):
                    new_k1[new_idx] = spinio.k1[old_idx]
                    new_k1dir[new_idx] = spinio.k1dir[old_idx]

            spinio.k1 = new_k1
            spinio.k1dir = new_k1dir

    if (
        hasattr(spinio, "has_sia_tensor")
        and spinio.has_sia_tensor
        and spinio.sia_tensor is not None
    ):
        new_sia_tensor = {}
        for old_idx, tensor in spinio.sia_tensor.items():
            if old_idx in old_to_new_spin_index:
                new_idx = old_to_new_spin_index[old_idx]
                new_sia_tensor[new_idx] = tensor
        spinio.sia_tensor = new_sia_tensor
        # If no tensors remain, set has_sia_tensor to False
        if not spinio.sia_tensor:
            spinio.has_sia_tensor = False

    spinio._ind_atoms = {}
    for iatom, ispin in enumerate(spinio.index_spin):
        if ispin >= 0:
            spinio._ind_atoms[ispin] = iatom


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
    from TB2J.symmetrize_J import symmetrize_exchange as sym_exchange

    sym_exchange(spinio, atoms, symprec)
