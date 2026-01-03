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
    "toggle_DMI",
    "toggle_Jani",
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
    if spinio.has_uniaxial_anistropy:
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
        spinio.has_dmi = True
    elif not enabled and spinio.has_dmi:
        # Disable: backup and clear
        spinio._dmi_backup = spinio.dmi_ddict
        spinio.dmi_ddict = {}
        spinio.has_dmi = False


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
        spinio.has_bilinear = bool(spinio.Jani_dict)
    elif not enabled and spinio.has_bilinear:
        # Disable: backup and clear
        spinio._jani_backup = spinio.Jani_dict
        spinio.Jani_dict = {}
        spinio.has_bilinear = False


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
