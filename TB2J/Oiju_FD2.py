"""
Module for calculating spin-phonon coupling using finite difference method.

This module provides functionality to calculate spin-phonon coupling parameters
by using finite difference of Wannier90 tight-binding models with atomic distortions.
It uses the same interface as Oiju_epw2.py but implements the finite difference approach
from Oiju_FD.py.
"""

import os

import numpy as np
from ase.io import read
from supercellmap import SupercellMaker

from TB2J.epw import epw_to_dHdx, generate_TB_with_distortion
from TB2J.exchange import ExchangeNCL
from TB2J.FDTB import dHdx
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.utils import auto_assign_basis_name


def merge_dHdx_spin(dH_up, dH_dn):
    """
    Merge spin-up and spin-down dH/dx matrices into a single spin-resolved dH/dx.

    The merged matrix has the structure:
    [[dH_up,  0   ],
     [0,      dH_dn]]

    Parameters
    ----------
    dH_up : dHdx
        dH/dx for spin-up channel
    dH_dn : dHdx
        dH/dx for spin-down channel

    Returns
    -------
    dHdx
        Merged spin-resolved dH/dx
    """
    assert dH_up.nbasis == dH_dn.nbasis, "Spin up and down must have same basis size"

    nb_single = dH_up.nbasis
    nb_total = nb_single * 2

    # Collect all R vectors from both spin channels
    all_R = set(dH_up.dHR.keys()) | set(dH_dn.dHR.keys())
    Rlist = np.array(list(all_R))

    # Merge dHR dictionaries
    merged_dHR = {}
    for R in all_R:
        d = np.zeros((nb_total, nb_total), dtype=complex)
        if R in dH_up.dHR:
            d[::2, ::2] = dH_up.dHR[R]  # Spin-up block
        if R in dH_dn.dHR:
            d[1::2, 1::2] = dH_dn.dHR[R]  # Spin-down block
        merged_dHR[R] = d

    # Merge wsdeg if available
    if dH_up.wsdeg is not None and dH_dn.wsdeg is not None:
        # Use average of wsdeg from both channels (should be the same)
        Rdict_up = {tuple(R): i for i, R in enumerate(dH_up.Rlist)}
        Rdict_dn = {tuple(R): i for i, R in enumerate(dH_dn.Rlist)}
        wsdeg = np.ones(len(Rlist))
        for i, R in enumerate(Rlist):
            if tuple(R) in Rdict_up:
                wsdeg[i] = dH_up.wsdeg[Rdict_up[tuple(R)]]
            elif tuple(R) in Rdict_dn:
                wsdeg[i] = dH_dn.wsdeg[Rdict_dn[tuple(R)]]
    else:
        wsdeg = None

    merged_dH = dHdx(
        Rlist=Rlist, nbasis=nb_total, dHR=merged_dHR, dHR2=None, wsdeg=wsdeg
    )

    return merged_dH


def gen_exchange_Oiju_epw(
    path: str,
    colinear=True,
    posfile="POSCAR",
    prefix_up="wannier90.up",
    prefix_dn="wannier90.dn",
    prefix_SOC="wannier90",
    epw_up_path="./",
    epw_down_path="./",
    epw_prefix_up="SrMnO3_up",
    epw_prefix_dn="SrMnO3_dn",
    idisp=0,
    Ru=(0, 0, 0),
    min_hopping_norm=1e-6,
    Rcut=None,
    efermi=3.0,
    magnetic_elements=[],
    kmesh=[5, 5, 5],
    emin=-12.0,
    emax=0.0,
    nz=50,
    np=1,
    exclude_orbs=[],
    description: str = "",
    list_iatom: list = None,
    output_path: str = "TB2J_results",
    # Additional parameters for finite difference method
    supercell_matrix=None,
    amplitude=0.01,
    max_distance=None,
):
    """Calculate spin-phonon coupling parameters using finite difference method.

    This function uses the same interface as Oiju_epw2.py but implements the
    finite difference approach from Oiju_FD.py. It generates distorted tight-binding
    models and compares them to compute spin-phonon coupling.

    Parameters
    ----------
    path : str
        Path to the Wannier90 calculation directory
    colinear : bool, default=True
        Whether the system is collinear magnetic
    posfile : str, default='POSCAR'
        Name of the structure file
    prefix_up : str, default='wannier90.up'
        Prefix for spin-up Wannier90 files
    prefix_dn : str, default='wannier90.dn'
        Prefix for spin-down Wannier90 files
    prefix_SOC : str, default='wannier90'
        Prefix for SOC Wannier90 files (not used in colinear mode)
    epw_up_path : str, default='./'
        Path to the EPW calculation directory (for spin-up)
    epw_down_path : str, default='./'
        Path to the EPW calculation directory (for spin-down)
    epw_prefix_up : str, default='SrMnO3_up'
        Prefix for spin-up EPW files
    epw_prefix_dn : str, default='SrMnO3_dn'
        Prefix for spin-down EPW files
    idisp : int, default=0
        Index of the atomic displacement pattern to analyze (maps to imode)
    Ru : tuple, default=(0,0,0)
        R vector for the phonon displacement (maps to Rp)
    min_hopping_norm : float, default=1e-6
        Minimum norm for hopping terms
    Rcut : float, optional
        Cutoff radius for interactions (not used in finite difference)
    efermi : float, default=3.0
        Fermi energy in eV
    magnetic_elements : list, default=[]
        List of magnetic elements
    kmesh : list, default=[5,5,5]
        k-point mesh dimensions
    emin : float, default=-12.0
        Minimum energy for integration
    emax : float, default=0.0
        Maximum energy for integration
    nz : int, default=50
        Number of energy points
    np : int, default=1
        Number of processors for parallel calculation
    exclude_orbs : list, default=[]
        List of orbitals to exclude
    description : str, default=''
        Description of the calculation
    list_iatom : list, optional
        List of atoms to include in calculation
    output_path : str, default='TB2J_results'
        Path for output files
    supercell_matrix : array_like, optional
        Supercell matrix for calculations. If None, uses identity (no supercell)
    amplitude : float, default=0.01
        Amplitude of atomic displacement for finite difference
    max_distance : float, optional
        Maximum distance for interactions (not used)

    Returns
    -------
    None
        Results are written to the specified output_path

    Notes
    -----
    - This implementation uses finite difference instead of EPW perturbation theory
    - The parameter names match Oiju_epw2.py for compatibility:
      * idisp (displacement index) corresponds to imode (phonon mode)
      * Ru corresponds to Rp (phonon R vector)
      * epw_up_path and epw_down_path are used to locate spin-resolved EPW files
    - Spin-resolved dH/dx: Both spin-up and spin-down EPW data are read and merged
    - Two calculations are performed: original and distorted structures
    - Results are saved in separate subdirectories
    """
    atoms = read(os.path.join(path, posfile))

    if supercell_matrix is None:
        supercell_matrix = np.eye(3)

    if colinear:
        # Read Wannier90 tight-binding models
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, atoms=atoms, nls=False
        )
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, atoms=atoms, nls=False
        )
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        # Create supercell maker
        scmaker = SupercellMaker(supercell_matrix, center=True)

        # Get dH/dx from EPW data for both spin channels
        # Note: idisp maps to imode, Ru maps to Rp in the original Oiju_FD.py
        dH_up = epw_to_dHdx(epw_up_path, epw_prefix_up, idisp, Ru, scmaker=scmaker)
        dH_dn = epw_to_dHdx(epw_down_path, epw_prefix_dn, idisp, Ru, scmaker=scmaker)

        # Merge spin-up and spin-down dH/dx
        dH = merge_dHdx_spin(dH_up, dH_dn)

        # Make supercell
        tbmodel = tbmodel.make_supercell(scmaker)
        atoms = scmaker.sc_atoms(atoms)
        basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)

        # Generate distorted tight-binding model
        dtb = generate_TB_with_distortion(tbmodel, amplitude, dH)

        # Calculate exchange for original structure
        exchange = ExchangeNCL(
            tbmodels=tbmodel,
            atoms=atoms,
            basis=basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            np=np,
            exclude_orbs=exclude_orbs,
            list_iatom=list_iatom,
            description=description,
        )
        exchange.run(os.path.join(output_path, "original"))

        # Calculate exchange for distorted structure
        exchange2 = ExchangeNCL(
            tbmodels=dtb,
            atoms=atoms,
            basis=basis,
            efermi=efermi,
            magnetic_elements=magnetic_elements,
            kmesh=kmesh,
            emin=emin,
            emax=emax,
            nz=nz,
            np=np,
            exclude_orbs=exclude_orbs,
            list_iatom=list_iatom,
            description=description,
        )
        exchange2.run(
            os.path.join(output_path, f"idisp{idisp}_Ru{Ru[0]}_{Ru[1]}_{Ru[2]}")
        )

    else:
        raise NotImplementedError("Non-collinear case not yet implemented")


if __name__ == "__main__":
    # Example usage with interface matching Oiju_epw2.py
    path = "/home/hexu/projects/TB2J_examples/Wannier/SrMnO3_QE_Wannier90/W90"

    for idisp in range(15):
        gen_exchange_Oiju_epw(
            path=path,
            colinear=True,
            posfile="SrMnO3.scf.pwi",
            prefix_up="SrMnO3_up",
            prefix_dn="SrMnO3_down",
            prefix_SOC="wannier90",
            epw_up_path="/home/hexu/projects/SrMnO3/epw",
            epw_down_path="/home/hexu/projects/SrMnO3/epw",
            epw_prefix_up="SrMnO3",
            epw_prefix_dn="SrMnO3",
            idisp=idisp,
            Ru=(0, 0, 0),
            supercell_matrix=np.eye(3) * 3,
            amplitude=0.01,
            max_distance=None,
            efermi=10.67,
            magnetic_elements=["Mn"],
            kmesh=[3, 3, 3],
            emin=-7.3363330034071295,
            emax=0.0,
            nz=70,
            np=1,
            exclude_orbs=[],
            description="",
            list_iatom=[13 * 5 + 1],
            output_path=f"FD333_idisp{idisp}",
        )
