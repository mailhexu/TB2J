"""
Module for calculating spin-phonon coupling using EPW (Electron-Phonon Wannier) data.

This module provides functionality to calculate spin-phonon coupling parameters
by combining Wannier90 electronic structure data with EPW electron-phonon
matrix elements. It uses perturbation theory to compute the coupling between
spin and lattice degrees of freedom.
"""

import os

from ase.io import read
from HamiltonIO.epw.epwparser import Epmat

from TB2J.exchange_pert2 import ExchangePert2
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.utils import auto_assign_basis_name


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
):
    """Calculate spin-phonon coupling parameters using EPW data.

    This function combines Wannier90 electronic structure with EPW electron-phonon
    matrix elements to compute the spin-phonon coupling. It can handle both
    collinear and non-collinear magnetic systems.

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
        Prefix for SOC Wannier90 files
    epw_path : str, default='./'
        Path to the EPW calculation directory
    epw_prefix_up : str, default='SrMnO3_up'
        Prefix for spin-up EPW files
    epw_prefix_dn : str, default='SrMnO3_dn'
        Prefix for spin-down EPW files
    idisp : int, default=0
        Index of the atomic displacement pattern to analyze
    Ru : tuple, default=(0,0,0)
        R vector for the unit cell
    min_hopping_norm : float, default=1e-6
        Minimum norm for hopping terms
    Rcut : float, optional
        Cutoff radius for interactions
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

    Returns
    -------
    None
        Results are written to the specified output_path
    """
    atoms = read(os.path.join(path, posfile))
    if colinear:
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, atoms=atoms, nls=False
        )
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, atoms=atoms, nls=False
        )
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        # Load spin-resolved EPW data
        epw_up = Epmat()
        epw_up.read(path=epw_up_path, prefix=epw_prefix_up, epmat_ncfile="epmat.nc")
        epw_dn = Epmat()
        epw_dn.read(path=epw_down_path, prefix=epw_prefix_dn, epmat_ncfile="epmat.nc")

        basis, _ = auto_assign_basis_name(tbmodel.xred, atoms)

        exchange = ExchangePert2(
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
            Rcut=Rcut,
            exclude_orbs=exclude_orbs,
            list_iatom=list_iatom,
            description=description,
        )
        exchange.set_epw(Ru, imode=idisp, epmat_up=epw_up, epmat_dn=epw_dn)
        exchange.run(output_path)


if __name__ == "__main__":
    # for imode in range(15):
    # for imode in range(3, 15):
    path = "/home/hexu/spinphon/2025-10-02_newdata/k555q555"
    for idisp in [0]:
        gen_exchange_Oiju_epw(
            path=path,
            colinear=True,
            posfile="scf.pwi",
            # prefix_up='wannier90.up',
            # prefix_dn='wannier90.dn',
            prefix_up="up/SrMnO3",
            prefix_dn="down/SrMnO3.down",
            prefix_SOC="wannier90",
            epw_up_path=f"{path}/up",
            epw_down_path=f"{path}/down",
            epw_prefix_up="epmat",
            epw_prefix_dn="epmat",
            idisp=idisp,
            Ru=(0, 0, 0),
            Rcut=8,
            efermi=11.26,
            magnetic_elements=["Mn"],
            kmesh=[5, 5, 5],
            emin=-7.3363330034071295,
            emax=0.0,
            nz=70,
            np=1,
            exclude_orbs=[],
            description="",
            # list_iatom=[1],
            output_path=f"VT_{idisp}_withcutoff0.1",
        )
