"""
Module for calculating spin-phonon coupling using double-sided finite difference method.

This module provides functionality to calculate spin-phonon coupling parameters
by using double-sided finite difference of Wannier90 tight-binding models with atomic distortions.
It computes exchange parameters for both positive and negative displacements, and then
calculates dJ/dx = (J(+dx) - J(-dx)) / (2*dx).
"""

import os
from collections import defaultdict

import numpy as np
from ase.io import read
from HamiltonIO.epw.epwparser import Epmat
from supercellmap import SupercellMaker

from TB2J.epw import generate_TB_with_distortion
from TB2J.exchange import ExchangeNCL
from TB2J.FDTB import dHdx
from TB2J.myTB import MyTB, merge_tbmodels_spin
from TB2J.utils import auto_assign_basis_name


def epw_to_dHdx(
    epw_path, epw_prefix, imode, Rpprime=(0, 0, 0), scmaker=SupercellMaker(np.eye(3))
):
    """
    Convert EPW data to dHdx using HamiltonIO Epmat parser.

    For a given perturbation at Rp',
    <Rm|Rp'=Rp+Rm|Rk+Rm>
    =H(Rp,Rk)=<0|Rp|Rk> is a matrix of nbasis*nbasis
    First: Rm = Rp'-Rp, Rk+Rm = Rp'-Rp+Rm
    Input: Rplist, Rklist, H
    H: [iRg, iRk, ibasis, ibasis]

    Parameters
    ----------
    epw_path : str
        Path to EPW data files
    epw_prefix : str
        Prefix for EPW files
    imode : int
        Phonon mode index
    Rpprime : tuple, default=(0,0,0)
        R vector for perturbation
    scmaker : SupercellMaker
        Supercell maker object

    Returns
    -------
    dHdx
        Derivative of Hamiltonian with respect to displacement
    """
    ep = Epmat()
    ep.read(path=epw_path, prefix=epw_prefix, epmat_ncfile="epmat.nc")
    n_basis = ep.nwann
    sc_nbasis = n_basis * scmaker.ncell
    sc_RHdict = defaultdict(lambda: np.zeros((sc_nbasis, sc_nbasis), dtype=complex))

    for iRp, Rp in enumerate(ep.Rq):
        Rm = np.array(Rpprime) - np.array(Rp)
        sc_part_i, pair_ind_i = scmaker._sc_R_to_pair_ind(tuple(np.array(Rm)))
        ii = pair_ind_i * n_basis

        for iRk, Rk in enumerate(ep.Rk):
            Rn = np.array(Rk) + np.array(Rm)
            sc_part_j, pair_ind_j = scmaker._sc_R_to_pair_ind(tuple(Rn))

            if tuple(sc_part_i) == (0, 0, 0):
                jj = pair_ind_j * n_basis
                sc_RHdict[tuple(sc_part_j)][ii : ii + n_basis, jj : jj + n_basis] += (
                    ep.get_epmat_Rv_from_RgRk(imode, Rm, Rk)
                )

    Rlist = np.array(list(sc_RHdict.keys()))
    dH = dHdx(Rlist, nbasis=sc_nbasis, dHR=sc_RHdict, dHR2=None, wsdeg=None)
    return dH


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


def compute_dJdx_from_exchanges(
    exchange_orig, exchange_neg, exchange_pos, amplitude, output_path, compute_d2J=True
):
    """
    Compute dJ/dx and optionally d2J/dx2 from three exchange calculations using double-sided finite difference.

    dJ/dx = (J(+dx) - J(-dx)) / (2*dx)
    d2J/dx2 = (J(+dx) - 2*J(0) + J(-dx)) / (dx)^2

    Parameters
    ----------
    exchange_orig : ExchangeNCL
        Exchange object for original (zero displacement) structure
        Can be None if compute_d2J=False
    exchange_neg : ExchangeNCL
        Exchange object for negative displacement
    exchange_pos : ExchangeNCL
        Exchange object for positive displacement
    amplitude : float
        Amplitude of displacement in Angstrom
    output_path : str
        Path to save dJ/dx results
    compute_d2J : bool, default=True
        Whether to compute second derivative d2J/dx2

    Notes
    -----
    Units:
    - Exchange parameters J: meV (output), eV (internal storage in TB2J)
    - Displacement amplitude: Angstrom
    - Output dJ/dx: meV/Angstrom
    - Output d2J/dx2: meV/Angstrom^2

    Important: TB2J stores exchange values internally in eV but outputs them in meV.
    This function multiplies by 1000 to convert from eV/Angstrom to meV/Angstrom.
    """
    os.makedirs(output_path, exist_ok=True)

    # Get exchange dictionaries from the ExchangeNCL objects
    # The attributes are: exchange_Jdict (isotropic), DMI, Jani
    # IMPORTANT: These values are stored in eV internally!
    Jiso_neg = exchange_neg.exchange_Jdict
    Jiso_pos = exchange_pos.exchange_Jdict

    # Get distance information (use negative exchange object if original is not available)
    if exchange_orig is not None:
        distance_dict = exchange_orig.distance_dict
        Jiso_orig = exchange_orig.exchange_Jdict
    else:
        distance_dict = exchange_neg.distance_dict
        Jiso_orig = None

    # Compute dJ_iso/dx and optionally d2J_iso/dx2 for each pair
    # Note: Exchange values are stored internally in eV, so multiply by 1000 to get meV
    # Note: If ispin0_only=True, the exchange objects will only contain pairs where ispin=0
    dJiso_dx = {}
    d2Jiso_dx2 = {}

    for key in Jiso_pos.keys():
        if key in Jiso_neg:
            # First derivative: (J_pos - J_neg) / (2*dx)
            # Convert from eV/Angstrom to meV/Angstrom
            dJiso_dx[key] = (Jiso_pos[key] - Jiso_neg[key]) / (2.0 * amplitude) * 1e3

            # Second derivative: (J_pos - 2*J_0 + J_neg) / (dx)^2
            # Convert from eV/Angstrom^2 to meV/Angstrom^2
            if compute_d2J and Jiso_orig is not None:
                if key in Jiso_orig:
                    d2Jiso_dx2[key] = (
                        (Jiso_pos[key] - 2.0 * Jiso_orig[key] + Jiso_neg[key])
                        / (amplitude**2)
                        * 1e3
                    )
        else:
            if key not in Jiso_neg:
                print(f"Warning: Key {key} not found in negative displacement results")

    # Write results to text file
    output_file = os.path.join(output_path, "exchange.out")
    with open(output_file, "w") as f:
        # Write header with appropriate information
        if compute_d2J:
            f.write(
                "# dJ/dx and d2J/dx2 computed using double-sided finite difference\n"
            )
        else:
            f.write("# dJ/dx computed using double-sided finite difference\n")
        f.write(f"# dJ/dx = (J(+{amplitude}) - J(-{amplitude})) / (2*{amplitude})\n")
        if compute_d2J:
            f.write(
                f"# d2J/dx2 = (J(+{amplitude}) - 2*J(0) + J(-{amplitude})) / ({amplitude})^2\n"
            )
        f.write("# Units: J in meV, distances in Angstrom, dJ/dx in meV/Angstrom")
        if compute_d2J:
            f.write(", d2J/dx2 in meV/Angstrom^2\n")
        else:
            f.write("\n")
        f.write("# Sorted by: first atom index (ispin), then by distance\n")

        # Format line depends on whether d2J is computed and whether J_0 is available
        if compute_d2J and Jiso_orig is not None:
            # 14 columns: includes J_0 and d2J/dx2
            f.write(
                "# Format: ispin jspin Rx Ry Rz distance(A) dx(A) dy(A) dz(A) J_0(meV) J_neg(meV) J_pos(meV) dJ/dx d2J/dx2\n"
            )
        elif Jiso_orig is None:
            # 12 columns: no J_0, optionally has d2J/dx2 (though d2J needs J_0)
            if compute_d2J:
                f.write(
                    "# Format: ispin jspin Rx Ry Rz distance(A) dx(A) dy(A) dz(A) J_neg(meV) J_pos(meV) dJ/dx d2J/dx2\n"
                )
            else:
                f.write(
                    "# Format: ispin jspin Rx Ry Rz distance(A) dx(A) dy(A) dz(A) J_neg(meV) J_pos(meV) dJ/dx\n"
                )
        else:
            # 13 columns: includes J_0 but no d2J/dx2
            f.write(
                "# Format: ispin jspin Rx Ry Rz distance(A) dx(A) dy(A) dz(A) J_0(meV) J_neg(meV) J_pos(meV) dJ/dx\n"
            )
        f.write("#" + "=" * 79 + "\n")

        # Sort keys by first atom index (ispin), then by distance
        sorted_keys = sorted(dJiso_dx.keys(), key=lambda x: (x[1], distance_dict[x][1]))

        for key in sorted_keys:
            # Key format is (R, ispin, jspin) where R is a tuple
            R, ispin, jspin = key
            dJ_iso = dJiso_dx[key]

            # Get J values (convert to meV)
            J_neg = Jiso_neg[key] * 1e3
            J_pos = Jiso_pos[key] * 1e3

            # Get distance vector and norm
            vec, distance = distance_dict[key]

            f.write(f"{ispin:4d} {jspin:4d} {R[0]:4d} {R[1]:4d} {R[2]:4d} ")
            f.write(f"{distance:8.4f} {vec[0]:10.6f} {vec[1]:10.6f} {vec[2]:10.6f} ")

            # Only output J_0 if it was computed
            if Jiso_orig is not None:
                J_0 = Jiso_orig[key] * 1e3
                f.write(f"{J_0.real:12.6f} ")

            f.write(f"{J_neg.real:12.6f} {J_pos.real:12.6f} ")
            f.write(f"{dJ_iso.real:12.6f}")

            if compute_d2J and key in d2Jiso_dx2:
                d2J_iso = d2Jiso_dx2[key]
                f.write(f" {d2J_iso.real:12.6f}")
            f.write("\n")

    if compute_d2J:
        print(f"dJ/dx and d2J/dx2 results written to {output_file}")
    else:
        print(f"dJ/dx results written to {output_file}")

    # Also save as pickle for easy loading
    import pickle

    results = {
        "dJiso_dx": dJiso_dx,
        "Jiso_neg": {k: v * 1e3 for k, v in Jiso_neg.items()},  # Convert to meV
        "Jiso_pos": {k: v * 1e3 for k, v in Jiso_pos.items()},  # Convert to meV
        "amplitude": amplitude,
    }
    if Jiso_orig is not None:
        results["Jiso_orig"] = {
            k: v * 1e3 for k, v in Jiso_orig.items()
        }  # Convert to meV
    if compute_d2J:
        results["d2Jiso_dx2"] = d2Jiso_dx2

    with open(os.path.join(output_path, "dJdx.pickle"), "wb") as f:
        pickle.dump(results, f)

    print(f"dJ/dx dictionary saved to {os.path.join(output_path, 'dJdx.pickle')}")
    # Build keys list based on what was computed
    keys_list = "'dJiso_dx', 'Jiso_neg', 'Jiso_pos', 'amplitude'"
    if Jiso_orig is not None:
        keys_list = "'dJiso_dx', 'Jiso_orig', 'Jiso_neg', 'Jiso_pos', 'amplitude'"
    if compute_d2J:
        if Jiso_orig is not None:
            keys_list = "'dJiso_dx', 'd2Jiso_dx2', 'Jiso_orig', 'Jiso_neg', 'Jiso_pos', 'amplitude'"
        else:
            keys_list = "'dJiso_dx', 'd2Jiso_dx2', 'Jiso_neg', 'Jiso_pos', 'amplitude'"
    print(f"  Keys: {keys_list}")


def gen_exchange_Oiju_FD_double_sided(
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
    compute_d2J=True,
    ispin0_only=False,
):
    """Calculate spin-phonon coupling and dJ/dx using double-sided finite difference.

    This function implements double-sided finite difference to compute exchange parameters
    for both negative and positive displacements, and then calculates the derivative
    dJ/dx = (J(+dx) - J(-dx)) / (2*dx).

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
    supercell_matrix : array_like, optional
        Supercell matrix for calculations. If None, uses identity (no supercell)
    amplitude : float, default=0.01
        Amplitude of atomic displacement for finite difference in Angstrom
    max_distance : float, optional
        Maximum distance for interactions (not used)
    compute_d2J : bool, default=True
        Whether to compute second derivative d2J/dx2
    ispin0_only : bool, default=False
        If True, only compute exchanges where either ispin=0 or jspin=0.
        This significantly reduces computation time when only interactions
        involving the first magnetic atom are needed.

    Returns
    -------
    None
        Results are written to output_path/idisp{idisp}_Ru{Rx}_{Ry}_{Rz}/ with four subdirectories:
        - original/: Exchange parameters (meV) at zero displacement
        - negative/: Exchange parameters (meV) at -amplitude displacement
        - positive/: Exchange parameters (meV) at +amplitude displacement
        - dJdx/: Computed derivatives dJ/dx and optionally d2J/dx2

    Notes
    -----
    Units:
    - Exchange parameters J: meV
    - Displacement amplitude: Angstrom
    - Output dJ/dx: meV/Angstrom
    - Output d2J/dx2: meV/Angstrom^2 (if computed)

    Method:
    - Uses double-sided finite difference:
      * dJ/dx = (J(+dx) - J(-dx)) / (2*dx)
      * d2J/dx2 = (J(+dx) - 2*J(0) + J(-dx)) / (dx)^2 (optional)
    - Three calculations are performed: original, negative, and positive displacements
    - First derivative is always computed; second derivative is optional
    - Results are saved in output_path/idisp{idisp}_Ru{Rx}_{Ry}_{Rz}/ subdirectories
    """
    atoms = read(os.path.join(path, posfile))

    if supercell_matrix is None:
        supercell_matrix = np.eye(3)

    # Create output directory with idisp and Ru information
    idisp_output_path = os.path.join(
        output_path, f"idisp{idisp}_Ru{Ru[0]}_{Ru[1]}_{Ru[2]}"
    )
    os.makedirs(idisp_output_path, exist_ok=True)

    if colinear:
        # Read Wannier90 tight-binding models
        tbmodel_up = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_up, atoms=atoms, nls=False
        )
        tbmodel_dn = MyTB.read_from_wannier_dir(
            path=path, prefix=prefix_dn, atoms=atoms, nls=False
        )
        tbmodel = merge_tbmodels_spin(tbmodel_up, tbmodel_dn)

        # Set atoms in tbmodel (needed for make_supercell)
        tbmodel.set_atoms(atoms)

        # Create supercell maker
        scmaker = SupercellMaker(supercell_matrix, center=False)

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

        # Generate tight-binding models with positive and negative displacements
        dtb_neg = generate_TB_with_distortion(tbmodel, -amplitude, dH)
        dtb_pos = generate_TB_with_distortion(tbmodel, amplitude, dH)

        # Calculate exchange for original (zero displacement) structure
        # Only needed if computing second derivative d2J/dx2
        if compute_d2J:
            exchange_orig = ExchangeNCL(
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
                ispin0_only=ispin0_only,
            )
            exchange_orig.run(os.path.join(idisp_output_path, "original"))
        else:
            exchange_orig = None

        # Calculate exchange for negative displacement
        exchange_neg = ExchangeNCL(
            tbmodels=dtb_neg,
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
            ispin0_only=ispin0_only,
        )
        exchange_neg.run(os.path.join(idisp_output_path, "negative"))

        # Calculate exchange for positive displacement
        exchange_pos = ExchangeNCL(
            tbmodels=dtb_pos,
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
            ispin0_only=ispin0_only,
        )
        exchange_pos.run(os.path.join(idisp_output_path, "positive"))

        # Compute dJ/dx and optionally d2J/dx2 using double-sided finite difference
        print("\n" + "=" * 80)
        if compute_d2J:
            print("Computing dJ/dx and d2J/dx2 using double-sided finite difference")
            print(f"dJ/dx = (J(+{amplitude}) - J(-{amplitude})) / (2*{amplitude})")
            print(
                f"d2J/dx2 = (J(+{amplitude}) - 2*J(0) + J(-{amplitude})) / ({amplitude})^2"
            )
        else:
            print("Computing dJ/dx using double-sided finite difference")
            print(f"dJ/dx = (J(+{amplitude}) - J(-{amplitude})) / (2*{amplitude})")
        print("=" * 80 + "\n")

        compute_dJdx_from_exchanges(
            exchange_orig,
            exchange_neg,
            exchange_pos,
            amplitude,
            os.path.join(idisp_output_path, "dJdx"),
            compute_d2J=compute_d2J,
        )

    else:
        raise NotImplementedError("Non-collinear case not yet implemented")


if __name__ == "__main__":
    # Example usage with double-sided finite difference
    path = "/home/hexu/projects/TB2J_examples/Wannier/SrMnO3_QE_Wannier90/W90"

    for idisp in range(15):
        gen_exchange_Oiju_FD_double_sided(
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
            output_path="FD333_results",
        )
