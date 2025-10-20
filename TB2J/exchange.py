import os
import pickle
from collections import defaultdict
from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.contour import Contour
from TB2J.exchange_params import ExchangeParams
from TB2J.external import p_map
from TB2J.green import TBGreen
from TB2J.io_exchange import SpinIO
from TB2J.mycfr import CFR
from TB2J.orbmap import map_orbs_matrix
from TB2J.pauli import pauli_block_all, pauli_block_sigma_norm
from TB2J.utils import (
    kmesh_to_R,
    symbol_number,
)


class Exchange(ExchangeParams):
    def __init__(self, tbmodels, atoms, **params):
        self.atoms = atoms
        super().__init__(**params)
        self._prepare_kmesh(self._kmesh)
        self._prepare_Rlist()
        self.set_tbmodels(tbmodels)
        self._adjust_emin()
        # self._prepare_elist(method="CFR")
        self._prepare_elist(method="legendre")
        self._prepare_basis()
        self._prepare_orb_dict()
        self._prepare_distance()

        # whether to calculate J and DMI with NJt method.
        # self._prepare_NijR()
        self._is_collinear = True
        self.has_elistc = False

        # Store overlap matrix before cleaning tbmodels
        if hasattr(self, "tbmodel") and hasattr(self.tbmodel, "SR"):
            # Find R=0 index in tbmodel.Rlist
            iR_S0 = np.argmin(np.linalg.norm(self.tbmodel.Rlist, axis=1))
            self.S_R0 = self.tbmodel.SR[iR_S0]  # R=0 overlap matrix
        else:
            self.S_R0 = None

        self._clean_tbmodels()

        # Initialize storage for Green's function diagonals (for charge and magnetic moment calculation)
        self.G_diagonal = {iatom: [] for iatom in range(len(self.atoms))}

    def _prepare_Jorb_file(self):
        os.makedirs(self.output_path, exist_ok=True)
        self.orbpath = os.path.join(self.output_path, "OrbResolve")
        os.makedirs(self.orbpath, exist_ok=True)

    def _adjust_emin(self):
        self.emin = self.G.adjusted_emin
        # self.emin = self.G.find_energy_ingap(rbound=self.efermi - 15.0) - self.efermi
        # print(f"A gap is found at {self.emin}, set emin to it.")

    def set_tbmodels(self, tbmodels):
        pass

    def _clean_tbmodels(self):
        # del self.tbmodel
        # del self.G.tbmodel
        pass

    def _prepare_kmesh(self, kmesh, ibz=False):
        for k in kmesh:
            self.kmesh = list(map(lambda x: x // 2 * 2 + 1, kmesh))

    def _prepare_elist(self, method="CFR"):
        """
        prepare list of energy for integration.
        The path has three segments:
         emin --1-> emin + 1j*height --2-> emax+1j*height --3-> emax
        """
        # if method.lower() == "rectangle":
        #    self.contour.build_path_rectangle(
        #        height=self.height, nz1=self.nz1, nz2=self.nz2, nz3=self.nz3
        #    )
        if method.lower() == "semicircle":
            self.contour = Contour(self.emin, self.emax)
            self.contour.build_path_semicircle(npoints=self.nz, endpoint=True)
        elif method.lower() == "legendre":
            self.contour = Contour(self.emin, self.emax)
            self.contour.build_path_legendre(npoints=self.nz, endpoint=True)
        elif method.lower() == "cfr":
            self.contour = CFR(nz=self.nz, T=600)
        else:
            raise ValueError(f"The path cannot be of type {method}.")

    def _prepare_Rlist(self):
        """
        prepare R list for J(i, j, R)
        [-Rx, Rx] * [-Ry, Ry] * [-Rz, Rz]
        """
        self.Rlist = kmesh_to_R(self.kmesh)

    def _prepare_basis(self):
        if self.basis is None:
            pass

    def _prepare_orb_dict(self):
        """
        Generate orbital and magnetic atom mappings needed for exchange calculations.

        Creates:
        - self.orb_dict: Maps atom indices to their orbital indices
        - self.labels: Maps atom indices to their orbital labels
        - self.ind_mag_atoms: List of indices of magnetic atoms
        - self._spin_dict: Maps atom indices to spin indices
        - self._atom_dict: Maps spin indices back to atom indices
        """
        self._create_orbital_mappings()
        self._identify_magnetic_atoms()
        self._validate_orbital_assignments()
        self._create_spin_mappings()
        self._prepare_orb_mmat()

    def _create_orbital_mappings(self):
        """Create mappings between atoms and their orbitals."""
        self.orb_dict = {}  # {atom_index: [orbital_indices]}
        self.labels = {}  # {atom_index: [orbital_labels]}
        atom_symbols = symbol_number(self.atoms)

        for orb_idx, base in enumerate(self.basis):
            if orb_idx in self.exclude_orbs:
                continue

            # Extract atom and orbital info
            if isinstance(base, str):
                atom_sym, orb_sym = base.split("|")[:2]
                atom_idx = atom_symbols[atom_sym]
            else:
                try:
                    atom_sym, orb_sym = base[:2]
                    atom_idx = atom_symbols[atom_sym]
                except Exception:
                    atom_idx = base.iatom
                    atom_sym = base.iatom
                    orb_sym = base.sym

            # Update orbital mappings
            if atom_idx not in self.orb_dict:
                self.orb_dict[atom_idx] = [orb_idx]
                self.labels[atom_idx] = [orb_sym]
            else:
                self.orb_dict[atom_idx].append(orb_idx)
                self.labels[atom_idx].append(orb_sym)

    def _identify_magnetic_atoms(self):
        """Identify which atoms are magnetic based on elements/tags."""
        if self.index_magnetic_atoms is not None:
            self.ind_mag_atoms = self.index_magnetic_atoms
        else:
            self.ind_mag_atoms = []
            symbols = self.atoms.get_chemical_symbols()
            tags = self.atoms.get_tags()

            for atom_idx, (sym, tag) in enumerate(zip(symbols, tags)):
                if (
                    sym in self.magnetic_elements
                    or f"{sym}{tag}" in self.magnetic_elements
                ):
                    self.ind_mag_atoms.append(atom_idx)

    def _validate_orbital_assignments(self):
        """Validate that magnetic atoms have proper orbital assignments."""
        # Check all magnetic atoms have orbitals
        for atom_idx in self.ind_mag_atoms:
            if atom_idx not in self.orb_dict:
                raise ValueError(
                    f"Atom {atom_idx} is magnetic but has no orbitals assigned. "
                    "Check Wannier function definitions."
                )

        # For non-collinear case, check spin-orbital pairing
        if not self._is_collinear:
            for atom_idx, orbitals in self.orb_dict.items():
                if len(orbitals) % 2 != 0:
                    raise ValueError(
                        f"Atom {atom_idx} has {len(orbitals)} spin-orbitals "
                        "(should be even). Check Wannier function localization."
                    )

    def _create_spin_mappings(self):
        """Create mappings between atom indices and spin indices."""
        self._spin_dict = {}  # {atom_index: spin_index}
        self._atom_dict = {}  # {spin_index: atom_index}

        for spin_idx, atom_idx in enumerate(self.ind_mag_atoms):
            self._spin_dict[atom_idx] = spin_idx
            self._atom_dict[spin_idx] = atom_idx

    def _prepare_orb_mmat(self):
        self.mmats = {}
        self.orbital_names = {}
        self.norb_reduced = {}
        if self.backend_name.upper() in ["SIESTA", "ABACUS", "LCAOHAMILTONIAN"]:
            # print(f"magntic_elements: {self.magnetic_elements}")
            # print(f"include_orbs: {self.include_orbs}")
            syms = self.atoms.get_chemical_symbols()
            for iatom, orbs in self.labels.items():
                if (self.include_orbs is not None) and syms[iatom] in self.include_orbs:
                    mmat, reduced_orbs = map_orbs_matrix(
                        orbs,
                        spinor=not (self._is_collinear),
                        include_only=self.include_orbs[syms[iatom]],
                    )
                else:
                    mmat, reduced_orbs = map_orbs_matrix(
                        orbs, spinor=not (self._is_collinear), include_only=None
                    )

                self.mmats[iatom] = mmat
                self.orbital_names[iatom] = reduced_orbs
                # Note that for siesta, spin up and spin down has same orb name.
                # Therefor there is no nedd to /2
                self.norb_reduced[iatom] = len(reduced_orbs)
        else:
            self.orbital_names = self.labels
            for iatom, orbs in self.labels.items():
                # Note that for siesta, spin up and spin down has same orb name.
                # thus //2
                self.norb_reduced[iatom] = len(orbs) // 2

    def ispin(self, iatom):
        return self._spin_dict[iatom]

    def iatom(self, ispin):
        return self._atom_dict[ispin]

    def _prepare_distance(self):
        """
        prepare the distance between atoms.
        """
        self.distance_dict = {}
        self.short_Rlist = []  # Will contain actual R vectors, not indices
        self.R_ijatom_dict = defaultdict(lambda: [])
        ind_matoms = self.ind_mag_atoms

        # First pass: identify which R vectors are within Rcut
        # Add both R and -R when within cutoff
        valid_R_vectors = set()
        for R in self.Rlist:
            for ispin, iatom in enumerate(ind_matoms):
                for jspin, jatom in enumerate(ind_matoms):
                    pos_i = self.atoms.get_positions()[iatom]
                    pos_jR = self.atoms.get_positions()[jatom] + np.dot(
                        R, self.atoms.get_cell()
                    )
                    vec = pos_jR - pos_i
                    distance = np.sqrt(np.sum(vec**2))
                    if self.Rcut is None or distance < self.Rcut:
                        R_tuple = tuple(R)
                        valid_R_vectors.add(R_tuple)
                        valid_R_vectors.add(tuple(-x for x in R_tuple))

        # Sort the valid_R_vectors
        self.short_Rlist = sorted(valid_R_vectors)
        # print(f"short_Rlist contains {len(self.short_Rlist)} R vectors, which are: {self.short_Rlist}")

        # Second pass: build dictionaries using the clean indexing
        for iR, R_vec in enumerate(self.short_Rlist):
            for ispin, iatom in enumerate(ind_matoms):
                for jspin, jatom in enumerate(ind_matoms):
                    pos_i = self.atoms.get_positions()[iatom]
                    pos_jR = self.atoms.get_positions()[jatom] + np.dot(
                        R_vec, self.atoms.get_cell()
                    )
                    vec = pos_jR - pos_i
                    distance = np.sqrt(np.sum(vec**2))
                    if self.Rcut is None or distance < self.Rcut:
                        self.distance_dict[(R_vec, ispin, jspin)] = (vec, distance)
                        self.R_ijatom_dict[iR].append((iatom, jatom))

        # Create lookup dictionary for negative R vectors
        self.Rvec_to_shortlist_idx = {
            R_vec: iR for iR, R_vec in enumerate(self.short_Rlist)
        }
        self.R_negative_index = {}
        for iR, R_vec in enumerate(self.short_Rlist):
            Rm_vec = tuple(-x for x in R_vec)
            if Rm_vec in self.Rvec_to_shortlist_idx:
                self.R_negative_index[iR] = self.Rvec_to_shortlist_idx[Rm_vec]
            else:
                self.R_negative_index[iR] = None  # No negative R found

        # Verify the R vector pairing
        pairing_good = True
        for iR, R_vec in enumerate(self.short_Rlist):
            neg_idx = self.R_negative_index[iR]
            if neg_idx is not None:
                expected_neg = tuple(-x for x in R_vec)
                actual_neg = self.short_Rlist[neg_idx]
                if expected_neg != actual_neg:
                    print(
                        f"  R[{iR}] = {R_vec} -> -R[{neg_idx}] = {actual_neg} ✗ (expected {expected_neg})"
                    )
                    pairing_good = False
            else:
                print(f"  R[{iR}] = {R_vec} -> No negative R found")

        if not pairing_good:
            raise ValueError("R vector pairing check failed.")

    def iorb(self, iatom):
        """
        given an index of atom, return all the orbital indices of it.
        :param iatom:  index of atom.
        """
        return np.array(self.orb_dict[iatom], dtype=int)

    def simplify_orbital_contributions(self, Jorbij, iatom, jatom):
        """
        sum up the contribution of all the orbitals with same (n,l,m)
        """
        if self.backend_name.upper() in ["SIESTA", "ABACUS", "LCAOHAMILTONIAN"]:
            mmat_i = self.mmats[iatom]
            mmat_j = self.mmats[jatom]
            Jorbij = mmat_i.T @ Jorbij @ mmat_j
        return Jorbij

    def calculate_all(self):
        raise NotImplementedError(
            "calculate all is not implemented in a abstract class. "
        )

    def write_output(self):
        raise NotImplementedError(
            "write_output is not implemented in a abstract class. "
        )


class ExchangeNCL(Exchange):
    """
    Non-collinear exchange
    """

    def set_tbmodels(self, tbmodels):
        """
        tbmodels should be in spinor form.
        The basis should be orb1_up, orb2_up,...orbn_up, orb1_dn, orb2_dn....
        """
        self.tbmodel = tbmodels
        self.backend_name = self.tbmodel.name
        self.G = TBGreen(
            tbmodel=self.tbmodel,
            kmesh=self.kmesh,
            ibz=self.ibz,
            gamma=True,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
            initial_emin=self.emin,
        )
        if self.efermi is None:
            self.efermi = self.G.efermi
        self.norb = self.G.norb
        self.nbasis = self.G.nbasis
        # self.rho = np.zeros((self.nbasis, self.nbasis), dtype=complex)
        self.rho = self.G.get_density_matrix()
        self.A_ijR_list = defaultdict(lambda: [])
        self.A_ijR = defaultdict(lambda: np.zeros((4, 4), dtype=complex))
        self.A_ijR_orb = dict()
        # self.HR0 = self.tbmodel.get_H0()
        # if hasattr(self.tbmodel, "get_H0"):
        #    self.HR0 = self.tbmodel.get_H0()
        # else:
        #    self.HR0 = self.G.H0
        self.HR0 = self.G.H0
        self._is_collinear = False
        self.Pdict = {}
        if self.write_density_matrix:
            self.G.write_rho_R()

    def get_MAE(self, thetas, phis):
        """
        Calculate the magnetic anisotropy energy.
        """
        pass

    def _prepare_NijR(self):
        self.N = {}
        for R in self.Rlist:
            self.N[R] = np.zeros((self.nbasis, self.nbasis), dtype=complex)

    def _prepare_Patom(self):
        for iatom in self.ind_mag_atoms:
            self.Pdict[iatom] = pauli_block_sigma_norm(self.get_H_atom(iatom))

    def get_H_atom(self, iatom):
        orbs = self.iorb(iatom)
        # return self.HR0[self.orb_slice[iatom], self.orb_slice[iatom]]
        return self.HR0[np.ix_(orbs, orbs)]

    def get_P_iatom(self, iatom):
        """Calculate the norm of the Hamiltonian vector.
        For each atom, the local hamiltonian of each orbital H(2*2 matrix)
        can be written as H0* I + H1* sigma1 + H2*sigma2 + H3 *sigma3
        where sigma is a Pauli matrix. return the norm of (H1, H2, H3) vector
        :param iatom: index of atom
        :returns: a matrix of norms P. P[i, j] is for orbital i and j.
        :rtype: complex matrix of shape norb_i * norb_i.
        """
        if self.Pdict == {}:
            self._prepare_Patom()
        return self.Pdict[iatom]

    def GR_atom(self, GR, iatom, jatom):
        """Given a green's function matrix, return the [iatom, jatom] component.

        :param GR:  Green's function matrix
        :param iatom:  index of atom i
        :param jatom:  index of atom j
        :returns:  G_ij
        :rtype:  complex matrix.

        """
        orbi = self.iorb(iatom)
        orbj = self.iorb(jatom)
        return GR[np.ix_(orbi, orbj)]
        # return GR[self.orb_slice[iatom], self.orb_slice[jatom]]

    def get_A_ijR(self, G, iR, iatom, jatom):
        """calculate A from G for a energy slice (de).
        It take the
        .. math::
           A^{uv} = p T^u p T^v dE / pi

        where u, v are I, x, y, z (index 0, 1,2,3). p(i) = self.get_P_iatom(iatom)
        T^u(ijR)  (u=0,1,2,3) = pauli_block_all(G)

        :param G: Green's function for all R, i, j (numpy array).
        :param iR: index in short_Rlist (position in G array)
        :param iatom: i
        :param jatom: j
        :returns: a matrix of A_ij(u, v), where u, v =(0)0, x(1), y(2), z(3)
        :rtype:  4*4 matrix
        """
        GR = G[iR]
        Gij = self.GR_atom(GR, iatom, jatom)
        Gij_Ixyz = pauli_block_all(Gij)

        # G(j, i, -R) - use optimized lookup
        iRm = self.R_negative_index[iR]
        if iRm is None:
            R_vec = self.short_Rlist[iR]
            Rm_vec = tuple(-x for x in R_vec)
            raise KeyError(f"Negative R vector {Rm_vec} not found in short_Rlist")

        GRm = G[iRm]
        Gji = self.GR_atom(GRm, jatom, iatom)
        Gji_Ixyz = pauli_block_all(Gji)

        ni = self.norb_reduced[iatom]
        nj = self.norb_reduced[jatom]

        tmp = np.zeros((4, 4), dtype=complex)

        if self.orb_decomposition:
            torb = np.zeros((4, 4, ni, nj), dtype=complex)
            # a, b in (0,x,y,z)
            for a in range(4):
                piGij = self.get_P_iatom(iatom) @ Gij_Ixyz[a]
                for b in range(4):
                    pjGji = self.get_P_iatom(jatom) @ Gji_Ixyz[b]
                    torb[a, b] = self.simplify_orbital_contributions(
                        np.einsum("ij, ji -> ij", piGij, pjGji) / np.pi, iatom, jatom
                    )
                    tmp[a, b] = np.sum(torb[a, b])

        else:
            for a in range(4):
                pGp = self.get_P_iatom(iatom) @ Gij_Ixyz[a] @ self.get_P_iatom(jatom)
                for b in range(4):
                    AijRab = pGp @ Gji_Ixyz[b]
                    tmp[a, b] = np.trace(AijRab) / np.pi
            torb = None
        return tmp, torb

    def get_all_A(self, G):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function (numpy array).
        :param de: energy step.
        """
        A_ijR_list = {}
        Aorb_ijR_list = {}
        for iR in self.R_ijatom_dict:
            for iatom, jatom in self.R_ijatom_dict[iR]:
                A, A_orb = self.get_A_ijR(G, iR, iatom, jatom)
                # Store with actual R vector for compatibility with existing code
                R_vec = self.short_Rlist[iR]
                A_ijR_list[(R_vec, iatom, jatom)] = A
                Aorb_ijR_list[(R_vec, iatom, jatom)] = A_orb
        return A_ijR_list, Aorb_ijR_list

    def get_all_A_vectorized(self, GR):
        """
        Vectorized calculation of all A matrix elements.
        Fully vectorized version based on TB2J_optimization_prototype.ipynb.
        Now works with properly ordered short_Rlist.

        :param GR: Green's function array of shape (nR, nbasis, nbasis)
        :returns: tuple of (A_ijR_list, Aorb_ijR_list) with R vector keys
        """

        # Get magnetic sites and their orbital indices
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]

        # Build the P matrices for all magnetic sites using the same method as original
        P = [self.get_P_iatom(site) for site in magnetic_sites]

        # Initialize results dictionary
        A = {}
        A_orb = {}

        # Batch compute all A tensors following the prototype
        for i, j in product(range(len(magnetic_sites)), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]
            Gij = GR[:, idx][:, :, jdx]
            Gji = GR[:, jdx][:, :, idx]
            Gij = pauli_block_all(Gij)
            Gji = pauli_block_all(Gji)
            # NOTE: becareful: this assumes that short_Rlist is properly ordered so that
            # the ith R vector's negative is at -i index.
            Gji = np.flip(Gji, axis=0)
            Pi = P[i]
            Pj = P[j]
            X = Pi @ Gij
            Y = Pj @ Gji
            mi, mj = (magnetic_sites[i], magnetic_sites[j])

            if self.orb_decomposition:
                # Vectorized orbital decomposition over all R vectors at once
                # X.shape: (nR, 4, ni, nj), Y.shape: (nR, 4, nj, ni)
                A_orb_tensor = (
                    np.einsum("ruij,rvji->ruvij", X, Y) / np.pi
                )  # Shape: (nR, 4, 4, ni, nj)
                # Vectorized sum over orbitals for simplified A values
                A_val_tensor = np.sum(A_orb_tensor, axis=(-2, -1))  # Shape: (nR, 4, 4)
            else:
                # Compute A_tensor for all R vectors at once
                A_tensor = (
                    np.einsum("...uij,...vji->...uv", X, Y) / np.pi
                )  # Shape: (nR, 4, 4)
                A_val_tensor = A_tensor  # Use pre-computed A_tensor directly
                A_orb_tensor = None

            # Store results for each R vector
            for iR, R_vec in enumerate(self.short_Rlist):
                A_val = A_val_tensor[iR]  # Shape: (4, 4)
                A_orb_val = A_orb_tensor[iR] if A_orb_tensor is not None else None

                # Store with R vector key for compatibility
                A[(R_vec, mi, mj)] = A_val
                if A_orb_val is not None:
                    A_orb[(R_vec, mi, mj)] = A_orb_val

        return A, A_orb

    def A_to_Jtensor_orb(self):
        """
        convert the orbital composition of A into J, DMI, Jani
        """
        self.Jiso_orb = {}
        self.Jani_orb = {}
        self.DMI_orb = {}

        if self.orb_decomposition:
            for key, val in self.A_ijR_orb.items():
                R, iatom, jatom = key
                # Rm = tuple(-x for x in R)
                # valm = self.A_ijR_orb[(Rm, jatom, iatom)]
                ni = self.norb_reduced[iatom]
                nj = self.norb_reduced[jatom]

                is_nonself = not (R == (0, 0, 0) and iatom == jatom)
                ispin = self.ispin(iatom)
                jspin = self.ispin(jatom)
                keyspin = (R, ispin, jspin)

                # isotropic J
                Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

                # off-diagonal anisotropic exchange
                Ja = np.zeros((3, 3, ni, nj), dtype=float)
                for i in range(3):
                    for j in range(3):
                        Ja[i, j] = np.imag(val[i + 1, j + 1] + val[j + 1, i + 1])
                # DMI

                Dtmp = np.zeros((3, ni, nj), dtype=float)
                for i in range(3):
                    Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

                if is_nonself:
                    self.Jiso_orb[keyspin] = Jiso
                    self.Jani_orb[keyspin] = Ja
                    self.DMI_orb[keyspin] = Dtmp

    def A_to_Jtensor(self):
        """
        Calculate J tensors from A.
        If we assume the exchange can be written as a bilinear tensor form,
        J_{isotropic} = Tr Im (A^{00} - A^{xx} - A^{yy} - A^{zz})
        J_{anisotropic}_uv = Tr Im (2A)
        DMI =  Tr Re (A^{0z} - A^{z0} )
        """
        self.Jani = {}
        self.DMI = {}

        self.Jprime = {}
        self.B = {}
        self.exchange_Jdict = {}
        self.Jiso_orb = {}

        self.debug_dict = {"DMI2": {}}

        for key, val in self.A_ijR.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key

            Rm = tuple(-x for x in R)
            valm = self.A_ijR[(Rm, jatom, iatom)]

            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)

            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            Jiso = 0.0
            Ja = np.zeros((3, 3), dtype=float)
            Dtmp = np.zeros(3, dtype=float)
            Dtmp2 = np.zeros(3, dtype=float)
            # Heisenberg like J.
            Jiso = np.imag(val[0, 0] - val[1, 1] - val[2, 2] - val[3, 3])

            if is_nonself:
                self.exchange_Jdict[keyspin] = Jiso

            # off-diagonal anisotropic exchange
            for i in range(3):
                for j in range(3):
                    Ja[i, j] = np.imag(val[i + 1, j + 1] + valm[i + 1, j + 1])
            if is_nonself:
                self.Jani[keyspin] = Ja

            # DMI
            for i in range(3):
                Dtmp[i] = np.real(val[0, i + 1] - val[i + 1, 0])

            # Dx = Jyz-Jzy
            # Dy = Jzx-Jxz
            # Dz = Jxy-Jyx
            Dtmp2[0] = np.imag(val[2, 3] - val[3, 2])
            Dtmp2[1] = np.imag(val[3, 1] - val[1, 3])
            Dtmp2[2] = np.imag(val[1, 2] - val[2, 1])
            if is_nonself:
                self.DMI[keyspin] = Dtmp
                self.debug_dict["DMI2"][keyspin] = Dtmp2

            # isotropic exchange into bilinear and biqudratic parts:
            # Jprime SiSj and B (SiSj)^2
            if is_nonself:
                Si = self.spinat[iatom]
                Sj = self.spinat[jatom]
                Jprime = np.imag(val[0, 0] - val[3, 3]) - 2 * np.sign(
                    np.dot(Si, Sj)
                ) * np.imag(val[3, 3])
                # Jprime = np.imag(val[0, 0] - 3*val[3, 3])
                B = np.imag(val[3, 3])
                self.B[keyspin] = Jprime, B

    # def get_N_e(self, GR, de):
    #    """
    #    calcualte density matrix for all R,i, j
    #    """
    #    self.N = defaultdict(lambda: 0.0)
    #    for R, G in GR.items():
    #        self.N[R] += -1.0 / np.pi * np.imag(G * de)

    # def get_rho_e(self, rhoR):
    #    """add component to density matrix from a green's function
    #    :param GR: Green's funciton in real space.
    #    """
    #    return -1.0 / np.pi * rhoR[0, 0, 0]

    # def get_total_charges(self):
    #    return np.sum(np.imag(np.diag(self.rho)))

    def get_rho_atom(self):
        """
        calculate charge and spin for each atom.
        """
        rho = {}
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tmp = self.rho[np.ix_(iorb, iorb)]
            # *2 because there is a 1/2 in the paui_block_all function
            rho[iatom] = np.array([np.trace(x) * 2 for x in pauli_block_all(tmp)]).real
            self.charges[iatom] = rho[iatom][0]
            self.spinat[iatom, :] = rho[iatom][1:]
        self.rho_dict = rho
        return self.rho_dict

    def integrate(self, AijRs, AijRs_orb=None, method="simpson"):
        """
        AijRs: a list of AijR,
        wherer AijR: array of ((nR, n, n, 4,4), dtype=complex)
        """
        # if method == "trapezoidal":
        #    integrate = trapezoidal_nonuniform
        # elif method == "simpson":
        #    integrate = simpson_nonuniform
        #

        # self.rho = integrate(self.contour.path, rhoRs)
        for iR in self.R_ijatom_dict:
            R_vec = self.short_Rlist[iR]
            for iatom, jatom in self.R_ijatom_dict[iR]:
                f = AijRs[(R_vec, iatom, jatom)]
                # self.A_ijR[(R_vec, iatom, jatom)] = integrate(self.contour.path, f)
                self.A_ijR[(R_vec, iatom, jatom)] = self.contour.integrate_values(f)

                if self.orb_decomposition:
                    # self.A_ijR_orb[(R_vec, iatom, jatom)] = integrate(
                    #    self.contour.path, AijRs_orb[(R_vec, iatom, jatom)]
                    # )
                    self.A_ijR_orb[(R_vec, iatom, jatom)] = (
                        self.contour.integrate_values(AijRs_orb[(R_vec, iatom, jatom)])
                    )

    def get_quantities_per_e(self, e):
        Gk_all = self.G.get_Gk_all(e)
        # mae = self.get_mae_kspace(Gk_all)
        mae = None
        # TODO: get the MAE from Gk_all
        # short_Rlist now contains actual R vectors
        GR = self.G.get_GR(self.short_Rlist, energy=e, Gk_all=Gk_all)

        # Save diagonal elements of Green's function for charge and magnetic moment calculation
        # Only if debug option is enabled
        if self.debug_options.get("compute_charge_moments", False):
            self.save_greens_function_diagonals(GR, e)

        # TODO: define the quantities for one energy.
        # Use vectorized method for better performance
        try:
            #
            AijR, AijR_orb = self.get_all_A_vectorized(GR)
            # AijR, AijR_orb = self.get_all_A(GR)
        except Exception as e:
            print(f"Vectorized method failed: {e}, falling back to original method")
            AijR, AijR_orb = self.get_all_A(GR)
        return dict(AijR=AijR, AijR_orb=AijR_orb, mae=mae)

    def save_greens_function_diagonals(self, GR, energy):
        """
        Save diagonal elements of Green's function for each atom.
        These will be used to compute charge and magnetic moments.

        :param GR: Green's function array of shape (nR, nbasis, nbasis)
        :param energy: Current energy value
        """
        # For proper charge and magnetic moment calculation, we need to sum over k-points
        # with weights: Σ_k S(k)·G(k)·w(k)
        # Since this function is called for each energy, we'll compute the k-sum here

        # Initialize the k-summed SG matrix for this energy
        nbasis = GR.shape[1]
        SG_ksum = np.zeros((nbasis, nbasis), dtype=complex)

        # Get k-points and weights from Green's function object
        kpts = self.G.kpts
        kweights = self.G.kweights

        # Use the passed energy parameter
        current_energy = energy

        # Sum over all k-points
        for ik, kpt in enumerate(kpts):
            # Get G(k) for current energy
            Gk = self.G.get_Gk(ik, energy=current_energy)

            if not self.G.is_orthogonal:
                Sk = self.G.get_Sk(ik)
                SG_ksum += Sk @ Gk * kweights[ik]
            else:
                # For orthogonal case, S is identity
                SG_ksum += Gk * kweights[ik]

        # Now SG_ksum contains Σ_k S(k)·G(k)·w(k) for this energy

        for iatom in self.orb_dict:
            # Get orbital indices for this atom
            orbi = self.iorb(iatom)
            # Extract diagonal elements for this atom
            G_diag = np.diag(SG_ksum[np.ix_(orbi, orbi)])
            self.G_diagonal[iatom].append(G_diag)

    def compute_charge_and_magnetic_moments(self):
        """
        Compute charge and magnetic moments from stored Green's function diagonals.
        Uses the relation:
        - Charge: n_i = -1/π ∫ Im[Tr(S·G_ii(E))] dE
        - Magnetic moment: m_i = -1/π ∫ Im[Tr(S·σ·G_ii(E))] dE
        where S is the overlap matrix.
        """
        # Only run if debug option is enabled
        if not self.debug_options.get("compute_charge_moments", False):
            # Just use density matrix method directly
            self.get_rho_atom()
            return

        if not hasattr(self, "G_diagonal") or not self.G_diagonal:
            print(
                "Warning: No Green's function diagonals stored. Cannot compute charge and magnetic moments."
            )
            return

        self.charges = np.zeros(len(self.atoms))
        self.spinat = np.zeros((len(self.atoms), 3))

        for iatom in range(len(self.atoms)):
            if not self.G_diagonal[iatom]:
                continue

            # Stack all diagonal elements for this atom
            G_diags = np.array(
                self.G_diagonal[iatom]
            )  # shape: (n_energies, n_orbitals)

            # Integrate over energy using the same contour as exchange calculation
            # Charge: -1/π Im[∫ diag(G) dE]
            integrated_diag = -np.imag(self.contour.integrate_values(G_diags)) / np.pi

            # Sum over orbitals to get total charge
            self.charges[iatom] = np.sum(integrated_diag)

            # For non-collinear case, compute magnetic moments from Green's function
            # Note: The stored diagonals only contain G_ii elements, not the full spin structure
            # For proper magnetic moment calculation, we need the full Green's function matrix
            # Here we'll compute the charge from diagonals and use density matrix for moments

            # The Green's function method can only compute charge from stored diagonals
            gf_charge = np.sum(integrated_diag)

            # For magnetic moments, we would need the full G matrix with spin structure
            # Since only diagonals are stored, we cannot compute magnetic moments from GF method
            # gf_spinat = np.array(
            #    [np.nan, np.nan, np.nan]
            # )  # Placeholder - cannot compute from diagonals

            # Compute using density matrix method
            self.get_rho_atom()  # This computes charges and spinat using density matrix
            dm_spinat = self.spinat[iatom].copy()
            dm_charge = self.charges[iatom]

            # Compare methods if difference is above threshold
            charge_diff = abs(gf_charge - dm_charge)
            threshold = self.debug_options.get("charge_moment_threshold", 1e-4)

            if charge_diff > threshold:
                print(f"Atom {iatom}:")
                print(f"  Green's function charge: {gf_charge:.6f}")
                print(f"  Density matrix charge: {dm_charge:.6f}")
                print(f"  Difference: {charge_diff:.6f} (threshold: {threshold})")
                print(
                    f"  Density matrix magnetic moment: [{dm_spinat[0]:.6f}, {dm_spinat[1]:.6f}, {dm_spinat[2]:.6f}]"
                )
                print(
                    "  Note: Magnetic moments from GF method require full Green's function matrix, not just diagonals"
                )

            # By default, use density matrix output unless debug option says otherwise
            if not self.debug_options.get("use_density_matrix_output", True):
                # Override with Green's function charge (not recommended)
                self.charges[iatom] = gf_charge
                # Magnetic moments cannot be computed from diagonals in non-collinear case

    def save_AijR(self, AijRs, fname):
        result = dict(path=self.contour.path, AijRs=AijRs)
        with open(fname, "wb") as myfile:
            pickle.dump(result, myfile)

    def validate(self):
        """
        Do some sanity check before proceding.
        """
        pass

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        AijRs = {}
        AijRs_orb = {}

        self.validate()

        npole = len(self.contour.path)
        if self.nproc > 1:
            results = p_map(
                self.get_quantities_per_e, self.contour.path, num_cpus=self.nproc
            )
        else:
            results = map(
                self.get_quantities_per_e, tqdm(self.contour.path, total=npole)
            )

        for i, result in enumerate(results):
            for iR in self.R_ijatom_dict:
                R_vec = self.short_Rlist[iR]
                for iatom, jatom in self.R_ijatom_dict[iR]:
                    if (R_vec, iatom, jatom) in AijRs:
                        AijRs[(R_vec, iatom, jatom)].append(
                            result["AijR"][(R_vec, iatom, jatom)]
                        )
                        if self.orb_decomposition:
                            AijRs_orb[(R_vec, iatom, jatom)].append(
                                result["AijR_orb"][(R_vec, iatom, jatom)]
                            )

                    else:
                        AijRs[(R_vec, iatom, jatom)] = []
                        AijRs[(R_vec, iatom, jatom)].append(
                            result["AijR"][(R_vec, iatom, jatom)]
                        )
                        if self.orb_decomposition:
                            AijRs_orb[(R_vec, iatom, jatom)] = []
                            AijRs_orb[(R_vec, iatom, jatom)].append(
                                result["AijR_orb"][(R_vec, iatom, jatom)]
                            )

        # self.save_AijRs(AijRs)
        self.integrate(AijRs, AijRs_orb)
        self.get_rho_atom()

        # Compute charge and magnetic moments from Green's function diagonals
        self.compute_charge_and_magnetic_moments()

        self.A_to_Jtensor()
        self.A_to_Jtensor_orb()

    def _prepare_index_spin(self):
        # index_spin: index in spin hamiltonian of atom. starts from 1. -1 means not considered.
        ind_matoms = []
        self.index_spin = []
        ispin = 0
        symbols = self.atoms.get_chemical_symbols()
        tags = self.atoms.get_tags()
        for i, (sym, tag) in enumerate(zip(symbols, tags)):
            if sym in self.magnetic_elements or f"{sym}{tag}" in self.magnetic_elements:
                ind_matoms.append(i)
                self.index_spin.append(ispin)
                ispin += 1
            else:
                self.index_spin.append(-1)

    def write_output(self, path="TB2J_results"):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=False,
            orbital_names=self.orbital_names,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            Jiso_orb=self.Jiso_orb,
            dmi_ddict=self.DMI,
            Jani_dict=self.Jani,
            DMI_orb=self.DMI_orb,
            Jani_orb=self.Jani_orb,
            biquadratic_Jdict=self.B,
            debug_dict=self.debug_dict,
            description=self.description,
        )
        output.write_all(path=path)
        # with open("TB2J_results/J_orb.pickle", 'wb') as myfile:
        #    pickle.dump({'Jiso_orb': self.Jiso_orb,
        #                 'DMI_orb': self.DMI_orb, 'Jani_orb': self.Jani_orb}, myfile)

    def finalize(self):
        self.G.clean_cache()

    def run(self, path="TB2J_results"):
        self.calculate_all()
        self.write_output(path=path)
        self.finalize()


class ExchangeCL(ExchangeNCL):
    def set_tbmodels(self, tbmodel):
        """
        only difference is a colinear tag.
        """
        super().set_tbmodels(tbmodel)
        self._is_collinear = True

    def write_output(self, path="TB2J_results"):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            orbital_names=self.orbital_names,
            colinear=True,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            dmi_ddict=None,
            biquadratic_Jdict=self.B,
            description=self.description,
        )
        output.write_all(path=path)
