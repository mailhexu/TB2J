"""
Exchange from Green's function
"""

import os
from collections import defaultdict
from itertools import product

import numpy as np
from tqdm import tqdm

from TB2J.external import p_map
from TB2J.green import TBGreen
from TB2J.io_exchange import SpinIO

from .exchange import ExchangeCL


class ExchangeCL2(ExchangeCL):
    def set_tbmodels(self, tbmodels):
        """
        only difference is a colinear tag.
        """
        self.tbmodel_up, self.tbmodel_dn = tbmodels
        self.backend_name = self.tbmodel_up.name
        self.Gup = TBGreen(
            tbmodel=self.tbmodel_up,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
        )
        self.Gdn = TBGreen(
            tbmodel=self.tbmodel_dn,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
        )
        if self.write_density_matrix:
            self.Gup.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_up.pickle")
            )
            self.Gdn.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_dn.pickle")
            )
        self.norb = self.Gup.norb
        self.nbasis = self.Gup.nbasis + self.Gdn.nbasis
        # self.rho_up_list = []
        # self.rho_dn_list = []
        self.rho_up = self.Gup.get_density_matrix()
        self.rho_dn = self.Gdn.get_density_matrix()
        self.Jorb_list = defaultdict(lambda: [])
        self.JJ_list = defaultdict(lambda: [])
        self.JJ = defaultdict(lambda: 0.0j)
        self.Jorb = defaultdict(lambda: 0.0j)
        self.HR0_up = self.Gup.H0
        self.HR0_dn = self.Gdn.H0
        self.Delta = self.HR0_up - self.HR0_dn
        if self.Gup.is_orthogonal and self.Gdn.is_orthogonal:
            self.is_orthogonal = True
        else:
            self.is_orthogonal = False
            # self.S0=self.Gup.S0

        self.exchange_Jdict = {}
        self.Jiso_orb = {}

        self.biquadratic = False

        self._is_collinear = True

    def _clean_tbmodels(self):
        del self.tbmodel_up
        del self.tbmodel_dn
        del self.Gup.tbmodel
        del self.Gdn.tbmodel

    def _adjust_emin(self):
        emin_up = self.Gup.adjusted_emin
        emin_dn = self.Gdn.adjusted_emin
        self.emin = min(emin_up, emin_dn)
        print(f"A gap is found at {self.emin}, set emin to it.")

    def get_Delta(self, iatom):
        orbs = self.iorb(iatom)
        return self.Delta[np.ix_(orbs, orbs)]
        # s = self.orb_slice[iatom]
        # return self.Delta[s, s]

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

    def get_A_ijR(self, Gup, Gdn, iatom, jatom):
        """
        ! Note: not used. In get_all_A, it is reimplemented.
        """
        Rij_done = set()
        Jorb_list = dict()
        JJ_list = dict()
        for iR, ijpairs in self.R_ijatom_dict.items():
            if (iatom, jatom) in ijpairs and (iR, iatom, jatom) not in Rij_done:
                Gij_up = self.GR_atom(Gup[iR], iatom, jatom)
                iRm = self.R_negative_index[iR]
                if iRm is None:
                    R_vec = self.R_vectors[iR]
                    Rm_vec = tuple(-x for x in R_vec)
                    raise KeyError(f"Negative R vector {Rm_vec} not found in R_vectors")
                Gji_dn = self.GR_atom(Gdn[iRm], jatom, iatom)
                tmp = 0.0j
                Deltai = self.get_Delta(iatom)
                Deltaj = self.get_Delta(jatom)
                t = np.einsum(
                    "ij, ji-> ij", np.matmul(Deltai, Gij_up), np.matmul(Deltaj, Gji_dn)
                )

                # if self.biquadratic:
                #    A = np.einsum(
                #        "ij, ji-> ij",
                #        np.matmul(Deltai, Gij_up),
                #        np.matmul(Deltaj, Gji_up),
                #    )
                #    C = np.einsum(
                #        "ij, ji-> ij",
                #        np.matmul(Deltai, Gij_down),
                #        np.matmul(Deltaj, Gji_down),
                #    )
                tmp = np.sum(t)
                self.Jorb_list[(iR, iatom, jatom)].append(t / (4.0 * np.pi))
                self.JJ_list[(iR, iatom, jatom)].append(tmp / (4.0 * np.pi))
                Rij_done.add((iR, iatom, jatom))
                if (iRm, jatom, iatom) not in Rij_done:
                    Jorb_list[(iRm, jatom, iatom)] = t.T / (4.0 * np.pi)
                    JJ_list[(iRm, jatom, iatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((iRm, jatom, iatom))
        return Jorb_list, JJ_list

    def get_all_A(self, Gup, Gdn):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        :param de: energy step.
        """
        Rij_done = set()
        Jorb_list = dict()
        JJ_list = dict()
        for iR in self.R_ijatom_dict:
            for iatom, jatom in self.R_ijatom_dict[iR]:
                if (iR, iatom, jatom) not in Rij_done:
                    iRm = self.R_negative_index[iR]
                    if iRm is None:
                        R_vec = self.short_Rlist[iR]
                        Rm_vec = tuple(-x for x in R_vec)
                        raise KeyError(
                            f"Negative R vector {Rm_vec} not found in short_Rlist"
                        )

                    if (iRm, jatom, iatom) in Rij_done:
                        raise KeyError(
                            f"Strange (iRm, jatom, iatom) has already been calculated! {(iRm, jatom, iatom)}"
                        )
                    Gij_up = self.GR_atom(Gup[iR], iatom, jatom)
                    Gji_dn = self.GR_atom(Gdn[iRm], jatom, iatom)
                    tmp = 0.0j
                    # t = self.get_Delta(iatom) @ Gij_up @ self.get_Delta(jatom) @ Gji_dn
                    t = np.einsum(
                        "ij, ji-> ij",
                        np.matmul(self.get_Delta(iatom), Gij_up),
                        np.matmul(self.get_Delta(jatom), Gji_dn),
                    )
                    tmp = np.sum(t)
                    # Use R vectors for keys for I/O compatibility
                    R_vec = self.short_Rlist[iR]
                    Rm_vec = self.short_Rlist[iRm]
                    Jorb_list[(R_vec, iatom, jatom)] = t / (4.0 * np.pi)
                    JJ_list[(R_vec, iatom, jatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((iR, iatom, jatom))
                    if (iRm, jatom, iatom) not in Rij_done:
                        Jorb_list[(Rm_vec, jatom, iatom)] = t.T / (4.0 * np.pi)
                        JJ_list[(Rm_vec, jatom, iatom)] = tmp / (4.0 * np.pi)
                        Rij_done.add((iRm, jatom, iatom))
        return Jorb_list, JJ_list

    def get_all_A_vectorized(self, Gup, Gdn):
        """
        Vectorized calculation of all A matrix elements for collinear exchange.
        Follows the pattern from TB2J/exchange.py get_all_A_vectorized method.

        :param Gup: Green's function array for spin up, shape (nR, nbasis, nbasis)
        :param Gdn: Green's function array for spin down, shape (nR, nbasis, nbasis)
        :returns: tuple of (Jorb_list, JJ_list) with R vector keys
        """
        # Get magnetic sites and their orbital indices
        magnetic_sites = self.ind_mag_atoms
        iorbs = [self.iorb(site) for site in magnetic_sites]

        # Build the Delta matrices for all magnetic sites
        Delta = [self.get_Delta(site) for site in magnetic_sites]

        # Initialize results dictionaries
        Jorb_list = {}
        JJ_list = {}

        # Batch compute all exchange tensors following the vectorized approach
        for i, j in product(range(len(magnetic_sites)), repeat=2):
            idx, jdx = iorbs[i], iorbs[j]

            # Extract Gij and Gji for all R vectors at once
            Gij = Gup[:, idx][:, :, jdx]  # Shape: (nR, ni, nj)
            # Since short_Rlist is properly ordered, we can flip Gji along R axis
            # to get Gji(-R) for Gij(R)
            Gji = np.flip(Gdn[:, jdx][:, :, idx], axis=0)  # Shape: (nR, nj, ni)

            # Compute exchange tensors for all R vectors at once
            # Following collinear formula: Delta_i @ Gij @ Delta_j @ Gji
            t_tensor = np.einsum(
                "ab,rbc,cd,rda->rac", Delta[i], Gij, Delta[j], Gji, optimize="optimal"
            ) / (4.0 * np.pi)
            tmp_tensor = np.sum(t_tensor, axis=(1, 2))  # Shape: (nR,)

            mi, mj = (magnetic_sites[i], magnetic_sites[j])

            # Store results for each R vector
            for iR, R_vec in enumerate(self.short_Rlist):
                # Store with R vector key for compatibility
                Jorb_list[(R_vec, mi, mj)] = t_tensor[iR]  # Shape: (ni, nj)
                JJ_list[(R_vec, mi, mj)] = tmp_tensor[iR]  # Scalar

        # Filter results to only include atom pairs within cutoff distance
        # This matches the behavior of the original get_all_A method
        filtered_Jorb_list = {}
        filtered_JJ_list = {}

        for iR, atom_pairs in self.R_ijatom_dict.items():
            R_vec = self.short_Rlist[iR]
            for iatom, jatom in atom_pairs:
                key = (R_vec, iatom, jatom)
                if key in Jorb_list:
                    filtered_Jorb_list[key] = Jorb_list[key]
                    filtered_JJ_list[key] = JJ_list[key]

        return filtered_Jorb_list, filtered_JJ_list

    def A_to_Jtensor(self):
        for key, val in self.JJ.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            Jorbij = np.imag(self.Jorb[key]) / np.sign(
                np.dot(self.spinat[iatom], self.spinat[jatom])
            )

            Jij = np.imag(val) / np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))

            if is_nonself:
                self.exchange_Jdict[keyspin] = Jij
                Jsimp = self.simplify_orbital_contributions(Jorbij, iatom, jatom)
                self.Jiso_orb[keyspin] = Jsimp
                self.exchange_Jdict[keyspin] = np.sum(Jsimp)

    def get_rho_e(self, rho_up, rho_dn):
        # self.rho_up_list.append(-1.0 / np.pi * np.imag(rho_up[(0,0,0)]))
        # self.rho_dn_list.append(-1.0 / np.pi * np.imag(rho_dn[(0,0,0)]))
        rup = -1.0 / np.pi * rho_up[(0, 0, 0)]
        rdn = -1.0 / np.pi * rho_dn[(0, 0, 0)]
        return rup, rdn

    def get_rho_atom(self):
        """
        charges and spins from density matrices
        """
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tup = np.real(np.trace(self.rho_up[np.ix_(iorb, iorb)]))
            tdn = np.real(np.trace(self.rho_dn[np.ix_(iorb, iorb)]))
            self.charges[iatom] = tup + tdn
            self.spinat[iatom, 2] = tup - tdn

    def finalize(self):
        self.Gup.clean_cache()
        self.Gdn.clean_cache()
        # path = 'TB2J_results/cache'
        # if os.path.exists(path):
        #    shutil.rmtree(path)

    def integrate(self, method="simpson"):
        # if method == "trapezoidal":
        #    integrate = trapezoidal_nonuniform
        # elif method == "simpson":
        #    integrate = simpson_nonuniform
        for iR in self.R_ijatom_dict:
            for iatom, jatom in self.R_ijatom_dict[iR]:
                # Use R vector key consistently
                R_vec = self.short_Rlist[iR]
                key = (R_vec, iatom, jatom)
                # self.Jorb[key] = integrate(
                #    self.contour.path, self.Jorb_list[key]
                # )
                # self.JJ[key] = integrate(
                #    self.contour.path, self.JJ_list[key]
                # )
                self.Jorb[key] = self.contour.integrate_values(self.Jorb_list[key])
                self.JJ[key] = self.contour.integrate_values(self.JJ_list[key])

    def get_quantities_per_e(self, e):
        # short_Rlist now contains actual R vectors
        GR_up = self.Gup.get_GR(self.short_Rlist, energy=e)
        GR_dn = self.Gdn.get_GR(self.short_Rlist, energy=e)

        # Save diagonal elements of Green's functions for charge and magnetic moment calculation
        # Only if debug option is enabled
        if self.debug_options.get("compute_charge_moments", False):
            self.save_greens_function_diagonals_collinear(GR_up, GR_dn, e)

        # Use vectorized method with fallback to original method
        try:
            Jorb_list, JJ_list = self.get_all_A_vectorized(GR_up, GR_dn)
            # Jorb_list, JJ_list = self.get_all_A(GR_up, GR_dn)
        except Exception as ex:
            print(f"Vectorized method failed: {ex}, falling back to original method")
            Jorb_list, JJ_list = self.get_all_A(GR_up, GR_dn)
        return dict(Jorb_list=Jorb_list, JJ_list=JJ_list)

    def save_greens_function_diagonals_collinear(self, GR_up, GR_dn, energy):
        """
        Save diagonal elements of spin-resolved Green's functions for each atom.
        These will be used to compute charge and magnetic moments in collinear case.

        :param GR_up: Spin-up Green's function array of shape (nR, nbasis, nbasis)
        :param GR_dn: Spin-down Green's function array of shape (nR, nbasis, nbasis)
        :param energy: Current energy value
        """
        # For proper charge and magnetic moment calculation, we need to sum over k-points
        # with weights: Σ_k S(k)·G(k)·w(k)

        # Initialize the k-summed SG matrices for this energy
        nbasis = GR_up.shape[1]
        SG_up_ksum = np.zeros((nbasis, nbasis), dtype=complex)
        SG_dn_ksum = np.zeros((nbasis, nbasis), dtype=complex)

        # Get k-points and weights from Green's function object
        kpts = self.Gup.kpts
        kweights = self.Gup.kweights

        # Use the passed energy parameter
        current_energy = energy

        # Sum over all k-points
        for ik, kpt in enumerate(kpts):
            # Get G(k) for current energy
            Gk_up = self.Gup.get_Gk(ik, energy=current_energy)
            Gk_dn = self.Gdn.get_Gk(ik, energy=current_energy)

            if not self.Gup.is_orthogonal:
                Sk = self.Gup.get_Sk(ik)  # Same overlap for both spins
                SG_up_ksum += Sk @ Gk_up * kweights[ik]
                SG_dn_ksum += Sk @ Gk_dn * kweights[ik]
            else:
                # For orthogonal case, S is identity
                SG_up_ksum += Gk_up * kweights[ik]
                SG_dn_ksum += Gk_dn * kweights[ik]

        # Store both spin channels
        if not hasattr(self, "G_diagonal_up"):
            self.G_diagonal_up = {iatom: [] for iatom in range(len(self.atoms))}
            self.G_diagonal_dn = {iatom: [] for iatom in range(len(self.atoms))}

        for iatom in self.orb_dict:
            # Get orbital indices for this atom
            orbi = self.iorb(iatom)
            # Extract diagonal elements for this atom
            G_up_diag = np.diag(SG_up_ksum[np.ix_(orbi, orbi)])
            G_dn_diag = np.diag(SG_dn_ksum[np.ix_(orbi, orbi)])

            self.G_diagonal_up[iatom].append(G_up_diag)
            self.G_diagonal_dn[iatom].append(G_dn_diag)

    def compute_charge_and_magnetic_moments(self):
        """
        Compute charge and magnetic moments from stored spin-resolved Green's function diagonals.
        Uses the relation for collinear case:
        - Charge: n_i = -1/π ∫ (Im[Tr(S·G_ii^↑(E))] + Im[Tr(S·G_ii^↓(E))]) dE
        - Magnetic moment: m_i = -1/π ∫ (Im[Tr(S·G_ii^↑(E))] - Im[Tr(S·G_ii^↓(E))]) dE
        where S is the overlap matrix.
        """
        # Only run if debug option is enabled
        if not self.debug_options.get("compute_charge_moments", False):
            # Just use density matrix method directly
            self.get_rho_atom()
            return

        if not hasattr(self, "G_diagonal_up") or not self.G_diagonal_up:
            print(
                "Warning: No Green's function diagonals stored. Cannot compute charge and magnetic moments."
            )
            return

        self.charges = np.zeros(len(self.atoms))
        self.spinat = np.zeros((len(self.atoms), 3))

        # Arrays to store Green's function results for comparison
        gf_charges = np.zeros(len(self.atoms))
        gf_spinat = np.zeros((len(self.atoms), 3))

        for iatom in range(len(self.atoms)):
            if not self.G_diagonal_up[iatom] or not self.G_diagonal_dn[iatom]:
                continue

            # Stack all diagonal elements for this atom
            G_up_diags = np.array(
                self.G_diagonal_up[iatom]
            )  # shape: (n_energies, n_orbitals)
            G_dn_diags = np.array(
                self.G_diagonal_dn[iatom]
            )  # shape: (n_energies, n_orbitals)

            # Integrate over energy using the same contour as exchange calculation
            # Charge: -1/π ∫ Im[G_up + G_dn] dE
            integrated_up = -np.imag(self.contour.integrate_values(G_up_diags)) / np.pi
            integrated_dn = -np.imag(self.contour.integrate_values(G_dn_diags)) / np.pi

            # Sum over orbitals and spin channels to get total charge
            gf_charge = np.sum(integrated_up) + np.sum(integrated_dn)

            # Magnetic moment (assuming z-direction for collinear case)
            # m_z = -1/π ∫ Im[G_up - G_dn] dE
            gf_spin_z = np.sum(integrated_up) - np.sum(integrated_dn)

            # Store Green's function results
            gf_charges[iatom] = gf_charge
            gf_spinat[iatom, 2] = gf_spin_z

            # Compute using density matrix method for comparison
            self.get_rho_atom()  # This computes charges and spinat using density matrix
            dm_charge = self.charges[iatom]
            dm_spin_z = self.spinat[iatom, 2]

            # Compare methods if difference is above threshold
            charge_diff = abs(gf_charge - dm_charge)
            spin_diff = abs(gf_spin_z - dm_spin_z)
            threshold = self.debug_options.get("charge_moment_threshold", 1e-4)

            if charge_diff > threshold or spin_diff > threshold:
                print(f"Atom {iatom}:")
                print(f"  Green's function charge: {gf_charge:.6f}")
                print(f"  Density matrix charge: {dm_charge:.6f}")
                if charge_diff > threshold:
                    print(
                        f"  Charge difference: {charge_diff:.6f} (threshold: {threshold})"
                    )
                print(f"  Green's function magnetic moment (z): {gf_spin_z:.6f}")
                print(f"  Density matrix magnetic moment (z): {dm_spin_z:.6f}")
                if spin_diff > threshold:
                    print(
                        f"  Spin difference: {spin_diff:.6f} (threshold: {threshold})"
                    )

            # By default, use density matrix output unless debug option says otherwise
            if not self.debug_options.get("use_density_matrix_output", True):
                # Override with Green's function values (not recommended)
                self.charges[iatom] = gf_charge
                self.spinat[iatom, 2] = gf_spin_z

        # Restore the final values from density matrix method
        self.get_rho_atom()

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        npole = len(self.contour.path)
        if self.nproc == 1:
            results = map(
                self.get_quantities_per_e, tqdm(self.contour.path, total=npole)
            )
        else:
            # pool = ProcessPool(nodes=self.nproc)
            # results = pool.map(self.get_AijR_rhoR ,self.contour.path)
            results = p_map(
                self.get_quantities_per_e, self.contour.path, num_cpus=self.nproc
            )
        for i, result in enumerate(results):
            Jorb_list = result["Jorb_list"]
            JJ_list = result["JJ_list"]
            for iR in self.R_ijatom_dict:
                for iatom, jatom in self.R_ijatom_dict[iR]:
                    # Use R vector key consistently across all dictionaries
                    R_vec = self.short_Rlist[iR]
                    key = (R_vec, iatom, jatom)
                    self.Jorb_list[key].append(Jorb_list[key])
                    self.JJ_list[key].append(JJ_list[key])
        self.integrate()
        self.get_rho_atom()

        # Compute charge and magnetic moments from Green's function diagonals
        self.compute_charge_and_magnetic_moments()

        self.A_to_Jtensor()

    def write_output(self, path="TB2J_results"):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            orbital_names=self.orbital_names,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            Jiso_orb=self.Jiso_orb,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            Jani_dict=None,
            biquadratic_Jdict=None,
            description=self.description,
        )
        output.write_all(path=path)
