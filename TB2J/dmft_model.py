from typing import Any, Dict, List, Optional

import numpy as np


class TBModelDMFT:
    """
    A wrapper for combining a static tight-binding model with a
    frequency-dependent self-energy.

    For siginp format (collinear spin with per-atom self-energies):
    - Self-energy is embedded into the full orbital basis
    - Only correlated atoms (e.g., Mn) have non-zero self-energy
    - Other atoms have zero self-energy

    Two modes:
    - Dynamic (default): uses full Σ(iωₙ) on Matsubara mesh
    - Static (use_static_sigma=True): uses only Σ(∞) or Σ(∞)-Vdc as a
      frequency-independent shift; compatible with CFR contour
    """

    def __init__(
        self,
        static_model,
        dmft_parser,
        basis: Optional[List[Any]] = None,
        atoms=None,
        magnetic_elements: Optional[List[str]] = None,
        use_static_sigma: bool = False,
    ):
        """
        Initialize DMFT model wrapper.

        Args:
            static_model: Static TB model (e.g., WannierHam)
            dmft_parser: DMFT parser with self-energy data
            basis: List of basis entries mapping orbitals to atoms
            atoms: ASE atoms object
            magnetic_elements: List of magnetic element symbols
            use_static_sigma: If True, use only Σ(∞) (or Σ(∞)-Vdc) as a
                              static frequency-independent self-energy
        """
        self.static_model = static_model
        self.dmft_parser = dmft_parser
        self.mu = dmft_parser.get_chemical_potential()
        self.is_static = use_static_sigma

        self.nbasis = static_model.nbasis
        self.is_orthogonal = static_model.is_orthogonal
        self.R2kfactor = static_model.R2kfactor
        self.norb = static_model.norb
        self.atoms = static_model.atoms
        self.nel = getattr(static_model, "nel", 0.0)
        self.efermi = getattr(static_model, "efermi", 0.0)

        if use_static_sigma:
            # Static mode: Σ(∞) or Σ(∞)-Vdc as a constant per-spin matrix
            sigma_raw, _ = dmft_parser.get_static_sigma()
            # sigma_raw shape: (n_spin, n_corr, n_corr)
            self.n_spin = sigma_raw.shape[0]
            self.n_freq = 1  # virtual; get_sigma ignores energy
            self.mesh = None
            self._sigma_full = self._build_full_sigma_static(
                sigma_raw, basis, atoms, magnetic_elements
            )
        else:
            # Dynamic mode: full Σ(iωₙ) on Matsubara mesh
            sigma_raw, self.mesh = dmft_parser.read_self_energy()
            self.n_spin = sigma_raw.shape[0] if sigma_raw.ndim == 4 else 1
            self.n_freq = (
                sigma_raw.shape[1] if sigma_raw.ndim == 4 else sigma_raw.shape[0]
            )
            self._sigma_full = self._build_full_sigma(
                sigma_raw, basis, atoms, magnetic_elements
            )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _build_orb_dict(self, basis, atom_symbols) -> Dict[int, List[int]]:
        """Build mapping from atom index (0-based) to list of basis orbital indices."""
        orb_dict: Dict[int, List[int]] = {}
        for orb_idx, base in enumerate(basis):
            if isinstance(base, str):
                parts = base.split("|")
                atom_sym = parts[0]
                atom_idx = atom_symbols[atom_sym]
            else:
                atom_idx = getattr(base, "iatom", base)

            if atom_idx not in orb_dict:
                orb_dict[atom_idx] = []
            orb_dict[atom_idx].append(orb_idx)

        return orb_dict

    def _embed_sigma_blocks(
        self,
        sigma_raw: np.ndarray,
        basis,
        atoms,
        magnetic_elements,
        n_freq_dim: int,
    ) -> np.ndarray:
        """
        Core embedding logic shared by both dynamic and static modes.

        Args:
            sigma_raw: shape (n_spin, n_freq_dim, n_corr, n_corr) — dynamic
                       or (n_spin, n_corr, n_corr) — static (n_freq_dim==0)
            basis: orbital basis list
            atoms: ASE atoms object
            magnetic_elements: not used currently (kept for future filtering)
            n_freq_dim: 1 if dynamic (freq axis present), 0 if static

        Returns:
            sigma_full: (n_spin, n_freq_dim, nbasis, nbasis) if dynamic,
                        (n_spin, nbasis, nbasis) if static
        """
        from TB2J.utils import symbol_number

        atom_symbols = symbol_number(atoms)
        orb_dict = self._build_orb_dict(basis, atom_symbols)
        orbital_map = self.dmft_parser.orbital_map

        if n_freq_dim > 0:
            # Dynamic: (n_spin, n_freq, nbasis, nbasis)
            sigma_full = np.zeros(
                (self.n_spin, n_freq_dim, self.nbasis, self.nbasis),
                dtype=np.complex128,
            )
        else:
            # Static: (n_spin, nbasis, nbasis)
            sigma_full = np.zeros(
                (self.n_spin, self.nbasis, self.nbasis),
                dtype=np.complex128,
            )

        # Resolve correlated-atom sequence indices (0,1,2,...) to global atom
        # indices in the full structure.  orbital_map['atom_indices'] contains
        # 0-based positions *within the list of correlated atoms*, not global
        # indices.  We must map them to the actual global indices using
        # magnetic_elements (e.g. Mn atoms at global positions 4,5,6,7 in
        # La4Mn4O12, not 0,1,2,3).
        atom_symbols_list = atoms.get_chemical_symbols()
        if magnetic_elements:
            mag_set = set(magnetic_elements)
            corr_global_indices = [
                i for i, s in enumerate(atom_symbols_list) if s in mag_set
            ]
        else:
            corr_global_indices = list(range(len(atom_symbols_list)))

        for corr_idx, _corr_seq_idx in enumerate(orbital_map.get("atom_indices", [])):
            if corr_idx >= len(corr_global_indices):
                continue
            atom_idx = corr_global_indices[corr_idx]
            if atom_idx not in orb_dict:
                continue

            wann_orb_indices = orb_dict[atom_idx]
            n_wann_orbs = len(wann_orb_indices)

            # Compare against per-atom orbital count, not total
            n_orbs_per_atom = orbital_map.get("n_orbitals_per_atom", [])
            if corr_idx < len(n_orbs_per_atom):
                n_expected = n_orbs_per_atom[corr_idx]
            else:
                n_expected = sigma_raw.shape[-1]  # fallback: total corr orbs

            if n_wann_orbs != n_expected:
                continue

            spin_channel = orbital_map["spin_channels"][corr_idx]

            # Embed sigma_raw (majority/minority in local frame) into global
            # spin channels.  sigma_raw[0] = majority-spin self-energy,
            # sigma_raw[1] = minority-spin self-energy (both from the same
            # DMFT impurity calculation).
            #
            # Mapping local → global spin channels:
            #
            #   spin-up atom   (spin_channel==0):
            #     majority = global up  → sigma_full[0] += sigma_raw[0]
            #     minority = global dn  → sigma_full[1] += sigma_raw[1]
            #
            #   spin-down atom (spin_channel==1):
            #     majority = global dn  → sigma_full[1] += sigma_raw[0]
            #     minority = global up  → sigma_full[0] += sigma_raw[1]
            #
            # green.py: G_s = inv(E+EF - H - sigma_full[s]), so both spin
            # channels of every correlated atom are correctly shifted.
            if spin_channel == 0:
                # majority → global up, minority → global dn
                spin_src_map = [(0, 0), (1, 1)]  # (global_spin, sigma_raw_spin)
            else:
                # majority → global dn, minority → global up
                spin_src_map = [(1, 0), (0, 1)]  # (global_spin, sigma_raw_spin)

            if n_freq_dim > 0:
                for global_s, src_s in spin_src_map:
                    for i_orb in range(n_wann_orbs):
                        for j_orb in range(n_wann_orbs):
                            i_full = wann_orb_indices[i_orb]
                            j_full = wann_orb_indices[j_orb]
                            sigma_full[global_s, :, i_full, j_full] = sigma_raw[
                                src_s, :, i_orb, j_orb
                            ]
            else:
                for global_s, src_s in spin_src_map:
                    for i_orb in range(n_wann_orbs):
                        for j_orb in range(n_wann_orbs):
                            i_full = wann_orb_indices[i_orb]
                            j_full = wann_orb_indices[j_orb]
                            sigma_full[global_s, i_full, j_full] = sigma_raw[
                                src_s, i_orb, j_orb
                            ]

        return sigma_full

    def _build_full_sigma(
        self,
        sigma_raw: np.ndarray,
        basis,
        atoms,
        magnetic_elements,
    ) -> np.ndarray:
        """
        Build full-size self-energy matrix (dynamic mode).

        Args:
            sigma_raw: Raw self-energy, shape (n_spin, n_freq, n_corr, n_corr)
            basis: Orbital basis mapping
            atoms: ASE atoms object
            magnetic_elements: List of magnetic element symbols

        Returns:
            sigma_full: (n_spin, n_freq, nbasis, nbasis)
        """
        if basis is None or atoms is None:
            return sigma_raw

        if not hasattr(self.dmft_parser, "orbital_map"):
            return sigma_raw

        orbital_map = self.dmft_parser.orbital_map
        if not orbital_map.get("atom_indices"):
            return sigma_raw

        return self._embed_sigma_blocks(
            sigma_raw, basis, atoms, magnetic_elements, n_freq_dim=self.n_freq
        )

    def _build_full_sigma_static(
        self,
        sigma_raw: np.ndarray,
        basis,
        atoms,
        magnetic_elements,
    ) -> np.ndarray:
        """
        Build full-size self-energy matrix (static mode).

        Args:
            sigma_raw: Static self-energy, shape (n_spin, n_corr, n_corr)
            basis: Orbital basis mapping
            atoms: ASE atoms object
            magnetic_elements: List of magnetic element symbols

        Returns:
            sigma_full: (n_spin, nbasis, nbasis)
        """
        if basis is None or atoms is None:
            return sigma_raw

        if not hasattr(self.dmft_parser, "orbital_map"):
            return sigma_raw

        orbital_map = self.dmft_parser.orbital_map
        if not orbital_map.get("atom_indices"):
            return sigma_raw

        return self._embed_sigma_blocks(
            sigma_raw, basis, atoms, magnetic_elements, n_freq_dim=0
        )

    # ------------------------------------------------------------------
    # Self-energy retrieval
    # ------------------------------------------------------------------

    def get_sigma(self, energy, ispin=0):
        """
        Get self-energy for a given energy.

        In static mode the returned matrix is the same for any energy.
        In dynamic mode the closest Matsubara point is selected.

        Args:
            energy: Complex energy (Matsubara frequency); ignored in static mode
            ispin: Spin index (0=up, 1=down, None=all spins)

        Returns:
            For ispin=None: (n_spin, nbasis, nbasis) array
            For specific ispin: (nbasis, nbasis) array
        """
        if self.is_static:
            # sigma_full is (n_spin, nbasis, nbasis)
            sigma_at_e = self._sigma_full
        else:
            idx = np.argmin(np.abs(self.mesh - energy))
            if self._sigma_full.ndim == 4:
                sigma_at_e = self._sigma_full[:, idx, :, :]
            else:
                sigma_at_e = self._sigma_full[idx, :, :]

        if not getattr(self.static_model, "colinear", True):
            if sigma_at_e.ndim == 3 and sigma_at_e.shape[0] == 2:
                n = sigma_at_e.shape[1]
                sigma_spinor = np.zeros((2 * n, 2 * n), dtype=complex)
                sigma_spinor[:n, :n] = sigma_at_e[0]
                sigma_spinor[n:, n:] = sigma_at_e[1]
                return sigma_spinor
            return sigma_at_e

        if sigma_at_e.ndim == 3:
            if ispin is None:
                return sigma_at_e
            return sigma_at_e[ispin]

        return sigma_at_e

    # ------------------------------------------------------------------
    # Delegate to static model
    # ------------------------------------------------------------------

    def HSE_k(self, kpt, convention=2):
        return self.static_model.HSE_k(kpt, convention=convention)

    def get_hamiltonian(self, kpt, ispin=0):
        if hasattr(self.static_model, "gen_ham"):
            try:
                return self.static_model.gen_ham(kpt, ispin=ispin)
            except TypeError:
                return self.static_model.gen_ham(kpt)
        else:
            H, S, _, _ = self.static_model.HSE_k(kpt)
            return H
