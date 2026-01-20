import numpy as np


class TBModelDMFT:
    """
    A wrapper for combining a static tight-binding model with a
    frequency-dependent self-energy.
    """

    def __init__(self, static_model, dmft_parser):
        """
        :param static_model: An object representing the static TB model
                             (e.g., from TB2J.interfaces)
        :param dmft_parser: A DMFTParser object containing Sigma(omega) and mu.
        """
        self.static_model = static_model
        self.dmft_parser = dmft_parser
        self.sigma, self.mesh = dmft_parser.read_self_energy()
        if self.sigma.ndim == 4 and self.sigma.shape[0] == 1:
            self.sigma = np.squeeze(self.sigma, axis=0)
        self.mu = dmft_parser.get_chemical_potential()

        # Properties to mimic the static model
        self.is_orthogonal = static_model.is_orthogonal
        self.R2kfactor = static_model.R2kfactor
        self.norb = static_model.norb
        self.nbasis = static_model.nbasis
        self.atoms = static_model.atoms
        self.nel = getattr(static_model, "nel", 0.0)
        self.efermi = getattr(static_model, "efermi", 0.0)

    def get_sigma(self, energy, ispin=0):
        """
        Interpolates or selects the self-energy for a given energy.
        If NCL, it returns a spinor matrix.
        If collinear, it returns the ispin channel.
        If ispin is None, it returns all channels.
        """
        idx = np.argmin(np.abs(self.mesh - (energy + self.efermi)))

        sigma_at_e = self.sigma[..., idx, :, :]

        # Scenario 1: NCL/SOC (Spinor)
        if not getattr(self.static_model, "colinear", True) or getattr(
            self.static_model, "nls", False
        ):
            if sigma_at_e.ndim == 3 and sigma_at_e.shape[0] == 2:
                n = sigma_at_e.shape[1]
                sigma_full = np.zeros((2 * n, 2 * n), dtype=complex)
                sigma_full[:n, :n] = sigma_at_e[0]
                sigma_full[n:, n:] = sigma_at_e[1]
                return sigma_full
            return sigma_at_e

        # Scenario 2: Collinear
        if sigma_at_e.ndim == 3 and sigma_at_e.shape[0] == 2:
            if ispin is None:
                return sigma_at_e
            return sigma_at_e[ispin]

        return sigma_at_e

    def HSE_k(self, kpt, convention=2):
        """
        Mock HSE_k that returns static results.
        Note: The frequency dependence is handled at the Green's function level.
        """
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
