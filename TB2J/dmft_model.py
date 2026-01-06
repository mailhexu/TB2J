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
        self.mu = dmft_parser.get_chemical_potential()

        # Properties to mimic the static model
        self.is_orthogonal = static_model.is_orthogonal
        self.R2kfactor = static_model.R2kfactor
        self.norb = static_model.norb
        self.nbasis = static_model.nbasis
        self.atoms = static_model.atoms
        self.nel = static_model.nel

    def get_sigma(self, energy):
        """
        Interpolates or selects the self-energy for a given energy.
        Since we currently use discrete Matsubara frequencies, we look for
        an exact match or nearest neighbor.

        :param energy: complex energy relative to Fermi level (e - efermi)
        :returns: Sigma(energy) matrix of shape (n_spin, n_orb, n_orb)
        """
        # Find index in mesh. Note: energy in TB2J is e - efermi.
        # DMFT mesh is usually Matsubara iwn.
        # In TB2J, G(z) = (z+efermi - H)^-1
        # So z is the energy parameter.

        # Simple exact match for now (Phase 2)
        idx = np.argmin(np.abs(self.mesh - (energy + self.static_model.efermi)))
        if not np.isclose(self.mesh[idx], energy + self.static_model.efermi, atol=1e-5):
            # Warning or interpolation could go here
            pass

        return self.sigma[:, idx, :, :]

    def HSE_k(self, kpt, convention=2):
        """
        Mock HSE_k that returns static results.
        Note: The frequency dependence is handled at the Green's function level.
        """
        return self.static_model.HSE_k(kpt, convention=convention)

    def get_hamiltonian(self, kpt):
        return self.static_model.gen_ham(kpt)
