"""
Unit tests for SigInpParser (DMFT text-based self-energy parser).
"""

import os

import numpy as np
import pytest

from TB2J.interfaces.dmft import SigInpParser


class TestHeaderParsing:
    """Test header parsing functionality."""

    def test_parse_nom_ncor_orb(self, tmp_path):
        """Test parsing of nom,ncor_orb header line."""
        sig_file = tmp_path / "test.inp"
        beta = 100.0
        n_freq = 10
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file))

        assert parser.header["nom"] == n_freq
        assert parser.header["ncor_orb"] == 2

    def test_parse_temperature(self, tmp_path):
        """Test parsing of temperature value."""
        sig_file = tmp_path / "test.inp"
        beta = 40.0
        n_freq = 5
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        lines.append("# T= 0.025\n")
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file))

        assert parser.header["temperature"] == pytest.approx(0.025)

    def test_parse_s_oo_brackets(self, tmp_path):
        """Test parsing of s_oo with different bracket styles."""
        sig_file = tmp_path / "test.inp"
        beta = 100.0
        n_freq = 5
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 4\n"]
        lines.append("# s_oo= [1.5, 2.5, 3.5, 4.5]\n")
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2 3.0 0.3 4.0 0.4\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file))

        expected = np.array([1.5, 2.5, 3.5, 4.5])
        assert np.allclose(parser.header["s_oo"], expected)

    def test_missing_optional_headers(self, tmp_path):
        """Test that missing optional headers don't cause failure."""
        sig_file = tmp_path / "test.inp"
        beta = 100.0
        n_freq = 3
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file))

        assert parser.header["nom"] == n_freq
        assert parser.header["temperature"] is None
        assert parser.header["s_oo"] is None

    def test_missing_required_header(self, tmp_path):
        """Test error when required header is missing."""
        sig_file = tmp_path / "test.inp"
        sig_file.write_text("# T= 0.01\n" "0.0314 1.0 0.1\n")

        with pytest.raises(ValueError, match="nom,ncor_orb"):
            SigInpParser(str(sig_file))


class TestDataParsing:
    """Test data section parsing."""

    def _make_matsubara_lines(self, n_freq, n_orb, beta=100.0):
        """Helper to generate valid Matsubara frequency test data."""
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]
        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i+1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")
        return "".join(lines)

    def test_parse_data_section(self, tmp_path):
        """Test parsing of Matsubara frequency data."""
        sig_file = tmp_path / "test.inp"
        beta = 100.0
        n_freq = 10
        n_orb = 2

        sig_file.write_text(self._make_matsubara_lines(n_freq, n_orb, beta))

        parser = SigInpParser(str(sig_file))

        assert len(parser.mesh) == n_freq
        assert parser.sigma_diag.shape == (n_freq, n_orb)

        # Check first frequency
        expected_freq = np.pi / beta  # First Matsubara frequency
        assert parser.mesh[0] == pytest.approx(1j * expected_freq, rel=1e-4)
        assert np.allclose(parser.mesh.real, 0.0)
        assert np.all(parser.mesh.imag > 0.0)

    def test_column_count_validation(self, tmp_path):
        """Test that wrong column count raises error."""
        sig_file = tmp_path / "test.inp"
        sig_file.write_text(
            "# nom,ncor_orb= 3 2\n"
            "0.0314 1.0 0.1 2.0 0.2\n"
            "0.0942 1.5 0.15\n"  # Missing columns
        )

        with pytest.raises(ValueError, match="expected 5 columns"):
            SigInpParser(str(sig_file))

    def test_complex_construction(self, tmp_path):
        """Test that complex numbers are constructed correctly."""
        sig_file = tmp_path / "test.inp"
        beta = 100.0
        n_freq = 5
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} -1.5 -0.5 -2.5 -0.6\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file))

        # Check complex values
        assert parser.sigma_diag[0, 0] == pytest.approx(-1.5 - 0.5j)
        assert parser.sigma_diag[0, 1] == pytest.approx(-2.5 - 0.6j)


class TestMatrixExpansion:
    """Test matrix expansion from diagonal to full form."""

    def _make_matsubara_lines(self, n_freq, n_orb, beta=100.0):
        """Helper to generate valid Matsubara frequency test data."""
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]
        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i+1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")
        return "".join(lines)

    def test_expand_diagonal_to_matrix(self, tmp_path):
        """Test expansion of diagonal to full matrix."""
        sig_file = tmp_path / "test.inp"
        sig_file.write_text(self._make_matsubara_lines(10, 4))

        parser = SigInpParser(str(sig_file))
        sigma, mesh = parser.read_self_energy()

        # Should have 2 spin channels
        assert sigma.shape[0] == 2
        assert sigma.shape[1] == 10  # frequencies
        assert sigma.shape[2] == sigma.shape[3]  # orbital matrix

        # Check diagonal elements
        # First 2 orbitals should be in spin-up (channel 0)
        assert sigma[0, 0, 0, 0] == pytest.approx(1.0 + 0.1j)
        assert sigma[0, 0, 1, 1] == pytest.approx(2.0 + 0.2j)

        # Next 2 orbitals should be in spin-down (channel 1)
        assert sigma[1, 0, 0, 0] == pytest.approx(3.0 + 0.3j)
        assert sigma[1, 0, 1, 1] == pytest.approx(4.0 + 0.4j)

    def test_spin_channel_separation(self, tmp_path):
        """Test that spin up/down are separated correctly."""
        sig_file = tmp_path / "test.inp"
        sig_file.write_text(self._make_matsubara_lines(5, 4))

        parser = SigInpParser(str(sig_file))
        sigma, mesh = parser.read_self_energy()

        # Off-diagonal should be zero
        assert np.allclose(sigma[0, :, 0, 1], 0.0)
        assert np.allclose(sigma[1, :, 0, 1], 0.0)

        # Different spin channels should have different values
        assert not np.allclose(sigma[0], sigma[1])


class TestParamsParsing:
    """Test dmft_params.dat parsing."""

    def test_parse_params_file(self, tmp_path):
        """Test parsing of dmft_params.dat."""
        sig_file = tmp_path / "sig.inp"
        params_file = tmp_path / "dmft_params.dat"

        beta = 100.0
        n_freq = 10
        n_orb = 10
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i+1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")

        sig_file.write_text("".join(lines))

        params_file.write_text(
            "# Number of total correlated atoms\n"
            "2\n"
            "# Number of correlated orbitals per atom\n"
            "5 5\n"
            "# Orbital index for the self-energy at each atom\n"
            "1 2 3 4 5\n"
            "6 7 8 9 10\n"
        )

        parser = SigInpParser(str(sig_file), params_file=str(params_file))

        assert parser.orbital_map["n_correlated_atoms"] == 2
        assert parser.orbital_map["n_orbitals_per_atom"] == [5, 5]
        assert len(parser.orbital_map["atom_orbital_map"]) == 2

        # Check spin channels (1-5=up, 6-10=down)
        assert parser.orbital_map["spin_channels"][0] == 0  # up
        assert parser.orbital_map["spin_channels"][1] == 1  # down

    def test_params_file_not_found(self, tmp_path):
        """Test behavior when params file doesn't exist."""
        sig_file = tmp_path / "sig.inp"
        beta = 100.0
        n_freq = 5
        n_orb = 4
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i+1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")

        sig_file.write_text("".join(lines))

        # Should not fail, should use default mapping
        parser = SigInpParser(
            str(sig_file), params_file=str(tmp_path / "nonexistent.dat")
        )

        assert parser.orbital_map["n_correlated_atoms"] == 1


class TestChemicalPotential:
    """Test chemical potential reading."""

    def test_read_mu_from_file(self, tmp_path):
        """Test reading chemical potential from file."""
        sig_file = tmp_path / "sig.inp"
        mu_file = tmp_path / "DMFT_mu.out"

        beta = 100.0
        n_freq = 5
        n_orb = 2
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))
        mu_file.write_text("8.49881948544077")

        parser = SigInpParser(str(sig_file), mu_file=str(mu_file))

        assert parser.get_chemical_potential() == pytest.approx(8.49881948544077)

    def test_mu_file_not_found(self, tmp_path):
        """Test behavior when mu file doesn't exist."""
        sig_file = tmp_path / "sig.inp"

        beta = 100.0
        n_freq = 5
        n_orb = 2
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        parser = SigInpParser(str(sig_file), mu_file=str(tmp_path / "nonexistent.out"))

        # Should default to 0.0
        assert parser.get_chemical_potential() == 0.0


class TestDoubleCounting:
    """Test double-counting correction."""

    def test_use_total_sigma(self, tmp_path):
        """Test adding Σ(∞) to self-energy."""
        sig_file = tmp_path / "sig.inp"
        sig_file.write_text(
            "# nom,ncor_orb= 2 2\n"
            "# s_oo= [1.0, 2.0]\n"
            "0.0314 1.0 0.1 2.0 0.2\n"
            "0.0942 1.5 0.15 2.5 0.25\n"
        )

        # Without Σ(∞)
        parser_no_dc = SigInpParser(str(sig_file), use_total_sigma=False)
        sigma_no_dc, _ = parser_no_dc.read_self_energy()

        # With Σ(∞)
        parser_with_dc = SigInpParser(str(sig_file), use_total_sigma=True)
        sigma_with_dc, _ = parser_with_dc.read_self_energy()

        # Difference should be s_oo
        # sigma shape: [2, 2, 1, 1] (n_spin, n_freq, n_orb_per_spin, n_orb_per_spin)
        # First orbital (spin up): index [0, :, 0, 0]
        diff_up = sigma_with_dc[0, :, 0, 0].real - sigma_no_dc[0, :, 0, 0].real
        assert np.allclose(diff_up, 1.0)

        # Second orbital (spin down): index [1, :, 0, 0]
        diff_dn = sigma_with_dc[1, :, 0, 0].real - sigma_no_dc[1, :, 0, 0].real
        assert np.allclose(diff_dn, 2.0)

    def test_use_total_sigma_no_soo(self, tmp_path, caplog):
        """Test warning when s_oo not found but use_total_sigma=True."""
        import logging

        sig_file = tmp_path / "sig.inp"
        beta = 100.0
        n_freq = 5
        n_orb = 2
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        with caplog.at_level(logging.WARNING):
            _ = SigInpParser(str(sig_file), use_total_sigma=True)

        assert "s_oo not found" in caplog.text


class TestAutoDetection:
    """Test auto-detection of auxiliary files."""

    def test_auto_detect_params_file(self, tmp_path):
        """Test auto-detection of dmft_params.dat."""
        sig_file = tmp_path / "sig.inp"
        params_file = tmp_path / "dmft_params.dat"

        beta = 100.0
        n_freq = 5
        n_orb = 4
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i+1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")

        sig_file.write_text("".join(lines))

        params_file.write_text(
            "# Number of total correlated atoms\n"
            "1\n"
            "# Number of correlated orbitals per atom\n"
            "4\n"
            "# Orbital index for the self-energy at each atom\n"
            "1 2 3 4\n"
        )

        # Should auto-detect params file
        parser = SigInpParser(str(sig_file))

        assert parser.params_file is not None
        assert "dmft_params.dat" in parser.params_file

    def test_auto_detect_mu_file(self, tmp_path):
        """Test auto-detection of DMFT_mu.out."""
        sig_file = tmp_path / "sig.inp"
        mu_file = tmp_path / "DMFT_mu.out"

        beta = 100.0
        n_freq = 5
        n_orb = 2
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))
        mu_file.write_text("5.5")

        # Should auto-detect mu file
        parser = SigInpParser(str(sig_file))

        assert parser.mu_file is not None
        assert "DMFT_mu.out" in parser.mu_file
        assert parser.get_chemical_potential() == pytest.approx(5.5)


class TestMatsubaraValidation:
    """Test Matsubara frequency mesh validation."""

    def test_valid_matsubara_mesh(self, tmp_path):
        """Test validation of correct Matsubara mesh."""
        sig_file = tmp_path / "sig.inp"
        # Create valid Matsubara mesh: ωₙ = (2n+1)π/β with β=100 (T=0.01)
        beta = 100.0
        n_freq = 10
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        # Should not raise
        parser = SigInpParser(str(sig_file))

        assert len(parser.mesh) == n_freq

    def test_unequal_spacing_raises_error(self, tmp_path):
        """Test that unequal spacing raises ValueError."""
        sig_file = tmp_path / "sig.inp"
        # Create INVALID mesh with unequal spacing
        lines = [
            "# nom,ncor_orb= 4 2\n",
            "0.0314 1.0 0.1 2.0 0.2\n",
            "0.0942 1.5 0.15 2.5 0.25\n",  # spacing = 0.0628
            "0.2000 2.0 0.2 3.0 0.3\n",  # spacing = 0.1058 (WRONG!)
            "0.2500 2.5 0.25 3.5 0.35\n",  # spacing = 0.05
        ]

        sig_file.write_text("".join(lines))

        with pytest.raises(ValueError, match="not equally spaced"):
            SigInpParser(str(sig_file))

    def test_negative_frequencies_raise_error(self, tmp_path):
        """Test that negative frequencies raise ValueError."""
        sig_file = tmp_path / "sig.inp"
        lines = [
            "# nom,ncor_orb= 3 2\n",
            "-0.0314 1.0 0.1 2.0 0.2\n",  # Negative!
            "0.0942 1.5 0.15 2.5 0.25\n",
            "0.1571 2.0 0.2 3.0 0.3\n",
        ]

        sig_file.write_text("".join(lines))

        with pytest.raises(ValueError, match="must be positive"):
            SigInpParser(str(sig_file))

    def test_non_increasing_frequencies_raise_error(self, tmp_path):
        """Test that non-increasing frequencies raise ValueError."""
        sig_file = tmp_path / "sig.inp"
        lines = [
            "# nom,ncor_orb= 3 2\n",
            "0.0314 1.0 0.1 2.0 0.2\n",
            "0.0942 1.5 0.15 2.5 0.25\n",
            "0.0500 2.0 0.2 3.0 0.3\n",  # Decreasing!
        ]

        sig_file.write_text("".join(lines))

        with pytest.raises(ValueError, match="strictly increasing"):
            SigInpParser(str(sig_file))

    def test_temperature_consistency_warning(self, tmp_path, caplog):
        """Test warning when mesh beta differs from header temperature."""
        import logging

        sig_file = tmp_path / "sig.inp"
        # Create valid mesh with beta=50 (T=0.02), but header says T=0.01
        beta = 50.0
        n_freq = 5
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} 2\n"]
        lines.append(
            "# T= 0.01\n"
        )  # Header says T=0.01 (beta=100), but mesh has beta=50
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2\n")

        sig_file.write_text("".join(lines))

        with caplog.at_level(logging.WARNING):
            _ = SigInpParser(str(sig_file))

        # Should warn about inconsistency
        assert "inconsistent" in caplog.text or "beta" in caplog.text


class TestIntegration:
    """Integration tests with LaMnO3 reference data."""

    @pytest.mark.skipif(
        not os.path.exists("Refs/DMFT/LaMnO3_DMFT_data"),
        reason="LaMnO3 reference data not available",
    )
    def test_lamno3_full_parse(self):
        """Test parsing LaMnO3 reference data."""
        path = "Refs/DMFT/LaMnO3_DMFT_data"
        sig_file = os.path.join(path, "sig.inp")
        params_file = os.path.join(path, "dmft_params.dat")
        mu_file = os.path.join(path, "DMFT_mu.out")

        parser = SigInpParser(sig_file, params_file=params_file, mu_file=mu_file)

        sigma, mesh = parser.read_self_energy()

        # Validate dimensions
        assert sigma.ndim == 4
        assert sigma.shape[0] == 2  # spin channels
        assert len(mesh) > 0
        assert len(mesh) == sigma.shape[1]

        # Validate chemical potential
        mu = parser.get_chemical_potential()
        assert mu > 0  # Should be ~8.5 eV for LaMnO3

        # Validate orbital map
        assert parser.orbital_map["n_correlated_atoms"] == 4

        # Check that self-energy has reasonable values
        # (not NaN, not infinite, not all zeros)
        assert np.isfinite(sigma).all()
        assert not np.allclose(sigma, 0.0)


class TestAtomAssignment:
    """Test atom-level self-energy assignment."""

    def test_assign_to_atoms_basic(self, tmp_path):
        """Test basic atom assignment."""
        sig_file = tmp_path / "sig.inp"
        params_file = tmp_path / "dmft_params.dat"

        beta = 100.0
        n_freq = 10
        n_orb = 4
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2 3.0 0.3 4.0 0.4\n")

        sig_file.write_text("".join(lines))

        params_file.write_text(
            "# Number of total correlated atoms\n"
            "2\n"
            "# Number of correlated orbitals per atom\n"
            "2 2\n"
            "# Orbital index for the self-energy at each atom\n"
            "1 2\n"
            "3 4\n"
        )

        parser = SigInpParser(str(sig_file), params_file=str(params_file))

        wannier_to_atom = [0, 0, 1, 1]

        atom_sigma = parser.assign_to_atoms(
            n_wannier_orbitals=4, wannier_to_atom_map=wannier_to_atom
        )

        assert 0 in atom_sigma
        assert 1 in atom_sigma

        assert atom_sigma[0].shape == (2, n_freq, 2, 2)
        assert atom_sigma[1].shape == (2, n_freq, 2, 2)

    def test_get_sigma_for_atom(self, tmp_path):
        """Test getting self-energy for specific atom at energy."""
        sig_file = tmp_path / "sig.inp"
        params_file = tmp_path / "dmft_params.dat"

        beta = 100.0
        n_freq = 10
        n_orb = 4
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        for freq in frequencies:
            lines.append(f"{freq:.6f} 1.0 0.1 2.0 0.2 3.0 0.3 4.0 0.4\n")

        sig_file.write_text("".join(lines))

        params_file.write_text(
            "# Number of total correlated atoms\n"
            "1\n"
            "# Number of correlated orbitals per atom\n"
            "4\n"
            "# Orbital index for the self-energy at each atom\n"
            "1 2 3 4\n"
        )

        parser = SigInpParser(str(sig_file), params_file=str(params_file))

        wannier_to_atom = [0, 0, 0, 0]
        parser.assign_to_atoms(
            n_wannier_orbitals=4, wannier_to_atom_map=wannier_to_atom
        )

        sigma_atom = parser.get_sigma_for_atom(atom_idx=0, energy=frequencies[0] * 1j)

        assert sigma_atom.shape == (2, 4, 4)

        assert np.isfinite(sigma_atom).all()


class TestGetStaticSigma:
    """Test get_static_sigma() returns correct (2, n_orb_per_spin, n_orb_per_spin) shape."""

    def _make_sig_file(self, tmp_path, n_orb=10, with_soo_vdc=True):
        beta = 100.0
        n_freq = 5
        frequencies = [(2 * n + 1) * np.pi / beta for n in range(n_freq)]

        soo_vdc_vals = " ".join([f"{i+1}.0" for i in range(n_orb)])
        soo_vals = ", ".join([f"{i+20}.0" for i in range(n_orb)])
        vdc_vals = ", ".join(["19.0" for _ in range(n_orb)])

        lines = [f"# nom,ncor_orb= {n_freq} {n_orb}\n"]
        if with_soo_vdc:
            lines.append(f"# s_oo-Vdc= {soo_vdc_vals}\n")
        lines.append(f"# s_oo= [{soo_vals}]\n")
        lines.append(f"# Vdc= [{vdc_vals}]\n")
        for freq in frequencies:
            values = " ".join([f"{i+1}.0 0.{i % 9 + 1}" for i in range(n_orb)])
            lines.append(f"{freq:.6f} {values}\n")

        sig_file = tmp_path / "sig.inp"
        sig_file.write_text("".join(lines))
        return str(sig_file)

    def test_static_sigma_shape_with_soo_vdc(self, tmp_path):
        """get_static_sigma returns (2, 5, 5) when ncor_orb=10 and s_oo-Vdc present."""
        sig_file = self._make_sig_file(tmp_path, n_orb=10, with_soo_vdc=True)
        parser = SigInpParser(sig_file)
        sigma_static, _ = parser.get_static_sigma()

        assert sigma_static.shape == (
            2,
            5,
            5,
        ), f"Expected (2, 5, 5), got {sigma_static.shape}"

    def test_static_sigma_values_from_soo_vdc(self, tmp_path):
        """Spin-up block uses first 5 values, spin-down uses last 5 from s_oo-Vdc."""
        sig_file = self._make_sig_file(tmp_path, n_orb=10, with_soo_vdc=True)
        parser = SigInpParser(sig_file)
        sigma_static, _ = parser.get_static_sigma()

        # s_oo-Vdc = [1.0, 2.0, ..., 10.0]
        # spin-up diagonal should be [1, 2, 3, 4, 5]
        np.testing.assert_allclose(np.diag(sigma_static[0].real), [1, 2, 3, 4, 5])
        # spin-down diagonal should be [6, 7, 8, 9, 10]
        np.testing.assert_allclose(np.diag(sigma_static[1].real), [6, 7, 8, 9, 10])

    def test_static_sigma_shape_fallback_to_soo(self, tmp_path):
        """get_static_sigma returns (2, 5, 5) when s_oo-Vdc absent, falling back to s_oo."""
        sig_file = self._make_sig_file(tmp_path, n_orb=10, with_soo_vdc=False)
        parser = SigInpParser(sig_file)
        sigma_static, _ = parser.get_static_sigma()

        assert sigma_static.shape == (
            2,
            5,
            5,
        ), f"Expected (2, 5, 5), got {sigma_static.shape}"

    def test_static_sigma_off_diagonal_zero(self, tmp_path):
        """Off-diagonal elements should be zero (diagonal input)."""
        sig_file = self._make_sig_file(tmp_path, n_orb=10, with_soo_vdc=True)
        parser = SigInpParser(sig_file)
        sigma_static, _ = parser.get_static_sigma()

        # Mask diagonal, check off-diag is zero
        for spin in range(2):
            off_diag = sigma_static[spin].copy()
            np.fill_diagonal(off_diag, 0.0)
            assert np.allclose(off_diag, 0.0)

    def test_static_sigma_spin_split(self, tmp_path):
        """Spin-up and spin-down blocks must differ when s_oo-Vdc values differ."""
        sig_file = self._make_sig_file(tmp_path, n_orb=10, with_soo_vdc=True)
        parser = SigInpParser(sig_file)
        sigma_static, _ = parser.get_static_sigma()

        assert not np.allclose(
            sigma_static[0], sigma_static[1]
        ), "Spin-up and spin-down blocks should differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
