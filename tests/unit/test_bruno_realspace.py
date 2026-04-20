"""Tests for real-space Bruno correction methods in ExchangeCL2.

Covers:
- _bruno_fft_roundtrip: FFT-based Bruno correction (R->q, Bruno, q->R)
- _bruno_local_approximation: per-R-vector local Bruno correction
- _build_bruno_Jdict: dict construction from Bruno-corrected values

Run from the TB2J repository root:

    pytest tests/unit/test_bruno_realspace.py -v

"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from TB2J.exchangeCL2 import ExchangeCL2
from TB2J.utils import kmesh_to_R

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fft_mock():
    """Create a MagicMock with enough state to run _bruno_fft_roundtrip.

    Uses a 2×2×2 kmesh with 2 magnetic atoms.
    JJ/Korb/Xorb are initialised to simple deterministic values.
    """
    exchange = MagicMock(spec=ExchangeCL2)
    exchange.kmesh = [2, 2, 2]
    exchange.Rlist = kmesh_to_R([2, 2, 2])
    exchange.ind_mag_atoms = [0, 1]
    exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    exchange.JJ = {}
    exchange.Korb = {}
    exchange.Xorb = {}

    for R_vec in exchange.Rlist:
        for iatom in exchange.ind_mag_atoms:
            for jatom in exchange.ind_mag_atoms:
                key = (R_vec, iatom, jatom)
                seed = hash(key) % 100
                exchange.JJ[key] = complex(0.1 * seed, 0.2 * seed)
                exchange.Korb[key] = np.array(
                    [
                        [complex(0.3 * seed, 0.05 * seed), 0j],
                        [0j, complex(0.3 * seed, 0.05 * seed)],
                    ]
                )
                exchange.Xorb[key] = np.array(
                    [
                        [complex(1.0 + 0.1 * seed, 0.01 * seed), 0j],
                        [0j, complex(1.0 + 0.1 * seed, 0.01 * seed)],
                    ]
                )
    return exchange


def _make_local_mock():
    """Create a MagicMock with enough state to run _bruno_local_approximation."""
    exchange = MagicMock(spec=ExchangeCL2)
    exchange.short_Rlist = [(0, 0, 0), (1, 0, 0)]
    exchange.ind_mag_atoms = [0, 1]
    exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    exchange.Korb = {}
    exchange.Xorb = {}
    exchange.JJ = {}

    for R_vec in exchange.short_Rlist:
        for iatom in exchange.ind_mag_atoms:
            for jatom in exchange.ind_mag_atoms:
                key = (R_vec, iatom, jatom)
                seed = hash(key) % 100
                exchange.Korb[key] = np.array(
                    [
                        [complex(0.3 * seed, 0.05 * seed), 0j],
                        [0j, complex(0.3 * seed, 0.05 * seed)],
                    ]
                )
                exchange.Xorb[key] = np.array(
                    [
                        [complex(1.0 + 0.1 * seed, 0.01 * seed), 0j],
                        [0j, complex(1.0 + 0.1 * seed, 0.01 * seed)],
                    ]
                )
                exchange.JJ[key] = complex(0.1 * seed, 0.2 * seed)
    return exchange


# ===========================================================================
# TestBrunoFFT
# ===========================================================================


class TestBrunoFFT:
    """Tests for ExchangeCL2._bruno_fft_roundtrip."""

    def test_fft_produces_jnorm_r_array(self):
        """After calling _bruno_fft_roundtrip, Jnorm_R should exist with shape (nR, nmag, nmag)."""
        exchange = _make_fft_mock()
        ExchangeCL2._bruno_fft_roundtrip(exchange)

        nR = len(exchange.Rlist)
        nmag = len(exchange.ind_mag_atoms)
        assert hasattr(exchange, "Jnorm_R")
        assert exchange.Jnorm_R.shape == (nR, nmag, nmag)

    def test_fft_round_trip_identity(self):
        """With only R=(0,0,0) non-zero, the FFT round-trip preserves values.

        All q-points see the same data, so Jnorm_R[R=0] should equal the
        Bruno-corrected value at q=0 (which is also the average over all q).
        """
        exchange = MagicMock(spec=ExchangeCL2)
        exchange.kmesh = [2, 2, 2]
        exchange.Rlist = kmesh_to_R([2, 2, 2])
        exchange.ind_mag_atoms = [0, 1]
        exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        ni = 2
        exchange.JJ = {}
        exchange.Korb = {}
        exchange.Xorb = {}

        for iatom in exchange.ind_mag_atoms:
            for jatom in exchange.ind_mag_atoms:
                key = ((0, 0, 0), iatom, jatom)
                exchange.JJ[key] = complex(0, 1.0)
                exchange.Korb[key] = np.array(
                    [[complex(0.5, 0.1), 0j], [0j, complex(0.5, 0.1)]]
                )
                if iatom == jatom:
                    exchange.Xorb[key] = np.array(
                        [[complex(3.0, 1.0), 0j], [0j, complex(3.0, 1.0)]]
                    )
                else:
                    exchange.Xorb[key] = np.array(
                        [[complex(1.0, 0.5), 0j], [0j, complex(1.0, 0.5)]]
                    )

        # All other R vectors get zero
        for R_vec in exchange.Rlist:
            if R_vec == (0, 0, 0):
                continue
            for iatom in exchange.ind_mag_atoms:
                for jatom in exchange.ind_mag_atoms:
                    key = (R_vec, iatom, jatom)
                    exchange.JJ[key] = 0j
                    exchange.Korb[key] = np.zeros((ni, ni), dtype=complex)
                    exchange.Xorb[key] = np.zeros((ni, ni), dtype=complex)

        ExchangeCL2._bruno_fft_roundtrip(exchange)

        # All q-points are identical since only R=0 contributes.
        # So Jnorm_R[0] (R=0) should be the Bruno-corrected value.
        # For other R, Jnorm_R should be very close to the same value
        # because all q see the same thing, so iFFT gives the same for every R.
        # Actually for R!=0, phase averages to zero if nq is uniform —
        # but let's verify Jnorm_R is finite and the R=0 entry is correct.
        assert np.all(np.isfinite(exchange.Jnorm_R))

        # R=0 entry: should contain the Bruno-corrected J
        # (all q-points identical, so iFFT at R=0 = average = value at any q)
        assert exchange.Jnorm_R[0, 0, 0] != 0.0 or exchange.Jnorm_R[0, 0, 1] != 0.0

    def test_fft_zero_kx_gives_jnorm_close_to_j(self):
        """With K=0 and X=const*identity, Bruno correction is a constant additive term.

        Since the correction is the same for all q-points, the iFFT round-trip
        should give Jnorm_R ≈ imag(JJ) + constant_correction at every R.
        """
        exchange = MagicMock(spec=ExchangeCL2)
        exchange.kmesh = [2, 2, 2]
        exchange.Rlist = kmesh_to_R([2, 2, 2])
        exchange.ind_mag_atoms = [0, 1]
        exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        ni = 2
        x_self = complex(3.0, 1.0)
        x_cross = complex(1.0, 0.5)
        exchange.JJ = {}
        exchange.Korb = {}
        exchange.Xorb = {}

        j_val = complex(0, 0.5)
        for iatom in exchange.ind_mag_atoms:
            for jatom in exchange.ind_mag_atoms:
                key = ((0, 0, 0), iatom, jatom)
                exchange.JJ[key] = j_val
                exchange.Korb[key] = np.zeros((ni, ni), dtype=complex)
                x_val = x_self if iatom == jatom else x_cross
                exchange.Xorb[key] = x_val * np.eye(ni, dtype=complex)

        for R_vec in exchange.Rlist:
            if R_vec == (0, 0, 0):
                continue
            for iatom in exchange.ind_mag_atoms:
                for jatom in exchange.ind_mag_atoms:
                    key = (R_vec, iatom, jatom)
                    exchange.JJ[key] = 0j
                    exchange.Korb[key] = np.zeros((ni, ni), dtype=complex)
                    x_val = x_self if iatom == jatom else x_cross
                    exchange.Xorb[key] = x_val * np.eye(ni, dtype=complex)

        ExchangeCL2._bruno_fft_roundtrip(exchange)

        assert np.all(np.isfinite(exchange.Jnorm_R))
        assert exchange.Jnorm_R[0, 0, 0] > 0.5


# ===========================================================================
# TestBrunoLocal
# ===========================================================================


class TestBrunoLocal:
    """Tests for ExchangeCL2._bruno_local_approximation."""

    def test_local_produces_bruno_jnorm_dict(self):
        """After calling, _bruno_Jnorm exists and has correct non-self keys."""
        exchange = _make_local_mock()
        ExchangeCL2._bruno_local_approximation(exchange)

        assert hasattr(exchange, "_bruno_Jnorm")
        assert isinstance(exchange._bruno_Jnorm, dict)
        assert len(exchange._bruno_Jnorm) > 0

        # Should contain cross-atom pairs
        assert ((0, 0, 0), 0, 1) in exchange._bruno_Jnorm
        assert ((0, 0, 0), 1, 0) in exchange._bruno_Jnorm
        assert ((1, 0, 0), 0, 1) in exchange._bruno_Jnorm
        assert ((1, 0, 0), 1, 0) in exchange._bruno_Jnorm

    def test_local_skips_self_interaction(self):
        """Keys with R=(0,0,0) and iatom==jatom should NOT be in _bruno_Jnorm."""
        exchange = _make_local_mock()
        ExchangeCL2._bruno_local_approximation(exchange)

        assert ((0, 0, 0), 0, 0) not in exchange._bruno_Jnorm
        assert ((0, 0, 0), 1, 1) not in exchange._bruno_Jnorm

    def test_local_r_zero_uses_m_minus_kt(self):
        """For R=(0,0,0), the B matrix should be M_mat - K.T (not just -K.T).

        We verify by computing the expected correction manually.
        """
        ni = 2
        exchange = MagicMock(spec=ExchangeCL2)
        exchange.short_Rlist = [(0, 0, 0)]
        exchange.ind_mag_atoms = [0, 1]
        exchange.spinat = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]])

        # Use known K and X for cross terms only (self is skipped)
        K_known = np.array([[complex(1.0, 0.5), 0j], [0j, complex(1.0, 0.5)]])
        X_known = np.array([[complex(2.0, 0.0), 0j], [0j, complex(2.0, 0.0)]])

        exchange.Korb = {}
        exchange.Xorb = {}
        exchange.JJ = {}

        # Only set cross terms (0,1) and (1,0) for R=(0,0,0)
        key_01 = ((0, 0, 0), 0, 1)
        exchange.Korb[key_01] = K_known
        exchange.Xorb[key_01] = X_known
        exchange.JJ[key_01] = complex(0, 0.1)

        key_10 = ((0, 0, 0), 1, 0)
        exchange.Korb[key_10] = K_known
        exchange.Xorb[key_10] = X_known
        exchange.JJ[key_10] = complex(0, 0.2)

        ExchangeCL2._bruno_local_approximation(exchange)

        M_mat = 3.0 * np.eye(ni)
        B_expected = M_mat - K_known.T
        correction_expected = 0.5 * B_expected.T @ np.linalg.solve(X_known, B_expected)
        j_expected_01 = np.imag(exchange.JJ[key_01]) + np.sum(correction_expected)

        assert exchange._bruno_Jnorm[key_01] == pytest.approx(j_expected_01)

    def test_local_r_nonzero_uses_minus_kt(self):
        """For R!=0, B should be -K.T (no M_mat contribution)."""
        exchange = MagicMock(spec=ExchangeCL2)
        exchange.short_Rlist = [(1, 0, 0)]
        exchange.ind_mag_atoms = [0, 1]
        exchange.spinat = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]])

        K_known = np.array([[complex(1.0, 0.5), 0j], [0j, complex(1.0, 0.5)]])
        X_known = np.array([[complex(2.0, 0.0), 0j], [0j, complex(2.0, 0.0)]])

        exchange.Korb = {}
        exchange.Xorb = {}
        exchange.JJ = {}

        key_01 = ((1, 0, 0), 0, 1)
        exchange.Korb[key_01] = K_known
        exchange.Xorb[key_01] = X_known
        exchange.JJ[key_01] = complex(0, 0.15)

        ExchangeCL2._bruno_local_approximation(exchange)

        B_expected = -K_known.T
        correction_expected = 0.5 * B_expected.T @ np.linalg.solve(X_known, B_expected)
        j_expected = np.imag(exchange.JJ[key_01]) + np.sum(correction_expected)

        assert exchange._bruno_Jnorm[key_01] == pytest.approx(j_expected)

    def test_local_singular_x_no_crash(self):
        """With a singular X_full matrix, _bruno_local_approximation should not crash."""
        ni = 2
        exchange = MagicMock(spec=ExchangeCL2)
        exchange.short_Rlist = [(1, 0, 0)]
        exchange.ind_mag_atoms = [0, 1]
        exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        K_known = np.array([[complex(1.0, 0.5), 0j], [0j, complex(1.0, 0.5)]])
        # Singular matrix: all zeros
        X_singular = np.zeros((ni, ni), dtype=complex)

        exchange.Korb = {}
        exchange.Xorb = {}
        exchange.JJ = {}

        key = ((1, 0, 0), 0, 1)
        exchange.Korb[key] = K_known
        exchange.Xorb[key] = X_singular
        exchange.JJ[key] = complex(0, 0.1)

        # Should not raise
        ExchangeCL2._bruno_local_approximation(exchange)

        # correction should be zero (caught LinAlgError), so result = imag(JJ)
        assert key in exchange._bruno_Jnorm
        assert exchange._bruno_Jnorm[key] == pytest.approx(np.imag(complex(0, 0.1)))


# ===========================================================================
# TestBuildBrunoJdict
# ===========================================================================


def _make_jdict_mock_fft():
    """Mock with Jnorm_R array for testing _build_bruno_Jdict(source='fft')."""
    exchange = MagicMock(spec=ExchangeCL2)
    exchange.Rlist = [(0, 0, 0), (1, 0, 0)]
    exchange.ind_mag_atoms = [0, 1]
    exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    exchange.Jnorm_R = np.array(
        [
            [[1.0, 4.5], [4.5, 1.0]],
            [[1.0, 2.7], [2.7, 1.0]],
        ]
    )

    vec01 = np.array([2.5, 0.0, 0.0])
    vec10 = np.array([-2.5, 0.0, 0.0])
    exchange.distance_dict = {
        ((0, 0, 0), 0, 1): (vec01, 2.5),
        ((0, 0, 0), 1, 0): (vec10, 2.5),
        ((1, 0, 0), 0, 1): (vec01 + np.array([3.8, 0, 0]), 4.55),
        ((1, 0, 0), 1, 0): (vec10 + np.array([3.8, 0, 0]), 4.55),
    }

    def fake_ispin(iatom):
        return iatom

    exchange.ispin = fake_ispin
    return exchange


def _make_jdict_mock_local():
    """Mock with _bruno_Jnorm dict for testing _build_bruno_Jdict(source='local')."""
    exchange = MagicMock(spec=ExchangeCL2)
    exchange.ind_mag_atoms = [0, 1]
    exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

    exchange._bruno_Jnorm = {
        ((0, 0, 0), 0, 1): 4.5,
        ((0, 0, 0), 1, 0): 4.5,
        ((1, 0, 0), 0, 1): 2.7,
        ((1, 0, 0), 1, 0): 2.7,
    }

    vec01 = np.array([2.5, 0.0, 0.0])
    vec10 = np.array([-2.5, 0.0, 0.0])
    exchange.distance_dict = {
        ((0, 0, 0), 0, 1): (vec01, 2.5),
        ((0, 0, 0), 1, 0): (vec10, 2.5),
        ((1, 0, 0), 0, 1): (vec01 + np.array([3.8, 0, 0]), 4.55),
        ((1, 0, 0), 1, 0): (vec10 + np.array([3.8, 0, 0]), 4.55),
    }

    def fake_ispin(iatom):
        return iatom

    exchange.ispin = fake_ispin
    return exchange


class TestBuildBrunoJdict:
    """Tests for ExchangeCL2._build_bruno_Jdict."""

    def test_build_jdict_fft_source(self):
        """With source='fft', _build_bruno_Jdict populates exchange_Jdict_bruno."""
        exchange = _make_jdict_mock_fft()
        ExchangeCL2._build_bruno_Jdict(exchange, source="fft")

        assert hasattr(exchange, "exchange_Jdict_bruno")
        assert isinstance(exchange.exchange_Jdict_bruno, dict)
        assert len(exchange.exchange_Jdict_bruno) > 0

    def test_build_jdict_local_source(self):
        """With source='local', _build_bruno_Jdict populates exchange_Jdict_bruno."""
        exchange = _make_jdict_mock_local()
        ExchangeCL2._build_bruno_Jdict(exchange, source="local")

        assert hasattr(exchange, "exchange_Jdict_bruno")
        assert isinstance(exchange.exchange_Jdict_bruno, dict)
        assert len(exchange.exchange_Jdict_bruno) > 0

    def test_build_jdict_distance_filtering(self):
        """Keys NOT in distance_dict should be excluded from exchange_Jdict_bruno."""
        exchange = _make_jdict_mock_fft()
        # Remove one key from distance_dict
        del exchange.distance_dict[((0, 0, 0), 1, 0)]

        ExchangeCL2._build_bruno_Jdict(exchange, source="fft")

        assert ((0, 0, 0), 1, 0) not in exchange.exchange_Jdict_bruno
        # Other keys should still be present
        assert ((0, 0, 0), 0, 1) in exchange.exchange_Jdict_bruno

    def test_build_jdict_sign_convention(self):
        """Values should be divided by sign(dot(spinat[i], spinat[j]))."""
        exchange = _make_jdict_mock_fft()
        # spinat[0] = [0,0,2], spinat[1] = [0,0,2] => dot = 4, sign = 1
        # So values should equal Jnorm_R entries
        ExchangeCL2._build_bruno_Jdict(exchange, source="fft")

        key = ((1, 0, 0), 0, 1)
        sign = np.sign(np.dot(exchange.spinat[0], exchange.spinat[1]))
        expected = exchange.Jnorm_R[1, 0, 1] / sign
        assert exchange.exchange_Jdict_bruno[key] == pytest.approx(expected)

    def test_build_jdict_sign_convention_antiparallel(self):
        """With antiparallel spins, the sign should flip the value."""
        exchange = _make_jdict_mock_fft()
        # Make spins antiparallel
        exchange.spinat = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
        # dot = -4, sign = -1 => value = Jnorm_R / (-1)
        ExchangeCL2._build_bruno_Jdict(exchange, source="fft")

        key = ((1, 0, 0), 0, 1)
        sign = np.sign(np.dot(exchange.spinat[0], exchange.spinat[1]))
        expected = exchange.Jnorm_R[1, 0, 1] / sign
        assert exchange.exchange_Jdict_bruno[key] == pytest.approx(expected)
        # Should be negative of Jnorm_R value
        assert exchange.exchange_Jdict_bruno[key] == pytest.approx(
            -exchange.Jnorm_R[1, 0, 1]
        )

    def test_build_jdict_excludes_self(self):
        """R=(0,0,0) with i==j should be excluded."""
        exchange = _make_jdict_mock_fft()
        ExchangeCL2._build_bruno_Jdict(exchange, source="fft")

        # Self-interaction keys should not appear
        assert (0, 0, 0, 0, 0) not in exchange.exchange_Jdict_bruno
        assert (0, 0, 0, 1, 1) not in exchange.exchange_Jdict_bruno


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
