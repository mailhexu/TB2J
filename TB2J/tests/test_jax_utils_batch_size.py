"""
Tests for dynamic k-point batch sizing in TB2J.gpu.jax_utils.
"""

import pytest

from TB2J.gpu.jax_utils import _estimate_kpt_batch_size


class TestKptBatchSize:
    """Test the batch-size scaling formula."""

    @pytest.mark.parametrize("non_orth", [False, True])
    def test_calibration_anchor(self, non_orth):
        """24 GB, nbasis=728 reproduces the original empirical calibration."""
        expected = 90 if non_orth else 100
        assert _estimate_kpt_batch_size(728, non_orth, gpu_bytes=24e9) == expected

    def test_scales_linearly_with_gpu_memory(self):
        """Halving GPU memory halves the batch."""
        assert _estimate_kpt_batch_size(728, False, gpu_bytes=12e9) == 50

    def test_scales_quadratically_with_nbasis(self):
        """Doubling nbasis drops the batch ~4x."""
        assert _estimate_kpt_batch_size(1456, False, gpu_bytes=24e9) == 25

    def test_non_orthogonal_smaller_than_orthogonal(self):
        """Non-orthogonal case gets a smaller batch (extra Sk matrix)."""
        orth = _estimate_kpt_batch_size(728, False, gpu_bytes=24e9)
        non_orth = _estimate_kpt_batch_size(728, True, gpu_bytes=24e9)
        assert non_orth < orth

    def test_tiny_system_does_not_underflow(self):
        """Tiny systems return at least 1."""
        assert _estimate_kpt_batch_size(50, False, gpu_bytes=8e9) >= 1

    def test_large_system_does_not_overflow(self):
        """Very large systems still return a sane (>=1) batch."""
        assert _estimate_kpt_batch_size(3000, True, gpu_bytes=24e9) >= 1
