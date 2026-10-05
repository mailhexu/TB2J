"""Reference channel filters shared by projector Green-function mode tests.

Mirrors the documented per-mode numerics of ``ProjectorGreen._build_channel_filter``
(exact inverse, SVD truncation, Loewdin pairing, Tikhonov damping) so the
mode-parity tests compute their expected dressed Green functions from the
published formulas rather than from backend internals.
"""

from __future__ import annotations

import numpy as np


def expected_channel_filter(S, mode, rcond):
    """Return the reference filter for one overlap matrix and mode."""
    if mode == "inverse":
        return np.linalg.inv(S)
    if mode == "svd":
        left, singular, right = np.linalg.svd(S)
        cutoff = rcond * singular[0]
        inverse = np.divide(
            1.0, singular, out=np.zeros_like(singular), where=singular > cutoff
        )
        return (right.conj().T * inverse) @ left.conj().T
    hermitian = (S + S.conj().T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    scale = np.max(np.abs(eigenvalues))
    cutoff = rcond * scale
    eigenvalues = np.maximum(eigenvalues, 0.0)
    if mode == "tikhonov":
        filtered = eigenvalues / (eigenvalues**2 + cutoff**2)
        return (eigenvectors * filtered) @ eigenvectors.conj().T
    inverse_sqrt = np.sqrt(
        np.divide(
            1.0,
            eigenvalues,
            out=np.zeros_like(eigenvalues),
            where=eigenvalues > cutoff,
        )
    )
    lowdin = (eigenvectors * inverse_sqrt) @ eigenvectors.conj().T
    return lowdin @ lowdin
