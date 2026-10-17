"""Magnon-phonon physics package (Architecture ADR-006).

Numeric kernels are filesystem- and SCF-free; data adapters live in
:mod:`TB2J.magnon_phonon.loaders`.  Story 009 provides the explicit
magnon-phonon scattering vertex; story 011 adds harmonic mixing here.
"""

# Unconditional seam attachment (Story 010): importing the scattering
# module binds ``MagnonPhononModel.scattering_spectra`` (and its private
# helpers) onto the Story-009 model facade, so the method always exists
# on :class:`MagnonPhononModel`.
from TB2J.magnon_phonon import scattering as _scattering  # noqa: F401
