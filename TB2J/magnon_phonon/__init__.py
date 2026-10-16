"""Magnon-phonon physics package (Architecture ADR-006).

Numeric kernels are filesystem- and SCF-free; data adapters live in
:mod:`TB2J.magnon_phonon.loaders`.  Story 009 provides the explicit
magnon-phonon scattering vertex; story 011 adds harmonic mixing here.
"""
