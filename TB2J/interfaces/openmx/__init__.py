#!/usr/bin/env python3
"""
TB2J OpenMX interface using HamiltonIO.

This module provides an alternative OpenMX interface for TB2J that uses
the new HamiltonIO OpenMX parser with pure-Python scfout reader.

The legacy TB2J_OpenMX package remains available for backward compatibility.
"""

from .gen_exchange_openmx import gen_exchange_openmx

__all__ = ["gen_exchange_openmx"]
