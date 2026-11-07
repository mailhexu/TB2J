"""Topological magnon calculations for TB2J (draft implementation).

Public consumer seam (architecture section 4):

- ``prepare_topology_model``: topology-ready, input-validated model
  preparation from TB2J results (ADR-001).
- ``calculate_bulk``: bulk Berry curvature and band-group Chern numbers.
- ``calculate_strip``: open-strip spectra and edge localization.
- ``compare_bulk_edge``: signed bulk-edge correspondence of completed
  results.

Names and contracts follow the approved architecture document; they are not
present on ``main``.
"""

from TB2J.magnon.topology.bosonic import (
    BosonicModes,
    ModeSolverError,
    solve_bosonic_modes,
)
from TB2J.magnon.topology.bulk import (
    BulkRequest,
    BulkTopologyResult,
    calculate_bulk,
)
from TB2J.magnon.topology.embedding import LatticeEmbedding, identity_embedding
from TB2J.magnon.topology.preparation import (
    CANONICALIZATION_VERSION,
    SUPPORTED_CHANNELS,
    UNSUPPORTED_CHANNELS,
    TopologyModel,
    TopologyPreparationError,
    prepare_topology_model,
)

__all__ = [
    "CANONICALIZATION_VERSION",
    "SUPPORTED_CHANNELS",
    "UNSUPPORTED_CHANNELS",
    "BosonicModes",
    "BulkRequest",
    "BulkTopologyResult",
    "LatticeEmbedding",
    "ModeSolverError",
    "TopologyModel",
    "TopologyPreparationError",
    "calculate_bulk",
    "identity_embedding",
    "prepare_topology_model",
    "solve_bosonic_modes",
]
