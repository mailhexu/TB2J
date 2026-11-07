Topological Magnons (draft)
===========================

This page documents the draft topological-magnon workflow of the
``magnon_topology`` branch: Berry curvature, band-group Chern numbers, open
strip edge states and bulk-edge comparison, built on TB2J exchange results
and linear spin-wave theory.

.. note::
   The numerical conventions, thresholds and validation gates are pinned by
   SymPy-checked certificates under ``docs/sympy/`` before results may be
   interpreted. The topological calculations are opt-in; existing band/DOS
   workflows are unchanged.

Topology-ready model preparation
--------------------------------

Topological calculations consume a validated, immutable model snapshot built
by :func:`TB2J.magnon.topology.prepare_topology_model` from a TB2J results
pickle:

::

   from TB2J.io_exchange import SpinIO
   from TB2J.magnon.topology import prepare_topology_model

   exc = SpinIO.load_pickle(path="TB2J_results")
   model = prepare_topology_model(exc, source_path="TB2J_results")

Preparation validates the source before any information is discarded:

- Translations are built from the union of all selected channels, so
  DMI-only and anisotropic-exchange-only shells are retained even when the
  isotropic channel has no bond in that shell.
- Magnetic-site coverage follows the explicit atom/spin mapping
  (``index_spin``), not the pair keys: on-site-only and bond-free magnetic
  sites enter the model.
- Positions, moments and on-site tensors follow tensor spin order; the
  atom/spin map and its inverse are preserved.
- Single-ion anisotropy is canonicalized exactly once: uniaxial ``k1`` /
  ``k1dir`` terms merged by the ``SpinIO`` constructor are not re-added, and
  uniaxial terms added after construction are summed with existing tensor
  entries.
- Nonzero unsupported interactions (biquadratic exchange, NJT DMI/exchange)
  are refused. Pass ``exclude=("biquadratic",)`` to record an explicit
  selection; the resulting model is then labeled a selected approximation.
- Pair tensors are validated against reversal symmetry
  ``J_ij(R) = J_ji(-R)^T`` without fabricating missing reverse bonds.

The returned ``TopologyModel`` carries a content digest of its effective
physical content (geometry, mappings, moments, translations, canonical
tensors). Changing any coupling changes the digest; file names and channel
selection flags are provenance, not identity.

Bulk and strip calculations
---------------------------

The public calculation seam (implemented incrementally on this branch):

::

   calculate_bulk(model, bulk_request)    -> BulkTopologyResult
   calculate_strip(model, strip_request)  -> StripSpectrumResult
   compare_bulk_edge(bulk, strip, gap, edge) -> BulkEdgeComparison

Requests state the reciprocal-space mesh, the selected bands or band groups,
the strip width, orientation and termination explicitly. Results are data
objects carrying energies, curvature/flux arrays, localization diagnostics,
validity measures and provenance, and can be serialized to versioned JSON.
Invalid geometry, unstable references, unresolved gaps and vanishing links
never produce a validated topological invariant.
