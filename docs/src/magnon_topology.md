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

``calculate_bulk`` (implemented) runs the endpoint-exclusive periodic mesh of
a ``BulkRequest(mesh=(nu, nv), bands=..., groups=..., embedding=...,
gap_tol=..., link_tol=...)`` over a topology-ready model. A request carries
ONE band group (``bands``) or several independent groups (``groups``,
mutually exclusive); groups may hold internal degeneracies, and their
invariant is basis-independent under unitary mixing within the group
(certified symbolically). A single-band group is refused when that band
touches another band anywhere on the mesh, while the full group of all
positive bands remains valid with its exactly vanishing total Chern number.
Per-group records (Chern value, plaquette flux, vertex curvature, minimum
external gap, minimum link overlap, failures) appear under ``groups`` in the
JSON; the group-curvature convention is documented in the record
(``vertex value = average of the four adjacent plaquette fluxes``).

Each mesh point is solved with the certified Colpa metric construction
(``docs/sympy/magnon_topology_lswt_ferromagnet.py``): with ``H = L L^dag`` the
physical energies are the ascending top half of ``eig(L^dag g L)``, and the
mode vectors ``sqrt(omega) L^{-dag} u`` carry unit metric norm. The Berry flux
uses Fukui-Hatsugai-Suzuki links ``det[Psi(k)^dag Psi(k')] / |...|`` over the
pinned plaquette order ``(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)`` on the
``(a1,a2)`` torus; vertex curvature is the gauge-safe average of adjacent
plaquette fluxes.

Refinement policy: ``refine_bulk(model, request, factor=2)`` reruns the
request on the factor-refined mesh and reports band Chern values, measured
minimum external gaps and their agreement as convergence EVIDENCE. The
measured minimum gap converges downward toward the continuum minimum; integer
agreement on one or two meshes is never a certificate of continuous
isolation, and non-finite data or vanishing links always invalidate the
affected groups. The CLI flags ``--topology-groups "0;1"`` and
``--topology-refine`` expose both.

Pinned executable conventions (certificate-verified): for the gapped
honeycomb ferromagnet with NNN Kane-Mele DMI ``D_z`` and easy-axis SIA,
``omega-(Gamma) = 2K``, ``omega+(Gamma) = 12 J1 + 2K`` at ``mu = 2 muB``, and
the band Chern numbers are ``C_lower = +sign(D_z)`` and
``C_upper = -sign(D_z)``. A gapped ``D = 0`` reference must never be fed
unrefined: the valley gap closing is detected only when the mesh resolves
``K = (1/3, 2/3)`` and ``K' = (2/3, 1/3)``; the JSON record carries the
measured minimum gap and a mesh-refinement note.

A result is ``valid=False`` with explicit reasons (and no Chern value) when
the reference is nonstationary on the retained bonds, the spectrum is
energetically unstable (no diagonal shift is ever applied), the requested
band group touches a stable zero mode, the band group is not isolated within
``gap_tol``, or a plaquette link determinant vanishes within ``link_tol``.
Requests and results serialize to versioned JSON
(``tb2j-topology.bulk/1``).

Mode solver and validity classes
--------------------------------

The shared solver ``solve_bosonic_modes`` (certified in
``docs/sympy/magnon_topology_bosonic_modes.py``) classifies each mesh point:

- ``positive_definite``: stable gapped set; metric-normalized positive modes
  attached (generalized residual and metric-norm error reported).
- ``zero_mode``: stable zero-frequency directions exist; they are reported as
  a non-normalizable null space, never as invented normalized vectors.
  Strictly positive modes still carry normalizable vectors, so a separated
  band group (gap from the zero level above ``gap_tol``) can be validated;
  a group touching zero is refused.
- ``energetically_unstable``: the Hessian has a resolved negative direction.
  The dynamical frequency character (``real`` or ``complex`` from ``gH``) is
  reported as a diagnostic; a diagonal shift is refused because it would not
  legitimize topology of the unshifted model. The legacy band/DOS shift
  fallback is untouched.
- ``dynamically_unstable``: the metric spectrum violates the +-pairing
  structure of a bosonic BdG set.

DMI never pairs a collinear ferromagnet (``u^T eps(D) u = 0``); genuine
anomalous terms require symmetric anisotropic exchange, exercised by the
Jani-paired honeycomb benchmark in ``tests/tests/test_magnon_topology_modes.py``.

The opt-in CLI mode writes the same JSON record:

.. code-block:: bash

   TB2J_magnon -p TB2J_results --topology --topology-mesh 12 12 \
       --topology-bands 0 --topology-out topology_bulk.json

exit code 0 for a validated invariant, 1 for a refused one (reasons printed
and recorded).

``calculate_strip`` and ``compare_bulk_edge`` are specified but not yet
implemented (tickets 06 and 07).
