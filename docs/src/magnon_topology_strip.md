Open-strip magnon spectra and edge localization
===============================================

``calculate_strip`` constructs a finite-width magnetic strip from the
prepared model's real-space interaction tensors. It cuts only bonds whose
physical target lies outside the width interval. The ordered ``LatticeEmbedding``
rows remain (edge, width, normal): the edge stays Bloch-periodic and, for a
three-dimensional source slice, every retained normal-translation coupling
keeps its prescribed fixed-normal-momentum phase. DMI tensor direction,
reverse-bond counting and on-site anisotropy are inherited from the canonical
parent model; no projected-exchange model or plotted periodic band path is used.

Python API
----------

.. code-block:: python

   from TB2J.io_exchange import SpinIO
   from TB2J.magnon.topology import (
       LatticeEmbedding,
       StripRequest,
       calculate_strip,
       prepare_topology_model,
   )

   source = SpinIO.load_pickle(path="TB2J_results")
   model = prepare_topology_model(source, source_path="TB2J_results")
   embedding = LatticeEmbedding(
       basis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], normal_k=0.25
   )
   strip = calculate_strip(
       model,
       StripRequest(
           width=8,
           kpoints=81,
           embedding=embedding,
           termination=(0.0, 0.0),
           edge_layers=1,
       ),
   )
   if not strip.valid:
       raise RuntimeError(strip.diagnostics["failures"])
   record = strip.to_json_dict()

``width`` counts magnetic-cell translations along the embedding width row.
The explicit physical cut is ``left <= x_width < width + right``, where
``termination=(left, right)`` supplies fractional-cell offsets in ``[0, 1)``.
The edge regions are the first and last ``edge_layers`` of this width coordinate;
the result records their coordinate bounds and site indices, and each site map
entry records the source spin, source-cell image, input-cell fractional and
Cartesian coordinates, and left/right/interior assignment. A primitive,
unimodular embedding is required; an explicit commensurate magnetic cell must
already be prepared for a supported nonzero ordering vector.

The strip reference defaults to copies of the parent model's moments. Local
fields and the constrained transverse gradient are recomputed from the
retained physical neighbours and on-site terms before the matrix solver runs.
Exchange fields use ``Jbar_ij = J_ij/(S_i S_j)``, with ``S_i=|mu_i|/2``, the
same normalization used by the magnon Hamiltonian.
If cutting a boundary makes the supplied reference nonstationary, the result is
invalid and no mode solve is performed. No edge relaxation is attempted.
Supply an explicit corrected ``reference_magmoms`` array only after deriving
or independently relaxing the physical reference; its rows follow
``strip_site_map(model, width, embedding, termination)`` order, and its moment
lengths must match the mapped parent moments. The saved local field, stationarity
threshold/residual, corrected reference and interior-to-parent vector mismatch
are available in the JSON. An interior mismatch leaves a standalone spectrum
inspectable but makes it ineligible for a matched bulk-edge comparison.

Localization uses physical transverse spin-amplitude matrix elements. For a
Colpa positive mode with Nambu components ``(u_i, v_i)`` and spin length
``S_i=|mu_i|/2``, the observable is

.. math::

   w_i = S_i (|u_i|^2 + |v_i|^2), \qquad
   p_{i,L/R} = \frac{\sum_{i\in L/R} w_i}{\sum_i w_i}.

The executable derivation is ``docs/sympy/magnon_topology_strip_localization.py``.
These are not Euclidean weights of Cholesky intermediate vectors. A resolved
mode's left, right and interior fractions sum to one. At a zero mode without a
normalizable vector, or at an exact/numerically unresolved positive-mode
degeneracy, per-vector fractions are null. The result instead reports the
eigenvalue spectra of the projected physical edge observables relative to the
total physical-amplitude metric; those spectra are invariant under arbitrary
unitary basis changes inside the degenerate subspace. An unresolved projection
is explicitly marked ambiguous.

CLI and JSON
------------

Use the opt-in topology CLI with ``--topology-strip-width``. For example:

.. code-block:: bash

   TB2J_magnon -p TB2J_results --topology \
       --topology-strip-width 8 --topology-strip-kpoints 81 \
       --topology-strip-termination 0 0 --topology-strip-edge-layers 1 \
       --topology-normal-k 0.25 --topology-out .tmp/strip.json

A strip-only run writes ``tb2j-topology.strip/1`` JSON to
``--topology-out`` (default ``topology_strip.json``); the separate
``--topology-strip-out`` option is available to paired workflows. The record
contains positive-mode energies in eV, endpoint-exclusive edge momenta,
left/right/interior localization arrays, physical-amplitude normalization,
degenerate-subspace projections, edge-site coordinates, the cut and retained
bond translations, the parent digest/snapshot/provenance, the full strip
reference and local fields, interior-reference mismatch, and measured
stationarity/stability thresholds. It does not serialize wavefunctions.
A valid calculation exits 0; an invalid physical result exits 1; invalid
requests or missing corrected-reference files exit 2. An invalid result does
not truncate an existing output file.

For an explicitly corrected CLI reference, pass
``--topology-strip-reference moments.json``; the file is a JSON array with
shape ``(n_strip_sites, 3)`` in the serialized ``geometry.site_map`` order.
The Python helper ``strip_site_map`` provides the same ordering before a
calculation. No file, geometry, or reference is silently relaxed.

Finite-width interpretation
---------------------------

One width is not evidence of convergence. Compare spectra and left/right
subspace localization at multiple widths and at distinct physical
terminations. The result reports the resolved per-mode edge fractions and the
basis-invariant degenerate-group projections; it explicitly does not infer
hybridization from arbitrary eigenvectors. A state with substantial weight on
both edges is evidence to inspect across widths, not by itself a hybridization
claim. The initial implementation uses dense Colpa solves: for ``N`` strip
sites each edge momentum requires a ``2N``-dimensional eigensolve, with
``O((2N)^3)`` time and ``O((2N)^2)`` working-matrix memory (plus eigensolver
work arrays). Use modest widths and record the actual width/termination rather
than implying an unbounded sparse solver.

Runnable nontrivial/trivial example
-----------------------------------

The self-contained synthetic honeycomb example varies strip width and
termination, records the stable anomalous bulk benchmark, and runs the
nontrivial model through bulk Chern, physical strip, and signed comparison
using both Python and the actual paired CLI. It also runs a ``D_z=0`` trivial
control with sublattice-dependent easy-axis anisotropy; the comparison refuses
its unresolved in-gap states rather than asserting an edge correspondence:

.. code-block:: bash

   python examples/magnon/magnon_topology_strip.py \
       --output-dir .tmp/magnon-topology-strip

The output JSON records retain the resolved scientific arrays and source
provenance. The selected edge regions use two width layers for the signed-flow
benchmark; the default standalone strip localization region remains one layer.
The reported width change is finite-sample evidence, not a continuum
convergence certificate. DMI reversal, additional counterpropagating pairs,
and material-specific agreement require separate calculations.
