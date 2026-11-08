# Signed magnon bulk–edge comparison

`compare_bulk_edge(bulk, strip, gap, edge)` compares two completed topology
results. It does not run a second Hamiltonian or silently reuse one result in
place of the other: call `calculate_bulk` and `calculate_strip` on the same
prepared `TopologyModel`, then pass those exact results to the comparison.

Given a loaded `SpinIO` object `spinio`, the Python workflow is:

```python
from TB2J.magnon.topology import (
    BulkRequest,
    StripRequest,
    calculate_bulk,
    calculate_strip,
    compare_bulk_edge,
    prepare_topology_model,
)

model = prepare_topology_model(spinio)
bulk = calculate_bulk(model, BulkRequest(mesh=(18, 18), bands=[0]))
strip = calculate_strip(model, StripRequest(width=16, kpoints=101, edge_layers=2))
comparison = compare_bulk_edge(bulk, strip, gap=1, edge="left")
record = comparison.to_json_dict()
```

`gap` is the number of positive-frequency bands below the selected gap
(`gap=1` selects the gap between bands 0 and 1); it must be a resolved global
frequency gap on the bulk mesh. The comparison uses the midpoint between the
maximum energy below and minimum energy above the gap. Bulk Chern values must
cover every band below that gap in complete, valid groups; an unresolved or
missing group never contributes an inferred value.

`edge` is `"left"` or `"right"` in the strip's declared width coordinate:
left is the minimum-width boundary and right the maximum-width boundary.
The signed crossing count adds `+1` for an edge-localized branch whose energy
increases as fractional `k_edge` increases and `-1` for a decreasing branch.
The bulk FHS chart is the ordered `(k_edge, k_width)` torus with plaquettes
`(i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)`. Under the package's overlap-link
orientation, the min-width edge is compared to `-sum(Chern below gap)` and the
max-width edge to `+sum(Chern below gap)`. The sign is pinned by the executed
Dirac domain-wall certificate `docs/sympy/magnon_topology_bulk_edge.py` and
full lattice regressions exercise the calculations on both edges.

The comparison is invalid, rather than a correspondence claim, if either
standalone result is invalid; the parent content digest/canonicalization,
magnetic-cell mapping, embedding, fixed normal momentum, orientation or strip
interior reference do not match; the bulk gap or lower-group Chern coverage is
unresolved; a crossing cannot be assigned to a resolved edge subspace; or the
signed counts do not match. Extra counterpropagating pairs are allowed because
the check compares the net signed flow, not an unsigned mode count. An
independently valid strip in a different magnetic domain remains a valid strip
result but is not comparable to the bulk.

A comparison is serialized as versioned `tb2j-topology.comparison/1` JSON.
It identifies each source result and keeps parent-model provenance separate
for bulk and strip, reports every consistency check, the resolved gap and
bulk sum, signed crossing records, unmatched/ambiguous counts, expected flow,
observed flow and final validity. Keep the bulk and strip records as standalone
outputs as well: an invalid comparison does not discard useful spectra or
curvature data.

The strip source entry also records the declared edge regions, retained/cut
directed-bond counts, retained translations and a SHA-256 fingerprint of the
reference moments actually used for that strip.

## Paired CLI

Run both calculations from one prepared exchange pickle and keep the bulk,
strip and comparison records separate:

```text
TB2J_magnon -p exchange_results \
  --topology-compare --topology-gap 1 \
  --topology-mesh 18 18 \
  --topology-strip-edge-layers 2 \
  --topology-edge left \
  --topology-out bulk.json \
  --topology-strip-out strip.json \
  --topology-comparison-out comparison.json
```

When no `--topology-bands` or `--topology-groups` is supplied, the CLI selects
all bands below the requested gap for the bulk Chern calculation. An explicit
band selection must cover all bands below the gap in complete groups that do
not straddle it. The three output paths
must be distinct. Invalid comparisons return a nonzero status and retain any
standalone bulk/strip records that were successfully written.

`valid` certifies the signed count resolved on the requested strip mesh; it is
not an automatic width- or k-mesh-convergence claim. For a converged result,
repeat the comparison with increased strip width and edge-k sampling and
confirm the same signed flow. The regression suite exercises this refinement
for both boundaries.
