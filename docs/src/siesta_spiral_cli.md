# Certified SIESTA spiral CLI (`tb2j-siesta-spiral`)

This workflow turns a producer-recorded SIESTA spiral reference into a
**certified, intra-reference SpinIO exchange export**. It executes the gates;
a manifest or a supplied pass report is not itself a certificate. This is
separate from the TBUpy `.spiral.nc` workflow in
[Additive frozen-spiral response](spiral_response.md) and from the existing
collinear `siesta2J.py` command.

## Command and inputs

```bash
tb2j-siesta-spiral manifest.json --output certified-reference
```

`--output` is a **new directory**, not a filename. It must not already exist,
and its parent directory must exist. There is no overwrite or raw-response-only
export option. The CLI stages its artifacts and publishes the directory by
atomic rename only after the actual SpinIO pickle writer and subsequent load
validation succeed. A refusal exits nonzero with an actionable diagnostic and
publishes no exchange directory.

Publication requires Linux `renameat2(RENAME_NOREPLACE)` support. Unsupported
platform/libc/kernel support is a refusal, not a fallback to overwriting an
existing destination.

Prepare these inputs before writing the manifest:

- `bundle`: the versioned, non-pickled NPZ accepted by
  `TB2J.interfaces.siesta_spiral_io.load_fixture`. It contains the sparse
  spiral Hamiltonian/overlap, geometry and conventions, plus the producer's
  recorded k points, weights, EIG bands, Fermi reference, electron count and
  temperature. This command does not ingest an arbitrary HSX directly as its
  undisplaced reference, nor manufacture missing producer records.
- `structure`: a structure readable by `ase.io.read`. Supply the actual
  structure corresponding to the reference and its caller-owned per-atom
  moments, charges and atom-to-spin mapping in `spin`.
- Producer evidence: the actual pseudopotential files, basis definitions and
  producer input files named by the mandatory `producer` record. Every
  referenced role must also have a byte-digest descriptor in `inputs`.
- For displacement: `displacement_plus` and `displacement_minus` are the
  matching **+h and -h displaced-SCF HSX files**, respectively. Their geometry
  must be anchored to the undisplaced reference, not just to their own midpoint.
- Optional gate-report roles: complete JSON verdict dictionaries from the
  producer/API gate workflow, named in `gate_reports` and digest-pinned in
  `inputs`. They are consumed evidence, not a replacement for executing gates.

Every descriptor has exactly `path` and `sha256`. Relative paths are resolved
against the **manifest's directory**, not the shell's current directory;
absolute paths are accepted. SHA-256 is the lowercase, 64-character digest of
**file bytes**, not of a filename or a reserialized object. All declared input
bytes, including structure, reports and displaced endpoints, are checked
before the numerical pipeline starts and consumed from digest-checked snapshots.

## Version-1 manifest schema

The following JSON Schema describes the document shape. Cross-record scientific
checks listed after it are also mandatory. Unknown settings, duplicate JSON
keys, unsupported modes and non-finite numerical values are not a way to
request a best-effort calculation.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "tb2j-siesta-spiral manifest version 1",
  "type": "object",
  "additionalProperties": false,
  "required": ["version", "inputs", "producer", "operator", "axes_by_site", "ensemble", "response", "mapping", "spin", "required_gates", "requires_plusu"],
  "properties": {
    "version": {"type": "integer", "const": 1},
    "inputs": {
      "type": "object",
      "required": ["bundle", "structure"],
      "additionalProperties": {"$ref": "#/$defs/input"}
    },
    "producer": {
      "type": "object",
      "additionalProperties": false,
      "required": ["kind", "software_revisions", "pseudopotentials", "basis_definitions", "numerical_settings", "input_files"],
      "properties": {
        "kind": {"enum": ["siesta", "analytic"]},
        "software_revisions": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "string", "minLength": 1}},
        "pseudopotentials": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "basis_definitions": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "numerical_settings": {"type": "object", "minProperties": 1},
        "input_files": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}
      },
      "allOf": [{
        "if": {"properties": {"kind": {"const": "siesta"}}},
        "then": {"properties": {"pseudopotentials": {"minItems": 1}}}
      }]
    },
    "operator": {
      "type": "object",
      "additionalProperties": false,
      "required": ["mode", "orbitals_by_site"],
      "properties": {
        "mode": {"enum": ["onsite_delta", "weighted_total_h"]},
        "orbitals_by_site": {
          "type": "object",
          "minProperties": 1,
          "propertyNames": {"minLength": 1},
          "additionalProperties": {"type": "array", "items": {"type": "integer", "minimum": 0}}
        }
      }
    },
    "axes_by_site": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "type": "object",
        "minProperties": 1,
        "propertyNames": {"minLength": 1},
        "additionalProperties": {"$ref": "#/$defs/vector3"}
      }
    },
    "ensemble": {
      "type": "object",
      "additionalProperties": false,
      "required": ["kind", "n_occupied", "electron_count", "fermi_energy_ev", "smearing_width_ev", "gap_ev"],
      "properties": {
        "kind": {"enum": ["gapped_fixed_occupations", "metallic_fermi_dirac_fixed_n"]},
        "n_occupied": {"type": "integer", "minimum": 1},
        "electron_count": {"type": "number", "exclusiveMinimum": 0},
        "fermi_energy_ev": {"type": "number"},
        "smearing_width_ev": {"type": "number", "minimum": 0},
        "gap_ev": {"type": "number"},
        "declaration": {"type": "string"}
      }
    },
    "response": {
      "type": "object",
      "additionalProperties": false,
      "required": ["sampling", "finite_steps"],
      "properties": {
        "sampling": {"const": "recorded"},
        "finite_steps": {"type": "array", "minItems": 2, "items": {"type": "number", "exclusiveMinimum": 0}},
        "contour": {"$ref": "#/$defs/contour"},
        "contour_node_rungs": {"type": "array", "items": {"type": "integer", "minimum": 32}}
      }
    },
    "mapping": {
      "type": "object",
      "additionalProperties": false,
      "required": ["reference_directions_by_site"],
      "properties": {
        "reference_directions_by_site": {"type": "object", "minProperties": 1, "additionalProperties": {"$ref": "#/$defs/vector3"}},
        "chi2_floor_eV_per_rad2": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
        "torque_chi2_floor_eV_per_rad": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]}
      }
    },
    "spin": {
      "type": "object",
      "additionalProperties": false,
      "required": ["spinat", "charges", "index_spin"],
      "properties": {
        "spinat": {"type": "array", "items": {"$ref": "#/$defs/vector3"}},
        "charges": {"type": "array", "items": {"type": "number"}},
        "index_spin": {"type": "array", "items": {"type": "integer", "minimum": -1}}
      }
    },
    "required_gates": {
      "type": "array",
      "items": {"enum": ["AC-003", "AC-005", "AC-007", "AC-008", "AC-009"]},
      "allOf": [{"contains": {"const": "AC-003"}}, {"contains": {"const": "AC-005"}}, {"contains": {"const": "AC-007"}}]
    },
    "requires_plusu": {"type": "boolean"},
    "gate_reports": {"type": "array", "items": {"type": "string"}},
    "displacement": {
      "type": "object",
      "additionalProperties": false,
      "required": ["pattern", "gauge", "reference_fermi_ev"],
      "properties": {
        "pattern": {
          "type": "object",
          "additionalProperties": false,
          "required": ["atom_index", "axis", "step_bohr"],
          "properties": {
            "atom_index": {"type": "integer", "minimum": 1},
            "axis": {"enum": ["x", "y", "z"]},
            "step_bohr": {"type": "number", "exclusiveMinimum": 0},
            "branch": {"const": "central", "default": "central"}
          }
        },
        "gauge": {"const": "periodic-image-sum"},
        "reference_fermi_ev": {"type": "number"}
      }
    }
  },
  "allOf": [
    {
      "if": {"properties": {"ensemble": {"properties": {"kind": {"const": "gapped_fixed_occupations"}}}}},
      "then": {"properties": {"response": {"required": ["contour"]}, "ensemble": {"properties": {"gap_ev": {"exclusiveMinimum": 0}}}}},
      "else": {"properties": {"response": {"not": {"anyOf": [{"required": ["contour"]}, {"required": ["contour_node_rungs"]}]}}, "ensemble": {"properties": {"smearing_width_ev": {"exclusiveMinimum": 0}}}}}
    },
    {
      "if": {"required": ["displacement"]},
      "then": {"properties": {"inputs": {"required": ["displacement_plus", "displacement_minus"]}, "required_gates": {"contains": {"const": "AC-009"}}, "operator": {"properties": {"mode": {"const": "onsite_delta"}}}}},
      "else": {"properties": {"required_gates": {"not": {"contains": {"const": "AC-009"}}}}}
    },
    {
      "if": {"properties": {"requires_plusu": {"const": true}}},
      "then": {"properties": {"required_gates": {"contains": {"const": "AC-008"}}}}
    }
  ],
  "$defs": {
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["path", "sha256"],
      "properties": {"path": {"type": "string", "minLength": 1}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}
    },
    "vector3": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
    "contour": {
      "type": "object",
      "additionalProperties": false,
      "required": ["z0", "occupied_min", "gap_ev", "margin_ev", "n_nodes"],
      "properties": {
        "z0": {"type": "number"},
        "occupied_min": {"type": "number"},
        "gap_ev": {"type": "number", "exclusiveMinimum": 0},
        "margin_ev": {"type": "number", "exclusiveMinimum": 0},
        "n_nodes": {"type": "integer", "minimum": 32}
      }
    }
  }
}
```

Additional contract rules:

- `inputs` contains **exactly** `bundle`, `structure`, roles referenced by
  `producer.pseudopotentials`, `producer.basis_definitions` or
  `producer.input_files`, the two displaced roles when requested, and roles
  listed in `gate_reports`. Every referenced role needs its own descriptor;
  unrelated or misspelled roles are refused. A `producer_record` role is not
  implicitly special: include it in `producer.input_files` to consume it.
- `producer` is required, not reconstructed from a bundle label. It declares
  `kind` (`siesta` or `analytic`), actual `software_revisions` (a nonempty
  name-to-nonempty-revision-string object), role lists `pseudopotentials`,
  `basis_definitions`, `input_files`, and a nonempty `numerical_settings`
  object. Basis/input lists are always nonempty. SIESTA pseudopotentials are
  also nonempty; only an actual analytic producer may have none. Preserve
  actual producer revisions/binary identity and all setup parameters; missing
  evidence is refused, never replaced by an invented revision or a change
  from `siesta` to `analytic`.
- `orbitals_by_site` is a disjoint, complete partition of all **zero-based
  spatial orbital indices**, not atom indices or spinor-band indices. Site
  labels agree with `axes_by_site` and `reference_directions_by_site`. Supply
  the two nondegenerate transverse chart axes per site required by the mapping,
  together with the producer-certified reference moment directions; do not
  infer a reference direction from the axes alone.
- `spin` has one entry per structure atom in all three arrays. `index_spin`
  assigns each magnetic atom exactly one contiguous response-site index in
  recorded `site_order`; `-1` marks a nonmagnetic atom. The structure cell and
  each site's owned orbital centers must match the bundle geometry. Each
  nonzero magnetic moment must point along its declared reference direction.
  These data are caller-supplied, not inferred from orbital ownership.
- Gapped mode uses step occupations, with `electron_count == n_occupied` and
  positive `gap_ev` matching the sampled gap. The contour must enclose occupied
  poles only: `occupied_min < z0`, with `z0` inside the gap, and
  `0 < margin_ev < gap_ev / 2`. All contour energies are in eV.
- Metallic mode uses positive `smearing_width_ev` in eV and **one global fixed-N
  Fermi-Dirac chemical potential**. It must equal the producer's recorded
  electronic temperature times `k_B` (`8.617333262145e-5 eV/K`) within
  `1e-10 eV`; missing or invalid producer temperature is refused. Declare the
  producer's Fermi reference, count, width and sampled gap, not a convenient
  fictitious insulating gap. Remove both `contour` and `contour_node_rungs`:
  the metallic evaluator has no occupied-pole contour.
- `finite_steps` are positive, pairwise-distinct rotation angles in radians,
  with at least two independent rungs. Values within `1e-12 rad` are refused
  as duplicates. Optional `contour_node_rungs` changes gapped contour quadrature
  node counts only. `sampling` is always `recorded`: no interpolation, new k
  mesh, k-point ladder or unrecorded integration setting is silently substituted.
- The optional mapping floors are precisely the two `map_response_set`
  keyword arguments shown above; omission or `null` uses the API defaults.
  They are statistic denominator floors, not new material uncertainty data.
- `requires_plusu` is always explicit. Although the manifest can declare a +U
  requirement, **`requires_plusu: true` or required `AC-008` is refused by this
  CLI**: the NPZ loader has no certified Hubbard ingestion. Use the Python
  `ingest_dftu_run` → `run_ac008_gate` → `build_exchange_export` workflow with
  matching XC+Hubbard provenance instead. Never relabel +U as XC-only to pass.

## Example: analytic gapped, two-site reference

The following writes a JSON manifest with real byte digests rather than dummy
hash strings. Run it beside `reference.npz` and `structure.extxyz` containing
the analytic two-site pencil declared in
`tests/test_siesta_spiral_cli.py::analytic_manifest`: onsite spin splitting
2 eV, spin-independent hopping 0.2 eV, identity overlap, two occupied spinor
bands, sampled gap 1.6 eV and gap center 0 eV. The bundle itself records this
analytic basis and source input. The producer revision names that model, not
an invented upstream SIESTA version. This is a structural usage example, not
material-validation or successful-run evidence.

```bash
bundle_digest=$(sha256sum reference.npz)
structure_digest=$(sha256sum structure.extxyz)
cat > manifest.json <<JSON
{
  "version": 1,
  "inputs": {
    "bundle": {"path": "reference.npz", "sha256": "${bundle_digest%% *}"},
    "structure": {"path": "structure.extxyz", "sha256": "${structure_digest%% *}"}
  },
  "producer": {
    "kind": "analytic",
    "software_revisions": {"analytic_model": "two-site pencil v1"},
    "pseudopotentials": [],
    "basis_definitions": ["bundle"],
    "numerical_settings": {
      "onsite_spin_splitting_ev": 2.0,
      "spin_independent_hopping_ev": 0.2,
      "overlap": "identity",
      "electron_count": 2.0,
      "recorded_kpoints": [[0.0, 0.0, 0.0]]
    },
    "input_files": ["bundle"]
  },
  "operator": {
    "mode": "onsite_delta",
    "orbitals_by_site": {"A": [0], "B": [1]}
  },
  "axes_by_site": {
    "A": {"x": [1, 0, 0], "y": [0, 1, 0]},
    "B": {"x": [1, 0, 0], "y": [0, 1, 0]}
  },
  "ensemble": {
    "kind": "gapped_fixed_occupations",
    "n_occupied": 2,
    "electron_count": 2.0,
    "fermi_energy_ev": 0.0,
    "smearing_width_ev": 0.008617333262,
    "gap_ev": 1.6
  },
  "response": {
    "sampling": "recorded",
    "finite_steps": [0.004, 0.002],
    "contour": {"z0": 0.0, "occupied_min": -1.2, "gap_ev": 1.6, "margin_ev": 0.3, "n_nodes": 256},
    "contour_node_rungs": [128, 256]
  },
  "mapping": {
    "reference_directions_by_site": {"A": [0, 0, 1], "B": [0, 0, 1]}
  },
  "spin": {
    "spinat": [[0, 0, 1], [0, 0, 1]],
    "charges": [0.0, 0.0],
    "index_spin": [0, 1]
  },
  "required_gates": ["AC-003", "AC-005", "AC-007"],
  "requires_plusu": false
}
JSON

tb2j-siesta-spiral manifest.json --output certified-reference
```

For a **SIESTA** reference, use `producer.kind: "siesta"` and the actual
producer revision/binary identity and numerical settings from its provenance.
For example, the curated O2 producer uses `nc/O.psf` as input role
`oxygen_psf` and `nc/o2_fm.fdf` as `producer_fdf`: add both descriptors,
set `pseudopotentials` to `["oxygen_psf"]`, and reference `producer_fdf` in
both `basis_definitions` and `input_files`. Include any other consumed setup
files as well. Do not copy the analytic producer declaration onto DFT data,
guess software revisions, or substitute a new pseudopotential with the same
filename.

For a displaced-reference calculation, add both endpoint descriptors with their
actual byte digests, add `AC-009` to `required_gates`, and add, for example:

```json
{
  "displacement": {
    "pattern": {"atom_index": 1, "axis": "x", "step_bohr": 0.01, "branch": "central"},
    "gauge": "periodic-image-sum",
    "reference_fermi_ev": 0.0
  }
}
```

This is an addition to the manifest, not a standalone manifest. `atom_index`
is **one-based SIESTA indexing**, `step_bohr` is the positive displacement
magnitude in Bohr, and `reference_fermi_ev` must be the matching energy
reference. Omitting `branch` selects `central`. Displacement is supported only
with `operator.mode: "onsite_delta"`; `weighted_total_h` plus displacement
is refused during manifest validation, before numerical work.

The CLI explicitly carries the manifest-declared `periodic-image-sum` gauge
into the loaded reference metadata before executing
`DisplacedReferenceFDProvider`, `vertex_displacement_derivative` and
`run_ac009_validation`; the NPZ loader does not supply a gauge, and none is
inferred from filenames or data. Supplied reports cannot skip these steps.
The localized observable requires displaced-projector `dP` endpoints, which
this CLI does not ingest, so `gauge: "localized"` is refused rather than
relabeled.

## Executed gates and refusal precedence

Exceptions short-circuit in the stage order below. AC-005/AC-007 verdicts are
recorded and enforced at the API export stage, rather than immediately
short-circuiting subsequent calculations. Thus a displacement exception may
precede rejection of an already recorded failed response/mapping verdict.
No later successful report can override a failed verdict.

1. **Destination check:** an existing output path or symlink is never
   overwritten, and the parent directory must already exist.
2. **Manifest/settings/producer/operator checks:** strict version and keys,
   complete producer evidence/role declarations, orbital ownership, baseline
   gates, recorded sampling and supported displacement gauge/operator.
   Explicit +U/required AC-008 and weighted-total-H displacement are refused
   before numerical work, even if a supplied report claims to pass them.
3. **Input identity:** verify every declared file's byte digest before using
   the reference, structure, reports or endpoints.
4. **Producer/AC-003:** load the bundle, construct the ensemble and require its
   electron count/Fermi energy to match the producer reference. Execute the
   EIG/eigenvalue oracle, then the electron count oracle; do not repair a
   corrupt pencil or change the producer mesh.
5. **Response/AC-005:** extract the declared vertex, check structure/spin
   identity, then evaluate the gapped contour or metallic fixed-N response.
   Record finite-rotation agreement plus adjacent finite-step convergence.
   Missing or failed agreement is not a pass, even if later stages complete.
6. **AC-007 mapping:** construct the mapping with the declared reference
   directions, preserving compatibility, residuals and identifiability.
7. **Optional displacement/AC-009:** compute and validate the matching central
   derivative and vertex derivative with its `periodic-image-sum` label.
8. **Certified API export:** `build_exchange_export` is the sole authority for
   assembling exchange. Its audits order response/mapping identity and
   operator provenance before consumed-verdict shape, ensemble consistency,
   +U identity/AC-008, failed consumed gates, Heisenberg compatibility and
   displacement/AC-009 certification. A supplied ensemble mismatch can
   therefore precede a supplied gate failure. Consumed verdicts are ordered
   as executed AC-003, AC-005, AC-007, then supplied reports in manifest order,
   generated AC-009 when present, and the response's AC-005 cross-check.
   The generated failing AC-007 verdict therefore raises `failed_gate` before
   the later `non_heisenberg_compatible_response` check. Required gates are
   executed regardless of the supplied reports.
9. **Writer/load/publication:** construct the existing SpinIO with caller data,
   invoke its actual pickle writer, load and validate the written pickle, and
   only then atomically publish the new directory. A writer returning without
   a loadable artifact is not success.

API diagnostics retain their stable reason identifiers:

| Reason | Action |
| --- | --- |
| `missing_operator_provenance` | Restore the matching source/vertex declaration and certified orbital ownership. |
| `incompatible_ensemble` | Use one matched producer ensemble across response and consumed verdicts. |
| `uncertified_plusu` | Supply matching AC-008 XC+Hubbard certification through the supported Python API; this CLI cannot ingest it. |
| `failed_gate` | Resolve and rerun the named gate; a supplied failure remains a refusal. |
| `non_heisenberg_compatible_response` | Obtain an AC-007 compatible, identifiable mapping; the CLI has no raw-only exchange escape. |
| `uncertified_displacement` | Supply the matching executed AC-009 chain and supported displacement gauge. |

When reached after earlier audits/verdicts, an explicitly failed AC-008 report
is `uncertified_plusu`; an explicitly failed AC-009 or generic report is
`failed_gate`; missing AC-009 certification is `uncertified_displacement`.
Malformed manifests, unsupported CLI settings, byte mismatches and writer
errors have actionable CLI diagnostics and need not be API refusal classes.
Neither a missing report nor a boolean pass flag can stand in for the required
producer-bound certification.

## Artifacts and reproducibility

A successful directory contains:

| Artifact | Meaning |
| --- | --- |
| `TB2J.pickle` | Actual `SpinIO.write_pickle` output, reloaded and validated before publication. Certified metadata travels in `debug_dict`. |
| `provenance.json` | The API export metadata except tuple-keyed `spinio_kwargs`, plus the CLI identity record. Contains raw response, mapping conventions/residuals/identifiability, units, reference and vertex provenance, ensemble, convergence evidence and consumed verdicts; optional displacement records retain their gauge. |
| `run.json` | Actual UTC run timestamp and the output-artifact SHA-256 record, separate from scientific reproducibility identity. |

`provenance.json` retains the API's `provenance` record with `input_digests`,
`code_revision`, independent `code_sha256`, `created_utc`, `gate_verdicts` and
`content_sha256`. These API input digests identify consumed scientific payloads.
Its additional `cli` object contains:

- `manifest_sha256`: the original manifest bytes;
- `input_digests`: role-to-digest identities of the input file bytes;
- `producer`: the explicit producer kind, software revisions, pseudopotential/
  basis/input role lists and numerical settings, retained without fallback;
- `settings`: the complete parsed manifest, including input descriptors;
- `settings_sha256`: canonical manifest settings **excluding `inputs`**;
- `code_revision` and `code_sha256`: revision identity and independent code hash
  covering the `siesta_spiral*.py` interfaces, CLI script and SpinIO module;
- `reproducibility_sha256`: the canonical complete export plus CLI identities,
  computed before adding this hash itself, with exactly
  `provenance.created_utc` and `provenance.content_sha256` removed. Other
  scientific records and input identities remain included.

Canonical JSON uses sorted keys, compact separators and no non-finite numbers.
Actual timestamps are retained; they are not replaced by a fixed time to make
hashes look stable. `run.json` contains `created_utc` and `output_sha256`, a
filename-to-byte-digest mapping for `TB2J.pickle` and `provenance.json`.
Artifact-byte hashes, including the timestamp-bearing pickle and provenance,
are integrity checks, **not** the timestamp-independent scientific identity.

To reproduce, keep the exact manifest, every digest-pinned input, declared
settings and recorded code revision/content. Run again into a different,
nonexistent output directory and compare `cli.reproducibility_sha256` along
with the preserved scientific records. Merely matching a Git HEAD does not
account for modified implementation bytes. Editing manifest formatting changes
its byte digest, even if parsed settings are otherwise unchanged.

## Scope and interpretation

- A pipeline-consistency pass is **not material validation**, a self-consistent
  total-energy exchange measurement, or proof of k-mesh/contour/material
  convergence. The recorded mesh cannot be refined by a hidden CLI option;
  regenerate and certify the producer inputs for a different mesh. Inspect
  the recorded ladders and perform the separate material/convergence campaign.
- `onsite_delta` is the established onsite rotating operator. The explicitly
  labeled `weighted_total_h` variant rotates incident magnetic blocks with
  their declared weights; it must not be silently compared as the same
  operator or promoted to a validated collinear anchor.
- SpinIO contains only determined intra-reference pairs at `R=(0,0,0)`, in
  both ordered directions at `j_stored_eV` (half the pair-once coupling), with
  eV/unit-direction conventions. This is **not a real-space Fourier inversion**
  or a translation-resolved exchange table.
- A marginal/incompatible mapping does not become exchange by exporting raw
  data. Partial chart-frame DMI/Gamma stays labeled metadata, not a fabricated
  full Cartesian tensor. `dH/dR`, `dS/dR` and `dV/dR` are not `dJdx`.
- Explicit +U CLI ingestion remains unsupported; the certified Python API
  remains supported. The API's diagnostic-only `raw_response_only` path is
  not an option on this exchange-producing CLI.

