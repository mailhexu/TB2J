# Symmetry-shell exchange files and native conversion

`TB2J_shell` and `TB2J.io_exchange.shell_workflow` provide an editable TOML representation of TB2J exchange data and standard results export. This workflow does not calculate exchange parameters.

The shell document stores one value per **symmetry orbit of ordered atom pairs**. Equal bond lengths do not define an orbit: equal-distance pairs in different orbits remain separate shells, while pairs related by a space-group or magnetic-space-group operation (including centering translations) belong to one shell. Reversing a pair is a separate physical transpose convention, not an automatic orbit operation.

Generate from a structure or a **trusted** TB2J pickle; export the edited shell file to standard TB2J results. Solver-specific outputs are included only when their representation is supported; omissions are reported.

## Generate a structure template from Python

```python
from TB2J.io_exchange.shell_workflow import generate_shells

result = generate_shells(
    "POSCAR", "exchange.shells.toml", magnetic_species=["Fe"], cutoff=6.0
)
```

Select magnetic sites with `magnetic_species` or, alternatively,
`magnetic_atoms` containing one-based full-structure indices. The cutoff is in
Angstrom. For magnetic space-group symmetry, pass `symmetry="msg"` and complete
`reference_spins` in full atom order; `symprec` and `magnetic_tol` are configurable.
Use `mode="tensor"` for 3×3 exchange values. Each shell remains `unfilled` until edited.

Both this entry point and `write_shells` add format explanations. Set
`equivalent_comments=False` to omit equivalent-pair lists without omitting
the explanations. Comments are informational and can be deleted.

## Generate from a trusted TB2J pickle

```python
from TB2J.io_exchange.shell_workflow import generate_shells

result = generate_shells("TB2J_results/TB2J.pickle", "exchange.shells.toml")
```

Only load a pickle from a trusted source: Python pickle loading can execute code.
The generator writes `exchange.shells.context.npz` beside the TOML. Keep the two
files together, including when moving or renaming them: the sibling name is the
TOML filename with its final suffix replaced by `.context.npz`. The companion
contains data, not the pickle, and its digest detects accidental mismatches; it
does not establish authenticity. The original pickle is not needed after generation.
Selectors, reference-spin overrides and cutoff apply to structure input only.
Native import preserves anisotropic exchange by selecting tensor mode when needed.
Symmetry-inconsistent values are rejected unless `project=True` explicitly requests
projection. A companion requiring unsupported objects, constraints, or a calculator
is rejected instead of silently dropping them.

## A complete hand-authored scalar document

The following file is complete and independent: copy it, edit the `J` values,
and export it. It describes one Fe atom in the primitive cell of body-centered
cubic iron (a = 2.87 Å), hand-written as TOML schema v2 (the `schema_version`
key is optional for hand-written v2 input) with space-group symmetry. Every
explanatory comment may be deleted without changing the physics.

```toml
# Complete hand-authored scalar (isotropic) example: one Fe atom in the
# primitive cell of body-centered cubic iron (a = 2.87 Angstrom).
#
# One [[shell]] stores ONE exchange value for one symmetry orbit of ordered
# atom pairs. Pairs related by a space-group operation (including lattice
# centering) share a shell; equal bond lengths alone never merge shells, and
# a reversed pair is a separate shell unless an operation maps it back.

# Exchange model of this file: "isotropic" = one scalar J per shell.
# (The alternative is "tensor" = one Cartesian 3x3 matrix per shell.)
exchange = "isotropic"

# Wire unit of every J below: "meV" (default, may be omitted) or "eV".
# Distances/cutoff are Angstrom, moments are mu_B; Python values and the
# exported native results are always eV (1 meV = 0.001 eV).
energy_unit = "meV"

# Optional, informational neighbor radius in Angstrom. It records the radius
# used when a template was generated; it does NOT filter the shells below.
cutoff = 4.5

[structure]
coordinates = "fractional"

# Exactly three rows of three Cartesian lattice-vector components (Angstrom):
# body-centered cubic primitive vectors of Fe (conventional a = 2.87 Angstrom).
lattice = """
-1.435 1.435 1.435
1.435 -1.435 1.435
1.435 1.435 -1.435
"""

# Atom rows, exactly seven whitespace-separated columns:
#   atom-index label x y z magnetic spin-index
# atom-index  sequential one-based index in row order (1, 2, ...).
# label       unique element symbol + positive integer (Fe1); gives the species.
# x y z       finite raw fractional coordinates; values outside [0, 1) are kept,
#             never wrapped, so native fixed-R bonds keep their geometry.
# magnetic    "Y" for a selected magnetic site, "N" otherwise.
# spin-index  zero-based index into the native spin map: every Y row together
#             must map exactly onto 0..nspin-1; an N row carries a negative
#             sentinel, "-" spelling -1. This mapping is never reordered.
# Blank lines and '#' comments are ignored inside row blocks.
atoms = """
1 Fe1 0.0 0.0 0.0 Y 0
"""

[reference_spins]
# One row per atom in full atom order (every atom, magnetic or not):
#   label m         signed collinear moment along z, or
#   label mx my mz  Cartesian axial moment vector.
# Moments are in mu_B. Rows must be complete; missing atoms are not zero-filled.
moments = """
Fe1 2.2
"""

[symmetry]
# "spacegroup" (CLI/Python: "sg") detects the crystal space group and ignores
# magnetic moments. The alternative "magnetic" (CLI/Python: "msg") detects the
# magnetic space group: it REQUIRES the complete [reference_spins] moments above
# plus a positive magnetic_tol (mu_B), and may include time-reversal (primed)
# operations. mode and the scalar/tensor choice are independent.
mode = "spacegroup"
# Structural symmetry tolerance in Angstrom (default 1e-5).
symprec = 1e-5

# One [[shell]] per symmetry orbit of ordered atom pairs. The representative
# names ONE member: "label-first label-second Rx Ry Rz", where (Rx Ry Rz) is
# the lattice translation moving the SECOND atom into its bonded image.
[[shell]]
representative = "Fe1 Fe1 1 0 0"
# Editable exchange of this orbit, in the wire unit above (meV).
# While editing, J = "unfilled" marks an untouched shell; export rejects
# unfilled shells until every one carries a real number.
J = 45.0

[[shell]]
# Second-nearest neighbors: R = (0, 1, 1) is the Cartesian vector a2 + a3 =
# (2.87, 0, 0) Angstrom, the bcc second-neighbor bond along x.
representative = "Fe1 Fe1 0 1 1"
J = 15.0

[metadata]
# Optional free-form metadata; never affects exchange or symmetry.
source_note = "hand-edited scalar example"

# Generator/writer comments, including the per-shell equivalent-pair lists with
# atom/spin cross-references, are informational: pass --no-equivalent-comments
# (or equivalent_comments=False in Python) to suppress them, or delete any
# comment; identity, support, and values never depend on comments.
```

No `name`, `distance`, or `n_equivalent_pairs` is supplied here: all three are
optional. `name` is display text only; TB2J computes `distance` (Angstrom) and
`n_equivalent_pairs` (the size of the full ordered-pair orbit, including the
representative) for every shell when the document is canonicalized or written.
No `name` is invented. With the geometry above the first shell validates as an
8-pair orbit at 2.4855 Å and the second as a 6-pair orbit at 2.87 Å. A row
flagged `N` carries a negative spin sentinel, and marking an atom nonmagnetic
does not imply a physical zero moment: moment rows stay complete for every
atom, magnetic or not.

## A complete hand-authored tensor document

The tensor variant is equally complete and independent. This file declares the
same cell with `exchange = "tensor"` and magnetic symmetry, which requires the
complete reference moments and a `magnetic_tol`. For the collinear moment
written here the detector reports magnetic space group 229.140 with 48
operations and no primed operation; primed operations (`time_reversal = true`)
appear for other magnetic configurations.

```toml
# Complete hand-authored tensor example: the same single-Fe body-centered
# cubic primitive cell (a = 2.87 Angstrom), with a magnetic space group and
# full 3x3 exchange tensors. Self-contained and independent of any other
# example: data lines and '#' comments only; comments may be deleted freely.
#
# One [[shell]] stores ONE tensor for one symmetry orbit of ordered atom
# pairs. Space-group/magnetic-group operations (including centering) map pair
# images into the same shell; equal bond lengths alone never merge shells.

# Exchange model of this file: "tensor" = one Cartesian 3x3 matrix J per shell.
# (The alternative is "isotropic" = one scalar per shell. There is no implicit
# scalar-to-tensor conversion: pick the model that matches your data.)
exchange = "tensor"

# Wire unit of every J entry below: "meV" (default, may be omitted) or "eV".
# Distances are Angstrom, moments mu_B; Python values and exported native
# results are always eV (1 meV = 0.001 eV).
energy_unit = "meV"

[structure]
coordinates = "fractional"

# Exactly three rows of three Cartesian lattice-vector components (Angstrom):
# body-centered cubic primitive vectors of Fe (conventional a = 2.87 Angstrom).
lattice = """
-1.435 1.435 1.435
1.435 -1.435 1.435
1.435 1.435 -1.435
"""

# Atom rows, exactly seven whitespace-separated columns:
#   atom-index label x y z magnetic spin-index
# atom-index  sequential one-based index in row order (1, 2, ...).
# label       unique element symbol + positive integer (Fe1); gives the species.
# x y z       finite raw fractional coordinates; values outside [0, 1) are kept,
#             never wrapped, so native fixed-R bonds keep their geometry.
# magnetic    "Y" for a selected magnetic site, "N" otherwise.
# spin-index  zero-based index into the native spin map: every Y row together
#             must map exactly onto 0..nspin-1; an N row carries a negative
#             sentinel, "-" spelling -1. This mapping is never reordered.
atoms = """
1 Fe1 0.0 0.0 0.0 Y 0
"""

[reference_spins]
# One row per atom in full atom order (every atom, magnetic or not):
#   label m         signed collinear moment along z, or
#   label mx my mz  Cartesian axial moment vector.
# Moments are in mu_B. Magnetic symmetry REQUIRES them complete, and collinear
# values are valid even when primed operations are detected.
moments = """
Fe1 2.2
"""
# Provenance of the moments: "explicit" (default), "native:spinat", or
# "model:unit-along-z".
source = "explicit"

[symmetry]
# "magnetic" (CLI/Python: "msg") detects the magnetic space group from the
# structure AND the reference moments; an operation that flips the moments is
# combined with time reversal ("primed", time_reversal = true). The alternative
# "spacegroup" (CLI/Python: "sg") ignores moments entirely. Magnetic mode
# requires the complete [reference_spins] moments above and a positive
# magnetic_tol (mu_B, default 1e-5); "spacegroup" may omit both. mode and the
# scalar/tensor choice are independent.
mode = "magnetic"
symprec = 1e-5
magnetic_tol = 1e-5

# One [[shell]] per symmetry orbit of ordered atom pairs. The representative
# names ONE member: "label-first label-second Rx Ry Rz", where (Rx Ry Rz) is
# the lattice translation moving the SECOND atom into its bonded image.
[[shell]]
# Optional display name; never used as identity, never generated when absent.
name = "1NN"
representative = "Fe1 Fe1 1 0 0"
# Editable exchange tensor, in the wire unit above (meV): rows/columns are the
# Cartesian spin components, J transforms as an axial matrix, and the
# representative bond's stabilizer decides which entries are allowed.
# Cubic symmetry does not force every bond tensor to be isotropic; equivalent
# bonds have tensors related by their Cartesian symmetry rotations.
# Use J = "unfilled" while editing; export rejects unfilled shells.
J = [[45.0, 0.0, 0.0], [0.0, 45.0, 0.0], [0.0, 0.0, 45.0]]

[[shell]]
# Second-nearest neighbors: R = (0, 1, 1) is the Cartesian vector a2 + a3 =
# (2.87, 0, 0) Angstrom, the bcc second-neighbor bond along x.
name = "2NN"
representative = "Fe1 Fe1 0 1 1"
# Optional input summaries: distance (Angstrom) and n_equivalent_pairs (size
# of the FULL ordered-pair orbit, including the representative - not a
# coordination number and not the sparse support size). Supplying them is
# optional; supplied values are validated against geometry and symmetry, and
# TB2J computes both for every shell when the file is canonicalized/written.
distance = 2.87
n_equivalent_pairs = 6
J = [[16.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]]

# Generator/writer comments, including the per-shell equivalent-pair lists with
# atom/spin cross-references, are informational: pass --no-equivalent-comments
# (or equivalent_comments=False in Python) to suppress them, or delete any
# comment; identity, support, and values never depend on comments.
```

The second-neighbor tensor is anisotropic along its representative bond; symmetry
rotates that tensor onto equivalent bonds. Tensor mode always requires the
representative. Scalar and tensor modes are explicit; no scalar-to-tensor or
tensor-to-scalar conversion is implicit.

Export validates the file automatically and writes standard results:

```python
from TB2J.io_exchange.shell_workflow import export_shells

result = export_shells("exchange.shells.toml")
print(result.results.omitted)
```

The equivalent CLI workflow is:

```bash
TB2J_shell generate POSCAR --magnetic-species Fe --cutoff 6
# Edit every J = "unfilled" in exchange.shells.toml.
TB2J_shell export exchange.shells.toml
```

Use `--magnetic-atoms 1 3` instead of species when selecting individual
one-based atoms. A full atom-ordered `moments = [...]` TOML file supplied via
`--reference-spins` supports magnetic symmetry. Export rejects unfilled values,
incompatible context and output collisions before publishing results.

## V2 document grammar and indexing

The document is UTF-8 TOML. Scientific tables accept only their declared fields; unknown scientific keys and the superseded v1 array grammar/field names are rejected. There are no compatibility aliases for old names such as `exchange_mode`, `Jiso`, `tensor`, or `orbit_size`.

- Top-level `exchange` is required and is `"isotropic"` or `"tensor"`. `schema_version` may be omitted (means 2); canonical output writes `2`. Explicit other versions fail. `energy_unit` is `"meV"` (default) or `"eV"`; canonical output writes it explicitly. Optional `cutoff` records a positive radius in Angstrom and does not re-filter imported or existing shells.
- `[structure]` requires `coordinates = "fractional"`, `lattice`, and `atoms`. The lattice is exactly three rows of three Cartesian lattice-vector components in Angstrom. There is no POSCAR scale factor or selective-dynamics syntax.
- Each atom row has exactly seven whitespace-separated fields: `atom-index label x y z magnetic spin-index`. Atom indices are sequential, one-based, in source row order. A label is a unique valid element symbol followed by a positive integer (for example `Fe7`); the element symbol provides species. Coordinates are finite raw fractional coordinates and may be outside `[0, 1)`. `Y` requires a unique nonnegative spin index; all selected rows map exactly onto `0..nspin-1`. `N` requires a negative index; `-` spells `-1`. Native negative sentinels are retained when written. Labels resolve to atom rows and never reorder or renumber spin indices.
- Blank lines and `#` comments are ignored inside row blocks. Row numbers count data rows. The rows accept decimal/scientific real numbers, not Python expressions, complex numbers, NaN/Inf, booleans, or locale-formatted numbers.
- Optional `[reference_spins]` contains `moments` as one label row per atom: either `label m` for signed collinear moments along z, or `label mx my mz` for Cartesian axial vectors. All rows must have the same width, labels must be unique and complete for every atom, and values are in μB. Missing atoms are not filled with zeros. `source` records `explicit`, `native:spinat`, or `model:unit-along-z` provenance.
- `[symmetry]` requires `mode = "spacegroup"` or `"magnetic"`; these correspond to `sg` and `msg` in Python and CLI arguments. `symprec` defaults to `1e-5` Å. Magnetic mode requires complete reference moments and a positive `magnetic_tol` (default `1e-5` μB). Space-group mode may omit moments. Collinear scalar moments are valid for MSG even when operations are primed; Cartesian vectors use the axial-vector convention.
- Optional `[metadata]` accepts finite TOML scalar/list/table data, including dates and times. Its values do not affect exchange or symmetry and are semantically preserved through shell read/write.
- Optional `[symmetry].producer` is informational and does not select the group. Canonical writes place the validated `[[symmetry.operations]]` snapshot after shell entries; each operation has a 3×3 integer `rotation`, three-component fractional `translation`, and Boolean `time_reversal` flag.
- Optional `[tolerances]` records `value_atol` and `zero_tol` in eV and dimensionless `value_rtol`; defaults are `1e-8`, `1e-10`, and `1e-6`, and canonical writes emit all three.
- Optional `[source]` records `format` (`template`, `pickle`, `text`, or `xml`), a nonempty source `precision_profile`, and/or the unique `required_base_fields` manifest. These fields record provenance and unencoded source context; they do not bypass scientific validation.

Each `[[shell]]` has one `J` value: a finite scalar in isotropic mode, a finite Cartesian 3×3 tensor in tensor mode, or the template string `"unfilled"`. Tensor mode requires `representative`; the same label-based representative is emitted for generated scalar templates. An isotropic hand-authored shell may omit the representative only when it supplies an authoritative canonical `id`. A representative has the form `"Fe7 Co4 1 0 -1"`: two site labels followed by the three signed integer translation components `R`, which translates the second site in the raw site-image frame. Optional `id` must equal the canonical minimum numeric pair key `i:j:Rx,Ry,Rz` for the full orbit; atom indices in that ID are one-based. Duplicate entries for the same orbit fail.

`name` is optional display text, not identity, and is never generated if absent. Input may omit `distance` and `n_equivalent_pairs`; TB2J writes both computed summaries for every shell, even when equivalent-pair comments are suppressed. `distance` is the representative bond length in Angstrom. `n_equivalent_pairs` counts the full orbit's ordered pairs, including the representative—not the sparse support size or a coordination number. Both supplied summaries are validated against geometry and symmetry and never select or merge shells.

By default, equivalent-pair and atom/spin cross-reference comments are emitted. They are informational; deleting them does not change shell identity, support, or values. `--no-equivalent-comments` suppresses these generated comments without changing data. Rewriting canonicalizes TOML and regenerates its own comments; arbitrary comments, whitespace, and formatting are not preserved.

## Symmetry snapshots, sparse support, and completion

A shell is the full orbit of an ordered key `(R, i, j)` under the declared space or magnetic group. Equal distances are only geometry summaries; they never merge distinct orbits. Centering translations participate when they are operations of the group. A reversed pair is included only if an operation maps it into the orbit; otherwise it remains a separate shell, with the native reverse-pair tensor-transpose convention checked separately.

A hand-authored **snapshot-free** file can omit `symmetry.operations` only for complete orbit entries without ordinal support/presence masks. Its first read constructs the group once from the declared structure, magnetic-site selection, symmetry mode, reference moments, and tolerances, then derives/validates canonical IDs. Writing that document records the detected operations. Thereafter the stored operation snapshot is authoritative and replayed; reading does not replace it with the current detector result. Python `validate_shells(..., check_producer_ops=True)` compares current detection with the snapshot without changing it.

Canonical full orbits are sorted. `support_members` and channel-presence masks are sorted, unique, **zero-based ordinals into that full orbit**; they require an explicit operation snapshot. `support_members` defaults to the full orbit. `Jiso_members`, `DMI_members`, and `Jani_members` record independently present channels within support (DMI/Jani are tensor-mode channels). Omitted channel masks mean present on the supported members. Empty masks distinguish absent channels from observed zeros; a nonzero authored channel cannot be hidden behind an empty mask. The representative must lie in declared support. Sparse support and masks survive ordinary export unchanged.

## Units and tolerances

Shell wire values use the file-wide `energy_unit` (`meV` by default, alternatively `eV`). Python `ShellEntry` values and native `SpinIO` exchange values are in eV. `value_atol` (default `1e-8` eV) and `zero_tol` (default `1e-10` eV) are always in eV, independent of the wire unit; `value_rtol` defaults to `1e-6` and is dimensionless. Symmetry tolerances are `symprec` in Angstrom and `magnetic_tol` in μB. The template `cutoff` is a real-space radius **always in Angstrom**; there is no cutoff-unit option. Cutoff selects candidate pairs/orbits when building a template, not distance bins or a later filter on imported support.

## Python interfaces and CLI options

The high-level `generate_shells` and `export_shells` functions are in
`TB2J.io_exchange.shell_workflow`. The latter returns an `ExpansionResult` with
the expanded `spinio`, conversion `report`, and standard-writer `results`
containing written targets and omissions.

For scientific in-memory work, `TB2J.io_exchange.shells` retains `template`,
`compress`, `read_shells`, `write_shells`, `validate_shells`, and `expand`.
Their advanced controls do not alter the ordinary CLI defaults.

`TB2J_shell generate` accepts `--magnetic-species` or `--magnetic-atoms`,
`--cutoff`, `--mode`, `--symmetry`, `--reference-spins`, `--symprec`,
`--magnetic-tol`, `--project`, `-o`, and `--no-equivalent-comments` where
applicable. Structure input requires a selector and cutoff; pickle input uses
its stored map and rejects structure-only controls. `--project` explicitly
requests symmetry projection for pickle input.

`TB2J_shell export FILE [-o DIRECTORY]` validates automatically. The default
output is `TB2J_results`. The output report names solver omissions. Existing
unrelated files are preserved; ambiguous or modified output ownership requires
a fresh destination. Generation accepts only structure files or explicitly
supplied trusted `.pickle`/`.pkl` files, never binary-to-pickle fallback.

## Base updates, metadata, and non-preservation

A compatible base is not merely periodically equivalent. It must match species order, cell, raw Cartesian site representatives in the same image frame, exact `index_spin` topology, and declared reference moments within the document's structural/magnetic tolerances. A lattice-image shift at the same atom or a different mapping is a compatibility error; coordinates are not silently wrapped/rebased while fixed native `R` keys are retained. Shell expansion replaces the base's Jiso/DMI/Jani pair values and support with shell values, while retaining compatible context where allowed.

`[source].required_base_fields` records source-present native content that the shell format does not encode. It is validated against this closed vocabulary:

- `biquadratic`: `biquadratic_Jdict`.
- `charges`: `charges`.
- `damping`: `damping`.
- `djdx`: `dJdx`, `dJdx2`, `dJdx_orb`.
- `gyromag`: `gyro_ratio`.
- `njt`: `NJT_Jdict`, `NJT_ddict`.
- `orbital`: `Jiso_orb`, `DMI_orb`, `Jani_orb`.
- `sia`: `sia_tensor`, `k1`, `k1dir`.

The file-level exporter requires the companion when declared and never silently
drops incompatible ancillary data. In-memory `expand` exposes explicit advanced
loss controls; the ordinary CLI does not. Charges and moments remain absent
when not supplied. Edited exchange invalidates dependent orbital or derivative
data rather than retaining it silently. Existing SIA is pass-through context,
not an editable shell value.

The exporter stages standard results beside the destination. It records owned
files and their digests so a later export can remove stale solver outputs
without deleting unrelated or user-modified files. A conflicting file without
verified ownership is rejected with a fresh-output alternative. Publication is
not a crash-proof multi-file transaction.

## Further references

- Python implementation: `TB2J/io_exchange/shell_workflow.py`, `shells.py` and `shell_text.py`.
- CLI implementation and installed command: `TB2J/scripts/TB2J_shell.py` and the `TB2J_shell` entry in `pyproject.toml`.
- Consumer regression suites: `tests/test_shell_cli.py`, `tests/test_shell_workflow_export.py`, and `tests/test_shell_base_updates.py`.
- Shared SG/MSG and ordered-pair conventions: `TB2J/symmetry_core.py` and `docs/sympy/04_spacegroup_symmetrization.md`.
