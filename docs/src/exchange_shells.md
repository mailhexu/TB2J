# Symmetry-shell exchange files and native conversion

`TB2J_shell` and `TB2J.io_exchange.shell_workflow` provide an editable TOML representation of TB2J exchange data and standard results export. This workflow does not calculate exchange parameters.

The shell document stores one editable value per bond **and its reverse**: each shell unites the symmetry orbit of an ordered atom pair with the orbit of its reversed direction under the declared space group or magnetic space group (including centering translations). Equal bond lengths do not define a shell: symmetry-inequivalent bonds stay separate even at the same distance. Reversal is part of the shell, not a second entry: the isotropic value is even under reversal, the DMI sign flips, and a tensor (hence its Jani part) transposes, matching the native reverse-pair convention.

Generate from a structure or a **trusted** TB2J pickle; export the edited shell file to standard TB2J results. Solver-specific outputs are included only when their representation is supported; omissions are reported.

## Generate a structure template from Python

```python
from TB2J.io_exchange.shell_workflow import generate_shells

result = generate_shells(
    "POSCAR", "exchange.shells.toml", magnetic_species=["Fe"], cutoff=6.0
)
```

Select magnetic sites with `magnetic_species` or, alternatively,
`magnetic_atoms` containing one-based full-structure indices. The cutoff uses
`distance_unit` (`"Angstrom"` by default, `"Bohr"` or `"nm"`). For magnetic
space-group symmetry, pass `symmetry="msg"` and complete
`reference_spins` in full atom order; `symprec` and `magnetic_tol` are configurable.
Use `mode="tensor"` for 3×3 exchange values. Each shell remains `unfilled` until edited.

For structure input, spin indices are not entered: `Y` rows receive contiguous
zero-based indices in atom order, and `N` rows are excluded from the spin map.
Pickle-based generation preserves its native mapping in the companion, as
described below.

Both this entry point and `write_shells` add format explanations. The header
identifies the TB2J version and, for generation from a file, the input path;
that path is informational and is never reopened during export. Equivalent
physical-pair comments (each bond listed once; its reverse shares the shell)
are grouped by shell after the complete shell list. Each group has
aligned atom labels, translation components, a deterministic stored
symmetry-operation number, and whether the listed pair is reversed relative
to the generated representative.
These witnesses describe the generated representatives: editing a
representative can make the comments stale. Refresh them without replacing
edited exchange values by reading and rewriting the existing document (do not
generate a new file from the source structure). For a pickle-derived document,
keep its context companion beside the TOML:

```python
from TB2J.io_exchange.shells import read_shells, write_shells
doc = read_shells('exchange.shells.toml')
write_shells(doc, 'exchange.shells.toml')
```

Set `equivalent_comments=False` to omit the pair lists without omitting format
explanations. The atom block describes its six columns and Y/N magnetic marker
immediately above the rows; spin mapping is not an atom-row field. Rotation
matrices use one row per line and translations use a single vector line. All
comments are informational and can be deleted.

Generated floating-point values use fixed notation with eight digits after the
decimal point, including exchange and structural/symmetry data. This is a lossy
write: values smaller than the displayed resolution can round to zero. Exchange
rounding is in the declared energy unit (normally meV), and geometry rounding is
in the units stated in the file. Validation tolerances retain their exact values
with additional decimal places where needed. The generated representation is
validated before publication; if rounding violates symmetry or native-context
tolerances, generation fails without replacing the existing output.

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
Pickle import symmetrizes exchange under the selected symmetry by default; this
can change values. Python callers can use `project=False` to reject inconsistent
values instead and inspect `result.report.residuals` after projection. A companion
requiring unsupported objects, constraints, or a calculator is rejected.

## A complete hand-authored scalar document

The following file is complete and independent: copy it, edit the `J` values,
and export it. It describes one Fe atom in the primitive cell of body-centered
cubic iron (a = 2.87 Å), hand-written as TOML schema v4 (an explicit
`schema_version = 4` is required; earlier schemas are not supported). Every
explanatory comment may be deleted without changing the physics.

```toml
# Complete hand-authored scalar (isotropic) example: one Fe atom in the
# primitive cell of body-centered cubic iron (a = 2.87 Angstrom).
#
# One [[shell]] stores ONE exchange value for one bond and its reverse: the
# shell unites the space-group orbit of the bond (including lattice
# centering) with the orbit of its reversed direction, and Jiso is even under
# reversal. Equal bond lengths alone never merge shells.

# Format version. Version 4 is required; earlier schemas are not supported.
schema_version = 4

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

# Atom rows, exactly six whitespace-separated columns:
#   atom-index label x y z magnetic
# atom-index  sequential one-based index in row order (1, 2, ...).
# label       unique element symbol + positive integer (Fe1); gives the species.
# x y z       finite raw fractional coordinates; values outside [0, 1) are kept,
#             never wrapped, so native fixed-R bonds keep their geometry.
# magnetic    "Y" for a selected magnetic site, "N" otherwise.
# Spin indices are not supplied here: Y rows receive contiguous zero-based
# indices in atom order; N rows are excluded from the spin map.
# Blank lines and '#' comments are ignored inside row blocks.
atoms = """
1 Fe1 0.0 0.0 0.0 Y
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

# One [[shell]] stores one J for the chosen representative pair and its reverse.
# Any member of the symmetry-equivalent orbit, including an unobserved pair,
# may be chosen. J is expressed in the global Cartesian frame.
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

# Generator/writer comments, including per-shell equivalent-pair lists with
# symmetry witnesses, are informational. They
# describe the generated representative; editing it can make the witness comments
# stale. Pass --no-equivalent-comments (or equivalent_comments=False in Python)
# to suppress generated lists, or delete any comments; values and native sparse
# context do not depend on comments.
```

No `name`, `distance`, or `n_equivalent_pairs` is supplied here: all three are
optional. `name` is display text only; TB2J computes `distance` (in the declared unit) and
`n_equivalent_pairs` (the number of unique physical pairs, counting each bond
and its reverse once) for every shell when the document is canonicalized or
written. No `name` is invented. With the geometry above, the first shell
covers 4 physical pairs at 2.4855 Å (an 8-member directed orbit that already
contains every reverse) and the second covers 3 physical pairs at 2.87 Å (a
6-member directed orbit).
A row flagged `N` is excluded from the automatically derived spin mapping; it
does not imply a physical zero moment. Moment rows remain complete for every
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
# One [[shell]] stores ONE tensor for one bond and its reverse:
# space-group/magnetic-group operations (including centering) map pair images
# into the same orbit, and the reversed bond joins the shell with its tensor
# transposed (DMI sign flipped, isotropic trace unchanged). Equal bond
# lengths alone never merge shells.

# Format version. Version 4 is required; earlier schemas are not supported.
schema_version = 4

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

# Atom rows, exactly six whitespace-separated columns:
#   atom-index label x y z magnetic
# atom-index  sequential one-based index in row order (1, 2, ...).
# label       unique element symbol + positive integer (Fe1); gives the species.
# x y z       finite raw fractional coordinates; values outside [0, 1) are kept,
#             never wrapped, so native fixed-R bonds keep their geometry.
# magnetic    "Y" for a selected magnetic site, "N" otherwise.
# Spin indices are not supplied here: Y rows receive contiguous zero-based
# indices in atom order; N rows are excluded from the spin map.
# Blank lines and '#' comments are ignored inside row blocks.
atoms = """
1 Fe1 0.0 0.0 0.0 Y
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

# One [[shell]] stores one J for the chosen representative pair and its reverse.
# Any member of the symmetry-equivalent orbit, including an unobserved pair,
# may be chosen. J is expressed in the global Cartesian frame.
[[shell]]
# Optional display name; never used as identity, never generated when absent.
name = "1NN"
representative = "Fe1 Fe1 1 0 0"
# Editable exchange tensor, in the wire unit above (meV): rows and columns are
# global Cartesian spin components. J applies to the chosen representative pair;
# symmetry relates values on equivalent pairs; reversal transposes the tensor
# and flips DMI. Choose any pair in the orbit, including one not observed.
J = [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]

[[shell]]
# Second-nearest neighbors: R = (0, 1, 1) is the Cartesian vector a2 + a3 =
# (2.87, 0, 0) Angstrom, the bcc second-neighbor bond along x.
name = "2NN"
representative = "Fe1 Fe1 0 1 1"
# Optional input summaries: distance (Angstrom) and n_equivalent_pairs
# (unique physical pairs, each bond and its reverse counted once - not the
# directed orbit size, a coordination number, or the sparse support size).
# Supplying them is optional; supplied values are validated against geometry
# and symmetry, and TB2J computes both for every shell when the file is
# canonicalized/written.
distance = 2.87
n_equivalent_pairs = 3
J = [[16.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]]

# Generator/writer comments include equivalent-pair lists and symmetry
# witnesses. They are informational and can become
# stale if the representative is edited. Pass --no-equivalent-comments (or
# equivalent_comments=False in Python) to suppress them; values and native
# sparse context do not depend on comments.
```

The second-neighbor tensor is anisotropic in the global Cartesian frame for its
chosen representative; symmetry maps it onto equivalent pairs, and the reversed
bond of each pair carries the transpose. Tensor mode requires a representative.
Scalar and tensor modes are explicit; no conversion between them is implicit.

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

The CLI prints the TB2J version and license, source, absolute output destination,
and symmetry before processing. On success it reports the exchange mode, number
of shells, and any context companion. Pickle import symmetrizes automatically and
reports the largest absolute component change in eV; the original pickle is not
modified. Review that change before using the generated exchange values.

Structure templates and pickle imports order shells by increasing geometric
distance. For example, use `TB2J_shell generate POSCAR -s Fe -c 0.6 -d nm`;
`TB2J_shell generate --help` lists every short option and its default or
automatic behavior.

## V4 document grammar and indexing

The document is UTF-8 TOML. Scientific tables accept only their declared fields; unknown scientific keys and the superseded v1 array grammar/field names are rejected. There are no compatibility aliases for old names such as `exchange_mode`, `Jiso`, `tensor`, or `orbit_size`. An explicit `schema_version = 4` is required; earlier schemas and the former seven-column atom-row grammar are not supported.

- Top-level `exchange` is required and is `"isotropic"` or `"tensor"`. An explicit `schema_version = 4` is required, and canonical output writes `4`. `energy_unit` is `"meV"` (default) or `"eV"`. `distance_unit` is `"Angstrom"` (default), `"Bohr"`, or `"nm"`; it applies to `cutoff` and shell distances. Optional `cutoff` does not re-filter imported or existing shells.
- `[structure]` requires `coordinates = "fractional"`, `lattice`, and `atoms`. The lattice is exactly three rows of three Cartesian lattice-vector components in Angstrom. There is no POSCAR scale factor or selective-dynamics syntax.
- Each atom row has exactly six whitespace-separated fields: `atom-index label x y z magnetic`. Atom indices are sequential, one-based, in source row order. A label is a unique valid element symbol followed by a positive integer (for example `Fe7`); the element symbol provides species. Coordinates are finite raw fractional coordinates and may be outside `[0, 1)`. `magnetic` is exactly `Y` or `N`. No spin index is supplied in the row: `Y` sites receive contiguous zero-based spin indices in atom order, and `N` sites are excluded from the derived mapping (internal sentinel `-1`). For structure-derived documents this mapping is automatic; native pickle mappings, including permutations and negative nonmagnetic sentinels, are preserved in the digest-protected companion context.
- Blank lines and `#` comments are ignored inside row blocks. Row numbers count data rows. Coordinate values accept decimal/scientific real numbers, not Python expressions, complex numbers, NaN/Inf, booleans, or locale-formatted numbers.
- Optional `[reference_spins]` contains `moments` as one label row per atom: either `label m` for signed collinear moments along z, or `label mx my mz` for Cartesian axial vectors. All rows must have the same width, labels must be unique and complete for every atom, and values are in μB. Missing atoms are not filled with zeros. `source` records `explicit`, `native:spinat`, or `model:unit-along-z` provenance.
- `[symmetry]` requires `mode = "spacegroup"` or `"magnetic"`; these correspond to `sg` and `msg` in Python and CLI arguments. `symprec` defaults to `1e-5` Å. Magnetic mode requires complete reference moments and a positive `magnetic_tol` (default `1e-5` μB). Space-group mode may omit moments. Collinear scalar moments are valid for MSG even when operations are primed; Cartesian vectors use the axial-vector convention.
- Optional `[metadata]` accepts finite TOML scalar/list/table data, including dates and times. Its values do not affect exchange or symmetry; floating-point values are rounded on writing.
- Optional `[symmetry].producer` is informational and does not select the group. Canonical writes place the validated `[[symmetry.operations]]` snapshot after shell entries; each operation has a 3×3 integer `rotation`, three-component fractional `translation`, and Boolean `time_reversal` flag.
- Optional `[tolerances]` records `value_atol` and `zero_tol` in eV and dimensionless `value_rtol`; defaults are `1e-8`, `1e-10`, and `1e-6`, and canonical writes emit all three.
- Optional `[source]` records `format` (`template`, `pickle`, `text`, or `xml`), a nonempty source `precision_profile`, and/or the unique `required_base_fields` manifest. These fields record provenance and unencoded source context; they do not bypass scientific validation.

Each `[[shell]]` has one `representative` and one `J` value: a finite scalar in isotropic mode, a finite Cartesian 3×3 tensor in tensor mode, or the template string `"unfilled"`. The representative has the form `"Fe7 Co4 1 0 -1"`: two site labels followed by the three signed integer translation components `R`, which translates the second site in the raw site-image frame. Any symmetry-equivalent directed pair may be selected, whether or not that pair was observed in the native data. Shell blocks have no `id` or ordinal support/channel masks. `J` is expressed for the chosen representative pair in the global Cartesian frame; symmetry operations map it to equivalent pairs, and reversal applies the native convention (isotropic even, DMI sign reversed, tensor transposed). Duplicate entries for the same orbit fail.

`name` is optional display text, not identity, and is never generated if absent. Input may omit `distance` and `n_equivalent_pairs`; TB2J writes both computed summaries for every shell, even when equivalent-pair comments are suppressed. `distance` is the representative bond length in the declared `distance_unit`. `n_equivalent_pairs` counts the shell's unique physical pairs—each bond and its reverse counted once—not a directed-orbit size, sparse support size, or coordination number. Both supplied summaries are validated against geometry and symmetry and never select or merge shells.

By default, equivalent-pair comments are emitted. Pair comments include a deterministic operation witness and whether that pair is reversed relative to the generated representative. They describe the representative at generation time; editing the representative can make them stale. Comments are informational; deleting them does not change shell identity, support, or values. `--no-equivalent-comments` suppresses generated comments without changing data. Rewriting canonicalizes TOML and regenerates its own comments; arbitrary comments, whitespace, and formatting are not preserved.

## Symmetry snapshots, sparse support, and completion

A shell is the union of the full orbit of a chosen ordered key `(R, i, j)` under the declared space or magnetic group and the orbit of its reversed key `(j, i, -R)`; each shell entry contains a representative that may be any member of this orbit, including an unobserved pair. Equal distances are only geometry summaries; they never merge distinct orbits, and symmetry-inequivalent shells stay separate even at equal distance. Centering translations participate when they are operations of the group. The reversed bond always belongs to the same shell: it shares the isotropic value, flips the DMI sign, and carries the transposed tensor. Export to native results writes both directions for supported pairs and channels, creating a missing reverse entry from the observed channels; it never invents unrelated symmetry images or absent channel families.

A hand-authored **snapshot-free** file can omit `symmetry.operations` only for complete orbit entries. Its first read constructs the group once from the declared structure, magnetic-site selection, symmetry mode, reference moments, and tolerances. Writing that document records the detected operations. Thereafter the stored operation snapshot is authoritative and replayed; reading does not replace it with the current detector result. Canonical operations are sorted and stable in the document. Python `validate_shells(..., check_producer_ops=True)` compares current detection with the snapshot without changing it.

For structure-derived documents, `Y` rows define the spin mapping automatically: selected sites receive contiguous zero-based indices in full atom order, and `N` rows are excluded. Native pickle mappings are different: the digest-protected companion retains the exact original atom-to-spin mapping, including permutations and negative nonmagnetic sentinels; that mapping is not encoded in shell rows. Before validating or using spin-indexed native pair keys, export restores the mapping from the verified companion. If the editable Y/N selection differs from the native context, export fails rather than silently remapping native keys. Keep the companion with its TOML to preserve sparse pair support, per-channel presence (including absent versus zero), and the native mapping.

## Units and tolerances

Shell wire values use the file-wide `energy_unit` (`meV` by default, alternatively `eV`). Python `ShellEntry` values and native `SpinIO` exchange values are in eV. `value_atol` and `zero_tol` are always in eV; `value_rtol` is dimensionless. `symprec` is in Angstrom and `magnetic_tol` in μB. `distance_unit` applies to wire cutoff and shell distances; lattice vectors and Python geometry remain in Angstrom. Cutoff selects candidate pairs/orbits when building a template, not distance bins or a later filter on imported support.

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
requests the same default symmetry projection for pickle input.

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
