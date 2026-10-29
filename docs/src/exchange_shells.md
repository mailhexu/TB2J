# Symmetry-shell exchange files and native conversion

`TB2J_shell` and `TB2J.io_exchange.shells` provide an editable TOML representation of TB2J exchange data and guarded conversion to the native formats. This is an exchange-format/conversion workflow; it does not calculate exchange parameters.

The shell document stores one value per **symmetry orbit of ordered atom pairs**. Equal bond lengths do not define an orbit: equal-distance pairs in different orbits remain separate shells, while pairs related by a space-group or magnetic-space-group operation (including centering translations) belong to one shell. Reversing a pair is a separate physical transpose convention, not an automatic orbit operation.

Supported shell/native formats are `shells`, trusted `pickle`, `text` (`exchange.out`), `xml` (`exchange.xml`), and the composed native `tb2j` output preset. External-solver formats are not targets of this interface. The former solver-export scenario A17 is **withdrawn, not passed**; existing solver writers and `write_all` are unchanged and outside this contract.

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
Use `mode="tensor"` for 3×3 exchange values. Each shell remains `unfilled` until
edited. This Python entry point currently generates structure templates; the
native conversion commands below retain their existing behavior.

Both this entry point and `write_shells` add format explanations. Set
`equivalent_comments=False` to omit equivalent-pair lists without omitting
the explanations. Comments are informational and can be deleted.

## A small hand-authored document

This complete scalar example is TOML schema v2 (the `schema_version` key is optional for hand-written v2 input). It has one explicit representative, a nonmagnetic atom with its own negative spin sentinel, and complete reference moments. The moment rows include every atom, including O2; a nonmagnetic selection does not imply a physical zero moment.

```toml
exchange = "isotropic"

[structure]
coordinates = "fractional"
lattice = """
4 0 0
0.3 5 0
0.2 0.4 6
"""
atoms = """
1 Fe7 -0.87 0.21 0.17 Y 1
2 O2 0.27 1.11 0.41 N -2
3 Co4 1.39 0.36 0.23 Y 0
"""

[reference_spins]
moments = """
Fe7 2
O2 0.03
Co4 -1
"""

[symmetry]
mode = "spacegroup"

[[shell]]
representative = "Fe7 Co4 1 0 -1"
J = 2

[metadata]
source_note = "hand-edited"
```

With no `distance` or `n_equivalent_pairs` supplied, both are computed when the document is canonicalized or written. No `name` is invented. To make this a tensor document, change `exchange` to `"tensor"` and replace the scalar `J` with a 3×3 Cartesian tensor, for example:

```toml
exchange = "tensor"

[[shell]]
representative = "Fe7 Co4 1 0 -1"
J = [[2, 0.4, 0.3], [-0.4, 1, 0.2], [0.3, 0.2, 3]]
```

Use the tensor replacement in the same file, not as a second `[[shell]]` entry. Scalar and tensor modes are explicit; no scalar-to-tensor or tensor-to-scalar conversion is implicit. The wire values above are in meV by default. The corresponding scalar value in Python/native storage is `0.002` eV.

Read, validate, expand, and write the scalar file to a native pickle from Python:

```python
from TB2J.io_exchange.shells import (
    expand,
    read_exchange,
    read_shells,
    validate_shells,
    write_exchange,
)

document = read_shells("pair.shells.toml")
validate_shells(document, for_export=True)
expanded = expand(document)
write_exchange(expanded.spinio, "native", "pickle")
reloaded = read_exchange("native/TB2J.pickle", "pickle").spinio
assert reloaded.exchange_Jdict == expanded.spinio.exchange_Jdict
```

The same calls work for the tensor variant. `ShellDocument` and its entries are frozen values; make edits by constructing a replacement document/entry (for example with `dataclasses.replace`) and validate again. Expansion always revalidates the current in-memory document.

The corresponding CLI path is:

```bash
TB2J_shell validate pair.shells.toml
TB2J_shell validate pair.shells.toml --for-export
TB2J_shell convert pair.shells.toml --from shells --to pickle -o native
```

`--from shells` is required for this schema-version-less hand-authored file: auto-detection requires a top-level `schema_version` key. For template-first use, generate a file, edit every `unfilled` value, then run export validation and conversion:

```bash
TB2J_shell template POSCAR --magnetic-atoms 1 3 --cutoff 6 \
  --symmetry sg --mode tensor --reference-spins moments.toml \
  -o exchange.shells.toml
TB2J_shell validate exchange.shells.toml
# Edit the TOML, replacing every J = "unfilled" with a real 3x3 tensor.
TB2J_shell validate exchange.shells.toml --for-export
TB2J_shell convert exchange.shells.toml --from shells --to tb2j \
  -o native-results --base TB2J.pickle --allow-loss damping gyromag charges \
  --report conversion.json
```

For that CLI template command, `moments.toml` is a separate TOML file with a `moments = [...]` array in **full atom order**, for example `moments = [2.0, 0.0, -1.0]`. It is not the label-row syntax embedded in a shell file. A compatible base is useful when native ancillary data or writer inputs (notably charges for `exchange.out`) must be kept. A generated template contains unfilled values: ordinary structural validation and `write_shells` allow them; `validate --for-export` and every native conversion reject them, naming the shell and value field, before output creation.

The preset example explicitly permits text to omit damping/gyromagnetic metadata and XML to omit charges. Review those named losses before using it; omit permissions you do not intend to grant. Without the `charges` permission, a charge-bearing base is rejected by the XML member before the preset creates output.

## V2 document grammar and indexing

The document is UTF-8 TOML. Scientific tables accept only their declared fields; unknown scientific keys and the superseded v1 array grammar/field names are rejected. There are no compatibility aliases for old names such as `exchange_mode`, `Jiso`, `tensor`, or `orbit_size`.

- Top-level `exchange` is required and is `"isotropic"` or `"tensor"`. `schema_version` may be omitted (means 2); canonical output writes `2`. Explicit other versions fail. `energy_unit` is `"meV"` (default) or `"eV"`; canonical output writes it explicitly. Optional `cutoff` records a positive radius in Angstrom and does not re-filter imported or existing shells.
- `[structure]` requires `coordinates = "fractional"`, `lattice`, and `atoms`. The lattice is exactly three rows of three Cartesian lattice-vector components in Angstrom. There is no POSCAR scale factor or selective-dynamics syntax.
- Each atom row has exactly seven whitespace-separated fields: `atom-index label x y z magnetic spin-index`. Atom indices are sequential, one-based, in source row order. A label is a unique valid element symbol followed by a positive integer (for example `Fe7`); the element symbol provides species. Coordinates are finite raw fractional coordinates and may be outside `[0, 1)`. `Y` requires a unique nonnegative spin index; all selected rows map exactly onto `0..nspin-1`. `N` requires a negative index; `-` spells `-1`. Native negative sentinels are retained when written. Labels resolve to atom rows and never reorder or renumber spin indices.
- Blank lines and `#` comments are ignored inside row blocks. Row numbers count data rows. The rows accept decimal/scientific real numbers, not Python expressions, complex numbers, NaN/Inf, booleans, or locale-formatted numbers.
- Optional `[reference_spins]` contains `moments` as one label row per atom: either `label m` for signed collinear moments along z, or `label mx my mz` for Cartesian axial vectors. All rows must have the same width, labels must be unique and complete for every atom, and values are in μB. Missing atoms are not filled with zeros. `source` records `explicit`, `native:spinat`, or `model:unit-along-z` provenance.
- `[symmetry]` requires `mode = "spacegroup"` or `"magnetic"`; these correspond to `sg` and `msg` in Python and CLI arguments. `symprec` defaults to `1e-5` Å. Magnetic mode requires complete reference moments and a positive `magnetic_tol` (default `1e-5` μB). Space-group mode may omit moments. Collinear scalar moments are valid for MSG even when operations are primed; Cartesian vectors use the axial-vector convention. `--reference-spins-model unit-along-z` is an explicit modeled context for template generation in SG mode only, not a substitute for calculated/observed moments and never enables MSG.
- Optional `[metadata]` accepts finite TOML scalar/list/table data, including dates and times. Its values do not affect exchange or symmetry and are semantically preserved through shell read/write.
- Optional `[symmetry].producer` is informational and does not select the group. Canonical writes place the validated `[[symmetry.operations]]` snapshot after shell entries; each operation has a 3×3 integer `rotation`, three-component fractional `translation`, and Boolean `time_reversal` flag.
- Optional `[tolerances]` records `value_atol` and `zero_tol` in eV and dimensionless `value_rtol`; defaults are `1e-8`, `1e-10`, and `1e-6`, and canonical writes emit all three.
- Optional `[source]` records `format` (`template`, `pickle`, `text`, or `xml`), a nonempty source `precision_profile`, and/or the unique `required_base_fields` manifest. These fields record provenance and unencoded source context; they do not bypass scientific validation.

Each `[[shell]]` has one `J` value: a finite scalar in isotropic mode, a finite Cartesian 3×3 tensor in tensor mode, or the template string `"unfilled"`. Tensor mode requires `representative`; the same label-based representative is emitted for generated scalar templates. An isotropic hand-authored shell may omit the representative only when it supplies an authoritative canonical `id`. A representative has the form `"Fe7 Co4 1 0 -1"`: two site labels followed by the three signed integer translation components `R`, which translates the second site in the raw site-image frame. Optional `id` must equal the canonical minimum numeric pair key `i:j:Rx,Ry,Rz` for the full orbit; atom indices in that ID are one-based. Duplicate entries for the same orbit fail.

`name` is optional display text, not identity, and is never generated if absent. Input may omit `distance` and `n_equivalent_pairs`; TB2J writes both computed summaries for every shell, even when equivalent-pair comments are suppressed. `distance` is the representative bond length in Angstrom. `n_equivalent_pairs` counts the full orbit's ordered pairs, including the representative—not the sparse support size or a coordination number. Both supplied summaries are validated against geometry and symmetry and never select or merge shells.

By default, equivalent-pair and atom/spin cross-reference comments are emitted. They are informational; deleting them does not change shell identity, support, or values. `--no-equivalent-comments` suppresses these generated comments without changing data. Rewriting canonicalizes TOML and regenerates its own comments; arbitrary comments, whitespace, and formatting are not preserved.

## Symmetry snapshots, sparse support, and completion

A shell is the full orbit of an ordered key `(R, i, j)` under the declared space or magnetic group. Equal distances are only geometry summaries; they never merge distinct orbits. Centering translations participate when they are operations of the group. A reversed pair is included only if an operation maps it into the orbit; otherwise it remains a separate shell, with the native reverse-pair tensor-transpose convention checked separately.

A hand-authored **snapshot-free** file can omit `symmetry.operations` only for complete orbit entries without ordinal support/presence masks. Its first read constructs the group once from the declared structure, magnetic-site selection, symmetry mode, reference moments, and tolerances, then derives/validates canonical IDs. Writing that document records the detected operations. Thereafter the stored operation snapshot is authoritative and replayed; reading does not replace it with the current detector result. `validate --check-producer-ops` explicitly compares current detection with the stored snapshot and reports drift without changing it. Removing a stored snapshot is not a way to reassign old masks or reinterpret the geometry.

Canonical full orbits are sorted. `support_members` and channel-presence masks are sorted, unique, **zero-based ordinals into that full orbit**; they require an explicit operation snapshot. `support_members` defaults to the full orbit. `Jiso_members`, `DMI_members`, and `Jani_members` record independently present channels within support (DMI/Jani are tensor-mode channels). Omitted channel masks mean present on the supported members. Empty masks distinguish absent channels from observed zeros; a nonzero authored channel cannot be hidden behind an empty mask. The representative must lie in declared support. Sparse support and masks survive expansion unchanged by default. `--complete-orbits` / `complete_orbits=True` explicitly creates missing model keys and reports them as created, never as calculated observations; an entirely absent channel is not backfilled as calculated zeros.

## Units and tolerances

Shell wire values use the file-wide `energy_unit` (`meV` by default, alternatively `eV`). Python `ShellEntry` values and native `SpinIO` exchange values are in eV. `value_atol` (default `1e-8` eV) and `zero_tol` (default `1e-10` eV) are always in eV, independent of the wire unit; `value_rtol` defaults to `1e-6` and is dimensionless. Symmetry tolerances are `symprec` in Angstrom and `magnetic_tol` in μB. The template `cutoff` is a real-space radius **always in Angstrom**; there is no cutoff-unit option. Cutoff selects candidate pairs/orbits when building a template, not distance bins or a later filter on imported support.

## Python interfaces and CLI options

The public functions are in `TB2J.io_exchange.shells`:

- `template(atoms, magnetic_atoms, cutoff, *, symmetry="sg", reference_spins=None, mode="isotropic", ...)` creates an unfilled full-orbit document. `magnetic_atoms` uses one-based indices in the full atom list.
- `compress(spinio, *, symmetry="sg", reference_spins=None, mode="tensor", on_inconsistent="error", snap_zero=False, ...)` reads observed native channels into a `ShellResult(document, report)`. Inconsistency fails by default; `on_inconsistent="project"` averages onto an orbit and reports residuals. `snap_zero=True` is explicit and reports every changed value. Python `compress` defaults to tensor mode; the CLI `convert ... --to shells` defaults to isotropic unless `--mode tensor` is supplied.
- `read_shells`, `write_shells`, `validate_shells` are the TOML codec/validation functions. `validate_shells(..., for_export=True)` is the unfilled-value gate; `check_producer_ops=True` adds the non-mutating operation comparison.
- `expand(document, *, base=None, complete_orbits=False, allow_loss=())` returns `ExpansionResult(spinio, report)`. It revalidates the document and does not snap values.
- `read_exchange(path, format, context=None, allow_loss=())` reads an explicit `pickle`, `text`, or `xml` input into a `NativeReadResult(spinio, report)`. `write_exchange(spinio, output_dir, format, context=None, allow_loss=())` writes one native target or the `tb2j` preset.
- `NativeContext` can supply `atoms` (ASE `Atoms` or a structure-file path), `species`, an exact `index_spin` map, one-based `magnetic_atoms`, `reference_spins`, and `charges`. Supplied context must agree with native information; it does not silently replace conflicting structure or physical data. The Python API exposes `index_spin`; the CLI has no `--index-spin` flag.

The installed `TB2J_shell` command has three subcommands: `template`, `validate`, and `convert`.

- `template STRUCTURE --cutoff ANGSTROM`: `--magnetic-atoms` is required. Also accepts `--symmetry sg|msg`, `--mode isotropic|tensor`, `--reference-spins FILE`, `--reference-spins-model unit-along-z`, `--symprec`, `--magnetic-tol`, `--energy-unit`, `--zero-tol`, `--value-atol`, `--value-rtol`, `-o`, and `--no-equivalent-comments`. Defaults: SG, isotropic, meV.
- `validate FILE --for-export`: rejects unfilled values. Without this option, structural validation accepts unfilled draft values.
- `validate FILE --check-producer-ops`: reports detector drift without replacing the stored snapshot.
- `convert INPUT --from auto|shells|pickle|text|xml --to shells|pickle|text|xml|tb2j -o OUTPUT`: shell output is a file; native output is a directory. The preset writes all three native files plus `structure.vasp` directly in that directory.
- `convert ... --base PICKLE`: shell-to-native only; requires trusted input with compatible raw structure, spin map and declared moments.
- `convert ... --complete-orbits`: shell-to-native only; reports created model pair/channel keys.
- `convert ... --project`: native-to-shell only; enables reported averaging instead of default inconsistency failure.
- `convert ... --snap-zero`: native-to-shell only; reports every modified value. Expansion never snaps.
- `convert ... --structure FILE`: missing native structure context; invalid for shell input.
- `convert ... --magnetic-atoms N ... --reference-spins FILE`: missing native magnetic context, also header inputs for compression. The reference file contains a full atom-ordered `moments = [...]` array with uniform scalar or vector entries. Invalid for shell input.
- `convert ... --symmetry --mode --symprec --magnetic-tol --energy-unit --zero-tol --value-atol --value-rtol`: header settings for native-to-shell compression only; these options also apply to `template`. Shell headers cannot be overridden.
- `convert ... --allow-loss NAME ...`: explicit named input-block, expansion-manifest or target losses. Cannot supply missing required writer inputs; invalid for shell-to-shell rewriting.
- `template` or `convert ... --to shells --no-equivalent-comments`: suppresses pair comments only; invalid for non-shell output.
- `convert ... --report FILE`: complete stage-wise JSON, including residuals, created keys, precision/quanta, losses, metadata actions and messages. A report may not replace an input, context/base or generated artifact.

`--from auto` recognizes only a TOML top-level `schema_version`, the native XML root, or the outer `Information:` header in `exchange.out`. A hand-written file without `schema_version` requires `--from shells`. Pickle is never guessed by auto-detection: select `--from pickle` explicitly and only for trusted input. In Python, `read_exchange` also requires an explicit format.

Header-producing conversion flags and shell-only flags are rejected when inapplicable; they are not silently ignored. Native-to-native conversion may use `--structure`, `--magnetic-atoms`, and `--reference-spins` as missing context, but symmetry/mode/unit/tolerance overrides are not accepted on that route. CLI validation/context/unfilled/loss preflight failures occur before requested output creation. The whole `tb2j` preset preflights every member before making the output directory; later filesystem or writer failures are not a transaction and have no rollback guarantee.

## Native import, context, and trusted pickle

Only `pickle`, `text`, and `xml` are native input formats. Pickles can execute code during loading: never load an untrusted pickle. Use `--from pickle` explicitly (or `format="pickle"` in Python); neither auto-detection nor the reader treats unknown bytes as pickle.

`exchange.out` has no version identifier. Its complete atom table supplies labels and geometry, but does not always establish magnetic membership or signed moment directions. Supply missing structure, magnetic-site selection, and signed reference moments explicitly rather than inferring them from nonzero moments or observed pair endpoints. Its scalar/DMI/Jani data are authoritative; combined tensor blocks are consistency checks. A structurally complete empty `Exchange` table is valid when explicit magnetic-map context is provided; a missing Exchange header is not an empty model.

Multibinit XML has no chemical symbols. Supply structure/species explicitly when not available from context; masses are not used to guess elements. The adapter restores exchange values to eV and translates the XML pair-key order into native `(R, i, j)` order. A supplied structure is checked against XML cell and position tokens. Missing moments, magnetic selection, or charges are never manufactured. Native files with meaningful unsupported blocks fail unless their named input loss is allowed and reported.

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

On standalone expansion, every manifest entry needs a compatible base or an explicit matching `allow_loss` name. Loss permission drops the unencoded content; it does not create charges or moments required by a writer. For example, `allow_loss=("charges",)` cannot make text output possible if actual charges are absent.

<table>
<thead><tr><th>Native field/data</th><th>Without base</th><th>With compatible base</th></tr></thead>
<tbody>
<tr><td>Jiso, DMI, Jani</td><td>Shell values and support</td><td>Replaced by shell values and support</td></tr>
<tr><td>Cell, raw positions, species, <code>index_spin</code>, reference moments</td><td>Represented in the document when present</td><td>Validated, then preserved</td></tr>
<tr><td>Charges</td><td>Absent unless supplied as real context</td><td>Preserved; text requires actual charges, XML needs named charge-loss permission</td></tr>
<tr><td>Mass, damping, gyromagnetic ratio</td><td>Not part of the shell schema</td><td>Preserved; XML reports its existing 1.0 damping/gyro defaults when arrays are absent</td></tr>
<tr><td>SIA</td><td>Not editable</td><td>Preserved after legacy <code>k1</code>/<code>k1dir</code> normalization; text/XML need named <code>sia_tensor</code> loss</td></tr>
<tr><td>Biquadratic, NJT</td><td>Not represented</td><td>Retained; target capability controls named losses</td></tr>
<tr><td>Orbital, derivatives</td><td>Not represented</td><td>Retained only when exchange agrees; edited exchange requires named <code>orbital</code>/<code>djdx</code> losses to drop stale data</td></tr>
<tr><td>Description, version, collinearity</td><td>Fresh native description/version; collinearity follows scalar/vector moments</td><td>Base description/version preserved; collinearity follows declared moment shape and differences are reported</td></tr>
<tr><td><code>Rlist</code>, atom maps, <code>distance_dict</code></td><td>Rebuilt</td><td>Rebuilt; derived state may differ</td></tr>
<tr><td>Shell labels and metadata</td><td>Document controls annotations</td><td>Document replaces base annotations, even empty metadata; pickle retains them, text/XML report omission</td></tr>
</tbody>
</table>

Shell TOML read/write preserves metadata values semantically, not comments or formatting. Native pickle carries those user annotations when they are present. Text/XML cannot carry the shell annotation namespace. No route promises byte-identical pickles, arbitrary native ancillary data, timestamps, generated descriptions, or all derived maps. Native text/XML values are also subject to their finite printer precision below. Existing SIA data is pass-through context only; this feature does not edit it.

## Native targets and exact loss names

The closed `allow_loss` vocabulary is `dmi`, `jani`, `sia`, `sia_tensor`, `biquadratic`, `njt`, `orbital`, `djdx`, `damping`, `gyromag`, and `charges`. Populated losses fail by default and proceed only with their matching explicit opt-in and a report. Losses are checked against the actual requested target and, for some text channels, the actual pair keys.

<table>
<thead><tr><th>Target</th><th>Populated losses requiring permission</th><th>Additional constraint</th></tr></thead>
<tbody>
<tr><td><code>shells</code></td><td>Unencoded ancillary data becomes <code>required_base_fields</code></td><td>Use <code>write_shells</code>, not the native writer</td></tr>
<tr><td><code>pickle</code></td><td>None in the native target table</td><td>Trusted input; retains native model and shell annotations</td></tr>
<tr><td><code>text</code></td><td><code>sia_tensor biquadratic orbital damping gyromag</code></td><td>Requires real charges/moments; DMI/Jani-only keys need named <code>dmi</code>/<code>jani</code> drops; NJT key mismatches and incomplete derivatives are guarded/reported</td></tr>
<tr><td><code>xml</code></td><td><code>sia_tensor biquadratic njt orbital djdx charges</code></td><td>Independent Jiso/DMI/Jani lists; requires reference moments; writes structure, mass, damping and gyro</td></tr>
<tr><td><code>tb2j</code></td><td>Union of pickle/text/XML losses</td><td>Preflights all native targets and structure; direct four-file output, no <code>write_all</code> or auxiliary XML files</td></tr>
</tbody>
</table>

The `sia` name represents legacy `k1`/`k1dir` input blocks or the shell source manifest; after native normalization the guarded output loss is named `sia_tensor`. `dmi`/`jani` pair-key drops do not fabricate zero Jiso entries. A zero written as part of a declared isotropic model is that model's convention, not a claim that a missing calculated channel was observed. Native import also accepts named `allow_loss` for explicitly identified unsupported source blocks; import does not guarantee that every output target can represent the resulting data.

## Printed precision and reports

Reports contain the precision profile and field quanta used for comparisons; CLI text output is concise and `--report FILE` stores full JSON (Decimal quanta are serialized as exact decimal strings). These are source/output quantization bounds, not symmetry tolerances. They do not justify loosening structural or symmetry tolerances.

<table>
<thead><tr><th>Native token regime</th><th>Field</th><th>Half-quantum bound</th></tr></thead>
<tbody>
<tr><td>Text <code>{:7.4f}</code> meV</td><td>Jiso and DMI</td><td><code>5e-5 meV = 5e-8 eV</code></td></tr>
<tr><td>NumPy precision 3, fixed tokens including suppressed <code>0.</code></td><td>Jani and combined matrices</td><td><code>5e-4 meV = 5e-7 eV</code></td></tr>
<tr><td>NumPy scientific token, exponent e and f fractional mantissa digits</td><td>Matrix values</td><td><code>0.5 × 10^(e-f) meV = 0.5 × 10^(e-f-3) eV</code>; actual Decimal bound per entry</td></tr>
<tr><td>Text <code>{:10.8f}</code>/<code>{:13.8f}</code> Å</td><td>Cell and positions</td><td><code>5e-9 Å</code></td></tr>
<tr><td>XML <code>%.5e</code></td><td>Each token</td><td><code>0.5 × 10^(e-5)</code> in declared units; exchange eV, cell/positions converted from Bohr to Å</td></tr>
</tbody>
</table>

Text input accepts the current fixed/scientific NumPy matrix regimes; mixed fixed/scientific matrices, unsupported precision patterns, and unknown printer regimes are rejected rather than assigned guessed tolerances. Suppressed `0.` matrix entries use the configured fixed precision, not their apparent integer spelling. XML bounds use each emitted token's exponent and trailing digits. Conversion reports compare/reconcile using these declared per-token intervals and report differences; they do not silently rescale or requantize exchange values.

## Further references

- Python implementation: `TB2J/io_exchange/shells.py` (`ShellDocument`, `template`, `compress`, `expand`, `read_exchange`, `write_exchange`), `TB2J/io_exchange/shell_text.py` (TOML codec), and `TB2J/io_exchange/native_readers.py` (text/XML readers and precision profiles).
- CLI implementation and installed command: `TB2J/scripts/TB2J_shell.py` and the `TB2J_shell` entry in `pyproject.toml`.
- Consumer regression suites: `tests/test_shell_authoring.py`, `tests/test_shells.py`, `tests/test_shell_cli.py`, `tests/test_shell_native_readers.py`, `tests/test_shell_native_exports.py`, and `tests/test_shell_base_updates.py`.
- Shared SG/MSG and ordered-pair conventions: `TB2J/symmetry_core.py` and `docs/sympy/04_spacegroup_symmetrization.md`.
