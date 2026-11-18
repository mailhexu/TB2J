# TB2J Standalone Magnon HTML Viewer

This package is a browser-only viewer for TB2J magnon eigenstate JSON files. It includes a CrI3 monolayer example file:

- `index.html`: viewer page and controls.
- `viewer.js`: UI, band chart drawing, and Three.js rendering.
- `parser.js`: parser and frame builder for TB2J magnon eigenstate JSON.
- `data/CrI3_monolayer_magnon.json`: bundled example magnon eigenstate file.

The viewer loads Three.js from a CDN, so internet access is needed unless you vendor Three.js locally.

## How To Run

Serve this directory with a local static file server:

```bash
python -m http.server 8765 -d html_viewer
```

Then open:

```text
http://127.0.0.1:8765/
```

Use either path:

- Click `Load bundled CrI3 example` to load `data/CrI3_monolayer_magnon.json`.
- Or choose another exported TB2J magnon JSON file with the file picker.

The page should usually be served over HTTP. Opening `index.html` directly with `file://` may fail because browsers restrict ES module imports and `fetch()` for local files.

## What Is In The JSON File

The bundled JSON uses the TB2J schema `tb2j.magnon.eigenstates`. Important top-level fields are:

- `schema_name`, `schema_version`: schema identity and version.
- `calculation_type`: usually `band` for a band-path calculation.
- `metadata`: units, conventions, spin-site metadata, magnetic moments, atomic positions/symbols, and cell vectors.
- `kpoints`: fractional reciprocal-space k-points.
- `energies`: magnon energies in eV, shaped as `[nkpt][nmode]`.
- `wavefunctions`: complex split wavefunctions with `real` and `imag` arrays. These are required for animation.
- `plot`: plotting payload, including `energies_mev`, `kpath_labels`, and optional `xcoords`.

The viewer uses `metadata.positions` and `metadata.symbols` for magnetic spin sites. If present, `metadata.atom_positions` and `metadata.atom_symbols` are used for the displayed atomic structure.

## JSON Parser

The JSON parser is implemented in `parser.js` so it can be reused independently from the rendering code.

Key exported functions:

- `parseEigenstateJson(data)`: validates the schema and requires wavefunctions.
- `buildBandRows(eigenstates)`: converts the band payload to clickable chart rows.
- `buildLabelTicks(eigenstates)`: extracts k-path labels such as `Γ`, `M`, and `K`.
- `buildScene(eigenstates, options)`: builds repeated supercell spin frames and structure data for the renderer.

The spin animation uses the complex transverse amplitude and Bloch phase:

```text
phase = exp(i 2π q · R)
frame(t) = Re[A(q, mode, site, R) exp(i t)]
```

The 3D arrow displays the full spin direction: static reference spin plus the rotating transverse component.

## Generate A Compatible JSON File From TB2J

Generate a band-path eigenstate JSON with wavefunctions enabled:

```bash
TB2J_magnon.py --bands \
  -p /path/to/TB2J_results \
  --kpath GMKG \
  --npoints 60 \
  --export-format json \
  --export-prefix my_magnon \
  --save-wavefunctions
```

This writes:

```text
my_magnon.json
```

Load that file in the viewer with the file picker.

For the newer unified magnon CLI, the same requirements apply: export JSON and include wavefunctions. Without `--save-wavefunctions`, the file can still store energies but cannot animate spin motion.
