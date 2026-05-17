export function parseEigenstateJson(data) {
  if (data.schema_name !== "tb2j.magnon.eigenstates") {
    throw new Error("Unsupported JSON schema: expected tb2j.magnon.eigenstates");
  }
  if (!data.wavefunctions) {
    throw new Error("Wavefunctions are required. Export with --save-wavefunctions.");
  }
  return data;
}

export function buildBandRows(eigenstates) {
  const plot = eigenstates.plot || {};
  const energies = plot.energies_mev || eigenstates.energies.map((row) => row.map((e) => e * 1000));
  const xvalues = flattenXCoords(plot.xcoords, energies.length);
  const rows = [];
  for (let ik = 0; ik < energies.length; ik++) {
    for (let ib = 0; ib < energies[ik].length; ib++) {
      rows.push({ kIndex: ik, bandIndex: ib, x: xvalues[ik] ?? ik, energy: energies[ik][ib] });
    }
  }
  return rows;
}

export function buildLabelTicks(eigenstates) {
  const labels = eigenstates.plot?.kpath_labels || [];
  const xvalues = flattenXCoords(eigenstates.plot?.xcoords, eigenstates.kpoints.length);
  return labels
    .map(([index, label]) => ({ x: xvalues[Number(index)] ?? Number(index), label: cleanLabel(label) }))
    .sort((a, b) => a.x - b.x);
}

export function buildScene(eigenstates, options) {
  const nspin = Number(eigenstates.metadata?.nspin ?? eigenstates.energies[0].length);
  const kpoint = eigenstates.kpoints[options.kIndex];
  const coeffs = getWavefunctionCoeffs(eigenstates, options.kIndex, options.bandIndex, nspin);
  const magmoms = eigenstates.metadata?.magmoms ?? Array.from({ length: nspin }, () => [0, 0, 1]);
  const positions = eigenstates.metadata?.positions ?? Array.from({ length: nspin }, (_, i) => [i, 0, 0]);
  const symbols = eigenstates.metadata?.symbols ?? Array.from({ length: nspin }, () => "X");
  const cell = eigenstates.metadata?.cell ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const atomPositions = eigenstates.metadata?.atom_positions ?? positions;
  const atomSymbols = eigenstates.metadata?.atom_symbols ?? symbols;

  const spins = normalizeRows(magmoms);
  const [e1, e2] = transverseAxes(spins);
  const repeated = repeatSpinData(positions, spins, coeffs, e1, e2, symbols, cell, options.repetitions, kpoint, options.amplitude);
  const structure = repeatStructureData(atomPositions, atomSymbols, cell, options.repetitions);
  const frames = buildFrames(repeated.referenceSpins, repeated.amplitudes, options.nframes);
  const supercell = cell.map((v, i) => scale(v, options.repetitions[i]));
  return {
    sites: { positions: repeated.positions, reference_spins: repeated.referenceSpins, symbols: repeated.symbols },
    structure: { positions: structure.positions, symbols: structure.symbols, cell: supercell, unit_cell: cell },
    frames,
    display: { atoms: true, cell: true },
    mode: { kpoint_index: options.kIndex, band_index: options.bandIndex, kpoint, frequency: eigenstates.energies[options.kIndex][options.bandIndex] },
  };
}

function flattenXCoords(xcoords, n) {
  if (!xcoords) return Array.from({ length: n }, (_, i) => i);
  if (Array.isArray(xcoords[0])) return xcoords.flat();
  return xcoords;
}

function cleanLabel(label) {
  const text = String(label);
  if (["G", "Gamma", "$\\Gamma$"].includes(text)) return "Γ";
  return text.replaceAll("$", "").replaceAll("\\Gamma", "Γ");
}

function getWavefunctionCoeffs(eigenstates, ik, ib, nspin) {
  const wf = eigenstates.wavefunctions;
  if (!wf) throw new Error("Wavefunctions are required for animation. Export with --save-wavefunctions.");
  if (wf.encoding === "complex_split") {
    return wf.real[ik][ib].slice(0, nspin).map((real, i) => [real, wf.imag[ik][ib][i]]);
  }
  return wf[ik][ib].slice(0, nspin).map((value) => Array.isArray(value) ? value : [value, 0]);
}

function normalizeRows(rows) {
  return rows.map((v) => {
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  });
}

function transverseAxes(spins) {
  const e1 = [];
  const e2 = [];
  for (const spin of spins) {
    const trial = Math.abs(dot(spin, [0, 0, 1])) > 0.9 ? [1, 0, 0] : [0, 0, 1];
    const a = normalize(cross(spin, trial));
    e1.push(a);
    e2.push(normalize(cross(spin, a)));
  }
  return [e1, e2];
}

function repeatSpinData(positions, spins, coeffs, e1, e2, symbols, cell, reps, kpoint, amplitude) {
  const out = { positions: [], referenceSpins: [], amplitudes: [], symbols: [] };
  for (let i = 0; i < reps[0]; i++) for (let j = 0; j < reps[1]; j++) for (let k = 0; k < reps[2]; k++) {
    const image = [i, j, k];
    const shift = matmul(image, cell);
    const phase = 2 * Math.PI * dot(kpoint, image);
    const phaseComplex = [Math.cos(phase), Math.sin(phase)];
    for (let s = 0; s < positions.length; s++) {
      const c = cmul(coeffs[s], phaseComplex);
      const ampReal = add(scale(e1[s], c[0]), scale(e2[s], -c[1])).map((x) => x * amplitude);
      const ampImag = add(scale(e2[s], c[0]), scale(e1[s], c[1])).map((x) => x * amplitude);
      out.positions.push(add(positions[s], shift));
      out.referenceSpins.push(spins[s]);
      out.amplitudes.push({ real: ampReal, imag: ampImag });
      out.symbols.push(symbols[s]);
    }
  }
  return out;
}

function repeatStructureData(positions, symbols, cell, reps) {
  const out = { positions: [], symbols: [] };
  for (let i = 0; i < reps[0]; i++) for (let j = 0; j < reps[1]; j++) for (let k = 0; k < reps[2]; k++) {
    const shift = matmul([i, j, k], cell);
    for (let a = 0; a < positions.length; a++) {
      out.positions.push(add(positions[a], shift));
      out.symbols.push(symbols[a]);
    }
  }
  return out;
}

function buildFrames(referenceSpins, amplitudes, nframes) {
  const frames = [];
  for (let iframe = 0; iframe < nframes; iframe++) {
    const time = 2 * Math.PI * iframe / nframes;
    frames.push(referenceSpins.map((spin, i) => {
      const realPart = scale(amplitudes[i].real, Math.cos(time));
      const imagPart = scale(amplitudes[i].imag, -Math.sin(time));
      return add(realPart, imagPart);
    }));
  }
  return frames;
}

function add(a, b) { return a.map((x, i) => x + b[i]); }
function scale(a, s) { return a.map((x) => x * s); }
function dot(a, b) { return a.reduce((sum, x, i) => sum + x * b[i], 0); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(a) { const n = Math.hypot(...a) || 1; return a.map((x) => x / n); }
function matmul(v, m) { return [0, 1, 2].map((j) => v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]); }
function cmul(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]; }
