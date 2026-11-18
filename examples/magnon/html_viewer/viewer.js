import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { buildBandRows, buildLabelTicks, buildScene, parseEigenstateJson } from "./parser.js";

const state = {
  eigenstates: null,
  bandRows: [],
  labelTicks: [],
  sceneData: null,
  rendererState: null,
};

const ui = {
  fileInput: document.getElementById("fileInput"),
  nx: document.getElementById("nx"),
  ny: document.getElementById("ny"),
  nz: document.getElementById("nz"),
  amplitude: document.getElementById("amplitude"),
  frames: document.getElementById("frames"),
  kIndex: document.getElementById("kIndex"),
  bandIndex: document.getElementById("bandIndex"),
  applyButton: document.getElementById("applyButton"),
  loadExampleButton: document.getElementById("loadExampleButton"),
  bandCanvas: document.getElementById("bandCanvas"),
  status: document.getElementById("status"),
  viewer3d: document.getElementById("viewer3d"),
  viewerxy: document.getElementById("viewerxy"),
};

ui.fileInput.addEventListener("change", loadSelectedFile);
ui.applyButton.addEventListener("click", rebuildScene);
ui.loadExampleButton.addEventListener("click", loadBundledExample);
ui.bandCanvas.addEventListener("click", selectBandPoint);
for (const input of [ui.nx, ui.ny, ui.nz, ui.amplitude, ui.frames]) {
  input.addEventListener("change", rebuildScene);
}

async function loadSelectedFile() {
  const file = ui.fileInput.files[0];
  if (!file) return;
  loadEigenstates(parseEigenstateJson(JSON.parse(await file.text())), file.name);
}

async function loadBundledExample() {
  const response = await fetch("./data/CrI3_monolayer_magnon.json");
  if (!response.ok) throw new Error(`Failed to load bundled example: ${response.status}`);
  loadEigenstates(parseEigenstateJson(await response.json()), "CrI3_monolayer_magnon.json");
}

function loadEigenstates(eigenstates, label) {
  state.eigenstates = eigenstates;
  state.bandRows = buildBandRows(eigenstates);
  state.labelTicks = buildLabelTicks(eigenstates);
  ui.kIndex.max = Math.max(0, eigenstates.kpoints.length - 1);
  ui.bandIndex.max = Math.max(0, eigenstates.energies[0].length - 1);
  ui.status.textContent = `Loaded ${label}`;
  drawBandChart();
  rebuildScene();
}

function rebuildScene() {
  if (!state.eigenstates) return;
  state.sceneData = buildScene(state.eigenstates, {
    kIndex: Number(ui.kIndex.value),
    bandIndex: Number(ui.bandIndex.value),
    amplitude: Number(ui.amplitude.value),
    nframes: Number(ui.frames.value),
    repetitions: [Number(ui.nx.value), Number(ui.ny.value), Number(ui.nz.value)],
  });
  drawBandChart();
  renderScene(state.sceneData);
}

function drawBandChart() {
  const canvas = ui.bandCanvas;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.bandRows.length) return;

  const rows = state.bandRows;
  const margin = { left: 52 * dpr, right: 18 * dpr, top: 18 * dpr, bottom: 44 * dpr };
  const xs = rows.map((r) => r.x);
  const ys = rows.map((r) => r.energy);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys);
  const ymax = Math.max(...ys);
  const sx = (x) => margin.left + ((x - xmin) / Math.max(xmax - xmin, 1e-12)) * (canvas.width - margin.left - margin.right);
  const sy = (y) => canvas.height - margin.bottom - ((y - ymin) / Math.max(ymax - ymin, 1e-12)) * (canvas.height - margin.top - margin.bottom);

  ctx.strokeStyle = "#d8d8d8";
  ctx.lineWidth = dpr;
  ctx.strokeRect(margin.left, margin.top, canvas.width - margin.left - margin.right, canvas.height - margin.top - margin.bottom);
  ctx.fillStyle = "#333333";
  ctx.font = `${12 * dpr}px sans-serif`;
  ctx.fillText("Energy (meV)", 8 * dpr, 18 * dpr);

  for (const tick of state.labelTicks) {
    const x = sx(tick.x);
    ctx.strokeStyle = "#999999";
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, canvas.height - margin.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#111111";
    ctx.textAlign = "center";
    ctx.fillText(tick.label, x, canvas.height - 18 * dpr);
  }

  const nband = state.eigenstates.energies[0].length;
  ctx.strokeStyle = "rgba(31, 119, 180, 0.75)";
  ctx.lineWidth = 1.6 * dpr;
  for (let ib = 0; ib < nband; ib++) {
    const band = rows.filter((r) => r.bandIndex === ib).sort((a, b) => a.kIndex - b.kIndex);
    ctx.beginPath();
    for (let i = 0; i < band.length; i++) {
      const x = sx(band[i].x);
      const y = sy(band[i].energy);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const kIndex = Number(ui.kIndex.value);
  const bandIndex = Number(ui.bandIndex.value);
  const selected = rows.find((r) => r.kIndex === kIndex && r.bandIndex === bandIndex);
  if (selected) {
    ctx.fillStyle = "#d62728";
    ctx.beginPath();
    ctx.arc(sx(selected.x), sy(selected.energy), 5 * dpr, 0, 2 * Math.PI);
    ctx.fill();
  }
  canvas._plot = { sx, sy, xmin, xmax, ymin, ymax, margin };
}

function selectBandPoint(event) {
  if (!state.bandRows.length || !ui.bandCanvas._plot) return;
  const rect = ui.bandCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const x = (event.clientX - rect.left) * dpr;
  const y = (event.clientY - rect.top) * dpr;
  let best = null;
  let bestDist = Infinity;
  for (const row of state.bandRows) {
    const dx = ui.bandCanvas._plot.sx(row.x) - x;
    const dy = ui.bandCanvas._plot.sy(row.energy) - y;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      best = row;
      bestDist = dist;
    }
  }
  if (!best) return;
  ui.kIndex.value = best.kIndex;
  ui.bandIndex.value = best.bandIndex;
  rebuildScene();
}

function renderScene(sceneData) {
  if (state.rendererState) state.rendererState.dispose();
  state.rendererState = makeRenderer(ui.viewer3d, ui.viewerxy, sceneData);
}

function makeRenderer(root, xyCanvas, sceneData) {
  root.replaceChildren();
  const width = root.clientWidth || 900;
  const height = root.clientHeight || 600;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  root.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));

  const positions = sceneData.sites.positions;
  const frames = sceneData.frames;
  const allPositions = positions.concat(sceneData.structure.positions || []);
  const extentBox = new THREE.Box3();
  for (const pos of allPositions) extentBox.expandByPoint(new THREE.Vector3(...pos));
  const extentSize = extentBox.isEmpty() ? 1.0 : Math.max(extentBox.getSize(new THREE.Vector3()).length(), 1.0);
  const atomRadius = Math.max(0.12, extentSize * 0.035);
  const spinScale = Math.max(0.7, extentSize * 0.18);
  const headRadius = Math.max(0.05, extentSize * 0.015);
  const headLength = headRadius * 3.0;

  const lines = [];
  const heads = [];
  const trails = [];
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x1f77b4 });
  const headGeometry = new THREE.ConeGeometry(headRadius, headLength, 18);
  const headMaterial = new THREE.MeshBasicMaterial({ color: 0x1f77b4 });
  const trailGeometry = new THREE.SphereGeometry(headRadius * 0.45, 10, 10);
  for (const pos of positions) {
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...pos), new THREE.Vector3(...pos)]), lineMaterial);
    const head = new THREE.Mesh(headGeometry, headMaterial);
    scene.add(line, head);
    lines.push(line);
    heads.push(head);
    trails.push([]);
  }
  drawAtoms(scene, sceneData, atomRadius);
  drawCell(scene, sceneData.structure.cell);
  centerCamera(scene, camera, controls);

  const xyState = makeXYState(xyCanvas, positions, atomRadius);
  let iframe = 0;
  let active = true;
  function animate() {
    if (!active) return;
    const frame = frames[iframe % frames.length];
    for (let i = 0; i < lines.length; i++) {
      const start = new THREE.Vector3(...positions[i]);
      const axis = new THREE.Vector3(...sceneData.sites.reference_spins[i]).multiplyScalar(spinScale * 0.8);
      const spin = new THREE.Vector3(...frame[i]).multiplyScalar(spinScale);
      const end = start.clone().add(axis).add(spin);
      lines[i].geometry.setFromPoints([start, end]);
      const direction = end.clone().sub(start).normalize();
      heads[i].position.copy(end.clone().sub(direction.clone().multiplyScalar(headLength * 0.5)));
      heads[i].quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      if (iframe % 2 === 0) addTrailDot(scene, trails[i], trailGeometry, end);
    }
    drawXYView(xyCanvas, xyState, sceneData, frame, iframe, atomRadius);
    iframe += 1;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
  return { dispose: () => { active = false; renderer.dispose(); root.replaceChildren(); } };
}

function drawAtoms(scene, sceneData, atomRadius) {
  const colors = { C: 0x555555, Cr: 0x7f3c8d, I: 0x9467bd, Fe: 0xd62728, Mn: 0x2ca02c, Ni: 0x1f77b4, Co: 0x17becf, O: 0xff7f0e };
  const symbols = sceneData.structure.symbols || [];
  for (let i = 0; i < sceneData.structure.positions.length; i++) {
    const color = colors[symbols[i]] || 0x444444;
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(atomRadius, 24, 24), new THREE.MeshBasicMaterial({ color }));
    sphere.position.set(...sceneData.structure.positions[i]);
    scene.add(sphere);
  }
}

function drawCell(scene, cell) {
  const vecs = cell.map((v) => new THREE.Vector3(...v));
  const o = new THREE.Vector3(0, 0, 0);
  const [a, b, c] = vecs;
  const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
  const edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
  const material = new THREE.LineBasicMaterial({ color: 0x999999 });
  for (const [i, j] of edges) scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([corners[i], corners[j]]), material));
}

function centerCamera(scene, camera, controls) {
  const box = new THREE.Box3().setFromObject(scene);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  camera.position.copy(center.clone().add(new THREE.Vector3(size, size, size)));
  camera.near = Math.max(size / 1000, 0.001);
  camera.far = Math.max(size * 10, 100);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function addTrailDot(scene, trail, geometry, position) {
  const dot = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color: 0x1f77b4, transparent: true, opacity: 0.55 }));
  dot.position.copy(position);
  scene.add(dot);
  trail.push(dot);
  if (trail.length > 14) {
    const old = trail.shift();
    scene.remove(old);
    old.material.dispose();
  }
  for (let i = 0; i < trail.length; i++) trail[i].material.opacity = 0.55 * (i + 1) / trail.length;
}

function makeXYState(canvas, positions, atomRadius) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const xs = positions.map((p) => p[0]);
  const ys = positions.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const margin = 36 * dpr;
  const scaleXY = Math.min((canvas.width - 2 * margin) / Math.max(maxX - minX, 1e-6), (canvas.height - 2 * margin) / Math.max(maxY - minY, 1e-6));
  return { dpr, margin, scale: scaleXY, minX, maxY, atomRadius };
}

function drawXYView(canvas, xy, sceneData, frame, iframe, atomRadius) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = `${12 * xy.dpr}px sans-serif`;
  ctx.fillStyle = "#222222";
  ctx.fillText("XY view: dot = out of plane, cross = into plane", 12 * xy.dpr, 20 * xy.dpr);
  for (let i = 0; i < sceneData.sites.positions.length; i++) {
    const pos = sceneData.sites.positions[i];
    const x = xy.margin + (pos[0] - xy.minX) * xy.scale;
    const y = xy.margin + (xy.maxY - pos[1]) * xy.scale;
    const ref = sceneData.sites.reference_spins[i];
    const spin = frame[i];
    const radius = Math.max(5 * xy.dpr, Math.min(10 * xy.dpr, atomRadius * xy.scale * 0.35));
    const circleRadius = radius * 1.45;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = ref[2] >= 0 ? "#fff4cc" : "#eeeeff";
    ctx.fill();
    ctx.strokeStyle = "#666666";
    ctx.stroke();
    if (ref[2] >= 0) {
      ctx.beginPath();
      ctx.arc(x, y, radius * 0.28, 0, 2 * Math.PI);
      ctx.fillStyle = "#111111";
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(x - radius * 0.42, y - radius * 0.42);
      ctx.lineTo(x + radius * 0.42, y + radius * 0.42);
      ctx.moveTo(x + radius * 0.42, y - radius * 0.42);
      ctx.lineTo(x - radius * 0.42, y + radius * 0.42);
      ctx.strokeStyle = "#111111";
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, circleRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = "rgba(31, 119, 180, 0.45)";
    ctx.stroke();
    const angle = Math.atan2(-spin[1], spin[0]) || (iframe / sceneData.frames.length) * 2 * Math.PI;
    ctx.beginPath();
    ctx.arc(x + circleRadius * Math.cos(angle), y + circleRadius * Math.sin(angle), radius * 0.35, 0, 2 * Math.PI);
    ctx.fillStyle = "#1f77b4";
    ctx.fill();
  }
}
