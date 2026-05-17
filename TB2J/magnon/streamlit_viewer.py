"""Streamlit/Three.js helpers for magnon spin-wave visualization."""

from __future__ import annotations

import json
import os

from TB2J.magnon.eigenstates import MagnonEigenstateData


def band_dataframe(eigenstates):
    """Return a DataFrame with one row per band/k-point for selection."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for the Streamlit band chart") from exc

    plot = eigenstates.plot or {}
    energies = plot.get("energies_mev", (eigenstates.energies * 1000.0).tolist())
    xcoords = plot.get("xcoords")
    if xcoords is None:
        xvalues = list(range(len(energies)))
    elif xcoords and isinstance(xcoords[0], list):
        xvalues = [x for segment in xcoords for x in segment]
    else:
        xvalues = xcoords

    rows = []
    for ik, band_values in enumerate(energies):
        for ib, energy in enumerate(band_values):
            rows.append(
                {
                    "k_index": ik,
                    "band_index": ib,
                    "x": xvalues[ik] if ik < len(xvalues) else ik,
                    "energy_mev": energy,
                }
            )
    return pd.DataFrame(rows)


def band_label_ticks(eigenstates):
    """Return x-axis tick positions and labels for a band path."""
    plot = eigenstates.plot or {}
    labels = plot.get("kpath_labels") or []
    if not labels:
        return [], None
    xcoords = plot.get("xcoords")
    if xcoords is None:
        xvalues = list(range(len(eigenstates.kpoints)))
    elif xcoords and isinstance(xcoords[0], list):
        xvalues = [x for segment in xcoords for x in segment]
    else:
        xvalues = xcoords

    ticks = []
    for index, label in labels:
        index = int(index)
        if 0 <= index < len(xvalues):
            ticks.append((float(xvalues[index]), _clean_kpath_label(label)))
    return ticks, _altair_label_expr(ticks)


def band_chart(df, label_ticks=None):
    """Return an Altair band chart with point selection enabled."""
    try:
        import altair as alt
        import pandas as pd
    except ImportError as exc:
        raise ImportError("altair is required for the Streamlit band chart") from exc

    selector = alt.selection_point(
        name="band_pick",
        fields=["k_index", "band_index"],
        nearest=True,
        on="click",
        clear=False,
    )
    x_axis = alt.Axis(title="k-path")
    if label_ticks:
        tick_values = [tick[0] for tick in label_ticks]
        label_expr = _altair_label_expr(label_ticks)
        x_axis = alt.Axis(
            values=tick_values,
            labelExpr=label_expr,
            labelAngle=0,
            title="k-path",
        )

    lines = (
        alt.Chart(df)
        .mark_line(color="#1f77b4", opacity=0.6)
        .encode(
            x=alt.X("x:Q", axis=x_axis),
            y=alt.Y("energy_mev:Q", title="Energy (meV)"),
            detail="band_index:N",
        )
    )
    points = (
        alt.Chart(df)
        .mark_circle(size=45)
        .encode(
            x=alt.X("x:Q", axis=x_axis),
            y="energy_mev:Q",
            color=alt.condition(selector, alt.value("#d62728"), alt.value("#1f77b4")),
            tooltip=["k_index:Q", "band_index:Q", "energy_mev:Q"],
        )
        .add_params(selector)
    )
    chart = lines + points
    if label_ticks:
        label_df = pd.DataFrame(
            {
                "x": [tick[0] for tick in label_ticks],
                "label": [tick[1] for tick in label_ticks],
            }
        )
        rules = (
            alt.Chart(label_df)
            .mark_rule(color="#888888", strokeDash=[3, 3])
            .encode(x="x:Q")
        )
        texts = (
            alt.Chart(label_df)
            .mark_text(align="center", baseline="top", dy=6, fontSize=13)
            .encode(x="x:Q", y=alt.value(0), text="label:N")
        )
        chart = chart + rules + texts
    return chart.properties(height=320)


def _clean_kpath_label(label):
    label = str(label)
    if label in {"G", "Gamma", "$\\Gamma$", r"$\Gamma$"}:
        return "Γ"
    return label.replace("$", "").replace("\\Gamma", "Γ")


def _altair_label_expr(ticks):
    if not ticks:
        return None
    expr = "''"
    for value, label in reversed(ticks):
        label = label.replace("'", "\\'")
        expr = f"abs(datum.value - {value:.16g}) < 1e-8 ? '{label}' : {expr}"
    return expr


def selected_band_from_event(event, fallback=(0, 0)):
    """Extract selected k-point and band indices from Streamlit chart events."""
    selection = getattr(event, "selection", None)
    if not selection:
        return fallback
    points = None
    if isinstance(selection, dict):
        points = selection.get("band_pick") or selection.get("points")
    else:
        points = getattr(selection, "band_pick", None) or getattr(
            selection, "points", None
        )
    if not points:
        return fallback
    point = points[0]
    return int(point["k_index"]), int(point["band_index"])


def build_scene_from_file(
    filename,
    kpoint_index=0,
    band_index=0,
    amplitude=1.0,
    nframes=40,
    repetitions=(1, 1, 1),
    display=None,
):
    """Build Three.js scene data from an exported magnon eigenstate file."""
    eigenstates = MagnonEigenstateData.load(filename)
    rotation = eigenstates.spin_rotation(
        kpoint_index=kpoint_index,
        band_index=band_index,
        amplitude=amplitude,
        nframes=nframes,
        repetitions=repetitions,
    )
    return rotation.to_threejs_scene(display=display)


def scene_to_html(scene, height=850):
    """Return a minimal Three.js HTML document for a scene.

    The initial viewer intentionally keeps rendering code small. It exposes the
    complete scene data to the browser and draws animated spin vectors as lines.
    """
    scene_json = json.dumps(scene)
    return f"""
<div id="tb2j-magnon-viewer" style="width: 100%; height: {int(height)}px; display: flex; flex-direction: column; gap: 10px;">
  <div id="tb2j-magnon-viewer-3d" style="width: 100%; height: 68%;"></div>
  <canvas id="tb2j-magnon-viewer-xy" style="width: 100%; height: 32%; border: 1px solid #dddddd; background: #ffffff;"></canvas>
</div>
<script type="importmap">
{{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

const sceneData = {scene_json};
const root = document.getElementById('tb2j-magnon-viewer-3d');
const xyCanvas = document.getElementById('tb2j-magnon-viewer-xy');
const width = root.clientWidth || 800;
const height = root.clientHeight || Math.floor({int(height)} * 0.68);
const renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(width, height);
root.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
camera.position.set(3, 3, 3);
camera.lookAt(0, 0, 0);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.9));

const material = new THREE.LineBasicMaterial({{color: 0x1f77b4}});
const lines = [];
const heads = [];
const trails = [];
const positions = sceneData.sites.positions;
const frames = sceneData.frames;
const allPositions = positions.concat(sceneData.structure.positions || []);
const extentBox = new THREE.Box3();
for (const pos of allPositions) {{
  extentBox.expandByPoint(new THREE.Vector3(...pos));
}}
const extentSize = extentBox.isEmpty() ? 1.0 : Math.max(extentBox.getSize(new THREE.Vector3()).length(), 1.0);
const atomRadius = Math.max(0.12, extentSize * 0.035);
const spinScale = Math.max(0.7, extentSize * 0.18);
const headRadius = Math.max(0.05, extentSize * 0.015);
const headLength = headRadius * 3.0;
const trailLength = 14;
const headGeometry = new THREE.ConeGeometry(headRadius, headLength, 18);
const headMaterial = new THREE.MeshBasicMaterial({{color: 0x1f77b4}});
const trailGeometry = new THREE.SphereGeometry(headRadius * 0.45, 10, 10);
const xyState = makeXYState();

for (let i = 0; i < positions.length; i++) {{
  const geometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(...positions[i]),
    new THREE.Vector3(...positions[i])
  ]);
  const line = new THREE.Line(geometry, material);
  scene.add(line);
  lines.push(line);
  const head = new THREE.Mesh(headGeometry, headMaterial);
  scene.add(head);
  heads.push(head);
  trails.push([]);
}}

if (sceneData.display.atoms) {{
  const elementColors = {{C: 0x555555, Cr: 0x7f3c8d, I: 0x9467bd, Fe: 0xd62728, Mn: 0x2ca02c, Ni: 0x1f77b4, Co: 0x17becf, O: 0xff7f0e}};
  const atomSymbols = sceneData.structure.symbols || [];
  for (let i = 0; i < sceneData.structure.positions.length; i++) {{
    const pos = sceneData.structure.positions[i];
    const color = elementColors[atomSymbols[i]] || 0x444444;
    const atomMaterial = new THREE.MeshBasicMaterial({{color}});
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(atomRadius, 24, 24), atomMaterial);
    sphere.position.set(...pos);
    scene.add(sphere);
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const context = canvas.getContext('2d');
    context.font = 'bold 32px sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillStyle = '#111111';
    context.fillText(atomSymbols[i] || 'X', 64, 32);
    const texture = new THREE.CanvasTexture(canvas);
    const labelMaterial = new THREE.SpriteMaterial({{map: texture, transparent: true}});
    const label = new THREE.Sprite(labelMaterial);
    label.position.set(pos[0], pos[1], pos[2] + atomRadius * 1.6);
    label.scale.set(atomRadius * 2.8, atomRadius * 1.4, 1);
    scene.add(label);
  }}
}}

if (sceneData.display.cell) {{
  const cell = sceneData.structure.cell;
  const a = new THREE.Vector3(...cell[0]);
  const b = new THREE.Vector3(...cell[1]);
  const c = new THREE.Vector3(...cell[2]);
  const o = new THREE.Vector3(0, 0, 0);
  const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
  const edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
  const cellMaterial = new THREE.LineBasicMaterial({{color: 0x999999}});
  for (const [i, j] of edges) {{
    const geometry = new THREE.BufferGeometry().setFromPoints([corners[i], corners[j]]);
    scene.add(new THREE.Line(geometry, cellMaterial));
  }}
}}

const box = new THREE.Box3().setFromObject(scene);
if (!box.isEmpty()) {{
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 1.0;
  camera.position.copy(center.clone().add(new THREE.Vector3(size, size, size)));
  camera.near = Math.max(size / 1000.0, 0.001);
  camera.far = Math.max(size * 10.0, 100.0);
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.target.copy(center);
  controls.update();
}}

let iframe = 0;
function animate() {{
  const frame = frames[iframe % frames.length];
  for (let i = 0; i < lines.length; i++) {{
    const start = new THREE.Vector3(...positions[i]);
    const spin = new THREE.Vector3(...frame[i]);
    const end = start.clone().add(spin.multiplyScalar(spinScale));
    lines[i].geometry.setFromPoints([start, end]);
    const direction = end.clone().sub(start).normalize();
    heads[i].position.copy(end.clone().sub(direction.clone().multiplyScalar(headLength * 0.5)));
    heads[i].quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    if (iframe % 2 === 0) {{
      const trailMaterial = new THREE.MeshBasicMaterial({{color: 0x1f77b4, transparent: true, opacity: 0.55}});
      const dot = new THREE.Mesh(trailGeometry, trailMaterial);
      dot.position.copy(end);
      scene.add(dot);
      trails[i].push(dot);
      if (trails[i].length > trailLength) {{
        const old = trails[i].shift();
        scene.remove(old);
        old.material.dispose();
      }}
      for (let j = 0; j < trails[i].length; j++) {{
        trails[i][j].material.opacity = 0.55 * (j + 1) / trails[i].length;
      }}
    }}
  }}
  drawXYView(frame, iframe);
  iframe += 1;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}}
animate();

function makeXYState() {{
  const canvas = xyCanvas;
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
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const margin = 36 * dpr;
  const scale = Math.min((canvas.width - 2 * margin) / spanX, (canvas.height - 2 * margin) / spanY);
  return {{canvas, dpr, margin, scale, minX, minY, maxY}};
}}

function xyPoint(pos) {{
  return [
    xyState.margin + (pos[0] - xyState.minX) * xyState.scale,
    xyState.margin + (xyState.maxY - pos[1]) * xyState.scale,
  ];
}}

function drawXYView(frame, iframe) {{
  const ctx = xyCanvas.getContext('2d');
  ctx.clearRect(0, 0, xyCanvas.width, xyCanvas.height);
  ctx.save();
  ctx.font = `${{12 * xyState.dpr}}px sans-serif`;
  ctx.fillStyle = '#222222';
  ctx.fillText('XY view: dot = spin axis out of plane, cross = into plane; blue marker shows rotation phase', 12 * xyState.dpr, 20 * xyState.dpr);
  for (let i = 0; i < positions.length; i++) {{
    const [x, y] = xyPoint(positions[i]);
    const spin = frame[i];
    const ref = sceneData.sites.reference_spins[i] || [0, 0, 1];
    const radius = Math.max(5 * xyState.dpr, Math.min(10 * xyState.dpr, atomRadius * xyState.scale * 0.35));
    const circleRadius = radius * 1.45;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = ref[2] >= 0 ? '#fff4cc' : '#eeeeff';
    ctx.fill();
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 1.2 * xyState.dpr;
    ctx.stroke();
    if (ref[2] >= 0) {{
      ctx.beginPath();
      ctx.arc(x, y, radius * 0.28, 0, 2 * Math.PI);
      ctx.fillStyle = '#111111';
      ctx.fill();
    }} else {{
      ctx.beginPath();
      ctx.moveTo(x - radius * 0.42, y - radius * 0.42);
      ctx.lineTo(x + radius * 0.42, y + radius * 0.42);
      ctx.moveTo(x + radius * 0.42, y - radius * 0.42);
      ctx.lineTo(x - radius * 0.42, y + radius * 0.42);
      ctx.strokeStyle = '#111111';
      ctx.lineWidth = 2 * xyState.dpr;
      ctx.stroke();
    }}
    ctx.beginPath();
    ctx.arc(x, y, circleRadius, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(31, 119, 180, 0.45)';
    ctx.lineWidth = 1.4 * xyState.dpr;
    ctx.stroke();
    const dx = spin[0] - ref[0];
    const dy = spin[1] - ref[1];
    let angle = Math.atan2(-dy, dx);
    if (Math.abs(dx) + Math.abs(dy) < 1e-9) {{
      angle = (iframe / frames.length) * 2 * Math.PI;
    }}
    const px = x + circleRadius * Math.cos(angle);
    const py = y + circleRadius * Math.sin(angle);
    ctx.beginPath();
    ctx.arc(px, py, radius * 0.35, 0, 2 * Math.PI);
    ctx.fillStyle = '#1f77b4';
    ctx.fill();
    const ahead = angle + 0.42;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(x + circleRadius * Math.cos(ahead), y + circleRadius * Math.sin(ahead));
    ctx.strokeStyle = '#1f77b4';
    ctx.lineWidth = 2 * xyState.dpr;
    ctx.stroke();
  }}
  ctx.restore();
}}
</script>
"""


def render_scene(scene, height=850):
    """Render scene data in Streamlit using a Three.js HTML component."""
    try:
        import streamlit.components.v1 as components
    except ImportError as exc:
        raise ImportError(
            "streamlit is required for magnon animation rendering"
        ) from exc
    components.html(scene_to_html(scene, height=height), height=height)


def main():
    """Small Streamlit entry point for exported magnon eigenstate files."""
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "streamlit is required for magnon animation rendering"
        ) from exc

    st.set_page_config(layout="wide")
    st.title("TB2J Magnon Spin-Wave Viewer")
    controls, viewer = st.columns([1, 2], gap="large")
    with controls:
        filename = st.text_input(
            "Magnon eigenstate file",
            value=os.environ.get("TB2J_MAGNON_VIEWER_FILE", ""),
        )
        nx = st.number_input("supercell nx", min_value=1, value=2, step=1)
        ny = st.number_input("supercell ny", min_value=1, value=2, step=1)
        nz = st.number_input("supercell nz", min_value=1, value=1, step=1)
        amplitude = st.slider("amplitude", min_value=0.0, max_value=2.0, value=1.0)
        nframes = st.slider("frames", min_value=4, max_value=200, value=40)
        scene = None
        if filename:
            eigenstates = MagnonEigenstateData.load(filename)
            df = band_dataframe(eigenstates)
            st.subheader("Band structure")
            label_ticks, _ = band_label_ticks(eigenstates)
            event = st.altair_chart(
                band_chart(df, label_ticks=label_ticks),
                use_container_width=True,
                on_select="rerun",
            )
            manual_k = int(st.number_input("k-index", min_value=0, value=0, step=1))
            manual_band = int(st.number_input("band", min_value=0, value=0, step=1))
            kpoint_index, band_index = selected_band_from_event(
                event,
                fallback=(manual_k, manual_band),
            )
            st.caption(f"Selected k-index {kpoint_index}, band {band_index}")
            scene = build_scene_from_file(
                filename,
                kpoint_index=int(kpoint_index),
                band_index=int(band_index),
                amplitude=float(amplitude),
                nframes=int(nframes),
                repetitions=(int(nx), int(ny), int(nz)),
            )
    with viewer:
        st.subheader("Animation")
        if scene is not None:
            render_scene(scene, height=900)
        else:
            st.info("Select a magnon eigenstate file to show the animation.")


if __name__ == "__main__":
    main()
