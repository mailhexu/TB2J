from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from TB2J.interfaces.sprkkr import read_sprkkr_exchange, sprkkr_to_spinio
from TB2J.magnon.magnon3 import Magnon

MOMENT_RU = 0.5674
STATES = {
    "FM": [MOMENT_RU, MOMENT_RU],
    "AFM": [MOMENT_RU, -MOMENT_RU],
}
JZZ_SCALES = [0.0, 1.0]
MIXED_JZZ_SCALES = [0.0, 1.0]
MIXED_ALPHAS = [0.25, 0.5, 1.0, -1.0]
MIXED_MODES = [
    "xz_only",
    "yz_only",
    "zx_only",
    "zy_only",
    "all_symmetric",
    "antisymmetric_xz_zx",
    "antisymmetric_yz_zy",
]


@dataclass(frozen=True)
class BandCase:
    state: str
    phase: str
    jzz_scale: float
    mixed_mode: str = "none"
    mixed_alpha: float = 0.0

    @property
    def name(self) -> str:
        if self.phase == "jzz":
            return f"{self.state}_jzz_{self.jzz_scale:g}"
        return (
            f"{self.state}_jzz_{self.jzz_scale:g}_"
            f"{self.mixed_mode}_alpha_{self.mixed_alpha:g}"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _reference_files(root: Path) -> tuple[Path, Path]:
    ref_dir = root / "Refs" / "SPRKKR_RuO2"
    return ref_dir / "RuO2.str", ref_dir / "RuO2_JXC_Jij.dat"


def _base_spinio(structure_file: Path, exchange_file: Path, moments: list[float]):
    data = read_sprkkr_exchange(
        structure_file=structure_file,
        exchange_file=exchange_file,
        magnetic_species=["Ru"],
        moment=moments,
    )
    return sprkkr_to_spinio(data, tensor_policy="transverse-block")


def _jzz_base(tensor: np.ndarray) -> float:
    return 0.5 * float(tensor[0, 0] + tensor[1, 1])


def _set_mixed_entries(tensor: np.ndarray, mode: str, value: float) -> None:
    if mode == "none":
        return
    if mode == "xz_only":
        tensor[0, 2] = value
    elif mode == "yz_only":
        tensor[1, 2] = value
    elif mode == "zx_only":
        tensor[2, 0] = value
    elif mode == "zy_only":
        tensor[2, 1] = value
    elif mode == "all_symmetric":
        tensor[0, 2] = value
        tensor[1, 2] = value
        tensor[2, 0] = value
        tensor[2, 1] = value
    elif mode == "antisymmetric_xz_zx":
        tensor[0, 2] = value
        tensor[2, 0] = -value
    elif mode == "antisymmetric_yz_zy":
        tensor[1, 2] = value
        tensor[2, 1] = -value
    else:
        raise ValueError(f"Unknown mixed mode: {mode}")


def _keep_diagonal_only(tensor: np.ndarray) -> None:
    tensor[:] = np.diag(np.diag(tensor))


def _apply_case(spinio, case: BandCase, diagonal_only_j: bool) -> None:
    if spinio.Jani_dict is None:
        raise ValueError("Expected anisotropic tensors in SpinIO.Jani_dict")
    for tensor in spinio.Jani_dict.values():
        base = _jzz_base(tensor)
        tensor[2, 2] = case.jzz_scale * base
        _set_mixed_entries(tensor, case.mixed_mode, case.mixed_alpha * base)
        if diagonal_only_j:
            _keep_diagonal_only(tensor)


def _torque_proxy(spinio) -> float:
    if spinio.Jani_dict is None:
        return 0.0
    torque = np.zeros((spinio.nspin, 2), dtype=float)
    for (_, i, _), tensor in spinio.Jani_dict.items():
        torque[i, 0] += tensor[0, 2]
        torque[i, 1] += tensor[1, 2]
    return float(np.max(np.linalg.norm(torque, axis=1)))


def _magnon_from_spinio(spinio) -> Magnon:
    magnon = Magnon.load_from_io(spinio, Jiso=True, Jani=True, DMI=False, SIA=False)
    magnon.set_reference(
        Q=np.zeros(3),
        uz=np.array([[0.0, 0.0, 1.0]]),
        n=np.array([0.0, 0.0, 1.0]),
        magmoms=magnon.magmom,
    )
    return magnon


def _special_points_for_path(path: str) -> dict[str, np.ndarray] | None:
    if "S" not in path:
        return None
    return {
        "G": np.array([0.0, 0.0, 0.0]),
        "X": np.array([0.5, 0.0, 0.0]),
        "S": np.array([0.5, 0.5, 0.0]),
    }


def _compute_case(
    case: BandCase,
    structure_file: Path,
    exchange_file: Path,
    path: str,
    npoints: int,
    diagonal_only_j: bool,
):
    spinio = _base_spinio(structure_file, exchange_file, STATES[case.state])
    _apply_case(spinio, case, diagonal_only_j=diagonal_only_j)
    magnon = _magnon_from_spinio(spinio)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        special_points = _special_points_for_path(path)
        if special_points is None:
            labels, bands_ev, _ = magnon.get_magnon_bands(
                path=path,
                npoints=npoints,
            )
        else:
            labels, bands_ev, _ = magnon.get_magnon_bands(
                path=path,
                npoints=npoints,
                special_points=special_points,
            )
    labels = sorted(labels, key=lambda item: item[0])
    bands_mev = np.asarray(bands_ev, dtype=float) * 1000.0
    return {
        "case": case,
        "labels": labels,
        "bands_mev": bands_mev,
        "torque_proxy_ev": _torque_proxy(spinio),
        "diagonal_only_j": diagonal_only_j,
        "warnings": [str(item.message) for item in caught],
    }


def _write_case_json(output_file: Path, result: dict, path: str, npoints: int) -> None:
    case = result["case"]
    payload = {
        "case": case.__dict__,
        "kpath": path,
        "npoints": npoints,
        "diagonal_only_j": result["diagonal_only_j"],
        "labels": [(int(index), str(label)) for index, label in result["labels"]],
        "bands_mev": result["bands_mev"].tolist(),
        "torque_proxy_ev": result["torque_proxy_ev"],
        "warnings": result["warnings"],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_jzz_overlay(results: list[dict], output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for iresult, result in enumerate(results):
        case = result["case"]
        bands = result["bands_mev"]
        x = np.arange(bands.shape[0])
        color = colors[iresult % len(colors)]
        for iband in range(bands.shape[1]):
            label = f"Jzz x {case.jzz_scale:g}" if iband == 0 else None
            ax.plot(x, bands[:, iband], color=color, lw=1.0, label=label)
    _format_band_axis(ax, results[0]["labels"])
    ax.set_ylabel("Energy (meV)")
    suffix = " diagonal J only" if results[0]["diagonal_only_j"] else ""
    ax.set_title(f"RuO2 {results[0]['case'].state} z-aligned, Jzz variation{suffix}")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_single_band(result: dict, output_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bands = result["bands_mev"]
    x = np.arange(bands.shape[0])
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    for iband in range(bands.shape[1]):
        ax.plot(x, bands[:, iband], color=color, lw=1.2)
    _format_band_axis(ax, result["labels"])
    ax.set_ylabel("Energy (meV)")
    case = result["case"]
    suffix = " diagonal J only" if result["diagonal_only_j"] else ""
    ax.set_title(f"RuO2 {case.state} z-aligned, Jzz=(Jxx+Jyy)/2{suffix}")
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_mixed_deviation(rows: list[dict], state: str, output_file: Path) -> None:
    selected = [
        row for row in rows if row["state"] == state and row["phase"] == "mixed"
    ]
    labels = [
        f"{row['mixed_mode']}\nα={row['mixed_alpha']:g}, z={row['jzz_scale']:g}"
        for row in selected
    ]
    deviations = [row["max_deviation_mev"] for row in selected]
    torques = [row["torque_proxy_mev"] for row in selected]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(labels)), 4.8))
    x = np.arange(len(labels))
    ax.bar(x, deviations, label="max band deviation")
    ax.plot(x, torques, color="tab:red", marker="o", lw=1.0, label="torque proxy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("meV")
    suffix = " diagonal J only" if selected and selected[0]["diagonal_only_j"] else ""
    ax.set_title(f"RuO2 {state} z-aligned, mixed z-transverse components{suffix}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _format_band_axis(ax, labels: list[tuple[int, str]]) -> None:
    if labels:
        ax.set_xticks([index for index, _ in labels])
        ax.set_xticklabels([label for _, label in labels])
        for index, _ in labels:
            ax.axvline(index, color="0.8", lw=0.8)
    ax.set_xlim(left=0)
    ax.set_xlabel("k path")
    ax.grid(axis="y", alpha=0.25)


def _summary_row(result: dict, baseline: np.ndarray | None = None) -> dict:
    case = result["case"]
    bands = result["bands_mev"]
    deviation = 0.0 if baseline is None else float(np.max(np.abs(bands - baseline)))
    return {
        "state": case.state,
        "phase": case.phase,
        "diagonal_only_j": result["diagonal_only_j"],
        "jzz_scale": case.jzz_scale,
        "mixed_mode": case.mixed_mode,
        "mixed_alpha": case.mixed_alpha,
        "min_energy_mev": float(np.min(bands)),
        "gamma_band0_mev": float(bands[0, 0]),
        "gamma_band1_mev": float(bands[0, 1]) if bands.shape[1] > 1 else "",
        "max_energy_mev": float(np.max(bands)),
        "max_deviation_mev": deviation,
        "torque_proxy_mev": result["torque_proxy_ev"] * 1000.0,
        "n_warnings": len(result["warnings"]),
        "warnings": " | ".join(result["warnings"]),
    }


def _write_metadata(output_file: Path, rows: list[dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    output_root: Path, path: str, npoints: int, diagonal_only_j: bool
) -> None:
    root = _repo_root()
    structure_file, exchange_file = _reference_files(root)
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    rows: list[dict] = []
    baselines: dict[tuple[str, float], np.ndarray] = {}

    for state in STATES:
        jzz_results = []
        for scale in JZZ_SCALES:
            case = BandCase(state=state, phase="jzz", jzz_scale=scale)
            result = _compute_case(
                case,
                structure_file,
                exchange_file,
                path,
                npoints,
                diagonal_only_j=diagonal_only_j,
            )
            jzz_results.append(result)
            baselines[(state, scale)] = result["bands_mev"]
            _write_case_json(
                output_root / "jzz" / state / f"{case.name}.json", result, path, npoints
            )
            rows.append(_summary_row(result))
        _plot_jzz_overlay(jzz_results, figures_dir / f"jzz_{state}_overlay.png")
        baseline = next(
            result for result in jzz_results if result["case"].jzz_scale == 1.0
        )
        _plot_single_band(baseline, figures_dir / f"jzz_base_{state}.png")

    for state in STATES:
        for jzz_scale in MIXED_JZZ_SCALES:
            baseline = baselines[(state, jzz_scale)]
            for mode in MIXED_MODES:
                for alpha in MIXED_ALPHAS:
                    case = BandCase(
                        state=state,
                        phase="mixed",
                        jzz_scale=jzz_scale,
                        mixed_mode=mode,
                        mixed_alpha=alpha,
                    )
                    result = _compute_case(
                        case,
                        structure_file,
                        exchange_file,
                        path,
                        npoints,
                        diagonal_only_j=diagonal_only_j,
                    )
                    _write_case_json(
                        output_root / "mixed" / state / f"{case.name}.json",
                        result,
                        path,
                        npoints,
                    )
                    rows.append(_summary_row(result, baseline=baseline))

    _write_metadata(output_root / "metadata.csv", rows)
    for state in STATES:
        _plot_mixed_deviation(rows, state, figures_dir / f"mixed_deviation_{state}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vary RuO2 z-component exchange tensors and plot magnon bands."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--kpath",
        default="GXSG",
        help="k-path for band plots (default: GXSG, i.e. Gamma-X-S-Gamma)",
    )
    parser.add_argument("--npoints", type=int, default=200)
    parser.add_argument(
        "--diagonal-only-j",
        action="store_true",
        help="Use only diagonal tensor entries Jxx, Jyy, and Jzz; zero Jxy, Jyx, and mixed z-transverse entries before computing bands.",
    )
    args = parser.parse_args()
    output_root = args.output_root
    if output_root is None:
        output_name = (
            "ruo2_z_component_outputs_diag"
            if args.diagonal_only_j
            else "ruo2_z_component_outputs"
        )
        output_root = Path(__file__).resolve().parent / output_name
    run_experiment(
        output_root, args.kpath, args.npoints, diagonal_only_j=args.diagonal_only_j
    )


if __name__ == "__main__":
    main()
