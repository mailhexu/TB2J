"""Run the real fcc Ni split-SOC GPAW rank-nine merge and MAE gate.

First run ``--build`` to generate a no-SOC legacy GPAW checkpoint, then reuse
that one checkpoint for every x/y/z SOC leg. Serial execution is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from TB2J.interfaces.gpaw_split_soc import gen_exchange_gpaw_split_soc


def build_collinear_ni(path: Path) -> None:
    """Save one converged 6x6x6 fcc Ni PBE calculation, without SOC."""
    from gpaw import GPAW, PW, FermiDirac

    a = 3.52
    cell = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    atoms = Atoms("Ni", scaled_positions=[(0, 0, 0)], cell=cell, pbc=True)
    atoms.set_initial_magnetic_moments([0.6])
    path.parent.mkdir(parents=True, exist_ok=True)
    calc = GPAW(
        mode=PW(500),
        xc="PBE",
        kpts=(6, 6, 6),
        symmetry="off",
        spinpol=True,
        nbands=12,
        occupations=FermiDirac(0.05),
        convergence={"energy": 1e-5, "density": 1e-5, "bands": "all"},
        txt=str(path.with_suffix(".log")),
        legacy_gpaw=True,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(str(path), mode="all")


def _nearest_neighbour_pairs(merged) -> list:
    """Non-on-site (R, i, j) keys nearest-neighbour along the ±x/±y/±z axes."""
    return [
        key
        for key in merged.exchange_Jdict
        if key[0] != (0, 0, 0) and int(np.abs(np.asarray(key[0])).max()) == 1
    ]


def run_gate(checkpoint: Path, output_path: Path) -> dict:
    """Run x/y/z second variation, then check full-BZ cubic/inversion nulls."""
    from gpaw import GPAW
    from gpaw.spinorbit import soc_eigenstates

    from TB2J.io_exchange.io_exchange import SpinIO

    rpts = np.array(
        [(0, 0, 0)]
        + [tuple(sign * np.eye(3, dtype=int)[i]) for i in range(3) for sign in (1, -1)]
    )
    result = gen_exchange_gpaw_split_soc(
        checkpoint,
        output_path=output_path,
        Rpts=rpts,
        Rcut=3.0,
        nz=12,
        smearing_eV=0.05,
        magnetic_sites=[0],
    )
    report = json.loads((Path(output_path) / "split_soc_report.json").read_text())
    diag = report["merge_diagnostics"]
    assert diag["min_rank"] == 9, diag
    assert diag["max_repeat_deviation"] <= 5e-5, diag
    assert diag["max_transverse_mask_residual"] < 1e-8, diag

    merged = SpinIO.load_pickle(path=str(result["merged_path"]))
    nn_keys = _nearest_neighbour_pairs(merged)
    assert len(nn_keys) == 6, nn_keys
    jiso = [merged.exchange_Jdict[key] for key in nn_keys]
    pairs = {
        str(key): {
            "Jiso_meV": float(merged.exchange_Jdict[key] * 1e3),
            "DMI_norm_meV": float(np.linalg.norm(merged.dmi_ddict[key]) * 1e3),
            "Jani_norm_meV": float(np.linalg.norm(merged.Jani_dict[key]) * 1e3),
        }
        for key in nn_keys
    }
    # cubic shells share one Jiso; inversion + O_h null the anisotropic parts
    assert (max(jiso) - min(jiso)) * 1e3 < 0.005, jiso
    assert max(row["DMI_norm_meV"] for row in pairs.values()) < 1e-6
    assert max(row["Jani_norm_meV"] for row in pairs.values()) < 0.02

    calc = GPAW(str(checkpoint), legacy_gpaw=True)
    for name, (theta, phi) in {"x": (90, 0), "y": (90, 90), "z": (0, 0)}.items():
        direct = soc_eigenstates(
            calc, theta=theta, phi=phi, projected=False
        ).calculate_band_energy()
        assert result["mae"][name]["band_energy_eV"] == direct
        assert result["mae"][name]["contour_within_tolerance"]

    gate = {
        "merge_diagnostics": diag,
        "pairs": pairs,
        "mae": result["mae"],
    }
    (output_path / "ni_symmetry_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n"
    )
    return gate


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("ni_fcc_pbe_nosoc.gpw"))
    parser.add_argument(
        "--output", type=Path, default=Path("TB2J_results_ni_split_soc")
    )
    parser.add_argument(
        "--build", action="store_true", help="run the collinear no-SOC Ni SCF first"
    )
    args = parser.parse_args(argv)
    if args.build:
        build_collinear_ni(args.input)
    if not args.input.is_file():
        parser.error(f"missing no-SOC checkpoint {args.input}; use --build")
    print(json.dumps(run_gate(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
