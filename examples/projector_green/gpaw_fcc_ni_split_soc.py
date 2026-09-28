"""Run the real fcc Ni split-SOC GPAW symmetry, rank-six, and MAE gate.

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
from TB2J.io_merge import Merger, read_pickle


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


def run_gate(checkpoint: Path, output_path: Path) -> dict:
    """Run x/y/z second variation, then check full-BZ cubic/inversion nulls."""
    from gpaw import GPAW
    from gpaw.spinorbit import soc_eigenstates

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
    legs = {name: read_pickle(str(path)) for name, path in result["leg_paths"].items()}
    merged = read_pickle(str(result["merged_path"]))
    ranks = [
        int(np.linalg.matrix_rank(matrix, tol=1e-2))
        for matrix in Merger(
            *(str(path) for path in result["leg_paths"].values())
        ).coeff_matrix.values()
    ]
    assert ranks and all(rank == 6 for rank in ranks), ranks
    pairs = {}
    for key, value in merged.exchange_Jdict.items():
        if key[0] == (0, 0, 0):
            continue
        jlegs = [leg.exchange_Jdict[key] for leg in legs.values()]
        pairs[str(key)] = {
            "Jiso_meV": float(value * 1e3),
            "Jiso_spread_meV": float((max(jlegs) - min(jlegs)) * 1e3),
            "DMI_norm_meV": float(np.linalg.norm(merged.dmi_ddict[key]) * 1e3),
            "Jani_norm_meV": float(np.linalg.norm(merged.Jani_dict[key]) * 1e3),
        }
    assert pairs and max(row["Jiso_spread_meV"] for row in pairs.values()) < 0.005
    assert max(row["DMI_norm_meV"] for row in pairs.values()) < 1e-6
    assert max(row["Jani_norm_meV"] for row in pairs.values()) < 0.02
    calc = GPAW(str(checkpoint), legacy_gpaw=True)
    for name, (theta, phi) in {"x": (90, 0), "y": (90, 90), "z": (0, 0)}.items():
        direct = soc_eigenstates(
            calc, theta=theta, phi=phi, projected=False
        ).calculate_band_energy()
        assert result["mae"][name]["band_energy_eV"] == direct
        assert result["mae"][name]["contour_within_tolerance"]
    report = {"rank": ranks, "pairs": pairs, "mae": result["mae"]}
    (output_path / "ni_symmetry_gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


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
