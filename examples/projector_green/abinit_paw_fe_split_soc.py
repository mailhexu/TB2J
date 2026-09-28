"""Run the ABINIT PAW split-SOC three-leg driver on a real schema-1.1 export.

The input ``<prefix>_SAVETB2J.nc`` is produced by one collinear PAW
ground state (``usepaw 1``, ``nsppol 2``, ``nspinor 1``, ``kptopt 0``)
exported with ``savetb2j 1`` and ``savetb2j_soc 1`` at the frozen
strength-zero density; see ``docs/src/split_soc_abinit_paw.rst``.  A
Gamma-only export has a vanishing SOC-off exchange and is not a useful
fixture: use a full-Brillouin-zone k-point list.

With ``--anchor`` the script additionally reruns the three legs with
``lam=0`` and compares them, shell by shell, against the collinear
``delta_total`` projector exchange computed directly from the same file.
The exported total energy is identical with and without the SOC export, so
this anchor must close to numerical precision; a Gamma-only fixture would
make the comparison vacuous (baseline below ``--anchor_baseline``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from TB2J.interfaces.abinit_paw_split_soc import gen_exchange_abinit_paw_split_soc


def _first_shell_rows(pickle_path: Path, site: int = 0) -> dict:
    """Exchange values (eV) of the shortest nonzero R of one site pair."""
    from TB2J.io_merge import read_pickle

    jdict = read_pickle(str(pickle_path)).exchange_Jdict
    shells = {
        key: value
        for key, value in jdict.items()
        if key[1] == site and key[2] == site and any(key[0])
    }
    if not shells:
        return {}
    rmin = min(np.linalg.norm(key[0]) for key in shells)
    return {
        str(key): float(value * 1e3)
        for key, value in shells.items()
        if np.linalg.norm(key[0]) == rmin
    }


def run_anchor(filename, sites, rpts, nz, smearing_eV, output_path, tol, baseline):
    """Compare lam=0 legs against the collinear delta_total exchange."""
    from TB2J.interfaces.abinit_savetb2j import load_abinit_savetb2j
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict
    from TB2J.io_merge import read_pickle

    rpts = np.asarray(rpts, dtype=int).reshape(-1, 3)
    data = load_abinit_savetb2j(filename)
    reference = compute_projector_exchange_jdict(
        data,
        Rpts=rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        sites=sites,
        operator_component="delta_total",
    )
    scale = max(abs(value) for value in reference.values())
    if scale < baseline:
        raise SystemExit(
            f"SOC-off baseline |J|={scale:.3e} eV is below {baseline:.1e} eV: "
            "the k grid is too coarse for a meaningful anchor (Gamma-only "
            "exports are vacuous). Use a full-BZ k-point list."
        )
    out = gen_exchange_abinit_paw_split_soc(
        filename,
        output_path=output_path,
        index_magnetic_atoms=list(sites),
        Rpts=rpts,
        nz=nz,
        smearing_eV=smearing_eV,
        lam=0.0,
    )
    deviations = {}
    for direction, path in out["leg_paths"].items():
        actual = read_pickle(str(path)).exchange_Jdict
        missing = set(actual) - set(reference)
        if missing:
            raise SystemExit(
                f"{direction}: anchor shells {sorted(map(tuple, missing))} "
                "are missing from the collinear reference"
            )
        deviations[direction] = max(abs(actual[key] - reference[key]) for key in actual)
        print(
            f"anchor {direction}: max |J_leg - J_collinear| = "
            f"{deviations[direction]:.3e} eV over {len(actual)} shells"
        )
    worst = max(deviations.values())
    if worst > tol:
        raise SystemExit(f"SOC-off anchor failed: {worst:.3e} eV exceeds {tol:.1e} eV")
    print(f"SOC-off anchor closed within {tol:.1e} eV.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fe_k8_soc1o_SAVETB2J.nc"),
        help="schema-1.1 ABINIT savetb2j export (savetb2j 1 + savetb2j_soc 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("TB2J_results_fe_paw_split_soc"),
    )
    parser.add_argument("--Rcut", type=float, default=8.0)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--smearing", type=float, default=0.05)
    parser.add_argument(
        "--index_magnetic_atoms",
        nargs="+",
        type=int,
        default=[1],
        help="1-based magnetic site indices (default: first atom)",
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="also run the lam=0 SOC-off anchor against the collinear exchange",
    )
    parser.add_argument(
        "--anchor_rpts",
        nargs="+",
        type=int,
        default=[0, 0, 0, 1, 0, 0, -1, 0, 0],
        help="R vectors (flat x y z triplets) for the anchor comparison",
    )
    parser.add_argument(
        "--anchor_tol", type=float, default=1e-8, help="anchor closure bound in eV"
    )
    parser.add_argument(
        "--anchor_baseline",
        type=float,
        default=1e-3,
        help="minimum collinear |J| (eV) for a non-vacuous anchor",
    )
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"missing savetb2j export {args.input}")

    sites = [i - 1 for i in args.index_magnetic_atoms]
    result = gen_exchange_abinit_paw_split_soc(
        args.input,
        output_path=args.output,
        index_magnetic_atoms=sites,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        lam=args.lam,
    )
    merged = Path(result["output_path"])
    print(f"Merged exchange: {merged / 'exchange.out'}")
    for direction, path in result["leg_paths"].items():
        rows = _first_shell_rows(Path(path), site=sites[0])
        for key, value in rows.items():
            print(f"leg {direction} J_iso({key}) = {value:+.6f} meV")
    rows = _first_shell_rows(merged, site=sites[0])
    for key, value in rows.items():
        print(f"merged J_iso({key}) = {value:+.6f} meV")

    if args.anchor:
        run_anchor(
            args.input,
            sites,
            args.anchor_rpts,
            args.nz,
            args.smearing,
            args.output.with_name(args.output.name + "_anchor"),
            args.anchor_tol,
            args.anchor_baseline,
        )


if __name__ == "__main__":
    main()
