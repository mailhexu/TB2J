"""Run the ABINIT PAW split-SOC three-leg driver on a real schema-1.1 export.

The input ``<prefix>_SAVETB2J.nc`` is produced by one collinear PAW
ground state (``usepaw 1``, ``nsppol 2``, ``nspinor 1``, ``kptopt 0``)
exported with ``savetb2j 1`` and ``savetb2j_soc 1`` at the frozen
strength-zero density; see ``docs/src/split_soc_abinit_paw.rst``.  A
Gamma-only export has a vanishing SOC-off exchange and is not a useful
fixture: use a full-Brillouin-zone k-point list.

With ``--anchor`` the script additionally reruns the three legs with
``lam=0`` and compares each leg's raw transverse block, entry by entry,
against the collinear ``delta_total`` projector exchange computed
directly from the same file.  The exported total energy is identical
with and without the SOC export, so this anchor must close to numerical
precision; a Gamma-only fixture would make the comparison vacuous
(baseline below ``--anchor_baseline``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from TB2J.interfaces.abinit_paw_split_soc import gen_exchange_abinit_paw_split_soc


def _leg_tensors(leg_dir: Path):
    """Load one leg's raw lattice-frame ``(pair_keys, J_leg, axis)`` arrays."""
    npz = np.load(Path(leg_dir) / "split_soc_leg.npz")
    keys = [tuple(int(v) for v in key) for key in npz["pair_keys"]]
    return (
        keys,
        np.asarray(npz["J_leg"], dtype=float),
        np.asarray(npz["axis"], dtype=float),
    )


def _first_shell_indices(keys, site: int) -> list:
    """Indices/keys of the shortest nonzero R of one site pair."""
    shells = [
        item
        for item in enumerate(keys)
        if item[1][1] == site and item[1][2] == site and any(item[1][0])
    ]
    if not shells:
        return []
    rmin = min(np.linalg.norm(np.asarray(key[0])) for _, key in shells)
    return [item for item in shells if np.linalg.norm(np.asarray(item[1][0])) == rmin]


def _first_shell_rows(pickle_path: Path, site: int = 0) -> dict:
    """Merged J_iso (meV) of the shortest nonzero R of one site pair."""
    from TB2J.io_exchange.io_exchange import SpinIO

    jdict = SpinIO.load_pickle(path=str(pickle_path)).exchange_Jdict
    keys = list(jdict)
    return {
        str(key): float(jdict[key] * 1e3) for _, key in _first_shell_indices(keys, site)
    }


def _leg_transverse_deviation(leg_dir: Path, reference: dict) -> float:
    """Max |J_leg[a, b] - J_ref| over transverse (a, b != n) entries."""
    keys, j_leg, axis = _leg_tensors(leg_dir)
    n = int(np.argmax(np.abs(axis)))
    if set(keys) != set(reference):
        missing = set(reference) - set(keys)
        raise SystemExit(
            f"{leg_dir}: anchor shells {sorted(missing)} "
            "are missing from the collinear reference"
        )
    return max(
        abs(float(j_leg[idx, a, b]) - float(reference[key]))
        for idx, key in enumerate(keys)
        for a in range(3)
        for b in range(3)
        if a != n and b != n
    )


def run_anchor(filename, sites, rpts, nz, smearing_eV, output_path, tol, baseline):
    """Compare lam=0 leg transverse blocks against the collinear exchange."""
    from TB2J.interfaces.abinit_savetb2j import load_abinit_savetb2j
    from TB2J.interfaces.gpaw_projector import compute_projector_exchange_jdict

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
        deviations[direction] = _leg_transverse_deviation(Path(path), reference)
        print(
            f"anchor {direction}: max |J_leg_transverse - J_collinear| = "
            f"{deviations[direction]:.3e} eV over {len(reference)} shells"
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
        keys, j_leg, axis = _leg_tensors(Path(path))
        n = int(np.argmax(np.abs(axis)))
        for idx, key in _first_shell_indices(keys, site=sites[0]):
            diag = [j_leg[idx, a, a] * 1e3 for a in range(3) if a != n]
            print(
                f"leg {direction} transverse diag {key} = "
                + " ".join(f"{value:+.6f}" for value in diag)
                + " meV"
            )
    for key, value in _first_shell_rows(merged, site=sites[0]).items():
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
