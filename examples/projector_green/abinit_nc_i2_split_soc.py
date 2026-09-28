"""Run the ABINIT NC (PAO) split-SOC three-leg driver on a real fixture.

The inputs are the two strength-zero artifacts of the norm-conserving PAO
workflow (see ``docs/src/split_soc_abinit_nc.rst``):

* ``--pao_hs``: an ``abinit.nc_pao_hs`` v2 projection file written by
  ``abinao project-pao`` from one collinear (``nsppol 2``, ``nspinor 1``)
  full-Brillouin-zone strength-0 WFK;
* ``--soc_kernel``: an ``abinao.nc_soc_ks`` v1 sidecar carrying the
  all-atom band-window SOC operator ``W_SO(k)`` for the x/y/z legs, with
  the SHA-256 provenance that TB2J checks before accepting the pairing.

The driver runs the three split-SOC legs as global SU(2) rotations of one
no-SOC spinor reference on the dualized PAO maps and merges the measured
transverse blocks with the rank-9 ``merge_transverse_legs`` core.  Gates
run by default and their measured reports are printed:

* the SOC-off collinear anchor (W=0 z reference vs the existing collinear
  NC PAO exchange kernel, same strength-0 data);
* the rank-9 merge invariance gate: repeated diagonal rows (measured in
  two different legs) must agree within ``merge_consistency_atol``
  (default 5e-2 eV — documenting reference-state spread per pair in
  ``merge_diagnostics``, not certifying exactness);
* the FR-032 projection-only (tangent-block) gate: the merged raw
  transverse block ``[:2, :2]`` vs the z one-shot leg, tolerance auto by
  default (2x the worst repeat deviation, reported as ``tol_source``).
  This gate is **projection-only** and is never a full-tensor or DMI
  certificate.

The script prints only the gate reports and provenance summaries; it never
reports per-leg Jiso/DMI/Jani as final observables (per legs only the raw
transverse ``J_leg`` blocks are stored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "ABINIT NC split-SOC three-leg exchange from an abinao "
            "abinit.nc_pao_hs v2 file and an abinao.nc_soc_ks v1 SOC sidecar"
        )
    )
    parser.add_argument("--pao_hs", required=True, help="abinit.nc_pao_hs v2 file")
    parser.add_argument(
        "--soc_kernel", required=True, help="abinao.nc_soc_ks v1 sidecar"
    )
    parser.add_argument(
        "--wfk",
        default=None,
        help="strength-0 WFK; when given its SHA-256 is checked against the "
        "sidecar provenance",
    )
    parser.add_argument("--output", default="TB2J_results_nc_split_soc")
    parser.add_argument(
        "--index-magnetic-atoms",
        type=int,
        nargs="+",
        default=None,
        help="1-based indices of the magnetic sites (converted to the "
        "0-based Python convention internally)",
    )
    parser.add_argument("--Rcut", type=float, default=10.0)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument("--smearing", type=float, default=0.05)
    parser.add_argument(
        "--tangent-tol",
        type=float,
        default=None,
        help="explicit FR-032 tangent-block tolerance in eV; default: auto, "
        "derived as 2x the worst repeated-diagonal deviation and reported "
        "as tol_source in the gate report",
    )
    parser.add_argument(
        "--skip-anchor",
        action="store_true",
        help="disable the SOC-off collinear anchor gate (not recommended)",
    )
    args = parser.parse_args(argv)

    from TB2J.interfaces.abinit_nc_split_soc import gen_exchange_abinit_nc_split_soc

    result = gen_exchange_abinit_nc_split_soc(
        args.pao_hs,
        args.soc_kernel,
        output_path=args.output,
        Rcut=args.Rcut,
        nz=args.nz,
        smearing_eV=args.smearing,
        index_magnetic_atoms=(
            None
            if args.index_magnetic_atoms is None
            else [i - 1 for i in args.index_magnetic_atoms]
        ),
        tangent_tol_eV=args.tangent_tol,
        soc_off_anchor=not args.skip_anchor,
        wfk=args.wfk,
    )

    ok = True
    pairing = result["metadata"]["sidecar_pairing"]
    print("Sidecar pairing:")
    print(f"  pao_hs sha256: {pairing['pao_hs']['sha256']}")
    print(f"  wfk:           {pairing['wfk']['name']} ({pairing['wfk']['sha256']})")
    if "eigenvalues_max_dev_eV" in pairing:
        print(f"  eigenvalue dev: {pairing['eigenvalues_max_dev_eV']:.3e} eV")

    anchor = result["soc_off_anchor"]
    print(
        f"SOC-off anchor: passed={anchor.get('passed')} "
        f"(max rel Jiso dev {anchor.get('max_rel_Jiso_dev', float('nan')):.3e} "
        f"over {anchor.get('pairs_compared')} pairs)"
    )
    if anchor.get("passed") is False:
        ok = False

    tangent = result["tangent_projection_check"]
    if tangent is not None:
        print(
            "FR-032 tangent projection gate (projection-only, NOT a full "
            "tensor or DMI proof): "
            f"passed={tangent['passed']} over {tangent['pairs_compared']} pairs "
            f"(max |dT[:2,:2]|={tangent['max_transverse_dev_eV']:.3e} eV, "
            f"tol {tangent['tol_eV']:.1e} eV [{tangent.get('tol_source', 'n/a')}])"
        )
        if tangent["passed"] is False:
            ok = False

    print(f"Merged exchange: {Path(result['output_path']) / 'exchange.out'}")
    print(
        f"Provenance:      {Path(result['output_path']) / 'split_soc_provenance.json'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
