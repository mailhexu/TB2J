"""Compute a SIESTA FC mode, optionally certify it against independent SCFs.

Run with --help. Pickle values are eV and eV/Angstrom; this report is meV.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from TB2J.interfaces.siesta_spin_lattice import (
    gen_exchange_spin_lattice_siesta,
    load_siesta_spin_lattice_model,
)
from TB2J.spin_lattice_lcao import gen_exchange_spin_lattice_lcao


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fdf", type=Path, required=True, help="Undisplaced SCF input")
    p.add_argument("--dhsdr", type=Path, required=True)
    p.add_argument("--atom", type=int, default=1, help="1-based displaced atom")
    p.add_argument("--axis", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--elements", nargs="+", default=["Fe"])
    p.add_argument("--kmesh", nargs=3, type=int, default=[9, 9, 9])
    p.add_argument("--nz", type=int, default=320)
    p.add_argument("--integration-method", choices=("CFR", "legendre"), default="CFR")
    p.add_argument(
        "--smearing", type=float, default=0.0258519997864, help="kBT in eV (CFR only)"
    )
    p.add_argument(
        "--rcut", type=float, default=3.0, help="Exchange pair radius in Angstrom"
    )
    p.add_argument("--output", type=Path, default=Path("spin_lattice_results"))
    p.add_argument("--plus-fdf", type=Path)
    p.add_argument("--minus-fdf", type=Path)
    p.add_argument("--delta", type=float, help="Positive half displacement in Angstrom")
    p.add_argument("--fd-rtol", type=float, default=0.01)
    p.add_argument(
        "--fd-atol", type=float, default=0.1, help="Absolute tolerance in meV/Angstrom"
    )
    a = p.parse_args()
    if any(v is not None for v in (a.plus_fdf, a.minus_fdf, a.delta)):
        if a.plus_fdf is None or a.minus_fdf is None or a.delta is None or a.delta <= 0:
            p.error("FD requires --plus-fdf, --minus-fdf and positive --delta")
    params = dict(
        magnetic_elements=a.elements,
        kmesh=tuple(a.kmesh),
        nz=a.nz,
        emin=-12.0,
        emax=0.0,
        Rcut=a.rcut,
        smearing=a.smearing,
        integration_method=a.integration_method,
    )
    sio = gen_exchange_spin_lattice_siesta(
        a.fdf,
        a.dhsdr,
        atom=a.atom,
        axis=a.axis,
        output_path=a.output / "analytic",
        **params,
    )
    ref, ef = load_siesta_spin_lattice_model(a.fdf)
    results = {
        "response": "periodic image sum",
        "reference_Ef_eV": ef,
        "kmesh": a.kmesh,
        "nz": a.nz,
        "integration_method": a.integration_method,
        "smearing_eV": a.smearing,
        "pair_radius_A": a.rcut,
        "derivative_file_sha256": hashlib.sha256(a.dhsdr.read_bytes()).hexdigest(),
        "units": {"J": "meV", "O": "meV/Angstrom"},
        "bonds": [],
    }
    fd = None
    if a.plus_fdf is not None:
        models = [
            load_siesta_spin_lattice_model(f, reference_fermi=ef)[0]
            for f in (a.plus_fdf, a.minus_fdf)
        ]
        target = np.zeros_like(ref.atoms.positions)
        target[a.atom - 1, a.axis - 1] = a.delta
        for model, sign in zip(models, (1, -1)):
            np.testing.assert_allclose(model.atoms.cell, ref.atoms.cell, atol=1e-10)
            np.testing.assert_allclose(
                model.atoms.positions - ref.atoms.positions, sign * target, atol=1e-8
            )
        j = []
        for model, name in zip(models, ("plus", "minus")):
            value = gen_exchange_spin_lattice_lcao(
                tbmodel=model,
                atoms=model.atoms,
                disp=None,
                basis=model.orbs,
                efermi=0.0,
                Jonly=True,
                response="periodic",
                output_path=a.output / name,
                **params,
            )
            j.append(value.exchange_Jdict)
        if set(j[0]) != set(j[1]) or set(j[0]) != set(sio.dJdx):
            raise ValueError(
                "FD pair sets differ; choose a radius away from a neighbour shell"
            )
        fd = {k: 1000 * (j[0][k] - j[1][k]) / (2 * a.delta) for k in j[0]}
        results["delta_A"] = a.delta
    passed = True
    for key in sorted(sio.exchange_Jdict):
        R, i, j = key
        row = {
            "R": list(R),
            "i": i,
            "j": j,
            "J": float(1000 * sio.exchange_Jdict[key]),
            "O": float(1000 * sio.dJdx[key]),
        }
        if fd is not None:
            error = row["O"] - fd[key]
            row.update(FD=float(fd[key]), error=float(error))
            passed &= abs(error) <= a.fd_atol + a.fd_rtol * abs(fd[key])
        results["bonds"].append(row)
    if fd is not None:
        results["passed"] = bool(passed)
        results["max_error_meV_A"] = max(abs(b["error"]) for b in results["bonds"])
        results["tolerances"] = {"relative": a.fd_rtol, "absolute_meV_A": a.fd_atol}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "bonds"}, indent=2))
    if not passed:
        raise SystemExit("Spin-lattice finite-difference comparison failed")


if __name__ == "__main__":
    main()
