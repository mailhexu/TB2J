"""Production bccFe spin-lattice convergence workflow (SCF, FC, FD, certificate).

Stages
------
scf          Generate and run independent SIESTA SCF cases (ref/plus/minus/fc)
             for every requested physical case (DFT k-mesh, MeshCutoff,
             displacement magnitude).
contour      Converge the contour integral (nz list) at fixed settings.
kmesh        Converge the exchange k-mesh at fixed nz.
displacement Adaptive central finite difference over displacement magnitudes.
dft          Converge the underlying DFT settings (k-mesh x MeshCutoff grid).
final        Recheck nz, exchange mesh and step at the final physical case;
             write production SpinIO only after these checks pass.

Certified observable
--------------------
The periodic image-summed dJ/dR at the pair radius ``--rcut`` computed from
the FC dHSdR vertex, certified against central finite differences from
independent plus/minus displaced SCFs at a fixed reference Fermi gauge.  The
local (home-cell) response is NOT certified by this workflow.

Run with --help.  All list-valued protocol parameters can be overridden on
the command line; the workflow fails (exit != 0) if any requested stage has
no plateau within the requested lists.  Every measured row is written to the
certificate JSON.  ``passed: true`` in the certificate means the FULL stage
set (contour, kmesh, displacement, dft) passed and final converged settings
were reached; a stage subset yields an explicitly labelled PARTIAL
certificate with ``passed: false`` regardless of the stages' own results.
The certified observable is the periodic image-summed dJ/dR, never the
local (home-cell) physical response.
The contour plateau selects the cheaper member of its agreeing pair; all
other plateaus select the last measured setting. Final checks require a
larger nz/mesh and a larger displacement from the supplied protocol lists.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.units import kB
from HamiltonIO.siesta.dhsdr import read_dHSdR
from sisl import get_sile

from TB2J.interfaces.siesta_spin_lattice import (
    gen_exchange_spin_lattice_siesta,
    load_siesta_spin_lattice_model,
)
from TB2J.spin_lattice_lcao import ExchangePert2Sl, LocalizedDisplacementDerivative

STAGES = ("scf", "contour", "kmesh", "displacement", "dft")

# Minimal complete two-atom cubic bccFe (a=2.848 Ang) FDF template with the
# production SCF settings baked in (DM/H/EDM tolerances, 300 K, FC options).
BASE_FDF = """SystemName\tbccFe2
SystemLabel\tsiesta

CDF.Compress\t9
CDF.Save\tTrue
MaxSCFIteration\t300
SCF.DM.Tolerance\t1e-8
SCF.EDM.Tolerance\t1e-8 eV
SCF.H.Tolerance\t1e-7 eV
SCF.Mixer.History\t16
SCF.Mixer.Method\tPulay
SCF.Mixer.Spom\tspinor
SCF.Mixer.Weight\t0.4
SaveHS\tTrue
SCFMustConverge\tTrue

Spin     \tCOLLINEAR
XC.functional\tGGA
XC.authors\tPBE

MeshCutoff\t{cutoff} Ry
PAO.EnergyShift\t0.1 eV

NumberOfSpecies\t1
NumberOfAtoms\t2
%block ChemicalSpecieslabel
    1 26 {pseudo_name}
%endblock ChemicalSpecieslabel

%block PAO.BasisSizes
    {pseudo_name}\tDZP
%endblock PAO.BasisSizes

LatticeConstant\t1.0 Ang
%block LatticeVectors
    2.848  0.0  0.0
    0.0  2.848  0.0
    0.0  0.0  2.848
%endblock LatticeVectors

AtomicCoordinatesFormat  Ang
%block AtomicCoordinatesAndAtomicSpecies
     {dx:.12f} 0.0 0.0 1
     1.424 1.424 1.424 1
%endblock AtomicCoordinatesAndAtomicSpecies

%block DM.InitSpin
    1 2.2
    2 2.2
%endblock DM.InitSpin

%block kgrid_Monkhorst_Pack
     {k}       0       0  0.0
     0       {k}       0  0.0
     0       0       {k}  0.0
%endblock kgrid_Monkhorst_Pack

ElectronicTemperature 300 K
DM.UseSaveDM false
"""

FC_FDF = """MD.TypeOfRun FC
FC.Save.dHS true
FC.First 1
FC.Last 1
FC.Displacement {h:g} Ang
FC.dHdR.Tolerance 0.0 Ry/Bohr
FC.dSdR.Tolerance 0.0 1/Bohr
"""

SCF_MDRUN = "MD.TypeOfRun CG\nMD.NumCGsteps 0\n"


def build_fdf(pseudo_name, k, cutoff, h, mode, temperature=300.0):
    dx = {"plus": h, "minus": -h}.get(mode, 0.0)
    fdf = BASE_FDF.format(pseudo_name=Path(pseudo_name).stem, cutoff=cutoff, k=k, dx=dx)
    fdf = fdf.replace(
        "ElectronicTemperature 300 K", f"ElectronicTemperature {temperature:g} K"
    )
    if mode == "fc":
        fdf += FC_FDF.format(h=h)
    else:
        fdf += SCF_MDRUN
    return fdf


def case_name(k, cutoff, h, mode):
    return f"k{k:g}-cut{cutoff:g}-h{h:g}-{mode}"


def case_dir(root, k, cutoff, h, mode):
    return root / case_name(k, cutoff, h, mode)


def _tol(text, keyword, value):
    m = re.search(rf"{re.escape(keyword)}\s+(\S+)", text)
    try:
        return m is not None and float(m.group(1)) == value
    except ValueError:
        return False


def provenance_ok(directory, k, cutoff, h, mode, pseudo=None, temperature=300.0):
    """Verify geometry, basis, pseudopotential, SCF settings and clean output."""
    fdf = directory / "siesta.fdf"
    if not fdf.exists():
        return "missing siesta.fdf"
    text = fdf.read_text()
    cutoff_match = re.search(r"MeshCutoff\s+(\S+)\s+Ry", text)
    temp_match = re.search(r"ElectronicTemperature\s+(\S+)\s+K", text)
    checks = [
        ("MeshCutoff", cutoff_match and float(cutoff_match.group(1)) == cutoff),
        (
            "kgrid",
            all(
                len(re.findall(pat, text, re.M)) == 1
                for pat in (
                    rf"^\s+{k}\s+0\s+0\s+0\.0$",
                    rf"^\s+0\s+{k}\s+0\s+0\.0$",
                    rf"^\s+0\s+0\s+{k}\s+0\.0$",
                )
            ),
        ),
        ("DM tol", _tol(text, "SCF.DM.Tolerance", 1e-8)),
        ("EDM tol", _tol(text, "SCF.EDM.Tolerance", 1e-8)),
        ("H tol", _tol(text, "SCF.H.Tolerance", 1e-7)),
        ("must converge", re.search(r"SCFMustConverge\s+True", text, re.I)),
        ("temperature", temp_match and float(temp_match.group(1)) == temperature),
        ("no saved DM", re.search(r"DM.UseSaveDM\s+false", text, re.I)),
        ("max SCF", re.search(r"MaxSCFIteration\s+300", text)),
        ("basis", re.search(r"PAO.BasisSizes[\s\S]*?\sDZP\b", text)),
        ("energy shift", re.search(r"PAO.EnergyShift\s+0\.1\s+eV", text)),
        ("PBE", re.search(r"XC.authors\s+PBE", text, re.I)),
        ("collinear", re.search(r"Spin\s+COLLINEAR", text, re.I)),
    ]
    if mode == "fc":
        checks += [
            ("FC.First", re.search(r"FC\.First\s+1\b", text)),
            ("FC.Last", re.search(r"FC\.Last\s+1\b", text)),
            ("FC displacement", re.search(rf"FC\.Displacement\s+{h:g}\s+Ang", text)),
            ("FC.Save.dHS", re.search(r"FC\.Save\.dHS\s+true", text, re.I)),
        ]
    for label, ok in checks:
        if not ok:
            return f"setting mismatch: {label}"
    dx = {"plus": h, "minus": -h}.get(mode, 0.0)
    geometry = get_sile(str(fdf)).read_geometry()
    if geometry.na != 2 or not np.allclose(
        geometry.lattice.cell, np.eye(3) * 2.848, atol=1e-10
    ):
        return "bccFe cell mismatch"
    if not np.allclose(geometry.xyz, [[dx, 0, 0], [1.424, 1.424, 1.424]], atol=1e-10):
        return "displaced geometry mismatch"
    if pseudo is not None:
        stored = directory / Path(pseudo).name
        if (
            not stored.is_file()
            or hashlib.sha256(stored.read_bytes()).digest()
            != hashlib.sha256(Path(pseudo).read_bytes()).digest()
        ):
            return "pseudopotential mismatch"
    expected = directory / ("siesta.dHSdR.nc" if mode == "fc" else "siesta.nc")
    if not expected.exists() or not (directory / "0_NORMAL_EXIT").exists():
        return "missing clean SIESTA output"
    out = directory / "siesta.out"
    if not out.exists():
        return "missing SCF convergence record"
    log = out.read_text()
    if (
        "SCF_NOT_CONV" in log
        or "SCF did not converge" in log
        or "SCF cycle converged" not in log
    ):
        return "unconverged SCF in existing output"
    return None


def check_reuse(args, cases):
    """Reuse-only mode: verify every needed case exists and matches settings."""
    reuse_root = (
        Path(args.output) if args.reuse_existing == "" else Path(args.reuse_existing)
    )
    rows = []
    for k, cutoff, h in cases:
        for mode in ("ref", "plus", "minus", "fc"):
            source = reuse_root / case_name(k, cutoff, h, mode)
            note = provenance_ok(
                source, k, cutoff, h, mode, args.pseudo, args.temperature
            )
            if note is not None:
                raise RuntimeError(
                    f"reuse-existing provenance failed for {source}: {note}"
                )
            for item in source.iterdir():
                if item.is_file():
                    link = args.output / source.name / item.name
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if not link.exists():
                        link.symlink_to(item.resolve())
            rows.append(
                {
                    "case": source.name,
                    "k": k,
                    "cutoff_Ry": cutoff,
                    "h_Ang": h,
                    "mode": mode,
                    "status": f"reused from {source}",
                }
            )
            print(f"REUSE {source}: provenance ok", flush=True)
    return rows


def run_siesta(binary, directory):
    expected = directory / (
        "siesta.dHSdR.nc" if directory.name.endswith("-fc") else "siesta.nc"
    )
    with (directory / "siesta.fdf").open("rb") as inp, (directory / "siesta.out").open(
        "wb"
    ) as out:
        cp = subprocess.run(
            [str(binary)],
            stdin=inp,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=directory,
        )
    text = (directory / "siesta.out").read_text()
    if cp.returncode or not expected.exists():
        raise RuntimeError(f"SIESTA failed in {directory}: returncode={cp.returncode}")
    if "SCF_NOT_CONV" in text or "SCF did not converge" in text:
        raise RuntimeError(f"SCF did not converge in {directory}")
    return "computed"


def ensure_cases(args, cases):
    """Prepare only requested physical cases; never relabel stale outputs."""
    rows = []
    pseudo_path = Path(args.pseudo).resolve()
    for k, cutoff, h in cases:
        for mode in ("ref", "plus", "minus", "fc"):
            directory = case_dir(args.output, k, cutoff, h, mode)
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "siesta.fdf").exists():
                note = provenance_ok(
                    directory, k, cutoff, h, mode, pseudo_path, args.temperature
                )
                if args.reuse_existing is None or note is not None:
                    raise RuntimeError(
                        f"Existing case {directory} cannot be overwritten: {note or 'explicit reuse required'}"
                    )
                status = "reused validated output"
            else:
                source = None
                if args.reuse_existing is not None:
                    reuse_root = (
                        args.output
                        if args.reuse_existing == ""
                        else Path(args.reuse_existing)
                    )
                    candidate = reuse_root / case_name(k, cutoff, h, mode)
                    if (
                        provenance_ok(
                            candidate, k, cutoff, h, mode, pseudo_path, args.temperature
                        )
                        is None
                    ):
                        source = candidate
                if source is not None:
                    for item in source.iterdir():
                        if item.is_file():
                            link = directory / item.name
                            if not link.exists():
                                link.symlink_to(item.resolve())
                    status = f"reused from {source}"
                else:
                    (directory / "siesta.fdf").write_text(
                        build_fdf(
                            pseudo_path.name, k, cutoff, h, mode, args.temperature
                        )
                    )
                    link = directory / pseudo_path.name
                    if not link.exists():
                        link.symlink_to(pseudo_path)
                    status = run_siesta(args.siesta, directory)
            rows.append(
                {
                    "case": directory.name,
                    "k": k,
                    "cutoff_Ry": cutoff,
                    "h_Ang": h,
                    "mode": mode,
                    "status": status,
                }
            )
            print(f"SCF {directory.name}: {status}", flush=True)
    return rows


def realspace_grid_metadata(directory):
    """Read the grid actually used; different requested cutoffs may alias."""
    log = (directory / "siesta.out").read_text()
    grid = re.search(r"InitMesh:\s*MESH\s*=\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", log)
    cutoff = re.search(
        r"Mesh cutoff \(required, used\)\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+Ry", log
    )
    if grid is None or cutoff is None:
        raise ValueError(f"Missing actual real-space mesh evidence in {directory}")
    return {
        "realspace_grid": [int(value) for value in grid.groups()],
        "effective_cutoff_Ry": float(cutoff.group(2)),
    }


def grid_converged(rows, atol, rtol):
    """A cutoff plateau must compare distinct actual real-space grids."""
    return (
        converged(rows, atol, rtol)
        and rows[-2]["realspace_grid"] != rows[-1]["realspace_grid"]
    )


def analyze_case(args, k, cutoff, h, mesh, nz, with_fd=True):
    """Periodic-response analysis of one physical case; returns row dict."""
    physical = (int(k), float(cutoff), float(h))
    if physical not in args.prepared_cases:
        if "scf" in args.stages:
            records = ensure_cases(args, [physical])
        elif args.reuse_existing is not None:
            records = check_reuse(args, [physical])
        else:
            records = []
            for mode in ("ref", "plus", "minus", "fc"):
                directory = case_dir(args.output, *physical, mode)
                note = provenance_ok(
                    directory, *physical, mode, args.pseudo, args.temperature
                )
                if note is not None:
                    raise RuntimeError(f"Unvalidated case {directory}: {note}")
                records.append(
                    {
                        "case": directory.name,
                        "k": k,
                        "cutoff_Ry": cutoff,
                        "h_Ang": h,
                        "mode": mode,
                        "status": "validated existing",
                    }
                )
        args.scf_records.extend(records)
        args.prepared_cases.add(physical)
    params = dict(
        magnetic_elements=["Fe"],
        kmesh=(mesh, mesh, mesh),
        emin=-12.0,
        emax=0.0,
        nz=nz,
        nproc=1,
        Rcut=args.rcut,
        smearing=args.temperature * kB,
        integration_method="CFR",
    )
    ref_fdf = case_dir(args.output, k, cutoff, h, "ref") / "siesta.fdf"
    ref, ef = load_siesta_spin_lattice_model(ref_fdf)

    # Dense meshes hold multi-GB eigensystems; an explicit TB2J_SL_CACHE_DIR
    # spills eigenvectors/overlap to disk-backed memmaps so successive
    # reference/plus/minus solves release memory between prepare phases.
    cache_dir = os.environ.get("TB2J_SL_CACHE_DIR")
    if cache_dir is not None:
        cache_dir = str(Path(cache_dir).resolve())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # ExchangeParams passes use_cache but not cache_path to TBGreen; its
        # documented TMPDIR fallback is the actual cache-location seam.
        os.environ["TMPDIR"] = cache_dir

    def run(tbmodel, disp):
        ex = ExchangePert2Sl(
            tbmodels=tbmodel,
            atoms=tbmodel.atoms,
            basis=tbmodel.orbs,
            efermi=0.0,
            use_cache=cache_dir is not None,
            **params,
        )
        ex.set_displacement(disp, J_only=disp is None, response="periodic")
        ex.calculate_all()
        return ex

    block = next(
        read_dHSdR(
            case_dir(args.output, k, cutoff, h, "fc") / "siesta.dHSdR.nc",
            displacements=[(1, 1)],
        )
    )
    disp = LocalizedDisplacementDerivative.from_siesta_dhsdr(
        block, atoms=ref.atoms, reference_fermi=ef
    )
    ex_ref = run(ref, disp)
    keys = sorted(ex_ref.exchange_Jdict)
    row = {
        "k": k,
        "cutoff_Ry": cutoff,
        "h_Ang": h,
        "mesh": mesh,
        "nz": nz,
        "reference_Ef_eV": ef,
        **realspace_grid_metadata(ref_fdf.parent),
        "bonds": [
            {
                "R": list(key[0]),
                "i": key[1],
                "j": key[2],
                "J_meV": float(1000 * ex_ref.exchange_Jdict[key]),
                "O_meV_A": float(1000 * ex_ref.dJdx[key]),
            }
            for key in keys
        ],
    }
    ex_ref.finalize()
    del ex_ref, ref, disp, block
    if with_fd:
        displaced = {}
        for mode in ("plus", "minus"):
            model, _ = load_siesta_spin_lattice_model(
                case_dir(args.output, k, cutoff, h, mode) / "siesta.fdf",
                reference_fermi=ef,
            )
            result = run(model, None)
            displaced[mode] = result.exchange_Jdict
            result.finalize()
            del result, model
        if set(displaced["plus"]) != set(keys) or set(displaced["minus"]) != set(keys):
            raise ValueError(
                "FD pair sets differ from reference; "
                "choose rcut away from a neighbour shell"
            )
        for bond, key in zip(row["bonds"], keys):
            fd = 1000.0 * (displaced["plus"][key] - displaced["minus"][key]) / (2 * h)
            bond["FD_meV_A"] = float(fd)
            bond["error_meV_A"] = float(bond["O_meV_A"] - fd)
        row["max_FD_error_meV_A"] = max(abs(b["error_meV_A"]) for b in row["bonds"])
        row["fd_passed"] = all(
            abs(b["error_meV_A"]) <= args.fd_atol + args.fd_rtol * abs(b["FD_meV_A"])
            for b in row["bonds"]
        )
    evidence = args.output / "measured_rows"
    evidence.mkdir(parents=True, exist_ok=True)
    name = f"k{k}-cut{cutoff:g}-h{h:g}-mesh{mesh}-nz{nz}-fd{int(with_fd)}.json"
    (evidence / name).write_text(json.dumps(row, indent=2) + "\n")
    return row


def converged(rows, atol, rtol, key="O_meV_A"):
    """True if the last two rows agree within atol + rtol * max signal."""
    if len(rows) < 2:
        return False
    delta = stage_delta(rows, key=key)
    scale = max(max(abs(b[key]) for b in r["bonds"]) for r in rows[-2:])
    return delta <= atol + rtol * scale


def stage_delta(rows, key="O_meV_A"):
    def indexed(row):
        return {(tuple(b["R"]), b["i"], b["j"]): b[key] for b in row["bonds"]}

    latest, previous = indexed(rows[-1]), indexed(rows[-2])
    if set(latest) != set(previous):
        raise ValueError("Convergence pair sets differ between settings")
    if not all(np.isfinite(value) for value in (*latest.values(), *previous.values())):
        raise ValueError("Nonfinite coupling cannot certify convergence")
    return max(abs(latest[k] - previous[k]) for k in latest)


def cross_validate_final(args, baseline):
    """Reject common-mode integration errors at the final physical setting."""
    probes = {}
    for name, field, choices, atol, rtol in (
        ("contour", "nz", args.nz_list, args.contour_atol, args.contour_rtol),
        ("kmesh", "mesh", args.kmesh_list, args.conv_atol, args.conv_rtol),
        (
            "displacement",
            "h_Ang",
            args.displacement_list,
            args.conv_atol,
            args.conv_rtol,
        ),
    ):
        larger = sorted(value for value in choices if value > baseline[field])
        if not larger:
            probes[name] = {
                "passed": False,
                "reason": f"Extend the {name} list beyond {baseline[field]} for a final-setting check",
            }
            continue
        settings = {
            key: baseline[key] for key in ("k", "cutoff_Ry", "h_Ang", "mesh", "nz")
        }
        settings[field] = larger[0]
        row = analyze_case(
            args,
            settings["k"],
            settings["cutoff_Ry"],
            settings["h_Ang"],
            settings["mesh"],
            settings["nz"],
            with_fd=name == "displacement",
        )
        pair = [baseline, row]
        ok = converged(pair, atol, rtol)
        if name == "displacement":
            ok &= baseline["fd_passed"] and row["fd_passed"]
            ok &= converged(pair, atol, rtol, key="FD_meV_A")
        probes[name] = {
            "rows": pair,
            "delta_meV_A": stage_delta(pair),
            "passed": bool(ok),
        }
        print(
            f"final {name} cross-check: delta={stage_delta(pair):.6g} meV/Ang, passed={ok}",
            flush=True,
        )
    return {
        "probes": probes,
        "passed": all(probe["passed"] for probe in probes.values()),
    }


def write_certificate(args, certificate):
    path = args.output / "convergence_certificate.json"
    certificate["certified_quantity"] = (
        "periodic image-summed dJ/dR within pair radius "
        f"{args.rcut} Ang; the local (home-cell) response is NOT certified"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in certificate.items() if k not in ("stages",)}, indent=2
        )
    )
    print(f"certificate written to {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Certified quantity: PERIODIC image-summed dJ/dR (pair radius "
        "--rcut). The local home-cell response is not certified here.",
    )
    p.add_argument("--siesta", type=Path, required=True, help="SIESTA binary")
    p.add_argument(
        "--pseudo",
        type=Path,
        required=True,
        help="Fe PSML pseudopotential (e.g. Fe.1.psml)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="runtime output directory (SCF cases, results, certificate)",
    )
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument(
        "--reuse-existing",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help="reuse an existing artifact tree laid out like this "
        "output dir (provenance-checked); bare flag reuses "
        "--output itself",
    )
    p.add_argument(
        "--rcut", type=float, default=3.0, help="exchange pair radius in Angstrom"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="electronic temperature in K (smearing)",
    )
    p.add_argument(
        "--contour-base",
        nargs=3,
        type=float,
        default=[5, 200, 0.005],
        metavar=("K", "CUTOFF_RY", "H_ANG"),
        help="physical case for the contour stage",
    )
    p.add_argument(
        "--kmesh-base",
        nargs=3,
        type=float,
        default=[5, 200, 0.005],
        metavar=("K", "CUTOFF_RY", "H_ANG"),
    )
    p.add_argument("--nz-list", nargs="+", type=int, default=[320, 640])
    p.add_argument("--contour-mesh", type=int, default=7)
    p.add_argument(
        "--kmesh-list", nargs="+", type=int, default=[5, 7, 9, 11, 13, 15, 17, 19, 21]
    )
    p.add_argument(
        "--displacement-list", nargs="+", type=float, default=[0.01, 0.005, 0.0025]
    )
    p.add_argument(
        "--displacement-k",
        type=int,
        default=None,
        help="Exchange k-mesh for the displacement stage "
        "(default: kmesh-stage plateau or 13)",
    )
    p.add_argument("--dft-ks", nargs="+", type=int, default=[5, 7, 9, 11, 13, 15])
    p.add_argument("--dft-cutoffs", nargs="+", type=float, default=[200, 300, 400])
    p.add_argument(
        "--displacement-nz",
        type=int,
        default=None,
        help="nz for the displacement stage " "(default: contour-stage plateau)",
    )
    p.add_argument(
        "--fd-atol", type=float, default=0.1, help="FD acceptance, meV/Angstrom"
    )
    p.add_argument("--fd-rtol", type=float, default=0.01)
    p.add_argument(
        "--conv-atol", type=float, default=0.1, help="convergence plateau, meV/Angstrom"
    )
    p.add_argument("--conv-rtol", type=float, default=0.01)
    p.add_argument(
        "--contour-atol",
        type=float,
        default=0.01,
        help="contour plateau (tighter), meV/Angstrom",
    )
    p.add_argument("--contour-rtol", type=float, default=0.001)
    p.add_argument(
        "--no-final-output",
        action="store_true",
        help="skip the final production SpinIO write",
    )
    a = p.parse_args()
    a.output = a.output.resolve()
    a.siesta = a.siesta.resolve()

    certificate = {
        "inputs": {
            "siesta": str(a.siesta.resolve()),
            "pseudo": str(Path(a.pseudo).resolve()),
            "output": str(a.output),
            "rcut_A": a.rcut,
            "temperature_K": a.temperature,
            "stages": a.stages,
        },
        "tolerances": {
            "fd_abs_meV_A": a.fd_atol,
            "fd_rel": a.fd_rtol,
            "conv_abs_meV_A": a.conv_atol,
            "conv_rel": a.conv_rtol,
            "contour_abs_meV_A": a.contour_atol,
            "contour_rel": a.contour_rtol,
        },
        "scf_cases": [],
        "stages": {},
        "final_settings": None,
        "passed": True,
    }
    a.prepared_cases = set()
    a.scf_records = certificate["scf_cases"]
    plateau = {}

    full_needed = (
        {tuple(a.contour_base), tuple(a.kmesh_base)}
        | {(a.dft_ks[0], a.dft_cutoffs[0], h) for h in a.displacement_list}
        | {(k, c, 0.005) for k in a.dft_ks for c in a.dft_cutoffs}
    )
    needed = set()
    if {"contour", "kmesh"} & set(a.stages):
        needed |= {tuple(a.contour_base), tuple(a.kmesh_base)}
    if "displacement" in a.stages:
        needed |= {(a.dft_ks[0], a.dft_cutoffs[0], h) for h in a.displacement_list}
    if "dft" in a.stages:
        needed |= {(k, c, 0.005) for k in a.dft_ks for c in a.dft_cutoffs}
    needed = (
        {(int(k), float(c), float(h)) for k, c, h in needed}
        if "scf" not in a.stages
        else {(int(k), float(c), float(h)) for k, c, h in full_needed}
    )
    if a.stages == ["scf"]:
        certificate["scf_cases"].extend(ensure_cases(a, sorted(needed)))

    if "contour" in a.stages:
        k, cutoff, h = (int(a.contour_base[0]), a.contour_base[1], a.contour_base[2])
        rows = []
        for nz in a.nz_list:
            rows.append(
                analyze_case(a, k, cutoff, h, a.contour_mesh, nz, with_fd=False)
            )
            print(
                f"contour nz={nz}: max|O| = "
                f"{max(abs(b['O_meV_A']) for b in rows[-1]['bonds']):.6f} "
                "meV/Ang",
                flush=True,
            )
            if converged(rows, a.contour_atol, a.contour_rtol):
                break
        ok = converged(rows, a.contour_atol, a.contour_rtol)
        plateau["nz"] = rows[-2]["nz"] if ok else None
        certificate["stages"]["contour"] = {
            "rows": rows,
            "delta_meV_A": stage_delta(rows) if len(rows) > 1 else None,
            "plateau_nz": plateau["nz"],
            "passed": ok,
        }
        certificate["passed"] &= ok
        if not ok:
            print("contour stage: NO PLATEAU within requested nz list", file=sys.stderr)

    if "kmesh" in a.stages:
        k, cutoff, h = (int(a.kmesh_base[0]), a.kmesh_base[1], a.kmesh_base[2])
        nz = plateau.get("nz") or max(a.nz_list)
        rows = []
        for mesh in a.kmesh_list:
            rows.append(analyze_case(a, k, cutoff, h, mesh, nz, with_fd=False))
            print(
                f"kmesh {mesh}^3: max|O| = "
                f"{max(abs(b['O_meV_A']) for b in rows[-1]['bonds']):.6f} "
                "meV/Ang",
                flush=True,
            )
            if converged(rows, a.conv_atol, a.conv_rtol):
                break
        ok = converged(rows, a.conv_atol, a.conv_rtol)
        plateau["mesh"] = a.kmesh_list[len(rows) - 1] if ok else None
        certificate["stages"]["kmesh"] = {
            "rows": rows,
            "delta_meV_A": stage_delta(rows) if len(rows) > 1 else None,
            "plateau_mesh": plateau["mesh"],
            "passed": ok,
        }
        certificate["passed"] &= ok
        if not ok:
            print("kmesh stage: NO PLATEAU within requested list", file=sys.stderr)

    if "displacement" in a.stages:
        mesh = a.displacement_k or plateau.get("mesh") or 13
        nz = a.displacement_nz or plateau.get("nz") or max(a.nz_list)
        k, cutoff = a.dft_ks[0], a.dft_cutoffs[0]
        rows = []
        for h in a.displacement_list:
            rows.append(analyze_case(a, k, cutoff, h, mesh, nz, with_fd=True))
            r = rows[-1]
            print(
                f"displacement h={h}: max FD error = "
                f"{r['max_FD_error_meV_A']:.6f} meV/Ang, "
                f"fd_passed={r['fd_passed']}",
                flush=True,
            )
            if (
                len(rows) >= 2
                and all(x["fd_passed"] for x in rows[-2:])
                and converged(rows, a.conv_atol, a.conv_rtol, key="FD_meV_A")
            ):
                break
        ok = len(rows) >= 2 and all(r["fd_passed"] for r in rows[-2:])
        ok &= converged(rows, a.conv_atol, a.conv_rtol, key="FD_meV_A")
        plateau["h"] = a.displacement_list[len(rows) - 1] if ok else None
        certificate["stages"]["displacement"] = {
            "k": k,
            "cutoff_Ry": cutoff,
            "mesh": mesh,
            "nz": nz,
            "rows": rows,
            "delta_meV_A": stage_delta(rows, "FD_meV_A") if len(rows) > 1 else None,
            "plateau_h_Ang": plateau["h"],
            "passed": ok,
        }
        certificate["passed"] &= ok
        if not ok:
            print(
                "displacement stage: no FD-accepted plateau within " "requested list",
                file=sys.stderr,
            )

    if "dft" in a.stages:
        mesh = plateau.get("mesh") or 13
        nz = plateau.get("nz") or max(a.nz_list)
        h = plateau.get("h") or 0.005
        rows = []
        series_ok = True
        for cutoff in a.dft_cutoffs:
            cut_rows = []
            for k in a.dft_ks:
                cut_rows.append(analyze_case(a, k, cutoff, h, mesh, nz, with_fd=True))
                r = cut_rows[-1]
                print(
                    f"dft k={k} cutoff={cutoff}: max FD error = "
                    f"{r['max_FD_error_meV_A']:.6f} meV/Ang, "
                    f"fd_passed={r['fd_passed']}",
                    flush=True,
                )
                if (
                    len(cut_rows) >= 2
                    and all(x["fd_passed"] for x in cut_rows[-2:])
                    and converged(cut_rows, a.conv_atol, a.conv_rtol)
                ):
                    break
            series_ok &= converged(cut_rows, a.conv_atol, a.conv_rtol)
            series_ok &= len(cut_rows) >= 2 and all(
                r["fd_passed"] for r in cut_rows[-2:]
            )
            rows.extend(cut_rows)
        # cutoff convergence: compare the last (plateau) k of each series
        tails = []
        for cutoff in a.dft_cutoffs:
            series = [r for r in rows if r["cutoff_Ry"] == cutoff]
            kmax = max(r["k"] for r in series)
            tails.append(next(r for r in series if r["k"] == kmax))
        cut_ok = grid_converged(tails, a.conv_atol, a.conv_rtol)
        ok = series_ok and cut_ok
        plateau["dft"] = (tails[-1]["k"], tails[-1]["cutoff_Ry"]) if ok else None
        certificate["stages"]["dft"] = {
            "mesh": mesh,
            "nz": nz,
            "h_Ang": h,
            "rows": rows,
            "plateau": {"k": plateau["dft"][0], "cutoff_Ry": plateau["dft"][1]}
            if ok
            else None,
            "passed": ok,
        }
        certificate["passed"] &= ok
        if not ok:
            print(
                "dft stage: NO PLATEAU within requested k/cutoff lists", file=sys.stderr
            )

    final = {
        "mesh": plateau.get("mesh"),
        "nz": plateau.get("nz"),
        "h_Ang": plateau.get("h"),
        "dft": plateau.get("dft"),
    }
    certificate["final_settings"] = final
    if certificate["passed"] and all(final.values()):
        cross_check = cross_validate_final(a, tails[-1])
        certificate["stages"]["final_cross_check"] = cross_check
        certificate["passed"] &= cross_check["passed"]
    required = {"contour", "kmesh", "displacement", "dft"}
    scope_complete = required.issubset(a.stages)
    certificate["scope_complete"] = scope_complete
    if not scope_complete:
        certificate["certificate_scope"] = "PARTIAL: stages " + ", ".join(
            a.stages
        ) + "; overall pass " "requires stages " + ", ".join(sorted(required))
    certificate["passed"] &= scope_complete and all(final.values())
    if not a.no_final_output and certificate["passed"]:
        k, cutoff = final["dft"]
        sio = gen_exchange_spin_lattice_siesta(
            case_dir(a.output, k, cutoff, final["h_Ang"], "ref") / "siesta.fdf",
            case_dir(a.output, k, cutoff, final["h_Ang"], "fc") / "siesta.dHSdR.nc",
            atom=1,
            axis=1,
            response="periodic",
            output_path=a.output / "spin_lattice_results",
            magnetic_elements=["Fe"],
            kmesh=(final["mesh"],) * 3,
            emin=-12.0,
            emax=0.0,
            nz=final["nz"],
            np=1,
            Rcut=a.rcut,
            smearing=a.temperature * kB,
            integration_method="CFR",
        )
        certificate["final_settings"]["spinio_output"] = str(
            a.output / "spin_lattice_results"
        )
        print("final SpinIO written:", sorted(sio.exchange_Jdict), flush=True)

    requested_passed = all(stage["passed"] for stage in certificate["stages"].values())
    certificate["requested_stages_passed"] = requested_passed
    write_certificate(a, certificate)
    if not requested_passed or (scope_complete and not certificate["passed"]):
        raise SystemExit(
            "spin-lattice convergence certificate FAILED; see convergence_certificate.json rows"
        )


if __name__ == "__main__":
    main()
