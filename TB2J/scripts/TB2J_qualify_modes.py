#!/usr/bin/env python3
"""TB2J_qualify_modes: one mode qualification report (Story-008).

Thin CLI over ``TB2J.magnon_phonon.qualification`` following the existing
argparse/JSON-result script conventions. It reads existing results only: a
TB2J magnetic-interaction result directory (magnon side) and, optionally, a
JSON record of the Story-007 ``PhononModes.to_dict()`` output (phonon side).
"""

import argparse
import json
import sys

import numpy as np

from TB2J.magnon.magnon3 import Magnon
from TB2J.magnon_phonon.qualification import qualification_report
from TB2J.versioninfo import print_license


class _PhononModesFromRecord:
    """Adapter from a Story-007 ``PhononModes.to_dict()`` JSON record."""

    def __init__(self, record: dict):
        self.q = np.asarray(record["q_reduced"], float)
        self.frequencies_thz = np.asarray(record["frequencies_thz"], float)
        self.frequencies_meV = np.asarray(record["frequencies_meV"], float)
        self.eigenvalues_eV_per_Ang2_per_amu = np.asarray(
            record["eigenvalues_eV_per_Ang2_per_amu"], float
        )
        self.eigenvectors = np.asarray(
            record["eigenvectors_real"], float
        ) + 1j * np.asarray(record["eigenvectors_imag"], float)
        self.force_constants = np.asarray(record["force_constants"], float)
        corrected = record.get("corrected_force_constants")
        self.corrected_force_constants = (
            None if corrected is None else np.asarray(corrected, float)
        )
        self.masses_amu = np.asarray(record["masses_amu"], float)
        self.atom_map = record["atom_map"]
        self.normalization = record["normalization"]
        self.diagnostics = record["diagnostics"]

    @property
    def instabilities(self):
        return list(self.diagnostics["instabilities"])


def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "TB2J_qualify_modes: qualify canonical magnon and phonon modes "
            "(stability, Goldstone/acoustic residuals, reciprocity/translation, "
            "degenerate-subspace determinism, structure identity)"
        )
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="TB2J_results",
        help="path to the TB2J magnetic interaction results (default: TB2J_results)",
    )
    parser.add_argument(
        "--kpoints",
        type=str,
        default="0,0,0",
        help=(
            "fractional reciprocal k-points, semicolon-separated triplets of "
            "comma-separated numbers, e.g. '0,0,0; 0.2,0,0'"
        ),
    )
    parser.add_argument(
        "--phonon-json",
        type=str,
        default=None,
        help=(
            "JSON record of the Story-007 PhononModes.to_dict() output; "
            "phonon diagnostics and cross-checks are reported only when given"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="magnon_phonon_qualification.json",
        help="output JSON report path (default: magnon_phonon_qualification.json)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of repeated phonon diagonalizations for determinism (default: 3)",
    )
    return parser


def parse_kpoints(text):
    kpoints = [
        [float(x) for x in triplet.replace(",", " ").split()]
        for triplet in text.split(";")
        if triplet.strip()
    ]
    for kpoint in kpoints:
        if len(kpoint) != 3:
            raise ValueError(f"each k-point must have three components: {kpoint}")
    return kpoints


def main(args=None):
    parser = create_parser()
    args = parser.parse_args(args)
    print_license()

    spin_model = Magnon.from_TB2J_results(path=args.fname)
    # load_from_io does not set the reference state; default to the
    # collinear z-frame used by the magnon band scripts.
    spin_model.set_reference(
        np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
    )
    phonon_modes = None
    if args.phonon_json:
        with open(args.phonon_json) as stream:
            phonon_modes = _PhononModesFromRecord(json.load(stream))

    report = qualification_report(
        spin_model,
        parse_kpoints(args.kpoints),
        phonon_modes,
        repeats=args.repeats,
    )
    with open(args.output, "w") as stream:
        json.dump(report, stream, indent=2)

    n_warnings = len(report["warnings"])
    print(f"Qualification report written to {args.output}")
    print(f"Warnings: {n_warnings}")
    for warning in report["warnings"]:
        print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
