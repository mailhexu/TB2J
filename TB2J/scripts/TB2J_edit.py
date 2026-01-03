#!/usr/bin/env python
"""
Command-line interface for TB2J_edit.

Usage:
    TB2J_edit load -i INPUT.pickle -o OUTPUT_DIR
    TB2J_edit set-anisotropy -i INPUT.pickle -s Sm 5.0 -d "0 0 1" -o OUTPUT_DIR
    TB2J_edit set-anisotropy -i INPUT.pickle -s Sm 5.0 -s Fe 1.0 -o OUTPUT_DIR
    TB2J_edit toggle-dmi -i INPUT.pickle --disable -o OUTPUT_DIR
    TB2J_edit toggle-jani -i INPUT.pickle --disable -o OUTPUT_DIR
    TB2J_edit symmetrize -i INPUT.pickle -S STRUCTURE.cif -o OUTPUT_DIR
"""

import argparse
import sys

import numpy as np


def cmd_load(args):
    """Load TB2J results and save to output directory."""
    from TB2J.io_exchange.edit import load, save

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)
    print(f"  Loaded {len(spinio.atoms)} atoms")
    print(f"  Has DMI: {spinio.has_dmi}")
    print(f"  Has Jani: {spinio.has_bilinear}")

    print(f"Saving to: {args.output}")
    save(spinio, args.output)
    print("  Done!")


def cmd_set_anisotropy(args):
    """Set single ion anisotropy for one or more species."""
    from TB2J.io_exchange.edit import load, save, set_anisotropy

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)

    # Parse species specifications
    # args.species is a list of lists: [['Sm', '5.0'], ['Fe', '1.0']]
    # or with direction: [['Sm', '5.0', '0,0,1'], ['Fe', '1.0']]
    specs = []
    default_dir = None
    if args.dir:
        default_dir = [float(x) for x in args.dir.split()]

    for spec_list in args.species:
        if len(spec_list) < 2:
            print(
                f"Error: Invalid specification {spec_list}. Expected: species k1 [dir]"
            )
            sys.exit(1)

        species = spec_list[0]
        k1_val = float(spec_list[1])
        k1_eV = k1_val * 1e-3 if args.mev else k1_val

        # Check if direction vector is provided
        k1dir = default_dir
        if len(spec_list) >= 3:
            # Third element is the direction vector
            dir_str = spec_list[2].replace(",", " ")
            k1dir = [float(x) for x in dir_str.split()]

        specs.append((species, k1_eV, k1dir))

    print(f"Setting anisotropy for {len(specs)} species:")
    for species, k1, k1dir in specs:
        unit = "meV" if args.mev else "eV"
        k1_display = k1 * 1000 if args.mev else k1
        print(f"  {species}: k1 = {k1_display:.4f} {unit}", end="")
        if k1dir is not None:
            print(f", k1dir = {k1dir}")
        else:
            print()

    for species, k1, k1dir in specs:
        set_anisotropy(spinio, species=species, k1=k1, k1dir=k1dir)

    print(f"Saving to: {args.output}")
    save(spinio, args.output)
    print("  Done!")


def cmd_toggle_dmi(args):
    """Toggle DMI on/off."""
    from TB2J.io_exchange.edit import load, save, toggle_DMI

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)

    if args.disable:
        enabled = False
        action = "Disabling"
    elif args.enable:
        enabled = True
        action = "Enabling"
    else:
        enabled = None
        action = "Toggling"

    print(f"{action} DMI...")
    toggle_DMI(spinio, enabled=enabled)
    print(f"  DMI is now: {'enabled' if spinio.has_dmi else 'disabled'}")

    print(f"Saving to: {args.output}")
    save(spinio, args.output)
    print("  Done!")


def cmd_toggle_jani(args):
    """Toggle anisotropic exchange on/off."""
    from TB2J.io_exchange.edit import load, save, toggle_Jani

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)

    if args.disable:
        enabled = False
        action = "Disabling"
    elif args.enable:
        enabled = True
        action = "Enabling"
    else:
        enabled = None
        action = "Toggling"

    print(f"{action} anisotropic exchange...")
    toggle_Jani(spinio, enabled=enabled)
    print(f"  Jani is now: {'enabled' if spinio.has_bilinear else 'disabled'}")

    print(f"Saving to: {args.output}")
    save(spinio, args.output)
    print("  Done!")


def cmd_symmetrize(args):
    """Symmetrize exchange using a reference structure."""
    from ase.io import read

    from TB2J.io_exchange.edit import load, save, symmetrize_exchange

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)

    print(f"Loading reference structure from: {args.structure}")
    atoms_ref = read(args.structure)
    print(f"  Loaded {len(atoms_ref)} atoms")

    print(f"Symmetrizing exchange (symprec={args.symprec})...")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        symmetrize_exchange(spinio, atoms=atoms_ref, symprec=args.symprec)
    print("  Done!")

    print(f"Saving to: {args.output}")
    save(spinio, args.output)
    print("  Done!")


def cmd_info(args):
    """Show information about TB2J results."""
    from TB2J.io_exchange.edit import load

    print(f"Loading TB2J results from: {args.input}")
    spinio = load(args.input)

    print("\n" + "=" * 60)
    print("TB2J Results Information")
    print("=" * 60)

    print(f"\nAtoms: {len(spinio.atoms)}")
    print(f"Chemical formula: {spinio.atoms.get_chemical_formula()}")

    # Count atoms by species
    symbols = spinio.atoms.get_chemical_symbols()
    from collections import Counter

    counts = Counter(symbols)
    print(
        f"Composition: {', '.join(f'{sym}: {count}' for sym, count in sorted(counts.items()))}"
    )

    # Magnetic atoms
    mag_atoms = [i for i, idx in enumerate(spinio.index_spin) if idx >= 0]
    print(f"Magnetic atoms: {len(mag_atoms)}")

    # Exchange info
    print("\nExchange parameters:")
    print(
        f"  Isotropic exchange: {len(spinio.exchange_Jdict) if spinio.exchange_Jdict else 0} pairs"
    )
    print(f"  DMI: {'enabled' if spinio.has_dmi else 'disabled'}")
    print(f"  Anisotropic exchange: {'enabled' if spinio.has_bilinear else 'disabled'}")

    # Anisotropy
    if spinio.k1:
        print("\nSingle ion anisotropy:")
        for i, (sym, idx) in enumerate(zip(symbols, spinio.index_spin)):
            if idx >= 0 and idx < len(spinio.k1):
                if spinio.k1[idx] != 0:
                    print(f"  {sym}{i}: k1={spinio.k1[idx]:.4f} eV")
    else:
        print("\nSingle ion anisotropy: None")

    # Cell info
    print("\nCell parameters:")
    cell = spinio.atoms.get_cell()
    a, b, c = [np.linalg.norm(v) for v in cell]
    print(f"  a = {a:.4f} Å")
    print(f"  b = {b:.4f} Å")
    print(f"  c = {c:.4f} Å")

    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="TB2J_edit",
        description="Command-line interface for modifying TB2J results",
    )
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # load command
    parser_load = subparsers.add_parser("load", help="Load and save TB2J results")
    parser_load.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_load.add_argument(
        "-o", "--output", default="output", help="Output directory"
    )
    parser_load.set_defaults(func=cmd_load)

    # set-anisotropy command
    parser_aniso = subparsers.add_parser(
        "set-anisotropy", help="Set single ion anisotropy", aliases=["set-aniso"]
    )
    parser_aniso.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_aniso.add_argument("-o", "--output", required=True, help="Output directory")
    parser_aniso.add_argument(
        "-s",
        "--species",
        action="append",
        nargs="+",
        required=True,
        help="Species and k1 value: -s species k1 (can be used multiple times)",
    )
    parser_aniso.add_argument("-d", "--dir", help='Default direction vector as "x y z"')
    parser_aniso.add_argument(
        "-m",
        "--mev",
        action="store_true",
        help="Interpret k1 values in meV (default: eV)",
    )
    parser_aniso.set_defaults(func=cmd_set_anisotropy)

    # toggle-dmi command
    parser_dmi = subparsers.add_parser(
        "toggle-dmi", help="Enable/disable DMI", aliases=["dmi"]
    )
    parser_dmi.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_dmi.add_argument("-o", "--output", required=True, help="Output directory")
    parser_dmi.add_argument("-e", "--enable", action="store_true", help="Enable DMI")
    parser_dmi.add_argument("-d", "--disable", action="store_true", help="Disable DMI")
    parser_dmi.set_defaults(func=cmd_toggle_dmi)

    # toggle-jani command
    parser_jani = subparsers.add_parser(
        "toggle-jani", help="Enable/disable anisotropic exchange", aliases=["jani"]
    )
    parser_jani.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_jani.add_argument("-o", "--output", required=True, help="Output directory")
    parser_jani.add_argument(
        "-e", "--enable", action="store_true", help="Enable anisotropic exchange"
    )
    parser_jani.add_argument(
        "-d", "--disable", action="store_true", help="Disable anisotropic exchange"
    )
    parser_jani.set_defaults(func=cmd_toggle_jani)

    # symmetrize command
    parser_symm = subparsers.add_parser(
        "symmetrize", help="Symmetrize exchange parameters", aliases=["symm"]
    )
    parser_symm.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_symm.add_argument("-o", "--output", required=True, help="Output directory")
    parser_symm.add_argument(
        "-S",
        "--structure",
        required=True,
        help="Reference structure file (CIF, VASP, etc.)",
    )
    parser_symm.add_argument(
        "-p",
        "--symprec",
        type=float,
        default=1e-3,
        help="Symmetry precision in Angstrom (default: 1e-3)",
    )
    parser_symm.set_defaults(func=cmd_symmetrize)

    # info command
    parser_info = subparsers.add_parser(
        "info", help="Show information about TB2J results"
    )
    parser_info.add_argument("-i", "--input", required=True, help="Input pickle file")
    parser_info.set_defaults(func=cmd_info)

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if args.func is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
