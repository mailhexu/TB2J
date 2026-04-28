#!/usr/bin/env python3
"""
dmft2J: Calculate exchange parameters using DMFT self-energies.

Supports multiple DMFT output formats:
- w2dynamics: HDF5 format from w2dynamics code
- siginp: Text-based sig.inp format (TRIQS, EDMFTF, custom solvers)
"""

import argparse
import logging
import sys

from TB2J.exchange_params import add_exchange_args_to_parser
from TB2J.interfaces.dmft import DMFTManager
from TB2J.versioninfo import print_license

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_dmft2J():
    print_license()
    parser = argparse.ArgumentParser(
        description=(
            "dmft2J: Calculate exchange parameters using DMFT self-energies. "
            "Supports w2dynamics (HDF5) and sig.inp (text) formats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect parser from file extension
  dmft2J --dmft_file=sig.inp --path=./data --prefix=wannier90
  
  # Explicitly specify parser type
  dmft2J --dmft_file=self_energy.dat --parser-type=siginp --path=./data
  
  # With double-counting correction
  dmft2J --dmft_file=sig.inp --total-sigma --path=./data
  
  dmft2J --dmft_file=sig.inp --spin-channels=+1,-1,-1,+1 --magnetic-elements=Mn --path=./LaMnO3
""",
    )

    # Input file options
    parser.add_argument(
        "--path", help="Path to Wannier90 files (default: ./)", default="./", type=str
    )
    parser.add_argument(
        "--posfile",
        help="Name of the position file (default: POSCAR)",
        default="POSCAR",
        type=str,
    )
    parser.add_argument(
        "--prefix",
        help="Prefix for Wannier90 files (default: wannier90)",
        default="wannier90",
        type=str,
    )

    # DMFT-specific options
    parser.add_argument(
        "--dmft_file",
        help=(
            "Path to DMFT self-energy file. "
            "Format auto-detected from extension (.h5/.hdf5=w2dynamics, .inp=siginp)"
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "--parser-type",
        help=(
            "DMFT parser type (default: auto-detect from file extension). "
            "Options: auto, w2dynamics, siginp"
        ),
        default="auto",
        type=str,
        choices=["auto", "w2dynamics", "siginp"],
    )
    parser.add_argument(
        "--dmft-params",
        help="Path to dmft_params.dat file (for siginp parser, auto-detected if not specified)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--mu-file",
        help="Path to chemical potential file (for siginp parser, auto-detected if not specified)",
        default=None,
        type=str,
    )

    # DMFT calculation options
    parser.add_argument(
        "--nspin",
        help="Number of spins in DMFT calculation (1=collinear, 2=spin-polarized, default: 2)",
        default=2,
        type=int,
        choices=[1, 2],
    )
    parser.add_argument(
        "--total-sigma",
        action="store_true",
        help="Use total self-energy (add Σ(∞) to Σ(iω) - Σ(∞)). Default: use as-is",
    )
    parser.add_argument(
        "--static-sigma",
        action="store_true",
        help=(
            "Use only the static part of self-energy (Σ(∞) or Σ(∞)-Vdc). "
            "This enables using CFR contour instead of Matsubara. "
            "Useful for quick calculations or when frequency dependence is negligible."
        ),
    )
    parser.add_argument(
        "--mode",
        default="dmft",
        choices=["dmft", "static-hamiltonian"],
        help=(
            "Calculation mode. "
            "'dmft': full DMFT Green's function (default). "
            "'static-hamiltonian': fold Σ(∞)−Vdc into Wannier H(R=0) and run "
            "standard collinear TB2J exchange (requires --static-sigma or implies it)."
        ),
    )
    parser.add_argument(
        "--spin-channels",
        help=(
            "Comma-separated list of spin channels for each correlated atom. "
            "Use +1 for spin-up, -1 for spin-down. "
            "Example: '+1,-1,-1,+1' for AFM LaMnO3 (↑↓ ↓↑). "
            "If not specified, uses spin channels from dmft_params.dat"
        ),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--magnetic-elements",
        help="Comma-separated list of magnetic elements (e.g., 'Mn,Fe'). Auto-detected if not specified",
        default=None,
        type=str,
    )

    # Add standard exchange arguments
    parser = add_exchange_args_to_parser(parser)

    args = parser.parse_args()

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)

    # Parse magnetic elements
    magnetic_elements = None
    if args.magnetic_elements:
        magnetic_elements = [elem.strip() for elem in args.magnetic_elements.split(",")]

    spin_channels = None
    if args.spin_channels:
        spin_channels = []
        for s in args.spin_channels.split(","):
            s = s.strip()
            if s == "+1" or s == "1":
                spin_channels.append(1)
            elif s == "-1":
                spin_channels.append(-1)
            else:
                raise ValueError(
                    f"Invalid spin channel '{s}': must be +1 (up) or -1 (down)"
                )
        logger.info(f"Using manual spin channels: {spin_channels}")

    try:
        # ── static-hamiltonian mode ───────────────────────────────────────────
        if args.mode == "static-hamiltonian":
            if spin_channels is None:
                logger.error(
                    "--spin-channels is required for --mode=static-hamiltonian"
                )
                sys.exit(1)

            from TB2J.exchange_params import parser_argument_to_dict
            from TB2J.interfaces.dmft import DMFTstaticManager

            params = parser_argument_to_dict(args)
            # Remove args that DMFTstaticManager does not accept or handles explicitly
            for key in [
                "dmft_file",
                "parser_type",
                "dmft_params_file",
                "mu_file",
                "use_total_sigma",
                "nspin",
                "afm_symmetry",
                "static_sigma",
                "mode",
                "description",
                "spin_channels",
                "magnetic_elements",
                "path",
                "prefix",
                "posfile",
                "output_path",
            ]:
                params.pop(key, None)

            logger.info("Mode: static-hamiltonian — folding Σ(∞)−Vdc into H(R=0)")
            print("\n" + "=" * 60)
            print("Starting DMFT static-Hamiltonian exchange calculation...")
            print("=" * 60)

            DMFTstaticManager(
                path=args.path,
                prefix=args.prefix,
                dmft_file=args.dmft_file,
                spin_channels=spin_channels,
                magnetic_elements=magnetic_elements,
                posfile=args.posfile,
                dmft_params_file=args.dmft_params,
                mu_file=args.mu_file,
                use_total_sigma=args.total_sigma,
                output_path=args.output_path,
                **params,
            )

            print("\n" + "=" * 60)
            print("DMFT static-Hamiltonian exchange completed!")
            print(f"Results written to: {args.output_path}/")
            print("=" * 60)
            return

        # ── full DMFT mode (default) ──────────────────────────────────────────
        # Initialize DMFT manager
        logger.info(f"Initializing DMFT manager with parser type: {args.parser_type}")
        dmft_manager = DMFTManager(
            atoms=None,  # Will be read from Wannier files
            path=args.path,
            prefix=args.prefix,
            dmft_file=args.dmft_file,
            parser_type=args.parser_type,
            dmft_params_file=args.dmft_params,
            mu_file=args.mu_file,
            use_total_sigma=args.total_sigma,
            nspin=args.nspin,
            magnetic_elements=magnetic_elements,
            spin_channels=spin_channels,
            static_sigma=args.static_sigma,
            **{
                k: v
                for k, v in vars(args).items()
                if k
                not in [
                    "path",
                    "prefix",
                    "dmft_file",
                    "parser_type",
                    "dmft_params_file",
                    "mu_file",
                    "use_total_sigma",
                    "nspin",
                    "magnetic_elements",
                    "spin_channels",
                    "static_sigma",
                ]
            },
        )

        # Get model, basis, description
        logger.info("Reading Wannier90 model and DMFT self-energy...")
        dmft_model, basis, description = dmft_manager()

        print("\n" + "=" * 60)
        print("Starting DMFT exchange calculation...")
        print("=" * 60)

        from TB2J.exchange_dmft import ExchangeCLDMFT, ExchangeDMFTNCL
        from TB2J.exchange_params import parser_argument_to_dict

        if args.nspin == 1:
            ExchangeClass = ExchangeCLDMFT
            logger.info("Using collinear exchange calculator (ExchangeCLDMFT)")
        else:
            ExchangeClass = ExchangeDMFTNCL
            logger.info("Using non-collinear exchange calculator (ExchangeDMFTNCL)")

        params = parser_argument_to_dict(args)
        params.pop("description", None)
        params.pop("dmft_file", None)
        params.pop("parser_type", None)
        params.pop("dmft_params_file", None)
        params.pop("mu_file", None)
        params.pop("use_total_sigma", None)
        params.pop("nspin", None)
        params.pop("afm_symmetry", None)
        params.pop("static_sigma", None)

        if args.static_sigma:
            params["method"] = "cfr"
            logger.info("Using CFR contour for static self-energy mode")
        else:
            params["method"] = "matsubara"
        if magnetic_elements:
            params["magnetic_elements"] = magnetic_elements

        exchange = ExchangeClass(
            tbmodels=dmft_model,
            atoms=dmft_model.atoms,
            basis=basis,
            description=description,
            **params,
        )

        logger.info(
            f"Calculating exchange parameters with {params.get('kmesh', 'default')} k-mesh..."
        )
        exchange.run(path=args.output_path)

        print("\n" + "=" * 60)
        print("DMFT exchange calculation completed successfully!")
        print(f"Results written to: {args.output_path}/")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during DMFT calculation: {e}", exc_info=True)
        sys.exit(1)


def validate_args(args):
    """
    Validate command-line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If arguments are invalid
    """
    # Check if dmft_file exists
    import os

    if not os.path.exists(args.dmft_file):
        raise ValueError(f"DMFT file not found: {args.dmft_file}")

    # Check parser-type compatibility
    if args.parser_type == "w2dynamics" and not (
        args.dmft_file.endswith(".h5") or args.dmft_file.endswith(".hdf5")
    ):
        logger.warning(
            f"Parser type 'w2dynamics' specified but file '{args.dmft_file}' "
            f"doesn't have .h5 or .hdf5 extension"
        )

    if args.parser_type == "siginp" and not args.dmft_file.endswith(".inp"):
        logger.warning(
            f"Parser type 'siginp' specified but file '{args.dmft_file}' "
            f"doesn't have .inp extension"
        )

    # Check for required auxiliary files for siginp
    if args.parser_type == "siginp" or args.dmft_file.endswith(".inp"):
        import os

        dir_path = os.path.dirname(args.dmft_file) or "."

        if args.dmft_params:
            if not os.path.exists(args.dmft_params):
                logger.warning(
                    f"Specified dmft_params file not found: {args.dmft_params}"
                )
        else:
            auto_params = os.path.join(dir_path, "dmft_params.dat")
            if not os.path.exists(auto_params):
                logger.warning(
                    f"dmft_params.dat not found in {dir_path}. "
                    f"Using default orbital mapping."
                )

        if args.spin_channels:
            logger.info("Using manual spin channels instead of dmft_params.dat values")


if __name__ == "__main__":
    run_dmft2J()
