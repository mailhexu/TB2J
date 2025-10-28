import os
from os.path import expanduser

# Import the enhanced functions from Oiju_epw2
from TB2J.Oiju_epw2 import (
    gen_exchange_Oiju_epw,
    gen_exchange_Oiju_epw_multiple,
    run_from_toml,
)


def example_single_idisp():
    """Example: Calculate spin-phonon coupling for a single displacement mode."""
    print("=== Example: Single displacement mode ===")

    gen_exchange_Oiju_epw(
        path="./",
        colinear=True,
        posfile="scf.pwi",
        prefix_up="up/SrMnO3",
        prefix_dn="down/SrMnO3.down",
        epw_path="up",
        epw_prefix_up="up/SrMnO3",
        epw_prefix_dn="down/SrMnO3.down",
        idisp=3,
        Ru=(0, 0, 0),
        Rcut=8,
        efermi=11.26,
        magnetic_elements=["Mn"],
        kmesh=[6, 6, 6],
        emin=-8.3363330034071295,
        emax=0.0,
        nz=50,
        np=1,
        exclude_orbs=[],
        description="Single displacement mode example - mode 3",
        output_path="dJdx_single_mode",
    )


def example_multiple_idisp_original():
    """Example: Original calculation with multiple displacement modes."""
    print("=== Example: Multiple displacement modes (original parameters) ===")

    outpath = "dJdx_666"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    gen_exchange_Oiju_epw_multiple(
        path="./",
        colinear=True,
        posfile="scf.pwi",
        prefix_up="up/SrMnO3",
        prefix_dn="down/SrMnO3.down",
        epw_path="up",
        epw_prefix_up="up/SrMnO3",
        epw_prefix_dn="down/SrMnO3.down",
        idisp_list=[0, 3, 4, 5, 6, 7, 8],
        Ru=(0, 0, 0),
        Rcut=8,
        efermi=11.26,
        magnetic_elements=["Mn"],
        kmesh=[6, 6, 6],
        emin=-8.3363330034071295,
        emax=0.0,
        nz=50,
        np=1,
        exclude_orbs=[],
        description="Multiple displacement modes - original parameters",
        output_path=outpath,
    )


def example_srmno3_original():
    """Example: Reproduce the original SrMnO3 calculation parameters."""
    print("=== Example: Original SrMnO3 calculation ===")

    gen_exchange_Oiju_epw_multiple(
        path=expanduser("~/projects/TB2J_examples/Wannier/SrMnO3_QE_Wannier90/W90"),
        colinear=True,
        posfile="SrMnO3.scf.pwi",
        prefix_up="SrMnO3_up",
        prefix_dn="SrMnO3_down",
        epw_path=expanduser("~/projects/projects/SrMnO3/epw"),
        epw_prefix_up="SrMnO3_up",
        epw_prefix_dn="SrMnO3_dn",
        idisp_list=[3, 6, 7],
        Ru=(0, 0, 0),
        Rcut=8,
        efermi=10.67,
        magnetic_elements=["Mn"],
        kmesh=[5, 5, 5],
        emin=-7.3363330034071295,
        emax=0.0,
        nz=70,
        np=1,
        exclude_orbs=[],
        description="Original SrMnO3 calculation with cutoff 0.1",
        output_path="VT_withcutoff0.1",
    )


def example_from_toml():
    """Example: Calculate spin-phonon coupling using TOML configuration file."""
    print("=== Example: Using TOML configuration file ===")

    config_file = "Oije_example_config.toml"

    # Check if the configuration file exists
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found.")
        print("Please create the configuration file first.")
        return

    run_from_toml(config_file)


def main():
    """Main function to run examples."""
    print("Spin-Phonon Coupling Calculation Examples")
    print("=" * 50)

    print("\nAvailable examples:")
    print("1. Single displacement mode")
    print("2. Multiple displacement modes (original parameters)")
    print("3. Original SrMnO3 calculation")
    print("4. Using TOML configuration file")
    print("5. Run all examples")

    try:
        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            example_single_idisp()
        elif choice == "2":
            example_multiple_idisp_original()
        elif choice == "3":
            example_srmno3_original()
        elif choice == "4":
            example_from_toml()
        elif choice == "5":
            print("\nRunning all examples...")
            example_single_idisp()
            example_multiple_idisp_original()
            example_from_toml()
            example_srmno3_original()
        else:
            print("Invalid choice. Please run the script again.")
            return

        print("\nCalculation completed successfully!")

    except KeyboardInterrupt:
        print("\nCalculation interrupted by user.")
    except Exception as e:
        print(f"\nError during calculation: {e}")
        print("Please check your input parameters and try again.")


if __name__ == "__main__":
    main()
