from pathlib import Path

from TB2J.interfaces import write_sprkkr_tb2j_results

root = Path(__file__).resolve().parents[2]
ref_dir = root.parent / "Refs" / "SPRKKR_RuO2"
output_path = root / "TB2J_sprkkr_results"

spinio = write_sprkkr_tb2j_results(
    structure_file=ref_dir / "RuO2.str",
    exchange_file=ref_dir / "RuO2_JXC_Jij.dat",
    output_path=output_path,
    magnetic_species=["Ru"],
    moment=[0.5674, -0.5674],
    tensor_policy="isotropic",
)

print(f"Wrote TB2J results to {output_path}")
print(f"Main text output: {output_path / 'exchange.out'}")
print(f"Pickle output: {output_path / 'TB2J.pickle'}")
print(f"Magnetic spins: {spinio.nspin}")
print(f"Spin mapping: {list(spinio.index_spin)}")
