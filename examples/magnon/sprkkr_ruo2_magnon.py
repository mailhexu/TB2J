from pathlib import Path

from TB2J.interfaces.sprkkr import magnon_from_sprkkr

root = Path(__file__).resolve().parents[2]
ref_dir = root.parent / "Refs" / "SPRKKR_RuO2"

magnon = magnon_from_sprkkr(
    structure_file=ref_dir / "RuO2.str",
    exchange_file=ref_dir / "RuO2_JXC_Jij.dat",
    magnetic_species=["Ru"],
    moment=[0.5674, -0.5674],
    tensor_policy="isotropic",
)

labels, bands, _ = magnon.get_magnon_bands(path="GX", npoints=20)
print(labels)
print(bands.shape)
