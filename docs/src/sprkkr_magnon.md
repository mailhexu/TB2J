# SPRKKR Magnon Workflow

TB2J can read the SPRKKR RuO2 reference format and convert it into the existing
`SpinIO -> Magnon.load_from_io()` workflow. The first supported input contract is
the file set used in `Refs/SPRKKR_RuO2`: `RuO2.str`, `RuO2_JXC_Jij.dat`, and the
full-table variant `RuO2_JXC_XCPLTEN_Jij.dat` for parsing/filtering checks.

The MVP does not require `shells.dat`, `Jijatom.*`, potential files, or shape
function sidecars. The exchange table columns are interpreted as
`IT IQ JT JQ N1 N2 N3 DRX DRY DRZ DR J_xx J_yy J_xy J_yx`; `DRX`, `DRY`, `DRZ`,
and `DR` are Cartesian quantities in units of the SPRKKR lattice parameter `A`.
Exchange values are read in meV and converted to the eV convention used by the
TB2J magnon path.

The reference README states the LKAG convention with prefactor `c = 1`, unit spin
direction vectors, and Ru moment `0.5674 mu_B`. Pass the moment explicitly so the
magnon normalization is visible in scripts and command lines.

## Python API

```python
from TB2J.interfaces import (
    magnon_from_sprkkr,
    read_sprkkr_exchange,
    write_sprkkr_tb2j_results,
)

data = read_sprkkr_exchange(
    structure_file="Refs/SPRKKR_RuO2/RuO2.str",
    exchange_file="Refs/SPRKKR_RuO2/RuO2_JXC_Jij.dat",
    magnetic_species=["Ru"],
    moment=[0.5674, -0.5674],
)

magnon = magnon_from_sprkkr(
    structure_file="Refs/SPRKKR_RuO2/RuO2.str",
    exchange_file="Refs/SPRKKR_RuO2/RuO2_JXC_Jij.dat",
    magnetic_species=["Ru"],
    moment=[0.5674, -0.5674],
    tensor_policy="isotropic",
)

spinio = write_sprkkr_tb2j_results(
    structure_file="Refs/SPRKKR_RuO2/RuO2.str",
    exchange_file="Refs/SPRKKR_RuO2/RuO2_JXC_Jij.dat",
    output_path="TB2J_results",
    magnetic_species=["Ru"],
    moment=[0.5674, -0.5674],
)

labels, bands, xcoords = magnon.get_magnon_bands(path="GX", npoints=20)
```

`write_sprkkr_tb2j_results()` converts SPRKKR inputs into `SpinIO`, writes a full
TB2J result directory including `TB2J.pickle`, `exchange.out`, `structure.vasp`,
and `Multibinit/exchange.xml`, and returns the in-memory `SpinIO` for immediate use
or further inspection. A complete runnable example is available at
`examples/magnon/sprkkr_ruo2_tb2j_results.py`.

`moment` can be a single scalar, which is broadcast to all selected magnetic
sites, `N` signed scalar values for `N` selected magnetic sites, or `3N` values
for explicit moment vectors. Scalar values are placed along the z direction.
Values follow the selected SPRKKR magnetic site order. For two Ru sites, use
`moment=[0.5674, -0.5674]`; the equivalent explicit-vector form is
`moment=[0, 0, 0.5674, 0, 0, -0.5674]`.

Use `tensor_policy="transverse-block"` to place `J_xx`, `J_yy`, `J_xy`, and
`J_yx` into the transverse block of a 3x3 tensor with unavailable z-couplings set
to zero. Use `tensor_policy="isotropic"` to use `(J_xx + J_yy) / 2` as the scalar
exchange. Use `tensor_policy="transverse-block-jzz"` to keep the transverse block
and set `J_zz = (J_xx + J_yy) / 2`; all other unavailable entries remain zero.
Requests for a full tensor are rejected because the reference files do not contain
all tensor components.

## CLI

```bash
sprkkr2magnon.py \
  -s Refs/SPRKKR_RuO2/RuO2.str \
  -e Refs/SPRKKR_RuO2/RuO2_JXC_Jij.dat \
  -S Ru \
  -m 0.5674 -0.5674 \
  -t isotropic \
  -b \
  -k GX \
  -n 20 \
  --qpoints "G:0,0,0,X:0.5,0,0" \
  -o ruo2_sprkkr_magnon.png
```

The short aliases above are equivalent to `--structure`, `--exchange`,
`--magnetic-species`, `--moment`, `--tensor-policy`, `--bands`, `--kpath`,
`--npoints`, and `--output`.

Use `--qpoints` to override high-symmetry point coordinates with the same
`name:x,y,z` comma-separated format as `TB2J_magnon.py`.

The command writes the plot path and a matching JSON file with the same stem. To
write a full TB2J-compatible result directory for later reuse, use:

```bash
sprkkr2magnon.py \
  -s Refs/SPRKKR_RuO2/RuO2.str \
  -e Refs/SPRKKR_RuO2/RuO2_JXC_Jij.dat \
  -S Ru \
  -m 0.5674 -0.5674 \
  -w TB2J_sprkkr_results
```
