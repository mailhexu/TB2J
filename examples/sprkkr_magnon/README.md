# RuO2 SPRKKR Magnon Comparison

This directory is a RuO2 example. It contains the input files and a native TB2J CLI command script used to compute two magnon-band cases from SPRKKR reference data:

- `ruo2_isotropic_magnon_band.png`: only isotropic exchange, with scalar `J = (Jxx + Jyy) / 2`.
- `ruo2_with_offdiagonal_magnon_band.png`: transverse tensor block `Jxx`, `Jyy`, `Jxy`, `Jyx`, with `Jzz = (Jxx + Jyy) / 2`.
- `RuO2.str` and `RuO2_JXC_Jij.dat`: local SPRKKR input files used by the script.

## Install This TB2J Version
To install TB2J with the `sprkkr2magnon.py` CLI, use the following pip command to get the specific alpha version that includes this script:
```
pip install TB2J==0.9.14.0a1
```


```bash
sprkkr2magnon.py --help
```

## Run The Example

The one-line bash helper is `run_magnons.sh`:



The magnetic state is AFM Ru moments `[0.5674, -0.5674]`, and the k path is `G-X-S-G` with `S=(0.5,0.5,0)`. The CLI also writes numeric band data to `ruo2_isotropic_magnon_band.json` and `ruo2_with_offdiagonal_magnon_band.json`, plus `plot_magnon_band.py` for replotting the saved JSON data.

Note that the nominal magnetic moment is often used in magnon calculation. But for RuO2, It is not clear to me what the nominal moment should be, so I just used the DFT-calculated moment from the SPRKKR run. 


## `sprkkr2magnon.py --help`

```text
usage: sprkkr2magnon.py [-h] -s STRUCTURE -e EXCHANGE
                        [-S MAGNETIC_SPECIES [MAGNETIC_SPECIES ...]]
                        [-i MAGNETIC_SITES [MAGNETIC_SITES ...]]
                        [-m MOMENT [MOMENT ...]]
                        [-t {transverse-block,transverse-block-jzz,isotropic}]
                        [-b] [-k KPATH] [-n NPOINTS] [--qpoints QPOINTS]
                        [-o OUTPUT] [--show] [-w WRITE_TB2J_RESULTS]

Compute magnon bands from SPRKKR exchange output files

options:
  -h, --help            show this help message and exit
  -s STRUCTURE, --structure STRUCTURE
                        SPRKKR .str structure file
  -e EXCHANGE, --exchange EXCHANGE
                        SPRKKR Jij.dat exchange file
  -S MAGNETIC_SPECIES [MAGNETIC_SPECIES ...], --magnetic-species MAGNETIC_SPECIES [MAGNETIC_SPECIES ...]
                        Chemical symbols to treat as magnetic sites, for
                        example Ru
  -i MAGNETIC_SITES [MAGNETIC_SITES ...], --magnetic-sites MAGNETIC_SITES [MAGNETIC_SITES ...]
                        1-based SPRKKR site IDs to treat as magnetic
  -m MOMENT [MOMENT ...], --moment MOMENT [MOMENT ...]
                        Magnetic moment values: one scalar, N z moments, or 3N
                        vector components for N magnetic sites
  -t {transverse-block,transverse-block-jzz,isotropic}, --tensor-policy {transverse-block,transverse-block-jzz,isotropic}
                        How to convert SPRKKR J_xx/J_yy/J_xy/J_yx columns.
                        isotropic: use scalar J=(J_xx+J_yy)/2;
                        transverse-block: keep J_xx,J_yy,J_xy,J_yx as a transverse 3x3
                        tensor block and set unavailable z terms to zero;
                        transverse-block-jzz: same transverse block, plus
                        J_zz=(J_xx+J_yy)/2.
  -b, --bands           Compute magnon bands
  -k KPATH, --kpath KPATH
                        Band path such as GXMG
  -n NPOINTS, --npoints NPOINTS
                        Number of band points
  --qpoints QPOINTS     Custom q-points as name:coord pairs, e.g.
                        "G:0,0,0,X:0.5,0,0"
  -o OUTPUT, --output OUTPUT
                        Output image path for band structure
  --show                Show the band plot
  -w WRITE_TB2J_RESULTS, --write-tb2j-results WRITE_TB2J_RESULTS
                        Write a TB2J-compatible pickle directory
```
