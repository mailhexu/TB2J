# O2 collinear zero-pitch anchor (story 011, research fixture)

Producer: SIESTA `5.4.2-24-g71c860291`, local binary SHA-256
`98be67909dd7f0b17d73828630bc1b7d37119372fa64d182901bed788151bdd1`.
Both SCFs use the same O2 geometry, O.psf, PBE/SZP basis, 150 Ry mesh,
1×1×4 k-mesh and 100 K electronic temperature as `siesta_spiral_o2`.
The only spin difference: `nc/` uses noncollinear 4-component HSX with
q=0 and both initial moments along x; `polarized/` uses SIESTA's collinear
spin-polarized mode with moments along z. Both reached free-energy-only SCF
convergence in nine iterations. These are **two independent SCFs** even though
their printed total energies coincide at six decimals:
`E_nc = E_polarized = -868.179053 eV`. Thus the printed energy difference
is `0.000000 eV` with rounding uncertainty; do not claim bit identity or
sub-µeV agreement. Printed total spin moments are (2,0,0) and (0,0,2) µB.

`o2_fm_q0.npz` is reduced **only** from `nc/o2.HSX`, `.EIG`, `.KP` by
`make_fixture_from_siesta(hsx,eig,kp,(0,0,0),12,100.0,'o2_fm')`.
Story-003 importer oracles pass: electron count 12.000000000 (budget 1e-4)
and EIG vs pencil max deviation 5.213e-7 eV (budget 1e-5).
The occupied–empty gap on the recorded four-point k-mesh is 2.49684889 eV
(bands 12/13). On the matched five-point gamma-centered mesh used by
ExchangeCL2, the sampled gap is 2.50146452945 eV. The reference is gapped.
`SHA256SUMS` pins inputs, raw outputs and reduced NPZ.

**Research caveat, not an approved AC-006 certificate:** the new response
rotates the declared *total-H* magnetic operator, including intersite
spin-dependent blocks (max |Re H12| = 0.2456934 eV), whereas established
ExchangeCL2's `Delta` uses site-onsite orbital blocks at R=0. These
magnetic-operator/projector definitions are not identical. On a common
five-point mesh the new raw mixed Hessian is
`K(O1,O2) = -0.623057870 eV/rad²` in both transverse directions.
The repository's ordered-pair convention maps it to
`J_stored = -K/2 = +0.311528935 eV`; the established-path image-summed
`J_stored(O1,O2)` from **the same NC HSX** (H0±dx spin-channel adapter,
Fermi-zero shifted) is `+0.311637068 eV`. The 0.108133 meV residual
is *not* a second-SCF effect. The direct polarized-twin ExchangeCL2 run
yields an image sum around `+0.3116437 eV` (Rcut 8 Å, five-point mesh,
contour nz=100–800). Until the projector mismatch is dispositioned,
this is research evidence, not a matched-projector pass.

Reproduce the NC producer in a fresh directory containing `O.psf`:
`siesta < o2_fm.fdf > o2_fm.out` (likewise for `polarized/`).
The `.fdf` files retain `SystemLabel o2`, so run each in its own directory.
The O2 seed has free-energy-only SCF convergence as in the original spiral
fixture; no stationary-torque assumption is made.
