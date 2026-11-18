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

**Repaired weighted rotation, numeric comparison only:** the production
response rotates the saved total-H magnetic operator with onsite blocks
fully owned and interatomic blocks split half to each endpoint. The
largest interatomic $|H_3|$ at R=0 is 0.2456934 eV. Established
`ExchangeCL2` instead constructs its rotation vertex from onsite
R=0 blocks of Δ=H↑−H↓. Both engines use the same NC SCF H/S:
the latter consumes H0±dx spin channels shifted by -Ef S. Their
generalized spectra agree on all four recorded producer k-points
to at most 1.066e-13 eV.

On the matched actual five-point mesh, repaired offsite curvature is
`K(O1,O2) = -0.474663805554 eV/rad²` in both transverse directions.
The repository's ordered-pair convention yields a **one-directed**
stored value `J_new = -K/2 = +237.331902777 meV`; image-summed
same-SCF `ExchangeCL2` gives `J_LKAG = +311.637068174 meV`.
The signed LKAG-minus-new difference is **+74.305165397 meV**.
Both opposite ordered directions are present in the legacy J dict;
the image sum uses only O1→O2, never their double-counted total.
The methods' magnetic operators, integration (occupied-band vs LKAG
contour), ensemble (gapped T=0 step vs smearing), and phase gauges
are recorded separately as unmatched. User selected numbers-only:
**no agreement tolerance, no AC-006/NFR-005 compatibility claim**.
The independent polarized-twin SCF total agrees at printed precision,
but its H is not used for these values.

**Why the previous values were retired:** the version-1 sequential
full-incident action rotated a shared O1–O2 block twice for an
equal-angle site rotation while onsite blocks rotated once. At q=0,
SOC-free O2, a uniform 0.10-rad tilt gave -1.682616 meV energy
change, violating global-spin covariance. The old curvature
-0.623057870 eV/rad² and stored J +311.528935 meV are not valid
oracles for the repaired convention. Version-2 weighted contributions
restore exact common-angle covariance (occupied-band |ΔE| below
3e-11 meV at 0.10 rad on the four-point producer mesh), while
one site's rotation remains nontrivial (+2.373148 meV). The exact
weighted formula is pinned under `docs/sympy/siesta_spiral_vertex.py`
and `siesta_spiral_frozen_response.py`.

Reproduce the NC producer in a fresh directory containing `O.psf`:
`siesta < o2_fm.fdf > o2_fm.out` (likewise for `polarized/`).
The `.fdf` files retain `SystemLabel o2`, so run each in its own directory.
The O2 seed has free-energy-only SCF convergence as in the original spiral
fixture; no stationary-torque assumption is made.
