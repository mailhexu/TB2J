# WFSX benchmark run (Story009 opt-in command)

Measured on phys-ptm40.fsc.uliege.priv (32 CPUs), 6.17.0-40-generic, Python 3.11.11, generated 2026-11-02T15:30:11.955410+00:00.

- TB2J worktree HEAD: `5b5fde3e11e07985a19f2820c3a29554dab01ec5` (/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j)
- HamiltonIO HEAD: `da9881f77082e9516759e84381242ed3e2a02bfa` (/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-hamiltonio)
- Driver: run_benchmarks.py sha256 `5a5326ba74ba61d497c55ce1aac641f1528841dec767b024241b6ac6b69fb369`

Method: one fresh isolated process per input mode (ordinary complete-HS vs public WFSX entry), single process and one thread per BLAS pool; identical explicit full requested k points with uniform weights, identical CFR contour parameters (nz and smearing recorded per run) and identical odd-coerced R list in both modes. Cold means per-input POSIX_FADV_DONTNEED advice only (never a guaranteed global cold page cache); warm means the immediately repeated fresh process on the same inputs. Peak RSS is resource.getrusage(RUSAGE_SELF) process-lifetime high-water in bytes, children excluded; launch wall time is reported separately from the worker-measured phases. WFSX spectral factors are staged in an owned unique system-temporary parent and removed afterwards; the ordinary mode keeps the eigenbasis in memory (use_cache=False). All numbers below are measured on the named runs only; no speed target, no projected universal benefit.

## collinear_nspin2_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.009 | 0.004 | 0.008 | 0.257 | 0.278 | 10 | 0.001 | 276.3 | 0.0 |
| cold | wfsx | disk-spectral | 0.007 | 0.003 | 0.012 | 0.249 | 0.273 | 0 | 0.000 | 282.0 | 102.7 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.772443206959906e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.1048740354529252e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.981 (cold policy: cold).

## collinear_nspin2_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.008 | 0.244 | 0.261 | 10 | 0.001 | 275.8 | 0.0 |
| warm | wfsx | disk-spectral | 0.007 | 0.003 | 0.012 | 0.252 | 0.275 | 0 | 0.000 | 282.0 | 102.7 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.772443206959906e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.1048740354529252e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.056 (cold policy: warm).

## collinear_nspin2_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.015 | 0.003 | 0.010 | 0.250 | 0.278 | 10 | 0.001 | 277.0 | 0.0 |
| cold | wfsx | disk-spectral | 0.007 | 0.003 | 0.013 | 0.258 | 0.282 | 0 | 0.000 | 282.4 | 102.7 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007115220796549693 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.5581698366952907e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.012 (cold policy: cold).

## collinear_nspin2_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.009 | 0.250 | 0.267 | 10 | 0.001 | 276.3 | 0.0 |
| warm | wfsx | disk-spectral | 0.007 | 0.003 | 0.013 | 0.254 | 0.279 | 0 | 0.000 | 281.5 | 102.7 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007115220796549693 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.5581698366952907e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.043 (cold policy: warm).

## collinear_nspin2_reduced (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_reduced`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.017 | 0.003 | 0.008 | 0.249 | 0.277 | 10 | 0.001 | 275.6 | 0.0 |
| cold | wfsx | disk-spectral | 0.006 | 0.002 | 0.009 | 0.252 | 0.271 | 0 | 0.000 | 282.1 | 61.6 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.772443206959906e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.1048740366006917e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.977 (cold policy: cold).

## collinear_nspin2_reduced (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_reduced`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.007 | 0.250 | 0.265 | 10 | 0.001 | 276.0 | 0.0 |
| warm | wfsx | disk-spectral | 0.006 | 0.002 | 0.009 | 0.250 | 0.269 | 0 | 0.000 | 281.9 | 61.6 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.772443206959906e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.1048740366006917e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.017 (cold policy: warm).

## collinear_nspin2_reduced (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_reduced`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.015 | 0.002 | 0.009 | 0.249 | 0.276 | 10 | 0.001 | 277.4 | 0.0 |
| cold | wfsx | disk-spectral | 0.007 | 0.002 | 0.009 | 0.254 | 0.275 | 0 | 0.000 | 282.4 | 61.6 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007115220787454746 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.5581698399959355e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.995 (cold policy: cold).

## collinear_nspin2_reduced (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/collinear_nspin2_reduced`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.009 | 0.250 | 0.267 | 10 | 0.001 | 276.9 | 0.0 |
| warm | wfsx | disk-spectral | 0.006 | 0.002 | 0.009 | 0.252 | 0.271 | 0 | 0.000 | 282.2 | 61.6 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007115220787454746 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.5581698399959355e-06, populations charges max |Δ| 4.902744876744691e-13 e / spin max |Δ| 4.694022948115162e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.015 (cold policy: warm).

## noncollinear_nspin4_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/noncollinear_nspin4_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.007 | 0.003 | 0.006 | 0.330 | 0.345 | 5 | 0.001 | 279.0 | 0.0 |
| cold | wfsx | disk-spectral | 0.005 | 0.003 | 0.008 | 0.370 | 0.388 | 0 | 0.000 | 283.4 | 203.9 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.2014441495011852e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 882.6726743959599, populations charges max |Δ| 2.5757174171303632e-14 e / spin max |Δ| 6.38378239159465e-14 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.125 (cold policy: cold).

## noncollinear_nspin4_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/noncollinear_nspin4_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.006 | 0.329 | 0.343 | 5 | 0.001 | 278.3 | 0.0 |
| warm | wfsx | disk-spectral | 0.003 | 0.002 | 0.008 | 0.373 | 0.387 | 0 | 0.000 | 283.7 | 203.9 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.2014441495011852e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 882.6726743959599, populations charges max |Δ| 2.5757174171303632e-14 e / spin max |Δ| 6.38378239159465e-14 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.130 (cold policy: warm).

## noncollinear_nspin4_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/noncollinear_nspin4_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.016 | 0.003 | 0.008 | 0.330 | 0.357 | 5 | 0.001 | 278.7 | 0.0 |
| cold | wfsx | disk-spectral | 0.003 | 0.002 | 0.009 | 0.379 | 0.395 | 0 | 0.000 | 283.8 | 203.9 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007754649971047911 meV (tolerance abs 0.001 / rel 0.0001), max rel 19693.1220211524, populations charges max |Δ| 2.5757174171303632e-14 e / spin max |Δ| 6.38378239159465e-14 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.107 (cold policy: cold).

## noncollinear_nspin4_full (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-tb2j/tests/siesta_wfsx_fixtures/noncollinear_nspin4_full`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.005 | 0.002 | 0.008 | 0.329 | 0.345 | 5 | 0.001 | 278.8 | 0.0 |
| warm | wfsx | disk-spectral | 0.003 | 0.002 | 0.009 | 0.375 | 0.391 | 0 | 0.000 | 283.5 | 203.9 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 0.0007754649971047911 meV (tolerance abs 0.001 / rel 0.0001), max rel 19693.1220211524, populations charges max |Δ| 2.5757174171303632e-14 e / spin max |Δ| 6.38378239159465e-14 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.135 (cold policy: warm).

## o2_repeat20_1x1x81 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/large-repeat20-81-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.012 | 8.927 | 10.946 | 0.304 | 20.189 | 162 | 5.068 | 1100.8 | 0.0 |
| cold | wfsx | disk-spectral | 0.234 | 2.104 | 7.182 | 0.433 | 10.743 | 0 | 0.000 | 325.6 | 332330.6 |

Same-mesh equivalence (default): **PASS** — 4800 records/run, max |ΔJ| 2.104260062196772e-06 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.387286529790569e-07, populations charges max |Δ| 1.0107470416187425e-11 e / spin max |Δ| 1.0440093234365122e-11 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.532 (cold policy: cold).

## o2_repeat20_1x1x81 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/large-repeat20-81-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.010 | 8.911 | 10.679 | 0.299 | 19.900 | 162 | 4.919 | 1100.8 | 0.0 |
| warm | wfsx | disk-spectral | 0.262 | 2.152 | 7.237 | 0.348 | 10.707 | 0 | 0.000 | 325.5 | 332330.6 |

Same-mesh equivalence (default): **PASS** — 4800 records/run, max |ΔJ| 2.104260062196772e-06 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.387286529790569e-07, populations charges max |Δ| 1.0107470416187425e-11 e / spin max |Δ| 1.0440093234365122e-11 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.538 (cold policy: warm).

## o2_repeat20_1x1x81 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/large-repeat20-81-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.020 | 8.750 | 39.743 | 0.421 | 48.935 | 162 | 4.956 | 1100.2 | 0.0 |
| cold | wfsx | disk-spectral | 0.232 | 2.150 | 18.252 | 0.435 | 21.860 | 0 | 0.000 | 316.3 | 332330.6 |

Same-mesh equivalence (contravariant): **PASS** — 4800 records/run, max |ΔJ| 7.570571551696048e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.541899630914974e-07, populations charges max |Δ| 1.0107470416187425e-11 e / spin max |Δ| 1.0440093234365122e-11 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.447 (cold policy: cold).

## o2_repeat20_1x1x81 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/large-repeat20-81-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.012 | 8.756 | 39.808 | 0.316 | 48.893 | 162 | 4.953 | 1101.0 | 0.0 |
| warm | wfsx | disk-spectral | 0.247 | 2.167 | 18.462 | 0.393 | 21.975 | 0 | 0.000 | 316.6 | 332330.6 |

Same-mesh equivalence (contravariant): **PASS** — 4800 records/run, max |ΔJ| 7.570571551696048e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 3.541899630914974e-07, populations charges max |Δ| 1.0107470416187425e-11 e / spin max |Δ| 1.0440093234365122e-11 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 0.449 (cold policy: warm).

## o2_repeat1_1x1x1001 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/dense-1001-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.012 | 0.222 | 0.666 | 0.245 | 1.145 | 2002 | 0.089 | 299.1 | 0.0 |
| cold | wfsx | disk-spectral | 0.241 | 0.213 | 0.778 | 0.403 | 1.663 | 0 | 0.000 | 285.4 | 10286.2 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.3641113696394314e-06 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.5694319338794882e-08, populations charges max |Δ| 2.824407374646398e-13 e / spin max |Δ| 9.672262990534364e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.453 (cold policy: cold).

## o2_repeat1_1x1x1001 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/dense-1001-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.010 | 0.223 | 0.680 | 0.277 | 1.191 | 2002 | 0.090 | 299.6 | 0.0 |
| warm | wfsx | disk-spectral | 0.254 | 0.238 | 0.769 | 0.369 | 1.658 | 0 | 0.000 | 284.4 | 10286.2 |

Same-mesh equivalence (default): **PASS** — 12 records/run, max |ΔJ| 1.3641113696394314e-06 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.5694319338794882e-08, populations charges max |Δ| 2.824407374646398e-13 e / spin max |Δ| 9.672262990534364e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.392 (cold policy: warm).

## o2_repeat1_1x1x1001 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/dense-1001-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold | hs | in-memory-eigenbasis (use_cache=False) | 0.009 | 0.220 | 0.947 | 0.255 | 1.432 | 2002 | 0.089 | 299.6 | 0.0 |
| cold | wfsx | disk-spectral | 0.243 | 0.225 | 0.811 | 0.298 | 1.603 | 0 | 0.000 | 285.4 | 10286.2 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 2.217705809925974e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.608063369711159e-07, populations charges max |Δ| 2.824407374646398e-13 e / spin max |Δ| 9.672262990534364e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.119 (cold policy: cold).

## o2_repeat1_1x1x1001 (`/home/hexu/projects/TB2J_dev/worktrees/siesta-wfsx-vault/.tmp/kosmic/runs/20261030T193117Z-siesta-wfsx-benchmark-co-899b70b05a3c3b45279d2c16dae6636a/scratch/dense-1001-final`)

| policy | mode | cache mode | import s | preparation s | integration s | output s | total s | eigensolver calls | eigensolver s | peak RSS MiB | spectral disk KiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm | hs | in-memory-eigenbasis (use_cache=False) | 0.008 | 0.218 | 0.933 | 0.294 | 1.453 | 2002 | 0.088 | 300.0 | 0.0 |
| warm | wfsx | disk-spectral | 0.246 | 0.229 | 0.813 | 0.365 | 1.681 | 0 | 0.000 | 284.9 | 10286.2 |

Same-mesh equivalence (contravariant): **PASS** — 12 records/run, max |ΔJ| 2.217705809925974e-05 meV (tolerance abs 0.001 / rel 0.0001), max rel 1.608063369711159e-07, populations charges max |Δ| 2.824407374646398e-13 e / spin max |Δ| 9.672262990534364e-13 mu_B (tolerance abs 0.0001 / rel 1e-05).
Measured total-wall ratio wfsx/hs on this run: 1.157 (cold policy: warm).

