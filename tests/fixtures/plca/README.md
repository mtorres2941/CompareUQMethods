# pLCA fixtures

Three artifacts, per Stage 1 amendment A2. Stored gzipped; `pandas.read_csv`
reads them directly.

| File | neccs | Seeded | What it is |
|---|---|---|---|
| `PLCA_unseeded_archive_neccs1000.csv.gz` | 1,000 | no | **Archive, not a fixture.** The last run of the code as it stood before seeding. Cannot be reproduced by rerunning. Kept because it is the closest surviving record of the configuration the current manuscript draft reports. |
| `PLCA_seeded_neccs1000.csv.gz` | 1,000 | yes, seed 20260911 | First reproducible pLCA result. Exists to isolate the effect of the neccs change from the effect of seeding. |
| `PLCA_seeded_neccs10000.csv.gz` | 10,000 | yes, seed 20260911 | The configuration the manuscript states. Created in Phase 3. |

## Monte Carlo noise floor

Comparing the unseeded archive against the seeded run, both at neccs = 1000,
isolates pure Monte Carlo variation, since nothing else differs:

| Result | mean abs diff | p95 | max | sd of the result itself |
|---|---|---|---|---|
| `eci_rank_1` | 0.01488 | 0.03700 | 0.08800 | 0.09593 |
| `eci_perc_mean` | 0.00304 | 0.00853 | 0.02503 | 0.01226 |
| `ui` | 0.02026 | 0.05831 | 0.21336 | 0.23658 |
| `eci_mean` | 0.01353 | 0.04510 | 0.17318 | 0.08129 |

At neccs = 1000 the noise in `eci_rank_1`, the headline result, is 0.0149,
which is 15.5 percent of the standard deviation of the result across datasets.
The theoretical standard error for a proportion near 0.5 at n = 1000 is 0.0158,
so the observed noise matches expectation.

## Why this matters for the paper

The central claim is that the choice of UQ method changes pLCA outcomes. That
claim is only meaningful if the differences between methods exceed the Monte
Carlo noise. Measured on the seeded run at neccs = 1000, mean absolute
difference in `eci_rank_1` between each pair of UQ methods:

| Method pair | difference | multiple of noise |
|---|---|---|
| Lognormal Uniform vs Normal Variable | 0.0600 | 4.0x |
| KDE Variable vs Lognormal Uniform | 0.0599 | 4.0x |
| KDE Uniform vs Lognormal Variable | 0.0597 | 4.0x |
| ... | ... | ... |
| KDE Uniform vs KDE Variable | 0.0341 | 2.3x |
| KDE Variable vs Normal Variable | 0.0326 | 2.2x |

All 15 pairs clear the noise floor, so the conclusions hold. But the weakest
comparisons sit only 2.2x above it, which is thin for a headline result.
Raising neccs to 10,000 drops the noise floor to about 0.0047 and lifts those
same comparisons to roughly 7x. That is a substantive argument for the change,
independent of the manuscript already claiming n = 10,000.
