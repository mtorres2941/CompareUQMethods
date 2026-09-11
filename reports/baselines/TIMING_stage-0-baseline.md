# Baseline timing, Stage 1 Phase 0

Unmodified notebooks (apart from the A1 clean-kernel fix), executed headless
with `jupyter nbconvert --execute` under the pinned `compareuq` environment on
the author's current laptop, 2026-09-11.

| Notebook | Wall clock | Notes |
|---|---|---|
| `01_CompareUQ_CreateData.ipynb` | 18.2 s | `generate_dontread = False`, so the 15,000 datasets are read from `DATA_all.json` rather than regenerated |
| `02_CompareUQ_AnalyzeData.ipynb` | 72.9 s | Full fit and score of 10,000 synthetic and 138 empirical datasets, plus 6 figures at dpi 1200 |
| `03_CompareUQ_PerformPLCA.ipynb` | 655.6 s (10.9 min) | 2,500 pLCA combos at `neccs = 1000`, plus 15 figures at dpi 1200 |
| **Total** | **746.7 s (12.4 min)** | |

## The disagreement this was meant to settle

Stage 0 section 3.5 recorded a conflict: the author remembered NB3 taking an
extremely long time, while the Stage 0 measurements put the pLCA loop at about
3.5 minutes. This run is the arbitration.

Conclusion: **the pipeline is not slow on the current stack.** The whole
analysis, all three notebooks, runs in 12.4 minutes. NB3 is the slowest at
10.9 minutes, of which the pLCA loop itself accounts for roughly 3.5 minutes
(measured separately in Stage 0); the remainder is figure rasterization at
dpi 1200 and repeated reconstruction of a 60,000-cell DataFrame.

The frame reconstruction is worse than Stage 0 recorded. `compare_results` and
`compare_results_bypewt` each rebuild the full
`index = 10,000 datasets x columns = 6 methods` frame by scalar
`.loc[dataset, pewt] = value` assignment, 60,000 writes per call. Between the
direct calls and the two loops over all 38 result categories, that is about 81
rebuilds per run, not the 44 estimated in Stage 0.

The author's recollection that NB3 took an extremely long time is still not
reproduced. The most likely explanation remains the previous laptop combined
with an older pandas, where `df.loc[len(df)] = row` and scalar `.loc`
assignment into object-dtype frames were quadratic rather than linear. On
pandas 3.0.5 both are linear. This measurement does not refute the
recollection, it simply cannot reproduce it on current hardware and software.

This does not reduce the case for the refactor, which rests on reproducibility
and correctness, not speed.

## Reproducibility of the result tables

The pinned environment is not the one that produced the shipped tables, which
was lost. Agreement between the shipped tables and this rebuild:

| Quantity | Max relative difference |
|---|---|
| Metric columns | 1.3e-14 |
| Normal and KDE Wasserstein-1 | 2.0e-13 |
| Lognormal Wasserstein-1 | 4.3e-08 |

The lognormal term dominates because `weighted_lognorm_fit` calls
`scipy.optimize.minimize`, whose convergence path is version dependent. The
regression tolerance is set at 1e-6, an order of magnitude above the worst
observed value.

**The analysis is therefore reproducible across environments**, which was not
previously demonstrated.
