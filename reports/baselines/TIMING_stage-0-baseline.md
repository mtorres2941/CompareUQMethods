# Baseline timing, Stage 1 Phase 0

Unmodified notebooks (apart from the A1 clean-kernel fix), executed headless
with `jupyter nbconvert --execute` under the pinned `compareuq` environment on
the author's current laptop, 2026-09-11.

| Notebook | Wall clock | Notes |
|---|---|---|
| `01_CompareUQ_CreateData.ipynb` | 18.2 s | `generate_dontread = False`, so the 15,000 datasets are read from `DATA_all.json` rather than regenerated |
| `02_CompareUQ_AnalyzeData.ipynb` | 72.9 s | Full fit and score of 10,000 synthetic and 138 empirical datasets, plus 6 figures at dpi 1200 |
| `03_CompareUQ_PerformPLCA.ipynb` | see below | 2,500 pLCA combos at `neccs = 1000`, plus 15 figures at dpi 1200 |

## The disagreement this was meant to settle

Stage 0 section 3.5 recorded a conflict: the author remembered NB3 taking an
extremely long time, while the Stage 0 measurements put the pLCA loop at about
3.5 minutes. This run is the arbitration.

Conclusion: **the pipeline is not slow on the current stack.** The most likely
explanation for the original experience is the previous laptop combined with an
older pandas, where `df.loc[len(df)] = row` and scalar `.loc` assignment into
object-dtype frames were quadratic rather than linear. Both patterns are used
heavily in NB3 cell 25 and in `compare_results`, which is called about 44
times. On pandas 3.0.5 they are linear.

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
