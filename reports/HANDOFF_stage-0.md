# HANDOFF stage-0 - Orientation and inventory

## 1. Stage and branch

- Stage: 0, orientation (inventory, dependency map, refactor plan, coding assessment)
- Branch: `stage-0-1-refactor`
- Branched from: `1ab6bf5` (on `main`), itself one commit after `4a20bdd`
  "Prepare repository for journal submission to Building & Environment"
- Commits in this stage:
  - `1ab6bf5` Exclude local reference PDFs and third-party datasets from version
    control (made on `main` before branching)
  - `ec358cf` Add CLAUDE.md with project brief, reference manifest, and handoff
    spec
  - this handoff file

No analysis logic was changed in this stage.

## 2. What was asked

Set up CLAUDE.md; inventory every notebook and script; build a dependency map;
flag duplication, dead code, hardcoded paths, and execution-order dependence;
identify runtime bottlenecks and their causes; locate every entry point for
randomness and say whether it is seeded; describe the synthetic data generation
algorithm in detail; inventory `src/customstats.py`; propose a refactor plan
without executing it; and assess the author's coding honestly.

## 3. What was done

### 3.1 Uncommitted work

The only uncommitted item was an untracked `refs/` directory: 73 MB of
publisher PDFs plus a 104 MB third-party supplementary spreadsheet
(`refs/28462145/full_lca_results.xlsx`). These were gitignored rather than
committed. Reason: this repository is a public Zenodo deposit, and committing
copyrighted publisher PDFs would be both a licensing problem and a permanent
177 MB addition to history. The files remain on disk for consultation.

All nine papers named in the reference manifest are present in `refs/`, so no
manifest lines were deleted. Three files are present that the manifest does not
name: Silverman (1986) "Density Estimation for Statistics and Data Analysis",
`jcgs.2009.08054.pdf` (Journal of Computational and Graphical Statistics, 2009),
and "The Embodied Carbon Benchmark Report 2025".

### 3.2 File inventory

| File | Role | Reads | Writes | Measured runtime |
|---|---|---|---|---|
| `notebooks/01_CompareUQ_CreateData.ipynb` (29 cells) | Generates 15,000 synthetic ECC datasets, pulls and cleans 138 empirical EC3 datasets, computes 20 metrics per dataset, flags metric outliers, trims to 10,000, builds 2,500 four-material combos | EC3 CSVs at `../../EPDsFromEC3/EPD_AllOfEC3` (external, absent); `data/processed/*.json` when not regenerating; `src/dct_metriclabels.json` | `data/processed/DATA_all.json`, `dct_realeccs_trimmed.json`, `datasets_outliers.json`, `datasets_trimto10k.json`, `combos.txt`; `outputs/tables/TABLE_EmpiricalECCMetrics.xlsx`; 2 figures | Generation + metadata ~0.6 min for 15,000 (the in-code comment claiming ~6 min is stale) |
| `notebooks/02_CompareUQ_AnalyzeData.ipynb` (69 cells) | Fits 6 PEWT models to all 10,000 synthetic and 138 empirical datasets; scores each by W1, W2 and KS; ranks; correlates metrics against fit quality | `data/processed/*`, `outputs/tables/TABLE_EmpiricalECCMetrics.xlsx`, `src/dct_metriclabels.json` | `outputs/tables/TABLE_SyntheticECCMetricsAndW1.xlsx`, `TABLE_EmpiricalECCMetricsAndW1.xlsx`; 6 figures | Fit+score ~0.5 min; dominated by figure rasterization |
| `notebooks/03_CompareUQ_PerformPLCA.ipynb` (81 cells) | Runs 2,500 four-material pLCAs under each of 6 PEWT methods; computes ECI, rank frequency, uncertainty index, ECC-cap and material-reduction results; compares across methods by NRMSE | `data/processed/*`, `src/dct_metriclabels.json` | 15 figures. **No tables.** | pLCA loop ~3 min; figures dominate |
| `src/customstats.py` (873 lines, 18 functions) | Weighted statistics and fitting: `weighted_lognorm_fit`, `shapiro_wilk_weighted`, `_royston_pvalue`, `empirical_metadata`, `weighted_ecdf`, `estimate_maxima`, `weighted_skew`, `weighted_kurtosis`, `wasserstein1_weighted`, `wasserstein2_weighted`, `weighted_mean`, `weighted_var`, `weighted_distance_norm`, `weighted_quantile`, `weighted_bw`, `weighted_std`, `bw_dirichlet`, `NestedDictValues` | - | - |
| `src/datageneration.py` (208 lines) | `random_irregular_dataset`, `generate_random_numbers`, `random_logcount`, `random_samples` | - | - |
| `src/datavisualization.py` (32 lines) | One function, `scale_lightness`. 26 of 32 lines are empty separator comments. | - | - |
| `src/funcs_unit_conversion.py` (17.6 kB) | EC3 unit normalization across area, density, volume, weight, emissions, length, pressure, thermal resistance, time | - | - |
| `src/dct_metriclabels.json` | Display labels for the 20 metrics | - | - |
| `src/VOID_dct_metriclabels.json` | Dead, superseded by the above | - | - |
| `notebooks/VOID_CompareUQMethods_wbLCA.ipynb` (193 cells) | Dead, gitignored | - | - |
| `notebooks/CompareUQMethods_backup.ipynb` (154 cells) | Dead, gitignored | - | - |
| `notebooks/_SCRATCHPAD.ipynb` (12 cells) | Dead, gitignored | - | - |

Dataset counts confirmed from the files on disk: 15,000 generated, 4,131
flagged as metric outliers, 869 further trimmed, 10,000 retained, 138 empirical,
2,500 combos, 1,446,833 total synthetic values, dataset size n from 4 to 749
(median 62).

### 3.3 Dependency map

```
  EC3 CSV dump (../../EPDsFromEC3/, EXTERNAL, NOT PRESENT)
        |  only when generate_dontread = True
        v
  NB1 cell 12 ---> dct_realeccs_trimmed.json ---> TABLE_EmpiricalECCMetrics.xlsx
        |                     |                            |
  NB1 cell 18 ---> DATA_all.json (122 MB, GITIGNORED)      |
        |                     |                            |
  NB1 cells 20-22 ---> datasets_outliers.json              |
                       datasets_trimto10k.json             |
  NB1 cell 24 ---> combos.txt                              |
        |                     |                            |
        +---------------------+----------------------------+
                              v
                 NB2 (independent of NB3)      NB3 (independent of NB2)
                   fits + W1/W2/KS               fits + pLCA
                   -> 2 XLSX, 6 figures          -> 15 figures, NO TABLE
```

Hard ordering: NB1 must run before NB2 and NB3. NB2 and NB3 are independent of
each other; NB3 refits the models itself (NB3 cell 15 duplicates NB2 cell 12
verbatim). Within NB1, cell 12 must precede cell 27 (which needs
`df_realeccmetrics`), and cells 18-23 must run in order.

### 3.4 Duplication, dead code, and execution-order dependence

- **Model fitting block duplicated three times verbatim**: NB2 cell 12, NB2
  cell 20, NB3 cell 15. Any change to the fitting method must be made in three
  places.
- **`df_metrics` rebuilt from scratch three times in NB2** (cells 9, 50, 51),
  identical code each time.
- **`rankth` ordinal dictionary defined identically in all three notebooks.**
- **`PEWT` / `dct_colors` construction duplicated** in NB2 cell 12 and NB3
  cells 5 and 15.
- **NB3 cells 17 and 18 are byte-identical apart from one dataset name**, and
  both are annotated as being for the PhD defense, not the manuscript.
- **NB2 cannot run from a clean kernel.** Cell 65 uses the variable `metrics`,
  which is only ever assigned inside cell 62, and cell 62 is entirely commented
  out. Running NB2 top to bottom raises `NameError` at cell 65, which is the
  cell that writes
  `CompareUQMethods_SUPP_WassDistanceVsMetric_ALLMETRICS.png`.
- **NB3 cell 21 references an undefined `W`**: `for dataset, w in zip(datasets, W)`.
  `W` survives from the fitting loop in cell 15, where it is the weight array of
  the last dataset processed. `w` is never used in the loop body, so the result
  is currently correct, but `zip` silently truncates: if that leftover array had
  fewer than 4 elements, materials would be dropped from the pLCA without any
  error. The minimum dataset size is 4, so this has not yet fired.
- **Leaked loop variables used as real inputs**: `dataset` and `pewt` leak out
  of loops and are then read by later cells (NB2 cells 27, 31, 50, 51; NB3
  cell 25). Results depend on which cell was executed last.
- **Hardcoded relative paths throughout** (`'../src'`, `'../data/processed/...'`,
  `'../outputs/figures/...'`), so every notebook only runs with the working
  directory set to `notebooks/`. Also `sys.path.insert(0, '../../shared')`,
  a directory that does not exist in this repository.
- **Hardcoded external data path** `'../../EPDsFromEC3/EPD_AllOfEC3'`, outside
  the repository and not present.
- **`generate_dontread` flag** (NB1, NB3) switches between regenerating and
  reading cached data. It is a module-level boolean edited by hand; there is no
  record in the outputs of which mode produced them.
- **Two bare `except:` blocks in NB1 cell 12** swallow every error during the
  EC3 pull and the metrics computation, silently dropping materials.
- **15 orphaned figures** sit in `outputs/figures/` that no current notebook
  writes, from two earlier naming generations (`FIG1_`..`FIG7_`,
  `Supplement2_`..`Supplement5ii_`). The README's "Key outputs" lists the
  orphaned names, not the ones the notebooks now produce.
- `outputs/figures/` is 267 MB for 34 PNGs because everything is saved at
  `dpi=1200`; the largest single figure is 37 MB.

### 3.5 Runtime bottlenecks, measured

Measured on this machine (2026 laptop, Python 3.12, pandas 2.x), total compute
across all three notebooks is on the order of 20 to 35 minutes. The full NB3
cell 21 pLCA loop, reproduced verbatim, runs at 0.08 s per combo at
`neccs=1000` (3.5 min for 2,500 combos) and 0.25 s per combo at `neccs=10000`
(10.2 min). NB3 cell 25's growing-DataFrame append is linear, not quadratic, in
this pandas version: about 25 s in total.

**This contradicts the author's recollection that NB3 took extremely long, and
that discrepancy is unresolved.** The measurements above were taken on a
different machine and a newer software stack than the one that produced the
published results, and the original `waterweed` environment no longer exists to
test against. The most likely explanations are (a) an older pandas, where
`df.loc[len(df)] = row` and scalar `.loc` assignment on object-dtype frames
were genuinely quadratic rather than linear, which would turn cell 25 and
`compare_results` from seconds into tens of minutes; (b) the previous laptop;
(c) earlier runs at a larger `neccs` or over all 15,000 datasets. Figure
rasterization at dpi=1200 is the one cost that is real on any machine.

The practical consequence: **do not treat these timings as the baseline.** The
first action in Stage 1 is a single timed end-to-end run on the author's
current laptop in the new pinned environment, recorded per cell. Optimization
targets should be set from that, not from this section. The refactor is
justified by correctness and reproducibility regardless of what the timings
say.

The costs, ranked as measured here:

1. **Figure rasterization at dpi=1200 is the single largest cost.** NB3
   cell 39 builds one 6x7 panel figure in which each of 38 panels scatters 30
   pairwise combinations of 10,000 points: 11.4 million points onto a 136
   megapixel canvas. Measured 6.3 s for that one figure at dpi=1200 versus
   1.7 s at dpi=300, producing a 30 MB PNG. Cause: publication dpi applied to
   exploratory multi-panel figures, and overplotting instead of
   summarizing.
2. **`compare_results()` (NB3 cell 33) rebuilds its entire input frame on every
   call**: 60,000 scalar `.loc[dataset, pewt] = value` writes into an
   object-dtype DataFrame, measured 0.5 s per call against 0.01 s for the
   vectorized equivalent, a 50x penalty. It is called about 44 times across
   cells 33, 35, 37, 39 and 45, re-deriving the same frame each time. Cause:
   scalar indexed assignment in a triple loop, and no caching.
3. **Row-by-row DataFrame growth**, `df.loc[key] = dict`, used in NB1 cell 19
   and NB2 cells 9, 50 and 51 (10,000 rows each), and
   `dct_allresults[pewt].loc[len(...)] = ...` in NB3 cell 25 (60,000 rows).
   Measured 3 s versus 0.04 s for the vectorized construction, a 75x penalty.
4. **Redundant reductions inside per-dataset loops** in NB3 cell 21:
   `df_eci.mean()`, `df_eci.std()` and `.sort_values()` are recomputed in full
   for each of the four datasets, roughly 8 redundant full-frame reductions per
   combo per method, 15,000 times.
5. **Python-level filtering in the sampling hot loop**:
   `np.array([ele for ele in eccs if ele > 0])` appears six times in NB3 where
   `eccs[eccs > 0]` would do the same work in C.
6. **`gaussian_kde.evaluate` on a 1,000-point grid** inside `estimate_maxima`
   is the top cost within `empirical_metadata` (measured 2.1 ms per dataset
   overall). This one is inherent to the mode-count metric, not waste.
7. **Embarrassingly parallel work run serially.** The 15,000 dataset
   generations, the 10,000 model fits, and the 2,500 pLCA combos are fully
   independent and single-threaded.

### 3.6 Where randomness enters, and whether it is seeded

| Location | Call | Seeded? |
|---|---|---|
| `datageneration.py:19` `generate_random_numbers` | `np.random.default_rng(seed)` with **`seed=0` as the default** | Seeded, but see 3.7 - this is a defect, not reproducibility |
| `datageneration.py:35` lognormal branch | `stats.lognorm.rvs(...)` with **no `random_state`** | Unseeded, uses global numpy state |
| `datageneration.py:78` `random_irregular_dataset` | `np.random.default_rng(seed)`, called from NB1 with `seed=None` | **Unseeded** (fresh OS entropy) |
| `datageneration.py:170` | `np.random.dirichlet(...)` for the per-point weights | **Unseeded**, global state |
| `datageneration.py:192` `random_logcount` | `np.random.uniform(...)` | **Unseeded**, global state |
| NB1 cell 10 | `np.random.seed(42)` | The only seed in the project, and it governs a figure, not the data |
| NB1 cell 12 | `np.random.dirichlet` for empirical ECC weights | **Unseeded** |
| NB1 cell 24 | `np.random.shuffle(combos)` | **Unseeded** |
| NB2 cells 13, 15 | `np.random.dirichlet`, `np.random.uniform` in figure cells | **Unseeded** |
| NB3 cells 17, 18, 21, 73 | `func.rvs(n)` and `kde.resample(n)`, all Monte Carlo draws | **Unseeded**, global state |
| NB3 cell 37 | `np.random.uniform(-0.25, 0.25)` jitter | **Unseeded** |

Conclusion: no part of the pipeline is reproducible from scratch. The results
are stable only because `DATA_all.json`, `dct_realeccs_trimmed.json` and
`combos.txt` were cached to disk. `DATA_all.json` is gitignored, so a fresh
clone of this repository can neither read the synthetic datasets nor regenerate
them to the same values. This violates the standing constraint that all
randomness come from an explicitly passed Generator.

### 3.7 Synthetic ECC dataset generation, independent description

Per dataset, `NB1 cell 18` draws `n = random_logcount(lo=4, hi=1000)`, a value
log-uniform on [4, 1000] rounded to an integer, then calls
`random_irregular_dataset(n=n)`, which does the following.

1. `rng = np.random.default_rng(seed)` with `seed=None`, so fresh entropy.
2. `k = rng.integers(1, 6)`, giving 1 to 5 mixture components. (The inline
   comment says "1-6 modes"; it is 1 to 5.)
3. `locs = rng.uniform(5, 20, k)` and `scales = rng.uniform(0.2, 1.5, k)`.
4. `weights = rng.dirichlet(np.ones(k) * 10)`. The concentration 10 is flat but
   fairly tight, so components tend toward comparable weights. An earlier
   scheme that tied concentration to `1/locs` is commented out just above.
5. `types = rng.choice(["gauss","skewnorm","studentt","lognorm"], size=k,
   p=[0.40, 0.25, 0.25, 0.10])`.
6. `counts = rng.multinomial(n, weights)`.
7. For each component, `generate_random_numbers(t, loc, scale, cnt)` is called
   **without a seed argument, so it uses the default `seed=0`**.
8. Extreme values are trimmed to `[max(0, Q1 - 3*IQR), Q3 + 3*IQR]`, computed
   once from the initial pooled sample. A `while` loop then rejects out-of-range
   values and redraws from a randomly chosen component until exactly `n` values
   remain in range.
9. **Variance inflation**: `exp = rng.uniform(0.9, 4.0)` and every value is
   raised to that power, `data = np.array([ele**exp for ele in data])`. With
   locations in [5, 20] and exponents up to 4, this maps values as high as
   20 to 160,000, and it is what supplies the heavy right skew. The comment
   says the purpose is "to align with empirical ECC data".
10. **Reflection**: with probability 0.25, `data = max(data) - data + min(data)`,
    producing left-skewed datasets.
11. Weights: `np.random.dirichlet(np.ones_like(data))`, a flat Dirichlet over
    the `n` points, drawn from **global** numpy state, not from `rng`.
12. Normalization: `data = data + 1` then `data = data / np.mean(data)`.

Two observations I want to raise before Stage 2a asks about them.

**(a) The normalization is by the unweighted mean, not the weighted mean.**
Step 12 divides by `np.mean(data)`, so the *unweighted* mean is exactly 1.0 and
the weighted mean is not. The same is true of the empirical path (NB1 cell 12,
`data = data/np.mean(data)`). The project brief states each dataset is
"normalized to a weighted mean of 1.0". The code does not do that. This is
visible in the metrics file: `mean_uw` is 1.0 for every dataset to floating
point precision, while `mean` varies with an interquartile range of about
0.978 to 1.022.

**(b) `generate_random_numbers` defaults to `seed=0`, which collapses the
diversity of the generated data.** Because `rng = np.random.default_rng(0)` is
constructed fresh on every call, two things follow, both verified empirically:

- The shape parameters are constants across the entire study, not random
  draws. Every skew-normal component has `a = 3.458732`, every Student-t
  component has `df = 3.366328`, and every lognormal component has
  `s = 1.136962`. The `rng.uniform` calls that appear to randomize them return
  the same first variate every time.
- The component samples themselves are deterministic given `(type, count)`.
  For the Gaussian branch, `samples = loc + scale * Z` where `Z` is one fixed
  standard normal vector determined only by `cnt`. Two Gaussian components with
  entirely different `loc` and `scale` are exact affine images of each other:
  measured Pearson correlation 1.0, and standardized values agreeing to 9e-15.

So the mixture components are not 15,000 independent random shapes. The
available library of standardized component shapes is 4 types times the number
of distinct counts, and all diversity between datasets comes from `k`, the
locations, the scales, the mixture weights, the multinomial counts, the
variance-inflation exponent, the reflection coin, and the rejection loop. The
lognormal branch is the sole exception, because it omits `random_state=rng` and
therefore draws from global numpy state, making it the only genuinely random
component type.

I have not yet quantified how much this reduces the effective coverage of the
metric space. That belongs in Stage 2a.

### 3.8 `src/customstats.py` inventory and findings

Correct and unremarkable: `weighted_mean`, `weighted_var`, `weighted_std`,
`weighted_ecdf`, `wasserstein1_weighted` (delegates to
`scipy.stats.wasserstein_distance`), `wasserstein2_weighted` (explicit
quantile-grid integration, correct), `weighted_lognorm_fit` (weighted MLE in
log space with method-of-moments initialization), `NestedDictValues`.

The bias corrections in `weighted_skew` and `weighted_kurtosis` were checked
algebraically and both reduce to the standard G1 and G2 estimators.

Findings, in descending order of consequence:

1. **`weighted_quantile` (line 677) is order-dependent and wrong for unsorted
   input.** It sorts `(X, W)` pairs by value, then overwrites the cumulative
   sum with `cdf[i+1] = cdf[i] + W[i]`, indexing the *original, unsorted*
   weight array against the *sorted* values. On one 8-point example the
   returned 25th percentile was 4.03 against a correct 1.05. Sweeping 2,000
   random lognormal datasets, the Silverman bandwidth computed from it differs
   between sorted and unsorted presentations of the same weighted sample in
   93.1% of cases, median relative error 10.6%, maximum 172%.
   **This does not currently move any published number**, because the only
   bandwidth path the analysis uses is `bw_method='scott'`, which uses only the
   standard deviation and never calls `weighted_quantile`. It is a live trap
   for Stage 2: the moment the bandwidth rule is switched to Silverman for
   consistency with KL2, this bug starts driving every KDE fit. It must be
   fixed before that switch, not after.
2. **The bandwidth rules are correctly named; only cross-referencing scipy is
   a hazard.** `weighted_bw` implements `'scott'` as `1.06 * std * n_eff**-0.2`
   and `'silverman'` as `0.9 * min(std, IQR/1.34) * n_eff**-0.2`. Both match
   the standard convention: Scott (1992) gives `1.059 * sigma * n**(-1/5)`, and
   Silverman's robust rule of thumb is Silverman (1986) eq. 3.31. The
   implementation and the README are right, and an earlier draft of this
   handoff wrongly implied otherwise.

   The one live hazard is that `scipy.stats.gaussian_kde` uses these two words
   for different formulas: its `bw_method='scott'` is `sigma * n**(-1/5)` with
   no constant, and its `'silverman'` is `sigma * (3n/4)**(-1/5)`, which is
   about `1.06 * sigma * n**(-1/5)`. So scipy's `'silverman'` is numerically
   this project's `'scott'`. The code sidesteps this by computing the bandwidth
   itself and passing `bw_method=1.0` before calling `set_bandwidth`, which is
   correct. The trap is documentation and review: never describe the method by
   pointing at a scipy keyword, and never compare this bandwidth to a scipy
   default without converting.

   Relevant to the KL1/KL2 divergence: on skewed data `0.9 * IQR/1.34` and
   `1.06 * std` can differ substantially, so switching to Silverman for KL2
   consistency is not a cosmetic change.
3. **`shapiro_wilk_weighted` silently switches estimator between the weighted
   and unweighted columns.** With uniform weights it returns
   `scipy.stats.shapiro`, the true Shapiro-Wilk W. With non-uniform weights it
   returns a Shapiro-Francia statistic, the squared weighted correlation with
   normal scores. `empirical_metadata` therefore produces `fit_norm_SW` and
   `fit_norm_SW_uw` from two different estimators, and the same for the
   lognormal pair. Comparing a column against its `_uw` counterpart, which the
   metric-correlation analysis does, compares two different statistics. The
   docstring states the equivalence holds "for n >= 20"; the median dataset
   size here is 62 but the minimum is 4.
4. **`_royston_pvalue` uses the wrong branch for 4 <= n <= 11.** Royston (1992)
   specifies distinct polynomials in `n` for the small-sample case and the
   transform `-log(gamma - log(1-W))`. The implementation instead reuses the
   `n >= 12` polynomials in `log(n)` and applies `gamma` as a plain additive
   shift. p-values for small datasets are therefore unreliable. No published
   number depends on this, because `empirical_metadata` keeps only
   `shapiro_wilk_weighted(...)[0]`, the statistic, and discards the p-value.
5. **`weighted_skew` and `weighted_kurtosis` assume weights already sum to 1**
   (`mean = np.sum(data*weights)` with no division by `sum(weights)`) and they
   apply the bias correction using the raw `n` rather than an effective sample
   size. `empirical_metadata` does normalize before calling them, so current
   results are unaffected, but either function called directly with raw weights
   returns silently wrong values.
6. **`empirical_metadata` can return non-finite quartiles.**
   `q1, q3 = np.interp([0.25, 0.75], ycdf, xcdf)` interpolates against an
   `xcdf` whose first and last entries are `-inf` and `+inf`. If the smallest
   data point carries more than 25% of the weight, which a flat Dirichlet on a
   small `n` can produce, `q1` interpolates into `-inf`.
7. **`wasserstein2_weighted(unitless=True)` divides by `std**2`** where W2 has
   the units of the data, so the result is not dimensionless. `unitless=True`
   is not used in the current analysis.
8. **`estimate_maxima` does not return a mode count.** It returns
   `(sum of local maxima heights - sum of local minima heights) / max height`,
   a continuous modality index. The metric is labelled "Mode Count" in
   `dct_metriclabels.json`, in the README, and in the project brief.
9. **`bw_dirichlet` is dead in this repository.** It implements the KL2-style
   variable-bandwidth scheme and is never imported by any notebook.
10. `weighted_distance_norm` computes AD, CvM, W1, W2, ED and KS against a
    normal fit. It is imported by all three notebooks and called by none; the
    block in `empirical_metadata` that used it is commented out.

### 3.9 Other findings that bear on the manuscript

- **NB3 writes no results table.** All 15 of its figures, including the
  headline "ECI Rank #1 Frequency" result, are drawn from the in-memory
  `results_meta` dictionary, which is never persisted. This directly violates
  the standing constraint that every analysis write a tidy table to disk and
  that figures be generated from those tables. Combined with 3.6, the pLCA
  results cannot currently be reproduced or re-plotted at all.
- **The pLCA uses `neccs = 1000`, not 10,000.** NB3 cell 21, the loop that
  produces every manuscript pLCA result, sets `neccs = 1000`. `nsamples =
  10_000` appears only in cells 17 and 18, which are annotated as being for the
  PhD defense rather than the manuscript. The project brief states the pLCA is
  run at n = 10,000. Either the brief and manuscript are wrong, or the
  headline results are at a tenth of the stated Monte Carlo sample size. This
  needs resolving before anything else in Stage 2.
- **Monte Carlo draws are independent across UQ methods and across materials.**
  Each PEWT method draws its own samples for each dataset. Differences reported
  between methods therefore carry independent Monte Carlo noise rather than
  being compared on common random numbers. Both Marsh, Lewis, Hattam and Allen
  (in press) and Henriksson et al. (2015) use dependent sampling for exactly
  this comparison. Flagged for Stage 2e.
- **The outlier filter discards 4,131 of 15,000 datasets (27.5%)**, and it is
  defined by the IQR of the *synthetic* metric distribution, widened to
  `max(q3-q1, 1.35*std)`, not by the empirical ranges. Its effects are not
  neutral: it removes 1,061 datasets on `weight_outliers` and 824 on `n`. The
  `n` cutoff lands at 749, so although generation draws sizes up to 1,000, no
  dataset larger than 749 survives. Filtering on `weight_outliers`
  preferentially discards the datasets where weighting matters most, which are
  the cases the paper is about.
- **14 datasets are discarded for floating-point noise.** `mean_uw` is
  identically 1.0 by construction, so its IQR and standard deviation are both
  0 and the cutoffs collapse to exactly 1.0. Fourteen datasets whose `mean_uw`
  differs from 1.0 in the 16th decimal place are flagged as outliers on that
  basis alone.
- **The lognormal fit is applied to `X + 0.5` and then shifted back**
  (`logfit_offset = 0.5`, NB2 cells 12 and 20, NB3 cell 15). This is an
  undocumented methodological choice that materially changes the lognormal fit,
  and it is not mentioned in the README.
- **`results_plca[dataset]['wbeci_mean']` and `['wbeci_stdev']` are assigned
  outside the per-dataset loop** in NB3 cell 21, so they are recorded for only
  one of the four datasets in each combo.
- **The `capecc` rank frequencies are divided by an extra `(1-capecc)` factor**
  with the inline comment "Percentages look off because not all reduction
  strategies apply in all scenarios". This is an ad hoc normalization.
- **The environment is gone.** All three notebooks record kernel
  `waterweed`, Python 3.11.9/3.11.15. No such conda environment exists on this
  machine, and the repository pins no versions: `setup.py` declares no
  dependencies and there is no lockfile, `requirements.txt` or
  `environment.yml`. The README lists packages without versions.
- **The README documents output filenames that are no longer produced**
  (`FIG1_`..`FIG7_`), and its figure list does not match what the notebooks
  write.
- **The manuscript `.docx` carries 98 unresolved comments from the author's
  advisor, and this repository is public and Zenodo-archived.** Committing the
  file as it stands would publish a named third party's private editorial
  feedback, permanently and without their consent, and git history would retain
  it even after a later deletion. The file should be gitignored until the
  comments are resolved and accepted, and only a clean copy committed, if any
  copy is committed at all. This is flagged for decision, not acted on.

### 3.10 Citation check

The manuscript was added by the author on 2026-09-11, after this inventory was
first written, at `outputs/manuscript/2026-09-10_Manuscript_CompareUQMethods.docx`
(15,314 words, 98 unresolved advisor comments). The citation pass over it has
**not** been performed: by the author's instruction the analysis is to be
settled first, and the manuscript revised afterwards in light of whatever the
analysis changes. The "Torres et al. (in press)" to RC&R 234, 109022 update is
therefore carried forward as a manuscript-stage task, together with a sweep for
any other placeholder or in-press citations.

Within the repository, the only citation-like strings are in `README.md`:
- line 167, the present paper cited as "(submitted)". Still accurate while in
  revision, but it should become the Building and Environment citation on
  acceptance.
- lines 8 and 169, the Zenodo DOI 10.5281/zenodo.19226429.
The README does not cite KL1 or KL2 at all, although it describes the KDE
method that KL1 introduced. No `TODO`, `FIXME` or other placeholder markers
exist in any source file or notebook code cell.

## 4. Numbers that moved

None. No analysis logic, data file, table or figure was modified in this stage.
The only committed changes are `.gitignore`, `CLAUDE.md` and this handoff file.

## 5. Open questions and flags

Decisions taken (author, 2026-09-11), superseding the open questions as first
written:

1. **The pLCA runs at 10,000 draws.** `neccs` moves from 1,000 to 10,000.
   Every pLCA number in the manuscript will move. Measured cost of the change
   is about 10 min for all 2,500 combos (see 3.5), so it is affordable; if a
   later change makes it prohibitive, reassess.
2. **Normalization is by the weighted mean,** because the weighted values are
   the synthesized ground truth. Both the synthetic path
   (`datageneration.random_irregular_dataset`) and the empirical path (NB1
   cell 12) change from `np.mean(data)` to the weighted mean. This moves every
   metric and every W1 in the paper.
3. **Bandwidth naming corrected** (3.8 item 2). The implementation was right.
4. **A dedicated conda environment will be created for this repository** in
   Stage 1, with pinned versions, replacing the lost `waterweed` kernel.
5. **`tqdm` is to be removed.** It dates from a pre-VS-Code workflow.
6. **The manuscript is now in the repository** at
   `outputs/manuscript/2026-09-10_Manuscript_CompareUQMethods.docx`: 15,314
   words with 98 unresolved comments from the author's advisor. The analysis is
   to be straightened out first; the manuscript is revisited afterwards, in
   light of whatever the analysis changes. **See the flag in 3.9 about
   publishing advisor comments in a public repository.**

Still blocking before code is touched:

7. **Approval of the refactor plan.**

Non-blocking, carried into Stage 2:

4. Bandwidth rule: this analysis uses `1.06 * std * n_eff**-0.2`; KL2 uses
   Silverman's rule and justifies it. `weighted_quantile` must be fixed before
   any switch (3.8 item 1). Stage 2a/2h.
5. The `seed=0` collapse in `generate_random_numbers` (3.7b). Whether the
   synthetic datasets need regenerating is a Stage 2a decision, and
   regenerating them moves every number in the paper.
6. Dependent versus independent sampling across compared UQ methods, against
   Marsh et al. (in press) and Henriksson et al. (2015). Stage 2e.
7. The outlier filter's effect on the coverage claim, and its preferential
   removal of high-`weight_outliers` datasets. Stage 2a.
8. `logfit_offset = 0.5` needs a justification in the manuscript or removal.
9. "Mode Count" should be renamed to a modality index, or the metric changed to
   an actual count.
10. Overlap area alongside W1 as a robustness check, per Prado-Lopez et al.
    (2014). Stage 2c.

## 6. Inputs and outputs

Read: all files under `notebooks/`, `src/`, `data/processed/`,
`outputs/`, plus `README.md`, `setup.py`, `.gitignore`, and the git history.
No reference PDF was opened in this stage.

Written:
- `.gitignore` (added `refs/`)
- `CLAUDE.md` (new)
- `reports/HANDOFF_stage-0.md` (this file)

Nothing under `data/`, `outputs/`, `src/` or `notebooks/` was modified.

Added by the author during this stage, not by this stage's work:
`outputs/manuscript/2026-09-10_Manuscript_CompareUQMethods.docx`.

## 7. Next stage

Stage 1 is the refactor, and it must not begin until the plan is approved and
questions 1 and 2 above are answered.

The first task in Stage 1, before any restructuring, is to establish a
regression baseline: pin the current environment, capture the existing
`TABLE_SyntheticECCMetricsAndW1.xlsx` and
`TABLE_EmpiricalECCMetricsAndW1.xlsx` as reference fixtures, and add a
characterization test that reproduces them to floating point tolerance. Without
that, the constraint "never silently change a result" cannot be enforced,
because there is currently no artifact recording what the pLCA results are.

The second task is to persist the pLCA results from NB3 to a tidy table, since
they presently exist nowhere on disk.
