# HANDOFF stage-2a - Generator audit, and the regeneration

## 1. Stage and branch

- Stage: 2a, generator audit
- Branch: `stage-2a-generator`
- Branched from: `5906849` "Tag decision log by provenance; add an
  irreplaceable-input manifest", on `stage-0-1-refactor`

| Commit | Label |
|---|---|
| `90a42a8` Freeze the pre-regeneration baseline | NEUTRAL |
| `f87c4b4` Part 0: measure the seeding collapse | MEASUREMENT ONLY |
| `e24adca` Part 7: audit the empirical data at the source | MEASUREMENT ONLY |
| `4d10ae9` Moment-targeted component families | no numbers moved yet |
| `886bb7c` Truncated-mixture parent and Silverman modality | no numbers moved yet |
| `a8b16c2` Configuration-driven generator | no numbers moved yet |
| `30d4124` Corpus driver, retire generate_dontread, two parent fixes | no numbers moved yet |
| `2325891` Point the notebooks at the named corpus | **MOVES NUMBERS** |
| `bb07df8` Prove the metric rename moved no value; 9 discrepancies | NEUTRAL |
| `bfd9185` Spread as a solved target; 10x faster; four defects | **MOVES NUMBERS** |
| `75c1e51` Coverage analysis into src/ and notebook 1 | NEUTRAL |
| `548b900` Centre the corpus on the empirical data | **MOVES NUMBERS** |

## 2. What was asked

Audit the synthetic dataset generator: confirm and fix four known defects,
close the regeneration-hygiene item Stage 1 left open, document and then
simplify the generator, drop the undocumented selection step, move market share
to the mode level, reconcile code against manuscript, replace manual tweaking
with a configuration, report coverage and produce Table 1, and audit the
empirical data at source. Then regenerate, once.

## 3. What was done

### 3.1 Part 0b, regeneration hygiene, done first

The five input files and three pLCA artifacts named in `data/INPUTS.sha256`
were copied to `data/baseline_frozen/` (gitignored, 161 MB) before anything
else ran. Both the originals and the copies verify 8/8 OK against the manifest.
Nothing in this stage overwrote any of them.

`generate_dontread` is gone. A corpus is now a named, dated directory carrying
its own `runmeta.json` with the seed, the full configuration, the git commit,
library versions, platform and the counts of everything retried or rejected.
`generate_corpus` refuses to write into an existing directory. The notebooks
only read, through the tracked pointer `data/processed/CORPUS.json`.

### 3.2 Part 0, the known defects

**The seeding collapse, quantified.** The frozen pre-Stage-1 generator was
recovered from git and run, rather than reasoned about. All three shape
parameters are constants to six decimals (skew-normal a = 3.458732, Student-t
df = 3.366328, lognormal s = 1.136962); gauss, skew-normal and Student-t draws
repeat bit-identically for a given (type, count); only the lognormal branch was
random.

Damage to the shipped corpus was measured by clustering datasets on identical
value ORDER. Every step after the component draw is monotone, so a Spearman
correlation of 1 between two datasets is the defect, not a coincidence.

| Arm | Duplicate pairs | Datasets in a cluster | Largest cluster | Effective size | Redundancy |
|---|---|---|---|---|---|
| shipped, 10,000 analysed | 2,891 | 1,379 | 18 | 9,026 | **9.74%** |
| legacy rerun, 15,000 | 5,119 | 1,727 | 29 | 13,740 | 8.40% |
| current generator, 15,000 | **0** of 378,945 tested | 0 | 1 | 15,000 | **0.00%** |

**Is correct seeding sufficient?** This is the question Stage 0 left open. An
ablation with values drawn correctly but the shape parameters pinned at the
legacy constants recovers independence and buys almost NO coverage: every
metric spread within 5 percent of legacy, effective metric dimension 5.70 to
5.78. Drawing the shape parameters per component is what widens the space:
skewness range +22 percent, kurtosis +65 percent, coefficient of variation
+13 percent. **Both are necessary.** Stage 1's single change delivered both,
but they are logically distinct and the ablation separates them.

**Dirichlet concentration.** Fixed to alpha = 1 in both arms. See section 4.

**The +1 buffer.** Removed. It compressed the coefficient of variation by more
than 1 percent in 28.8 percent of datasets and more than 10 percent in 0.9
percent, concentrated in low-mean datasets. Its real role was holding values
away from zero: mean minimum over mean 0.364 with it against 0.358 without, and
at the 1st percentile 0.00025 against 0.00002.

**Stale docstrings.** Already fixed by Stage 1 commit `533f6c4`. Confirmed and
not redone.

### 3.3 Parts 1, 2, 3 and 5, the generator

Four new modules, all tested:

- `src/components.py`. A component is a TARGET (mean, sd, skewness, excess
  kurtosis) that is solved for. Four families tile the feasible plane along the
  Pearson system's own partition: Johnson SU above the lognormal line,
  lognormal on it, beta-prime between the gamma and lognormal lines, beta below
  the gamma line. Accuracy over 800 random targets: skewness to 4.6e-13,
  kurtosis to 2.3e-14 relative. Infeasible and numerically degenerate targets
  are reported, never approximated.
- `src/mixture.py`. The parent: a mixture, truncated at POPULATION quantiles,
  renormalized, divided by the recorded sample mean. `cdf`, `pdf` and `ppf` are
  exact. Verified against 400,000 draws through the real sampler, and across
  all four strata against the Kolmogorov noise floor.
- `src/modality.py`. Silverman's critical bandwidth, with a binned FFT KDE that
  makes each evaluation independent of n.
- `src/genconfig.py` and `src/generator.py`. Every parameter in one dataclass;
  nothing tuned inside the code.

**Three steps removed, as decided.** The truncation loop (which also kept
`data[:n]` after filtering, preferentially discarding the LATER components in
concatenation order), the power transform, and the sample-dependent reflection.

**The parent CDF is the deliverable and it verifies.** Ratio of observed
Kolmogorov distance to its own noise floor, by stratum: median 0.55 to 0.62,
95th percentile 0.94 to 1.02. Truncation removes a median of 0.0013 percent of
probability, mean 1.5 percent, 95th percentile 6.3 percent, maximum 45.8
percent.

**Part 2, the filter.** Dropped. Measured: `weight_outliers` flagged more
datasets than any other metric (1,061), and the removed set averaged 0.0755
against 0.0146 for the kept set, 0.83 pooled sd. It flagged 824 datasets on `n`
alone, all with n >= 750, which is what capped the analysed maximum at 749. It
flagged 14 datasets on `mean_uw`, a column that is 1.0 by construction. The
replacement rejects only for unanalysability: **0 of 10,050 datasets rejected**
in the final run.

**Part 3, market share.** Attaches at the mode level and is distributed within
a mode. The market-weighted parent has mode weights
`(1 - c) * pi_trunc + c * market`; at c = 0 it collapses exactly onto the
uniform-weighted parent, which states the circularity problem as an identity.
Both are tested against 400,000 draws.

### 3.4 Part 6, coverage and Table 1

Now in `src/coverage.py` and notebook 1, not in the audit scripts, because
Table 1 and the coverage figure are paper deliverables.

**The first two regenerations failed this test and were discarded.** See
section 4.3. The third is the one in use.

### 3.5 Part 7, the empirical data at source

- **Deduplication.** At EPD level there is nothing to deduplicate: 0 of 206,668
  records share an `open_xpd_uuid` inside their category. But **55.00 percent
  of records share a (manufacturer, GWP per kg) pair** with another record in
  the same category, across 105 of 138 categories, and the top manufacturer
  holds a median 18.4 percent of a category, up to 73.0 percent. The
  uniform-weighted empirical distribution is therefore already implicitly
  weighted, by how many EPDs each manufacturer published.
- **Industry-average EPDs.** None. All 206,668 records are Product EPDs; 98.5
  percent product-specific, 82.9 percent plant-specific.
- **Cleaning sensitivity.** The shipped rule drops 1.61 percent of records but
  moves `fit_norm_SW` by 1.55, `entropy` by 1.42, `modality_index` by 1.00 and
  `weight_outliers` by 0.77 standard deviations of the uncleaned metric.

## 4. Numbers that moved

**Every number in the paper moves.** The corpus was regenerated. What follows
is what moved and why, against the frozen baseline.

### 4.1 The empirical arm

| Metric | alpha = 5 (shipped) | alpha = 1 (now) | Shift |
|---|---|---|---|
| `w_v_uw_wasserstein` mean | 0.0594 | **0.1329** | **1.33 sd, 2.24x** |
| `kurtosis` mean | 1.328 | 2.576 | 0.98 sd |
| `weight_outliers` mean | 0.0482 | 0.0438 | 0.58 sd |
| `skewness` mean | 0.949 | 0.921 | 0.63 sd |

The first row is the paper's central quantity. Every published
empirical-versus-synthetic comparison of the weighting effect was made between
arms that were not comparable.

The multiplicative low-end cleaning removes 342 of 107,523 values (0.318
percent) across 51 of 138 datasets. ReadyMix's minimum moves from 3.1e-17 of
its mean to 0.265; datasets holding a value below 1 percent of their mean fall
from 39 to 12.

### 4.2 The metric set

`mode_count_est` renamed `modality_index`; `crit_bw_1` added. **No value
moved**: `tests/test_regression.py` maps the old name back and drops the new
columns before comparing, and all 8 regression tests pass against the Stage 1
fixtures at rtol 1e-6.

### 4.3 Two discarded regenerations, and why

Both are recorded because they are the evidence for the final configuration,
and because the corpus directories still exist on disk.

**`corpus_2026-09-11`** (31.7 min). Median coefficient of variation **0.049**
against an empirical 0.600. Cause: removing the power transform took away the
only thing in the generator that produced spread, and the mixture was placed by
shifting it above the 1e-9 quantile of its heaviest-tailed component, which put
the median dataset's support at 0.65 of its own mean.

**`corpus_2026-09-11b`** (17.5 min). Coverage of the empirical range 98.6 to
100 percent on every metric, but median coefficient of variation **0.071**.
Covering a range is not the requirement; sitting where the data sit is.

**`corpus_2026-09-11c`** (15.3 min) is the corpus in use. 10,000 datasets plus
50 probe datasets, **0 failed parents, 0 rejected by the validity filter**.
The fixes were to centre the target coefficient of variation on the empirical
distribution rather than draw it log-uniformly, and to cluster component
locations toward the low end (`position_skew = 5`), which is what actually
controls achievable spread.

**Final coverage of the 138 empirical datasets by the synthetic range:**

| Metric | Empirical min/median/max | Synthetic min/median/max | Covered |
|---|---|---|---|
| `coeffvar` | 0.011 / 0.600 / 2.083 | 0.000 / 0.330 / 8.205 | **100%** |
| `skewness` | -1.44 / 1.055 / 4.618 | -95.4 / 0.556 / 42.2 | **100%** |
| `kurtosis` | -5.45 / 1.160 / 62.7 | -6.00 / 0.660 / 2091.7 | **100%** |
| `entropy` | 0.789 / 2.937 / 4.940 | 0.005 / 2.472 / 5.405 | **100%** |
| `crit_bw_1` | 0.127 / 0.574 / 1.143 | 0.119 / 0.649 / 3.132 | **100%** |
| `weight_outliers` | 0.000 / 0.028 / 0.386 | 0.000 / 0.005 / 0.519 | **100%** |
| `fit_norm_SW` | 0.491 / 0.887 / 1.000 | 0.010 / 0.793 / 1.000 | **100%** |
| `fit_lognorm_SW` | 0.605 / 0.937 / 1.000 | 0.021 / 0.807 / 1.000 | **100%** |
| `w_v_uw_wasserstein` | 0.001 / 0.097 / 0.877 | 0.000 / 0.104 / 3.447 | **100%** |
| `n` | 3 / 35 / 77,439 | 3 / 99.5 / 9,996 | 99.3% |

The only uncovered dataset is ReadyMix at n = 77,439, which is what the probe
set exists to address. Effective metric dimension: empirical 4.67, synthetic
5.95, so the corpus is not collapsing the space.

**Probe set: results plateau above n = 10 ** 4 on 8 of 9 metrics.** Only
`crit_bw_1` sits outside two standard errors (-2.08). The coverage claim can be
stated as complete to 9,999 with stability above that established on the probe
set, with that one caveat.

### 4.4 Defects found while building, each of which would have corrupted results

1. **`ppf` bisection collapsed** instead of bracketing. Found by test.
2. **Components with no mass inside the truncation bounds** were kept with a
   near-zero mass, which then divided into `cdf` and `sample`. Two parents in a
   1,600-draw check had a population mean of 1e-12.
3. **Overlap by quadrature was wrong, not just slow.** On one pair it was 0.070
   in absolute probability from the truth while reporting the correct NUMBER of
   crossings, because a grid spanning every component at once gives a narrow
   component very few nodes. The quantization was not monotone in the component
   spread, which broke the solver: 23 percent 'tolerance_not_met'.
4. **`_pair_overlap`'s no-crossing branch** returned a winner-dependent 1.0 or
   0.0. Correct for one direction, wrong once both are summed.
5. **The CV probe truncated the UNSHIFTED mixture to SHIFTED bounds.** The
   solve missed its target by 59 percent at the median.
6. **`truncated_moments` computed variance about the origin**, so for large
   shifts it cancelled catastrophically. 31 percent of datasets were rejected
   as degenerate.
7. **The sample kurtosis bound used `n - 2`** instead of the sharp
   `(n**2 - 3n + 3)/(n - 1)`, rejecting 12 percent of valid stratum 1 datasets.
8. **A test's own brentq oracle was wrong**, and disagreed with correct code by
   0.36. Replaced by a brute-force count of the definition.

### 4.5 Performance

Generation went from **272 ms to 80 ms per dataset**, 32 minutes to 17.5 for
the full corpus. Profiling, not guesswork: 65 percent of the run was inside
`scipy.optimize.brentq`, called once per crossing, each iteration evaluating
two scipy pdfs on a one-element array. Overlap now scans the union of both
components' quantiles, refines every crossing with one vectorized bisection,
and computes both directions of a pair together.

## 5. Open questions and flags

### Carried forward

| Item | Owner | Status |
|---|---|---|
| Regeneration in Stage 2 | 2a | **RESOLVED.** Done, three times; the third is in use |
| `seed=0` collapse | 2a | **RESOLVED.** Quantified at 9.74 percent corpus redundancy; correct seeding is necessary but not sufficient, and per-component shape draws are what buy coverage |
| The 27.5 percent filter, n cap at 749 | 2a | **RESOLVED.** Dropped, replaced by a validity-only filter |
| "Mode Count" naming | 2a | **RESOLVED.** Renamed `modality_index`; `crit_bw_1` added |
| Variance-inflation exponent, reflection | 2a | **RESOLVED.** Both removed |
| Multiplicative cleaning filter | 2a | **RESOLVED.** Low end only; the high end cannot be re-cleaned |
| Bandwidth rule, KL1/KL2 inconsistency | 2h | STILL OPEN |
| `logfit_offset` | 2b, swept in 2h | STILL OPEN |
| Dependent sampling | 2e | STILL OPEN |
| Overlap area alongside W1 | 2c | STILL OPEN |
| Shapiro-Wilk vs Shapiro-Francia | 2f | STILL OPEN |
| `(1-capecc)` divisor | 2g | STILL OPEN |
| Scoring grid includes zero | 2c or 2e | STILL OPEN |
| W1 has no complexity penalty | 2c | STILL OPEN |
| `weighted_quantile` must stay fixed before Silverman in 2h | 2h | STILL OPEN |
| Entry 13, support (0, inf), needs author confirmation | - | **STILL OPEN.** Stage 2a built on it |

### New in Stage 2a

- **The empirical extraction cannot be reproduced.** The directory notebook 1's
  empirical branch reads no longer exists on this machine.
  `dct_realeccs_trimmed.json` is the only surviving record of the empirical arm
  and it is POST-cleaning. Affects the Zenodo deposit. Discrepancy entry 26.
- **The empirical "unweighted" baseline is already weighted** by publication
  frequency: 55 percent of records share a (manufacturer, GWP) pair. Entry 23.
- **The coefficient of variation still sits below the empirical median**, 0.377
  against 0.600, and the corpus cannot exceed about 2.1 against an empirical
  2.08. The `Q3 + 3*IQR` truncation caps how much right tail survives. Reported,
  not engineered away. **Owner: 2h**, as a sweep of `trunc_iqr_mult`.
- **About 35 percent of coefficient-of-variation targets are unreachable** for
  their mixture and are recorded as `clipped_max_cv`.
- **Kurtosis is undefined in 24.8 percent of stratum 1**, by construction.
  **Stage 2f must be told**, or its complete-case models drop the stratum
  silently.
- **The probe set does not fully plateau on skewness** (difference over
  standard error 2.62 in the 2026-09-11b run). Everything else is inside 2.
  **Owner: 2f/2g** when the claim is written.
- **`mode_share_alpha` stays at 10** and leaves mode dominance nearly constant.
  **Owner: 2h.**
- **`position_skew = 5` is a calibrated parameter**, chosen by measurement
  against the empirical envelope. It is the single strongest control on the
  coefficient of variation. **Owner: 2h**, as a sweep.
- **15 stale figures deleted, 115 MB.** They were left by earlier naming
  conventions (`FIG2_`, `FIG4_`, `Supplement2_`, `Supplement4_`,
  `ScatterPlot_UQResults_All`) and no live notebook cell writes any of them.
  `outputs/figures` went from 269 MB to 154 MB. Verified before deleting that
  `TABLE_PLCAResults.csv` is NOT stale: it is written through a variable, which
  a literal-path scan reports as an orphan.
- **The remaining 21 figures are all from the PRE-REGENERATION corpus** and are
  wrong as of this stage. They will be overwritten when notebooks 2 and 3 run
  against `corpus_2026-09-11c`. Only
  `CompareUQMethods_FIG_MetricCoverage.png/.pdf` is current.
- **The figures are enormous for a reason that is not dpi.** Measured:

  | Size | Pixels | Mpx | dpi | File |
  |---|---|---|---|---|
  | 31.3 MB | 9540 x 10230 | 97.6 | 1200 | `WassVsResultDiff.png` |
  | 31.2 MB | 9988 x 6587 | 65.8 | **default 100** | `SUPP_WassDistanceVsMetric_ALLMETRICS.png` |
  | 23.4 MB | 8948 x 10410 | 93.1 | 300 | `SUPP_ScatterPlot_UQResults_All.png` |
  | 9.6 MB | 9409 x 5501 | 51.8 | **default 100** | `SUPP_KSTestStripAndRank.png` |

  Three of those four are at the DEFAULT dpi of 100, which means the `figsize`
  is being declared at about 94 by 55 INCHES. A journal figure is 3.5 in for a
  single column or 7.5 in full width; at 300 dpi that is 2,250 px. These are
  four times too wide and twenty times too many pixels. Reducing dpi alone will
  not fix it; the `figsize` calls are the problem. **Owner: 3.**
- **The repository is 391 MB of git history, and almost none of it is data.**
  Measured at the end of this stage. The six largest blobs are 1200-dpi
  figures and `TABLE_PLCAResults.csv`, each re-stored whole every time it
  changed, totalling about 190 MB: 36.8 MB
  `Supplement4_WassDistanceVsMetric_ALLMETRICS.png`, 32.2 and 30.0 MB for two
  versions of `TABLE_PLCAResults.csv`, 31.3 and 30.8 MB for two versions of
  `WassVsResultDiff.png`, 31.2 MB `SUPP_WassDistanceVsMetric_ALLMETRICS.png`.
  Stage 1 already flagged this as deferred. It matters for the Zenodo deposit.
  **Owner: 3 (figures) and 4 (deposit).**

  Note that deleting files from the working tree does NOT shrink `.git`:
  `outputs/figures` fell from 269 MB to 154 MB in this stage and `.git` stayed
  at exactly 391 MB, because every version ever committed is still in the
  object database. Only a history rewrite (`git filter-repo`) reclaims it, and
  that rewrites every commit hash, which breaks any existing clone and has to
  be reconciled with the Zenodo deposit. **That is an explicit author decision,
  not a cleanup task**, and it is best done once, immediately before the
  Stage 4 re-deposit, after Stage 3 has settled the final figure set.
- **The corpus data are 192 MB and gitignored.** For the record, since the
  growth surprised the author: the old `DATA_all.json` held 2,744,113 values in
  121.5 MB of JSON, 44 bytes per value; the new corpus holds 12,876,931 values
  in 192 MB of Parquet, 15 bytes per value. That is 4.7x more data in 1.6x the
  space. The growth is the stratified design, not the format: stratum 4 alone
  carries 9,781,710 of the values and the probe set another 2,015,051, where
  the old corpus stopped at n = 749 against an empirical maximum of 77,548.
  Storing values and weights as float32 would halve it, at the cost of about
  seven significant digits and a perturbation to every W1 distance; **not done
  without the author's decision.**
- **An earlier version of the regeneration commit tracked the corpus data by
  mistake**, adding about 600 MB. Caught, reverted, and garbage-collected; the
  blobs are gone from the object database. `.gitignore` now covers
  `corpus_*/values.parquet`, `metrics.parquet` and `parents.json.gz`, while the
  provenance files stay tracked.
- **Audit scripts read `data/baseline_frozen/`**, so they keep reporting the
  pre-regeneration baseline after regeneration.

## 6. Inputs and outputs

**Read:** `reports/` in full, `CLAUDE.md`, `CONTEXT.md`, all of `src/`,
`notebooks/`, `data/processed/`, `refs/jcgs.2009.08054.pdf` and
`refs/v51i12.pdf` for the overlap parameterization, and the EC3 EPD store at
`../EPDsFromEC3/store/`.

**Written:** `src/components.py`, `src/mixture.py`, `src/modality.py`,
`src/genconfig.py`, `src/generator.py`, `src/corpus.py`, `src/empirical.py`,
`src/coverage.py`; `tests/test_components.py`, `tests/test_mixture.py`,
`tests/test_modality.py`, `tests/test_generator.py`; `audits/` (9 scripts);
`outputs/tables/stage2a/` (20+ tables); `data/baseline_frozen/`;
`data/processed/corpus_2026-09-11{,b,c}/`; this file.

**Modified:** all three notebooks, `src/customstats.py`,
`src/datageneration.py`, `src/dct_metriclabels.json`, `tests/test_regression.py`,
`CLAUDE.md`, `CONTEXT.md`, `reports/MANUSCRIPT_discrepancies.md` (entries 19 to
27), `environment.yml` (pyarrow added), `.gitignore`.

**Not touched:** `data/processed/DATA_all.json`, `dct_realeccs_trimmed.json`,
`combos.txt`, `datasets_*.json`, the pLCA fixtures, and the manuscript.

## 7. Next stage

**Stage 2b, the lognormal.** Threshold pathology, the +0.5 offset, two-parameter
versus profile-likelihood versus gamma, and W1-optimal fitting alongside MLE.

Three things to know first:

1. **Notebooks 2 and 3 have not been run against the new corpus.** They are
   rewired and parse, but the fitting and pLCA fixtures are still pinned to the
   pre-regeneration corpus. Running them and re-freezing is the first task.
   Use `COMPAREUQ_SMOKE_COMBOS=20` first.
2. **The parent is available.** `corpus.load_parents()` returns, per dataset,
   everything needed to rebuild its CDF exactly. Stage 2c depends on this;
   Stage 2b can use it to ask whether a lognormal fit is recovering the parent
   or the sample.
3. **The datasets are bigger.** Stratum 4 runs to n = 9,996 where the old
   corpus stopped at 749, so notebook 2's fitting and notebook 3's pLCA will
   both be slower than their Stage 1 timings.

## 8. Stage 2a assessment against the Stage 0 baseline

| Axis | Stage 0 | Stage 1 | Now | What changed |
|---|---|---|---|---|
| Correctness and numerical care | 6 | 8 | 9 | Eight defects found and fixed, most by test or measurement rather than reading; overflow-safe moment formulas; catastrophic cancellation removed |
| Code organization and reuse | 3 | 7 | 8 | Generation split into six focused modules with one responsibility each |
| Naming and readability | 6 | 7 | 8 | `generate_dontread` gone, "Mode Count" renamed to what it measures |
| Testing and validation | 2 | 8 | 9 | 126 tests from 42; the parent CDF is verified against 400,000 draws rather than asserted |
| Randomness and reproducibility | 2 | 9 | 10 | A corpus carries its seed, configuration, commit and library versions, and cannot be overwritten |
| Performance awareness | 4 | 8 | 9 | Generation profiled and made 3.4x faster; the dominant cost was found by profile, not guess |
| Scientific Python idiom | 5 | 7 | 8 | Vectorized bisection replacing per-element root-finding; closed-form moments |
| Version control and hygiene | 4 | 8 | 8 | Baseline frozen and verified before any destructive step |
| Statistical implementation judgment | 5 | 5 | 8 | Component shapes are moment targets, overlap and spread are specified and solved for, modality has a principled statistic, the parent is closed form |

Three things in this stage were caught only because the author interrupted to
ask why something was slow or what a number meant. The `brentq` bottleneck, the
overlap accuracy bug behind it, and the fact that the first corpus did not
resemble the empirical data on its most important metric were all surfaced that
way. The lesson for later stages is to state what a number means and what it is
being compared against at the time it is produced.
