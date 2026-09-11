# HANDOFF stage-1 - Refactor with results locked

## 1. Stage and branch

- Stage: 1, refactor
- Branch: `stage-0-1-refactor`, the same branch as Stage 0, as instructed
- Branched from: `1ab6bf5` on `main`
- Stage 1 begins after `de26b4f` (the last Stage 0 commit) and ends at the
  commit adding this file. 18 commits.

| Commit | Label |
|---|---|
| `a6f8363` Freeze shipped result tables as regression fixtures | NEUTRAL |
| `8936ef0` Make NB2 runnable from a clean kernel | NEUTRAL (A1) |
| `68ac3e2` Pinned environment, regression suite, baseline timing | NEUTRAL |
| `279df5c` Manuscript discrepancy log | docs |
| `111ad0f` NB3 baseline timing | docs |
| `9fd3b6f` Persist the pLCA results | NEUTRAL (A2) |
| `533f6c4` Thread an explicit Generator through everything | NEUTRAL |
| `c096428` Seeded pLCA fixtures, Monte Carlo noise floor | NEUTRAL |
| `323563a` Fix weighted_quantile ordering, add unit tests | NEUTRAL |
| `c313690` Fix infinite kurtosis for n < 4 | **MOVES NUMBERS** |
| `113dbeb` Remove unused imports from NB2, delete a dead file | NEUTRAL |
| `58fde9c` Consolidate the PEWT fit into src/fitting.py | NEUTRAL |
| `23c6482` CLAUDE.md decision log, CONTEXT.md | docs |
| `7c57afc` Restore box-drawing in CONTEXT.md | docs |
| `c84fb20` Update discrepancy log | docs |
| `4af6f6b` neccs to 10,000, wbeci fix, loop optimization | **MOVES NUMBERS** |
| `4f78808` NB3 on src/fitting.py, leaked W removed, environment trimmed | NEUTRAL |
| `83eb419` Pipeline roadmap in CLAUDE.md | docs |

## 2. What was asked

Execute the Stage 1 plan in section 10 of the Stage 0 handoff, phase by phase,
under four amendments: A1 make NB2 runnable before taking a baseline, A2 treat
the Phase 1 pLCA table as an archive rather than a fixture and keep three pLCA
artifacts, A3 keep unweighted-mean normalization and fix the text instead, A4
close the `weighted_quantile` scope question. Create the manuscript discrepancy
log. Allow exactly two number-moving changes. Extend CLAUDE.md and write this
handoff.

## 3. What was done

### 3.1 Phase 0, baseline and environment

The plan's ordering was reversed deliberately: fixtures were frozen **before**
the baseline run, because executing the notebooks overwrites the very tables
being frozen. That turned the baseline run into a verification rather than a
destructive act.

- `environment.yml` and `environment.lock.yml` replace the lost `waterweed`
  kernel. Python 3.11.16, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5.
- **The analysis is reproducible across environments**, which had never been
  demonstrated. The pinned environment reproduces the shipped tables to a
  maximum relative difference of **4.3e-08**: metric columns to 1.3e-14, Normal
  and KDE W1 to 2.0e-13, and lognormal W1 to 4.3e-08. The lognormal term
  dominates because `weighted_lognorm_fit` calls `scipy.optimize.minimize`,
  whose convergence path shifts between scipy versions. Regression tolerance
  set at `rtol = 1e-6`.
- Baseline timing, all three notebooks unmodified: NB1 18.2 s, NB2 72.9 s,
  NB3 655.6 s, total 12.4 min.

**A1.** NB2 could not run from a clean kernel: cell 65 used `metrics`, assigned
only inside cell 62, which is entirely commented out. The definition was
recovered verbatim from that commented block, and the recovery was verified
three ways before editing: it survives in the source so nothing was
reconstructed; it resolves to exactly 19 metrics, all of which have labels; and
19 metrics at 5 columns gives 4 rows, consistent with the committed figure's
aspect ratio of 1.516 and inconsistent with 3 or 5.

### 3.2 Phase 1, persisting the pLCA results

NB3 wrote fifteen figures and no table, so the headline results existed only in
a kernel. It now writes `TABLE_PLCAResults.csv`, 60,000 rows by 43 columns, one
row per (pLCA, UQ method, dataset), plus a run-metadata record.

Per **A2** the first such table is an archive, not a fixture: at that point the
draws were unseeded, so rerunning produces different numbers.

### 3.3 Phase 2, seeding

- Every function in `src/datageneration.py` now takes a required Generator.
  None creates one or touches global numpy state.
- **The `seed=0` default is gone.** It built a fresh Generator on every call,
  which made the shape parameters constants across the entire study
  (skew-normal a = 3.458732, Student-t df = 3.366328, lognormal s = 1.136962)
  and made two Gaussian components of equal length exact affine images of one
  another. Verified removed: two such components now differ by 3.36 in
  standardized units, against about 1e-15 before.
- The lognormal branch got its missing `random_state`; it was the one component
  type still drawing from global state.
- 32 substitutions across the three notebooks leave exactly one
  `np.random.default_rng(SEED)` per notebook and no other global randomness.
- `datasets_outliers.json` was non-deterministic between runs because
  `list(set(...))` over strings depends on `PYTHONHASHSEED`. Now `sorted(...)`.
  The set is identical and the 10,000 analysed datasets are unchanged.

**No regeneration.** `DATA_all.json` is untouched, dated March 2026.
`generate_dontread` remains `False`. Regeneration is Stage 2a.

### 3.4 Phase 3, correctness

`weighted_quantile` sorted value and weight pairs together but built the
cumulative sum from the original unsorted weight array, so the answer depended
on input ordering. **The first version of the fix was itself still wrong**:
sorting alone is insufficient because `np.argsort` is not stable, so tied
values had their weights attached in different orders. On `dataset71` the 25th
percentile still came out as 0.8293 or 0.8146 depending on presentation, a 7.6
percent difference in the resulting bandwidth. The correct treatment aggregates
tied values into a single atom. Now order-invariant on 2,000 of 2,000 datasets
at `rtol = 1e-12`.

Per **A4** no metric columns were audited for downstream effects: the analysis
uses the `'scott'` path, which never calls this function.

### 3.5 Phase 4, consolidation

`src/fitting.py` holds one implementation of the PEWT fit, replacing three
verbatim copies. `LOGFIT_OFFSET` and `BW_METHOD` are named constants with
comments, rather than values buried in a loop. Verified to reproduce the frozen
W1 fixture on 400 datasets to 6.1e-09, and the NB3 consolidation verified
**bit-identical** against the committed pLCA fixture.

Consolidating **removed a latent bug**: NB3's loop read
`for dataset, w in zip(datasets, W)`, where `W` leaked from the fitting loop
above. Results were correct because `w` was unused, but `zip` would have
silently dropped materials had that leftover array held fewer than four
elements. Removing the fitting loop turned it into a NameError.

Also removed: 15 unused imports across NB2 and NB3 (openturns, xgboost,
scikit-learn, requests, pprint, stdlib random), all verified used zero times;
`src/VOID_dct_metriclabels.json`; and the three defence-presentation cells,
annotated as not for the manuscript and byte-identical apart from one dataset
name. `environment.yml` drops four dependencies, three of them heavy.

### 3.6 Performance

Per-cell timings of the baseline run, rather than guesswork, located the cost:
pLCA loop 270 s (41 percent), frame rebuilds 170 s (26 percent),
`compare_results` plus savefig 159 s (24 percent).

- `compare_results` and `compare_results_bypewt` rebuilt a 60,000-cell frame by
  scalar `.loc` assignment about 80 times per run. They now pivot off the tidy
  table.
- `pandas.rank` was 30 percent of the pLCA loop and was recomputed four times
  per combo on the same frame. Hoisted at three sites, as were the mean,
  standard deviation, median and their sorts.
- The rejection filter uses a boolean mask rather than a Python list
  comprehension over every draw. Measured 53x faster per call but only about
  0.9 min overall, because the rejection loop rarely runs more than a round or
  two. I had assumed this was the dominant cost; it was not.
- Supplementary figures dropped to dpi 300; publication figures stay at 1200.

**Net: 813 s at neccs=10000 against 655 s at neccs=1000 before optimization.**
A tenfold increase in draws costs 24 percent more wall clock.

A **smoke mode** was added, `COMPAREUQ_SMOKE_COMBOS=20`, running NB3 on 20
pLCAs in about 25 seconds. It found two defects that had previously only
surfaced eleven minutes into a full run. It validates the pipeline through the
results table; the correlation and figure cells below that assume full coverage
and are expected to fail under it.

### 3.7 Testing

42 tests, about 8 seconds, in four files. See CONTEXT.md section 8.

## 4. Numbers that moved

Three changes moved numbers. Two were the sanctioned ones; the third was
approved by the author during the stage.

### 4.1 Infinite kurtosis at n < 4, commit `c313690`

`weighted_kurtosis` divides by `(n-3)`. Five of the 138 empirical datasets have
exactly n = 3. Nine cells moved, all from infinite to NaN:

| Column | Dataset | Before | After |
|---|---|---|---|
| kurtosis | Electrical | +inf | nan |
| kurtosis | AluminiumSiding | -inf | nan |
| kurtosis | WindTurbines | -inf | nan |
| kurtosis | WallBase | +inf | nan |
| kurtosis | Flooring | -inf | nan |
| kurtosis_uw | Electrical | +inf | nan |
| kurtosis_uw | AluminiumSiding | +inf | nan |
| kurtosis_uw | WallBase | +inf | nan |
| kurtosis_uw | Flooring | -inf | nan |

No finite value changed, no W1 value changed, synthetic datasets unaffected
(minimum n is 4). NaN was returned rather than silently substituting the biased
estimator, which would place two different quantities in one column.

### 4.2 wbeci assignment moved inside the loop, commit `4af6f6b`

`wbeci_mean` and `wbeci_stdev` move from **25.0 percent to 100 percent
populated** in `TABLE_PLCAResults.csv`. Verified before the fix that the value
is constant within each (pLCA, method) group, so filling the other three rows
cannot change any existing number.

A consequence worth recording: with the columns populated they enter the result
set, and their absence from `dct_resultlabels` failed a full run eleven minutes
in. Labels were added.

### 4.3 neccs from 1,000 to 10,000, commit `4af6f6b`

Same seed, same input data, measured between the two seeded fixtures:

| Result | mean abs diff | p95 | max | sd of the result |
|---|---|---|---|---|
| `eci_rank_1` | 0.01115 | 0.02780 | 0.06890 | 0.09584 |
| `eci_perc_mean` | 0.00227 | 0.00639 | 0.01772 | 0.01226 |
| `ui` | 0.01500 | 0.04302 | 0.15715 | 0.23654 |
| `eci_mean` | 0.01010 | 0.03401 | 0.11905 | 0.08124 |
| `capecc_perc_mean` | 0.00369 | 0.01141 | 0.03502 | 0.08504 |
| `matred_perc_mean` | 0.00057 | 0.00160 | 0.00443 | 0.00306 |

These are Monte Carlo convergence, not a change of method: the n = 1000 result
was the noisier estimate of the same quantity.

**No other number moved.** Every NEUTRAL phase left the fixtures passing.

## 5. Open questions and flags

### Carried forward from Stage 0

| Stage 0 item | Status |
|---|---|
| 1. pLCA sample size 1,000 vs 10,000 | **RESOLVED.** 10,000 |
| 2. Unweighted vs weighted mean normalization | **RESOLVED** by A3. Unweighted stands; the manuscript text is wrong. Removed from the Stage 2 regeneration list |
| 7. Notebooks remain the entry point | **RESOLVED.** Implemented |
| 8. Regeneration in Stage 2 | **STILL OPEN.** Owned by 2a |
| 9. Manuscript numbers may be invalidated | **RESOLVED.** Accepted |
| 10. Approval of the refactor plan | **RESOLVED.** Approved with four amendments |
| Bandwidth rule, KL1/KL2 inconsistency | **STILL OPEN.** Owned by 2h. `weighted_quantile` is fixed and must stay fixed before any switch to Silverman |
| `seed=0` collapse | **PARTIALLY RESOLVED.** Seeding fixed in Stage 1; whether the generation algorithm needs further change is 2a |
| Dependent sampling | **STILL OPEN.** Owned by 2e |
| The 27.5 percent outlier filter, n cap at 749 | **STILL OPEN.** Owned by 2a |
| `logfit_offset` | **STILL OPEN.** Owned by 2b, swept in 2h |
| "Mode Count" naming | **STILL OPEN.** Owned by 2a |
| Overlap area alongside W1 | **STILL OPEN.** Owned by 2c |
| Shapiro-Wilk vs Shapiro-Francia, `_royston_pvalue` | **STILL OPEN.** Owned by 2f |

### New in Stage 1, with the stage that owns each

- **The additive low-end cleaning bound never binds.** `Q1 - 3*IQR` is negative
  in 128 of 138 empirical datasets, so near-zero values are never removed while
  high ones are. `ReadyMix` retains a value at 3.1e-17 of its mean. Owned by
  **2a**. Discrepancy log entry 16.
- **Empirical dataset sizes reach 77,548 while synthetic stop at 749.** Six
  empirical datasets exceed the synthetic maximum, including the two largest and
  most carbon-significant. Owned by **2a**. Entry 14.
- **Monte Carlo noise was never quantified.** Owned by **2e**. Entry 17.
- **The scoring grid includes zero while the sampler excludes it.** The author's
  position is support on (0, inf), open at zero. Owned by **2c** or **2e**.
  Entry 18.
- **W1 is an in-sample criterion with no complexity penalty**, and the three
  families differ in flexibility. A held-out or cross-validated W1 would close
  the objection that the ranking is partly a flexibility ranking. Owned by
  **2c**.

### A disclosure about scope

The roadmap arrived near the end of this stage, after the author had asked
direct questions about several items that the roadmap assigns to later stages.
Measurements were taken and are recorded in
`reports/MANUSCRIPT_discrepancies.md` entries 14 to 18 and in the decision log,
covering the lognormal threshold (2b, 2h), multimodality measures (2a), the
multiplicative cleaning filter (2a), a candidate decision-reversal metric (2g),
and the Monte Carlo noise floor (2e).

**Treat these as inputs, not conclusions.** They were made on the pre-
regeneration data, they do not bind the owning stage, and each owning stage
should reach its own decision. They are recorded so the work is not repeated,
not to pre-empt it.

Decision log entries 10 to 15 in CLAUDE.md are **tagged by provenance** for
exactly this reason: `[AUTHOR]` is settled and should be implemented,
`[RECOMMENDED]` is a Stage 1 suggestion the owning stage may overrule,
`[DELEGATED]` is a choice the author explicitly left to judgment, and
`[CONTEXT]` is a measured fact rather than a decision. The tags were added
after a review pointed out that the original entries conflated the author's
decisions with mine. Four of the six did.

**One entry is flagged for confirmation.** Entry 13, that ECC support is
(0, inf) open at zero, was stated in conversation rather than in a prompt, and
it constrains the lognormal and gamma fits in 2b and the W1 evaluation grid in
2c. Confirm it with the author before building on it.

### Deferred Stage 1 work

Recommended to the author and accepted as deferrable. None blocks Stage 2.

- Phase 5, compute and plotting separation. Figures still read in-memory state
  in places, which the standing constraint asks to be fixed.
- Phase 6, deleting the 15 orphaned figures, reconciling the README's output
  list, reducing the remaining large figures.
- Remaining Phase 4 tidying: the `generate_dontread` flag, the
  `sys.path.insert(0, '../../shared')` pointing at a directory that does not
  exist, `tqdm`, the three duplicate rebuilds of `df_metrics` in NB2, and the
  `rankth` dictionary defined identically in all three notebooks.

## 6. Inputs and outputs

**Read:** everything under `notebooks/`, `src/`, `data/processed/`,
`outputs/`, plus `reports/HANDOFF_stage-0.md` and the manuscript.

**Written:** `environment.yml`, `environment.lock.yml`, `src/fitting.py`,
`tests/` (4 test files, 6 fixtures, 2 READMEs), `CONTEXT.md`,
`reports/MANUSCRIPT_discrepancies.md`, `reports/baselines/`, this file.
**Modified:** all three notebooks, `src/customstats.py`,
`src/datageneration.py`, `CLAUDE.md`, `data/processed/datasets_outliers.json`
(ordering only), `outputs/tables/`. **Deleted:**
`src/VOID_dct_metriclabels.json`.

**Not touched:** `DATA_all.json`, `dct_realeccs_trimmed.json`, `combos.txt`,
`datasets_trimto10k.json`, and the manuscript.

### Irreplaceable inputs

`data/INPUTS.sha256` records SHA-256 checksums and provenance for the five
input files and the three pLCA artifacts. `DATA_all.json` is 122 MB,
gitignored, dated 2026-03-13, and **cannot be reproduced by any means**: it was
generated before seeding existed, with `default_rng(None)` for the structural
parameters and a `seed=0` default for the component draws, both since changed.
Every regression fixture is pinned to it and Stage 2a's verification story
depends on it.

The repository lives inside a synced Dropbox folder and the file carries
`com.dropbox.attrs`, so it is replicated off-machine already. That is sync, not
backup: a deletion or corruption propagates. The manifest lets any copy be
verified as the genuine pre-regeneration artifact with
`shasum -a 256 -c data/INPUTS.sha256`. Stage 0 plan item 11 asked that this
file stop being both required and untracked; the manifest closes the
verification half, not the durability half.

## 7. Next stage

**Stage 2a, the generator audit**, in a new window. Read `reports/` in full and
CLAUDE.md's pipeline roadmap before starting.

Three things to know before touching anything:

1. **The regression fixtures are a change detector, not a correctness claim.**
   Stage 2a will deliberately move most of these numbers. When it does,
   re-freeze the fixture in the same commit and update `SHA256SUMS.txt`.
2. **Use smoke mode.** `COMPAREUQ_SMOKE_COMBOS=20` turns an eleven-minute
   failure into a twenty-five-second one. It earned its place twice in Stage 1.
3. **Regenerate exactly once, at the end of 2a.** Every number in the paper
   moves when it does.

Generation now takes a required Generator and is provably reproducible from a
recorded seed, so a regeneration can be tied to a seed in the run metadata and
repeated exactly. That machinery is the main thing Stage 1 hands over.

## 8. Stage 1 assessment against the Stage 0 baseline

Measured against the axes in `reports/HANDOFF_stage-0.md` section 8.1. Only
axes Stage 1 was scoped to move are scored; the rest are unchanged by design.

| Axis | Stage 0 | Now | What changed |
|---|---|---|---|
| Correctness and numerical care | 6 | 8 | `weighted_quantile` order dependence and tie handling fixed; infinite kurtosis guarded; the leaked `W` removed; non-finite quartile case still open |
| Code organization and reuse | 3 | 7 | Three copies of the fit collapsed into `src/fitting.py`; dead files, 15 unused imports and the duplicate defence cells removed; four notebook-level duplications remain |
| Naming and readability | 6 | 7 | Named constants replace buried literals; stale docstrings corrected; `generate_dontread` and the shadowed `type` remain |
| Testing and validation | 2 | 8 | 42 tests where there were none, including regression, determinism, hand-computed unit tests and static notebook guards |
| Randomness and reproducibility | 2 | 9 | One Generator per notebook, no global state, required rng, determinism tested, the seed recorded beside every table. Held back from 10 only because `DATA_all.json` is still an untracked artifact rather than regenerable on demand |
| Performance awareness | 4 | 8 | Cost located by measurement rather than assumption; the three real bottlenecks fixed; smoke mode added |
| Scientific Python idiom | 5 | 7 | Scalar `.loc` rebuilds replaced by pivots, reductions hoisted, list comprehensions vectorized; pandas-as-dict patterns remain in the untouched cells |
| Version control and project hygiene | 4 | 8 | Pinned and locked environment, a smaller dependency set, one commit per number-moving change with the delta recorded; 267 MB of figures and 15 orphans remain |
| Statistical implementation judgment | 5 | 5 | Deliberately unchanged. Every judgment call was deferred to its owning stage |

Three defects in this stage came from my own edits rather than from the
original code: a patch script that dropped the last line of any cell without a
trailing newline, a first `weighted_quantile` fix that was still order
dependent under ties, and a regex that matched only as far as the first `func`
assignment. All three were caught by tests or by measurement rather than by
reading, which is the argument for both the test suite and smoke mode.
