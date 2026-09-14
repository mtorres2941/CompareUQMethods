# CONTEXT.md - How this repository works

Stable reference material. CLAUDE.md holds the project brief, the standing
constraints and the decision log; this file holds the mechanics. Split out at
the end of Stage 1, when CLAUDE.md grew past a comfortable size.

---

## 1. Package layout

```
CompareUQMethods/
├── CLAUDE.md                  Project brief, constraints, decision log
├── CONTEXT.md                 This file
├── environment.yml            Pinned environment (see section 4)
├── environment.lock.yml       Full transitive solve, osx-arm64
├── notebooks/
│   ├── 01_CompareUQ_CreateData.ipynb    generate/read data, compute metrics
│   ├── 02_CompareUQ_AnalyzeData.ipynb   fit 6 methods, score by W1/W2/KS
│   └── 03_CompareUQ_PerformPLCA.ipynb   2,499 pLCAs, downstream results
├── src/
│   ├── components.py          moment-targeted component families (Stage 2a)
│   ├── mixture.py             the truncated-mixture parent (Stage 2a)
│   ├── genconfig.py           every generation parameter (Stage 2a)
│   ├── generator.py           parent -> dataset, plus the validity filter
│   ├── corpus.py              generate, write and read a named corpus
│   ├── categorysplit.py       resolve a category into specifiable products,
│   │                          on metadata only (Stage 2a-3)
│   ├── empirical.py           prepare the empirical EC3 datasets
│   ├── modality.py            Silverman critical bandwidth, and the VISIBLE
│   │                          mode count that the generator is tuned against
│   ├── customstats.py         weighted statistics, distances, bandwidths
│   ├── datageneration.py      legacy generation helpers, empirical cleaning
│   ├── families.py            the support (0, inf), the parametric families,
│   │                          the weighted KDE with a CDF, and the
│   │                          estimators (Stage 2b)
│   ├── fitting.py             the six PEWT fits and W1 scoring
│   ├── datavisualization.py   one colour helper
│   ├── funcs_unit_conversion.py  EC3 unit normalization
│   └── dct_metriclabels.json  display labels for the 22 metrics
├── audits/stage2a/            one-off measurement scripts, see audits/README.md
├── audits/stage2a2/           empirical extract and generator audits, see its README
├── audits/stage2a3/           the category split and the envelope it moved
├── audits/stage2b/            the plausibility ceiling and the lognormal
├── data/processed/            inputs, see section 5
├── outputs/tables/            tidy results, see section 6
├── outputs/figures/           publication and supplementary figures
├── reports/                   handoffs, baselines, discrepancy log
└── tests/                     regression, determinism, unit, notebook guards
```

Notebooks are the entry point by design: the author values seeing inputs and
outputs inline, and considers that more reviewable by an outside reader of the
code. The target shape for a cell is: load a table, call one tested function
from `src/`, display, write a table. Computation belongs in `src/` where it can
be tested; narrative and display belong in the notebook.

## 2. The fitting interface

`src/fitting.py` is the single implementation. It replaced three verbatim
copies in Stage 1. `src/families.py` holds the distributions it fits.

```python
from fitting import fit_pewt, score_all_models, score_grid_open, PEWT

models, params = fit_pewt(x, weights_variable)      # 6 fitted models
w1 = score_all_models(models, x, weights_variable)  # 6 W1 distances
```

| Label | Probability estimation | Weights |
|---|---|---|
| `Normal, Uniform` | weighted mean and standard deviation | equal |
| `Normal, Variable` | same | supplied |
| `Lognormal, Uniform` | 3-parameter lognormal, threshold by profile likelihood | equal |
| `Lognormal, Variable` | same | supplied |
| `KDE, Uniform` | Gaussian KDE at `weighted_bw(..., BW_METHOD)` | equal |
| `KDE, Variable` | same | supplied |

### The support is (0, inf), open at zero

Decision 13, confirmed by the author in the Stage 2b prompt. **Every model is
an explicit truncation of its parent to (0, inf), renormalized**, and every one
of the six exposes the same interface:

    .pdf(x)  .cdf(x)  .ppf(q)  .rvs(size, random_state)  .rvs_from_uniform(u)

Before Stage 2b the normal and the KDE were one object when JUDGED and another
when APPLIED: the scoring grid started at zero, so they were implicitly
truncated without anything saying so, and the pLCA rejected non-positive draws,
so they were truncated there too by a different mechanism. The two agreed, but
neither was written down and nothing would have caught them drifting apart.

**Sampling is by inverse CDF and never by rejection.** They give the same
distribution, but Stage 2e's common random numbers need ONE uniform variate per
material per iteration pushed through every method's inverse CDF, and rejection
sampling consumes an unpredictable number of variates per draw.
`rvs_from_uniform` is that entry point. `WeightedKDE` exists because
`scipy.stats.gaussian_kde` offers a density and a resampler and no CDF; its
density matches gaussian_kde to machine precision and its CDF is the closed-form
weighted sum of normal CDFs, so the KDE that is scored is the KDE that is
sampled.

### The lognormal

`LOGNORMAL_FAMILY = 'lognormal_3p'`, the three-parameter lognormal whose
threshold is chosen by profile likelihood over a closed interval bounded
strictly below `min(x)`, taking the interior local maximum. **The likelihood of
a three-parameter lognormal is unbounded**: as the threshold approaches the
smallest observation from below, one term of the density diverges while the
others stay bounded, so the global MLE does not exist and a naive optimizer
returns whatever it stopped at. `families.fit_lognorm3_profile` carries the
treatment and the guard; the guard is reported per fit in `params[...]['status']`
and is swept in Stage 2h.

`LOGFIT_OFFSET = 0.5` and `fit_pewt_models` are the Stage 1 method, kept so the
change from it can be measured. What the offset actually was: a three-parameter
lognormal with the threshold fixed at -0.5 and never estimated, patching
near-zero values rather than the threshold pathology. See discrepancy entries
37, 38 and 40 and `audits/stage2b/r4_lognormal_offset.py`.

`FAMILIES` also holds `lognormal_2p`, `lognormal_offset` and `gamma`, which are
reported alongside rather than used: `audits/stage2b/r5_family_comparison.py`.

### Fitting by the criterion we score by

`fit_family(name, x, w, method='w1')` minimizes W1 directly instead of the
likelihood, starting from the MLE fit so it can never score worse. Every
parametric family in this study is estimated by maximum likelihood and judged by
W1, which are different criteria, so a family could lose the comparison because
it was never fitted under the rule it is judged by. `FIT_METHOD = 'mle'` is the
study's method; the W1-optimal results are reported beside it, not instead.

**Scoring.** Every model is scored against the **variable-weighted** empirical
CDF, including the uniform-weighted fits. `score_grid_open` is 1,000 points from
`hi / 1000` to `hi = max(x) + 10 * spread`, **open at zero**: the lower bound is
the first point of the grid's own lattice, chosen that way so that it is not a
new free parameter. The lattice is LINEAR, so its resolution near zero is the
same for every dataset; on a dataset spanning several orders of magnitude that
is coarse, which is unchanged from Stage 1 and belongs to Stage 2c.

## 3. Seeding and caching

**Seeding.** All randomness comes from an explicitly passed
`numpy.random.Generator`. No function in `src/` creates a Generator or touches
global numpy state. Each notebook creates exactly one:

```python
SEED = 20260911
rng = np.random.default_rng(SEED)
```

`tests/test_notebooks.py` enforces this: exactly one `np.random.default_rng`
per notebook and no other `np.random.*` call anywhere. `tests/test_determinism.py`
enforces that generation is reproducible from the seed and neither reads nor
advances the legacy global stream.

Record the seed in the run-metadata file beside any table the notebook writes.

**Corpus, not caching.** `generate_dontread` was removed in Stage 2a. It was a
hand-edited module-level boolean that left no record in the outputs of which
mode had produced them, which is the wrong mechanism for the one irreversible
operation in this project.

A corpus is now a named, dated directory that carries its own provenance, and
the notebooks only ever read. Regeneration is explicit:

```bash
cd src && python corpus.py 2026-09-11          # writes data/processed/corpus_2026-09-11/
cd src && python corpus.py <label> 1000        # a 1,000-dataset DRAFT, 80 s not 850
python -c "import sys; sys.path.insert(0,'src'); import corpus; corpus.set_active('2026-09-11')"
```

`generate_corpus` refuses to write into a directory that already exists, so a
corpus can never be overwritten. `data/processed/CORPUS.json`, which IS tracked,
names the active one, so repointing the whole analysis is a one-line change.

Each corpus directory holds:

| File | What |
|---|---|
| `values.parquet` | long format, `dataset_id, value, weight` (decision 15) |
| `metrics.parquet` | one row per dataset: stratum, metrics, generation record |
| `parents.json.gz` | the parent of each dataset, enough to rebuild its CDF exactly |
| `combos.csv` | the 2,500 disjoint pLCA groups of four |
| `runmeta.json` | seed, full config, git commit, library versions, platform, counts |
| `invalid_datasets.json` | what the validity filter rejected, and why |

## 4. Running

```bash
conda env create -f environment.yml
conda activate compareuq
python -m ipykernel install --user --name compareuq --display-name compareuq
python -m pytest tests/          # 128 tests, about 90 seconds
```

Headless execution, from `notebooks/`:

```bash
python -m nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=compareuq \
  --output-dir=/tmp/nbrun --output=out.ipynb 03_CompareUQ_PerformPLCA.ipynb
```

**Smoke configuration.** Notebook 3 is the expensive one. Set
`COMPAREUQ_SMOKE_COMBOS=20` to run it on 20 pLCAs instead of 2,500, which
exercises the pipeline end to end in well under a minute:

```bash
COMPAREUQ_SMOKE_COMBOS=20 python -m nbconvert --to notebook --execute ...
```

Use it before any full run. It caught two defects in Stage 1 that had
previously only surfaced eleven minutes into a full execution.

Smoke mode validates the pLCA loop, the results table and the inter-method
distance cell. The correlation and figure cells further down assume every
dataset appears in some pLCA, which is only true of the datasets the GROUPING
covers, so they are expected to fail under smoke mode. **Smoke results must
never be committed.**

Stage 2b found that the same assumption also broke a FULL run, because the
corpus no longer divides by four: cell 37 indexed `df_resultstd` by every
dataset in the corpus while the results covered the 9,996 the grouping reaches,
and cell 56 fed the two to `pearsonr` 18 minutes into the run. It now indexes by
`df_stds.index`. `corpus.describe_combos` names the held-out datasets and both
notebooks print it, so the remainder is stated rather than inferred.

Approximate runtimes on a 2026 laptop, all three notebooks, after the Stage 1
optimizations: NB1 about 80 s, NB2 about 160 s, NB3 about 19 min at `neccs = 10000`.
All three roughly doubled in Stage 2b, because stratum 4 now reaches n = 9,996
where the pre-regeneration corpus stopped at 749.

## 5. Input data

| File | What | Tracked |
|---|---|---|
| `CORPUS.json` | names the active corpus directory | yes |
| `corpus_<label>/` | the synthetic corpus, see section 3 | no, large |
| `raw/ec3_raw_ecc_<pull date>.csv.gz` | the raw empirical ECC extract, one row per EPD | yes |
| `raw/ec3_record_metadata_<pull date>.csv.gz` | product name, description, concrete strength and EC3 path; read by the split rules | yes |
| `raw/ec3_category_tree_<pull date>.csv` | EC3 category hierarchy; its parent/child relation identifies a residual bin | yes |
| `dct_realeccs_trimmed.json` | SUPERSEDED. The 2026-03 EC3 pull the manuscript reports | yes |
| `empirical_<label>.json` | a prepared empirical arm, written by `src/empirical.py` | yes |

**Retired at the end of Stage 2a, kept on disk as the pre-regeneration record:**
`DATA_all.json`, `datasets_outliers.json`, `datasets_trimto10k.json` and
`combos.txt`. Nothing reads them any more. Byte-identical copies with verified
checksums are in `data/baseline_frozen/`; see `data/INPUTS.sha256`.

The analysed set is the corpus minus the probe set: **9,999 datasets, not
10,000**, sizes 3 to 9,999, stratified 2,500 per stratum over 3-9, 10-99,
100-999 and 1000-9999 except the second, which holds 2,499 because one parent
failed to solve and was reported rather than approximated. Plus a 50-dataset
probe set at 10,000 to 100,000 that is excluded from every aggregate.

**It does not divide by four**, so the pLCA grouping is 2,499 groups covering
9,996 datasets and three are held out. `corpus.describe_combos` names them and
both notebooks print it; `corpus.make_combos` carries the reason a short last
group would be worse.

### The empirical arm

`src/empirical.py` reads a frozen, dated raw extract under `data/raw/`. Raw
means no outlier rule has been applied to it, which is what lets the cleaning
rule treat both ends of the distribution the same way. An ECC is the declared
GWP divided by the declared unit, each value converted by its own unit, with
each category restricted to the declared-unit type most of its products use.

Cleaning is a multiplicative 3 x IQR bound in log space, applied at BOTH ends.
An ECC is strictly positive and right skewed, so the additive form is the wrong
shape: `Q1 - 3*IQR` is negative in most categories and never binds, which
removes high outliers while leaving values orders of magnitude below the mean.
A dataset is kept only if at least three values survive, which is why 136 of the
138 extracted categories are retained.

**The arm is 149 datasets.** The 136 categories are resolved into specifiable
products by three metadata rules in `src/categorysplit.py`: EC3 residual bins
are dropped (15 of them, including `Insulation` and `Steel`, whose children are
already datasets here), concrete is split by specified 28-day compressive
strength, and insulation by material type read from the product name. **The test
is whether a category is something a specifier could name, NOT whether it is
tight**: splitting `ReadyMix` by strength moves its coefficient of variation only
from 0.29 to 0.27 and is still right. Decision 46. `src/categorysplit.py` holds the screen, the axis and
the binding constraint: a split may read only record metadata, never the ECC
values, because this study measures the modality and dispersion of ECC
distributions and splitting on those would be circular. Stage 2a-3, decision 43.

Each dataset's Dirichlet weights are keyed by its NAME, not by its position in
the iteration, so adding or splitting a category does not perturb the weights of
every dataset after it alphabetically. That coupling was real: before the change,
splitting six categories moved every weighted metric of 130 untouched datasets.

The current extract is a slice of the consolidated store at
`../EPDsFromEC3/store`, pulled 2026-08-13/14 through the LucidLCA wrapper.
`../EPDsFromEC3/PULLING_EPDS.md` documents three ways a paginated pull fails
while reporting success, and must be read before writing anything that talks to
that API. Note also that EC3 rate limits per ACCOUNT rather than per process, so
nothing should query it while a pull is running in another repository.

**The extract is frozen for the remainder of the project** (decision 44). The
procedure below is recorded for the one deliberate pre-submission refresh, if
the author calls for it, and for nothing else. To build a new extract, adapt
`audits/stage2a2/p1_build_raw_extract.py`, which refuses to overwrite an
existing dated file, then validate it with `p2_validate_extract.py` and
`p3_diagnose_changes.py` before pointing `src/empirical.SOURCE` at it.

### Two modality measures, and which one to tune against

`src/modality.py` provides both, and they answer different questions.

- `n_modes_silverman` is Silverman's critical-bandwidth test: is the data
  multimodal at ANY bandwidth. It is sensitive to fine structure that never
  appears in a plot.
- `n_modes_visible` counts local maxima of a Scott's-bandwidth KDE, keeping
  peaks whose prominence is at least 5 percent of the tallest. It answers how
  many humps a reader sees.

**They disagree profoundly on real ECC data**: about half the empirical datasets
are multimodal by Silverman, while 94.9 percent have exactly one visible mode.
Their structure is shoulders on a right-skewed body, not separated humps.

**Tune the generator against `n_modes_visible`.** Stage 2a-2 spent most of its
length tuning against Silverman, which was already matched, while the visible
distribution drifted to 58.9 percent unimodal against an empirical 94.3 and the
corpus filled with separated humps. `n_modes_silverman` remains a reported
characteristic and belongs in the metric set; it is not a steering signal.

`n_modes_visible` is the author's original `estimate_maxima` with a prominence
threshold in place of a continuous index. Stage 2a's decision 23 discarded that
metric for spanning only 1.000 to 1.159 across the empirical datasets, which was
a defect in the readout rather than in the idea.

### Both arms are cleaned by the same rule

`genconfig.trunc_rule = 'log'`. The synthetic parent is truncated at
`Q1 / (Q3/Q1)**3` and `Q3 * (Q3/Q1)**3`, which is the rule
`datageneration.clean_empirical_symmetric` applies to the empirical values. Do
not let these diverge: when they did, the two arms' characteristics differed
because of the cleaning and generator tuning was compensating for it. Restoring
consistency moved mean W1 across the characteristics from 0.488 to 0.270.

## 6. Output tables

| File | Written by | Shape |
|---|---|---|
| `TABLE_EmpiricalECCMetrics.xlsx` | NB1 | 149 x 22 |
| `TABLE_EmpiricalCategorySplit.csv` | NB1 | one row per split population |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | NB2 | 149 x 28 |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | NB2 | 9,999 x 37 |
| `TABLE_PLCAResults.csv` | NB3 | 59,976 x 43, which is 2,499 groups x 6 methods x 4 datasets |
| `TABLE_PLCAResults_runmeta.json` | NB3 | seed, neccs, versions, platform |

`TABLE_PLCAResults.csv` is tidy long format, one row per
(pLCA, UQ method, dataset). It did not exist before Stage 1: notebook 3 wrote
fifteen figures and no table, so its results lived only in a kernel.

## 7. Regression fixtures

`tests/fixtures/`. These are a change detector, not a correctness claim: they
were produced by code with known defects, and Stage 2 will deliberately move
many of these numbers. When a number is meant to move, the fixture is
re-frozen **in the same commit** that moves it, with the delta recorded in the
commit message, and `SHA256SUMS.txt` updated.

| Fixture | Pins |
|---|---|
| `TABLE_EmpiricalECCMetrics.xlsx` | the metrics for the empirical datasets |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | those metrics plus the six W1 scores |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | metrics and W1 for the 10,000 synthetic datasets |
| `SHA256SUMS.txt` | checksums, so a fixture cannot be edited silently |
| `plca/PLCA_unseeded_archive_neccs1000.csv.gz` | **archive, not a fixture.** The last run before seeding; unreproducible. The closest surviving record of what the current manuscript draft reports |
| `plca/PLCA_seeded_neccs1000.csv.gz` | the first reproducible pLCA result, kept to separate the effect of seeding from the effect of the sample-size change |
| `plca/PLCA_seeded_neccs10000.csv.gz` | the configuration the manuscript states |

Stage 2a renamed `mode_count_est` to `modality_index` and added `crit_bw_1`.
The recomputation tests map the old name back and drop the new columns before
comparing, so every column the fixtures and the current code share is still
checked value for value. All 8 pass, which is the proof that the rename and the
addition moved nothing.

**Tolerance** is `rtol = 1e-6`. The environment that produced the original
tables was lost, so the fixtures cannot be reproduced bit for bit. Measured
agreement under the pinned environment: metric columns 1.3e-14, Normal and KDE
W1 2.0e-13, lognormal W1 4.3e-08. The lognormal term dominates because
`weighted_lognorm_fit` calls `scipy.optimize.minimize`, whose convergence path
shifts between scipy versions. The tolerance sits an order of magnitude above
the worst observed value.

## 8. Test suite

| File | Tests | Guards |
|---|---|---|
| `test_regression.py` | 8 | fixture integrity, outputs against fixtures, independent recomputation driving `src/` without a notebook |
| `test_determinism.py` | 7 | same seed reproduces, different seeds differ, global numpy state neither affects nor is consumed, rng is required, the seed=0 collapse is gone, output contract |
| `test_customstats.py` | 17 | hand-computed quantiles, order invariance with and without ties, moments against scipy, both bandwidth formulas, Wasserstein identities |
| `test_notebooks.py` | 10 | every code cell parses, no global numpy randomness, exactly one Generator per notebook |
| `test_components.py` | 48 | moment targets hit exactly, infeasible targets refused not approximated, every accepted component inverts its own CDF, the four families partition the Pearson plane |
| `test_mixture.py` | 9 | the parent CDF matches a 400,000-draw sample, the market-weighted parent is a real population object, coupling 0 collapses the two parents, inverse-CDF sampling agrees with the truncation loop it replaced, overlap is symmetric and monotone in separation |
| `test_modality.py` | 8 | binned KDE matches direct evaluation, mode count ignores FFT round-off and is non-increasing in bandwidth, Silverman recovers known mode counts, the statistic is scale free and defined at n = 3 |
| `test_families.py` | 105 | the support is open at zero and no sampler can emit an inadmissible value, cdf inverts ppf on every family, inverse-CDF sampling reproduces the model CDF, `rvs_from_uniform` is the same map `rvs` uses, truncation renormalizes rather than discarding mass, the weighted KDE matches gaussian_kde's density and integrates to its own CDF, the closed-form lognormal and gamma estimators beat their neighbours on the likelihood, the profile threshold stays strictly below min(x) and reaches the normal limit when the data asks for it, an unguarded joint fit walks into the pathology and the guarded one does not, the W1-optimal fit never scores worse than the MLE fit |
| `test_generator.py` | 18 | strata allocate and cover their endpoints, the probe set sits outside the corpus, generated datasets are valid and normalized, the record reconstructs the parent, the validity filter passes extreme-but-analysable data and catches unanalysable data, undefined kurtosis at n = 3 is not a failure, generation is reproducible and never touches global numpy state |

`test_notebooks.py::test_all_code_cells_parse` exists because a Stage 1 patch
script silently dropped the final line of any cell whose source did not end in
a newline, truncating a cell mid-statement. The only symptom was a SyntaxError
twelve minutes into a headless run.
