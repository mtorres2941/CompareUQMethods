# CONTEXT.md - How this repository works

Stable reference material. CLAUDE.md holds the project brief, the standing
constraints and the decision log; this file holds the mechanics. Split out at
the end of Stage 1, when CLAUDE.md grew past a comfortable size.

---

## 1. Package layout

```
CompareUQMethods/
|-- CLAUDE.md                  Project brief, constraints, decision log
|-- CONTEXT.md                 This file
|-- environment.yml            Pinned environment (see section 4)
|-- environment.lock.yml       Full transitive solve, osx-arm64
|-- notebooks/
|   |-- 01_CompareUQ_CreateData.ipynb    generate/read data, compute metrics
|   |-- 02_CompareUQ_AnalyzeData.ipynb   fit 6 methods, score by W1/W2/KS
|   \-- 03_CompareUQ_PerformPLCA.ipynb   2,500 pLCAs, downstream results
|-- src/
|   |-- customstats.py         weighted statistics, distances, bandwidths
|   |-- datageneration.py      synthetic ECC dataset generation
|   |-- fitting.py             the six PEWT fits and W1 scoring
|   |-- datavisualization.py   one colour helper
|   |-- funcs_unit_conversion.py  EC3 unit normalization
|   \-- dct_metriclabels.json  display labels for the 20 metrics
|-- data/processed/            inputs, see section 5
|-- outputs/tables/            tidy results, see section 6
|-- outputs/figures/           publication and supplementary figures
|-- reports/                   handoffs, baselines, discrepancy log
\-- tests/                     regression, determinism, unit, notebook guards
```

Notebooks are the entry point by design: the author values seeing inputs and
outputs inline, and considers that more reviewable by an outside reader of the
code. The target shape for a cell is: load a table, call one tested function
from `src/`, display, write a table. Computation belongs in `src/` where it can
be tested; narrative and display belong in the notebook.

## 2. The fitting interface

`src/fitting.py` is the single implementation. It replaced three verbatim
copies in Stage 1.

```python
from fitting import fit_pewt_models, score_w1, score_all, score_grid, PEWT

models = fit_pewt_models(x, weights_variable)   # -> dict of 6 fitted models
w1     = score_all(models, x, weights_variable) # -> dict of 6 W1 distances
```

| Label | Probability estimation | Weights |
|---|---|---|
| `Normal, Uniform` | weighted mean and standard deviation | equal |
| `Normal, Variable` | same | supplied |
| `Lognormal, Uniform` | weighted MLE on `x + LOGFIT_OFFSET`, location shifted back | equal |
| `Lognormal, Variable` | same | supplied |
| `KDE, Uniform` | Gaussian KDE at `weighted_bw(..., BW_METHOD)` | equal |
| `KDE, Variable` | same | supplied |

Every model exposes `.pdf()`. The KDE objects are `scipy.stats.gaussian_kde`;
the others are frozen scipy distributions.

Two module constants carry the methodological choices that are under review in
Stage 2:

- `LOGFIT_OFFSET = 0.5`. The data are shifted before the lognormal fit and the
  location shifted back, which is a 3-parameter lognormal with the threshold
  fixed at -0.5 rather than estimated. It exists because near-zero values drag
  the log-space mean down and inflate sigma, collapsing the fitted mode toward
  zero. See `reports/MANUSCRIPT_discrepancies.md` entry 6.
- `BW_METHOD = 'scott'`, meaning `1.06 * sigma * n_eff ** -0.2`, which is
  Scott (1992). **`scipy.stats.gaussian_kde` uses the words 'scott' and
  'silverman' for different formulas**: its `'scott'` carries no 1.06 factor
  and its `'silverman'` is approximately this project's `'scott'`. Never
  describe the method by pointing at a scipy keyword. See entry 10.

**Scoring.** Every model is scored against the **variable-weighted** empirical
CDF, including the uniform-weighted fits. The model is discretized onto a
1,000-point grid running from 0 to `max(x) + 10 * sd`, so it is implicitly
truncated at zero and renormalized. That matches the rejection sampling in the
pLCA, which discards non-positive draws, so the model scored is the model
sampled.

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

**Caching.** Notebook 1 has a `generate_dontread` flag. When `False`, the
default, it reads `data/processed/` rather than regenerating. Regeneration is a
Stage 2 activity and invalidates every number in the paper, so it should happen
once, deliberately, at a known commit.

## 4. Running

```bash
conda env create -f environment.yml
conda activate compareuq
python -m ipykernel install --user --name compareuq --display-name compareuq
python -m pytest tests/          # 42 tests, about 8 seconds
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
dataset appears in some pLCA, which is only true in a full run, so they are
expected to fail under smoke mode. **Smoke results must never be committed.**

Approximate runtimes on a 2026 laptop, all three notebooks, after the Stage 1
optimizations: NB1 about 20 s, NB2 about 75 s, NB3 about 11 min at
`neccs = 10000`.

## 5. Input data

| File | What | Tracked |
|---|---|---|
| `DATA_all.json` | 15,000 synthetic datasets, values, weights, 20 metrics | no, 122 MB |
| `dct_realeccs_trimmed.json` | 138 empirical EC3 datasets | yes |
| `datasets_outliers.json` | 4,131 datasets flagged by the metric filter | yes |
| `datasets_trimto10k.json` | 869 further datasets dropped to reach 10,000 | yes |
| `combos.txt` | 2,500 groups of four, as a flat list | yes |

The analysed set is `DATA_all` minus the two exclusion lists: exactly 10,000
datasets, sizes 4 to 749, median 62.

## 6. Output tables

| File | Written by | Shape |
|---|---|---|
| `TABLE_EmpiricalECCMetrics.xlsx` | NB1 | 138 x 20 |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | NB2 | 138 x 26 |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | NB2 | 10,000 x 26 |
| `TABLE_PLCAResults.csv` | NB3 | 60,000 x 43 |
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
| `TABLE_EmpiricalECCMetrics.xlsx` | the 20 metrics for the 138 empirical datasets |
| `TABLE_EmpiricalECCMetricsAndW1.xlsx` | those metrics plus the six W1 scores |
| `TABLE_SyntheticECCMetricsAndW1.xlsx` | metrics and W1 for the 10,000 synthetic datasets |
| `SHA256SUMS.txt` | checksums, so a fixture cannot be edited silently |
| `plca/PLCA_unseeded_archive_neccs1000.csv.gz` | **archive, not a fixture.** The last run before seeding; unreproducible. The closest surviving record of what the current manuscript draft reports |
| `plca/PLCA_seeded_neccs1000.csv.gz` | the first reproducible pLCA result, kept to separate the effect of seeding from the effect of the sample-size change |
| `plca/PLCA_seeded_neccs10000.csv.gz` | the configuration the manuscript states |

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

`test_notebooks.py::test_all_code_cells_parse` exists because a Stage 1 patch
script silently dropped the final line of any cell whose source did not end in
a newline, truncating a cell mid-statement. The only symptom was a SyntaxError
twelve minutes into a headless run.
