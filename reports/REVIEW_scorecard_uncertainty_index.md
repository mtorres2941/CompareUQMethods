# REVIEW: the claim scorecard's uncertainty-index cell

Written 2026-10-05 by a fresh window whose only job was to attack one number.
Ran against `corpus_2026-09-25`, `WEIGHT_RHO = 0.5`, and `outputs/` as committed
on 2026-10-02. **No code, table, figure or notebook was changed by this review.**

---

## The first page

**The computation is correct and nothing has to be recomputed.** The author
asked whether a probabilistic LCA's uncertainty index can really be wrong by
half. It can, it is, and the 48.7 percent on the scorecard is arithmetically
right, built on the right truth, in the right units, by the same code path as
the fitted side.

**What is wrong is the figure the number sits on, in one specific place.** The
right-hand bar labeled "what the choice costs (worst minus best)" is an
AVERAGED quantity standing beside cells that are explicitly PER DECISION. For
the uncertainty index it reads 2.2 where the per-decision figure is 27.9 to
40.2 -- a factor of 13 to 18. This is the same one-decision-against-average-of-
many defect that decision 174 corrected in the CELLS; the bar was never brought
across with them.

**And one finding that should change how the whole figure is read.** 85 percent
of the uncertainty index's cell is an error all six methods make, and a model
that reproduced the practitioner's declarations EXACTLY would score 50.1
percent -- slightly WORSE than the six real methods at 48.7. There is nothing
left for a better method to win. This is not special to that row: the shared
share runs 64 to 91 percent across all 15 claims.

**What needs an author decision.** Which of three bar designs to adopt; the
mockups were shown in chat and their numbers are in section 2 below so they can
be rebuilt without them. Everything else here is either confirmation or a
documentation fix.

---

## 1. The truth is the right truth (no defect)

`plca.truth_rows` (`src/plca.py:693`) builds the truth with the SAME `outputs`
and `draw_contributions` the fitted methods go through, substituting
`samplers[d]` for the model and passing the same `u` and `mui`. There is no
second code path to drift. The truth is per (pLCA, material): the true
uncertainty index runs from -0.0008 to 0.9916 with a standard deviation of
0.215, and it is identical across the six methods to 0.000e+00.

**Units are right, and the silent default never fires.** All 10,050 parent
specs carry a `normalizer`, none is 1.0, and they span 0.051 to 199. The stored
values sit uniformly on their own parent's CDF -- mean 0.5011, median 0.4994 --
and the KS distance between a dataset and its parent falls as one over the
square root of n (median 0.220 at n = 3-9, 0.015 at n >= 1000). A wrong scale
factor could not produce that.

**Recomputed two independent ways** over 160 materials in 40 groups: through
`ParentSampler` on fresh variates, and analytically as `Var_j / sum_k Var_k`
from the parent's own `ppf` on a 200,001-point quantile lattice, with no
sampler and no Monte Carlo. Stored against Monte Carlo 0.0161 mean absolute
difference; stored against analytic 0.0130; the two of mine against each other
0.0095. That is Monte Carlo noise.

**So what.** The experiment does what it says: it measures the error a fitted
model causes, with the sampling noise removed by common random numbers.

    python - <<'PY'
    import gzip, json, numpy as np, pandas as pd
    s = json.load(gzip.open('data/processed/corpus_2026-09-25/parents_spec.json.gz','rt'))['specs']
    n = np.array([float(v['normalizer']) for v in s.values()])
    print(len(s), 'specs; missing normalizer:',
          sum('normalizer' not in v for v in s.values()),
          '; exactly 1.0:', int((n == 1.0).sum()),
          '; range %.4f to %.4f' % (n.min(), n.max()))
    t = pd.read_csv('outputs/tables/TABLE_PLCATruth.csv.gz')
    m = t[t.truth_parent == 'market']
    print('truth spread across methods: %.3e'
          % m.groupby(['plca','dataset'])['ui__truth'].apply(lambda s: s.max()-s.min()).max())
    PY

---

## 2. THE DEFECT: the bar is an average, the cells are per decision

The bar is `max_m(mean error) - min_m(mean error)`. On one building the six
methods disagree far more than their averages do. All figures are percent of
the claim's own true level.

    claim                                   bar now  mean pair  worst pair
    the uncertainty index                       2.2       27.9        40.2
    the total: its 90th percentile              0.8        8.3        11.6
    a material: its 95th percentile             2.6       13.3        17.8
    a material: its share at the building 95th  3.4       17.3        22.8
    a material: its standard deviation          7.0       17.1        22.4
    a material: its share of the total          4.4       11.1        15.5
    the total: its mean                         4.8        9.6        14.7
    the probability B beats A                   7.0       14.0        20.0
    a material: its chance of being largest    18.1       34.8        49.9
    a cap: its mean saving                     19.5       34.7        52.5
    a material: its mean contribution           7.5       13.0        19.6
    a cap: how often it binds                  27.3       38.5        57.2
    a cap: its chance of saving 5 pct          28.8       39.5        59.1
    the total: its standard deviation          11.4       13.8        17.6
    the chance of meeting a budget              5.0        5.9         8.5

**It cross-checks against a number the study already publishes.**
`TABLE_PLCANRMSE.csv` gives the uncertainty index an NRMSE of 0.5504, which
with a table standard deviation of 0.1999 is an RMS between-method difference
of 0.1101, or 44.0 percent of the 0.25 true level. The bar says 2.2.

**So what.** As it stands the figure tells a reader that for the uncertainty
index the choice of method is nearly free. Per decision it is not. What is
nearly free is the average over many buildings, which is a narrower claim and a
different reader's question.

### The three bar designs put to the author

Numbers below are per claim, percent of true level, and are everything needed
to rebuild the mockups.

    claim                                  bar_now pair_mean worst_cell shared specific
    the total: its mean                        4.8       9.6       14.2    9.3      4.9
    the total: its standard deviation         11.4      13.8       35.0   27.3      7.7
    the total: its 90th percentile             0.8       8.3       13.4   11.4      1.9
    the chance of meeting a budget             5.0       5.9       10.1    6.1      4.0
    a material: its mean contribution          7.5      13.0       21.2   14.8      6.4
    a material: its standard deviation         7.0      17.1       36.4   30.3      6.2
    a material: its 95th percentile            2.6      13.3       24.1   20.6      3.5
    a material: its share of the total         4.4      11.1       16.7   11.8      4.9
    a material: its share at the bldg 95th     3.4      17.3       29.5   24.7      4.7
    a material: its chance of being largest   18.1      34.8       50.5   32.2     18.4
    the uncertainty index                      2.2      27.9       49.8   44.5      5.2
    a cap: how often it binds                 27.3      38.5       58.4   35.0     23.4
    a cap: its mean saving                    19.5      34.7       56.5   38.3     18.1
    a cap: its chance of saving 5 pct         28.8      39.5       61.3   37.0     24.3
    the probability B beats A                  7.0      14.0       19.6   12.6      7.0

**A, minimal.** One bar, same slot, `pair_mean`. Label "what the choice costs in
one decision".

**B, both readings.** Wide accent bar at `pair_mean`, narrow dark bar inside it
at `bar_now`, annotated "28 / 2". Declares which statistic is which, which is
the project's standing convention.

**C, reframe.** Replace the bar with `shared` stacked under `specific`, summing
to the worst method's own cell. Answers finding 3 instead of finding 2.

**A and C give different cost numbers for the same row and must not be quoted
for each other.** A is a per-case distance between two methods; C is a
difference of averaged magnitudes.

    python - <<'PY'
    import sys, itertools; sys.path.insert(0, 'src')
    import numpy as np, pandas as pd, plca
    r = pd.read_csv('outputs/tables/TABLE_PLCATruth.csv.gz'); r = r[r.truth_parent=='market']
    w = r.pivot_table(index=['plca','dataset'], columns='method', values='ui').to_numpy(float)
    d = w[:,:,None] - w[:,None,:]; off = ~np.eye(6, dtype=bool)
    print('NRMSE %.4f ; pooled RMS between two methods %.4f = %.1f pct of 0.25'
          % (plca.nrmse(r,'ui'), np.sqrt(np.nanmean(d[:,off]**2)),
             100*np.sqrt(np.nanmean(d[:,off]**2))/0.249955))
    print('bar as shipped %.1f pct'
          % (100*pd.read_csv('outputs/tables/TABLE_ClaimScorecardWithRule.csv')
             .query("claim=='the uncertainty index'").stakes.iloc[0]))
    PY

---

## 3. The denominator is defensible, and it is not why the number is large

The scorecard divides by the mean true level, which for three attribution rows
and the uncertainty index is mechanically 0.25. Replacing it with the MEDIAN
per-case ratio moves the uncertainty index from 48.7 to **45.3**. The number
survives better than any other row on the figure.

**The MEAN per-case ratio is unusable and must not be published.** It is 114.8
percent, and the 4.05 percent of materials whose true value sits below a
hundredth of the mean carry 24 percent of it, while their absolute error
(0.0026) is forty times smaller than everyone else's. 0.38 percent of true
values are negative. The median is the usable statistic.

**Where the denominator does matter is the rest of the figure, in the opposite
direction.** The published statistic overstates most claims relative to the
median per-case reading -- by a factor of two on a material's 95th percentile
(22.9 to 10.8) and its mean contribution (17.1 to 9.0). The claim ordering
survives (Spearman 0.968, uncertainty index hardest either way), **but the
boxed method changes on 4 of 15 claims**: a cap's mean saving, a material's
95th percentile, a material's chance of being largest, and the total's standard
deviation all move to `KDE, Variable`.

**So what.** "10 of 15" is a property of the chosen denominator, not only of
the methods. The title asserts it without qualification.

    python - <<'PY'
    import pandas as pd, numpy as np
    r = pd.read_csv('outputs/tables/TABLE_PLCATruth.csv.gz'); r = r[r.truth_parent=='market']
    e, t = r.ui__error.abs(), r.ui__truth.abs()
    print('published %.1f ; median per-case %.1f ; mean per-case %.1f'
          % (100*e.mean()/abs(r.ui__truth.mean()), 100*np.median(e/t), 100*np.mean(e/t)))
    PY

---

## 4. Decision 159 is right in direction, 40 percent of the magnitude, and its
## wording points at the wrong culprit

Its signed-error pattern reproduces on the current corpus almost exactly:
-0.0926 at n = 3-9, +0.0043 at 10-99, +0.0402 at 100-999, +0.0483 at 1000+,
against its -0.093, +0.005, +0.042, +0.046. Its shared/specific table
reproduces too: shared 0.1113 here against its 0.1017, per-method specific
0.039 to 0.061.

**Two qualifications.**

**The systematic small-n shrinkage explains only 40 percent of the error.**
Replacing each material's variance distortion by its band-and-method median --
keeping the systematic part, deleting the idiosyncratic -- leaves 0.0487 of the
observed 0.1218.

**It is the DATA that understates the variance, not the methods.** Split the
distortion into the sample's own market-weighted standard deviation against the
parent's, and the fitted model's against the sample's:

    n          sample sd / parent sd   model sd / sample sd, six methods
    3-9                        0.492   0.99 to 1.39
    10-99                      0.807   0.90 to 1.13
    100-999                    0.948   0.86 to 1.04
    1000+                      0.994   0.84 to 1.01

Five declarations drawn from a heavy-tailed parent show about half its spread,
and every method faithfully reproduces or exceeds what the data shows. Decision
159's last sentence already says this; its table is headed "every method
understates the variance", and that half is what gets quoted.

**The consequence, which is the most important number in this review.** A model
that reproduced the declarations exactly would score **50.1 percent** against
the six methods' 48.7. It holds under both weighting schemes separately (50.5
for a uniform-weighted benchmark against published 48.6 to 49.8; 50.1 for a
market-weighted one against 47.6 to 49.6). The same benchmark takes 81 percent
of a material's mean contribution, 94 percent of its standard deviation and 102
percent of its share of the total.

**The worked example.** `dataset1019` in pLCA 0 has five values at 0.072,
0.753, 0.919, 1.584, 1.672, with market weights of 0.78 and 0.21 on the top
two. Their market-weighted standard deviation is 0.313. The parent's is 3.830.
The six methods report 0.296 to 0.603 -- between 0.95 and 1.93 times what the
data shows. No estimator could have recovered 3.83 from those five numbers.

**So what.** The scorecard is, to first order, a map of how little a handful of
EPDs tells you about a product population, with a thin layer of method choice on
top. That is a finding worth printing, not a weakness to hide.

    python - <<'PY'
    import numpy as np, pandas as pd
    r = pd.read_csv('outputs/tables/TABLE_PLCATruth.csv.gz'); r = r[r.truth_parent=='market'].copy()
    v = pd.read_parquet('data/processed/corpus_2026-09-25/values.parquet')
    w = v.groupby('dataset_id')['weight'].sum()
    mu = (v.assign(p=v.value*v.weight).groupby('dataset_id')['p'].sum())/w
    sq = (v.assign(p=v.value**2*v.weight).groupby('dataset_id')['p'].sum())/w
    r = r.join(np.sqrt(sq-mu**2).rename('s_sd'), on='dataset')
    k = [r.plca, r.method]; sh = lambda x: x/x.groupby(k).transform('sum')
    lvl = abs(r.ui__truth.mean())
    print('published    %.1f pct' % (100*(r.ui-r.ui__truth).abs().mean()/lvl))
    print('perfect data %.1f pct' % (100*(sh(r.s_sd**2)-sh(r.eci_std__truth**2)).abs().mean()/lvl))
    PY

---

## 5. Numbers that moved

**None.** No table, figure or constant was changed. Every figure in this review
is read off `outputs/` as committed on 2026-10-02 or computed from it.

---

## 6. What is still open

| item | who |
|---|---|
| Choose bar design A, B or C, and have a session implement it in notebook 3's scorecard cell | author, then a code window |
| Append the discrepancy entry in section 7 to `reports/MANUSCRIPT_discrepancies.md` | the next code window |
| Decide whether the median per-case column goes in the supplement | manuscript |
| Decide whether the title's "10 of 15" carries a qualifier, given it moves on 4 of 15 rows under the other denominator | manuscript |

---

## 7. The discrepancy entry, ready to append

> **N. The scorecard's right-hand bar is an averaged quantity beside per-decision
> cells.** `metricset`'s `stakes` is `max_m(mean error) - min_m(mean error)`,
> which is the spread of the AVERAGE error across methods, while every cell of
> the same figure is the error in ONE decision (decision 174). For the
> uncertainty index the bar reads 2.2 percent of true level where the mean
> per-pair per-decision difference is 27.9 and the worst pair is 40.2, and the
> study's own published NRMSE of 0.5504 implies 44.0. The figure therefore tells
> a reader the choice of method is nearly free on that claim when what is nearly
> free is the average over many buildings. Found 2026-10-05 by an independent
> review; no number on disk is wrong and no table moves. Reproduce with the
> command in `reports/REVIEW_scorecard_uncertainty_index.md` section 2.

---

## 8. Inputs and outputs

**Inputs**, all corpus `corpus_2026-09-25` at `WEIGHT_RHO = 0.5`:
`outputs/tables/TABLE_PLCATruth.csv.gz`, `TABLE_PLCATruthBuilding.csv.gz`,
`TABLE_PLCATruthIntervention.csv.gz`, `TABLE_PLCADesignSwap.csv.gz`,
`TABLE_ClaimScorecardWithRule.csv`, `TABLE_PLCANRMSE.csv`,
`data/processed/corpus_2026-09-25/{values.parquet,parents_spec.json.gz}`.

**Outputs:** this file. Nothing else.

---

## 9. What the next window picks up first

The bar, once the author has chosen a design. It is one cell of notebook 3
(the `# FIGURE: every claim a probabilistic LCA makes` cell) plus whatever
`metricset` column the new bar reads, and it is a FIGURE change, so
`audits/render_figures.py --only "every claim"` redraws it in seconds rather
than re-running the notebook. Append the section 7 entry in the same commit.

---

## 10. CLOSING: what was acted on, 2026-10-05

Written by the code window that implemented the fix, after the review above.
Corpus `corpus_2026-09-25`, `WEIGHT_RHO = 0.5`.

### What was reproduced before anything was changed

Every number this review rests on, re-run in the pinned environment:

- **NRMSE 0.5504.** `TABLE_PLCANRMSE.csv` gives the uncertainty index
  0.550353; recomputing it from the row-level truth table through
  `plca.nrmse` gives 0.5505. Same number.
- **Pooled per-decision RMS 0.1101, or 44.0 percent of the 0.25 true level.**
- **The shipped bar read 2.2 percent.**
- **All fifteen rows by all five columns of the section 2 table**, to within
  the 0.1 the review rounds to. One correction to how it should be read: its
  `specific` column is the RESIDUAL `worst_cell - shared`, not a mean
  deviation from the cross-method mean, which is what makes design C's stack
  close exactly. Recomputing it the other way gives 24.4 on the uncertainty
  index rather than 5.2.

### What the author chose, and one thing the review did not flag

**Design A, with the rows re-sorted.** One bar at `pair_mean`, the x axis
reading "what the choice costs in one decision".

**Design C would not have fixed the defect**, which is worth recording because
it was offered as an equal option. Its orange segment, `worst_cell - shared`,
is 5.2 percent on the uncertainty index where the per-decision disagreement is
27.3: a difference of two averaged magnitudes, so a reader would again have
concluded the choice is nearly free on that row. C answers finding 3 -- that
most of the error is shared -- which is real and important, but it answers it
instead of the bar's own question rather than as well as.

**Re-sorting was forced by the change, not chosen on top of it.** The rows were
ordered by `stakes` within each question block. Once the bar draws a different
quantity those bars no longer read descending -- the first block ran 14, 6, 9,
8 -- so the sort key had to follow the bar. No cell value moves with it.

### What changed

| | |
|---|---|
| `src/metricset.py` | `choice_cost`, new, with `CLAIM_UNIT_KEYS`. Differences the methods UNIT BY UNIT and returns `pair_mean`, `pair_worst`, `pair_best`, `shared`, `specific`, `worst_cell` and `stakes_mean`, all on the scorecard's own divisor |
| `tests/test_metricset.py` | five tests. The first plants the defect: two methods 0.6 apart on every building whose average errors are equal, so `stakes` reads one value and the per-decision statistic another |
| notebook 3, new compute cell | reads the four row-level truth frames and their four Rule siblings from disk, writes `TABLE_ClaimChoiceCost.csv`. Depends on the setup cells and nothing else, so it re-runs in seconds |
| notebook 3, figure cell | reads that table, draws `pair_mean`, sorts rows by it |
| `CONTEXT.md` | the `stakes` description, plus two staleness corrections found while editing it -- see below |
| `reports/MANUSCRIPT_discrepancies.md` | entry 186 |

**The fifteen rows, before and after, as a percentage of each claim's own true
level.** `bar_was` is `stakes`; `bar_now` is `pair_mean`:

    claim                                       bar_was  bar_now  worst pair
    the total: its mean                             4.8      9.0        14.7
    the total: its standard deviation              11.4     13.6        18.6
    the total: its 90th percentile                  0.8      7.9        11.6
    the chance of meeting a budget                  5.0      5.5         8.5
    a material: its mean contribution               7.5     12.2        19.6
    a material: its standard deviation              7.0     16.6        22.8
    a material: its 95th percentile                 2.6     12.6        17.8
    a material: its share of the total              4.4     10.5        15.5
    a material: its share at the building 95th      3.4     16.7        22.8
    a material: its chance of being largest        18.1     33.0        49.9
    the uncertainty index                           2.2     27.3        40.2
    a cap: how often it binds                      27.3     36.4        57.2
    a cap: its mean saving                         19.5     32.6        52.5
    a cap: its chance of saving 5 pct              28.8     37.3        59.1
    the probability B beats A                       7.0     13.2        20.0

The understatement ran from a factor of **1.11** on the chance of meeting a
budget to **12.66** on the uncertainty index.

**These are over all SEVEN policies**, which is the set `stakes` spanned, so
the statistic changed and the method set did not. The review's section 2 table
is over the six fixed methods, which is why its uncertainty-index `pair_mean`
reads 27.9 against the 27.3 now on the figure.

### The controls

- **`TABLE_ClaimScorecardWithRule.csv` is byte-identical**, so no cell of the
  figure moved: the cells pivot that table. `git diff --exit-code` on it
  passes.
- **The compute cell recomputes `stakes` from the row-level frames and
  reproduces the scorecard's to 2.5e-15**, which is the check that it reads
  the same experiment and differs in the statistic alone. It is an assert, so
  a future run fails rather than drifts.
- **`figstyle.check_overlaps` reports no `OVERLAPPING TEXT`** on the redrawn
  figure. The right-hand block header does not collide, because the bar's
  label is its x axis rather than a panel title -- which is the arrangement
  the cell already used and the reason the review's own mockups hit it.
- **`audits/figure_manifest.py`**: 0 orphans, 0 duplicates, 37 of 37 figure
  cells renderable against the setup block alone.
- Nothing else on disk moved. `git status` shows the notebook, `metricset`,
  its tests, `CONTEXT.md`, the discrepancy file, the figure's PNG and PDF, and
  the one new table.

### Two staleness corrections found in `CONTEXT.md` while fixing it

Both are decision 247 not having reached that file, and both are in the
sentences being edited:

- The `TABLE_MetricClaimScorecard.csv` row said **SIXTEEN** claims; the table
  holds **fifteen**. So did the `TABLE_MixedPolicyScorecard.csv` row.
- That second row also said **SEVEN** policies; the table holds **62** -- the
  six fixed methods plus every swept cutoff and one-axis variant.

### What is still open

| item | who |
|---|---|
| Whether the median per-case denominator goes in the supplement, and whether the title's count carries a qualifier given it moves on 4 of 15 rows under that denominator (section 3) | manuscript |
| Whether the paper prints finding 4 -- that a model reproducing the declarations exactly scores 50.1 percent on the uncertainty index against the six methods' 48.7 (section 4) | manuscript |
| Figure numbering, full `FIGURE_STYLE.md` compliance, confidence intervals on figure aggregates | deferred to the figure selection, decision 235. Unchanged by this work |

Nothing here is waiting on a code window.
